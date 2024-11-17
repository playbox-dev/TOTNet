import torch
import torch.nn as nn
import sys
import cv2 
import torch.nn.functional as F
import time
import torchvision.models as models
import torchvision.ops.deform_conv
from einops import rearrange

sys.path.append('../')
from model.backbone_positional_encoding import create_positional_encoding

class MotionModel(nn.Module):
    def __init__(self):
        super(MotionModel, self).__init__()    
    
    def forward(self, x):
        """_summary_

        Args:
            x (tensor): shape [Bs, N, C, H, W], where N is the number of frames in the sequence

        Returns:
            tensor: _description_
        """
        motion_features = []
        
        # Loop through consecutive frames and compute absolute difference
        for i in range(x.size(1) - 1):  # size(1) is N (number of frames)
            motion_difference = torch.abs(x[:, i] - x[:, i + 1])  # Compute frame difference
    
            motion_features.append(motion_difference)
        
        # Stack the motion features to create a tensor of shape [Bs, N-1, C, H, W]
        motion_features = torch.stack(motion_features, dim=1)
        
        return motion_features
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1, 1), padding=(padding, 0, 0), bias=bias),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
        )
    def forward(self, x):
        return self.block(x)
    

class EfficientAttentionMask(nn.Module):
    def __init__(self, input_channels, threshold=0.5):
        super(EfficientAttentionMask, self).__init__()
        
        # Pooling layer to downsample the input first
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downscale by factor of 2
            nn.MaxPool2d(kernel_size=2, stride=2)   # Downscale again by factor of 2, total factor of 4
        )
        
        # Conv2d to project downsampled spatial data to 1D
        self.input_proj = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=1),
                nn.GroupNorm(32, 32),
            )
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        
        # Fully connected layer to produce the mask
        self.fc = nn.Linear(32, 1)
        
        # Threshold for masking
        self.threshold = threshold

    def forward(self, x):
        B, N, C, H, W = x.shape
        x_rehsaped = x.view(B*N,C,H,W)
        
        # Downsample the input with pooling
        x_pooled = self.pool(x_rehsaped)  # [B, N, C, H//2, W//2]
        BN, C_pooled, H_pooled, W_pooled = x_pooled.shape
        
        # Project to 1D
        x_projected = self.input_proj(x_pooled)  # [B*N, C_proj, H_pooled, W_pooled]
        # Add positional encoding
        x_projected = x_projected + create_positional_encoding(x_projected)
        x_projected = x_projected.flatten(2)

        # Reshape to [B*N, H*W, C_proj] for attention
        x_projected = x_projected.permute(0, 2, 1)
        
        # Self-attention
        attn_output, _ = self.attention(x_projected, x_projected, x_projected)  # [B*N, H*W, C_proj]
        
        # Fully connected layer to get attention scores
        attention_scores = self.fc(attn_output).squeeze(-1)  # [B*N, H*W]
        
        # Reshape back to original spatial dimensions
        attention_map = attention_scores.view(B, N, H_pooled, W_pooled)
        
        # Upsample attention map to match original dimensions
        attention_map_upsampled = F.interpolate(attention_map, size=(H, W), mode="nearest")
        
        # Apply threshold to create binary mask
        mask = (attention_map_upsampled > self.threshold).float()

        # Apply mask to original input
        masked_output = x * mask.unsqueeze(2)  # Mask over the original resolution
        return masked_output


class SaliencyMask(nn.Module):
    def __init__(self, threshold=0.5):
        super(SaliencyMask, self).__init__()
        self.saliency_model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
        self.saliency_model.eval()
        for param in self.saliency_model.parameters():
            param.requires_grad = False
        self.threshold = threshold

    def forward(self, x):
        B, N, C, H, W = x.shape
        x_flat = x.view(B * N, C, H, W)

        # Generate saliency maps
        with torch.no_grad():
            saliency_output = self.saliency_model(x_flat)['out']  # Saliency map

        saliency_map = torch.sigmoid(saliency_output).max(dim=1, keepdim=True)[0]  # [0, 1] scale
        saliency_map = saliency_map.view(B, N, 1, H, W)
        mask = (saliency_map > self.threshold).float()

        # Apply the mask
        masked_output = x * mask

        return masked_output



class EncoderBlock(nn.Module):
    def __init__(self, num_frames, in_channels, out_channels, kernel_size, padding, bias=True):
        super().__init__()
        self.num_frames = num_frames
        self.out_channels = out_channels
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.temp_conv = TemporalConvBlock(out_channels, out_channels, kernel_size, padding, bias)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # input in shape [BN, C, H, W]
        BN, C, H, W = x.shape
        B = BN//self.num_frames
        x = self.conv1(x)
        x = self.conv2(x)
        spatial_out = x.clone()
        x = x.view(B, self.num_frames, self.out_channels, H, W).permute(0, 2, 1, 3, 4)  # [B, C', N, H, W]
        x_res = x  # Residual connection
        # Temporal Convolution using Conv3d
        x_temporal = self.temp_conv(x)
        temporal_out = x_temporal.clone()
        x = x_temporal + x_res  # Add residual

        # reshape to [B*N, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(BN, self.out_channels, H, W)
        x = self.pool(x)

        return x, spatial_out, temporal_out

class DecoderBlock(nn.Module):
    def __init__(self, num_frames, in_channels, out_channels, kernel_size, padding, bias=True):
        super().__init__()
        self.num_frames = num_frames
        self.out_channels = out_channels
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.temp_conv = TemporalConvBlock(out_channels*2, out_channels, kernel_size, padding, bias)

    def forward(self, x, target_H, target_W, spatial_concat, temporal_concat):
        # input in shape [BN, C, H, W]
        BN, _, _, _ = x.shape
        B = BN//self.num_frames
        x = F.interpolate(x, size=(target_H, target_W), mode='nearest')
        x = torch.concat((x, spatial_concat), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(B, self.num_frames, x.size(1), target_H, target_W).permute(0, 2, 1, 3, 4)  # [B, C', N, H, W]
        x_res = x  # Residual connection
        # Temporal Convolution using Conv3d
        x = torch.concat((x, temporal_concat), dim=1)
        x = self.temp_conv(x)

        x = x.permute(0, 2, 1, 3, 4).reshape(BN, self.out_channels, target_H, target_W)

        return x
        



class TemporalConvNet(nn.Module):
    def __init__(self, input_channels, spatial_channels=64, num_frames=5):
        super(TemporalConvNet, self).__init__()

        self.spatial_channels = spatial_channels
        self.convblock1_out_channels = spatial_channels * 2
        self.convblock2_out_channels = spatial_channels * 4
        self.convblock3_out_channels = spatial_channels * 8
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=-1)

        # Weighted pooling parameters across frames
        self.temporal_weights = nn.Parameter(torch.ones(num_frames), requires_grad=True)
        

        # block 1
        # Spatial convolutions
        self.block1 = EncoderBlock(num_frames=num_frames, in_channels=3, out_channels=spatial_channels, kernel_size=3, padding=1)

        # block 2 
        self.block2 = EncoderBlock(num_frames=num_frames, in_channels=spatial_channels, out_channels=self.convblock1_out_channels, kernel_size=5, padding=5//2)

        #block 3
        self.block3 = EncoderBlock(num_frames, in_channels=self.convblock1_out_channels, out_channels=self.convblock2_out_channels, kernel_size=7, padding=7//2)

        #block 4
        self.conv8 = ConvBlock(in_channels=self.convblock2_out_channels, out_channels=self.convblock3_out_channels)
        # self.conv9 = ConvBlock(in_channels=self.convblock3_out_channels, out_channels=self.convblock3_out_channels)
        self.conv10 = ConvBlock(in_channels=self.convblock3_out_channels, out_channels=self.convblock3_out_channels)

        self.temporal_conv4 = TemporalConvBlock(in_channels=self.convblock3_out_channels, out_channels=self.convblock3_out_channels, kernel_size=9, padding=9//2)

        #block 5
        self.block5 = DecoderBlock(num_frames, self.convblock3_out_channels+self.convblock2_out_channels, self.convblock2_out_channels, 7, padding=7//2)

        #block 6
        self.block6 = DecoderBlock(num_frames, self.convblock2_out_channels+self.convblock1_out_channels, self.convblock1_out_channels, 5, 5//2 )

        #block 7
        self.block7 = DecoderBlock(num_frames, self.convblock1_out_channels+self.spatial_channels, self.spatial_channels, 3, padding=1)
        
        #projection
        self.final = nn.Conv3d(in_channels=spatial_channels, out_channels=1, kernel_size=(num_frames, 1, 1), stride=(1, 1, 1), padding=(num_frames//2, 0, 0))
        self.temp_pool = nn.AdaptiveAvgPool3d((1, None, None))  # Temporal size becomes 1
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0) 

    def patch_processing(self, input, conv1dlayer, batchnorm1dlayer, factor=4):
        # Calculate chunk size based on the factor
        chunk_size = input.size(0) // factor
        processed_patch = []

        # Iterate through the input in chunks
        for i in range(0, input.size(0), chunk_size):
            # Handle the last chunk, which may be smaller than chunk_size
            patch = input[i:min(i + chunk_size, input.size(0))]
            
            # Apply Conv1D and BatchNorm to the patch
            x_temporal_patch = batchnorm1dlayer(F.relu(conv1dlayer(patch)))
            processed_patch.append(x_temporal_patch)

        # Concatenate all processed patches along the 0th dimension
        x_temporal = torch.cat(processed_patch, dim=0)

        return x_temporal



    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, N, C, H, W]
        Returns:
            tuple: heatmap in both x and y directions 
        """
        B, N, C, H, W = x.shape
        
        # Reshape to [B*N, C, H, W] for spatial convolutions
        x = x.view(B * N, C, H, W)  # Merge batch and frame dimensions

        # Block 1
        x, spatial_out1, temporal_out1 = self.block1(x)

        # Block 2
        H2, W2 = H//2, W//2
        x, spatial_out2, temporal_out2 = self.block2(x)

        # Block 3
        H3,W3 = H2//2, W2//2
        x, spatial_out3, temporal_out3 = self.block3(x)

        # Block 4 which is the bottleneck block
        H4,W4 = H3//2, W3//2
        x = self.conv8(x)
        # x = self.conv9(x)
        x = self.conv10(x)

        x = x.view(B, N, self.convblock3_out_channels, H4, W4).permute(0, 2, 1, 3, 4)  # [B, C', N, H, W]
        x_res = x  # Residual connection
        # Temporal Convolution using Conv3d
        x_temporal = self.temporal_conv4(x)
        x = x_temporal + x_res  # Add residual

        x = x.permute(0, 2, 1, 3, 4).reshape(B * N, self.convblock3_out_channels, H4, W4)

        # block 5
        x = self.block5(x, H3, W3, spatial_out3, temporal_out3)
        
        # block 6
        x = self.block6(x, H2, W2, spatial_out2, temporal_out2)

        # block 7
        x = self.block7(x, H, W, spatial_out1, temporal_out1) #outputs [B*N, C, H, W]

        # final project 
        x = rearrange(x, "(b n) c h w -> b c n h w", b=B, n=N)
        x = self.final(x) # outputs [B, 1, N, H, W]
        x = self.temp_pool(x)  # Temporal size becomes 1 so [B, 1, 1, H, W]
        out = x.squeeze(dim=1).squeeze(dim=1) #[B,H,W]

        vertical_heatmap = out.max(dim=-1)[0]  # Max over width
        horizontal_heatmap = out.max(dim=-2)[0]  # Max over height
        vertical_heatmap = self.softmax(vertical_heatmap)
        horizontal_heatmap = self.softmax(horizontal_heatmap) 

        return (horizontal_heatmap, vertical_heatmap)   

def post_process(logits_output):
    horizontal_heatmap, vertical_heatmap = logits_output

    vertical_heatmap = torch.sigmoid(vertical_heatmap)
    horizontal_heatmap = torch.sigmoid(horizontal_heatmap)     

    return (horizontal_heatmap, vertical_heatmap)  


def build_motion_model(args):
    # motion_model = MotionModel()
    model = TemporalConvNet(input_channels=3, spatial_channels=64, num_frames=args.num_frames).to(args.device)
    return model


# Sample Visualization Function (Optional)
def visualize_feature_maps(features_frame1, features_frame2, motion_features, save_dir='visualizations'):
    import matplotlib.pyplot as plt
    import os

    os.makedirs(save_dir, exist_ok=True)

    # Visualize the first sample in the batch
    sample_idx = 0

    # Frame1 Features
    frame1_sample = features_frame1[sample_idx].detach().cpu().numpy()
    plt.imshow(frame1_sample.mean(axis=0), cmap='viridis')
    plt.title('Frame1 Features (Mean Across Channels)')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, 'frame1_sample.png'))
    plt.close()

    # Frame2 Features
    frame2_sample = features_frame2[sample_idx].detach().cpu().numpy()
    plt.imshow(frame2_sample.mean(axis=0), cmap='viridis')
    plt.title('Frame2 Features (Mean Across Channels)')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, 'frame2_sample.png'))
    plt.close()

    # Motion Features
    motion_sample = motion_features[sample_idx].detach().cpu().numpy()
    plt.imshow(motion_sample.mean(axis=0), cmap='viridis')
    plt.title('Motion Features (Mean Across Channels)')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, 'motion_sample.png'))
    plt.close()



if __name__ == '__main__':
    from config.config import parse_configs
    from data_process.dataloader import create_occlusion_train_val_dataloader
    from model.model_utils import get_num_parameters

    configs = parse_configs()
    configs.num_frames = 5
    configs.device = 'cpu'
    configs.batch_size = 1
    configs.img_size = (360, 6400)
    # Create dataloaders
    # train_dataloader, val_dataloader, train_sampler = create_occlusion_train_val_dataloader(configs)
    # batch_data, (masked_frameids, labels, _, _) = next(iter(train_dataloader))
    # batch_data = batch_data.to(configs.device)

    # print(torch.unique(batch_data))
    batch_data = torch.randn([8, 5, 3, 360, 640])

    B, N, C, H, W = batch_data.shape

    # network = SaliencyMask().to(configs.device)
    # print(f"attention model num params is {get_num_parameters(network)}")
    # output = network(batch_data.float())

    motion_model = build_motion_model(configs)
    print(f"motion model num params is {get_num_parameters(motion_model)}")
    # Start timer for data loading
    start_time = time.time()
    #Forward pass through the backbone
    motion_features = motion_model(batch_data.float())
    forward_pass_time = time.time() - start_time
    print(f"Forward pass time: {forward_pass_time:.4f} seconds")
    print(f"Features stacked_features Shape: {motion_features[0].shape}")  # Expected: [B*P, 3, 2048, 34, 60]
    print(torch.unique(motion_features[0]))
    
