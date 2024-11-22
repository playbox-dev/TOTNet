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
        x = rearrange(x, "(b n) c h w -> b c n h w", b=B, n=self.num_frames)  # [B, C', N, H, W]
        x_res = x  # Residual connection
        # Temporal Convolution using Conv3d
        x_temporal = self.temp_conv(x)
        temporal_out = x_temporal.clone()
        x = x_temporal + x_res  # Add residual

        # reshape to [B*N, C, H, W]
        x = rearrange(x, 'b c n h w -> (b n) c h w')
        x = self.pool(x)

        return x, spatial_out, temporal_out

class DecoderBlock(nn.Module):
    def __init__(self, num_frames, up_channels, in_channels, out_channels, kernel_size, padding, bias=True, final=False):
        super().__init__()
        self.num_frames = num_frames
        self.out_channels = out_channels
        self.final = final
        # self.up = nn.ConvTranspose2d(
        #     in_channels=up_channels,
        #     out_channels=up_channels,
        #     kernel_size=2,  # Kernel size for 2x upsampling
        #     stride=2
        # )
        self.up = nn.Upsample(scale_factor=2)
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        if final == True:
            self.residual_proj = nn.Conv3d(out_channels, 1, kernel_size=1) 
            self.temp_conv = TemporalConvBlock(out_channels*2, 1, kernel_size, padding, bias)
        else:
            self.temp_conv = TemporalConvBlock(out_channels*2, out_channels, kernel_size, padding, bias)

    def forward(self, x, spatial_concat, temporal_concat):
        # input in shape [BN, C, H, W]
        BN, _, _, _ = x.shape
        B = BN//self.num_frames

        x = self.up(x)
        x = torch.concat((x, spatial_concat), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)

        x = rearrange(x, '(b n) c h w -> b c n h w', b=B, n=self.num_frames)  # [B, C', N, H, W]
        x_res = x
        # Temporal Convolution using Conv3d
        x = torch.concat((x, temporal_concat), dim=1)
        x = self.temp_conv(x)

        if self.final:
            x_res = self.residual_proj(x_res)  # Project to [B, 1, N, H, W]
        
        x = x + x_res

        x = rearrange(x, 'b c n h w -> (b n) c h w')

        return x
        



class TemporalConvNet(nn.Module):
    def __init__(self, input_channels=3, spatial_channels=64, num_frames=5):
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
        self.block1 = EncoderBlock(num_frames=num_frames, in_channels=input_channels, out_channels=spatial_channels, kernel_size=3, padding=1)

        # block 2 
        self.block2 = EncoderBlock(num_frames=num_frames, in_channels=spatial_channels, out_channels=self.convblock1_out_channels, kernel_size=5, padding=5//2)

        #block 3
        self.block3 = EncoderBlock(num_frames, in_channels=self.convblock1_out_channels, out_channels=self.convblock2_out_channels, kernel_size=7, padding=7//2)

        #block 4
        self.conv8 = ConvBlock(in_channels=self.convblock2_out_channels, out_channels=self.convblock3_out_channels)
        self.conv10 = ConvBlock(in_channels=self.convblock3_out_channels, out_channels=self.convblock3_out_channels)

        self.temporal_conv4 = TemporalConvBlock(in_channels=self.convblock3_out_channels, out_channels=self.convblock3_out_channels, kernel_size=9, padding=9//2)

        #block 5
        self.block5 = DecoderBlock(num_frames, self.convblock3_out_channels, self.convblock3_out_channels+self.convblock2_out_channels, self.convblock2_out_channels, 7, padding=7//2)

        #block 6
        self.block6 = DecoderBlock(num_frames, self.convblock2_out_channels, self.convblock2_out_channels+self.convblock1_out_channels, self.convblock1_out_channels, 5, 5//2 )

        #block 7
        self.block7 = DecoderBlock(num_frames, self.convblock1_out_channels, self.convblock1_out_channels+self.spatial_channels, self.spatial_channels, 3, padding=1, final=True)

        # projection block
        self.temp_reduce = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(num_frames, 1, 1), stride=(1,1,1), padding=(0,0,0))
        
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


    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, N, C, H, W]
        Returns:
            tuple: heatmap in both x and y directions 
        """
        B, N, C, H, W = x.shape
        
        # Reshape to [B*N, C, H, W] for spatial convolutions
        x = rearrange(x, 'b n c h w -> (b n) c h w', b=B, n=N) # Merge batch and frame dimensions

        # Block 1
        x, spatial_out1, temporal_out1 = self.block1(x)

        # Block 2
        x, spatial_out2, temporal_out2 = self.block2(x)

        # Block 3
        x, spatial_out3, temporal_out3 = self.block3(x)

        # Block 4 which is the bottleneck block
        x = self.conv8(x)
        x = self.conv10(x)

        x = rearrange(x, '(b n) c h w -> b c n h w',b=B, n=N)
        x_res = x  # Residual connection
        # Temporal Convolution using Conv3d
        x_temporal = self.temporal_conv4(x)
        x = x_temporal + x_res  # Add residual

        x = rearrange(x, 'b c n h w -> (b n) c h w', b=B, n=N)
        # block 5
        x = self.block5(x, spatial_out3, temporal_out3)
        
        # block 6
        x = self.block6(x, spatial_out2, temporal_out2)

        # block 7
        x = self.block7(x, spatial_out1, temporal_out1) #outputs [B*N, C, H, W] C = 1 

        # final project 
        x = rearrange(x, "(b n) c h w -> b n c h w", b=B, n=N)
        temporal_weights = torch.softmax(self.temporal_weights, dim=0).view(1, N, 1, 1, 1)
        x = (x * temporal_weights).sum(dim=1)  # Weighted sum across frames  #[B, C, H, W]
        # x = x[:, -1, :, :, :]  # [B, C, H, W]
        out = x.squeeze(dim=1) # [B, H, W]

        # use 3dconv to reduce temporal dim
        # x = rearrange(x, "(b n) c h w -> b c n h w", b=B, n=N)
        # x = self.temp_reduce(x)
        # out = x.squeeze(dim=1).squeeze(dim=1)
        
      
        # Sum along the width to get a vertical heatmap (along H dimension)
        vertical_heatmap = out.max(dim=2)[0]   # Shape: [B, H]
        # Sum along the height to get a horizontal heatmap (along W dimension)
        horizontal_heatmap = out.max(dim=1)[0]   # Shape: [B, W]
        
        vertical_heatmap = self.softmax(vertical_heatmap)
        horizontal_heatmap = self.softmax(horizontal_heatmap) 

        return (horizontal_heatmap, vertical_heatmap), None

def post_process(logits_output):
    horizontal_heatmap, vertical_heatmap = logits_output

    vertical_heatmap = torch.sigmoid(vertical_heatmap)
    horizontal_heatmap = torch.sigmoid(horizontal_heatmap)     

    return (horizontal_heatmap, vertical_heatmap)


def build_motion_model(args):
    # motion_model = MotionModel()
    model = TemporalConvNet(input_channels=3, spatial_channels=64, num_frames=args.num_frames).to(args.device)
    return model



if __name__ == '__main__':
    from config.config import parse_configs
    from data_process.dataloader import create_occlusion_train_val_dataloader
    from model.model_utils import get_num_parameters

    configs = parse_configs()
    configs.num_frames = 3
    configs.device = 'cpu'
    configs.batch_size = 5
    configs.img_size = (288, 512)
    configs.dataset_choice = 'tennis'
    # Create dataloaders
    train_dataloader, val_dataloader, train_sampler = create_occlusion_train_val_dataloader(configs)
    batch_data, (masked_frameids, labels, _, _) = next(iter(train_dataloader))
    batch_data = batch_data.to(configs.device)

    # print(torch.unique(batch_data))
    # batch_data = torch.randn([8, 5, 3, 288, 512])

    B, N, C, H, W = batch_data.shape

    # network = SaliencyMask().to(configs.device)
    # print(f"attention model num params is {get_num_parameters(network)}")
    # output = network(batch_data.float())

    motion_model = build_motion_model(configs)
    print(f"motion model num params is {get_num_parameters(motion_model)}")
    # Start timer for data loading
    start_time = time.time()
    #Forward pass through the backbone
    motion_features, cls = motion_model(batch_data.float())
    forward_pass_time = time.time() - start_time
    print(f"Forward pass time: {forward_pass_time:.4f} seconds")
    print(f"Features stacked_features Shape: horizontal {motion_features[0].shape},   vertical {motion_features[1].shape}")  # Expected: [B*P, 3, 2048, 34, 60]
    print(torch.unique(motion_features[0]))
    
