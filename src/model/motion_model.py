import torch
import torch.nn as nn
import sys
import cv2 
import torch.nn.functional as F
import time

sys.path.append('../')
from model.backbone_positional_encoding import ChosenFeatureExtractor

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
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.BatchNorm3d(num_features=in_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.block(x)
    

class TemporalConvNet(nn.Module):
    def __init__(self, input_channels, spatial_channels=64, num_frames=5):
        super(TemporalConvNet, self).__init__()
        
        self.spatial_channels = spatial_channels
        self.convblock1_out_channels = spatial_channels * 2
        self.convblock2_out_channels = spatial_channels * 4
        self.convblock3_out_channels = spatial_channels * 8
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool = nn.MaxPool3d()

        # Weighted pooling parameters across frames
        self.temporal_weights = nn.Parameter(torch.ones(num_frames), requires_grad=True)
        
        # Spatial convolutions
        self.conv1 = ConvBlock(in_channels=input_channels, out_channels=spatial_channels)
        self.conv2 = ConvBlock(in_channels=spatial_channels, out_channels=spatial_channels)
        
        # Temporal convolution for block 1
        # self.temporal_conv1 = nn.Conv1d(spatial_channels, spatial_channels, kernel_size=3, padding=1)
        # self.temporal_conv1 = nn.Conv1d(spatial_channels, spatial_channels, kernel_size=3, padding=1, groups=spatial_channels)
        # self.temporal_bn1 = nn.BatchNorm1d(spatial_channels)
        self.temporal_conv1 = TemporalConvBlock(in_channels=spatial_channels, out_channels=spatial_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        

        # block 2 
        self.conv3 = ConvBlock(in_channels=spatial_channels, out_channels=self.convblock1_out_channels)
        self.conv4 = ConvBlock(in_channels=self.convblock1_out_channels, out_channels=self.convblock1_out_channels)
        
        # Temporal convolution for block 2
        self.temporal_conv2 = nn.Conv1d(self.convblock1_out_channels, self.convblock1_out_channels, kernel_size=5, padding=5//2)
        # self.temporal_conv2 = nn.Conv1d(self.convblock1_out_channels, self.convblock1_out_channels, kernel_size=3, padding=2, dilation=2, groups=self.convblock1_out_channels)
        self.temporal_bn2 = nn.BatchNorm1d(self.convblock1_out_channels)

        #block 3
        self.conv5 = ConvBlock(in_channels=self.convblock1_out_channels, out_channels=self.convblock2_out_channels)
        # self.conv6 = ConvBlock(in_channels=self.convblock2_out_channels, out_channels=self.convblock2_out_channels)
        self.conv7 = ConvBlock(in_channels=self.convblock2_out_channels, out_channels=self.convblock2_out_channels)
        
        # Temporal convolution for block 3
        self.temporal_conv3 = nn.Conv1d(self.convblock2_out_channels, self.convblock2_out_channels, kernel_size=7, padding=7//2)
        # self.temporal_conv3 = nn.Conv1d(self.convblock2_out_channels, self.convblock2_out_channels, kernel_size=3, padding=3, dilation=3, groups=self.convblock2_out_channels)
        self.temporal_bn3 = nn.BatchNorm1d(self.convblock2_out_channels)

        #block 4
        self.conv8 = ConvBlock(in_channels=self.convblock2_out_channels, out_channels=self.convblock3_out_channels)
        # self.conv9 = ConvBlock(in_channels=self.convblock3_out_channels, out_channels=self.convblock3_out_channels)
        self.conv10 = ConvBlock(in_channels=self.convblock3_out_channels, out_channels=self.convblock3_out_channels)

        self.temporal_conv4 = nn.Conv1d(self.convblock3_out_channels, self.convblock3_out_channels, kernel_size=9, padding=9//2)
        # self.temporal_conv4 = nn.Conv1d(self.convblock3_out_channels, self.convblock3_out_channels, kernel_size=3, padding=4, dilation=4, groups=self.convblock3_out_channels)
        self.temporal_bn4 = nn.BatchNorm1d(self.convblock3_out_channels)

        #block 5
        self.conv11 = ConvBlock(in_channels=self.convblock3_out_channels, out_channels=self.convblock2_out_channels)
        # self.conv12 = ConvBlock(in_channels=self.convblock2_out_channels, out_channels=self.convblock2_out_channels)
        self.conv13 = ConvBlock(in_channels=self.convblock2_out_channels, out_channels=self.convblock2_out_channels)
        self.temporal_conv5 = nn.Conv1d(self.convblock2_out_channels, self.convblock2_out_channels, kernel_size=7, padding=7//2)
        # self.temporal_conv5 = nn.Conv1d(self.convblock2_out_channels, self.convblock2_out_channels, kernel_size=3, padding=3, dilation=3, groups=self.convblock2_out_channels)
        self.temporal_bn5 = nn.BatchNorm1d(self.convblock2_out_channels)

        #block 6
        self.conv14 = ConvBlock(in_channels=self.convblock2_out_channels, out_channels=self.convblock1_out_channels)
        # self.conv15 = ConvBlock(in_channels=self.convblock1_out_channels, out_channels=self.convblock1_out_channels)
        self.conv16 = ConvBlock(in_channels=self.convblock1_out_channels, out_channels=self.convblock1_out_channels)
        self.temporal_conv6 = nn.Conv1d(self.convblock1_out_channels, self.convblock1_out_channels, kernel_size=5, padding=5//2)
        # self.temporal_conv6 = nn.Conv1d(self.convblock1_out_channels, self.convblock1_out_channels, kernel_size=3, padding=2, dilation=2, groups=self.convblock1_out_channels)
        self.temporal_bn6 = nn.BatchNorm1d(self.convblock1_out_channels)

        #block 7
        self.conv17 = ConvBlock(in_channels=self.convblock1_out_channels, out_channels=self.spatial_channels)
        # self.conv18 = ConvBlock(in_channels=self.spatial_channels, out_channels=self.spatial_channels)
        self.conv19 = ConvBlock(in_channels=self.spatial_channels, out_channels=self.spatial_channels)
        self.temporal_conv7 = nn.Conv1d(self.spatial_channels, self.spatial_channels, kernel_size=3, padding=3//2)
        # self.temporal_conv7 = nn.Conv1d(self.spatial_channels, self.spatial_channels, kernel_size=3, padding=3//2, groups=spatial_channels)
        self.temporal_bn7 = nn.BatchNorm1d(self.spatial_channels)


        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
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
        x = self.conv1(x)
        x = spatial_out1 = self.conv2(x) # shape [B*N, C, H, W]

        x = x.reshape(B, N, self.spatial_channels, H, W).permute(0, 3, 4, 2, 1) # Reshape for temporal conv
        x_res = x.reshape(B*H*W, self.spatial_channels, N)  # Reshape for Conv1d
        x_temporal = temporal_out1 = self.temporal_bn1(F.relu(self.temporal_conv1(x_res)))
        # x_temporal = temporal_out1 = self.patch_processing(x_res, self.temporal_conv1, self.temporal_bn1, factor=4)
        x = x_temporal + x_res

        x = x.view(B, H, W, self.spatial_channels, N)
        x = x.permute(0, 4, 3, 1, 2).reshape(B * N, self.spatial_channels, H, W)  # Reshape back to spatial
        x = self.pool(x)

        # Block 2
        H2,W2 = H//2, W//2
        x = self.conv3(x)
        x = spatial_out2 = self.conv4(x)
        x = x.view(B, N, self.convblock1_out_channels, H2, W2).permute(0, 3, 4, 2, 1)

        x_res = x.reshape(B*H2*W2, self.convblock1_out_channels, N)
        x_temporal = temporal_out2 = self.temporal_bn2(F.relu(self.temporal_conv2(x_res)))
        # x_temporal = temporal_out2 = self.patch_processing(x_res, self.temporal_conv2, self.temporal_bn2, factor=4)
        x = x_temporal + x_res

        x = x.view(B, H2, W2, self.convblock1_out_channels, N)
        x = x.permute(0, 4, 3, 1, 2).reshape(B * N, self.convblock1_out_channels, H2, W2)
        x = self.pool(x)

        # Block 3
        H3,W3 = H2//2, W2//2
        x = self.conv5(x)
        # x = self.conv6(x)
        x = spatial_out3 = self.conv7(x)
        x = x.view(B, N, self.convblock2_out_channels, H3, W3).permute(0, 3, 4, 2, 1)

        x_res = x.reshape(B*H3*W3, self.convblock2_out_channels, N)
        x_temporal = temporal_out3 = self.temporal_bn3(F.relu(self.temporal_conv3(x_res)))
        # x_temporal = temporal_out3 = self.patch_processing(x_res, self.temporal_conv3, self.temporal_bn3, factor=4)
        x = x_temporal + x_res

        x = x.view(B, H3, W3, self.convblock2_out_channels, N)
        x = x.permute(0, 4, 3, 1, 2).reshape(B * N, self.convblock2_out_channels, H3, W3)
        x = self.pool(x)

        # Block 4 which is the bottleneck block
        H4,W4 = H3//2, W3//2
        x = self.conv8(x)
        # x = self.conv9(x)
        x = self.conv10(x)
        x = x.view(B, N, self.convblock3_out_channels, H4, W4).permute(0, 3, 4, 2, 1)

        x_res = x.reshape(B*H4*W4, self.convblock3_out_channels, N)
        x_temporal = self.temporal_bn4(F.relu(self.temporal_conv4(x_res)))
        # x_temporal = self.patch_processing(x_res, self.temporal_conv4, self.temporal_bn4, factor=4)
        x = x_temporal + x_res

        x = x.view(B, H4, W4, self.convblock3_out_channels, N)
        x = x.permute(0, 4, 3, 1, 2).reshape(B * N, self.convblock3_out_channels, H4, W4)

        # block 5
        x = F.interpolate(x, size=(H3, W3), mode='nearest')
        x = self.conv11(x)
        # x = self.conv12(x)
        x = self.conv13(x)
        x = x + spatial_out3

        x = x.view(B, N, self.convblock2_out_channels, H3, W3).permute(0, 3, 4, 2, 1)
        x_res = x.reshape(B*H3*W3, self.convblock2_out_channels, N)
        x_temporal = self.temporal_bn5(F.relu(self.temporal_conv5(x_res)))
        # x_temporal = self.patch_processing(x_res, self.temporal_conv5, self.temporal_bn5, factor=4)
        x_temporal = x_temporal + temporal_out3
        x = x_temporal + x_res

        x = x.view(B, H3, W3, self.convblock2_out_channels, N)
        x = x.permute(0, 4, 3, 1, 2).reshape(B * N, self.convblock2_out_channels, H3, W3)
        
        # block 6
        x = F.interpolate(x, size=(H2, W2), mode='nearest')
        x = self.conv14(x)
        # x = self.conv15(x)
        x = self.conv16(x)
        x = x + spatial_out2

        x = x.view(B, N, self.convblock1_out_channels, H2, W2).permute(0, 3, 4, 2, 1)
        x_res = x.reshape(B*H2*W2, self.convblock1_out_channels, N)
        x_temporal = self.temporal_bn6(F.relu(self.temporal_conv6(x_res)))
        # x_temporal = self.patch_processing(x_res, self.temporal_conv6, self.temporal_bn6, factor=4)
        x_temporal = x_temporal + temporal_out2
        x = x_temporal + x_res

        x = x.view(B, H2, W2, self.convblock1_out_channels, N)
        x = x.permute(0, 4, 3, 1, 2).reshape(B * N, self.convblock1_out_channels, H2, W2)

        # block 7
        x = F.interpolate(x, size=(H, W), mode='nearest')
        x = self.conv17(x)
        # x = self.conv18(x)
        x = self.conv19(x)
        x = x + spatial_out1
        
        x = x.view(B, N, self.spatial_channels, H, W).permute(0, 3, 4, 2, 1)
        x_res = x.reshape(B*H*W, self.spatial_channels, N)
        x_temporal = self.temporal_bn7(F.relu(self.temporal_conv7(x_res)))
        # x_temporal = self.patch_processing(x_res, self.temporal_conv7, self.temporal_bn7, factor=4)
        x_temporal = x_temporal + temporal_out1
        x = x_temporal + x_res
        
        x = x.view(B, H, W, self.spatial_channels, N)
        x = x.permute(0, 4, 3, 1, 2)

        x = x.reshape(B, N, self.spatial_channels, H, W)  # Reshape for pooling on N
        temporal_weights = torch.softmax(self.temporal_weights, dim=0).view(1, N, 1, 1, 1)
        x = (x * temporal_weights).sum(dim=1)  # Weighted sum across frames

        out = x

        # Sum along the width to get a vertical heatmap (along H dimension)
        vertical_heatmap = out.sum(dim=1).sum(dim=-1)   # Shape: [B, H]
        # Sum along the height to get a horizontal heatmap (along W dimension)
        horizontal_heatmap = out.sum(dim=1).sum(dim=-2)   # Shape: [B, W]

        # # Min-max normalization for vertical and horizontal heatmaps
        vertical_heatmap = (vertical_heatmap - vertical_heatmap.min()) / (vertical_heatmap.max() - vertical_heatmap.min() + 1e-8)
        horizontal_heatmap = (horizontal_heatmap - horizontal_heatmap.min()) / (horizontal_heatmap.max() - horizontal_heatmap.min() + 1e-8)

        vertical_heatmap = torch.softmax(vertical_heatmap, dim=-1)
        horizontal_heatmap = torch.softmax(horizontal_heatmap, dim=-1)

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
    from losses_metrics.losses import Heatmap_Ball_Detection_Loss_2D
    from losses_metrics.metrics import heatmap_calculate_metrics_2d

    configs = parse_configs()
    configs.num_frames = 5
    configs.device = 'cuda'
    configs.batch_size = 4
    configs.img_size = (360, 640)
    # Create dataloaders
    train_dataloader, val_dataloader, train_sampler = create_occlusion_train_val_dataloader(configs)
    batch_data, (masked_frameids, labels) = next(iter(train_dataloader))
    batch_data = batch_data.to(configs.device)

    B, N, C, H, W = batch_data.shape

    motion_model = build_motion_model(configs)
    print(f"motion model num params is {get_num_parameters(motion_model)}")
    # Start timer for data loading
    start_time = time.time()
    #Forward pass through the backbone
    motion_features = motion_model(batch_data.float())
    forward_pass_time = time.time() - start_time
    print(f"Forward pass time: {forward_pass_time:.4f} seconds")
    print(f"Features stacked_features Shape: {motion_features[0].shape}")  # Expected: [B*P, 3, 2048, 34, 60]
 
    
