import torch
import torch.nn as nn
import sys
import cv2 

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
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)
    

class TemporalConvNet(nn.Module):
    def __init__(self, input_channels, spatial_channels=64):
        super(TemporalConvNet, self).__init__()
        
        self.convblock1_out_channels = spatial_channels * 2
        self.convblock2_out_channels = spatial_channels * 4
        self.convblock3_out_channels = spatial_channels * 8
        
        # Spatial convolutions
        self.conv1 = ConvBlock(in_channels=input_channels, out_channels=spatial_channels)
        self.conv2 = ConvBlock(in_channels=spatial_channels, out_channels=self.convblock1_out_channels)
        
        # Temporal convolution for block 1
        self.temporal_conv1 = nn.Conv1d(self.convblock1_out_channels, self.convblock1_out_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(in_channels=self.convblock1_out_channels, out_channels=self.convblock1_out_channels)
        self.conv4 = ConvBlock(in_channels=self.convblock1_out_channels, out_channels=self.convblock2_out_channels)
        
        # Temporal convolution for block 2
        self.temporal_conv2 = nn.Conv1d(self.convblock2_out_channels, self.convblock2_out_channels, kernel_size=5, padding=5//2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = ConvBlock(in_channels=self.convblock2_out_channels, out_channels=self.convblock2_out_channels)
        self.conv6 = ConvBlock(in_channels=self.convblock2_out_channels, out_channels=self.convblock3_out_channels)
        
        # Temporal convolution for block 3
        self.temporal_conv3 = nn.Conv1d(self.convblock3_out_channels, self.convblock3_out_channels, kernel_size=7, padding=7//2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, N, C, H, W]
        Returns:
            output: Tensor of shape [B, N, C, H, W] after applying spatial and temporal convolutions
        """
        B, N, C, H, W = x.shape

        # Reshape to [B*N, C, H, W] for spatial convolutions
        x = x.view(B * N, C, H, W)  # Merge batch and frame dimensions

        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(B, N, self.convblock1_out_channels, H, W).permute(0, 3, 4, 2, 1).contiguous()  # Reshape for temporal conv

        x_res = x.view(B*H*W, self.convblock1_out_channels, N)  # Reshape for Conv1d
        x_temporal = self.temporal_conv1(x_res)
        x = x_temporal + x_res

        x = x.view(B, H, W, self.convblock1_out_channels, N)
        x = x.permute(0, 4, 3, 1, 2).contiguous().reshape(B * N, self.convblock1_out_channels, H, W)  # Reshape back to spatial
        x = out1 = self.pool1(x)
        
    
        # Block 2
        H2,W2 = H//2, W//2
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.reshape(B, N, self.convblock2_out_channels, H2, W2).permute(0, 3, 4, 2, 1).contiguous()

        x_res = x.view(B*H2*W2, self.convblock2_out_channels, N)
        x_temporal = self.temporal_conv2(x_res)
        x = x_temporal + x_res

        x = x.view(B, H2, W2, self.convblock2_out_channels, N)
        x = x.permute(0, 4, 3, 1, 2).contiguous().reshape(B * N, self.convblock2_out_channels, H2, W2)
        x = out2 = self.pool2(x)

        # Block 3
        H3,W3 = H2//2, W2//2
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.reshape(B, N, self.convblock3_out_channels, H3, W3).permute(0, 3, 4, 2, 1).contiguous()

        x_res = x.view(B*H3*W3, self.convblock3_out_channels, N)
        x_temporal = self.temporal_conv3(x_res)
        x = x_temporal + x_res

        x = x.view(B, H3, W3, self.convblock3_out_channels, N)
        x = x.permute(0, 4, 3, 1, 2).contiguous().reshape(B * N, self.convblock3_out_channels, H3, W3)
        x = self.pool3(x)

        return x.view(B*N, self.convblock3_out_channels, H3//2, W3//2)  # Reshape back to [B, N, C, H, W]


def build_motion_model(args):
    # motion_model = MotionModel()
    model = TemporalConvNet(input_channels=3, spatial_channels=64).to(args.device)
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
    # Create dataloaders
    train_dataloader, val_dataloader, train_sampler = create_occlusion_train_val_dataloader(configs)
    batch_data, (masked_frameids, labels) = next(iter(train_dataloader))

    B, N, C, H, W = batch_data.shape

    motion_model = build_motion_model(configs)
    print(f"motion model num params is {get_num_parameters(motion_model)}")
    #Forward pass through the backbone
    with torch.no_grad():  # Disable gradient computation for testing
        motion_features = motion_model(batch_data.float())
    
    # Verify output shapes, the output shape is [B*P, 3, 2048, 34, 60] where B*P is batch and pair numbers, 3 means frame1, frame2 and motion feature
    print(f"Features stacked_features Shape: {motion_features.shape}")  # Expected: [B*P, 3, 2048, 34, 60]
    
