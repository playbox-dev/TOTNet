import torch
import torch.nn as nn
import sys
import cv2 

sys.path.append('../')
from model.backbone_positional_encoding import ChosenFeatureExtractor

class MotionModel(nn.Module):
    def __init__(self, num_output_channels, num_input_channels=3):
        super(MotionModel, self).__init__()

        # First set of convolutional layers for aggressive downsampling
        self.conv = nn.Sequential(
            nn.Conv2d(num_input_channels, 512, kernel_size=3, stride=2, padding=1),  # Downsample H, W by 2
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # Further downsample H, W by 2
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),  # Downsample to H/8, W/8
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(2048, num_output_channels, kernel_size=3, stride=2, padding=1),  # Further downsample
            nn.BatchNorm2d(num_output_channels),
            nn.ReLU()
        )

        # Final convolution to ensure the output is exactly (9, 15)
        self.final_conv = nn.Conv2d(num_output_channels, num_output_channels, kernel_size=3, stride=2, padding=1)
    
    
    
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
            downsampled_motion = self.conv(motion_difference)
            final_motion_features = self.final_conv(downsampled_motion)
            motion_features.append(final_motion_features)
        
        # Stack the motion features to create a tensor of shape [Bs, N-1, C, H, W]
        motion_features = torch.stack(motion_features, dim=1)
        
        return motion_features

def build_motion_model(args):
    motion_model = MotionModel(num_output_channels=args.backbone_out_channels)
    return motion_model

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
    from data_process.dataloader import create_masked_train_val_dataloader

    configs = parse_configs()
    # Create dataloaders
    train_dataloader, val_dataloader, train_sampler = create_masked_train_val_dataloader(configs)
    batch_data, (masked_frameids, masked_frames, labels) = next(iter(train_dataloader))

    B, N, C, H, W = batch_data.shape

    motion_model = MotionModel()

    #Forward pass through the backbone
    with torch.no_grad():  # Disable gradient computation for testing
        motion_features = motion_model(batch_data)
    
    # Verify output shapes, the output shape is [B*P, 3, 2048, 34, 60] where B*P is batch and pair numbers, 3 means frame1, frame2 and motion feature
    print(f"Features stacked_features Shape: {motion_features.shape}")  # Expected: [B*P, 3, 2048, 34, 60]

