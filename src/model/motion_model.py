import torch
import torch.nn as nn
import sys

sys.path.append('../')
from model.backbone_positional_encoding import ChosenFeatureExtractor

class MotionModel(nn.Module):
    def __init__(self, channels=2048, pretrained=True):
        super(MotionModel, self).__init__()
        self.spatial_extractor = ChosenFeatureExtractor(choice="single", pretrained=pretrained, out_channels=channels)
        # Fusion module to combine spatial and motion features
        # Using a 1x1 convolution to reduce concatenated channels back to 2048
        self.fusion_module = nn.Conv2d(channels * 3, channels, kernel_size=1)
    
    def forward(self, frame1, frame2):
        # Extract spatial features
        frame1 = frame1.float()
        frame2 = frame2.float()
        features_frame1 = self.spatial_extractor(frame1)  # [B*7, 2048, 34, 60]
        features_frame2 = self.spatial_extractor(frame2)  # [B*7, 2048, 34, 60]
    
        # Compute motion features via feature difference
        motion_features = features_frame2 - features_frame1  # [B*7, 2048, 34, 60]
    
        # Fuse features by concatenation
        fused_features = torch.cat((features_frame1, features_frame2, motion_features), dim=1)  # [B*7, 6144, 34, 60]
    
        # Apply fusion module to reduce channels back to 2048
        fused_features = self.fusion_module(fused_features)  # [B*7, 2048, 34, 60]

        # Stack the three feature maps
        # Shape: [B*7, 3, 2048, 34, 60]
        stacked_features = torch.stack((features_frame1, features_frame2, motion_features), dim=1)  # [B*7, 3, 2048, 34, 60]
    
        return stacked_features  # [B*7, 3, 2048, 34, 60]


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
    from data_process.dataloader import create_train_val_dataloader

    configs = parse_configs()
    # Create dataloaders
    train_dataloader, val_dataloader, train_sampler = create_train_val_dataloader(configs)
    batch_data, (masked_frameids, masked_frames, labels) = next(iter(train_dataloader))

    B, num_pairs, num_images, C, H, W = batch_data.shape
    data_reshaped = batch_data.view(B * num_pairs, num_images, C, H, W)
    
    # Split into frame1 and frame2
    frame1 = data_reshaped[:, 0, :, :, :]  # [B*num_pairs, C, H, W]
    frame2 = data_reshaped[:, 1, :, :, :]  # [B*num_pairs, C, H, W]
    print(f"batch_data shape {batch_data.shape}, frame1 shape {frame1.shape}, frame2 shape {frame2.shape}")

    # # Create dummy input frames
    # batch_size = 2
    # pair_number = 7
    # stacked_input = torch.randn(batch_size, pair_number, 2, 3, 1080, 1920)
    # B, num_pairs, num_images, C, H, W = stacked_input .shape
    # data_reshaped = stacked_input.view(B * num_pairs, num_images, C, H, W)
    # frame1 = data_reshaped[:, 0, :, :, :]  # [B*num_pairs, C, H, W]
    # frame2 = data_reshaped[:, 1, :, :, :]  # [B*num_pairs, C, H, W]


    # Instantiate the MotionModel
    height = 1080 // 32  # 34
    width = 1920 // 32   # 60
    motion_model = MotionModel(channels=2048, pretrained=True)

    #Forward pass through the backbone
    with torch.no_grad():  # Disable gradient computation for testing
        stacked_features = motion_model(frame1, frame2)
    
    # Verify output shapes, the output shape is [B*P, 3, 2048, 34, 60] where B*P is batch and pair numbers, 3 means frame1, frame2 and motion feature
    print(f"Features stacked_features Shape: {stacked_features.shape}")  # Expected: [B*P, 3, 2048, 34, 60]
    # example_stacked = stacked_features[0]
    # print(f"example stacked shape {example_stacked.shape}")
    # example_feature_map1 = example_stacked[0]
    # example_feature_map2 = example_stacked[1]
    # motion_feature = example_stacked[2]
