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

def build_motion_model(args):
    motion_model = MotionModel()
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
    torch.set_printoptions(threshold=torch.inf)
    with open('motion_features.txt', 'w') as f:
        for i, batch in enumerate(motion_features):
            for j, frame in enumerate(batch):
                f.write(f"Motion feature batch {i}, frame {j}:\n")
                f.write(f"{frame}\n")
