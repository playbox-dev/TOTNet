import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SpatialFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(SpatialFeatureExtractor, self).__init__()
        # Load pre-trained ResNet-50
        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            resnet.eval()
        else:
            resnet = models.resnet50(weights=None)
        # Remove the fully connected layer and average pooling
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Output: [Batch, 2048, H/32, W/32]
        
    def forward(self, x):
        """
        Forward pass for spatial feature extraction.

        Args:
            x (Tensor): Input tensor of shape [Batch * Pair number, 3, H, W]

        Returns:
            Tensor: Extracted spatial features of shape [Batch * Pair number, 2048, H/32, W/32]
        """
        features = self.feature_extractor(x)
        return features


class PositionEncoding2D(nn.Module):
    def __init__(self, channels, height, width):
        super(PositionEncoding2D, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width
        
        if channels % 4 != 0:
            raise ValueError("Channels must be divisible by 4 for 2D positional encoding.")
        
        self.register_buffer("pos_encoding", self.create_position_encoding())
        
    def create_position_encoding(self):
        """
        Creates a 2D sine-cosine positional encoding.

        Returns:
            Tensor: Positional encoding tensor of shape [1, C, H, W]
        """
        C, H, W = self.channels, self.height, self.width
        pe = torch.zeros(1, C, H, W)
        y_embed = torch.linspace(0, 1, steps=H).unsqueeze(1).repeat(1, W)
        x_embed = torch.linspace(0, 1, steps=W).unsqueeze(0).repeat(H, 1)
        
        dim_range = torch.arange(0, C // 4, 1).float() / (C // 4)
        div_term = torch.exp(-torch.log(torch.tensor(10000.0)) * dim_range)
        
        pe_sin = torch.sin(x_embed.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))  # [C//4, H, W]
        pe_cos = torch.cos(x_embed.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))  # [C//4, H, W]
        pe_sin_y = torch.sin(y_embed.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))  # [C//4, H, W]
        pe_cos_y = torch.cos(y_embed.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))  # [C//4, H, W]
        
        pe[0, 0::4, :, :] = pe_sin
        pe[0, 1::4, :, :] = pe_cos
        pe[0, 2::4, :, :] = pe_sin_y
        pe[0, 3::4, :, :] = pe_cos_y
        
        return pe  # [1, C, H, W]

    def forward(self, x):
        """
        Adds positional encoding to the input feature map.

        Args:
            x (Tensor): Input feature map of shape [Batch * Pair number, C, H, W]

        Returns:
            Tensor: Feature map with positional encoding added, shape [Batch * Pair number , C, H, W]
        """
        return self.pos_encoding.to(x.device)



class BackboneWithOptimizedPositionalEncoding(nn.Module):
    def __init__(self, pretrained=True, height=34, width=60):
        super(BackboneWithOptimizedPositionalEncoding, self).__init__()
        self.spatial_extractor = SpatialFeatureExtractor(pretrained=pretrained)
        self.position_encoding = PositionEncoding2D(channels=2048, height=height, width=width)
        
    def forward(self, frame1, frame2):
        """
        Forward pass for the backbone network with optimized positional encoding.

        Args:
            frame1 (Tensor): First frame of shape [Batch * Pair number, 3, H, W]
            frame2 (Tensor): Second frame of shape [Batch * Pair number, 3, H, W]

        Returns:
            Tuple[Tensor, Tensor]: Tuple containing:
                - features_frame1: [Batch * Pair number, 2048, H/32, W/32]
                - features_frame2: [Batch * Pair number, 2048, H/32, W/32]
        """
        # Extract spatial features
        features_frame1 = self.spatial_extractor(frame1)  # [Batch * Pair number, 2048, H/32, W/32]
        features_frame2 = self.spatial_extractor(frame2)  # [Batch * Pair number, 2048, H/32, W/32]
        
        # Apply positional encoding
        position_encoding = self.position_encoding(features_frame1)  # [Batch * Pair number, 2048, H/32, W/32]
        
        return features_frame1, features_frame2, position_encoding

    

if __name__ == '__main__':
    # Instantiate the backbone
    height = 1080 // 32  # 33.75, typically rounded down to 33
    width = 1920 // 32   # 60
    backbone = BackboneWithOptimizedPositionalEncoding(pretrained=True, height=34, width=60)  # Using 34 to match integer division

    # Create dummy input frames
    batch_size = 2
    pair_number = 7
    frame1 = torch.randn(batch_size * pair_number, 3, 1080, 1920)  # [Batch * Pair number, 3, H, W]
    frame2 = torch.randn(batch_size * pair_number, 3, 1080, 1920)

    # Forward pass through the backbone
    with torch.no_grad():  # Disable gradient computation for testing
        features_frame1, features_frame2, positional_encoding = backbone(frame1, frame2)

    # Verify output shapes
    print(f"Features Frame 1 Shape: {features_frame1.shape}")  # Expected: [14, 2048, 34, 60]
    print(f"Features Frame 2 Shape: {features_frame2.shape}")  # Expected: [14, 2048, 34, 60]
    print(f"Positional Encoding Map Shape: {positional_encoding.shape}")  # Expected: [14, 2048, 34, 60]



    # import matplotlib.pyplot as plt

    # # Instantiate the positional encoding module
    # channels = 2048
    # height = 34
    # width = 60
    # pos_encoding_module = PositionEncoding2D(channels=channels, height=height, width=width)
    # # Access the positional encoding tensor
    # pos_encoding = pos_encoding_module.pos_encoding  # Shape: [1, C, H, W]


    # # Define a list of channels to visualize
    # channels_to_visualize = [
    #     (0, 'Channel 0 (Sine x-axis)'),
    #     (1, 'Channel 1 (Cosine x-axis)'),
    #     (2, 'Channel 2 (Sine y-axis)'),
    #     (3, 'Channel 3 (Cosine y-axis)')
    # ]

    # # Iterate through the selected channels, visualize, and save each
    # for idx, title in channels_to_visualize:
    #     # Extract the positional encoding for the current channel
    #     pe_channel = pos_encoding[0, idx].cpu().numpy()  # Convert to NumPy for plotting
        
    #     # Create the plot
    #     plt.figure(figsize=(6, 4))
    #     plt.imshow(pe_channel, cmap='viridis', aspect='auto')
    #     plt.title(f'Positional Encoding - {title}')
    #     plt.colorbar()
        
    #     # Save the figure
    #     filename = f'pos_encoding_channel_{idx}.png'
    #     plt.savefig(filename, bbox_inches='tight')
    #     print(f"Saved {filename}")
        
    #     # Display the plot
    #     plt.show()
        
    #     # Close the figure to free up memory
    #     plt.close()
