import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from collections import OrderedDict


class MultiLevelSpatialFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, out_channels=256):
        """
        Initializes the SpatialFeatureExtractor with ResNet-50 backbone and FPN.

        Args:
            pretrained (bool): If True, loads pre-trained ResNet-50 weights.
            out_channels (int): Number of output channels for each FPN level.
        """
        super(MultiLevelSpatialFeatureExtractor, self).__init__()
        
        # Load pre-trained ResNet-50
        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            resnet.eval()  # Set to evaluation mode
        else:
            resnet = models.resnet50(weights=None)
        
        # Extract layers from ResNet-50 backbone
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # C2 (H/4, W/4)
            resnet.layer2,  # C3 (H/8, W/8)
            resnet.layer3,  # C4 (H/16, W/16)
            resnet.layer4   # C5 (H/32, W/32)
        )
        
        # Define the input channels for FPN (from C3, C4, C5)
        in_channels_list = [512, 1024, 2048]
        self.num_channels = [512, 1024, 2048]

       
        # # Freeze backbone if not training
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        # Forward pass through the ResNet backbone
        c2 = self.backbone[0:5](x)  # Output from layer1 (C2), shape: [Batch, 256, H/4, W/4]
        c3 = self.backbone[5](c2)   # Output from layer2 (C3), shape: [Batch, 512, H/8, W/8]
        c4 = self.backbone[6](c3)   # Output from layer3 (C4), shape: [Batch, 1024, H/16, W/16]
        c5 = self.backbone[7](c4)   # Output from layer4 (C5), shape: [Batch, 2048, H/32, W/32]

        # Return the multi-scale feature maps
        return [c3, c4, c5, c5]

class SingleLevelSpatialFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, out_channels=2048):
        """
        Initializes the SingleLevelSpatialFeatureExtractor with ResNet-50 backbone.
        
        Args:
            pretrained (bool): If True, loads pre-trained ResNet-50 weights.
            out_channels (int): Desired number of output channels. Must be one of [256, 512, 1024, 2048].
        
        Raises:
            ValueError: If out_channels is not among the supported values.
        """
        super(SingleLevelSpatialFeatureExtractor, self).__init__()
        
        # Define a mapping from out_channels to corresponding ResNet-50 layers
        self.channel_to_layer = {
            256: 'layer1',
            512: 'layer2',
            1024: 'layer3',
            2048: 'layer4'
        }
        
        if out_channels not in self.channel_to_layer:
            raise ValueError(f"Unsupported out_channels: {out_channels}. Supported values are {list(self.channel_to_layer.keys())}.")
        
        # Load pre-trained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        
        # Extract layers up to the specified layer  
        layers = []
        for name, module in resnet.named_children():
            layers.append(module)
            if name == self.channel_to_layer[out_channels]:
                break
        
        self.feature_extractor = nn.Sequential(*layers)  # Sequential up to the desired layer
        
    def forward(self, x):
        """
        Forward pass for spatial feature extraction.
        
        Args:
            x (torch.Tensor): Input tensor of shape [Batch * Pair number, 3, H, W]
        
        Returns:
            torch.Tensor: Extracted features of shape [1, Batch * Pair number, out_channels, H', W'] where 1 represents num_feature level
        """
        features = self.feature_extractor(x)
        # unsqueeze on dim 0 so it has the num_feature level
        return features.unsqueeze(dim=0)

def create_positional_encoding(feature):
    """
    Args:
        feature(tensor): (B, C, H, W)
    Returns:
        Tensor: Positional encoding of the same shape as the input feature
    """
    # Get the last three dimensions (C, H, W) from the input
    batch_size, channels, height, width = feature.size()

    device = feature.device
    batch_size = feature.size(0)  # Handle batch size dynamically
    if channels % 4 != 0:
        raise ValueError("Channels must be divisible by 4 for 2D positional encoding.")

    # Create an empty tensor for positional encoding with the same batch size
    pe = torch.zeros(batch_size, channels, height, width, device=device)

    # Create y and x embeddings
    y_embed = torch.linspace(0, 1, steps=height).unsqueeze(1).repeat(1, width)
    x_embed = torch.linspace(0, 1, steps=width).unsqueeze(0).repeat(height, 1)

    # Calculate the division term for the sine and cosine functions
    dim_range = torch.arange(0, channels // 4, 1).float() / (channels // 4)
    div_term = torch.exp(-torch.log(torch.tensor(10000.0)) * dim_range)

    # Calculate sine and cosine positional encodings
    pe_sin = torch.sin(x_embed.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))  # [C//4, H, W]
    pe_cos = torch.cos(x_embed.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))  # [C//4, H, W]
    pe_sin_y = torch.sin(y_embed.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))  # [C//4, H, W]
    pe_cos_y = torch.cos(y_embed.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))  # [C//4, H, W]

    # Assign the positional encodings to the positional encoding tensor
    pe[:, 0::4, :, :] = pe_sin
    pe[:, 1::4, :, :] = pe_cos
    pe[:, 2::4, :, :] = pe_sin_y
    pe[:, 3::4, :, :] = pe_cos_y

    return pe.to(device)
    


class ChosenFeatureExtractor(nn.Module):
    def __init__(self, choice, pretrained=True, out_channels=256, num_feature_levels=4):
        super(ChosenFeatureExtractor, self).__init__()
        self.out_channels = out_channels
        self.num_channels = [512, 1024, 2048]
        self.num_backbone_outputs = num_feature_levels-1
        if choice == "multi":
            self.spatialExtractor = MultiLevelSpatialFeatureExtractor(pretrained, out_channels=out_channels)
        elif choice == "single":
            self.spatialExtractor = SingleLevelSpatialFeatureExtractor(pretrained, out_channels=out_channels)
    
    def forward(self, x):
        """
        Forward pass for spatial feature extractions
        
        Args:
            x: Input tensor of shape Expected shape [Bs, Pair Number (number of frames in this list), C, H ,W]
        """
        out = self.spatialExtractor(x)

        return out


def build_backbone(args):
    return ChosenFeatureExtractor(choice=args.backbone_choice, pretrained=args.backbone_pretrained, out_channels=args.backbone_out_channels, num_feature_levels=args.num_feature_levels).to(args.device)

    

if __name__ == '__main__':
    # Instantiate the backbone and positional encoding
    # multiLevelbackbone = MultiLevelSpatialFeatureExtractor(pretrained=True, out_channels=256)
    # singleLevelbackbone = SingleLevelSpatialFeatureExtractor(pretrained=True, level=-2)
    chosenFramesBackbone = ChosenFeatureExtractor(choice="single")
    # Create dummy input frames
    batch_size = 2
    pair_number = 3
    frame1 = torch.randn(batch_size * pair_number, 3, 1080, 1920)  # [Batch * Pair number, 3, H, W]
    frame2 = torch.randn(batch_size * pair_number, 3, 1080, 1920)

    # Forward pass through the backbone
    with torch.no_grad():  # Disable gradient computation for testing
        # fpn_features_frame1, fpn_features_frame2 = multiLevelbackbone(frame1, frame2)
        # single_feature_frame1, single_feature_frame2 = singleLevelbackbone(frame1, frame2)
        feature_frame1, feature_frame2 = chosenFramesBackbone(frame1), chosenFramesBackbone(frame2)

    # Verify output shapes
    # for name, feature in fpn_features_frame1.items():
    #     _, _, height, width = feature.shape
    #     pos = create_positional_encoding(feature)
    #     print(f"{name}: {feature.shape}, pos {pos.shape}") # Expected shape [B*P, out_channels, height, width], pos shape [1, out_channels, height, width]
    
    pos = create_positional_encoding(feature_frame1)
    print(f"single feature map shape 1 {feature_frame1.shape}, single feature map shape 2 {feature_frame2.shape}, pos {pos.shape}")
        
    
    # print(f"Positional Encoding Map Shape: {positional_encoding.shape}")  # Expected: [14, 2048, 34, 60]



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
