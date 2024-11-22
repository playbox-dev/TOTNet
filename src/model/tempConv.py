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
            nn.BatchNorm3d(num_features=in_channels),
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

        # Weighted pooling parameters across frames
        self.temporal_weights = nn.Parameter(torch.ones(num_frames), requires_grad=True)
        
        # Spatial convolutions
        self.conv1 = ConvBlock(in_channels=input_channels, out_channels=spatial_channels)
        self.conv2 = ConvBlock(in_channels=spatial_channels, out_channels=spatial_channels)
        
        # Temporal convolution for block 1
        self.temporal_conv1 = TemporalConvBlock(in_channels=spatial_channels, out_channels=spatial_channels, kernel_size=3, padding=1)
        

        # block 2 
        self.conv3 = ConvBlock(in_channels=spatial_channels, out_channels=self.convblock1_out_channels)
        self.conv4 = ConvBlock(in_channels=self.convblock1_out_channels, out_channels=self.convblock1_out_channels)
        
        # Temporal convolution for block 2
        self.temporal_conv2 = TemporalConvBlock(in_channels=self.convblock1_out_channels, out_channels=self.convblock1_out_channels, kernel_size=5, padding=5//2)

        #block 3
        self.conv5 = ConvBlock(in_channels=self.convblock1_out_channels, out_channels=self.convblock2_out_channels)
        # self.conv6 = ConvBlock(in_channels=self.convblock2_out_channels, out_channels=self.convblock2_out_channels)
        self.conv7 = ConvBlock(in_channels=self.convblock2_out_channels, out_channels=self.convblock2_out_channels)
        
        # Temporal convolution for block 3
        self.temporal_conv3 = TemporalConvBlock(in_channels=self.convblock2_out_channels, out_channels=self.convblock2_out_channels, kernel_size=7, padding=7//2)

        #block 4
        self.conv8 = ConvBlock(in_channels=self.convblock2_out_channels, out_channels=self.convblock3_out_channels)
        # self.conv9 = ConvBlock(in_channels=self.convblock3_out_channels, out_channels=self.convblock3_out_channels)
        self.conv10 = ConvBlock(in_channels=self.convblock3_out_channels, out_channels=self.convblock3_out_channels)

        self.temporal_conv4 = TemporalConvBlock(in_channels=self.convblock3_out_channels, out_channels=self.convblock3_out_channels, kernel_size=9, padding=9//2)


        #block 5
        self.conv11 = ConvBlock(in_channels=self.convblock3_out_channels, out_channels=self.convblock2_out_channels)
        # self.conv12 = ConvBlock(in_channels=self.convblock2_out_channels, out_channels=self.convblock2_out_channels)
        self.conv13 = ConvBlock(in_channels=self.convblock2_out_channels, out_channels=self.convblock2_out_channels)

        self.temporal_conv5 = TemporalConvBlock(in_channels=self.convblock2_out_channels, out_channels=self.convblock2_out_channels, kernel_size=7, padding=7//2)

        #block 6
        self.conv14 = ConvBlock(in_channels=self.convblock2_out_channels, out_channels=self.convblock1_out_channels)
        # self.conv15 = ConvBlock(in_channels=self.convblock1_out_channels, out_channels=self.convblock1_out_channels)
        self.conv16 = ConvBlock(in_channels=self.convblock1_out_channels, out_channels=self.convblock1_out_channels)

        self.temporal_conv6 = TemporalConvBlock(in_channels=self.convblock1_out_channels, out_channels=self.convblock1_out_channels, kernel_size=5, padding=5//2)

        #block 7
        self.conv17 = ConvBlock(in_channels=self.convblock1_out_channels, out_channels=self.spatial_channels)
        # self.conv18 = ConvBlock(in_channels=self.spatial_channels, out_channels=self.spatial_channels)
        self.conv19 = ConvBlock(in_channels=self.spatial_channels, out_channels=self.spatial_channels)
        self.temporal_conv7 = TemporalConvBlock(in_channels=self.spatial_channels, out_channels=self.spatial_channels, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
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
        x = x.view(B * N, C, H, W)  # Merge batch and frame dimensions

        # Block 1
        x = self.conv1(x)
        x = spatial_out1 = self.conv2(x) # shape [B*N, C, H, W]

        x = rearrange(x, '(b n) c h w -> b c n h w', b=B, n=N)
        x_res = x
        x_temporal = temporal_out1 = self.temporal_conv1(x)
        x = x_temporal + x_res

        # reshape back
        x = rearrange(x, 'b c n h w -> (b n) c h w', b=B, n=N)
        x = self.pool(x)

        # Block 2
        H2,W2 = H//2, W//2
        x = self.conv3(x)
        x = spatial_out2 = self.conv4(x)

        x = rearrange(x, '(b n) c h w -> b c n h w', b=B, n=N)

        x_res = x
        x_temporal = temporal_out2 = self.temporal_conv2(x)
        x = x_temporal + x_res

        # reshape back
        x = rearrange(x, 'b c n h w -> (b n) c h w', b=B, n=N)
        x = self.pool(x)

        # Block 3
        H3,W3 = H2//2, W2//2
        x = self.conv5(x)
        # x = self.conv6(x)
        x = spatial_out3 = self.conv7(x)

        x = rearrange(x, '(b n) c h w -> b c n h w', b=B, n=N)
        x_res = x

        x_temporal = temporal_out3 = self.temporal_conv3(x_res)
        x = x_temporal + x_res

        x = rearrange(x, 'b c n h w -> (b n) c h w', b=B, n=N)
        x = self.pool(x)

        # Block 4 which is the bottleneck block
        H4,W4 = H3//2, W3//2
        x = self.conv8(x)
        # x = self.conv9(x)
        x = self.conv10(x)

        x = rearrange(x, '(b n) c h w -> b c n h w', b=B, n=N)
        x_res = x

        x_temporal = self.temporal_conv4(x_res)
        x = x_temporal + x_res

        x = rearrange(x, 'b c n h w -> (b n) c h w', b=B, n=N)
       
        # block 5
        x = F.interpolate(x, size=(H3, W3), mode='nearest')
        x = self.conv11(x)
        # x = self.conv12(x)
        x = self.conv13(x)
        x = x + spatial_out3

        x = rearrange(x, '(b n) c h w -> b c n h w', b=B, n=N)
        x_res = x
        x_temporal = self.temporal_conv5(x)
        x_temporal = x_temporal + temporal_out3
        x = x_temporal + x_res

        x = rearrange(x, 'b c n h w -> (b n) c h w', b=B, n=N)
        
        # block 6
        x = F.interpolate(x, size=(H2, W2), mode='nearest')
        x = self.conv14(x)
        # x = self.conv15(x)
        x = self.conv16(x)
        x = x + spatial_out2

        x = rearrange(x, '(b n) c h w -> b c n h w', b=B, n=N)
        x_res = x
        x_temporal = self.temporal_conv6(x_res)
        # x_temporal = self.patch_processing(x_res, self.temporal_conv6, self.temporal_bn6, factor=4)
        x_temporal = x_temporal + temporal_out2
        x = x_temporal + x_res

        x = rearrange(x, 'b c n h w -> (b n) c h w', b=B, n=N)

        # block 7
        x = F.interpolate(x, size=(H, W), mode='nearest')
        x = self.conv17(x)
        # x = self.conv18(x)
        x = self.conv19(x)
        x = x + spatial_out1
        
        x = rearrange(x, '(b n) c h w -> b c n h w', b=B, n=N)
        x_res = x
        x_temporal = self.temporal_conv7(x)
        x_temporal = x_temporal + temporal_out1
        x = x_temporal + x_res
        
        x = rearrange(x, 'b c n h w -> b n c h w', b=B, n=N)

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

        return (horizontal_heatmap, vertical_heatmap), None

def build_motion_model(args):
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
    configs.dataset_choice='tennis'
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
    print(f"Features stacked_features Shape: {motion_features[0].shape}, {motion_features[1].shape}")  # Expected: [B*P, 3, 2048, 34, 60]
    print(torch.unique(motion_features[0]))
    

    