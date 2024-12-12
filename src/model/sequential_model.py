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
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.block(x)
    


class EncoderBlock(nn.Module):
    def __init__(self, pool_size, in_channels, out_channels, spatial_kernel_size, temporal_kernel_size, 
                 padding='same', spatial_padding='same', bias=True, num_spatial_layers=2, num_temporal_layers=1):
        super().__init__()
        self.out_channels = out_channels

        self.conv_layers = nn.ModuleList()
        self.temp_layers = nn.ModuleList()
        for i in range(num_spatial_layers):
            self.conv_layers.append(ConvBlock(
                in_channels=in_channels if i == 0 else out_channels,  # Input channels for the first layer
                out_channels=out_channels,
                kernel_size=spatial_kernel_size,
                pad=spatial_padding
            ))
        
        for i in range(num_temporal_layers):
            self.temp_layers.append(TemporalConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=temporal_kernel_size,
                padding=padding,
                bias=bias
            ))
       
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3d = nn.AdaptiveMaxPool3d(pool_size)

    def forward(self, x, num_frames, previous_infor=None):
        # input in shape [BN, C, H, W]
        BN, C, H, W = x.shape
        B = BN//num_frames

        for layer in self.conv_layers:
            x = layer(x)

        spatial_out = x.clone()
        x = rearrange(x, "(b n) c h w -> b c n h w", b=B, n=num_frames)  # [B, C', N, H, W]
        x_res = x  # Residual connection

        # Temporal Convolution using Conv3d
        for layer in self.temp_layers:
            x_temporal = layer(x)
       
        temporal_out = x_temporal.clone()
        x = x_temporal + x_res  # Add residual

        if previous_infor != None:
            x += previous_infor
     
        x = self.pool3d(x)
        _, _, N, _, _ = x.shape
       
        # reshape to [B*N, C, H, W]
        x = rearrange(x, 'b c n h w -> (b n) c h w')

        return x, spatial_out, temporal_out, N

class DecoderBlock(nn.Module):
    def __init__(self, up_size, in_channels, out_channels, spatial_kernel_size, temporal_kernel_size, 
                padding='same', spataial_padding='same', bias=True, final=False, num_spatial_layers=2, num_temporal_layers=1):
        super().__init__()
        self.out_channels = out_channels
        self.final = final
        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.Upsample(size=up_size, mode='trilinear')

        self.conv_layers = nn.ModuleList()
        self.temp_layers = nn.ModuleList()

        for i in range(num_spatial_layers):
            self.conv_layers.append(ConvBlock(
                in_channels=in_channels if i == 0 else out_channels,  # Input channels for the first layer
                out_channels=out_channels,
                kernel_size=spatial_kernel_size,
                pad=spataial_padding
            ))
        
        for i in range(num_temporal_layers):
            self.temp_layers.append(TemporalConvBlock(
                in_channels=out_channels*2 if i ==0 else out_channels,
                out_channels=out_channels,
                kernel_size=temporal_kernel_size,
                padding=padding,
                bias=bias
            ))
        
        if final == True:
            self.residual_proj = TemporalConvBlock(in_channels=out_channels, out_channels=1, kernel_size=(1, 1, 1), padding=(0, 0, 0))
            self.temp_layers.append(TemporalConvBlock(out_channels, 1, temporal_kernel_size, padding, bias))


    def forward(self, x, spatial_concat, temporal_concat):
        # input in shape [BN, C, H, W]
        x = self.up(x)
        B, C, N, H, W = x.shape
        x = rearrange(x, 'b c n h w -> (b n) c h w', b=B, n=N)
        x = torch.concat((x, spatial_concat), dim=1)

        for layer in self.conv_layers:
            x = layer(x)

        x = rearrange(x, '(b n) c h w -> b c n h w', b=B, n=N)  # [B, C', N, H, W]
        x_res = x
        # Temporal Convolution using Conv3d
        x = torch.concat((x, temporal_concat), dim=1)

        for layer in self.temp_layers:
            x = layer(x)

        if self.final:
            x_res = self.residual_proj(x_res)  # Project to [B, 1, N, H, W]
        
        x = x + x_res
        # x = rearrange(x, 'b c n h w -> (b n) c h w')

        return x

class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_kernel_size, temporal_kernel_size, 
                 padding='same', bias=True, num_spatial_layers=2, num_temporal_layers=1):
        super().__init__()
        self.out_channels = out_channels

        self.conv_layers = nn.ModuleList()
        self.temp_layers = nn.ModuleList()

        for i in range(num_spatial_layers):
            self.conv_layers.append(ConvBlock(
                in_channels=in_channels if i == 0 else out_channels,  # Input channels for the first layer
                out_channels=out_channels,
                kernel_size=spatial_kernel_size,
                pad=padding
            ))
        
        for i in range(num_temporal_layers):
            self.temp_layers.append(TemporalConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=temporal_kernel_size,
                padding=padding,
                bias=bias
            ))

    def forward(self, x, N, previous_infor=None):
        # Block 4 which is the bottleneck block
        BN, C, H, W = x.shape
        B = BN//N


        for layer in self.conv_layers:
            x = layer(x)

        x = rearrange(x, '(b n) c h w -> b c n h w',b=B, n=N)
        x_res = x  # Residual connection

        x_temporal = None
        # Temporal Convolution using Conv3d
        for layer in self.temp_layers:
            x_temporal = layer(x)
        if x_temporal != None:
            x = x_temporal + x_res  # Add residual
        else:
            x = x_res
        

        if previous_infor != None:
            x = x + previous_infor

        # x = rearrange(x, 'b c n h w -> b c n h w', b=B, n=self.num_frames)

        return x

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super(ClassificationHead, self).__init__()
        
        # Create a list of layers
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))  # Linear layer
            layers.append(nn.ReLU())                     # Activation
            layers.append(nn.Dropout(dropout))               # Optional dropout
            in_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(in_dim, output_dim))  # Linear layer for output classes
        
        # Combine layers into a Sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TemporalConvNet(nn.Module):
    def __init__(self, input_shape=(288, 512), spatial_channels=64, num_frames=5):
        super(TemporalConvNet, self).__init__()

        self.spatial_channels = spatial_channels
        self.convblock1_out_channels = spatial_channels * 2
        self.convblock2_out_channels = spatial_channels * 4
        self.convblock3_out_channels = spatial_channels * 8
        self.softmax = nn.Softmax(dim=-1)
        self.num_frames = num_frames

        size = (num_frames, input_shape[0], input_shape[1])
        size1 = (num_frames, 144, 256)
        size2 = (3, 72, 128)
        size3 = (1, 36, 64)


        # block 1
        # Spatial convolutions
        self.block1 = EncoderBlock(pool_size=size1, in_channels=3, out_channels=spatial_channels, 
                                spatial_kernel_size=3, temporal_kernel_size=(self.num_frames, 3, 3))

        # block 2 
        self.block2 = EncoderBlock(pool_size=size2, in_channels=spatial_channels, out_channels=self.convblock1_out_channels, 
                                   spatial_kernel_size=2, temporal_kernel_size=(3, 2, 2), 
                                   num_spatial_layers=2, num_temporal_layers=2)

        #block 3
        self.block3 = EncoderBlock(pool_size=size3, in_channels=self.convblock1_out_channels, out_channels=self.convblock2_out_channels, 
                                   spatial_kernel_size=2, temporal_kernel_size=(3, 2, 2), 
                                   num_spatial_layers=2, num_temporal_layers=2)

        self.bottle_neck = BottleNeckBlock(in_channels=self.convblock2_out_channels, out_channels=self.convblock3_out_channels,
                                           spatial_kernel_size=1, temporal_kernel_size=(1, 1, 1), 
                                           num_spatial_layers=3, num_temporal_layers=0)

        #block 5
        self.block5 = DecoderBlock(size2, self.convblock3_out_channels+self.convblock2_out_channels, self.convblock2_out_channels, 
                                   spatial_kernel_size=2, temporal_kernel_size=(3, 2, 2), 
                                   num_spatial_layers=2, num_temporal_layers=2)

        #block 6
        self.block6 = DecoderBlock(size1, self.convblock2_out_channels+self.convblock1_out_channels, self.convblock1_out_channels, 
                                   spatial_kernel_size=2, temporal_kernel_size=(3, 2, 2), 
                                   num_spatial_layers=2, num_temporal_layers=2)

        #block 7
        self.block7 = DecoderBlock(size, self.convblock1_out_channels+self.spatial_channels, self.spatial_channels, 
                                   spatial_kernel_size=3, temporal_kernel_size=(self.num_frames, 3, 3), final=False)


        # projection block
        self.temp_reduce = TemporalConvBlock(in_channels=self.spatial_channels, out_channels=1, kernel_size=(num_frames, 1, 1), padding=(0, 0, 0))
        
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
            elif isinstance(module, nn.Linear):
                # Initialize linear layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


    def forward(self, x, previous_infor=None):
        """
        Args:
            x: Tensor of shape [B, N, C, H, W]
        Returns:
            tuple: heatmap in both x and y directions 
        """
        B, N, C, H, W = x.shape

        if previous_infor != None:
            x = x + previous_infor.expand_as(x)

        # Reshape to [B*N, C, H, W] for spatial convolutions
        x = rearrange(x, 'b n c h w -> (b n) c h w', b=B, n=N) # Merge batch and frame dimensions

        # Block 1
        x, spatial_out1, temporal_out1, N = self.block1(x, N)

        # Block 2
        x, spatial_out2, temporal_out2, N = self.block2(x, N)

        # Block 3
        x, spatial_out3, temporal_out3, N = self.block3(x, N)

        # block 4 bottleneck
        x = self.bottle_neck(x, N)
       
        # block 5
        x= self.block5(x, spatial_out3, temporal_out3)
        
        # block 6
        x = self.block6(x, spatial_out2, temporal_out2)

        # block 7
        x = self.block7(x, spatial_out1, temporal_out1) #outputs [B*N, C, H, W] 


        x = self.temp_reduce(x) 
        information = x
        out = x.squeeze(dim=1).squeeze(dim=1) #[B, H, W]

        # Sum along the width to get a vertical heatmap (along H dimension)
        vertical_heatmap = out.max(dim=2)[0]   # Shape: [B, H]
        # Sum along the height to get a horizontal heatmap (along W dimension)
        horizontal_heatmap = out.max(dim=1)[0]   # Shape: [B, W]
        
        vertical_heatmap = self.softmax(vertical_heatmap)
        horizontal_heatmap = self.softmax(horizontal_heatmap) 

        return (horizontal_heatmap, vertical_heatmap), None, information


class BounceConvNet(nn.Module):
    def __init__(self):
        super().__init__()
    




class SequentialConvNet(nn.Module):
    def __init__(self, input_shape=(288, 512), spatial_channels=64, total_num_frames=5, heatmap_num_frames=3):
        super(SequentialConvNet, self).__init__()

        self.heatmap_num_frames = heatmap_num_frames
        self.num_frames = total_num_frames
        self.heatmap_model = TemporalConvNet(input_shape=input_shape, spatial_channels=spatial_channels, num_frames=heatmap_num_frames)
        self.bounce_net = BounceConvNet()

    
    def forward(self, x):
        """
        Process video frames to generate ball position heatmaps
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C, H, W]
        """
        B, N, C, H, W = x.shape
        heatmaps = None
        cls = None

        # Ensure we have enough frames to process
        for i in range(N):
            # Correctly slice input with proper boundary handling
            start = max(0, i - self.heatmap_num_frames + 1)
            end = i + 1
            input = x[:, start:end, :, :, :]

            # Pad if not enough frames
            if input.shape[1] < self.heatmap_num_frames:
                # Create padding tensor with same device and dtype as input
                needed_frame_num = self.heatmap_num_frames - input.shape[1]
                pad_input = torch.zeros(
                    [B, needed_frame_num, C, H, W], 
                    dtype=input.dtype, 
                    device=input.device
                )
                input = torch.cat([input, pad_input], dim=1)
            # Process the input through heatmap model
            ball_xys_heatmaps, _, heatmaps = self.heatmap_model(input, heatmaps)

        return ball_xys_heatmaps, cls



def build_motion_model_light(args):
    # motion_model = MotionModel()
    model = TemporalConvNet(input_shape=(288, 512), spatial_channels=64, num_frames=args.num_frames).to(args.device)
    return model

def build_sequential_model(args):
    sequential_model = SequentialConvNet(input_shape=(288, 512), spatial_channels=64, total_num_frames=args.num_frames).to(args.device)
    return sequential_model

if __name__ == '__main__':
    from config.config import parse_configs
    from data_process.dataloader import create_occlusion_train_val_dataloader
    from model.model_utils import get_num_parameters

    configs = parse_configs()
    configs.num_frames = 5
    configs.device = 'cuda'
    configs.batch_size = 1
    configs.img_size = (288, 512)
    configs.dataset_choice = 'tennis'
    # Create dataloaders
    train_dataloader, val_dataloader, train_sampler = create_occlusion_train_val_dataloader(configs)
    batch_data, (masked_frameids, labels, _, _) = next(iter(train_dataloader))
 

    # print(torch.unique(batch_data))
    # batch_data = torch.randn([configs.batch_size, configs.num_frames, 3, 288, 512])

    batch_data = batch_data.to(configs.device)

    B, N, C, H, W = batch_data.shape

    # network = SaliencyMask().to(configs.device)
    # print(f"attention model num params is {get_num_parameters(network)}")
    # output = network(batch_data.float())

    # motion_model = build_motion_model_light(configs)
    sequential_model = build_sequential_model(configs)
    print(f" model num params is {get_num_parameters(sequential_model)}")
    # Start timer for data loading
    start_time = time.time()
    #Forward pass through the backbone
    motion_features, cls = sequential_model(batch_data.float())
    print(f"length of output is {len(motion_features)}")
    forward_pass_time = time.time() - start_time
    print(f"Forward pass time: {forward_pass_time:.4f} seconds")
    print(f"Features stacked_features Shape: horizontal {motion_features[0].shape},   vertical {motion_features[1].shape}")  # Expected: [B*P, 3, 2048, 34, 60]
    print(f"cls score is {cls}")
    print(torch.unique(motion_features[0]))
    
