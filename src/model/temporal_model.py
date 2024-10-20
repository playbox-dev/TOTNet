import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')

from model.ConvLSTM import ConvLSTM


class ConvLSTM_Model(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, num_layers=1):
        super(ConvLSTM_Model, self).__init__()


        self.convlstm = ConvLSTM(input_dim=input_channels,
                                 hidden_dim=hidden_channels,
                                 kernel_size=(3,3),
                                 num_layers=num_layers,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False)

        self.conv_out = nn.Conv2d(hidden_channels[-1], output_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        """this function is used to learn both the spatial and temporal information of the input sequce of frame

        Args:
            x (tensor): shape [B, N, H, W, C], the embedding from encoder, where each batch contains number of feature maps

        Returns:
            tensor: outputs [B, HW, C], the embedding that contains both spatial and temporal information of the sequence
        """
        B, N, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3) # convert to [B, N, C, H, W] 
        
        # ConvLSTM preserves the spatial structure
        lstm_out, _ = self.convlstm(x)
        lstm_out = lstm_out[0]  # Get the output for the last time step

        # Apply a final convolution to map to the desired output channels
        output = self.conv_out(lstm_out[:, -1, :, :, :])  # Take the last frame's output
        output = output.permute(0, 2, 3, 1).view(B, H*W, C)
        return output



class TemporalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        """
        Args:
            d_model (int): Dimension of the model (C).
            num_heads (int): Number of attention heads.
            num_layers (int): Number of temporal self-attention layers.
        """
        super(TemporalSelfAttention, self).__init__()
        self.num_layers = num_layers

        # Create a list of multi-head attention layers and normalization layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, batch_first=True) 
            for _ in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

    def forward(self, memory):
        """
        Args:
            memory: [B, N, H*W, C] – Feature maps for all frames.
        Returns:
            temporal_output: [B, N, H*W, C] – Temporal-attended feature maps.
        """
        B, N, HW, C = memory.shape
        memory = memory.view(B, N, -1)  # Flatten spatial dimensions (N, HW*C)

        # Apply multiple layers of temporal self-attention
        for i in range(self.num_layers):
            attn_output, _ = self.attention_layers[i](memory, memory, memory)  # [B, N, HW*C]
            memory = self.norm_layers[i](attn_output + memory)  # Residual connection

        temporal_output = memory.view(B, N, HW, C)  # Reshape back to original shape
        return temporal_output
    
    

def create_temporal_model(args):
    temporal_model = ConvLSTM_Model(args.transfromer_dmodel, hidden_channels=[512, 512, 512], output_channels=args.transfromer_dmodel, num_layers=3)

    return temporal_model


if __name__ == '__main__':
    from config.config import parse_configs
    configs = parse_configs()
    dummy_input = torch.randn([8, 8, 135, 512], dtype=torch.float32)
    temporal_model = ConvLSTM_Model(input_channels=512, hidden_channels=[64, 64, 128], output_channels=512, num_layers=3)

    output = temporal_model(dummy_input)
    print(output.shape)
