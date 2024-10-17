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



class Temporal_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=512, num_layers=6, bidirectional=True, teacher_forcing_ratio=0):
        super(Temporal_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
        self.teacher_forcing_ratio = teacher_forcing_ratio
        if bidirectional:
            self.fc = nn.Linear(hidden_size*2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

    
    def forward(self, x):
        """_summary_

        Args:
            x (tensor): shape [B, N, H*W, C]

        Returns:
            tensor_: embeddings of an embedding represents
        """
        B, N, HW, C = x.shape
        reshaped_x = x.view(B, N, HW*C)
    
        h0 = torch.zeros(self.num_layers * 2, B, self.hidden_size).to(x.device)  # 2 for bidirectional LSTM
        c0 = torch.zeros(self.num_layers * 2, B, self.hidden_size).to(x.device)  # 2 for bidirectional LSTM
        out, (h0, c0) = self.lstm(reshaped_x, (h0, c0))  # Pass reshaped input
        out = out[:, -1, :]  # Shape: [batch_size, hidden_size * 2]
        out = out.view(B, N, HW, C)
        out = self.fc(out)  # Optional fully connected layer after RNN

        
        return out

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
