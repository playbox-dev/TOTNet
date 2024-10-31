import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import copy
sys.path.append('../')

from model.ConvLSTM import ConvLSTM
from model.ops.modules.ms_deform_attn import DeformAttn


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


class GatedFusion(nn.Module):
    def __init__(self, feature_dim):
        super(GatedFusion, self).__init__()
        
        # Learnable linear layer to compute gating weights
        self.gate_fc = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, spatial_features, temporal_features):
        # Concatenate spatial and temporal features along the channel dimension
        combined_features = torch.cat([spatial_features, temporal_features], dim=-1)  # Shape: [B, HW, C*2]
        
        # Compute gating weights
        gate = torch.sigmoid(self.gate_fc(combined_features))  # Shape: [B, HW, C]
        
        # Apply gated fusion
        fused_features = gate * spatial_features + (1 - gate) * temporal_features  # Shape: [B, HW, C]
        
        return fused_features

class PixelTemporalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, num_frames):
        """
        Args:
            d_model (int): Dimension of the model (C).
            num_heads (int): Number of attention heads.
            num_layers (int): Number of temporal self-attention layers.
        """
        super(PixelTemporalSelfAttention, self).__init__()
        self.num_layers = num_layers
        self.gated_fusion = GatedFusion(d_model)
        # Learnable weights for each frame
        self.frame_weights = nn.Parameter(torch.ones(num_frames))

        # Create a list of multi-head attention layers and normalization layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, batch_first=True) 
            for _ in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

    def compute_motion_difference(self, memory):
        """
        Args:
            memory: [B, N, HW, C] – Feature maps for all frames in the batch.
        Returns:
            motion_diff: [B, N-1, HW, C] – Motion difference between consecutive frames.
            motion_magnitude: [B, N-1, HW] – Magnitude of motion difference (L2 norm).
        """
        # Calculate the difference between consecutive frames: memory[:, t+1, ...] - memory[:, t, ...]
        motion_diff = memory[:, 1:, :, :] - memory[:, :-1, :, :]  # [B, N-1, HW, C]

        # Calculate the L2 norm (motion magnitude) along the feature dimension
        motion_magnitude = torch.norm(motion_diff, dim=-1)  # [B, N-1, HW]

        return motion_diff, motion_magnitude

    def forward(self, memory):
        """
        Args:
            memory: [B, N, HW, C] – Feature maps for all frames.
        Returns:
            last_frame_output: [B, HW, C] – Temporal-attended feature map for the last frame.
        """
        B, N, HW, C = memory.shape  # B: Batch, N: Frames, HW: Pixels, C: Channels

        # Normalize the frame weights
        normalized_weights = torch.softmax(self.frame_weights, dim=0)  # Shape: [N]

        # Reshape to [B*HW, N, C] – Each spatial location becomes a sequence over frames
        memory = memory.permute(0, 2, 1, 3).contiguous()  # [B, HW, N, C]
        memory = memory.view(B * HW, N, C)  # [B*HW, N, C]

        # Apply frame weights to each frame in memory
        weighted_memory = memory * normalized_weights.view(1, N, 1)  # Broadcasting [N] to [B*HW, N, C]

        # Use the last frame as the query
        query = weighted_memory[:, -1:, :].clone()  # Last frame as query, shape: [B*HW, 1, C]

        # Apply multiple layers of temporal self-attention
        for i in range(self.num_layers):
            # Apply multi-head attention using the last frame as the query
            query, _ = self.attention_layers[i](query, weighted_memory, weighted_memory)  # [B*HW, 1, C]


        # Apply gated fusion
        output = self.gated_fusion(weighted_memory[:, -1:, :], query).view(B, HW, C)

        return output
    



class TemporalDeformableSelfAttentionEncoder(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, num_layers=4, num_queries=50):
        """
        Args:

            num_layers (int): Number of temporal self-attention layers.
        """
        super(TemporalDeformableSelfAttentionEncoder, self).__init__()
        encoder_layer = EncoderLayer(d_model=d_model, d_ffn=d_ffn,
                                    dropout=dropout, activation=activation,
                                    n_levels=n_levels, n_heads=n_heads, n_points=n_points)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.num_queries = num_queries
        # Learnable linear layer to predict reference points (x, y) coordinates
        self.reference_predictor = nn.Linear(d_model, num_queries * 2)  # Predicts [num_queries, 2] per frame
    
    def compute_motion_difference(self, memory):
        """
        Args:
            memory: [B, N, HW, C] – Feature maps for all frames in the batch.
        Returns:
            motion_diff: [B, N-1, HW, C] – Motion difference between consecutive frames.
            motion_magnitude: [B, N-1, HW] – Magnitude of motion difference (L2 norm).
        """
        B, N, HW, C = memory.shape
        # Initialize motion difference with zeros
        motion_diff = torch.zeros(B, N-1, HW, C, device=memory.device)
        # Compute differences for the first frame (only the next frame difference)
        motion_diff[:, 0, :, :] = torch.abs(memory[:, 1, :, :] - memory[:, 0, :, :])

        # Compute differences for intermediate frames (both previous and next frame differences)
        for t in range(1, N - 1):
            prev_diff = torch.abs(memory[:, t, :, :] - memory[:, t - 1, :, :])
            next_diff = torch.abs(memory[:, t + 1, :, :] - memory[:, t, :, :])
            # Combine both differences 
            motion_diff[:, t, :, :] = (prev_diff + next_diff)

        # Compute difference for the last frame (only the previous frame difference)
        motion_diff[:, N - 2, :, :] = torch.abs(memory[:, N - 1, :, :] - memory[:, N - 2, :, :])
        # Calculate the L2 norm (motion magnitude) along the feature dimension
        motion_magnitude = torch.norm(motion_diff, dim=-1)  # [B, N-1, HW]

        return motion_diff, motion_magnitude
    
    def predict_reference_points(self, motion_magnitude):
        """
        Predicts normalized reference points based on motion magnitude.
        Args:
            motion_magnitude: [B, N-1, HW, d_model] – Motion features for each frame.
        Returns:
            reference_points: [B * N-1, num_queries, 2] – Normalized reference points for each frame.
        """
        B, N_minus_1, HW, d_model = motion_magnitude.shape

        # Flatten the input to [B * N-1 * HW, d_model] for the linear layer
        flattened_input = motion_magnitude.view(B * N_minus_1 * HW, d_model)
        # Apply the linear layer to predict reference points
        predicted_points = self.reference_predictor(flattened_input)  # [B * N-1 * HW, num_queries * 2]

        # Reshape to [B, N-1, HW, num_queries, 2]
        predicted_points = predicted_points.view(B, N_minus_1, HW, self.num_queries, 2)

        # Take the mean over the spatial dimension to aggregate the reference points [B, N-1, num_queries, 2]
        reference_points = predicted_points.mean(dim=2)  # [B, N-1, num_queries, 2]

        # Normalize the reference points to [0, 1] using sigmoid
        reference_points = torch.sigmoid(reference_points)

        # Reshape to [B * N-1, num_queries, 2] to align with deformable attention input
        reference_points = reference_points.view(B * N_minus_1, self.num_queries, 2)

        return reference_points
    
    @staticmethod
    def get_reference_points(spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        
            # remove since all valid
            ref_y = ref_y.reshape(-1)[None] 
            ref_x = ref_x.reshape(-1)[None]
  
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points

    def forward(self, memory, spatial_shapes, level_start_index, padding_mask=None, pos=None):
        """
        Args:
            memory: [B, N, HW, C] – Feature maps for all frames.
        Returns:
            temporal_output: [B, N, HW, C] – Temporal-attended feature maps.
        """
        B, N, HW, C = memory.shape  # B: Batch, N: Frames, HW: Pixels, C: Channels
        # Step 2: Initialize output memory
        output = memory.clone()
        # Step 3: all other frames pixel can be referecne points
        # reference_points = self.predict_reference_points(motion_difference).view(B, N-1, self.num_queries, 2)  # [B*N, num_quereis, 2]
        reference_points = self.get_reference_points(spatial_shapes, memory.device) # all position as reference points
        reference_points = reference_points.repeat(B, 1, 1, 1)  # [B, HW, 1, 2]

        # Step 3: Loop through each frame as the query
        for query_idx in range(N):
            query = memory[:, query_idx, :, :]  # Query is the current frame [B, HW, C]
            # Accumulate attended queries for the current frame
            attended_query = query.clone()

            # Loop through all other frames as key-value pairs
            for key_value_idx in range(N):
                if key_value_idx == query_idx:
                    continue  # Skip self-attention within the same frame

                # Extract the key-value frame [B, HW, C]
                key_value = memory[:, key_value_idx, :, :].reshape(B, HW, C)

                # Perform deformable attention between query and key-value frames
                for _, layer in enumerate(self.layers):
                    attended_query = layer(
                        attended_query, key_value, pos, reference_points,
                        spatial_shapes, level_start_index, padding_mask
                    )  # Output: [B, HW, C]

            output = torch.cat([output[:, query_idx, :, :], attended_query])

        return output


class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        """
        Args:
            d_model: the dimensionality of the embeddings used throughout the model
        """

        # self attention
        self.self_attn = DeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, key_value, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        """
        Args: 
            src: query represents the src, which will be used with pos as the query [B, N, HW, C] – Feature maps for all frames.
            key_value: used as the value
            pos: pos encoding information, already used in the deformable encoder, so not sure if we need it
            reference_points: referecne points for this frame

        """
        # self attention
        src2 = self.self_attn(src, reference_points, key_value, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
    

def create_temporal_model(args):
    # temporal_model = ConvLSTM_Model(args.transfromer_dmodel, hidden_channels=[512, 512, 512], output_channels=args.transfromer_dmodel, num_layers=3)
    # temporal_model = TemporalDeformableSelfAttentionEncoder(d_model=args.transfromer_dmodel)
    temporal_model = PixelTemporalSelfAttention(d_model=512, num_heads=4, num_layers=4, num_frames=9)

    return temporal_model


if __name__ == '__main__':
    from config.config import parse_configs
    configs = parse_configs()
    device = 'cuda'
    dummy_input = torch.randn([8, 9, 135, 512], dtype=torch.float32, device=device)
    spatial_shape = torch.tensor([[9, 15]], dtype=torch.long, device=device)  # Shape: [1, 2]
    level_start_index = torch.tensor([0], dtype=torch.long, device=device)
    temporal_model = create_temporal_model(configs).to(device)
    print(dummy_input[0][4])
    output = temporal_model(dummy_input)
    print(output.shape)
    print(output)
