import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformableAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, num_points):
        super(DeformableAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        self.d_k = out_channels // num_heads

        # Linear projections
        self.query_proj = nn.Linear(in_channels, out_channels)
        self.key_proj = nn.Linear(in_channels, out_channels)
        self.value_proj = nn.Linear(in_channels, out_channels)

        # Offset prediction for deformable attention
        self.offset_fc = nn.Linear(in_channels, num_heads * num_points * 2)

        # Position and velocity prediction
        self.position_fc = nn.Linear(in_channels, 2)  # Predict (x, y) position
        self.velocity_fc = nn.Linear(in_channels, 2)  # Predict (vx, vy) velocity

        # Output projection
        self.out_proj = nn.Linear(out_channels, out_channels)

    def forward(self, x, feature_map, reference_point=None):
        '''
        x: Input feature vector of shape (batch_size, in_channels)
        feature_map: Input feature map of shape (batch_size, channels, height, width)
        reference_point: Previous reference point of shape (batch_size, 2) or None
        '''
        batch_size, in_channels = x.size()
        
        # Linear projections
        query = self.query_proj(x)  # (batch_size, out_channels)
        query = query.view(batch_size, self.num_heads, self.d_k)  # (batch_size, num_heads, d_k)
        
        # Predict offsets
        offsets = self.offset_fc(x)  # (batch_size, num_heads * num_points * 2)
        offsets = offsets.view(batch_size, self.num_heads, self.num_points, 2)
        
        # Predict initial reference point if not provided
        if reference_point is None:
            reference_point = self.position_fc(x)  # (batch_size, 2)
        else:
            # Ensure reference_point has the correct shape
            reference_point = reference_point  # (batch_size, 2)
        
        # Apply offsets to reference point
        reference_point_expanded = reference_point.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, 2)
        sampling_locations = reference_point_expanded + offsets  # (batch_size, num_heads, num_points, 2)
        
        # Normalize sampling locations to [-1, 1] for grid_sample
        # Assuming sampling_locations are in normalized coordinates
        sampling_locations_normalized = sampling_locations  # Ensure proper normalization
        
        # Reshape sampling locations for grid_sample
        sampling_grid = sampling_locations_normalized.view(batch_size * self.num_heads, self.num_points, 2)
        
        # Reshape feature_map for grid_sample
        feature_map = feature_map.repeat(self.num_heads, 1, 1, 1)  # (batch_size * num_heads, channels, height, width)
        
        # Sample features at sampling locations
        sampled_features = F.grid_sample(
            feature_map,  # (batch_size * num_heads, channels, height, width)
            sampling_grid.unsqueeze(2),  # (batch_size * num_heads, num_points, 1, 2)
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # Output shape: (batch_size * num_heads, channels, num_points, 1)
        
        # Reshape sampled features
        sampled_features = sampled_features.squeeze(3)  # (batch_size * num_heads, channels, num_points)
        sampled_features = sampled_features.view(batch_size, self.num_heads, self.d_k, self.num_points)  # (batch_size, num_heads, d_k, num_points)
        sampled_features = sampled_features.permute(0, 1, 3, 2)  # (batch_size, num_heads, num_points, d_k)
        
        # Compute attention scores between query and sampled keys
        query = query.unsqueeze(2)  # (batch_size, num_heads, 1, d_k)
        attention_scores = torch.einsum('bhqd,bhpd->bhqp', query, sampled_features) / (self.d_k ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, 1, num_points)
        
        # Compute attention output
        attention_output = torch.einsum('bhqp,bhpd->bhqd', attention_weights, sampled_features)
        attention_output = attention_output.squeeze(2)  # (batch_size, num_heads, d_k)
        attention_output = attention_output.contiguous().view(batch_size, -1)  # (batch_size, out_channels)
        
        # Final output projection
        output = self.out_proj(attention_output)  # (batch_size, out_channels)
        
        # Predict velocity
        estimated_velocity = self.velocity_fc(x)  # (batch_size, 2)
        
        return output, reference_point, estimated_velocity