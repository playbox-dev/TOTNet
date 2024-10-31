import torch
import torch.nn as nn

class PhysicsLoss(nn.Module):
    def __init__(self, fps=25, velocity_threshold=5.0, epsilon=1e-8, loss_scale=1e-7):
        """
        Args:
            time_step (float): Constant time difference between frames.
            velocity_threshold (float): Threshold to detect abrupt velocity changes (e.g., ball hit).
        """
        super(PhysicsLoss, self).__init__()
        self.fps = fps  # Constant time step
        self.velocity_threshold = velocity_threshold  # Threshold for adaptive loss
        self.epsilon = epsilon
        self.loss_scale = loss_scale

        
    def heatmap_to_coord(self, heatmaps_x, heatmaps_y):
        """
        Convert heatmap to (x, y) coordinates based on the maximum value.
        Args:
            heatmapx (tensor): [B, N, W] heatmap tensor
            heatmapy (tensor): [B, N, H] heatmap tensor
        Returns:
            coordinates (tensor): [B, N, 2] tensor of (x, y) coordinates
        """
        B, N, W = heatmaps_x.shape
        _, _, H = heatmaps_y.shape

        # Get the index of the maximum value along the width (x-axis) and height (y-axis)
        max_x = torch.argmax(heatmaps_x, dim=-1)  # Shape: [B, N]
        max_y = torch.argmax(heatmaps_y, dim=-1)  # Shape: [B, N]

        # Stack the (x, y) coordinates along the last dimension
        coordinates = torch.stack((max_x, max_y), dim=-1)  # Shape: [B, N, 2]

        return coordinates

    def forward(self, heatmapsx, heatmapsy, labels):
        """
        Args:
            heatmapsx (tensor): [B, N, W] heatmap tensor.
            heatmapsy (tensor): [B, N, H] heatmap tensor.
            labels (tensor): [B, N, 2] actual ground truth labels (with possible [0, 0] for missing values).
        Returns:
            total_loss (tensor): Combined velocity and acceleration loss.
        """
        # Convert heatmaps to coordinates: [B, N, 2]
        predicted_coords = self.heatmap_to_coord(heatmapsx, heatmapsy)

        # Mask for valid labels (exclude [0, 0])
        valid_mask = (labels != 0).all(dim=-1)  # [B, N]

        # Compute predicted velocity and acceleration
        predicted_velocity, predicted_acceleration = self.compute_velocity_acceleration(predicted_coords, valid_mask, self.fps)

        # Compute actual velocity and acceleration from labels
        actual_velocity, actual_acceleration = self.compute_velocity_acceleration(labels, valid_mask, self.fps)

        # Compute loss between predicted and actual values
        velocity_loss = self.compute_loss(predicted_velocity, actual_velocity, valid_mask[:, 1:])
        acceleration_loss = self.compute_loss(predicted_acceleration, actual_acceleration, valid_mask[:, 2:])

        # Combine losses with a regularization weight
        total_loss = velocity_loss + 0.1 * acceleration_loss

        total_loss = self.loss_scale * total_loss

        return total_loss
    
    def compute_velocity_acceleration(self, coordinates, valid_mask, fps):
        """
        Computes velocity and acceleration from coordinates while respecting the valid mask.
        Args:
            coordinates (tensor): [B, N, 2] coordinates for each frame.
            valid_mask (tensor): [B, N] mask indicating valid frames (1 for valid, 0 for invalid).
            fps (float): Frames per second (FPS) rate of the video.
        Returns:
            velocity (tensor): [B, N-1, 2] velocities between consecutive valid frames.
            acceleration (tensor): [B, N-2, 2] accelerations between consecutive valid velocities.
        """
        # Initialize lists to store velocities and accelerations
        all_velocities = []
        all_accelerations = []

        for batch_idx in range(coordinates.shape[0]):
            valid_coords = coordinates[batch_idx][valid_mask[batch_idx] == 1]  # [V, 2]
            valid_frame_indices = torch.nonzero(valid_mask[batch_idx]).squeeze(-1)  # [V]

            # Compute time intervals between consecutive valid frames
            time_intervals = (valid_frame_indices[1:] - valid_frame_indices[:-1]) / fps  # [V-1]

            # Compute velocity: v = Δx / Δt
            delta_positions = valid_coords[1:] - valid_coords[:-1]  # [V-1, 2]
            velocities = delta_positions / time_intervals.unsqueeze(-1)  # [V-1, 2]

            # Compute acceleration: a = Δv / Δt
            delta_velocities = velocities[1:] - velocities[:-1]  # [V-2, 2]
            acceleration_intervals = time_intervals[1:]  # [V-2]
            accelerations = delta_velocities / acceleration_intervals.unsqueeze(-1)  # [V-2, 2]

            # Pad velocities and accelerations to maintain consistent dimensions
            padded_velocity = torch.zeros((coordinates.shape[1] - 1, 2), device=coordinates.device)
            padded_acceleration = torch.zeros((coordinates.shape[1] - 2, 2), device=coordinates.device)

            padded_velocity[:velocities.shape[0]] = velocities
            padded_acceleration[:accelerations.shape[0]] = accelerations

            all_velocities.append(padded_velocity)
            all_accelerations.append(padded_acceleration)

        # Stack all velocities and accelerations for the batch
        velocity = torch.stack(all_velocities)  # [B, N-1, 2]
        acceleration = torch.stack(all_accelerations)  # [B, N-2, 2]

        return velocity, acceleration
    
    def compute_loss(self, predicted, actual, valid_mask):
        """
        Computes L2 loss between predicted and actual values, ignoring invalid frames.
        Args:
            predicted (tensor): Predicted values (velocity or acceleration).
            actual (tensor): Actual values (velocity or acceleration).
            valid_mask (tensor): Mask indicating valid frames.
        Returns:
            loss (tensor): Loss value.
        """
        # Compute squared error only on valid frames
        squared_error = (predicted - actual) ** 2
        masked_error = squared_error * valid_mask.unsqueeze(-1)  # Apply mask

        # Compute mean loss over valid entries
        loss = masked_error.sum() / valid_mask.sum()

        return loss
    

# Example usage
if __name__ == "__main__":
    # Dummy inputs
    B = 4  # Batch size
    d_model = 512  # Input feature size
    hidden_size = 256  # Hidden layer size
    num_layers = 3  # Number of layers in the MLP

    # Initialize the model
    model = PhysicsLoss()

    # Example data
    initial_position = torch.tensor([[0.0, 0.0]] * B)  # Initial coordinates (x_0, y_0)
    heatmapx = torch.randn([B,5,320])
    heatmapy = torch.randn([B,5,128])
    labels = torch.rand([B,5,2])


    # Compute physics-informed loss
    loss = model(heatmapx, heatmapy, labels)
    print(f"Physics-informed loss: {loss.item()}")
