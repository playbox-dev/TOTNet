import torch
import torch.nn as nn

class PhysicsInformedModel(nn.Module):
    def __init__(self, d_model, hidden_size, num_layers):
        """
        A PINN that predicts ball position and enforces physics-based constraints.
        Args:
            d_model (int): Dimension of the input features.
            hidden_size (int): Hidden size for the MLP.
            num_layers (int): Number of hidden layers in the MLP.
        """
        super(PhysicsInformedModel, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            *layers,
            nn.Linear(hidden_size, 2)  # Output x, y coordinates
        )

    def forward(self, x):
        return self.network(x)

    def physics_loss(self, predicted, initial, velocity, acceleration, time_steps):
        """
        Args:
            predicted: Predicted positions [B, N, 2] (x, y).
            initial: Initial positions [B, 2].
            velocity: Initial velocity [B, 2].
            acceleration: Acceleration [B, 2].
            time_steps: Time steps [B].

        Returns:
            Physics-based loss enforcing the motion equations.
        """
        # Compute expected positions using physics equations
        expected = initial + velocity * time_steps.unsqueeze(-1) + 0.5 * acceleration * (time_steps ** 2).unsqueeze(-1)
        
        # MSE loss between predicted and physics-constrained expected positions
        loss = nn.functional.mse_loss(predicted, expected)
        return loss

# Example usage
if __name__ == "__main__":
    # Dummy inputs
    B = 4  # Batch size
    d_model = 512  # Input feature size
    hidden_size = 256  # Hidden layer size
    num_layers = 3  # Number of layers in the MLP

    # Initialize the model
    model = PhysicsInformedModel(d_model=d_model, hidden_size=hidden_size, num_layers=num_layers)

    # Example data
    initial_position = torch.tensor([[0.0, 0.0]] * B)  # Initial coordinates (x_0, y_0)
    velocity = torch.tensor([[5.0, 2.0]] * B)  # Initial velocity (v_x, v_y)
    acceleration = torch.tensor([[0.0, -9.8]] * B)  # Gravity on y-axis
    time_steps = torch.tensor([1.0, 2.0, 3.0, 4.0])  # Time steps for each batch

    # Forward pass (using random input features for demo)
    input_features = torch.randn(B, d_model)
    predicted_positions = model(input_features)

    # Compute physics-informed loss
    loss = model.physics_loss(predicted_positions, initial_position, velocity, acceleration, time_steps)
    print(f"Physics-informed loss: {loss.item()}")
