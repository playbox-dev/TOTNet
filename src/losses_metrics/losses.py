import torch.nn as nn
import torch


import torch
import torch.nn as nn

class Heatmap_Ball_Detection_Loss(nn.Module):
    def __init__(self, h, w):
        super(Heatmap_Ball_Detection_Loss, self).__init__()
        self.h = h  # Image height
        self.w = w  # Image width
        self.bce_loss = nn.BCELoss()  # Use BCEWithLogitsLoss for logits

    def forward(self, output, target_ball_position):
        """
        Args:
        - output: tuple of (pred_x, pred_y)
            - pred_x: [B, W] predicted logits across the width (x-axis)
            - pred_y: [B, H] predicted logits across the height (y-axis)
        - target_ball_position: [B, 2] true (x, y) integer pixel coordinates of the ball
        """
        # Correctly unpack the output logits
        pred_x, pred_y = output

        # Ensure target positions are of type LongTensor and on the same device
        device = pred_x.device
        target_x = target_ball_position[:, 0].long().to(device)  # [B]
        target_y = target_ball_position[:, 1].long().to(device)  # [B]

        # Create one-hot encoded ground truth for x and y
        batch_size = pred_x.size(0)

        # Clamp the indices to valid ranges
        target_x = torch.clamp(target_x, 0, pred_x.shape[1] - 1)
        target_y = torch.clamp(target_y, 0, pred_y.shape[1] - 1)

        # For x-axis predictions
        target_x_one_hot = torch.zeros_like(pred_x)  # [B, W]
        target_x_one_hot.scatter_(1, target_x.unsqueeze(1), 1.0)

        # For y-axis predictions
        target_y_one_hot = torch.zeros_like(pred_y)  # [B, H]
        target_y_one_hot.scatter_(1, target_y.unsqueeze(1), 1.0)

        # Compute binary cross-entropy loss for x and y
        loss_x = self.bce_loss(pred_x, target_x_one_hot)
        loss_y = self.bce_loss(pred_y, target_y_one_hot)

        # Return the combined loss
        return loss_x + loss_y


class HeatmapBallDetectionLoss(nn.Module):
    def __init__(self, h, w):
        super(HeatmapBallDetectionLoss, self).__init__()
        self.h = h  # Image height
        self.w = w  # Image width
        self.bce_loss = nn.BCELoss()  # Use BCEWithLogitsLoss for logits

    def forward(self, output, target_ball_position):
        """
        Args:
        - output: tuple of (pred_x, pred_y)
            - pred_x: [B, N, W] predicted logits across the width (x-axis)
            - pred_y: [B, N, H] predicted logits across the height (y-axis)
        - target_ball_position: [B, N, 2] true (x, y) integer pixel coordinates of the ball.
            [-1, -1] entries indicate missing ball positions.
        """
        # Unpack the output logits
        pred_x, pred_y = output  # [B, N, W] and [B, N, H]

        # Ensure target positions are of type LongTensor and on the same device
        device = pred_x.device
        target_x = target_ball_position[..., 0].long().to(device)  # [B, N]
        target_y = target_ball_position[..., 1].long().to(device)  # [B, N]

        # Clamp the indices to valid ranges
        target_x = torch.clamp(target_x, 0, pred_x.shape[2] - 1)  # W dimension
        target_y = torch.clamp(target_y, 0, pred_y.shape[2] - 1)  # H dimension

        # Initialize the total loss
        total_loss_x, total_loss_y = 0.0, 0.0
        valid_count = 0

        # Iterate over each frame and batch to compute the loss only on valid frames
        batch_size, num_frames = target_x.shape
        for b in range(batch_size):
            for n in range(num_frames):
                if (target_x[b, n] == 0) and (target_y[b, n] == 0):
                    # Skip invalid frames with [-1, -1] labels
                    continue

                # Create one-hot encoded ground truth for the current frame
                target_x_one_hot = torch.zeros_like(pred_x[b, n])  # [W]
                target_y_one_hot = torch.zeros_like(pred_y[b, n])  # [H]

                # Set the ground truth positions in the one-hot vectors
                target_x_one_hot[target_x[b, n]] = 1.0
                target_y_one_hot[target_y[b, n]] = 1.0

                # Compute the BCE loss for the current frame
                loss_x = self.bce_loss(pred_x[b, n], target_x_one_hot)
                loss_y = self.bce_loss(pred_y[b, n], target_y_one_hot)
        
                # Accumulate the total loss
                total_loss_x += loss_x
                total_loss_y += loss_y
                valid_count += 1

        # Avoid division by zero
        if valid_count > 0:
            total_loss_x /= valid_count
            total_loss_y /= valid_count

        # Return the combined loss
        return total_loss_x + total_loss_y

def calculate_rmse_from_heatmap(output, labels, scale=None):
    """
    Extract coordinates from the predicted logits and compute RMSE with ground truth labels.

    Args:
        output (tuple): Tuple of (pred_x, pred_y), where:
                        - pred_x: [B, W] logits across width (x-axis).
                        - pred_y: [B, H] logits across height (y-axis).
        labels (tensor): Ground truth coordinates of shape [B, 2].

    Returns:
        rmse (tensor): RMSE loss between the predicted and ground truth coordinates.
    """

    pred_x, pred_y = output  # Unpack the tuple
    B, W = pred_x.shape  # Shape of x-axis logits
    _, H = pred_y.shape  # Shape of y-axis logits

    # Ensure the labels are on the correct device
    labels = labels.to(pred_x.device)

    # Apply softmax to get probability distributions along each axis
    prob_x = torch.softmax(pred_x, dim=-1)  # [B, W]
    prob_y = torch.softmax(pred_y, dim=-1)  # [B, H]

    # Create coordinate grids for x and y axes
    x_grid = torch.linspace(0, W - 1, W, device=pred_x.device)  # [W]
    y_grid = torch.linspace(0, H - 1, H, device=pred_y.device)  # [H]

    # Compute soft-argmax to get predicted coordinates
    x_pred = torch.sum(prob_x * x_grid, dim=-1)  # [B]
    y_pred = torch.sum(prob_y * y_grid, dim=-1)  # [B]

    # Stack predicted x and y coordinates into a [B, 2] tensor
    predicted_coords = torch.stack([x_pred, y_pred], dim=-1)  # Shape: [B, 2]
    if scale is not None:
        predicted_coords = predicted_coords * scale

    # Compute RMSE between predicted coordinates and ground truth labels
    rmse = torch.sqrt(torch.mean((predicted_coords - labels) ** 2))  # RMSE loss

    return rmse
    
def extract_coords_from_heatmap(output):
    """
    Extracts the (x, y) coordinates from the heatmap logits.

    Args:
        output (tuple): Tuple of (pred_x, pred_y), where:
                        - pred_x: [B, W] logits across width (x-axis).
                        - pred_y: [B, H] logits across height (y-axis).

    Returns:
        coords (tensor): Extracted coordinates of shape [B, 2] (x, y) for each sample.
    """

    pred_x, pred_y = output  # Unpack the tuple
    B, W = pred_x.shape  # Batch size and width
    _, H = pred_y.shape  # Height

    # Apply softmax to get probability distributions along each axis
    prob_x = torch.softmax(pred_x, dim=-1)  # [B, W]
    prob_y = torch.softmax(pred_y, dim=-1)  # [B, H]

    # Create coordinate grids for x and y axes
    x_grid = torch.linspace(0, W - 1, W, device=pred_x.device)  # [W]
    y_grid = torch.linspace(0, H - 1, H, device=pred_y.device)  # [H]

    # Compute soft-argmax to get predicted coordinates
    x_pred = torch.sum(prob_x * x_grid, dim=-1)  # [B]
    y_pred = torch.sum(prob_y * y_grid, dim=-1)  # [B]

    # Stack predicted x and y coordinates into a [B, 2] tensor
    coords = torch.stack([x_pred, y_pred], dim=-1)  # Shape: [B, 2]

    return coords

def gaussian_1d(pos, muy, sigma):
    """Create 1D Gaussian distribution based on ball position (muy), and std (sigma)"""
    target = torch.exp(- (((pos - muy) / sigma) ** 2) / 2)
    return target

def create_target_ball_right(ball_position_xy, sigma, w, h, thresh_mask, device):
    """Create target for the ball detection stages

    :param ball_position_xy: Position of the ball (x,y)
    :param sigma: standard deviation (a hyperparameter)
    :param w: width of the resize image
    :param h: height of the resize image
    :param thresh_mask: if values of 1D Gaussian < thresh_mask --> set to 0 to reduce computation
    :param device: cuda() or cpu()
    :return:
    """
    w, h = int(w), int(h)
    target_ball_position_x = torch.zeros(w, device=device)
    target_ball_position_y = torch.zeros(h, device=device)
    # Only do the next step if the ball is existed
    if (w > ball_position_xy[0] > 0) and (h > ball_position_xy[1] > 0):
        # For x
        x_pos = torch.arange(0, w, device=device)
        target_ball_position_x = gaussian_1d(x_pos, ball_position_xy[0], sigma=sigma)
        # For y
        y_pos = torch.arange(0, h, device=device)
        target_ball_position_y = gaussian_1d(y_pos, ball_position_xy[1], sigma=sigma)

        target_ball_position_x[target_ball_position_x < thresh_mask] = 0.
        target_ball_position_y[target_ball_position_y < thresh_mask] = 0.

    return target_ball_position_x, target_ball_position_y