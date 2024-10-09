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