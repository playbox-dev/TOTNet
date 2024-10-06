import torch.nn as nn
import torch


class Ball_Detection_Loss(nn.Module):
    def __init__(self):
        super(Ball_Detection_Loss, self).__init__()
        # Choose between MSELoss or L1Loss (or any other appropriate loss function)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, coord_logits, target_ball_position):
        """
        Args:
        - pred_ball_position: [B, 2] where 2 represents (x, y) predicted coordinates
        - target_ball_position: [B, 2] where 2 represents (x, y) true coordinates (integer)
        """

        loss_x = self.loss_fn(coord_logits[:, 0], target_ball_position[:, 0])
        loss_y = self.loss_fn(coord_logits[:, 1], target_ball_position[:, 1])

        total_loss = loss_x + loss_y

        return total_loss