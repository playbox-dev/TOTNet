import torch.nn as nn
import torch
import torch.nn.functional as F


class Heatmap_Ball_Detection_Loss_Weighted(nn.Module):
    def __init__(self, weighted_list=[1, 2, 2, 3], sigma=0.3):
        """
        Args:
        - weighted_list: List of weights corresponding to the four visibility classes.
        """
        super(Heatmap_Ball_Detection_Loss_Weighted, self).__init__()
        self.loss = nn.BCELoss(reduction="none")  # Avoid reduction for per-sample weighting
        self.weighted_list = weighted_list
        self.sigma = sigma

    def forward(self, output, target_ball_position, visibility):
        """
        Args:
        - output: tuple of (pred_x, pred_y)
            - pred_x: [B, W] predicted logits across the width (x-axis)
            - pred_y: [B, H] predicted logits across the height (y-axis)
        - target_ball_position: [B, 2] true (x, y) integer pixel coordinates of the ball
        - visibility: [B] visibility labels for the ball (0 to 3 corresponding to visibility classes).
        """
        # Correctly unpack the output logits
        pred_x, pred_y = output

        # Ensure target positions are of type LongTensor and on the same device
        device = pred_x.device
        target_x = target_ball_position[:, 0].long().to(device)  # [B]
        target_y = target_ball_position[:, 1].long().to(device)  # [B]

        # Clamp the indices to valid ranges
        target_x = torch.clamp(target_x, 0, pred_x.shape[1] - 1)
        target_y = torch.clamp(target_y, 0, pred_y.shape[1] - 1)

        gaussian_mask = (visibility == 3)

        # Gaussian-based target generation
        if gaussian_mask.any():
            x_coords = torch.arange(pred_x.shape[1], device=device).float()  # Width
            y_coords = torch.arange(pred_y.shape[1], device=device).float()  # Height

            target_x_map_gaussian = torch.exp(
                -((x_coords.unsqueeze(0).unsqueeze(0) - target_x[gaussian_mask].unsqueeze(1).unsqueeze(2)) ** 2)
                / (2 * self.sigma ** 2)
            )  # Shape: [masked_B, 1, W]

            target_y_map_gaussian = torch.exp(
                -((y_coords.unsqueeze(0).unsqueeze(0) - target_y[gaussian_mask].unsqueeze(1).unsqueeze(2)) ** 2)
                / (2 * self.sigma ** 2)
            )  # Shape: [masked_B, 1, H]

       
        # For x-axis predictions
        target_x_map = torch.zeros_like(pred_x)  # [B, W]
        target_x_map.scatter_(1, target_x.unsqueeze(1), 1.0)

        # For y-axis predictions
        target_y_map = torch.zeros_like(pred_y)  # [B, H]
        target_y_map.scatter_(1, target_y.unsqueeze(1), 1.0)

        # Replace targets with Gaussian maps where applicable
        if gaussian_mask.any():
            target_x_map[gaussian_mask] = target_x_map_gaussian.squeeze(1)  # Replace for Gaussian targets
            target_y_map[gaussian_mask] = target_y_map_gaussian.squeeze(1)

            # Normalize each map to sum to 1
            target_x_map[gaussian_mask] /= target_x_map[gaussian_mask].sum(dim=-1, keepdim=True)
            target_y_map[gaussian_mask] /= target_y_map[gaussian_mask].sum(dim=-1, keepdim=True)

        # Compute binary cross-entropy loss for x and y without reduction
        loss_x = self.loss(pred_x, target_x_map).sum(dim=1)  # Sum across width [B]
        loss_y = self.loss(pred_y, target_y_map).sum(dim=1)  # Sum across height [B]

        # Compute per-sample weights based on visibility classes
        visibility_weights = torch.tensor(self.weighted_list, device=device)  # [4]
        sample_weights = visibility_weights[visibility]  # Map visibility to weights [B]

        # Apply weights to the losses
        weighted_loss_x = (loss_x * sample_weights).mean()  # Average across batch
        weighted_loss_y = (loss_y * sample_weights).mean()  # Average across batch

        # Return the combined weighted loss
        return weighted_loss_x + weighted_loss_y



class Heatmap_Ball_Detection_Loss(nn.Module):
    def __init__(self):
        super(Heatmap_Ball_Detection_Loss, self).__init__()
        self.loss = nn.BCELoss() # Use BCEWithLogitsLoss for logits

    def forward(self, output, target_ball_position, visibility):
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
        loss_x = self.loss(pred_x, target_x_one_hot)
        loss_y = self.loss(pred_y, target_y_one_hot)

        # Return the combined loss
        return loss_x + loss_y

class Heatmap_Ball_Detection_Loss_Gaussian(nn.Module):
    def __init__(self, sigma=0.25, weighted_list=[1,2,3,3]):

        super(Heatmap_Ball_Detection_Loss_Gaussian, self).__init__()
        self.loss = nn.BCELoss(reduction='none') # More stable for logits
        self.sigma = sigma  # Standard deviation for Gaussian spread
        self.weighted_list = weighted_list

    def forward(self, output, target_ball_position, visibility):
        """
        Args:
        - output: tuple of (pred_x, pred_y)
            - pred_x: [B, W] predicted logits across the width (x-axis)
            - pred_y: [B, H] predicted logits across the height (y-axis)
        - target_ball_position: [B, 2] true (x, y) integer pixel coordinates of the ball
        """
        pred_x, pred_y = output
        device = pred_x.device
        target_x = target_ball_position[:, 0].long().to(device)
        target_y = target_ball_position[:, 1].long().to(device)

        # Clamp to valid ranges
        target_x = torch.clamp(target_x, 0, pred_x.shape[1] - 1)
        target_y = torch.clamp(target_y, 0, pred_y.shape[1] - 1)

        # Create coordinate ranges for Gaussian distribution
        x_coords = torch.arange(pred_x.shape[1], device=device).float()  # Width
        y_coords = torch.arange(pred_y.shape[1], device=device).float()  # Height

        # Generate Gaussian distributions centered at the ground truth positions
        target_x_gaussian = torch.exp(-((x_coords.unsqueeze(0) - target_x.unsqueeze(1))**2) / (2 * self.sigma**2))
        target_y_gaussian = torch.exp(-((y_coords.unsqueeze(0) - target_y.unsqueeze(1))**2) / (2 * self.sigma**2))

        # Normalize to sum to 1
        target_x_gaussian /= target_x_gaussian.sum(dim=1, keepdim=True)
        target_y_gaussian /= target_y_gaussian.sum(dim=1, keepdim=True)

        #Compute per-pixel BCE loss
        loss_x = self.loss(pred_x, target_x_gaussian)  # [B, W]
        loss_y = self.loss(pred_y, target_y_gaussian)  # [B, H]

        # Sum losses across the width and height
        loss_x = loss_x.sum(dim=1)  # [B]
        loss_y = loss_y.sum(dim=1)  # [B]

        # Create mask based on visibility and weights
        visibility_weights = torch.tensor(self.weighted_list, device=device)  # [num_visibility_states]
        mask = visibility_weights[visibility]  # [B]

        # Apply mask to the losses
        weighted_loss_x = (loss_x * mask).mean()
        weighted_loss_y = (loss_y * mask).mean()

    
        return weighted_loss_x + weighted_loss_y

class Heatmap_Ball_Detection_Loss_2D(nn.Module):
    def __init__(self, h, w, sigma=2.0):
        super(Heatmap_Ball_Detection_Loss_2D, self).__init__()
        self.h = h  # Image height
        self.w = w  # Image width
        self.sigma = sigma
        self.bce_loss = nn.BCELoss()  # Use BCEWithLogitsLoss if your output is logits

    def forward(self, output, target_ball_position):
        """
        Args:
        - output: [B, H, W] predicted heatmap with probabilities for each pixel
        - target_ball_position: [B, 2] true (x, y) integer pixel coordinates of the ball
        """
        device = output.device
        batch_size = output.size(0)

        # Ensure sigma is a tensor
        sigma_tensor = torch.tensor(self.sigma, dtype=torch.float32, device=device)

        # Create coordinate grids
        y_coords = torch.arange(self.h, device=device).view(1, self.h, 1).expand(batch_size, self.h, self.w)
        x_coords = torch.arange(self.w, device=device).view(1, 1, self.w).expand(batch_size, self.h, self.w)

        # Extract target positions and reshape for broadcasting
        x_targets = target_ball_position[:, 0].view(batch_size, 1, 1).float()
        y_targets = target_ball_position[:, 1].view(batch_size, 1, 1).float()

        # Compute squared distances
        squared_distances = ((x_coords - x_targets) ** 2 + (y_coords - y_targets) ** 2)

        # Compute Gaussian heatmaps
        target_heatmap = torch.exp(-squared_distances / (2 * sigma_tensor ** 2))

        # Compute binary cross-entropy loss between the predicted and target heatmaps
        loss = self.bce_loss(output, target_heatmap)

        return loss


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


def events_spotting_loss(pred_events, target_events, weights=(1, 3), epsilon=1e-9):
    """
    Weighted binary cross-entropy loss for event spotting.

    Args:
        pred_events (torch.Tensor): Predicted probabilities, shape [B, num_events].
        target_events (torch.Tensor): Ground truth labels, shape [B, num_events].
        weights (tuple): Weights for the event classes, e.g., (1, 3).
        epsilon (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Loss value (scalar).
    """
    # Convert weights to a tensor and normalize
    weights = torch.tensor(weights, dtype=torch.float32).view(1, -1)
    weights = weights / weights.sum()
    
    # Move weights to the same device as predictions
    weights = weights.to(pred_events.device)
    
    # Compute the weighted binary cross-entropy loss
    loss = -torch.mean(
        weights * (
            target_events * torch.log(pred_events + epsilon) +
            (1.0 - target_events) * torch.log(1.0 - pred_events + epsilon)
        )
    )
    
    return loss

def generate_gaussian_map(width, target_x, sigma=0.5):
    """
    Generate a 1D Gaussian map.

    Args:
        width (int): The width of the map (number of points).
        target_x (int): The center position of the Gaussian.
        sigma (float): The standard deviation of the Gaussian.

    Returns:
        torch.Tensor: A 1D Gaussian map.
    """
    x_coords = torch.arange(width).float()  # 1D coordinate space
    gaussian_map = torch.exp(-((x_coords - target_x)**2) / (2 * sigma**2))  # Gaussian formula
    normalized_gaussian_map = gaussian_map / gaussian_map.sum()  # Normalize to sum to 1
    return normalized_gaussian_map


# Example usage
def probability_loss(pred_probs, true_probs):
    """
    Compute KL Divergence Loss.
    Args:
        pred_probs: Model output probabilities (after softmax) [B, C]
        true_probs: Ground truth probabilities [B, C]
    Returns:
        KL divergence loss.
    """
    pred_probs = pred_probs + 1e-12
    return F.kl_div(pred_probs.log(), true_probs, reduction='batchmean')



def focal_loss(pred_logits, labels, alpha=1.0, gamma=2.0, num_classes=3):
    """
    Focal loss for classification tasks.

    Args:
        pred_logits (tensor): Logits in shape [B, num_classes].
        labels (tensor): Ground truth labels in shape [B].
        alpha (float, optional): Balancing factor for the loss. Defaults to 1.0.
        gamma (float, optional): Focusing parameter to down-weight easy examples. Defaults to 2.0.
        num_classes (int): Number of classes in the classification task.

    Returns:
        torch.Tensor: Scalar loss value (mean over the batch).
    """
    # Convert logits to probabilities
    pred_probs = F.softmax(pred_logits, dim=-1)  # [B, num_classes]
    
    # Ensure labels are 1D
    labels = labels.squeeze(-1) if labels.dim() > 1 else labels  # [B]
    
    # One-hot encode the labels
    labels_one_hot = F.one_hot(labels.long(), num_classes=num_classes).float()  # [B, num_classes]
    
    # Compute probabilities of true classes
    pt = (pred_probs * labels_one_hot).sum(dim=1)  # [B]
    
    # Clamp pt for numerical stability
    pt = torch.clamp(pt, min=1e-7, max=1.0)  # Avoid log(0)
    
    # Compute focal loss
    loss = -alpha * (1 - pt) ** gamma * torch.log(pt)
    
    # Return mean loss over the batch
    return loss.mean()

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


if __name__ == "__main__":
    # Initialize the loss function
    # loss_func = Heatmap_Ball_Detection_Loss_Weighted(weighted_list=[1, 2, 3, 3])

    # # Heatmap dimensions
    # width = 512
    # height = 288

    # # Generate a target in the middle of the heatmap
    # middle_target = torch.tensor([[300, 156]])  # Example ground truth

    # # Generate a target at (0, 0)
    # out_of_frame_target = torch.tensor([[0, 0]])  # Example for out-of-frame

    # # Create coordinate ranges
    # x_coords = torch.arange(width).float()
    # y_coords = torch.arange(height).float()

    # # Generate perfect Gaussian heatmaps
    # middle_x = middle_target[0, 0].float()
    # middle_y = middle_target[0, 1].float()

    # # Gaussian centered at middle_target
    # heat_map_x_perfect = torch.exp(-((x_coords - middle_x) ** 2) / (2 * (2 ** 2)))  # Sigma=2
    # heat_map_y_perfect = torch.exp(-((y_coords - middle_y) ** 2) / (2 * (2 ** 2)))  # Sigma=2

    # # Normalize heatmaps
    # heat_map_x_perfect /= heat_map_x_perfect.sum()
    # heat_map_y_perfect /= heat_map_y_perfect.sum()

    # # Reshape for batch dimension
    # heat_map_x_perfect = heat_map_x_perfect.unsqueeze(0)  # [1, W]
    # heat_map_y_perfect = heat_map_y_perfect.unsqueeze(0)  # [1, H]

    # # Visibility labels (e.g., 0: visible, 1: partially visible, 2: occluded, 3: gaussian-based)
    # visibility_values = [1]  # Example visibility values for the batch
    # visibility = torch.tensor(visibility_values).unsqueeze(1)  # Shape [B, 1]

    # # Compute the loss for a perfect match
    # loss_perfect = loss_func((heat_map_x_perfect, heat_map_y_perfect), middle_target, visibility)
    # print(f"Loss for perfect match: {loss_perfect.item()}")

    # # Compute the loss for a perfect match
    # loss_perfect = loss_func((heat_map_x_perfect, heat_map_y_perfect), middle_target, 2)
    # print(f"Loss for perfect match: {loss_perfect.item()}")

    # # Simulate predicted heatmaps and normalize using softmax
    # heat_map_x = torch.softmax(torch.randn([1, width]), dim=-1)  # Predicted x-axis probabilities
    # heat_map_y = torch.softmax(torch.randn([1, height]), dim=-1)  # Predicted y-axis probabilities

    # # Compute the loss for the middle target with noisy predictions
    # loss_middle = loss_func((heat_map_x, heat_map_y), middle_target, 1)
    # print(f"Loss for middle target with noisy predictions: {loss_middle.item()}")

    # # Compute the loss for the out-of-frame target
    # loss_out_of_frame = loss_func((heat_map_x, heat_map_y), out_of_frame_target, 0)
    # print(f"Loss for out-of-frame target: {loss_out_of_frame.item()}")

    print(generate_gaussian_map(width=288, target_x=50, sigma=0.45))










