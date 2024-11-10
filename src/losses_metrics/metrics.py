import torch


def extract_coords(pred_heatmap):
    """_summary_

    Args:
        pred_heatmap : tuple of tensors (pred_x_logits, pred_y_logits)
        - pred_x_logits: Tensor of shape [B, W] with predicted logits for x-axis
        - pred_y_logits: Tensor of shape [B, H] with predicted logits for y-axis
    Return:
        out (tensor) : Tensor in shape [B,2] which represents coords for each 
    """
    pred_x_logits, pred_y_logits = pred_heatmap

    # Predicted coordinates are extracted by taking the argmax over logits
    x_pred_indices = torch.argmax(pred_x_logits, dim=1)  # [B]
    y_pred_indices = torch.argmax(pred_y_logits, dim=1)  # [B]

    # Convert indices to float for calculations
    x_pred = x_pred_indices.float()
    y_pred = y_pred_indices.float()

    # Stack the predicted x and y coordinates
    pred_coords = torch.stack([x_pred, y_pred], dim=1)  # [B, 2]

    return pred_coords

def heatmap_calculate_metrics(pred_logits, target_coords, scale=None):
    """
    Calculates evaluation metrics between predicted logits for x and y coordinates
    and target pixel coordinates (integers).
    
    Args:
    - pred_logits: tuple of tensors (pred_x_logits, pred_y_logits)
        - pred_x_logits: Tensor of shape [B, W] with predicted logits for x-axis
        - pred_y_logits: Tensor of shape [B, H] with predicted logits for y-axis
    - target_coords: Tensor of shape [B, 2] with ground truth (x, y) coordinates (integers)
    - scale: Tensor
    
    Returns:
    - mse: Mean Squared Error
    - rmse: Root Mean Squared Error
    - mae: Mean Absolute Error
    - euclidean_distance: Average Euclidean distance between predictions and ground truth
    """
    pred_x_logits, pred_y_logits = pred_logits

    # Ensure tensors are on the same device
    device = pred_x_logits.device
    target_coords = target_coords.to(device)

    # Predicted coordinates are extracted by taking the argmax over logits
    x_pred_indices = torch.argmax(pred_x_logits, dim=1)  # [B]
    y_pred_indices = torch.argmax(pred_y_logits, dim=1)  # [B]

    # Convert indices to float for calculations
    x_pred = x_pred_indices.float()
    y_pred = y_pred_indices.float()

    # Stack the predicted x and y coordinates
    pred_coords = torch.stack([x_pred, y_pred], dim=1)  # [B, 2]

    if scale is not None:
        pred_coords = pred_coords * scale

    # Convert target coordinates to float
    target_coords = target_coords.float()

    # Difference between predicted and ground truth coordinates
    diff = pred_coords - target_coords  # [B, 2]

    # Mean Squared Error (MSE) per sample
    mse_per_sample = torch.mean(diff ** 2, dim=1)  # [B]

    # Mean MSE over all samples
    mse = mse_per_sample.mean()

    # Root Mean Squared Error (RMSE) per sample
    rmse_per_sample = torch.sqrt(mse_per_sample)  # [B]

    # Mean RMSE over all samples
    rmse = rmse_per_sample.mean()

    # Mean Absolute Error (MAE) per sample
    mae_per_sample = torch.mean(torch.abs(diff), dim=1)  # [B]

    # Mean MAE over all samples
    mae = mae_per_sample.mean()

    # Euclidean distance per sample
    euclidean_distance_per_sample = torch.norm(diff, dim=1)  # [B]

    # Mean Euclidean distance over all samples
    euclidean_distance = euclidean_distance_per_sample.mean()

    return mse.item(), rmse.item(), mae.item(), euclidean_distance.item()

def precision_recall_f1(pred_heatmap, target_coords, threshold=0.5):
    """
    Calculates precision, recall, and F1 score for a tracking model with separate x and y heatmaps.
    
    Args:
    - pred_heatmap: Tuple of tensors (x_heatmap, y_heatmap) of shapes [B, W] and [B, H] respectively.
    - target_coords: Tensor of shape [B, 2] with ground truth (x, y) coordinates.
    - threshold: Float, threshold to binarize predictions.
    
    Returns:
    - precision: Float, precision across the batch.
    - recall: Float, recall across the batch.
    - f1_score: Float, F1 score across the batch.
    """
    x_heatmap, y_heatmap = pred_heatmap
    B, W = x_heatmap.shape
    _, H = y_heatmap.shape
    device = x_heatmap.device

    # Convert target coordinates to one-hot encoded x and y heatmaps
    x_target = torch.zeros_like(x_heatmap, device=device)
    y_target = torch.zeros_like(y_heatmap, device=device)
    
    # Extract x and y coordinates from target_coords
    x_coords = target_coords[:, 0].long()  # x-coordinates
    y_coords = target_coords[:, 1].long()  # y-coordinates

    # Create one-hot encoding for target positions
    x_target[torch.arange(B), x_coords] = 1
    y_target[torch.arange(B), y_coords] = 1

    # Binarize predicted heatmaps based on the threshold
    x_pred_binary = (x_heatmap >= threshold).float()
    y_pred_binary = (y_heatmap >= threshold).float()

    # Calculate true positives, false positives, and false negatives for x and y dimensions
    true_positives_x = (x_pred_binary * x_target).sum(dim=1)
    predicted_positives_x = x_pred_binary.sum(dim=1)
    actual_positives_x = x_target.sum(dim=1)

    true_positives_y = (y_pred_binary * y_target).sum(dim=1)
    predicted_positives_y = y_pred_binary.sum(dim=1)
    actual_positives_y = y_target.sum(dim=1)

    # Calculate precision, recall, and F1 score for x and y axes separately
    precision_x = true_positives_x / (predicted_positives_x + 1e-8)
    recall_x = true_positives_x / (actual_positives_x + 1e-8)
    f1_score_x = 2 * (precision_x * recall_x) / (precision_x + recall_x + 1e-8)

    precision_y = true_positives_y / (predicted_positives_y + 1e-8)
    recall_y = true_positives_y / (actual_positives_y + 1e-8)
    f1_score_y = 2 * (precision_y * recall_y) / (precision_y + recall_y + 1e-8)

    # Average metrics across the batch and axes
    precision = (precision_x + precision_y).mean().item() / 2
    recall = (recall_x + recall_y).mean().item() / 2
    f1_score = (f1_score_x + f1_score_y).mean().item() / 2

    return precision, recall, f1_score


def precision_recall_f1_tracknet(pred_coords, target_coords, distance_threshold=5):
    """
    Calculates precision, recall, and F1 score for TrackNet-style tracking,
    where a detection is counted as true positive if within a certain distance
    threshold from the ground truth.
    
    Args:
    - pred_coords: Tensor of shape [B, 2] with predicted (x, y) coordinates.
    - target_coords: Tensor of shape [B, 2] with ground truth (x, y) coordinates.
    - distance_threshold: Float, maximum allowed distance for a detection to be considered a true positive.
    
    Returns:
    - precision: Float, precision across the batch.
    - recall: Float, recall across the batch.
    - f1_score: Float, F1 score across the batch.
    """
    device = pred_coords.device

    # Calculate Euclidean distances between predicted and target coordinates
    distances = torch.norm(pred_coords - target_coords, dim=1)  # Shape: [B]

    # Determine true positives, false positives, and false negatives
    tp = (distances <= distance_threshold).sum().float()  # True positives
    fp = (distances > distance_threshold).sum().float()   # False positives
    fn = (target_coords.shape[0] - tp).float()            # False negatives

    # Precision, recall, and F1 score calculations
    precision = tp / (tp + fp + 1e-8)  # Adding epsilon to avoid division by zero
    recall = tp / (tp + fn + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision.item(), recall.item(), f1_score.item()

def heatmap_calculate_metrics_2d(pred_heatmap, target_coords, scale=None):
    """
    Calculates evaluation metrics between predicted heatmap and target pixel coordinates.

    Args:
    - pred_heatmap: Tensor of shape [B, H, W] with predicted heatmap values
    - target_coords: Tensor of shape [B, 2] with ground truth (x, y) coordinates (integers)
    - scale: Tensor (optional), scaling factor for predicted coordinates

    Returns:
    - mse: Mean Squared Error
    - rmse: Root Mean Squared Error
    - mae: Mean Absolute Error
    - euclidean_distance: Average Euclidean distance between predictions and ground truth
    """
    # Ensure tensors are on the same device
    device = pred_heatmap.device
    target_coords = target_coords.to(device)

    # Get height and width from the heatmap shape
    B, H, W = pred_heatmap.shape

    # Flatten the heatmap to find the index of the maximum value
    pred_flat = pred_heatmap.view(B, -1)  # Shape: [B, H*W]
    max_indices = torch.argmax(pred_flat, dim=1)  # Shape: [B]

    # Convert flat indices to 2D coordinates (y, x)
    y_pred = (max_indices // W).float()  # Divide by width to get y-coordinate
    x_pred = (max_indices % W).float()   # Modulus by width to get x-coordinate

    # Stack the predicted x and y coordinates
    pred_coords = torch.stack([x_pred, y_pred], dim=1)  # Shape: [B, 2]

    # Apply scaling if scale tensor is provided
    if scale is not None:
        pred_coords = pred_coords * scale

    # Convert target coordinates to float
    target_coords = target_coords.float()

    # Difference between predicted and ground truth coordinates
    diff = pred_coords - target_coords  # Shape: [B, 2]

    # Mean Squared Error (MSE) per sample
    mse_per_sample = torch.mean(diff ** 2, dim=1)  # Shape: [B]
    mse = mse_per_sample.mean()  # Mean MSE over all samples

    # Root Mean Squared Error (RMSE) per sample
    rmse_per_sample = torch.sqrt(mse_per_sample)  # Shape: [B]
    rmse = rmse_per_sample.mean()  # Mean RMSE over all samples

    # Mean Absolute Error (MAE) per sample
    mae_per_sample = torch.mean(torch.abs(diff), dim=1)  # Shape: [B]
    mae = mae_per_sample.mean()  # Mean MAE over all samples

    # Euclidean distance per sample
    euclidean_distance_per_sample = torch.norm(diff, dim=1)  # Shape: [B]
    euclidean_distance = euclidean_distance_per_sample.mean()  # Mean Euclidean distance over all samples

    return mse.item(), rmse.item(), mae.item(), euclidean_distance.item()

# Example code to calculate RMSE for a single sample
def calculate_rmse(original_x, original_y, rescaled_x_pred, rescaled_y_pred):
    # Calculate the squared differences
    x_diff = (rescaled_x_pred - original_x) ** 2
    y_diff = (rescaled_y_pred - original_y) ** 2

    # Sum the squared differences
    squared_error = x_diff + y_diff

    # Take the square root to compute RMSE
    rmse = torch.sqrt(squared_error)

    return rmse.item()

def calculate_rmse_batched(pred_coords, label_coords):
    """
    Calculates the RMSE between predicted coordinates and ground truth labels for a batch.

    Args:
        pred_coords (tensor): Predicted coordinates of shape [B, 2].
        label_coords (tensor): Ground truth coordinates of shape [B, 2].

    Returns:
        rmse (float): Mean RMSE across the batch.
    """

    # Ensure both tensors are on the same device
    pred_coords = pred_coords.to(label_coords.device)

    # Calculate the squared differences for x and y
    squared_diff = (pred_coords - label_coords) ** 2  # [B, 2]

    # Sum the squared differences along the coordinate axis (x and y)
    sum_squared_diff = torch.sum(squared_diff, dim=-1)  # [B]

    # Take the square root to get RMSE for each sample
    rmse_per_sample = torch.sqrt(sum_squared_diff)  # [B]

    # Calculate the mean RMSE across the batch
    mean_rmse = torch.mean(rmse_per_sample)

    return mean_rmse.item()