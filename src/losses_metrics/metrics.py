import torch

def heatmap_calculate_metrics(pred_logits, target_coords, img_width, img_height):
    """
    Calculates evaluation metrics between predicted logits for x and y coordinates
    and target pixel coordinates (integers).
    
    Args:
    - pred_logits: tuple of tensors (pred_x_logits, pred_y_logits)
        - pred_x_logits: Tensor of shape [B, W] with predicted logits for x-axis
        - pred_y_logits: Tensor of shape [B, H] with predicted logits for y-axis
    - target_coords: Tensor of shape [B, 2] with ground truth (x, y) coordinates (integers)
    - img_width: Image width in pixels
    - img_height: Image height in pixels
    
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