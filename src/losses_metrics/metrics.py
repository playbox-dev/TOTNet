import torch
import numpy as np
import csv
import os
from sklearn.metrics import precision_score, recall_score, f1_score

import csv
import os
from sklearn.metrics import precision_score, recall_score, f1_score

def bounce_metrics(ball_positions, csv_file_path, frame_tolerance=5):
    """
    Calculate bounce detection metrics (precision, recall, F1-score) based on predicted and actual bounce frames.

    Args:
        ball_positions (list): List of tuples, where each tuple is (frame_id, (x, y)) representing
                               the frame ID and ball coordinates.
        csv_file_path (str): Path to the CSV file containing ground truth bounce information.
        frame_tolerance (int): The allowed tolerance (in frames) to consider a bounce as correct.

    Returns:
        dict: A dictionary containing precision, recall, and F1-score.
    """
    # Get all predicted frame IDs from ball_positions
    predict_frame_list = [frame for frame, _ in ball_positions]

    # Get all actual bounce frame IDs from the CSV file
    actual_frame_list = []
    with open(csv_file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)  # Use DictReader to load as a list of dictionaries
        for row in csv_reader:
            file_name = row['img']
            image_name = os.path.basename(file_name)  # Get image name
            ball_frameidx = int(image_name[4:10])  # Extract the frame index from the image name
            status_mapping = {'Empty': 0, 'Bounce': 1}
            status = status_mapping.get(row.get('event-type'), 2)
            if status == 1:
                actual_frame_list.append(ball_frameidx)

    # Match predicted frames to actual frames within tolerance
    matched_frames = set()
    for predicted_frame in predict_frame_list:
        for actual_frame in actual_frame_list:
            if abs(predicted_frame - actual_frame) <= frame_tolerance:
                matched_frames.add(actual_frame)
                break

    # Calculate metrics
    y_true = [1 if frame in actual_frame_list else 0 for frame in range(max(max(predict_frame_list, default=0), max(actual_frame_list, default=0)) + 1)]
    y_pred = [1 if frame in predict_frame_list else 0 for frame in range(len(y_true))]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }



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

    # Calculate Euclidean distances between predicted and target coordinates
    distances = torch.norm(pred_coords - target_coords, dim=1)  # Shape: [B]

    # Determine true positives, false positives, false negatives, and true negatives
    tp = ((distances <= distance_threshold) & (target_coords != (0, 0))).sum().float()  # True positives: ball is within the threshold and ball is in the frame
    fp = ((distances > distance_threshold) & (target_coords != (0, 0))).sum().float()   # False positives: ball is in the frame but prediction is not correct
    fn = ((target_coords != (0, 0)) & (pred_coords == (0, 0)))          # False negatives: ball is in the frame, but prediction says it’s not
    tn = ((target_coords == (0, 0)) & (pred_coords == (0, 0)))          # True negatives: ball is not in the frame, and prediction correctly says so

    # Precision, recall, and F1 score calculations
    precision = tp / (tp + fp + 1e-8)  # Adding epsilon to avoid division by zero
    recall = tp / (tp + fn + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Accuracy calculation
    accuracy = tp / (tp + fp + fn + 1e-8)

    return precision.item(), recall.item(), f1_score.item(), accuracy.item()

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


def classification_metrics(preds, labels, num_classes=4):
    """
    Calculate accuracy, precision, recall, and F1-score for classification.

    Args:
        preds (torch.Tensor): Predictions of shape [B, 3] (logits or probabilities).
        labels (torch.Tensor): Ground truth of shape [B].
        num_classes (int): Number of classes.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1-score.
    """
    # Get predicted classes
    _, predicted_classes = torch.max(preds, dim=1)  # [B]
    labels = labels.squeeze(-1) if labels.dim() > 1 else labels  # [B]

    # Calculate accuracy
    correct = (predicted_classes == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total

    # Confusion matrix
    confusion_matrix = torch.zeros((num_classes, num_classes), device=preds.device)
    for t, p in zip(labels, predicted_classes):
        confusion_matrix[t.long(), p.long()] += 1

    # True positives, false positives, false negatives
    true_positives = torch.diag(confusion_matrix)
    false_positives = confusion_matrix.sum(dim=0) - true_positives
    false_negatives = confusion_matrix.sum(dim=1) - true_positives

    # Precision, recall, F1-score
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Macro averages
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()

    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1
    }


def post_process_event_prediction(preds):
    _, predicted_classes = torch.max(preds, dim=1)  # [B]

    return predicted_classes



def PCE(sample_prediction_events, sample_target_events):
    """
    Percentage of Correct Events for PyTorch tensors.

    :param sample_prediction_events: Predicted events, tensor of shape [2,]
    :param sample_target_events: Ground truth events, tensor of shape [2,]
    :return: Integer (1 for correct, 0 for incorrect)
    """
    # Threshold predictions and targets
    sample_prediction_events = (sample_prediction_events >= 0.5).float()
    sample_target_events = (sample_target_events >= 0.5).float()
    
    # Compute the difference
    diff = sample_prediction_events - sample_target_events
    
    # Check if all values are correct
    if torch.sum(diff) != 0:  # Incorrect
        ret_pce = 0
    else:  # Correct
        ret_pce = 1
    return ret_pce


def SPCE(sample_prediction_events, sample_target_events, thresh=0.25):
    """
    Smooth Percentage of Correct Events for PyTorch tensors.

    :param sample_prediction_events: Predicted events, tensor of shape [2,]
    :param sample_target_events: Ground truth events, tensor of shape [2,]
    :param thresh: Threshold for the difference between prediction and ground truth
    :return: Integer (1 for correct, 0 for incorrect)
    """
    # Compute the absolute difference
    diff = torch.abs(sample_prediction_events - sample_target_events)
    
    # Check if all differences are within the threshold
    if torch.sum(diff > thresh) > 0:  # Incorrect
        ret_spce = 0
    else:  # Correct
        ret_spce = 1
    return ret_spce


def batch_PCE(batch_prediction_events, batch_target_events):
    """
    Batch Percentage of Correct Events (PCE) using PyTorch tensors.
    
    :param batch_prediction_events: Batch of predictions, size: (B, N)
    :param batch_target_events: Batch of ground truths, size: (B, N)
    :return: Tensor of PCE values for each sample in the batch, size: (B,)
    """
    # Threshold predictions and targets
    batch_prediction_events = (batch_prediction_events >= 0.5).float()
    batch_target_events = (batch_target_events >= 0.5).float()
    
    # Compute difference
    diff = batch_prediction_events - batch_target_events  # Shape: (B, N)
    
    # Check correctness for each sample
    batch_pce = (diff.abs().sum(dim=1) == 0).float()  # 1 if correct, 0 if incorrect
    
    return batch_pce.mean()


def batch_SPCE(batch_prediction_events, batch_target_events, thresh=0.25):
    """
    Batch Smooth Percentage of Correct Events (SPCE) using PyTorch tensors.
    
    :param batch_prediction_events: Batch of predictions, size: (B, N)
    :param batch_target_events: Batch of ground truths, size: (B, N)
    :param thresh: Threshold for the difference between prediction and ground truth.
    :return: Tensor of SPCE values for each sample in the batch, size: (B,)
    """
    # Compute absolute difference
    diff = torch.abs(batch_prediction_events - batch_target_events)  # Shape: (B, N)
    
    # Check if all differences are within the threshold for each sample
    batch_spce = (diff <= thresh).all(dim=1).float()  # 1 if all within threshold, 0 otherwise
    
    return batch_spce.mean()