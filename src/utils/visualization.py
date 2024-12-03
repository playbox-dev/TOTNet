import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import cv2

def visualize_and_save_2d_heatmap(output_heatmap, save_dir='src/results/visualization', figsize=(20, 20)):
    """
    Visualizes the heatmap output from the model as a 2D heatmap image and saves to a specified directory.
    
    Args:
    - output_heatmap: tuple of (pred_x, pred_y), each of shape ([B, W], [B, H])
    - save_dir: string, directory path to save the images.
    - figsize: tuple, size of the plot (default: (6, 6))
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    pred_x, pred_y = output_heatmap  # Unpack the outputs
    batch_size = pred_x.shape[0]

    for i in range(batch_size):
        heatmap_x = pred_x[i].cpu().detach().numpy()  # [W]
        heatmap_y = pred_y[i].cpu().detach().numpy()  # [H]

        # Create a 2D heatmap by taking the outer product of heatmap_x and heatmap_y
        heatmap_2d = np.outer(heatmap_y, heatmap_x)

        # Get the predicted coordinates (indices of max probability)
        pred_x_index = np.argmax(heatmap_x)
        pred_y_index = np.argmax(heatmap_y)

        # Plot the 2D heatmap   
        plt.figure(figsize=figsize)
        plt.imshow(heatmap_2d, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Probability')
        plt.title(f'2D Heatmap Visualization for Sample {i+1}')
        plt.xlabel('Width')
        plt.ylabel('Height')

        # Highlight the predicted point with a red dot
        plt.scatter(pred_x_index, pred_y_index, color='red', s=200, label='Predicted Coord')
        plt.legend(loc='upper right')

        # Save the figure
        file_path = os.path.join(save_dir, f"heatmap_sample_{i+1}.png")
        plt.savefig(file_path)
        print(f"Saved heatmap for sample {i+1} to {file_path}")
        plt.close()  # Close the figure to free memory



def visualize_optical_flow(flow, max_flow=None):
    """
    Visualizes optical flow using HSV colormap.
    Args:
        flow: Optical flow tensor of shape [2, H, W] (dx, dy).
        max_flow: Maximum flow for normalization (if None, compute from flow).
    Returns:
        BGR image representing the optical flow.
    """
    # Convert to numpy
    flow = flow.detach().cpu().numpy()  # Ensure it's numpy
    dx, dy = flow[0], flow[1]  # Split flow into x and y components

    # Compute magnitude and angle
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)  # Angle in radians

    # Normalize magnitude
    if max_flow is None:
        max_flow = np.max(magnitude)
    magnitude = np.clip(magnitude / (max_flow + 1e-5), 0, 1)

    # Create HSV image
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.float32)
    hsv[..., 0] = (angle + np.pi) / (2 * np.pi)  # Hue: Angle normalized to [0, 1]
    hsv[..., 1] = magnitude  # Saturation: Magnitude normalized to [0, 1]
    hsv[..., 2] = 1.0  # Value: Set to 1 for brightness

    # Convert HSV to BGR for visualization
    bgr = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return bgr

def save_optical_flow_visualization(flow, save_path='src/results/visualization', max_flow=None):
    """
    Visualizes optical flow using HSV colormap and saves the image.
    Args:
        flow: Optical flow tensor of shape [2, H, W] (dx, dy).
        save_path: Path to save the visualized image (e.g., 'flow_visualization.png').
        max_flow: Maximum flow for normalization (if None, compute from flow).
    """
    # Convert flow to HSV visualization
    flow_viz = visualize_optical_flow(flow, max_flow=max_flow)

    # Save the visualized image
    cv2.imwrite(save_path, flow_viz)
    print(f"Optical flow visualization saved at: {save_path}")


def save_batch_optical_flow_visualization(flow_batch, save_dir='src/results/visualization', max_flow=None):
    """
    Visualizes and saves optical flow for a batch of flow tensors.
    Args:
        flow_batch: Optical flow tensor of shape [N, 2, H, W] (batch of dx, dy).
        save_dir: Directory to save the visualized images.
        max_flow: Maximum flow for normalization (if None, compute from each sample).
    """
    import os
    os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists

    # Iterate through the batch
    for i, flow in enumerate(flow_batch):
        # Generate visualization for each flow tensor
        flow_viz = visualize_optical_flow(flow, max_flow=max_flow)

        # Save with an indexed filename
        save_path = os.path.join(save_dir, f'optical_flow_{i}.png')
        cv2.imwrite(save_path, flow_viz)
        print(f"Optical flow visualization saved at: {save_path}")