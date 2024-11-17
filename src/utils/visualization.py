import matplotlib.pyplot as plt
import torch
import os
import numpy as np


def visualize_and_save_2d_heatmap(output_heatmap, save_dir, figsize=(20, 20)):
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

        # Plot the 2D heatmap
        plt.figure(figsize=figsize)
        plt.imshow(heatmap_2d, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Probability')
        plt.title(f'2D Heatmap Visualization for Sample {i+1}')
        plt.xlabel('Width')
        plt.ylabel('Height')

        # Save the figure
        file_path = os.path.join(save_dir, f"heatmap_sample_{i+1}.png")
        plt.savefig(file_path)
        print(f"Saved heatmap for sample {i+1} to {file_path}")
        plt.close()  # Close the figure to free memory