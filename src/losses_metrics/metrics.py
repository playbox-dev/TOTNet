import torch
import torch.nn.functional as F

def calculate_metrics(pred_coords, true_coords):
    # Mean Squared Error (MSE)
    mse = F.mse_loss(pred_coords, true_coords)
    
    # Root Mean Squared Error (RMSE)
    rmse = torch.sqrt(mse)

    # Mean Absolute Error (MAE)
    mae = F.l1_loss(pred_coords, true_coords)

    # Euclidean Distance
    euclidean_distance = torch.sqrt(((pred_coords - true_coords) ** 2).sum(dim=1)).mean()

    return mse.item(), rmse.item(), mae.item(), euclidean_distance.item()