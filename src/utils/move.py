import os
import shutil

def move_files_to_subfolder(parent_folder, subfolder_name):
    """
    Moves all files from the parent folder into a specified subfolder.

    Args:
        parent_folder (str): Path to the parent folder.
        subfolder_name (str): Name of the subfolder where files will be moved.
    """
    # Path to the subfolder
    subfolder_path = os.path.join(parent_folder, subfolder_name)

    # Ensure the subfolder exists
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # Iterate over all items in the parent folder
    for file_name in os.listdir(parent_folder):
        source_path = os.path.join(parent_folder, file_name)

        # Skip the subfolder itself
        if source_path == subfolder_path:
            continue

        # Check if it's a file (not a directory)
        if os.path.isfile(source_path):
            destination_path = os.path.join(subfolder_path, file_name)
            print(f"Moving: {source_path} -> {destination_path}")
            shutil.move(source_path, destination_path)

    print(f"All files have been moved to: {subfolder_path}")

# Example usage
parent_folder = "/home/s224705071/github/PhysicsInformedDeformableAttentionNetwork/data/tta_dataset/training/images/24Paralympics_FRA_F9_Lei_AUS_v_Xiong_CHN"  # Replace with the path to the parent folder
subfolder_name = "whole_game"  # Replace with the name of the subfolder
move_files_to_subfolder(parent_folder, subfolder_name)