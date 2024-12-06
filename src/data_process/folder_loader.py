import os
import cv2
import numpy as np
from collections import deque
import sys
sys.path.append('../')


import os
import cv2
import numpy as np
from collections import deque


class Folder_Loader:
    """The loader for demo with a folder of frames as input."""

    def __init__(self, folder_path, input_size=(288, 512), num_frames=5, 
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        Args:
            folder_path (str): Path to the folder containing frames.
            input_size (tuple): Desired input size (height, width).
            num_frames (int): Number of frames per sequence.
            mean (tuple): Mean values for normalization.
            std (tuple): Standard deviation values for normalization.
        """
        assert os.path.isdir(folder_path), f"No folder at {folder_path}"

        self.frame_paths = sorted([
            os.path.join(folder_path, fname) for fname in os.listdir(folder_path)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        assert len(self.frame_paths) >= num_frames, "Not enough frames in the folder!"

        self.width = input_size[1]
        self.height = input_size[0]
        self.count = 0
        self.num_frames_sequence = num_frames
        self.num_total_frames = len(self.frame_paths)
        self.mean = np.array(mean).reshape(1, 1, 3)  # Reshape for broadcasting
        self.std = np.array(std).reshape(1, 1, 3)    # Reshape for broadcasting

        print(f'Number of frames in the folder: {self.num_total_frames}')

        self.images_sequence = deque(maxlen=num_frames)
        self.get_first_images_sequence()

    def normalize(self, img):
        """Normalize an individual frame."""
        img = img.astype(np.float32) / 255.0  # Scale to [0, 1]
        return (img - self.mean) / self.std  # Normalize using mean and std

    def get_first_images_sequence(self):
        """Load the first `num_frames_sequence` frames and normalize them."""
        for _ in range(self.num_frames_sequence):
            frame_path = self.frame_paths[self.count]
            frame = cv2.imread(frame_path)  # BGR
            assert frame is not None, f'Failed to load frame at {frame_path}'

            # Resize, convert to RGB, and normalize
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.width, self.height))
            normalized_frame = self.normalize(frame)
            self.images_sequence.append(normalized_frame)
            self.count += 1

    def __iter__(self):
        self.count = self.num_frames_sequence - 1  # Start from the first available sequence
        return self

    def __next__(self):
        self.count += 1
        if self.count >= self.num_total_frames:
            raise StopIteration

        # Read the next frame
        frame_path = self.frame_paths[self.count]
        frame = cv2.imread(frame_path)  # BGR
        assert frame is not None, f'Failed to load frame at {frame_path}'

        # Resize, convert to RGB, and normalize
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.width, self.height))
        normalized_frame = self.normalize(frame)
        self.images_sequence.append(normalized_frame)

        # Prepare images in [N, C, H, W] format
        frames_np = np.array(self.images_sequence)  # [N, H, W, C]
        frames_np = frames_np.transpose(0, 3, 1, 2)  # [N, C, H, W]

        return self.count, frames_np, frame

    def __len__(self):
        return self.num_total_frames - self.num_frames_sequence + 1  # Number of sequences


if __name__ == '__main__':
    import time

    import matplotlib.pyplot as plt
    from config.config import parse_configs

    configs = parse_configs()
    configs.num_frames = 5

    folder_loader = Folder_Loader("/home/s224705071/github/PhysicsInformedDeformableAttentionNetwork/data/tennis_data/game1/Clip1")

