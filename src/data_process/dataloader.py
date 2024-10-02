import sys

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader

sys.path.append('../')

from data_process.dataset import PIDA_dataset, Masked_Dataset
from data_process.data_utils import get_all_detection_infor, train_val_data_separation
from data_process.transformation import Compose, Random_Crop, Resize, Normalize, Random_Rotate, Random_HFlip


def create_train_val_dataloader(configs):
    """Create dataloader for training and validate"""

    train_transform = Compose([
        Random_Crop(max_reduction_percent=0.15, p=0.5),
        Random_HFlip(p=0.5),
        Random_Rotate(rotation_angle_limit=10, p=0.5),
    ], p=1.)

    train_events_infor, val_events_infor, train_events_label, val_events_label = train_val_data_separation(configs)
    train_dataset = PIDA_dataset(train_events_infor, train_events_label, transform=train_transform,
                                  num_samples=configs.num_samples)
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=train_sampler)

    val_dataloader = None
    if not configs.no_val:
 
        val_transform = None
        val_sampler = None
        val_dataset = PIDA_dataset(val_events_infor, val_events_label, transform=val_transform,
                                    num_samples=configs.num_samples)
        if configs.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                    pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler)

    return train_dataloader, val_dataloader, train_sampler


def create_test_dataloader(configs):
    """Create dataloader for testing phase"""

    test_transform = None
    dataset_type = 'test'
    test_events_infor, test_events_labels = get_all_detection_infor(configs.test_game_list, configs, dataset_type)
    test_dataset = PIDA_dataset(test_events_infor, test_events_labels, transform=test_transform,
                                 num_samples=configs.num_samples)
    test_sampler = None
    if configs.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False,
                                 pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=test_sampler)

    return test_dataloader


def create_masked_train_val_dataloader(configs):
    """Create dataloader for training and validate"""

    train_transform = Compose([
        Random_Crop(max_reduction_percent=0.15, p=0.5),
        Random_HFlip(p=0.5),
        Random_Rotate(rotation_angle_limit=10, p=0.5),
    ], p=1.)

    train_events_infor, val_events_infor, train_events_label, val_events_label = train_val_data_separation(configs)
    train_dataset = Masked_Dataset(train_events_infor, train_events_label, transform=train_transform,
                                  num_samples=configs.num_samples)
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=train_sampler)

    val_dataloader = None
    if not configs.no_val:
 
        val_transform = None
        val_sampler = None
        val_dataset = Masked_Dataset(val_events_infor, val_events_label, transform=val_transform,
                                    num_samples=configs.num_samples)
        if configs.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                    pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler)

    return train_dataloader, val_dataloader, train_sampler


def create_masked_test_dataloader(configs):
    """Create dataloader for testing phase"""

    test_transform = None
    dataset_type = 'test'
    test_events_infor, test_events_labels = get_all_detection_infor(configs.test_game_list, configs, dataset_type)
    test_dataset = Masked_Dataset(test_events_infor, test_events_labels, transform=test_transform,
                                 num_samples=configs.num_samples)
    test_sampler = None
    if configs.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False,
                                 pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=test_sampler)

    return test_dataloader


def draw_image_with_ball(image_tensor, ball_location_tensor, out_images_dir, example_index):
    # Convert tensors to numpy arrays
    print(image_tensor.shape, ball_location_tensor.shape)
    image = image_tensor.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
    ball_location = ball_location_tensor.cpu().numpy()    # Ensure ball location is on CPU

    # Ensure the image is in uint8 format for OpenCV
    if image.dtype != 'uint8':
        image = (image * 255).astype('uint8')

    # Draw the ball on the image
    ball_xy = tuple(ball_location.astype(int))  # Convert coordinates to int
    img_with_ball = cv2.circle(image.copy(), ball_xy, radius=5, color=(255, 0, 0), thickness=2)

    # Convert the image to BGR format for saving with OpenCV
    img_with_ball = cv2.cvtColor(img_with_ball, cv2.COLOR_RGB2BGR)

    # Save the image
    output_path = os.path.join(out_images_dir, f'example_label_{example_index}.jpg')
    cv2.imwrite(output_path, img_with_ball)

    return output_path  # Optionally return the saved path

if __name__ == '__main__':
    from config.config import parse_configs

    configs = parse_configs()
    configs.distributed = False  # For testing

    # Create dataloaders
    train_dataloader, val_dataloader, train_sampler = create_train_val_dataloader(configs)
    print('len train_dataloader: {}, val_dataloader: {}'.format(len(train_dataloader), len(val_dataloader)))

    test_dataloader = create_test_dataloader(configs)
    print(f"len test_loader {len(test_dataloader)}")


    # Create Masked dataloaders 
    train_dataloader, val_dataloader, train_sampler = create_masked_train_val_dataloader(configs)
    print('len train_dataloader: {}, val_dataloader: {}'.format(len(train_dataloader), len(val_dataloader)))

    test_dataloader = create_masked_test_dataloader(configs)
    print(f"len test_loader {len(test_dataloader)}")

    # # Get one batch from train_dataloader
    # for batch in train_dataloader:
    #     # Assuming batch contains both input data and labels
    #     inputs, (masked_frameids, masked_frames, labels) = batch
    #     print(f"Train batch data shape: {inputs.shape}")
    #     print(f"Train batch labels shape: {labels.shape}")
    #     break  # Exit after printing the first batch


    # # Get one batch from val_dataloader
    # for batch in val_dataloader:
    #     inputs, (masked_frameids, masked_frames, labels) = batch
    #     print(f"Val batch data shape: {inputs.shape}")
    #     print(f"Val batch labels shape: {labels.shape}")
    #     break

    # # Get one batch from test_dataloader
    # for batch in test_dataloader:
    #     inputs, (masked_frameids, masked_frames, labels) = batch
    #     # Test dataloader might have only inputs
    #     print(f"Test batch data shape: {inputs.shape}")
    #     print(f"Test batch labels shape: {labels.shape}")
    #     break

    # show example
    
    batch_data, (masked_frameids, masked_frames, labels) = next(iter(train_dataloader))

    # Check the shapes
    print(f'Batch data shape: {batch_data.shape}')      # Expected: [8, 7, 2, 3, 1080, 1920]
    print(f'Batch labels shape: {labels.shape}, batch masked frames shape {masked_frames.shape}')  # Expected: [8, 2], 2 represents X and Y of the coordinaties 

    # Select the first sample in the batch
    sample_data = batch_data[0]  # Shape: [7, 2, 3, 1080, 1920]

    # Select the first paire in the sequence
    frame = sample_data[0]  # Shape: [2, 3, 1080, 1920]

    # Select the first frame
    img = frame[0]  # Shape: [3, 1080, 1920]

    # Transpose the dimensions to [H, W, C]
    image = np.transpose(img, (1, 2, 0))  # Shape: [1080, 1920, 3]
    image = image.cpu().numpy()
    
    out_images_dir = os.path.join(configs.results_dir, 'debug', 'ttnet_dataset')
    if not os.path.isdir(out_images_dir):
        os.makedirs(out_images_dir)
    cv2.imwrite(os.path.join(out_images_dir, f'example.jpg'), image)

    example_index = 0
    masked_image = masked_frames[example_index] # Shape(3,1080,1920)
    masked_frame = np.transpose(masked_image.cpu().numpy(), (1,2,0))
    ball_xy = labels[example_index].cpu().numpy()
    img_with_ball = cv2.circle(masked_frame.copy(), tuple(ball_xy), radius=5, color=(255, 0, 0), thickness=2)
    img_with_ball = cv2.cvtColor(img_with_ball, cv2.COLOR_RGB2BGR)  # Convert to BGR for saving
    cv2.imwrite(os.path.join(out_images_dir, f'example_label_{example_index}.jpg'), img_with_ball)