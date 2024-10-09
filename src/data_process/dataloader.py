import sys

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader, Subset

sys.path.append('../')

from data_process.dataset import PIDA_dataset, Masked_Dataset, Normal_Dataset
from data_process.data_utils import get_all_detection_infor, train_val_data_separation
from data_process.transformation import Compose, Random_Crop, Resize, Normalize, Random_Rotate, Random_HFlip, Random_VFlip


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


def create_masked_train_val_dataloader(configs, subset_size=None):
    """Create dataloader for training and validation, with an option to use a subset of the data."""

    train_transform = Compose([
        Resize(new_size=configs.img_size, p=1.0),
        Random_Crop(max_reduction_percent=0.15, p=0.5),
        Random_HFlip(p=0.5),
        Random_Rotate(rotation_angle_limit=10, p=0.5),
    ], p=1.)

    # Load train and validation data information
    train_events_infor, val_events_infor, train_events_label, val_events_label = train_val_data_separation(configs)

    # Create train dataset
    train_dataset = Masked_Dataset(train_events_infor, train_events_label, transform=train_transform,
                                   num_samples=configs.num_samples)
    
    # If subset_size is provided, create a subset for training
    if subset_size is not None:
        train_indices = torch.randperm(len(train_dataset))[:subset_size].tolist()
        train_dataset = Subset(train_dataset, train_indices)
    
    # Create train sampler if distributed
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    # Create train dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=train_sampler, drop_last=True)

    # Create validation dataloader (without transformations)
    val_dataloader = None
    if not configs.no_val:
        val_transform = Compose([
            Resize(new_size=configs.img_size, p=1.0),
        ], p=1.)
        val_dataset = Masked_Dataset(val_events_infor, val_events_label, transform=val_transform,
                                     num_samples=configs.num_samples)

        # If subset_size is provided, create a subset for validation
        if subset_size is not None:
            val_indices = torch.randperm(len(val_dataset))[:subset_size].tolist()
            val_dataset = Subset(val_dataset, val_indices)
        
        # Create validation sampler if distributed
        val_sampler = None
        if configs.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        
        # Create validation dataloader
        val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                    pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler, drop_last=True)

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


def create_normal_train_val_dataloader(configs, subset_size=None):
    """Create dataloader for training and validation, with an option to use a subset of the data."""

    train_transform = Compose([
        Resize(new_size=configs.img_size, p=1.0),
        Random_Crop(max_reduction_percent=0.15, p=0.5),
        Random_HFlip(p=0.5),
        Random_VFlip(p=0.5),
        Random_Rotate(rotation_angle_limit=10, p=0.5),
    ], p=1.)

    # Load train and validation data information
    train_events_infor, val_events_infor, train_events_label, val_events_label = train_val_data_separation(configs)

    # Create train dataset
    train_dataset = Normal_Dataset(train_events_infor, train_events_label, transform=train_transform,
                                   num_samples=configs.num_samples)
    
    # If subset_size is provided, create a subset for training
    if subset_size is not None:
        train_indices = torch.randperm(len(train_dataset))[:subset_size].tolist()
        train_dataset = Subset(train_dataset, train_indices)
    
    # Create train sampler if distributed
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    # Create train dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=train_sampler, drop_last=False)

    # Create validation dataloader (without transformations)
    val_dataloader = None
    if not configs.no_val:
        val_transform = Compose([
            Resize(new_size=configs.img_size, p=1.0),
        ], p=1.)
        val_dataset = Normal_Dataset(val_events_infor, val_events_label, transform=val_transform,
                                     num_samples=configs.num_samples)

        # If subset_size is provided, create a subset for validation
        if subset_size is not None:
            val_indices = torch.randperm(len(val_dataset))[:subset_size].tolist()
            val_dataset = Subset(val_dataset, val_indices)
        
        # Create validation sampler if distributed
        val_sampler = None
        if configs.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        
        # Create validation dataloader
        val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                    pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler, drop_last=True)

    return train_dataloader, val_dataloader, train_sampler

def create_normal_test_dataloader(configs):
    """Create dataloader for testing phase"""

    test_transform = Compose([
            Resize(new_size=configs.img_size, p=1.0),
        ], p=1.)
    dataset_type = 'test'
    test_events_infor, test_events_labels = get_all_detection_infor(configs.test_game_list, configs, dataset_type)
    test_dataset = Normal_Dataset(test_events_infor, test_events_labels, transform=test_transform,
                                 num_samples=configs.num_samples)
    test_sampler = None
    if configs.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False,
                                 pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=test_sampler)

    return test_dataloader



def draw_image_with_ball(image_tensor, ball_location_tensor, out_images_dir, example_index):
    # Convert tensors to numpy arrays

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
    train_dataloader, val_dataloader, train_sampler = create_normal_train_val_dataloader(configs)
    print('len train_dataloader: {}, val_dataloader: {}'.format(len(train_dataloader), len(val_dataloader)))

    # test_dataloader = create_test_dataloader(configs)
    # print(f"len test_loader {len(test_dataloader)}")

    # show example
    # normal dataset doesnt contain masked frame
    batch_data, (masked_frameids, _, labels) = next(iter(train_dataloader))

    # Check the shapes
    print(f'Batch data shape: {batch_data.shape}')      # Expected: [B, Number of images, C, H, W]
    print(f'Batch labels shape: {labels.shape}')  # Expected: [8, 2], 2 represents X and Y of the coordinaties 

    # Select the first sample in the batch
    sample_data = batch_data[0]  # Shape: [N, C, H, W]

    # Select the first frame
    img = sample_data[0]  # Shape: [C, H, W]

    # Transpose the dimensions to [H, W, C]
    image = np.transpose(img, (1, 2, 0))  # Shape: [H, W, C]
    image = image.cpu().numpy()
    
    out_images_dir = os.path.join(configs.results_dir, 'debug', 'ttnet_dataset')
    if not os.path.isdir(out_images_dir):
        os.makedirs(out_images_dir)
    cv2.imwrite(os.path.join(out_images_dir, f'example.jpg'), image)

    example_index = 0
    masked_image = sample_data[masked_frameids].squeeze() # Shape(3,1080,1920)
    masked_frame = np.transpose(masked_image.cpu().numpy(), (1,2,0))
    ball_xy = labels[example_index].cpu().numpy()
    img_with_ball = cv2.circle(masked_frame.copy(), tuple(ball_xy), radius=5, color=(255, 0, 0), thickness=2)
    img_with_ball = cv2.cvtColor(img_with_ball, cv2.COLOR_RGB2BGR)  # Convert to BGR for saving
    cv2.imwrite(os.path.join(out_images_dir, f'normal_data_example_label_{example_index}.jpg'), img_with_ball)



    # Create dataloaders
    train_dataloader, val_dataloader, train_sampler = create_train_val_dataloader(configs)
    print('len train_dataloader: {}, val_dataloader: {}'.format(len(train_dataloader), len(val_dataloader)))

    test_dataloader = create_test_dataloader(configs)
    print(f"len test_loader {len(test_dataloader)}")

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




    # Create Masked dataloaders 
    train_dataloader, val_dataloader, train_sampler = create_masked_train_val_dataloader(configs)
    print('len train_dataloader: {}, val_dataloader: {}'.format(len(train_dataloader), len(val_dataloader)))

    test_dataloader = create_masked_test_dataloader(configs)
    print(f"len test_loader {len(test_dataloader)}")


    batch_data, (masked_frameids, masked_frames, labels) = next(iter(train_dataloader))

    # Check the shapes
    print(f'Batch data shape: {batch_data.shape}')      # Expected: [B, N, 3, 1080, 1920]
    print(f'Batch labels shape: {labels.shape}, batch masked frames shape {masked_frames.shape}')  # Expected: [8, 2], 2 represents X and Y of the coordinaties 
    print(masked_frameids, labels)
    # Select the first sample in the batch
    sample_data = batch_data[0]  # Shape: [8, 3, 1080, 1920]

    # Select the first frame
    img = sample_data[0]  # Shape: [3, 1080, 1920]

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
    cv2.imwrite(os.path.join(out_images_dir, f'example_label_{example_index}_masked.jpg'), img_with_ball)
