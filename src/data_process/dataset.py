import sys
import os
import numpy as np
import time

from torch.utils.data import Dataset
import cv2

sys.path.append('../')


class PIDA_dataset(Dataset):
    def __init__(self, events_infor, events_label, transform=None, num_samples=None):
        self.events_infor = events_infor
        self.events_label = events_label
        self.transform = transform

        if num_samples is not None:
            self.events_infor = self.events_infor[:num_samples]

    def __len__(self):
        return len(self.events_infor)

    def __getitem__(self, index):
        img_path_list = self.events_infor[index]
        ball_xy = self.events_label[index]
        imgs = []
        for img_path in img_path_list:
            img = cv2.imread(img_path)
            imgs.append(img)
        # Apply augmentation
        if self.transform:
            imgs, ball_xy= self.transform(imgs, ball_xy)
        
        converted_imgs = []
        for img in imgs:    
            img = np.transpose(img, (2, 0, 1))  # Now img is (C, H, W)
            converted_imgs.append(img)

        numpy_imgs = np.stack(converted_imgs, axis=0)  # Stack along the new axis (N)
        
        return numpy_imgs, ball_xy.astype(int)


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    import torch
    from config.config import parse_configs
    from data_process.data_utils import train_val_data_separation
    from data_process.transformation import Compose, Random_Crop, Resize, Random_HFlip, Random_Rotate

    configs = parse_configs()
    game_list = ['game_1']
    dataset_type = 'training'
    train_events_infor, val_events_infor, train_events_label, val_events_label = train_val_data_separation(configs)
    print('len(train_events_infor): {}'.format(len(train_events_infor)))
    # Test transformation
    transform = Compose([
        Random_Crop(max_reduction_percent=0.15, p=1.),
        Random_HFlip(p=1.),
        Random_Rotate(rotation_angle_limit=15, p=1.)
    ], p=1.)

    ttnet_dataset = PIDA_dataset(train_events_infor, train_events_label, transform=transform)

    print('len(ttnet_dataset): {}'.format(len(ttnet_dataset)))
    example_index = 100
    imgs, ball_xy = ttnet_dataset.__getitem__(example_index)
    if 1:
        # Test F.interpolate, we can simply use cv2.resize() to get origin_imgs from resized_imgs
        # Achieve better quality of images and faster
        print(f"dataset images shape is {imgs.shape}")
        origin_imgs = F.interpolate(torch.from_numpy(imgs).float(), (1080, 1920))
        origin_imgs = origin_imgs.numpy().transpose(0, 2, 3, 1).astype(np.uint8)  # (N, C, H, W) to (N, H, W, C)
        print('F.interpolate - origin_imgs shape: {}'.format(origin_imgs.shape))
        resized_imgs = imgs.transpose(0, 2, 3, 1)  # (N, H, W, C)
        print('resized_imgs shape: {}'.format(resized_imgs.shape))
    else:
        # Test cv2.resize
        resized_imgs = imgs.transpose(0, 2, 3, 1)  # (N, H, W, C)
        print('resized_imgs shape: {}'.format(resized_imgs.shape))
        origin_imgs = np.array([cv2.resize(img, (1920, 1080)) for img in resized_imgs])  # Resize each image in the batch
        print('cv2.resize - origin_imgs shape: {}'.format(origin_imgs.shape))  # Should print (9, 1080, 1920, 3)

    # Set the output directory
    out_images_dir = os.path.join(configs.results_dir, 'debug', 'ttnet_dataset')
    if not os.path.isdir(out_images_dir):
        os.makedirs(out_images_dir)

    # Plot the original images
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
    axes = axes.ravel()

    # Loop over frames to plot the original images
    for i in range(configs.num_frames_sequence):
        img = origin_imgs[i]  # Assuming origin_imgs is in the format (N, H, W, C) or (N, C, H, W)
        axes[i].imshow(img)
        axes[i].set_title(f'Original image {i}')
    fig.suptitle(
        f'Ball Position: (x= {ball_xy[0]}, y= {ball_xy[1]})', fontsize=16
    )
    plt.savefig(os.path.join(out_images_dir, f'org_all_imgs_{example_index}.jpg'))
    plt.close(fig)  # Close the figure after saving

    # Plot the resized images
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
    axes = axes.ravel()

    for i in range(configs.num_frames_sequence):
        img = resized_imgs[i]  # Assuming resized_imgs is also (N, H, W, C) or (N, C, H, W)
        axes[i].imshow(img)
        axes[i].set_title(f'Resized image {i}')
        
        # If it's the middle frame, add a circle for the ball position
        if i == (configs.num_frames_sequence//2):
            print(f"ball frame is {i}")
            img_with_ball = cv2.circle(img.copy(), tuple(ball_xy), radius=5, color=(255, 0, 0), thickness=2)
            img_with_ball = cv2.cvtColor(img_with_ball, cv2.COLOR_RGB2BGR)  # Convert to BGR for saving
            cv2.imwrite(os.path.join(out_images_dir, f'augment_img_{example_index}.jpg'), img_with_ball)

    fig.suptitle(
        f'Ball Position: (x= {ball_xy[0]}, y= {ball_xy[1]})', fontsize=16
    )
    plt.savefig(os.path.join(out_images_dir, f'augment_all_imgs_{example_index}.jpg'))
    plt.close(fig)  # Close the figure after saving