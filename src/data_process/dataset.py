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
            if img is None:
                raise ValueError(f"Image not found or can't be read at path: {img_path}")
            imgs.append(img)
        # Apply augmentation
        if self.transform:
            imgs, ball_xy= self.transform(imgs, ball_xy)
        
        converted_imgs = []
        for img in imgs:    
            img = np.transpose(img, (2, 0, 1))  # Now img is (C, H, W)
            converted_imgs.append(img)
        # stack them to form the shape (1,num_frames, C, H, W)
        # numpy_imgs = np.stack(converted_imgs, axis=0)  # Stack along the new axis (N)
        # convert them into pairs formation
        image_pairs = []
        masked_frameid = len(converted_imgs)//2 
        i = 1
        while i < len(converted_imgs):
            # Handle the masked frame case by skipping the masked frame
            if i == masked_frameid:
                # Convert to NumPy arrays
                image_pair = (np.array(converted_imgs[i-1]), np.array(converted_imgs[i+1]))
                image_pairs.append(image_pair)
                i += 2  # Skip the masked frame and move to the next
            else:
                # Convert to NumPy arrays
                image_pair = (np.array(converted_imgs[i-1]), np.array(converted_imgs[i]))
                image_pairs.append(image_pair)
                i += 1  # Standard increment
        
        image_pairs_np = np.array(image_pairs)
        masked_frame = np.array(converted_imgs[masked_frameid])
        return image_pairs_np, (masked_frameid, masked_frame, np.array(ball_xy.astype(int)))


class Masked_Dataset(Dataset):
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

            if img is None:
                raise ValueError(f"Image not found or can't be read at path: {img_path}")
            imgs.append(img)
        # Apply augmentation
        if self.transform:
            imgs, ball_xy= self.transform(imgs, ball_xy)
        
        converted_imgs = []
        for img in imgs:    
            # after transform all images will be in shape (H, W, C)
            img = np.transpose(img, (2, 0, 1))  # Now img is (C, H, W)
            converted_imgs.append(img)
        # stack them to form the shape (1,num_frames, C, H, W)
        # numpy_imgs = np.stack(converted_imgs, axis=0)  # Stack along the new axis (N)
        # convert them into pairs formation
        image_list=[]
        masked_frameid = len(converted_imgs)//2 
        i = 0
        while i < len(converted_imgs):
            # Handle the masked frame case by skipping the masked frame
            if i != masked_frameid:
                # Convert to NumPy arrays
                image_list.append(np.array(converted_imgs[i]))
            i+=1
        
        image_list_np = np.array(image_list)
        masked_frame = np.array(converted_imgs[masked_frameid])
        return image_list_np, (masked_frameid, masked_frame, np.array(ball_xy.astype(int)))
    
class Normal_Dataset(Dataset):
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

            if img is None:
                raise ValueError(f"Image not found or can't be read at path: {img_path}")
            imgs.append(img)
        # Apply augmentation
        if self.transform:
            imgs, ball_xy= self.transform(imgs, ball_xy)
        
        converted_imgs = []
        for img in imgs:    
            # after transform all images will be in shape (H, W, C)
            img = np.transpose(img, (2, 0, 1))  # Now img is (C, H, W)
            converted_imgs.append(img)
        # stack them to form the shape (1,num_frames, C, H, W)
        # numpy_imgs = np.stack(converted_imgs, axis=0)  # Stack along the new axis (N)
        # convert them into pairs formation
        # add a padded frame so the number is equal and can be processed with, only when the images is in odd length
        image_list=[]
        if len(converted_imgs)//2 != 0:
            pad_frame = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)
            image_list.append(pad_frame)

        masked_frameid = len(converted_imgs)//2 
        i = 0
        while i < len(converted_imgs):
            image_list.append(np.array(converted_imgs[i]))
            i+=1
        
        image_list_np = np.array(image_list)
        masked_frame = np.array(converted_imgs[masked_frameid])
        return image_list_np, (masked_frameid, masked_frame, np.array(ball_xy.astype(int)))


class Occlusion_Dataset(Dataset):
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

            if img is None:
                raise ValueError(f"Image not found or can't be read at path: {img_path}")
            imgs.append(img)
        # Apply augmentation
        if self.transform:
            imgs, ball_xy= self.transform(imgs, ball_xy)
        
        converted_imgs = []
        for img in imgs:    
            # after transform all images will be in shape (H, W, C)
            img = np.transpose(img, (2, 0, 1))  # Now img is (C, H, W)
            converted_imgs.append(img)
        # stack them to form the shape (1,num_frames, C, H, W)
        # numpy_imgs = np.stack(converted_imgs, axis=0)  # Stack along the new axis (N)
        # convert them into pairs formation
        # add a padded frame so the number is equal and can be processed with, only when the images is in odd length
        image_list=[]

        masked_frameid = len(converted_imgs)//2 
        i = 0
        while i < len(converted_imgs):
            image_list.append(np.array(converted_imgs[i]))
            i+=1
        
        image_list_np = np.array(image_list)
        masked_frame = np.array(converted_imgs[masked_frameid])
        return image_list_np, (masked_frameid, masked_frame, np.array(ball_xy.astype(int)))


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    import torch
    from config.config import parse_configs
    from data_process.data_utils import train_val_data_separation
    from data_process.transformation import Compose, Random_Crop, Resize, Random_HFlip, Random_Rotate

    configs = parse_configs()
    train_events_infor, val_events_infor, train_events_label, val_events_label = train_val_data_separation(configs)
    print('len(train_events_infor): {}'.format(len(train_events_infor)))
    # Test transformation
    transform = Compose([
        Random_Crop(max_reduction_percent=0.15, p=1.),
        Random_HFlip(p=1.),
        Random_Rotate(rotation_angle_limit=15, p=1.)
    ], p=1.)

    ttnet_dataset = PIDA_dataset(train_events_infor, train_events_label, transform=transform)
    masked_dataset = Masked_Dataset(train_events_infor, train_events_label, transform=transform)

    print('len(ttnet_dataset): {}'.format(len(ttnet_dataset)))
    example_index = 200
    image_pairs_np, (masked_frameid, masked_frame, ball_xy) = ttnet_dataset.__getitem__(example_index)

    if 1:
        # Test F.interpolate (Torch-based resizing)
        print(f"dataset images shape is {image_pairs_np.shape}")

        # Reshape (7, 2, C, H, W) to (7*2, C, H, W) to handle each image independently for resizing
        imgs = image_pairs_np.reshape(-1, image_pairs_np.shape[2], image_pairs_np.shape[3], image_pairs_np.shape[4])
        
        origin_imgs = F.interpolate(torch.from_numpy(imgs).float(), size=(1080, 1920))
        
        # Convert back to NumPy and reshape to (7, 2, 1080, 1920, C) for further usage
        origin_imgs = origin_imgs.numpy().transpose(0, 2, 3, 1).astype(np.uint8)  # (N, C, H, W) to (N, H, W, C)
        origin_imgs = origin_imgs.reshape(7, 2, 1080, 1920, -1)  # Reshape back to (7, 2, 1080, 1920, C)
        
        print('F.interpolate - origin_imgs shape: {}'.format(origin_imgs.shape))
        
        # Reshape for resized images (from (7, 2, C, H, W) to (7*2, H, W, C))
        resized_imgs = imgs.transpose(0, 2, 3, 1)  # (N, C, H, W) to (N, H, W, C)
        resized_imgs = resized_imgs.reshape(7, 2, resized_imgs.shape[1], resized_imgs.shape[2], resized_imgs.shape[3])
        print('resized_imgs shape: {}'.format(resized_imgs.shape))

    else:
        # Test cv2.resize (CV2-based resizing)
        resized_imgs = image_pairs_np.transpose(0, 1, 3, 4, 2)  # Convert to (7, 2, H, W, C)
        
        # Reshape to (7*2, H, W, C) to process all images
        resized_imgs = resized_imgs.reshape(-1, resized_imgs.shape[2], resized_imgs.shape[3], resized_imgs.shape[4])
        
        # Resize each image in the batch
        origin_imgs = np.array([cv2.resize(img, (1920, 1080)) for img in resized_imgs])  # (7*2, 1080, 1920, 3)
        
        # Reshape back to (7, 2, 1080, 1920, 3)
        origin_imgs = origin_imgs.reshape(7, 2, 1080, 1920, -1)
        
        print('cv2.resize - origin_imgs shape: {}'.format(origin_imgs.shape))

    # Set the output directory
    out_images_dir = os.path.join(configs.results_dir, 'debug', 'ttnet_dataset')
    if not os.path.isdir(out_images_dir):
        os.makedirs(out_images_dir)

    # Assuming you want to plot N=9 frames and are forming 8 overlapping pairs
    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(100, 100))  # Keep the original 3x3 grid for 9 frames
    axes = axes.ravel()

    for idx in range(len(origin_imgs)):  # This will iterate over 7 pairs
        img1 = origin_imgs[idx, 0]      # First frame in the pair
        img2 = origin_imgs[idx, 1]  # Second frame in the pair

        # Plot the first frame in the pair
        axes[2 * idx].imshow(img1)
        axes[2 * idx].set_title(f'Original image {idx}')

        # Plot the second frame in the pair
        axes[2 * idx + 1].imshow(img2)
        axes[2 * idx + 1].set_title(f'Original image {idx + 1}')

    fig.suptitle(
        f'Ball Position: (x= {ball_xy[0]}, y= {ball_xy[1]})', fontsize=16
    )
    plt.savefig(os.path.join(out_images_dir, f'org_all_imgs_{example_index}.jpg'))
    plt.close(fig)  # Close the figure after saving

    # Plot the resized images (following the same logic)
    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(20, 20))  # Keep the original 3x3 grid for 9 frames
    axes = axes.ravel()

    for idx in range(len(origin_imgs)):  # This will iterate over 7 pairs
        img1 = resized_imgs[idx, 0]      # First frame in the pair
        img2 = resized_imgs[idx, 1]  # Second frame in the pair

        # Plot the first frame in the pair
        axes[2 * idx].imshow(img1)
        axes[2 * idx].set_title(f'Resized image {idx}')

        # Plot the second frame in the pair
        axes[2 * idx + 1].imshow(img2)
        axes[2 * idx + 1].set_title(f'Resized image {idx + 1}')

    # draw masked frame
    masked_frame = masked_frame.transpose(1,2,0)
    print(f"masked frame id is {masked_frameid}, shape is {masked_frame.shape}")
    img_with_ball = cv2.circle(masked_frame.copy(), tuple(ball_xy), radius=5, color=(255, 0, 0), thickness=2)
    img_with_ball = cv2.cvtColor(img_with_ball, cv2.COLOR_RGB2BGR)  # Convert to BGR for saving
    cv2.imwrite(os.path.join(out_images_dir, f'augment_img_{example_index}.jpg'), img_with_ball)

    fig.suptitle(
        f'Ball Position: (x= {ball_xy[0]}, y= {ball_xy[1]})', fontsize=16
    )
    plt.savefig(os.path.join(out_images_dir, f'augment_all_imgs_{example_index}.jpg'))
    plt.close(fig)  # Close the figure after saving