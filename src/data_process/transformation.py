import random

import cv2
import numpy as np
import torchvision.transforms.functional as F
import torch


class Compose(object):
    def __init__(self, transforms, p=1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, imgs, ball_position_xy):
        if random.random() <= self.p:
            for t in self.transforms:
                imgs, ball_position_xy = t(imgs, ball_position_xy)
        return imgs, ball_position_xy


class Normalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), num_frames_sequence=9, p=1.0):
        self.p = p
        self.mean = np.array(mean).reshape(1, 1, 3)  # For individual image normalization
        self.std = np.array(std).reshape(1, 1, 3)

    def __call__(self, imgs, ball_position_xy):
        if random.random() < self.p:
            imgs = [((img / 255.) - self.mean) / self.std for img in imgs]

        return imgs, ball_position_xy


class Denormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0):
        self.p = p
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)

    def __call__(self, img):
        img = (img * self.std + self.mean) * 255.
        img = img.astype(np.uint8)

        return img


class Resize(object):
    def __init__(self, new_size, p=0.5, interpolation=cv2.INTER_LANCZOS4):
        self.new_size = new_size  # new_size should be (width, height)
        self.p = p
        self.interpolation = interpolation

    def __call__(self, imgs, ball_position_xy):
        """_summary_

        Args:
            imgs (numpy): list of images
            ball_position_xy (numpy): ball position of intended frame 

        Returns:
            _type_: _description_
        """
        # If random value is greater than p, return original imgs and ball position
        if random.random() > self.p:
            return imgs, ball_position_xy

        # Original image dimensions (assuming imgs[0] has the original size)
        original_w, original_h, _ = imgs[0].shape

        # New image dimensions
        new_w, new_h = self.new_size
        
        # Resize a sequence of images
        transformed_imgs = []
        for img in imgs:
            transformed_img = cv2.resize(img, (new_h, new_w), interpolation=self.interpolation)
            transformed_imgs.append(transformed_img)

        # Adjust ball position
        w_ratio = float(original_w) / float(new_w)  # New width divided by original width
        h_ratio = float(original_h) / float(new_h) # New height divided by original height

        transformed_ball_pos = np.array([ball_position_xy[0] / w_ratio, ball_position_xy[1] / h_ratio])
        # Round the coordinates to the nearest integer
        # transformed_ball_pos = np.round(transformed_ball_pos).astype(int)
        
        return transformed_imgs, transformed_ball_pos


class Random_Crop(object):
    def __init__(self, max_reduction_percent=0.15, p=0.5, interpolation=cv2.INTER_LINEAR):
        self.max_reduction_percent = max_reduction_percent
        self.p = p
        self.interpolation = interpolation

    def __call__(self, imgs, ball_position_xy):
        transformed_imgs = imgs.copy()
        transformed_ball_pos = ball_position_xy.copy()
        # imgs are before resizing
        if random.random() <= self.p:
            h, w, c = imgs[0].shape
            # Calculate min_x, max_x, min_y, max_y
            remain_percent = random.uniform(1. - self.max_reduction_percent, 1.)
            new_w = remain_percent * w
            min_x = int(random.uniform(0, w - new_w))
            max_x = int(min_x + new_w)
            w_ratio = w / new_w

            new_h = remain_percent * h
            min_y = int(random.uniform(0, h - new_h))
            max_y = int(new_h + min_y)
            h_ratio = h / new_h
            # crop a sequence of images
            transformed_imgs = []
            for i, img in enumerate(imgs):
                img_cropped = img[min_y:max_y, min_x:max_x, :]
                # Resize the image to the original dimensions
                img_resized = cv2.resize(img_cropped, (w, h), interpolation=self.interpolation)
                transformed_imgs.append(img_resized)
                if i == len(imgs):
                    transformed_ball_pos = np.array([(ball_position_xy[0] - min_x) * w_ratio,
                                            (ball_position_xy[1] - min_y) * h_ratio])

        return transformed_imgs, transformed_ball_pos


class Random_Rotate(object):
    def __init__(self, rotation_angle_limit=15, p=0.5):
        self.rotation_angle_limit = rotation_angle_limit
        self.p = p

    def __call__(self, imgs, ball_position_xy):
        transformed_imgs = imgs.copy()
        if random.random() <= self.p:
            random_angle = random.uniform(-self.rotation_angle_limit, self.rotation_angle_limit)
            # Rotate a sequence of imgs
            h, w, c = imgs[0].shape
            center = (int(w / 2), int(h / 2))
            rotate_matrix = cv2.getRotationMatrix2D(center, random_angle, 1.)

            transformed_imgs = []
            for img in imgs:
                transformed_img = cv2.warpAffine(img, rotate_matrix, (w, h), flags=cv2.INTER_LINEAR)
                transformed_imgs.append(transformed_img)

            # Adjust ball position, apply the same rotate_matrix for the sequential images
            ball_position_xy = rotate_matrix.dot(np.array([ball_position_xy[0], ball_position_xy[1], 1.]).T)

        return transformed_imgs, ball_position_xy


class Random_HFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs, ball_position_xy):
        transformed_imgs = imgs.copy()
        if random.random() <= self.p:
            h, w, c = imgs[0].shape
            transformed_imgs = []
            for img in imgs:
                # Horizontal flip a sequence of imgs
                transformed_img = cv2.flip(img, 1)
                transformed_imgs.append(transformed_img)

            # Adjust ball position: Same y, new x = w - x
            ball_position_xy[0] = w - ball_position_xy[0]

        return transformed_imgs, ball_position_xy


class Random_VFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs, ball_position_xy):
        transformed_imgs = imgs.copy()
        if random.random() <= self.p:
            h, w, c = imgs[0].shape
            transformed_imgs = []
            for img in imgs:
                # Horizontal flip a sequence of imgs
                transformed_img = cv2.flip(img, 0)
                transformed_imgs.append(transformed_img)

            # Adjust ball position: Same x, new y = h - y
            ball_position_xy[1] = h - ball_position_xy[1]

        return transformed_imgs, ball_position_xy



class Random_Ball_Mask:
    def __init__(self, mask_size=(20, 20), p=0.5, mask_type='mean'):
        """
        Args:
            mask_size (tuple): Height and width of the mask area (blackout area).
            p (float): Probability of applying the mask.
            mask_type (str): Type of mask ('zero', 'noise', 'mean').
        """
        self.mask_size = mask_size
        self.p = p
        self.mask_type = mask_type

    def __call__(self, imgs, ball_position_xy):
        """
        Args:
            imgs : List of numpy arrays where the length represents num frames
            ball_position_xy (numpy): (x, y) ball position for the labeled frame.

        Returns:
            masked_imgs: Tensor with the ball area masked in some frames.
        """
        H, W, C = imgs[0].shape  # Get the shape from the input tensor
        movement_range = self.mask_size[0]//2
        mask_h = random.randint(self.mask_size[0] - movement_range, self.mask_size[0] + movement_range)
        mask_w = random.randint(self.mask_size[1] - movement_range, self.mask_size[1] + movement_range)
        x, y = int(ball_position_xy[0]), int(ball_position_xy[1])
        # Iterate over all frames and apply masking with some probability
        for i, (img) in enumerate(imgs):
            if random.random() < self.p:
                if i == len(imgs)-1:
                    x, y = int(ball_position_xy[0]), int(ball_position_xy[1])
                else:
                    # Apply mask at a random position in non-labeled frames
                    x = random.randint(0, W - mask_w)
                    y = random.randint(0, H - mask_h)

                # Ensure the mask is within the image boundaries
                top = max(0, min(H - mask_h, y - mask_h // 2))
                left = max(0, min(W - mask_w, x - mask_w // 2))

                # Apply the chosen mask type
                if self.mask_type == 'zero':
                    img[top:top + mask_h, left:left + mask_w, :] = 0

                elif self.mask_type == 'noise':
                    noise = np.random.randn(mask_h, mask_w, C) * 255  # Generate noise in the same shape
                    img[top:top + mask_h, left:left + mask_w, :] = noise.clip(0, 255)  # Apply noise

                elif self.mask_type == 'mean':
                    # Calculate the mean value along the spatial dimensions
                    mean_value = img[top:top + mask_h, left:left + mask_w, :].mean(axis=(0, 1))
                    noise = np.random.randn(mask_h, mask_w, C) * 10  # Small noise
                    img[top:top + mask_h, left:left + mask_w, :] = (mean_value + noise).clip(0, 255)  # Apply mean

        return imgs, ball_position_xy



class RandomColorJitter(object):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1, p=0.5):
        """
        Initializes the RandomColorJitter augmentation.

        Args:
            brightness (float): Brightness adjustment factor (default 0.2).
            contrast (float): Contrast adjustment factor (default 0.2).
            saturation (float): Saturation adjustment factor (default 0.1).
            hue (float): Hue shift factor (default 0.1).
            p (float): Probability of applying the jitter (default 0.5).
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, imgs, ball_position_xy):
        """
        Applies random color jitter to a sequence of images.

        Args:
            imgs (list of np.ndarray): List of images to augment.

        Returns:
            list of np.ndarray: Augmented images.
        """
        transformed_imgs = imgs.copy()

        # Apply jitter with probability p
        if random.random() <= self.p:
            transformed_imgs = []
            for img in imgs:
                jittered_img = self.apply_jitter(img)
                transformed_imgs.append(jittered_img)

        return transformed_imgs, ball_position_xy

    def apply_jitter(self, img):
        """
        Applies brightness, contrast, saturation, and hue jitter to a single image.

        Args:
            img (np.ndarray): Input image (H, W, C) in range [0, 255].

        Returns:
            np.ndarray: Jittered image.
        """
        # Convert to float32 and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Apply brightness jitter
        brightness_factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
        img = np.clip(img * brightness_factor, 0, 1)

        # Apply contrast jitter
        contrast_factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = np.clip((img - mean) * contrast_factor + mean, 0, 1)

        # Convert to HSV to apply saturation and hue jitter
        hsv_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)

        # Apply saturation jitter
        saturation_factor = 1.0 + np.random.uniform(-self.saturation, self.saturation)
        hsv_img[..., 1] = np.clip(hsv_img[..., 1] * saturation_factor, 0, 255)

        # Apply hue jitter
        hue_shift = np.random.uniform(-self.hue * 180, self.hue * 180)
        hsv_img[..., 0] = (hsv_img[..., 0] + hue_shift) % 180

        # Convert back to RGB
        jittered_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

        # Convert back to uint8 and scale to [0, 255]
        jittered_img = np.clip(jittered_img * 255, 0, 255).astype(np.uint8)

        return jittered_img


