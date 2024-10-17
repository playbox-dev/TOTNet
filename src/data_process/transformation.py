import random

import cv2
import numpy as np
import torchvision.transforms.functional as F


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
        self.mean = np.repeat(np.array(mean).reshape(1, 1, 3), repeats=num_frames_sequence, axis=-1)
        self.std = np.repeat(np.array(std).reshape(1, 1, 3), repeats=num_frames_sequence, axis=-1)

    def __call__(self, imgs, ball_position_xy):
        if random.random() < self.p:
            imgs = ((imgs / 255.) - self.mean) / self.std

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
    def __init__(self, new_size, p=0.5, interpolation=cv2.INTER_LINEAR):
        self.new_size = new_size  # new_size should be (width, height)
        self.p = p
        self.interpolation = interpolation

    def __call__(self, imgs, ball_position_xy):
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
        w_ratio = new_w / original_w  # New width divided by original width
        h_ratio = new_h / original_h  # New height divided by original height

        # Scale ball position to match the resized image
        ball_position_xy = np.array([ball_position_xy[0] * w_ratio, ball_position_xy[1] * h_ratio])

        return transformed_imgs, ball_position_xy


class Random_Crop(object):
    def __init__(self, max_reduction_percent=0.15, p=0.5, interpolation=cv2.INTER_LINEAR):
        self.max_reduction_percent = max_reduction_percent
        self.p = p
        self.interpolation = interpolation

    def __call__(self, imgs, ball_position_xy):
        transformed_imgs = imgs.copy()
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
            for img in imgs:
                img_cropped = img[min_y:max_y, min_x:max_x, :]
                # Resize the image to the original dimensions
                img_resized = cv2.resize(img_cropped, (w, h), interpolation=self.interpolation)
                transformed_imgs.append(img_resized)
           
            # Adjust ball position
            ball_position_xy = np.array([(ball_position_xy[0] - min_x) * w_ratio,
                                         (ball_position_xy[1] - min_y) * h_ratio])

        return transformed_imgs, ball_position_xy


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
    def __init__(self, mask_size=(20, 20), p=0.5):
        """
        Args:
            mask_size (tuple): Height and width of the mask area (blackout area).
            p (float): Probability of applying the mask.
        """
        self.mask_size = mask_size
        self.p = p

    def __call__(self, imgs, ball_position_xy):
        """
        Args:
            imgs (tensor): Tensor of shape [N, C, H, W], where N is the number of frames.
            ball_position_xy (tuple): (x, y) ball position for the labeled frame.

        Returns:
            masked_imgs: Tensor with the ball area masked in some frames.
        """
        H, W, C = imgs[0].shape  # Get height and width from a single image
        mask_h, mask_w = self.mask_size

        # Iterate over all frames and apply masking with some probability
        for i, img in enumerate(imgs):
            if random.random() < self.p:
                # Slightly jitter the ball position to simulate slight movement
                jitter_x = random.randint(-2, 2)
                jitter_y = random.randint(-2, 2)
                x = int(ball_position_xy[0] + jitter_x)
                y = int(ball_position_xy[1] + jitter_y)

                # Ensure the mask is within the image boundaries
                top = max(0, min(H - mask_h, y - mask_h // 2))
                left = max(0, min(W - mask_w, x - mask_w // 2))

                # Apply the mask (set pixel values to 0)
                img[top:top + mask_h, left:left + mask_w, :] = 0

        return imgs, ball_position_xy