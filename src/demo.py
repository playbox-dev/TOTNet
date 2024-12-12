import os
import sys
from collections import deque
import subprocess

import cv2
import numpy as np
import torch
import time

sys.path.append('./')

from data_process.video_loader import Video_Loader
from data_process.folder_loader import Folder_Loader
from model.model_utils import load_pretrained_model
from model.motion_model_light import build_motion_model_light
from model.motion_model import build_motion_model
from model.motion_model_v3 import build_motion_model_light_opticalflow
from model.tracknet import build_TrackNetV2
from model.wasb import build_wasb
from config.config import parse_configs
from utils.misc import time_synchronized
from losses_metrics.metrics import extract_coords


def demo(configs):

    if configs.dataset_choice == 'tt' or configs.dataset_choice == 'badminton' or configs.dataset_choice == 'tta':
        data_loader = Video_Loader(configs.video_path, configs.img_size, configs.num_frames)
    elif configs.dataset_choice == 'tennis':
        data_loader = Folder_Loader(configs.video_path, configs.img_size, configs.num_frames)
    

    if configs.save_demo_output:
        configs.frame_dir = os.path.join(configs.save_demo_dir, 'frame')
        if not os.path.isdir(configs.frame_dir):
            os.makedirs(configs.frame_dir)

    configs.device = torch.device('cuda:{}'.format(configs.gpu_idx))

    # Model
    if configs.model_choice == 'motion_light':
        model = build_motion_model_light(configs)
    elif configs.model_choice == 'motion':
        model = build_motion_model(configs)
    elif configs.model_choice == 'wasb':
        model = build_wasb(configs)
    elif configs.model_choice == 'motion_light_opticalflow':
        print("Building Motion Light Optical Flow model...")
        model = build_motion_model_light_opticalflow(configs)
    elif configs.model_choice == 'tracknetv2':
        model = build_TrackNetV2(configs)
    else:
        raise ValueError(f"Unknown model choice: {configs.model_choice}")
    model.cuda()


    assert configs.pretrained_path is not None, "Need to load the pre-trained model"
    try:
        model = load_pretrained_model(model, configs.pretrained_path, configs.gpu_idx)
        print(f"Model loaded successfully from {configs.pretrained_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

    model.eval()
    frame_idx = int(configs.num_frames - 1)

    with torch.no_grad():
        for count, resized_imgs, current_frame in data_loader:
            resized_imgs = torch.from_numpy(resized_imgs).to(configs.device, non_blocking=True).float().unsqueeze(0)
            batched_data = resized_imgs
            t1 = time.time()

            if configs.model_choice == 'wasb' or configs.model_choice == 'tracknetv2':
                B, N, C, H, W = batched_data.shape
                # Permute to bring frames and channels together
                batched_data = batched_data.permute(0, 2, 1, 3, 4).contiguous()  # Shape: [B, C, N, H, W]
                # Reshape to combine frames into the channel dimension
                batched_data = batched_data.view(B, N * C, H, W)  # Shape: [B, N*C, H, W]
   
            heatmap_output, pred_event = model(batched_data)
            t2 = time.time()
          

            post_processed_coord = extract_coords(heatmap_output)
            
            x_pred, y_pred = post_processed_coord[0][0], post_processed_coord[0][1]
            ball_pos = (int(x_pred), int(y_pred))  # Ensure integer coordinates
            print(ball_pos)

          
            events = pred_event.cpu().numpy() if pred_event is not None else (0.0, 0.0)

            ploted_img = plot_detection(current_frame.copy(), ball_pos, events)

            ploted_img = cv2.cvtColor(ploted_img, cv2.COLOR_RGB2BGR)
            if configs.show_image:
                cv2.imshow('ploted_img.png', ploted_img)
                time.sleep(0.01)
            if configs.save_demo_output:
                cv2.imwrite(os.path.join(configs.frame_dir, '{:06d}.jpg'.format(frame_idx)), ploted_img)

            if count == 3000:
                break

            frame_idx += 1
            print('Done frame_idx {} - time {:.3f}s'.format(frame_idx, t2 - t1))

    if configs.output_format == 'video':
        output_video_path = os.path.join(configs.save_demo_dir, 'result.mp4')
        frames_dir = configs.frame_dir

        print(f"Frames directory: {frames_dir}")
        if not os.path.isdir(frames_dir):
            print(f"Error: Frame directory {frames_dir} does not exist!")
        else:
            print(f"Frame files: {os.listdir(frames_dir)}")

        if not os.path.isdir(configs.save_demo_dir):
            print(f"Output directory does not exist. Creating: {configs.save_demo_dir}")
            os.makedirs(configs.save_demo_dir)

        cmd_str = f'ffmpeg -f image2 -i {frames_dir}/%06d.jpg -b:v 5000k -c:v mpeg4 {output_video_path}'
        print(f"Running ffmpeg command: {cmd_str}")
        process = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)

        if process.returncode != 0:
            print("Error: ffmpeg command failed.")
            print(f"ffmpeg stdout: {process.stdout}")
            print(f"ffmpeg stderr: {process.stderr}")
        else:
            if os.path.isfile(output_video_path):
                print(f"Video saved at: {output_video_path}")
            else:
                print(f"Error: Video file not found at {output_video_path}")


def plot_detection(img, ball_pos, events):
    """Show the predicted information in the image"""
    if not isinstance(img, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")
    img = img.astype(np.uint8)
    if ball_pos != (0, 0):
        img = cv2.circle(img, ball_pos, 5, (255, 0, 255), -1)
    # event_name = 'is bounce: {:.2f}, is net: {:.2f}'.format(events[0], events[1])
    # img = cv2.putText(img, event_name, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    return img



if __name__ == '__main__':
    configs = parse_configs()
    demo(configs=configs)
