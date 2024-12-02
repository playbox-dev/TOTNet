import os
import sys
from collections import deque

import cv2
import numpy as np
import torch
import time

sys.path.append('./')

from data_process.video_loader import Video_Loader
from model.model_utils import load_pretrained_model
from model.motion_model_light import build_motion_model_light
from model.motion_model import build_motion_model
from model.motion_model_v3 import build_motion_model_light_opticalflow
from config.config import parse_configs
from utils.misc import time_synchronized
from losses_metrics.metrics import extract_coords


def demo(configs):
    video_loader = Video_Loader(configs.video_path, configs.img_size, configs.num_frames)
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
    elif configs.model_choice == 'motion_light_opticalflow':
        print("Building Motion Light Optical Flow model...")
        model = build_motion_model_light_opticalflow(configs)
    else:
        raise ValueError(f"Unknown model choice: {configs.model_choice}")
    model.cuda()


    assert configs.pretrained_path is not None, "Need to load the pre-trained model"
    model = load_pretrained_model(model, configs.pretrained_path, configs.gpu_idx)

    model.eval()
    frame_idx = int(configs.num_frames - 1)

    with torch.no_grad():
        for count, resized_imgs in video_loader:
            resized_imgs = torch.from_numpy(resized_imgs).to(configs.device, non_blocking=True).float().unsqueeze(0)
            t1 = time.time()
   
            heatmap_output, pred_event = model(resized_imgs)
            t2 = time.time()
          

            post_processed_coord = extract_coords(heatmap_output)
            
            x_pred, y_pred = post_processed_coord[0][0], post_processed_coord[0][1]
            ball_pos = (int(x_pred), int(y_pred))  # Ensure integer coordinates
            print(ball_pos)

            # resized_imgs [B, N, C, H, W]
            last_frame = resized_imgs[0, -1, :, :, :].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            
          
            events = pred_event.cpu().numpy() if pred_event is not None else (0.0, 0.0)

            ploted_img = plot_detection(last_frame.copy(), ball_pos, events)

            ploted_img = cv2.cvtColor(ploted_img, cv2.COLOR_RGB2BGR)
            if configs.show_image:
                cv2.imshow('ploted_img.png', ploted_img)
                time.sleep(0.01)
            if configs.save_demo_output:
                cv2.imwrite(os.path.join(configs.frame_dir, '{:06d}.jpg'.format(frame_idx)), ploted_img)

            frame_idx += 1
            print('Done frame_idx {} - time {:.3f}s'.format(frame_idx, t2 - t1))

    if configs.output_format == 'video':
        output_video_path = os.path.join(configs.save_demo_dir, 'result.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%06d.jpg -b:v 5000k -c:v mpeg4 {}'.format(
            os.path.join(configs.frame_dir), output_video_path)
        os.system(cmd_str)


def plot_detection(img, ball_pos, events):
    """Show the predicted information in the image"""
    if not isinstance(img, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")
    img = img.astype(np.uint8)
    img = cv2.circle(img, ball_pos, 5, (255, 0, 255), -1)
    # event_name = 'is bounce: {:.2f}, is net: {:.2f}'.format(events[0], events[1])
    # img = cv2.putText(img, event_name, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    return img



if __name__ == '__main__':
    configs = parse_configs()
    demo(configs=configs)
