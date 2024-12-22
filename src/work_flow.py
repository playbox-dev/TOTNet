import torch 
import os
import sys
import time
import cv2
import numpy as np
import subprocess
from collections import deque
sys.path.append('./')

from model.motion_model_light import build_motion_model_light
from model.model_utils import load_pretrained_model
from post_process.bounce_detection import Bounce_Detection
from post_process.table_detection import Table_ball_transform, read_img
from config.config import parse_configs
from data_process.video_loader import Video_Loader
from data_process.folder_loader import Folder_Loader

from losses_metrics.metrics import extract_coords

def demo(configs):

    if configs.dataset_choice == 'tt' or configs.dataset_choice == 'badminton' or configs.dataset_choice == 'tta':
        data_loader = Video_Loader(configs.video_path, configs.img_size, configs.num_frames)
    elif configs.dataset_choice == 'tennis':
        data_loader = Folder_Loader(configs.video_path, configs.img_size, configs.num_frames)

    x_scale, y_scale = 1920/512, 1020/288

    # output_folder = "/home/august/github/PhysicsInformedDeformableAttentionNetwork/results/demo/logs/output"
    # image = read_img("/home/august/github/PhysicsInformedDeformableAttentionNetwork/data/tta_dataset/images/img_000000.jpg")
    # table_ball_transform = Table_ball_transform(output_folder=output_folder, table_image=image) 
    table_corners = [(734, 397), (1119, 399), (1150, 581), (742, 577)]
    bounce_detection = Bounce_Detection(table_corners)
    ball_queue = deque(maxlen=10)  # Queue to store the last 10 scaled_ball_pos

    if configs.save_demo_output:
        configs.frame_dir = os.path.join(configs.save_demo_dir, 'frame')
        if not os.path.isdir(configs.frame_dir):
            os.makedirs(configs.frame_dir)

    configs.device = torch.device('cuda:{}'.format(configs.gpu_idx))

    # Model

    model = build_motion_model_light(configs)

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

            heatmap_output, pred_event = model(batched_data)
            t2 = time.time()
          

            post_processed_coord = extract_coords(heatmap_output)
            
            x_pred, y_pred = post_processed_coord[0][0], post_processed_coord[0][1]
            ball_pos = (int(x_pred), int(y_pred))  # Ensure integer coordinates
            scaled_ball_pos = (int(ball_pos[0]*x_scale), int(ball_pos[1]*y_scale))
            print(f"ball position is {ball_pos}, scaled back pos is {scaled_ball_pos}")

            # Update the ball queue
            ball_queue.append(scaled_ball_pos)

            # Check for bounces
            bounces = bounce_detection.bounce_detection(ball_queue)
            if bounces:
                print(f"Bounces detected at positions: {bounces}, ball queue is {ball_queue}")
                last_bounce_index = bounces[-1]
                ball_queue = deque(list(ball_queue)[last_bounce_index:], maxlen=10)

            events = pred_event.cpu().numpy() if pred_event is not None else (0.0, 0.0)

            ploted_img = plot_detection(current_frame.copy(), ball_pos, events, bounces, scaled_ball_pos)

            ploted_img = cv2.cvtColor(ploted_img, cv2.COLOR_RGB2BGR)
            if configs.show_image:
                cv2.imshow('ploted_img.png', ploted_img)
                time.sleep(0.01)
            if configs.save_demo_output:
                cv2.imwrite(os.path.join(configs.frame_dir, '{:06d}.jpg'.format(frame_idx)), ploted_img)

            if count == 1000:
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


def plot_detection(img, ball_pos, events, bounces, scaled_ball_pos):
    """Show the predicted information in the image"""
    if not isinstance(img, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")
    img = img.astype(np.uint8)
    if ball_pos != (0, 0):
        img = cv2.circle(img, ball_pos, 5, (255, 0, 255), -1)
    if bounces:
        text = f"Bounce detected at {scaled_ball_pos}"
        img = cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return img



if __name__ == '__main__':
    configs = parse_configs()
    demo(configs=configs)
