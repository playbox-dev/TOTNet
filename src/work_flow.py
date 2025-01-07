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
from model.tracknet import build_TrackNetV2
from model.wasb import build_wasb
from model.motion_model_v3 import build_motion_model_light_opticalflow
from model.model_utils import load_pretrained_model
from model.TTNet import build_TTNet
from post_process.bounce_detection import Bounce_Detection
from post_process.table_detection import Table_ball_transform, read_img
from config.config import parse_configs
from data_process.video_loader import Video_Loader
from data_process.folder_loader import Folder_Loader

from losses_metrics.metrics import extract_coords, bounce_metrics


def extract_confidence_score(heatmap):
    """
    Extract confidence scores from the heatmap.

    Args:
        heatmap (tuple): A tuple of tensors in shape [(B, W), (B, H)] representing
                         the horizontal and vertical heatmaps for the batch.

    Returns:
        torch.Tensor: Confidence scores for the batch, one score per sample in the batch.
    """
    horizontal_heatmap, vertical_heatmap = heatmap  # Unpack the tuple

    # Get the maximum values (confidence) from both heatmaps
    horizontal_confidence, _ = horizontal_heatmap.max(dim=1)  # Max along the width dimension (W)
    vertical_confidence, _ = vertical_heatmap.max(dim=1)  # Max along the height dimension (H)

    # Combine the confidence scores (e.g., mean of the two heatmaps' max values)
    confidence_scores = (horizontal_confidence + vertical_confidence) / 2

    return confidence_scores


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
    scaled_ball_queue = deque(maxlen=10)  # Queue to store the last 10 scaled_ball_pos
    ball_queue = deque(maxlen=10)
    ball_bounce_list = []

    if configs.save_demo_output:
        configs.frame_dir = os.path.join(configs.save_demo_dir, 'frame')
        if not os.path.isdir(configs.frame_dir):
            os.makedirs(configs.frame_dir)

    configs.device = torch.device('cuda:{}'.format(configs.gpu_idx))


    # Model
    if configs.model_choice == 'motion_light':
        model = build_motion_model_light(configs)
    elif configs.model_choice == 'wasb':
        model = build_wasb(configs)
    elif configs.model_choice == 'motion_light_opticalflow':
        print("Building Motion Light Optical Flow model...")
        model = build_motion_model_light_opticalflow(configs)
    elif configs.model_choice == 'tracknetv2':
        model = build_TrackNetV2(configs)
    elif configs.model_choice == 'TTNet':
        print("Building TTNet")
        model = build_TTNet(configs)
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
            if configs.model_choice == 'wasb' or configs.model_choice == 'tracknetv2' or configs.model_choice == 'TTNet':
                B, N, C, H, W = batched_data.shape
                # Permute to bring frames and channels together
                batched_data = batched_data.permute(0, 2, 1, 3, 4).contiguous()  # Shape: [B, C, N, H, W]
                # Reshape to combine frames into the channel dimension
                batched_data = batched_data.view(B, N * C, H, W)  # Shape: [B, N*C, H, W]
            t1 = time.time()

            heatmap_output, pred_event = model(batched_data)
            t2 = time.time()

            confidence_scores = extract_confidence_score(heatmap_output)
            
            post_processed_coord = extract_coords(heatmap_output)
            
            x_pred, y_pred = post_processed_coord[0][0], post_processed_coord[0][1]
            ball_pos = (int(x_pred), int(y_pred))  # Ensure integer coordinates
            scaled_ball_pos = (int(ball_pos[0]*x_scale), int(ball_pos[1]*y_scale))
            print(f"ball position is {ball_pos}, scaled back pos is {scaled_ball_pos}")
            

            if confidence_scores>0.3:
                # Update the ball queue
                ball_queue.append(ball_pos)
                scaled_ball_queue.append(scaled_ball_pos)
            else:
                ball_queue.clear()
                scaled_ball_queue.clear()

            # Check for bounces
            bounces = bounce_detection.detect_bounce(list(scaled_ball_queue))
            if bounces:
                print(f"Bounces detected at positions: {bounces}, ball queue is {scaled_ball_queue}")
                last_bounce_index = bounces[-1]
                
                # filter out bounces thats too close to each other
                if not ball_bounce_list or frame_idx - ball_bounce_list[-1][0] > 10:
                    ball_bounce_pos = list(scaled_ball_queue)[last_bounce_index]
                    ball_bounce_list.append([frame_idx, ball_bounce_pos])
                    print(f"Frame {frame_idx}: Recorded bounce at {ball_bounce_pos}. Total bounces: {len(ball_bounce_list)}")
                    # Truncate the queue
                    if len(scaled_ball_queue) > last_bounce_index + 1:
                        scaled_ball_queue = deque(list(scaled_ball_queue)[last_bounce_index:], maxlen=10)

            events = pred_event.cpu().numpy() if pred_event is not None else (0.0, 0.0)

            if ball_queue:
                ploted_img = plot_detection_list(current_frame.copy(), list(ball_queue), events, bounces, scaled_ball_pos)
            else:
                ploted_img = current_frame.copy()
            
            ploted_img = cv2.cvtColor(ploted_img, cv2.COLOR_RGB2BGR)

            if configs.show_image:
                cv2.imshow('ploted_img.png', ploted_img)
                time.sleep(0.01)
            if configs.save_demo_output:
                cv2.imwrite(os.path.join(configs.frame_dir, '{:06d}.jpg'.format(frame_idx)), ploted_img)

            if count == 2000:
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

    return ball_bounce_list

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


def filter_bounces(ball_positions, frame_threshold=10):
    """
    Filters bounces to ensure there are no multiple bounces detected within close frame ranges.

    Args:
        ball_positions (list): List of tuples, where each tuple is (frame_id, (x, y)) representing
                               the frame ID and ball coordinates.
        frame_threshold (int): The minimum number of frames required between two bounces.

    Returns:
        list: A filtered list of bounces, keeping only one bounce per close range of frames.
    """
    if not ball_positions:
        return []

    # Sort ball positions by frame ID to ensure order
    ball_positions = sorted(ball_positions, key=lambda x: x[0])

    filtered_bounces = []
    last_kept_frame = -frame_threshold  # Initialize to a value far enough to allow the first bounce

    for frame_id, position in ball_positions:
        # Check if the current frame is far enough from the last kept bounce
        if frame_id - last_kept_frame >= frame_threshold:
            filtered_bounces.append((frame_id, position))
            last_kept_frame = frame_id  # Update the last kept bounce frame

    return filtered_bounces





def plot_detection_list(img, ball_positions, events, bounces, scaled_ball_pos):
    """Show the predicted information in the image"""
    if not isinstance(img, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")
    img = img.astype(np.uint8)

    # Draw all previous ball positions
    for pos in ball_positions:
        if pos != (0, 0):
            img = cv2.circle(img, pos, 3, (0, 255, 255), -1)  # Yellow for history positions

    # Highlight the current ball position
    if ball_positions[-1] != (0, 0):
        img = cv2.circle(img, ball_positions[-1], 5, (255, 0, 255), -1)  # Magenta for the latest position

    if bounces:
        text = f"Bounce detected at {scaled_ball_pos}"
        img = cv2.circle(img, ball_positions[bounces[-1]], 5, (0, 0, 255), -1)  # Magenta for the latest position
        img = cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return img



if __name__ == '__main__':
    start_time = time.time()  # Start the timer

    configs = parse_configs()
    ball_bounce_list = demo(configs=configs)
    print(ball_bounce_list)

    bounce_results = bounce_metrics(
        ball_bounce_list,
        "/home/s224705071/github/PhysicsInformedDeformableAttentionNetwork/data/tta_dataset/training/annotations/24Paralympics_FRA_F9_Lei_AUS_v_Xiong_CHN/labels.csv"
    )
    print(bounce_results)

    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    print(f"Process completed in {elapsed_time:.2f} seconds.")