import os
import json
import sys
import csv
import ast
from collections import Counter
from collections import defaultdict

import cv2
from sklearn.model_selection import train_test_split
import torch
import numpy as np

sys.path.append('../')


def load_raw_img(img_path):
    """Load raw image based on the path to the image"""
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)  # BGR --> RGB
    return img


def gaussian_1d(pos, muy, sigma):
    """Create 1D Gaussian distribution based on ball position (muy), and std (sigma)"""
    target = torch.exp(- (((pos - muy) / sigma) ** 2) / 2)
    return target


def create_target_ball(ball_position_xy, sigma, w, h, thresh_mask, device):
    """Create target for the ball detection stages

    :param ball_position_xy: Position of the ball (x,y)
    :param sigma: standard deviation (a hyperparameter)
    :param w: width of the resize image
    :param h: height of the resize image
    :param thresh_mask: if values of 1D Gaussian < thresh_mask --> set to 0 to reduce computation
    :param device: cuda() or cpu()
    :return:
    """
    w, h = int(w), int(h)
    target_ball_position = torch.zeros((w + h,), device=device)
    # Only do the next step if the ball is existed
    if (w > ball_position_xy[0] > 0) and (h > ball_position_xy[1] > 0):
        # For x
        x_pos = torch.arange(0, w, device=device)
        target_ball_position[:w] = gaussian_1d(x_pos, ball_position_xy[0], sigma=sigma)
        # For y
        y_pos = torch.arange(0, h, device=device)
        target_ball_position[w:] = gaussian_1d(y_pos, ball_position_xy[1], sigma=sigma)

        target_ball_position[target_ball_position < thresh_mask] = 0.

    return target_ball_position

def create_target_ball_right(ball_position_xy, sigma, w, h, thresh_mask, device):
    """Create target for the ball detection stages

    :param ball_position_xy: Position of the ball (x,y)
    :param sigma: standard deviation (a hyperparameter)
    :param w: width of the resize image
    :param h: height of the resize image
    :param thresh_mask: if values of 1D Gaussian < thresh_mask --> set to 0 to reduce computation
    :param device: cuda() or cpu()
    :return:
    """
    w, h = int(w), int(h)
    target_ball_position_x = torch.zeros(w, device=device)
    target_ball_position_y = torch.zeros(h, device=device)
    # Only do the next step if the ball is existed
    if (w > ball_position_xy[0] > 0) and (h > ball_position_xy[1] > 0):
        # For x
        x_pos = torch.arange(0, w, device=device)
        target_ball_position_x = gaussian_1d(x_pos, ball_position_xy[0], sigma=sigma)
        # For y
        y_pos = torch.arange(0, h, device=device)
        target_ball_position_y = gaussian_1d(y_pos, ball_position_xy[1], sigma=sigma)

        target_ball_position_x[target_ball_position_x < thresh_mask] = 0.
        target_ball_position_y[target_ball_position_y < thresh_mask] = 0.

    return target_ball_position_x, target_ball_position_y


def smooth_event_labelling(event_class, smooth_idx, event_frameidx):
    target_events = np.zeros((2,))
    if event_class < 2:
        n = smooth_idx - event_frameidx
        target_events[event_class] = np.cos(n * np.pi / 8)
        target_events[target_events < 0.01] = 0.
    return target_events


def get_events_infor(game_list, configs, dataset_type):
    """Get information of sequences of images based on events

    :param game_list: List of games (video names)
    :return:
    [
        each event: [[img_path_list], ball_position, target_events, segmentation_path]
    ]
    """
    # the paper mentioned 25, but used 9 frames only
    num_frames_from_event = int((configs.num_frames - 1) / 2)

    annos_dir = os.path.join(configs.dataset_dir, dataset_type, 'annotations')
    images_dir = os.path.join(configs.dataset_dir, dataset_type, 'images')
    events_infor = []
    events_labels = []
    for game_name in game_list:
        ball_annos_path = os.path.join(annos_dir, game_name, 'ball_markup.json')
        events_annos_path = os.path.join(annos_dir, game_name, 'events_markup.json')
        # Load ball annotations
        json_ball = open(ball_annos_path)
        ball_annos = json.load(json_ball)

        # Load events annotations
        json_events = open(events_annos_path)
        events_annos = json.load(json_events)
        for event_frameidx, event_name in events_annos.items():
            event_frameidx = int(event_frameidx)
            smooth_frame_indices = [event_frameidx]  # By default

            # smooth labeling 
            if (event_name != 'empty_event') and (configs.smooth_labelling):
                smooth_frame_indices = [idx for idx in range(event_frameidx - num_frames_from_event,
                                                             event_frameidx + num_frames_from_event + 1)]

            for smooth_idx in smooth_frame_indices:
                sub_smooth_frame_indices = [idx for idx in range(smooth_idx - num_frames_from_event,
                                                                 smooth_idx + num_frames_from_event + 1)]
                img_path_list = []
                for sub_smooth_idx in sub_smooth_frame_indices:
                    img_path = os.path.join(images_dir, game_name, 'img_{:06d}.jpg'.format(sub_smooth_idx))
                    img_path_list.append(img_path)

                last_f_idx = smooth_idx + num_frames_from_event
                # Get ball position for the last frame in the sequence
                if '{}'.format(last_f_idx) not in ball_annos.keys():
                    print('smooth_idx: {} - no ball position for the frame idx {}'.format(smooth_idx, last_f_idx))
                    continue
                ball_position_xy = ball_annos['{}'.format(last_f_idx)]
                ball_position_xy = np.array([ball_position_xy['x'], ball_position_xy['y']], dtype=int)
                # Ignore the event without ball information
                if (ball_position_xy[0] < 0) or (ball_position_xy[1] < 0):
                    continue

                # Get segmentation path for the last frame in the sequence
                seg_path = os.path.join(annos_dir, game_name, 'segmentation_masks', '{}.png'.format(last_f_idx))
                if not os.path.isfile(seg_path):
                    print("smooth_idx: {} - The segmentation path {} is invalid".format(smooth_idx, seg_path))
                    continue
                event_class = configs.events_dict[event_name]

                target_events = smooth_event_labelling(event_class, smooth_idx, event_frameidx)
                events_infor.append([img_path_list, ball_position_xy, target_events, seg_path])
                # Re-label if the event is neither bounce nor net hit
                if (target_events[0] == 0) and (target_events[1] == 0):
                    event_class = 2
                events_labels.append(event_class)

    return events_infor, events_labels



def get_events_infor_noseg(game_list, configs, dataset_type):
    """Get information of sequences of images based on events

    :param game_list: List of games (video names)
    :return:
    [
        each event: [[img_path_list], ball_position, target_events, segmentation_path]
    ]
    """
    # the paper mentioned 25, but used 9 frames only
    num_frames_from_event = int((configs.num_frames - 1) / 2)

    annos_dir = os.path.join(configs.dataset_dir, dataset_type, 'annotations')
    images_dir = os.path.join(configs.dataset_dir, dataset_type, 'images')
    events_infor = []
    events_labels = []
    for game_name in game_list:
        ball_annos_path = os.path.join(annos_dir, game_name, 'ball_markup.json')
        events_annos_path = os.path.join(annos_dir, game_name, 'events_markup.json')
        # Load ball annotations
        json_ball = open(ball_annos_path)
        ball_annos = json.load(json_ball)

        # Load events annotations
        json_events = open(events_annos_path)
        events_annos = json.load(json_events)
        for event_frameidx, event_name in events_annos.items():
            event_frameidx = int(event_frameidx)
            smooth_frame_indices = [event_frameidx]  # By default

            # smooth labeling 
            if (event_name != 'empty_event') and (configs.smooth_labelling):
                smooth_frame_indices = [idx for idx in range(event_frameidx - num_frames_from_event,
                                                             event_frameidx + num_frames_from_event + 1)]

            for smooth_idx in smooth_frame_indices:
                sub_smooth_frame_indices = [idx for idx in range(smooth_idx - num_frames_from_event,
                                                                 smooth_idx + num_frames_from_event + 1)]
                img_path_list = []
                for sub_smooth_idx in sub_smooth_frame_indices:
                    img_path = os.path.join(images_dir, game_name, 'img_{:06d}.jpg'.format(sub_smooth_idx))
                    img_path_list.append(img_path)

               
                # Get ball position for the last frame in the sequence
                if '{}'.format(smooth_idx) not in ball_annos.keys():
                    # print('smooth_idx: {} - no ball position for the frame idx {}'.format(smooth_idx, smooth_idx))
                    continue
                ball_position_xy = ball_annos['{}'.format(smooth_idx)]
                ball_position_xy = np.array([ball_position_xy['x'], ball_position_xy['y']], dtype=int)
                # Ignore the event without ball information
                if (ball_position_xy[0] < 0) or (ball_position_xy[1] < 0):
                    continue

                event_class = configs.events_dict[event_name]

                target_events = smooth_event_labelling(event_class, smooth_idx, event_frameidx)
                events_infor.append(img_path_list)
                # Re-label if the event is neither bounce nor net hit
                if (target_events[0] == 0) and (target_events[1] == 0):
                    event_class = 2
                events_labels.append([ball_position_xy, target_events, event_class])

    return events_infor, events_labels


def get_all_detection_infor_bidirect(game_list, configs, dataset_type):
    num_frames_from_event = (configs.num_frames - 1) // 2
    interval = configs.interval  # Get interval value from configs

    annos_dir = os.path.join(configs.dataset_dir, dataset_type, 'annotations')
    images_dir = os.path.join(configs.dataset_dir, dataset_type, 'images')
    events_infor = []
    events_labels = []
    skipped_frame = 0
    # Initialize a counter for occurrences of [-1, -1]
    invalid_count_tracker = defaultdict(int)

    def find_next_valid_frame(start_idx, game_name):
        """Find the next available frame file from the given start index."""
        while not os.path.exists(
                os.path.join(images_dir, game_name, f'img_{start_idx:06d}.jpg')):
            start_idx += 1  # Increment to the next frame index
        return start_idx

    for game_name in game_list:
        ball_annos_path = os.path.join(annos_dir, game_name, 'ball_markup.json')

        # Load ball annotations
        with open(ball_annos_path) as json_ball:
            ball_annos = json.load(json_ball)

        for ball_frameidx, ball_location in ball_annos.items():
            ball_frameidx = int(ball_frameidx)

            # Create frame indices with the correct interval
            sub_ball_frame_indices = [
                ball_frameidx + i * interval
                for i in range(-num_frames_from_event, num_frames_from_event + 1)
            ]

            img_path_list = []
            for idx in sub_ball_frame_indices:
                # Find the next valid frame if the current one doesn't exist
                valid_idx = find_next_valid_frame(idx, game_name)
                img_path = os.path.join(images_dir, game_name, f'img_{valid_idx:06d}.jpg')
                img_path_list.append(img_path)

            # Check if any valid frames were found
            if not img_path_list:
                print(f"No valid frames found for event at frame {ball_frameidx}.")
                continue

            # Check if the ball position exists for the target frame
            if str(ball_frameidx) not in ball_annos:
                print(f'No ball position for frame idx {ball_frameidx}.')
                continue

            ball_positions = []
            invalid_count = 0  # Track the number of [-1, -1] in this event
            for idx in sub_ball_frame_indices:
                # Find the next valid frame if the current one doesn't exist
                valid_idx = find_next_valid_frame(idx, game_name)
                if str(valid_idx) not in ball_annos:
                    ball_positions.append(np.array([-1, -1], dtype=int))
                    invalid_count += 1  # Increment invalid count
                else:
                    ball_position_xy = ball_annos[str(valid_idx)]
                    ball_position_xy = np.array([ball_position_xy['x'], ball_position_xy['y']], dtype=int)
                    ball_positions.append(ball_position_xy)

            # Track the number of invalid frames for this event
            invalid_count_tracker[invalid_count] += 1 
            # Check if the middle frame label is [-1, -1]
            middle_idx = len(ball_positions) // 2
            
            if (ball_positions[middle_idx] == np.array([-1, -1])).all():
                # print(f"Skipping event at frame {ball_frameidx} due to invalid middle label.")
                skipped_frame += 1
                continue  # Skip this event if the middle frame is invalid

            events_infor.append(img_path_list)
            events_labels.append(ball_positions)
    print(f"{skipped_frame} skipped frame due to due to invalid middle label")
    # Print the occurrences of [-1, -1] across events
    print("Count of [-1, -1] occurrences in events:")
    for count, occurrence in sorted(invalid_count_tracker.items()):
        print(f"{count} occurrence(s) of [-1, -1]: {occurrence} event(s)")
    return events_infor, events_labels



def get_all_detection_infor(game_list, configs, dataset_type):
    num_frames = configs.num_frames - 1

    annos_dir = os.path.join(configs.dataset_dir, dataset_type, 'annotations')
    images_dir = os.path.join(configs.dataset_dir, dataset_type, 'images')
    events_infor = []
    events_labels = []
    skipped_frame = 0

    for game_name in game_list:
        ball_annos_path = os.path.join(annos_dir, game_name, 'ball_markup.json')

        # Load ball annotations
        with open(ball_annos_path) as json_ball:
            ball_annos = json.load(json_ball)

        for ball_frameidx, ball_location in ball_annos.items():
            ball_frameidx = int(ball_frameidx)

            if configs.bidirect:
                middle_frame = num_frames // 2  # Middle frame index for bidirectional setting
                sub_ball_frame_indices = [
                    ball_frameidx - (middle_frame - i)  # Adjust to have the middle as key frame
                    for i in range(num_frames + 1)
                ]
            else:
                sub_ball_frame_indices = [
                    ball_frameidx - (num_frames - i)  # Adjust to have the last as key frame
                    for i in range(num_frames + 1)
                ]

            img_path_list = []
            for idx in sub_ball_frame_indices:
                img_path = os.path.join(images_dir, game_name, f'img_{idx:06d}.jpg')
                img_path_list.append(img_path)
        
            # Check if any valid frames were found
            if not img_path_list:
                print(f"No valid frames found for event at frame {ball_frameidx}.")
                continue

            # Check if the ball position exists for the target frame
            if str(ball_frameidx) not in ball_annos:
                print(f'No ball position for frame idx {ball_frameidx}.')
                continue
            
            ball_position = np.array([ball_location['x'], ball_location['y']], dtype=int)

            visibility = 1
            if (ball_position  == np.array([-1, -1])).all():
                # print(f"Skipping event at frame {ball_frameidx} due to invalid last label.")
                skipped_frame += 1
                continue  # Skip this event if the last frame is invalid

            events_infor.append(img_path_list)
            events_labels.append([ball_position, visibility])

    print(f"{skipped_frame} skipped frame due to due to invalid last label")

    return events_infor, events_labels


def get_all_detection_infor_tennis(game_list, configs):
    num_frames = configs.num_frames - 1

    dir = os.path.join(configs.tennis_dataset_dir)
    events_infor = []
    events_labels = []
    skipped_frame = 0

    for game_name in game_list:
        game_dir = os.path.join(dir, game_name)
        clips_list = [name for name in os.listdir(game_dir)]
        for clip_name in clips_list:
            clip_dir = os.path.join(game_dir, clip_name)
            ball_annos_path = os.path.join(clip_dir, 'Label.csv')

            # Load ball annotations from CSV
            ball_annos = []
            with open(ball_annos_path, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)  # Use DictReader to load as a list of dictionaries
                for row in csv_reader:
                    ball_annos.append(row)
            
            for row in ball_annos:
                file_name = row['file name']  # Assuming `file_name` is a column in the CSV
                visibility = int(row['visibility']) if row['visibility'] else -1  # Convert visibility to integer
                x = int(row['x-coordinate']) if row['x-coordinate'] else -1  # Default to -1 if empty Convert x-coordinate to integer
                y = int(row['y-coordinate']) if row['y-coordinate'] else -1  # Default to -1 if empty 
                status = int(row['status']) if row['status'] else -1

                # Extract the first four characters from file_name and convert to an integer
                ball_frameidx = int(file_name[:4])

                # Create frame indices with the correct interval, with the key frame as the last frame
                if configs.bidirect:
                    middle_frame = num_frames // 2  # Middle frame index for bidirectional setting
                    sub_ball_frame_indices = [
                        ball_frameidx - (middle_frame - i)  # Adjust to have the middle as key frame
                        for i in range(num_frames + 1)
                    ]
                else:
                    sub_ball_frame_indices = [
                        ball_frameidx - (num_frames - i)  # Adjust to have the last as key frame
                        for i in range(num_frames + 1)
                    ]


                img_path_list = []
                for idx in sub_ball_frame_indices:
                    img_path = os.path.join(clip_dir, f'{idx:04d}.jpg')
                    img_path_list.append(img_path)
                
                # Check if any valid frames were found
                if not img_path_list:
                    print(f"No valid frames found for event at frame {ball_frameidx}.")
                    continue


                ball_position = np.array([x, y], dtype=int)
                # if (ball_position  == np.array([-1, -1])).all():
                #     # print(f"Skipping event at frame {ball_frameidx} due to invalid last label.")
                #     skipped_frame += 1
                #     continue  # Skip this event if the last frame is invalid
               
                events_infor.append(img_path_list)
                events_labels.append([ball_position, visibility, status])

    print(f"{skipped_frame} skipped frame due to due to invalid last label")

    return events_infor, events_labels


def get_all_detection_infor_badminton(level_list, configs):
    num_frames = configs.num_frames - 1

    dir = os.path.join(configs.badminton_dataset_dir)
    events_infor = []
    events_labels = []
    skipped_frame = 0
    for level_name in level_list:
        level_dir = os.path.join(dir, level_name)
        games_list = [name for name in os.listdir(level_dir)]
        for game_name in games_list:
            game_dir = os.path.join(level_dir, game_name)
            images_dir = os.path.join(game_dir, 'images')
            clips_list = [name for name in os.listdir(images_dir)]
            for clip_name in clips_list:
                clip_dir = os.path.join(images_dir, clip_name)
                ball_annos_dir = os.path.join(game_dir, 'csv')
                
                file_path = os.path.join(ball_annos_dir, clip_name+'_ball.csv')
         
                ball_annos = []
                # Load ball annotations from CSV
                with open(file_path, mode='r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)  # Use DictReader to load as a list of dictionaries
                    for row in csv_reader:
                        ball_annos.append(row)
                
                for row in ball_annos:
                    frame = row['Frame']  # Assuming `file_name` is a column in the CSV
                
                    visibility = int(row['Visibility']) if row['Visibility'] else -1  # Convert visibility to integer
                    x = int(float(row['X'])) if row['X'] else -1  # Default to -1 if empty Convert x-coordinate to integer
                    y = int(float(row['Y'])) if row['Y'] else -1  # Default to -1 if empty 
                    status = -1
    

                    # Extract the first four characters from file_name and convert to an integer
                    ball_frameidx = int(frame)
        
                    sub_ball_frame_indices = [
                        ball_frameidx - (num_frames - i)  # Adjust to have the last as key frame
                        for i in range(num_frames + 1)
                    ]


                    img_path_list = []
    
                    for idx in sub_ball_frame_indices:
                        img_path = os.path.join(clip_dir, f'img_{idx:06d}.jpg')

                        img_path_list.append(img_path)
            
                    # Check if any valid frames were found
                    if not img_path_list:
                        print(f"No valid frames found for event at frame {ball_frameidx}.")
                        continue


                    ball_position = np.array([x, y], dtype=int)
 
                    events_infor.append(img_path_list)
                    events_labels.append([ball_position, visibility, status])

               
                
               

    print(f"{skipped_frame} skipped frame due to due to invalid last label")

    return events_infor, events_labels


def get_all_detection_infor_tta(configs, dataset_type):
    num_frames = configs.num_frames - 1
    annos_dir = os.path.join(configs.tta_dataset_dir, dataset_type, 'annotations')
    images_dir = os.path.join(configs.tta_dataset_dir, dataset_type, 'images')
    events_infor = []
    events_labels = []
    skipped_frame = 0

    for game_name in configs.tta_game_list:
        ball_annos_path = os.path.join(annos_dir, game_name, 'labels.csv')
        game_image_dir = os.path.join(images_dir, game_name)
        # Load ball annotations from CSV
        ball_annos = []
        with open(ball_annos_path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)  # Use DictReader to load as a list of dictionaries
            for row in csv_reader:
                ball_annos.append(row)
        
        for row in ball_annos:
            file_name = row['img']
            image_name = os.path.basename(file_name)  # get image name 
            visibility = int(row.get('visibility', [0])[0]) if row.get('visibility') and len(row['visibility']) > 0 else 0
            status = -1
            ball_annotation = row['kp-1']
            if ball_annotation == '':
                x = 0
                y = 0
            else:
                ball_annotation = ast.literal_eval(ball_annotation)[0]
                x = int(ball_annotation['x'] * ball_annotation['original_width'] / 100)  # Convert to pixel
                y = int(ball_annotation['y'] * ball_annotation['original_height'] / 100)  # Convert to pixel
            
            ball_frameidx = int(image_name[4:10])
            # Create frame indices with the correct interval, with the key frame as the last frame
            if configs.bidirect:
                middle_frame = num_frames // 2  # Middle frame index for bidirectional setting
                sub_ball_frame_indices = [
                    ball_frameidx - (middle_frame - i)  # Adjust to have the middle as key frame
                    for i in range(num_frames + 1)
                ]
            else:
                sub_ball_frame_indices = [
                    ball_frameidx - (num_frames - i)  # Adjust to have the last as key frame
                    for i in range(num_frames + 1)
                ]
        
            img_path_list = []
            for idx in sub_ball_frame_indices:
                img_path = os.path.join(game_image_dir, f'img_{idx:06d}.jpg')
                img_path_list.append(img_path)
            
            # Check if any valid frames were found
            if not img_path_list:
                print(f"No valid frames found for event at frame {ball_frameidx}.")
                continue

            ball_position = np.array([x, y], dtype=int)

            events_infor.append(img_path_list)
            events_labels.append([ball_position, visibility, status])

    print(f"{skipped_frame} skipped frame due to due to invalid last label")

    return events_infor, events_labels
 

def train_val_data_separation(configs):
    """Seperate data to training and validation sets"""
    if configs.dataset_choice == 'tt':
        dataset_type = 'training'
        if configs.event == True:
            events_infor, events_labels = get_events_infor_noseg(configs.train_game_list, configs, dataset_type)
        else:
            events_infor, events_labels = get_all_detection_infor(configs.train_game_list, configs, dataset_type)

        if configs.no_val:
            train_events_infor = events_infor
            train_events_labels = events_labels
            val_events_infor = None
            val_events_labels = None
        else:
            train_events_infor, val_events_infor, train_events_labels, val_events_labels = train_test_split(events_infor,
                                                                                                            events_labels,
                                                                                                            shuffle=True,
                                                                                                            test_size=configs.val_size,
                                                                                                            random_state=configs.seed,
                                                                                                            )
    elif configs.dataset_choice == 'tennis':
        events_infor, events_labels = get_all_detection_infor_tennis(configs.tennis_train_game_list, configs)
        if configs.no_val:
            train_events_infor = events_infor
            train_events_labels = events_labels
            val_events_infor = None
            val_events_labels = None
        else:
            train_events_infor, val_events_infor, train_events_labels, val_events_labels = train_test_split(events_infor,
                                                                                                            events_labels,
                                                                                                            shuffle=True,
                                                                                                            test_size=configs.val_size,
                                                                                                            random_state=configs.seed,
                                                                                                            )
    elif configs.dataset_choice == 'badminton':
        events_infor, events_labels = get_all_detection_infor_badminton(configs.badminton_train_game_list, configs)
        if configs.no_val:
            train_events_infor = events_infor
            train_events_labels = events_labels
            val_events_infor = None
            val_events_labels = None
        else:
            train_events_infor, val_events_infor, train_events_labels, val_events_labels = train_test_split(events_infor,
                                                                                                            events_labels,
                                                                                                            shuffle=True,
                                                                                                            test_size=configs.val_size,
                                                                                                            random_state=configs.seed,
                                                                                                            )
    elif configs.dataset_choice == 'tta':
        events_infor, events_labels = get_all_detection_infor_tta(configs, 'training')
        if configs.no_val:
            train_events_infor = events_infor
            train_events_labels = events_labels
            val_events_infor = None
            val_events_labels = None
        else:
            train_events_infor, val_events_infor, train_events_labels, val_events_labels = train_test_split(events_infor,
                                                                                                            events_labels,
                                                                                                            shuffle=True,
                                                                                                            test_size=configs.val_size,
                                                                                                            random_state=configs.seed,
                                                                                                            )
            
    return train_events_infor, val_events_infor, train_events_labels, val_events_labels

def get_visibility_distribution(events_labels):
    """
    Calculate the distribution of visibility labels in the dataset.

    Args:
        events_labels (list): A list of labels where each label is of the form:
                              [ball_position, visibility, status].

    Returns:
        dict: A dictionary with visibility levels as keys and their counts as values.
    """
    # Extract visibility values
    visibility_values = [label[1] for label in events_labels]
    
    # Count occurrences of each visibility level
    visibility_distribution = Counter(visibility_values)
    
    return dict(visibility_distribution)

if __name__ == '__main__':
    from config.config import parse_configs

    configs = parse_configs()
    configs.num_frames = 5
    configs.interval = 1
    configs.dataset_choice ='tta'
    # configs.event = True
    # configs.bidirect = True


    train_events_infor, val_events_infor, train_events_labels, val_events_labels = train_val_data_separation(configs)
    print(len(train_events_infor), len(train_events_labels), len(val_events_infor), len(val_events_labels))
    test_events_infor, test_events_labels = get_all_detection_infor_badminton(configs.badminton_test_game_list, configs)
    # print(len(test_events_infor))

    # dataset_type = 'test'
    # if configs.event:
    #     test_events_infor, test_events_labels = get_events_infor_noseg(configs.test_game_list, configs, dataset_type)
    # else:
    #     test_events_infor, test_events_labels = get_all_detection_infor(configs.test_game_list, configs, dataset_type)
    
    print(train_events_infor[30])
    print(train_events_labels[30])

    # Get distributions for train and validation datasets
    train_visibility_distribution = get_visibility_distribution(train_events_labels)
    val_visibility_distribution = get_visibility_distribution(val_events_labels)
    test_visibility_distribution = get_visibility_distribution(test_events_labels)

    # Print the results
    print("Train Visibility Distribution:", train_visibility_distribution)
    print("Validation Visibility Distribution:", val_visibility_distribution)
    print("Test", test_visibility_distribution)

    