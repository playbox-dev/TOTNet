import os
from glob import glob

import cv2


def make_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)


def extract_images_from_videos(video_path, out_images_dir):
    video_fn = os.path.basename(video_path)[:-4]
    sub_images_dir = os.path.join(out_images_dir, video_fn)

    make_folder(sub_images_dir)

    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    n_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    f_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Processing video: {video_fn}.mp4')
    print(f'Number of frames: {n_frames}, Width: {f_width}, Height: {f_height}')

    frame_idx = 0
    while frame_idx < n_frames:
        ret, img = video_cap.read()
        if not ret:
            print(f"Warning: Failed to read frame {frame_idx} from video {video_path}")
            break

        image_path = os.path.join(sub_images_dir, f'img_{frame_idx:06d}.jpg')
        if os.path.isfile(image_path):
            # Image already exists, skip writing but continue extracting
            print(f"Frame {frame_idx} already exists. Skipping...")
        else:
            success = cv2.imwrite(image_path, img)
            if not success:
                print(f"Error: Failed to write frame {frame_idx} to {image_path}")
                # Optionally, you can choose to break or continue based on your needs
                # break

        frame_idx += 1

    video_cap.release()
    print(f'Done extracting frames from: {video_path}')


if __name__ == '__main__':
    dataset_dir = '../data/badminton/TrackNetV2'
    for dataset_type in ['Amateur', 'Professional', 'Test']:
        video_dir = os.path.join(dataset_dir, dataset_type)

        # Recursively iterate through all match folders
        for root, dirs, _ in os.walk(video_dir):
            if 'video' in dirs:  # Check if the `video` folder exists in the current directory
                video_folder = os.path.join(root, 'video')
                video_paths = glob(os.path.join(video_folder, '*.mp4'))  # Find all videos in the `video` folder
           
                # Skip if no videos are found
                if not video_paths:
                    continue

                # Create output images directory for the current match
                relative_path = os.path.relpath(root, video_dir)  # Relative path of the match folder
                out_images_dir = os.path.join(root, 'images')  # Same level as `video`
                os.makedirs(out_images_dir, exist_ok=True)

                # Process each video in the `video` folder
                for video_idx, video_path in enumerate(video_paths):
                    extract_images_from_videos(video_path, out_images_dir)