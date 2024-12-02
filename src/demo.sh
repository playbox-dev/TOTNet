#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=August
#SBATCH -w boromir

#tt video path 
# /home/s224705071/github/TT/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch/dataset/test/videos/test_1.mp4
python demo.py \
    --save_demo_output    \
    --gpu_idx 0   \
    --model_choice motion \
    --num_frames 5  \
    --video_path '/home/s224705071/github/TT/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch/dataset/test/videos/test_1.mp4' \
    --pretrained_path '../checkpoints/normal_tracking_288_512_motion_tt(5)/normal_tracking_288_512_motion_tt(5)_best.pth' \
