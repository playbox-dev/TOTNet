#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=August_demo
#SBATCH -w boromir

#tt video path 
# /home/s224705071/github/TT/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch/dataset/test/videos/test_1.mp4
# tt training video path
# /home/s224705071/github/TT/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch/dataset/training/videos/game_1.mp4
# tennis video path 
# /home/s224705071/github/PhysicsInformedDeformableAttentionNetwork/data/tennis_data/game1/Clip1

# badminton 
# /home/s224705071/github/PhysicsInformedDeformableAttentionNetwork/data/badminton/TrackNetV2/Professional/match1/video/1_01_00.mp4
python demo.py \
    --save_demo_output    \
    --output_format video \
    --gpu_idx 0   \
    --model_choice 'motion_light' \
    --num_frames 5  \
    --dataset_choice tt \
    --video_path '/home/s224705071/github/PhysicsInformedDeformableAttentionNetwork/data/tta_dataset/videos/24Paralympics_FRA_F9_Lei_AUS_v_Xiong_CHN.MP4' \
    --pretrained_path '../checkpoints/normal_tracking_288_512_motion_light_tt(5)/normal_tracking_288_512_motion_light_tt(5)_best.pth' \
