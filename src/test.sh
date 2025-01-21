#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=August
##SBATCH -w boromir

#### legolas
#### boromir

nvidia-smi
export NCCL_P2P_DISABLE=1

python test.py \
  --working-dir '../' \
  --saved_fn 'tracking_288_512_motion_light_TTA(5)_new_data' \
  --model_choice 'motion_light'  \
  --gpu_idx 0   \
  --batch_size 8   \
  --img_size 288 512    \
  --num_frames 5  \
  --dataset_choice 'tta' \
  --ball_size 5 \
  --pretrained_path '../checkpoints/tracking_288_512_motion_light_TTA(5)_new_data/tracking_288_512_motion_light_TTA(5)_new_data_best.pth' \

# deformable 
# python test.py \
#   --working-dir '../' \
#   --saved_fn 'normal_tracking_360_640_wasb_tennis' \
#   --gpu_idx 0   \
#   --num_queries 100    \
#   --batch_size 8   \
#   --transfromer_dmodel 512    \
#   --img_size 360 640    \
#   --num_frames 3  \
#   --interval 1   \
#   --occluded_prob 0 \
#   --dataset_choice 'tennis' \
#   --ball_size 4 \
#   --pretrained_path '../checkpoints/normal_tracking_360_640_wasb_tennis/normal_tracking_360_640_wasb_tennis_best.pth' \