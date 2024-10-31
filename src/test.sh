#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=August

python test.py \
  --working-dir '../' \
  --saved_fn 'normal_tracking_135_240_baseline' \
  --gpu_idx 0   \
  --num_queries 50    \
  --batch_size 16   \
  --transfromer_dmodel 512    \
  --img_size 135 240    \
  --num_frames 5  \
  --interval 1   \
  --occluded_prob 0 \
  --pretrained_path '../checkpoints/normal_tracking_135_240_baseline/normal_tracking_135_240_baseline_best.pth' \

