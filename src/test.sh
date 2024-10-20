#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=August

python test.py \
  --working-dir '../' \
  --saved_fn 'occluded_mutliframes_train_270_480_motion_2' \
  --gpu_idx 0   \
  --num_queries 50    \
  --batch_size 16   \
  --transfromer_dmodel 256    \
  --img_size 270 480    \
  --num_frames 5  \
  --interval 5   \
  --pretrained_path ../checkpoints/occluded_mutliframes_train_270_480_motion_2/occluded_mutliframes_train_270_480_motion_2_best.pth \

