#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=August

python test.py \
  --working-dir '../' \
  --saved_fn 'mutli_frames_masked_train_270_480_single' \
  --gpu_idx 0   \
  --num_queries 50    \
  --batch_size 32   \
  --img_size 270 480    \
  --num_frames 9  \
  --pretrained_path ../checkpoints/mutli_frames_masked_train_270_480_single/mutli_frames_masked_train_270_480_single_best.pth \

