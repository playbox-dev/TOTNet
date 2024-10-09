#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=August

python test.py \
  --working-dir '../' \
  --saved_fn 'normal_train_540_960_single' \
  --gpu_idx 0   \
  --num_queries 50    \
  --batch_size 32    \
  --img_size 540 960    \
  --pretrained_path ../checkpoints/normal_train_540_960_single/normal_train_540_960_single_best.pth \

