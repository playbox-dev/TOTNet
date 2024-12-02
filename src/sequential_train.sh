#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --job-name=Aug_occl


torchrun --nproc_per_node=2 sequential_train.py     \
    --num_epochs 20   \
    --saved_fn 'test'   \
    --interval 1   \
    --num_frames 5  \
    --optimizer_type adamw  \
    --lr 5e-4 \
    --weight_decay 5e-5 \
    --img_size 288 512 \
    --batch_size 1 \
    --print_freq 100 \
    --dist_url 'env://' \
    --dist_backend 'nccl' \
    --multiprocessing_distributed \
    --distributed \
    --dataset_choice 'tennis' \
    --model_choice 'motion_sequential'  \
    --occluded_prob 0 \
    --ball_size 4 \
    --no_test   \
    --val-size 0.2 \