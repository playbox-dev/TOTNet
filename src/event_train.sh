#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --job-name=Aug_occl
#SBATCH -w wulf-2gpu-vm02


# first train a ball tarcking model withoout events
# torchrun --nproc_per_node=2 event_train.py     \
#     --num_epochs 2   \
#     --saved_fn 'normal_tracking_288_512_motion_tt_events(5)_stage1'   \
#     --interval 1   \
#     --num_frames 5  \
#     --optimizer_type adamw  \
#     --lr 5e-4 \
#     --weight_decay 5e-5 \
#     --img_size 288 512 \
#     --batch_size 16 \
#     --print_freq 100 \
#     --dist_url 'env://' \
#     --dist_backend 'nccl' \
#     --multiprocessing_distributed \
#     --distributed \
#     --dataset_choice 'tt' \
#     --model_choice 'motion'  \
#     --bidirect \
#     --occluded_prob 0 \
#     --ball_size 4 \
#     --no_test   \
#     --val-size 0.2 \


# Finetune on previous
torchrun --nproc_per_node=2 event_train.py     \
    --num_epochs 20   \
    --saved_fn 'normal_tracking_288_512_motion_tt_events(5)_stage2(2)'   \
    --interval 1   \
    --num_frames 5  \
    --optimizer_type adamw  \
    --lr 5e-4 \
    --weight_decay 5e-5 \
    --img_size 288 512 \
    --batch_size 16 \
    --print_freq 50 \
    --dist_url 'env://' \
    --dist_backend 'nccl' \
    --multiprocessing_distributed \
    --distributed \
    --dataset_choice 'tt' \
    --event \
    --smooth_labelling \
    --model_choice 'motion'  \
    --occluded_prob 0 \
    --ball_size 4 \
    --no_test   \
    --val-size 0.2 \
    --pretrained_path '../checkpoints/normal_tracking_288_512_motion_tt_events(5)_stage1/normal_tracking_288_512_motion_tt_events(5)_stage1_best.pth' \