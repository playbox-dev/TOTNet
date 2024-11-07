#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --job-name=Aug_occl

# single node multiple gpu
python main.py     \
    --num_epochs 30   \
    --saved_fn 'normal_tracking_135_240_proposed_motion(2)'   \
    --backbone_choice 'single' \
    --num_feature_levels 1  \
    --interval 1   \
    --num_frames 5  \
    --lr 1e-4 \
    --img_size 135 240 \
    --num_queries 100    \
    --batch_size 32 \
    --transfromer_dmodel 512    \
    --print_freq 50 \
    --dist_url 'tcp://127.0.0.1:29500' \
    --dist_backend 'nccl' \
    --multiprocessing_distributed \
    --world_size 1 \
    --rank 0 \
    --distributed \
    --occluded_prob 0 \
    --no_test    \
    # --num_samples 500  \
    # --no_val    \

# single node single gpu, train data total length
# python main.py     \
#     --num_epochs 30   \
#     --saved_fn 'single_frame_train_128_320_single'   \
#     --backbone_choice 'single' \
#     --num_feature_levels 1  \
#     --num_frames 1  \
#     --lr 1e-4 \
#     --img_size 128 320 \
#     --num_queries 50    \
#     --batch_size 64 \
#     --transfromer_dmodel 512    \
#     --no-test \
#     --print_freq 50 \
#     --gpu_idx 0 \

