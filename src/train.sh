#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --job-name=Aug_mask

# single node multiple gpu
python main.py     \
    --num_epochs 30   \
    --saved_fn 'mutli_frames_masked_train_270_480_proposed'   \
    --backbone_choice 'single' \
    --num_feature_levels 1  \
    --interval 10   \
    --num_frames 9  \
    --lr 1e-4 \
    --img_size 270 480 \
    --num_queries 50    \
    --batch_size 16 \
    --transfromer_dmodel 512    \
    --print_freq 50 \
    --dist_url 'tcp://127.0.0.1:29500' \
    --dist_backend 'nccl' \
    --multiprocessing_distributed \
    --world_size 1 \
    --rank 0 \
    --distributed \
    --no-val    \

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

