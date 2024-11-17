#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --job-name=Aug_occl

# single node multiple gpu
torchrun --nproc_per_node=2 main.py     \
    --num_epochs 30   \
    --saved_fn 'normal_tracking_360_640_tracknetv2_tt'   \
    --interval 1   \
    --num_frames 5  \
    --lr 1e-3 \
    --img_size 360 640 \
    --batch_size 16 \
    --print_freq 100 \
    --dist_url 'env://' \
    --dist_backend 'nccl' \
    --multiprocessing_distributed \
    --distributed \
    --dataset_choice 'tt' \
    --model_choice 'tracknetv2'  \
    --occluded_prob 0 \
    --ball_size 1 \
    --val-size 0.2 \
    --no_test     \
    # --num_samples 100  \


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


### deformable attention detr 
# torchrun --nproc_per_node=2 main.py     \
#     --num_epochs 30   \
#     --saved_fn 'normal_tracking_270_480_wasb'   \
#     --backbone_choice 'test' \
#     --num_feature_levels 1  \
#     --interval 1   \
#     --num_frames 5  \
#     --lr 1e-4 \
#     --img_size 240 480 \
#     --num_queries 100    \
#     --batch_size 16 \
#     --transfromer_dmodel 512    \
#     --print_freq 100 \
#     --dist_url 'env://' \
#     --dist_backend 'nccl' \
#     --multiprocessing_distributed \
#     --distributed \
#     --dataset_choice 'tt' \
#     --occluded_prob 0 \
#     --ball_size 1 \
#     --no_test    \
#     # --num_samples 100  \
#     # --no_val    \