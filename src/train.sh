#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --job-name=Aug_occl
####SBATCH -w wulf-2gpu-vm02

nvidia-smi
export NCCL_P2P_DISABLE=1



# torchrun --nproc_per_node=2 main.py     \
#     --num_epochs 30   \
#     --saved_fn 'tracking_288_512_tracknetv2_epoch_30_TTA(5)'   \
#     --interval 1   \
#     --num_frames 5  \
#     --lr 1e-3 \
#     --weight_decay 1e-5 \
#     --loss_function BCE   \
#     --img_size 288 512 \
#     --batch_size 16 \
#     --print_freq 100 \
#     --dist_url 'env://' \
#     --dist_backend 'nccl' \
#     --multiprocessing_distributed \
#     --distributed \
#     --dataset_choice 'tta' \
#     --model_choice 'tracknetv2'  \
#     --occluded_prob 0 \
#     --ball_size 4 \
#     --no_test   \
#     --val-size 0.2 \
#     --pretrained_path '../checkpoints/normal_tracking_288_512_tracknetv2_tt(5)/normal_tracking_288_512_tracknetv2_tt(5)_best.pth' \

torchrun --nproc_per_node=3 main.py     \
    --num_epochs 30   \
    --saved_fn 'tracking_288_512_motion_light_TTA(5)_new_data'   \
    --num_frames 5  \
    --optimizer_type adamw  \
    --lr 5e-4 \
    --loss_function WBCE  \
    --weight_decay 5e-5 \
    --img_size 288 512 \
    --batch_size 24 \
    --print_freq 100 \
    --dist_url 'env://' \
    --dist_backend 'nccl' \
    --multiprocessing_distributed \
    --distributed \
    --dataset_choice 'tta' \
    --weighting_list 1 2 2 3   \
    --model_choice 'motion_light'  \
    --occluded_prob 0.1 \
    --ball_size 4 \
    --val-size 0.2 \
    --no_test   \
    # --pretrained_path '../checkpoints/normal_tracking_288_512_motion_light_tt(5)/normal_tracking_288_512_motion_light_tt(5)_best.pth' \


# torchrun --nproc_per_node=2 main.py     \
#     --num_epochs 20   \
#     --saved_fn 'tracking_288_512_motion_tennis(5)'   \
#     --num_frames 5  \
#     --loss_function WBCE  \
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
#     --dataset_choice 'tennis' \
#     --model_choice 'motion'  \
#     --occluded_prob 0.1 \
#     --ball_size 4 \
#     --no_test   \
#     --val-size 0.2 \


# torchrun --nproc_per_node=2 main.py     \
#     --num_epochs 50   \
#     --saved_fn 'tracking_288_512_TTNet_sigmoid_tta(5)'   \
#     --num_frames 5  \
#     --loss_function BCE  \
#     --optimizer_type adam  \
#     --lr 1e-4 \
#     --img_size 128 320 \
#     --batch_size 16 \
#     --print_freq 100 \
#     --dist_url 'env://' \
#     --dist_backend 'nccl' \
#     --multiprocessing_distributed \
#     --distributed \
#     --dataset_choice 'tta' \
#     --model_choice 'TTNet'  \
#     --ball_size 4 \
#     --no_test   \
#     --val-size 0.2 \
#     # --pretrained_path '../checkpoints/tracking_288_512_TTNet_tennis(5)/tracking_288_512_TTNet_tennis(5)_epoch_30.pth' \


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

