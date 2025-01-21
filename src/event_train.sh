#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --job-name=Aug_occl
#SBATCH -w boromir

nvidia-smi
export NCCL_P2P_DISABLE=1

torchrun --nproc_per_node=2 event_train.py     \
    --num_epochs 100   \
    --saved_fn 'Bounce_Detection_288_512_motion_light_opticalflow_event_bidirect_TTA(5)(2)_epoch_100'   \
    --num_frames 5  \
    --optimizer_type adamw  \
    --lr 1e-4 \
    --loss_function WBCE  \
    --weight_decay 5e-5 \
    --img_size 288 512 \
    --batch_size 16 \
    --print_freq 40 \
    --dist_url 'env://' \
    --dist_backend 'nccl' \
    --multiprocessing_distributed \
    --distributed \
    --dataset_choice 'tta' \
    --event \
    --bidirect \
    --weighting_list 1 2 2 3   \
    --model_choice 'motion_light_opticalflow'  \
    --occluded_prob 0 \
    --ball_size 4 \
    --val-size 0.2 \
    # --no_test   \