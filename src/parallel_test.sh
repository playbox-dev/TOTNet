#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --job-name=Aug_occl



# print GPU information 
nvidia-smi
export NCCL_P2P_DISABLE=1
# Set environment variables for distributed training
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500


CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS torchrun --nproc_per_node=2 parallel_test.py     \
    --num_epochs 30   \
    --saved_fn 'tracking_288_512_motion_light_TTA(5)_new_data'   \
    --num_frames 5  \
    --optimizer_type adamw  \
    --lr 5e-4 \
    --loss_function WBCE  \
    --weight_decay 5e-5 \
    --img_size 288 512 \
    --batch_size 18 \
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
