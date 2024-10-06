#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --job-name=August

# single node multiple gpu
python main.py     \
    --lr 1e-2 \
    --img_size 270 480 \
    --batch_size 32 \
    --dist_url 'tcp://127.0.0.1:29500' \
    --dist_backend 'nccl' \
    --multiprocessing_distributed \
    --world_size 1 \
    --rank 0 \
    --distributed \
    --no-test \
    --print_freq 100

# single node single gpu, train data total length
# python main.py     \
#     --batch_size 1 \
#     --gpu_idx 0

