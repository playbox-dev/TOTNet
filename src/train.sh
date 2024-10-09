#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --job-name=August

# single node multiple gpu
python main.py     \
    --num_epochs 30   \
    --saved_fn 'normal_train_540_960_multi'   \
    --backbone_choice 'multi' \
    --num_feature_levels 4  \
    --lr 1e-4 \
    --img_size 540 960 \
    --num_queries 50    \
    --batch_size 32 \
    --transfromer_dmodel 512    \
    --dist_url 'tcp://127.0.0.1:29500' \
    --dist_backend 'nccl' \
    --multiprocessing_distributed \
    --world_size 1 \
    --rank 0 \
    --distributed \
    --no-test \
    --print_freq 20

# single node single gpu, train data total length
# python main.py     \
#     --batch_size 1 \
#     --gpu_idx 0

