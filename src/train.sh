#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --job-name=Aug_occl
#SBATCH -w wulf-2gpu-vm02



# torchrun --nproc_per_node=2 main.py     \
#     --num_epochs 30   \
#     --saved_fn 'normal_tracking_288_512_wasb_badminton(5)'   \
#     --interval 1   \
#     --num_frames 5  \
#     --lr 1e-3 \
#     --weight_decay 1e-5 \
#     --img_size 288 512 \
#     --batch_size 16 \
#     --print_freq 100 \
#     --dist_url 'env://' \
#     --dist_backend 'nccl' \
#     --multiprocessing_distributed \
#     --distributed \
#     --dataset_choice 'badminton' \
#     --model_choice 'wasb'  \
#     --occluded_prob 0 \
#     --ball_size 4 \
#     --no_test   \
#     --val-size 0.2 \


# torchrun --nproc_per_node=2 main.py     \
#     --num_epochs 30   \
#     --saved_fn 'normal_tracking_288_512_tracknetv2_badminton(5)'   \
#     --interval 1   \
#     --num_frames 5  \
#     --lr 1e-3 \
#     --weight_decay 1e-5 \
#     --img_size 288 512 \
#     --batch_size 16 \
#     --print_freq 100 \
#     --dist_url 'env://' \
#     --dist_backend 'nccl' \
#     --multiprocessing_distributed \
#     --distributed \
#     --dataset_choice 'badminton' \
#     --model_choice 'tracknetv2'  \
#     --occluded_prob 0 \
#     --ball_size 4 \
#     --no_test   \
#     --val-size 0.2 \


# torchrun --nproc_per_node=2 main.py     \
#     --num_epochs 20   \
#     --saved_fn 'normal_tracking_288_512_motion_light_badminton_weighted(5)(2)'   \
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
#     --dataset_choice 'badminton' \
#     --model_choice 'motion_light'  \
#     --occluded_prob 0.1 \
#     --ball_size 4 \
#     --no_test   \
#     --val-size 0.2 \



torchrun --nproc_per_node=2 main.py     \
    --num_epochs 20   \
    --saved_fn 'normal_tracking_288_512_motion_light_opticalflow_tt(5)'   \
    --interval 1   \
    --num_frames 5  \
    --optimizer_type adamw  \
    --lr 5e-4 \
    --weight_decay 5e-5 \
    --img_size 288 512 \
    --batch_size 16 \
    --print_freq 100 \
    --dist_url 'env://' \
    --dist_backend 'nccl' \
    --multiprocessing_distributed \
    --distributed \
    --dataset_choice 'tt' \
    --model_choice 'motion_light_opticalflow'  \
    --occluded_prob 0 \
    --ball_size 4 \
    --no_test   \
    --val-size 0.2 \




# # # single node multiple gpu
# torchrun --nproc_per_node=2 main.py     \
#     --num_epochs 30   \
#     --saved_fn 'normal_tracking_288_512_2sm_tennis(5)'   \
#     --interval 1   \
#     --num_frames 5  \
#     --lr 1e-3 \
#     --weight_decay 1e-5 \
#     --img_size 288 512 \
#     --batch_size 16 \
#     --print_freq 100 \
#     --dist_url 'env://' \
#     --dist_backend 'nccl' \
#     --multiprocessing_distributed \
#     --distributed \
#     --dataset_choice 'tennis' \
#     --model_choice 'two_stream_model'  \
#     --occluded_prob 0 \
#     --ball_size 4 \
#     --no_test   \
#     --val-size 0.2 \
#     # --num_samples 1000  \

# # # single node multiple gpu
# torchrun --nproc_per_node=2 main.py     \
#     --num_epochs 30   \
#     --saved_fn 'normal_tracking_288_512_2sm_tennis(7)'   \
#     --interval 1   \
#     --num_frames 7  \
#     --lr 1e-3 \
#     --weight_decay 1e-5 \
#     --img_size 288 512 \
#     --batch_size 16 \
#     --print_freq 100 \
#     --dist_url 'env://' \
#     --dist_backend 'nccl' \
#     --multiprocessing_distributed \
#     --distributed \
#     --dataset_choice 'tennis' \
#     --model_choice 'two_stream_model'  \
#     --occluded_prob 0 \
#     --ball_size 4 \
#     --no_test   \
#     --val-size 0.2 \
#     # --num_samples 1000  \




# torchrun --nproc_per_node=2 main.py     \
#     --num_epochs 30   \
#     --saved_fn 'normal_tracking_288_512_wasb_tt'   \
#     --interval 1   \
#     --num_frames 5  \
#     --lr 1e-3 \
#     --img_size 288 512 \
#     --batch_size 64 \
#     --print_freq 100 \
#     --dist_url 'env://' \
#     --dist_backend 'nccl' \
#     --multiprocessing_distributed \
#     --distributed \
#     --dataset_choice 'tt' \
#     --model_choice 'wasb'  \
#     --occluded_prob 0 \
#     --ball_size 1 \
#     --val-size 0.2 \
#     --no_test     \


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