#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=August
#SBATCH -w legolas

nvidia-smi
export NCCL_P2P_DISABLE=1

python event_test.py \
  --working-dir '../' \
  --saved_fn 'Bounce_Detection_288_512_motion_light_event_bidirect_TTA(5)(2)_epoch_100' \
  --model_choice 'motion_light'  \
  --gpu_idx 0   \
  --batch_size 8   \
  --img_size 288 512    \
  --num_frames 5  \
  --occluded_prob 0 \
  --dataset_choice 'tta' \
  --event \
  --bidirect \
  --test  \
  --ball_size 5 \
  --pretrained_path '../checkpoints/Bounce_Detection_288_512_motion_light_event_bidirect_TTA(5)(2)_epoch_100/Bounce_Detection_288_512_motion_light_event_bidirect_TTA(5)(2)_epoch_100_epoch_42.pth' \