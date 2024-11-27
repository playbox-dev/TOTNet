#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=August
#SBATCH -w legolas


python event_test.py \
  --working-dir '../' \
  --saved_fn 'normal_tracking_288_512_motion_tt_events(5)_stage2(2)' \
  --model_choice 'motion'  \
  --gpu_idx 0   \
  --batch_size 8   \
  --img_size 288 512    \
  --num_frames 5  \
  --interval 1   \
  --occluded_prob 0 \
  --dataset_choice 'tt' \
  --event \
  --smooth_labelling \
  --ball_size 5 \
  --pretrained_path '../checkpoints/normal_tracking_288_512_motion_tt_events(5)_stage2(2)/normal_tracking_288_512_motion_tt_events(5)_stage2(2)_epoch_14.pth' \