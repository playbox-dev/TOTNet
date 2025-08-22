#!/bin/bash

cd "$(dirname "$0")" || exit
cd ../ || exit

nvidia-smi

# worked with RTX4080 Super 16GB
python src/main.py \
  --num_epochs 20 \
  --saved_fn 'TOTNet' \
  --num_frames 5 \
  --optimizer_type adamw \
  --lr 5e-4 \
  --loss_function WBCE \
  --weight_decay 5e-5 \
  --img_size 288 512 \
  --batch_size 3 \
  --print_freq 100 \
  --dataset_choice 'badminton' \
  --weighting_list 1 2 2 3 \
  --model_choice 'TOTNet' \
  --occluded_prob 0.1 \
  --ball_size 4 \
  --val-size 0.2 \
  --gpu_idx 0 \
  --no_test
