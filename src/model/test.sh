#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=August

# python temporal_transformer.py
# python propose_model.py
# python temporal_model.py
# python motion_model.py --model_choice 'motion'
# python tracknet.py
# python mamba_model.py  --model_choice 'mamba'
# python two_stream_network.py --model_choice 'two_stream_model'
python sequential_model.py --model_choice 'motion'