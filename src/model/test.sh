#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=August


# python temporal_transformer.py
# python propose_model.py
# python temporal_model.py
# python motion_model.py
python tracknet.py