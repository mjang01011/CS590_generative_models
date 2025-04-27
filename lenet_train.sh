#!/bin/bash
#SBATCH -p gpu-common
#SBATCH --gres=gpu:1
#SBATCH --exclusive

#SBATCH --job-name=lenet5_train
#SBATCH --output=logs/lenet5_train_%j.out
#SBATCH --error=logs/lenet5_train_%j.err

# Make sure environment is clean from Miniconda
unset PYTHONHOME
unset PYTHONPATH

# Activate environment if needed
source ~/.bashrc

# ✅ Use system Python or known-good environment
/usr/bin/python lenet_train.py