#!/bin/bash
#SBATCH -p gpu-common
#SBATCH --gres=gpu:1
#SBATCH --exclusive

#SBATCH --job-name=lenet5_train
#SBATCH --output=logs/latent_ddpm_hf_search%j.out
#SBATCH --error=logs/latent_ddpm_hf_search%j.err

# Make sure environment is clean from Miniconda
unset PYTHONHOME
unset PYTHONPATH

# Activate environment if needed
source ~/.bashrc

# ✅ Use system Python or known-good environment
/usr/bin/python latent_ddpm_hf_search.py