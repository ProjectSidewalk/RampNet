#!/bin/bash
#SBATCH -p gpu-l40s
#SBATCH --time=100:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --output=download_dataset.out
#SBATCH --error=download_dataset.err
#SBATCH --requeue
#SBATCH --exclude=g3110,g3097

source activate sidewalkcv2

python -u download_dataset.py
