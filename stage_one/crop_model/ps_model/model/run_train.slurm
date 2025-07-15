#!/bin/bash
#SBATCH -p gpu-l40s
#SBATCH --time=100:00:00
#SBATCH --mem=50G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --output=train.out
#SBATCH --error=train.err

source activate sidewalkcv2

python -u train.py
