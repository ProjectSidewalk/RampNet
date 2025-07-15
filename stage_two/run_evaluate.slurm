#!/bin/bash
#SBATCH -p gpu-l40s
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err
#SBATCH --exclude=g3110,g3097

source activate sidewalkcv2

python -u evaluate.py
