#!/bin/bash
#SBATCH -p gpu-l40s
#SBATCH --time=100:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --output=generate_dataset_meta.out
#SBATCH --error=generate_dataset_meta.err

source activate sidewalkcv2

python -u generate_dataset_meta.py
