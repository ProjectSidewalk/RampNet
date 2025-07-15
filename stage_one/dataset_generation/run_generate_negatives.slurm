#!/bin/bash
#SBATCH -p ckpt
#SBATCH --time=100:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --output=generate_negatives.out
#SBATCH --error=generate_negatives.err
#SBATCH --requeue

source activate sidewalkcv2

python -u generate_negative_panos.py
