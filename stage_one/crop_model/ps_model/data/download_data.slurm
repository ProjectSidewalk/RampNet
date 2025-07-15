#!/bin/bash
#SBATCH -p gpu-l40s            # Partition name
#SBATCH --time=100:00:00       # Time limit
#SBATCH --mem=50G            # Memory allocation
#SBATCH -c 16                # Number of CPU cores
#SBATCH --gpus=1             # Number of GPUs
#SBATCH -o slurm-%j.out      # Standard output log
#SBATCH -e slurm-%j.err      # Standard error log

source activate sidewalkcv2
python -u download_data.py
