#!/bin/bash
#SBATCH -p ckpt-all
#SBATCH --job-name=train_curb_ramp_detector
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=100:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --requeue
#SBATCH --constraint='l40s|l40|a40|a100'

echo "Loading Conda environment..."
source activate sidewalkcv2

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

NPROC_PER_NODE=${SLURM_GPUS_PER_NODE:-4}

WORLD_SIZE=$(($SLURM_NNODES * $NPROC_PER_NODE))

echo "--- Job Configuration ---"
echo "Job ID: ${SLURM_JOBID}"
echo "Node List: ${SLURM_NODELIST}"
echo "Number of Nodes: ${SLURM_NNODES}"
echo "GPUs per Node: ${NPROC_PER_NODE}"
echo "Total Processes (World Size): ${WORLD_SIZE}"
echo "Master Node Address: ${MASTER_ADDR}"
echo "Master Port: ${MASTER_PORT}"
echo "CPUs per node task: ${SLURM_CPUS_PER_TASK}"
echo "Memory per node: ${SLURM_MEM_PER_NODE}"
echo "-------------------------"


echo "Launching torchrun..."
srun --export=ALL \
    torchrun --nnodes $SLURM_NNODES \
             --nproc_per_node $NPROC_PER_NODE \
             --rdzv_id $SLURM_JOB_ID \
             --rdzv_backend c10d \
             --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
             train.py

echo "--- Slurm job finished ---"
