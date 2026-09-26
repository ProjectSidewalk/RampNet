#!/bin/bash
# The resolution arms' crops for the #86 context experiment (docs/context_fov_86.md section 4.2):
# `context_fov_86.sh downsample` on a klone CPU allocation, not a login node (klone reaps heavy
# login-node processes). Reads the fov25 crops in $WORK/crops and writes the fov25px122 and
# fov25px57 crops beside them, plus labels_<arm>.csv, crops_as_trained_<arm>.sha256 and
# downsample_<arm>.json under analysis_out/context_fov_86/ in $REPO.
#
# As run on 2026-09-25 (job 40599892, 11.9 min on 4 CPUs): the file below, with WORK, REPO and PY
# written inline at their defaults here, submitted with a bare
#   sbatch /gscratch/makelab/jonf/context_fov_86/downsample.slurm
# The two resolution-arm training jobs waited on it with --dependency=afterok:<this job>
# (context_fov_86.sh header).
#
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --job-name=ctx_downsample
#SBATCH --output=logs/ctx_downsample_%j.out
set -eo pipefail
WORK=${WORK:-/gscratch/makelab/jonf/context_fov_86}
REPO=${REPO:-/gscratch/makelab/jonf/context_fov_86/RampNet}
PY=${PY:-/gscratch/makelab/jonf/envs/tagger/bin/python}
cd "$REPO"
date -u +%FT%TZ; hostname
WORK="$WORK" PY="$PY" bash scripts/analysis/context_fov_86.sh downsample
echo "EXIT 0"; date -u +%FT%TZ
