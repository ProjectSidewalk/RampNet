#!/bin/bash
# Re-stage the Stage 2 dataset on klone for issue #263 (the scrubbed copy was partly purged:
# train had 114,667 of ~300,126 files and val was empty on 2026-10-20).
#
# Writes to a FRESH directory so counts can be verified before anything points at it:
#   /gscratch/scrubbed/jfroehli/rampnet_dataset_v2/{train,val,test}
# Expected: 150,063 / 42,875 / 21,438 panos, i.e. 2x that many files (jpg + json).
# download_dataset.py writes ./dataset relative to its cwd and takes no arguments, so this
# job runs it from a staging dir whose `dataset` entry is a symlink to the target.
# Parquet shards cache under HF_HOME, so a requeued job skips the download and redoes only
# the jpg/json writing. CPU-only scavenger queue; no GPU.
#
#   cd /gscratch/makelab/jonf/RampNet263 && mkdir -p logs && sbatch stage_two/run_restage_263.slurm
#
#SBATCH -p ckpt
#SBATCH -A ckpt-makelab
#SBATCH --qos=ckpt
#SBATCH --job-name=rampnet_restage263
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH -o logs/restage263_%j.out
#SBATCH -e logs/restage263_%j.err
set -euo pipefail
export HF_HOME=/gscratch/scrubbed/jfroehli/hf
REPO=/gscratch/makelab/jonf/RampNet263
PY=/gscratch/makelab/jonf/envs/sidewalkcv2/bin/python
OUT=/gscratch/scrubbed/jfroehli/rampnet_dataset_v2
STAGE=/gscratch/scrubbed/jfroehli/restage263
mkdir -p "$OUT" "$STAGE"
ln -sfn "$OUT" "$STAGE/dataset"
cd "$STAGE"
echo "host $(hostname)  start $(date -u +%FT%TZ)  python $PY"
"$PY" "$REPO/download_dataset.py"
echo "download_dataset.py exit $?  end $(date -u +%FT%TZ)"
for s in train val test; do
  n=$(ls "$OUT/$s" | wc -l); echo "$s: $n files ($((n/2)) panos)"
done
du -sh "$OUT"
