#!/bin/bash
# Re-stage the Stage 2 dataset on klone from the cached Hugging Face parquet shards (issue
# #263), without the Arrow conversion cache that run_restage_263.slurm's load_dataset() path
# needs. That path died on a disk quota (job 42083469, 2026-10-20: 431 GB of shards had
# downloaded, then the ~460 GB Arrow conversion hit `OSError: [Errno 122] Disk quota exceeded`
# on a filesystem at 99 %). This launcher ran as job 42090420 on 2026-10-20: 4 h 14 m on one
# CPU node (12 workers) until ckpt preempted it, then 37 m after the requeue, resuming by
# skipping panos already written; 462 GB in all.
#
# Reads:  /gscratch/scrubbed/jfroehli/hf/hub/datasets--projectsidewalk--rampnet-dataset/snapshots/<rev>/
#         (download it first with `huggingface-cli download projectsidewalk/rampnet-dataset
#         --repo-type dataset` under HF_HOME=/gscratch/scrubbed/$USER/hf, or let
#         download_dataset.py's load_dataset() fetch it and fail at the Arrow step)
# Writes: /gscratch/scrubbed/jfroehli/rampnet_dataset_v2/{train,val,test}  (jpg + json pairs,
#         the parquet's own jpg bytes) and _quarantine/<split>/ for the Hub's three
#         placeholder rows per split (pano_id file_1..file_3, 19-byte blobs).
# Verify: the script exits 0 only when every split holds exactly 150,063 / 42,875 / 21,438
#         pairs with no jpg-only or json-only strays. Nothing should point at the new
#         directory until that line reads OK.
# Resumable: a requeued job skips panos whose json already exists.
#
#   cd /gscratch/makelab/jonf/RampNet263 && mkdir -p logs && sbatch stage_two/run_restage_parquet_263.slurm
#   # afterwards, counts only:
#   python scripts/restage_from_parquet.py --snapshot <snapshot> --out <out> --check
#
#SBATCH -p ckpt
#SBATCH -A ckpt-makelab
#SBATCH --qos=ckpt
#SBATCH --job-name=rampnet_restage_pq263
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH -o logs/restage_pq263_%j.out
#SBATCH -e logs/restage_pq263_%j.err
set -euo pipefail
PY=${PY:-/gscratch/makelab/jonf/envs/sidewalkcv2/bin/python}
REPO=${REPO:-/gscratch/makelab/jonf/RampNet263}
HF_HOME=${HF_HOME:-/gscratch/scrubbed/$USER/hf}
SNAP=$(ls -d "$HF_HOME"/hub/datasets--projectsidewalk--rampnet-dataset/snapshots/*/ | head -1)
OUT=${OUT:-/gscratch/scrubbed/$USER/rampnet_dataset_v2}
echo "host $(hostname)  start $(date -u +%FT%TZ)  snapshot $SNAP  out $OUT"
"$PY" "$REPO/scripts/restage_from_parquet.py" --snapshot "$SNAP" --out "$OUT" --workers "$SLURM_CPUS_PER_TASK"
echo "end $(date -u +%FT%TZ)"
du -sh "$OUT"
