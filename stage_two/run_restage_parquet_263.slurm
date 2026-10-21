#!/bin/bash
# Re-stage the Stage 2 dataset on klone from the cached Hugging Face parquet shards (issue
# #263), without the Arrow conversion cache that run_restage_263.slurm's load_dataset() path
# needs. That path died on a disk quota (job 42083469, 2026-10-20: the shards had downloaded,
# then the Arrow conversion hit `OSError: [Errno 122] Disk quota exceeded` on a filesystem at
# 99 %). This launcher ran as job 42090420 on 2026-10-20: 4 h 14 m on one CPU node (12
# workers) until ckpt preempted it, then 37 m after the requeue, resuming by skipping panos
# already written. The header, the overridable variables and the provenance lines below were
# added after that run; the #SBATCH directives and the python command are the ones that ran
# (then with REPO and PY hardcoded to Jon's checkout and env).
#
# Reads:  $HF_HOME/hub/datasets--projectsidewalk--rampnet-dataset/snapshots/$HF_REVISION/
#         Download first, under the same HF_HOME, with
#           huggingface-cli download projectsidewalk/rampnet-dataset --repo-type dataset --revision <sha>
#         The run of record used revision ee882e3f3c779dc13182f307bca616e50d9b8c5c (the Hub
#         head on 2026-10-20, last modified 2025-08-15). HF_REVISION defaults to the only
#         snapshot present and refuses to guess when there are several.
# Writes: $OUT/{train,val,test} (jpg + json pairs, the parquet's own jpg bytes),
#         $OUT/_quarantine/<split>/ for the Hub's three placeholder rows per split
#         (pano_id file_1..file_3, 19-byte blobs), and $OUT/COUNTS.json (per-split id hashes).
# Verify: the script exits 0 only when every split holds exactly 150,063 / 42,875 / 21,438
#         pairs with no strays or .tmp files. Nothing should point at $OUT until that reads OK.
#         Afterwards, read-only and without the shard cache:
#           python scripts/restage_from_parquet.py --out $OUT --check
# Resumable: a requeued job skips panos whose jpg and json both exist.
#
# USAGE, from the root of the checkout to run (logs/ must already exist):
#   mkdir -p logs
#   RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2 sbatch stage_two/run_restage_parquet_263.slurm
# Optional: HF_HOME (default /gscratch/scrubbed/$USER/hf), HF_REVISION, OUT (default
# /gscratch/scrubbed/$USER/rampnet_dataset_v2), REPO (default the submitting checkout).
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
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
RAMPNET_ENV="${RAMPNET_ENV:?set RAMPNET_ENV to the conda prefix, e.g. RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2}"
PY="$RAMPNET_ENV/bin/python"
[ -x "$PY" ] || { echo "FATAL: RAMPNET_ENV=$RAMPNET_ENV has no bin/python" >&2; exit 1; }
[ -f "$REPO/scripts/restage_from_parquet.py" ] || { echo "FATAL: REPO=$REPO has no scripts/restage_from_parquet.py" >&2; exit 1; }
HF_HOME="${HF_HOME:-/gscratch/scrubbed/$USER/hf}"
SNAP_ROOT="$HF_HOME/hub/datasets--projectsidewalk--rampnet-dataset/snapshots"
if [ -n "${HF_REVISION:-}" ]; then
    SNAP="$SNAP_ROOT/$HF_REVISION"
else
    shopt -s nullglob
    snaps=("$SNAP_ROOT"/*/)
    shopt -u nullglob
    [ "${#snaps[@]}" -eq 1 ] || { echo "FATAL: $SNAP_ROOT holds ${#snaps[@]} snapshots (download the dataset, or set HF_REVISION)" >&2; exit 1; }
    SNAP="${snaps[0]}"
fi
[ -d "$SNAP/train" ] || { echo "FATAL: $SNAP has no train/ shards; download the dataset first" >&2; exit 1; }
OUT="${OUT:-/gscratch/scrubbed/$USER/rampnet_dataset_v2}"
if commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null); then
    dirty=$(git -C "$REPO" status --porcelain | wc -l)
    echo "repo $REPO at $commit (${dirty} uncommitted paths)"
fi
echo "host $(hostname)  start $(date -u +%FT%TZ)  snapshot $SNAP  out $OUT  python $PY"
"$PY" "$REPO/scripts/restage_from_parquet.py" --snapshot "$SNAP" --out "$OUT" --workers "$SLURM_CPUS_PER_TASK"
echo "end $(date -u +%FT%TZ)"
