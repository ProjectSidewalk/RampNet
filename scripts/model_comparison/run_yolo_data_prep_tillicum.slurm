#!/bin/bash
# Rebuild the YOLO training dataset ON TILLICUM, from Hugging Face. Download + prep.
#
# WHY REGENERATE RATHER THAN TRANSFER FROM KLONE
# There is no automated path from klone to Tillicum: no shared filesystem (/gscratch and
# /mmfs1 do not exist here), and Tillicum cannot authenticate to klone non-interactively
# — both ends are Duo/keyboard-interactive with no publickey, so BatchMode fails. Globus
# CLI is on neither side. Verified 2026-07-30.
#
# Regenerating sidesteps all of it, and is SAFE because the prep is deterministic by
# construction: prepare_yolo_dataset.py thins background tiles with an md5 of the file
# stem specifically so the choice is stable across processes and runs (see
# `_keep_background`, and its comment about salted hash()).
#
# VERIFY, DO NOT ASSUME. The committed record gives the exact counts klone produced:
#     tiles 557,413 train / 161,002 val      pano 150,063 train / 42,875 val
# This job prints its own counts at the end. If they do not match, the datasets are NOT
# equivalent and nothing trained here is comparable to the #51 runs — stop and diff.
#
# WHY A GPU JOB FOR CPU WORK
# Tillicum prohibits CPU-only jobs; every job must request >=1 GPU. The runbook also
# warns that login nodes reap heavy long-running processes, and on Tillicum that would
# take the Duo-authenticated SSH master with it. So we hold one GPU and eat the cost:
# roughly 6-12 h at $0.90/h, i.e. $5-11. Cheap against a $1,500/month cap.
#
#   sbatch scripts/model_comparison/run_yolo_data_prep_tillicum.slurm
#
#SBATCH --job-name=yolo_data_prep
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=NONE
#SBATCH --output=logs/yolo_data_prep_%j.out
#SBATCH --error=logs/yolo_data_prep_%j.err

set -euo pipefail

export REPO="${REPO:-/gpfs/home/$USER/RampNet}"
# /gpfs/scrubbed, NOT the 1 TB /gpfs/projects/makelab quota: the HF source is ~462 GB
# and the prepared YOLO layout another ~286 GB, so this needs ~750 GB. scrubbed has
# ~484 TB free and is what UW-IT recommends for active many-small-file datasets.
export SCRATCH="${SCRATCH:-/gpfs/scrubbed/$USER}"
export PYBIN="${PYBIN:-/gpfs/projects/makelab/$USER/envs/rampnet-yolo/bin/python}"
export HF_HOME="$SCRATCH/hf"
export YOLO_CONFIG_DIR="$SCRATCH/ultralytics"

mkdir -p "$SCRATCH"

echo "=== node ==="
hostname; date
df -h "$SCRATCH" | tail -1
echo "=== interpreter (must match the klone #51 runs exactly) ==="
"$PYBIN" -c "import sys, torch, ultralytics; print('python', sys.version.split()[0], '| torch', torch.__version__, '| ultralytics', ultralytics.__version__)"

cd "$REPO"

echo
echo "=== STAGE: data (HF download, ~462 GB) ==="
date
bash hyak_yolo_runbook.sh data

echo
echo "=== STAGE: prep (tiles + pano) ==="
date
bash hyak_yolo_runbook.sh prep

echo
echo "=== VERIFICATION — counts must match the klone record ==="
date
for ds in tiles pano; do
    for split in train val; do
        d="$SCRATCH/yolo/$ds/images/$split"
        printf "  %-6s %-6s %s\n" "$ds" "$split" "$(ls -U "$d" 2>/dev/null | grep -c '\.jpg$')"
    done
done
cat <<'EOF'

  Expected, from scripts/model_comparison/yolo_baseline/README.md provenance:
    tiles  train  557413
    tiles  val    161002
    pano   train  150063
    pano   val    42875

  A mismatch means the regenerated dataset is NOT the one the #51 arms trained on.
  Do not train against it until the difference is understood.
EOF
date
echo "Cost so far: run `hyakusage`"
