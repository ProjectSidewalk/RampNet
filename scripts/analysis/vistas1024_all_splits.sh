#!/bin/bash
# The Mapillary Vistas parity arm (mask2former-vistas-curb-cut-1024x1024, #126/#137/#163)
# on the nine splits it had not yet run on, completing it across every benchmark split.
# richmond (2026-09-20, makelab2) and bend/paterson/gainesville/annapolis (#35 transfer,
# klone job 40774146, scripts/analysis/cascade_transfer_35.slurm) are already published.
# The flags are exactly that run's: --models rampnet,vistas:curb-cut --vistas-input-size
# 1024 1024, fp16 and min_area_px 16 by default, no --vistas-revision (setting it changes the
# signature and so the leg; the snapshot actually loaded is recorded in env.txt instead).
#
# Run of record: makelab2 (A40), 2026-10-20, from a scratch worktree at origin/claude-newest-ids
# with the panoramas symlinked/unpacked into benchmark/<split>/panos and checked with
#   python scripts/analysis/imagery_manifest.py --verify --cities <the nine splits>
# (all nine OK). Then:
#
#   W=$HOME/vistas1024_run PY=$HOME/RampNet/.venv-eval/bin/python \
#       bash scripts/analysis/vistas1024_all_splits.sh
#
# Variables (all optional):
#   W        scratch root for model_cache/, usage_log.jsonl, env.txt, logs/
#            default: ./vistas1024_run
#   PY       python with torch/transformers/skimage    default: python on PATH
#   SPLITS   splits to run   default: the nine below
#   HF_OFFLINE  1 = load the checkpoint from the local HF cache only   default: 1
#
# One compare.py process per split, so each split's usage row carries its own model load,
# inference seconds and s/pano (paid: false; appended to analysis_out/usage_log.jsonl
# afterwards). Resumable: compare.py skips panos already in $W/model_cache. Afterwards, on a
# clone that has the cache:
#   python scripts/analysis/export_model_cache.py --write --models vistas:curb-cut \
#       --vistas-input-size 1024 1024 --cache-dir $W/model_cache --splits <the nine>
#   python scripts/analysis/export_model_cache.py --verify (same flags)
set -o pipefail
W=${W:-$PWD/vistas1024_run}
PY=${PY:-python}
SPLITS=${SPLITS:-clovis morgantown laurens_mapillary juneau laurens_gsv budapest_district5 sao_paulo bayonne manual_gold}
HF_OFFLINE=${HF_OFFLINE:-1}
if [ "$HF_OFFLINE" = 1 ]; then export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
mkdir -p "$W/logs"
HUB=${HF_HOME:-$HOME/.cache/huggingface}/hub
snapshot() {  # the commit refs/main resolves to in the HF cache, i.e. what an offline load uses
  local d="$HUB/models--${1//\//--}"
  echo "$1 refs/main=$(cat "$d/refs/main" 2>/dev/null || echo MISSING) snapshots=$(ls "$d/snapshots" 2>/dev/null | tr '\n' ' ')"
}
{
  echo "=== $(date -u +%FT%TZ) host=$(hostname -f) commit=$(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  $PY -c "import sys,torch,torchvision,transformers,skimage;print('python',sys.version.split()[0]);print('torch',torch.__version__,'cuda',torch.version.cuda);print('torchvision',torchvision.__version__);print('transformers',transformers.__version__);print('scikit-image',skimage.__version__)"
  lscpu | grep -E '^Model name|^CPU\(s\)'
  echo "HF hub=$HUB HF_OFFLINE=$HF_OFFLINE"
  snapshot facebook/mask2former-swin-large-mapillary-vistas-semantic
} > "$W/env.txt" 2>&1
cat "$W/env.txt"

for s in $SPLITS; do
  echo "=== vistas $s $(date -u +%FT%TZ)"
  $PY -u scripts/model_comparison/compare.py "benchmark/$s" \
      --models rampnet,vistas:curb-cut --vistas-input-size 1024 1024 \
      --cache-dir "$W/model_cache" --usage-log "$W/usage_log.jsonl" \
      2>&1 | tee "$W/logs/vistas1024_${s}.log" \
      || echo "compare.py exited non-zero on $s: a leg was not runnable or aborted (see log)"
done
echo "=== nvidia-smi at end"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
echo "ALL_DONE $(date -u +%FT%TZ)"
