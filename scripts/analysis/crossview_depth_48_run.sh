#!/bin/bash
# usage: bash run.sh "<models>" [limit]
R=/homes/gws/jonf/crossview48/depth
export HF_HOME=$R/hf
export TORCH_HOME=$R/torch
export PYTHONUNBUFFERED=1
cd $R/repo
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
for m in $1; do
  OUT=$R/out${2:+_smoke}
  echo "=== $m $(date -u +%FT%TZ)"
  $R/venv/bin/python scripts/analysis/crossview_depth_48.py extract --model $m \
      --archive-root /projects/makeabilitylab/sidewalk-auto-labeler/runs --src-root $R/src \
      --out-dir $OUT ${2:+--limit $2} 2>&1 | grep -v "^  " | tail -40
  echo "=== $m exit ${PIPESTATUS[0]}"
done
echo ALL_DONE
