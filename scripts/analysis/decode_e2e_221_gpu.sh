#!/bin/bash
# GPU phase of the #221 end-to-end check: one fresh evaluate.py --decode gaussian pass per
# TTA setting on manual_gold. Each pass writes the heatmap cache AND the coarse cache, so
# every later argmax/gaussian/threshold run reads them without touching the GPU.
set -u
WT=/homes/gws/jonf/wt-e2e221
W=/homes/gws/jonf/nobackup/e2e221
PY=/homes/gws/jonf/RampNet/.venv/bin/python
export PYTHONPATH=$WT
cd $WT/stage_two
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader > $W/gpu_info.txt
for tta in tta no-tta; do
  t0=$(date +%s)
  echo "START $tta $t0 $(date -u +%FT%TZ)" >> $W/timing.txt
  $PY evaluate.py --checkpoint $W/released_606a119.pth --dataset manual \
      --data-root $W/data --manual-labels $WT/manual_labels \
      --cache-dir $W/cache --results-dir $W/results/gpu_gaussian_$tta \
      --decode gaussian --threshold 0.0 --$tta --fresh > $W/log_gpu_$tta.txt 2>&1
  rc=$?
  t1=$(date +%s)
  echo "END $tta $t1 $(date -u +%FT%TZ) rc=$rc elapsed_s=$((t1-t0))" >> $W/timing.txt
done
echo DONE >> $W/timing.txt
