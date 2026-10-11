#!/bin/bash
# #221 review S1: does evaluate.py reproduce stage_two/evaluation_results_new/ when the
# manual_gold images go through download_dataset.py's path (PIL decode, JPEG quality 95)
# instead of the raw HF parquet bytes in benchmark/manual_gold/panos/?
#
# Edit the four paths below. Needs the released-weights .pth from decode_e2e_221.md step 2.
set -u
WT=/homes/gws/jonf/wt-e2e221                  # checkout of this branch
W=/homes/gws/jonf/nobackup/e2e221             # scratch root
PY=/homes/gws/jonf/RampNet/.venv/bin/python
RAW=/homes/gws/jonf/RampNet/benchmark/manual_gold/panos   # raw HF bytes (fetch_manual_gold.py)
export PYTHONPATH=$WT
mkdir -p $W/data_q95/test $W/logs
# Exactly download_dataset.py:save_example: the decoded PIL image saved as JPEG quality 95.
$PY - <<EOF
import os, PIL
from PIL import Image
src, dst = "$RAW", "$W/data_q95/test"
n = 0
for f in sorted(os.listdir(src)):
    if f.endswith(".jpg"):
        Image.open(os.path.join(src, f)).save(os.path.join(dst, f), format="JPEG", quality=95)
        n += 1
print("re-encoded", n, "with Pillow", PIL.__version__)
EOF
t0=$(date +%s); echo "Q95_START $t0 $(date -u +%FT%TZ)" >> $W/timing.txt
cd $WT/stage_two
$PY evaluate.py --checkpoint $W/released_606a119.pth --dataset manual \
    --data-root $W/data_q95 --manual-labels $WT/manual_labels --cache-dir $W/cache_q95 \
    --results-dir $W/results/q95_argmax_tta --decode argmax --threshold 0.0 --tta --fresh \
    > $W/logs/q95_0.0.txt 2>&1
echo "rc=$? q95 0.0" >> $W/logs/rc.txt
t1=$(date +%s); echo "Q95_GPU_END $t1 $(date -u +%FT%TZ) elapsed_s=$((t1-t0))" >> $W/timing.txt
$PY evaluate.py --checkpoint $W/released_606a119.pth --dataset manual \
    --data-root $W/data_q95 --manual-labels $WT/manual_labels --cache-dir $W/cache_q95 \
    --results-dir $W/results/q95_argmax_tta --decode argmax --threshold 0.55 --tta \
    > $W/logs/q95_0.55.txt 2>&1
echo "rc=$? q95 0.55" >> $W/logs/rc.txt
cd $W/results/q95_argmax_tta
for f in *.csv; do
  cmp -s $f $WT/stage_two/evaluation_results_new/$f && echo "SAME $f" || echo "DIFF $f"
done > $W/q95_vs_committed.txt
sha256sum *.csv >> $W/q95_vs_committed.txt
echo "Q95_DONE $(date -u +%FT%TZ)" >> $W/timing.txt
