#!/bin/bash
# #221 end-to-end check: run the committed-era evaluate.py (a9ed8a5, the commit that wrote
# stage_two/evaluation_results_new/) on the heatmaps decode_e2e_221_gpu.sh cached, to tell
# a code change from an input change. Run from a clone of this repo that has a9ed8a5
# (a shallow clone does not; `git fetch --unshallow` first). These are the commands that
# produced analysis_out/decode_e2e_221/metrics/a9ed8a5_argmax_tta__*.json.
set -eu
REPO=$(git rev-parse --show-toplevel)
W=/homes/gws/jonf/nobackup/e2e221            # same scratch root as the other two scripts
PY=/homes/gws/jonf/RampNet/.venv/bin/python
FP=f7f255c586ba                               # checkpoint_fingerprint of $W/released_606a119.pth
WO=$W/a9ed8a5
mkdir -p $WO
git -C $REPO archive a9ed8a5adf1275543656ab0fb2ff79b2f2601912 \
    stage_two/evaluate.py stage_two/evaluation_results_new rampnet manual_labels | tar x -C $WO
# a9ed8a5's cache key has no dataset id: <fp>_tta, not <fp>_manual_tta
mkdir -p $W/cache_old/heatmaps
ln -sfn $W/cache/heatmaps/${FP}_manual_tta $W/cache_old/heatmaps/${FP}_tta
cd $WO/stage_two
for th in 0.0 0.55; do
  PYTHONPATH=$WO $PY evaluate.py --checkpoint $W/released_606a119.pth --dataset manual \
      --data-root $W/data --manual-labels $WO/manual_labels --cache-dir $W/cache_old \
      --results-dir $W/results/a9ed8a5_argmax_tta --threshold $th --tta
done
cd $W/results/a9ed8a5_argmax_tta
for f in *.csv; do
  cmp -s $f $WO/stage_two/evaluation_results_new/$f && echo "same-as-committed $f" \
      || echo "differs-from-committed $f"
done
