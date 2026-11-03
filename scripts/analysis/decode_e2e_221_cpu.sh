#!/bin/bash
# CPU phase of the #221 end-to-end check. Waits for the GPU phase, then:
#  B. every (decode x threshold x TTA) evaluate.py run from the caches, plus main's
#     evaluate.py on the same caches for the bit-identity check, plus positions;
#  C. the StaleCoarseCache / --fresh checks on a 3-pano subset (separate cache root).
set -u
WT=/homes/gws/jonf/wt-e2e221
WM=/homes/gws/jonf/wt-e2e221-main
W=/homes/gws/jonf/nobackup/e2e221
PY=/homes/gws/jonf/RampNet/.venv/bin/python
CK=$W/released_606a119.pth
FP=f7f255c586ba
until grep -q DONE $W/timing.txt; do sleep 60; done

ev() {  # ev <worktree> <results-subdir> <tta|no-tta> <threshold> [extra args]
  local wt=$1 out=$2 tta=$3 thr=$4; shift 4
  (cd $wt/stage_two && PYTHONPATH=$wt $PY evaluate.py --checkpoint $CK --dataset manual \
      --data-root $W/data --manual-labels $WT/manual_labels --cache-dir $W/cache \
      --results-dir $W/results/$out --threshold $thr --$tta "$@") > $W/logs/${out}_$thr.txt 2>&1
  echo "rc=$? $out $thr" >> $W/logs/rc.txt
}
mkdir -p $W/logs
echo "CPU_START $(date +%s) $(date -u +%FT%TZ)" >> $W/timing.txt
for tta in tta no-tta; do
  (
    for thr in 0.0 0.3 0.55; do
      ev $WT pr_argmax_$tta $tta $thr --decode argmax
      ev $WM main_argmax_$tta $tta $thr
      ev $WT pr_gaussian_$tta $tta $thr --decode gaussian
    done
    flag=--tta; [ $tta = no-tta ] && flag=--no-tta
    (cd $WT && PYTHONPATH=$WT $PY scripts/analysis/decode_e2e_221.py positions \
        --cache-dir $W/cache --fingerprint $FP $flag \
        --out $W/positions_${tta}.json) > $W/logs/positions_$tta.txt 2>&1
    echo "rc=$? positions $tta" >> $W/logs/rc.txt
  ) &
done
wait
echo "CPU_END $(date +%s) $(date -u +%FT%TZ)" >> $W/timing.txt

# ---- C. stale-cache guard on 3 panos -------------------------------------------------
S=$W/stale; rm -rf $S; mkdir -p $S/labels
ls $WT/manual_labels/*.txt | head -3 | xargs -I{} cp {} $S/labels/
P1=$(ls $S/labels | head -1); P1=${P1%.txt}; P2=$(ls $S/labels | sed -n 2p); P2=${P2%.txt}
KEY=${FP}_manual_tta
st() {  # st <name> <expect: ok|stale> [extra]
  local name=$1 expect=$2; shift 2
  (cd $WT/stage_two && PYTHONPATH=$WT $PY evaluate.py --checkpoint $CK --dataset manual \
      --data-root $W/data --manual-labels $S/labels --cache-dir $S/cache \
      --results-dir $S/results_$name --threshold 0.0 --tta "$@") > $S/log_$name.txt 2>&1
  local rc=$?
  local got=ok; grep -q StaleCoarseCache $S/log_$name.txt && got=stale
  [ $rc -ne 0 ] && [ $got = ok ] && got=error
  echo "$name expect=$expect got=$got rc=$rc coarse_files=$(ls $S/cache/coarse/$KEY 2>/dev/null | wc -l)" >> $S/summary.txt
}
pert() {  # pert <python expr on c (coarse of P1), others> ; edits P1's coarse in place
  PYTHONPATH=$WT $PY -c "
import numpy as np, shutil
d='$S/cache/coarse/$KEY/'; c=np.load(d+'${P1}_coarse.npy'); c2=np.load(d+'${P2}_coarse.npy')
$1
np.save(d+'${P1}_coarse.npy', c.astype(np.float32))"
}
st 1_fresh_gaussian ok --decode gaussian --fresh
cp $S/cache/coarse/$KEY/${P1}_coarse.npy $S/orig_coarse.npy
pert "c = c + 2e-3";                       st 2_offset_2e-3 stale --decode gaussian
cp $S/orig_coarse.npy $S/cache/coarse/$KEY/${P1}_coarse.npy
pert "c = c + 5e-4";                       st 3_offset_5e-4_below_atol ok --decode gaussian
cp $S/orig_coarse.npy $S/cache/coarse/$KEY/${P1}_coarse.npy
pert "c = c2";                             st 4_swapped_pano stale --decode gaussian
cp $S/orig_coarse.npy $S/cache/coarse/$KEY/${P1}_coarse.npy
pert "c = c[:1]";                          st 5_single_branch_in_tta_key stale --decode gaussian
cp $S/orig_coarse.npy $S/cache/coarse/$KEY/${P1}_coarse.npy
pert "c = c[::-1]";                        st 6_branches_swapped_order_invariant ok --decode gaussian
cp $S/orig_coarse.npy $S/cache/coarse/$KEY/${P1}_coarse.npy
pert "c = c + 2e-3";                       st 7_argmax_fresh_clears_coarse ok --decode argmax --fresh
st 8_gaussian_after_argmax_fresh ok --decode gaussian
echo "STALE_DONE $(date -u +%FT%TZ)" >> $W/timing.txt
