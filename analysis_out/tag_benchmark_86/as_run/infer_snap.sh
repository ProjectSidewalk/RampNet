#!/bin/bash
# usage: infer_snap.sh EPOCH  -- score each arm best_after_epEPOCH.pth on all 10,857 crops
E=$1; W=/homes/gws/jonf/nobackup/tagbench86; D=$W/sidewalk-tagger-ai/datasets/crops-curbramp-tags
cd /homes/gws/jonf/wt-tagbench2
until [ $(grep -c " ep$E " $W/snap.log) -ge 3 ]; do sleep 30; done
for A in control pano cell; do
  /homes/gws/jonf/envs/tagger/bin/python scripts/analysis/tag_benchmark_86.py infer --tagger-repo $W/sidewalk-tagger-ai --checkpoint $W/train_$A/best_after_ep$E.pth --csv analysis_out/tag_benchmark_86/hf_curbramp_labels.csv --images $D/train $D/test --out $W/snap_ep${E}_${A}_predictions.csv 2>&1 | grep -E "\"(elapsed_s|ts)\""
done
echo DONE $E
