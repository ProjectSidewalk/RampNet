#!/bin/bash
# YOLO pano-trio benchmark eval (issue #51), per the pre-registered protocol
# (scripts/model_comparison/yolo_baseline/README.md, issue #71):
# best.pt as-saved, pano geometry = --tiling none / imgsz 1280,
# headline F1 at conf 0.25 (--op-threshold 0.25), sweep printed as exploratory.
# Checkpoints: yolo_ckpts/*.pt, sha256-verified against the durable snapshot's
# .sha256.verified (klone /gscratch/makelab/jonf/rampnet_yolo_baseline_51).
set -u
cd /homes/gws/jonf/RampNet
PY=.venv-eval/bin/python
MODELS="yolo:yolo_ckpts/y11l_pano.pt,yolo:yolo_ckpts/y26_pano.pt,yolo:yolo_ckpts/y11x_pano_h200.pt"
OUT=yolo_eval_results
mkdir -p "$OUT"
$PY -c "import torch,ultralytics; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ultralytics', ultralytics.__version__)" > "$OUT/env.txt" 2>&1
for b in bend richmond annapolis budapest_district5 clovis gainesville morgantown paterson sao_paulo; do
  echo "=== $b start $(date -Is)"
  $PY scripts/model_comparison/compare.py "benchmark/$b" \
    --models "$MODELS" --tiling none --yolo-imgsz 1280 \
    --op-threshold 0.25 --sweep --pr-out "$OUT/pr_$b" \
    > "$OUT/$b.txt" 2>&1
  echo "=== $b exit=$? $(date -Is)"
done
echo ALL_CITY_BUNDLES_DONE
