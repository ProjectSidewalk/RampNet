#!/bin/bash
# The GPU half of #159's pre-review work, as run on makelab2 (A40) on 2026-10-02.
# Everything it writes is committed under analysis_out/bayonne_159/, plus one
# usage-ledger row per challenger leg in analysis_out/usage_log.jsonl (paid: false).
#
#   bash scripts/analysis/bayonne_159_gpu.sh 2>&1 | tee analysis_out/bayonne_159/gpu_run.log
#
# Needs, in this checkout: benchmark/bayonne/panos/ (125 native panos; check with
# `bayonne_159.py verify`) and benchmark/laurens_mapillary/panos/ (the replication
# control). Environments are makelab2's: RampNet/.venv-eval (py3.12, torch 2.13 cu130,
# transformers 5.15, ultralytics 8.4) for everything except Molmo, which needs its own
# env at transformers 4.57.1 (docs/running_model_comparison.md). The YOLO checkpoints
# are the sha256-verified snapshot in ~/RampNet/yolo_ckpts/ (yolo_baseline/README.md).
#
# Steps are independent; set STEPS to run a subset, e.g. STEPS="extract yolo".
set -u
cd "$(dirname "$0")/../.."
PY="${PY:-$HOME/RampNet/.venv-eval/bin/python}"
MOLMO_PY="${MOLMO_PY:-$HOME/envs/molmo/bin/python}"
YOLO_DIR="${YOLO_DIR:-$HOME/RampNet/yolo_ckpts}"
OUT=analysis_out/bayonne_159
LEDGER=analysis_out/usage_log.jsonl
STEPS="${STEPS:-repro extract parity yolo open qwen molmo export}"
mkdir -p "$OUT"
has() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }
stamp() { echo "=== $1 $(date -Is)"; }

$PY -c "import sys,torch;print('python',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true

if has repro; then
  # Prove this pipeline reproduces an existing split's committed op_cache before
  # reading Bayonne off it. laurens_mapillary: the other GoPro Max split, 94 panos.
  stamp "repro laurens_mapillary"
  $PY -u scripts/analysis/operating_point_curve.py extract --cities laurens_mapillary \
      --cache "$OUT/repro_op_cache" --force
  stamp "repro done"
fi

if has extract; then
  stamp "extract bayonne"
  $PY -u scripts/analysis/operating_point_curve.py extract --cities bayonne --unreviewed \
      --cache "$OUT/op_cache" --force
  stamp "extract done"
fi

if has parity; then
  # The committed records.jsonl came from the labeler's own run of the same model on
  # the same native panos, so this is the Mapillary-path parity check (expected exact).
  $PY scripts/analysis/low_floor_sweep.py parity --cities bayonne --cache "$OUT/op_cache"
fi

if has yolo; then
  # The #51/#71 protocol: best.pt as-saved, pano geometry, imgsz 1280.
  stamp "yolo"
  ( cd "$(pwd)" && ln -sfn "$YOLO_DIR" yolo_ckpts )
  $PY -u scripts/model_comparison/compare.py benchmark/bayonne --unreviewed \
      --models "yolo:yolo_ckpts/y11l_pano.pt,yolo:yolo_ckpts/y26_pano.pt,yolo:yolo_ckpts/y11x_pano_h200.pt" \
      --tiling none --yolo-imgsz 1280 --usage-log "$LEDGER"
  stamp "yolo done"
fi

if has open; then
  stamp "owlv2 + gdino"
  $PY -u scripts/model_comparison/compare.py benchmark/bayonne --unreviewed \
      --models "owlv2,gdino" --usage-log "$LEDGER"
  stamp "open done"
fi

if has qwen; then
  stamp "qwen 8B"
  $PY -u scripts/model_comparison/compare.py benchmark/bayonne --unreviewed \
      --models "qwen:Qwen/Qwen3-VL-8B-Instruct" --usage-log "$LEDGER"
  stamp "qwen done"
fi

if has molmo; then
  # Its own env, or the leg "completes" with Molmo skipped (the documented trap):
  # check the log for "[allenai/Molmo2-8B] not runnable" before trusting it.
  stamp "molmo"
  $MOLMO_PY -u scripts/model_comparison/compare.py benchmark/bayonne --unreviewed \
      --models "molmo:allenai/Molmo2-8B" --usage-log "$LEDGER"
  stamp "molmo done"
fi

if has export; then
  # One JSON per (leg, split), the benchmark/model_detections format, kept OUT of
  # that directory until the split is registered (tests/test_roster.py requires every
  # file there to belong to a registered split's scored run).
  stamp "export"
  $PY scripts/analysis/export_model_cache.py --out "$OUT/model_detections" --splits bayonne \
      --allow-partial --tiling none --yolo-imgsz 1280 \
      --models "yolo:yolo_ckpts/y11l_pano.pt,yolo:yolo_ckpts/y26_pano.pt,yolo:yolo_ckpts/y11x_pano_h200.pt"
  $PY scripts/analysis/export_model_cache.py --out "$OUT/model_detections" --splits bayonne \
      --allow-partial \
      --models "owlv2,gdino,qwen:Qwen/Qwen3-VL-8B-Instruct,molmo:allenai/Molmo2-8B"
  stamp "export done"
fi
echo ALL_STEPS_DONE
