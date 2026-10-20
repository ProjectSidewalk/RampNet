#!/bin/bash
# The GPU half of #254's post-review work. First started on makelab2 (A40) and abandoned when
# that host wedged; the run of record is klone, through scripts/analysis/juneau_254_klone.slurm
# (docs/juneau_split_254.md section 5). The same steps as
# scripts/analysis/bayonne_159_gpu.sh, pointed at juneau. Everything it writes is
# committed under analysis_out/juneau_254/, plus one usage-ledger row per challenger
# leg in analysis_out/usage_log.jsonl (paid: false).
#
#   bash scripts/analysis/juneau_254_gpu.sh 2>&1 | tee analysis_out/juneau_254/gpu_run.log
#
# Needs, in this checkout: benchmark/juneau/panos/ (125 native panos; check with
# `juneau_254.py verify`) and benchmark/laurens_mapillary/panos/ (the replication
# control). PY / MOLMO_PY / YOLO_DIR default to makelab2's paths; on klone pass the envs
# from scripts/analysis/juneau_254_klone_env.sh (see the .slurm header). Steps are
# independent and a failed step does not stop the rest (no `set -e`), so read the log:
# compare.py exits 3 when a leg is not runnable or aborts after consecutive failures.
#
# Unlike bayonne's, this runs AFTER the review: verdicts.json is committed, so the
# extraction writes the reviewed cache straight to analysis_out/op_cache/juneau.json
# (extract --unreviewed refuses a reviewed bundle, and attach-gt is not needed) and
# the challenger legs run without --unreviewed.
#
# Steps are independent; set STEPS to run a subset, e.g. STEPS="extract yolo".
set -u
cd "$(dirname "$0")/../.."
PY="${PY:-$HOME/RampNet/.venv-eval/bin/python}"
MOLMO_PY="${MOLMO_PY:-$HOME/envs/molmo/bin/python}"
YOLO_DIR="${YOLO_DIR:-$HOME/RampNet/yolo_ckpts}"
OUT=analysis_out/juneau_254
LEDGER=analysis_out/usage_log.jsonl
STEPS="${STEPS:-repro extract parity yolo open qwen molmo export}"
mkdir -p "$OUT"
has() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }
stamp() { echo "=== $1 $(date -Is)"; }

$PY -c "import sys,torch;print('python',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true

if has repro; then
  # Prove this pipeline reproduces an existing split's committed op_cache before
  # reading Juneau off it. laurens_mapillary is the one committed cache built with
  # the post-f4c71c8 extractor, so it should agree cell for cell.
  stamp "repro laurens_mapillary"
  $PY -u scripts/analysis/operating_point_curve.py extract --cities laurens_mapillary \
      --cache "$OUT/repro_op_cache" --force
  stamp "repro done"
fi

if has extract; then
  stamp "extract juneau"
  $PY -u scripts/analysis/operating_point_curve.py extract --cities juneau --force
  stamp "extract done"
fi

if has parity; then
  # records.jsonl came from the labeler's own Mapillary run of the same model on the
  # same native panos, so this is the Mapillary-path parity check (expected exact).
  $PY scripts/analysis/low_floor_sweep.py parity --cities juneau
fi

if has yolo; then
  # The #51/#71 protocol: best.pt as-saved, pano geometry, imgsz 1280.
  stamp "yolo"
  ( cd "$(pwd)" && ln -sfn "$YOLO_DIR" yolo_ckpts )
  $PY -u scripts/model_comparison/compare.py benchmark/juneau \
      --models "yolo:yolo_ckpts/y11l_pano.pt,yolo:yolo_ckpts/y26_pano.pt,yolo:yolo_ckpts/y11x_pano_h200.pt" \
      --tiling none --yolo-imgsz 1280 --usage-log "$LEDGER"
  stamp "yolo done"
fi

if has open; then
  stamp "owlv2 + gdino"
  $PY -u scripts/model_comparison/compare.py benchmark/juneau \
      --models "owlv2,gdino" --usage-log "$LEDGER"
  stamp "open done"
fi

if has qwen; then
  stamp "qwen 8B"
  $PY -u scripts/model_comparison/compare.py benchmark/juneau \
      --models "qwen:Qwen/Qwen3-VL-8B-Instruct" --usage-log "$LEDGER"
  stamp "qwen done"
fi

if has molmo; then
  # Its own env, or the leg "completes" with Molmo skipped (the documented trap):
  # check the log for "[allenai/Molmo2-8B] not runnable" before trusting it. On
  # bayonne it failed every pano when device_map="auto" offloaded to the CPU on a
  # shared GPU (docs/bayonne_split_159.md section 5).
  stamp "molmo"
  $MOLMO_PY -u scripts/model_comparison/compare.py benchmark/juneau \
      --models "molmo:allenai/Molmo2-8B" --usage-log "$LEDGER"
  stamp "molmo done"
fi

if has export; then
  # One JSON per (leg, split), the benchmark/model_detections format, written to a
  # scratch dir and moved into benchmark/model_detections/ with the registration.
  stamp "export"
  $PY scripts/analysis/export_model_cache.py --out "$OUT/model_detections" --splits juneau \
      --allow-partial --tiling none --yolo-imgsz 1280 \
      --models "yolo:yolo_ckpts/y11l_pano.pt,yolo:yolo_ckpts/y26_pano.pt,yolo:yolo_ckpts/y11x_pano_h200.pt"
  $PY scripts/analysis/export_model_cache.py --out "$OUT/model_detections" --splits juneau \
      --allow-partial \
      --models "owlv2,gdino,qwen:Qwen/Qwen3-VL-8B-Instruct,qwen:Qwen/Qwen3-VL-32B-Instruct,molmo:allenai/Molmo2-8B"
  stamp "export done"
fi
echo ALL_STEPS_DONE
