#!/usr/bin/env bash
# makelab2 runbook for #48's challenger legs on the Richmond neighbourhood bundle.
#
#   bash scripts/analysis/multiview_challengers_48.sh            # the real run
#   bash scripts/analysis/multiview_challengers_48.sh --smoke    # 2 judged + 2 unjudged panos per leg
#
# Run from a checkout of the branch that holds benchmark/richmond_neighbourhood/
# (records.jsonl + bundle.json, built by `multiview_challengers_48.py bundle`). The
# defaults below are makelab2's; override any of them in the environment:
#
#   ARCHIVE     the labeler's native-res Richmond archive (<pano_id>.jpg)
#   PY          RampNet eval env: torch + transformers 5.x + ultralytics
#   MOLMOPY     Molmo2's own env, transformers==4.57.1. On 5.x Molmo's Hub code fails at
#               import, compare.py skips the model with a note and exits 0 -- the leg
#               "completes" with no Molmo row. The check after the leg catches that.
#   YOLO_CKPTS  dir holding y11l_pano.pt, y11x_pano_h200.pt, y26_pano.pt (#51 snapshots)
#   WORK        detection cache, logs and the usage ledger, OUTSIDE any worktree, so a
#               removed worktree cannot take them with it
#
# Legs run one at a time on the one A40 (another process may hold part of it; the GPU
# check below reads free memory, it never touches other processes). Order: YOLO trio
# (minutes), OWLv2 + Grounding DINO, Qwen3-VL-8B, Molmo2-8B (hours). Every leg is
# resumable -- detections are cached per pano -- so a re-run continues where it stopped.
# compare.py writes one usage row per leg (wall-clock, s/pano, hardware; paid: false) to
# $WORK/usage_log.jsonl; those rows are copied into the main checkout's
# analysis_out/usage_log.jsonl and committed (docs/compute_cost.md: makelab2 has no
# sacct, so its GPU time goes in the usage ledger, not compute_log.jsonl).
set -uo pipefail
cd "$(dirname "$0")/../.."
ARCHIVE=${ARCHIVE:-/projects/makeabilitylab/sidewalk-auto-labeler/runs/richmond/panos}
PY=${PY:-/homes/gws/jonf/RampNet/.venv-eval/bin/python}
MOLMOPY=${MOLMOPY:-/homes/gws/jonf/envs/molmo/bin/python}
YOLO_CKPTS=${YOLO_CKPTS:-/homes/gws/jonf/RampNet/yolo_ckpts}
WORK=${WORK:-/homes/gws/jonf/mv48}
export HF_HOME=${HF_HOME:-/homes/gws/jonf/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BUNDLE=benchmark/richmond_neighbourhood
LIMIT=()
LOGS=$WORK/logs
if [[ ${1:-} == --smoke ]]; then LIMIT=(--limit 2); LOGS=$WORK/logs_smoke; fi
mkdir -p "$WORK/model_cache" "$LOGS" "$BUNDLE/panos"
DRIVER=$LOGS/driver.log
say() { echo "=== $* $(date -Is)" | tee -a "$DRIVER"; }

say "start $(hostname) $(git rev-parse HEAD) limit=${LIMIT[*]:-none}"

# 1. imagery: link every bundle pano from the archive; refuse to run on a partial set
"$PY" - "$BUNDLE" "$ARCHIVE" <<'EOF' | tee -a "$DRIVER"
import json, os, sys
bundle, archive = sys.argv[1], sys.argv[2]
missing = linked = 0
for line in open(os.path.join(bundle, "records.jsonl"), encoding="utf-8"):
    if not line.strip():
        continue
    pid = json.loads(line)["pano"]["panorama_id"]
    src = os.path.join(archive, pid + ".jpg")
    dst = os.path.join(bundle, "panos", pid + ".jpg")
    if not os.path.exists(src):
        missing += 1
        print("missing", src)
        continue
    if not os.path.lexists(dst):
        os.symlink(src, dst)
    linked += 1
print(f"imagery: {linked} linked, {missing} missing")
sys.exit(1 if missing else 0)
EOF
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then say "ABORT: imagery incomplete"; exit 1; fi
[[ -e yolo_ckpts ]] || ln -s "$YOLO_CKPTS" yolo_ckpts

# 2. are the judged panos the same bytes the published richmond legs saw?
"$PY" - <<'EOF' | tee -a "$DRIVER"
import hashlib, json, os, sys
man = json.load(open("benchmark/richmond/imagery_manifest.json"))["panos"]
same = diff = 0
for pid, rec in sorted(man.items()):
    p = os.path.join("benchmark/richmond_neighbourhood/panos", rec["file"])
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if h == rec["sha256"]:
        same += 1
    else:
        diff += 1
        print("  differs from benchmark/richmond:", pid)
print(f"judged imagery vs benchmark/richmond/imagery_manifest.json: {same} identical, {diff} differ")
sys.exit(1 if diff else 0)
EOF
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then say "ABORT: judged imagery differs from benchmark/richmond"; exit 1; fi

# 3. the GPU: log what else is on it, and wait (never kill) until 30 GB are free
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | tee -a "$DRIVER"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a "$DRIVER"
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if (( free >= 30000 )); then break; fi
  say "only ${free} MiB free on the GPU; waiting 10 min"
  sleep 600
done

leg() {  # leg <name> <python> <compare args...>
  local name=$1 py=$2; shift 2
  say "$name start"
  "$py" -c "import sys,torch,transformers;print('python',sys.version.split()[0],'torch',torch.__version__,'transformers',transformers.__version__,'cuda',torch.cuda.is_available())" \
      > "$LOGS/${name}_env.txt" 2>&1
  "$py" -u scripts/model_comparison/compare.py "$BUNDLE" --detect-unjudged \
      --cache-dir "$WORK/model_cache" --usage-log "$WORK/usage_log.jsonl" \
      "${LIMIT[@]}" "$@" > "$LOGS/$name.txt" 2>&1
  say "$name exit=$?"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$DRIVER"
}

YOLO="yolo:yolo_ckpts/y11l_pano.pt,yolo:yolo_ckpts/y11x_pano_h200.pt,yolo:yolo_ckpts/y26_pano.pt"
# the #71 protocol's pano geometry: whole pano, imgsz 1280, 0.05 cache floor
leg yolo "$PY" --models "$YOLO" --tiling none --yolo-imgsz 1280 --op-threshold 0.25
leg open "$PY" --models owlv2,gdino
leg qwen8b "$PY" --models qwen:Qwen/Qwen3-VL-8B-Instruct
leg molmo "$MOLMOPY" --models molmo:allenai/Molmo2-8B
if grep -q "Molmo2-8B" "$LOGS/molmo.txt" && ! grep -q "not runnable" "$LOGS/molmo.txt"; then
  say "MOLMO_OK"
else
  say "MOLMO_SUSPECT: check $LOGS/molmo.txt for a skip note"
fi

# 4. export every leg's detections for every bundle pano (committed from the branch)
"$PY" scripts/analysis/multiview_challengers_48.py export --cache-dir "$WORK/model_cache" \
    2>&1 | tee -a "$DRIVER"
say "ALL_DONE"
