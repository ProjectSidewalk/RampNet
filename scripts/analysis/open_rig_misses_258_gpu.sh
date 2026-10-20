#!/bin/bash
# The GPU half of #263 Track 1 (#258): the laurens_gsv low-floor op_cache, which no
# earlier run produced, so laurens_gsv can be bucketed by open_rig_misses_258.py like
# every other open split.
#
# STATUS: NOT YET RUN TO COMPLETION. A launch on makelab2 on 2026-10-20 (from the
# worktree ~/wt-openrig258 at d74d680, an earlier version of this script) was never
# confirmed: the launching ssh call timed out client-side and, by the standing rule,
# nobody contacted the host again. Nothing from it is committed. Re-run from here.
#
#   LEDGER=/path/to/main/checkout/analysis_out/usage_log.jsonl \
#     bash scripts/analysis/open_rig_misses_258_gpu.sh 2>&1 | tee open_rig_258_gpu.log
#
# Inputs, in this checkout:
#   benchmark/laurens_gsv/panos/       86 native GSV panos
#   benchmark/laurens_mapillary/panos/ 94 native panos, the repro control
# Neither split is on Hugging Face yet (docs/replication.md, "benchmark panoramas"); the
# copies of record are in makelab2's ~/RampNet/benchmark/<split>/panos/. Symlink or copy
# them in, then confirm the bytes against the committed manifests (step `manifest`).
#
# Environment: PY defaults to makelab2's RampNet/.venv-eval (py3.12, torch cu130), the env
# bayonne_159_gpu.sh and juneau_254_gpu.sh used. The model is the published Hub checkpoint
# (projectsidewalk/rampnet-model), the extractor default.
#
# Steps (STEPS="manifest repro extract parity score ledger" by default). Every gate exits
# non-zero, and `set -euo pipefail` stops the run there:
#   manifest  imagery_manifest.py --verify on both splits: the panos are the reviewed bytes.
#   repro     re-extract laurens_mapillary into a scratch cache; exit 1 unless every pano's
#             peak list equals the committed analysis_out/op_cache/laurens_mapillary.json
#             (it is a post-f4c71c8, Mapillary-path cache, so it should agree exactly).
#   extract   laurens_gsv is reviewed (verdicts.json committed), so it is written straight
#             to analysis_out/op_cache/laurens_gsv.json with verdict-derived GT.
#   parity    low_floor_sweep.py parity: the cache's 0.55 peaks against records.jsonl. GSV
#             splits drift by a heatmap cell (the records came from a 4096x2048 intermediate,
#             the cache from a native-res downsample), so this is a tolerance gate, not
#             bit-exactness; see parity_for() in low_floor_sweep.py.
#   score     tp/fp/fn at 0.55 from the cache against analysis_out/scoreboard.json's
#             laurens_gsv RampNet cell (records-based). Exit 1 if they differ. A GSV cache
#             can legitimately differ (bend's committed cache gives tp 247 / fp 5 against
#             the scoreboard's 249 / 10; paterson and gainesville are 2 and 7 TPs short), so
#             a difference must be explained in the doc before the cache is committed, not
#             silently accepted. juneau, a Mapillary-path cache, matches exactly (204 / 7).
#   ledger    appends one paid:false row (host, GPU, elapsed_s of the extract) to $LEDGER
#             via rampnet.ledger.append_rows. Point LEDGER at the MAIN checkout's
#             analysis_out/usage_log.jsonl (docs/compute_cost.md: makelab2 has no Slurm,
#             so its GPU time goes in usage_log.jsonl, not compute_log.jsonl), then copy
#             the row into the branch's copy by its ts.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-$HOME/RampNet/.venv-eval/bin/python}"
LEDGER="${LEDGER:-analysis_out/usage_log.jsonl}"
STEPS="${STEPS:-manifest repro extract parity score ledger}"
has() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }
stamp() { echo "=== $1 $(date -Is)"; }

$PY -c "import sys,torch;print('python',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
git rev-parse HEAD

if has manifest; then
  stamp "manifest"
  $PY scripts/analysis/imagery_manifest.py --verify --cities laurens_gsv,laurens_mapillary
fi

if has repro; then
  stamp "repro laurens_mapillary"
  $PY -u scripts/analysis/operating_point_curve.py extract --cities laurens_mapillary \
      --cache analysis_out/open_rig_258_repro_op_cache --force
  $PY - <<'PYEOF'
import json, sys
a = json.load(open("analysis_out/op_cache/laurens_mapillary.json"))
b = json.load(open("analysis_out/open_rig_258_repro_op_cache/laurens_mapillary.json"))
pa = {p["pano"]: p["preds"] for p in a["panos"]}
pb = {p["pano"]: p["preds"] for p in b["panos"]}
same = sum(1 for k in pa if pa[k] == pb.get(k))
print(f"repro: {same}/{len(pa)} panos with identical peak lists; "
      f"peaks committed {sum(map(len, pa.values()))} vs repro {sum(map(len, pb.values()))}")
if same != len(pa) or set(pa) != set(pb):
    sys.exit("repro FAILED: this environment does not reproduce the committed cache")
PYEOF
  stamp "repro done"
fi

if has extract; then
  stamp "extract laurens_gsv"
  T0=$(date +%s.%N)
  $PY -u scripts/analysis/operating_point_curve.py extract --cities laurens_gsv --force
  T1=$(date +%s.%N)
  echo "$T0 $T1" > analysis_out/open_rig_258_extract_times.txt
  stamp "extract done"
fi

if has parity; then
  stamp "parity laurens_gsv"
  $PY scripts/analysis/low_floor_sweep.py parity --cities laurens_gsv
fi

if has score; then
  stamp "score laurens_gsv vs scoreboard"
  $PY - <<'PYEOF'
import json, os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts/analysis")
from operating_point_curve import read_cache
from rampnet.detection_eval import aggregate, score_pano
panos, _ = read_cache("analysis_out/op_cache/laurens_gsv.json")
scores = [score_pano([(x, y, s) for x, y, s in pd["preds"] if s >= 0.55], pd["gt"])
          for pd in panos]
tp = sum(s.tp for s in scores); fp = sum(s.fp for s in scores)
n_gt = sum(len(pd["gt"].gt_points) for pd in panos if pd["gt"].fn_confirmed)
want = json.load(open("analysis_out/scoreboard.json"))["per_split"]["rampnet"]["laurens_gsv"]
print(f"cache:      tp {tp} fp {fp} fn {n_gt - tp}")
print(f"scoreboard: tp {want['tp']} fp {want['fp']} fn {want['fn']}")
if (tp, fp, n_gt - tp) != (want["tp"], want["fp"], want["fn"]):
    sys.exit("score MISMATCH: explain the difference in the doc before committing the cache")
PYEOF
fi

if has ledger; then
  stamp "ledger"
  $PY - "$LEDGER" <<'PYEOF'
import datetime, json, socket, subprocess, sys
sys.path.insert(0, ".")
from rampnet.ledger import append_rows
t0, t1 = map(float, open("analysis_out/open_rig_258_extract_times.txt").read().split())
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip().splitlines()[:1]
n = json.load(open("analysis_out/op_cache/laurens_gsv.json"))["meta"]["n_panos"]
ts = datetime.datetime.fromtimestamp(t1, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
row = {"ts": ts, "bundle": "laurens_gsv", "label": "open-rig-258:extract",
       "provider": "rampnet", "model_id": "projectsidewalk/rampnet-model", "paid": False,
       "hardware": {"host": socket.getfqdn(), "gpus": gpu}, "status": "ok",
       "est_cost_usd": 0.0, "pricing": None, "panos_scored": n,
       "elapsed_s": round(t1 - t0, 3), "s_per_pano": round((t1 - t0) / n, 4),
       "what": "operating_point_curve.py extract --cities laurens_gsv (0.05 floor) for #263 Track 1",
       "run_id": f"open-rig-258:extract:{ts}"}
append_rows(sys.argv[1], [row])
print(json.dumps(row))
PYEOF
fi
echo ALL_STEPS_DONE
