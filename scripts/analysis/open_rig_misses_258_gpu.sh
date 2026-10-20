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
#   bash scripts/analysis/open_rig_misses_258_gpu.sh 2>&1 | tee open_rig_258_gpu.log
#   (LEDGER=<path> overrides the default ledger, the main checkout's usage_log.jsonl)
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
# Steps (STEPS="manifest repro extract ledger parity score" by default). `set -euo pipefail`
# stops the run at the first failing gate:
#   manifest  imagery_manifest.py --verify on both splits: the panos are the reviewed bytes.
#   repro     re-extract laurens_mapillary into a scratch cache; exit 1 unless every pano's
#             peak list equals the committed analysis_out/op_cache/laurens_mapillary.json
#             (it is a post-f4c71c8, Mapillary-path cache, so it should agree exactly).
#   extract   laurens_gsv is reviewed (verdicts.json committed), so it is written straight
#             to analysis_out/op_cache/laurens_gsv.json with verdict-derived GT.
#   ledger    one paid:false row (host, GPU, elapsed_s of the extract) appended to $LEDGER
#             via rampnet.ledger.append_rows, BEFORE any gate that reads the result, so a
#             failing gate cannot cost the row (the #139 / #143 rule: a ledger row outlives
#             the run). If the extract itself dies part-way, an EXIT trap still writes the
#             row, with status "failed" and the time spent until then. LEDGER defaults to the
#             MAIN checkout's analysis_out/usage_log.jsonl (rampnet.ledger.canonical_repo_root,
#             so a worktree run lands there); makelab2 has no Slurm, so its GPU time goes in
#             usage_log.jsonl, not compute_log.jsonl (docs/compute_cost.md). Copy the row into
#             the branch's copy by its ts.
#   parity    THE GATE. low_floor_sweep.py parity: the cache's 0.55 peaks against
#             records.jsonl, with the repo's tolerance. GSV splits drift by a heatmap cell (the
#             records came from a 4096x2048 intermediate, the cache from a native-res
#             downsample), so this is a tolerance gate, not bit-exactness; see parity_for().
#   score     REPORTS, does not abort. tp/fp/fn at 0.55 from the cache against
#             analysis_out/scoreboard.json's laurens_gsv RampNet cell. It uses score_pano's
#             default wrap_x=True, the scoreboard's own convention. The scoreboard scores
#             records.jsonl, so the two differ wherever the cache's peaks differ from the
#             records', and on committed caches they do: bayonne 114 / 25 against 113 / 23
#             (the cache keeps 3 seam peaks the labeler's extractor drops), bend 247 / 5
#             against 249 / 10, and paterson and gainesville 2 and 7 TPs short (GSV drift).
#             juneau and laurens_mapillary match exactly. laurens_gsv is a GSV split, so a
#             small difference is expected. Parity decides whether the cache is usable; the
#             score difference is written to the log and must be quoted in the doc beside the
#             laurens_gsv row before the cache is committed.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-$HOME/RampNet/.venv-eval/bin/python}"
if [ -z "${LEDGER:-}" ]; then
  LEDGER="$($PY -c 'import sys; sys.path.insert(0, "."); from rampnet.ledger import canonical_repo_root; print(canonical_repo_root("."))')/analysis_out/usage_log.jsonl"
fi
STEPS="${STEPS:-manifest repro extract ledger parity score}"
TIMES=analysis_out/open_rig_258_extract_times.txt
LEDGER_DONE=0
has() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }
stamp() { echo "=== $1 $(date -Is)"; }
echo "LEDGER=$LEDGER"

write_ledger() {   # $1 = status (ok | failed)
  $PY - "$LEDGER" "$1" "$TIMES" <<'PYEOF'
import datetime, json, os, socket, subprocess, sys, time
sys.path.insert(0, ".")
from rampnet.ledger import append_rows
ledger, status, times = sys.argv[1:4]
parts = open(times).read().split()
t0 = float(parts[0])
t1 = float(parts[1]) if len(parts) > 1 else time.time()
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip().splitlines()[:1]
cache = "analysis_out/op_cache/laurens_gsv.json"
n = json.load(open(cache))["meta"]["n_panos"] if status == "ok" and os.path.exists(cache) else None
ts = datetime.datetime.fromtimestamp(t1, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
row = {"ts": ts, "bundle": "laurens_gsv", "label": "open-rig-258:extract",
       "provider": "rampnet", "model_id": "projectsidewalk/rampnet-model", "paid": False,
       "hardware": {"host": socket.getfqdn(), "gpus": gpu}, "status": status,
       "est_cost_usd": 0.0, "pricing": None, "panos_scored": n,
       "elapsed_s": round(t1 - t0, 3), "s_per_pano": round((t1 - t0) / n, 4) if n else None,
       "what": "operating_point_curve.py extract --cities laurens_gsv (0.05 floor) for #263 Track 1",
       "run_id": f"open-rig-258:extract:{ts}"}
append_rows(ledger, [row])
print(json.dumps(row))
PYEOF
  LEDGER_DONE=1
}
# An extract that started but never reached the ledger step still spent GPU time.
on_exit() {
  rc=$?
  # Only an extract that started and never finished (one timestamp in $TIMES); a
  # finished extract is recorded by the ledger step, or deliberately skipped via STEPS.
  if [ "$LEDGER_DONE" = 0 ] && [ -f "$TIMES" ] && [ "$(wc -w < "$TIMES")" -eq 1 ]; then
    echo "=== exit $rc before the ledger step: recording the extract as failed"
    write_ledger failed || true
  fi
}
trap on_exit EXIT
rm -f "$TIMES"

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
  date +%s.%N > "$TIMES"
  $PY -u scripts/analysis/operating_point_curve.py extract --cities laurens_gsv --force
  echo "$(cat "$TIMES") $(date +%s.%N)" > "$TIMES"
  stamp "extract done"
fi

if has ledger && [ -f "$TIMES" ]; then
  stamp "ledger"
  write_ledger ok
fi

if has parity; then
  stamp "parity laurens_gsv"
  $PY scripts/analysis/low_floor_sweep.py parity --cities laurens_gsv
fi

if has score; then
  stamp "score laurens_gsv vs scoreboard (report only)"
  $PY - <<'PYEOF'
import json, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts/analysis")
from operating_point_curve import read_cache
from rampnet.detection_eval import score_pano
panos, _ = read_cache("analysis_out/op_cache/laurens_gsv.json")
# wrap_x=True is score_pano's default and the scoreboard's convention.
scores = [score_pano([(x, y, s) for x, y, s in pd["preds"] if s >= 0.55], pd["gt"],
                     wrap_x=True) for pd in panos]
tp = sum(s.tp for s in scores); fp = sum(s.fp for s in scores)
n_gt = sum(len(pd["gt"].gt_points) for pd in panos if pd["gt"].fn_confirmed)
want = json.load(open("analysis_out/scoreboard.json"))["per_split"]["rampnet"]["laurens_gsv"]
print(f"cache:      tp {tp} fp {fp} fn {n_gt - tp}")
print(f"scoreboard: tp {want['tp']} fp {want['fp']} fn {want['fn']}")
if (tp, fp, n_gt - tp) == (want["tp"], want["fp"], want["fn"]):
    print("score: exact match")
else:
    print(f"score: DIFFERS by tp {tp - want['tp']:+d} fp {fp - want['fp']:+d}; "
          "quote this in the doc beside the laurens_gsv row (parity is the gate)")
PYEOF
fi
echo ALL_STEPS_DONE
