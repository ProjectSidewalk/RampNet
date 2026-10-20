#!/bin/bash
# The GPU half of #263 Track 1 (#258): the laurens_gsv low-floor op_cache, which
# no earlier run produced, so laurens_gsv can be bucketed by miss_taxonomy.py like
# every other split. Run on makelab2 (A40) on 2026-10-20; writes
# analysis_out/op_cache/laurens_gsv.json (committed) and a scratch repro cache.
#
#   bash scripts/analysis/open_rig_misses_258_gpu.sh 2>&1 | tee open_rig_258_gpu.log
#
# Needs, in this checkout: benchmark/laurens_gsv/panos/ (86 native panos) and
# benchmark/laurens_mapillary/panos/ (94, the repro control). PY defaults to
# makelab2's RampNet/.venv-eval (py3.12, torch cu130), the env bayonne_159_gpu.sh
# and juneau_254_gpu.sh used. The model is the published Hub checkpoint
# (projectsidewalk/rampnet-model), the extractor default.
#
# repro: re-extract laurens_mapillary into a scratch cache and compare it to the
# committed analysis_out/op_cache/laurens_mapillary.json before trusting this env.
# extract: laurens_gsv is a reviewed bundle (verdicts.json committed), so it goes
# straight to the shared op_cache with verdict-derived GT; no attach-gt step.
set -u
cd "$(dirname "$0")/../.."
PY="${PY:-$HOME/RampNet/.venv-eval/bin/python}"
STEPS="${STEPS:-repro extract}"
has() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }
stamp() { echo "=== $1 $(date -Is)"; }

$PY -c "import sys,torch;print('python',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
git rev-parse HEAD

if has repro; then
  stamp "repro laurens_mapillary"
  $PY -u scripts/analysis/operating_point_curve.py extract --cities laurens_mapillary \
      --cache analysis_out/open_rig_258_repro_op_cache --force
  $PY - <<'PYEOF'
import json
a = json.load(open("analysis_out/op_cache/laurens_mapillary.json"))
b = json.load(open("analysis_out/open_rig_258_repro_op_cache/laurens_mapillary.json"))
pa = {p["pano"]: p["preds"] for p in a["panos"]}
pb = {p["pano"]: p["preds"] for p in b["panos"]}
same = sum(1 for k in pa if pa[k] == pb.get(k))
print(f"repro: {same}/{len(pa)} panos with identical peak lists; "
      f"peaks committed {sum(map(len, pa.values()))} vs repro {sum(map(len, pb.values()))}")
PYEOF
  stamp "repro done"
fi

if has extract; then
  stamp "extract laurens_gsv"
  $PY -u scripts/analysis/operating_point_curve.py extract --cities laurens_gsv --force
  stamp "extract done"
fi
echo ALL_STEPS_DONE
