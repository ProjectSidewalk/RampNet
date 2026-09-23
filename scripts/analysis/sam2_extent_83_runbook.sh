#!/bin/bash
# SAM2 extent vs the whole-apron gold (#83, RampNet 2.0 plan item 7) -- the exact
# commands, in order, that produced analysis_out/sam2_extent_83/ and
# docs/sam2_extent_83.md. Written for makelab2 (1x A40, no Slurm); any Linux + CUDA
# box works, adjust the three paths below.
#
# Inputs a clean clone does NOT have:
#   - benchmark/<city>/panos/  (git-ignored; each bundle's imagery_manifest.json says
#     how they were fetched and pins every file's sha256 -- `run` refuses to start if a
#     pano is missing or differs).
#   - the SAM2.1 Hiera-L checkpoint (public, fetched below, sha256 pinned).
#
# Usage (from the repo root):  bash scripts/analysis/sam2_extent_83_runbook.sh
set -euo pipefail

WORK="${WORK:-/homes/gws/jonf/nobackup/sam2}"          # env build dir + checkpoint + run outputs
ENV="${ENV:-/homes/gws/jonf/envs/sam2}"                # python venv
PANOS_ROOT="${PANOS_ROOT:-$PWD}"                       # checkout holding benchmark/<city>/panos
SAM2_COMMIT=2b90b9f5ceec907a1c18123530e92e794ad901a4   # facebookresearch/sam2 main, 2024-12-15
CKPT_SHA256=2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318

# 1. Environment (uv; python 3.11, torch 2.5.1 cu124). SAM2's optional CUDA extension is
#    skipped (SAM2_BUILD_CUDA=0): it only does mask post-processing hole filling, and
#    the as-run environment did not have it. docs/data/sam2_extent_83_env_freeze.txt is
#    the full as-run `uv pip freeze`.
mkdir -p "$WORK"
if [ ! -x "$ENV/bin/python" ]; then
  uv venv --python 3.11 "$ENV"
  [ -d "$WORK/sam2-src" ] || git clone https://github.com/facebookresearch/sam2.git "$WORK/sam2-src"
  git -C "$WORK/sam2-src" checkout "$SAM2_COMMIT"
  uv pip install --python "$ENV/bin/python" torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cu124
  uv pip install --python "$ENV/bin/python" setuptools wheel
  SAM2_BUILD_CUDA=0 uv pip install --python "$ENV/bin/python" --no-build-isolation -e "$WORK/sam2-src"
  uv pip install --python "$ENV/bin/python" pillow numpy pytest
fi

# 2. Checkpoint.
CKPT="$WORK/sam2.1_hiera_large.pt"
[ -f "$CKPT" ] || curl -sSL -o "$CKPT" \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
echo "$CKPT_SHA256  $CKPT" | sha256sum -c -

# 3. GPU: every boxed item, 4 arms x 3 FOVs x 4 prompt variants. Each city appends one
#    paid:false time row to $WORK/runs/usage_rows.jsonl (outside any worktree, so it
#    outlives the run); copy those rows into analysis_out/usage_log.jsonl and commit.
OUT="$WORK/runs/full"
for city in richmond annapolis sao_paulo paterson; do
  "$ENV/bin/python" scripts/analysis/sam2_extent_83.py run --city "$city" \
      --arm boxcenter_gnomonic,boxcenter_equirect,point_gnomonic,point_equirect \
      --fov 90,76,60 --panos-root "$PANOS_ROOT" --checkpoint "$CKPT" \
      --out "$OUT" --usage-log "$WORK/runs/usage_rows.jsonl"
done
# Committed: $OUT/<city>_rows.csv and <city>_run.json -> analysis_out/sam2_extent_83/

# 4. CPU (any machine, from the committed CSVs): tables + paired deltas.
python scripts/analysis/sam2_extent_83.py summarize --city richmond --out analysis_out/sam2_extent_83
python scripts/analysis/sam2_extent_83.py summarize --city annapolis,sao_paulo,paterson \
    --name partial3 --out analysis_out/sam2_extent_83
python scripts/analysis/sam2_extent_83.py summarize --city richmond,annapolis,sao_paulo,paterson \
    --name all4 --out analysis_out/sam2_extent_83
for city in annapolis sao_paulo paterson; do
  python scripts/analysis/sam2_extent_83.py summarize --city "$city" --out analysis_out/sam2_extent_83
done

# 5. CPU + Richmond panos: the committed contact sheets.
for variant in pt_multi ptbox_multi; do
  python scripts/analysis/sam2_extent_83.py gallery --city richmond --gallery-variant "$variant" \
      --panos-root "$PANOS_ROOT" --out analysis_out/sam2_extent_83 --assets docs/assets
done
