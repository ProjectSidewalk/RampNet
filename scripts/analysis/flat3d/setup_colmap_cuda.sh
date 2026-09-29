#!/bin/bash
# CUDA COLMAP (for patch-match MVS; the pycolmap pip wheel has no CUDA) in a scratch
# micromamba env on makelab2. Pinned. Run once:  bash setup_colmap_cuda.sh
set -e
B=${FLAT3D_ROOT:-/homes/gws/jonf/flat3d}
mkdir -p "$B/bin" && cd "$B"
if [ ! -x bin/micromamba ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/2.0.5 | tar -xj -C "$B" bin/micromamba
fi
export MAMBA_ROOT_PREFIX=$B/mamba
[ -d "$B/mamba/envs/colmap" ] || bin/micromamba create -y -q -n colmap -c conda-forge \
    "colmap=3.11.1=*cuda*"
bin/micromamba run -n colmap colmap -h | head -3
bin/micromamba run -n colmap colmap patch_match_stereo -h 2>&1 | head -3
echo COLMAP_SETUP_DONE
