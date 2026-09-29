#!/bin/bash
# CUDA COLMAP (for patch-match MVS; the pycolmap pip wheel has no CUDA) in a scratch
# micromamba env on makelab2. Pinned. Run once:  bash setup_colmap_cuda.sh
# 3.13 reads the rig/frame model format pycolmap 4.2 writes (3.11 does not).
set -e
B=${FLAT3D_ROOT:-/homes/gws/jonf/flat3d}
mkdir -p "$B/bin" && cd "$B"
if [ ! -x bin/micromamba ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/2.0.5 | tar -xj -C "$B" bin/micromamba
fi
export MAMBA_ROOT_PREFIX=$B/mamba
[ -d "$B/mamba/envs/colmap313" ] || bin/micromamba create -y -q -n colmap313 -c conda-forge \
    "colmap=3.13.0=*cuda*"
bin/micromamba run -n colmap313 colmap -h | head -3
bin/micromamba list -n colmap313 | grep -i -E "^ *colmap"
echo COLMAP_SETUP_DONE
