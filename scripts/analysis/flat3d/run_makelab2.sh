#!/bin/bash
# Run reconstruct.py on makelab2 (A40, shared: check nvidia-smi first). Arguments are
# passed through, e.g.:  bash run_makelab2.sh --corner richmond:180 richmond:10
#                        bash run_makelab2.sh --no-flat --no-gs        # the control
B=${FLAT3D_ROOT:-/homes/gws/jonf/flat3d}
export CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.6
export PATH=$B/venv/bin:/usr/local/cuda-12.8/bin:$PATH
export TORCH_HOME=$B/torch_home
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$B/RampNet"
exec "$B/venv/bin/python" scripts/analysis/flat3d/reconstruct.py \
    --flat-dir "$B/flat" --views /homes/gws/jonf/crossview48/views \
    --archive-root /projects/makeabilitylab/sidewalk-auto-labeler/runs \
    --out "$B/corners" --mly-pano-dir "$B/mly_panos" \
    --colmap "env MAMBA_ROOT_PREFIX=$B/mamba $B/bin/micromamba run -n colmap313 colmap" \
    "$@"
