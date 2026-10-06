#!/bin/bash
# Gaussian splatting of the dense corners on makelab2 (A40), in the cross-view sweep's
# InstantSplat environment with gsplat 1.4.0 JIT-built into it. Detached:
#   setsid nohup bash /homes/gws/jonf/talk_dense/makelab2_run_splat.sh > /homes/gws/jonf/talk_dense/logs/run_splat.log 2>&1 < /dev/null &
B=/homes/gws/jonf/crossview48_sfm
D=/homes/gws/jonf/talk_dense
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST="8.6"
export TORCH_EXTENSIONS_DIR=$B/torch_ext UV_CACHE_DIR=$B/uv_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. $B/venv_splat/bin/activate
uv pip install -q scipy pillow
mkdir -p $D/logs $D/splat
for s in ${SLUGS:-richmond_99 bend_7}; do
  python $D/klone_splat_train.py train --data $D/$s/colmap --out $D/splat/$s --iters ${ITERS:-15000} > $D/logs/splat_train_$s.log 2>&1
  python $D/klone_splat_train.py render --out $D/splat/$s --path $D/$s/path.json > $D/logs/splat_render_$s.log 2>&1
  tar czf $D/splat/${s}_frames.tgz -C $D/splat/$s frames train.json
done
echo DONE > $D/logs/splat.done
