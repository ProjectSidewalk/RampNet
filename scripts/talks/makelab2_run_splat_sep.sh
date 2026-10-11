#!/bin/bash
# richmond_99 trained on its seven 2025-09 panoramas only (one lighting), 30k iterations,
# lower densification threshold; then training-view checks and the path with the vehicle pruned.
B=/homes/gws/jonf/crossview48_sfm
D=/homes/gws/jonf/talk_dense
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST="8.6"
export TORCH_EXTENSIONS_DIR=$B/torch_ext PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. $B/venv_splat/bin/activate
P=1335429861397399,780580238100270,2615441925465755,3896106377348000,777747894880623,1122077376171358,1340497397658839
O=$D/splat/richmond_99_sep
python $D/klone_splat_train.py train --data $D/richmond_99/colmap --out $O --iters 30000 --grow-grad2d 0.00012 --panos $P > $D/logs/splat_train_sep.log 2>&1
python $D/klone_splat_train.py check --out $O --data $D/richmond_99/colmap --views 0 1 2 3 > $D/logs/splat_check_sep.log 2>&1
python $D/klone_splat_train.py render --out $O --path $D/richmond_99/path.json --data $D/richmond_99/colmap --prune-near 3.5 --step 8 > $D/logs/splat_render_sep.log 2>&1
echo DONE > $D/logs/splat_sep.done
