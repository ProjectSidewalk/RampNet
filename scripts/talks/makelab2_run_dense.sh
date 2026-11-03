#!/bin/bash
# Dense MapAnything reconstructions for the UChicago fly-around, on makelab2. Detached:
#   setsid nohup bash /homes/gws/jonf/talk_dense/makelab2_run_dense.sh > /homes/gws/jonf/talk_dense/logs/run_dense.log 2>&1 < /dev/null &
B=/homes/gws/jonf/crossview48_sfm
D=/homes/gws/jonf/talk_dense
export HF_HOME=$B/hf TORCH_HOME=$B/torch_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $D/logs
for s in ${SLUGS:-richmond_99 bend_7}; do
  /usr/bin/time -v $B/venv/bin/python $D/uchicago_2026_dense_scene.py --slug $s \
    --harness $B/RampNet/scripts/analysis --scenes /homes/gws/jonf/crossview48/scenes \
    --archive /projects/makeabilitylab/sidewalk-auto-labeler/runs --out $D $DENSE_ARGS \
    > $D/logs/dense_$s.log 2>&1
done
echo DONE > $D/logs/dense.done
