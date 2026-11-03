#!/bin/bash
# Pilot (#214): 5 corners, three variants, sequentially on makelab2's A40.
B=${FLAT3D_ROOT:-/homes/gws/jonf/flat3d}
R=$B/RampNet/scripts/analysis/flat3d/run_makelab2.sh
P="richmond:150 richmond:186 richmond:201 richmond:96 richmond:25"
cd "$B"
date +%s > pilot_t0
bash "$R" --corner richmond:186 richmond:201 richmond:96 richmond:25
bash "$R" --no-flat --corner $P
bash "$R" --mly-panos --corner $P
date +%s > pilot_t1
echo PILOT_DONE
