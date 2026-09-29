#!/bin/bash
# All 31 Richmond harness corners (#214), three variants, sequentially on makelab2's A40.
# Re-runs the 5 pilot corners too, so every corner uses the same code (the first pilot
# corner ran MVS over all views with geometric consistency; the rest use the source view).
B=${FLAT3D_ROOT:-/homes/gws/jonf/flat3d}
R=$B/RampNet/scripts/analysis/flat3d/run_makelab2.sh
cd "$B"
date +%s > full_t0
bash "$R"
bash "$R" --no-flat
bash "$R" --mly-panos
date +%s > full_t1
echo FULL_DONE
