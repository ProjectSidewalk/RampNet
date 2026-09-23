#!/usr/bin/env bash
# Snapshot watcher for the #86 retrain arms (docs/tag_benchmark_86.md §5).
#
# The tagger's recipe keeps only best.pth (best *training* exact-match accuracy so far), so an
# interim read of a run needs a copy of it taken right after a given epoch. This polls each
# arm's train_<arm>.log and copies train_<arm>/best.pth to train_<arm>/best_after_ep<E>.pth
# as soon as the line for epoch index E appears. cmd_train saves best.pth *before* printing
# the epoch's line, and an epoch takes ~10 min, so a 20 s poll always copies the checkpoint
# as of epoch E.
#
# This is the script that ran beside the committed arms on makelab2 (2026-09-23), with WORK,
# EPOCHS and ARMS made parameters, a stop for an arm that exits before reaching an epoch, and
# no overwrite of an existing snapshot (so a restarted watcher cannot replace one).
# `tag_benchmark_86.sh train` starts it; it can also be started by hand:
#
#   WORK=/path/to/scratch [EPOCHS="4 9 19 49"] [ARMS="control pano cell"] \
#     bash scripts/analysis/tag_benchmark_86_snap.sh >> "$WORK/snap.log"
#
# Epoch indices are 0-based: "ep4" is the checkpoint after 5 epochs.
set -euo pipefail

WORK=${WORK:?set WORK to the runbook scratch directory}
EPOCHS=${EPOCHS:-4 9 19 49}
ARMS=${ARMS:-control pano cell}
cd "$WORK"

for E in $EPOCHS; do
  for A in $ARMS; do
    until grep -q "\"epoch\": $E," "train_$A.log"; do
      if grep -q '^EXIT ' "train_$A.log"; then
        echo "$(date -u +%FT%TZ) $A ep$E never reached: $(grep '^EXIT ' "train_$A.log")"
        continue 2
      fi
      sleep 20
    done
    [ -f "train_$A/best_after_ep$E.pth" ] || cp "train_$A/best.pth" "train_$A/best_after_ep$E.pth"
    echo "$(date -u +%FT%TZ) $A ep$E $(grep "\"epoch\": $E," "train_$A.log")"
  done
done
