#!/bin/bash
# Predict every flat3d arm from the committed per-corner JSON, move the predictions to
# analysis_out/flat_mapillary_3d/predictions/ (the shared crossview_align_48/predictions/
# and its results.json are left alone), then score the Richmond pairs against the
# reference arms (other branches' predictions read in place via git show).
# Run from the repo root after `git fetch origin`.
set -e
PY=${PY:-python}
ARMS="flat_sfm flat_gs flat_gsmed flat_mvs noflat_sfm noflat_gs noflat_gsmed noflat_mvs
      mlypano_sfm mlypano_gs mlypano_gsmed mlypano_mvs"
SRC=analysis_out/crossview_align_48/predictions
DST=analysis_out/flat_mapillary_3d/predictions
mkdir -p $DST
for a in $ARMS; do
  $PY scripts/analysis/crossview_align_48.py predict --arm $a
  mv $SRC/$a.jsonl $SRC/$a.meta.json $DST/
done
M=origin/analysis/crossview-matching-48
S=origin/analysis/crossview-sfm-48
$PY scripts/analysis/flat_mapillary_48.py score --arms proj_height_auto lg $ARMS \
    roma@$M roma_magsac@$M roma_hyb@$M \
    sfm_colmap@$S sfm_colmap_prior@$S mast3r_pair@$S vggt_corner@$S mv3d_consensus@$S
