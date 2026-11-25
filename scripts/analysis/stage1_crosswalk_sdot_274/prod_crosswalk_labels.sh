#!/bin/bash
# Run ON makelab1 (read-only; RampNet#274), e.g. wsl-ssh.ps1 makelab1 script scripts/analysis/stage1_crosswalk_sdot_274/prod_crosswalk_labels.sh
# Read-only: every live, non-tutorial Crosswalk label in Laurens and Juneau, with its point and pano, as CSV.
export PGOPTIONS='-c default_transaction_read_only=on'
for S in sidewalk_laurens_ia sidewalk_juneau_ak; do
  psql -X -d sidewalk_prod -p 5434 --csv -c "SELECT '$S' AS schema, l.label_id, l.pano_id, p.source, p.width, p.height, lp.pano_x, lp.pano_y, lp.lat, lp.lng, l.correct, l.agree_count, l.disagree_count, l.user_id, l.time_created FROM $S.label l JOIN $S.label_point lp USING (label_id) LEFT JOIN $S.pano_data p ON p.pano_id = l.pano_id WHERE l.label_type = 'Crosswalk' AND NOT l.deleted AND NOT l.tutorial ORDER BY l.label_id;"
done
