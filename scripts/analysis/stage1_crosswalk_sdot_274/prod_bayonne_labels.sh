#!/bin/bash
# Run ON makelab1 (read-only; RampNet#274 s4, reported only), e.g.
#   wsl-ssh.ps1 makelab1 script scripts/analysis/stage1_crosswalk_sdot_274/prod_bayonne_labels.sh
# Every live, non-tutorial Crosswalk label on a Panoramax pano in Bayonne, with its point and pano, as CSV
# (the same columns as prod_crosswalk_labels.sh), for s4's recall probe on open imagery it never trained on.
export PGOPTIONS='-c default_transaction_read_only=on'
S=sidewalk_bayonne_fr
psql -X -d sidewalk_prod -p 5434 --csv -c "SELECT '$S' AS schema, l.label_id, l.pano_id, p.source, p.width, p.height, lp.pano_x, lp.pano_y, lp.lat, lp.lng, l.correct, l.agree_count, l.disagree_count, l.user_id, l.time_created, p.has_backup FROM $S.label l JOIN $S.label_point lp USING (label_id) LEFT JOIN $S.pano_data p ON p.pano_id = l.pano_id WHERE l.label_type = 'Crosswalk' AND NOT l.deleted AND NOT l.tutorial AND p.source::text = 'panoramax' ORDER BY l.label_id;"
