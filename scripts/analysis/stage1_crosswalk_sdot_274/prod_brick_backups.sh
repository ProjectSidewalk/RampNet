#!/bin/bash
# Run ON makelab1 (read-only; RampNet#274), e.g. wsl-ssh.ps1 makelab1 script scripts/analysis/stage1_crosswalk_sdot_274/prod_brick_backups.sh
# Read-only: for the cities with brick/cobblestone Crosswalk labels, how many of those labels (and panos) the
# crowd validated (agree > disagree), and how many of those panos have a stored backup image (pano_data.has_backup).
export PGOPTIONS='-c default_transaction_read_only=on'
Q() { psql -X -d sidewalk_prod -p 5434 -At -F '|' -c "$1"; }
echo "schema|validated_brick_labels|their_panos|panos_with_backup|unvalidated_brick_labels(no votes)"
for S in sidewalk_chicago sidewalk_cuenca sidewalk_columbia sidewalk_columbus sidewalk_gainesville sidewalk_amsterdam sidewalk_seattle sidewalk_pittsburgh sidewalk_knox sidewalk_maywood sidewalk_oradell sidewalk_laurens_ia; do
  Q "SELECT '$S', count(*) FILTER (WHERE l.agree_count > l.disagree_count), count(DISTINCT l.pano_id) FILTER (WHERE l.agree_count > l.disagree_count), count(DISTINCT l.pano_id) FILTER (WHERE l.agree_count > l.disagree_count AND p.has_backup), count(*) FILTER (WHERE l.agree_count = 0 AND l.disagree_count = 0) FROM $S.label l LEFT JOIN $S.pano_data p ON p.pano_id = l.pano_id WHERE l.label_type = 'Crosswalk' AND NOT l.deleted AND NOT l.tutorial AND 'brick/cobblestone' = ANY(l.tags);"
done
