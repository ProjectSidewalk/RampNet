#!/bin/bash
# Run ON makelab1 (read-only; RampNet#274), e.g. wsl-ssh.ps1 makelab1 script scripts/analysis/stage1_crosswalk_sdot_274/prod_brick_census.sh
# Read-only: per live city schema, Crosswalk labels (live, non-tutorial), how many carry brick/cobblestone, on how
# many panos, how validation went (agree > disagree), and the pano sources of the brick ones.
export PGOPTIONS='-c default_transaction_read_only=on'
Q() { psql -X -d sidewalk_prod -p 5434 -At -F '|' -c "$1"; }
echo "schema|crosswalk|brick|brick_panos|brick_agree_gt_disagree|brick_disagree_gt_agree|brick_sources"
for S in $(Q "SELECT table_schema FROM information_schema.columns WHERE table_name='label' AND column_name='tags' AND table_schema LIKE 'sidewalk_%' AND table_schema !~ '(bak|backup|old|test)' ORDER BY 1;"); do
  Q "SELECT '$S', count(*), count(*) FILTER (WHERE 'brick/cobblestone' = ANY(l.tags)), count(DISTINCT l.pano_id) FILTER (WHERE 'brick/cobblestone' = ANY(l.tags)), count(*) FILTER (WHERE 'brick/cobblestone' = ANY(l.tags) AND l.agree_count > l.disagree_count), count(*) FILTER (WHERE 'brick/cobblestone' = ANY(l.tags) AND l.disagree_count > l.agree_count), coalesce(string_agg(DISTINCT p.source::text, ',') FILTER (WHERE 'brick/cobblestone' = ANY(l.tags)), '') FROM $S.label l LEFT JOIN $S.pano_data p ON p.pano_id = l.pano_id WHERE l.label_type = 'Crosswalk' AND NOT l.deleted AND NOT l.tutorial;" 2>&1
done
