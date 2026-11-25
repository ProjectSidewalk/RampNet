#!/bin/bash
# Run ON makelab1 (read-only; RampNet#274), e.g. wsl-ssh.ps1 makelab1 script scripts/analysis/stage1_crosswalk_sdot_274/s4_prod_probe.sh
# Read-only (RampNet#274 s4 planning): (1) live Crosswalk labels on non-GSV panos, per schema and source;
# (2) Columbia SC brick/cobblestone Crosswalk labels: count, panos, validation, stored backups, capture years.
export PGOPTIONS='-c default_transaction_read_only=on'
Q() { psql -X -d sidewalk_prod -p 5434 -At -F '|' -c "$1"; }
echo "== non-GSV crosswalk labels: schema|source|labels|panos|agree>disagree|disagree>agree"
for S in $(Q "SELECT table_schema FROM information_schema.columns WHERE table_name='label' AND column_name='tags' AND table_schema LIKE 'sidewalk_%' AND table_schema !~ '(bak|backup|old|test)' ORDER BY 1;"); do
  Q "SELECT '$S', p.source::text, count(*), count(DISTINCT l.pano_id), count(*) FILTER (WHERE l.agree_count > l.disagree_count), count(*) FILTER (WHERE l.disagree_count > l.agree_count) FROM $S.label l JOIN $S.pano_data p ON p.pano_id = l.pano_id WHERE l.label_type = 'Crosswalk' AND NOT l.deleted AND NOT l.tutorial AND p.source::text <> 'gsv' GROUP BY 2;" 2>&1
done
echo "== columbia brick: labels|panos|kept(disagree<=agree)|kept_panos|kept_panos_with_backup|min_year|max_year"
S=$(Q "SELECT table_schema FROM information_schema.tables WHERE table_name='label' AND table_schema LIKE 'sidewalk_columbia%' ORDER BY 1;")
echo "schemas: $S"
for s in $S; do
  Q "SELECT '$s', count(*), count(DISTINCT l.pano_id), count(*) FILTER (WHERE l.disagree_count <= l.agree_count), count(DISTINCT l.pano_id) FILTER (WHERE l.disagree_count <= l.agree_count), count(DISTINCT l.pano_id) FILTER (WHERE l.disagree_count <= l.agree_count AND p.has_backup), min(p.capture_date), max(p.capture_date), string_agg(DISTINCT p.source::text, ',') FROM $s.label l LEFT JOIN $s.pano_data p ON p.pano_id = l.pano_id WHERE l.label_type = 'Crosswalk' AND NOT l.deleted AND NOT l.tutorial AND 'brick/cobblestone' = ANY(l.tags);" 2>&1
done
