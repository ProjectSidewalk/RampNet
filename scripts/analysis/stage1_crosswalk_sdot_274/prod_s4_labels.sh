#!/bin/bash
# Run ON makelab1 (read-only; RampNet#274 s4 Step A), e.g. wsl-ssh.ps1 makelab1 script scripts/analysis/stage1_crosswalk_sdot_274/prod_s4_labels.sh
# Every live, non-tutorial Crosswalk label not voted incorrect by majority (disagree <= agree) for:
#   columbia-sc  tagged brick/cobblestone   (the s4 brick test set; never trained on)
#   richmond-va  on a Mapillary pano        (s4 training positives, after Jon's confirmation)
# as one CSV with the label's point, its pano, and whether PS stores a backup image.
export PGOPTIONS='-c default_transaction_read_only=on'
SEL="l.label_id, l.pano_id, p.source::text AS source, p.width, p.height, lp.pano_x, lp.pano_y, lp.lat, lp.lng, l.agree_count, l.disagree_count, l.user_id, l.time_created, p.capture_date, p.has_backup, array_to_string(l.tags, ';') AS tags"
WHERE="l.label_type = 'Crosswalk' AND NOT l.deleted AND NOT l.tutorial AND l.disagree_count <= l.agree_count"
psql -X -d sidewalk_prod -p 5434 --csv -c "SELECT 'columbia-sc' AS city, $SEL FROM sidewalk_columbia.label l JOIN sidewalk_columbia.label_point lp USING (label_id) LEFT JOIN sidewalk_columbia.pano_data p ON p.pano_id = l.pano_id WHERE $WHERE AND 'brick/cobblestone' = ANY(l.tags) UNION ALL SELECT 'richmond-va', $SEL FROM sidewalk_richmond.label l JOIN sidewalk_richmond.label_point lp USING (label_id) LEFT JOIN sidewalk_richmond.pano_data p ON p.pano_id = l.pano_id WHERE $WHERE AND p.source::text = 'mapillary' ORDER BY 1, 2;"
