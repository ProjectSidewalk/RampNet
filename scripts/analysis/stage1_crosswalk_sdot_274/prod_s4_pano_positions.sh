#!/bin/bash
# Run ON makelab1 (read-only; RampNet#274 s4 root_v4), e.g.
#   wsl-ssh.ps1 makelab1 script scripts/analysis/stage1_crosswalk_sdot_274/prod_s4_pano_positions.sh > pano_positions.csv
# The position of every pano that root_v4 may add, from each city's pano_data, for the 35 m Phase 1 GT exclusion
# (export_stage2.py add-panos). A label's lat/lng is the label's ground point, not the camera's, so it cannot
# stand in. Covers every pano that carries a live Crosswalk label in the brick cities and Richmond (a superset
# of the brick and Richmond positives' panos), plus every Laurens pano (the car-top negatives).
export PGOPTIONS='-c default_transaction_read_only=on'
# census city key -> schema
CITIES="chicago-il:chicago cuenca:cuenca gainesville-fl:gainesville columbus-oh:columbus pittsburgh-pa:pittsburgh
knox-oh:knox maywood-nj:maywood oradell-nj:oradell danville-il:danville niagara-falls-ny:niagara_falls
west-chester-pa:west_chester st-louis-mo:st_louis auckland:auckland cdmx:cdmx cliffside-park-nj:cliffside_park
clifton-nj:clifton la-ca:la santiago-chile:santiago tucson-az:tucson richmond-va:richmond"
Q=""
for cs in $CITIES; do
  city=${cs%%:*}; s=sidewalk_${cs##*:}
  Q="$Q SELECT '$city' AS city, p.pano_id, p.lat, p.lng, p.capture_date, p.source::text AS source FROM $s.pano_data p
       WHERE p.pano_id IN (SELECT pano_id FROM $s.label WHERE label_type = 'Crosswalk' AND NOT deleted) UNION ALL"
done
Q="$Q SELECT 'laurens-ia', p.pano_id, p.lat, p.lng, p.capture_date, p.source::text FROM sidewalk_laurens_ia.pano_data p
     ORDER BY 1, 2;"
psql -X -d sidewalk_prod -p 5434 --csv -c "$Q"
