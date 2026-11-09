#!/bin/bash
# Run ON makelab1 (read-only; RampNet#274 s5 root_v5), e.g.
#   wsl-ssh.ps1 makelab1 script scripts/analysis/stage1_crosswalk_sdot_274/prod_s5_pano_positions.sh > pano_positions_s5.csv
# Every Juneau pano's position from prod's pano_data, in prod_s4_pano_positions.sh's columns, for the 35 m Phase 1
# GT exclusion in `export_stage2.py add-panos` (the s5 mined Juneau panos are a subset).
export PGOPTIONS='-c default_transaction_read_only=on'
psql -X -d sidewalk_prod -p 5434 --csv -c "SELECT 'juneau-ak' AS city, p.pano_id, p.lat, p.lng, p.capture_date,
  p.source::text AS source FROM sidewalk_juneau_ak.pano_data p ORDER BY 2;"
