# Crosswalk Phase 0 (#274): census, offset read, Vistas probe

The scripts behind [`docs/crosswalk_phase0_274.md`](../../../docs/crosswalk_phase0_274.md). Items 1
and 2 are stdlib-only and CPU; item 3 needs the Vistas checkpoint and a GPU (an RTX 3070 with 2 GB
free ran it in fp16 at ~2–3 s per pano). Every network pull is cached under `--out`, so a re-run
is offline once the pulls exist; the committed tables under
[`analysis_out/crosswalk_phase0_274/`](../../../analysis_out/crosswalk_phase0_274/) are what the
doc quotes, and the raw pulls (~200 MB of rawLabels, OSM tiles and ways) are not committed.

| script | item | network | what it writes |
|---|---|---|---|
| `crossing_census.py --out OUT [--only city]` | 1 | PS `/v3/api` rawLabels + Overpass | `OUT/rawlabels/`, `OUT/osm/`, `OUT/per_label/`, `OUT/census.json` |
| `make_table.py OUT` / `footprint.py OUT` | 1 | no | the per-city table and the OSM footprint table as Markdown |
| `offset_read.py --out OUT [--only city] [--sample 1500]` | 2 | Overpass `way(bn)` | `OUT/ways/`, `OUT/offsets/<city>.csv`, `OUT/offset_read.json` |
| `offset_table.py OUT` | 2 | no | the per-city and pooled offset tables as Markdown |
| `aerial_sample.py --out OUT --city seattle-wa --per-class 40` | 2 | no | the aerial rating sheet `OUT/aerial_sample_<city>.csv` |
| `assemble.py` / `assemble2.py` | 1, 2 | no | fill the posted comments' placeholders from the tables |
| `vistas_crosswalk_probe.py --rampnet . --rawlabels OUT/rawlabels --split <split>=<city> ... --out P [--all]` | 3 | HF hub once (checkpoint) | `P/components/<split>/<pano>.json`, `P/labels_scored.csv`, `P/panos.csv`, `P/summary.json`, `P/meta.json` |
| `score_extra.py --probe-out P --rawlabels OUT/rawlabels --rampnet . --split ...` | 3 | no | `P/summary_extra.json` (chance floor, sensitivity) |
| `figures.py` | 2, 3 | no | `docs/figures/crosswalk_phase0_274/*.png` from the committed tables |

Overpass is shared infrastructure and was overloaded during the census: `crossing_census.overpass_query`
quarters a bbox that times out and pauses between calls; mirrors are listed in `OVERPASS`. The probe
imports the six-view rig from `scripts/model_comparison/equirect_tiling.py` and the pano loader from
`scripts/model_comparison/detectors.py`, so its views are the #126 parity rig by construction.

Label ids are per city: every row carries `label_uid` = `<city>:<label_id>`.

`aerial_spotcheck.py build --sheet <rated_ai.csv> --out <html>` writes a self-contained spot-check page
(committed as `analysis_out/crosswalk_phase0_274/aerial_spotcheck_seattle-wa.html`; open it in a browser): one
row per screen, the crop requested live from King County's 2023 or 2025 ortho service, node / label / 2 m and
5 m rings drawn over it, the AI's call pre-selected (Enter accepts, Y/N/U and 1-4/0 override). Verdicts live
in localStorage; "Export verdicts" downloads them as JSON, and `aerial_spotcheck.py merge --sheet ... --verdicts
<json> --out <rated.csv>` adds the `human_*` columns and prints agreement with the AI pass, overall and per
OSM class. Same pattern as `scripts/gt_gallery.py`.

`aerial_crops.py --sheet OUT/aerial_sample_seattle-wa.csv --out CROPS` renders the rating sheet as 36 m
crops (node blue, label red) plus 2×2 contact sheets from King County's 2023 ortho service, for a rater;
the crops are licensed imagery and stay out of the repo. `aerial_sample_seattle-wa.rated_ai.csv` is the
AI first-pass rating of that sheet (rater column says so); the unrated original is beside it for a person.
