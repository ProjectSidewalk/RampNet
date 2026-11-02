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
| `make_table.py OUT` / `footprint.py OUT` / `confirmed_table.py OUT` | 1 | no | the per-city table, the OSM footprint table (Reading 1) and the confirmed-label table + Reading 3/4 numbers (from `per_label/`, committed) as Markdown |
| `offset_read.py --out OUT [--only city] [--sample 1500]` | 2 | Overpass `way(bn)` | `OUT/ways/`, `OUT/offsets/<city>.csv`, `OUT/offset_read.json` |
| `offset_table.py OUT` | 2 | no | the per-city and pooled offset tables as Markdown |
| `aerial_sample.py --out OUT --city seattle-wa --per-class 40` | 2 | no | the aerial rating sheet `OUT/aerial_sample_<city>.csv` |
| `assemble.py` / `assemble2.py` | 1, 2 | no | filled the posted comments' placeholders from the tables (the `post_*.md` parts are not committed; the assembled comments are, as `comment_item1_census.md` / `comment_item2_offset.md`) |
| `vistas_crosswalk_probe.py --rampnet . --rawlabels OUT/rawlabels --split <split>=<city> ... --out P [--all]` | 3 | HF hub once (checkpoint) | `P/components/<split>/<pano>.json`, `P/labels_scored.csv`, `P/panos.csv`, `P/summary.json`, `P/meta.json` |
| `score_extra.py --probe-out P --rawlabels OUT/rawlabels --rampnet . --split ...` | 3 | no | `P/summary_extra.json` (chance floor, sensitivity) |
| `figures.py` | 2, 3 | no | `docs/figures/crosswalk_phase0_274/*.png` from the committed tables |

Overpass is shared infrastructure and was overloaded during the census: `crossing_census.overpass_query`
quarters a bbox that times out and pauses between calls; mirrors are listed in `OVERPASS`. The probe
imports the six-view rig from `scripts/model_comparison/equirect_tiling.py` and the pano loader from
`scripts/model_comparison/detectors.py`, so its views are the #126 parity rig by construction.

Label ids are per city: every row carries `label_uid` = `<city>:<label_id>`.

`aerial_spotcheck.py build --sheet <rated_ai.csv> --rawlabels <csv> --review <json> --verdicts <export> --out <html>`
writes a self-contained spot-check page (committed as `analysis_out/crosswalk_phase0_274/aerial_spotcheck_seattle-wa.html`;
open it in a browser): one row per screen, the 2023 and 2025 King County orthos side by side (36 m / 18 m Web Mercator, about 24 m / 12 m on
the ground at Seattle) with
node / label / 2 m and 5 m rings (ground metres since 2026-11-02; before that they were Mercator metres,
about 1.48x too small at Seattle, and the 120 human verdicts were saved against those), the street-level crop Project Sidewalk serves for the label, the pano and
label dates and the human votes, and the AI's call pre-selected (Enter accepts, Y/N/U and 1-4/0 override; B
tags a label on a bike crossing). Verdicts live in localStorage; `--verdicts` embeds an export so a rebuilt
page carries its state (newest timestamp wins per row); "Export verdicts" downloads them as JSON, and
`aerial_spotcheck.py merge --sheet ... --verdicts <json> --out <rated.csv>` adds the `human_*` columns
(verdicts, flag, bike tag, note, the imagery each row was saved on, timestamp) and prints agreement with the
AI pass, overall and per OSM class. `build --archive <dir>` renders the page from `archive_spotcheck.py`'s
replay archive instead of the live services. Same pattern as `scripts/gt_gallery.py`.

`aerial_rerate.py sheets --archive <dir> --sheet <rated_ai.csv> --out <dir>` draws the images for a blind
re-rate of the AI first pass's no / unclear rows (2023, 2025, label crop), and `aerial_rerate.py score
--rerate OUT/aerial_rerate_seattle-wa.csv --rated OUT/aerial_sample_seattle-wa.rated.csv` prints agreement
with the human verdicts before and after it. The King County 2025 service returns a blank tile at 24 Mercator m or less (720 px; imagery from 28 m,
probed 2026-11-02), so the 2025 panel is the 36 m tile and a 2x enlargement of it; the spot-check page
does the same for its zoomed view (`MIN_TILE_M`).

`aerial_crops.py --sheet OUT/aerial_sample_seattle-wa.csv --out CROPS` renders the rating sheet as 36 m
(Mercator) crops (node blue, label red) plus 2×2 contact sheets from King County's 2023 ortho service, for a rater;
the crops are licensed imagery and stay out of the repo. `aerial_sample_seattle-wa.rated_ai.csv` is the
AI first-pass rating of that sheet (rater column says so); the unrated original is beside it for a person.
