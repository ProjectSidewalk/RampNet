# Stage 1 for crosswalks from SDOT: Seattle pilot (#274)

The paper's Stage 1 turned government curb-ramp inventories into auto-labelled GSV panos. This pilot does the
same for crosswalks on one city. Seattle's SDOT "Marked Crosswalks" layer supplies the locations. It is
GT-grade inside its coverage under the amended rule on branch `crosswalk-inventory-274`. GSV supplies the
pixels. Two placers put each crosswalk into the pano:

- **B1:** the SDOT point projected into the pano. The bearing is `download_dataset.py`'s; the dip comes from
  a 2.5 m camera height.
- **B2:** B1 snapped to the nearest Mapillary Vistas crosswalk component within 0.05 of the pano width,
  using the Phase 0 probe's checkpoint and code.

Written 2026-11-03; fork A of [`stage1_label_type_274.md`](stage1_label_type_274.md) is decided as A1. Numbers
are in `analysis_out/stage1_crosswalk_sdot_274/report.md`.

## Findings

**0. Stage 1's pano search is dead.** Google has decommissioned `GeoPhotoService.SingleImageSearch`, the
endpoint `stage_one/dataset_generation/search_panos.search_panoramas` calls. Its answer is "decommissioned and
turned down". `generate_dataset_meta.py` therefore cannot discover panos as written. The pilot discovers
through streetlevel's z17 coverage tiles plus `find_panorama_by_id` for historical captures, as the labeler
does. GetMetadata (heading, date) and the tile fetch in `rampnet.gsv.fetch_panorama` still work. On all 937
panos fetched, the production heading equals streetlevel's.

**1. Yield is large because historical captures count.** The median is 21 panos per crosswalk within 10 m
and 56 within 20 m, from captures between 2007 and 2026. 10,646 unique panos were found for 200 crosswalks;
0.687 pass the clean-pano rule. Most failures (2,930 panos) have OSM paint within 35 m that SDOT doesn't
record. The pixel arms ran on at most 6 clean panos per crosswalk, spread over capture dates: 937 panos and
2,890 placements.

**2. B1 alone is not a placer.** By the pre-registered rule the city-wide run does not proceed: B1's exact
pixel is on Vistas paint for 0.371 [0.35, 0.39] of placements within 20 m, against a bar of 0.70. B1 is
usually near the paint but not on it:

| reading (within 20 m) | share |
|---|---|
| exact pixel on paint | 0.371 [0.35, 0.39] |
| paint within 0.022 width (RampNet's matcher radius) | 0.771 [0.75, 0.79] |
| a component within 0.05 (B2 exists) | 0.830 [0.81, 0.84] |

The exact-pixel test was too strict for a thin band seen at a grazing angle: at 20–35 m it is 0.16 while the
tolerance reading stays 0.83. It is still the rule that was written down, and the verdict stands.

**3. B2 passes as the default placer.** Where a component exists, the B1→B2 shift is ≤ 0.05 of the pano width
in 0.816 [0.80, 0.83] of cases, against 0.80; the median shift is 0.021. The eyeball shows what it fixes:
- B1 lands on a car parked over the crosswalk, and B2 moves to the visible paint.
- B1 falls between two adjacent crosswalks, and B2 picks one. Which one is ambiguous.

**4. Temporal misalignment is real but moderate.** SDOT is a 2026 snapshot and ~94% of its points are undated.
The exact-pixel hit is 0.28 for captures up to 2015, 0.37 for 2016–2020 and ~0.47 from 2021 on. The secondary
reading (pre-declared, captures from 2021 on) is 0.473, which would not pass either. The "component exists"
rate moves less, 0.78–0.89 across years.

**Failure modes seen in the gallery:**
- **SDOT lists a crosswalk with no visible paint** in a 2022 residential capture.
- **The pano is in a tunnel or underpass** within 20 m horizontally of the crosswalk, so the projection lands
  on a wall. A grade-separation filter is needed before a city-wide run.

## What this means for the next step

- **Projection needs Vistas.** SDOT projection is a usable *bearing*, as the inventory was for ramps, but
  only with B2 as the placer. A crosswalk with no Vistas component should be an ignore region, not
  background.
- **Better first training set: crowd labels on stored panos.** Prefer the crowd Crosswalk labels on Project
  Sidewalk's stored panos (makelab2 `sidewalk_panos/Panoramas/seattle-wa/`, ~170k native-res panos). 3,292 of
  the 3,301 Seattle panos with a crowd Crosswalk label are there, and 4,810 of those labels have an SDOT
  crosswalk within 10 m. A person clicked paint in that exact pano, so neither the date nor the placement is
  in doubt. B2 then labels the other SDOT crosswalks in the same pano.

## First crosswalk detector: training set and pre-registered evaluation (fixed 2026-11-04, before any checkpoint)

**Training set** (`export_stage2.py build`, `analysis_out/stage1_crosswalk_crowd_274/stage2_build/manifest.json`):
- 5,145 panos with 10,410 positive points: train 3,601, val 1,029, test 515. The split is `split_dataset.py`'s
  60 m groups, seed 42.
- Sources:
  - Seattle arm C. Complete panos train as `dense_ignore`; incomplete panos that have positives train as
    `sparse`.
  - 543 Vistas-verified empty Seattle panos (`verified_empty`).
  - Teaneck, Amsterdam and Burnaby crowd labels that pass the adopted confirmer `O&P`. These are all `sparse`.
- Dropped:
  - 340 panos within 35 m of a Phase 1 GT node (all in Amsterdam);
  - 2,386 incomplete panos with no positive;
  - 55 negatives where Vistas saw paint.
- Teaneck's yield under `O&P` is low: 546 of 3,251 candidates. The cause is OSM, not the crowd. Only 28% of
  Teaneck clicks are within 10 m of an OSM-marked crossing, against 86% in Amsterdam and 94% in Burnaby, while
  the exact-pixel paint rate is similar (0.58 vs 0.56–0.71). The rule's yield depends on local OSM tagging.

**Run.** `stage_two/run_finetune_crosswalk_274.slurm`, a single run, set up as follows:
- Initialisation: from the released RampNet weights (Hub 606a119, sha256 `f2119e3b…`).
- Schedule: seed 1, 1,000 optimizer steps at global batch 16 (about 4.4 epochs), LR 1e-5 constant.
- Loss: `--loss-mask --mask-norm pixels`.
- Only the final checkpoint is scored. No checkpoint is picked on val or test.

**Evaluation.** Detections are heatmap peaks via `peak_local_max`, as Stage 2 inference does. Each detection is
then held to the reading below that applies to it.

- **(a) Pixels, test split.** Matching is one-to-one with RampNet's matcher (seam wrap) at radius 0.05, Phase
  0's crosswalk radius.
  - Recall is over all 515 test panos' positives.
  - Precision is over complete panos only (`dense_ignore` and `verified_empty`). A detection inside an ignore
    region counts as neither TP nor FP. On `sparse` panos, unmatched detections are not scored, because their
    background is unverified.
  - Confidence threshold: the one that maximises F1 on the **val** split, applied unchanged to test.
  - Reported overall, per city, per supervision mode, and per capture year (≤2018, 2019–2022, 2023+).
  - The released curb-ramp weights, scored the same way, give the floor.
- **(b) Per OSM node, Seattle test split, against the map-only baseline.**
  - Nodes: the rows of `baseline_nodes.csv` (inside SDOT coverage) within 25 m of a test-split Seattle pano.
  - Prediction: a node is predicted marked when some detection, ground-raycast at 2.5 m camera height (the
    pilot's B1 geometry) and kept only within 25 m range, lands within 10 m of it.
  - Truth: a current SDOT point within 10 m, as in the baseline.
  - Compared with `osm_marked_signal` **on the same node subset**, not with the city-wide 0.846.
  - **Passes** if the detector's node F1 is ≥ the baseline's on that subset. Also reported:
    detector OR `osm_marked_signal`, and the detector on nodes OSM leaves untagged (where a detector adds most).
- **(c) Phase 1 GT** (Jon's blind ratings across four cities, none of it trained on). Pre-registered separately
  once the ratings are in. The ratings are not looked at before then.

## Draft issue for Jon to post (not posted)

> **Title:** Stage 1 pano discovery is broken: `GeoPhotoService.SingleImageSearch` has been decommissioned
>
> `stage_one/dataset_generation/search_panos.search_panoramas` calls
> `maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch`. Since at least 2026-11-03 every call
> returns HTTP 200 with the body
> `[[5,"generic","GeoPhotoService.SingleImageSearch is decommissioned and turned down."]]`, which
> `extract_panoramas` turns into an `IndexError`. `generate_dataset_meta.py` catches it per row and prints
> "Error processing row", so a run looks like it worked while discovering nothing.
>
> Still working: `GetMetadata` (`get_pano_heading`, `get_date_of_panorama`) and the tile fetch in
> `rampnet.gsv.fetch_panorama`.
>
> Replacement used by the crosswalk pilot (`scripts/analysis/stage1_crosswalk_sdot_274/pilot.py meta`):
> - streetlevel 0.12.10 `get_coverage_tile` (z17) for the current panos near a point;
> - `find_panorama_by_id` for each pano's historical captures, and for each capture's own position, heading
>   and date.
>
> On 937 panos the production `get_pano_heading` equals streetlevel's heading exactly. Two side effects:
> - Discovery then includes every historical capture, about 56 panos per point within 20 m in Seattle.
> - streetlevel's `pyequilib` dependency pulls a newer CPU torch on Windows.
>
> Also worth knowing: the panos Project Sidewalk already stores (makelab2 `sidewalk_panos/`) cover the crowd's
> labelled panos almost completely (3,292 of 3,301 in Seattle). GSV no longer serves metadata for about half of
> them, but the PS server's `/backupImage/<id>/metadata` does.
>
> 🤖 Generated with [Claude Code](https://claude.com/claude-code) — Opus 5.5, claude-opus-5-5

## Replication

```bash
P=scripts/analysis/stage1_crosswalk_sdot_274/pilot.py; OUT=analysis_out/stage1_crosswalk_sdot_274
INV=<crosswalk-inventory-274>/analysis_out/crosswalk_inventory_274
python $P sample --inventory $INV --out $OUT [--phase1-sheet <crosswalk-gt-phase1>/analysis_out/crosswalk_gt_phase1_274/sheet.csv]
python $P meta --inventory $INV --census <Phase 0 census OUT> --out $OUT    # network: streetlevel tiles + metadata
python $P fetch --out $OUT                                                  # network: GSV pixels (~1 GB/100 panos)
python $P place --out $OUT                                                  # GPU: Vistas, ~25 min on an RTX 3070
python $P report --out $OUT && python $P figures --out $OUT && python $P gallery --out $OUT
python -m pytest tests/test_sdot_source_274.py && python -m doctest stage_one/dataset_generation/sdot_source.py
```

Needs RampNet's environment plus `streetlevel==0.12.10` (its `pyequilib` dependency pulls a newer CPU torch on
Windows; reinstall `torch==2.6.0+cu126` after it). `search_cache.jsonl` is the network record, and coverage
churns, so a fresh `meta` moves the yield numbers.

| number | lives in |
|---|---|
| sample, strata | `sample.csv`, `sample_meta.json` |
| yield | `discovery.csv` |
| clean rule, drop reasons | `dataset.jsonl` |
| the pixel-arm subset | `fetch_selection.txt` |
| on-paint, B2, shifts, per-year table | `placements.csv`, `report.md` §2–4b |
| figures | `docs/figures/stage1_crosswalk_sdot_274/` (`yield_hist.png`, `range_hit_rate.png`) via `pilot.py figures` |
| gallery (local, untracked: it points into `views/`) | `gallery.html` via `pilot.py gallery` |

![](figures/stage1_crosswalk_sdot_274/range_hit_rate.png)
