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

### Scoring implementation (fixed 2026-11-04, after training, before any score was read)

`scripts/analysis/stage1_crosswalk_sdot_274/score_274.py`. The text above left these choices open. They are fixed
here, in the commit that adds the scorer, before it read any model output.
1. **Peaks.** RampNet's frozen benchmark extractor (`operating_point_curve.py extract` via `threshold_sweep`):
   - resize to 2048×4096, ImageNet normalisation, fp32, no flip-TTA (the deployed labeler runs none);
   - `peak_local_max(min_distance=10, exclude_border=False)` down to a 0.05 floor, argmax decode, no seam NMS.
2. **Ignore regions.** Matching gives GT priority. An unmatched detection on a complete pano is ignored when
   either:
   - it is within the match radius (0.05) of an ignore point, as `detection_eval.score_pano` does; or
   - it falls inside one of the pano's `ignore_boxes`.
3. **Precision counts only complete panos**, TPs included. A TP on a `sparse` pano counts toward recall only.
4. **Threshold.**
   - Picked on a 0.05–0.95 grid in steps of 0.01, by F1 on val; ties go to the lowest threshold.
   - Each model gets its own val-chosen threshold, the released curb-ramp floor included.
5. **(b) geometry.**
   - Detections at the model's val threshold are inverted through the pilot's B1 projection (`pilot.project`,
     a round trip pinned by a doctest).
   - Pano position and heading come from streetlevel (`crowd_meta.jsonl`, the frame the SDOT arm projected in),
     else Project Sidewalk's record (`ps_meta.jsonl`, `negatives/neg_meta.jsonl`).
   - Node coordinates come from the Phase 0 census tiles that `baseline_nodes.csv` was built from.
   - "Untagged" means `osm_class == unknown`.
   - A node bootstrap of the F1 difference is reported as exploratory and is not part of the pass rule.
6. **A label-ceiling row.** The training labels themselves, scored as detections (`score_274.py oracle`), show
   what the geometry allows. Checked on the plumbing run:
   - pixels P = R = 1;
   - nodes F1 0.914 against `osm_marked_signal`'s 0.898, on the same 463 Seattle test nodes (315 true).
   So (b)'s bar sits 0.016 below the best a perfect pixel detector could score through this geometry.

### Results (scored 2026-11-05)

The run used the checkpoint `final_step_1000.pth` (sha256 `6f467ff4…`). Inference was klone job 42861556, A100,
0.63 s per pano. Full tables are in `analysis_out/stage1_crosswalk_detector_274/report.md`.

**(a) Pixels: a large lift over the curb-ramp floor.** Each model is scored on test at its own val-F1 threshold:

| model | threshold | precision (complete panos) | recall (all positives) | F1 |
|---|---|---|---|---|
| fine-tune `crosswalk_s1` | 0.31 | 0.889 [0.861, 0.912] | 0.761 [0.736, 0.785] | 0.820 |
| released curb-ramp weights | 0.05 (grid floor) | 0.198 [0.178, 0.220] | 0.413 [0.386, 0.442] | 0.268 |

- **Per city**, recall is Amsterdam 0.786, Burnaby 0.839, Seattle 0.752 and Teaneck 0.730 (37 positives). Only
  Seattle has complete panos, so precision exists only there.
- **By capture year**, F1 is 0.771 for ≤2018 (30 panos), 0.821 for 2019–2022 and 0.823 for 2023+.
- **On the 31 verified-empty test panos**, the fine-tune makes no detection at all.
- Test F1 (0.820) is above val F1 (0.778).

**(b) Per node: FAILS as pre-registered.**

| | P / R / F1 |
|---|---|
| detector | 0.832 / 0.927 / 0.877 |
| `osm_marked_signal` | 0.854 / 0.946 / 0.898 |
| detector OR OSM | F1 0.895 |

- The scores are over 463 Seattle test nodes, 315 of them marked.
- On the 38 nodes OSM leaves untagged (9 marked), the detector scores 0.889 / 0.889 / 0.889 against OSM's
  0.625 / 0.556 / 0.588.

**Post hoc, explaining (b) without re-deciding it.**
- **The node subset is not the city.** 68% of these nodes are marked, against 30% city-wide, and OSM tags 74% of them
  marked, against 33%. The cause is that the test panos were chosen where the crowd clicked crosswalks, which is
  OSM's easiest case. Saying yes at every node already scores F1 0.810 here, against 0.459 city-wide.
- **The ceiling is tight.** The label ceiling is 0.914, so (b) has 0.016 of headroom.
- **No threshold rescues it.** The detector's node F1 peaks at 0.885, at threshold 0.25.
- **Where the two disagree** (at the val threshold):
  - the detector alone is right on 13 marked nodes that OSM misses;
  - OSM alone is right on 19;
  - the detector alone says yes on 18 nodes that are not marked, and OSM alone on 10.

Proposed next test: (b′) below, pending Jon's approval.

![](figures/stage1_crosswalk_detector_274/pr_f1.png)

### (b′) A representative Seattle node sample (fixed 2026-11-05 before any (b′) inference)

Why: (b)'s nodes came with the test panos, which were picked where the crowd clicked crosswalks. The question (b)
asks, whether a street-level detector beats the free map at a Seattle crossing, needs Seattle's nodes as they
are. This is a new test. It does not re-decide (b), which stands as failed.

- **Population.** A node is eligible when all of these hold:
  - it is in `baseline_nodes.csv` (inside the regions and inside SDOT coverage);
  - it is more than 60 m from every Seattle train/val pano;
  - some Project Sidewalk store pano lies within 25 m of it.

  That leaves 13,550 eligible nodes. Of the 22,020, 6,455 were dropped as near train/val and 2,015 for having no
  store pano. 27% of the eligible nodes are marked, against 30% city-wide.
- **Sample.** `score_274.py bprime-sample`, seed 274: 150 nodes per (OSM class × SDOT truth) cell, or all of a
  smaller cell. Untagged-and-marked has 126.
  - 876 nodes on 793 panos (`analysis_out/stage1_crosswalk_detector_274/bprime/sample.csv`, `sample.json`).
  - Each node carries the weight `eligible ÷ sampled` of its cell.
  - Truth-stratified sampling is valid because the weights restore the population. It gives every cell enough
    nodes to read both recall and false positives.
- **Pano.** Each node is scored from one pano: the newest Project Sidewalk store pano within 25 m, ties to the
  nearest.
  - Position, heading and date come from `sidewalk_seattle.pano_data`, exported read-only on 2026-11-05.
  - Pixels are the makelab2 store copy, written at 4096×2048 exactly as the training root was
    (`bprime-prep` → `export_stage2.write_image`).
- **Prediction.** The fine-tune at its pre-registered val threshold 0.31 (the released weights at theirs, 0.05),
  with the frozen extractor. Detections on the node's own pano go through (b)'s B1 inversion (2.5 m, ≤ 25 m).
  The node is called marked when one lands within 10 m. Truth is a current SDOT point within 10 m, as before.
- **Metrics.** P, R, F1 and MCC, population-weighted, for:
  - the detector;
  - `osm_marked_signal`;
  - detector OR OSM;
  - always-yes.

  95% intervals come from a within-cell bootstrap (2,000 draws, seed 274). Also reported: per-cell call rates,
  and recall on marked nodes by the pano's capture year.
- **Pass rule (adopted 2026-11-05 with Jon's go-ahead, before any (b′) inference).**
  - **Primary:** "does the detector improve the map?" Weighted F1 of detector OR `osm_marked_signal` exceeds
    weighted `osm_marked_signal` F1, and the 95% within-cell bootstrap interval of the difference lies above 0.
  - **Secondary, reported:** weighted detector F1 ≥ weighted `osm_marked_signal` F1, (b)'s rule.
  - The thresholds are the pre-registered val picks; nothing is tuned on (b′).
- **Plumbing check** (a truth oracle, no model output): weighted F1 is 0.854 for OSM (city-wide 0.846) and 0.424
  for always-yes. The test has headroom where (b) had 0.016.

#### (b′) results (klone job 42902924, scored 2026-11-05): FAILS both rules

`analysis_out/stage1_crosswalk_detector_274/bprime/report.md`. Population-weighted, 876 nodes:

| predictor | P | R | F1 | MCC |
|---|---|---|---|---|
| `osm_marked_signal` | 0.814 | 0.898 | 0.854 | 0.799 |
| detector (val threshold 0.31) | 0.745 | 0.636 | 0.686 | 0.586 |
| detector OR OSM | 0.732 | 0.940 | 0.823 | 0.758 |
| always-yes | | | 0.424 | |

- **Primary fails:** the OR − OSM F1 difference is −0.031, with 95% interval [−0.059, −0.004].
- **Secondary fails:** the detector − OSM difference has interval [−0.224, −0.115].
- **The released curb-ramp weights:** detector F1 0.463.

Per cell (unweighted), the detector finds crosswalks OSM misses:
- 37% of the marked nodes OSM tags unmarked (OSM: 0%);
- 63% of the marked untagged nodes (OSM: 40%).

But it also says yes at 6% and 4% of those classes' unmarked nodes. Those classes hold 6,959 and 2,198 unmarked
nodes, against 296 and 126 marked, so the false alarms outweigh the finds. OSM-marked nodes that SDOT does not
mark draw a yes 39% of the time.

**Post hoc (explains, never re-decides):**
- **Using the detector only where OSM is silent.**
  - OSM, plus a detector yes on untagged nodes: weighted F1 0.850.
  - The detector's call replacing OSM on untagged nodes: 0.849.
  - Neither beats OSM alone (0.854).
- **Recall dropped from (b)'s 0.927 to 0.636. About half of that is the 10 m match, not missed paint.**
  - Of 192 missed marked nodes, 81 have no detection ≥ 0.31 on the pano.
  - The other 111 do, but the nearest lands a median 13.2 m from the node (74 within 15 m).
  - The pano–node distance does not explain the misses (median 6.7 m for misses, 8.3 m for hits).
- **Recall on marked nodes by pano capture year:** 0.59 for 2019–2022, 0.45 for ≤2018, 0.41 for 2023+ (56
  nodes).

**Reading.** At Seattle's crossing nodes, OSM is already strong (F1 0.85), and only about 4% of the nodes OSM
calls unmarked or leaves untagged are marked. Any per-node false alarm therefore costs more than a find gains. So
this detector does not improve Seattle's map through a node-level rule. Its demonstrated value is per-pano
labelling, (a).

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
# first crosswalk detector (scores): inference on klone, then CPU scoring anywhere
S=scripts/analysis/stage1_crosswalk_sdot_274/score_274.py; D=analysis_out/stage1_crosswalk_detector_274
RAMPNET_ENV=<conda prefix> sbatch scripts/analysis/stage1_crosswalk_sdot_274/score_274.slurm   # -> $D/preds (committed)
python $S oracle --root <Stage 2 root> --preds $D/preds
python $S score --root <Stage 2 root> --preds $D/preds --labels crosswalk_s1,released,label_ceiling     --baseline-nodes <crosswalk-inventory-274>/analysis_out/crosswalk_inventory_274/baseline_nodes.csv     --census <Phase 0 census out> --out $D
python $S figures --out $D && python $S gallery --root <root> --preds $D/preds --scores $D --label crosswalk_s1 --out <outside the repo>
python -m doctest $S
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
| detector predictions (peaks >= 0.05, per pano) | `analysis_out/stage1_crosswalk_detector_274/preds/<label>/{val,test}.jsonl` + `meta.json` (sha256) |
| (a) thresholds, overall / per-group P-R-F1 | `report.md`, `scores.json`, `threshold_sweep.csv`, `pixel_breakdown.csv` |
| (b) per-node calls, post-hoc node sweep and agreement | `nodes.csv`, `scores.json` (`nodes`, `nodes_exploratory`) |
| detector figures | `docs/figures/stage1_crosswalk_detector_274/` via `score_274.py figures` |
| detector gallery (local, untracked: embeds imagery) | `score_274.py gallery` |

![](figures/stage1_crosswalk_sdot_274/range_hit_rate.png)
