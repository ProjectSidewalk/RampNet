# Argon as a teacher for RampNet's open-rig misses: Stage 0

**Plan:** `argon-teacher-2026-11-15` (revised after review), Stage 0 only. **Run date:** 2026-11-15.
**Script:** `scripts/analysis/argon_teacher_s0.py` (tests: `tests/test_argon_teacher_s0.py`).
**Outputs:** `analysis_out/argon_teacher_s0/results.json`, plus two derived inputs in the same
directory (`neighbours.json`, `heatmap_features.json`); `docs/figures/argon_teacher/` (two PNGs).
**Compute:** one RampNet forward pass per benchmark pano (469 panos) on the local RTX 3070,
10 minutes; everything else on the CPU in under 10 seconds. No model API call, no remote host,
no spend. Lyon was not read.
**Status:** Stage 0 of a pre-registered plan. The gate bars are the plan's. They were set after
the per-split Argon-only precisions were published (plan section 3), and nothing was tuned here.
Stage 1 was not started.

## Result

**Gate B passes. Gates P, Y and S all fail at their kill bars. Under the plan, the teacher run
stops here.**

| Gate | Plan bar (go / kill) | Measured | Verdict |
|---|---|---|---|
| B, leak | H2 ≤ 5%, else E ≤ 5%, else stop | H2 0.6% (7 / 1,147), with `budapest_district5` as an upper bound; H alone 3.0% | **go** (via H2) |
| P, precision | ≥ 0.80, no split with 10+ boxes under 0.60 / < 0.65 | 0.41 (7 TP / 10 FP), held out | **kill** |
| Y, yield | ≥ 30 / < 20 of the 65 reached blind ramps kept | 8 | **kill** |
| S, strict-silent | ≥ 10 kept and ≥ 25% of those Argon reaches / fewer | 8 of 42 (19%) | **kill** |

The filter result is not a near miss, and it does not depend on the leave-one-split-out
mechanics. **No candidate reaches even the 0.65 kill bar in sample.** The best of the 80, scored
on all four splits at once, is `f2<=3m & f4_yolo` at 0.64 (25 TP / 14 FP, 23 blind ramps kept).
Under the selection rule, three of the four folds find no candidate at precision ≥ 0.80 on
their training splits. Only the `bayonne` fold selects one (`f2<=3m & f4_yolo +snap`, 0.82 in
training), and it scores 0.41 on `bayonne`.

**What the plan says happens next:** gate B alone does not keep the plan alive. The plan stops
when a filter gate is at kill. "Between go and kill on P, the default is stop; Jon may choose
rating R0" applies only to the band between the bars, and P is below it. Stage 1 and Stage 2 do
not start.

![Leak by rule](figures/argon_teacher/leak.png)

*Figure 1. Gate B. Share of sure ramps left in background, by rule. `budapest_district5` has no
labeler run with stored peaks, so its H2 bar equals its H + far band bar (an upper bound). E is
0 wherever it is drawn, and `budapest_district5` has no matched empty pano. Regenerate with
`argon_teacher_s0.py figures`.*

![Candidate filters](figures/argon_teacher/filters.png)

*Figure 2. Gates P and Y. Every candidate scored in sample on all four splits (pooled), against
the go lines. No dot reaches 0.65. The star is the held-out result the gates read.*

## Gate B: background leak

"Sure ramp" means a GT point (a reviewer-confirmed RampNet detection, or a missed mark not
flagged unsure) on a recall-confirmed pano. All 469 panos are recall-confirmed, so the
denominator is the plan's 1,147.

| split | GT | H | H, both Argon legs | H + far band | H + neighbours | H2 | E: matched empties (with a ramp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| laurens_mapillary | 249 | 4 (1.6%) | 2 (0.8%) | 3 (1.2%) | 0 (0.0%) | 0 (0.0%) | 6 (0) |
| juneau | 297 | 6 (2.0%) | 2 (0.7%) | 2 (0.7%) | 1 (0.3%) | 0 (0.0%) | 11 (0) |
| bayonne | 301 | 17 (5.6%) | 12 (4.0%) | 7 (2.3%) | 5 (1.7%) | 2 (0.7%) | 1 (0) |
| budapest_district5 | 300 | 7 (2.3%) | 5 (1.7%) | 5 (1.7%) | no data | 5 (1.7%), upper bound | 0 |
| **pooled** | 1,147 | 34 (3.0%) | 21 (1.8%) | 17 (1.5%) | – | **7 (0.6%)** | 18 (0) |

- **H passes as well, against the plan's expectation.** The review estimated "about 10%" from
  counts made at radius R with YOLO at conf 0.25. The plan's H uses 2R discs and YOLO boxes down
  to 0.05, and that is what is measured here. With one YOLO arm instead of the union, H is 3.1-3.6%
  (`y26_pano` 35, `y11x_pano_h200` 40, `y11l_pano` 41 leaked).
- **H2 is read twice.** Over all four splits, with `budapest_district5` at its H + far band value,
  it is 0.6%. Over the three splits with neighbour data it is 0.2% (2 / 847). Both reads pass, so
  the gate does not have to fall through to E.
- **E is uninformative.** It is 0 of 18 matched empty panos. Eighteen panos is too few to bound a
  5% leak, and E's weakness (easy scenes only) is stated in the plan. It is reported, not used.

## Gates P, Y, S: filtering Argon-only boxes

**Box funnel** (zero-shot Argon leg, recall-confirmed panos): 915 boxes; 236 with no RampNet
peak ≥ 0.05 within R (Argon-only); 10 dropped by the rig mask and 51 within 2R of a peak ≥ 0.30;
**175 boxes** carry features (`laurens_mapillary` 28, `juneau` 19, `bayonne` 68,
`budapest_district5` 60).

**Leave-one-split-out** (80 candidates; held out at R):

| held-out split | selected on the other three (training TP / FP, blind kept) | eligible | held-out TP / FP / ignored | precision | blind kept | strict-silent kept |
|---|---|---:|---|---:|---:|---:|
| laurens_mapillary | none | 0 | 0 / 0 / 0 | – | 0 | 0 |
| juneau | none | 0 | 0 / 0 / 0 | – | 0 | 0 |
| bayonne | `f2<=3m & f4_yolo +snap` (18 / 4, 10) | 6 | 7 / 10 / 0 | 0.412 | 8 | 8 |
| budapest_district5 | none | 0 | 0 / 0 / 0 | – | 0 | 0 |
| **pooled** | | | **7 / 10 / 0** | **0.412** | **8** | **8** |

The `bayonne` fold's choice scores 0.89 on `laurens_mapillary` (16 / 2), 0.50 on `juneau`
(2 / 2) and keeps nothing on `budapest_district5`, which has no feature 2. Precision on one
split does not carry to the next.

**Without feature 2.** Feature 2 is unavailable on `budapest_district5`, so every box there
fails it. With the 32 feature-2 candidates removed, no fold has an eligible candidate, and all
three gates fail at kill with nothing kept. The verdict is the same either way.

**Which features help.** The highest in-sample precisions all pair feature 4 (a YOLO box ≥ 0.05
within 2R) with feature 2 (support from a neighbouring pano) or feature 3 (Argon's own
confidence). Features 2 and 4 depend on RampNet and on YOLO, which was trained on RampNet's data.
Feature 3 is the only independent feature that appears there, and only paired with feature 4.
Feature 5 alone (elevation in range) is close to no filter: 0.32 (48 / 101).

## Method

Inputs, all committed except the labeler runs behind `neighbours.json`:

- GT: `benchmark/<split>/` via `cascade_cost_35.load_gts`. RampNet peaks: `analysis_out/op_cache/`
  (floor 0.05). Argon: `benchmark/model_detections/gemini-4-argon__<split>.json` (box centres) and
  `gemini-4-argon-confidence-on__<split>.json` (feature 3's scores). YOLO: the three `*_pano`
  arms in the same directory, published down to 0.05.
- `heatmap_features.json`: one forward pass of the published RampNet checkpoint per pano, with
  the op_cache's preprocessing and decode (`threshold_sweep.heatmap_for`, `peaks_to_dets`). Every
  image is checked against `imagery_manifest.json`'s sha256. It reproduces all 1,959 op_cache
  peaks of the four splits to within one pixel. It keeps, per zero-shot Argon box, the raw
  heatmap maximum within R and 2R, plus the heatmap modes ≥ 0.01 lying within 2R of some box.
- `neighbours.json`: for each benchmark pano with a labeler run, every stored peak (floor 0.10)
  of every other pano within 50 m, raycast to flat ground at 2.6 m and projected into this pano
  with the labeler's own `geo.detection_ground_point` / `geo.ground_point_to_pano`. Every fused
  site is projected the same way. It also keeps each zero-shot Argon box's distance from its
  ground point to the nearest other-pano peak, or to a site with a member from another pano
  (feature 2).

Rules, as the plan states them:

- **R** = 0.022 of pano width; distances wrap at the seam (`rampnet.geometry.dist_sq`).
- **H**: a sure ramp leaks if no disc of radius 2R holds it. Discs are centred on RampNet peaks
  ≥ 0.05, zero-shot Argon boxes, and YOLO boxes ≥ 0.05 from any of the three arms.
- **H2**: H plus 2R discs at every projected neighbour peak and fused site. A ramp less than 6°
  below the horizon (y < 0.5 + 6/180) is ignored. At 2.6 m, 6° below the horizon is 24.7 m out,
  so the far band starts about where the 25 m raycast ends.
- **E**: "empty" means no RampNet peak ≥ 0.05, no zero-shot Argon box and no YOLO box ≥ 0.05
  (rig region included). An empty pano is matched when its `(sequence_id, camera_model)` also
  holds a non-empty pano of the split.
- **Gate B**: H2 ≤ 5%, else E ≤ 5%, else stop.
- **Features** (per Argon-only box, after the rig mask, dip > 49°, and the ≥ 0.30-within-2R drop):
  (1) heatmap max within R ≥ 0.01 / 0.02 / 0.03; (2) support within 3 / 5 m;
  (3) score of the nearest confidence-leg box within R ≥ the median (0.68) / upper quartile
  (0.76) of all 1,360 confidence-leg scores; (4) a YOLO box ≥ 0.05 within 2R;
  (5) y inside the 5-95% range of all 915 zero-shot boxes (0.526-0.715); (6) snap to the nearest
  heatmap mode ≥ 0.01 within 2R, where one exists.
- **Candidates**: the 9 single tests and the 31 AND-pairs of tests from different features, each
  with and without the snap (80).
- **Selection** per held-out split: among candidates whose pooled precision on the other three
  splits is ≥ 0.80, take the one keeping the most blind ramps there. Ties go to higher precision,
  then a single test over a pair, then no snap. The choice is scored at R on the held-out split
  with `score_pano`. A fold with no eligible candidate keeps nothing.
- **Gates** read the four held-out results pooled. Y counts blind ramps (no peak ≥ 0.05 within R)
  with a kept box within R. S counts strict-silent ramps (none within 2R) kept, against the 42
  strict-silent ramps any zero-shot box reaches.

## Amendments

Places where the plan was ambiguous or did not fit the data, the reading taken, and whether the
other reading changes a verdict.

1. **"Argon box" means the zero-shot leg**, the leg behind the plan's 65 (reproduced: 14 + 6 +
   32 + 13). The confidence leg supplies feature 3's score only. H with both legs is reported
   (1.8%). No verdict changes.
2. **"YOLO box" means any of the three YOLO pano arms** (union), for H, E and feature 4. H with
   one arm is 3.1-3.6%. Gate B does not change.
3. **Feature 5 is elevation only.** The published Argon outputs, and the local detector cache,
   keep box centres, not box extents, so box size cannot be computed. The H discs are centred on
   box centres for the same reason.
4. **Feature 3**: a zero-shot box takes the score of the nearest confidence-leg box within R, and
   fails if there is none. The median and upper quartile, and feature 5's range, are taken over
   all four splits rather than per fold. They are statistics of Argon's output alone and use no GT.
5. **Pairs are ANDs.** The plan says "in pairs" without an operator. The OR reading was checked
   once, by hand, in sample on all four splits (not committed): the best OR pair is
   `f3>=q75 | f4_yolo` at 0.48 (37 / 40). The verdict does not change.
6. **The selection rule's objective.** The plan fixes the 0.80 eligibility bar but not what to
   maximise among eligible candidates. Blind ramps kept (gate Y's quantity) was used, with the
   tie-breaks above. Three folds have no eligible candidate, so the objective only matters for the
   `bayonne` fold.
7. **P's kill reading.** "With every unmatched box counted true" taken literally makes every
   precision 1, and then the kill could never fire. It is read as: boxes the scorer ignores (near
   a reviewer's unsure mark) count as true. There are none among the kept boxes, so both
   precisions are 0.41.
8. **Neighbour data.** `budapest_district5` has no usable labeler run: its `results.jsonl` holds
   300 panos at the old 0.55 floor, 2 of them benchmark panos. H2 is therefore read two ways,
   above, and feature 2 fails on every box there (sensitivity read without feature 2, above). The
   `juneau` run was read from the labeler worktree `sal-juneau`, because the main checkout's
   `runs/juneau/` holds only submission records. Fused sites existed only for Laurens, made with
   older parameters, so all three were re-fused with the labeler's current `fuse_sites.py`
   defaults into scratch (not committed; hashes below).
9. **The far band** is measured in the pano frame (y < 0.5 + 6/180), with no tilt correction, and
   it includes everything above the horizon.
10. **E's "empty"** uses the same 0.05 floors as H and counts detections in the rig region, which
    is the stricter reading.
11. **Gate S** has no in-between band in the plan ("fewer" is the kill), so it reads go or kill.

## Caveats

- **The gates score at R.** [`cascade_argon_35.md`](cascade_argon_35.md) shows Argon's box
  centres are looser than YOLO's, and a 2R match would raise every precision here. The plan
  fixes R, and feature 6 (the snap) is the plan's answer to localisation. The snapped variants do
  no better in sample.
- **FPs are upper bounds.** GT is RampNet's reviewed detections plus the reviewer's missed marks,
  so a real ramp that neither marked counts as an FP. This holds for every split and every
  candidate alike.
- **Small held-out counts.** The 175 featured boxes split 28 / 19 / 68 / 60. No interval is
  computed, and the plan's gates are on point values.
- **E rests on 18 panos**, too few to bound a 5% leak. Gate B passes on H2 regardless.
- **One Argon run per leg**, as in the cascade read.

## Replication

Commit: the commit that adds this file, on `analysis/argon-teacher-s0` (based on `cascade-argon-b0`,
`b3868c8`).

```bash
# 1. CPU, committed inputs only: leak + features + filters -> results.json, then the check
python scripts/analysis/argon_teacher_s0.py run
python scripts/analysis/argon_teacher_s0.py --check      # byte-compares results.json (~1 s)
python scripts/analysis/argon_teacher_s0.py leak         # or: features, filters (print tables)
python scripts/analysis/argon_teacher_s0.py figures      # needs matplotlib
pytest -q tests/test_argon_teacher_s0.py

# 2. Derived inputs (only to rebuild them; both are committed)
#    neighbours.json: needs the labeler runs; sites re-fused first with the labeler at 1e0e96e
python <labeler>/scripts/fuse_sites.py <labeler>/runs/laurens/results.jsonl --out <tmp>/laurens_mapillary.sites.jsonl
python <labeler>/scripts/fuse_sites.py <labeler>/runs/bayonne/results.jsonl --out <tmp>/bayonne.sites.jsonl
python <labeler>/scripts/fuse_sites.py <sal-juneau>/runs/juneau/results.jsonl --out <tmp>/juneau.sites.jsonl
python scripts/analysis/argon_teacher_s0.py extract-neighbours --labeler-root <labeler> --sites-dir <tmp>
#    heatmap_features.json: GPU, the benchmark panos (checked against imagery_manifest.json)
python scripts/analysis/argon_teacher_s0.py extract-heatmaps --panos-root <dir holding <split>/panos/>
```

Input hashes (sha256, first 16 hex). `results.json` records the full hashes of its two derived
inputs. `neighbours.json` records the full hashes of the run and site files.

| input | sha256 |
|---|---|
| `analysis_out/argon_teacher_s0/neighbours.json` | `88ffb71e7449a147` |
| `analysis_out/argon_teacher_s0/heatmap_features.json` | `7676c6a739f9dcdf` |
| labeler `runs/laurens/results.jsonl` (4,495 panos) | `16c5a348b739274b` |
| labeler `runs/bayonne/results.jsonl` (50,409 panos) | `cb03768cc4cefd0b` |
| `sal-juneau` `runs/juneau/results.jsonl` (4,999 panos; worktree at `2e3f238`) | `5f7822fb3f977d74` |
| re-fused sites: laurens_mapillary (813) / bayonne (14,644) / juneau (1,335) | `cf0e07b120f88591` / `2ff67c08c4dc62f4` / `8f7f83b9f4ecb116` |
| `benchmark/model_detections/gemini-4-argon__{laurens_mapillary,juneau,bayonne,budapest_district5}.json` | `21a622145d797982` / `49d8fe9db2659e4e` / `9bfde72653caea44` / `8075badffc4e0d2d` |

`--check` (also `check_all.py --only argon_teacher_s0`) covers every number in this note except
the two derived inputs, which need the labeler runs or the GPU. The figures are not
byte-compared.

## Where each number lives

All keys are in `analysis_out/argon_teacher_s0/results.json` unless stated.

| number | key |
|---|---|
| 469 panos, 1,147 GT, 196 blind, 65 reached, 105 strict-silent, 42 reached | `counts.pooled` (`n_panos`, `n_gt`, `n_blind`, `blind_reached`, `n_strict_silent`, `ss_reached`); per split in `counts.per_split` |
| leak table (H, both legs, far band, neighbours, H2, E) | `leak.per_split.<split>.*`, `leak.pooled.*` |
| H with one YOLO arm | `leak.pooled.H_yolo_<arm>_only` |
| H2 over the splits with data | `leak.pooled.H2_splits_with_neighbours` |
| gate B verdict | `leak.gate_B` |
| feature thresholds (0.68, 0.76, 0.526-0.715) | `thresholds` |
| box funnel (915 / 236 / 10 / 51 / 175) | `feature_drops.<split>` |
| per-box features | `features[]` (also `argon_teacher_s0.py features`) |
| every candidate on every split | `filters.candidates.<name>.<split>` |
| LOSO choices and held-out rows | `filters.loso.<split>` |
| pooled held-out row, gates P / Y / S | `filters.held_out_pooled`, `filters.gates` |
| sensitivity without feature 2 | `filters_without_f2` |
| op_cache peaks reproduced (1,959 / 1,959) | `heatmap_sanity`; source `heatmap_features.json` |
| figures | `docs/figures/argon_teacher/{leak,filters}.png`, from `leak` and `filters.candidates` |
