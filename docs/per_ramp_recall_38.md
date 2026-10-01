# Per-ramp recall and how correlated the per-view misses are ([#38](https://github.com/ProjectSidewalk/RampNet/issues/38))

[#38](https://github.com/ProjectSidewalk/RampNet/issues/38) argued that the benchmark reports
per-pano recall (about 0.77 at the time) while the product is a city inventory, which needs
per-ramp recall over every pano that sees a ramp. It listed four items: measure per-ramp
deployment recall, quantify how correlated the misses across views are, sweep sampling density,
and consider distance-aware aggregation.

Most of this has since been measured elsewhere. This document first lists what is already
answered, where, and with which caveats (§1). It then adds what was missing (§2-§5), computed
from one committed table on CPU: confidence intervals and a permutation null for the
miss correlation, a description of the ramps every view missed, a thinning sweep, and
nearest-view vs any-view recall with CIs.

Code: `scripts/analysis/per_ramp_recall_38.py`. Outputs: `analysis_out/per_ramp_recall_38/`
(`results.json`, `ramps_other_views.csv`). Test: `tests/test_per_ramp_recall_38.py`.

## Takeaways

- **Per-ramp deployment recall is measured, and it is far above per-pano recall.** On five cities
  the union of views finds 0.945-1.000 of the GT ramps within 25 m
  ([labeler#27](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/27)). On Laurens,
  against one auditor's own labels, the deployment has an AI cluster within 7.5 m of 0.913 of
  them, where the same ramps are found in their own pano 0.517 of the time
  ([labeler PR #119](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/119), not
  merged). Both rest on one rater per city.
- **Misses are correlated, about 2x the independence prediction, and the interval excludes 1.**
  Ramps missed by every other view: 153 observed vs 76.5 predicted, ratio 2.00
  [95% CI 1.74, 2.24]; no permutation of 5,000 reached 153 (null mean 76.3, 95th percentile 88).
  GSV 1.80 [1.57, 2.04], Mapillary (Richmond) 4.62 [2.92, 6.92].
- **#38's 0.3% was too low by about 50x.** The three nearest other views all miss 15.9% of ramps
  [13.9, 17.8]; independence on the same instrument predicts 7.8%. Most of the gap from 0.3% is
  that other views miss far more often than the per-pano benchmark (§2), and the rest is
  correlation.
- **The ramps no view finds are not far-field ramps.** Of the 55 ramps missed by every view
  including the GT-source view, the nearest other camera is at a median 7.1 m (6.6 m for all
  ramps). 27 have a stored peak below 0.55 in some other view.
- **Thinning costs recall about in proportion to the captures removed, and the data cannot say
  whether denser sampling would add any.** Re-thinning Richmond's Mapillary run on a new 5 m grid
  keeps 80% of its panos and costs 1.4 points of other-view recall [0.6, 2.4]; at 10 m it keeps
  52% and costs 6.2 points. GSV is already at about 10 m spacing and is not thinned in production,
  so no setting changes it below 10 m. No un-thinned run exists to test denser than native.
- **A nearest-view-only rule loses a quarter of the recall.** The nearest other view alone finds
  0.654 [0.628, 0.679] of ramps; any other view within 25 m finds 0.904 [0.887, 0.920].
  Distance-weighted evidence was already tested for precision in #48 §6 and did not beat k-of-n.

## 1. What is already answered

| #38 item | answered by | number | caveats that travel with it |
|---|---|---|---|
| 1. Per-ramp deployment recall | [labeler#27](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/27), re-read in [`docs/multiview_48.md`](multiview_48.md) §1 ([PR #200](https://github.com/ProjectSidewalk/RampNet/pull/200), merged) | Union of views per ramp, no fusion: paterson 1.000, gainesville 0.945, sao_paulo 0.980, richmond 0.972, bend 0.983. After fusion into one site per ramp: 0.927-0.957. Decoy GT points displaced 30 m are "found" 0.02-0.11 of the time, so the union figure is not the match radius doing the work. | GT is one reviewer per city, built from RampNet detections reviewed at 0.55 plus the reviewer's missed marks, so it is anchored to RampNet. The pool holds only GT points that raycast within 25 m of their source camera; 257 of 1,584 ramps (richmond 57, paterson 91, gainesville 53, bend 30, sao_paulo 26) are outside it, and those are the far-field ramps that dominate per-pano misses. 5 m match radius on a flat-ground 2.6 m raycast (GT placement error p50 1.9 m / p90 4.4 m). Four GSV cities, one Mapillary city. |
| 1. Per-ramp deployment recall, independent GT | [labeler PR #119](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/119) (`docs/server-agree-check.md`, open), summarised on [#158](https://github.com/ProjectSidewalk/RampNet/issues/158) | Laurens: of the auditor's 230 CurbRamp labels, an AI cluster lies within 5 m of 0.830, within 7.5 m of 0.913 [0.870, 0.943], within 10 m of 0.974. In the pano frame the same comparison is 92/178 = 0.517. | One rater (the auditor also cast nearly all the human votes); 44 of 169 streets completed at the pull, so provisional. Laurens only, Mapillary GoPro Max rig. The auditor labels each ramp once, so this is recall only; it gives no false-positive rate. The server's 7.5 m clustering is the unit, not RampNet's fusion. |
| 2. Failure correlation | [`docs/multiview_48.md`](multiview_48.md) §5 | Ramps missed by every other view: 153 vs 76.5 predicted (2.0x); GSV 1.8x, Mapillary 4.6x. Joint-miss ratio by camera separation 1.27-1.79. Pairs from different months are more correlated than pairs from one month (1.45 vs 1.24). An 8 m world test halves the count and leaves the ratio at 2.0x. | No confidence interval or significance test was given. Same GT and pool caveats as item 1. This document adds both (§2). |
| 3. Sampling density | Partly: [`docs/multiview_48.md`](multiview_48.md) §4 (recall vs the k nearest other captures) | The first other view finds 0.652 of ramps, two 0.781, three 0.833, eight 0.870 (pooled, 0.55). | Says how recall grows with the number of captures used, not with pano spacing, and not at production's thinning. This document adds a thinning sweep (§4). |
| 4. Distance-aware aggregation | [`docs/multiview_48.md`](multiview_48.md) §4 (R = 12 / 18 / 25 m) and §6 (evidence score) | Capping qualifying captures at 12 m holds recall to 0.807, 25 m lifts it to 0.888 (k = 8). A per-capture log-likelihood score with confidence and range bins, counting misses in range against a site, showed no detectable precision gain over k-of-n at the 0.30 tier. | §6's false class is 15-59 sites per city, and its GT under-credits sub-0.55 sites. This document adds a nearest-only vs any-view comparison with CIs (§5). |

Related but not used here: per-pano recall by distance on GSV's own depth
([`detection_recall_analysis.md`](detection_recall_analysis.md) §0) and by input resolution
([`input_res_sweep_25.md`](input_res_sweep_25.md)). The distance table #38 quoted was on the
older flat-ground axis.

## 2. Data and instrument

**One input.** `analysis_out/multiview_48/captures_R25.csv` (committed with
[PR #200](https://github.com/ProjectSidewalk/RampNet/pull/200)): one row per pair of a pool ramp
and a run pano whose camera is within 25 m of it, 12,823 rows, 1,327 ramps, five cities. Each row
carries the best stored detection confidence that claims the ramp in world space (its raycast
lands within 5 m, one-to-one claims in confidence order), the camera position and the capture
month. `captures_R25_hit8.csv` is the same with an 8 m world test. sha256 of every input is in
`results.json` → `inputs`. Ramps are keyed by `city:index`; no id is used without its city.

**Other views.** As in #48 §4-§5, the GT-source view is left out of every "other views" number,
because a verdict-true ramp's GT world point was raycast from that view's own detection, so the
source view's world test passes by construction. The "union" rows add it back. Its hit rate
reproduces per-pano recall (0.722-0.830 by city, #48 §3), so it is an honest per-pano
measurement; it is kept out of the correlation headline only because its geometry differs.

**Why other views miss so often.** At 0.55 the pooled marginal miss rate in other views is 0.375
at 0-6 m, 0.370 at 6-12 m and 0.601 at 12-18 m (#48 §5), against per-pano miss rates of about
0.16-0.28 in the source views. Other views are scored against a GT point carried over from another
pano by a flat-ground raycast, so GT placement error (p90 4.4 m against a 5 m radius) turns some
detections into misses, and a dual ramp claims only one detection. The 8 m arm is the check on
how much of the correlation that explains.

**The independence prediction.** For each ramp, the product of its views' marginal miss rates,
each taken from all other-view captures in the same city and range bin (0-6, 6-12, 12-18 m). The
count of ramps every view missed is compared with the sum of those products.

**Two uncertainty measures.**
- *Ramp-cluster bootstrap*, 2,000 resamples of ramps within each city. The marginals are
  re-estimated from each resample, so the CI covers their sampling error too.
- *Stratified permutation null*, 5,000 shuffles: within each (city, range bin) stratum the miss
  flags of all captures are shuffled across ramps. Every capture keeps its city and range bin and
  every stratum keeps its miss count; only which ramp a miss belongs to changes. The p-value is
  (1 + shuffles at or above the observed count) / (1 + 5,000), so 0.0002 is the smallest it can
  be.

## 3. Miss correlation with intervals

Ramps with at least two other captures within 18 m, world test at 5 m unless stated.

| population | floor | ramps | all other views missed | predicted, independent | ratio [95% CI] | permutation null mean / 95th pct | p |
|---|---|---|---|---|---|---|---|
| pooled, 5 cities | 0.55 | 1,298 | 153 (11.8%) | 76.5 (5.9%) | **2.00 [1.74, 2.24]** | 76.3 / 88 | 0.0002 |
| GSV (4 cities) | 0.55 | 1,058 | 128 | 71.1 | 1.80 [1.57, 2.04] | 70.9 / 82 | 0.0002 |
| Mapillary (richmond) | 0.55 | 240 | 25 | 5.4 | 4.62 [2.92, 6.92] | 5.4 / 9 | 0.0002 |
| paterson | 0.55 | 303 | 30 | 16.1 | 1.86 [1.36, 2.44] | 16.0 / 22 | 0.001 |
| gainesville | 0.55 | 213 | 49 | 33.7 | 1.45 [1.20, 1.72] | 33.3 / 40 | 0.001 |
| sao_paulo | 0.55 | 253 | 27 | 13.5 | 2.00 [1.43, 2.59] | 13.4 / 19 | 0.001 |
| bend | 0.55 | 289 | 22 | 7.8 | 2.83 [1.88, 3.96] | 7.8 / 12 | 0.001 |
| pooled, 4 cities (no bend) | 0.30 | 1,009 | 98 | 43.4 | 2.26 [1.92, 2.64] | 43.3 / 52 | 0.0002 |
| pooled, 4 cities (no bend) | 0.10 | 1,009 | 67 | 29.1 | 2.30 [1.85, 2.77] | 29.1 / 37 | 0.0002 |
| pooled, 3 m range bins | 0.55 | 1,298 | 153 | 75.2 | 2.03 [1.78, 2.29] | 74.9 / 87 | 0.0002 |
| pooled, 8 m world test | 0.55 | 1,298 | 83 | 41.3 | 2.01 [1.65, 2.37] | 41.0 / 50 | 0.0002 |
| pooled, union with the source view | 0.55 | 1,298 | 55 (4.2%) | 16.2 (1.2%) | 3.40 [2.62, 4.27] | 16.1 / 23 | 0.0002 |

The per-city rows use 1,000 shuffles, so their smallest possible p is 0.001. Bend and the 0.30 /
0.10 rows: bend stores nothing below 0.55, so it is left out of the lower floors; richmond's
sub-0.55 detections are the labeler's re-inference (#48 §3).

The pooled row reproduces #48 §5 exactly (153 and 76.52); the script asserts this before writing.

**#38's arithmetic, done per ramp.** #38 multiplied three per-view miss rates (0.158 x 0.158 x
0.121 = 0.3%) for the three nearest views. On the 1,215 ramps with at least three other captures
within 18 m (median distances of the three nearest: 6.5, 9.6 and 12.4 m):

| | all three nearest other views missed |
|---|---|
| #38's estimate (benchmark per-pano miss rates, independent) | 0.3% |
| independence, with this instrument's range-matched other-view miss rates | 7.8% (95.2 ramps) |
| observed | **15.9% [13.9, 17.8]** (193 ramps); ratio 2.03 [1.82, 2.21], p = 0.0002 |

At 0.30 (four cities, 958 ramps): 13.5% [11.3, 15.7] observed vs 5.8% independent, ratio 2.32
[2.03, 2.60].

**Reading.**
- The correlation is real at every floor, in every city, with both range binnings and with both
  world tests. The ratio sits near 2 pooled. The finer range bins move the prediction by 1.3
  ramps, so the coarse bins do not hide much distance confounding.
- It is not only GT placement. An 8 m world test removes 70 of the 153 all-missed ramps, but the
  ratio stays at 2.0.
- GSV and Mapillary differ: GSV 1.80 [1.57, 2.04] and Mapillary 4.62 [2.92, 6.92] do not overlap.
  Richmond's consecutive Mapillary frames are often under a metre apart (#48 §5), so its views
  are less independent by construction, and it is one city.
- Gainesville has the highest all-missed share (23%) and the lowest ratio (1.45): its other views
  miss often, more or less independently.
- Most of the distance between #38's 0.3% and the observed 15.9% is the marginal miss rate, not
  the correlation. Under independence this instrument already predicts 7.8%, because other views
  miss 37-60% of the time (§2), not 12-16%. Correlation doubles that.

## 4. The ramps every view missed

`ramps_other_views.csv` has one row per ramp with at least two other captures within 18 m: city,
imagery, number of other captures, nearest and median other-camera distance, capture months, the
source view's distance and hit, the best stored other-view confidence, whether every other view
missed, and its [#48](https://github.com/ProjectSidewalk/RampNet/issues/48) residual class.

| | all other views missed (153) | the other 1,145 |
|---|---|---|
| nearest other camera, median [IQR] | 7.3 m [4.8, 9.8] | 6.6 m [4.9, 8.3] |
| median other-camera distance, median | 12.2 m | 11.9 m |
| other captures within 18 m, median (mean) | 5 (5.0) | 5 (5.5) |
| distinct capture months, mean | 1.8 | 1.9 |
| nearest other camera under 6 m | 62 of 153 | |

By city: bend 22 of 289 (7.6%), gainesville 49 of 213 (23.0%), paterson 30 of 303 (9.9%),
richmond 25 of 240 (10.4%), sao_paulo 27 of 253 (10.7%). GSV 128, Mapillary 25.

**The real floor is the 55 the source view also missed.** The other 98 were found in their
source pano, so the deployment finds them. The 55 (bend 10, gainesville 13, paterson 12, richmond
10, sao_paulo 10) are 4.2% of these 1,298 ramps, a union recall of 0.958 on this population,
consistent with labeler#27's 0.945-1.000. Their nearest other camera is at a median 7.1 m
[IQR 4.7, 9.0] and their source camera at 14.1 m. By #48's residual class: 27 sub-threshold only,
15 association / placement (a 0.55 candidate claimed the ramp in some
capture within 25 m by the world or pixel test, but no other view within 18 m did by the world
test), 7
never fired with nothing stored below 0.55 (bend), 2 never fired at any stored floor (paterson),
4 counted as recalled by a fused site. 27 of the 55 have a stored other-view peak between 0.10 and
0.55.

**Reading.** The ramps no view finds are seen from close range by several captures. Distance does
not explain them. #48 §8's one-rater check found every residual GT click on a ramp (97 of 97), so
they are not wrong labels either, though the world GT point used here was not re-checked. What is
left is appearance or setting, threshold (27 have a sub-threshold peak), and GT world placement.

## 5. Density: re-thinning at coarser spacings

**What production does.** The labeler thins Mapillary and Panoramax to one pano per grid cell,
5 m by default, newest capture wins (`sources/mapillary.thin_panos`). Downtown Richmond went from
35k to 9k panos. GSV is not thinned; its spacing is whatever Google captured.

**Native spacing in these runs** (median distance from a pano to its nearest neighbour, among
panos within 25 m of a pool ramp): richmond 3.5 m, the four GSV cities 9.9-10.0 m. The GSV runs
include several capture dates per location (mean 1.9 distinct months among a ramp's other views).

**Method.** For each spacing, the labeler's rule is re-applied on a grid with a random offset:
one pano per cell, newest capture month wins, ties broken at random (the labeler breaks them on a
quality score these data do not carry). The result is averaged over 40 offsets (20 for the 0.30
arm). The cross-check `random` keeps the same number of panos chosen uniformly, which has no grid
edge effects. The population is fixed: every ramp with at least one other capture within 18 m at
native density (1,317). Recall is "found by a kept other view within 18 m". Deltas are paired
against native on the same ramps, with a ramp-cluster bootstrap CI (offset-to-offset variation is
averaged out, not included).

0.55, other views, grid rule (`results.json` → `thinning_055`, which also has the union and
random rows):

| spacing | GSV panos kept | GSV recall | GSV delta [95% CI] | richmond panos kept | richmond recall | richmond delta [95% CI] |
|---|---|---|---|---|---|---|
| native | 1.000 | 0.875 | | 1.000 | 0.875 | |
| 5 m | 0.996 | 0.874 | -0.000 [-0.001, -0.000] | 0.798 | 0.861 | -0.014 [-0.024, -0.006] |
| 7.5 m | 0.986 | 0.871 | -0.003 [-0.006, -0.001] | 0.639 | 0.836 | -0.039 [-0.057, -0.023] |
| 10 m | 0.940 | 0.860 | -0.015 [-0.019, -0.011] | 0.516 | 0.813 | -0.062 [-0.082, -0.043] |
| 15 m | 0.711 | 0.787 | -0.088 [-0.097, -0.079] | 0.366 | 0.749 | -0.126 [-0.152, -0.099] |
| 20 m | 0.562 | 0.691 | -0.184 [-0.197, -0.172] | 0.284 | 0.673 | -0.201 [-0.236, -0.169] |
| 30 m | 0.401 | 0.525 | -0.350 [-0.366, -0.335] | 0.198 | 0.545 | -0.329 [-0.365, -0.294] |

With the source view added back (union), richmond goes 0.960 at native, 0.937 at 5 m, 0.880 at
10 m; GSV 0.956, 0.956, 0.946. The 0.30 arm (four cities) has the same shape: richmond 0.903 at
native, 0.891 at 5 m, 0.852 at 10 m.

**Reading.**
- Re-thinning Richmond on a fresh 5 m grid drops 20% of its panos (the production grid and the
  re-applied one do not line up) and costs 1.4 points of other-view recall [0.6, 2.4] and 2.3 points
  of union recall [1.1, 3.7]. Halving the panos (10 m) costs 6.2 and 8.0 points.
- For GSV, nothing below 10 m changes the pano set, so thinning is not a lever there, and denser
  GSV is not a setting at all.
- At equal pano counts the grid rule beats uniform random selection on Richmond (10 m: 0.813 vs
  0.786), because it spreads the kept panos out. On GSV at 20-30 m random does better (0.599 vs
  0.525 at 30 m); the grid keeps the newest of co-located captures, which this data cannot
  separate further.
- **Denser than native cannot be measured here.** Richmond's run was already thinned at 5 m, so
  the panos between grid cells were never downloaded or scored. The curve's slope near native
  (about 1.4 points for the last 20% of panos) and the diminishing returns in #48 §4 suggest a
  small gain, and the correlation in §3 says why: the extra views are of ramps that nearby views
  already missed. That is an extrapolation, not a measurement.
- **What would measure it:** the labeler's `scripts/thinning_experiment.py` protocol, which has a
  design and no run: detect a Richmond sub-area un-thinned (`--thin-spacing 0`, about one pano
  per 1.5 m of street), then score per-ramp recall at 0, 2.5 and 5 m against the richmond GT
  pool. That needs Mapillary downloads and GPU time on the order of 1.5 s per pano.

**Caveats.** The capture table holds only panos within 25 m of some pool ramp. In production, a
grid cell can also hold a newer pano that is more than 25 m from every pool ramp and so is not in
the table; that pano would win the cell and the qualifying one would be dropped. The simulation
cannot see this, so at coarse spacings it is optimistic. The bias needs a cell diagonal of at
least 7 m to matter (a qualifying pano within 18 m and a cell-mate beyond 25 m), so it does not
affect the 5 m rows and grows with spacing. The `random` rows have no such effect and agree with
the grid rows to within 3 points up to 10 m.

## 6. Nearest view vs any view

Population: the 1,307 ramps with at least three other captures within 25 m, so every rule has a
view to use. 0.55, world test, ramp-cluster bootstrap CI.

| rule | ramps with a qualifying view | recall [95% CI] |
|---|---|---|
| nearest other view only | 1,307 | 0.654 [0.628, 0.679] |
| any other view within 6 m | 533 | 0.276 [0.252, 0.302] |
| any other view within 12 m | 1,250 | 0.774 [0.752, 0.797] |
| any other view within 18 m | 1,304 | 0.878 [0.859, 0.895] |
| any other view within 25 m | 1,307 | 0.904 [0.887, 0.920] |

The 6 m row is mostly coverage: only 533 of the ramps have another camera that close.

**Reading.** Far views add recall: the 18-25 m views add 2.6 points over the 18 m cap, and any
view beats the nearest view by 25 points. A rule that trusts only the nearest view, or drops far
views, gives that up. #38's point that "a ramp seen once at 8 m is worth more than three sightings
at 30 m" is about precision. #48 §6 tested a distance-binned evidence score for precision and found
no gain over k-of-n at the 0.30 tier. Wegner et al. 2016 (trees, #48 §2 row 21) found the single
closest street view beat pooling for detection; for curb-ramp recall these data say the opposite.

## 7. What was not done, and why

- **No new detection, no GPU, no paid calls.** Every number is from committed tables.
- **Far-field ramps.** The pool excludes 257 GT ramps whose source-view click does not raycast
  within 25 m. Those are the ramps per-pano recall misses most, so per-ramp recall on the full GT
  is unmeasured. Raycasting them needs a better lift than flat ground (depth or the #48 cross-view
  work) and the labeler's runs.
- **Denser-than-native sampling** (§5): no un-thinned run exists.
- **A second rater** for any GT used here, and an independent audit in a GSV city. The Laurens
  read is the only per-ramp number not anchored to RampNet's review.
- **Fetch and inference cost** of density: the thinning sweep reports panos kept, which is the
  cost driver (the labeler measures about 1.5 s per pano on one GPU); no cost was measured here.
- **Precision.** Project Sidewalk labels and the benchmark reviewer label each ramp once, so none
  of this gives a per-ramp false-positive rate. #48 §6 and labeler#27 have world precision at the
  site level.

## 8. Reproduction

From a clean clone, CPU only, about 35 seconds:

```bash
python scripts/analysis/per_ramp_recall_38.py run     # writes analysis_out/per_ramp_recall_38/
python scripts/analysis/per_ramp_recall_38.py check   # re-derives both files and compares bytes
pytest -q tests/test_per_ramp_recall_38.py
```

The seed (38), bootstrap and permutation counts and spacings are constants at the top of the
script and recorded in `results.json` → `params`. The input table itself comes from
`scripts/analysis/multiview_evidence_48.py run`, which needs the labeler's unpublished run files
([`multiview_48.md`](multiview_48.md) §10 says what would unblock that).

## 9. Cost and time

| step | where | wall-clock | GPU-h | $ |
|---|---|---|---|---|
| `run` and `check` | desktop CPU | about 35 s each | 0 | 0 |

No cluster job, no model leg, so nothing goes in `compute_log.jsonl` or `usage_log.jsonl`.
