# Per-ramp recall and how correlated the per-view misses are ([#38](https://github.com/ProjectSidewalk/RampNet/issues/38))

[#38](https://github.com/ProjectSidewalk/RampNet/issues/38) argued that the benchmark reports
per-pano recall (about 0.77 at the time) while the product is a city inventory, which needs
per-ramp recall over every pano that sees a ramp. It listed four items: measure per-ramp
deployment recall, quantify how correlated the misses across views are, sweep sampling density,
and consider distance-aware aggregation.

Most of this has since been measured elsewhere. This document first lists what is already
answered, where, and with which caveats (§1). It then adds what was missing (§3-§6), computed
from one committed table on CPU: confidence intervals and a permutation null for the miss
correlation, the same at matched view counts, a description of the ramps every view missed, a
thinning sweep, and nearest-view vs any-view recall with CIs.

Code: `scripts/analysis/per_ramp_recall_38.py`. Outputs: `analysis_out/per_ramp_recall_38/`
(`results.json`, `ramps_other_views.csv`). Test: `tests/test_per_ramp_recall_38.py`, which also
checks that every table row below is quoted verbatim from `results.json` (`doc-numbers`).

## Takeaways

- **Per-ramp deployment recall is measured, and it is far above per-pano recall.** On five cities
  the union of views finds 0.945-1.000 of the GT ramps within 25 m
  ([labeler#27](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/27)). On Laurens,
  against one auditor's own labels, the deployment has an AI cluster within 7.5 m of 0.913 of
  them ([labeler PR #119](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/119), not
  merged). Both rest on one rater per city.
- **Misses are correlated beyond the independence prediction, and the interval excludes 1.**
  Ramps missed by every other view within 18 m: 153 observed vs 76.5 predicted, ratio 2.00
  [95% CI 1.74, 2.25]; no permutation of 5,000 reached 153. **The ratio grows with the number of
  views per ramp**, so "about 2x" is specific to this population's view counts: matched on the
  two nearest other views it is 1.52 [1.43, 1.61], on three 2.03, on four 2.63.
- **GSV and Mapillary differ less than the raw ratios suggest.** Raw: GSV 1.80, Mapillary 4.62.
  Richmond ramps have about twice as many views. At two views per ramp: GSV 1.47 [1.37, 1.59],
  Mapillary 1.69 [1.48, 1.91], intervals overlapping; the gap opens from three views on.
- **#38's 0.3% was too low by 33-53x, depending on the match test.** The three nearest other views
  all miss 15.9% [13.9, 17.9] of ramps with the 5 m world test and 10.0% [8.4, 11.8] with an 8 m
  test; independence on the same instrument predicts 7.8% and 4.4%. Most of the gap from 0.3% is
  that other views miss far more often than the per-pano benchmark (§2), and part of that is GT
  placement error. The ratio to independence (2.03 and 2.29) is stable across the two tests; the
  level is not.
- **The deployment floor is 46 of 1,298 ramps (3.5%), missed by the source view and every other
  view within 25 m**; union recall 0.965 on this population. Counting only other views within
  18 m gives 55, of which 9 are found by a view at 18-25 m. Within the 25 m pool, the floor ramps
  are seen from about as close as the rest (nearest other camera median 7.8 m vs 6.6 m). 27 have a
  stored peak below 0.55. Far-field ramps are outside this pool by construction (§7).
- **Thinning costs recall about in proportion to the captures removed, and the data cannot say
  whether denser sampling would add any.** Removing 20% of Richmond's panos (a fresh 5 m grid)
  costs 1.3 points of other-view recall [0.5, 2.3]; keeping 52% (10 m) costs 6.3 points. GSV is
  already at about 10 m spacing and is not thinned in production, so no setting changes it below
  10 m. No un-thinned run exists to test denser than native.
- **A nearest-view-only rule loses a quarter of the recall.** The nearest other view alone finds
  0.654 [0.630, 0.680] of ramps; any other view within 25 m finds 0.904 [0.887, 0.919]. This is the
  recall side only, where adding views cannot hurt; the precision side is #48 §6, where a
  distance-weighted evidence score did not beat k-of-n.

## 1. What is already answered

| #38 item | answered by | number | caveats that travel with it |
|---|---|---|---|
| 1. Per-ramp deployment recall | [labeler#27](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/27), re-read in [`docs/multiview_48.md`](multiview_48.md) §1 ([PR #200](https://github.com/ProjectSidewalk/RampNet/pull/200), merged) | Union of views per ramp, no fusion: paterson 1.000, gainesville 0.945, sao_paulo 0.980, richmond 0.972, bend 0.983. After fusion into one site per ramp: 0.927-0.957. Decoy GT points displaced 30 m are "found" 0.02-0.11 of the time, so the union figure is not the match radius doing the work. | GT is one reviewer per city, built from RampNet detections reviewed at 0.55 plus the reviewer's missed marks, so it is anchored to RampNet. The pool holds only GT points that raycast within 25 m of their source camera; 257 of 1,584 ramps (richmond 57, paterson 91, gainesville 53, bend 30, sao_paulo 26) are outside it, and those are the far-field ramps that dominate per-pano misses. 5 m match radius on a flat-ground 2.6 m raycast (GT placement error p50 1.9 m / p90 4.4 m). Four GSV cities, one Mapillary city. |
| 1. Per-ramp deployment recall, independent GT | [labeler PR #119](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/119) (`docs/server-agree-check.md`, open), summarised on [#158](https://github.com/ProjectSidewalk/RampNet/issues/158) | Laurens: of the auditor's 230 CurbRamp labels, an AI cluster lies within 5 m of 0.830, within 7.5 m of 0.913 [0.870, 0.943], within 10 m of 0.974. For reference, of the 178 labels on a pano the AI labelled at all, 92 (0.517) have an AI label within RampNet's match radius in that same pano. | One rater (the auditor also cast nearly all the human votes); 44 of 169 streets completed at the pull, so provisional. Laurens only, Mapillary GoPro Max rig. The auditor labels each ramp once, so this is recall only; it gives no false-positive rate. The server's 7.5 m clustering is the unit, not RampNet's fusion. The 0.517 is on 178 labels, not the 230. There is no decoy control (labeler#27 has a 30 m displaced-GT check; this does not), so at dense corners a neighbouring ramp's cluster can earn the credit. |
| 2. Failure correlation | [`docs/multiview_48.md`](multiview_48.md) §5 | Ramps missed by every other view: 153 vs 76.5 predicted (2.0x); GSV 1.8x, Mapillary 4.6x; the excess is concentrated in ramps with many captures. Joint-miss ratio by camera separation 1.27-1.79. Pairs from different months are more correlated than pairs from one month (1.45 vs 1.24). An 8 m world test halves the count and leaves the ratio at 2.0x. | No confidence interval or significance test was given. Same GT and pool caveats as item 1. This document adds both, and view-count-matched ratios (§3). |
| 3. Sampling density | Partly: [`docs/multiview_48.md`](multiview_48.md) §4 (recall vs the k nearest other captures) | The first other view finds 0.652 of ramps, two 0.781, three 0.833, eight 0.870 (pooled, 0.55). | Says how recall grows with the number of captures used, not with pano spacing, and not at production's thinning. This document adds a thinning sweep (§5). |
| 4. Distance-aware aggregation | [`docs/multiview_48.md`](multiview_48.md) §4 (R = 12 / 18 / 25 m) and §6 (evidence score) | Capping qualifying captures at 12 m holds recall to 0.807, 25 m lifts it to 0.888 (k = 8). A per-capture log-likelihood score with confidence and range bins, counting misses in range against a site, showed no detectable precision gain over k-of-n at the 0.30 tier. | §6's false class is 15-59 sites per city, and its GT under-credits sub-0.55 sites. This document adds a nearest-only vs any-view comparison with CIs (§6). |

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
because a verdict-true ramp's GT world point was raycast from that view's own detection, so its
geometry differs from every other view's. It is not a hit by construction: the GT includes the
reviewer's missed marks, and its hit rate reproduces per-pano recall (0.722-0.830 by city, #48
§3). The "union" rows add it back in its own (city, range bin) stratum.

**Why other views miss so often.** At 0.55 the pooled marginal miss rate in other views is 0.375
at 0-6 m, 0.370 at 6-12 m and 0.601 at 12-18 m (#48 §5), against per-pano miss rates of about
0.16-0.28 in the source views. Other views are scored against a GT point carried over from another
pano by a flat-ground raycast, so GT placement error (p90 4.4 m against a 5 m radius) turns some
detections into misses, and a dual ramp claims only one detection. The 8 m arm is the check on
how much of the correlation that explains.

**The independence prediction.** For each ramp, the product of its views' marginal miss rates,
each taken from all other-view captures in the same city and range bin (0-6, 6-12, 12-18 m; 18-25 m
in the 25 m union row). The count of ramps every view missed is compared with the sum of those
products.

**Two uncertainty measures.**
- *Ramp-cluster bootstrap*, 2,000 resamples of ramps within each city (1,000 for the
  view-count-matched rows). The marginals are re-estimated from each resample, so the CI covers
  their sampling error too. In the PR review (one-off, not in the committed script), clustering by (city, source pano) instead (433 clusters)
  gave [1.73, 2.29] for the pooled ratio, so ramp-level clustering does not under-cover.
- *Stratified permutation null*, 5,000 shuffles (1,000 for single cities): within each (city,
  range bin) stratum the miss flags of all captures are shuffled across ramps. Every capture keeps
  its city and range bin and every stratum keeps its miss count; only which ramp a miss belongs to
  changes. The p-value is (1 + shuffles at or above the observed count) / (1 + shuffles), so
  0.0002 and 0.0010 are the smallest it can be.

Each output block draws from its own generator, seeded by the block's name, so adding a block does
not move another block's draws. Byte-identical re-runs are checked on the numpy version recorded in
`results.json` → `params`; numpy does not promise the same random streams across versions.

## 3. Miss correlation with intervals

Ramps with at least two other captures within 18 m, world test at 5 m unless stated.

| population | floor | ramps | all views missed | predicted, independent | ratio [95% CI] | permutation null mean / 95th pct | p |
|---|---|---|---|---|---|---|---|
| pooled, 5 cities | 0.55 | 1,298 | 153 (11.8%) | 76.5 (5.9%) | 2.00 [1.74, 2.25] | 76.2 / 88 | 0.0002 |
| GSV (4 cities) | 0.55 | 1,058 | 128 (12.1%) | 71.1 (6.7%) | 1.80 [1.55, 2.02] | 71.1 / 82 | 0.0002 |
| Mapillary (richmond) | 0.55 | 240 | 25 (10.4%) | 5.4 (2.3%) | 4.62 [3.01, 7.01] | 5.4 / 9 | 0.0002 |
| paterson | 0.55 | 303 | 30 (9.9%) | 16.1 (5.3%) | 1.86 [1.39, 2.37] | 16.0 / 22 | 0.0010 |
| gainesville | 0.55 | 213 | 49 (23.0%) | 33.7 (15.8%) | 1.45 [1.19, 1.72] | 33.7 / 40 | 0.0010 |
| sao_paulo | 0.55 | 253 | 27 (10.7%) | 13.5 (5.3%) | 2.00 [1.41, 2.57] | 13.4 / 19 | 0.0010 |
| bend | 0.55 | 289 | 22 (7.6%) | 7.8 (2.7%) | 2.83 [1.86, 3.95] | 7.7 / 12 | 0.0010 |
| pooled, 4 cities (no bend) | 0.30 | 1,009 | 98 (9.7%) | 43.4 (4.3%) | 2.26 [1.90, 2.65] | 43.1 / 52 | 0.0002 |
| pooled, 4 cities (no bend) | 0.10 | 1,009 | 67 (6.6%) | 29.1 (2.9%) | 2.30 [1.85, 2.76] | 29.1 / 37 | 0.0002 |
| pooled, 3 m range bins | 0.55 | 1,298 | 153 (11.8%) | 75.2 (5.8%) | 2.03 [1.79, 2.30] | 75.1 / 87 | 0.0002 |
| pooled, 8 m world test | 0.55 | 1,298 | 83 (6.4%) | 41.3 (3.2%) | 2.01 [1.63, 2.39] | 41.1 / 51 | 0.0002 |
| pooled, union with the source view, other views within 18 m | 0.55 | 1,298 | 55 (4.2%) | 16.2 (1.2%) | 3.40 [2.58, 4.25] | 16.2 / 23 | 0.0002 |
| pooled, union with the source view, other views within 25 m | 0.55 | 1,298 | 46 (3.5%) | 9.9 (0.8%) | 4.63 [3.42, 5.99] | 9.9 / 15 | 0.0002 |

The Mapillary row is all of richmond; there is no separate richmond row. Bend stores nothing below
0.55, so it is left out of the lower floors; richmond's sub-0.55 detections are the labeler's
re-inference (#48 §3). The pooled row reproduces #48 §5 exactly (153 and 76.52); the script
asserts this before writing.

**At matched view counts.** The all-missed ratio rises with the number of views a ramp has under
any per-ramp heterogeneity, and the populations differ: richmond ramps have a mean of 10.3 other
views within 18 m, the GSV cities 4.0-4.6 (on the ramps with at least two). Each cell below keeps only each ramp's k nearest other
views, on the ramps that have at least k (ratio [95% CI], observed vs predicted, ramps):

| population | k = 2 | k = 3 | k = 4 |
|---|---|---|---|
| pooled | 1.52 [1.43, 1.61] (277 vs 182.6, n 1,298) | 2.03 [1.82, 2.22] (193 vs 95.2, n 1,215) | 2.63 [2.28, 2.98] (140 vs 53.2, n 1,036) |
| gsv | 1.47 [1.37, 1.59] (215 vs 145.9, n 1,058) | 1.91 [1.69, 2.13] (150 vs 78.6, n 981) | 2.23 [1.91, 2.57] (104 vs 46.6, n 815) |
| mapillary | 1.69 [1.48, 1.91] (62 vs 36.7, n 240) | 2.58 [2.05, 3.21] (43 vs 16.7, n 234) | 5.42 [3.93, 7.36] (36 vs 6.6, n 221) |
| paterson | 1.44 [1.22, 1.70] (52 vs 36.0, n 303) | 2.02 [1.57, 2.49] (36 vs 17.8, n 286) | 2.26 [1.61, 2.95] (25 vs 11.1, n 243) |
| gainesville | 1.38 [1.25, 1.57] (75 vs 54.2, n 213) | 1.51 [1.29, 1.74] (53 vs 35.1, n 193) | 1.63 [1.33, 1.97] (39 vs 23.9, n 162) |
| sao_paulo | 1.37 [1.18, 1.58] (53 vs 38.6, n 253) | 1.80 [1.42, 2.17] (37 vs 20.6, n 245) | 2.41 [1.75, 3.14] (24 vs 10.0, n 218) |
| bend | 2.05 [1.66, 2.54] (35 vs 17.1, n 289) | 4.74 [3.44, 6.42] (24 vs 5.1, n 257) | 9.70 [6.20, 15.15] (16 vs 1.6, n 192) |

**#38's arithmetic, done per ramp.** #38 multiplied three per-view miss rates (0.158 x 0.158 x
0.121 = 0.3%) for the three nearest views. On the ramps with at least three other captures within
18 m (median distances of the three nearest: 6.5, 9.6 and 12.4 m):

| instrument | ramps | all three nearest other views missed [95% CI] | independent, same instrument | ratio [95% CI] | vs #38's 0.3% |
|---|---|---|---|---|---|
| 5 m world test, 0.55 | 1,215 | 15.9% [13.9, 17.9] | 7.8% | 2.03 [1.82, 2.23] | 53x |
| 8 m world test, 0.55 | 1,215 | 10.0% [8.4, 11.8] | 4.4% | 2.29 [1.97, 2.58] | 33x |
| 5 m world test, 0.30 (4 cities) | 958 | 13.5% [11.4, 15.8] | 5.8% | 2.32 [2.02, 2.61] | 45x |

**Reading.**
- The correlation is real at every floor, in every city, at every matched view count, with both
  range binnings and with both world tests: no interval in either table includes 1.
- How large it is depends on how many views are pooled. Two views of a ramp both miss about 1.5x
  as often as independence predicts; with four views every-view-missed is 2.63x pooled (GSV 2.23,
  Mapillary 5.42). The pooled
  "2.00" is a property of this population's mix of view counts, not a constant.
- In the PR review (a one-off computation, not in the committed script), splitting ramps by whether the source view found them (detected vs reviewer-added,
  other-view miss rates 0.454 vs 0.618) left the ratio at 1.95 and 1.70 within each group, so the
  correlation is not a mixture of those two populations.
- It is not only GT placement. An 8 m world test removes 70 of the 153 all-missed ramps, but the
  ratio stays at 2.0.
- GSV vs Mapillary: at two views the intervals overlap (1.47 [1.37, 1.59] vs 1.69 [1.48, 1.91]); the
  gap opens at three and four views. That fits consecutive Mapillary frames being near-duplicates
  (often under a metre apart, #48 §5), but it is one Mapillary city. The raw 1.80 vs 4.62 mostly
  measures view count.
- Gainesville has the highest all-missed share (23%) and the lowest ratios at every matched k
  (1.38-1.63); bend the highest (2.05-9.70). Their other-view marginal miss rates are the highest
  and lowest of the five cities (0.63 and 0.35, vs 0.49-0.50 elsewhere), so the same
  absolute excess is a larger ratio there; the per-city ratios are not directly comparable without
  matching the marginal miss rate too.
- Most of the distance between #38's 0.3% and the observed three-nearest share is the marginal miss
  rate, not the correlation: under independence this instrument already predicts 7.8% (5 m) or
  4.4% (8 m), because other views miss far more often than per-pano recall implies (§2).
  Correlation roughly doubles that.

## 4. The ramps every view missed

`ramps_other_views.csv` has one row per ramp with at least two other captures within 18 m: city,
imagery, number of other captures, nearest and median other-camera distance, capture months, the
source view's distance and hit, the best stored other-view confidence, whether every other view
within 18 m missed, the same out to 25 m (`all_views_missed_25m`, source view included), and its
[#48](https://github.com/ProjectSidewalk/RampNet/issues/48) residual class.

**All other views within 18 m missed (153).** Their nearest other camera is at a median 7.3 m
[IQR 4.8, 9.8] (6.6 m for the other 1,145); median 5 other captures in both groups; 62 have an other
camera under 6 m. By city: bend 22 of 289 (7.6%), gainesville 49 of 213 (23.0%), paterson 30 of 303
(9.9%), richmond 25 of 240 (10.4%), sao_paulo 27 of 253 (10.7%). 98 of the 153 were found in their
source pano, so the deployment has them.

**The floor: missed by the source view and every other view in the table, out to 25 m.**

| | |
|---|---|
| missed by the source view and every other view within 25 m | 46 of 1,298 (3.5%) |
| union recall on this population | 0.965 |
| missed by the source view and every other view within 18 m | 55, of which 9 are found by a view at 18-25 m |
| nearest other camera, median [IQR], floor vs rest | 7.8 m [5.1, 9.4] vs 6.6 m |
| source camera, median, floor vs rest | 13.1 m vs 11.9 m |
| other views within 25 m, median (mean), floor vs rest | 7 (7.3) vs 7 (8.8) |
| with a stored other-view peak in [0.10, 0.55) | 27 |
| by city | bend 9, gainesville 11, paterson 8, richmond 9, sao_paulo 9 |
| #48 residual class | association_placement 8, never_fired 2, never_fired_unknown_below_055 7, recalled_by_a_site (not a #48 residual) 2, sub_threshold_only 27 |

The 0.965 is consistent with labeler#27's 0.945-1.000, which also uses every view. An earlier
version of this document quoted the 18 m count (55, 4.2%) as "missed by every view"; the 25 m count
is the one that matches the union definition.

**Reading.** Within the 25 m pool, distance does not separate the floor from the rest: the floor
ramps' nearest other camera is a little farther (7.8 vs 6.6 m median) and they have a few fewer
views (mean 7.3 vs 8.8), but most are seen from under 10 m by several captures. With 46 ramps no
test of the difference is attempted. Far-field ramps cannot appear here, because the pool requires
a source raycast within 25 m (§7). #48 §8's one-rater check found every residual GT click on a ramp
(97 of 97), so these are not wrong labels either, though the world GT point used here was not
re-checked. What is left is appearance or setting, threshold (27 have a sub-threshold peak), and GT
world placement.

## 5. Density: re-thinning at coarser spacings

**What production does.** The labeler thins Mapillary and Panoramax to one pano per grid cell,
5 m by default, newest capture wins by full timestamp, quality score breaks ties
(`sources/mapillary.thin_panos`). Downtown Richmond went from 35k to 9k panos. GSV is not thinned;
its spacing is whatever Google captured.

**Native spacing in these runs** (median distance from a pano to its nearest neighbour, among
panos within 25 m of a pool ramp): richmond 3.5 m, the four GSV cities 9.9-10.0 m. The GSV runs
include several capture dates per location (mean 1.9 distinct months among a ramp's other views).

**Method.** For each spacing, the rule is re-applied, approximated at month resolution: one pano
per cell of a grid with a random offset, newest capture month wins, ties broken at random. The
committed table has capture months, not timestamps or quality scores, so within one Richmond drive
the simulation keeps a random frame where production keeps the latest. The ENU origin of
production's grid is not in the table either, so the 5 m row measures grid misalignment (a fresh
grid drops 20% of Richmond's panos), not a density change. The result is averaged over 40 offsets
(20 for the 0.30 arm). The cross-check `random` keeps the same number of panos chosen uniformly,
which has no grid edge effects. The population is fixed: every ramp with at least one other capture
within 18 m at native density (1,317). Recall is "found by a kept other view within 18 m". Deltas
are paired against native on the same ramps, with a ramp-cluster bootstrap CI (offset-to-offset
variation is averaged out, not included).

0.55, other views, grid rule (`results.json` → `thinning_055`, which also has the union and random
rows):

| spacing | GSV panos kept | GSV recall | GSV delta [95% CI] | richmond panos kept | richmond recall | richmond delta [95% CI] |
|---|---|---|---|---|---|---|
| native | 1.000 | 0.875 |  | 1.000 | 0.875 |  |
| 5 m | 0.996 | 0.875 | -0.000 [-0.001, -0.000] | 0.798 | 0.861 | -0.013 [-0.023, -0.005] |
| 7.5 m | 0.985 | 0.872 | -0.003 [-0.005, -0.002] | 0.638 | 0.836 | -0.038 [-0.055, -0.023] |
| 10 m | 0.939 | 0.860 | -0.015 [-0.019, -0.011] | 0.518 | 0.812 | -0.063 [-0.085, -0.043] |
| 15 m | 0.711 | 0.785 | -0.090 [-0.100, -0.081] | 0.367 | 0.741 | -0.134 [-0.163, -0.105] |
| 20 m | 0.561 | 0.690 | -0.185 [-0.198, -0.172] | 0.283 | 0.673 | -0.201 [-0.236, -0.170] |
| 30 m | 0.401 | 0.528 | -0.347 [-0.363, -0.331] | 0.198 | 0.542 | -0.333 [-0.369, -0.299] |

Native, 5 m and 10 m for the union and the 0.30 arm (grid rule):

| arm | native | 5 m | 10 m |
|---|---|---|---|
| richmond, 0.55, union | 0.960 | 0.937 | 0.882 |
| GSV, 0.55, union | 0.956 | 0.956 | 0.946 |
| richmond, 0.30, other views | 0.903 | 0.887 | 0.859 |

**Reading.**
- Removing 20% of Richmond's panos (a fresh 5 m grid) costs 1.3 points of other-view recall
  [0.5, 2.3] and 2.2 points of union recall. Halving the panos (10 m) costs 6.3 and 7.8 points.
- For GSV, nothing below 10 m changes the pano set, so thinning is not a lever there, and denser
  GSV is not a setting at all.
- At equal pano counts the grid rule beats uniform random selection on Richmond (10 m: 0.812 vs
  0.787), because it spreads the kept panos out. On GSV at 20-30 m random does better (0.596 vs
  0.528 at 30 m); the grid keeps the newest of co-located captures, which these data cannot
  separate further.
- **Denser than native cannot be measured here.** Richmond's run was already thinned at 5 m, so
  the panos between grid cells were never downloaded or scored. The curve's slope near native
  (about 1.3 points for the last 20% of panos) and the diminishing returns in #48 §4 suggest a
  small gain, and the correlation in §3 says why: the extra views are of ramps that nearby views
  already missed. That is an extrapolation, not a measurement.
- **What would measure it:** the labeler's `scripts/thinning_experiment.py` protocol, which has a
  design and no run: detect a Richmond sub-area un-thinned (`--thin-spacing 0`, about one pano
  per 1.5 m of street), then score per-ramp recall at 0, 2.5 and 5 m against the richmond GT
  pool. That needs Mapillary downloads and GPU time on the order of 1.5 s per pano.

**Caveats.** The capture table holds only panos within 25 m of some pool ramp. In production, a
grid cell can also hold a newer pano that is more than 25 m from every pool ramp and so is not in
the table; that pano would win the cell and the qualifying one would be dropped. The simulation
cannot see this, so at coarse spacings it is optimistic. It needs a qualifying pano within 18 m and
a cell-mate beyond 25 m, so a cell diagonal of at least 7 m: it barely affects the 5 m rows (7.07 m
diagonal) and grows with spacing. The `random` rows have no such effect and agree with the grid
rows to within 3 points up to 10 m.

## 6. Nearest view vs any view

Population: the 1,307 ramps with at least three other captures within 25 m, so every rule has a
view to use. 0.55, world test, ramp-cluster bootstrap CI.

| rule | ramps with a qualifying view | recall [95% CI] |
|---|---|---|
| nearest other view only | 1,307 | 0.654 [0.630, 0.680] |
| any other view within 6 m | 533 | 0.276 [0.252, 0.301] |
| any other view within 12 m | 1,250 | 0.774 [0.751, 0.796] |
| any other view within 18 m | 1,304 | 0.878 [0.860, 0.894] |
| any other view within 25 m | 1,307 | 0.904 [0.887, 0.919] |

The 6 m row is mostly coverage: only 533 of the ramps have another camera that close.

**Reading.** This measures only the recall side, where an "any view" rule cannot lose recall as
views are added. The size is the point: the 18-25 m views add 2.6 points over the 18 m cap, and
any view beats the nearest view by 25 points, so a rule that trusts only the nearest view, or
drops far views, gives that up. #38's point that "a ramp seen once at 8 m is worth more than three
sightings at 30 m" is about precision. #48 §6 tested a distance-binned evidence score for precision
and found no gain over k-of-n at the 0.30 tier.

## 7. What was not done, and why

- **No new detection, no GPU, no paid calls.** Every number is from committed tables.
- **Far-field ramps.** The pool excludes 257 GT ramps whose source-view click does not raycast
  within 25 m. Those are the ramps per-pano recall misses most, so per-ramp recall on the full GT
  is unmeasured, and nothing here says whether far-field ramps are in the floor. Raycasting them
  needs a better lift than flat ground (depth or the #48 cross-view work) and the labeler's runs.
- **Denser-than-native sampling** (§5): no un-thinned run exists.
- **A second rater** for any GT used here, and an independent audit in a GSV city. The Laurens
  read is the only per-ramp number not anchored to RampNet's review.
- **Fetch and inference cost** of density: the thinning sweep reports panos kept, which is the
  cost driver (the labeler measures about 1.5 s per pano on one GPU); no cost was measured here.
- **Precision.** Project Sidewalk labels and the benchmark reviewer label each ramp once, so none
  of this gives a per-ramp false-positive rate. #48 §6 and labeler#27 have world precision at the
  site level.

## 8. Reproduction

From a clean clone, CPU only, about a minute per command:

```bash
python scripts/analysis/per_ramp_recall_38.py run           # writes analysis_out/per_ramp_recall_38/
python scripts/analysis/per_ramp_recall_38.py check         # re-derives both files and compares bytes
python scripts/analysis/per_ramp_recall_38.py doc-numbers   # the table rows this doc quotes
pytest -q tests/test_per_ramp_recall_38.py
```

The seed (38), bootstrap and permutation counts and spacings are constants at the top of the
script and recorded in `results.json` → `params`, with the numpy version. The input table itself
comes from `scripts/analysis/multiview_evidence_48.py run`, which needs the labeler's unpublished
run files ([`multiview_48.md`](multiview_48.md) §10 says what would unblock that).

## 9. Cost and time

| step | where | wall-clock | GPU-h | $ |
|---|---|---|---|---|
| `run` and `check` | desktop CPU | about 75 s each | 0 | 0 |

No cluster job, no model leg, so nothing goes in `compute_log.jsonl` or `usage_log.jsonl`.
