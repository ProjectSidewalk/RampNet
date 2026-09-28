# Multi-view evidence per physical ramp (#48, Phase 1)

[#48](https://github.com/ProjectSidewalk/RampNet/issues/48) proposed treating repeated
captures of one curb ramp as independent evidence: recall by disjunction across views,
precision by k-of-n agreement, and a per-ramp evaluation unit. Most of its Tier 1 was built
and measured in
[sidewalk-auto-labeler#27](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/27).
This document records what that work answered, what the literature says, and four
measurements it left open, made with the labeler's own code on the same five cities. A fifth
measurement, the free challengers in world space on Richmond, is in §7.

Scope, as Jon set it: the challenger experiment runs on Richmond only, with free models only
and no paid API legs. Tier 2 (feed-forward 3D) and Tier 3 (semantic 3D) are not in this
phase; §9 gives a read on them, marked as proposed.

Code: `scripts/analysis/multiview_evidence_48.py` (B.1–B.4) and
`scripts/analysis/multiview_challengers_48.py` + `multiview_challengers_48.sh` (C). Outputs:
`analysis_out/multiview_48/`. Figures: `docs/figures/multiview_48/`.

## 1. What labeler#27 already answered

All numbers below are from the labeler#27 thread and the labeler's
`runs/<city>/fusion_eval/report.md` files, re-read for this document.

- **Recall by disjunction is already realized in production.** The labeler processes every
  pano and submits every operational detection, so a city already receives the union of
  views. Union coverage per ramp with no fusion: paterson 1.000, gainesville 0.945,
  sao_paulo 0.980, richmond 0.972, bend 0.983. Displaced-GT decoys at 30 m hit 0.02–0.11, so
  the union coverage is real. Fused world recall is 0.927–0.957, so fusion *costs* 1.8–4.7
  points of coverage against the raw union (labeler#27's text says 1.5–4.7; its own two tables
  give 1.8 for gainesville, 0.945 − 0.927).
- **What fusion buys:** deduplication (2.33–6.07 operational labels per ramp become one site;
  richmond 6.07) and p90 placement 20–30% tighter than the best single view (the median is a
  wash). World precision is 0.893–0.975.
- **Stage 4, promoting sub-0.55 sites on multi-view support,** was measured and not built:
  k ≥ 2 at confidence ≥ 0.25 adds 4 ramps on gainesville and 3 on sao_paulo, inside the CIs;
  paterson is saturated.
- **Ghost check:** other-pano support at ≥ 0.25 for 0.87–0.97 of verdict-true detections and
  0.64–0.83 of verdict-false ones.
- **Placement:** GT-anchored p50 1.89 m / p90 4.41 m (labeler `docs/reprojection-residual.md`,
  n = 866 references at 2.6 m). Far views range about 11% long
  ([#101](https://github.com/ProjectSidewalk/RampNet/issues/101)).
- **Dual ramps:** same-pano GT pairs under 5 m apart are kept separate 77–83% of the time.

So the "big win" #48 expected from disjunction is already in the production numbers. Two
RampNet-side numbers bound what is left: the pooled per-pano recall ceiling is 0.849 at the
0.10 storage floor (`docs/operating_point.md`, "The storage floor and the recall ceiling"),
and 57.8% of misses are far-field, ≥ 18 m (`scripts/analysis/miss_decomposition.py`).

## 2. Literature

V = read in full or the relevant section, S = snippet or abstract only, as recorded in the
2026-09-26 sweep.

| # | work | what it contributes | read |
|---|---|---|---|
| 1 | Krylov, Kenny & Dahyot 2018, *Remote Sensing* 10(5):661, arXiv 1708.08417, code `vlkryl/streetview_objectmapping` | MRF over pairwise ray intersections with monocular depth as the unary; ≥ 2 views; traffic lights R 0.94 / P 0.922, poles R 0.926 / P 0.973 at 2 m; camera-position error is the main error source. **The baseline.** | V |
| 2 | Krylov & Dahyot 2018, ECML-PKDD Urban Reasoning workshop (Springer LNCS 2019) | the same on Mapillary; geolocation worse "due to excessive camera position noise" | S |
| 3 | Liu, Ulicny, Manzke & Dahyot 2021, IMVIP, arXiv 2108.06302 | SfM pose refinement plus OSM context snapping | S |
| 4 | Ulicny et al. 2023, arXiv 2305.08232 | the MRF extended to object height, < 20 cm against LiDAR | V |
| 5 | Ahmad & Krylov, EUSIPCO 2024 | noisy council asset records as weak supervision: 82 inventoried bins, mean offset 5.45 m, 43 real bins missing from the record, 46 FP. Inventory incompleteness has to be handled explicitly | V |
| 6 | Murphy, Viola & Krylov, IMVIP 2025, arXiv 2509.10310 | birth-and-death point process over map-space energy maps with GIS priors; simulation only | V (abstract) |
| 7 | Hebbalaguppe et al., WACV 2017 | triangulation from consecutive GSV pairs dedups and suppresses FPs | S |
| 8 | Lumnitz et al., *ISPRS J.* 175:144, 2021 | trees: Mask R-CNN + depth + triangulation on GSV and Mapillary; 70% of inventory, 4–6 m error | S |
| 9 | Wilson et al. 2021, arXiv 2107.06257 | learned similarity + Hungarian association across low-frame-rate video; ARTS dataset | V (abstract) |
| 10 | Chaabane et al., WACV 2021, arXiv 2004.05232 | end-to-end pose + association + tracking | V (abstract) |
| 11 | Nassar, Lefèvre & Wegner, ICCV 2019, arXiv 1907.10892; GeoGraph, ECCV 2020, arXiv 2003.10151 | GNN multi-view detection with approximate poses (Mapillary signs, Pasadena trees) | V (abstracts) |
| 12 | Liu, Fu, Jia, Dong & Yang, "SVII-3D", arXiv 2601.10535 (Jan 2026) | geometry-only attention association (positions, bearing, box size) + ray triangulation + split/merge repair; explicitly rejects dense reconstruction for sparse wide-baseline input; F1 0.848 / 0.839 at 1 m, on survey-grade poses. The cleanest 2026 template for association | V |
| 13 | Toso et al., "Maps from Motion", arXiv 2411.12620 | object-layout registration; COLMAP fails on 80% of sparse, viewpoint-varying sequences | V (abstract) |
| 14 | VGGT, CVPR 2025, arXiv 2503.11651 | feed-forward pose + depth + point maps | V |
| 15a | MapAnything (Keetha et al., Meta/CMU), **arXiv 2509.13414** | factored output with one global metric scale; can be conditioned on given poses and intrinsics | V |
| 15b | Carnot et al., "MapAnything: Evaluating Monocular Metric Depth Models for 3D Urban Asset Localization", arXiv 2509.14839 | sign error 3.0–3.7 m under 10 m range, 5.7 m beyond 20 m; ≤ 10 m match, 3 m dedup. **#48's text cites this ID for the Meta model above; they are different papers** | V |
| 16 | π³, ICLR 2026, arXiv 2507.13347, `yyfz/Pi3` | feed-forward reconstruction with no reference view | V |
| 17 | Depth Anything 3, arXiv 2511.10647 | monocular / multi-view depth | V (abstract) |
| 18 | PanoVGGT, CVPR 2026, arXiv 2603.17571 | equirect input, but its outdoor data is synthetic (UE5/AirSim) with sufficient overlap; no GSV, Mapillary or multi-date evaluation | V |
| 19 | Li et al., "When Wider Views Fail", arXiv 2609.24839 (21 Sep 2026) | feed-forward reconstruction degrades and hallucinates geometry as angular span widens at a fixed view count | V (abstract) |
| 20 | UrbanVGGT, arXiv 2603.22531 | **single-image** GSV perspective crops (640×640, 90°), VGGT ground plane + 2.5 m camera-height scale; MAE 0.252 m, 95.5% within 0.5 m on ~300 DC images. Not evidence that VGGT can pose several captures | V |
| 21 | Wegner, Branson, Hall, Schindler & Perona, CVPR 2016 (trees) | CRF over world locations with a spacing prior, road distance, aerial and street-view scores; mAP street-only 0.581 → full 0.706; the **single closest** street view beat all views within a radius (occlusion and heading error). The principled log-linear form of k-of-n plus priors | V |
| 22 | Dabeer et al., IROS 2017, arXiv 1703.10193 | per-journey triangulation, cross-journey clustering and joint BA; < 20 cm from 25 journeys | S |
| 23 | Crowdsourced HD-map existence filters: (a) *Decision Support Systems* 2020, doi S0167923620301986; (b) *Sensors* 23(1):438, 2023, doi 10.3390/s23010438 | (a) time-aware Bayesian confidence per clustered sign; (b) recursive Bayesian existence filter per landmark with explicit miss and false-alarm rates, cluster size vs number of traversals. The principled replacement for fixed k-of-n: a miss in a pass where the object should have been visible is negative evidence, and the denominator is the passes with visibility | S |
| 24 | Mapillary map-features help page | ≥ 2 images (signs) / ≥ 3 (other point features); a production rule with no published calibration | S |

Gaps the sweep found: no paper plots P/R against the number of observations or calibrates k
(§4 below is that curve). No peer-reviewed multi-view curb-ramp work was found. Moreira et al.
2025, "ACAMAI", *IET Smart Cities*, doi 10.1049/smc2.70020 (YOLOv8 on GSV crops, ramp
R 0.91 / P 0.85) could not be fetched, so its geolocation and fusion method are unverified.
Evaluation practice: a match radius about 2× the expected position error, reported as a
sweep (2/3/5/8 m); one-to-one matching; duplicates inside the radius count as FP; nobody
applies a formal GT-completeness correction (best practice is auditing unmatched
predictions). Pose: Tsai & Chang, ISPRS Annals I-4, 2012, measured one GSV pair at
±0.52 / ±1.23 m with Google's pose error dominating (n = 1); no modern GSV pose audit exists,
and no independent measurement of Mapillary's computed vs raw pose exists. Turksever, SotM
2022: Mapillary street-light features about 2.0–2.2 m mean error against 323 municipal
records, completeness 72–88%.

## 3. Data and instrument

**Cities and runs.** richmond (Mapillary), paterson, gainesville, bend, sao_paulo (GSV). The
runs are the `results.jsonl` files the committed `fusion_eval/report.md` files scored: the
copies in the labeler's native-res archive on makelab2
(`/projects/makeabilitylab/sidewalk-auto-labeler/runs/<city>/results.jsonl`), pinned by sha256
in `ARCHIVED_RESULTS_SHA256`. The labeler checkout's own `runs/` have since been gap-filled for
paterson (+260 records), gainesville (+2,231) and sao_paulo (+7,293), so they no longer
reproduce the reports; richmond and bend are unchanged. The script refuses any other file.

**Instrument check.** Before writing anything, `run` requires eval_sites to reproduce each
report's headline exactly (world recall, precision, TP / FP and the four buckets). All five
pass, including the "self-detected with no site" diagnostic, on the archived runs (on the
gap-filled gainesville run that diagnostic reads 3 against the report's 2).

**Side note, gap-fill.** On the gap-filled runs, paterson and gainesville score exactly as
before, and sao_paulo's world recall rises from 0.933 to 0.953 (other-view recoveries 54 → 59)
with precision unchanged at 0.893. More captures of the same area did buy recall there.
Recorded in `meta.json` → `city_info.<city>.gap_filled_run`.

**Sub-threshold detections.** paterson, gainesville and sao_paulo store every peak down to
the 0.10 floor. bend stores nothing below 0.55. richmond's run stops at 0.55, but the labeler
re-inferred every richmond pano at the 0.10 floor into `results.f01.jsonl` (2026-09-22). Those
are fresh forward passes, so their confidences are new numbers; their ≥ 0.55 set matches
`results.jsonl` in (x, y) on 9,089 of 9,091 panos. The script keeps `results.jsonl`'s
operational detections exactly (the verdicts are keyed to them) and adds the re-inferred ones
below 0.55; the two non-matching panos get nothing added. With them, richmond's 15 unmatched
ramps split into 5 unmatched and 10 sub-threshold-only; nothing at 0.55 changes.

**World GT** is eval_sites' own: verdict-true operational detections and non-unsure missed
marks raycast at 2.6 m with the 25 m envelope, merged across panos within 2.5 m. The recall
pool is 1,327 ramps (richmond 253, paterson 304, gainesville 219, bend 296, sao_paulo 255).
**GT points that do not raycast inside 25 m are not in it** (richmond 57, paterson 91,
gainesville 53, bend 30, sao_paulo 26), so the far-field ramps that dominate per-pano misses
are outside every world-space number here.

**Qualifying capture and the two hit tests.** A capture qualifies for a ramp when its camera
is within R of the ramp's world position (R = 18 m unless stated; 12 and 25 m are in the
JSON). A capture *sees* the ramp at a floor when one of its stored detections claims the ramp:
- **world test (primary):** the detection's production raycast lands within 5 m (eval_sites'
  match radius);
- **pixel test:** the detection lies within the benchmark's 0.022 radius of the ramp projected
  into that pano (`geo.ground_point_to_pano`).

Claims are one-to-one within a capture, made in descending confidence like `score_pano`, so one
detection between two dual ramps counts for only one. Because claims go in confidence order,
one pass answers every floor. **Check:** the hit rate of the GT-source views reproduces
own-view recall: 0.838 / 0.766 / 0.785 / 0.822 / 0.733 against 0.830 / 0.763 / 0.785 / 0.818 /
0.722 (richmond, paterson, gainesville, bend, sao_paulo). The pixel test undercounts in other
views, because a GT position with a p90 error of 4.4 m projects well outside a 22-px radius at
short range; it is reported, not used for the headline. No projection fell at or above the
horizon or outside the envelope (every capture is within 25 m of its ramp).

## 4. B.1 Recall vs number of qualifying captures

Other views only (the GT-source view is excluded, because a verdict-true detection is GT by
construction). World test, R = 18 m. "k nearest": the share of ramps seen by at least one of
their k nearest other captures; a ramp with fewer than k uses all it has.

| k nearest other captures | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| pooled, ≥ 0.55 (1,317 ramps) | 0.652 | 0.781 | 0.833 | 0.854 | 0.865 | 0.868 | 0.870 |
| GSV, ≥ 0.55 (1,070) | 0.671 | 0.793 | 0.843 | 0.863 | 0.874 | 0.875 | 0.875 |
| Mapillary (richmond), ≥ 0.55 (247) | 0.571 | 0.725 | 0.789 | 0.814 | 0.826 | 0.838 | 0.850 |
| pooled, ≥ 0.30 (1,025; four cities) | 0.677 | 0.817 | 0.857 | 0.873 | 0.882 | 0.885 | 0.889 |
| pooled, ≥ 0.10 (1,025; four cities) | 0.754 | 0.864 | 0.896 | 0.910 | 0.914 | 0.918 | 0.922 |

The denominators are the pool ramps with at least one other capture within 18 m (10 of the
1,327 have none).

On a fixed population (the 1,036 ramps with ≥ 4 other captures, so every point is the same
ramps), ≥ 0.55: k = 1 0.645 [0.615, 0.673] → k = 4 0.865 [0.843, 0.884]. With ≥ 8 other
captures: richmond's 165 ramps go 0.588 (k = 1) → 0.855 (k = 4) → 0.903 (k = 8); at R = 25 m,
398 GSV ramps have ≥ 8 and go 0.601 → 0.842 → 0.902. Figure:
`docs/figures/multiview_48/recall_k_nearest.png` (≥ 4 solid, richmond ≥ 8 dotted).

**Reading.** Returns diminish: the first other capture recovers about two thirds of the ramps,
the second adds about 13 points, the third about 5, and captures five to eight add a few points
more (0.855 → 0.903 on richmond's ≥ 8 population). The all-ramps rows flatten sooner only
because most GSV ramps have no fifth or sixth capture to add (median 5 captures within 18 m,
source included; richmond 11). On populations with the same number of captures, GSV and
Mapillary follow the same shape (≥ 8 captures: 0.902 and 0.903 at k = 8), so the lower
Mapillary rows are about which ramps have how many captures, not about the imagery source.
R = 12 m caps the pooled curve at 0.807 and R = 25 m lifts it to 0.888 at k = 8. Since
production already takes the union, this curve says where the union's recall comes from, not
a new lever. The per-city union (source view or any other) is 0.93–0.99 for ramps with ≥ 3
other captures.

**Caveats beside the numbers.** GT is one rater per city. Range is flat-ground at 2.6 m and
rig-specific (labeler `docs/reprojection-residual.md`: the 2025–26 GSV rig and Mapillary rigs
sit lower, range scale k ≈ 1.18–1.35), so R is approximate. Mapillary pose error is
unmeasured. The world test can be satisfied by a detection of a real, non-GT ramp within 5 m;
the one-to-one claims only stop it double-counting GT ramps. With an 8 m world test
(`recall_vs_captures_hit8.json`) every number rises (pooled k = 8: 0.927), because GT positions
carry their own placement error (p90 4.4 m).

## 5. B.2 Failure correlation (#38's open checkbox)

For ramps with ≥ 2 other qualifying captures (1,298 at ≥ 0.55), are misses in two views of one
ramp independent? The prediction for a pair is the product of the two views' marginal miss
rates, each taken for its own city and range bin, so neither "both views were far" nor "both
views are in the city with the most captures" counts as correlation (richmond contributes 63%
of the pairs). Pooled marginal miss rate at ≥ 0.55 by range: 0–6 m 0.375, 6–12 m 0.370,
12–18 m 0.601.

| camera separation (pooled, ≥ 0.55) | pairs | P(miss j \| miss i) | P(miss j), city- and range-matched | joint miss, observed / independent |
|---|---|---|---|---|
| 0–3 m | 645 | 0.803 | 0.467 | 1.79 |
| 3–6 m | 1,930 | 0.686 | 0.468 | 1.42 |
| 6–12 m | 6,940 | 0.623 | 0.462 | 1.32 |
| 12–25 m | 11,118 | 0.592 | 0.487 | 1.27 |
| 25–51 m | 2,411 | 0.697 | 0.570 | 1.43 |

GSV alone the ratio is 1.09–1.33 by separation (6–12 m 1.29, 12–25 m 1.14); richmond alone
1.34–1.80. (The P(miss j) column is the mean marginal; the exact independence conditional
differs by < 0.01.)

**Every other view missed:** 153 ramps observed against 76.5 predicted under independence
(2.0×). GSV 128 vs 71.1 (1.8×); Mapillary 25 vs 5.4 (4.6×). The excess is concentrated in
ramps with many captures: with 5–8 other captures, 66 observed vs 15.0 predicted; with 2, 15 vs
19.0. At ≥ 0.30 the pooled figure is 98 vs 43.4 (2.3×). Figure:
`docs/figures/multiview_48/failure_correlation.png`.

**Not one pass.** Pairs of captures from different months are *more* correlated than pairs
from the same month (joint-miss ratio 1.45 vs 1.24 pooled; GSV 1.25 vs 1.19; richmond 1.58 vs
1.26). A transient cause in one drive (a parked car, glare) would show the opposite.

**GT position is part of it, not all of it.** A GT position error is shared by every view and
can make world-test misses look correlated. With an 8 m world test the all-missed count halves
(153 → 83) but the ratio does not move (2.0×; GSV 1.8×, Mapillary 4.9×).

**Reading.** Misses are correlated across views, moderately on GSV and strongly on Mapillary,
and more so across dates than within one. The misses that survive the union look like
properties of the ramp or its GT point (appearance, setting, or a GT position or verdict that
is off), not of one pass. That is why adding captures gives diminishing returns (§4), and it
is the quantitative reason #48's "independent evidence" premise overstates what more captures
can buy. Caveats as in §4; the 0–3 m bin is mostly richmond (627 of 645 pairs), where
consecutive Mapillary frames can be under a metre apart; wrong GT points (verdicts) are not
separable from appearance here — the one-rater gallery (§8) is where that would show.

## 6. B.3 Evidence accumulation vs k-of-n at the 0.30 tier

Cities with sub-threshold detections: gainesville, paterson, richmond (re-inference), sao_paulo.
Sites are fused at the 0.30 tier (`FuseParams(min_confidence=0.30)`, floor 0.10). Precision and
recall use eval_sites' structure, generalized so that any detector can be scored: a judged-pano
detection is TP / FP / ignored by the benchmark's per-pano matcher; a site is TP if any judged
member is a TP, FP if all decided members are FPs; a pool ramp is recalled if one of its GT
points is claimed by a detection whose site is submitted, or a submitted site is within 5 m
one-to-one. On richmond at 0.55 this rule gives exactly eval_sites' 0.941 / 0.959; on the other
cities it differs from eval_sites' verdict-based precision by 0.001–0.011 (gainesville
0.945 vs 0.956, paterson 0.971 vs 0.975), mostly because a `duplicate` verdict is a TP there and
an FP here.

Policies:
- `operational_055`: eval_sites' own sites at 0.55;
- `flat_030`: every site with a member ≥ 0.30;
- `kofn_k`: sites with a member ≥ 0.55, plus sites with ≥ k distinct panos at ≥ 0.30;
- **evidence score:** per site, the sum over every capture within 18 m of a per-capture
  log-likelihood ratio: a hit contributes by its confidence bin and range, **a miss in range
  contributes negatively** (naive Bayes, add-one smoothing; `fit_evidence_model`). It is
  calibrated on the labeled sites' non-judged captures (the judged views labeled the site, so
  they are left out of calibration), **in-sample** per city, and **leave-one-city-out**.

| pooled (4 cities, 1,031 pool ramps) | recall | precision | TP / FP sites |
|---|---|---|---|
| operational 0.55 | 0.941 | 0.944 | 801 / 48 |
| 0.55 sites within the 0.30 fuse | 0.947 | 0.898 | 845 / 96 |
| flat 0.30 | 0.966 | 0.865 | 864 / 135 |
| k-of-n, k = 2 | 0.958 | 0.885 | 855 / 111 |
| k-of-n, k = 3 | 0.952 | 0.890 | 848 / 105 |

At each k-of-n point's recall, the best precision the evidence score reaches at or above it
(FP sites in brackets; "keep 0.55" always submits sites with a member ≥ 0.55 and scores only the
rest):

| city, k | k-of-n | score, in-sample | score, LOCO | score keep-0.55, in-sample | score keep-0.55, LOCO |
|---|---|---|---|---|---|
| gainesville, 2 | 0.880 (25) | 0.849 (33) | 0.853 (32) | 0.883 (24) | 0.852 (32) |
| gainesville, 3 | 0.891 (22) | 0.866 (28) | 0.859 (30) | 0.892 (22) | 0.879 (25) |
| paterson, 2 | 0.948 (13) | 0.948 (13) | 0.941 (15) | 0.956 (11) | 0.956 (11) |
| richmond, 2 | 0.904 (24) | 0.901 (25) | 0.901 (25) | 0.915 (21) | 0.915 (21) |
| richmond, 3 | 0.908 (23) | 0.918 (20) | 0.918 (20) | 0.918 (20) | 0.915 (21) |
| sao_paulo, 2 | 0.809 (49) | 0.792 (55) | 0.798 (53) | 0.801 (52) | 0.801 (52) |
| sao_paulo, 3 | 0.811 (48) | 0.792 (55) | 0.807 (50) | 0.803 (51) | 0.803 (51) |

(paterson k = 3 is 0.952 (12) for k-of-n against 0.948 (13) / 0.941 (15) / 0.956 (11) /
0.956 (11).)

**No detectable gain.** Counting misses-in-range does not show a precision gain at the 0.30
tier over k-of-n. Out of sample (LOCO, keep-0.55) the score saves 1–3 FP sites on paterson and
richmond and costs 3–7 on gainesville and sao_paulo; in-sample (keep-0.55) it ranges from 3 FP
sites better (richmond) to 3 worse (sao_paulo). These are differences of a few sites against
15–59 false-class sites per city, and no confidence interval is computed for them; the honest
reading is "not distinguishable from k-of-n here", not "worse".

**What the 0.30 tier costs.** The two 0.55 rows differ because scoring at the 0.30 tier makes
the judged panos' 0.30–0.55 detections scorable, not because more sites are submitted (1,570
richmond sites in both). Against the like-for-like row (0.55 sites within the 0.30 fuse),
flat 0.30 adds 1.9 points of recall (+19 TP sites) for +39 FP sites, k = 2 adds 1.1 points
(+10 / +15), k = 3 adds 0.5 (+3 / +9). Figure: `docs/figures/multiview_48/evidence_vs_kofn.png`.

**Caveats.** Only 15–59 FP sites per city label the false class, so the calibration is thin;
the in-sample variant is exactly that, and the LOCO variant pools two GSV-heavy groups against
one Mapillary city. Calibration leaves out the captures of every judged pano (not only the
one that labeled the site), while scoring uses all captures; the threshold curve samples 40
quantiles of the scores, which can move a cell by one site. The GT gap below 0.55 hits hardest
the sites an evidence score promotes (multi-view-supported sub-0.55 sites that are real but
unlabelled count as FP), so this comparison is biased against promotion of any kind; a
re-score with `benchmark/<city>/incremental_fp_tags.json` where it covers those detections is
the proposed follow-up. **Below 0.55 precision is a lower bound**: the city GT was assembled from
RampNet detections at ≥ 0.55, so a sub-threshold detection is credited only if a reviewer
marked that ramp missed (`low_floor_sweep.py gtbias`; re-reviewing the sub-0.55 detections
found real, unlabelled ramps at 12.5–35% per split, 17% on richmond; `docs/model_comparison.md`). Vintage is not modelled; labeler#27 argued a miss in an
older view should count less. Richmond's sub-threshold detections come from the re-inference.

## 7. C. Challengers in world space (Richmond, free models)

**Question.** #48 claims cross-capture agreement "would kill much of the chat-VLMs'
false-positive flood" and so narrow the RampNet-vs-VLM gap. Measured here for the free legs.

**Bundle.** `benchmark/richmond_neighbourhood/`: every richmond run pano whose camera is within
30 m (the 25 m raycast envelope plus the 5 m match radius) of a pool ramp **or of a judged
pano's camera**, plus the judged panos: 2,867 panos, 124 judged. It was built in two passes: a
first 1,560-pano bundle (20 m of a pool ramp only) went to makelab2 first, and the PR review
showed it would flatter k-of-n precision (a false site away from the ramps loses the captures
that could support it). The widened bundle adds 1,307 panos; the first pass's detections stay
in the cache, so only the new panos run. `bundle.json` borrows
`benchmark/richmond/verdicts.json` (`compare.py` loads it with `verdicts_from`), so only the
judged panos are scored and `--detect-unjudged` runs the rest detect-only. Records are copied
from the labeler run (the judged ones verbatim from `benchmark/richmond/records.jsonl`) and
tagged with the ramps each pano is within 30 m of. Imagery is linked from the native-res
archive; the runbook refuses to run unless the 124 judged panos are byte-identical to
`benchmark/richmond/imagery_manifest.json` (they are).

**Precision scope.** No finite bundle holds every capture of every judged site, so world
precision is scored only on **covered sites**: sites with every full-run pano within 33 m
(25 m envelope + 8 m association cap) in the bundle (`covered_sites`). Recall is unaffected.

**Control.** RampNet from its own run detections (≥ 0.55 from `results.jsonl`, below from the
re-inference). eval_sites on the full run reproduces the report's 0.941 / 0.959. The
generalized scorer on the full run and on the bundle alone agree exactly at every tier
(0.30 / 0.40 / 0.55 / 0.70) and every k (1–3), recall and covered precision both
(`challengers/scores.json` → `legs.rampnet`). The recall agreement is the informative half:
covered sites are by definition those whose every full-run capture within 33 m is in the
bundle, so both scopes fuse them from the same detections and their precision agrees almost
by construction. The the per-pano row on the bundle reproduces
richmond's published 0.964 / 0.768 / 0.855. At 0.55, k = 1: world recall 0.941, covered
precision 0.971 (132 TP / 4 FP sites; the covered scope keeps 136 of the 220 judged sites).

**Results.** All seven legs scored all 2,867 bundle panos with none missing
(`challengers/scores.json`). The table uses each model's pre-set operating point from the
scoreboard. The per-pano column is the usual score on the 124 judged panos. The world columns
are eval_sites' recall over the 253 pool ramps and precision over covered sites (TP / FP site
counts in brackets). k is the minimum number of captures a fused site needs before it is
accepted.

| model | tier | per-pano P / R / F1 | world k=1 P (TP/FP) / R / F1 | world k=3 P (TP/FP) / R / F1 |
|---|---|---|---|---|
| RampNet | 0.55 | 0.964 / 0.768 / 0.855 | 0.971 (132/4) / 0.941 / 0.955 | 0.992 (117/1) / 0.842 / 0.911 |
| y11l_pano | 0.25 | 0.925 / 0.439 / 0.595 | 0.944 (84/5) / 0.842 / 0.890 | 0.971 (67/2) / 0.668 / 0.791 |
| y11x_pano_h200 | 0.25 | 0.952 / 0.384 / 0.547 | 0.973 (73/2) / 0.794 / 0.875 | 0.984 (62/1) / 0.625 / 0.764 |
| y26_pano | 0.25 | 0.680 / 0.384 / 0.491 | 0.724 (76/29) / 0.846 / 0.780 | 0.743 (52/18) / 0.648 / 0.692 |
| Molmo2-8B | none | 0.410 / 0.516 / 0.457 | 0.528 (94/84) / 0.901 / 0.666 | 0.593 (83/57) / 0.814 / 0.686 |
| Qwen3-VL-8B | none | 0.319 / 0.445 / 0.371 | 0.423 (74/101) / 0.830 / 0.560 | 0.489 (66/69) / 0.763 / 0.596 |
| OWLv2 | 0.05 | 0.033 / 0.971 / 0.064 | 0.044 (131/2869) / 0.992 / 0.084 | 0.046 (123/2562) / 0.961 / 0.087 |
| Grounding DINO | 0.05 | 0.028 / 0.848 / 0.053 | 0.035 (115/3128) / 0.992 / 0.069 | 0.037 (104/2709) / 0.961 / 0.071 |

The full sweep (every tier, k = 1 to 3, with Wilson CIs) is in `scores.json`.

1. **RampNet first and the open-vocab detectors last are robust; the middle order is not
   resolved.** At the scoreboard operating points the point estimates keep one order per pano,
   in world space at k = 1 and at k = 3. The ends hold across the whole sweep: RampNet's world
   F1 at every swept tier is above every challenger row at the same k (its lowest, 0.949 at
   0.70, k = 1, against the best challenger row, y11x at 0.10, 0.913), and the open-vocab
   detectors' best world F1 (OWLv2 0.307 at 0.30, k = 1) is below every other leg's headline
   row. The neighbours in between are not separated. y11l vs y11x flips with the operating
   point: at 0.10 y11x beats y11l per pano (F1 0.777 vs 0.737) and in world space at k = 1
   (0.913 vs 0.850). y26 and Molmo at k = 3 are 0.692 vs 0.686. So #48's conjecture that "best
   single-image model" and "best model in a multi-view system" could differ is not borne out
   for the top model on Richmond; among the challengers these data cannot say.
2. **The gap narrows, and the narrowing is recall by disjunction, not precision by agreement.**
   Like for like, on the same 253-ramp pool: own-view recall (a pool ramp whose judged-pano GT
   point is matched by that pano's own detection, with the detection's fused site accepted;
   `self` in `scores.json`) is 211 / 253 = 0.834 for RampNet and 136 / 253 = 0.538 for y11l, a
   gap of 0.296. Fused world recall at k = 1 is 0.941 vs 0.842, a gap of 0.099: y11l recovers
   77 ramps from other views, RampNet 27. The F1 leads point the same way, 0.259 per pano
   (0.8546 − 0.5952) and 0.065 in world space at k = 1 (0.955 − 0.890), but they compare
   different denominators (310 GT instances on 124 judged panos vs 253 ramps within 25 m), so
   read them as secondary and do not compare a model's recall across those two columns.
   RampNet's 211 + 27 differs from the labeler report's 210 + 28 by one ramp because the two
   define "self" differently: eval_sites' `self_detected` counts a ramp whose GT point came
   from a reviewed RampNet detection, whether or not its site is accepted, and this scorer uses
   the definition above, which also applies to a challenger. The recalled totals agree (238).
3. **In world space a YOLO arm can match RampNet's recall or its precision, but not both.**
   Taking the best tier from the sweep, y11x at 0.10 matches RampNet's world recall (0.945, CI
   0.909–0.967, vs 0.941) at precision 0.884 (0.820–0.927) against 0.971 (0.927–0.989); the
   precision CIs just touch at 0.927. World F1 is 0.913 vs 0.955. At 0.25 it matches the
   precision (0.973) and falls to recall 0.794. The tier was chosen on the same data, which
   flatters every row, RampNet's sweep included (its best is 0.963 at 0.40).
4. **k-of-n agreement removes part of the chat VLMs' false positives, not most of them.** From
   k = 1 to k = 3, Molmo's false sites fall from 84 to 57 (−32%) while its true sites fall from
   94 to 83 (−12%); Qwen's fall from 101 to 69 (−32%) and from 74 to 66 (−11%). Precision rises
   (Molmo 0.528 to 0.593, Qwen 0.423 to 0.489; the Wilson CIs overlap) and so does F1 (0.666 to
   0.686, 0.560 to 0.596), at a cost of 0.09 and 0.07 recall, and 57 and 69 false sites survive.
   That is a partial precision-by-agreement effect; #48's "cross-capture agreement would kill
   much of the chat-VLMs' false-positive flood" overstates it on this data. For OWLv2 and
   Grounding DINO precision is flat in k at every tier (OWLv2 at 0.20: 0.146 at k = 1, 0.151 at
   k = 3): their false sites have as much multi-capture support as their true ones. This
   measurement does not say why. The detectors may fire on the same non-ramp objects from
   several captures (correlated errors, like the correlated misses in §5), or they may emit so
   many boxes per pano (OWLv2 at 0.05: 8,799 false detections on the 124 judged panos) that
   unrelated boxes from different captures land within the 8 m association gate by chance.
5. **k-of-n does not help RampNet's F1 either.** RampNet's F1 falls from k = 1 to k = 3 at every
   tier (0.955 to 0.911 at 0.55): the precision it buys (0.971 to 0.992) is smaller than the recall
   it costs (0.099). This matches §6.

**Reproduction of the published per-pano rows.** On the 124 judged panos the exported
detections reproduce each leg's published richmond per-pano score at its headline: exactly for
Molmo, OWLv2 and the YOLO trio; Qwen F1 0.371 vs 0.377; Grounding DINO 0.053 vs 0.053 (recall
0.848 vs 0.852). Detection by detection (`published_richmond.judged_panos_agreement` in
`scores.json`), the share of judged panos with the same detection count and every sorted pair
within 1e-4 in x and y (20× the 5-dp export rounding, about 0.4 px) is 1.00 for y11l, y11x and
Molmo, 0.99 for y26, 0.97 for OWLv2, 0.70 for Grounding DINO (same count on 0.90) and 0.03 for
Qwen (same count on 0.90). So only Qwen differs throughout, and Grounding DINO partly; at the
headline the differences move F1 by at most 0.006. An earlier version of this paragraph gave
much lower shares (0 for both open-vocab detectors, 0.52–0.71 for YOLO). That comparison
rounded both sides to 4 dp after the export had already rounded one side to 5 dp, so it
measured rounding, not re-run drift (review of PR 200).

**Where the published legs ran.** The YOLO trio's published richmond detections came from
makelab2's A40 on 2026-08-14 (commit 3d7c7bf, per its message). OWLv2, Grounding DINO, Molmo
and Qwen3-VL-8B ran on klone's `gpu-l40s` partition (one L40S; `docs/model_comparison.md`,
"Run on Hyak (L40S)") and were published in commit 2811f73. `analysis_out/compute_log.jsonl`
holds the matching klone jobs of 2026-07-23: `open_rich` (37604867), `molmo_rich` (37606420)
and the `qwen_curb_ramp_compare` jobs from 37596141 on. Neither the published files'
signatures nor the usage ledger records a host for those four legs, so the job-to-leg mapping
rests on the job names.

**Mixed hardware.** The re-run detections do not all come from one GPU. makelab2's A40
produced every YOLO detection and all four other legs on the first 1,560 panos, which include
all 124 judged panos; Molmo ran partly on the CPU there. The four non-YOLO legs on the 1,307
panos the 30 m widening added ran on klone's `ckpt-all` in nine tasks on L40S, A40 and A100
GPUs (commit 471e8b6, `scripts/analysis/multiview_48_klone/README.md`,
`docs/compute_cost.md`). The per-pano column is therefore single-host, and the world columns
mix hosts. The reproduction above compares makelab2 with the published legs, so for the YOLO
trio it is a same-host re-run, and for the four other legs it compares makelab2's A40 with
klone's L40S: the same kind of change the widening introduced, but not the same GPUs (the
widening also used A40 and A100). Across that change OWLv2 and Molmo match on 0.97 and 1.00 of
judged panos, Grounding DINO on 0.70 and Qwen on 0.03, and headline F1 moves by at most 0.006.
Every usage ledger row for this re-run records its host.

**Caveats that apply to every challenger row.** The fusion error model and gates were tuned on
RampNet's heatmap peaks; challenger points come from boxes (the chat VLMs, OWLv2, Grounding
DINO and YOLO emit a box centre, which sits above the ground contact point and raycasts long,
roughly 0.7–1.2 m at 10–18 m), which penalizes their world recall and association and not
RampNet's. The city GT is anchored to RampNet's review, so a challenger's real ramps that the
review did not mark count as FP, and every fused site containing one inherits that. Qwen3-VL-32B
is not run (it does not fit makelab2's A40 in bf16).

## 8. B.4 Residual misses

The pool ramps that no operational site recovered (eval_sites' `unmatched` and
`subthreshold_only`, 74 across the five cities) plus the 23 self-detected ramps whose nearest
operational site landed more than 5 m away. Classified mechanically (`residual_class`,
precedence in this order), using both hit tests at R ≤ 25 m:

| class | meaning | ramps |
|---|---|---|
| association / placement | a ≥ 0.55 candidate claimed the ramp in some capture, but no operational site is within 5 m | 35 (21 unmatched + 14 sub-threshold-only) |
| sub-threshold only | a stored candidate in [0.10, 0.55) claimed it, nothing ≥ 0.55 did | 28 |
| never fired | no candidate at any stored floor in any capture | 2 (paterson) |
| never fired, unknown below 0.55 | bend: nothing ≥ 0.55 anywhere; bend stores nothing lower | 8 |
| coverage gap, unknown below 0.55 | bend: no other capture within 18 m | 1 |
| self-detected, site displaced | counted as recalled by eval_sites; the fused site sits > 5 m away | 23 |

By city (`residual_misses.json`): richmond 6 / 9 / 0 / 11 (association / sub-threshold / other /
displaced), paterson 9 / 2 / 2 / 4, gainesville 7 / 9 / 0 / 2, sao_paulo 9 / 8 / 0 / 4,
bend 4 / 0 / 9 / 2.

**Reading.** 63 of the 74 residual ramps fired somewhere: at the operational tier in some view
(35) or just below it (28). Of the other 11, 2 (paterson) show no candidate at any stored
floor, and 9 are bend, where nothing below 0.55 is stored, so whether they fired lower is
unknown. The residual is mostly an association / placement problem and a threshold problem,
not a detection problem, which agrees with labeler#27's finding that fusion costs 1.8–4.7
points against the raw union. The
"association / placement" class is an upper bound: a claiming detection may be of an adjacent
non-GT ramp.

**One-rater gallery.** `benchmark/multiview_residual_48/gallery.html`: one card per ramp with
the GT-source view and up to four nearest other captures (440 crops, 36° × 24° equirect windows
cut on makelab2 from the native-res archive, ring at the GT point or its projection). The
rubric (occluded / flush-minimal-reveal (#151) / far / construction-changed / gt-error / other,
plus unclear) travels in `analysis_out/multiview_48/residual_taxonomy__jonf.json`, which is
committed with **empty verdicts**: this pass is Jon's. The page exports that file's format;
the item list and manifest digest (`d03e2546f74df4c3`) are fixed so a second rater can repeat
it.

## 9. What this says about Tier 2 and Tier 3 (proposed, not decided)

- **More captures give diminishing returns.** Recall from other views rises most with the first
  two or three captures and a few points more after that, and misses are correlated across
  views, more so across dates (§4–5). More captures, or reconstruction that uses more of them,
  cannot recover a ramp every view misses.
- **The addressable residual is association and placement** (35 residual ramps plus 23 displaced
  sites, §8). That is what Tier 2 feed-forward geometry would improve, if it improves anything.
  Proposed next step, cheapest first: test SVII-3D-style geometry-only association on these 58
  cases against the current chi-square associator, scored with the GT-anchored placement
  instrument (labeler `docs/reprojection-residual.md`), before any dense reconstruction.
- **Feed-forward 3D on our input is unproven.** Our captures are sparse, wide-baseline and
  multi-date; "When Wider Views Fail" and "Maps from Motion" both report the failure regime we
  would be in, PanoVGGT's outdoor evaluation is synthetic, and UrbanVGGT is single-image.
  Proposed: if Tier 2 is tried, condition MapAnything (2509.13414) on the given poses rather than
  estimating them, and measure on the 58 cases first.
- **Evidence scoring showed no detectable gain over k-of-n** at the 0.30 tier (§6), on thin
  false-class counts and a GT that under-credits promoted sites; proposed: do not build it
  unless the tag-corrected re-score changes that. Attaching a multi-view
  support count to each submitted label, as labeler#27 closed on, costs nothing and keeps the
  option open for a server-side re-threshold.
- **Tier 3 is not motivated by these data.** Nothing here points at a failure a 3D scene
  representation would fix that association would not.
- **Multi-view does not change who is first or last among the challengers (§7)**; the middle
  order is not resolved. Like for like on the 253-ramp pool, it narrows RampNet's recall lead
  over y11l from 0.296 (own view) to 0.099 (fused), through recall by disjunction. k-of-n
  agreement removes about a third of Molmo's and Qwen's false sites and does not raise the
  open-vocab detectors' precision. At its best swept tier (0.10) y11x matches RampNet's world
  recall, and the precision CIs just touch (0.927), so on this one city a YOLO arm in a
  multi-view system comes close to RampNet; the tier was chosen on the same data, so that is an
  upper bound.
- Proposed follow-up: Jon's one-rater pass on the gallery.

## 10. Reproduction

Inputs: the labeler checkout (read-only) at `b6bf5cf` (sidewalk-auto-labeler `main`; any
checkout whose `geo.py`, `depth.py`, `detectors/__init__.py`, `scripts/fuse_sites.py` and
`scripts/eval_sites.py` match the sha256s in `analysis_out/multiview_48/meta.json` → `labeler`),
its `runs/<city>/` (for `fusion_eval/report.md` and richmond's `results.f01.jsonl`), and copies of
the archived `results.jsonl` files from makelab2 (`--results-root`). The labeler's `runs/` and the
archive are not published; they are the inputs that keep `run` from being replicable from this
repo alone. **What would unblock it:** publishing, e.g. as a Hugging Face dataset config next to
`projectsidewalk/rampnet-benchmark`, the five archived `results.jsonl` (richmond 13 MB, paterson
43 MB, gainesville 42 MB, bend 75 MB, sao_paulo 41 MB; sha256s in `ARCHIVED_RESULTS_SHA256`),
their `fusion_eval/report.md`, and richmond's `results.f01.jsonl` (14 MB, sha256 in `meta.json`). **What is replicable from a clean clone:** every B.1 / B.2 table re-derives from the
committed `captures_R25.csv` (`tests/test_multiview_48.py` checks three keys), and the challenger
scores re-derive from the committed detections under
`analysis_out/multiview_48/challengers/detections/` plus the labeler inputs above.

```bash
# 0. snapshot the labeler code at main (read-only) and copy the archived runs
git -C ../sidewalk-auto-labeler archive -o labeler_main.tar b6bf5cf \
    geo.py depth.py detectors scripts/fuse_sites.py scripts/eval_sites.py
mkdir labeler_main && tar -xf labeler_main.tar -C labeler_main
for c in paterson gainesville sao_paulo; do mkdir -p runs_archive/$c; \
    scp makelab2:/projects/makeabilitylab/sidewalk-auto-labeler/runs/$c/results.jsonl runs_archive/$c/; done

# 1. B.1-B.4 (CPU, about 3-5 min); add --hit-radius 8 for the sensitivity arm
python scripts/analysis/multiview_evidence_48.py run --labeler-root labeler_main \
    --runs-root ../sidewalk-auto-labeler/runs --results-root runs_archive
python scripts/analysis/multiview_evidence_48.py figures
python scripts/analysis/multiview_evidence_48.py crop-plan

# 2. the gallery: cut on makelab2, render locally
python scripts/analysis/multiview_evidence_48.py cut-crops \
    analysis_out/multiview_48/residual_crop_plan.json \
    --archive-root /projects/makeabilitylab/sidewalk-auto-labeler/runs --out crops   # makelab2
python scripts/analysis/multiview_evidence_48.py gallery --crops crops

# 3. C: bundle (local), legs (makelab2), export (makelab2), score (local)
python scripts/analysis/multiview_challengers_48.py bundle --labeler-root labeler_main \
    --runs-root ../sidewalk-auto-labeler/runs
bash scripts/analysis/multiview_challengers_48.sh          # makelab2; --smoke first
python scripts/analysis/multiview_challengers_48.py score --labeler-root labeler_main \
    --runs-root ../sidewalk-auto-labeler/runs
```

## 11. Cost and time

| step | where | wall-clock | GPU-h | $ |
|---|---|---|---|---|
| B.1–B.4 `run` | desktop CPU | ~3 min | 0 | 0 |
| crop cutting (440 crops) | makelab2 CPU | 4 min 13 s | 0 | 0 |
| challenger smoke (2 + 2 panos × 7 legs) | makelab2 A40 | 15 min | 0.245 (sum of the 7 legs' elapsed, 882 s) | 0 |
| challenger legs, pass 1 (1,560 panos, 7 legs; 1,556 called, 4 cached by the smoke run) + pass 2 YOLO trio (1,307 panos) | makelab2 A40 (shared; Molmo partly on CPU, 46.4 s/pano) | 28 h 2 min, legs in sequence (2026-09-26 23:24Z to 09-28 03:27Z) | 28.0 (sum of the 10 legs' elapsed, 100,811 s; Molmo alone 20.1) | 0 |
| challenger legs, pass 2, Molmo / Qwen / OWLv2 / Grounding DINO (1,307 panos) | klone `ckpt-all`, 9 tasks, L40S / A40 / A100 | 80 min (2026-09-27 08:40 to 10:00 PT) | 9.26 (`compute_log.jsonl`) | 0 |
| challenger scoring (`score`) | desktop CPU | 3 min | 0 | 0 |

makelab2 has no `sacct`, so per `docs/compute_cost.md` its GPU time goes in
`analysis_out/usage_log.jsonl` as `paid: false` rows, one per leg, with wall-clock, s/pano and
hardware, not in `compute_log.jsonl`.
