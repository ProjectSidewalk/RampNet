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
  the union coverage is real. Fused world recall is 0.927–0.957, so fusion *costs* 1.5–4.7
  points of coverage against the raw union.
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
| pooled, ≥ 0.55 (1,327 ramps) | 0.652 | 0.781 | 0.833 | 0.854 | 0.865 | 0.868 | 0.870 |
| GSV, ≥ 0.55 (1,074) | 0.671 | 0.793 | 0.843 | 0.863 | 0.874 | 0.875 | 0.875 |
| Mapillary (richmond), ≥ 0.55 (253) | 0.571 | 0.725 | 0.789 | 0.814 | 0.826 | 0.838 | 0.850 |
| pooled, ≥ 0.30 (1,031; four cities) | 0.677 | 0.817 | 0.857 | 0.873 | 0.882 | 0.885 | 0.889 |
| pooled, ≥ 0.10 (1,031; four cities) | 0.754 | 0.864 | 0.896 | 0.910 | 0.914 | 0.918 | 0.922 |

On a fixed population (the 1,036 ramps with ≥ 4 other captures, so every point is the same
ramps), ≥ 0.55: k = 1 0.645 [0.615, 0.673] → k = 4 0.865 [0.843, 0.884]. Richmond's 165 ramps
with ≥ 8 other captures reach 0.909 at k = 8. Figure: `docs/figures/multiview_48/recall_k_nearest.png`.

**Reading.** Most of the gain comes from the first two or three other captures; the curve is
flat by four or five. GSV reaches its plateau faster than Mapillary (GSV cities have a median
of 5 captures within 18 m, richmond 11), and Mapillary's single other view is weaker (0.571
vs 0.671). R = 12 m caps the pooled curve at 0.807 and R = 25 m lifts it to 0.888 at k = 8:
views beyond 18 m still add a little. Since production already takes the union, this curve
says where the union's recall comes from, not a new lever. The per-city union (source view or
any other) is 0.93–0.99 for ramps with ≥ 3 other captures.

**Caveats beside the numbers.** GT is one rater per city. Range is flat-ground at 2.6 m and
rig-specific (labeler `docs/reprojection-residual.md`: the 2025–26 GSV rig and Mapillary rigs
sit lower, range scale k ≈ 1.18–1.35), so R is approximate. Mapillary pose error is
unmeasured. The world test can be satisfied by a detection of a real, non-GT ramp within 5 m;
the one-to-one claims only stop it double-counting GT ramps.

## 5. B.2 Failure correlation (#38's open checkbox)

For ramps with ≥ 2 other qualifying captures (1,298 at ≥ 0.55), are misses in two views of one
ramp independent? The prediction for a pair is the product of the two views' marginal miss
rates at their ranges, so "both views were far" is not counted as correlation. Marginal miss
rate at ≥ 0.55 by range: 0–6 m 0.375, 6–12 m 0.370, 12–18 m 0.601.

| camera separation | pairs | P(miss j \| miss i) | P(miss j), range-matched | joint miss, observed / independent |
|---|---|---|---|---|
| 0–3 m | 645 | 0.803 | 0.466 | 1.79 |
| 3–6 m | 1,930 | 0.686 | 0.464 | 1.44 |
| 6–12 m | 6,940 | 0.623 | 0.458 | 1.37 |
| 12–25 m | 11,118 | 0.592 | 0.486 | 1.30 |
| 25–51 m | 2,411 | 0.697 | 0.571 | 1.43 |

**Every other view missed:** 153 ramps observed against 68.8 predicted under independence
(2.2×). GSV 128 vs 62.7 (2.0×); Mapillary 25 vs 5.4 (4.6×). The excess grows with capture
count: for ramps with 5–8 other captures, 66 observed vs 10.4 predicted. At ≥ 0.30 the pooled
figure is 98 vs 37.1 (2.6×). Figure: `docs/figures/multiview_48/failure_correlation.png`.

**Reading.** Misses are strongly correlated across views even when the cameras are 12–25 m
apart: a ramp one view misses tends to be missed by the others. The misses that survive the
union are properties of the ramp (its appearance, its setting), not of one pass (a parked car).
That is why capture count stops helping after four or five, and it is the quantitative reason
#48's "independent evidence" premise overstates what more captures can buy. Caveats as in §4;
the 0–3 m bin is mostly richmond (627 of 645 pairs), where consecutive Mapillary frames can be
under a metre apart.

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

**Answer: no.** Counting misses-in-range does not buy precision at the 0.30 tier that k-of-n
cannot. Out of sample (LOCO, keep-0.55) the score saves 2–3 FP sites on paterson and richmond
and costs 3–7 on gainesville and sao_paulo; in-sample the gains are 0–3 FP sites where there
are any. Every 0.30 policy trades
precision for recall against the operational 0.55: flat 0.30 adds 2.5 points of recall
(+63 TP sites) for +87 FP sites. Figure: `docs/figures/multiview_48/evidence_vs_kofn.png`.

**Caveats.** Only 15–59 FP sites per city label the false class, so the calibration is thin;
the in-sample variant is exactly that, and the LOCO variant pools two GSV-heavy groups against
one Mapillary city. **Below 0.55 precision is a lower bound**: the city GT was assembled from
RampNet detections at ≥ 0.55, so a sub-threshold detection is credited only if a reviewer
marked that ramp missed (`low_floor_sweep.py gtbias`; re-reviewing the sub-0.55 detections
found real, unlabelled ramps at 12.5–35% per split, 17% on richmond; `docs/model_comparison.md`). Vintage is not modelled; labeler#27 argued a miss in an
older view should count less. Richmond's sub-threshold detections come from the re-inference.

## 7. C. Challengers in world space (Richmond, free models)

**Question.** #48 claims cross-capture agreement "would kill much of the chat-VLMs'
false-positive flood" and so narrow the RampNet-vs-VLM gap. Measured here for the free legs.

**Bundle.** `benchmark/richmond_neighbourhood/`: every richmond run pano whose camera is within
20 m of a pool ramp (1,514), plus the 46 judged panos not already in that set: 1,560 panos, 124
judged. `bundle.json` borrows `benchmark/richmond/verdicts.json` (`compare.py` loads it with
`verdicts_from`), so only the judged panos are scored and `--detect-unjudged` runs the rest
detect-only. Records are copied from the labeler run (the judged ones verbatim from
`benchmark/richmond/records.jsonl`) and tagged with the ramps each pano qualifies for. Imagery
is linked from the native-res archive; the 124 judged panos are byte-identical to
`benchmark/richmond/imagery_manifest.json`.

**Control.** RampNet from its own run detections (≥ 0.55 from `results.jsonl`, below from the
re-inference). eval_sites on the full run reproduces the report's 0.941 / 0.959; the
generalized scorer gives the same 0.941 / 0.959 on the full run **and** on the neighbourhood
bundle alone, and the per-pano row on the bundle reproduces richmond's published
0.964 / 0.768 / 0.855. So restricting fusion to the neighbourhood loses nothing.

**Results.** Pending: the legs are running on makelab2 (see the run-status comment on #48).
This section is filled in when they finish.

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

**Reading.** Only 11 of the 74 residual ramps show no candidate anywhere. The rest fired: at the
operational tier in some view (35) or just below it (28). The residual is mostly an
association / placement problem and a threshold problem, not a detection problem, which
agrees with labeler#27's finding that fusion costs 1.5–4.7 points against the raw union. The
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

- **Capture count is not the constraint.** Recall from other views flattens after four or five
  captures and misses are correlated across views (§4–5). More captures, or reconstruction that
  uses more of them, cannot recover a ramp every view misses.
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
- **Evidence scoring is not worth building** at the 0.30 tier (§6). Attaching a multi-view
  support count to each submitted label, as labeler#27 closed on, costs nothing and keeps the
  option open for a server-side re-threshold.
- **Tier 3 is not motivated by these data.** Nothing here points at a failure a 3D scene
  representation would fix that association would not.
- Proposed follow-ups: Jon's one-rater pass on the gallery; §7's challenger read.

## 10. Reproduction

Inputs: the labeler checkout (read-only) at `b6bf5cf` (sidewalk-auto-labeler `main`; any
checkout whose `geo.py`, `depth.py`, `detectors/__init__.py`, `scripts/fuse_sites.py` and
`scripts/eval_sites.py` match the sha256s in `analysis_out/multiview_48/meta.json` → `labeler`),
its `runs/<city>/` (for `fusion_eval/report.md` and richmond's `results.f01.jsonl`), and copies of
the archived `results.jsonl` files from makelab2 (`--results-root`). The labeler's `runs/` and the
archive are not published; they are the inputs that keep `run` from being replicable from this
repo alone. **What is replicable from a clean clone:** every B.1 / B.2 table re-derives from the
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

# 1. B.1-B.4 (CPU, ~5 min)
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
| challenger smoke (2 + 2 panos × 7 legs) | makelab2 A40 | 15 min | ~0.25 | 0 |
| challenger legs (1,560 panos × 7 legs) | makelab2 A40 | pending | pending | 0 |

makelab2 has no `sacct`, so per `docs/compute_cost.md` its GPU time goes in
`analysis_out/usage_log.jsonl` as `paid: false` rows, one per leg, with wall-clock, s/pano and
hardware, not in `compute_log.jsonl`.
