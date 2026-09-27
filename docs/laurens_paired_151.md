# Laurens, paired: the rig effect on the corners both rigs saw (#151)

**Status:** measured 2026-09-27. Plan: the 2026-09-27 comment on
[#151](https://github.com/ProjectSidewalk/RampNet/issues/151). Script
`scripts/analysis/laurens_paired_151.py`; rows and tables `analysis_out/laurens_paired_151.json`
(`.md` beside it); tests `tests/test_laurens_paired_151.py`. Related: #112 (the depth axis), #158
(the rig-native retrain, which cites these Laurens findings).

## What this answers

Laurens is the one benchmark city imaged by two rigs over one footprint: `laurens_gsv` (Google,
September 2024, 16384 px) and `laurens_mapillary` (GoPro Max, November 2025, 5760 px). The
2026-09-03 comparison on #151 read the rig effect off the two whole arms. Those are different
panorama sets with different ground truth (220 against 249 ramps), so it compared two samples of
one town. Here both arms are restricted to the corners both rigs saw, and every number is a paired
difference with an interval.

| question | answer on the paired corners |
|---|---|
| Does RampNet's rig effect survive pairing? | **Yes.** ΔF1 (GSV minus Mapillary) **+0.112 [+0.033, +0.199]** on 47 pano pairs, against +0.115 on the whole arms. All of it is recall (ΔR +0.120 [+0.036, +0.211]); precision does not move (ΔP +0.007 [−0.038, +0.055]). |
| Is RampNet more rig-sensitive than the YOLO pano arms? | **Yes against two of the three at 95%, and in 97% of draws against the third.** The YOLO arms' paired ΔF1 are +0.028, −0.000 and −0.020, and every one of their intervals spans zero. RampNet minus YOLO: +0.084 [−0.001, +0.178] (`y11x_pano_h200`), +0.112 [+0.019, +0.211] (`y11l_pano`), +0.132 [+0.026, +0.251] (`y26_pano`). |
| Is "3–5×" the right way to say it? | **No.** Two of the three YOLO deltas are zero or negative on the paired corners, so a ratio is undefined or negative. The difference with its interval, in the table above, is the paired form of the claim. |
| Does "the town is not the problem" survive? | **Yes.** No zero-shot challenger gains more than +0.009 on the paired corners. Three lose with an interval that excludes zero: `gemini-3.1-pro-preview` −0.106, `Qwen3-VL-32B` −0.085, `owlv2` −0.011. `claude-opus-5` (effort low) reads −0.046 [−0.134, +0.046], against +0.008 on the whole arms. |
| How many physical ramps does the rig cost? | On the ramps both arms' GT contain (matched in world space), RampNet at 0.55 finds **23 on GSV only and 16 on Mapillary only**, out of 86. At a tighter 3 m match, with fewer chance matches, it is **17 and 8** out of 54. |
| Is the "near ramps are missed" signature real? | **Not on the paired corners.** The whole-arm deltas reproduce (Mapillary **+0.0038**, GSV **−0.0053**). On the paired panos both arms read about −0.001, with intervals that span zero on both sides. The Mapillary inversion comes from Mapillary panos outside the paired set. This supports the 2026-09-03 withdrawal. |
| What camera height is Laurens's GSV rig? | **2.41 m** (median, 60 measured benchmark panos; 2.40 m over all 1,418 measured panos of the run). That is the older 2.3–2.4 m rig, not Google's 2025–26 one. The flat axis is about 4% long, so 18 m / 25 m read 17.4 m / 23.4 m. See `docs/detection_recall_analysis.md` §0. |
| Can the GSV depth payload measure curb reveal? | **No.** At the 28 GT ramps within 8 m, where a curb face would span several payload rows, 24 windows hold one ground-like plane or none, and **none** shows a curb-sized step. The payload does not separate sidewalk from road at the ramps, so the curb-reveal measurement #151 asked for cannot come from it. The probe stops there, as planned. |

## How it was built

Every step reads committed inputs except the depth harvest and the curb probe, which need the GSV
depth payloads (below).

1. **Pairs.** Each `laurens_gsv` bundle pano is paired with the nearest `laurens_mapillary`
   bundle pano within 20 m, one-to-one, greedy by ascending distance. Pano positions and headings
   come from the bundle records.
2. **Physical ramps.** Each arm's GT points (`rampnet.detection_eval.build_ground_truth` over its
   verdict review: verdict-true detections plus non-unsure missed marks) are placed on the ground
   with the labeler's flat-ground convention (`sidewalk-auto-labeler/scripts/eval_sites.py`, read
   only). The bearing is heading + (x − 0.5)·360°, and the range is h / tan(depression). Points
   above 0.02 rad of depression or beyond 25 m are dropped as unplaceable. Within each pair the two
   arms' points are matched one-to-one by ascending distance within 5 m (the labeler's
   `--match-radius-m`). Camera height is the GSV pano's measured depth-payload height where it has
   one, else the labeler's 2.6 m. Mapillary uses 2.6 m, because the labeler's GoPro Max estimate
   (`runs/laurens/camera_heights.json`, h_scale 2.95 m) failed its own gate and is not applied.
3. **Paired scores.** Every committed leg on both arms is scored with the benchmark scorer
   (`score_pano`/`aggregate`) at the scoreboard's operating points: RampNet 0.55, YOLO 0.25,
   open-vocabulary detectors at their 0.05 export floor, chat VLMs unthresholded. Only the paired
   panos count. The paired deltas are GSV minus Mapillary. The intervals are a **pano-pair
   bootstrap**: 10,000 draws that resample the 47 pairs with replacement, **seed 151**, percentile
   2.5–97.5. The same draws are used for every leg, so RampNet-minus-YOLO is itself paired.
4. **The per-ramp 2×2** over the matched ramps: hit on both arms, GSV only, Mapillary only,
   neither. A hit is the benchmark matcher claiming that GT point.
5. **The near-miss delta**: median normalized y of missed GT minus median y of detected GT, per
   arm, on the whole arm and on the paired panos. The primary definition is the issue's:
   detected = verdict-true detections, missed = non-unsure missed marks. It reproduces the issue's
   +0.0038 and −0.0053 exactly. Beside it is the scorer-hit definition.
6. **The curb-reveal probe** (GSV only). Around each GT point on a measured-ground pano, a window of
   7 × 13 payload cells (about ±2° × ±4°) is searched for distinct ground-like planes (tilt ≤ 18°),
   and for the vertical step between two ground-like planes at every boundary they share inside
   it. The null is three windows on the same image row at azimuth +90°, +180° and +270°. It uses
   `recall_by_depth_112.py`'s conventions: image column = raw payload column, and planes
   `n · p + d = 0` with +z down.

## 1. The pairs

| | laurens_gsv | laurens_mapillary |
|---|---:|---:|
| bundle panos | 86 | 94 |
| with a pano of the other arm within 20 m | 51 | 49 |
| in a one-to-one pair | 47 | 47 |
| minimum spacing between two panos of the same arm | 30.2 m | 26.2 m |
| capture months of the paired panos | 2024-09 | 2025-11 |
| pair distance, median / max | 9.6 / 18.8 m | |

The 20 m counts reproduce the radius table of the 2026-09-03 comment (51 GSV / 49 Mapillary
panos). The one-to-one matching keeps 47 pairs. The minimum spacing within each arm (30.2 m GSV,
26.2 m Mapillary) is larger than the 20 m pairing radius, and the matching is one-to-one, so no
corner contributes two pairs.

## 2. The physical ramps

| | laurens_gsv | laurens_mapillary |
|---|---:|---:|
| GT ramps on the paired panos | 183 | 193 |
| not placeable (above the horizon or beyond 25 m) | 14 | 3 |
| placed | 169 | 190 |
| **matched across arms within 5 m (physical ramps both GTs have)** | **86** | **86** |
| placed but in this arm's GT only | 83 | 104 |

| check | matched ramps |
|---|---:|
| as placed (GSV measured height else 2.6 m; Mapillary 2.6 m) | 86 |
| sensitivity: Mapillary 2.95 m | 93 |
| sensitivity: all cameras 2.6 m | 79 |
| null: GSV points rotated 180 deg about their camera | 33 |
| null: GSV points rotated 270 deg about their camera | 37 |
| null: GSV points rotated 90 deg about their camera | 40 |
| match distance, median / p90 | 2.49 / 4.16 m |

| match radius | matched | null (mean of 3 rotations) | excess over null | share of matches in excess |
|---:|---:|---:|---:|---:|
| 2 m | 28 | 4.7 | 23.3 | 0.83 |
| 3 m | 54 | 14.7 | 39.3 | 0.73 |
| 4 m | 71 | 24.0 | 47.0 | 0.66 |
| 5 m | 86 | 36.7 | 49.3 | 0.57 |

**Read the matched count with its null.** Ramps cluster at corners, so a GSV point rotated by
90°, 180° or 270° about its own camera still lands within 5 m of some Mapillary GT point
33–40 times out of 169. At 5 m, about 57% of the 86 matches are in excess of chance; at 3 m,
73% of 54 are. Chance matches pair two different real ramps, so they dilute the 2×2 below toward
independence. They do not create a rig effect. The 3 m 2×2 is reported for that reason.

"This arm's GT only" (83 GSV, 104 Mapillary placed points) mixes three things this analysis does
not separate: ramps the other rig's pano could not see (different position, occlusion), ramps its
reviewer did not mark, and placement error beyond the match radius. It is not a GT-completeness
estimate.

The GSV placement uses the payload-measured camera heights. Those come from an **unpublished
input** (the GSV depth payloads, below). The "all cameras 2.6 m" row does not use them and gives 79
matches instead of 86.

## 3. Paired scores, every leg

GSV minus Mapillary on the 47 pairs. P/R/F1 are each arm's aggregate over its 47 paired panos.
The last column is the unpaired whole-arm ΔF1, which reproduces the 2026-09-03 table.

| leg | op | GSV P / R / F1 | Mapillary P / R / F1 | ΔP [95% CI] | ΔR [95% CI] | **ΔF1 [95% CI]** | whole-arm ΔF1 |
|---|---:|---|---|---|---|---|---:|
| rampnet@0.55 | 0.55 | 0.969 / 0.519 / 0.676 | 0.963 / 0.399 / 0.564 | +0.007 [-0.038, +0.055] | +0.120 [+0.036, +0.211] | **+0.112** [+0.033, +0.199] | +0.115 |
| rampnet@0.30 | 0.30 | – | 0.946 / 0.544 / 0.691 | – | – | – | – |
| rampnet_r2048@0.55 | 0.55 | 0.978 / 0.475 / 0.640 | 0.963 / 0.399 / 0.564 | +0.015 [-0.036, +0.073] | +0.076 [-0.007, +0.167] | **+0.076** [-0.005, +0.164] | +0.072 |
| rampnet_r2048@0.30 | 0.30 | 0.946 / 0.672 / 0.786 | 0.946 / 0.544 / 0.691 | +0.000 [-0.064, +0.060] | +0.128 [+0.055, +0.202] | **+0.095** [+0.040, +0.152] | +0.106 |
| gemini-3.6-flash | 0.00 | 0.372 / 0.208 / 0.267 | 0.483 / 0.218 / 0.300 | -0.110 [-0.236, +0.028] | -0.010 [-0.074, +0.060] | **-0.033** [-0.113, +0.055] | -0.003 |
| gemini-3.1-pro-preview | 0.00 | 0.484 / 0.164 / 0.245 | 0.520 / 0.264 / 0.350 | -0.036 [-0.206, +0.134] | -0.100 [-0.172, -0.024] | **-0.106** [-0.201, -0.005] | -0.064 |
| Qwen/Qwen3-VL-8B-Instruct | 0.00 | 0.224 / 0.142 / 0.174 | 0.289 / 0.212 / 0.245 | -0.065 [-0.162, +0.041] | -0.070 [-0.137, -0.002] | **-0.071** [-0.147, +0.009] | -0.049 |
| Qwen/Qwen3-VL-32B-Instruct | 0.00 | 0.000 / 0.000 / 0.000 | 0.450 / 0.047 / 0.085 | -0.450 [-0.688, -0.185] | -0.047 [-0.084, -0.014] | **-0.085** [-0.147, -0.026] | -0.048 |
| allenai/Molmo2-8B | 0.00 | 0.379 / 0.273 / 0.318 | 0.449 / 0.321 / 0.375 | -0.070 [-0.189, +0.048] | -0.048 [-0.139, +0.044] | **-0.057** [-0.156, +0.042] | -0.032 |
| google/owlv2-large-patch14-ensemble | 0.00 | 0.041 / 0.853 / 0.077 | 0.047 / 0.845 / 0.088 | -0.006 [-0.011, -0.001] | +0.008 [-0.057, +0.070] | **-0.011** [-0.020, -0.002] | -0.007 |
| IDEA-Research/grounding-dino-base | 0.00 | 0.041 / 0.891 / 0.079 | 0.036 / 0.798 / 0.069 | +0.005 [-0.000, +0.010] | +0.093 [+0.025, +0.161] | **+0.009** [-0.001, +0.018] | +0.009 |
| gemini-3.7-flash | 0.00 | 0.596 / 0.169 / 0.264 | 0.552 / 0.192 / 0.285 | +0.044 [-0.159, +0.234] | -0.022 [-0.094, +0.052] | **-0.021** [-0.123, +0.084] | -0.020 |
| y11l_pano | 0.25 | 0.955 / 0.459 / 0.620 | 0.947 / 0.461 / 0.620 | +0.008 [-0.066, +0.078] | -0.002 [-0.085, +0.086] | **-0.000** [-0.078, +0.082] | +0.023 |
| y11x_pano_h200 | 0.25 | 0.988 / 0.437 / 0.606 | 0.952 / 0.414 / 0.578 | +0.035 [-0.021, +0.102] | +0.023 [-0.075, +0.124] | **+0.028** [-0.066, +0.130] | +0.039 |
| y26_pano | 0.25 | 0.779 / 0.481 / 0.595 | 0.778 / 0.508 / 0.614 | +0.001 [-0.094, +0.100] | -0.027 [-0.131, +0.084] | **-0.020** [-0.102, +0.067] | -0.036 |
| claude-opus-5-effort-low | 0.00 | 0.511 / 0.388 / 0.441 | 0.569 / 0.425 / 0.487 | -0.059 [-0.172, +0.061] | -0.037 [-0.121, +0.050] | **-0.046** [-0.134, +0.046] | +0.008 |

- **`rampnet@0.30` has no GSV side.** `laurens_gsv` has no `op_cache`: its bundle stops at the
  deployed 0.55, and producing the low-floor cache needs GPU inference, which this CPU-only
  analysis did not run. So RampNet at the recommended 0.30 is Mapillary-only here.
- **`rampnet_r2048` is the #25 sweep's re-extraction** (`analysis_out/input_res_sweep_25/cache/r2048/`):
  the same model, run on the committed benchmark JPEGs resized to 2048×4096, on both arms, at both
  thresholds. On `laurens_mapillary` it reproduces the deployed records at 0.55 (whole-arm F1 0.543
  both ways). **On `laurens_gsv` it does not**: whole-arm F1 0.616 against the deployed run's
  0.659. The deployed GSV detections came from the labeler's production pipeline, not from these
  JPEGs, and why the two differ is not measured here. The practical consequence is that part of
  RampNet's GSV advantage depends on how the GSV imagery reached the model. On this same-code-path
  leg the paired rig effect is +0.076 [−0.005, +0.164] at 0.55 and **+0.095 [+0.040, +0.152] at
  0.30**, so the effect is present at the recommended operating point too.

### The headline: RampNet against the three YOLO pano arms

| YOLO arm | RampNet ΔF1 (0.55) | YOLO ΔF1 (0.25) | RampNet minus YOLO [95% CI] | draws with RampNet larger | ratio of point estimates |
|---|---:|---:|---|---:|---:|
| y11x_pano_h200 | +0.112 | +0.028 | +0.084 [-0.001, +0.178] | 0.973 | 3.95 |
| y11l_pano | +0.112 | -0.000 | +0.112 [+0.019, +0.211] | 0.991 | – |
| y26_pano | +0.112 | -0.020 | +0.132 [+0.026, +0.251] | 0.993 | – |

"Draws with RampNet larger" is the share of the 10,000 bootstrap draws in which RampNet's paired
ΔF1 exceeds the YOLO arm's. The ratio is shown only where the YOLO delta is positive.

## 4. The rig cost in ramps: the per-ramp 2×2

| RampNet leg | matched ramps | hit on both | GSV only | Mapillary only | neither | recall GSV | recall Mapillary |
|---|---:|---:|---:|---:|---:|---:|---:|
| rampnet@0.55 | 86 | 21 | 23 | 16 | 26 | 0.512 | 0.430 |
| rampnet@0.55 (3 m) | 54 | 14 | 17 | 8 | 15 | 0.574 | 0.407 |
| rampnet_r2048@0.30 | 86 | 40 | 22 | 14 | 10 | 0.721 | 0.628 |
| rampnet_r2048@0.55 | 86 | 21 | 19 | 16 | 30 | 0.465 | 0.430 |

On the same physical ramps, RampNet at 0.55 finds 7 more on GSV than on Mapillary out of 86, or 9
more out of 54 at the 3 m match. Both counts lean the same way as the recall delta in §3. On these
small counts the 2×2 is a description of the matched ramps, not a separate test. The 26 ramps
(5 m) that neither arm finds are the part of the Laurens deficit that a better rig does not
recover.

## 5. The near-miss delta, paired

Median normalized y of missed minus detected GT, RampNet at 0.55. Negative means misses sit nearer
the horizon (farther away), which is the normal direction in the other nine splits. The interval is
a pano-pair bootstrap on the paired subset (2,000 draws, seed 151, medians resampled directly).

| definition | arm | whole arm: delta (detected / missed) | paired panos: delta [95% CI] (detected / missed) |
|---|---|---|---|
| verdict (the issue's) | laurens_gsv | -0.0053 (111 / 109) | -0.0014 [-0.0179, +0.0191] (94 / 89) |
| verdict (the issue's) | laurens_mapillary | +0.0038 (97 / 152) | -0.0011 [-0.0099, +0.0198] (77 / 116) |
| scorer hit at 0.55 | laurens_gsv | -0.0065 (112 / 108) | -0.0017 [-0.0192, +0.0181] (95 / 88) |
| scorer hit at 0.55 | laurens_mapillary | +0.0038 (97 / 152) | -0.0011 [-0.0099, +0.0198] (77 / 116) |

On the corners both rigs saw, neither arm shows the inverted signature. Both deltas are about
−0.001, and both intervals include zero. The whole-arm Mapillary inversion (+0.0038) is carried by
Mapillary panos outside the paired set. This is consistent with the 2026-09-03 withdrawal of "near,
well-resolved ramps are being missed". The two intervals differ in what else they rule out. The GSV
interval (−0.018 to +0.019) is wide enough to include the far-field tilt the other nine splits
show (−0.013 to −0.030). The paired Mapillary interval (−0.010 to +0.020) excludes it: on those
panos, misses are neither nearer nor farther than hits.

The depth axis adds one more observation (`docs/detection_recall_analysis.md` §0.2): on
`laurens_gsv`, RampNet's recall is low at every range, 0.607 at 0–8 m and 0.478 at 8–12 m
(measured-ground panos, depth axis), against bend's 0.885 and 0.922. Laurens's deficit is not a
far-field effect on either arm.

## 6. The curb-reveal probe: the payload cannot supply it

| windows | n | median depth range | ≤ 1 ground-like plane | ≥ 2 ground-like planes | with a ground/ground boundary | curb-sized step (5–30 cm) | largest step per window: p25 / median / p75 |
|---|---:|---:|---:|---:|---:|---:|---|
| GT ramps RampNet detected (0.55) | 73 | 11.4 m | 49 | 24 | 25 | 10 (0.137) | 0.011 / 0.037 / 0.068 m |
| GT ramps RampNet missed | 78 | 13.7 m | 43 | 35 | 35 | 21 (0.269) | 0.034 / 0.082 / 0.133 m |
| all GT ramps | 151 | 12.5 m | 92 | 59 | 60 | 31 (0.205) | 0.020 / 0.058 / 0.105 m |
| GT ramps within 8 m (depth axis) | 28 | 6.5 m | 24 | 4 | 4 | 0 (0.000) | 0.018 / 0.025 / 0.030 m |
| null: same image row, azimuth +90°/180°/270° | 453 | 12.5 m | 326 | 127 | 131 | 70 (0.154) | 0.024 / 0.057 / 0.091 m |

"Median depth range" is the GT point's own range on the depth axis. The null windows share their
GT point's image row, so they share its range.

The deciding row is the near one. One payload row is 0.70°, so a 15 cm curb face spans about two
rows at 6.5 m (the median range of the near group) and about one at the 11–14 m median range of the other groups. The
near windows are where a sidewalk plane would show up if the payload modelled one. In 24 of those 28 windows there is one ground-like plane or none, and
no window shows a curb-sized step. Farther out, where one payload row covers more than a curb face,
two-plane windows are more common, and their steps look like the null windows at the same image
row: 15% curb-sized in the null against 21% at GT ramps. The payload's ground model is a small set
of large planes, fitted per pano. It does not separate sidewalk from road at the ramps, so it
cannot give a per-ramp curb-reveal measurement.

The missed and detected rows differ: curb-sized steps at 27% against 14%, and a median largest step
of 8 cm against 4 cm. **This is not read as curb reveal.** The missed ramps sit farther away (median
13.7 m against 11.4 m), the step grows with range in this payload, and the rows where a curb would
be resolvable show no steps at all. The flush-ramp hypothesis in #151 stays untested. Testing it
needs a measurement at the ramp: a crop-level estimate from the imagery, or survey data.

## Caveats that travel with these numbers

- **47 pairs.** Every interval here is wide. The RampNet rig effect and two of the three
  RampNet-minus-YOLO differences clear zero; the paired near-miss deltas do not.
- **Held out, still.** `laurens_gsv` stays out of every pooled number in the benchmark and in
  `recall_by_depth_112.json`. Nothing here changes a pooled figure.
- **One seed, verdict-review GT.** The GT on each arm was assembled during a RampNet review of that
  arm ("RampNet-anchored", `docs/model_comparison.md`), and each arm was reviewed separately. The
  paired design removes the between-arm difference in *which corners* were sampled, not the
  difference in *which ramps each review marked*. The "this arm's GT only" row in §2 is the size of
  that difference.
- **World placement is flat ground at an assumed or payload height**, with no pose correction
  (the labeler's `eval_sites.py` default). The rotation null and the radius sweep bound how much of
  the matching is chance. The 2×2 at 3 m is the conservative read.
- **The deployed GSV run and the r2048 re-extraction differ on `laurens_gsv`** (whole-arm F1 0.659
  against 0.616, §3). Which one to call "RampNet on GSV" is a real choice. The headline uses the
  deployed run, the one the benchmark and the 2026-09-03 comment report.
- **Unpublished input: the GSV depth payloads.** The camera heights used to place GSV points (§2),
  the curb probe (§6) and the laurens_gsv depth axis come from Google-derived depth payloads
  archived in the labeler (`runs/laurens_gsv/depth/`, 2,137 files, `index.csv` sha256
  `3e11306d…`). They are not published, for the same reason as the other four cities' archives
  (`docs/detection_recall_analysis.md` §0.5). Every row derived from them is committed, so the
  tables re-derive without them. Publishing the archive would unblock a from-scratch re-derivation.

## Left out, and why

- **RampNet at 0.30 on the deployed `laurens_gsv` run.** It needs a `laurens_gsv` `op_cache`, which
  needs GPU inference. That was out of scope for this CPU-only pass. The r2048 re-extraction gives a
  same-code-path 0.30 read on both arms instead, with the caveat above.
- **A curb-reveal measurement.** The payload cannot supply one (§6). This is a negative result, not
  a skipped step.
- **Why the deployed GSV run beats the r2048 re-extraction.** Observed, not investigated.

## Cost

| step | where | wall clock | spend |
|---|---|---:|---:|
| GSV depth harvest, 2,137 panos (`harvest_depth_launch_151.py` around the labeler's `harvest_depth.py`) | desktop CPU + the GSV metadata endpoint | 280 s | $0 |
| `recall_by_depth_112.py --only laurens_gsv` | desktop CPU | ~10 s | $0 |
| `depth_image_alignment_112.py --splits laurens_gsv` | desktop CPU | ~24 s | $0 |
| `laurens_paired_151.py` (derive, bootstrap included) | desktop CPU | ~2 s | $0 |

No GPU and no cluster. The harvest's wall clock is a `paid: false` row in
`analysis_out/usage_log.jsonl` (`depth-harvest-151:laurens_gsv`).

**The labeler's harvest script refuses every GSV run at its HEAD `29dc605`.** It compares each
record's `pano.source` to `"gsv"`, but GSV runs store GSV's own upload type there (`"launch"`, in
all 2,137 `laurens_gsv` records and in every bend, paterson, gainesville and sao_paulo record). The
harvest therefore ran the labeler's code unchanged through `scripts/analysis/harvest_depth_launch_151.py`,
which replaces only that check. It first verifies that the manifest says GSV and that every record
says `"launch"`. The labeler repo was not modified; this needs a one-line fix there.

## Reproduce

```bash
# 1. depth payloads (network: the GSV metadata endpoint; resumable), from the labeler checkout
cd D:/Git/sidewalk-auto-labeler
.venv/Scripts/python.exe D:/Git/RampNet/scripts/analysis/harvest_depth_launch_151.py runs/laurens_gsv

# 2. back in RampNet: append the laurens_gsv depth rows (asserts no pre-existing row or table moves)
python scripts/analysis/recall_by_depth_112.py --only laurens_gsv --labeler-root D:/Git/sidewalk-auto-labeler

# 3. the image<->payload alignment check on the held-out arm (needs benchmark/laurens_gsv/panos)
python scripts/analysis/depth_image_alignment_112.py --splits laurens_gsv --panos-root . \
    --out analysis_out/depth_image_alignment_151_laurens_gsv.json

# 4. the paired analysis (the curb probe reads the payloads; everything else is committed inputs)
python scripts/analysis/laurens_paired_151.py --labeler-root D:/Git/sidewalk-auto-labeler

# no payloads needed: re-derive every row but the probe from committed inputs, every table from
# the rows, and check the content hash
python scripts/analysis/laurens_paired_151.py --check
python scripts/analysis/recall_by_depth_112.py --check
```
