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

**Corrected after review** ([PR #201 review](https://github.com/ProjectSidewalk/RampNet/pull/201#issuecomment-5858671681)).
The first version of this document said RampNet's excess rig sensitivity over the YOLO pano arms
"survives pairing". It does not survive the two robustness checks the committed inputs allow, and
the rows below state what does.

| question | answer on the paired corners |
|---|---|
| Does RampNet's own rig effect survive pairing? | **Yes.** ΔF1 (GSV minus Mapillary) **+0.112 [+0.033, +0.199]** on 47 pano pairs for the deployed run, against +0.115 on the whole arms. The effect is recall (ΔR +0.120 [+0.036, +0.211]); ΔP's interval spans zero (+0.007 [−0.038, +0.055]). On the same-input re-run of the committed JPEGs (`rampnet_r2048`) it is +0.076 [−0.005, +0.164] at 0.55 and **+0.095 [+0.040, +0.152] at 0.30**. |
| Is RampNet more rig-sensitive than the YOLO pano arms? | **Not established.** With the deployed GSV run, RampNet minus YOLO clears zero against two of the three arms. With the same-input RampNet leg at 0.55, the leg that saw the same JPEGs the YOLO arms saw, **no difference clears zero**. At 0.30 two of three do. On the physical ramps both reviews contain, **the YOLO arms gain about as much as RampNet**: net GSV-only minus Mapillary-only is +7 for RampNet and +5, +5 and +2 for the YOLO arms, and no 2×2 asymmetry is significant (exact McNemar p 0.34 to 0.86). All of these intervals cover pano sampling only; each model is one training run. |
| Is "3–5×" the right way to say it? | **No.** Two of the three YOLO paired deltas are zero or negative, so a ratio is undefined or negative, and the difference itself is not robust (above). |
| What do the zero-shot models do on the same corners? | No zero-shot leg gains F1 on the GSV arm: the largest paired ΔF1 is +0.009 (`grounding-dino-base`). F1 barely moves for the open-vocabulary detectors, because their precision is about 0.04. `grounding-dino-base` does gain recall on GSV (ΔR +0.093 [+0.025, +0.161]). Both arms are the same town, so this compares rigs; **it does not test the town** (see §3.1). |
| Is part of the Laurens deficit independent of the rig? | **Yes, on RampNet's home rig.** On the depth axis, `laurens_gsv` recall is 0.607 at 0–8 m and 0.478 at 8–12 m, against bend's 0.885 and 0.922. About a third of the matched physical ramps (26 of 86 at 5 m) are missed by RampNet on both rigs. |
| How many physical ramps does the rig cost RampNet? | 23 found on GSV only and 16 on Mapillary only, out of 86 matched (**exact McNemar p 0.34**). At a tighter 3 m match it is 17 and 8, out of 54 (**p 0.11**). Neither asymmetry is significant. |
| Is the "near ramps are missed" signature real? | **Pairing cannot say.** The whole-arm deltas reproduce (Mapillary **+0.0038**, GSV **−0.0053**). The paired Mapillary estimate is −0.0011 [−0.0099, +0.0198], and that interval includes both zero and the whole-arm +0.0038. It does exclude the −0.013 to −0.030 far-field tilt of the other nine splits. |
| What camera height is Laurens's GSV rig? | **2.41 m** (median, 60 measured benchmark panos; 2.40 m over all 1,418 measured panos of the run). That is the older 2.3–2.4 m rig, not Google's 2025–26 one. The flat axis is about 4% long, so 18 m / 25 m read 17.4 m / 23.4 m. The image↔payload mapping on this split is carried over from the four pooled splits and is not independently confirmed; see `docs/detection_recall_analysis.md` §0. |
| Can the GSV depth payload measure curb reveal? | **No.** At the 28 GT ramps within 8 m, where a 15 cm curb face spans about two payload rows, 24 windows hold one ground-like plane or none, and **none** shows a curb-sized step. The payload does not separate sidewalk from road at the ramps where it could, so the curb-reveal measurement #151 asked for cannot come from it. |

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
   less than 0.02 rad below the horizon or beyond 25 m are dropped as unplaceable. Within each pair the two
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
   **The intervals cover pano sampling only.** Each model is a single training run, and this
   repo's seed-variance work (#51, #135) found seed variance to be the binding limit for
   RampNet-versus-YOLO comparisons. Any rig-sensitivity statement here is about these checkpoints.
4. **The per-ramp 2×2** over the matched ramps, for every RampNet leg and the three YOLO pano
   arms: hit on both arms, GSV only, Mapillary only, neither. A hit is the benchmark matcher
   claiming that GT point. The two discordant cells are tested with an exact two-sided McNemar
   test. The 2×2 is restricted to ramps both arms' reviews contain, which removes differences
   in how complete each review's missed-ramp pass was.
5. **The near-miss delta**: median normalized y of missed GT minus median y of detected GT, per
   arm, on the whole arm and on the paired panos. The primary definition is the issue's:
   detected = verdict-true detections, missed = non-unsure missed marks. It reproduces the issue's
   +0.0038 and −0.0053 exactly. Beside it is the scorer-hit definition.
6. **The curb-reveal probe** (GSV only). Around each GT point on a measured-ground pano, a window of
   7 × 13 payload cells (about ±2° × ±4°) is searched for distinct ground-like planes (tilt ≤ 18°),
   and for the vertical step between two ground-like planes at every boundary they share inside
   it (edges that leave the window are not counted). The null is three windows on the same image row at azimuth +90°, +180° and +270°. It uses
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
26.2 m Mapillary) is larger than the 20 m pairing radius, and the matching is one-to-one, so each
pano is in at most one pair and no two pairs share a pano.

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
independence. They do not create a rig effect. The 2×2 is reported at 3 m for that reason.

"This arm's GT only" (83 GSV, 104 Mapillary placed points) mixes three things this analysis does
not separate: ramps the other rig's pano could not see (different position, occlusion), ramps its
reviewer did not mark, and placement error beyond the match radius. It is not a GT-completeness
estimate.

The GSV placement uses the payload-measured camera heights. Those come from an **unpublished
input** (the GSV depth payloads, below). The "all cameras 2.6 m" row does not use them and gives 79
matches instead of 86.

## 3. Paired scores, every leg

GSV minus Mapillary on the 47 pairs. P/R/F1 are each arm's aggregate over its 47 paired panos.
"op" is the scoreboard's operating-point label: "0.05 floor" is the open-vocabulary detectors'
export floor, and "no score" means the leg emits no confidence. "Predictions" counts every point
the leg emitted on the paired panos. The last column is the unpaired whole-arm ΔF1. It
reproduces the 2026-09-03 table to ±0.001: that table took differences of rounded F1s, so
`claude-opus-5` reads +0.008 here against +0.007 there, `y11l_pano` +0.023 against +0.024, and
`grounding-dino-base` +0.009 against +0.008.

| leg | op | predictions GSV / Mly | GSV P / R / F1 | Mapillary P / R / F1 | ΔP [95% CI] | ΔR [95% CI] | **ΔF1 [95% CI]** | whole-arm ΔF1 |
|---|---:|---:|---|---|---|---|---|---:|
| rampnet@0.55 | 0.55 | 99 / 80 | 0.969 / 0.519 / 0.676 | 0.963 / 0.399 / 0.564 | +0.007 [-0.038, +0.055] | +0.120 [+0.036, +0.211] | **+0.112** [+0.033, +0.199] | +0.115 |
| rampnet@0.30 | 0.30 | – / 113 | – | 0.946 / 0.544 / 0.691 | – | – | – | – |
| rampnet_r2048@0.55 | 0.55 | 91 / 80 | 0.978 / 0.475 / 0.640 | 0.963 / 0.399 / 0.564 | +0.015 [-0.036, +0.073] | +0.076 [-0.007, +0.167] | **+0.076** [-0.005, +0.164] | +0.072 |
| rampnet_r2048@0.30 | 0.30 | 134 / 113 | 0.946 / 0.672 / 0.786 | 0.946 / 0.544 / 0.691 | +0.000 [-0.064, +0.060] | +0.128 [+0.055, +0.202] | **+0.095** [+0.040, +0.152] | +0.106 |
| gemini-3.6-flash | no score | 103 / 87 | 0.372 / 0.208 / 0.267 | 0.483 / 0.218 / 0.300 | -0.110 [-0.236, +0.028] | -0.010 [-0.074, +0.060] | **-0.033** [-0.113, +0.055] | -0.003 |
| gemini-3.1-pro-preview | no score | 63 / 100 | 0.484 / 0.164 / 0.245 | 0.520 / 0.264 / 0.350 | -0.036 [-0.206, +0.134] | -0.100 [-0.172, -0.024] | **-0.106** [-0.201, -0.005] | -0.064 |
| Qwen/Qwen3-VL-8B-Instruct | no score | 116 / 142 | 0.224 / 0.142 / 0.174 | 0.289 / 0.212 / 0.245 | -0.065 [-0.162, +0.041] | -0.070 [-0.137, -0.002] | **-0.071** [-0.147, +0.009] | -0.049 |
| Qwen/Qwen3-VL-32B-Instruct | no score | 4 / 20 | 0.000 / 0.000 / 0.000 | 0.450 / 0.047 / 0.085 | -0.450 [-0.688, -0.185] | -0.047 [-0.084, -0.014] | **-0.085** [-0.147, -0.026] | -0.048 |
| allenai/Molmo2-8B | no score | 132 / 140 | 0.379 / 0.273 / 0.318 | 0.449 / 0.321 / 0.375 | -0.070 [-0.189, +0.048] | -0.048 [-0.139, +0.044] | **-0.057** [-0.156, +0.042] | -0.032 |
| google/owlv2-large-patch14-ensemble | 0.05 floor | 3862 / 3522 | 0.041 / 0.853 / 0.077 | 0.047 / 0.845 / 0.088 | -0.006 [-0.011, -0.001] | +0.008 [-0.057, +0.070] | **-0.011** [-0.020, -0.002] | -0.007 |
| IDEA-Research/grounding-dino-base | 0.05 floor | 3976 / 4272 | 0.041 / 0.891 / 0.079 | 0.036 / 0.798 / 0.069 | +0.005 [-0.000, +0.010] | +0.093 [+0.025, +0.161] | **+0.009** [-0.001, +0.018] | +0.009 |
| gemini-3.7-flash | no score | 52 / 67 | 0.596 / 0.169 / 0.264 | 0.552 / 0.192 / 0.285 | +0.044 [-0.159, +0.234] | -0.022 [-0.094, +0.052] | **-0.021** [-0.123, +0.084] | -0.020 |
| y11l_pano | 0.25 | 88 / 94 | 0.955 / 0.459 / 0.620 | 0.947 / 0.461 / 0.620 | +0.008 [-0.066, +0.078] | -0.002 [-0.085, +0.086] | **-0.000** [-0.078, +0.082] | +0.023 |
| y11x_pano_h200 | 0.25 | 81 / 84 | 0.988 / 0.437 / 0.606 | 0.952 / 0.414 / 0.578 | +0.035 [-0.021, +0.102] | +0.023 [-0.075, +0.124] | **+0.028** [-0.066, +0.130] | +0.039 |
| y26_pano | 0.25 | 113 / 126 | 0.779 / 0.481 / 0.595 | 0.778 / 0.508 / 0.614 | +0.001 [-0.094, +0.100] | -0.027 [-0.131, +0.084] | **-0.020** [-0.102, +0.067] | -0.036 |
| claude-opus-5-effort-low | no score | 141 / 144 | 0.511 / 0.388 / 0.441 | 0.569 / 0.425 / 0.487 | -0.059 [-0.172, +0.061] | -0.037 [-0.121, +0.050] | **-0.046** [-0.134, +0.046] | +0.008 |

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
- **`Qwen3-VL-32B-Instruct` is a near-silent leg on these panos** (see the predictions column),
  so its ΔF1 reflects how few points it emits, not a rig loss.

### RampNet against the three YOLO pano arms (corrected after review)

| RampNet leg | YOLO arm (0.25) | RampNet ΔF1 | YOLO ΔF1 | RampNet minus YOLO [95% CI] | draws with RampNet larger | ratio of point estimates |
|---|---|---:|---:|---|---:|---:|
| rampnet@0.55 | y11x_pano_h200 | +0.112 | +0.028 | +0.084 [-0.001, +0.178] | 0.973 | 3.95 |
| rampnet@0.55 | y11l_pano | +0.112 | -0.000 | +0.112 [+0.019, +0.211] | 0.991 | – |
| rampnet@0.55 | y26_pano | +0.112 | -0.020 | +0.132 [+0.026, +0.251] | 0.993 | – |
| rampnet_r2048@0.55 | y11x_pano_h200 | +0.076 | +0.028 | +0.047 [-0.037, +0.134] | 0.858 | 2.66 |
| rampnet_r2048@0.55 | y11l_pano | +0.076 | -0.000 | +0.076 [-0.020, +0.172] | 0.939 | – |
| rampnet_r2048@0.55 | y26_pano | +0.076 | -0.020 | +0.095 [-0.008, +0.203] | 0.964 | – |
| rampnet_r2048@0.30 | y11x_pano_h200 | +0.095 | +0.028 | +0.067 [-0.025, +0.155] | 0.923 | 3.35 |
| rampnet_r2048@0.30 | y11l_pano | +0.095 | -0.000 | +0.096 [+0.012, +0.176] | 0.988 | – |
| rampnet_r2048@0.30 | y26_pano | +0.095 | -0.020 | +0.115 [+0.022, +0.206] | 0.993 | – |

"Draws with RampNet larger" is the share of the 10,000 bootstrap draws in which RampNet's paired
ΔF1 exceeds the YOLO arm's. The ratio is shown only where the YOLO delta is positive.

The deployed GSV run came from the labeler's production pipeline. The YOLO arms were run on the
committed benchmark JPEGs (`source_max_edge 4096` in their signatures). So on GSV the deployed row
compares the rig plus an input-path difference that only RampNet received. `rampnet_r2048` is
RampNet on the same JPEGs. On that leg at 0.55, no RampNet-minus-YOLO interval clears zero; at
0.30, two of the three do. Together with the 2×2 in §4, where the YOLO arms gain about as many
physical ramps as RampNet, **the data do not establish that RampNet is more rig-sensitive than the
YOLO arms trained on its data.** The 2026-09-03 reading ("three to five times larger") and the
first version of this document ("survives pairing as a difference") are withdrawn.

### 3.1 What the zero-shot rows show, and what they do not

No zero-shot leg gains F1 on the GSV arm of the same corners (largest +0.009). Three lose with an
interval that excludes zero: `gemini-3.1-pro-preview` −0.106, `owlv2` −0.011, and `Qwen3-VL-32B`
−0.085, but the last is a near-silent leg (above). Because both arms are the same town, these rows
compare rigs, and they cannot test whether the town is hard. Three numbers point to a deficit that
does not depend on the rig:

| evidence | value |
|---|---|
| `laurens_gsv` RampNet recall on the depth axis, 0–8 m / 8–12 m (bend in the same bands) | 0.607 / 0.478 (0.885 / 0.922) |
| matched physical ramps RampNet misses on both rigs (5 m) | 26 of 86 |
| `grounding-dino-base` paired ΔR (a zero-shot leg that does gain recall on GSV) | +0.093 [+0.025, +0.161] |

The town question needs a different comparison: for example, `laurens_gsv` against another GSV
split of the same 2024 rig vintage. That was not run here.

## 4. The rig effect in physical ramps: the per-ramp 2×2

| match radius | leg | matched ramps | hit on both | GSV only | Mapillary only | neither | net (GSV − Mly) | exact McNemar p | recall GSV | recall Mapillary |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 m | rampnet@0.55 | 86 | 21 | 23 | 16 | 26 | +7 | 0.34 | 0.512 | 0.430 |
| 5 m | rampnet_r2048@0.30 | 86 | 40 | 22 | 14 | 10 | +8 | 0.24 | 0.721 | 0.628 |
| 5 m | rampnet_r2048@0.55 | 86 | 21 | 19 | 16 | 30 | +3 | 0.74 | 0.465 | 0.430 |
| 5 m | y11l_pano | 86 | 26 | 20 | 15 | 25 | +5 | 0.50 | 0.535 | 0.477 |
| 5 m | y11x_pano_h200 | 86 | 23 | 19 | 14 | 30 | +5 | 0.49 | 0.488 | 0.430 |
| 5 m | y26_pano | 86 | 33 | 18 | 16 | 19 | +2 | 0.86 | 0.593 | 0.570 |
| 3 m | rampnet@0.55 | 54 | 14 | 17 | 8 | 15 | +9 | 0.11 | 0.574 | 0.407 |
| 3 m | rampnet_r2048@0.30 | 54 | 28 | 15 | 6 | 5 | +9 | 0.08 | 0.796 | 0.630 |
| 3 m | rampnet_r2048@0.55 | 54 | 13 | 15 | 9 | 17 | +6 | 0.31 | 0.518 | 0.407 |
| 3 m | y11l_pano | 54 | 22 | 9 | 8 | 15 | +1 | 1.00 | 0.574 | 0.556 |
| 3 m | y11x_pano_h200 | 54 | 19 | 12 | 7 | 16 | +5 | 0.36 | 0.574 | 0.481 |
| 3 m | y26_pano | 54 | 25 | 10 | 9 | 10 | +1 | 1.00 | 0.648 | 0.630 |

On the same physical ramps, RampNet at 0.55 finds 7 more on GSV than on Mapillary out of 86
(exact McNemar p 0.34), or 9 more out of 54 at the 3 m match (p 0.11). Neither asymmetry is
significant. At 5 m the YOLO pano arms lean the same way by about as much (+5, +5, +2). At 3 m
RampNet's lean (+9, and +9 for the same-input leg at 0.30, p 0.08) is larger than the YOLO arms'
(+5, +1, +1), but no asymmetry in the table is significant at 0.05. About a third of
the matched ramps are missed on both rigs (26 of 86 at 5 m, 15 of 54 at 3 m): that part of the
Laurens deficit does not depend on the rig.

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

The paired Mapillary estimate is −0.0011, and its interval [−0.0099, +0.0198] includes both zero
and the whole-arm +0.0038. **Pairing neither confirms nor removes the inversion**, and the data
cannot say where it comes from. What the interval does exclude is the far-field tilt of the other
nine splits (−0.013 to −0.030): on the paired Mapillary panos, misses do not sit clearly farther
away than hits. The GSV interval (−0.018 to +0.019) is wide enough to include that tilt.

The depth axis adds one more observation (`docs/detection_recall_analysis.md` §0.2): on
`laurens_gsv`, RampNet's recall is low at every range, 0.607 at 0–8 m and 0.478 at 8–12 m
(measured-ground panos, depth axis), against bend's 0.885 and 0.922. Laurens's deficit is not a
far-field effect on either arm.

## 6. The curb-reveal probe: the payload cannot supply it

| windows | n | median depth range | ≤ 1 ground-like plane | ≥ 2 ground-like planes | with a ground/ground boundary | curb-sized step (5–30 cm) | largest step per window: p25 / median / p75 |
|---|---:|---:|---:|---:|---:|---:|---|
| GT ramps RampNet detected (0.55) | 73 | 11.4 m | 49 | 24 | 24 | 10 (0.137) | 0.010 / 0.029 / 0.074 m |
| GT ramps RampNet missed | 78 | 13.7 m | 43 | 35 | 35 | 21 (0.269) | 0.034 / 0.082 / 0.133 m |
| all GT ramps | 151 | 12.5 m | 92 | 59 | 59 | 31 (0.205) | 0.019 / 0.058 / 0.105 m |
| GT ramps within 8 m (depth axis) | 28 | 6.5 m | 24 | 4 | 4 | 0 (0.000) | 0.018 / 0.025 / 0.030 m |
| null: same image row, azimuth +90°/180°/270° | 453 | 12.5 m | 326 | 127 | 127 | 69 (0.152) | 0.018 / 0.057 / 0.095 m |

"Median depth range" is the GT point's own range on the depth axis. The null windows share their
GT point's image row, so they share its range.

| depth range | GT windows | GT: curb-sized step | GT: largest step, median | null windows | null: curb-sized step | null: largest step, median |
|---|---:|---:|---:|---:|---:|---:|
| 0-8 m | 28 | 0 (0.000) | 0.025 m | 84 | 6 (0.071) | 0.050 m |
| 8-12 m | 46 | 2 (0.043) | 0.046 m | 138 | 15 (0.109) | 0.043 m |
| 12-18 m | 37 | 7 (0.189) | 0.035 m | 111 | 24 (0.216) | 0.055 m |
| 18 m+ | 40 | 22 (0.550) | 0.090 m | 120 | 24 (0.200) | 0.074 m |

The deciding rows are the near ones. One payload row is 0.70°, so a 15 cm curb face spans about
two rows at 6.5 m (the median range of the 0–8 m group) and about one row at 12 m. The near
windows are where a sidewalk plane would show up if the payload modelled one. In 24 of the 28 GT
windows within 8 m there is one ground-like plane or none, and none of the 28 shows a curb-sized
step. The null windows at the same range show some (6 of 84). The payload's ground model is a
small set of large planes, fitted per pano, and it does not separate sidewalk from road at the
ramps where the resolution would allow it. So it cannot give a per-ramp curb-reveal measurement.

**Beyond 18 m the GT windows show more curb-sized steps than the null** (22 of 40 against 24 of 120).
At that range one payload row covers more than 20 cm of height, so a step there is a boundary
between two large coarse planes near a corner, not a resolved curb face. It could be the payload
marking a curb line at coarse scale, or planes breaking up at intersections. This probe cannot
tell those apart, and it is recorded here as an open observation, not a measurement.

The missed and detected rows also differ: curb-sized steps at 27% against 14%, and a median
largest step of 8 cm against 3 cm. **This is not read as curb reveal.** The missed ramps sit
farther away (median 13.7 m against 11.4 m), and the table above shows the step share rising with
range at GT ramps (0% below 8 m, 55% beyond 18 m). The rows where a curb would be resolvable show
no steps at all. The flush-ramp hypothesis in #151 stays untested. Testing it needs a
measurement at the ramp: a crop-level estimate from the imagery, or survey data.

## Caveats that travel with these numbers

- **47 pairs.** Every interval here is wide. RampNet's own rig effect clears zero; the
  RampNet-minus-YOLO differences do not, on the same-input leg at 0.55; the paired near-miss
  deltas do not.
- **One training run per model.** The bootstrap resamples pano pairs, so the intervals cover
  pano sampling only. Seed variance (#51, #135) is not in them.
- **Held out, still.** `laurens_gsv` stays out of every pooled number in the benchmark and in
  `recall_by_depth_112.json`. Nothing here changes a pooled figure.
- **Verdict-review GT.** The GT on each arm was assembled during a RampNet review of that
  arm ("RampNet-anchored", `docs/model_comparison.md`), and each arm was reviewed separately. The
  paired design removes the between-arm difference in *which corners* were sampled, not the
  difference in *which ramps each review marked*. The "this arm's GT only" row in §2 is the size of
  that difference.
- **World placement is flat ground at an assumed or payload height**, with no pose correction
  (the labeler's `eval_sites.py` default). The rotation null and the radius sweep bound how much of
  the matching is chance. The 2×2 at 3 m is the conservative read.
- **The deployed GSV run and the r2048 re-extraction differ on `laurens_gsv`** (whole-arm F1 0.659
  against 0.616, §3). The deployed run is what the benchmark and the 2026-09-03 comment report.
  The r2048 leg is the one that saw the same input as the YOLO arms, so any comparison with them
  is read on it (§3).
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
check became an allowlist in labeler commit `6666bf7` (2026-09-06); before that it refused only
`"mapillary"`. The harvest therefore ran the labeler's code unchanged through
`scripts/analysis/harvest_depth_launch_151.py`, which replaces only that check. It first verifies
that the manifest says GSV and that every record says `"launch"`, and the manifest check
(`check_gsv`) still runs. The labeler repo was not modified. Filed as
[sidewalk-auto-labeler#99](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/99).

## Reproduce

```bash
# 1. depth payloads (network: the GSV metadata endpoint; resumable), from the labeler checkout
cd D:/Git/sidewalk-auto-labeler
.venv/Scripts/python.exe D:/Git/RampNet/scripts/analysis/harvest_depth_launch_151.py runs/laurens_gsv

# 2. back in RampNet: append the laurens_gsv depth rows (asserts the pre-existing rows and
#    tables still hash to their pre-#151 content, and that the parser reproduces the pooled rows)
python scripts/analysis/recall_by_depth_112.py --only laurens_gsv --labeler-root D:/Git/sidewalk-auto-labeler

# 3. the image<->payload alignment check on the held-out arm. Needs the native panos in
#    benchmark/laurens_gsv/panos. They are NOT on the Hub yet: projectsidewalk/rampnet-benchmark
#    carries nine splits and not the two Laurens arms (scripts/unpack_benchmark_panos.py
#    docstring), so this step needs the local bundle imagery. Verify it first with
#    python scripts/analysis/imagery_manifest.py --verify --cities laurens_gsv
python scripts/analysis/depth_image_alignment_112.py --splits laurens_gsv --panos-root . \
    --out analysis_out/depth_image_alignment_151_laurens_gsv.json

# 4. the paired analysis (the curb probe reads the payloads; everything else is committed inputs)
python scripts/analysis/laurens_paired_151.py --labeler-root D:/Git/sidewalk-auto-labeler

# no payloads needed: re-derive every row but the probe from committed inputs, every table from
# the rows, and check the content hash
python scripts/analysis/laurens_paired_151.py --check
python scripts/analysis/recall_by_depth_112.py --check
```
