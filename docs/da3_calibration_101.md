# Depth Anything 3 calibrated against GSV depth, and carried to the Mapillary splits (#101)

**Status: complete, 2026-09-27.** DA3METRIC-LARGE was run on every verdict-reviewed panorama of
the eleven reviewed benchmark bundles (1,289 panos; `manual_gold` is out of scope), sampled at
every GT point and every committed 0.55 RampNet detection (5,403 points), and a ground plane was
fitted to each pano's road band. On the four GSV splits with Google depth (#112) DA3 was compared
with Google's range at the same points and with Google's camera height per pano; the fitted
calibration was then applied to the six Mapillary splits. Everything below re-derives on CPU from
committed rows (`scripts/analysis/da3_calibration_101.py --check`). Implemented by an Opus 5.5
agent from the plan in [#101](https://github.com/ProjectSidewalk/RampNet/issues/101#issuecomment-5858165792).

## Summary

1. **DA3 reads range ~10.6% longer than Google's depth, and the difference is a scale.** At 1,188
   GSV locations, DA3/Google has median **1.106** (pano-clustered 95% CI 1.094–1.117); the
   log-log exponent is 0.981 (CI 0.959–1.001), and the ratio is 1.05–1.12 in every range bucket
   from 0 to 25 m+. Per split it is 1.03 (bend) to 1.18 (sao_paulo) (§2).
2. **The per-city part of that scale looks like Google's, not DA3's.** The labeler's
   bearing-only triangulation found Google's depth frame running short of the imagery's own
   geometry by 1.06 / 1.08 / 1.095 / 1.16 (bend / paterson / gainesville / sao_paulo). Divided by
   those factors, DA3/Google becomes **0.975 / 1.027 / 0.995 / 1.018**. Two instruments that
   share nothing agree on which cities Google reads short, and by how much (§2.2). The labeler's
   factors are themselves approximate, so this is agreement, not proof.
3. **As a distance axis, calibrated DA3 beats flat ground on GSV, except where flat ground is
   already right.** With the scale fitted on three GSV splits and tested on the fourth, DA3's
   range at the point is within 10% of Google's at **64%** of locations against **44%** for flat
   ground at 2.5 m (median |ln ratio| 0.070 vs 0.118, same 1,031 locations). It wins by a wide
   margin on paterson and gainesville (Google's 2025–26 rig at ~1.8 m), ties on sao_paulo, and
   loses on bend, whose cameras sit near the assumed 2.4–2.5 m (§4).
4. **DA3's fitted camera height separates rigs but not panoramas within a rig.** Pooled over
   GSV, DA3 height / Google height is 1.038 (CI 1.026–1.056), r = 0.64; but within Google's
   2025–26 rig r = 0.08 and within the older US vintages r = −0.13 (§3). So the per-pano height
   that #112 reads off Google's payload is not something DA3 can supply per pano; per rig class
   it can.
5. **Carried to Mapillary, the far field shrinks.** On the calibrated DA3 axis the published
   "18 m / 25 m" thresholds land at **14.5 m / 19.1 m** pooled over the six Mapillary splits
   (per split 11.9–15.9 m / 16.1–22.2 m), and recall past 18 m is lower than the flat axis
   shows: 0.470 at 18–25 m and 0.121 at 25–40 m (n = 181, 58), against 0.657 and 0.547 on the flat
   axis (§5.2). The flat axis puts near ramps into the far buckets on these rigs, which
   inflates far-field recall there.
6. **Mapillary camera heights by rig (calibrated DA3):** NCTech iStar Pulsar 2.47 m (richmond),
   Trimble MX7 2.12 m (annapolis), GoPro Fusion 2.02 m (clovis), GoPro Max 1.82–2.13 m by city,
   all below the 2.5 m this repo assumes and the 2.6 m the labeler assumes (§5.1).
7. **Laurens: DA3 and the labeler's instrument disagree.** DA3 puts the Laurens GoPro Max at
   2.13 m; the labeler's scale identity says 2.95 m (CI 2.90–3.00) and its bearing fixed point
   4.31 m, and the labeler's own gate rejects that group as not identifiable. Unresolved (§6).
8. **The published DA3 figures now re-derive.** `detection_recall_analysis.md`'s "agree to
   within 6.5–8.5%, Spearman 0.95 Bend / 0.81 Richmond" reproduces from the committed rows
   (1.065 / 1.086, ρ 0.953 / 0.813), and it turns out to compare DA3's *planar* depth with flat
   *horizontal* range; on a like-for-like horizontal range the agreement is 1.4% / 3.7% (§7).

Every number in §2–§4 depends on Google's depth, which reaches this repo only as the committed
per-point rows of `analysis_out/recall_by_depth_112.json`; the depth payloads themselves are an
unpublished input (they live in the sidewalk-auto-labeler run archive). Nothing here needs them to
re-derive.

## 1. Method

**Instrument.** `depth-anything/DA3METRIC-LARGE` (Hugging Face), DA3 code at
`ByteDance-Seed/Depth-Anything-3` commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`, loaded
from its `src/` as `scripts/analysis/README.md` §"Depth Anything 3 setup" describes. Each pano is
downscaled to 4096 px wide and rendered into the six perspective views of
`equirect_tiling.default_views()` (90° FOV, pitch −30°, 1024 px, yaws 0–300° every 60°), the
same rendering `depth_extract_da3.py` used. DA3 runs once per view with the exact intrinsics
(focal 512 px).

- **Intrinsics make no difference to this model.** On the first pano of each split, DA3's output
  with the intrinsics divided by its output without them is exactly 1.000 (11 of 11). So the value
  recorded is the network's raw output, and the `× focal / 300` question in DA3's README does not
  arise: whatever DA3's absolute scale is, §2 measures it against Google.
- **Points.** Every GT point (`build_ground_truth` over `verdicts.json`) and every committed
  detection of `records.jsonl` is sampled in the view where it is most central (7×7 median at
  DA3's 504×504 output), as `depth_extract_da3.py` did. The value is read as planar (z) depth:
  ray distance = z·|f + a·r + b·u| for the view's basis, and horizontal range = ray·cos(latitude),
  the same definition as #112's `depth_range`. Hits use `recall_by_depth_112.match` (greedy,
  confidence-ordered, radius 0.022). Rows join #112's on exact (split, pano, x, y).
- **z-depth, not ray depth.** Fitting the road plane under both readings, the z reading gives the
  flatter road (median inlier share 0.517 vs 0.440, residual 0.035 m vs 0.043 m over 1,289 panos),
  and on a synthetic scene the ray reading bends a level road (`tests/test_da3_calibration_101.py`).
- **Ground plane and camera height.** From each view, every 4th DA3 output pixel whose ray lies
  20–45° below the pano horizon, within ±30° of the view's yaw (so the six views partition the
  ring). Sequential RANSAC (fixed seed per pano, inliers within 0.10 m, normal within 20° of
  vertical, least-squares refit) finds up to three planes that each hold ≥ 15% of the band; the
  ground is the **lowest** of them (§8 says why). Height = the plane's distance from the camera,
  the same quantity as Google's ground-plane distance. A fit counts when its plane holds ≥ 25% of
  ≥ 200 band points: 1,043 of 1,289 panos. 20–60° is kept as a sensitivity band; the upper
  limit is 45° because the Laurens GoPro Max's capture vehicle is visible from ~49° down
  (labeler `runs/laurens/rig_labels.json`).
- **Candidate distance axes** at a point: `flat_2p5` (the doc's 2.5 m / tan(depression));
  `da3_point` (DA3 range at the point / k_point); `da3_plane` (the ray to the fitted plane, height
  / k_height, tilt kept); `flat_da3_height` (flat ground at the pano's height / k_height).
  k_point = 1.1059 and k_height = 1.038 are the pooled GSV medians of DA3/Google (§2, §3).
- **Statistics** are pure Python (medians, OLS, 1,000-replicate pano-clustered bootstrap, seed
  101), so every table is identical on every numpy build. Floats in committed rows are rounded to
  4 decimals; files are LF; `analysis_out/da3_calibration_101/SHA256SUMS` hashes every one plus the
  #112 JSON they join to.

## 2. DA3 range vs Google range at the same points

Unique GSV locations (a true-positive detection and its GT point count once) on measured-ground
panos where Google's range comes from the plane under the pixel (`pixel_plane`):

| group | locations (panos) | DA3/Google median [95% CI] | p10–p90 | OLS slope [CI] | intercept (m) | log-log exponent [CI] | median abs ln ratio |
|---|---:|---|---|---|---:|---|---:|
| bend | 266 (83) | 1.033 [1.010, 1.058] | 0.94–1.16 | 1.047 [1.007, 1.090] | 0.06 | 1.025 [0.998, 1.048] | 0.065 |
| paterson | 372 (92) | 1.109 [1.096, 1.131] | 1.01–1.25 | 1.078 [1.041, 1.122] | 0.64 | 1.003 [0.978, 1.029] | 0.104 |
| gainesville | 264 (92) | 1.089 [1.075, 1.106] | 0.97–1.32 | 0.938 [0.872, 1.008] | 2.07 | 0.880 [0.828, 0.935] | 0.089 |
| sao_paulo | 286 (88) | 1.180 [1.163, 1.199] | 1.04–1.36 | 1.173 [1.108, 1.238] | 0.29 | 1.013 [0.977, 1.050] | 0.166 |
| 2025-26 rig | 385 (116) | 1.111 [1.094, 1.131] | 1.00–1.27 | 1.009 [0.980, 1.039] | 1.39 | 0.919 [0.881, 0.954] | 0.108 |
| older US vintages | 517 (151) | 1.066 [1.050, 1.082] | 0.95–1.22 | 1.052 [0.999, 1.109] | 0.43 | 1.019 [0.983, 1.049] | 0.078 |
| gsv_pooled | 1188 (355) | 1.106 [1.094, 1.117] | 0.98–1.29 | 1.058 [1.027, 1.091] | 0.82 | 0.981 [0.959, 1.001] | 0.103 |

"2025-26 rig" is paterson 2025 + gainesville 2026 capture, as #112 grouped it; sao_paulo is its
own group.

### 2.1 By range

| Google range | n | DA3/Google median (p10–p90) | flat 2.5 m / Google median |
|---|---:|---|---:|
| 0-8 m | 297 | 1.098 (0.97–1.32) | 1.134 |
| 8-12 m | 277 | 1.121 (0.98–1.30) | 1.147 |
| 12-18 m | 311 | 1.122 (0.99–1.30) | 1.126 |
| 18-25 m | 215 | 1.096 (0.99–1.22) | 1.100 |
| 25 m+ | 88 | 1.054 (0.95–1.20) | 1.230 |

DA3's ratio is flat with range, which is what makes it a one-constant calibration. The exception
is gainesville, whose exponent is 0.88 (CI 0.83–0.94): DA3 over-reads near points more than far
ones there. gainesville is mostly Google's 2026 rig with a ~1.8 m camera, and that is what pulls
the 2025–26 row's exponent to 0.92.

### 2.2 Against the labeler's corrected depth frame

`recall_by_depth_112.py` carries the labeler's per-city estimate of how far Google's depth frame
runs short of the imagery's own geometry (`DEPTH_FRAME_SCALE`, from bearing-only triangulation
in the labeler's camera-height study; approximate, and whether it is a scale or an offset is open
there):

| split | locations | labeler scale | DA3/Google | DA3/(Google x scale) |
|---|---:|---:|---:|---:|
| bend | 266 | 1.06 | 1.033 | 0.975 |
| paterson | 372 | 1.08 | 1.109 | 1.027 |
| gainesville | 264 | 1.095 | 1.089 | 0.995 |
| sao_paulo | 286 | 1.16 | 1.180 | 1.018 |

The per-city spread of DA3/Google (1.03–1.18) collapses to 0.975–1.027 once Google is corrected
by the labeler's factors. DA3 is a monocular model and the labeler's instrument is multi-view
geometry, and neither uses Google's depth. That both find sao_paulo's Google depth the shortest and
bend's the least short is the strongest evidence here that **the per-city part of the DA3/Google
ratio belongs to Google's frame, and DA3 behaves as a single-scale instrument across these four
cities.**

**What that means for the calibration.** k_point = 1.106 maps DA3 onto *Google's* frame, because
that is the reference the plan chose and the one #112 reports in. If the labeler's factors are
right, raw DA3 is closer to the imagery's own geometry than calibrated DA3 is, by roughly 6–16%
depending on city. The Mapillary tables in §5 are on the Google-calibrated axis so they compare
directly with #112's GSV tables; multiply a calibrated distance by 1.106 to get raw DA3.

## 3. DA3 camera height vs Google camera height

Per pano, measured Google ground, DA3 fit passing:

| group | panos | DA3 h median | Google h median | DA3/Google median [CI] | p10–p90 | Pearson r | OLS slope | median abs ln ratio |
|---|---:|---:|---:|---|---|---:|---:|---:|
| bend | 78 | 2.29 | 2.38 | 0.959 [0.949, 0.977] | 0.91–1.06 | 0.135 | 0.112 | 0.053 |
| paterson | 97 | 2.15 | 2.05 | 1.032 [1.022, 1.061] | 0.93–1.18 | 0.749 | 0.866 | 0.061 |
| gainesville | 95 | 1.95 | 1.83 | 1.054 [1.026, 1.086] | 0.96–1.39 | 0.431 | 0.269 | 0.068 |
| sao_paulo | 82 | 2.55 | 2.26 | 1.109 [1.079, 1.132] | 0.95–1.25 | 0.479 | 0.654 | 0.118 |
| 2025-26 rig | 119 | 1.92 | 1.83 | 1.060 [1.036, 1.077] | 0.96–1.32 | 0.081 | 0.051 | 0.068 |
| older US vintages | 151 | 2.31 | 2.36 | 0.995 [0.976, 1.010] | 0.91–1.14 | -0.132 | -0.121 | 0.053 |
| gsv_pooled | 352 | 2.24 | 2.22 | 1.038 [1.026, 1.056] | 0.93–1.22 | 0.635 | 0.599 | 0.070 |

DA3 recovers the **rig-level** difference Google's payload shows (1.92 vs 2.31 m, against Google's
1.83 vs 2.36 m) and places a typical pano's height within ~7% (median |ln ratio| 0.070). What it
does not do is track panos *within* a rig: r is 0.08 and −0.13 inside the two US groups, so the
pooled r = 0.64 comes from the difference between groups. paterson's r = 0.75 is the same effect
inside one split, which mixes both rigs. **Read a DA3 height as a rig-class estimate, not a
per-pano measurement.**

Three height readings, same panos:

| split | reading | panos | DA3/Google median | median abs ln ratio | Pearson r |
|---|---|---:|---:|---:|---:|
| gsv_pooled | lowest_plane | 352 | 1.038 | 0.070 | 0.635 |
| gsv_pooled | dominant_plane | 352 | 1.027 | 0.071 | 0.639 |
| gsv_pooled | gt_points | 323 | 1.091 | 0.114 | 0.555 |

(Per split in `analysis_out/da3_calibration_101/tables.md`.) On GSV the lowest and the dominant
plane are almost always the same plane (Google's own vehicle is not in the imagery), so the §8
change matters only on consumer rigs. `gt_points` is the median of DA3 ray × sin(depression) over
a pano's GT points: it never looks at the nadir, but it reads ~5% higher than the plane and is
noisier, because ramps sit at or just above road level, often on a sidewalk.

## 4. Which axis reproduces Google's range: leave-one-split-out

For each GSV split, k_point and k_height are fitted on the other three and the axes are scored
against Google's range at that split's locations. `pooled_common` is the 1,031 locations where
every axis has a value (the plane axes need a passing fit):

| population | axis | n | median axis/Google (p10–p90) | median abs ln ratio | share within 10% |
|---|---|---:|---|---:|---:|
| pooled_common | flat_2p5 | 1031 | 1.125 (1.00–1.61) | 0.118 | 0.444 |
| pooled_common | da3_point | 1031 | 0.997 (0.87–1.17) | 0.070 | 0.640 |
| pooled_common | da3_plane | 1031 | 1.026 (0.80–1.36) | 0.119 | 0.433 |
| pooled_common | flat_da3_height | 1031 | 1.004 (0.86–1.25) | 0.097 | 0.517 |
| bend | flat_2p5 | 266 | 1.062 (1.00–1.22) | 0.060 | 0.669 |
| bend | da3_point | 266 | 0.919 (0.83–1.04) | 0.092 | 0.538 |
| paterson | flat_2p5 | 372 | 1.166 (1.00–1.56) | 0.154 | 0.360 |
| paterson | da3_point | 372 | 1.007 (0.92–1.13) | 0.052 | 0.761 |
| gainesville | flat_2p5 | 264 | 1.366 (1.05–1.93) | 0.312 | 0.151 |
| gainesville | da3_point | 264 | 0.982 (0.88–1.19) | 0.057 | 0.678 |
| sao_paulo | flat_2p5 | 286 | 1.082 (1.00–1.31) | 0.079 | 0.556 |
| sao_paulo | da3_point | 286 | 1.085 (0.95–1.25) | 0.088 | 0.549 |

(The plane axes per split are in `tables.md`.) The script picks the headline axis by rule, the
DA3 axis with the lowest pooled_common median |ln ratio|: **`da3_point`**. The two plane-based
axes are worse because DA3's per-pano height error (§3) enters every point on the pano, and a
tilt error of a degree moves a near-horizon point a long way; reading DA3 at the point itself
avoids both.

On bend, where Google's camera is ~2.4 m and the flat axis is already close, DA3 loses: the
held-out constant (fitted on the other three splits, 1.125) over-corrects bend, whose own ratio is
1.03. That is the per-city spread of §2.2 again. On the evidence there, it is Google's frame
that varies, so part of this loss is the reference moving, not DA3.

## 5. Carried to Mapillary

### 5.1 Camera height by split and by rig

Calibrated DA3 height (lowest plane / k_height):

| split | panos | fit ok | median h (p25–p75) | min–max | median tilt (deg) | Google h median |
|---|---:|---:|---|---|---:|---:|
| bend | 110 | 94 | 2.21 (2.13–2.29) | 1.92–2.92 | 2.1 | 2.38 |
| paterson | 125 | 108 | 2.11 (1.90–2.39) | 1.38–2.87 | 2.2 | 2.05 |
| gainesville | 125 | 106 | 1.89 (1.79–2.10) | 1.60–2.79 | 2.0 | 1.83 |
| sao_paulo | 125 | 95 | 2.46 (2.27–2.54) | 1.34–2.78 | 2.6 | 2.26 |
| laurens_gsv | 86 | 78 | 2.14 (2.09–2.19) | 1.82–2.59 | 2.1 | – |
| richmond | 124 | 93 | 2.38 (2.06–2.50) | 1.33–2.78 | 3.1 | – |
| annapolis | 125 | 96 | 2.12 (2.05–2.21) | 1.89–2.41 | 3.6 | – |
| morgantown | 125 | 92 | 2.07 (1.94–2.20) | 0.52–2.63 | 3.4 | – |
| clovis | 125 | 112 | 2.02 (1.94–2.11) | 1.70–2.62 | 4.3 | – |
| laurens_mapillary | 94 | 73 | 2.13 (2.01–2.26) | 1.85–2.66 | 4.5 | – |
| budapest_district5 | 125 | 96 | 1.88 (1.78–1.97) | 0.60–2.47 | 7.3 | – |

| split / rig | panos | median h (p25–p75) | min–max | median tilt (deg) |
|---|---:|---|---|---:|
| richmond / gopro/max | 23 | 1.82 (1.73–1.98) | 1.33–2.10 | 8.2 |
| richmond / nctech ltd/istar pulsar | 58 | 2.47 (2.38–2.58) | 2.10–2.78 | 2.1 |
| richmond / unknown | 12 | 2.34 (2.19–2.40) | 1.90–2.50 | 3.9 |
| annapolis / trimble/mx7 | 96 | 2.12 (2.05–2.21) | 1.89–2.41 | 3.6 |
| morgantown / gopro/max | 92 | 2.07 (1.94–2.20) | 0.52–2.63 | 3.4 |
| clovis / gopro/fusion | 112 | 2.02 (1.94–2.11) | 1.70–2.62 | 4.3 |
| laurens_mapillary / gopro/max | 73 | 2.13 (2.01–2.26) | 1.85–2.66 | 4.5 |
| budapest_district5 / gopro/max | 95 | 1.88 (1.78–1.97) | 0.60–2.47 | 7.3 |
| all Mapillary / gopro/max | 283 | 2.00 (1.85–2.16) | 0.52–2.66 | 4.8 |

Rig = camera make/model from the bundle records, normalized as the labeler's `rig_key` does
(one budapest pano on an LG R105 is left out of this table; it is in `tables.md`).

Caveats that belong to these numbers:

- **The calibration was fitted on Google's car rigs.** Nothing in this repo validates DA3 on a
  GoPro or an NCTech image directly. The heights are plausible (every Mapillary rig median lands
  in 1.8–2.5 m, and DA3 reads the same Laurens footprint at 2.14 m on Google's rig and 2.13 m
  on the GoPro Max), but "plausible" is not "measured".
- **Per-pano heights are rig-class estimates** (§3). The spread inside a rig here mixes real
  mounting differences with DA3's per-pano noise, and this measurement cannot separate them.
- **Tilt is larger on the consumer rigs** (median 3.4–8.2° against ~2° on GSV). That is the
  unleveled-rig effect the flat axis cannot see, and why a point near the horizon is where the
  flat axis goes most wrong.
- **4 of 1,043 fitted panos still read below 1.2 m** (morgantown 0.52, 0.68, 0.84; budapest
  0.60): the roof-selection failure of §8 not fully removed. They are left in, not filtered.

### 5.2 Recall by distance on the calibrated axis

Fn-confirmed GT points, hits at the deployed 0.55 point, pooled over the six Mapillary splits
(1,615 GT points; per split in `tables.md`):

| bucket | n (flat 2.5 m) | recall (flat 2.5 m) | n (DA3 point) | recall (DA3 point) | n (DA3 plane) | recall (DA3 plane) | n (flat at DA3 height) | recall (flat at DA3 height) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-8 m | 410 | 0.680 | 553 | 0.667 | 395 | 0.666 | 430 | 0.646 |
| 8-12 m | 440 | 0.718 | 438 | 0.712 | 344 | 0.683 | 353 | 0.691 |
| 12-18 m | 316 | 0.627 | 385 | 0.691 | 312 | 0.699 | 256 | 0.707 |
| 18-25 m | 239 | 0.657 | 181 | 0.470 | 146 | 0.541 | 110 | 0.509 |
| 25-40 m | 128 | 0.547 | 58 | 0.121 | 57 | 0.386 | 82 | 0.549 |
| 40 m+ | 65 | 0.231 |  |  | 23 | 0.000 | 35 | 0.257 |
| all | 1598 | 0.648 | 1615 | 0.643 | 1277 | 0.640 | 1266 | 0.642 |

Where the published thresholds land (median flat/axis ratio of the points within ±20% of the
threshold on the flat axis, window n in brackets, as #112 computed it):

| population | DA3 point: 18 m / 25 m become | DA3 plane | flat at DA3 height |
|---|---|---|---|
| mapillary_pooled | 14.5 m (461) / 19.1 m (240) | 15.5 m (369) / 20.9 m (194) | 15.1 m (369) / 21.2 m (194) |
| richmond | 15.4 m (116) / 21.4 m (69) | 16.0 m (78) / 22.0 m (49) | 17.6 m (78) / 24.5 m (49) |
| annapolis | 15.1 m (99) / 17.2 m (45) | 15.9 m (79) / 19.3 m (37) | 15.3 m (79) / 21.2 m (37) |
| morgantown | 14.7 m (70) / 18.9 m (35) | 15.9 m (55) / 20.9 m (31) | 15.5 m (55) / 21.3 m (31) |
| clovis | 13.1 m (68) / 17.6 m (28) | 14.8 m (64) / 18.8 m (28) | 14.6 m (64) / 20.0 m (28) |
| laurens_mapillary | 15.9 m (40) / 22.2 m (24) | 16.1 m (35) / 23.8 m (21) | 16.4 m (35) / 23.6 m (21) |
| budapest_district5 | 11.9 m (68) / 16.1 m (39) | 13.1 m (58) / 17.9 m (28) | 13.5 m (58) / 19.0 m (28) |
| laurens_gsv (GSV, no Google depth) | 16.0 m (72) / 22.0 m (41) | 15.3 m (65) / 21.3 m (37) | 15.5 m (65) / 21.6 m (37) |

What this says, with its limits:

- **On these rigs the flat axis overstates the far field.** 128 GT points are 25–40 m out on the
  flat axis and only 58 on the DA3 axis; recall there is 0.547 on the flat axis and 0.121 on
  DA3's. The flat axis is putting nearby ramps, which RampNet mostly finds, into the far buckets,
  so "blind past 25 m" is if anything *understated* on Mapillary, and it starts nearer: recall on
  the DA3 axis is 0.69 at 12–18 m and 0.47 at 18–25 m.
- **The far buckets are small.** 58 points at 25–40 m on the DA3 axis, pooled over six cities;
  per split they are 2–22. Read the cliff, not the third decimal.
- **The DA3 axis carries the §2.2 question.** It is in Google's frame. In the labeler's corrected
  frame every DA3 distance here would be ~6–16% longer, and the thresholds would move up by the
  same factor.
- The flat and DA3 plane / height columns have fewer points than the DA3 point column: flat drops
  the 17 GT points at or above the horizon, and the plane axes need a passing fit (1,277 and
  1,266 of 1,615).

## 6. Laurens cross-read against the labeler's height instrument

Read-only from `sidewalk-auto-labeler` at commit `29dc605bd64bc7b50b78ec25bfdaa45dcfc6169d`
(`runs/laurens/camera_height/groups.csv` sha256 and `runs/laurens/camera_heights.json` sha256 are
in `tables.json` → `labeler_laurens.files`). The labeler's Laurens run covers 4,495 panos; the
benchmark's 94 laurens_mapillary panos are a subset of the same sequences.

- **Rig level (GoPro Max):** DA3 2.13 m (73 panos). The labeler: scale identity (instrument B)
  **2.95 m** (CI 2.90–3.00), bearing fixed point (A) **4.31 m** (CI 3.80–4.84), slope 0.72. The
  labeler's decision rule rejects the group ("fails identifiable (slope 0.723); agreement (A
  4.314, B 2.949); SUSPECT: bearing height outside 1.0-3.5 m"), so its table keeps the 2.6 m
  default and `applied: false`.
- **Sequence level:** 16 sequences have both a DA3 median and a labeler h_scale; the correlation
  is r = −0.28. DA3's sequence medians span 1.95–2.36 m, the labeler's 2.78–3.19 m. With 1–11
  benchmark panos per sequence, and DA3's per-pano spread (§3), a correlation at this grain was
  not expected to be resolvable either way.
- **The same footprint on Google's rig** reads 2.14 m in DA3 (laurens_gsv, 78 panos; Google depth
  was never harvested for laurens_gsv, so there is no Google height to check it against).

The two instruments disagree by ~0.8 m on the GoPro Max, and the labeler already distrusts its
own number. DA3's number agrees with the other GoPro Max cities (1.82–2.07 m) and with the car-roof
mounting the rig labels show. **This is recorded as a disagreement, not resolved.** A direct test
would be a Laurens pano pair with a known ground distance (for example, a mapped curb-ramp pair
on both arms).

## 7. The published DA3 figures, re-derived

`detection_recall_analysis.md` reported DA3 and flat-ground geometry agreeing "to within
6.5–8.5% (Spearman ρ = 0.95 Bend / 0.81 Richmond)" from `gt_depth_da3.json`, which was never
committed. From this run's rows:

| split | n | flat/DA3 raw value, median | Spearman | flat/DA3 horizontal range, median | Spearman | GT at/above horizon with DA3 |
|---|---:|---:|---:|---:|---:|---:|
| bend | 327 | 1.065 | 0.953 | 1.014 | 0.961 | 0 |
| richmond | 307 | 1.086 | 0.813 | 1.037 | 0.847 | 3 |

The published numbers reproduce to the digit, which pins down what they measured: DA3's raw
planar depth against the flat *horizontal* range. Those are different quantities (a point off the
view's axis has planar depth shorter than its ray), and on a like-for-like horizontal range the
agreement is closer, 1.4% on bend and 3.7% on richmond. The doc counted 4 Richmond GT ramps above
the horizon rescued by DA3; this run finds 3 GT points at or above the horizon, all with a DA3
value. The difference was not chased; `gt_depth_da3.json` does not exist to compare.

## 8. The ground-fit change, and what the first run said

The first full run (job `40774944`, committed at `4ab47a2`) took the ground as the single dominant
plane of the band. On car-roof consumer rigs that plane is often the car's own roof, about 0.75 m
under the camera: morgantown (GoPro Max) read heights down to 0.53 m and only 23 of 125 panos
passed the fit, and richmond's GoPro Max panos read 1.50 m. The fit was changed to take the
lowest of up to three well-supported planes and the extraction re-run (job `40777019`). **This is a
change made after seeing the data**, so both readings are reported:

| split | fit ok, first run | median h, first run | min h, first run | fit ok, now | median h, now | min h, now |
|---|---:|---:|---:|---:|---:|---:|
| bend | 106 | 2.23 | 1.88 | 94 | 2.21 | 1.92 |
| paterson | 95 | 2.11 | 1.40 | 108 | 2.11 | 1.38 |
| gainesville | 116 | 1.88 | 1.44 | 106 | 1.89 | 1.60 |
| sao_paulo | 86 | 2.47 | 1.17 | 95 | 2.46 | 1.34 |
| laurens_gsv | 81 | 2.18 | 1.87 | 78 | 2.14 | 1.82 |
| richmond | 99 | 2.32 | 0.68 | 93 | 2.38 | 1.33 |
| annapolis | 96 | 2.14 | 1.91 | 96 | 2.12 | 1.89 |
| morgantown | 23 | 1.92 | 0.53 | 92 | 2.07 | 0.52 |
| clovis | 48 | 2.03 | 1.73 | 112 | 2.02 | 1.70 |
| laurens_mapillary | 62 | 2.11 | 1.64 | 73 | 2.13 | 1.85 |
| budapest_district5 | 105 | 1.85 | 0.61 | 96 | 1.88 | 0.60 |

(First run: pass rule inlier share ≥ 0.5 of the band on the single plane, k_height 1.022; now:
≥ 0.25 on the selected plane, k_height 1.038. The first-run table re-derives by checking out
`4ab47a2` and running `--check --markdown`.) The GSV splits barely move, which is the expected
signature of a fix aimed at a vehicle roof Google's imagery does not contain. The point
calibration (§2) does not use the plane at all. It is not bit-identical between the runs, because
the two ran on different GPUs (A40, then Quadro RTX 6000) and DA3's output differs slightly: over
5,402 points the run-to-run difference in the DA3 value has median 0.06%, p99 0.35%, max 0.82%.
Pooled DA3/Google moved from 1.105 to 1.106. §4's plane-based axes and §5.1 moved; the headline
axis did not.

## 9. Cost

Three jobs on klone `ckpt-all` (free, preemptible; none was preempted), 1.94 GPU-hours, $0.
The `sacct` dump is committed at `docs/data/compute/sacct_klone_2026-09-27_da3_101.txt` and
parsed into `analysis_out/compute_log.jsonl` (see [`compute_cost.md`](compute_cost.md)). The
script's own `paid: false` rows are in `analysis_out/usage_log.jsonl`:

| job | what | GPU | elapsed (sacct) | extract loop | s / pano |
|---|---|---|---:|---:|---:|
| `40774578` | smoke test, 2 bend + 2 richmond panos | A40 | 159 s | 20.2 s | 5.05 (includes warm-up) |
| `40774944` | first full run, 1,289 panos | A40 | 2,625 s | 2,502.6 s | 1.94 |
| `40777019` | re-run with the lowest-plane fit | Quadro RTX 6000 | 4,208 s | 4,075.2 s | 3.16 |

The re-run is slower for two reasons that were not separated: a slower GPU, and three sequential
RANSAC fits per band instead of one (the fits run on the CPU inside the loop). Model load was
12–21 s per job (20.8, 11.6, 13.2 s). The derive step (CPU, laptop) takes about 10 s; `--check` about the same.

## 10. How to reproduce

Inputs: the committed bundles (`benchmark/<split>/{records.jsonl,verdicts.json,imagery_manifest.json}`),
the committed #112 rows (`analysis_out/recall_by_depth_112.json`), the benchmark panos (Hugging
Face `projectsidewalk/rampnet-benchmark`, sha256-checked against each manifest), DA3 weights
`depth-anything/DA3METRIC-LARGE` and code commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.
Unpublished: Google's depth payloads (only their committed per-point rows are needed) and the
labeler's Laurens tables (needed only by `derive`; their values are copied into `tables.json`).

```bash
# 0. environment on klone (paths are Jon's; any CUDA box works)
ROOT=/gscratch/scrubbed/jfroehli/da3_101
git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git $ROOT/src/Depth-Anything-3
mkdir -p $ROOT/src/Depth-Anything-3/stubs/moviepy && touch $ROOT/src/Depth-Anything-3/stubs/moviepy/{__init__,editor}.py
python -m venv --system-site-packages $ROOT/env     # over a sidewalkcv2 env (torch 2.6, numpy 2.2)
$ROOT/env/bin/pip install omegaconf einops addict opencv-python-headless plyfile pycolmap trimesh evo
git clone --branch <this branch or main> https://github.com/ProjectSidewalk/RampNet.git $ROOT/repo

# 1. GPU: DA3 at every point + ground fits -> $ROOT/out/raw/<split>.jsonl (~45-70 min, one GPU)
cd $ROOT/repo && OUT=$ROOT/out/raw sbatch -A ckpt-makelab -p ckpt-all scripts/analysis/da3_calibration_101.slurm
#    copy $ROOT/out/raw/<split>.jsonl to analysis_out/da3_calibration_101/raw/

# 2. CPU: rows, tables, markdown, SHA256SUMS (reads the labeler's Laurens tables read-only)
python scripts/analysis/da3_calibration_101.py derive --labeler-root /path/to/sidewalk-auto-labeler

# 3. CPU, from a clean clone, no GPU, no panos, no payloads: everything re-derives
python scripts/analysis/da3_calibration_101.py --check
python scripts/analysis/da3_calibration_101.py --check --markdown   # every table, per split
```

`tests/test_da3_calibration_101.py` runs the `--check` path, pins the headline numbers of this
document, and checks the geometry on synthetic scenes (level and tilted ground, a car-sized
obstacle, a vehicle roof over a road).

## 11. Gaps, stated

- **Not validated on Mapillary imagery.** Mapillary serves no depth, so the Mapillary numbers rest
  on the GSV calibration transferring to other cameras. The one independent instrument available
  (§6) disagrees, and distrusts itself.
- **Google is the reference, and Google is not ground truth** (§2.2). The calibrated axis is in
  Google's frame by design.
- **laurens_gsv** has DA3 but no Google depth (never harvested), so it is carried like a Mapillary
  split, not validated.
- **Per-pano heights and tilts are not trustworthy per pano** (§3); the tilt DA3 fits was not
  compared with Google's plane tilt, which #112 has for every measured pano. That comparison is a
  cheap follow-up from the committed rows.
- **Apparent-size tables and the resolution forecast** of #112 were not re-issued on the DA3
  axis; only recall by distance was, as the plan specified.
- **`manual_gold`** is out of scope (no verdict review, no depth, 1,000 panos).
- **One model.** RANSAC is seeded per pano. DA3's output is not bit-identical across GPUs (§8:
  median 0.06%, max 0.82% between an A40 and a Quadro RTX 6000), far below every effect reported
  here. No other monocular depth model was tried.

🤖 Generated with [Claude Code](https://claude.com/claude-code) — Opus 5.5, claude-opus-5-5; plan by Fable 5.1, claude-fable-5-1
