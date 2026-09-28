# Depth Anything 3 calibrated against GSV depth, and carried to the Mapillary splits (#101)

**Status: complete, 2026-09-27; revised the same day after the review on
[#203](https://github.com/ProjectSidewalk/RampNet/pull/203#issuecomment-5859733995). Held-out check on
laurens_gsv added 2026-09-28 (§5.3).** DA3METRIC-LARGE was
run on every verdict-reviewed panorama of the eleven reviewed benchmark bundles (1,289 panos;
`manual_gold` is out of scope), sampled at every GT point and every committed 0.55 RampNet detection
(5,403 points), and a ground plane was fitted to each pano's road band. On the four GSV splits with
Google depth (#112) DA3 was compared with Google's range at the same points and with Google's camera
height per pano; the fitted calibration was then applied to the six Mapillary splits. Everything
below re-derives on CPU from committed rows (`scripts/analysis/da3_calibration_101.py --check`);
every table not printed here is in `analysis_out/da3_calibration_101/tables.md`. Implemented by an
Opus 5.5 agent from the plan in
[#101](https://github.com/ProjectSidewalk/RampNet/issues/101#issuecomment-5858165792).

## Summary

1. **DA3 reads range ~10.6% longer than Google's depth. Pooled, the difference is a scale; on
   Google's 2025–26 rig it is not.** At 1,188 GSV locations, DA3/Google has median **1.106**
   (pano-clustered 95% CI 1.094–1.117) and a log-log exponent of 0.981 (CI 0.959–1.001). On the
   2025–26 rig (paterson 2025 + gainesville 2026, a ~1.8 m camera) the exponent is **0.919**
   (CI 0.881–0.954) and the ratio falls with range, 1.144 at 0–8 m to 1.047 at 25 m+. That rig is
   in the same camera-height class DA3 reads on the Mapillary rigs (1.8–2.1 m), so a single
   constant is least safe exactly where it is carried (§2).
2. **The per-city part of the ratio lines up with the labeler's estimate of Google's own frame
   error, on four cities.** Divided by the labeler's per-city depth-frame factors (1.06 / 1.08 /
   1.095 / 1.16, from bearing-only triangulation, which the labeler calls approximate),
   DA3/Google becomes **0.975 / 1.027 / 0.995 / 1.018** (bend / paterson / gainesville /
   sao_paulo). Both put sao_paulo shortest and bend least short; paterson and gainesville swap
   rank. Four cities is agreement worth noting, not proof (§2.2).
3. **As a distance axis, calibrated DA3 beats flat ground on GSV, except where flat ground is
   already right.** With the scale fitted on three GSV splits and tested on the fourth, DA3's range
   at the point is within 10% of Google's at **64%** of locations against **44%** for flat ground at
   2.5 m (same 1,031 locations). It wins on paterson and gainesville, ties on sao_paulo, and loses on
   bend, whose cameras sit near the assumed height (§4). **The downstream method also checks out on
   GSV:** recall by distance and the threshold mapping on the DA3 axis track Google's axis (18 m /
   25 m become 15.9 / 21.3 m on DA3 against 16.2 / 21.7 m on Google, pooled; per split within
   about ±1–2 m), while the flat axis inflates far-field recall there too (§4.1).
4. **DA3's fitted camera height separates rigs but not panoramas within a rig.** Pooled over GSV,
   DA3 height / Google height is 1.038 (CI 1.026–1.056), r = 0.64; within Google's 2025–26 rig
   r = 0.08, within the older US vintages r = −0.13 (§3).
5. **Carried to Mapillary, the far field shrinks.** On the calibrated DA3 axis the published "18 m
   / 25 m" thresholds land at **14.5 m / 19.1 m** pooled over the six Mapillary splits (11.9–15.9 m
   / 16.1–22.2 m per split, each with the ±1–2 m error scale of §4.1). Recall past 18 m is lower
   than the flat axis shows: 0.470 at 18–25 m and 0.121 at 25–40 m (n = 181, 58), against 0.657
   and 0.547 on the flat axis (§5.2).
6. **Mapillary camera heights by rig (calibrated DA3):** NCTech iStar Pulsar 2.47 m (richmond),
   Trimble MX7 2.12 m (annapolis), GoPro Fusion 2.02 m (clovis), GoPro Max 1.82–2.13 m by city,
   all below the 2.5 m this repo assumes and the 2.6 m the labeler assumes (§5.1).
7. **Laurens: DA3 and the labeler differ, and the labeler has since found its instrument B
   unvalidated.** DA3 puts the Laurens GoPro Max at 2.13 m. The labeler's bearing fixed point reads
   4.31 m and its scale identity at 2.6 m 2.98 m, but sidewalk-auto-labeler#89 found no validated
   instrument-B estimator (both candidates read +0.25 to +0.53 m high at a true 1.8 m under its
   noisier setting) and its gate rejects the group (§6).
8. **The published DA3 figures now re-derive.** Under `depth_analysis.py`'s own filters,
   `detection_recall_analysis.md`'s "agree to within 6.5–8.5%, ρ 0.95 / 0.81, 4 Richmond ramps above
   the horizon" reproduces as 6.5% / 8.5%, ρ 0.953 / 0.812, 3 + 1. It compared DA3's *planar*
   depth with flat *horizontal* range; like for like the agreement is 1.4% / 3.7% (§7).
9. **The ground fit was changed after seeing the data, in two ways**: the lowest supported plane
   instead of the dominant one, and a pass rule of 25% instead of 50%. Both runs are committed, and
   §8 attributes every pano to one change or the other.
10. **Held out, laurens_gsv lands at the low edge of the fitted splits' spread.** laurens_gsv has
    Google depth (#201) but was never in the fit. At its 156 locations (45 panos) DA3/Google is
    **1.015** (CI 0.997–1.046), 0.918× the pooled 1.106. Paired on its 53 panos, DA3's camera height
    / Google's is **0.925** (CI 0.920–0.942), 0.891× the pooled 1.038. The four fitted splits already
    span −6.6% to +6.7% (points) and −7.6% to +6.8% (height) of the pooled values; laurens_gsv
    widens the low end to −8.2% and −10.9%. It is below bend beyond noise in height, not at points.
    The shape holds: the log-log exponent is 1.015 (CI 0.982–1.042). What this means for Mapillary
    depends on an open question. If DA3's scale varies by city, calibrated Mapillary values carry
    that observed spread, consistent with §4.1's ±1–2 m. If Google's frame varies instead, DA3's
    own spread is about ±3% and the Mapillary axis carries a common bias (§2.2). laurens_gsv's
    uncorrected ratios sit inside the fitted splits' frame-corrected bands, which fits the second
    reading. A third reading, a DA3 bias specific to Google's 2024 imagery or rig, would not carry
    to the GoPro Max at all. A Laurens frame factor from the labeler would decide it (§5.3, §11).

Every number in §2–§4 depends on Google's depth, which reaches this repo only as the committed
per-point rows of `analysis_out/recall_by_depth_112.json`; the depth payloads themselves are an
unpublished input (they live in the sidewalk-auto-labeler run archive). Nothing here needs them to
re-derive.

## 1. Method

**Instrument.** `depth-anything/DA3METRIC-LARGE` (Hugging Face, weights revision
`4010e39f3634a45bc60553321fb49fb760bd594e`, the snapshot both runs loaded), DA3 code at
`ByteDance-Seed/Depth-Anything-3` commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`, loaded from
its `src/` as `scripts/analysis/README.md` §"Depth Anything 3 setup" describes. Both are now pinned
in the script (`MODEL_REVISION`, `DA3_CODE_COMMIT`); the two runs predate that pin in code, and the
values are the ones read from the run's HF cache and DA3 clone. Each pano is downscaled to 4096 px
wide and rendered into the six perspective views of `equirect_tiling.default_views()` (90° FOV,
pitch −30°, 1024 px, yaws 0–300° every 60°), the same rendering `depth_extract_da3.py` used. DA3
runs once per view with the exact intrinsics (focal 512 px).

- **Intrinsics make no difference to this model.** On the first pano of each split, DA3's output
  with the intrinsics divided by its output without them is exactly 1.000 (11 of 11). So the value
  recorded is the network's raw output, and §2 measures its scale against Google.
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
  ground is the **lowest** of them. Height = the plane's distance from the camera, the same
  quantity as Google's ground-plane distance. A fit counts when its plane holds ≥ **25%** of ≥ 200
  band points: 1,043 of 1,289 panos. **Both the lowest-plane rule and the 25% pass rule were set
  after the first run** (it used the dominant plane at ≥ 50%); §8 separates their effects. The
  45° upper limit is because the Laurens GoPro Max's capture vehicle is visible from ~49° down
  (labeler `runs/laurens/rig_labels.json`); 20–60° is kept as a sensitivity band (§5.1).
- **Candidate distance axes** at a point: `flat_2p5` (the doc's 2.5 m / tan(depression));
  `da3_point` (DA3 range at the point / k_point); `da3_plane` (the ray to the fitted plane, height
  / k_height, tilt kept); `flat_da3_height` (flat ground at the pano's height / k_height).
  k_point = 1.1059 and k_height = 1.038 are the pooled GSV medians of DA3/Google (§2, §3).
- **Statistics** are pure Python (medians, OLS, 1,000-replicate pano-clustered bootstrap, seed
  101), so every table is identical on every numpy build. Floats in committed rows are rounded to
  4 decimals; files are LF and pinned LF in `.gitattributes`; `analysis_out/da3_calibration_101/SHA256SUMS`
  hashes every one (CRLF folded to LF before hashing) plus the #112 JSON they join to.

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

### 2.1 By range, and by rig

| Google range | pooled n | pooled DA3/Google (p10–p90) | 2025-26 rig n / ratio | older US n / ratio | sao_paulo n / ratio | flat 2.5 m / Google, pooled |
|---|---:|---|---|---|---|---:|
| 0-8 m | 297 | 1.098 (0.97–1.32) | 91 / 1.144 | 122 / 1.028 | 84 / 1.144 | 1.134 |
| 8-12 m | 277 | 1.121 (0.98–1.30) | 101 / 1.135 | 113 / 1.061 | 63 / 1.208 | 1.147 |
| 12-18 m | 311 | 1.122 (0.99–1.30) | 97 / 1.111 | 141 / 1.090 | 73 / 1.198 | 1.126 |
| 18-25 m | 215 | 1.096 (0.99–1.22) | 63 / 1.089 | 99 / 1.079 | 53 / 1.174 | 1.100 |
| 25 m+ | 88 | 1.054 (0.95–1.20) | 33 / 1.047 | 42 / 1.049 | 13 / 1.193 | 1.230 |

Pooled, the ratio is nearly flat with range, and the log-log exponent's CI touches 1. The groups
differ: on the 2025–26 rig the ratio falls steadily with range (1.144 → 1.047; exponent 0.919, CI
excludes 1), on the older US vintages it rises then falls, and on sao_paulo it is high throughout.
So **one pooled constant is a fair summary of the older rigs and a biased one for the 2025–26 rig**:
divided by k_point = 1.106, that rig's near points (0–8 m) come out ~3% longer than Google's range
and its far points (25 m+) ~5% shorter. The Mapillary rigs DA3 reads at 1.8–2.1 m are in that rig's height class; nothing
here says whether they share its range dependence.

### 2.2 Against the labeler's corrected depth frame

`recall_by_depth_112.py` carries the labeler's per-city estimate of how far Google's depth frame
runs short of the imagery's own geometry (`DEPTH_FRAME_SCALE`, from bearing-only triangulation in
the labeler's camera-height study; approximate, and whether it is a scale or an offset is open
there):

| split | locations | labeler scale | DA3/Google | DA3/(Google x scale) |
|---|---:|---:|---:|---:|
| bend | 266 | 1.06 | 1.033 | 0.975 |
| paterson | 372 | 1.08 | 1.109 | 1.027 |
| gainesville | 264 | 1.095 | 1.089 | 0.995 |
| sao_paulo | 286 | 1.16 | 1.180 | 1.018 |

The per-city spread of DA3/Google (1.03–1.18) narrows to 0.975–1.027 once Google is corrected by
the labeler's factors. DA3 is a monocular model and the labeler's instrument is multi-view
geometry, and neither uses Google's depth. Both put sao_paulo's Google depth shortest and bend's
least short; **paterson and gainesville swap rank** (labeler 1.08 < 1.095, DA3/Google 1.109 >
1.089). This is **n = 4 cities**, against factors the labeler itself calls approximate. It is
consistent with the per-city part of the DA3/Google ratio belonging to Google's frame, and it is
not strong enough to rest a correction on.

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

DA3 recovers **about 74%** of the rig-level height difference Google's payload shows (0.39 m, 1.92
vs 2.31, against Google's 0.53 m, 1.83 vs 2.36) and places a typical pano's height within ~7%
(median |ln ratio| 0.070). It does not track panos *within* a rig: r is 0.08 and −0.13 inside the
two US groups, so the pooled r = 0.64 comes from the difference between groups. paterson's r = 0.75
is the same effect inside one split, which mixes both rigs. **Read a DA3 height as a rig-class
estimate, not a per-pano measurement.**

Three height readings, same panos:

| split | reading | panos | DA3/Google median | median abs ln ratio | Pearson r |
|---|---|---:|---:|---:|---:|
| gsv_pooled | lowest_plane | 352 | 1.038 | 0.070 | 0.635 |
| gsv_pooled | dominant_plane | 352 | 1.027 | 0.071 | 0.639 |
| gsv_pooled | gt_points | 323 | 1.091 | 0.114 | 0.555 |

On GSV the lowest plane is **not** always the dominant one: on **76 of the 403** fitted panos of the
four Google-depth splits it is a different plane (gainesville 28 of 106), lying a median 9% lower
than the dominant plane (p10–p90 1.4–25%; what those surfaces are was not inspected) (§8). Measured against Google's height the
two readings end up close (1.038 vs 1.027). `gt_points` is the median of DA3 ray × sin(depression)
over a pano's GT points: it never looks at the nadir, but it reads ~5% higher than the plane and is
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

The script picks the headline axis by rule, the DA3 axis with the lowest pooled_common median
|ln ratio|: **`da3_point`**. The two plane-based axes are worse because DA3's per-pano height error
(§3) enters every point on the pano, and a tilt error of a degree moves a near-horizon point a long
way; reading DA3 at the point itself avoids both. The plane axes also moved between the two runs
(§8): within 10%, `flat_da3_height` 0.553 → 0.517 and `da3_plane` 0.417 → 0.433; `da3_point`
does not use the plane.

On bend, where Google's camera is ~2.4 m and the flat axis is already close, DA3 loses: the
held-out constant (fitted on the other three splits, 1.125) over-corrects bend, whose own ratio is
1.03. That is the per-city spread of §2.2 again.

### 4.1 The downstream method, checked on GSV

§5.2 reads recall by distance and maps the published thresholds on the DA3 axis. On GSV the same
computation can be run on Google's axis and compared. Fn-confirmed GT points on measured-ground
panos (1,101), DA3 axis with the leave-one-split-out k_point:

| bucket | n (Google) | recall (Google) | n (DA3) | recall (DA3) | n (flat 2.5 m) | recall (flat 2.5 m) |
|---|---:|---:|---:|---:|---:|---:|
| 0-8 m | 281 | 0.804 | 276 | 0.833 | 214 | 0.836 |
| 8-12 m | 254 | 0.791 | 244 | 0.766 | 245 | 0.792 |
| 12-18 m | 286 | 0.717 | 297 | 0.724 | 238 | 0.719 |
| 18-25 m | 194 | 0.598 | 207 | 0.585 | 234 | 0.705 |
| 25-40 m | 83 | 0.301 | 77 | 0.273 | 134 | 0.448 |
| 40 m+ | 2 | 0.000 |  |  | 35 | 0.114 |

| group | n | Google: 18 m / 25 m become | DA3: 18 m / 25 m become |
|---|---:|---|---|
| gsv_pooled | 1101 | 16.2 / 21.7 m | 15.9 / 21.3 m |
| bend | 254 | 16.9 / 23.3 m | 15.9 / 21.2 m |
| paterson | 360 | 15.9 / 21.7 m | 16.3 / 21.8 m |
| gainesville | 249 | 13.8 / 19.7 m | 13.1 / 18.5 m |
| sao_paulo | 238 | 16.9 / 22.8 m | 17.8 / 24.2 m |
| 2025-26 rig | 366 | 13.1 / 18.7 m | 13.3 / 18.2 m |

The DA3 axis reproduces Google's recall curve bucket by bucket, and the flat axis shows on GSV the
same far-field inflation §5.2 finds on Mapillary (0.448 at 25–40 m against Google's 0.301). The
thresholds agree pooled to 0.3–0.4 m; per split DA3 is off by −1.0 to +0.9 m at 18 m and −2.1 to
+1.4 m at 25 m. **That ±1–2 m is the error scale to read beside every per-split Mapillary threshold
in §5.2.** This validates the method on GSV imagery; it does not validate the transfer of the
calibration to GoPro or NCTech cameras.

## 5. Carried to Mapillary

### 5.1 Camera height by split and by rig

Calibrated DA3 height (lowest plane / k_height):

| split | panos | fit ok | median h (p25–p75) | min–max | median tilt (deg) | Google h median |
|---|---:|---:|---|---|---:|---:|
| bend | 110 | 94 | 2.21 (2.13–2.29) | 1.92–2.92 | 2.1 | 2.38 |
| paterson | 125 | 108 | 2.11 (1.90–2.39) | 1.38–2.87 | 2.2 | 2.05 |
| gainesville | 125 | 106 | 1.89 (1.79–2.10) | 1.60–2.79 | 2.0 | 1.83 |
| sao_paulo | 125 | 95 | 2.46 (2.27–2.54) | 1.34–2.78 | 2.6 | 2.26 |
| laurens_gsv | 86 | 78 | 2.14 (2.09–2.19) | 1.82–2.59 | 2.1 | 2.41¹ |
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

¹ laurens_gsv has Google depth since #201 (harvested 2026-09-27 for #151). This cell does not come
from the table the other rows come from. `tables.md`'s camera-height-by-split table still prints
"–" here, because that table is pinned and laurens_gsv's Google rows are not joined into the
committed rows. The value is from `tables()["laurens_gsv_held_out"]` (§5.3). It is the median over
the 60 of 86 benchmark panos with measured Google ground (min–max 1.99–2.49 m). The DA3 columns are
over the 78 fit-ok panos, a different set. On the 53 panos in both sets, DA3 reads 2.14 m calibrated
and Google 2.41 m, the same medians (§5.3).

Caveats that belong to these numbers:

- **The calibration was fitted on Google's car rigs.** Nothing in this repo validates DA3 on a
  GoPro or an NCTech image directly. The heights are plausible (every Mapillary rig median lands
  in 1.8–2.5 m, and DA3 reads the same Laurens footprint at 2.14 m on Google's rig and 2.13 m on
  the GoPro Max), but "plausible" is not "measured". On the one GSV split held out of the fit,
  paired against Google on the same 53 panos, calibrated DA3 reads 0.891× Google's height (CI
  0.875–0.912, §5.3). Google's frame in Laurens is uncorrected, because the labeler has no factor
  for it, so this is a reading against Google, not against the imagery's geometry. Whether any of
  it applies to the GoPro Max on that footprint depends on which of §5.3's three explanations holds.
- **Per-pano heights are rig-class estimates** (§3). The spread inside a rig here mixes real
  mounting differences with DA3's per-pano noise, and this measurement cannot separate them.
- **Tilt.** By split the Mapillary medians are 3.1–7.3° against 2.0–2.6° on GSV; by rig the NCTech
  iStar Pulsar reads 2.1°, the same as Google's rigs, and richmond's GoPro Max 8.2°. The noise floor
  under those numbers, measured on 352 GSV panos: DA3's plane tilt has median 2.18° where Google's
  ground plane reads 1.53°, r = 0.375, median |difference| 1.17°. So DA3 tilts of ~2° are at the
  floor, and only the budapest (7.3°) and GoPro-Max-in-richmond (8.2°) values stand clearly above it.
- **Band sensitivity.** Fitting the 20–60° band instead of 20–45° changes the median fitted height
  by 0.97–1.00× per split (bend 0.973, laurens_gsv 0.972, every Mapillary split 0.982–1.000) and
  lowers the inlier share, so the choice of 45° moves heights by at most ~3%.
- **4 of 1,043 fitted panos still read below 1.2 m** (morgantown 0.52, 0.68, 0.84; budapest 0.60):
  the lowest-plane rule did not reach the road there. They are left in, not filtered.
- **The Laurens panos are not a public input.** `laurens_gsv` and `laurens_mapillary` are not in
  HF `projectsidewalk/rampnet-benchmark` (`scripts/unpack_benchmark_panos.py`), so the Laurens rows
  here, their share of the Mapillary pool, and all of §6 cannot be re-extracted from public inputs.
  Their committed DA3 rows re-derive every table; re-running the GPU step for them needs the panos
  from the lab's mirror until those two splits are published.

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
threshold on the flat axis, window n in brackets, as #112 computed it). Read each per-split value
with the ±1–2 m error scale measured on GSV in §4.1:

| population | DA3 point: 18 m / 25 m become | DA3 plane | flat at DA3 height |
|---|---|---|---|
| mapillary_pooled | 14.5 m (461) / 19.1 m (240) | 15.5 m (369) / 20.9 m (194) | 15.1 m (369) / 21.2 m (194) |
| richmond | 15.4 m (116) / 21.4 m (69) | 16.0 m (78) / 22.0 m (49) | 17.6 m (78) / 24.5 m (49) |
| annapolis | 15.1 m (99) / 17.2 m (45) | 15.9 m (79) / 19.3 m (37) | 15.3 m (79) / 21.2 m (37) |
| morgantown | 14.7 m (70) / 18.9 m (35) | 15.9 m (55) / 20.9 m (31) | 15.5 m (55) / 21.3 m (31) |
| clovis | 13.1 m (68) / 17.6 m (28) | 14.8 m (64) / 18.8 m (28) | 14.6 m (64) / 20.0 m (28) |
| laurens_mapillary | 15.9 m (40) / 22.2 m (24) | 16.1 m (35) / 23.8 m (21) | 16.4 m (35) / 23.6 m (21) |
| budapest_district5 | 11.9 m (68) / 16.1 m (39) | 13.1 m (58) / 17.9 m (28) | 13.5 m (58) / 19.0 m (28) |
| laurens_gsv (GSV; held out of the fit, checked against Google in §5.3) | 16.0 m (72) / 22.0 m (41) | 15.3 m (65) / 21.3 m (37) | 15.5 m (65) / 21.6 m (37) |

What this says, with its limits:

- **On these rigs the flat axis overstates the far field**, as it does on GSV against Google
  (§4.1). 128 GT points are 25–40 m out on the flat axis and only 58 on the DA3 axis; recall there
  is 0.547 on the flat axis and 0.121 on DA3's. The flat axis is putting nearby ramps, which RampNet
  mostly finds, into the far buckets, so "blind past 25 m" is if anything *understated* on
  Mapillary, and it starts nearer: recall on the DA3 axis is 0.69 at 12–18 m and 0.47 at 18–25 m.
- **The far buckets are small.** 58 points at 25–40 m on the DA3 axis, pooled over six cities;
  per split they are 2–22. Read the cliff, not the third decimal.
- **The DA3 axis carries the §2 questions.** It is in Google's frame (§2.2), and on the rig class
  closest to these cameras the pooled constant leaves far points ~5% short and near points ~3%
  long (§2.1).
- The flat and DA3 plane / height columns have fewer points than the DA3 point column: flat drops
  the 17 GT points at or above the horizon, and the plane axes need a passing fit (1,277 and
  1,266 of 1,615).

### 5.3 Held-out check: laurens_gsv against Google depth

laurens_gsv is a GSV split with Google depth (#201) that the calibration never saw. It enters no
pooled statistic and no leave-one-split-out set, and its committed rows carry no Google fields.
`tables()["laurens_gsv_held_out"]` joins its DA3 rows to #201's committed per-pano and per-point
Google rows in `analysis_out/recall_by_depth_112.json`, and nothing else. The pooled constants of
§2 and §3 are the prediction.

**n at every step.** 86 benchmark panos. DA3's ground fit passes on 78. Google has measured ground
on 60. Both hold on **53**, and the height comparison uses those. For points, all 341 DA3 point
rows join a Google row (230 unique locations). Of those, **156 locations on 45 panos** have
measured ground and a pixel-plane Google range, the same filter as §2.

| read | n | laurens_gsv DA3/Google [95% CI] | pooled prediction [95% CI] | four fitted splits | laurens_gsv / pooled [95% CI] | older US vintages [CI] |
|---|---|---|---|---|---|---|
| range at the point | 156 locations (45 panos) | 1.015 [0.997, 1.046] | 1.106 [1.094, 1.117] | 1.033–1.180 | **0.918** [0.897, 0.947] | 1.066 [1.050, 1.082] |
| camera height, paired | 53 panos | 0.925 [0.920, 0.942] | 1.038 [1.026, 1.056] | 0.959–1.109 | **0.891** [0.875, 0.912] | 0.995 [0.976, 1.010] |

The CIs are pano-clustered, 1,000 replicates, seed 101, as in §2 and §3. The "laurens_gsv /
pooled" CI resamples both samples' panos, so it carries the pooled constant's uncertainty too.
"Older US vintages" is the rig group laurens_gsv's 2024 capture would join under #112's grouping;
it is a second reference, not the prediction.

By Google range, laid out as §2.1:

| Google range | n (panos) | laurens_gsv DA3/Google (p10–p90) | 2025-26 rig | older US | sao_paulo | flat 2.5 m / Google, laurens_gsv |
|---|---|---|---:|---:|---:|---:|
| 0-8 m | 28 (19) | 0.993 (0.92–1.07) | 1.144 | 1.028 | 1.144 | 1.053 |
| 8-12 m | 47 (26) | 1.032 (0.95–1.13) | 1.135 | 1.061 | 1.208 | 1.025 |
| 12-18 m | 40 (26) | 1.010 (0.94–1.12) | 1.111 | 1.090 | 1.198 | 1.040 |
| 18-25 m | 23 (17) | 1.084 (0.97–1.19) | 1.089 | 1.079 | 1.174 | 1.027 |
| 25 m+ | 18 (7) | 1.015 (0.86–1.06) | 1.047 | 1.049 | 1.193 | 1.074 |

The log-log exponent is 1.015 (CI 0.982–1.042), against 0.981 (CI 0.959–1.001) pooled and 0.919 on
the 2025–26 rig. The OLS slope is 1.008 (CI 0.950–1.065) with a 0.32 m intercept.

Calibrated with the pooled constants, each axis against Google's range at the 132 locations where
every axis has a value:

| axis | median axis/Google (p10–p90) | median abs ln ratio | share within 10% |
|---|---|---:|---:|
| flat_2p5 | 1.034 (1.00–1.16) | 0.034 | **0.742** |
| da3_point | 0.925 (0.85–1.02) | 0.089 | 0.591 |
| da3_plane | 0.899 (0.72–1.05) | 0.127 | 0.439 |
| flat_da3_height | 0.899 (0.84–1.01) | 0.108 | 0.462 |

**Where laurens_gsv falls against the fitted splits.** The pooled CI states how well the pooled
median is known, not how far a split can land from it. Fitted splits fall outside it too:

| read | fitted splits / pooled | fitted splits outside the pooled CI | laurens_gsv / pooled | laurens_gsv / bend [95% CI] |
|---|---|---|---|---|
| range at the point | 0.934–1.067 | 3 of 4 (bend, gainesville, sao_paulo) | 0.918 | [0.954, 1.016] |
| camera height, paired | 0.924–1.068 | 2 of 4 (bend, sao_paulo) | 0.891 | [0.945, 0.987] |

bend is the fitted split closest to laurens_gsv: mostly 2024 capture, with Google cameras at 2.38 m
and 2.41 m. The laurens_gsv / bend CI is two-sample and pano-clustered, like the pooled one.

**The second explanation, in numbers.** §2.2 divides each fitted city's DA3/Google by the labeler's
estimate of how far Google's depth frame runs short there. The labeler has no such factor for
Laurens. The fitted splits' frame-corrected values, beside laurens_gsv's uncorrected ones:

| read | bend | paterson | gainesville | sao_paulo | corrected range | laurens_gsv, uncorrected |
|---|---:|---:|---:|---:|---|---:|
| range at the point | 0.975 | 1.027 | 0.995 | 1.018 | 0.975–1.027 | 1.015 (inside) |
| camera height | 0.904 | 0.955 | 0.963 | 0.956 | 0.904–0.963 | 0.925 (inside) |

**What the check shows.**

- **laurens_gsv lands at the low edge of the fitted splits' spread, and past it.** Against the
  pooled constants the four fitted splits span −6.6% to +6.7% at points and −7.6% to +6.8% in
  height. laurens_gsv is at −8.2% and −10.9%. That widens the low end by 1.6 points at points and
  3.3 points in height. Its CIs do not overlap the pooled CIs, but that alone says little: by the
  same CI-overlap test 2 of 4 fitted splits fail at points (bend, sao_paulo) and 2 of 4 in height,
  and by point estimate outside the pooled CI (the table above) 3 of 4 and 2 of 4.
- **Against bend, the contrast holds in height only.** At points laurens_gsv's CI (0.997–1.046)
  contains bend's 1.033, and the laurens_gsv / bend CI (0.954–1.016) contains 1. In height the
  laurens_gsv / bend CI is 0.945–0.987 and excludes 1. So "below all four fitted splits" is true of
  the point estimates. It is supported beyond noise only in height.
- **The two reads agree with each other.** Points and heights both put DA3 about 8–11% shorter
  relative to Google than the pooled fit does. The paired height ratio (0.891×) matches the
  unpaired read that raised the flag (2.14 m over 78 panos against 2.41 m over 60, 0.890×). So the
  flag was not an artifact of comparing different pano sets.
- **The shape transfers.** On laurens_gsv the ratio is flat with range (exponent CI contains 1;
  0.99–1.08 by bucket). What differs is the constant, not a range dependence like the 2025–26
  rig's.
- **On this split the flat axis beats calibrated DA3.** Flat at 2.5 m is within 10% of Google at
  74% of locations, calibrated DA3 at 59%. Google puts this rig at 2.41 m, close to the 2.5 m the
  flat axis assumes, as on bend (§4). Calibrated DA3 reads 8% short of Google. So its published
  thresholds on laurens_gsv (16.0 / 22.0 m, §5.2) sit about 1.4 m below #112's thresholds on
  Google's axis (17.4 / 23.4 m, [`detection_recall_analysis.md`](detection_recall_analysis.md)
  §0.1). That is inside §4.1's ±1–2 m leave-one-split-out error scale.

**What it means for carrying the calibration.** Three explanations fit the data, and the committed
inputs cannot separate them.

1. **DA3's per-city scale varies.** DA3 reads Laurens short because of its scenes (rural streets,
   a mostly-sky upper half), and Google's frame is equally good everywhere.
2. **Google's frame varies by city, and DA3 does not.** Google's depth runs short in the four fitted
   cities, as the labeler estimates (§2.2), but not in Laurens, an older 2.41 m rig. The pooled
   constant has absorbed the fitted cities' shortfall. laurens_gsv's uncorrected ratios sit inside
   the fitted splits' frame-corrected bands, so this reading fits the data with no DA3 shortfall at
   all.
3. **DA3 has a bias specific to Google's 2024 imagery or rig.** It would be neither a scene effect
   nor Google's frame, and it would not carry to the GoPro Max.

The consequence for the Mapillary tables differs by reading:

- **Under reading 1**, the spread is DA3's. Every calibrated Mapillary distance and threshold in
  §5.2 carries a split-level spread about as wide as the observed range, −8% to +7%, and every
  height in §5.1 −11% to +7%. That range comes from five GSV splits, four of them in the fit. It is
  an observed range, not an interval with stated coverage. At 18 m, −8% is 1.4 m. That is consistent
  with §4.1's ±1–2 m, which already carried most of it: the four fitted splits alone spanned about
  ±7%. The rig-level ordering of §5.1 can change only where rigs differ by less than about 0.2 m.
- **Under reading 2**, the spread is Google's. DA3's own city-to-city spread, once Google is
  corrected, is about ±3% (0.975–1.027 at points, 0.904–0.963 in height). Mapillary has no Google
  frame, so what remains is a **common bias** of the Google-calibrated axis against the imagery's
  own geometry. That is §2.2's caveat: raw DA3 is closer to the imagery than calibrated DA3, by
  roughly 6–16% depending on city. Every calibrated Mapillary distance and height would then be
  short by about the same factor, and the per-split thresholds would not scatter by ±8%.
- **Under reading 3**, laurens_gsv says nothing about the Mapillary rigs.
- **laurens_mapillary in particular.** It shares laurens_gsv's footprint. Under reading 1 the GoPro
  Max reads low by about the same amount, 2.13 m / 0.891 ≈ 2.39 m. Under reading 2 the GoPro's
  2.13 m is low against the imagery's geometry by about k_height divided by the Laurens frame
  factor, which is not known. Under reading 3 no correction applies. **laurens_mapillary's 2.13 m
  is not corrected.**
- **The deciding measurement** is the labeler's depth-frame factor for Laurens (§11). A factor near
  1.0 would favour reading 2. A factor near the fitted cities' 1.06–1.16 would favour reading 1 or 3.

Caveats, beside the numbers:

- **Different pano sets.** The height comparison uses the 53 panos in both sets, not the 78 or
  the 60. The point comparison uses 45 panos. Both are smaller than any fitted split's (78–97 panos
  for heights, 83–92 for points).
- **Google's depth is an unpublished input.** Only its committed per-pano and per-point rows are
  used, as for the four fitted splits. `--check` re-derives the block from a clean clone without
  payloads.
- **The image-to-payload column mapping is assumed on laurens_gsv, not confirmed**
  ([`detection_recall_analysis.md`](detection_recall_analysis.md) §0.2: the edge check prefers it
  on 56 of 86 panos). The point comparison rests on that mapping. The height comparison does not,
  because Google's camera height is a per-pano ground-plane distance. It reads the same way.
- **The frame-corrected bands use factors the labeler calls approximate** (§2.2), over four
  cities. That laurens_gsv falls inside them is consistent with reading 2. It does not establish it.
- **One 2024 rig in one small town.** This is one held-out split. It widens the observed spread by
  1.6 points at points and 3.3 points in height. It does not measure that spread well.
- **The Laurens panos are not published** (§5.1), so DA3 cannot be re-extracted for this split from
  public inputs. Its committed DA3 rows re-derive the block.

## 6. Laurens cross-read against the labeler's height instrument

Read-only from sidewalk-auto-labeler's **tracked** `runs/laurens/camera_heights.json` at commit
`2653a49c420465bd792d165ece5681ad6c2ace4a` (sha256 `74900b62…c54d`, pinned in the script; `derive`
reads it with `git show` at that commit and refuses any other bytes, and `--check` refuses a
`tables.json` whose labeler block names another commit or hash). 2653a49 is the re-issue after
[sidewalk-auto-labeler#89](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/89)
(its results commit `1d8127e`, 2026-09-27).

The first version of this section read the labeler's working tree at `29dc605`, including
`runs/laurens/camera_height/groups.csv`. That file is an untracked run output (`.gitignore:
runs/**`) and is in no labeler commit, so the sequence-level rows it supplied (a DA3-vs-h_scale
correlation over 16 sequences) could not be reproduced and have been removed.

- **DA3:** GoPro Max 2.13 m (73 benchmark panos); the same footprint on Google's rig 2.14 m
  (laurens_gsv, 78 panos). Google's depth reads that rig at 2.41 m over 60 panos (§5.1, footnote ¹).
  Held out and paired on 53 panos, calibrated DA3 reads the GSV rig 0.891× Google's height, in
  Google's uncorrected Laurens frame (§5.3). Whether the GoPro Max reading shares that shortfall
  is not known (§5.3).
- **Labeler, gopro/max group (`n_panos` 644 in its table):** bearing fixed point
  (instrument A) **4.31 m** (CI 3.80–4.84, slope 0.723); scale identity evaluated at 2.6 m
  (instrument B) **2.98 m** (CI 2.93–3.03). #89 found **no validated instrument-B estimator**: under
  its pre-registered rule neither the line fixed point (max |mean error| 0.361 m) nor the local
  crossing (0.418 m) passes, and "at noise 1.0 both read +0.25 to +0.53 m high at h_true 1.8 m in
  every city and rig class". The group is not applied: the labeler keeps its 2.6 m default, and
  its reason string reads, in full, "fails identifiable (slope 0.723); agreement (b_unvalidated: no
  h*_B estimator passed rule V in this city; line 3.755, local 3.481); material (CI 3.80-4.84, h_g
  n/a); SUSPECT: bearing height outside 1.0-3.5 m".

So the ~0.8 m gap between DA3 (2.13 m) and the labeler's B value (2.98 m) sits beside the
labeler's own finding that B reads high by up to about half a metre at DA3-like heights, and that A
is outside its plausible range. Part of the gap is explained by that bias; how much cannot be said
from here. **Recorded as a difference between one uncalibrated-on-this-rig instrument and one
unvalidated one, not resolved.** A direct test would be a Laurens pano pair with a known ground
distance (for example, a mapped curb-ramp pair on both arms).

## 7. The published DA3 figures, re-derived

`detection_recall_analysis.md` reported DA3 and flat-ground geometry agreeing "to within
6.5–8.5% (Spearman ρ = 0.95 Bend / 0.81 Richmond)" and "4 Richmond ramps that geometry placed above
the horizon", from `gt_depth_da3.json`, which was never committed. Re-derived from this run's rows
under `depth_analysis.py`'s own filters (points with flat < 150 m; ratio over DA3 > 0.5 m;
"unusable" = at or above the horizon, or flat ≥ 150 m):

| split | n (flat < 150 m) | flat/DA3 raw value, median | Spearman | flat/DA3 horizontal range, median | Spearman | unusable on flat (above horizon + flat ≥ 150 m) |
|---|---:|---:|---:|---:|---:|---|
| bend | 327 | 1.065 | 0.953 | 1.014 | 0.961 | 0 (0 + 0) |
| richmond | 306 | 1.085 | 0.812 | 1.037 | 0.847 | 4 (3 + 1) |

The published numbers reproduce: 6.5% and 8.5%, ρ 0.95 / 0.81, and the "4 Richmond ramps" are 3
at or above the horizon plus 1 below it at a flat range of 405 m. (Without the 150 m filter
richmond reads 8.6%.) That pins down what they measured: DA3's raw planar depth against the flat
*horizontal* range. Those are different quantities (a point off the view's axis has planar depth
shorter than its ray), and on a like-for-like horizontal range the agreement is closer, 1.4% on
bend and 3.7% on richmond.

## 8. The ground-fit change: two parts, attributed

The first full run (job `40774944`) took the ground as the **single dominant plane** of the band
and passed it at **inlier share ≥ 0.5**. On car-roof consumer rigs that plane is often the car's own
roof, about 0.75 m under the camera: morgantown (GoPro Max) read heights down to 0.53 m and passed
only 23 of 125 panos. After seeing that, **two things were changed together** and the extraction
re-run (job `40777019`): the ground became the **lowest** of up to three planes that each hold
≥ 15% of the band, and the pass rule became **≥ 0.25** of the band on that plane. Both were set
after seeing the data. The first run's raw rows are committed under
`analysis_out/da3_calibration_101/raw_run1/`, and `tables()["ground_fit_change"]` attributes each
pano, from committed rows, to one change or the other:

| group | panos | run 1 pass @0.5 | run 1 plane pass @0.25 (of which < 1.3 m) | now pass (< 1.3 m) | same as run 1 | threshold only | switched plane | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bend | 110 | 106 | 110 (0) | 94 (0) | 80 | 3 | 11 | 0 |
| paterson | 125 | 95 | 125 (0) | 108 (0) | 74 | 13 | 20 | 1 |
| gainesville | 125 | 116 | 125 (0) | 106 (0) | 70 | 5 | 28 | 3 |
| sao_paulo | 125 | 86 | 125 (2) | 95 (0) | 60 | 16 | 17 | 2 |
| laurens_gsv | 86 | 81 | 86 (0) | 78 (0) | 68 | 1 | 8 | 1 |
| richmond | 124 | 99 | 124 (14) | 93 (0) | 60 | 18 | 15 | 0 |
| annapolis | 125 | 96 | 125 (0) | 96 (0) | 82 | 13 | 0 | 1 |
| morgantown | 125 | 23 | 125 (33) | 92 (3) | 15 | 62 | 15 | 0 |
| clovis | 125 | 48 | 125 (12) | 112 (0) | 48 | 55 | 7 | 2 |
| laurens_mapillary | 94 | 62 | 94 (0) | 73 (0) | 47 | 19 | 7 | 0 |
| budapest_district5 | 125 | 105 | 124 (10) | 96 (1) | 74 | 10 | 12 | 0 |
| gsv_google_depth | 485 | 403 | 485 (2) | 403 (0) | 284 | 37 | 76 | 6 |
| mapillary_all | 718 | 433 | 717 (69) | 562 (4) | 326 | 177 | 56 | 3 |

Heights uncalibrated. Of today's passing fits: *same as run 1* = the lowest plane is the dominant
one, run 1 passed at 0.5, same height within 3%; *threshold only* = the same plane, which run 1
failed only on the 0.5 rule; *switched plane* = the lowest plane is not the dominant one; *other* =
lowest is dominant but its height moved > 3% from run 1 (run-to-run DA3 / RANSAC noise).

What the two parts did:

- **The pass rule did most of the fit-count change.** morgantown's 23 → 92 is 62 threshold-only
  fits (run 1's own plane, which failed only the 0.5 rule) and 15 switched planes; clovis's 48 → 112
  is 55 threshold-only and 7 switched.
- **The two changes only make sense together.** Run 1's dominant plane under the 0.25 rule alone
  would pass 125 of 125 morgantown panos, 33 of them below 1.3 m (roofs); 69 across the Mapillary
  splits. The lowest-plane rule is what keeps those roofs out (4 remain).
- **GSV is affected too.** The lowest plane differs from the dominant one on 76 of 403 fitted GSV
  panos with Google depth (gainesville 28 of 106). §3's heights barely move (1.027 → 1.038 against
  Google), but the plane-based axes of §4 do, in both directions (within 10%: `flat_da3_height`
  0.553 → 0.517, `da3_plane` 0.417 → 0.433). The headline `da3_point` axis does not use the plane.

The first run's own camera-height table is in its committed `tables.json`:
`git show 4ab47a2:analysis_out/da3_calibration_101/tables.json` → `tables.camera_height_by_split`
(k_height 1.022). `--check` does not pass at `4ab47a2` itself (a dict-order bug in the markdown
check, fixed in `2c51a4a`), so that JSON, not a re-run of `--check` there, is the route. The table
above re-derives from the committed first-run raw rows with the current code.

| split | fit ok, run 1 | median h, run 1 | min h, run 1 | fit ok, now | median h, now | min h, now |
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

(Calibrated heights; each run by its own k_height.) The point calibration (§2) does not use the
plane. It is not bit-identical between the runs because they ran on different GPUs (A40, then Quadro
RTX 6000): over 5,402 points the run-to-run difference in the DA3 value has median 0.06%, p99 0.35%,
max 0.82%, and pooled DA3/Google moved from 1.105 to 1.106.

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

The script wrote every row with the label `da3-calibration-101:extract`. Two were **relabelled by
hand** when appended to `usage_log.jsonl`: the smoke test's to `da3-calibration-101:extract-smoke`
and the re-run's to `da3-calibration-101:extract-v2-lowest-plane`; the first full run kept the
script's label. `extract` now takes `--label` (and the launcher `LABEL=`), so a future run needs no
hand edit. The re-run is slower for two reasons that were not separated: a slower GPU, and three
sequential RANSAC fits per band instead of one (the fits run on the CPU inside the loop). Model load
was 12–21 s per job (20.8, 11.6, 13.2 s). The derive step (CPU, laptop) takes about 10 s;
`--check` about the same.

## 10. How to reproduce

Inputs: the committed bundles (`benchmark/<split>/{records.jsonl,verdicts.json,imagery_manifest.json}`),
the committed #112 rows (`analysis_out/recall_by_depth_112.json`), the benchmark panos (Hugging
Face `projectsidewalk/rampnet-benchmark` for nine splits; **laurens_gsv and laurens_mapillary are
not on the Hub**, see §5.1), sha256-checked against each manifest, DA3 weights
`depth-anything/DA3METRIC-LARGE` at revision `4010e39f3634a45bc60553321fb49fb760bd594e` and code
commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`. Unpublished: Google's depth payloads (only their
committed per-point rows are needed) and the Laurens panos. The labeler file is public at the pinned
commit and is needed only by `derive`; its values are copied into `tables.json`.

```bash
# 0. environment (paths are Jon's klone layout; any CUDA box works)
ROOT=/gscratch/scrubbed/jfroehli/da3_101
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git $ROOT/src/Depth-Anything-3
git -C $ROOT/src/Depth-Anything-3 checkout 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
mkdir -p $ROOT/src/Depth-Anything-3/stubs/moviepy && touch $ROOT/src/Depth-Anything-3/stubs/moviepy/{__init__,editor}.py
python -m venv --system-site-packages $ROOT/env     # over a sidewalkcv2 env (torch 2.6, numpy 2.2)
$ROOT/env/bin/pip install omegaconf einops addict opencv-python-headless plyfile pycolmap trimesh evo
git clone --branch <this branch or main> https://github.com/ProjectSidewalk/RampNet.git $ROOT/repo

# 1. GPU: DA3 at every point + ground fits -> $ROOT/out/raw/<split>.jsonl (~45-70 min, one GPU).
#    The launcher logs to logs/%x_%j.out under the submit directory, which must exist before
#    sbatch (Slurm does not create it, and a missing directory kills the job with no log). To log
#    elsewhere, create that directory and pass --output=<dir>/%x_%j.out instead.
cd $ROOT/repo && mkdir -p logs
OUT=$ROOT/out/raw sbatch -A ckpt-makelab -p ckpt-all scripts/analysis/da3_calibration_101.slurm
#    copy $ROOT/out/raw/<split>.jsonl to analysis_out/da3_calibration_101/raw/

# 2. CPU: rows, tables, markdown, SHA256SUMS (reads the labeler file with git show at the pin)
python scripts/analysis/da3_calibration_101.py derive --labeler-root /path/to/sidewalk-auto-labeler

# 3. CPU, from a clean clone, no GPU, no panos, no payloads: everything re-derives
python scripts/analysis/da3_calibration_101.py --check
python scripts/analysis/da3_calibration_101.py --check --markdown   # every table, per split
```

`tests/test_da3_calibration_101.py` runs the `--check` path, pins the headline numbers of this
document (including the §8 attribution, the §4.1 thresholds, the pinned labeler input, and the
§5.3 held-out n, ratios, CIs and verdict flags), and checks the geometry on synthetic scenes (level
and tilted ground, a car-sized obstacle, a vehicle roof over a road). The held-out join, the
two-sample bootstrap and the prediction flags are unit-tested on synthetic data. §5.3 needs no
extra input: its Google rows are the held-out split's rows already committed in
`analysis_out/recall_by_depth_112.json`, which `SHA256SUMS` already hashes.

## 11. Gaps, stated

- **Not validated on Mapillary imagery.** Mapillary serves no depth, so the Mapillary numbers rest
  on the GSV calibration transferring to other cameras. §4.1 validates the method on GSV only.
- **Google is the reference, and Google is not ground truth** (§2.2). The calibrated axis is in
  Google's frame by design.
- **A single pooled constant** is biased on Google's 2025–26 rig, −5% to +3% across range (§2.1);
  no per-rig or range-dependent calibration was fitted.
- **One held-out split, at the low edge of the fitted spread** (§5.3). laurens_gsv was held out of
  the fit and checked against Google on the same panos and points: DA3/Google 0.918× the pooled
  constant at points (CI 0.897–0.947, 156 locations on 45 panos) and 0.891× in height (CI
  0.875–0.912, 53 panos). The four fitted splits spanned −6.6% to +6.7% and −7.6% to +6.8%; this
  split widens the low end by 1.6 and 3.3 points. No per-split or per-city calibration was fitted
  in response. laurens_gsv was not added to the fit, the leave-one-split-out set or any pooled
  number, so every earlier table is unchanged.
- **Not measured: the labeler's Google depth-frame factor for Laurens.** It is the measurement that
  decides between §5.3's explanations. Without it, "DA3 reads Laurens short" and "Google's frame is
  short in the fitted cities but not in Laurens" fit the committed numbers equally well, and they
  lead to different Mapillary error models (a split-level spread against a common bias). The
  labeler's camera-height study did not cover laurens_gsv. Running its bearing-only triangulation
  on the laurens_gsv run would unblock it. **The Laurens panos are not published** (§5.1).
- **Per-pano heights and tilts are not trustworthy per pano** (§3, §5.1).
- **Apparent-size tables and the resolution forecast** of #112 were not re-issued on the DA3
  axis; only recall by distance was, as the plan specified.
- **`manual_gold`** is out of scope (no verdict review, no depth, 1,000 panos).
- **One model.** RANSAC is seeded per pano. DA3's output is not bit-identical across GPUs (§8:
  median 0.06%, max 0.82% between an A40 and a Quadro RTX 6000), far below every effect reported
  here. No other monocular depth model was tried.

🤖 Generated with [Claude Code](https://claude.com/claude-code) — Opus 5.5, claude-opus-5-5; plan by Fable 5.1, claude-fable-5-1
