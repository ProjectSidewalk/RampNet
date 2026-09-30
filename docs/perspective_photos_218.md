# RampNet on perspective photos ([#218](https://github.com/ProjectSidewalk/RampNet/issues/218))

Code:
- `rampnet/perspective.py`: camera model, canvas embed, hit tests;
- `scripts/analysis/perspective_photos_218.py`: Richmond `select`, `fetch`, `infer`,
  `merge`, `score`, `gallery`, `rates`, `reledger`;
- `scripts/analysis/perspective_photos_218.sh`: the GPU launcher;
- `scripts/analysis/seoul_photos_218.py`: Seoul `manifest`, `fetch`, `infer`, `gallery`,
  `rates`;
- `scripts/analysis/rating_page_218.py`: the shared one-question rating page, and the
  checks `rates` runs on a rater's export;
- `tests/test_perspective_218.py`: geometry, chance-floor and rating-path tests.

Outputs are in `analysis_out/perspective_photos_218/` (Richmond) and
`analysis_out/perspective_photos_218/seoul/`. The galleries are
`benchmark/richmond_flat_fp_218/` and `benchmark/seoul_presence_218/`.

Run 2026-09-30 on free compute (makelab2 A40, desktop for fetches and CPU scoring).
Revised the same day after the review of PR #227
(https://github.com/ProjectSidewalk/RampNet/pull/227#pullrequestreview-5371717851): a
chance floor for the bearing test, and a fix to the in-view rule. The re-review
(https://github.com/ProjectSidewalk/RampNet/pull/227#pullrequestreview-5372192810) added
a count-matched floor and a co-visibility clustering sensitivity. Every number below is
from the revised `score`; the corrections are listed at the end of §4.

## Summary

- **The released checkpoint mostly does not transfer to perspective photos, and no input
  mapping fixes it.** On Richmond's flat Mapillary photos the canvas embed hits **0.199**
  of in-view pool ramps at 0.30 under the bearing test (292 image-ramp pairs, 106 ramps).
  **Detections taken from an unrelated photo hit 0.093 of the same pairs.** So the rate
  above chance is **0.106 [0.027, 0.194]**. At 0.55 it is 0.147 against 0.065, 0.083
  [0.022, 0.153] above chance.
  - **Against the stricter count-matched floor** (the donor photo also has as many
    detections as this one), the floor is 0.138 and the rate above it **0.061 [0.016,
    0.110]** at 0.30; 0.050 [0.013, 0.089] at 0.55. Either way it is above chance, and
    somewhere between half and two thirds of the 0.199 is chance.
- **The same ramps from Richmond's 360 panos** are hit 0.681 of the time under the same
  test at 0.55, against a chance floor of about 0.22 (detections rotated 90° / 180° /
  270°). **Above chance, per ramp, that is 0.097 for the flat photos vs 0.458 for the
  panos: −0.361 [−0.435, −0.281]** (105 ramps). Against the count-matched flat floor it is
  0.061 vs 0.458, −0.397 [−0.464, −0.329]. Without the floors the gap is −0.515.
- **Canvas embed vs naive stretch: no difference in ramps found.** Both rates are close to
  their floors, and the stretch's floor is higher because it fires more.
  - Above each arm's own floor, stretch − canvas is −0.009 [−0.061, 0.040] at 0.30
    (+0.013 [−0.030, 0.059] above the count-matched floors).
    Without the floors it is +0.024 [−0.026, 0.074].
  - The stretch fires much more on images with no pool ramp in view: 0.175 vs 0.072 at
    0.30, paired +0.103 [0.049, 0.163]. So its higher presence recall (0.422 vs 0.293) is
    mostly detections that are not at a known ramp.
- **Placing the photo at its SfM pitch and roll, instead of level, changes little**:
  +0.014 [0.000, 0.031] in point hits at 0.30 (+0.014 [−0.001, 0.031] above chance).
- **No evidence of a scale deficit.** The stretch shows ramps at 3.6–6.8× the trained
  angular scale across (p10–p90) and finds the same ramps as the canvas, which shows them
  at the trained scale. Arm (c), the canvas at twice the scale, was not run (§5).
- **The hit test's pose tolerance is not the explanation.** Doubling the lateral
  tolerance to 10 m moves the canvas rate 0.199 → 0.223 and its floor 0.093 → 0.115; above
  chance it is unchanged (0.106 → 0.108). Pose error can still move ramps into or out of
  the in-view denominator; that is not tested (§5).
- **Precision is not measured.** The pool is not a complete ramp inventory. 158 of the
  canvas arm's 290 detections at 0.30 match no pool ramp. The first card looked at by eye
  (`d001`) is on a real ramp with tactile paving that is not in the pool; that is one card,
  not a precision. The 190-card gallery for a human rater is
  `benchmark/richmond_flat_fp_218/gallery.html`: 150 unmatched detections plus 40 matched
  ones as a blind control. **It has not been rated.** `rates` will score it (§7).
- **Seoul is prepared, not scored.**
  - All 514 photos were fetched by filename with sha256, and both arms were run.
  - A blind presence gallery with a written rubric is ready.
  - There are no curb ramp labels yet, so there is no Seoul accuracy number.
  - The canvas arm fires on 9% of the photos at 0.30, the stretch on 32%.
- **Reading (proposed).** For Richmond flat imagery the input mapping is not the
  bottleneck; the model is. RampNet learned ramps as they appear in car-roof 360
  panoramas. Dashcam and phone frames differ in mount height, field of view and context,
  and image quality. A frozen-model fix is not on offer here, so Seoul would need more than
  an input arm.

## 1. Question and design

RampNet was trained, and is deployed, on 2048×4096 equirect panoramas. That is one angular
scale: 4096 px per 360°, about 11.4 px per degree. The question is whether the released
checkpoint, unchanged, finds curb ramps in ordinary perspective photos.

**Richmond is the controlled test.** Its flat (non-360) Mapillary images change only the
input. The city, the ramps and the camera-on-a-vehicle viewpoint stay the same, and the
same ramps have world GT and 360 captures. **Seoul is the stretch test.** It changes the
input, the viewpoint (a pedestrian 1 m up) and the ramp design (Korean flush crossings) at
once, and it has no curb ramp labels. So it gets only the preparation (§6).

## 2. Images and arms

**Images.** The #216 census (`analysis_out/flat_mapillary_3d/census/images.csv`) lists every
Mapillary image within 30 m of a Richmond pool ramp. Of its 1,422 flat images, the 1,353
perspective ones are used; the 69 fisheye frames are dropped (a different camera model).
All 1,353 were fetched as Mapillary's 2048-px thumbnails, 525 MB, with no failures, and
each has a sha256 in `fetched.csv`. Every image carries Mapillary's SfM pose
(`computed_rotation`, `computed_compass_angle`, `computed_geometry`) and SfM-refined
intrinsics (`camera_parameters` = [f, k1, k2]). The 193 images that #216 had fetched are a
subset: those facing a harness corner, which would have been nearly all positives.

- Horizontal FOV, p5/p50/p95: 48° / 69° / 103°.
- SfM pitch p5–p95: −12° to +10°; roll −8° to +6°. So "level" is an assumption with a
  real error, and the two canvas arms test it.
- Cameras: Garmin VIRB 271, unnamed 248, GoPro HERO11 194, moto x4 176, GoPro Max
  single-lens 171, then iPhones.
- The heading implied by `computed_rotation` equals `computed_compass_angle` to 1e-11°
  on all 1,353, which pins the rotation convention (world-to-camera, OpenCV axes). A test
  builds poses independently (heading, then pitch, then roll) and checks that the
  scorer's photo → world path and the `canvas_sfm` embed both return the true bearing and
  depression to 1e-6°.

**Arms.** All run the released checkpoint unchanged, fp32. Peaks come from the same
extraction as the benchmark: floor 0.10, `min_distance=10`, `exclude_border=False`.

| arm | input | what it tests |
|---|---|---|
| `canvas_level` (a) | the photo reprojected into a 2048×4096 equirect canvas at its true FOV (SfM focal + k1/k2), heading on the centre column, **camera assumed level** | the principled arm: at the photo's centre a ramp subtends the angle it would in a panorama |
| `canvas_sfm` (a2) | the same, with the photo placed at its SfM pitch and roll | whether the level assumption costs anything |
| `stretch` (b) | the photo resized straight to 2048×4096 (`threshold_sweep.PRE`) | the strawman |
| `canvas_x2` (c) | the level canvas at twice the angular scale (4096×8192 input) | arm (c); **not run** (§5) |

Canvas details:
- The canvas outside the photo is the ImageNet mean colour, zero after normalisation.
  Peaks there are dropped and counted (§4).
- A 70° photo spans about 796 canvas columns (a test asserts it).
- Every detection is mapped back to the photo pixel it came from, so every arm is scored
  with the same geometry. For a canvas arm this uses the exact ray the canvas sampled; for
  the stretch, the resize inverse.

**What the canvas matches, and what it does not** (`results.md`, "Angular sampling").
- **At the photo's centre it matches the pano path.** The thumbnails have 14–37 px/deg at
  their centre by camera model (median camera 25–26). The photo is first shrunk with PIL
  bilinear to the canvas's 11.4 px/deg at its centre, and is then sampled bilinearly
  (`grid_sample`). That is the angular sampling of an 11,000-px pano downsampled to 4096,
  so at the centre ramps sit at the trained scale with no extra blur from resolution.
- **Toward the side edges it does not.** The prescale uses the centre focal, and a
  perspective photo has more pixels per degree off-axis. At the side edge the canvas
  samples the shrunk photo 1.2–1.6× more coarsely than at the centre for most cameras, and
  2.5× for the iPhone 13's 102° lens, with bilinear sampling and no antialiasing. So the
  edges are mildly aliased, and many flat-photo ramps sit near the edges.
- **The photo is resampled twice** (the prescale, then `grid_sample` at about 1:1). That
  adds a little low-pass that the pano path does not have.
- **The stretch** shows the photo at 41 / 59 / 77 px/deg across and 22 / 42 / 78 down
  (p10 / p50 / p90), i.e. 3.6–6.8× and 2.0–6.9× the trained scale.

## 3. Ground truth and what a hit means

**World GT** is the 253 Richmond pool ramps (`census/ramps.csv`). Their positions come
from eval_sites' merged world GT (`docs/multiview_48.md` §3).
- **The pool is not a complete ramp inventory.** It holds the ramps reviewers confirmed
  or marked missed in 124 judged panos, and only those whose GT raycasts inside 25 m.
  Other real ramps sit within view of many of these photos.
- **So detections that match no pool ramp are not false positives by construction.**
  Precision is not reported from the pool. Those detections go to a gallery for a human
  rater (§7).

**In view.** A pool ramp is in view of a photo when:
- its horizontal range from the camera is 3–18 m (18 m is the multiview analysis's R);
- a flat-ground point at the ramp, seen from 1.5 m with the SfM pose, projects inside
  the frame, at least 3% of the width from either side edge and above the bottom edge;
- **and that point lies inside the distortion model's monotonic range.** The Brown model
  `r·(1 + k1 r² + k2 r⁴)` turns over for k2 < 0, and past that point rays far outside the
  lens project back into the frame. The first version of `score` lacked this check and
  counted 32 such pairs as in view (60–70° off-axis, on GoPro Max, HERO11, VIRB, moto x4
  and iPhone frames; e.g. image `1011298886069150`, a 36° half-FOV, with `richmond:29`
  66° off-axis). `P.fold_radius` / `P.in_distortion_domain` implement it, and a test
  checks a k2 < 0 camera with a ramp 66° off-axis.

Occlusion is not checked, here or in the pano reference.

**Image classes.**
- *Positive*: at least one pool ramp in view.
- *Pool-negative*: no pool ramp within 40 m whose bearing lies inside the FOV widened by
  10° on each side. This still does not mean "no ramp in the photo" (see above).
- Images that are neither (a ramp at 18–40 m in view, near the edge, or folded) are left
  out of both image-level rates.

**The hit test (primary): bearing.** The flat cameras' heights are unknown. Dashcams,
roof mounts and handheld phones are all present, and a flat-ground raycast at the wrong
height moves a point by metres. So the primary test does not fix a height. A detection
hits an in-view ramp at range d when both hold:
- its bearing (photo pixel → ray → world via the SfM pose) is within atan(5 m / d) of the
  ramp's bearing. That is eval_sites' 5 m match radius, expressed laterally. At 12 m it is
  ±22.6°, about two thirds of a 70° photo's width;
- it is below the horizon at a depression that puts it on flat ground for **some** camera
  height from 0.5 to 4 m: h = d · tan(depression).

Claims are one-to-one, greedy in descending score, and each detection takes the unclaimed
ramp (within 30 m) with the smallest bearing error.

**Chance floor.** Because the lateral window is wide, the bearing test scores hits for
detections placed at random. `score` measures that floor three ways and reports every
bearing number beside it:
- **Swap null (flat, primary floor).** Each positive image is scored with the detections
  of another positive image whose nearest pool ramp differs, at the same normalised pixel
  positions, projected through the receiving image's own camera and SfM pose. 20 draws
  (donors drawn with replacement, seeded), the same donors for every arm and threshold.
  Reported as the mean over draws with its p5–p95.
- **Count-matched swap null (flat, sensitivity).** The same, but the donor must also have
  the receiving image's number of detections ≥ 0.30 (bucketed 0 / 1 / 2 / 3+, per arm;
  seeded separately, 20 draws). **It is the more conservative floor for a localisation
  claim.** Images with more ramps in view also carry more detections (the re-review
  measured 0.63 vs 0.47 per image, pair-weighted), so the unmatched null gives each pair
  fewer chances to be hit by chance than the real detections had. Firing more where more
  ramps are is real presence skill, but not localisation. The count-matched null keeps how much the model fires on
  each image and randomises only where. The unmatched null stays the primary label
  "above chance" in this doc because it is the one the first revision reported; both are
  given for every headline number. The pano rotation null keeps each pano's own
  detections, so it is count-matched by construction.
- **Mirror null (flat).** Every candidate ramp's bearing is reflected about the camera
  heading. It runs higher than the swap null, probably because intersections put real
  ramps at mirror positions.
- **Rotation null (panos).** Every stored pano detection is rotated by 90°, 180° and 270°.

*Above chance* is the real hit minus its per-pair swap-null mean, with a paired cluster
bootstrap CI. The CI treats the null mean as fixed; the spread across draws is the
separate p5–p95.

**The height gate barely changes the real rate, but it cuts what chance scores by about a third.**
Without it (any detection below the horizon), canvas_level at 0.30 moves 0.199 → 0.202,
while its swap null moves 0.093 → 0.135. So the test is close to azimuth-only for real
detections, and the gate is still worth keeping.

**Sensitivity: world tests.** A flat-ground raycast at a fixed height, with the 5 m radius:
- 2.6 m, the labeler's per-rig value for Mapillary;
- 1.5 m.

**The same test on the panos.** On the panos the bearing test can be applied like for
like: the ramp's bearing is its projected column (`captures_R25.csv` → `x_proj`), and the
detection's depression is read off the level equirect. On the 2,440 non-source Richmond
captures at 3–18 m, at 0.55:
- bearing test 0.627 [0.585, 0.668], against a rotation null of 0.230 / 0.178 / 0.245;
  above chance 0.410 [0.372, 0.447];
- world test 0.514 [0.473, 0.554];
- the two tests agree on 84% of captures.

So the flat-vs-pano contrast that matters is **bearing vs bearing, at 0.55, each above its
own floor**. Only detections ≥ 0.55 are stored for the panos
(`benchmark/richmond_neighbourhood/records.jsonl`), so there is no bearing reference at
0.30. The pano world-test hit rate at both thresholds comes from
`analysis_out/multiview_48/captures_R25.csv` (`world_conf`; richmond's sub-0.55
re-inference included).

**Metrics.**
- *Presence recall*: the share of positive images with any detection ≥ thr, anywhere.
- *Localized recall*: the share of positive images where a detection hits an in-view ramp.
- *Fire rate on pool-negative images*: the share with any detection ≥ thr. This is not a
  false-positive rate (see above).
- *Point recall*: the share of (image, in-view ramp) pairs that are hit, overall and by range.

**CIs.** 95% percentile cluster bootstrap, 2,000 reps, seed 218.
- Images cluster by their nearest pool ramp; pairs cluster by ramp. Ramps seen in the
  same photo share one detection set, so clustering pairs by ramp is slightly optimistic.
  Clustering them instead by co-visible component (79 groups of ramps linked through a
  shared photo) widens canvas_level @ 0.30 to 0.199 [0.111, 0.300], above chance 0.106
  [0.016, 0.211] (count-matched 0.061 [0.017, 0.111]); every arm is in `results.md`.
- Arm contrasts are paired: the same resampled clusters for both arms.
- The flat-vs-pano contrast is per ramp: the ramp's flat pair-hit rate minus its pano
  capture hit rate, over ramps with both, resampling ramps. The above-chance version
  subtracts each side's floor per pair or capture first.

## 4. Results: Richmond (`results.md`, `results.json`)

**Counts.**
- 1,353 images: 249 positive, 709 pool-negative, 395 neither.
- 292 in-view (image, ramp) pairs of 106 ramps. The positive images cluster on 80 nearest
  ramps. The fold check removed 32 pairs.

**Image level** (95% CIs; images clustered by nearest pool ramp):

| arm @ thr | presence recall (249) | localized recall (249) | localized, swap null (p5–p95) | localized, above chance | fire rate, pool-negative (709) | detections / image |
|---|---|---|---|---|---|---|
| canvas_level @ 0.30 | 0.293 [0.193, 0.414] | 0.193 [0.117, 0.292] | 0.105 (0.076–0.134) | 0.088 [0.011, 0.184] | 0.072 [0.048, 0.099] | 0.21 |
| canvas_sfm @ 0.30 | 0.301 [0.207, 0.416] | 0.205 [0.126, 0.304] | 0.103 (0.068–0.122) | 0.102 [0.023, 0.198] | 0.072 [0.051, 0.097] | 0.22 |
| stretch @ 0.30 | 0.422 [0.311, 0.546] | 0.217 [0.139, 0.306] | 0.142 (0.115–0.177) | 0.075 [−0.004, 0.167] | 0.175 [0.125, 0.234] | 0.47 |
| canvas_level @ 0.55 | 0.181 [0.110, 0.268] | 0.141 [0.077, 0.217] | 0.073 (0.052–0.101) | 0.068 [0.008, 0.143] | 0.025 [0.013, 0.042] | 0.10 |
| canvas_sfm @ 0.55 | 0.185 [0.112, 0.274] | 0.149 [0.084, 0.228] | 0.075 (0.051–0.105) | 0.073 [0.011, 0.152] | 0.027 [0.015, 0.041] | 0.10 |
| stretch @ 0.55 | 0.229 [0.143, 0.332] | 0.137 [0.075, 0.214] | 0.082 (0.052–0.109) | 0.054 [−0.006, 0.129] | 0.062 [0.032, 0.098] | 0.19 |

**Point hits, bearing test, beside its chance floor** (292 in-view pairs, clustered by ramp):

| arm @ thr | real | swap null (p5–p95) | mirror null | above chance | count-matched swap null (p5–p95) | above count-matched |
|---|---|---|---|---|---|---|
| canvas_level @ 0.30 | 0.199 [0.123, 0.285] | 0.093 (0.068–0.114) | 0.127 | **0.106 [0.027, 0.194]** | 0.138 (0.113–0.154) | **0.061 [0.016, 0.110]** |
| canvas_sfm @ 0.30 | 0.212 [0.134, 0.301] | 0.092 (0.061–0.111) | 0.130 | 0.120 [0.040, 0.209] | 0.143 (0.120–0.161) | 0.069 [0.025, 0.119] |
| stretch @ 0.30 | 0.223 [0.147, 0.307] | 0.126 (0.098–0.158) | 0.171 | 0.096 [0.016, 0.185] | 0.148 (0.133–0.165) | 0.074 [0.022, 0.128] |
| canvas_level @ 0.55 | 0.147 [0.086, 0.218] | 0.065 (0.044–0.089) | 0.082 | **0.083 [0.022, 0.153]** | 0.098 (0.075–0.117) | **0.050 [0.013, 0.089]** |
| canvas_sfm @ 0.55 | 0.158 [0.093, 0.233] | 0.067 (0.044–0.096) | 0.079 | 0.090 [0.027, 0.165] | 0.104 (0.082–0.124) | 0.053 [0.017, 0.094] |
| stretch @ 0.55 | 0.144 [0.083, 0.215] | 0.072 (0.044–0.093) | 0.096 | 0.072 [0.012, 0.144] | 0.089 (0.068–0.110) | 0.054 [0.009, 0.104] |

Against the mirror null, which runs higher, the canvas arm is 0.07 above chance at 0.30
and 0.065 at 0.55; against the count-matched null, 0.061 and 0.050. So the flat photos
are **about 5–11 points above chance**, depending on the null and the threshold.

**By range** (bearing test, canvas_level @ 0.30):

| | 3–6 m (15 pairs) | 6–12 m (103) | 12–18 m (174) |
|---|---|---|---|
| real | 0.133 | 0.194 | 0.207 |
| swap null | 0.083 (0.000–0.200) | 0.119 (0.077–0.157) | 0.078 (0.046–0.121) |
| above chance | 0.050 [−0.121, 0.288] | 0.075 [−0.019, 0.210] | 0.129 [0.047, 0.220] |
| count-matched null | 0.100 (0.000–0.203) | 0.152 (0.117–0.186) | 0.132 (0.103–0.161) |
| above count-matched | 0.033 [−0.104, 0.225] | 0.042 [−0.003, 0.102] | 0.075 [0.018, 0.136] |
| panos, bearing test @ 0.55, same ramps | 0.778 | 0.753 | 0.615 |

The near bin has 15 pairs (2 hits). The other arms and 0.55 are in `results.md`.

**Sensitivities** (canvas_level @ 0.30; every arm in `results.md`):

| test | real | swap null | above chance |
|---|---|---|---|
| bearing (primary) | 0.199 | 0.093 | 0.106 [0.027, 0.194] |
| bearing, 10 m lateral | 0.223 | 0.115 | 0.108 [0.029, 0.199] |
| bearing, no height gate | 0.202 | 0.135 | 0.067 [−0.014, 0.160] |
| world, 1.5 m | 0.168 [0.099, 0.244] | not measured | – |
| world, 2.6 m | 0.079 [0.040, 0.126] | not measured | – |

**The same ramps from the 360 panos** (non-source Richmond captures at 3–18 m, 1,088
captures of 105 of the 106 ramps; world test, `captures_R25.csv`):

| | all | 3–6 m | 6–12 m | 12–18 m |
|---|---|---|---|---|
| pano @ 0.30 | 0.615 [0.556, 0.673] | 0.778 | 0.712 | 0.507 |
| pano @ 0.55 | 0.566 [0.508, 0.625] | 0.707 | 0.678 | 0.451 |

**Range mix is not a confound.** The flat pairs are 15 / 103 / 174 across the bins, the
pano captures of the same ramps 99 / 441 / 548, with medians of 12.8 m and 12.1 m. The
pano bearing test at 0.55 re-weighted to the flat range mix is 0.672, against 0.686
unweighted.

**Paired contrasts** (differences of means, cluster bootstrap):

| contrast | 0.30 | 0.55 |
|---|---|---|
| canvas_sfm − canvas_level, point hits | +0.014 [0.000, 0.031] | +0.010 [0.000, 0.023] |
| stretch − canvas_level, point hits | +0.024 [−0.026, 0.074] | −0.003 [−0.042, 0.034] |
| stretch − canvas_level, point hits, each above its own floor | −0.009 [−0.061, 0.040] | −0.011 [−0.052, 0.026] |
| stretch − canvas_level, point hits, each above its own count-matched floor | +0.013 [−0.030, 0.059] | +0.005 [−0.032, 0.042] |
| stretch − canvas_level, presence recall | +0.129 [0.056, 0.205] | +0.048 [−0.005, 0.109] |
| stretch − canvas_level, fire rate on pool-negative | +0.103 [0.049, 0.163] | +0.037 [0.006, 0.073] |
| canvas_level (bearing) − pano (world), per ramp | −0.400 [−0.484, −0.312] | −0.397 [−0.479, −0.314] |
| canvas_level − pano, both bearing, per ramp | – | −0.515 [−0.595, −0.433] (0.166 vs 0.681) |
| **canvas_level − pano, both bearing, each above its floor, per ramp** | – | **−0.361 [−0.435, −0.281]** (0.097 vs 0.458) |
| canvas_level − pano, both bearing, flat above its count-matched floor, per ramp | – | −0.397 [−0.464, −0.329] (0.061 vs 0.458) |
| stretch − pano, both bearing, each above its floor, per ramp | – | −0.378 [−0.451, −0.299] |

**Camera height implied by the matched detections: not evidence for a camera height.**
At 0.30, range × tan(depression) over the canvas arm's 132 bearing-matched detections
gives p10 / p50 / p90 = 1.19 / 1.69 / 2.65 m. That sample is narrow and largely chance:
- the 132 matches come from 26 sequences, and 84 of them from one camera model (GoPro
  HERO11, 9 sequences, median 1.91 m). VIRB gives 1.51 m (20 matches, 2 sequences),
  unnamed cameras 1.53 m (11, 9), moto x4 1.40 m (9, 2);
- **on the positive images, only the HERO11's matches clearly exceed chance**: 63 real
  matches against 10 per swap-null draw. For VIRB (2 vs 3.3), unnamed (8 vs 9.6) and
  moto x4 (6 vs 5.4) the real count is at the null's;
- **and chance matches imply the same heights.** The swap null's matches have a median of
  1.55 m overall and 1.93 m on HERO11 frames, against 1.69 m and 1.91 m for the real
  matches. A match's depression is set by where detections sit in the frame, inside the
  0.5–4 m gate, as much as by the camera.

So the diagnostic does not show that these rigs sit at 1.7 m. What stands is narrower:
a single fixed height is wrong for a mixed set of rigs, as the two world tests disagree
by 2× (0.168 at 1.5 m vs 0.079 at 2.6 m). A 2.6 m raycast, the labeler's Mapillary
value, would score these photos at half the 1.5 m rate; the labeler does not ingest
flat imagery today (#216), so this is about a possible future use, not a current bug.
That is why the bearing test is primary.

**Peaks in the canvas fill.** The canvas arms produced 476 peaks outside the photo, in the
grey fill, on 281 of 1,353 images, most at the photo's border. They were dropped before
scoring. They are a canvas artefact to keep in mind for any deployment; the photo's border
is an edge the model never saw in training.

**Geometry checks.** Of the canvas arm's 290 detections at 0.30, none sits at a photo
pixel the distortion model cannot invert, and none came from a canvas ray beyond the
model's fold. (`unproject_cam` does not invert the border of 47 images, 45 GoPro HERO11
and 2 VIRB, whose corners lie beyond the fold; it now returns NaN there.)

**What changed in the review revision** (first version → now; every change comes from
the fold check or the new chance floors, not from new detections):
- in-view pairs 324 → 292, ramps 110 → 106, positive images 268 → 249;
- canvas_level bearing @ 0.30 0.182 → 0.199, @ 0.55 0.136 → 0.147; canvas_sfm @ 0.30
  0.194 → 0.212; stretch @ 0.30 0.204 → 0.223;
- flat − pano, both bearing, per ramp −0.519 → −0.515 (0.150 vs 0.669 → 0.166 vs 0.681);
- the implied-height diagnostic is withdrawn as evidence for a camera height;
- new: every bearing number's chance floor and above-chance difference;
- new after the re-review: the count-matched swap null, and the co-visible-component
  clustering sensitivity.

## 5. Reading, with the caveats beside it

- **The flat photos are 5–11 points above chance; the panos about 40.** Between half and
  two thirds of the canvas arm's 0.199 at 0.30 is chance: 0.093 of it would be scored for
  detections taken from an unrelated photo, and 0.138 for detections from an unrelated
  photo on which the model fired as often. Any flat number here should be read against its floor.
- **Input mapping is not the lever.** The canvas and the stretch find the same ramps above
  chance, and both are near their floors. The stretch mostly adds detections away from
  known ramps. The pose-true canvas adds at most about 3 points (CI touches zero).
- **No evidence of a scale deficit.** This is weaker than a positive result.
  - The direct evidence is the stretch: it shows ramps at 3.6–6.8× the trained angular
    scale across and finds the same ramps as the canvas, which shows them at the trained
    scale at the photo's centre. A larger scale does not help.
  - The range pattern is not good evidence either way. The near bin has 15 pairs, and
    above chance the flat rate rises with range (0.05 / 0.075 / 0.13). Near ramps in a
    dashcam frame sit at the bottom edge, where the hood and the frame edge crop them, so
    range confounds scale with framing.
  - Arm (c), the canvas at twice the scale, was specified as "run if (a) loses small
    ramps". (a) does not lose them preferentially, and the stretch already covers larger
    scales, so (c) was not run.
- **Pose error.** Doubling the lateral tolerance adds nothing above chance, so pose error
  does not explain misses **in the hit test**. SfM heading and position error, and GT
  error (p90 4.4 m), also move ramps into or out of the **in-view denominator** (the 3%
  edge margin, the 18 m cut); the 10 m test does not touch that. The visual in-view pass
  (§7) is the remaining check.
- **What the comparison does not control.**
  - The flat and pano captures of a ramp are different drives, years and seasons: flat
    2018–2025, panos mostly 2024–25.
  - The flat cameras are lower and often behind a windshield, with the hood in frame.
  - Occlusion by parked cars is not checked for either, and a low camera is blocked more
    often than one 2.6 m up.
  - "Controlled" here means same city, same ramps, vehicle-mounted; it does not mean same
    image content.
- **In view is geometric, not visual.** A pool ramp "in view" was not checked by eye. A
  human pass over the 292 pairs would turn the recall denominator into "visible ramps"
  and would likely raise every flat number. It would raise the pano numbers too, but by
  less, since those were admitted by reviewers who saw them in a pano.
- **The pool is incomplete and GT points carry error** (p90 4.4 m, `docs/multiview_48.md`).
  So some misses are wrong GT and some unmatched detections are real ramps; the gallery is
  the way to measure the second.
- **One threshold family.** 0.30 and 0.55 are the pano operating points. Nothing was tuned
  for flat photos.

## 6. Seoul: prepared, not scored

**Fetch.** `seoul_photos_218.py manifest` records the Zenodo record's file list.
- `summary_attributes.csv` is committed, md5-checked against Zenodo, and pinned binary in
  `.gitattributes`.
- Each zip's member list, with sizes and CRCs, is in `seoul/files.csv`.

`fetch` reads the 514 photos named in the summary out of the three zips by HTTP range
reads. There is no full download. The zip CRC is checked, and each photo's sha256 is
recorded in `seoul/fetched.csv`. It took 3,000 s on makelab2 and read 6.75 GB for 4.4 GB
of photos (4 MB read blocks).
- The summary names some files `.HEIC` / `.JPG`; the zips hold `.jpg`. They match on the
  case-folded stem, 514 of 514.
- **The images are not redistributed.** Nothing under `seoul_imgs/` or the gallery's `img/`
  is committed.

**Camera assumptions.**
- The photos carry **no EXIF** (0 of 514; stripped in the HEIC→JPG conversion), so every
  photo uses the assumed **70° horizontal FOV**, pinhole.
- The camera is assumed **level**, as the protocol holds the phone.
- The **1.0 m** camera height is used only for the range column of each detection.
- The 34 photos at 4032×3024 may come from a different lens; that is unknown and not
  modelled.

**Output** (`seoul/dets_<arm>.jsonl`, `seoul/summary.json`; no labels, so no accuracy):

| arm | photos with a detection ≥ 0.30 | ≥ 0.55 | detections / photo @ 0.30 | ground range of detections @ 0.30 at 1.0 m, p10 / p50 / p90 |
|---|---|---|---|---|
| canvas_level | 0.093 | 0.021 | 0.11 | 2.1 / 3.2 / 7.8 m |
| stretch | 0.315 | 0.105 | 0.46 | 3.0 / 7.2 / 24.2 m |

The canvas arm put 18 peaks in the fill (dropped). Jon's 18-photo sample in the issue
estimated that 15–20% of photos hold a ramp-like crossing. That is an estimate, not a
label, so these rates cannot be read as recall or precision.

**Gallery for rating.** `benchmark/seoul_presence_218/gallery.html` has 514 cards and
manifest digest `a1484360ce2df9a6`.
- **Blind.** The page shows no model output.
- **The question:** "Is there a curb ramp in this photo?"
- **The answers:**
  - Curb ramp;
  - Flush crossing only;
  - No;
  - Can't tell.
- **The rubric is in the page and in every export.** It uses Project Sidewalk's definition
  ("a curb ramp connecting sidewalk to street").
  - **Flush crossings:** Project Sidewalk's labeling guide gives a crossing that is level
    with the street no Curb Ramp label, so the primary score counts "Flush crossing only"
    as no ramp.
  - It is a separate answer so the result can also be scored the other way:
    `seoul_photos_218.py rates` reports both.
- **Exports** are per rater: `seoul_presence__<rater>.json`, to be committed beside the
  gallery in `benchmark/seoul_presence_218/` (the Richmond gallery's exports go beside it
  the same way).
- The display copies (`img/`, 1,400 px long side) are not committed. `fetch` then `gallery`
  rebuilds them; a copy is also on makelab2 in
  `/homes/gws/jonf/persp218/RampNet/benchmark/seoul_presence_218/img/`.

## 7. What was not run, and why

- **Seoul accuracy.** It needs a human presence pass on the gallery above, which only Jon
  (or another rater) can do. Point-level Seoul scoring would need a second pass with rings,
  after the presence pass, so that pass stays blind.
- **Richmond precision.** The 190-card detection gallery is built and has not been rated.
  Until it is, no precision is claimed for any arm. "Fire rate on pool-negative images" is
  the only false-positive-side number, and it is a ceiling on the FP rate, not the rate.
  - **The scorer is written:** `perspective_photos_218.py rates --verdicts
    benchmark/richmond_flat_fp_218/richmond_flat_fp__<rater>.json`. It reports precision
    per stratum (150 of the 158 unmatched detections, 40 of the 132 matched) and weights
    the two back to the 290 detections, with a stratified bootstrap CI. Can't tell is
    excluded and counted.
  - **Card `d001` was viewed before rating** (the tactile-paving ramp in §Summary). The
    rater should know that; its verdict is not blind.
- **A second rater.** Both `rates` commands take several exports and report pairwise
  agreement (Cohen's kappa), and both refuse an export whose digest, item list, question,
  rubric or rules differ from the committed gallery's. No second rater has been asked.
- **Arm (c), native-scale tiles.** Its condition, that (a) loses small ramps, did not hold,
  and the stretch already shows ramps at several times the trained scale (§5).
- **A visual check of "in view".** The 292 recall pairs are geometric (§5).
- **The optional Seoul pairing with GSV / Mapillary panos at the same corners.** Coverage
  in Seoul was not checked.
- **Any retraining or fine-tuning** on perspective imagery: out of scope for a frozen-model
  test.
- **Bit-equality between machines.** Every detection here is from makelab2. A desktop smoke
  run (RTX 3070, the first 40 images, canvas sampled on the CPU) put the peaks in the same
  positions on 40 of 40 images for both canvas arms and 39 of 40 for the stretch, with
  scores within 1.1e-4. That is cross-machine fp32 noise, not bit-equality, and the smoke
  output is not used. The review revision changed only CPU-side geometry: windowed and
  full canvas maps are still identical on 100 real cameras (60 of them the non-inverting
  and slow-converging ones), both canvas arms, so the committed detections are what the
  current code would produce.

## 8. Reproduction

Everything after `infer` reads committed files only (`dets_*.jsonl`, the census, `captures_R25.csv`,
`benchmark/richmond_neighbourhood/records.jsonl`).

```bash
# Richmond. Desktop: the image list (committed census; no network), then the thumbnails.
python scripts/analysis/perspective_photos_218.py select
python scripts/analysis/perspective_photos_218.py fetch --env ../sidewalk-auto-labeler/.env --out IMG
# GPU (makelab2 A40; ~50 min with 4 shards sharing the GPU). Checks every sha256 first.
NSHARD=4 bash scripts/analysis/perspective_photos_218.sh IMG -
# CPU (about 1 min on a desktop; 10 s with --n-null 0, which skips the chance floors):
# every table in section 4 -> results.json / results.md / images_scored.csv
python scripts/analysis/perspective_photos_218.py score
# CPU: the detection gallery (needs IMG)
python scripts/analysis/perspective_photos_218.py gallery --images IMG
# after a rating pass (several files -> agreement too):
python scripts/analysis/perspective_photos_218.py rates --verdicts benchmark/richmond_flat_fp_218/richmond_flat_fp__<rater>.json

# Seoul. Any machine with network: zip directories, then the photos by range read.
python scripts/analysis/seoul_photos_218.py manifest   # only to re-check; files.csv is committed
python scripts/analysis/seoul_photos_218.py fetch --out SEOUL
bash scripts/analysis/perspective_photos_218.sh - SEOUL
python scripts/analysis/seoul_photos_218.py summary
python scripts/analysis/seoul_photos_218.py gallery --images SEOUL
# after a rating pass:
python scripts/analysis/seoul_photos_218.py rates --verdicts benchmark/seoul_presence_218/seoul_presence__<rater>.json

pytest -q tests/test_perspective_218.py
```

**Inputs that are not in the repo.**
- **Richmond thumbnails.** They are re-fetched by image id; `fetched.csv` has every sha256,
  and `infer --verify-sha` refuses a mismatch. Mapillary may re-encode or remove images, so
  a later re-fetch can differ. The set was complete on 2026-09-30 (1,353 of 1,353).
  Fetching needs a Mapillary token.
- **Seoul photos.** They are fetched from Zenodo by filename and checked against
  `seoul/fetched.csv`.
- **The model.** `projectsidewalk/rampnet-model` on Hugging Face, via
  `threshold_sweep.load_model`.

## 9. Cost and time

All free compute. `paid: false` rows are in `analysis_out/usage_log.jsonl` (issue 218).

| step | where | wall-clock | GPU-h |
|---|---|---|---|
| Richmond thumbnail fetch (1,353, 525 MB, 136 API calls) | desktop | 886 s | 0 |
| Richmond infer, first single-process run, aborted at ~50 images (CPU-bound, 5.8 s/image) | makelab2 A40 | ~390 s | 0.11 |
| Richmond infer, 3 arms × 1,353, 4 shards | makelab2 A40, shared with two other agents' jobs | 3,004 s (19:00:35–19:50:39Z) | 0.83 (the shard rows' elapsed / 4) |
| Seoul fetch (514, range reads) | makelab2, CPU | 2,997 s | 0 |
| Seoul infer, 2 arms × 514 | makelab2 A40 | 1,599 s (19:35:33–20:02:12Z) | 0.41 (upper bound; the ledger's per-arm seconds) |
| Seoul gallery display copies | makelab2, CPU | ~10 min | 0 |
| score, gallery, tests, smoke runs, review re-score | desktop (RTX 3070 for smoke runs only) | ~20 min | not logged: smoke only (≤ 0.05) |

**Total logged: 1.35 GPU-hours (1.3481) on makelab2, $0.** GPU-hours are upper bounds:
canvas building, peak extraction and the per-shard overlap are CPU time counted as GPU
time, and the A40 was shared with other jobs throughout.

**How the ledger rows were made.**
- **Shard rows.** Each shard process's GPU-hours are its elapsed time × its share of the
  GPU (1/4), so the 12 shard rows sum to about the run's wall-clock (0.83 h) rather than
  4× it. `usage_row` writes this itself for `--shard k/N` (`gpu_share` = 1/N, plus
  `concurrent_with` from `--concurrent-with`).
- **Regenerated from the raw rows.** The run itself used an earlier `usage_row` that wrote
  the undivided value. Those raw rows are committed as
  `analysis_out/perspective_photos_218/usage_rows_raw.jsonl` and `usage_rows_raw_seoul.jsonl`.
  The ledger rows were rebuilt from them with:

  ```bash
  python scripts/analysis/perspective_photos_218.py reledger \
    --raw analysis_out/perspective_photos_218/usage_rows_raw.jsonl \
    --replace-in analysis_out/usage_log.jsonl \
    --concurrent-with "the other three shards of this run" \
    --concurrent-with "other agents' makelab2 jobs (#221, #217) on the same A40"
  python scripts/analysis/perspective_photos_218.py reledger \
    --raw analysis_out/perspective_photos_218/usage_rows_raw_seoul.jsonl \
    --replace-in analysis_out/usage_log.jsonl \
    --concurrent-with "other agents' makelab2 jobs (#221, #217) on the same A40" \
    --concurrent-with "the Richmond shards until 19:50Z"
  ```

  The first version of these rows had been edited by hand to the same numbers.
- **The aborted-run row is hand-written** (the run was killed, so nothing wrote a row);
  its times are approximate and it says so.
- **Seoul's two walls.** The rows' "run wall 1589 s" is timed by the script from after
  the model loads; the 1,599 s above is the launcher log's start and end, which includes
  about 10 s of model loading.
