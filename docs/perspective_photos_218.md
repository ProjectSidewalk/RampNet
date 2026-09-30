# RampNet on perspective photos ([#218](https://github.com/ProjectSidewalk/RampNet/issues/218))

Code:
- `rampnet/perspective.py`: camera model, canvas embed, hit tests;
- `scripts/analysis/perspective_photos_218.py`: Richmond `select`, `fetch`, `infer`,
  `merge`, `score`, `gallery`;
- `scripts/analysis/perspective_photos_218.sh`: the GPU launcher;
- `scripts/analysis/seoul_photos_218.py`: Seoul `manifest`, `fetch`, `infer`, `gallery`,
  `rates`;
- `scripts/analysis/rating_page_218.py`: the shared one-question rating page;
- `tests/test_perspective_218.py`: geometry tests.

Outputs are in `analysis_out/perspective_photos_218/` (Richmond) and
`analysis_out/perspective_photos_218/seoul/`. The galleries are
`benchmark/richmond_flat_fp_218/` and `benchmark/seoul_presence_218/`.

Run 2026-09-30 on free compute (makelab2 A40, desktop for fetches and CPU scoring).

## Summary

- **The released checkpoint mostly does not transfer to perspective photos, and no input
  mapping fixes it.** On Richmond's flat Mapillary photos it finds **18% of in-view pool
  ramps at 0.30** (canvas embed, bearing test: 0.182 [0.117, 0.260]; 324 image-ramp pairs,
  110 ramps). The **same ramps seen from Richmond's 360 panos** are found **67% of the time**
  under the same test at 0.55 (0.669), against 15% from the flat photos. Per ramp that is
  **−0.52 [−0.59, −0.44]** (109 ramps).
- **Canvas embed vs naive stretch: no difference in ramps found.** The paired difference
  in point hits (stretch − canvas) is +0.022 [−0.019, 0.062] at 0.30 and −0.006
  [−0.043, 0.029] at 0.55.
  - The stretch fires much more on images with no pool ramp in view: 0.175 vs 0.072 at
    0.30, paired +0.103 [0.049, 0.163].
  - So its higher presence recall (0.418 vs 0.291) is mostly detections that are not at a
    known ramp.
- **Placing the photo at its SfM pitch and roll, instead of level, changes nothing**:
  +0.012 [−0.003, 0.028] in point hits.
- **The deficit is not about scale.** The flat photos lose ramps at every range, and not
  more at long range: 0.11 (3–6 m, n = 18), 0.17 (6–12 m), 0.19 (12–18 m). The panos
  lose them at long range: 0.76 / 0.70 / 0.49 at 0.30.
  - So the native-scale tiles arm (c), conditional on (a) losing small ramps, was not run.
- **It is not pose error either.** Doubling the lateral tolerance to 10 m adds 2 points
  (0.182 → 0.204).
- **Precision is not measured.** The pool is not a complete ramp inventory. 158 of the
  canvas arm's 290 detections at 0.30 match no pool ramp. The first one looked at by eye
  (`d001`) is on a real ramp with tactile paving that is not in the pool. The 190-card
  gallery for a human rater is `benchmark/richmond_flat_fp_218/gallery.html`: 150
  unmatched detections plus 40 matched ones as a blind control. **It has not been rated.**
- **Seoul is prepared, not scored.**
  - All 514 photos were fetched by filename with sha256, and both arms were run.
  - A blind presence gallery with a written rubric is ready.
  - There are no curb ramp labels yet, so there is no Seoul accuracy number.
  - The canvas arm fires on 9% of the photos at 0.30, the stretch on 32%.
- **Reading (proposed).** For Richmond flat imagery, the input mapping is not the
  bottleneck; the model is. RampNet learned ramps as they appear in car-roof 360 panoramas.
  Dashcam and phone frames differ in height (the matched detections imply a median camera
  height of 1.7 m, against 2.6 m for the pano rigs), in field of view and context, and in
  image quality. A frozen-model fix is not on offer here, so Seoul would need more than an
  input arm.

## 1. Question and design

RampNet was trained, and is deployed, on 2048×4096 equirect panoramas. That is one angular
scale: 4096 px per 360°, about 11.4 px per degree. The question is whether the released
checkpoint, unchanged, finds curb ramps in ordinary perspective photos.

**Richmond is the controlled test.** Its flat (non-360) Mapillary images change only the
input. The city, the ramps and the camera-on-a-vehicle viewpoint stay the same, and the
same ramps have world GT and 360 captures. **Seoul is the stretch test.** It changes the
input, the viewpoint (a pedestrian 1 m up) and the ramp design (Korean flush crossings) at
once, and it has no curb ramp labels. So tonight it gets only the preparation (§6).

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
  on all 1,353, which pins the rotation convention (world-to-camera, OpenCV axes).

**Arms.** All run the released checkpoint unchanged, fp32. Peaks come from the same
extraction as the benchmark: floor 0.10, `min_distance=10`, `exclude_border=False`.

| arm | input | what it tests |
|---|---|---|
| `canvas_level` (a) | the photo reprojected into a 2048×4096 equirect canvas at its true FOV (SfM focal + k1/k2), heading on the centre column, **camera assumed level** | the principled arm: a ramp subtends the angle it would in a panorama |
| `canvas_sfm` (a2) | the same, with the photo placed at its SfM pitch and roll | whether the level assumption costs anything |
| `stretch` (b) | the photo resized straight to 2048×4096 (`threshold_sweep.PRE`) | the strawman |
| `canvas_x2` (c) | the level canvas at twice the angular scale (4096×8192 input) | arm (c), run only if (a) loses small ramps; **not run** (§5) |

Canvas details:
- The canvas outside the photo is the ImageNet mean colour, zero after normalisation.
  Peaks there are dropped and counted (§4).
- The photo is first shrunk with PIL bilinear to about the canvas's scale, so the
  resampling does not alias. It is then sampled bilinearly (`grid_sample`, on the GPU).
- A 70° photo spans about 796 canvas columns (a test asserts it).
- Every detection is mapped back to the photo pixel it came from, so every arm is scored
  with the same geometry. For a canvas arm this uses the exact ray the canvas sampled; for
  the stretch, the resize inverse.

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
  the frame, at least 3% of the width from either side edge and above the bottom edge.

Occlusion is not checked, here or in the pano reference.

**Image classes.**
- *Positive*: at least one pool ramp in view.
- *Pool-negative*: no pool ramp within 40 m whose bearing lies inside the FOV widened by
  10° on each side. This still does not mean "no ramp in the photo" (see above).
- Images that are neither (a ramp at 18–40 m in view, or near the edge) are left out of
  both image-level rates.

**The hit test (primary): bearing.** The flat cameras' heights are unknown. Dashcams,
roof mounts and handheld phones are all present, and a flat-ground raycast at the wrong
height moves a point by metres. So the primary test does not fix a height. A detection
hits an in-view ramp at range d when both hold:
- its bearing (photo pixel → ray → world via the SfM pose) is within atan(5 m / d) of the
  ramp's bearing. That is eval_sites' 5 m match radius, expressed laterally;
- it is below the horizon at a depression that puts it on flat ground for **some** camera
  height from 0.5 to 4 m: h = d · tan(depression).

Claims are one-to-one, greedy in descending score, and each detection takes the unclaimed
ramp (within 30 m) with the smallest bearing error.

**Sensitivity: world tests.** A flat-ground raycast at a fixed height, with the 5 m radius:
- 2.6 m, the labeler's per-rig value for Mapillary;
- 1.5 m.

**The same test on the panos.** The bearing test is looser than the world test. On the
panos it can be applied like for like: the ramp's bearing is its projected column
(`captures_R25.csv` → `x_proj`), and the detection's depression is read off the level
equirect. On the same 2,440 non-source Richmond captures at 3–18 m, at 0.55:
- bearing test 0.627 [0.585, 0.668];
- world test 0.514 [0.473, 0.554];
- they agree on 84% of captures.

So the flat-vs-pano contrast that matters is **bearing vs bearing, at 0.55**. Only detections
≥ 0.55 are stored for the panos (`benchmark/richmond_neighbourhood/records.jsonl`), so there
is no bearing reference at 0.30. The pano world-test hit rate at both thresholds comes from
`analysis_out/multiview_48/captures_R25.csv` (`world_conf`; richmond's sub-0.55 re-inference
included).

**Metrics.**
- *Presence recall*: the share of positive images with any detection ≥ thr, anywhere.
- *Localized recall*: the share of positive images where a detection hits an in-view ramp.
- *Fire rate on pool-negative images*: the share with any detection ≥ thr. This is not a
  false-positive rate (see above).
- *Point recall*: the share of (image, in-view ramp) pairs that are hit, overall and by range.

**CIs.** 95% percentile cluster bootstrap, 2,000 reps, seed 218.
- Images cluster by their nearest pool ramp; pairs cluster by ramp.
- Arm contrasts are paired: the same resampled clusters for both arms.
- The flat-vs-pano contrast is per ramp: the ramp's flat pair-hit rate minus its pano
  capture hit rate, over ramps with both, resampling ramps.

## 4. Results: Richmond (`results.md`, `results.json`)

**Counts.**
- 1,353 images: 268 positive, 709 pool-negative, 383 neither.
- 324 in-view (image, ramp) pairs of 110 ramps. The positive images cluster on 85 nearest
  ramps.

**Image level** (95% CIs; images clustered by nearest pool ramp):

| arm @ thr | presence recall (268) | localized recall (268) | fire rate, pool-negative (709) | detections / image |
|---|---|---|---|---|
| canvas_level @ 0.30 | 0.291 [0.202, 0.407] | 0.183 [0.114, 0.280] | 0.072 [0.048, 0.099] | 0.21 |
| canvas_sfm @ 0.30 | 0.298 [0.212, 0.414] | 0.194 [0.124, 0.289] | 0.072 [0.051, 0.097] | 0.22 |
| stretch @ 0.30 | 0.418 [0.326, 0.527] | 0.205 [0.132, 0.297] | 0.175 [0.125, 0.234] | 0.47 |
| canvas_level @ 0.55 | 0.179 [0.109, 0.273] | 0.134 [0.078, 0.211] | 0.025 [0.013, 0.042] | 0.10 |
| canvas_sfm @ 0.55 | 0.179 [0.104, 0.279] | 0.138 [0.077, 0.220] | 0.027 [0.015, 0.041] | 0.10 |
| stretch @ 0.55 | 0.228 [0.149, 0.324] | 0.127 [0.069, 0.202] | 0.062 [0.032, 0.098] | 0.19 |

**Point hits** (324 in-view pairs, clustered by ramp):

| arm @ thr | bearing (primary) | bearing, 10 m lateral | world 1.5 m | world 2.6 m | 3–6 m (18) | 6–12 m (121) | 12–18 m (185) |
|---|---|---|---|---|---|---|---|
| canvas_level @ 0.30 | 0.182 [0.117, 0.260] | 0.204 | 0.154 | 0.071 | 0.111 | 0.174 | 0.195 |
| canvas_sfm @ 0.30 | 0.194 [0.126, 0.273] | 0.216 | 0.148 | 0.059 | 0.167 | 0.182 | 0.205 |
| stretch @ 0.30 | 0.204 [0.136, 0.280] | 0.241 | 0.148 | 0.096 | 0.056 | 0.190 | 0.227 |
| canvas_level @ 0.55 | 0.136 [0.082, 0.202] | 0.145 | 0.120 | 0.056 | 0.111 | 0.149 | 0.130 |
| canvas_sfm @ 0.55 | 0.142 [0.083, 0.214] | 0.151 | 0.120 | 0.046 | 0.111 | 0.149 | 0.141 |
| stretch @ 0.55 | 0.130 [0.075, 0.197] | 0.142 | 0.096 | 0.059 | 0.056 | 0.132 | 0.135 |

CIs for every cell are in `results.md`. The 3–6 m bin has 18 pairs, so its CI spans
about 0 to 0.3.

**The same ramps from the 360 panos** (non-source Richmond captures at 3–18 m, 1,129
captures of 109 of the 110 ramps; world test, `captures_R25.csv`):

| | all | 3–6 m | 6–12 m | 12–18 m |
|---|---|---|---|---|
| pano @ 0.30 | 0.601 [0.541, 0.661] | 0.757 | 0.700 | 0.494 |
| pano @ 0.55 | 0.554 [0.496, 0.613] | 0.689 | 0.667 | 0.440 |

**Paired contrasts** (differences of means, cluster bootstrap):

| contrast | 0.30 | 0.55 |
|---|---|---|
| canvas_sfm − canvas_level, point hits | +0.012 [−0.003, 0.028] | +0.006 [−0.004, 0.020] |
| stretch − canvas_level, point hits | +0.022 [−0.019, 0.062] | −0.006 [−0.043, 0.029] |
| stretch − canvas_level, presence recall | +0.127 [0.054, 0.199] | +0.049 [−0.009, 0.103] |
| stretch − canvas_level, fire rate on pool-negative | +0.103 [0.049, 0.163] | +0.037 [0.006, 0.073] |
| canvas_level (bearing) − pano (world), per ramp | −0.405 [−0.484, −0.317] | −0.401 [−0.475, −0.321] |
| **canvas_level − pano, both bearing, per ramp** | – | **−0.519 [−0.591, −0.443]** (0.150 vs 0.669) |
| stretch − pano, both bearing, per ramp | – | −0.519 [−0.593, −0.443] |

**Camera height implied by the matched detections.** At 0.30, range × tan(depression)
over the bearing-matched detections gives these p10 / p50 / p90:
- canvas_sfm: 1.12 / 1.62 / 2.67 m (137 detections);
- canvas_level: 1.19 / 1.69 / 2.65 m.

So most flat rigs sit well below the 2.6 m the labeler uses for Mapillary. That is why the
2.6 m world test scores these arms at half the 1.5 m one (0.071 vs 0.154). Neither
fixed-height test is right for a mixed set of rigs, and it is why the bearing test is
primary.

**Peaks in the canvas fill.** The canvas arms produced 476 peaks outside the photo, in the
grey fill, on 281 of 1,353 images, most at the photo's border. They were dropped before
scoring. They are a canvas artefact to keep in mind for any deployment; the photo's border
is an edge the model never saw in training.

## 5. Reading, with the caveats beside it

- **Input mapping is not the lever.** The principled canvas and the strawman stretch find
  the same ramps. The stretch mostly adds detections away from known ramps. The pose-true
  canvas adds nothing over level.
- **Scale is not the lever.** The flat photos' hit rate is flat across range while the
  panos' falls with range, so the flat deficit is largest close up: 0.11 vs 0.76 at 3–6 m,
  though on only 18 pairs.
  - Arm (c) was specified as "tiles at native angular scale, if (a) loses small ramps".
    (a) does not lose small ramps preferentially, so (c) was not run.
  - A larger-scale input remains untested. It would test "the model wants ramps bigger",
    which the range pattern argues against, but does not rule out.
- **Pose error is not the explanation.** A 10 m lateral tolerance adds 2 points.
  Presence recall, which needs no pose at all, is 0.29 at 0.30.
- **What the comparison does not control.**
  - The flat and pano captures of a ramp are different drives, years and seasons: flat
    2018–2025, panos mostly 2024–25.
  - The flat cameras are lower and often behind a windshield, with the hood in frame.
  - Occlusion by parked cars is not checked for either, and a camera 1.3 m up is blocked
    more often than one 2.6 m up.
  - "Controlled" here means same city, same ramps, vehicle-mounted; it does not mean same
    image content.
- **In view is geometric, not visual.** A pool ramp "in view" was not checked by eye. A
  human pass over the 324 pairs would turn the recall denominator into "visible ramps"
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
- **Exports** are per rater: `seoul_presence__<rater>.json`, to be committed under
  `analysis_out/perspective_photos_218/seoul/`.
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
- **Arm (c), native-scale tiles.** Its condition, that (a) loses small ramps, did not hold
  (§5).
- **A visual check of "in view".** The 324 recall pairs are geometric (§5).
- **The optional Seoul pairing with GSV / Mapillary panos at the same corners.** Coverage
  in Seoul was not checked.
- **Any retraining or fine-tuning** on perspective imagery: out of scope for a frozen-model
  test.
- **Bit-equality between machines.** Every number here is from makelab2. A desktop smoke run
  (RTX 3070, the first 40 images, canvas sampled on the CPU) put the peaks in the same
  positions on 40 of 40 images for both canvas arms and 39 of 40 for the stretch, with
  scores within 1.1e-4. That is cross-machine fp32 noise, not bit-equality, and the smoke
  output is not used.

## 8. Reproduction

Everything after `infer` reads committed files only (`dets_*.jsonl`, the census, `captures_R25.csv`,
`benchmark/richmond_neighbourhood/records.jsonl`).

```bash
# Richmond. Desktop: the image list (committed census; no network), then the thumbnails.
python scripts/analysis/perspective_photos_218.py select
python scripts/analysis/perspective_photos_218.py fetch --env ../sidewalk-auto-labeler/.env --out IMG
# GPU (makelab2 A40; ~50 min with 4 shards sharing the GPU). Checks every sha256 first.
NSHARD=4 bash scripts/analysis/perspective_photos_218.sh IMG -
# CPU (~5 min): every table in section 4 -> results.json / results.md
python scripts/analysis/perspective_photos_218.py score
# CPU: the detection gallery (needs IMG)
python scripts/analysis/perspective_photos_218.py gallery --images IMG

# Seoul. Any machine with network: zip directories, then the photos by range read.
python scripts/analysis/seoul_photos_218.py manifest   # only to re-check; files.csv is committed
python scripts/analysis/seoul_photos_218.py fetch --out SEOUL
bash scripts/analysis/perspective_photos_218.sh - SEOUL
python scripts/analysis/seoul_photos_218.py summary
python scripts/analysis/seoul_photos_218.py gallery --images SEOUL
# after a rating pass:
python scripts/analysis/seoul_photos_218.py rates --verdicts analysis_out/perspective_photos_218/seoul/seoul_presence__<rater>.json

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
| Richmond infer, 3 arms × 1,353, 4 shards | makelab2 A40, shared with two other agents' jobs | 3,004 s (19:00:35–19:50:39Z) | 0.83 (the ledger rows' elapsed / 4) |
| Seoul fetch (514, range reads) | makelab2, CPU | 2,997 s | 0 |
| Seoul infer, 2 arms × 514 | makelab2 A40 | 1,599 s (19:35:33–20:02:12Z) | 0.41 (upper bound; the ledger's per-arm seconds) |
| Seoul gallery display copies | makelab2, CPU | ~10 min | 0 |
| score, gallery, tests, smoke runs | desktop (RTX 3070 for smoke runs only) | ~15 min | not logged: smoke only (≤ 0.05) |

**Total logged: 1.35 GPU-hours on makelab2, $0.** GPU-hours are upper bounds: canvas
building, peak extraction and the per-shard overlap are CPU time counted as GPU time, and
the A40 was shared with other jobs throughout.
