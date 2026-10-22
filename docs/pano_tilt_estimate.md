# Image-based camera tilt for equirectangular panos (#159)

Issue: [#159](https://github.com/ProjectSidewalk/RampNet/issues/159) (the Lyon label audit).
Code: `rampnet/pano_tilt.py` (estimator), `scripts/analysis/pano_tilt_estimate.py` (validation,
Lyon run, summaries, visual check), tests in `tests/test_pano_tilt.py`. Outputs:
`analysis_out/pano_tilt/`. Run 2026-10-22 on JONFHOME (Windows, OpenCV 5.0.0, numpy 2.5.1).

## 1. Result

**The estimator validates on GSV.** On the 427 held-out GSV panos (the test three quarters of 571
panos from five GSV splits whose records carry Google's measured pitch and roll), it accepts 322
(declines **24.6%**) and on those it is off Google's pose by **MAE 0.23° in pitch and 0.25° in
roll** (pooled median 0.17°, p90 0.56°, max 1.72°); no accepted pano is more than 3° off, and 2.5%
are more than 1° off in either angle. Some of that residual is Google's pose, not the estimator
(section 3.4). The regression slope of the estimate on Google's value is 0.96 for both angles,
intercept 0.02° or less, so the sign conventions agree.

**On Lyon it accepts 193 of 200 panos, and many are far from level.** Median tilt (the angle
between gravity and the rig's up axis) is 3.3°, p90 12.3°, max 27.1°; 79 panos are tilted 5° or
more and 42 are tilted 10° or more. Almost all of that is GoPro Max: ecartip's GoPro Max panos
have a median tilt of **10.4°** (56 of 56 accepted at 2° or more, 34 at 10° or more), other GoPro
Max accounts 4.8°, the Grand Lyon Ladybug panos 1.7° (none at 10° or more), the other cameras 2.8°.
The 63 records whose metadata says exactly 0.0 / 0.0 have a median estimated tilt of 10.2°, so
that 0 / 0 is not a measurement.

**This output is a covariate for after the review.** It is not joined to any verdict, and
precision and recall by tilt are not computed here (section 5).

## 2. Method

`rampnet.pano_tilt.METHOD_VERSION = "pano_tilt v1 (lsd zenith-vp, 8 views, sector-sqrt ransac)"`.
CPU only (numpy, OpenCV, Pillow); about 3 s per pano single-threaded.

1. The equirect is read as grey and reduced to 4096 x 2048.
2. Eight gnomonic views are rendered around the horizon, one every 45° of bearing, each 60° wide
   and covering elevation −45° to +60° (the vehicle and most of the road stay out). In a pitch-0
   pinhole view, a gravity-vertical edge (building corner, pole, door frame) is an image column
   when the rig is level and leans toward the zenith vanishing point when it is not.
3. OpenCV's LSD line segment detector runs on each view. A segment is kept if it is at least 1.5°
   long, within 35° of the view's vertical, and its midpoint is in that view's own 45° sector
   (overlapping views do not count a segment twice).
4. Each segment's endpoint rays span a plane through the camera; its normal `n` is orthogonal to
   gravity's up direction `u` if the edge is vertical. RANSAC over segment pairs (`u = n_i x n_j`,
   within 40° of the rig's up axis) scores each hypothesis by **the sum over the eight sectors of
   the square root of the inlier weight in that sector** (weight = segment length, capped at 15°;
   inlier = `asin|n.u|` below 0.75°). The square root makes support from many directions worth
   more than the same total from one direction, which is what a fence, an oblique facade or a tree
   produces. The winner is refined as the smallest-eigenvalue eigenvector of `sum w n n^T` over the
   inliers (0.35°).
5. `u` is converted to (pitch, roll) by inverting `rampnet.stage1_geometry.rig_from_gravity`, so
   the output is in the #113 convention (`docs/crop_tilt_113.md` §2) without redefining it:
   **pitch > 0 is nose down, roll > 0 is left side up**, and the gravity horizon at bearing
   `b = x / w * 360 - 180` sits `T(b) = pitch cos b + roll sin b` degrees above the image's middle
   row. `tests/test_pano_tilt.py` round-trips the inversion against `rig_from_gravity` and checks
   that `u` is orthogonal to the horizon `gravity_to_rig` produces.

**Self-reported reliability.** Every pano gets `status` `ok` or `declined` with the reason, and the
raw estimate is kept for declined panos so the gate can be re-tuned from the rows. The gate (all
must hold): at least 12 inlier segments; at least 3 of the 8 sectors holding 3 or more inliers; a
formal standard error of pitch and roll of 0.25° or less; and an **ambiguity** of 0.8 or less, the
score of the best RANSAC hypothesis at least 3° from the fit divided by the fit's own score. The
ambiguity term is the one that matters: in the validation it is among the reasons for 134 of the
137 declined panos (inliers 3, sectors 18; a pano can fail more than one test), and the formal SE
did not flag the one gross failure in the development quarter (roll 20° off Google's pose with a
formal SE of 0.04°, but an ambiguity of 1.08). Each row also carries the inlier count per sector and
the residual RMS.

`level_equirect(gray, pitch, roll)` re-renders a pano gravity-level with the estimate (it is the
inverse of `tilt_equirect`, tested). Nothing in the detection pipeline calls it yet.

## 3. Validation against Google's measured pose

### 3.1 Data

`analysis_out/pano_tilt/validation_ids.tsv` lists every pano of the five GSV benchmark splits
whose records carry `camera_pitch` and `camera_roll`: bend 110, gainesville 125, laurens_gsv 86,
paterson 125, sao_paulo 125 = **571**. All 571 have numeric values; none is a default (no exact
zeros; median |pitch| 0.72-1.13° and median |roll| 0.86-1.07° per split; roll is served in
[0, 360) and wrapped). The other splits with local panos do not qualify: annapolis, clovis,
morgantown, richmond, budapest_district5 and laurens_mapillary are Mapillary with no pose, and
`manual_gold`'s records carry no pose fields at all. Every image's sha256 matched its split's
`imagery_manifest.json` (0 mismatches). Google's tilts in this set run up to 8.25°.

The `role` column fixes a **development quarter** (`sha256(pano_id) % 4 == 0`, 144 panos) and a
**test set** (the other 427). The scoring rule, segment-length floor, inlier tolerance and the
whole decline gate were chosen on the development quarter. One disclosure: before that split was
in place, a first comparison of RANSAC scoring rules (plain inlier weight against per-sector
square root and caps) was looked at on all 571 panos, so the scoring rule was not chosen blind to
the test panos. Every later choice was.

### 3.2 Accuracy (`validation_summary.json`)

Error = estimate minus Google, per angle. "Worst" is the larger of the two angles' errors.

| set | n | accepted | declined | MAE pitch | MAE roll | median abs | p90 abs | max abs | worst > 1° | worst > 3° |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test | 427 | 322 | 24.6% | 0.23° | 0.25° | 0.17° | 0.56° | 1.72° | 2.5% | 0 |
| dev | 144 | 112 | 22.2% | 0.19° | 0.21° | 0.14° | 0.42° | 1.82° | 0.9% | 0 |
| test, no gate | 427 | 427 | — | — | — | 0.21° | 0.86° | 30.2° | 11.7% | 2.8% |

The ungated row is every pano's raw estimate. Without the gate, 12 of 427 test panos are more than
3° off (a wrong vanishing point, up to 30°); with it, none of the 322 accepted are. The declined
test panos with an estimate are off by MAE 1.25°, p90 2.1°, so the gate removes the bad ones, not
random ones. It does not select easy panos either: the median Google tilt is 1.61° on accepted and
1.62° on declined test panos.

By split (test panos only):

| split | n | declined | MAE pitch | MAE roll | p90 abs | max abs |
|---|---:|---:|---:|---:|---:|---:|
| bend | 73 | 28.8% | 0.30° | 0.41° | 0.81° | 1.50° |
| gainesville | 91 | 45.1% | 0.24° | 0.24° | 0.63° | 1.33° |
| laurens_gsv | 73 | 23.3% | 0.28° | 0.29° | 0.56° | 0.92° |
| paterson | 93 | 18.3% | 0.21° | 0.20° | 0.42° | 1.72° |
| sao_paulo | 97 | 9.3% | 0.17° | 0.17° | 0.40° | 1.35° |

Declines follow the scenery: dense São Paulo streets 9.3%, Gainesville's tree-canopied streets
45.1%. Rural Laurens (GSV) is not worse than the cities once accepted.

Error does not grow with tilt over the range GSV covers (accepted test panos, binned by Google's
tilt): MAE 0.23° below 1° (n 68), 0.25° at 1-2° (143), 0.23° at 2-3° (64), 0.24° at 3-5° (42), 0.16°
at 5° and above (5).

### 3.3 Sign and scale

OLS of the estimate on Google's value over the 434 accepted panos: pitch slope 0.97, roll slope
0.98, intercepts 0.014° and 0.008° (test only: 0.96 and 0.96). A sign error would give −1. The few
percent below 1 are what regression dilution from noise in Google's own value would produce; this
run cannot tell that apart from a small scale bias in the estimator.

### 3.4 Google's pose is not ground truth

Looking at the largest disagreements with `inspect` (one view levelled by Google's pose next to
the same view levelled by the estimate, plumb lines every 40 px): on
`gainesville/DSID5BsZ19qz5BuvC_ZDNA` (184 inliers in all 8 sectors; Google roll −2.71°, estimate
−4.53°) the brick pier and window columns lean visibly under Google's pose and stand plumb under
the estimate. On `gainesville/GRTC3zOCFSc8QMIrAtwtzw` (Google −1.44 / 0.25, estimate 0.88 / −1.54)
the stop-sign post and the building corner lean under Google's pose and are vertical under the
estimate (this pano is declined by the gate, ambiguity 0.84, so it is not in the accepted numbers).
In both, Google's stored pose is the one that is off. So the errors in section 3.2 are an
upper bound on the estimator's error. Only these two panos were inspected; how much of the 0.23°
MAE is Google's is not measured.

### 3.5 Does it beat 1° MAE?

Yes, by about a factor of four on accepted panos (0.23° and 0.25°), at the cost of declining a
quarter of GSV panos. The tilts it was validated on are at most 8.25°. Lyon has many larger tilts;
for those the evidence is the synthetic tests (known tilts up to 24.7° pitch and 9° roll recovered
within 0.25°), the agreement with Panoramax's own estimator (section 4.3), and visual checks
(section 4.4), not a measured pose.

## 4. Lyon (`lyon.json`, `lyon_summary.json`)

The 200 panos of the review bundle, read from `D:\Git\RampNet-wt\lyon159\benchmark\lyon\panos\`
without writing anything there; every image's sha256 matched `benchmark/lyon/imagery_manifest.json`.
Groups are `benchmark/lyon/group_map.json`'s (the test checks they equal the bundle's own).
Tilt = `acos(cos pitch cos roll)`.

### 4.1 What the metadata says

The records' `camera_pitch` / `camera_roll`: **125 absent, 63 exactly 0.0 / 0.0, 12 non-zero**.
Of the 12, 11 are **Panoramax's own image-based estimate**: the record's `semantics` name
`computed_by[pitch] = panoflat/0.1.0` (served as `pers:pitch` / `pers:roll`). The twelfth is a GoPro
MAX2 whose XMP `PosePitchDegrees` is −0.8. An XMP pose is present on 83 records: the MAX2's, and
82 that read exactly 0.0 / 0.0, among them the GoPro Max panos Panoramax rated at −24.7° and 11.8°. (This differs from the counts
#159 started from, 125 / 72 / 3: by these fields it is 125 / 63 / 12.)

### 4.2 Tilt by group and by camera (accepted panos)

| group | n | accepted | median tilt | p90 | max | median abs pitch | median abs roll | ≥ 2° | ≥ 5° | ≥ 10° |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ecartip_gopro_max | 57 | 56 | 10.4° | 19.7° | 27.1° | 9.6° | 3.1° | 56 | 54 | 34 |
| other_gopro_max | 35 | 33 | 4.8° | 9.7° | 25.2° | 2.6° | 1.4° | 26 | 14 | 4 |
| grand_lyon_ladybug | 75 | 72 | 1.7° | 3.4° | 7.9° | 0.7° | 1.1° | 28 | 3 | 0 |
| other_cameras | 33 | 32 | 2.8° | 10.7° | 17.3° | 1.5° | 1.9° | 22 | 8 | 4 |
| **all** | 200 | 193 | 3.3° | 12.3° | 27.1° | 1.9° | 1.6° | 132 | 79 | 42 |

| camera model | n | accepted | median tilt | p90 | max |
|---|---:|---:|---:|---:|---:|
| GoPro Max | 92 | 89 | 9.4° | 16.6° | 27.1° |
| Ladybug | 41 | 40 | 1.7° | 3.8° | 7.9° |
| (no make/model; Grand Lyon and bike-route batches) | 40 | 37 | 1.9° | 8.1° | 17.3° |
| QooCam 3 | 12 | 12 | 2.2° | 3.0° | 3.3° |
| insta360 x3 | 6 | 6 | 3.2° | 6.6° | 7.2° |
| VIRB 360 | 5 | 5 | 3.3° | 4.2° | 4.7° |
| THETA S | 2 | 2 | 2.4° | — | 2.6° |
| GoPro MAX2 | 1 | 1 | 1.9° | — | 1.9° |
| insta360 x5 | 1 | 1 | 6.1° | — | 6.1° |

The GoPro Max tilt is mostly pitch (median |pitch| 8.7° against |roll| 2.4°), consistent with a
camera on a helmet or handlebar pitched forward or back and the camera not levelling the image.
By metadata: the 63 "0.0 / 0.0" panos have median tilt 10.2° (61 of 62 accepted at 2° or more); the
125 with no pose 2.1°. The decline rate on Lyon is 3.5%, 7 panos (reasons: ambiguity 7, inliers 1,
sectors 1), far lower than on GSV; these are dense city streets.

### 4.3 Sanity check against the 12 non-zero metadata panos

All 12 are accepted. Against Panoramax's panoflat values (11 panos), the estimate agrees **with
pitch's sign flipped and roll's sign kept**: MAE 0.43° in pitch and 0.30° in roll, median abs 0.22°.
With the signs as served, pitch is off by MAE 13.0°. So Panoramax's `pers:pitch` is positive nose
up, the opposite of #113's convention, and its roll matches. The one larger gap is
`97e7f15b` (panoflat pitch 9.0°, estimate −12.3°, a 3.3° difference after the flip). The pano
#159 flagged at pitch −24.7° (`2b8c9798`, other_gopro_max) is estimated at **+25.2° pitch,
−0.4° roll**; levelled by that estimate its buildings stand vertical (section 4.4). The GoPro MAX2
(`d113e602`) is the one that disagrees: XMP pitch −0.8° / roll 0.0°, estimate −0.07° / −1.85°.
Two image estimators agreeing is not a measurement, but they are different methods.

### 4.4 Visual checks

Rendered with a scratch variant of `inspect` (raw against estimate-levelled, four views each):
`2b8c9798` (25.2°) and `0251def3` (ecartip, 17.0°, metadata 0 / 0) both show strongly leaning
facades in the raw pano and vertical facades and a horizontal horizon after levelling. These
images are not committed; `inspect --split lyon --panos-root <dir holding lyon/panos>` reproduces
the three-panel version for any pano.

## 5. What this is for, and what was not done

- **Not joined to verdicts.** `lyon.json` carries id, pitch, roll, tilt, status, reason, the
  confidence fields, method version and image sha256, and nothing from the review. Splitting
  Lyon precision and recall by tilt waits until Jon's review is final.
- **Levelling before detection is not tested.** The Lyon numbers say a large share of open-imagery
  panos (GoPro Max in particular) are tilted 5-25°, far beyond GSV's 0-8°, so whether levelling
  helps detection is now a real question. Answering it needs a run of the detector on levelled
  and unlevelled copies of a scored split with tilted rigs (Bayonne's GoPro Max panos are the
  obvious candidate); not run here.
- **Mapillary splits not run.** Their records carry no pose, so there is nothing to validate
  against, and they were outside this task.

## 6. Caveats

- **Reference noise.** Google's pose is itself off by a degree or two on some panos (section 3.4),
  so the validation error is an upper bound and the 2.5% of accepted test panos more than 1° off
  are not all estimator failures.
- **Validated range.** Measured accuracy covers tilts up to 8.25°. Beyond that the support is
  synthetic and the panoflat agreement, not a measured pose.
- **Scenery dependence.** The method needs vertical structure in several directions. It declined
  45% of Gainesville's test panos; on a split of open rural road with few poles it would decline
  more. A declined pano has no usable tilt; its raw estimate is in the row for re-gating only.
- **Gate tuned on 144 panos.** The development quarter held one gross (> 3°) failure before the
  gate; the gate's thresholds are therefore loosely constrained, and the held-out result (no gross
  failure among 322 accepted) is the evidence that they generalise.
- **Equirect assumption.** The input must be a full 2:1 equirect with the camera's own frame
  (centre column = bearing 0). All 200 Lyon panos and all 571 GSV panos are.
- **Bit-level reproducibility.** LSD's output can differ slightly between OpenCV builds, so a re-run
  elsewhere reproduces the rows closely but not byte for byte; the size of that difference was not
  measured (only one machine was used). On this machine the committed run matched the development
  pass exactly (112 of 144 dev and 322 of 427 test accepted in both). `summarize` is exact.

## 7. Gaps

- **laurens_gsv panos are not on Hugging Face** (`docs/replication.md`): its 86 rows can only be
  re-run from a local copy; the other four splits are in `projectsidewalk/rampnet-benchmark`.
- **The Lyon panos are not on Hugging Face.** They are public on Panoramax and pinned by
  `benchmark/lyon/imagery_manifest.json`; the run read a local staged copy.
- **No second human check** of the visual comparisons; two GSV and two Lyon panos were looked at.
- **Panoramax's convention** for `pers:pitch` is inferred from 11 panos, not from documentation.

## 8. Reproduce

```bash
# id list (checks against the committed one without --write)
python scripts/analysis/pano_tilt_estimate.py ids
# GSV validation: needs benchmark/<split>/panos/ for the five splits (HF rampnet-benchmark,
# laurens_gsv local only); about 9 min with 12 workers
python scripts/analysis/pano_tilt_estimate.py validate --panos-root benchmark --workers 12
# Lyon: the 200 bundle panos, read-only
python scripts/analysis/pano_tilt_estimate.py lyon --panos-dir D:/Git/RampNet-wt/lyon159/benchmark/lyon/panos --workers 12
# re-derive the id list and both summaries from the committed rows; exits 1 on any byte difference
python scripts/analysis/pano_tilt_estimate.py check
# the section 3.4 visual check
python scripts/analysis/pano_tilt_estimate.py inspect --split gainesville --pano-id DSID5BsZ19qz5BuvC_ZDNA --view 6 --out inspect.jpg
# tests: synthetic tilted panos, conventions, and the committed rows (CPU, no network)
pytest -q tests/test_pano_tilt.py
```

Cost: CPU only on a desktop, no paid model or cluster: 8.7 min wall-clock for the 571 GSV panos
and 3.2 min for Lyon's 200, with 12 worker processes (from the log files' timestamps).
