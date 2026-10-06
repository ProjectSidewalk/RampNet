# Rig tilt in the Stage 1 crop model's training targets (#113, measurement half)

Issue: [#113](https://github.com/ProjectSidewalk/RampNet/issues/113). Script:
`scripts/analysis/crop_tilt_113.py`. Outputs: `analysis_out/crop_tilt_113/`. Run 2026-10-05/06;
revised 2026-10-06 after the review on PR #244 (section 10 lists every correction).
Scope: the measurement half only. The architectural half of #113 (consume
sidewalk-panorama-tools' cropper) is not touched, and no training set is regenerated or corrected.

## 1. Result

Over the 14,505 vouched CurbRamp labels in the 12 deployments `download_data.py` reads that
carry a committed pose and pass its `agree - disagree >= 2` filter, rig tilt moves the round-1
training target off its object by **0.24 sigma on average in y (median 0.18, p90 0.50, p99 1.06)**
at beta = 1, and 0.21 sigma (p90 0.45) at the measured beta = 0.90. 10.1% of targets are more than
0.5 sigma off in y and 1.3% more than 1 sigma (7.7% and 0.9% at beta = 0.90). The x term is small:
mean 0.07 sigma, p90 0.15. Labels from the other cities in the pool (3,590, never in round 1) are
displaced the same way (mean 0.24 sigma, p90 0.51). The displacement is a sinusoid in heading as
#113 predicted, but **not zero-mean over headings**: the fleet's mean roll is about +0.5 degrees,
which leaves a heading-locked sinusoid of amplitude 9.8 render px (0.10 sigma) in the bin means.

**The model response is measured on panos drawn from the crop model's own training population.**
An estimated 48% of the 600 sampled labels are themselves round-1 crops (53.0% match a round-1
keypoint exactly against a 10.4% chance rate, so (0.530 - 0.104) / (1 - 0.104) = 0.48; held-out
cities match at chance). Run through the released checkpoints, **the round-1 crop model follows the
object only part of the way**: its peak moves by 0.47 [0.42, 0.52] of the analytic displacement in
y over the whole sample (slope 0 would be "reproduces the displaced target", 1 "sits on the
object"), **0.53 [0.45, 0.60] on the 282 unmatched labels** (not found among round-1 crops; the
test's sensitivity is unmeasured, so some may still be round-1 crops), and 0.38 [0.30, 0.46] on
the 233 that match a round-1 train crop, a difference of 0.15 (SE 0.056, z 2.7). **Round 2's manual
fine-tune moves it further toward the object**: 0.71 [0.66, 0.76] overall, 0.73 [0.66, 0.80] on
unmatched labels, a paired gain over round 1 of 0.24 [0.19, 0.29]. These
slopes come mostly from the |T| >= 3 degree stratum and vary with the search window (section 5);
read the round-2 y slope as about 0.65-0.85, not as one number. In x the slopes are about 0.4 for
both rounds, weakly identified because the x displacement is small.

**On 565 labels from 22 cities the crop model never trained on, the slopes are about the same as
on the unmatched in-population labels** (section 5.4): round 1 0.44 [0.34, 0.53] in the stored
window, 0.59 [0.52, 0.67] in the midpoint window; round 2 0.64 [0.51, 0.77] and 0.86 [0.79, 0.93].
So the "about half way" that remains for round 1 after the training-crop overlap is set aside is
not explained by missed round-1 labels or panos seen through other labels. Re-localizing on the
unclipped heatmap moved no in-population slope by more than 0.03.

## 2. What was already settled elsewhere (not re-measured here)

- **Frame.** sidewalk-panorama-tools' tilt error study (its PR #158, report
  `reports/2026-09-26-tilt-error-study.md`, commit `21d10aa`) established that Google's depth planes
  and the served GSV tiles are in the camera rig's frame (F1, F2: raw lean slopes exclude 0 by at
  least 5.5 SE in every arm), and that the labelled feature sits at the rig pixel, not at the stored
  `pano_y` (endpoint C: Jon, blind, 79 : 0, p = 1.7e-24). Sign: pose pitch > 0 is nose down, roll > 0
  is left side up, `T(b) = pitch cos b + roll sin b` with `b = pano_x / w * 360 - 180` (which is
  exactly `download_data.py`'s `theta`), and the rig pixel is `pano_y - T * h / 180` to first order.
- **Beta**, the share of `T` the stored `pano_y` is off by: 0.902 (SE 0.010) pooled over 14,878
  RampNet-detection vs human-label pairs (sidewalk-auto-labeler #113; 0.953 under npz poses; by era
  legacy 0.88, mid 0.91, post179 0.94); pano-tools' human beta batch (#191) gives 0.93 [0.86, 0.99],
  close to 1 but with a CI that just excludes it. This doc reports beta = 1 (ceiling) and beta = 0.90
  (measured) and does not re-estimate beta.
- Two of the issue's four measurement boxes are therefore answered elsewhere: the frame question
  for the tile stitch (F2), and the per-pano pose source (the committed pano-tools pose scan, no
  `streetlevel` calls needed).

## 3. The unit chain

`download_data.py` renders a 2048 x 2048, 90-degree perspective view at depression 30 degrees,
keeps columns 682..1364, and writes the projected label point in **render px**. `train.py` halves
both coordinates into the 1024 x 352 input (`adj = orig * 0.5`), the head is 4x down (256 x 88), and
the target Gaussian has sigma = 12 heatmap px. So:

| unit | sigma |
| :--- | ---: |
| heatmap px | 12 |
| input px | 48 |
| render px | **96** |
| degrees of elevation, strip centre (17.87 render px/deg) | 5.37 |
| degrees, label median (19.40 render px/deg; 93.2% of labels sit above the strip centre, median strip y 746 of 2048) | 4.95 |
| px on the paper's 8192 x 4096 pano (22.76 px/deg), strip centre | 122 |

`evaluate.py`'s match radius is 0.132 x 341/4 = 11.25 heatmap px = 90 render px, so the radius is
about 0.94 sigma.

**The issue's "0.2-0.6 sigma at 1-3 degrees of tilt" is confirmed.** At the strip centre 1 and 3
degrees of `T` are 0.19 and 0.56 sigma; at the label median scale they are 0.20 and 0.61 sigma.

**Correction to pano-tools' S3.** S3 (`s3_miscentering` in
`reports/data/2026-09-26-tilt-error-study.json`) reads "7-9 click sigmas for RampNet stage one"
from a sigma of 12 px on a 4096-high pano. Sigma is 12 *heatmap* px, which is 96 render px. In
degrees that depends on where in the strip the band sits, so matching each S3 band's depression
(`summary.json` `unit_chain.s3_by_depression_band`):

| S3 band | depression | sigma (deg) | S3 understates sigma by | S3's p90 shift in sigma |
| :--- | ---: | ---: | ---: | ---: |
| < 5 | 3.7 | 4.32 | 8.2x | 1.12 |
| 5-15 | 10.8 | 4.79 | 9.1x | 0.91 |
| 15-30 | 19.5 | 5.19 | 9.8x | 0.77 |
| > 30 | 34.0 | 5.35 | 10.1x | 0.71 |

So S3 understates sigma by 8-10x, and its p90 shifts are 0.7-1.1 sigma, not 7-9. That correction
belongs on sidewalk-panorama-tools #54 and has not been posted there yet.

**A related quirk, recorded and not acted on (corrected after review, see section 10).** `train.py`
scales target x by 0.5, but the 683-wide strip is resized to 352 (0.515), so an unflipped target
sits 3% of its distance from the left edge too far left. Both `train.py` files also flip
horizontally with p = 0.5, and in a flipped crop the same error points the other way. Averaged over
flips the learned target is pulled toward the crop's centre column. **Corrected in the second
review round:** the first revision compared its prediction in the wrong frame. The response run
measures x error as `(peak - stored) - d_x`, converting peaks to render px at x0.5, the same scale
as `train.py`'s targets. In that frame a peak on the flip-averaged training target predicts a
constant +10 render px with slope 0 on strip x, and a peak exactly on the object predicts intercept
0 and slope +0.031 (the image's true x scale is 0.515, not 0.5). Measured (section 5.3): intercepts
+16.3 (SE 3.2) and +15.3 (SE 3.1) render px at strip x 0, slopes -0.029 (SE 0.008) for round 1 and
-0.015 (SE 0.008) for round 2. At the strip's centre column (x 341) that is about +10.2 for round 2,
close to the flip-averaged target, and about +6.4 for round 1, between the two hypotheses. Those
slopes are not agreement with either hypothesis: they are an extra pull toward the centre column,
stronger in round 1.

## 4. Analytic displacement

Population (`summary.json` `funnel`): of the 70,422-label pano-tools pool, 27,701 are CurbRamp
(all GSV), 21,412 are in the 12 deployments, 17,234 have a pose (6,958 npz, 10,276 XML; 4,178
unposed), and **14,505** of those pass `agree - disagree >= 2`. All 14,505 project inside the kept
strip. Numbers below are for the 14,505 unless marked; `summary.json` carries the same blocks for
all 17,234 (mean |d_y| 0.24 sigma, p90 0.52, 10.9% above 0.5 sigma). The same computation for the
other cities in the pool is in `summary_heldout.json` (6,289 candidates, 4,484 posed, 3,590 passing
the filter: mean |d_y| 0.24 sigma, p90 0.51, 10.4% above 0.5 sigma).

Tilt itself (per label): |T| mean 1.17 degrees, median 0.89, p90 2.47, p99 5.20; |pitch| mean 1.21,
|roll| mean 1.12.

Displacement of the target from the object, in sigma (d = corrected - stored, render px / 96):

| | beta = 1 | beta = 0.90 |
| :--- | ---: | ---: |
| mean abs d_y | 0.24 | 0.21 |
| median abs d_y | 0.18 | 0.16 |
| p90 abs d_y | 0.50 | 0.45 |
| p99 abs d_y | 1.06 | 0.96 |
| share abs d_y above 0.25 sigma | 36.2% | 31.7% |
| share abs d_y above 0.5 sigma | 10.1% | 7.7% |
| share abs d_y above 1 sigma | 1.3% | 0.9% |
| mean abs d_x | 0.07 | 0.06 |
| p90 abs d_x | 0.15 | 0.14 |

By |T| (beta = 1): below 1 degree (n 7,895) mean |d_y| 0.09 sigma; 1-2 degrees (4,227) 0.29; 2-3
(1,529) 0.49; 3-5 (681) 0.75; 5 and up (173) 1.31. Among cities with n >= 100 the mean |d_y| runs
from 0.16 sigma (Chicago, n 2,144) to 0.30 (Pittsburgh, n 1,099); Seattle, the largest city (n
4,932), is 0.29 with 14.9% of its targets more than 0.5 sigma off. Era (legacy 0.25, mid 0.23,
post179 0.23) and pose source (npz 0.23, XML 0.24) barely matter.

**Heading-resolved.** Binned by the 30-degree strip heading, the mean signed d_y traces a sinusoid
from +11.1 render px (SE 0.8) at -90 degrees to -9.1 (SE 0.8) at +60. A fit of
`d_y = c + a cos b + s sin b` gives a = -2.05 (SE 0.52), s = -9.56 (SE 0.39), c = -0.17 (SE 0.28)
render px; the fleet-mean pose predicts a = -1.97 and s = -9.62 (mean pitch 0.10, mean roll 0.50
degrees, at the median 19.41 render px per degree), and d_y regressed on T has slope -19.50 (SE
0.03) render px per degree against an expected -19.41. **This is an internal-consistency check,
not a sign check** (corrected after review): d_y is a deterministic function of pitch, roll and b,
so these fits recover the mean pose by construction. The sign is confirmed empirically in section
5.3, where the model's peaks move with both the pitch part and the roll part of d_y. The RMS of
d_y per bin is 0.29-0.40 sigma at every heading, so most of the displacement is per-pano spread
around that small mean, as #113 said; the mean is not zero because the GSV rigs in this sample
lean slightly left side up on average.

## 5. Does the crop model follow the object or the displaced target?

### 5.1 The run

Sample (`sample.csv`, `sample` subcommand, seed 113): one label per pano from the 14,505, with a
store JPEG, 200 each at |T| below 1, 1-3 and at least 3 degrees; 600 panos, 10 cities (Seattle
269). On makelab2 each store JPEG was resized to 8192 x 4096 with `cv2.INTER_AREA` (482 were
16384 x 8192, 115 were 13312 x 6656, 3 were 3328 x 1664), rendered at
`(90, nearest_theta, -30, 2048, 2048)` with `rampnet.gsv.equirectangular_to_perspective`, sliced
to the middle third, resized to 1024 x 352 as `evaluate.py` does, and run once (no flip TTA)
through both released crop checkpoints (`projectsidewalk/rampnet-crop-model` @ `7aa79b8e`,
safetensors sha256 pinned in the script). The peak is the parabolic-refined argmax of the
**clipped** heatmap within `evaluate.py`'s match radius (90 render px) of the projected **stored**
target ("stored" window); a second read uses twice that radius ("stored x2"). `peak - stored`
(render px) is regressed on the analytic d (beta = 1), SE clustered by pano (one label per pano,
so this is HC1). Slope 1 means the peak is on the object (at beta = 1), slope 0 that it reproduces
the displaced target. All 600 JPEGs were found. **The pano store is an unpublished local input**;
`response_v1.csv` is committed, so every number below re-derives from the repo with `--check`.

Localization caveat (review N1, not fixed in these numbers): the heatmap was clipped to [0, 1]
before the argmax, and 27% of round-2 peaks (6% of round 1) sit on a plateau at exactly 1.0, where
argmax returns the plateau's top-left pixel. The script now localizes on the unclipped map, and
the second run (section 5.4) does so. Every y slope in the table below moved by at most 0.03 between
the two runs (round 2 stored: 0.71 to 0.72), which confirms the reviewer's measurement on
unsaturated peaks (0.708 against 0.710). The table keeps the first run's numbers, since the text
and review cite them.

### 5.2 Training-set overlap (review B1)

The response sample is drawn from the crop model's own training population: round 1 trained on
27,704 crops built with the same filter over the same 12 deployments, 70% of them in train. To
measure the overlap without images, `fetch-keypoints` reads the `crop_id` column (which carries every
keypoint) of all 27,704 round-1 crops from `projectsidewalk/rampnet-crop-model-dataset-round1` @
`521f74ff`, and `overlap` matches each label's integer point, computed exactly as `download_data.py`
writes it, against each crop's first keypoint (the crop's own label). Shifting every point by 16
offsets of 11-29 render px and matching again gives the chance rate (`overlap.json`):

| labels | n | exact match | chance | estimated overlap f |
| :--- | ---: | ---: | ---: | ---: |
| 12 deployments, crowd_ok | 14,505 | 52.6% | 10.6% | 46.9% |
| response sample | 600 | 53.0% | 10.4% | 47.6% |
| held-out cities, crowd_ok | 3,590 | 10.8% | 10.2% | 0.6% |
| held-out sample | 565 | 10.3% | 9.5% | 0.8% |

If a share f of labels truly have a round-1 crop, match = f + (1 - f) x chance, so
f = (match - chance) / (1 - chance). (The first revision reported match - chance, "about 43%", which
understates f; corrected in the second review round.)

The held-out cities match at chance, which validates the test's specificity. So an estimated 48%
of the response sample are round-1 crops. Of the 318 matching labels about 33 are expected to be
chance matches (`expected_chance_matches`), so the 233 that match a train crop are about 90% true
round-1 train labels. Of the 282 unmatched labels, only 1 appears even as a secondary keypoint of
another crop (4.75 expected by chance). **The test's sensitivity is not measured**: a true round-1
label whose integer point differs today (for example a label edited since the 2025 fetch) would be
missed. The match rate tops out at about 85% even among the most-agreed labels, so up to about 40
of the 282 unmatched labels may still be round-1 crops. Below, "unmatched" means "not found among
round-1 crops", not "never seen". Their panos may also have been seen through other labels at
other headings.

### 5.3 Slopes

y slope [95% CI] by subset (`response_v1.json`):

| checkpoint | window | all 600 | unmatched (282) | matches a round-1 train crop (233) | population-weighted |
| :--- | :--- | ---: | ---: | ---: | ---: |
| round 1 | stored | 0.47 [0.42, 0.52] | 0.53 [0.45, 0.60] | 0.38 [0.30, 0.46] | 0.46 [0.39, 0.54] |
| round 1 | stored x2 | 0.51 [0.44, 0.59] | 0.60 [0.50, 0.71] | 0.37 [0.26, 0.48] | 0.50 [0.41, 0.59] |
| round 2 | stored | 0.71 [0.66, 0.76] | 0.73 [0.66, 0.80] | 0.68 [0.59, 0.77] | 0.75 [0.68, 0.82] |
| round 2 | stored x2 | 0.81 [0.74, 0.88] | 0.84 [0.74, 0.95] | 0.75 [0.63, 0.87] | 0.81 [0.71, 0.90] |

By |T| stratum (stored window): round 1 0.52 [0.40, 0.65] at 1-3 degrees and 0.47 [0.41, 0.52] at
3 degrees and up; round 2 0.82 [0.70, 0.93] and 0.70 [0.64, 0.75]. The below-1-degree stratum has
almost no displacement to regress on (CIs about +-0.4). That |T| >= 3 stratum supplies most of the
regressor's variance but only 5.9% of the population, so the pooled slope is in effect its slope.
The population-weighted column reweights each stratum to its share of the 14,505 (54.4%, 39.7%,
5.9%); it moves the estimate little but widens the CI.

**Edge hits.** A peak within 8 render px of its window's boundary counts as an edge hit. In the
stored window: round 1 111 of 600 (28, 28 and 55 in the three strata), round 2 142 (24, 27 and 91;
91 of 200 at 3 degrees and up). In the stored x2 window: 38 and 33. Clipping by the window biases
slopes toward 0 for the largest displacements, which is part of why the doubled window reads
higher. Excluding edge hits gives 0.41 / 0.53 (round 1, stored / x2) and 0.65 / 0.84 (round 2), so
dropping them does not settle it either, and **the true slope is not established to lie between
the two windows** (corrected after review). A third window centred half way between the stored
target and the corrected point (`mid`, same radius) was read in the second run (section 5.4).

**Pitch and roll parts (the empirical sign check).** Splitting d_y into its pitch part and its
roll part (exact geometry with the other angle zeroed) and regressing on both: round 1 0.50 (SE
0.03) and 0.43 (SE 0.05), round 2 0.74 (SE 0.03) and 0.66 (SE 0.05), stored window. Both are
positive: the model's peak moves in the predicted direction for each tilt axis separately.

**x.** Slopes about 0.39 (round 1) and 0.43 (round 2), CIs about +-0.2, with no detectable round-2
gain (paired 0.04 [-0.16, 0.24]); the x displacement is small (SD 13 render px), so x is weakly
identified. The x error regressed on strip x has slope -0.029 (SE 0.008) for round 1 and -0.015
(SE 0.008) for round 2, intercepts +16.3 (SE 3.2) and +15.3 (SE 3.1) render px. Section 3 reads
these against the two hypotheses (on the flip-averaged target: +10, slope 0; on the object: 0,
slope +0.031).

### 5.4 The second run: unclipped peaks, the `mid` window, and held-out cities

After the review a second `respond` run was launched on makelab2 (2026-10-06, about 05:52Z) with
the fixes above: unclipped localization, edge flags, the `mid` window, and the 565-label
held-out-city sample (`sample_heldout.csv`: 22 cities, led by spgg 129, cdmx 90 and columbia-sc 88;
only 165 labels exist at |T| >= 3). Both halves finished (06:05Z and 06:08Z, no pano missing). A
status poll timed out at about 05:52Z, so the outputs were retrieved only later that day, once
makelab2 was reachable again. They are committed as `response.csv` / `response.json` and
`response_heldout.csv` / `response_heldout.json`, beside the first run's `response_v1.*`, and
`--check` covers all three.

y slope [95% CI], all labels in each sample (`response.json`, `response_heldout.json`):

| checkpoint | window | in-population, all 600 | in-population, unmatched (282) | held-out cities, all 565 |
| :--- | :--- | ---: | ---: | ---: |
| round 1 | stored | 0.47 [0.42, 0.52] | 0.53 [0.46, 0.60] | 0.44 [0.34, 0.53] |
| round 1 | stored x2 | 0.52 [0.45, 0.59] | 0.60 [0.50, 0.71] | 0.56 [0.47, 0.65] |
| round 1 | mid | 0.54 [0.48, 0.59] | 0.61 [0.53, 0.69] | 0.59 [0.52, 0.67] |
| round 2 | stored | 0.72 [0.67, 0.77] | 0.74 [0.67, 0.81] | 0.64 [0.51, 0.77] |
| round 2 | stored x2 | 0.82 [0.75, 0.89] | 0.85 [0.75, 0.96] | 0.83 [0.73, 0.93] |
| round 2 | mid | 0.84 [0.79, 0.89] | 0.86 [0.78, 0.94] | 0.86 [0.79, 0.93] |

Paired round 2 minus round 1, held-out: 0.20 [0.14, 0.27] stored, 0.26 [0.20, 0.33] mid
(in-population 0.25 and 0.30).

- **Edge hits** (stored / mid window): in-population round 1 110 / 80 and round 2 147 / 87 of 600;
  held-out round 1 168 / 146 and round 2 195 / 145 of 565. The held-out stored window clips far more
  often, which is why its stored-window slopes sit lowest; the `mid` window agrees across samples.
- **The `mid` window** reads highest for both checkpoints and has the fewest edge hits in-population.
  It moves the in-population unmatched-minus-train difference for round 1 from 0.15 (SE 0.056) to
  0.20 (SE 0.056).
- **The 45 held-out labels that "match" a round-1 crop** are the chance matches 5.2 predicts
  (held-out cities are not in round 1). Their slopes are not read.

### 5.4.1 What the held-out arm shows

Section 5.5 left three explanations for round 1's "about half way" on unmatched labels: visible
tilt, round-1 labels the overlap test missed, or panos seen through other labels. The held-out
cities rule out the second and third, since none of their panos are in round 1. Round 1 still
follows the object only about half way there (0.44-0.59 across windows, against 0.53-0.61 on
unmatched in-population labels). The remainder therefore does not need memorization. That points to
an image cue that carries the displacement, such as visible tilt, though the held-out cities also
differ in imagery and rig, which this run does not separate. Round 2 reaches the same 0.83-0.86 on
held-out cities as in-population in the two wider windows.

### 5.5 Reading (corrected after review)

Neither checkpoint reproduces the displaced target (slope 0 is far outside every CI) and neither
sits fully on the object. **What was written here first was wrong.** It said round 1 "splits the
difference, as a model trained on targets displaced by a heading- and pose-dependent amount it
cannot see would be expected to". The opposite is true. Under an L2 heatmap loss, a displacement the
model cannot see from the image averages out, and the peak should land on the object (slope about
beta). A slope well below 1 needs the image to carry information about the displacement:
memorized training targets, or tilt that is visible in the image (leaning verticals, a sloping
horizon).

The overlap split separates the two in part. Memorization is real: on labels whose crop was in
round-1 train the round-1 slope is 0.38, below the 0.53 on unmatched labels: a difference of 0.15
(SE 0.056, z 2.7), and the same 0.15 (SE 0.06) within the |T| >= 3 stratum alone. But it is not the
whole story, because even on unmatched labels round 1 follows the object
only about half way. That remainder could be visible tilt, round-1 labels the overlap test missed, or panos seen through other labels. The
held-out-city arm (5.4.1) rules out the last two: the same half-way slope appears in cities round 1
never trained on. Round 2's manual fine-tune moves the peak toward the
object on both subsets (0.73 unmatched, 0.68 train), so its gain is not only forgetting round-1
targets, though forgetting may contribute. If the true leak is beta = 0.90 rather than 1, "on the
object" is slope 0.90, not 1.

## 6. What it means for 2.0's regeneration

If 2.0 regenerates the crop-model training set from Project Sidewalk labels, correcting `pano_y`
by beta x T (beta between 0.90 and 1) before projecting removes a displacement whose p90 is half a
sigma and whose largest 1% exceed a sigma. The released round-2 model still leaves roughly
0.04-0.26 of that displacement in its peak on unmatched in-population labels, and 0.04-0.36 on
held-out cities. Both are computed as beta minus slope, with beta between 0.90 and 1 and point
estimates only (second-run y slopes 0.74-0.86 and 0.64-0.86 across the three windows, against
0.90-1 for "on the object"). The high end comes from the stored window, which clips the most; in
the `mid` window it is 0.04-0.14 on both. These ranges come mostly from labels with |T| >= 3 degrees
and are not population estimates. Whether to apply the correction is a
decision for 2.0, not made here. The effect on Stage 1 point placement, and from there on Stage 2,
is not measured.

## 7. Caveats

- **The response sample overlaps the training set** (section 5.2): an estimated 48% of it are
  round-1 crops, and the overlap test's sensitivity is unmeasured. The slopes on the 282 unmatched labels are the cleaner estimate, but their panos are
  from the same deployments and may appear in round 1 through other labels.
- **Vouched pool, not the paper's crowd sample.** The labels are pano-tools' lead-vouched pool
  (labels made or validated by two lead labellers), filtered with `download_data.py`'s
  `agree - disagree >= 2`; the paper's round-1 set is a crowd sample from the same 12 deployments
  read live in 2025. The round-1 crops on the Hub carry no pano id (`crop_uid` is random), so the
  paper's exact crops cannot be posed. The convention and cities are the same; the label mix is not.
- **Pose coverage.** 4,178 of 21,412 candidate labels (20%) have no committed pose and are left out;
  no `streetlevel` or Google call was made to fill them. Of the 14,505, 8,432 use the 2019 XML pose
  and 6,073 the npz pose; their displacement distributions agree (mean |d_y| 0.24 vs 0.23 sigma).
- **Pose noise attenuates the slopes** (review N3). On the 9,346 panos that carry both poses, npz
  minus XML has SD 0.69 degrees in pitch and 0.66 in roll (medians of the absolute difference only
  0.09 and 0.12, so the noise is heavy-tailed; `summary.json` `funnel.pose_npz_vs_xml_all_cities`).
  Against the sample's spread in d (SD 54 render px, about 2.8 degrees of T), that error variance
  attenuates the slopes by roughly 3% if the two sources share it equally and 6% if one carries it
  all.
- **Store JPEG vs paper-era tiles.** The response run renders from the makelab2 store JPEGs, resized
  with `INTER_AREA`; the paper fetched zoom-4 tiles and resized with `INTER_LINEAR`. Imagery dates
  may also differ from the 2025 fetch. Both are in the rig frame (pano-tools F2), which is what the
  measurement depends on. 3 of 600 store JPEGs are low resolution (3328 x 1664) and were upsampled.
- **Beta is borrowed.** 0.90 comes from a different city and era mix (sidewalk-auto-labeler #113);
  it is reported as a sensitivity row, not re-estimated here.
- **The analytic d is the first-order leak of a perfect click.** Click noise (about 0.33 sigma
  residual SD in the response run) is on top of it and is not modelled in section 4.
- **One forward per strip**, no flip TTA, unlike `evaluate.py`, so the response slopes describe the
  raw model; flip TTA's max-combine could move them.
- **One checkpoint per round.** The slopes are for the released checkpoints; there is no seed
  replicate of the crop model, so how much of round 2's gain is the manual data and how much is
  that one fine-tune is not separable here.

## 8. Cost

- `predict`: desktop CPU, about 1 minute. `sample`, `overlap`, `fit` and `--check`: seconds.
  `fetch-keypoints`: about 5 minutes of ranged Hub reads (the `crop_id` column only, no images).
- `respond`, first run: makelab2 A40, 232 s wall-clock for 600 panos (0.39 s per pano including
  JPEG decode; 164 s of render plus forward), recorded as `crop-tilt-113:respond` in
  `analysis_out/usage_log.jsonl` with `paid: false` and `gpu_hours` 0.0645. The A40 was shared the
  whole time with two other processes (about 15 GB, 100% utilization at start and end), so
  `gpu_hours` is an upper bound (`usage_respond_v1.json`). A 3-pano smoke run (about 4 s) and the one-time download of the two
  checkpoints from Hugging Face are not counted.
- `respond`, second run (5.4): the same shared A40. 833.5 s for the 600 in-population panos and
  120.0 s for the 565 held-out ones (`usage_respond.json`, `usage_respond_heldout.json`), 953.5 s
  and 0.265 GPU-hours in total, an upper bound because the GPU was shared. The in-population half
  ran 3.6x slower than the first run on the same panos, which is contention, not the code. Recorded
  as `crop-tilt-113:respond-v2`: the `in_progress` row written at launch, then a closing `ok` row
  with the same `run_id`.
- External calls: Google 0, Project Sidewalk API 0. Paid APIs: none.

## 9. Reproduction

Inputs: sidewalk-panorama-tools at commit `21d10aa3767167e67a098557f58f327def2396a5`
(public, `github.com/ProjectSidewalk/sidewalk-panorama-tools`), files
`reports/data/2026-09-29-tilt-jm-pool.csv.gz` and `reports/data/2026-09-29-tilt-pose-jm.csv.gz`,
both sha256-pinned in the script; the crop checkpoints from Hugging Face
`projectsidewalk/rampnet-crop-model` @ `7aa79b8edb10b384ed69c2e99f74945e9c527fd3`; the round-1
crop keypoints from `projectsidewalk/rampnet-crop-model-dataset-round1` @
`521f74ff752d57824400c8f7d5ca4717efa7bf16` (committed as `round1_keypoints.csv`). The `respond`
step also needs the makelab2 panorama store
(`/m-makeabilitylab/makeabilitylab/sidewalk_panos/Panoramas/<city>/<id[:2]>/<id>.jpg`), which is
an **unpublished local input**; the first run's output, `response_v1.csv`, is committed.

```bash
# analytic half (CPU, about 1 minute): labels*.csv and summary*.json
python scripts/analysis/crop_tilt_113.py predict --pano-tools-root ../sidewalk-panorama-tools
python scripts/analysis/crop_tilt_113.py sample
python scripts/analysis/crop_tilt_113.py sample --labels analysis_out/crop_tilt_113/labels_heldout.csv --out analysis_out/crop_tilt_113/sample_heldout.csv
# training-set overlap (no GPU, no images)
python scripts/analysis/crop_tilt_113.py fetch-keypoints
python scripts/analysis/crop_tilt_113.py overlap
# empirical half: needs the pano store and a GPU. On makelab2 the RampNet venv lacked cv2, so
# opencv-python-headless 4.10.0.84 was installed into a private --target dir on PYTHONPATH.
python scripts/analysis/crop_tilt_113.py respond --sample analysis_out/crop_tilt_113/sample.csv --store /m-makeabilitylab/makeabilitylab/sidewalk_panos/Panoramas --out analysis_out/crop_tilt_113 --usage-out analysis_out/crop_tilt_113/usage_respond.json
python scripts/analysis/crop_tilt_113.py respond --sample analysis_out/crop_tilt_113/sample_heldout.csv --store /m-makeabilitylab/makeabilitylab/sidewalk_panos/Panoramas --out analysis_out/crop_tilt_113 --suffix _heldout --usage-out analysis_out/crop_tilt_113/usage_respond_heldout.json
python scripts/analysis/crop_tilt_113.py fit --name response_v1   # the committed first run
python scripts/analysis/crop_tilt_113.py fit --name response
python scripts/analysis/crop_tilt_113.py fit --name response_heldout
# re-derive every summary from the committed CSVs, byte for byte
python scripts/analysis/crop_tilt_113.py --check
# and labels*.csv too, given the sibling checkout
python scripts/analysis/crop_tilt_113.py --check --pano-tools-root ../sidewalk-panorama-tools
```

What `--check` covers: without `--pano-tools-root` it takes `labels.csv` and `labels_heldout.csv`
as given and copies each summary's `funnel` block from the committed file, so CI (which runs it via
`tests/test_crop_tilt_113.py`) never re-derives the label tables or the funnel. Re-deriving those
needs the sibling checkout and the second command. `response_v1.csv` is the first run's file in its
original column layout; `--check` adapts it (`response_from_v1`) before fitting.

`tests/test_crop_tilt_113.py` pins the vendored tilt functions (`rampnet/stage1_geometry.py`)
against values computed by the original pano-tools module. It also checks the float and integer
projections against `download_data.py`'s, the renderer against the point projection, the heading
sinusoid on synthetic labels, the pitch/roll split, the weighted clustered OLS and the peak finder,
and it runs `--check`.

## 10. Corrections after review (2026-10-06)

The review on PR #244 found the following in the first version of this document. The text above is
corrected; what it said before is quoted here.

- **B1.** Section 5 said round 1 "splits the difference, as a model trained on targets displaced by
  a heading- and pose-dependent amount it cannot see would be expected to". That reasoning was
  backwards (section 5.5), and the sample's overlap with the training set was not stated. The
  overlap is now measured (5.2), the slopes are split by it (5.3), and the headline is scoped to
  "panos drawn from the crop model's own training population".
- **S1.** Section 1 said round 2 "still carries about 0.29 of the displacement in y, roughly 0.07
  sigma on average and 0.15 sigma at the p90 label", and section 6 said "about a fifth to a third".
  The pooled slope is in effect the |T| >= 3 stratum's slope, so a single population extrapolation
  is not supported. Both are replaced by per-stratum slopes, edge-hit counts, a population-weighted
  slope and a range. Section 5 also said "the true slopes are probably between the two rows". That
  is not established, because edge hits remain in the doubled window.
- **S2.** Section 4 called the sinusoid fit "the sign check the plan asked for". It is an
  internal-consistency check; the empirical sign check is now the pitch/roll split in 5.3.
- **S3.** The S3 correction said "understates sigma by a factor of 10.2, so its p90 shifts of
  3.8-4.8 degrees are 0.7-0.9 sigma", using the strip-centre scale only. Matched by depression band
  it is 8-10x and 0.7-1.1 sigma.
- **S4.** Section 3 said every round-1 target "sits 3% of its distance from the left edge too far
  left". That ignored the horizontal-flip augmentation, which makes the flip-averaged error a pull
  toward the centre column.
- **N2.** Section 3's table said "labels sit below the strip centre"; 93.2% sit above it.
- **N5.** Section 2 called pano-tools' 0.93 [0.86, 0.99] "consistent with 1"; that CI excludes 1.
- **N8.** The city range now says "cities with n >= 100" (cliffside-park, n 2, was outside it).

Second review round (same day):

- The first revision said "about 43%" of the sample are round-1 crops (match minus chance). The
  estimator is (match - chance) / (1 - chance): 47.6% for the sample, 46.9% for the 14,505. It also
  called unmatched labels "never saw" / "not a round-1 crop"; the test's sensitivity is unmeasured,
  so they are now "unmatched".
- The first revision's x-quirk prediction ("+10.5 - 0.031 x strip_x", and "the response run agrees
  on the slope") compared a target-minus-object prediction with a quantity measured in the
  stored-target frame. Section 3 now gives both hypotheses in the measured frame.
- Section 6 said "roughly 0.15-0.35"; that mixed a CI bound into a point-estimate range. It is now
  0.06-0.27, derived as beta minus slope.
- The held-out sample covers 22 cities, not 20. "The CIs touch" is replaced by the difference and
  its SE.
- The first run's files are renamed `response_v1.csv` / `response_v1.json`.

Second run retrieved (2026-10-06, later the same day):

- Section 5.4 was "The re-run that was not retrieved" and said "every slope in this document comes
  from the first run". The second run's in-population and held-out results are now in 5.4 and
  5.4.1. Section 5.3's table still shows the first run.
- Section 6 said "roughly 0.06-0.27 ... (y slopes 0.73-0.84 on that subset across the two
  windows)". With the third window and the second run it is 0.04-0.26 in-population and 0.04-0.36
  held-out.
- Section 5.5 said the held-out arm "would test" whether missed round-1 labels explain the
  remainder. It now rules out that explanation and the panos-seen-through-other-labels one.
