# Rig tilt in the Stage 1 crop model's training targets (#113, measurement half)

Issue: [#113](https://github.com/ProjectSidewalk/RampNet/issues/113). Script:
`scripts/analysis/crop_tilt_113.py`. Outputs: `analysis_out/crop_tilt_113/`. Run 2026-10-05/06.
Scope: the measurement half only. The architectural half of #113 (consume
sidewalk-panorama-tools' cropper) is not touched, and no training set is regenerated or corrected.

## 1. Result

Over the 14,505 vouched CurbRamp labels in the 12 deployments `download_data.py` reads that
carry a committed pose and pass its `agree - disagree >= 2` filter, rig tilt moves the round-1
training target off its object by **0.24 sigma on average in y (median 0.18, p90 0.50, p99 1.06)**
at beta = 1, and 0.21 sigma (p90 0.45) at the measured beta = 0.90. 10.1% of targets are more than
0.5 sigma off in y and 1.3% more than 1 sigma (7.7% and 0.9% at beta = 0.90). The x term is small:
mean 0.07 sigma, p90 0.15. The displacement is a sinusoid in heading as #113 predicted, but **not
zero-mean over headings**: the fleet's mean roll is about +0.5 degrees, which leaves a heading-locked
sinusoid of amplitude 9.8 render px (0.10 sigma) in the bin means. Run on 600 store panos through
the released checkpoints, **the round-1 crop model follows the object only about half way**: its
peak moves by 0.47 [0.42, 0.52] of the analytic displacement in y (slope 0 would be "reproduces
the displaced target", 1 "sits on the object"). **The round-2 manual fine-tune re-centres it
partly**, to 0.71 [0.66, 0.76], a paired gain of 0.24 [0.19, 0.29]. In x the slopes are 0.39
[0.18, 0.60] and 0.43 [0.22, 0.64], with no detectable round-2 gain (0.04 [-0.16, 0.24]); the x
displacement is small (SD 13 render px), so x is weakly identified. So the leak is real in the
released Stage 1 model: round 2, the model that placed every Stage 1 point, still carries about
0.29 of the displacement in y, roughly 0.07 sigma on average and 0.15 sigma at the p90 label.

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
  legacy 0.88, mid 0.91, post179 0.94); pano-tools' human beta batch (#191) is consistent with 1
  (0.93 [0.86, 0.99]). This doc reports beta = 1 (ceiling) and beta = 0.90 (measured) and does not
  re-estimate beta.
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
| degrees, label median (19.40 render px/deg; labels sit below the strip centre) | 4.95 |
| px on the paper's 8192 x 4096 pano (22.76 px/deg), strip centre | 122 |

`evaluate.py`'s match radius is 0.132 x 341/4 = 11.25 heatmap px = 90 render px, so the radius is
about 0.94 sigma.

**The issue's "0.2-0.6 sigma at 1-3 degrees of tilt" is confirmed.** At the strip centre 1 and 3
degrees of `T` are 0.19 and 0.56 sigma; at the label median scale they are 0.20 and 0.61 sigma.

**Correction to pano-tools' S3.** S3 (`s3_miscentering` in
`reports/data/2026-09-26-tilt-error-study.json`) reads "7-9 click sigmas for RampNet stage one"
from a sigma of 12 px on a 4096-high pano. Sigma is 12 *heatmap* px, which is 96 render px, about
5.4 degrees, or about 122 px on a 4096-high pano. S3 understates sigma by a factor of 10.2, so its
p90 shifts of 3.8-4.8 degrees are 0.7-0.9 sigma, not 7-9. That correction belongs on
sidewalk-panorama-tools #54 and has not been posted there yet.

A related quirk, recorded and not acted on: `train.py` scales target x by 0.5, but the 683-wide
strip is resized to 352 (0.515), so every round-1 target sits 3% of its distance from the left
edge too far left. It is a constant, not tilt-dependent, and is at most 11 input px at the right
edge.

## 4. Analytic displacement

Population (`summary.json` `funnel`): of the 70,422-label pano-tools pool, 27,701 are CurbRamp
(all GSV), 21,412 are in the 12 deployments, 17,234 have a pose (6,958 npz, 10,276 XML; 4,178
unposed), and **14,505** of those pass `agree - disagree >= 2`. All 14,505 project inside the kept
strip. Numbers below are for the 14,505 unless marked; `summary.json` carries the same blocks for
all 17,234 (mean |d_y| 0.24 sigma, p90 0.52, 10.9% above 0.5 sigma).

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
(1,529) 0.49; 3-5 (681) 0.75; 5 and up (173) 1.31. By city the mean |d_y| runs from 0.16 sigma
(Chicago, n 2,144) to 0.30 (Pittsburgh, n 1,099); Seattle, the largest city (n 4,932), is 0.29 with
14.9% of its targets more than 0.5 sigma off. Era (legacy 0.25, mid 0.23, post179 0.23) and pose
source (npz 0.23, XML 0.24) barely matter.

**Heading-resolved.** Binned by the 30-degree strip heading, the mean signed d_y traces a sinusoid
from +11.1 render px (SE 0.8) at -90 degrees to -9.1 (SE 0.8) at +60. A fit of
`d_y = c + a cos b + s sin b` gives a = -2.05 (SE 0.52), s = -9.56 (SE 0.39), c = -0.17 (SE 0.28)
render px; the fleet-mean pose predicts a = -1.97 and s = -9.62 (mean pitch 0.10, mean roll 0.50
degrees, at the median 19.41 render px per degree), so the bin means are explained by the mean
pose to within an SE. This is the sign check the plan asked for: predicted and measured
coefficients agree in sign and size, and d_y regressed on T has slope -19.50 (SE 0.03) render px
per degree against an expected -19.41. The RMS of d_y per bin is 0.29-0.40 sigma at every heading,
so most of the displacement is per-pano spread around that small mean, as #113 said; the mean is
not zero because the GSV rigs in this sample lean slightly left side up on average.

## 5. Does the crop model follow the object or the displaced target?

Sample (`sample.csv`, `sample` subcommand, seed 113): one label per pano from the 14,505, with a
store JPEG, 200 each at |T| below 1, 1-3 and at least 3 degrees; 600 panos, 10 cities (Seattle
269). On makelab2 each store JPEG was resized to 8192 x 4096 with `cv2.INTER_AREA` (482 were
16384 x 8192, 115 were 13312 x 6656, 3 were 3328 x 1664), rendered at
`(90, nearest_theta, -30, 2048, 2048)` with `rampnet.gsv.equirectangular_to_perspective`, sliced
to the middle third, resized to 1024 x 352 as `evaluate.py` does, and run once (no flip TTA)
through both released crop checkpoints (`projectsidewalk/rampnet-crop-model` @ `7aa79b8e`,
safetensors sha256 pinned in the script). The peak is the parabolic-refined argmax within
`evaluate.py`'s match radius (90 render px) of the projected **stored** target; a second read
uses twice that radius. `peak - stored` (render px) is regressed on the analytic d (beta = 1), SE
clustered by pano (one label per pano, so this is HC1). Slope 1 means the peak is on the object
(at beta = 1), slope 0 that it reproduces the displaced target. All 600 JPEGs were found; the
median peak value is 0.92 (round 1) and 0.97 (round 2).

| checkpoint | window | slope y [95% CI] | slope x [95% CI] |
| :--- | :--- | ---: | ---: |
| round 1 | radius | 0.47 [0.42, 0.52] | 0.39 [0.18, 0.60] |
| round 1 | 2 x radius | 0.51 [0.44, 0.59] | 0.41 [0.14, 0.68] |
| round 2 | radius | 0.71 [0.66, 0.76] | 0.43 [0.22, 0.64] |
| round 2 | 2 x radius | 0.81 [0.74, 0.88] | 0.45 [0.19, 0.72] |
| round 2 - round 1, paired | radius | 0.24 [0.19, 0.29] | 0.04 [-0.16, 0.24] |
| round 2 - round 1, paired | 2 x radius | 0.30 [0.23, 0.36] | 0.04 [-0.21, 0.30] |

Restricting to peaks of at least 0.3 (574 and 579 of 600 within the radius) moves the y slopes to
0.49 and 0.73. The intercepts are small: y +3.5 (SE 1.3) and +2.2 (SE 1.3) render px, x +6.7 and
+10.6 (SE 1.4); the x intercept has the same sign as the x-scaling quirk in section 3 and is not
investigated further. Residual SD is 31-35 render px (about 0.33 sigma) within the radius, which is
click noise plus model noise.

Reading. Neither checkpoint reproduces the displaced target (slope 0 is far outside every CI) and
neither sits fully on the object. Round 1 splits the difference, as a model trained on targets
displaced by a heading- and pose-dependent amount it cannot see would be expected to. Round 2's
manual fine-tune (1,212 crops with manually placed points, no Project Sidewalk coordinates) moves
it measurably toward the object, by 0.24-0.30 of the displacement, and leaves 0.19-0.29 of it in
y. If the true leak is beta = 0.90 rather than 1, "on the object" is slope 0.90, not 1, and the
remaining fractions shrink by about 0.1. The radius window biases slopes toward 0 for the largest
displacements (the object can sit outside a 90 px window around the stored target), which is why
the doubled window reads higher; the true slopes are probably between the two rows.

## 6. What it means for 2.0's regeneration

If 2.0 regenerates the crop-model training set from Project Sidewalk labels, correcting `pano_y`
by beta x T (beta between 0.90 and 1) before projecting removes a displacement whose p90 is half a
sigma and whose largest 1% exceed a sigma, and which the released round-2 model still carries at
about a fifth to a third of its size. Whether to apply it is a decision for 2.0, not made here. The
effect on Stage 1 point placement, and from there on Stage 2, is not measured.

## 7. Caveats

- **Vouched pool, not the paper's crowd sample.** The labels are pano-tools' lead-vouched pool
  (labels made or validated by two lead labellers), filtered with `download_data.py`'s
  `agree - disagree >= 2`; the paper's round-1 set is a crowd sample from the same 12 deployments
  read live in 2025. The round-1 crops on the Hub carry no pano id (`crop_uid` is random), so the
  paper's exact crops cannot be posed. The convention and cities are the same; the label mix is not.
- **Pose coverage.** 4,178 of 21,412 candidate labels (20%) have no committed pose and are left out;
  no `streetlevel` or Google call was made to fill them. Of the 14,505, 8,432 use the 2019 XML pose
  and 6,073 the npz pose; their displacement distributions agree (mean |d_y| 0.24 vs 0.23 sigma).
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

- `predict`: desktop CPU, 45 s. `sample` and `--check`: seconds. No GPU.
- `respond`: makelab2 A40, 232 s wall-clock for 600 panos (0.39 s per pano including JPEG decode;
  164 s of render plus forward), recorded as `crop-tilt-113:respond` in
  `analysis_out/usage_log.jsonl` with `paid: false` and `gpu_hours` 0.0645. The A40 was shared the
  whole time with two other processes (about 15 GB, 100% utilization at start and end), so
  `gpu_hours` is an upper bound. A 3-pano smoke run (about 4 s) and the one-time download of the two
  checkpoints from Hugging Face are not counted.
- External calls: Google 0, Project Sidewalk API 0. Paid APIs: none.

## 9. Reproduction

Inputs: sidewalk-panorama-tools at commit `21d10aa3767167e67a098557f58f327def2396a5`
(public, `github.com/ProjectSidewalk/sidewalk-panorama-tools`), files
`reports/data/2026-09-29-tilt-jm-pool.csv.gz` and `reports/data/2026-09-29-tilt-pose-jm.csv.gz`,
both sha256-pinned in the script; the crop checkpoints from Hugging Face
`projectsidewalk/rampnet-crop-model` @ `7aa79b8edb10b384ed69c2e99f74945e9c527fd3`. The `respond`
step also needs the makelab2 panorama store
(`/m-makeabilitylab/makeabilitylab/sidewalk_panos/Panoramas/<city>/<id[:2]>/<id>.jpg`), which is
an **unpublished local input**; its output, `response.csv`, is committed, so every number in
section 5 re-derives from the repo.

```bash
# analytic half (CPU, about 1 minute)
python scripts/analysis/crop_tilt_113.py predict --pano-tools-root ../sidewalk-panorama-tools --out analysis_out/crop_tilt_113
python scripts/analysis/crop_tilt_113.py sample --out analysis_out/crop_tilt_113/sample.csv --per-stratum 200 --seed 113
# empirical half: needs the pano store and a GPU. On makelab2 the RampNet venv lacked cv2, so
# opencv-python-headless 4.10.0.84 was installed into a private --target dir on PYTHONPATH.
python scripts/analysis/crop_tilt_113.py respond --sample analysis_out/crop_tilt_113/sample.csv --store /m-makeabilitylab/makeabilitylab/sidewalk_panos/Panoramas --out analysis_out/crop_tilt_113 --usage-out analysis_out/crop_tilt_113/usage_respond.json
# re-derive summary.json, sample.csv and response.json from the committed CSVs, byte for byte
python scripts/analysis/crop_tilt_113.py --check
# and labels.csv too, given the sibling checkout
python scripts/analysis/crop_tilt_113.py --check --pano-tools-root ../sidewalk-panorama-tools
```

`tests/test_crop_tilt_113.py` pins the vendored tilt functions (`rampnet/stage1_geometry.py`)
against values computed by the original pano-tools module, checks the float projection against
`download_data.py`'s integer one and the renderer against the point projection, checks the heading
sinusoid on synthetic labels, and runs `--check`.
