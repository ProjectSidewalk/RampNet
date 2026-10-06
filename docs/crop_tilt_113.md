# Rig tilt in the Stage 1 crop model's training targets (#113, measurement half)

Issue: [#113](https://github.com/ProjectSidewalk/RampNet/issues/113). Script:
`scripts/analysis/crop_tilt_113.py`. Outputs: `analysis_out/crop_tilt_113/`. Run 2026-10-05/06.
Scope: the measurement half only. The architectural half of #113 (consume
sidewalk-panorama-tools' cropper) is not touched, and no training set is regenerated or corrected.

## 1. Result

RESULT_PARAGRAPH

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

ANALYTIC_SECTION

## 5. Does the crop model follow the object or the displaced target?

RESPONSE_SECTION

## 6. What it means for 2.0's regeneration

MEANING_SECTION

## 7. Caveats

CAVEATS_SECTION

## 8. Cost

COST_SECTION

## 9. Reproduction

REPRO_SECTION
