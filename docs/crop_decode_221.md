# Sub-cell decode on the Stage 1 crop model (#221)

Run 2026-10-05 on branch `analysis/crop-decode-221`; revised the same night after the review on
PR #247 (section 10 lists what changed). Script:
[`scripts/analysis/crop_decode_221.py`](../scripts/analysis/crop_decode_221.py). Committed outputs:
[`analysis_out/crop_decode_221/`](../analysis_out/crop_decode_221/).

## 1. Result

The Gaussian sub-cell decode that shipped for the pano model (`rampnet/subcell.py`, PRs #229 and
#233) also moves the crop model's peaks toward the labelled points.

**Test split.** The round-2 test split is 190 files, but they hold only **147 distinct images
with 203 distinct points**. A crop with several ramps is stored once per ramp, as the same
bytes under a name that permutes the points. Every CI below therefore resamples distinct images
(sha256), not files, and a deduplicated read (one file per image) is reported beside the
all-files one. Pairs are fixed on the argmax peaks at score >= 0.30.

| read | arm | pairs (images) | argmax mean px | gaussian mean px | paired change [95% CI] |
|---|---|---|---|---|---|
| deduplicated | single pass | 173 (139) | 3.961 | 2.964 | **-0.997 [-1.213, -0.771]** |
| deduplicated | flip TTA | 178 (139) | 4.089 | 3.028 | **-1.061 [-1.301, -0.828]** |
| all files | single pass | 244 (139) | 3.844 | 2.858 | -0.987 [-1.213, -0.744] |
| all files | flip TTA | 252 (139) | 3.917 | 2.906 | -1.012 [-1.255, -0.775] |

That is about a 25% cut in the mean distance from detection to label. The pano model's cut was
14% (5.080 to 4.353 px on `manual_gold`).

**Val split.** The round-2 val split (142 files, 118 distinct images, 157 distinct points)
replicates it:
- deduplicated: -1.196 [-1.448, -0.941] single pass, -1.077 [-1.335, -0.800] TTA;
- all files: -1.108 [-1.423, -0.777] single pass, -0.960 [-1.292, -0.603] TTA.

The CI excludes zero on every split, arm, checkpoint and threshold measured. No image is in both
test and val.

**Units.** A heatmap pixel here is 4 model-input pixels, about 8 source pixels of a 682x2048
crop (7.75 in x, 8 in y). The training target's sigma is 12 heatmap px, so the mean error goes
from about 0.33 sigma to 0.25 sigma. The evaluator's match radius is 11.25 px.

**Detection metrics.** The peaks and scores are identical under both decodes; only the positions
move. The detection counts can still change, in either direction, for peaks near the 11.25-unit
match radius. Round-2 test, TTA: 5 peaks gained true-positive status and none lost it. Round-2
val, single pass: +1 / -0. Round-1 test, single pass: +2 / -3, a net loss of one. These are small
and depend on the split; the position gain is the finding (section 4.4).

**Nothing Stage 1 produces changes.** The decode is not wired into Stage 1 placement, and cannot
be dropped in (section 6). The committed crop evaluator gains an opt-in, `PEAK_DECODE`, whose
default (`"argmax"`) leaves its output unchanged.

## 2. Mechanism check on the real crop head

The crop head is the pano head (`rampnet/model.py`): `Conv3x3 -> ReLU -> Upsample(256x88,
bilinear) -> Conv1x1` on a stride-32 map of a 1024x352 input. The pre-upsample map is 32x11, and
the 256x88 output should be exactly its bilinear x8 upsample. Measured on every crop and both
flip branches, the largest relative residual `max|head - upsample(coarse)| / max|head|` is:

| split | torch upsample | numpy `subcell.upsample` |
|---|---|---|
| test (round-2 checkpoint) | 2.95e-07 | 2.44e-07 |
| val (round-2 checkpoint) | 3.17e-07 | 3.09e-07 |
| test (round-1 checkpoint) | 2.72e-07 | 2.50e-07 |

That is fp32 rounding, as on the pano (#226). The bilinear upsample also commutes with a
horizontal flip to 2.22e-16, so the mirrored branch's coarse map, flipped back, is a valid
decode source under TTA.

Before decoding, 238 of the 244 argmax peak columns sit on heatmap pixel 3 or 4 mod 8 (single
pass, test, all files). Under the gaussian decode all eight residues are populated
(`results_test.md`, mod-8 histograms). The labels themselves spread over all eight residues.

## 3. Method

What is the same as the pano read (`docs/subcell_decode_221.md` section 3):
- peaks with `detect_peaks(..., min_distance=10, clip=True, exclude_border=False)` from
  `rampnet/subcell.py`;
- pairs fixed once on the argmax positions, greedy by confidence;
- every decode rule in `subcell.METHODS` scored on the same pairs;
- residual = label - detection, per axis;
- a cluster bootstrap of the paired change (2,000 reps, seed 221);
- sub-cell regression slopes and mod-8 histograms.

What differs, all taken from the committed crop evaluator
(`stage_one/crop_model/ps_and_manual_model/evaluate.py`) verbatim:

- No x wrap. A crop is not a panorama.
- Matching geometry: radius 0.132 normalized with `scale_x = 341/4`, `scale_y = 1024/4`, so
  11.25 units. The crops are 682 px wide (the evaluator's 341 comes from the half-width
  generation), so in x one unit is 88/85.25 heatmap px. Used as is, not "fixed".
- Flip TTA as the evaluator does it: clip each branch to [0, 1], flip the mirrored one back,
  elementwise max. The decode reads each peak from the branch the max took it from (the
  shipped `coarse=(2, 32, 11)` branch-select rule).
- **The bootstrap unit is the distinct image (sha256).** Duplicate files of one image move
  together. The deduplicated read keeps the first file of each image by filename. No point set
  crosses splits.
- Residuals are in pixels of the 256x88 heatmap.
- GT is parsed from the filenames (`{uid}_-_{x}_{y}[_-_{x}_{y}...].jpg`) and normalized by the
  image's own width and height, as the evaluator does. One val crop is 704 px wide; all
  others are 682.

Two arms: **single pass**, which is what `inference_isolator.infer_image` runs, and **flip
TTA**, which is what the crop evaluator runs. Three thresholds: 0.30 (the pairs reported
above), 0.0 (the evaluator's) and 0.55.

Inputs, all pinned in `analysis_out/crop_decode_221/inputs.json` with the sha256 of every
file:

- `projectsidewalk/rampnet-crop-model` at `7aa79b8edb10b384ed69c2e99f74945e9c527fd3`:
  - `round2_ps_and_manual_best_model.safetensors`, sha256 `d129c0c6…`;
  - `round1_ps_best_model.safetensors`, sha256 `23e40b59…`.

  The safetensors hashes were read from the Hub's LFS metadata on 2026-10-05 and re-hashed on
  download. The `.pth` twin of round 2, downloaded for the evaluator check, hashed
  `3fc00ad6…`, which matches the model card.
- `projectsidewalk/rampnet-crop-model-dataset-round2` at
  `9e902acf3bf23bb38122d3a7ebd0d9b9dcd5cfce`:
  - `test/`: 190 files, 147 images, 203 points;
  - `val/`: 142 files, 118 images, 157 points.

  These are manual labels. Val is the split `train.py` selects the checkpoint on (lowest val
  loss), so it is a selection split, not a training split.

## 4. Results

All numbers are in `analysis_out/crop_decode_221/results_{test,val,test_round1}.{json,md}`. Those
files carry every decode rule, every threshold, and the deduplicated read. Pairs at 0.30.

### 4.1 Round-2 checkpoint, test split (primary)

All files (the per-axis split; the deduplicated read is in section 1 and `results_test.md`):

| arm | pairs (images) | decode | mean px | median px | mean abs x | mean abs y | SD x | SD y |
|---|---|---|---|---|---|---|---|---|
| single | 244 (139) | argmax | 3.844 | 3.567 | 2.782 | 2.103 | 3.515 | 2.581 |
| single | | gaussian | 2.858 | 2.191 | 2.289 | 1.259 | 3.002 | 1.886 |
| TTA | 252 (139) | argmax | 3.917 | 3.612 | 2.861 | 2.114 | 3.576 | 2.528 |
| TTA | | gaussian | 2.906 | 2.218 | 2.481 | 1.080 | 3.165 | 1.512 |

Paired change, gaussian minus argmax, with 95% CIs that resample distinct images:

| arm | mean px | mean abs x | mean abs y | SD x | SD y |
|---|---|---|---|---|---|
| single | -0.987 [-1.213, -0.744] | -0.493 [-0.700, -0.275] | -0.844 [-1.049, -0.627] | -0.513 [-0.747, -0.261] | -0.695 [-1.032, -0.355] |
| TTA | -1.012 [-1.255, -0.775] | -0.380 [-0.592, -0.153] | -1.034 [-1.255, -0.815] | -0.412 [-0.639, -0.176] | -1.016 [-1.256, -0.763] |

The other refining rules land within 0.03 px of gaussian (single pass: parabola -0.974, dark
-1.012, centroid -0.997, quarter -0.688). `centre`, which snaps to the coarse centre, is worse
than argmax (+0.267). The ordering is the same as on the pano.

Threshold sensitivity (mean px change, single pass / TTA):
- at 0.0: -0.980 [-1.211, -0.726] / -0.955 [-1.209, -0.706];
- at 0.55: -0.994 [-1.237, -0.746] / -1.022 [-1.267, -0.785].

Sub-cell regression (gaussian; the label's offset from the coarse centre regressed on the
decoded offset, in cells):
- single pass: slope 1.049 in x, 0.883 in y;
- TTA: slope 0.965 in x, 0.974 in y.

A slope near 1 means the decoded offset carries the label's own sub-cell offset at about the
right scale.

**y bias.** The decode shifts mean dy down by about 0.4-0.6 px on both splits. Where the argmax's
y bias comes from is not established. It is not the 8i+3 / 8i+4 straddle: those rows sit
symmetrically about the cell centre, and the measured residues predict only about +0.04 px.

| split | single pass, argmax → gaussian | TTA, argmax → gaussian |
|---|---|---|
| test | +0.434 → +0.027 | +0.411 → -0.008 |
| val | +0.018 → -0.346 | +0.359 → -0.214 |

So on test it removes a bias, and on val it leaves one of similar size with the other sign.
What the decode reliably cuts is the y spread: SD y falls by 0.57-1.02 px on both splits.

### 4.2 Round-2 checkpoint, val split (replication)

| read | arm | pairs (images) | argmax mean px | gaussian mean px | change [95% CI] |
|---|---|---|---|---|---|
| all files | single | 179 (113) | 3.872 | 2.764 | -1.108 [-1.423, -0.777] |
| all files | TTA | 182 (114) | 3.899 | 2.938 | -0.960 [-1.292, -0.603] |
| deduplicated | single | 139 (113) | 4.073 | 2.877 | -1.196 [-1.448, -0.941] |
| deduplicated | TTA | 140 (114) | 4.085 | 3.009 | -1.077 [-1.335, -0.800] |

### 4.3 Round-1 checkpoint on the same test split

The round-1 checkpoint was trained on Project Sidewalk crops only; round 2 fine-tuned it on the
manual crops. I did not check whether any round-2 test image also appears in round-1 training.
The crop ids are random, so it cannot be read off the names.

| read | arm | pairs (images) | argmax mean px | gaussian mean px | change [95% CI] |
|---|---|---|---|---|---|
| all files | single | 198 (130) | 5.007 | 4.307 | -0.700 [-0.992, -0.391] |
| all files | TTA | 222 (136) | 4.682 | 3.952 | -0.730 [-0.978, -0.478] |
| deduplicated | single | 146 (130) | 5.014 | 4.263 | -0.751 [-1.028, -0.452] |
| deduplicated | TTA | 160 (136) | 4.725 | 4.019 | -0.706 [-0.941, -0.461] |

The decode helps both checkpoints. Round 1 is less accurate to begin with, and its gain is
smaller in absolute terms.

### 4.4 Detection metrics at the evaluator's protocol

Radius 0.132; AP is computed over all peaks at threshold 0.0. These rows count every file, as
the evaluator does, so an image stored twice counts twice. The deduplicated detection rows are in
the results files.

| checkpoint / split / arm | decode | AP | TP@0.30 | FP@0.30 | F1@0.30 | TP@0.55 | FP@0.55 |
|---|---|---|---|---|---|---|---|
| r2 / test / single | argmax | 0.8324 | 244 | 23 | 0.8730 | 241 | 19 |
| r2 / test / single | gaussian | 0.8488 | 247 | 20 | 0.8837 | 244 | 16 |
| r2 / test / TTA | argmax | 0.8580 | 252 | 32 | 0.8750 | 249 | 29 |
| r2 / test / TTA | gaussian | 0.8758 | 257 | 27 | 0.8924 | 254 | 24 |
| r2 / val / single | argmax | 0.8303 | 179 | 14 | 0.8775 | 177 | 9 |
| r2 / val / single | gaussian | 0.8362 | 180 | 13 | 0.8824 | 178 | 8 |
| r2 / val / TTA | argmax | 0.8371 | 182 | 15 | 0.8835 | 180 | 11 |
| r2 / val / TTA | gaussian | 0.8371 | 182 | 15 | 0.8835 | 180 | 11 |
| r1 / test / single | argmax | 0.6799 | 198 | 23 | 0.7719 | 187 | 18 |
| r1 / test / single | gaussian | 0.6845 | 197 | 24 | 0.7680 | 187 | 18 |
| r1 / test / TTA | argmax | 0.7377 | 222 | 36 | 0.8073 | 209 | 26 |
| r1 / test / TTA | gaussian | 0.7427 | 222 | 36 | 0.8073 | 210 | 25 |

Per-peak TP flips at 0.30, gaussian vs argmax (gained / lost). The argmax distance is each
flipped peak's distance to its nearest label, in units where the match radius is 11.25:

| checkpoint / split / arm | gained / lost | argmax distance of flipped peaks |
|---|---|---|
| r2 / test / single | +3 / -0 | 11.40-12.05 |
| r2 / test / TTA | +5 / -0 | 11.34-12.55 |
| r2 / val / single | +1 / -0 | 11.69 |
| r2 / val / TTA | 0 / 0 | |
| r1 / test / single | +2 / -3 | 8.91-12.46 |
| r1 / test / TTA | +1 / -1 | 10.31-11.77 |

The peaks and scores are identical under both decodes; only the positions differ. A peak flips
only when its position crosses the radius. **The change can be negative** (round 1, single
pass). No CI was computed on these counts.

### 4.5 The committed evaluator agrees with this script

`stage_one/crop_model/ps_and_manual_model/evaluate.py` from `origin/main`, run on the round-2
test split with the round-2 `.pth`, reports **AP 0.8580** (292 GT points, 4,553 predictions).
That equals this script's argmax TTA AP. This branch's evaluator with
`PEAK_DECODE = "gaussian"` reports **AP 0.8758**, equal to the script's gaussian TTA AP. Its
heatmap cache was byte-identical to `main`'s (190 of 190 maps).

Those GPU runs predate the section 10 change to the gaussian path's cache handling. A full
GPU re-run after that change was started and then stopped: other sessions held the desktop GPU
at 100%, and it had written 8 maps in 20 minutes. The change was checked on the CPU on the first
12 test crops instead, by running each of these in turn:

- gaussian with a fresh cache;
- gaussian again, reading that cache;
- argmax with this branch's file, reading that cache;
- `main`'s file with an empty cache.

All four gave the same AP (0.8947 on those 12). The heatmaps written by the gaussian path were
byte-identical to `main`'s (12 of 12).

The evaluator counts every file, so these APs weight a two- to four-ramp image two to four
times. The deduplicated TTA AP is 0.8667 under argmax and 0.8927 under gaussian (203 points).

The repo already holds one earlier crop-model number: the PR-curve PNG
`stage_one/crop_model/ps_and_manual_model/evaluation_results/pr_curve_dataset_1_test_r0.132_pt0.0.png`,
from the initial commit (`fea2785`), is titled **AP 0.6753**. It came from the evaluator as it
was before the matching fix in `ed9ec43`, on a `dataset_1/test` whose contents are not recorded.
It is neither reproduced nor explained here. That it lies near this branch's round-1 AP (0.6799)
may be a coincidence.

## 5. The x bias in the training targets

**The mismatch.** Both crop `train.py` files scale label coordinates by 0.5, but the image is
resized from 682 to 352 px wide (x 0.516). On the 88-px heatmap a point at true column t
therefore gets its target at k·t, with k = 682/704 = 0.969 (`docs/replication.md` describes this
for the labels).

**Both rounds train with horizontal flips.** Round 1 sets `apply_horizontal_flip=True` at
`stage_one/crop_model/ps_model/model/train.py` L118; round 2 sets it in
`ps_and_manual_model/train.py`. The mirrored target sits at `351 - 0.5x` input px. Averaged over
the two views, both rounds predict the same learned bias:

dx = label - detection = (1 - k)·t - 87.75·(1 - k)/2 = **+0.0312·t - 1.371**

That is a contraction toward column 44, of up to ±1.4 px at the crop edges. It is the same
order as the decode effect, so it is reported here per axis.

**Use the GT-x regression.** dx is regressed on the label's own column. The first version of this
document regressed dx on the detection's column instead. That regressor carries the detection
error e = det - GT, which also appears (negated) in dx, so its slope is pulled toward zero by
about var(e)/var(det x). The detection-x slopes are kept in the results files as a comparison
only.

Slope and intercept of dx on GT x (heatmap px; 95% CIs resample distinct images):

| checkpoint / split / arm | decode | slope | intercept |
|---|---|---|---|
| r2 / test / single | argmax | +0.0214 [-0.0052, +0.0474] | -0.269 [-1.546, +1.025] |
| r2 / test / single | gaussian | +0.0275 [+0.0053, +0.0479] | -0.670 [-1.671, +0.405] |
| r2 / test / TTA | gaussian | +0.0313 [+0.0087, +0.0531] | -0.520 [-1.637, +0.602] |
| r2 / test / single, dedup | gaussian | +0.0356 [+0.0127, +0.0572] | -1.069 [-2.086, +0.045] |
| r2 / val / single | gaussian | +0.0196 [-0.0068, +0.0478] | -1.367 [-2.602, -0.088] |
| r2 / val / TTA | gaussian | +0.0203 [-0.0095, +0.0496] | -0.814 [-2.138, +0.475] |
| r1 / test / single | argmax | +0.0622 [+0.0280, +0.0951] | -1.658 [-3.200, -0.105] |
| r1 / test / single | gaussian | +0.0806 [+0.0489, +0.1108] | -2.644 [-4.094, -1.176] |
| r1 / test / TTA | gaussian | +0.0595 [+0.0360, +0.0830] | -1.665 [-2.793, -0.509] |

**Dilution check.** The gap between the two regressions matches the predicted dilution
var(e)/var(det x). For round-2 test, single pass:

| decode | slope on GT x | slope on det x | gap | predicted dilution |
|---|---|---|---|---|
| gaussian | +0.0275 | +0.0069 | 0.021 | 0.022 |
| argmax | +0.0214 | -0.0074 | 0.029 | 0.029 |

So the first version's statements that round 2 "shows no slope" and that "under argmax the 8-px
steps blur it" were both this artefact.

What the corrected regression shows:

- **Round 2's slope matches the predicted geometry; a mean offset remains on test.**
  - On test, every gaussian slope CI covers +0.031 and excludes zero.
  - On val the slopes are lower (+0.020), and their CIs include both zero and +0.031.
  - The intercept CIs are wide and correlated with the slopes, so each one covering -1.37 does
    not test the predicted line.

  The direct test is the mean of dx minus the predicted dx at the label's x (gaussian, CIs
  resample distinct images):

  | split | single pass | TTA |
  |---|---|---|
  | test | +0.545 [+0.093, +1.024] | +0.852 [+0.367, +1.336] |
  | val | -0.512 [-1.031, +0.001] | +0.077 [-0.443, +0.587] |

  So on test the labels sit about 0.5-0.9 px right of where the training geometry puts the
  detections. That offset is not explained. On val the single-pass offset has the opposite
  sign, and the TTA offset is near zero. The all-files val single-pass CI just touches zero, but
  the deduplicated read excludes it (-0.562 [-1.059, -0.048]). TTA moves the test offset by +0.31 px and the val
  offset by +0.59 px. One untested candidate for that TTA shift is the half-pixel asymmetry of
  `np.fliplr`, which maps index p to 87 - p, against a continuous mirror of the 682-px image.
  The first version's "unexplained uniform offset" was partly this offset and partly the
  detection-x regression's artefact.
- **Round 1 overshoots the prediction.** Its gaussian slope (+0.081 single pass, +0.060 TTA) has
  a CI that excludes +0.031, although its training geometry is the same as round 2's. Why round
  1 learned a stronger contraction is not established here. Its labels are Project Sidewalk
  points, which the manual round-2 labels replaced. Round 1 also has a mean offset beyond the
  predicted line: +0.748 [+0.100, +1.386] single pass, +0.884 [+0.369, +1.403] TTA.
- **The decode does not remove the bias, and the bias does not remove the decode's gain.** The
  slope is a property of the learned peak, not of the quantization. Correcting detections' x
  for the predicted bias does not reduce the round-2 mean error: the `flip_average` correction
  changes it by +0.033 [-0.042, +0.107] px (gaussian, single pass, test). The 704/width
  rescale (`label_scale`, which ignores the flips) changes it by +0.151 [-0.033, +0.332] px.
  The decode's gain holds under either: gaussian vs argmax is -1.036 [-1.262, -0.795] after
  `flip_average`.

## 6. What this means for Stage 1

Stage 1 placement never peak-picks the crop heatmap. `stage_one/dataset_generation/
download_dataset.py::process_line` (L34-76) does the following for each government ramp location:

1. Renders a 1024x1024 perspective at -30 degrees pitch and takes the middle 341 columns.
2. Runs `infer_image` (single pass, no TTA) and resizes the 256x88 output to 341x1024 with
   `cv2.INTER_CUBIC`.
3. Pads it back to 1024 wide, clips it to [0, 1], scales it to 0-255 and reprojects it to a
   2048x4096 equirectangular map.
4. Casts to `uint8` and takes the `np.maximum` across all crops of the pano.
5. Runs `peak_local_max(min_distance=40, threshold_abs=0.4*255)` on that pano-wide map.

A sub-cell decode cannot be dropped into that path. The peak is found after a cubic resize, a
reprojection and a `uint8` cast. There is no point-level crop-to-pano mapping in
`rampnet/gsv.py` to carry a refined crop position across, and no rule for merging refined points
from overlapping crops. Two designs are possible:

- refine each crop's peaks on the 32x11 map, map those points to the pano, then merge;
- keep the current map-level path and accept its quantization.

That choice is a design decision for Jon and is not built here. Whether the current path
inherits the 8-px grid after the cubic resize and reprojection was **not measured**. The two
cases look alike: on the bilinear surface the maxima sit at the coarse centres, and a cubic
resample of it should keep them near there.

## 7. Cost

All runs were on the desktop RTX 3070 in fp32. Every row in `analysis_out/usage_log.jsonl` is
`paid: false`.

| run | wall-clock | GPU-h |
|---|---|---|
| extract, test, round 2 (380 forwards) | 40.1 s | 0.0111 |
| extract, val, round 2 (284 forwards) | 54.1 s | 0.0150 |
| extract, test, round 1 (380 forwards) | 89.8 s | 0.0249 |
| evaluator check: three fresh `evaluate.py` runs (1,140 forwards, PNG writes included) | 650 s | 0.1806 |
| evaluator re-check after the section 10 cache fix: aborted GPU attempt (8 maps, GPU held by other sessions) plus a 12-crop CPU check | about 25 min (GPU, mostly waiting) + 124 s (CPU) | 0.42 (upper bound) |

- The GPU-hours are upper bounds, because other sessions on the desktop may have used the GPU at
  the same time and that was not monitored.
- The Hub fetch took 54 s (332 JPEGs and two 360 MB safetensors). The `.pth` download was not
  timed.
- Report and `--check` run on the CPU and take about 40 s for all three extracts.

**Environment.** During the first run, another process replaced torch in the shared desktop venv
(`D:/Git/RampNet/.venv`) with 2.14.1+cpu, which broke the torchvision import. Every number here
comes from a private venv with torch 2.6.0+cu126, torchvision 0.21.0+cu126 and timm 1.0.28, the
versions the shared venv had before.

## 8. Reproduce

From a clean clone, with `PYTHONPATH` set to the repo root:

```bash
python scripts/analysis/crop_decode_221.py fetch --cache .hf_cache \
    --out analysis_out/crop_decode_221/inputs.json
python scripts/analysis/crop_decode_221.py extract --split test --cache .hf_cache \
    --out analysis_out/crop_decode_221/extract_test.json \
    --usage-out analysis_out/crop_decode_221/usage_test.json
python scripts/analysis/crop_decode_221.py extract --split val --cache .hf_cache \
    --out analysis_out/crop_decode_221/extract_val.json \
    --usage-out analysis_out/crop_decode_221/usage_val.json
python scripts/analysis/crop_decode_221.py extract --split test --checkpoint round1 \
    --cache .hf_cache --out analysis_out/crop_decode_221/extract_test_round1.json \
    --usage-out analysis_out/crop_decode_221/usage_test_round1.json
for s in test val test_round1; do
  python scripts/analysis/crop_decode_221.py report \
      --extract analysis_out/crop_decode_221/extract_$s.json \
      --out analysis_out/crop_decode_221/results_$s.json \
      --md analysis_out/crop_decode_221/results_$s.md
done
python scripts/analysis/crop_decode_221.py --check     # CPU only, ~40 s
```

Each extract holds, for every file, its two 32x11 coarse maps (to 7 decimals), its GT points and
the image's sha256. The results re-derive from the extracts with no GPU, no checkpoint and no
download. `--check` compares all six results files byte for byte. A re-extraction on other
hardware will differ in the last stored decimals (fp32 noise), so compare its results, not its
extract.

For the evaluator check (section 4.5), run from a scratch directory:

```bash
mkdir -p evalrun/dataset_1/test && cp <.hf_cache round-2 snapshot>/test/*.jpg evalrun/dataset_1/test/
python -c "from huggingface_hub import hf_hub_download; import shutil; shutil.copy(hf_hub_download('projectsidewalk/rampnet-crop-model', 'round2_ps_and_manual_best_model.pth', revision='7aa79b8edb10b384ed69c2e99f74945e9c527fd3'), 'evalrun/best_model.pth')"
cd evalrun && python <repo>/stage_one/crop_model/ps_and_manual_model/evaluate.py      # argmax
python -c "import sys; sys.path.insert(0, '<repo>/stage_one/crop_model/ps_and_manual_model'); import evaluate as e; e.PEAK_DECODE = 'gaussian'; e.main()"
```

## 9. Not run, and why

- **Round-1 Parquet test crops (~1,350 Project Sidewalk-labelled crops).** Optional in the plan.
  Their labels carry the #113 tilt leak and the x mismatch, which would confound a position
  read, and the two manual splits already agree.
- **Round-2 train split.** It is training data, so it would not be an honest read.
- **Stage 1 placement error with and without the decode.** This needs the crop-to-pano point
  design in section 6.
- **A CI on the TP and AP changes.** Reported as counts only.
- **Why round 1's x slope exceeds the predicted geometry, why round-2 test keeps a mean x offset, and what causes the TTA x shift** (section 5).
- **Nothing was pushed to the Hub.** The crop model card template
  (`scripts/hf_package/README.crop_model_card.template.md`) now states the measured number. The
  live card changes only when someone republishes it.

## 10. Revisions after the PR #247 review

The first version of this document (commit `3947d02`) had these problems. All are corrected
above; the numbers re-derive from the same committed extracts, with no new GPU extraction.

- **CIs resampled files, not images.** The round-2 splits store a multi-point crop once per
  point, so test's 190 files are 147 images and its 292 points are 203. The bootstrap now
  resamples distinct images (sha256), and a deduplicated read is reported. The point estimates
  did not change. The CIs widened by 25-40%: test single pass went from [-1.167, -0.791] to
  [-1.213, -0.744]. Every CI still excludes zero.
- **The x-bias slope was regressed on detection x**, an errors-in-variables estimate biased
  toward zero. Section 5 now uses GT x. Round 2 does show the predicted slope, which reverses the
  first version's "no slope". A mean x offset beyond the predicted line remains on test and is
  not explained; it is now measured directly (section 5). On val it has the opposite sign in the
  single pass.
- **Round 1 was said to train without flips.** It trains with flips (`ps_model/model/train.py`
  L118). Both rounds predict the same geometry.
- **The document said the repo held no earlier crop-model metric.** The 0.6753 PNG from the
  initial commit is now cited (section 4.5).
- **Round-1 detection rows and per-peak TP flips were missing.** Both are added; the change can
  be negative.
- **The y-bias claim held on test only.** Both splits are now reported (section 4.1). The
  attribution of the argmax's y bias to the 8i+3 / 8i+4 straddle is withdrawn; it predicts about
  +0.04 px.
- **Smaller fixes:**
  - the evaluator's gaussian path no longer re-reads a heatmap it has just written;
  - `--check` reports a missing results file instead of raising;
  - the report renders an empty slice instead of failing on it;
  - section 8 gains the `.pth` download command.
