# Sub-cell decode on the Stage 1 crop model (#221)

Run 2026-10-05 on branch `analysis/crop-decode-221`. Script:
[`scripts/analysis/crop_decode_221.py`](../scripts/analysis/crop_decode_221.py). Committed outputs:
[`analysis_out/crop_decode_221/`](../analysis_out/crop_decode_221/).

## 1. Result

The Gaussian sub-cell decode that shipped for the pano model (`rampnet/subcell.py`, PRs #229 and
#233) also moves the crop model's peaks toward the labelled points. On the 190 manually labelled
round-2 test crops (292 points), with pairs fixed on the argmax peaks at score >= 0.30, the mean
distance from detection to label falls from **3.844 to 2.858 heatmap px** in a single pass, a
paired change of **-0.987 px [95% CI -1.167, -0.791]** (244 pairs in 182 crops), and from
**3.917 to 2.906 px** under the evaluator's flip TTA, **-1.012 px [-1.199, -0.819]** (252 pairs).
That is a 26% cut. The pano model's cut was 14% (5.080 to 4.353 px on `manual_gold`). The
round-2 val split replicates it: -1.108 px [-1.342, -0.874] single pass, -0.960 px
[-1.203, -0.713] TTA. The CI excludes zero on every split, arm and threshold measured.

A heatmap pixel here is 4 model-input pixels, about 8 source pixels of a 682x2048 crop
(7.75 in x, 8 in y). The training target's sigma is 12 heatmap px, so the mean error goes from
0.32 sigma to 0.24 sigma. The evaluator's match radius is 11.25 px.

Unlike the pano read, detection metrics move a little on the test split. Under TTA at 0.30 the
gaussian decode turns 5 false positives into true positives (TP 252 to 257 of 292, recall
0.863 to 0.880) and AP rises from 0.8580 to 0.8758. Single pass: TP 244 to 247, AP 0.8324 to
0.8488. Those are argmax peaks sitting just outside the 11.25 px radius that the decode moves
inside. The val split shows almost none of it (TTA: no change; single pass: TP 179 to 180), so
treat the detection gain as small and split-dependent. The position gain is the finding.

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

Before decoding, 238 of 244 argmax peak columns (single pass, test pairs) sit on heatmap pixel 3
or 4 mod 8; under the gaussian decode the eight residues are all populated
(`results_test.md`, mod-8 histograms). The labels themselves spread over all eight residues.

## 3. Method

What is the same as the pano read (`docs/subcell_decode_221.md` section 3): peaks with
`detect_peaks(..., min_distance=10, clip=True, exclude_border=False)` from `rampnet/subcell.py`;
pairs fixed once on the argmax positions, greedy by confidence; every decode rule in
`subcell.METHODS` scored on the same pairs; residual = label - detection, per axis; a cluster
bootstrap (2,000 reps, seed 221) of the paired change; sub-cell regression slopes; mod-8
histograms.

What differs, all taken from the committed crop evaluator
(`stage_one/crop_model/ps_and_manual_model/evaluate.py`) verbatim:

- No x wrap. A crop is not a panorama.
- Matching geometry: radius 0.132 normalized with `scale_x = 341/4`, `scale_y = 1024/4`, so
  11.25 units. The crops are 682 px wide (the evaluator's 341 is from the half-width
  generation), so in x one unit is 88/85.25 heatmap px. Used as is; not "fixed".
- Flip TTA as the evaluator does it: clip each branch to [0, 1], flip the mirrored one back,
  elementwise max. The decode reads each peak from the branch the max took it from (the
  shipped `coarse=(2, 32, 11)` branch-select rule).
- The bootstrap unit is the crop.
- Residuals are in pixels of the 256x88 heatmap.
- GT is parsed from the filenames (`{uid}_-_{x}_{y}[_-_{x}_{y}...].jpg`) and normalized by the
  image's own width and height, as the evaluator does. One val crop is 704 px wide; all
  others are 682.

Two arms: **single pass**, which is what `inference_isolator.infer_image` runs, and **flip
TTA**, which is what the crop evaluator runs. Three thresholds: 0.30 (the pairs reported
above), 0.0 (the evaluator's) and 0.55.

Inputs, all pinned in `analysis_out/crop_decode_221/inputs.json` with the sha256 of every
file:

- `projectsidewalk/rampnet-crop-model` at `7aa79b8edb10b384ed69c2e99f74945e9c527fd3`, files
  `round2_ps_and_manual_best_model.safetensors` (sha256 `d129c0c6…`) and
  `round1_ps_best_model.safetensors` (`23e40b59…`). The safetensors hashes were read from the
  Hub's LFS metadata on 2026-10-05 and re-hashed on download. The `.pth` twin of round 2,
  downloaded for the evaluator check, hashed `3fc00ad6…`, which matches the model card.
- `projectsidewalk/rampnet-crop-model-dataset-round2` at
  `9e902acf3bf23bb38122d3a7ebd0d9b9dcd5cfce`, `test/` (190 crops, 292 points) and `val/`
  (142 crops, 215 points). These are manual labels. Val is the split `train.py` selects the
  checkpoint on (lowest val loss), so it is a selection split, not a training split.

## 4. Results

All numbers are in `analysis_out/crop_decode_221/results_{test,val,test_round1}.{json,md}`,
which include every decode rule and every threshold. Pairs at 0.30.

### 4.1 Round-2 checkpoint, test split (primary)

| arm | pairs (crops) | decode | mean px | median px | mean abs x | mean abs y | SD x | SD y |
|---|---|---|---|---|---|---|---|---|
| single | 244 (182) | argmax | 3.844 | 3.567 | 2.782 | 2.103 | 3.515 | 2.581 |
| single | | gaussian | 2.858 | 2.191 | 2.289 | 1.259 | 3.002 | 1.886 |
| TTA | 252 (181) | argmax | 3.917 | 3.612 | 2.861 | 2.114 | 3.576 | 2.528 |
| TTA | | gaussian | 2.906 | 2.218 | 2.481 | 1.080 | 3.165 | 1.512 |

Paired change, gaussian minus argmax, 95% crop-cluster bootstrap CI:

| arm | mean px | mean abs x | mean abs y | SD x | SD y |
|---|---|---|---|---|---|
| single | -0.987 [-1.167, -0.791] | -0.493 [-0.668, -0.319] | -0.844 [-1.006, -0.673] | -0.513 [-0.725, -0.311] | -0.695 [-0.936, -0.438] |
| TTA | -1.012 [-1.199, -0.819] | -0.380 [-0.557, -0.209] | -1.034 [-1.211, -0.856] | -0.412 [-0.598, -0.220] | -1.016 [-1.209, -0.807] |

The other refining rules land within 0.03 px of gaussian (single pass: parabola -0.974, dark
-1.012, centroid -0.997, quarter -0.688). `centre`, which snaps to the coarse centre, is worse
than argmax (+0.267 [+0.167, +0.355]). Same ordering as the pano.

Threshold sensitivity (single pass / TTA, mean px change): at 0.0, -0.980 [-1.162, -0.790] /
-0.955 [-1.144, -0.761]; at 0.55, -0.994 [-1.182, -0.803] / -1.022 [-1.226, -0.825].

Sub-cell regression (gaussian, label offset from the coarse centre on decoded offset, in cells):
slope 1.049 in x and 0.883 in y single pass, 0.965 and 0.974 under TTA. A slope near 1 means
the decoded offset carries the label's own sub-cell offset at about the right scale.

### 4.2 Round-2 checkpoint, val split (replication)

| arm | pairs (crops) | argmax mean px | gaussian mean px | change [95% CI] |
|---|---|---|---|---|
| single | 179 (135) | 3.872 | 2.764 | -1.108 [-1.342, -0.874] |
| TTA | 182 (138) | 3.899 | 2.938 | -0.960 [-1.203, -0.713] |

### 4.3 Round-1 checkpoint on the same test split

The round-1 checkpoint was trained on Project Sidewalk crops only; round 2 fine-tuned it on the
manual crops. Whether any round-2 test image also appears in round-1 training was not checked
(the crop ids are random, so it cannot be read off the names).

| arm | pairs (crops) | argmax mean px | gaussian mean px | change [95% CI] |
|---|---|---|---|---|
| single | 198 (171) | 5.007 | 4.307 | -0.700 [-0.936, -0.463] |
| TTA | 222 (178) | 4.682 | 3.952 | -0.730 [-0.928, -0.531] |

The decode helps both checkpoints. Round 1 is less accurate to begin with, and its gain is
smaller in absolute terms.

### 4.4 Detection metrics at the evaluator's protocol

Radius 0.132, AP over all peaks at threshold 0.0.

| split / arm | decode | AP | TP@0.30 | FP@0.30 | F1@0.30 | TP@0.55 | FP@0.55 |
|---|---|---|---|---|---|---|---|
| test, single | argmax | 0.8324 | 244 | 23 | 0.8730 | 241 | 19 |
| test, single | gaussian | 0.8488 | 247 | 20 | 0.8837 | 244 | 16 |
| test, TTA | argmax | 0.8580 | 252 | 32 | 0.8750 | 249 | 29 |
| test, TTA | gaussian | 0.8758 | 257 | 27 | 0.8924 | 254 | 24 |
| val, single | argmax | 0.8303 | 179 | 14 | 0.8775 | 177 | 9 |
| val, single | gaussian | 0.8362 | 180 | 13 | 0.8824 | 178 | 8 |
| val, TTA | argmax | 0.8371 | 182 | 15 | 0.8835 | 180 | 11 |
| val, TTA | gaussian | 0.8371 | 182 | 15 | 0.8835 | 180 | 11 |

The peaks and scores are identical under both decodes (asserted in the script); only the
positions differ. No CI was computed on the TP or AP change.

### 4.5 The committed evaluator agrees, and this is its first committed number

`stage_one/crop_model/ps_and_manual_model/evaluate.py` from `origin/main`, run on the round-2
test split with the round-2 `.pth`, reports **AP 0.8580** (292 GT points, 4,553 predictions).
That equals this script's argmax TTA AP. This branch's evaluator with `PEAK_DECODE = "gaussian"`
reports **AP 0.8758**, equal to the script's gaussian TTA AP. Its heatmap cache was
byte-identical to `main`'s (190 of 190 maps). The repo had no committed metric number for the
crop model before this (only two PR-curve PNGs in `evaluation_results/`), so 0.8580 is the first.

## 5. The x bias in the training targets

Both crop `train.py` files scale label coordinates by 0.5, but the image is resized from 682 to
352 px wide (x 0.516). On the 88-px heatmap a point at true column t gets its target at
k·t with k = 682/704 = 0.969: up to 2.75 px left at the right edge (`docs/replication.md`
describes this for the labels). That is the same order as the decode effect, so it is reported
here per axis.

What the training geometry predicts depends on augmentation. Round 1 trains without flips, so
the expected learned bias is dx = label - detection = +0.031·t. Round 2 trains with
`apply_horizontal_flip=True`, and the mirrored target sits at `351 - 0.5x` input px. Averaged over
both views, that is a contraction toward column 44, with dx = 0.031·t - 1.37. The slope is
+0.031 in both cases; only the intercept differs.

Measured slope of dx on detection x (heatmap px per px, 95% crop-cluster bootstrap CI):

| checkpoint / split / arm | argmax | gaussian |
|---|---|---|
| round 2 / test / single | -0.0074 [-0.0280, +0.0156] | +0.0069 [-0.0114, +0.0250] |
| round 2 / test / TTA | -0.0052 [-0.0291, +0.0170] | +0.0087 [-0.0115, +0.0286] |
| round 2 / val / single | -0.0072 [-0.0329, +0.0197] | +0.0013 [-0.0220, +0.0255] |
| round 2 / val / TTA | -0.0045 [-0.0301, +0.0219] | +0.0008 [-0.0224, +0.0250] |
| round 1 / test / single | +0.0257 [-0.0018, +0.0556] | +0.0545 [+0.0306, +0.0828] |
| round 1 / test / TTA | +0.0115 [-0.0108, +0.0335] | +0.0376 [+0.0177, +0.0591] |

- **Round 1 shows the predicted slope.** Under the gaussian decode the CI covers +0.031. Under
  argmax the 8-px steps blur it.
- **Round 2 does not.** Every round-2 CI includes zero and stops short of +0.031, narrowly (the
  highest upper bound is +0.0286). How round 2's flip augmentation and fine-tuning removed it
  is not established here.
- **The x bias is not removed by the decode.** Round 2 has a roughly uniform x offset of
  +0.48 px (single pass) and +0.82 px (TTA) under gaussian on test, with no slope. On val it is
  -0.50 and +0.08. The test-vs-val sign change and the TTA shift are measured, not explained.
  One untested candidate for the TTA shift is the half-pixel asymmetry of `np.fliplr` (index
  p to 87 - p) against a continuous mirror of the 682-px image.
- **Neither correction helps round 2.** Rescaling detections' x by 704/width
  (`label_scale`) or inverting the flip-averaged model (`flip_average`) leaves the mean error
  the same or worse (`results_test.md`, "x corrected" lines). The decode's gain holds under
  both: gaussian vs argmax is -0.879 [-1.068, -0.683] (label_scale) and -1.036
  [-1.213, -0.839] (flip_average), single pass.

y has no such mismatch. The decode removes the argmax's y bias (+0.434 to +0.027 px, single
pass) along with most of its spread.

## 6. What this means for Stage 1

Stage 1 placement never peak-picks the crop heatmap. `stage_one/dataset_generation/
download_dataset.py::process_line` (L34-76) does the following for each government ramp
location:

1. Renders a 1024x1024 perspective at -30 degrees pitch and takes the middle 341 columns.
2. Runs `infer_image` (single pass, no TTA) and resizes the 256x88 output to 341x1024 with
   `cv2.INTER_CUBIC`.
3. Pads the result back to 1024 wide, clips it to [0, 1], scales it to 0-255 and reprojects it
   to a 2048x4096 equirectangular map.
4. Casts to `uint8` and takes the `np.maximum` across all crops of the pano.
5. Runs `peak_local_max(min_distance=40, threshold_abs=0.4*255)` on that pano-wide map.

A sub-cell decode cannot be dropped into that path. The peak is found after a cubic resize, a
reprojection and a `uint8` cast. There is no point-level crop-to-pano mapping in
`rampnet/gsv.py` to carry a refined crop position across, and no rule for merging refined
points from overlapping crops. Two designs would work:

- refine each crop's peaks on the 32x11 map, map those points to the pano, then merge them
  (needs the point mapping and a merge rule);
- keep the current map-level path and accept its quantization.

That is a design decision for Jon and is not built here. Whether the current path inherits the
8-px grid after the cubic resize and reprojection was **not measured**. The two cases look alike:
on the bilinear surface the maxima sit at the coarse centres, and a cubic resample of it should
keep them near there.

## 7. Cost

Desktop RTX 3070, fp32. Five rows in `analysis_out/usage_log.jsonl`, all `paid: false`:

| run | wall-clock | GPU-h |
|---|---|---|
| extract, test, round 2 (380 forwards) | 40.1 s | 0.0111 |
| extract, val, round 2 (284 forwards) | 54.1 s | 0.0150 |
| extract, test, round 1 (380 forwards) | 89.8 s | 0.0249 |
| evaluator check: three fresh `evaluate.py` runs (1,140 forwards, PNG writes included) | 650 s | 0.1806 |
| total | 834 s | 0.2317 |

Other sessions on the desktop may have used the GPU concurrently. That was not monitored, so
the GPU-hours are upper bounds. The Hub fetch took 54 s (332 JPEGs, two 360 MB safetensors).
The `.pth` download for the evaluator check was not timed. Report and `--check` are CPU-only and
take about 40 s for all three extracts.

The shared desktop venv (`D:/Git/RampNet/.venv`) had its torch replaced with 2.14.1+cpu by
another process during this session, which broke the torchvision import. Every number here
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

The extracts hold every crop's two 32x11 coarse maps (to 7 decimals), its GT points and the
image sha256. The results re-derive from them with no GPU, no checkpoint and no download.
`--check` compares all six results files byte for byte. A re-extraction on other hardware will
differ in the last stored decimals (fp32 noise), so compare its results, not its extract.

For the evaluator check (section 4.5): put the round-2 test JPEGs in `dataset_1/test/` and
`round2_ps_and_manual_best_model.pth` as `best_model.pth`, both from the pinned revisions. Run
the evaluator from that directory, then again with `PEAK_DECODE = "gaussian"`.

## 9. Not run, and why

- **Round-1 Parquet test crops (~1,350 Project Sidewalk-labelled crops).** Optional in the plan.
  Their labels carry the #113 tilt leak and the x mismatch, which would confound a position
  read, and two manual splits already agree.
- **Round-2 train split.** Training data; not an honest read.
- **Stage 1 placement error with and without the decode.** Needs the crop-to-pano point design
  in section 6.
- **A CI on the TP and AP changes.** Reported as counts only.
- **The cause of the uniform round-2 x offset and of the TTA x shift** (section 5).
- **Nothing was pushed to the Hub.** The crop model card template
  (`scripts/hf_package/README.crop_model_card.template.md`) now states the measured number. The
  live card changes only when someone republishes it.
