# Sub-cell decode of RampNet peaks (#221)

Issue [#221](https://github.com/ProjectSidewalk/RampNet/issues/221): the pano model's 512x1024
heatmap has an effective resolution of 64x128, so every detection sits on an 8-pixel grid. This
doc covers items 1 and 2 of that issue. It measures a sub-cell decode against human box centres
and records that `heatmap_size` is nominal. Item 3 (a head that can express sub-cell position
natively) is a RampNet 2.0 question and is not addressed here.

Run 2026-09-30 on makelab2 (one NVIDIA A40), released checkpoint `projectsidewalk/rampnet-model`
at Hub commit `606a11956743f7eb328d9207769034752f6191f4` (weights sha256 `f2119e3b...`), single
pass (no flip TTA), 2048x4096 input. Fixes from the independent
[review of PR #226](https://github.com/ProjectSidewalk/RampNet/pull/226#pullrequestreview-5371536820)
are folded in. No headline number changed.

## 1. Result in one paragraph

The quantization is real and it matters. The released model puts 99.0% of peaks (3,829 of 3,868
columns on manual_gold) on hi-res pixel 3 or 4 mod 8, and every exception is a plateau. Refining
each peak from its 3x3 coarse neighbourhood lowers the mean distance to human box centres on all
five splits measured. The CI excludes zero on four of them; annapolis is the exception. On the
1,000-pano `manual_gold` set, a
Gaussian (log-parabola) decode moves detections toward the box centres by **-0.727 px
[-0.783, -0.673]** on average (5.080 -> 4.353 px on the 512x1024 grid; 1.726 -> 1.473 degrees
great-circle), over 3,517 matched pairs in 787 panos. Regressing the GT's own sub-cell offset on the
decoded one gives a slope of about 1 (0.996 in x, 0.972 in y; no CI on the slopes). Matched detection
counts change by at most one per split, so no detection metric moves. On the cross-view harness's
reference-noise read ([#48](https://github.com/ProjectSidewalk/RampNet/issues/48)), the median
peak-to-box distance falls from 1.510 to **1.197 degrees** (-0.314 [-0.344, -0.280]).
**Recommendation:** make `gaussian` the decode rule in the labeler's `detections_from_heatmap`
([sidewalk-auto-labeler#111](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/111)).
It needs no retraining and about 60 lines of numpy.

## Examples

Six manual_gold ramps, chosen by a fixed rule from the 3,517 matched pairs: the two largest
improvements, the two closest to the median |d|, the one where the decode moved furthest away
from the box centre (all five from peaks with score <= 1, not climbed, not at the seam), and
one clipped plateau (score > 1, argmax off the 8i+3 / 8i+4 grid, d nearest that group's
median). At most one example is taken per pano; that rule did not change any pick here. d is
the change in distance to the box centre, Gaussian minus argmax.

To calibrate the six: of the 3,300 eligible pairs, 65.2% move closer and 34.8% move further
away, with median d -0.80 px. The rule takes two best cases and one worst, so the sheet leans
favourable by construction, and one of the two "typical" panels (d +1.49) is a regression.
The "large" and "moved away" panels show the decode at its reach: their Gaussian offsets are
0.43-0.50 cell, close to the 0.5-cell clamp (`MAX_OFFSET`), so |d| cannot get much larger.

![Contact sheet of six manual_gold crops. In four of the six, the Gaussian decode (circle) sits closer to the human box centre (plus) than the argmax (square); in the "typical" regression and the "moved away" crop it sits further. Every argmax sits 2 input px from the centre of its coarse cell on each axis, except the clipped plateau.](figures/subcell_decode_221/examples_contact_sheet.jpg)

- Many manual_gold "boxes" are only a few pixels across, so they read as a point label, and
  the box is often hidden under its centre mark.
- In the "moved away" example (`hjGc_EK1u9NfwcU5eciW1g`), the decode moves the peak up and
  left, further onto the tactile pad. The human mark sits lower, on the kerb line at the
  pad's front edge, and the argmax happened to fall between them (4.4 px to 8.8 px). The
  decode follows the coarse map faithfully here (the neighbours up and to the left are higher,
  giving an offset of 0.48 / 0.43 cell), so this is consistent with the concept offset in
  section 5, where the model and the labeller mark different points of the ramp. One image
  cannot show that this is the cause.

The mechanism panel uses the first "typical" example. Its heatmap is the released model's
output on that pano, re-run on a desktop RTX 3070. It agrees with the bilinear upsample of
the run's coarse map (the makelab2 A40 map; only its sha256 is committed, in
`detections.json`) to 2.0e-5, which is cross-GPU fp32 noise. The two histograms are the
manual_gold rows of the section 4.3 table.

![Mechanism panel. Top: the 512x1024 heatmap around one peak looks smooth, but the 64x128 coarse map it was upsampled from shows the same 8x8 cells; a profile down the peak column is piecewise linear with kinks at the coarse sample positions. Bottom: argmax positions mod 8 fall almost only on 3 and 4 on both axes, while Gaussian-decoded positions and human box centres spread across all eight.](figures/subcell_decode_221/mechanism_panel.png)

Both figures come from `scripts/analysis/subcell_decode_221_figures.py` (commands in
section 8). The crops are small documentation extracts of the manual_gold panos in the HF
`projectsidewalk/rampnet-dataset` test split.

## 2. Mechanism, verified

The head is `Conv2d(3x3) -> ReLU -> Upsample(512x1024, bilinear, align_corners=False) ->
Conv2d(1x1)`. The 1x1 conv is linear, and its bias commutes with bilinear resampling because the
weights sum to one. So the output equals a bilinear upsample of `Conv1x1(ReLU(Conv3x3(f)))`
evaluated at the 64x128 feature resolution. That 64x128 map is called the *coarse map* below.
A bilinear surface has its maxima at its sample points. Coarse centre `i` sits at hi-res index
`8i + 3.5`, so an integer argmax lands on `8i+3` or `8i+4`.

Measured, per pano, in `extract`:

- `max |model.head(f) - interpolate(coarse)|` is **at most 3.6e-7** per pano in torch (2.9e-8 to
  3.58e-7, non-zero on all 1,499 panos), which is fp32 rounding. The numpy operator in
  `rampnet/subcell.py` gives at most 3.16e-7 (`results.md`, Mechanism). An earlier version of this
  doc said "0". That was these values rounded to 4 decimals.
  The unit test `test_real_head_output_is_upsampled_coarse_map` checks the same identity on the
  real `KeypointModel.head` with random features.
- Peaks at score >= 0.30 with column mod 8 in {3, 4}: manual_gold 3,829 / 3,868, annapolis 270 /
  272, paterson 310 / 312, richmond 325 / 328, sao_paulo 320 / 320. Rows: manual_gold 3,848, annapolis
  271, paterson 312, richmond 328, sao_paulo 318. sao_paulo is the one split where rows are
  further off-grid than columns.
  **Every** off-grid peak (row or column) has one of two causes, both plateaus (`results.json`,
  `off_grid_other_than_plateau_or_col0` = 0 on all five splits):
  - **Clipped tops.** `peak_local_max` runs on `clip(h, 0, 1)`, so a peak above 1 becomes a flat
    top and the pixel it returns is arbitrary. This accounts for 40 of manual_gold's 58 off-grid
    peaks, out of 165 peaks above 1.
  - **Column 0.** Bilinear upsampling clamps at the edges, so hi-res columns 0-3 all equal coarse
    column 0. This accounts for the other 18 on manual_gold. The right edge has the same plateau
    (1020-1023), but the pixel returned there is 1020 = 8 x 127 + 4, which happens to count as
    on-grid.

  Neither case needed the climb in this data. In all 187 peaks above 1 (165 on manual_gold), the
  returned pixel was already in the coarse-max cell. The column-0 plateau is handled by the
  missing left neighbour (NaN), which leaves x unrefined. `rampnet.subcell.climb` did something
  else: it moved **48 peaks** (39 of 3,868 on manual_gold, 3 paterson, 2 richmond, 4 sao_paulo,
  0 annapolis), and every one was **on-grid** with score <= 1. Each hi-res pixel mixes four
  coarse values, so a strong diagonal neighbour can pull the integer argmax into the cell next to
  the true coarse maximum. Climbing re-anchors those peaks. On the 34 manual_gold pairs among
  them, the mean residual is 5.34 px under argmax and 4.31 px under `gaussian` (`results.md`,
  "Climbed pairs").
- A 1-px within-cell tie is visible in the instrument check. One manual_gold peak (pano
  `UaJ4KZfBet6CqD-BpX66xQ`) sits at row 292 here and 291 in the committed `op_cache`. Those are
  the two pixels flanking one coarse centre (8x36+3 and 8x36+4), with scores 0.579273 vs
  0.579275, so the only difference is cross-machine fp32 noise. Both pixels are in coarse cell
  36, so every refined decode places this peak identically, and the decode removes this 1-px flip
  outright. It is **not** the 7-px flip described in
  [#221](https://github.com/ProjectSidewalk/RampNet/issues/221), which is a change in *which*
  coarse cell is the maximum (8i+4 to 8(i+1)+3). The decode does not address that one.

### Which map to decode from

The coarse map and the 512x1024 heatmap carry the same information. The heatmap is an exact linear
function of the coarse map, and `rampnet.subcell.coarse_from_heatmap` inverts it by least squares
against the bilinear operator. The measurement decodes from the **coarse map**, for two reasons:

1. The refinement rules (parabola, Gaussian, DARK) assume samples of a smooth peak at unit
   spacing. On the coarse grid that is exactly what they get. On the hi-res grid the 3x3
   neighbourhood of the argmax lies within one bilinear facet, and it carries only first-order
   information.
2. It is 64x smaller. Each peak's 3x3 coarse neighbourhood is committed (9 floats), so every
   number here re-derives on CPU without the model.

A caller that only has `model(x)`, like the labeler, calls `refine_peaks(heatmap, peaks)`. That
recovers the coarse map from the heatmap, exactly up to float error, and then applies the same
rule.

## 3. Method

**Decode rules** (`rampnet/subcell.py`, `METHODS`). Each gives an offset in coarse cells from the
coarse centre of the peak's cell, clamped to +/-0.5 cell:

| rule | what it does |
|---|---|
| `argmax` | no refinement: the pixel `peak_local_max` returned (today's decode) |
| `centre` | the coarse centre `8i+3.5` itself (0.5 px from argmax; a control) |
| `quarter` | 0.25 cell toward the higher neighbour, per axis (SimpleBaseline / HRNet) |
| `parabola` | vertex of the parabola through the 3 values on each axis |
| `gaussian` | the same on log values. This is exact for a sampled Gaussian, which is the training target's shape (sigma 10 hi-res px = 1.25 cells) |
| `dark` | 2-D second-order Taylor step on the log map with the cross term (DARK, Zhang et al. CVPR 2020), without DARK's pre-smoothing |
| `centroid` | value-weighted centroid of the 3x3 neighbourhood, minus its minimum |

Coordinates follow the pipeline convention. `stage_two/train.py` places a target at pixel
`round(x * W)`, and every extractor reports `x = col / W`. So a refined position is
`(8 * (j + dx) + 3.5) / 1024`. Columns at the 360-degree seam (coarse col 0 or 127) are not refined
in x by default, because the network pads there rather than wrapping. That affects 34 manual_gold
peaks.

**Ground truth.**

- `manual_gold` (primary): centres of the YOLO boxes in `manual_labels/`. They were labelled
  without any model (1,000 panos, GSV at 2048x4096), so they are not anchored to the model's grid.
- `annapolis`, `paterson`, `richmond`, `sao_paulo` (secondary): centres of the reviewer boxes in
  `benchmark/<split>/boxes.json` with status `boxed` (box rule v2; 658 boxes in 198 panos; the
  other 10 of the 208 panos in those files hold only `cant` entries). Entries
  marked `cant` (extent undeterminable) are kept as match decoys so they cannot steal a detection,
  and are dropped from residuals. These are the only splits with a `boxes.json`.

**Pairs.** Detections are peaks >= 0.30 (the #79 recommended operating point), extracted exactly
as `analysis_out/op_cache` does: `clip(h, 0, 1)`, `min_distance=10`, `exclude_border=False`. They
are matched once, on their **argmax** positions, to GT: greedy by confidence, radius 0.022 (the
benchmark's), x wrapped. Every decode is then scored on the same pairs, so every comparison is
paired.

**Metrics.** Per pair, the residual GT minus detection on the 512x1024 grid, where 1 px = 0.3516
degrees on both axes. Reported: mean and median Euclidean px, great-circle degrees, per-axis bias
and SD. The paired difference against argmax gets a pano-cluster bootstrap 95% CI (2,000 reps,
seed 221). The **sub-cell fit** regresses the GT's offset from the peak's coarse centre on the
decoded offset: slope 1 means the decode is unbiased, and `r` is how much of the GT's sub-cell
position it explains. Position mod 8 histograms are reported before and after.

**Instrument check.** Our argmax peaks are compared with the committed `analysis_out/op_cache`.
All five splits pass: 3,832 / 3,833 manual_gold peaks at the same pixel and the remaining one 1 px
away (the 1-px tie above), and 100% same-pixel on the four box splits. Max score difference is 1e-4,
under the 2e-4 cross-machine tolerance `input_res_sweep_25.py` uses. Peaks within 10 px of the
heatmap edge are counted, not compared, because the op_caches were extracted with
`exclude_border=True`, the [#132](https://github.com/ProjectSidewalk/RampNet/issues/132) defect.
There are 43 such peaks here across five splits, and none in the op_caches.

## 4. Results

All numbers are from `analysis_out/subcell_decode_221/results.md` / `results.json`. Residuals are
in px on the 512x1024 grid. d is the paired change against argmax, with its 95% CI; negative means
closer to the box centre.

### 4.1 Mean residual, argmax vs Gaussian decode, per split

| split | GT | pairs (panos) | argmax mean px | gaussian mean px | d mean px [95% CI] | argmax mean deg | gaussian mean deg |
|---|---|---:|---:|---:|---|---:|---:|
| manual_gold | independent boxes | 3,517 (787) | 5.080 | 4.353 | **-0.727 [-0.783, -0.673]** | 1.726 | 1.473 |
| annapolis | reviewer boxes | 102 (40) | 7.133 | 6.945 | -0.188 [-0.635, +0.254] | 2.440 | 2.374 |
| paterson | reviewer boxes | 85 (29) | 5.665 | 4.874 | -0.791 [-1.108, -0.461] | 1.934 | 1.660 |
| richmond | reviewer boxes | 246 (85) | 5.327 | 4.796 | -0.531 [-0.757, -0.300] | 1.828 | 1.642 |
| sao_paulo | reviewer boxes | 93 (37) | 5.563 | 5.066 | -0.497 [-0.906, -0.087] | 1.899 | 1.725 |
| pooled, 4 box splits | | 526 (191) | 5.774 | 5.273 | -0.501 [-0.660, -0.331] | 1.976 | 1.802 |
| pooled, all 5 | | 4,043 (978) | 5.170 | 4.473 | -0.698 [-0.752, -0.641] | 1.759 | 1.516 |

Annapolis is the one split whose CI crosses zero. Its y-axis spread still falls (SD y -0.429
[-0.747, -0.107]); its x-axis does not (+0.053 [-0.334, +0.441]), and its x sub-cell fit is weak
(slope 0.463, r 0.141). Annapolis is Mapillary imagery at 8000x4000 with 102 pairs, and this run
cannot say whether the x-axis null is the imagery or the sample.

### 4.2 All decode rules, manual_gold

| decode | mean px | median px | mean deg | SD x | SD y | d mean px [95% CI] | slope x / y | r x / y |
|---|---:|---:|---:|---:|---:|---|---|---|
| argmax | 5.080 | 4.440 | 1.726 | 5.046 | 3.259 | -- | -- | -- |
| centre | 5.392 | 4.812 | 1.835 | 5.212 | 3.507 | +0.312 [+0.291, +0.333] | -- | -- |
| quarter | 4.629 | 3.864 | 1.569 | 4.824 | 2.873 | -0.452 [-0.500, -0.405] | 0.990 / 1.014 | 0.379 / 0.574 |
| parabola | 4.365 | 3.526 | 1.478 | 4.699 | 2.625 | -0.716 [-0.768, -0.661] | 1.062 / 1.040 | 0.433 / 0.664 |
| **gaussian** | **4.353** | **3.491** | **1.473** | 4.695 | 2.623 | **-0.727 [-0.783, -0.673]** | 0.996 / 0.972 | 0.434 / 0.664 |
| dark | 4.365 | 3.513 | 1.478 | 4.696 | 2.632 | -0.715 [-0.770, -0.658] | 0.991 / 0.964 | 0.434 / 0.661 |
| centroid | 4.396 | 3.618 | 1.488 | 4.722 | 2.650 | -0.684 [-0.731, -0.640] | 1.266 / 1.180 | 0.433 / 0.663 |

- `parabola`, `gaussian` and `dark` are indistinguishable here: their CIs overlap almost entirely.
  `gaussian` is recommended because it is the simplest rule that is exact for a sampled Gaussian,
  which is the target's shape. Its slopes are closest to 1 on manual_gold (0.996 / 0.972) but not
  on the pooled set, where `quarter` (0.985 / 1.006) is as close and `parabola` is closer in y
  (1.033 against 0.965). The slopes carry no CI, so they do not separate the rules.
- `quarter` recovers about 60% of what the others do. `centroid` explains as much of the GT's
  position (same r), but its slope above 1 means its offsets are systematically too small. The
  3x3 window truncates a peak 1.25 cells wide.
- `centre` is *worse* than argmax, by +0.312 px. So the argmax's half-pixel lean toward the higher
  neighbour already carries a little sub-cell information. Snapping to coarse centres would lose
  it.
- The spread falls on both axes, and the decode removes essentially all of the quantization
  variance (`results.md`, "Variance removed"). The baseline is the argmax's own quantization, not
  a centre snap. Snapping to the coarse centre leaves an offset u ~ U[-4, 4] px, variance
  64/12 = 5.33 px^2. The argmax already leans 0.5 px toward the true side (a = 0.5 sign(u)), so
  its quantization variance is E[(u - a)^2] = 5.33 - 2(0.5)(2) + 0.25 = **3.58 px^2**. The
  `centre` row checks this: centre minus argmax variance is predicted at 1.75 px^2 and measures
  27.16 - 25.46 = 1.70 in x and 12.30 - 10.62 = 1.68 in y. Against the 3.58 px^2 baseline, the
  Gaussian decode removes 25.46 - 22.04 = 3.42 px^2 in x (**95%**) and 10.62 - 6.88 = 3.74 px^2
  in y (**104%**). Against the centre snap it removes 96% and 102%. y above 100% means the
  uniform-quantization model, with the remaining error independent of it, does not hold exactly
  in y. The mechanism: with shift q = gaussian position - argmax position, the removed variance
  is var(q) + 2 cov(gaussian residual, q). On manual_gold that is 3.505 - 0.086 = 3.419 px^2 in x
  and **3.910** - 0.170 = 3.740 px^2 in y (`results.md`, the line under "Variance removed"). The
  decode really moves peaks by more than the uniform model's 3.58 px^2 in y, because decoded rows
  pile up at 5-6 mod 8, away from the argmax's 3-4: the same non-uniformity as section 4.3. An earlier version of this doc divided by the 5.33 px^2 centre-snap variance and reported
  64% / 70%, which understated the result.

### 4.3 Position mod 8, manual_gold

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x argmax | 12 | 3 | 18 | 1,712 | 1,772 | 0 | 0 | 0 |
| x gaussian | 418 | 414 | 441 | 481 | 465 | 415 | 487 | 396 |
| x GT (box centres) | 464 | 425 | 452 | 437 | 449 | 442 | 430 | 418 |
| y argmax | 1 | 2 | 17 | 1,595 | 1,902 | 0 | 0 | 0 |
| y gaussian | 440 | 385 | 337 | 343 | 459 | 524 | 562 | 467 |
| y GT (box centres) | 454 | 413 | 418 | 473 | 455 | 459 | 435 | 410 |

After decoding, x is close to uniform, like the GT. **y is not**: decoded rows pile up at 5-6 and
thin out at 2-3, while the GT stays uniform. This is a property of the model output, not a
decode defect. The y path is the same code as x, and on synthetic bilinear upsamples of sampled
Gaussians it recovers the position exactly on both axes (`tests/test_subcell.py`). The pile
comes from where the ramps are. The table is from `results.md`, "y profile by band", produced by
`report --y-bands 256,290,330` (the default). Bands are on the argmax row, and row 256 is the
horizon.

| argmax rows | pairs | mean decoded offset (cells) | mean GT offset (cells) | mean abs dy, argmax px | mean abs dy, gaussian px | decoded y mod 8 |
|---|---:|---:|---:|---:|---:|---|
| 256-290 (far field) | 1,212 | +0.140 | +0.239 | 2.474 | 1.484 | 73, 76, 70, 92, 173, 285, 296, 147 |
| 290-330 | 1,947 | -0.024 | +0.051 | 2.441 | 1.969 | 318, 261, 213, 202, 238, 204, 236, 275 |
| 330-512 | 358 | -0.034 | -0.012 | 3.680 | 3.509 | 49, 48, 54, 49, 48, 35, 30, 45 |

In the far-field band just below the horizon, the decoded peak sits +0.14 cell below the coarse
centre on average, and the GT sits lower still, +0.24 cell. The pile at rows 5-6 is that band;
elsewhere the decoded histogram is close to uniform. The decode is calibrated. Binned by decoded
y offset, the mean GT offset rises monotonically with it and sits +0.04 to +0.11 cell above it
(`results.md`, calibration table). The decode also has less spread than the GT (SD 0.299 against
0.438 cells). A calibrated estimate with lower variance piles up around its band's mean offset,
so its mod-8 marginal need not be uniform. The coarse vertical profile has the width the rule
assumes (median log-curvature -0.657 in y and -0.641 in x, against -0.640 for sigma 1.25 cells).
What differs is its asymmetry: the lower neighbour is the higher one in 56.1% of pairs, while the
right neighbour is in 50.9%. The far-field band is also where the decode helps most in y. Mean
|dy| goes from 2.47 to 1.48 px there, against 2.44 to 1.97 px in rows 290-330. No y-decode defect
was found. This decomposition is from the #226 review; the committed `report` step re-derives
every number in it.

### 4.4 The cross-view reference floor (#48), re-read

`docs/crossview_align_48.md` attributes a ~2 degree floor to reference-detection noise. The
number behind it is `analysis_out/crossview_align_48/reference_noise.json`: manual_gold detections
>= 0.55 against box centres, median 1.508 degrees. Re-read here with the same protocol (single
pass; 3,347 matches):

| decode | median deg | p90 deg | mean deg | d median [95% CI] |
|---|---:|---:|---:|---|
| argmax | 1.510 | 3.037 | 1.698 | -- |
| gaussian | 1.197 | 2.735 | 1.449 | -0.314 [-0.344, -0.280] |
| dark | 1.206 | 2.761 | 1.453 | -0.304 [-0.334, -0.267] |
| parabola | 1.213 | 2.711 | 1.453 | -0.297 [-0.325, -0.266] |

The argmax row lands on the committed median (1.510 vs 1.508) but not on its n (3,347 vs 3,420).
The committed file read `benchmark/manual_gold/records.jsonl`, which was exported with flip TTA
(`detections_meta.json`: `"tta": true`), and TTA raises scores, so more peaks clear 0.55. **Not
run:** re-scoring the placement arms themselves (`proj_height_auto` etc.) with refined source and
reference points. That needs the cross-view panos pushed through the model and the harness's
predictions re-made from refined source points, which is more than a small extension. What this
does show is that the part of the floor due to the reference peak shrinks by about a fifth. That
was measured on manual_gold GSV imagery only. The #48 pairs include other cities and Mapillary
imagery, and on annapolis (Mapillary) the x axis did not improve here (section 4.1). Note also
that 1.51 degrees is the per-peak reference noise; the ~2 degree floor involves a peak at each end
of a pair. The harness's between-arm differences of a few tenths of a degree were below the old
floor, and they may become resolvable with the refined decode.

## 5. Caveats

- **A box centre is not the point the model was trained on.** Stage 2 learned Gaussian targets at
  Stage 1 auto-placed points, not box centres. Residuals therefore include a systematic
  concept offset, visible as the per-split biases of about 1 px in `results.md`, plus labeller
  noise. Both are the same for every decode, which is why the comparison is paired. It is also
  why the absolute residuals cannot fall to zero, and why `r` in the sub-cell fit tops out near
  0.43 / 0.66.
- **The box splits are not blind to the model.** Their boxes were drawn in a crop around a
  *shown* detection point (`det:k`) or a reviewer-placed miss point. The box *extent* is human,
  but its centre may be pulled toward the shown point, which sits on the argmax grid. If
  anything, that biases the box splits toward argmax. manual_gold has no such anchoring, and it
  is the primary read.
- **One operating point.** Pairs are detections >= 0.30. The 0.55 subset is in 4.4.
- **Single pass.** The deployed labeler and this run use no TTA. Flip TTA max-combines two
  bilinear surfaces. Its peaks are still on the grid, because the flip maps coarse centres onto
  coarse centres, but the max-combined map is no longer one bilinear upsample. This decode was not
  measured under TTA.
- **The crop model has the same head** (x8, 32x11 effective) and is not measured here.
- **The seam.** Coarse columns 0 and 127 are not refined in x (`wrap_x=False`), which affects 34
  manual_gold peaks. `report --wrap-x` uses the neighbour across the seam instead, and it was not
  run for the committed numbers. The stored neighbourhoods were climbed without wrapping, so
  `report --wrap-x` cannot follow a coarse maximum that lies across the seam. After the review,
  `rampnet.subcell.refine_peaks(..., wrap_x=True)` passes `wrap_x` to `climb` as well, so the
  labeler path can (`tests/test_subcell.py::test_climb_crosses_the_seam_when_wrapping`).

## 6. `heatmap_size` is nominal

`KeypointModel(heatmap_size=(512, 1024))` and the model card describe a 512x1024 heatmap. The
effective resolution is **64x128** for the pano model and **32x11** for the crop model. Any
localization figure quoted with the plain argmax decode includes a +/-4 px (+/-1.4 degree)
quantization per axis. That covers the paper's metrics and every benchmark number, at matching
radii far larger than this effect. The matching radius (0.022, about 22.5 px) is wide enough that
no detection metric depends on it: matched counts with the Gaussian decode differ from argmax by
at most one per split (`results.md`). Text saying so has been added to both model-card templates
in `scripts/hf_package/`. It reaches Hugging Face on the next card export. **Nothing was pushed to
the Hub.**

## 7. Using the decode

```python
import numpy as np
from skimage.feature import peak_local_max
from rampnet.subcell import refine_peaks

h = model(x).squeeze().float().cpu().numpy()                  # 512x1024
pk = peak_local_max(np.clip(h, 0, 1), min_distance=10, threshold_abs=0.30,
                    exclude_border=False)
xy = refine_peaks(h, pk, method="gaussian")                    # (N, 2) normalized x, y
scores = h[pk[:, 0], pk[:, 1]]                                 # confidence is unchanged
```

`method="argmax"` returns the input peaks unchanged, so the rule can be switched off without a code
path change. Since section 10, `rampnet.subcell.detect_peaks(h, 0.30, decode="gaussian", clip=True)`
does the extraction and the refinement in one call, and it is what `stage_two/evaluate.py`,
`stage_two/demo.py` and the Hugging Face package use. The coarse map is recovered with a cached pseudo-inverse, costing two small matrix
products per heatmap.

## 8. Reproduction

**As run** (2026-09-30, makelab2, from a checkout of this branch at
`/homes/gws/jonf/wt-subcell221`, with the makelab2 RampNet venv). The panos were read from Jon's
makelab checkout, not fetched for this run:

```bash
python scripts/analysis/subcell_decode_221.py extract \
    --panos-root /homes/gws/jonf/RampNet --cache-dir /homes/gws/jonf/subcell221_cache \
    --out analysis_out/subcell_decode_221/detections.json \
    --usage-out analysis_out/subcell_decode_221/usage_row.json
```

(`--out` and `--usage-out` were defaults then. They are required now, so a replication cannot
overwrite the committed files.)

The model was `projectsidewalk/rampnet-model` at Hub `main`, which has been commit
`606a11956743f7eb328d9207769034752f6191f4` since 2026-07-24 (`model_info().last_modified`);
makelab2's HF cache `refs/main` held that commit on the run date, and its `model.safetensors` has
sha256 `f2119e3becb0b551fa1470f7b7ba85b82122a3f73a6ed2a85609dd57617866b5`. `detections.json`'s
`meta` predates the fix that records this, so the commit is pinned in the script instead
(`MODEL_REVISION`, the default of `extract --model-revision`) and reported in `results.md`.

**Imagery check.** After the review, every pano the extraction read was hashed against the
committed `benchmark/<split>/imagery_manifest.json`:

```bash
python scripts/analysis/subcell_decode_221.py verify-imagery --panos-root /homes/gws/jonf/RampNet \
    --out analysis_out/subcell_decode_221/imagery_check.json
```

All 1,499 match (manual_gold 1,000, annapolis 125, paterson 125, richmond 124, sao_paulo 125;
`analysis_out/subcell_decode_221/imagery_check.json`). No file under those `panos/` directories
was modified after the run started (newest mtime 2026-08-14), so the check describes the bytes the
run read. `extract` now runs the same check first and refuses mismatched imagery unless given
`--allow-imagery-mismatch`.

**From a clean clone** (inputs: the Hub model, the Hub imagery, and the committed labels and
boxes). Every output goes to a scratch directory (`/tmp/sc221` below), so nothing committed is
overwritten:

```bash
# imagery: manual_gold from the rampnet-dataset test split (checked against its
# imagery_manifest.json after the fetch; this script has no --revision, and the dataset's main
# was ee882e3f3c779dc13182f307bca616e50d9b8c5c on 2026-09-30); the four box splits at native
# resolution from rampnet-benchmark, pinned
python scripts/fetch_manual_gold.py --images-only
python scripts/unpack_benchmark_panos.py --out . --cities annapolis,paterson,richmond,sao_paulo \
    --revision 63d5ffd0e4b6795702db40be89ea4d9672d91ab5

# CPU: prove the imagery is the imagery the run used (exit 1 on any mismatch)
python scripts/analysis/subcell_decode_221.py verify-imagery --panos-root . \
    --out /tmp/sc221/imagery_check.json

# GPU: 43 min wall-clock on one (shared) A40 for 1,499 panos; native JPEG decode is not overlapped.
# --model-revision defaults to the commit above. The imagery check runs again first, and its
# result, the model commit and the weights hash go into the new detections.json's meta.
python scripts/analysis/subcell_decode_221.py extract --panos-root . \
    --cache-dir /tmp/sc221/coarse --out /tmp/sc221/detections.json \
    --usage-out /tmp/sc221/usage_row.json

# CPU, ~1 min: every number in this doc, from YOUR detections
python scripts/analysis/subcell_decode_221.py report --detections /tmp/sc221/detections.json \
    --out /tmp/sc221/results.json

# the Examples figures. Selection is CPU-only and prints with --select-only. --heatmap-source
# model runs the checkpoint on the mechanism pano and recovers its coarse map from that
# heatmap, checked against the 3x3 neighbourhoods stored in --detections (to 2e-4), so this
# needs neither the extract step nor its cache. The selection reads the committed detections;
# pass /tmp/sc221/detections.json instead to draw from your own.
python scripts/analysis/subcell_decode_221_figures.py --panos-root . \
    --detections analysis_out/subcell_decode_221/detections.json \
    --heatmap-source model --out-dir /tmp/sc221/figures

# compare with the committed report: every number, ignoring only the top-level "inputs"
python scripts/analysis/subcell_decode_221.py compare \
    analysis_out/subcell_decode_221/results.json /tmp/sc221/results.json
```

**What must match and what may differ.** `compare` ignores only the top-level `inputs` block:
the detections path and sha256, the model commit, the imagery-check path and panos root, and where
each of those came from. Those differ in any replication, and `results.md`'s header lines, which
print them, differ with them. Everything else is a measured number and is compared exactly.
On the same software stack (torch 2.6.0+cu124 on an A40) the extraction is expected to be
bit-identical, so `compare` should report 0 differences. Across machines, fp32 noise of about 1e-4
in peak scores (section 3) can flip a within-cell 1-px tie or move a peak across the 0.30 floor,
so a few counts and the last printed digit of some means may move. Use `compare --tol` to see
the size of the differences. A change in any paired difference beyond its last digit, or in a CI
beyond noise, is a real discrepancy. `report` alone, run on the committed `detections.json` with
default arguments, reproduces the committed `results.json` and `results.md` byte for byte.

**Figures, as run** (2026-10-01, desktop RTX 3070, torch 2.6.0+cu126). The coarse map was a
local copy of `/homes/gws/jonf/subcell221_cache/manual_gold/4ogBseRbooQ5Jz5eiu05ig_coarse.npy`
from makelab2, whose sha256 equals the one in `detections.json`; the panos were the desktop's
manual_gold copy, which matches `imagery_manifest.json`:

```bash
python scripts/analysis/subcell_decode_221_figures.py --panos-root D:/Git/RampNet \
    --detections analysis_out/subcell_decode_221/detections.json \
    --coarse-dir <local copy of subcell221_cache> --heatmap-source model \
    --out-dir docs/figures/subcell_decode_221
```

Re-running that command reproduces both committed figures byte for byte:
`examples_contact_sheet.jpg` sha256 `de0e5954df3c02090e4591d8750f804a334dd1dfc64b7b662898cc7350c19b95`
and `mechanism_panel.png` sha256 `c264bf52eb62a6e3720e5c211b04c47b6094d7999f6d30e355800f4daf1f6041`.
The clean-clone command above draws the same contact sheet byte for byte (it uses no coarse map).
Its mechanism panel differs in bytes, because its coarse map is recovered from a heatmap computed
on another GPU, which differs from the A40 map by about 2e-5. On that path the number that shows
the coarse map is the run's is the printed neighbourhood gap (about 1.55e-05 on the desktop,
against the 2e-4 tolerance); the `max |model heatmap - upsample(coarse)|` it also prints (about
2.15e-07) is a different quantity, which only checks the recovered map against the heatmap it
came from, and the 2.0e-5 quoted in Examples is that quantity on the as-run path. The script
checks the coarse map before it writes anything, so a failed check leaves `--out-dir` untouched.
`--out-dir` is required, so the figure script cannot overwrite the committed figures unless told
to.

`detections.json` (sha256 `57cfb968c5106a833a214aeea7801d20f8fb228cb3c28bdaf7a3474074cd72b1`)
holds every peak >= 0.30 with its 3x3 coarse neighbourhood (6 decimals), so `report` needs neither
GPU nor imagery. `tests/test_subcell_decode_221.py` re-derives the point estimates from it in CI.
Two things are not committed. The 64x128 coarse maps (1,499 x 32 KB) are on makelab2 at
`/homes/gws/jonf/subcell221_cache/`, and each one's sha256 is in `detections.json`. A regenerated
map can be proven identical if it was produced on the same software stack. Across machines, fp32
noise of about 1e-4 is expected (section 3).

## 9. Cost

| step | where | wall-clock | GPU-hours | $ |
|---|---|---:|---:|---:|
| `extract`, 1,499 panos (manual_gold 1,000 + 4 x ~125) | makelab2, 1x A40 (shared for part of the run: #217's segment-vistas at 18:53:48Z and #218's perspective shards from 19:00:50Z both ran on it before this run ended at ~19:18Z) | 2,579 s (43 min) | 0.72 | 0 |
| `report` | desktop CPU | ~1 min | 0 | 0 |
| smoke test (15 panos) | desktop RTX 3070 | 32 s | ~0.01 | 0 |
| Examples figures (one forward pass, plus plotting) | desktop RTX 3070 (not in the ledger) | ~22 s | ~0.01 | 0 |

The extract row is in `analysis_out/usage_log.jsonl`
(`run_id` `subcell-decode-221:extract:2026-09-30T18:35:42Z`, `paid: false`). Of the 2,579 s, the
forward passes took 1,812 s (1.2 s/pano, with the GPU shared) and single-threaded JPEG decode of
the 8k-16k native panos took 688 s. The GPU share was not measured, so the row keeps `gpu_share: 1.0` and its GPU-hours are an upper bound. `verify-imagery` (CPU, makelab2, after the review) is not in the ledger, and neither is the smoke test.

## 10. Shipping (items 1 and 2)

Added after the measurement above, on branch `feat/subcell-decode-ship-221` (stacked on PR #226).
It makes the decode available wherever a peak is extracted. **Nothing in sections 1-9 was re-run,
and no published number changes by default.**

**One entry point.** `rampnet.subcell.detect_peaks(heatmap, threshold, min_distance=10,
decode="argmax", *, exclude_border=False, clip=False, coarse=None, wrap_x=False, factor=8,
return_pixels=False)` returns float `(row, col, score)` rows on the heatmap's own grid.

- `decode="argmax"` runs `peak_local_max(heatmap, min_distance, threshold_abs=threshold,
  exclude_border=False)` and returns those pixels unchanged. It is bit-identical to the call
  `stage_two/evaluate.py` always made: same pixels, and the score read at that pixel.
- `decode="gaussian"` (or any rule in `METHODS`) applies `refine_peaks`. It finds the same peaks
  with the same scores and moves only their positions.
- The coarse map has to come from the **raw single-pass** head output, because only that is
  exactly a bilinear upsample. `clip=True` finds peaks on `clip(h, 0, 1)` and decodes from the raw
  `h`, which is the section 3 protocol. `coarse=` takes a precomputed map, or a `(B, 64, 128)`
  stack with one map per flip-TTA branch. With a stack, each peak is decoded from the branch whose
  upsampled value is highest at that pixel, which is the branch the max-combine took it from.
  **TTA decoding was not measured** (section 5).
- `wrap_x` defaults to off, as measured.
- The function only needs the heatmap to be a multiple of 8. The crop model's 256x88 heatmap
  (32x11 coarse) works the same way (`wrap_x` must stay off there). It is tested on synthetic maps
  only; the crop model's decode was never measured.

**Where it is wired, and the defaults.**

| path | flag / API | default | why |
|---|---|---|---|
| `stage_two/evaluate.py` | `--decode {argmax,gaussian}` | `argmax` | every published number and committed `evaluation_results*/` file used argmax |
| `stage_two/demo.py` | `--decode {argmax,gaussian}` | `argmax` | the demo draws exactly what it drew before (it keeps skimage's default `exclude_border=True`, the #132 defect, for that reason) |
| HF package | `RampNetModel.detect(inputs, threshold=None, decode="gaussian", min_distance=None, wrap_x=False)` | `gaussian` | a new API with no published number behind it, so it defaults to the measured, recommended rule. Jon may overrule this |

**evaluate.py's cache.** `evaluate_cache/heatmaps/<fingerprint>_<dataset>_<tta|notta>/` stores
the clipped, TTA max-combined heatmaps, not peaks. The decode does not change those maps, so it is
deliberately **not** part of their key: argmax reads exactly the cache it always read, and argmax
and gaussian runs share it. A refining decode cannot be read from those maps, which are clipped
and (with TTA) a max of two surfaces. So `--decode gaussian` adds
`evaluate_cache/coarse/<same key>/<pano>_coarse.npy`, a float32 `(B, 64, 128)` stack of the raw
branches' coarse maps. The maps are cast to float32 before use, so a decode read back from the
cache equals one computed fresh. If a heatmap is cached but its coarse map is not, the model is
re-run for the coarse map only, and the existing heatmap is kept so that peaks still come from the
map an argmax run used.

*Corrected after the [review of PR #229](https://github.com/ProjectSidewalk/RampNet/pull/229#pullrequestreview-5373695786) (S1).*
The first version said "`--fresh` clears both directories". That was false for an argmax run,
which cleared only `heatmaps/` and could leave old coarse maps to be paired with new heatmaps.
Two changes fix it:

- `--fresh` now clears `coarse/<key>` whenever it clears `heatmaps/<key>`, whatever the decode
  (`prepare_cache_dirs()`).
- Before a refining decode uses a coarse stack, it re-builds the heatmap from that stack
  (`max_b clip(upsample(coarse_b), 0, 1)`) at every peak pixel and compares the result with the
  cached heatmap (`rampnet.subcell.coarse_mismatch`). Agreement above `COARSE_ATOL` = 1e-3 is
  required; float32 storage noise is about 1e-7. On a mismatch the run raises `StaleCoarseCache`
  and names `--fresh`. The comparison covers the **whole map**, not only the peak pixels: when a
  peak is clipped at 1, a stale map and a fresh map agree at the peak and differ only on its flanks
  (re-review N3; `test_stale_coarse_with_saturated_peak_raises`).

The check covers both stale pairings: old coarse maps beside new heatmaps, and a coarse map
recomputed for a heatmap cached by a different model or preprocessing.

Two related guards came from the same review (S2):

- `detect_peaks` with `coarse=None` and a refining decode raises `ValueError` when the heatmap is
  not an exact x8 upsample. The threshold is a relative residual above `UPSAMPLE_RTOL` = 1e-4,
  and a raw fp32 head output measures about 1e-7. Examples are a clipped peak, a TTA max, or a
  heatmap from a non-2048x4096 input, where the factor is no longer 8. In those cases a
  least-squares coarse map can land further from the truth than argmax.
- `RampNetModel.detect` rejects `pixel_values` that are not `config.input_size` unless
  `decode="argmax"`. Result files from a refining decode are tagged
`_dgaussian` (e.g. `metrics_manual_r0.022_pt0.0_dgaussian.json`) so they cannot overwrite the
committed argmax files. `metrics.json` records `decode` either way. `cache_dirs()` and
`results_params_str()` are pinned by `tests/test_decode_ship_221.py`.

**Hugging Face package.** `scripts/export_hf_model.py` now ships `rampnet/subcell.py` verbatim as
`rampnet_subcell.py` (`VERBATIM_COPIES`, beside `rampnet_model.py`), and `modeling_rampnet.py`
imports `detect_peaks` from it. `subcell.py` imports only numpy and stdlib at module level.
scikit-image is imported inside a `try` in `detect_peaks`, so the remote-code loader (which skips
`try` blocks when listing required packages) does not start requiring it to load the model. Only
`detect()` needs it. Both the model card template and the README now say that `heatmap_size` is
nominal (64x128 effective, 8-px grid, half-cell floor 1.4 degrees) and show `model.detect(...,
decode="gaussian")` beside the `peak_local_max` snippet. The card's `peak_local_max` snippet now
passes `exclude_border=False`, matching evaluate.py (#132).

**How to turn it on.**

```bash
python stage_two/evaluate.py --checkpoint <ckpt> --dataset manual --decode gaussian
python stage_two/demo.py --decode gaussian
```

```python
from rampnet.subcell import detect_peaks
rcs = detect_peaks(h_raw, 0.30, decode="gaussian", clip=True)   # (N, 3) row, col, score
```

**What was not run.**

- No `evaluate.py --decode gaussian` run. No GPU or checkpoint was available on the machine that
  did this work. Section 6's claim that matched counts move by at most one per split comes from
  the section 4 pairs (single pass, >= 0.30), not from an evaluate.py run. evaluate.py defaults to
  flip TTA, and TTA decoding is unmeasured. Any gaussian AP or P/R figure would be new and has not
  been produced.
- No end-to-end export. `export_hf_model.py` needs a checkpoint. The file copy, the remote-code
  load (`AutoModel.from_pretrained(..., trust_remote_code=True)` on a package assembled from
  random weights) and `detect()` are tested on CPU. Exporting from the released weights and
  uploading to `projectsidewalk/rampnet-model` is a separate step for Jon, and nothing was pushed
  to the Hub. Until it is done, the published package has no `detect()`, and the README's second
  snippet (`rampnet.subcell.detect_peaks`) is the way to get the decode.
- The demo's `--decode gaussian` path was not launched (it needs gradio and a model). It uses the
  same `detect_peaks` call that the tests cover.
