# Context experiment: field of view vs curb-ramp tag accuracy (#86, RampNet 2.0 plan item 4)

**Status: four arms scored against the final #178 control (2026-09-24); two resolution arms
added 2026-09-25 (§4.2), which separate the lost pixels from the rest of the wider crop's cost
(more street, and a smaller ramp at the model's input, together); a second seed of fov25 and
fov90 added 2026-09-26 (§4.2), the benchmark's first seed-to-seed numbers: 0.025 and 0.006
mAP, so effects of that size in this document are within seed noise.** Crops cut on makelab2
(12 workers, 48 min; 32,544 of 32,559 label-centred crops and 10,848 of 10,853 viewport crops,
the rest `missing_pano`, `cut_summary_*.json`); the four arms trained on klone's `gpu-l40s`
allocation 2026-09-23/24 (§5). The control is #178's 100-epoch `train_control` (`best.pth`,
epoch index 89, sha256 `de59f2d3…`), re-scored on this experiment's common test rows. §4 was
first read against an interim control, the dead first #178 launch's epoch-49 snapshot; that
reading is kept as history in §4.1 and §6, and §4.1 lists what changed.
**Script:** `scripts/analysis/context_fov_86.py`; run order in `scripts/analysis/context_fov_86.sh`;
one Slurm job per arm, `scripts/analysis/context_fov_86.slurm`; environment,
`scripts/analysis/context_fov_86_env.slurm`. **Tests:** `tests/test_context_fov_86.py`.
**Outputs:** `analysis_out/context_fov_86/`.

## 1. Question

The plan (item 4) asks whether the street-dependent tags (*points into traffic*, *not level
with street*, *not enough landing space*) fail for lack of context or lack of vision. The
ASSETS'24 tagger, reproduced exactly as the benchmark of record
([`tag_benchmark_86.md`](tag_benchmark_86.md), mAP 0.3408), is strong only on *missing tactile
warning*, a tag that is visible on the ramp itself. This experiment holds everything else fixed
(labels, recipe, split, test rows, backbone, the model's 256×256 input) and varies only how much
of the scene the crop shows. One exception to "labels fixed": the arms train on the 8,666 train
labels every arm has, the control on the HF split's 8,674 (the 9 labels no arm has are 8 train
and 1 test, `dropped_labels.csv`). The control is *scored* on the same 2,182 test rows as the arms.

## 2. What the control actually sees

The HF `sidewalk-tagger-ai-validated` crops are 1440×960 screenshots of the labeling viewport,
and the tagger's `crop.py` takes a 640 px box around the label point before training. The
viewport's horizontal field of view depends on the labeler's zoom (`rampnet/crops.py`
`get_3d_fov`: 89.75° at zoom 1, 53° at zoom 2, 27.68° at zoom 3), so the box's field of view is
about **48° / 25° / 12.5°** at zoom 1 / 2 / 3 (`2·atan(320 / f)` with `f = 720 / tan(fov/2)`).
The benchmark's labels are 54% zoom 1, 28% zoom 2, 18% zoom 3 (`docs/crop_cutter.md`), so the
control is a mixture, not one field of view, and its framing is the labeler's, with the ramp
off-centre.

**The model never sees 640 px.** Every crop, control and arm alike, is resized to **256×256**
before the model sees it (`tag_benchmark_86.py`: `IMAGE_DIMENSION = 256`, the `Resize((256, 256))`
in `train` and in the evaluation transform, as in the tagger). So the model's pixel budget is
fixed, and a wider field of view spreads it over more of the scene. At the crop centre, in model
pixels per degree: **10.1** at 25°, **4.8** at 50°, **2.2** at 90°; the control's box is 5.0 /
10.1 / 20.4 at zoom 1 / 2 / 3. fov90 therefore has 4.5× coarser linear resolution than fov25,
about 20× fewer model pixels on the ramp; fov50 has 2.1× coarser, about 4.4× fewer.

## 3. Arms

All arms are cut by `scripts/crop_cutter.py` (plan item 2b, PR #177) from the makelab2 pano
store, gnomonic, rendered with the viewer's tilt (the cutter's default `--tilt mm`), on the
same 10,853 labels (the benchmark's 10,857 minus 4 with no rawLabels geometry,
`docs/data/crop_cutter/coverage_input.csv`), then restricted to the labels every arm has.

| arm | framing | crop, then model input | centre px/deg at the input | what it isolates |
|---|---|---|---:|---|
| control | HF screenshot, labeler's zoom, then the 640 px box | 640×640 at 12.5°–48° → 256×256 | 5.0–20.4 | the published benchmark (#178, `train_control`, 100 epochs) |
| `viewport` | the labeler's own view re-cut from the store (1440×960), then the same 640 px box (`crop640`, the tagger's arithmetic, saved at JPEG quality 92) | 640×640 at 12.5°–48° → 256×256 | 5.0–20.4 | re-cut from the archive vs the screenshot, at the control's field of view |
| `fov25` | label-centred, 25° horizontal | 640×640 → 256×256 | 10.1 | about the zoom-2 control, centred |
| `fov50` | label-centred, 50° | 640×640 → 256×256 | 4.8 | about the zoom-1 control, centred |
| `fov90` | label-centred, 90° | 640×640 → 256×256 | 2.2 | wider than any control |

Every arm trains the benchmark's recipe unchanged (`tag_benchmark_86.py train`: DINOv2-B/14
with registers, full fine-tune, Adam 1e-6, batch 4, 100 epochs, seed 86, checkpoint by best
training exact-match accuracy) on the **published HF split** restricted to the common labels
(`split_common.csv`), and is scored on the common test rows. The #178 control is re-scored on
those same rows (`control_scores.json`; the superseded interim read is
`control_interim_ep49_scores.json`, §4.1), so the comparison is paired on the test set. The pano-grouped re-split
is not used here: #178 found no measurable inflation from the pano leak.

## 4. Results

The control is #178's final 100-epoch control (`best.pth`, epoch index 89). Its mAP on these
2,182 rows is 0.3545, against 0.3552 for the interim epoch-49 snapshot this section was first
read against; the per-tag APs moved more than that (§4.1).

**Wider is worse, at a fixed 256 px input.** On the 2,182 common test rows, mAP falls
monotonically with field of view past the control's: the label-centred 25° arm matches the
control and the re-cut viewport arm, 50° loses 0.035, and 90° loses 0.074, with the loss
concentrated on the tags that are visible on the ramp itself. The street-dependent tags the
experiment was designed for (*points into traffic*, *not level with street*, *not enough
landing space*) show no detected gain from any wider crop, though the intervals do not rule out
gains of up to about 0.06–0.09 AP (reading 3). `summary.md`, from
`bash scripts/analysis/context_fov_86.sh report` (the full sequence is in §6):

| arm | n test | mAP | 95% CI | micro-F1 | macro-F1 | leak-free mAP | missing-tactile-warning | narrow | not-enough-landing-space | not-level-with-street | points-into-traffic | pooled-water | steep | surface-problem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control (#178, HF crops) | 2182 | 0.354 | [0.336, 0.390] | 0.659 | 0.315 | 0.370 | 0.973 | 0.234 | 0.182 | 0.141 | 0.323 | 0.323 | 0.062 | 0.597 |
| viewport | 2182 | 0.346 | [0.325, 0.381] | 0.650 | 0.330 | 0.346 | 0.968 | 0.232 | 0.168 | 0.146 | 0.310 | 0.284 | 0.099 | 0.556 |
| fov25 | 2182 | 0.349 | [0.329, 0.381] | 0.659 | 0.317 | 0.361 | 0.976 | 0.265 | 0.149 | 0.126 | 0.341 | 0.320 | 0.053 | 0.565 |
| fov50 | 2182 | 0.319 | [0.302, 0.350] | 0.636 | 0.293 | 0.331 | 0.957 | 0.236 | 0.195 | 0.110 | 0.333 | 0.198 | 0.032 | 0.494 |
| fov90 | 2182 | 0.280 | [0.266, 0.310] | 0.595 | 0.250 | 0.279 | 0.929 | 0.177 | 0.153 | 0.099 | 0.296 | 0.115 | 0.048 | 0.426 |

n = 2,182 test rows on 1,868 panoramas in every row; the CI is the pano-clustered bootstrap
(1,000 draws) on the arm's own mAP; leak-free = the 957 rows whose panorama has no train label.
*parallel-lines* and *tactile-warning* have too few positives to score and are outside the
fixed eight-tag average, as in #178. Test positives on the street-dependent tags: 297
(*points into traffic*), 84 (*not enough landing space*), 65 (*not level with street*).

**Paired contrasts** (`contrast_vs_control.json`, `contrast_vs_viewport.json`;
`context_fov_86.py contrast`): each arm minus the reference on the same test rows *and the same
1,000 pano resamples*, which is the interval that answers "is this arm different on this test
set". Separate CIs overlap for a reason unrelated to the arms (they share every panorama).
Per-tag AP differences whose interval excludes 0 are marked. An interval off 0, here and in
§4.1, says two checkpoints differ on this test set, not that two recipes differ: a change of
seed alone gave one on mAP (fov25, −0.025) and on single tags (up to −0.140 AP), §4.2 "Seed
noise".

| arm minus reference | mAP | micro-F1 | macro-F1 | per-tag AP with an interval off 0 |
|---|---:|---:|---:|---|
| viewport − control | −0.009 [−0.031, +0.017] | −0.010 [−0.024, +0.004] | +0.016 [−0.018, +0.052] | none (surface-problem −0.041 [−0.087, +0.006]) |
| fov25 − control | −0.005 [−0.031, +0.016] | −0.000 [−0.015, +0.015] | +0.002 [−0.030, +0.035] | none (points-into-traffic +0.017 [−0.031, +0.062]) |
| fov50 − control | **−0.035 [−0.063, −0.011]** | **−0.024 [−0.040, −0.008]** | −0.022 [−0.056, +0.012] | missing-tactile −0.016, surface-problem −0.103 |
| fov90 − control | **−0.074 [−0.101, −0.048]** | **−0.065 [−0.083, −0.048]** | **−0.064 [−0.098, −0.033]** | missing-tactile −0.045, narrow −0.057, pooled-water −0.208, surface-problem −0.171 |
| fov25 − viewport | +0.004 [−0.024, +0.025] | +0.009 [−0.005, +0.023] | −0.013 [−0.045, +0.018] | none |
| fov50 − viewport | **−0.026 [−0.053, −0.002]** | −0.014 [−0.030, +0.003] | **−0.038 [−0.073, −0.003]** | steep −0.067, surface-problem −0.062 |
| fov90 − viewport | **−0.065 [−0.093, −0.042]** | **−0.055 [−0.072, −0.038]** | **−0.080 [−0.114, −0.045]** | missing-tactile −0.040, narrow −0.055, pooled-water −0.169, surface-problem −0.130 |

The same contrast restricted to the 957 leak-free rows on 889 panoramas
(`contrast_vs_control_leak_free.json`, `contrast --subset leak_free`), where the control's own
mAP is 0.370:

| arm minus reference, leak-free rows | mAP | micro-F1 | macro-F1 | per-tag AP with an interval off 0 |
|---|---:|---:|---:|---|
| viewport − control | −0.025 [−0.051, +0.009] | −0.009 [−0.031, +0.012] | +0.024 [−0.025, +0.074] | surface-problem −0.094 [−0.164, −0.023] |
| fov25 − control | −0.009 [−0.044, +0.024] | +0.002 [−0.018, +0.022] | +0.008 [−0.037, +0.051] | none |
| fov50 − control | **−0.039 [−0.077, −0.008]** | **−0.021 [−0.044, −0.001]** | −0.019 [−0.066, +0.028] | missing-tactile −0.015, pooled-water −0.189, steep −0.048, surface-problem −0.089 |
| fov90 − control | **−0.092 [−0.130, −0.052]** | **−0.063 [−0.089, −0.041]** | **−0.070 [−0.119, −0.025]** | missing-tactile −0.036, narrow −0.079, pooled-water −0.315, surface-problem −0.175 |

The pattern is the full-set one, larger, with *pooled water* and *steep* also off 0 at 50°.
Two of the fov50 marks are at the edge of detection and depend on the bootstrap draws: the
micro-F1 upper end is −0.0009 with the committed seed (86) and −0.0004 / +0.0013 in the final
review's own re-implementation of the bootstrap at seeds 20260924 / 7 (not a committed script;
`contrast` has no seed flag), and *surface problem*'s is −0.012 with seed 86 and −0.0046 /
+0.0000 in that re-implementation. Read them as borderline, not as detected.

Against `viewport` on the same leak-free rows (`contrast_vs_viewport_leak_free.json`, added in
the final review; it does not read the control):

| arm minus reference, leak-free rows | mAP | micro-F1 | macro-F1 | per-tag AP with an interval off 0 |
|---|---:|---:|---:|---|
| fov25 − viewport | +0.015 [−0.020, +0.047] | +0.011 [−0.009, +0.032] | −0.015 [−0.062, +0.031] | missing-tactile +0.013 [+0.002, +0.027] |
| fov50 − viewport | −0.015 [−0.049, +0.011] | −0.012 [−0.035, +0.012] | −0.043 [−0.102, +0.015] | steep −0.064 [−0.226, −0.002] |
| fov90 − viewport | **−0.067 [−0.105, −0.033]** | **−0.054 [−0.079, −0.030]** | **−0.094 [−0.150, −0.038]** | missing-tactile −0.030, narrow −0.076, pooled-water −0.205 |

Reading, in the order the arms were built to be read:

1. **Re-cutting from the archive: no mAP loss detected, bounded at 0.031 on all test rows and
   0.051 on the leak-free rows.** `viewport` (the labeler's view rendered from the pano store,
   same 640 px box) is −0.009 [−0.031, +0.017] against the HF screenshot control on mAP, so a
   loss of up to 0.031 (about 9% of 0.354) is not ruled out. No tag's interval excludes 0 on
   the full set; the largest loss is *surface problem*, −0.041 [−0.087, +0.006]. On the 957
   leak-free rows mAP is 0.370 → 0.346, paired −0.025 [−0.051, +0.009], and *surface problem*
   loses −0.094 [−0.164, −0.023], a detected loss on that subset. *Surface problem* is a
   texture tag, which is where the viewport arm's second JPEG encode (§6) would show, so the
   encode is not ruled out as a cause. What plan item 5 inherits is this bound, not a go-ahead:
   crops cut by `crop_cutter.py` for the ~354k labels without a production crop cost no
   detected mAP against the screenshots the benchmark was built on, with the loss bounded at
   about 0.031 mAP on all test rows and 0.051 on the leak-free rows; *surface problem* loses
   about 0.04 AP on all rows (interval through 0) and 0.09 on the leak-free rows (off 0).
2. **Centring the ramp at the zoom-2 field of view changes nothing detectable on mAP.**
   `fov25`'s point estimates are within 0.016 of both the control and `viewport` on mAP,
   micro-F1 and macro-F1, on all rows and on the leak-free rows, and every one of those
   intervals includes 0. Against the control no tag's interval excludes 0, on all rows or on the
   leak-free rows. Against `viewport` none does on all rows, but on the leak-free rows *missing
   tactile warning* gains +0.013 [+0.002, +0.027]: the interim control's leak-free hint (§4.1),
   now against the re-cut. It is a tag visible on the ramp itself, so if it is real it is the
   centring, not street context; one seed per arm cannot say. *Points into traffic* is +0.017
   [−0.031, +0.062] against the control (+0.046 [−0.026, +0.118] on the leak-free rows) and
   +0.030 [−0.017, +0.076] against `viewport` (+0.042 [−0.020, +0.107] leak-free); none of these
   is a detected gain.
3. **At a fixed 256 px input, widening the crop costs the ramp-surface tags, and no tag gains
   from context.** Field of view and angular resolution move together in these arms (§2): at 50° the
   ramp gets about a quarter of the model pixels it had at 25°, at 90° about a twentieth. The
   losses are on the tags that need pixels on the ramp. At 50°, *surface problem* (−0.103) and
   *missing tactile warning* (−0.016) drop with intervals off 0, and mAP is −0.035
   [−0.063, −0.011]; *steep* is −0.029 [−0.068, +0.008], through 0 (on the leak-free rows it is
   off 0, −0.048, as is *pooled water*, −0.189). At 90°, *surface problem* (−0.171), *pooled
   water* (−0.208), *narrow* (−0.057) and *missing tactile warning* (−0.045) do, and mAP is
   −0.074 [−0.101, −0.048]; *steep* at 90° is −0.014 [−0.050, +0.017], through 0. The same four
   at 90° are off 0 against `viewport` too. The street-dependent tags show no detected gain at
   any field of view (against the control, for fov25 / fov50 / fov90): *points into traffic*
   +0.017 / +0.010 / −0.027, *not level with street* −0.015 / −0.032 / −0.042, *not enough
   landing space* −0.033 / +0.013 / −0.030, every interval through 0. The intervals do not rule
   out gains of up to about 0.06 AP on *points into traffic* (fov25, upper end +0.062) and *not
   level with street* (fov25, +0.057), and 0.09 on *not enough landing space* (fov50, +0.089);
   one seed per arm cannot resolve gains that size. How much of the loss is the lost pixels is
   §4.2's question: at 90° about 0.03 mAP goes with the pixels and the rest with the wider crop
   (more street and a smaller ramp at the input, together), point estimates with about ±0.025
   of seed noise on each; per tag, only *missing tactile warning*'s loss is clear of seed noise.

So the plan-item-4 question, "do the street-dependent tags fail for lack of context or lack of
vision", is **not answered by this design**. What it shows is narrower: delivering context as a
wider single crop at the tagger's 256 px input does not produce a detectable gain on any
street-dependent tag and costs the ramp-surface tags. In the four arms of this section that
loss could be resolution rather than context, because a wider crop is also a coarser one. §4.2
runs the arms that separate the two (fov25 crops downsampled to fov90's and fov50's centre
resolution, 57 and 122 px, then resized to 256 by the trainer) and finds both: about 0.03 mAP
of fov90's loss goes with the lost pixels, and the rest goes with the wider crop, which there
means more street *and* a smaller ramp at the model's input, not separable by that design.
With one or two seeds per arm and seed noise measured at 0.006–0.025 mAP, that split is a point
estimate (§4.2). Three readings survive for the street-dependent tags. They may be limited by
the labels (§2.2 of the plan: positive-unlabeled, rater-dependent), which no crop can fix and
item 5 addresses; they may need context *and* resolution together (a larger input or a
two-crop model); or context may not help them at all. The converse arm, fov90 at a larger input
so that its centre resolution matches fov25's (about 1,150 px), would test whether context
helps once resolution is restored; it is not run (§6), since it needs a different backbone
input size and far more GPU time.

### 4.1 Superseded: the first read, against the interim control

Section 4 was first written on 2026-09-24 against an interim control, the dead first #178
launch's epoch-index-49 snapshot (§6, first bullet). That is a different checkpoint from the
relaunch's own epoch-49 snapshot (`docs/tag_benchmark_86.md` §5.3), and from the final control.
Its files stay committed (`control_interim_ep49_*`, `contrast_vs_control_interim_ep49{,_leak_free}.json`,
`summary_interim_ep49.{json,md}`). The arm rows and the "− viewport" contrasts are the same in
both reads; only the control and the "− control" contrasts changed.

| control | mAP, all 2,182 rows | leak-free mAP | missing-tactile-warning | narrow | not-enough-landing-space | not-level-with-street | points-into-traffic | pooled-water | steep | surface-problem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| interim (first launch, epoch index 49) | 0.355 | 0.381 | 0.970 | 0.259 | 0.164 | 0.171 | 0.296 | 0.294 | 0.078 | 0.610 |
| final (relaunch, epoch index 89) | 0.354 | 0.370 | 0.973 | 0.234 | 0.182 | 0.141 | 0.323 | 0.323 | 0.062 | 0.597 |

The interim "− control" contrasts, all rows / leak-free rows (mAP, then the per-tag APs whose
interval excluded 0):

| arm minus interim control | mAP, all rows | per-tag off 0, all rows | mAP, leak-free | per-tag off 0, leak-free |
|---|---:|---|---:|---|
| viewport | −0.010 [−0.035, +0.017] | surface-problem −0.054 [−0.102, −0.009] | −0.036 [−0.069, +0.000] | surface-problem −0.111 |
| fov25 | −0.006 [−0.035, +0.020] | none | −0.020 [−0.060, +0.016] | missing-tactile +0.012 [+0.001, +0.023] |
| fov50 | −0.036 [−0.064, −0.009] | missing-tactile −0.013, steep −0.045, surface-problem −0.116 | −0.051 [−0.088, −0.017] | steep −0.147, surface-problem −0.106 |
| fov90 | −0.075 [−0.105, −0.048] | missing-tactile −0.041, narrow −0.082, pooled-water −0.179, surface-problem −0.184 | −0.103 [−0.147, −0.063] | missing-tactile −0.030, narrow −0.103, pooled-water −0.233, steep −0.131, surface-problem −0.192 |

**What changed against the final control.** The headline did not: mAP still falls with field of
view past 25°, fov50 and fov90 still lose with intervals off 0 on mAP (−0.035 and −0.074, were
−0.036 and −0.075), and no street-dependent tag gains. Four finer readings did change:

- **Viewport, all rows: the *surface problem* loss is no longer detected.** −0.054
  [−0.102, −0.009] became −0.041 [−0.087, +0.006]. On the leak-free rows it still is (−0.094,
  was −0.111).
- **Viewport, leak-free rows: the mAP loss is no longer borderline.** −0.036 [−0.069, +0.000]
  (upper end +0.0002 with the committed seed, −0.0006 with other draws) became −0.025
  [−0.051, +0.009]. The bound item 5 inherits tightens from 0.069 to 0.051.
- **fov25: the two hints are gone.** *Points into traffic* +0.045 [−0.003, +0.089] became
  +0.017 [−0.031, +0.062], because the final control is better on that tag (0.323, was 0.296);
  the leak-free *missing tactile warning* gain, +0.012 [+0.001, +0.023], became +0.006
  [−0.004, +0.015].
- **fov50, all rows: *steep* drops out** (−0.045 [−0.138, −0.001] became −0.029
  [−0.068, +0.008]), and on the leak-free rows *pooled water* and *missing tactile warning*
  come in (−0.189, −0.015) while *steep* stays. At 90° the four tags off 0 on all rows are the
  same four; on the leak-free rows *steep* drops out.

The upper ends that bound an undetected street-tag gain moved too: about 0.09 on *points into
traffic* and 0.10 on *not enough landing space* against the interim control, 0.06 and 0.09
against the final one (fov arms, as in reading 3); *not level with street* moved from +0.045
to +0.057 on the same basis (over all four arms, viewport included: +0.048 to +0.069).

### 4.2 Resolution arms: the wider arms' loss is not only the lost pixels (2026-09-25)

Reading 3 above could not tell resolution from context, because every arm is resized to 256 px
and a wider crop is also a coarser one. The separating arm it named is now run, twice: the
fov25 crops downsampled to the centre resolution a wider arm has at the model's input, then
trained with the same recipe on the same labels and scored on the same 2,182 common test rows
(`context_fov_86.py downsample`, `context_fov_86.sh downsample`, `res-contrast`, `res-report`).

| arm | what the model sees | side before the trainer's resize to 256 | information content at the centre, px/deg (sampled at 256 px) |
|---|---|---:|---:|
| `fov25px122` | the fov25 scene with fov50's information per degree | 122 px (LANCZOS from the 640 px fov25 crop, JPEG quality 92) | 4.8 |
| `fov25px57` | the fov25 scene with fov90's information per degree | 57 px | 2.2 |

The side is derived, not chosen: the ratio of the gnomonic focal lengths at 256 px,
`matched_px` in the script (`tests/test_context_fov_86.py` pins 122 and 57 and the 10.1 / 4.8 /
2.2 px/deg of §2). Because the focal lengths are equal, the central 25° of fov90's own crop is
sampled exactly as `fov25px57` is before the resize (and the central 25° of fov50's as
`fov25px122` is). Gnomonic px/deg rises away from the centre (as sec²θ), so fov90's periphery
is sampled more finely than its centre, up to 2× at ±45°; "centre" is the right figure only
for the shared central field.

**The match is of information, not of what the input looks like.** The trainer resizes every
crop to 256 px, so a resolution arm is upsampled, 4.5× from 57 px and 2.1× from 122 px: the
model sees it at fov25's 10.1 px/deg, band-limited to 2.2 (or 4.8). The ramp therefore reaches
the model at fov25's size, filling the frame: about 18×18 of DINOv2's 14 px patches, where the
same central 25° inside fov90's 256 px input is 57 px, about 4×4 patches (about 9×9 inside
fov50's). So the pairs differ as follows. `fov25px57` against fov25 is the same scene at the
same size, band-limited to fewer pixels per degree. `fov25px57` against fov90 has the same
information per degree on the shared field, but two things change together: less street, and a
ramp 4.5× larger (linear) at the input. Likewise `fov25px122` against fov25, and against fov50
(2.1×). This design cannot say which of the two is what fov90 loses to; the arm that would (fov90
with everything outside the central 25° greyed out) is not run (§6). The resolution arms are
also JPEG-encoded twice, where fov25 / fov50 / fov90 are encoded once, and the resize enlarges
the second encode's 8×8 blocks 4.5× / 2.1× (§6); that penalises the resolution arms, so it
would inflate the resolution term and deflate the other.

**Scores on the common test rows** (`summary_res.md`, which also has the viewport row; the first
four rows here are §4's):

| arm | mAP | 95% CI | micro-F1 | macro-F1 | leak-free mAP | missing-tactile | narrow | landing | not-level | into-traffic | pooled-water | steep | surface |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control (#178) | 0.354 | [0.336, 0.390] | 0.659 | 0.315 | 0.370 | 0.973 | 0.234 | 0.182 | 0.141 | 0.323 | 0.323 | 0.062 | 0.597 |
| fov25 | 0.349 | [0.329, 0.381] | 0.659 | 0.317 | 0.361 | 0.976 | 0.265 | 0.149 | 0.126 | 0.341 | 0.320 | 0.053 | 0.565 |
| fov50 | 0.319 | [0.302, 0.350] | 0.636 | 0.293 | 0.331 | 0.957 | 0.236 | 0.195 | 0.110 | 0.333 | 0.198 | 0.032 | 0.494 |
| fov90 | 0.280 | [0.266, 0.310] | 0.595 | 0.250 | 0.279 | 0.929 | 0.177 | 0.153 | 0.099 | 0.296 | 0.115 | 0.048 | 0.426 |
| `fov25px122` | 0.341 | [0.321, 0.374] | 0.650 | 0.328 | 0.362 | 0.962 | 0.286 | 0.163 | 0.114 | 0.335 | 0.262 | 0.059 | 0.548 |
| `fov25px57` | 0.322 | [0.302, 0.355] | 0.621 | 0.310 | 0.332 | 0.939 | 0.245 | 0.156 | 0.132 | 0.311 | 0.258 | 0.039 | 0.498 |

**Paired contrasts** (`contrast_res_vs_fov25.json`, `contrast_fov25px122_vs_fov50.json`,
`contrast_fov25px57_vs_fov90.json`, `contrast_res_vs_control.json`, and `_leak_free` for each;
same rows, same 1,000 pano resamples; per-tag differences whose interval excludes 0 listed;
"off 0" is about these checkpoints, not the recipe, see "Seed noise" below):

| contrast | what changes | mAP | micro-F1 | macro-F1 | per-tag AP off 0 |
|---|---|---:|---:|---:|---|
| `fov25px122` − fov25 | fewer px/deg (fov50's); same scene, same ramp size at the input | −0.008 [−0.029, +0.017] | −0.009 [−0.024, +0.005] | +0.011 [−0.023, +0.044] | missing-tactile −0.014 |
| `fov25px122` − fov50 | less street and a ramp 2.1× larger at the input; same px/deg | +0.022 [−0.001, +0.046] | +0.014 [−0.002, +0.031] | +0.035 [−0.001, +0.071] | steep +0.027, surface-problem +0.053 |
| `fov25px57` − fov25 | fewer px/deg (fov90's); same scene, same ramp size at the input | **−0.027 [−0.053, −0.001]** | **−0.038 [−0.054, −0.022]** | −0.007 [−0.043, +0.030] | missing-tactile −0.036, surface-problem −0.066 |
| `fov25px57` − fov90 | less street and a ramp 4.5× larger at the input; same px/deg | **+0.042 [+0.019, +0.067]** | **+0.026 [+0.010, +0.043]** | **+0.059 [+0.025, +0.094]** | narrow +0.068, pooled-water +0.143, surface-problem +0.072 |
| `fov25px122` − control | | −0.013 [−0.040, +0.014] | −0.010 [−0.026, +0.006] | +0.013 [−0.031, +0.054] | missing-tactile −0.011 |
| `fov25px57` − control | | **−0.032 [−0.064, −0.005]** | **−0.038 [−0.057, −0.020]** | −0.005 [−0.048, +0.040] | missing-tactile −0.034, surface-problem −0.099 |

On the 957 leak-free rows mAP shows the same pattern, with wider intervals: `fov25px122` − fov25
+0.001 [−0.028, +0.032], `fov25px122` − fov50 +0.031 [+0.005, +0.069], `fov25px57` − fov25
−0.029 [−0.068, +0.007], `fov25px57` − fov90 +0.053 [+0.019, +0.088], `fov25px122` − control
−0.009 [−0.044, +0.030], `fov25px57` − control −0.038 [−0.083, +0.001]. Per tag, the leak-free
rows include one *gain* from lower resolution, *narrow* +0.091 [+0.010, +0.154] for
`fov25px122` − fov25, which is no more than a seed's move on that subset (below) and reads as
noise.

**What this says**, each reading with its seed caveat (the seed pair is below):

1. **Halving the resolution costs nothing measurable; quartering it costs about 0.03 mAP, the
   size of one seed's move.** At fov50's pixels per degree the fov25 scene scores as fov25
   does (−0.008, interval through 0). At fov90's it loses 0.027 [0.001, 0.053], about what a
   seed change alone moved fov25 (−0.025), so the mAP term is not established beyond one seed.
   One per-tag effect does stand clear of seed noise: *missing tactile warning* loses −0.036
   [−0.051, −0.023] at 57 px and −0.014 [−0.025, −0.004] at 122 px, where a seed change moved
   it by −0.004 on both fov25 and fov90, with intervals of about ±0.01. *Surface problem*'s
   −0.066 at 57 px does not: a seed change alone moved that tag −0.050 on fov25 (−0.140 on the
   leak-free rows), interval off 0.
2. **The rest of the wider arms' loss goes with the wider crop, which here is more street and a
   smaller ramp at the input together.** At the same information per degree, the 25° scene
   beats the 90° scene by 0.042 [0.019, 0.067] on mAP and on both F1s, larger than either
   measured seed move, and beats the 50° scene by 0.022 [−0.001, +0.046] (off 0 on the
   leak-free rows, +0.031 [+0.005, +0.069]), which is within one seed. As point estimates,
   fov90's 0.074 against the control splits into about 0.03 that goes with the lost pixels and
   0.04 that goes with the wider crop, and 0.022 of fov50's 0.035 (about 60%) goes with the
   wider crop; each term carries about ±0.025 of seed noise. Which half of "the wider crop" it
   is, the street or the ramp's smaller size at the input, this design does not say. The
   per-tag split of that term (*narrow* +0.068, *pooled water* +0.143, *surface problem* +0.072
   at 90°; *steep* +0.027, *surface problem* +0.053 at 50°) is comparable to what a seed change
   alone does to the same tags (on fov25, all rows: *pooled water* −0.097, *narrow* −0.038,
   *surface problem* −0.050; leak-free, *pooled water* −0.163 and *surface problem* −0.140), so
   it is not a finding on its own.
3. **No street-dependent tag gains at matched resolution either.** *Points into traffic*, *not
   level with street* and *not enough landing space* are through 0 in every resolution-arm
   contrast here, in both directions. The bound is loose: the highest upper end is +0.108 on
   all rows (*not level with street*, `fov25px57` − fov90, +0.033 [−0.034, +0.108]) and +0.201
   on the leak-free rows (the same tag and contrast), and with one seed per resolution arm a
   gain of that size is not ruled out. So a wider single crop at this input does not deliver
   street context these tags can use, whatever its cost is made of: the pixels, the ramp's
   size, or the street.

**Seed noise, measured (2026-09-26).** A second seed (87) of fov25 and fov90 ran on the same
crops, labels and rows (`context_fov_86.slurm` with `SEED=87`, outputs `train_<arm>_s87*`,
klone jobs 40599943 / 40599944), the first seed-to-seed numbers for this benchmark
(`contrast_seed_fov25.json`, `contrast_seed_fov90.json`, `contrast_fov90_vs_fov25.json`,
`contrast_fov90_s87_vs_fov25_s87.json`, and `_leak_free` for each; all from
`context_fov_86.sh seed-contrast`):

| contrast | mAP, all rows | mAP, leak-free | per-tag AP off 0, all rows | per-tag AP off 0, leak-free |
|---|---:|---:|---|---|
| fov25 seed 87 − seed 86 | **−0.025 [−0.047, −0.000]** | **−0.042 [−0.069, −0.005]** | surface-problem −0.050 | surface-problem −0.140 |
| fov90 seed 87 − seed 86 | −0.006 [−0.025, +0.012] | −0.002 [−0.029, +0.026] | not-enough-landing-space −0.046 | landing −0.047, surface-problem −0.073 |
| fov90 − fov25, both at seed 86 | **−0.069 [−0.092, −0.046]** | **−0.082 [−0.116, −0.046]** | missing-tactile −0.047, narrow −0.088, pooled-water −0.205, surface-problem −0.139 | missing-tactile −0.042, pooled-water −0.288, surface-problem −0.136 |
| fov90 − fov25, both at seed 87 | **−0.050 [−0.074, −0.028]** | **−0.042 [−0.082, −0.007]** | missing-tactile −0.048, landing −0.047, surface-problem −0.098 | missing-tactile −0.046 |

Per tag, a seed change moves more than it does on mAP. On fov25: *surface problem* −0.050
[−0.101, −0.001] (−0.140 [−0.221, −0.059] on the leak-free rows), and, through 0, *pooled water*
−0.097 (−0.163), *narrow* −0.038 and *not level with street* −0.034. On fov90: *not enough
landing space* −0.046 [−0.107, −0.000] (−0.047 leak-free) and, on the leak-free rows, *surface
problem* −0.073. *Missing tactile warning* is the stable tag: −0.004 on fov25 and on fov90, both
intervals through 0 and about ±0.01 wide.

fov25 scores 0.349 at seed 86 and 0.324 at seed 87; fov90 0.280 and 0.274. Both seed-87 runs
train to the same end state as their seed-86 twins (final training exact-match accuracy 0.997 /
0.999 for fov25, best epoch 77 / 83; fov90 best epoch 93 / 47), so the fov25 move is
checkpoint-to-checkpoint noise, not a failed run. What that means for everything above:

- **A seed change alone can give a paired interval that excludes zero** (fov25, −0.025
  [−0.047, −0.000] on mAP; −0.050 and −0.140 on *surface problem*; −0.046 on fov90's *not enough
  landing space*). The pano-clustered bootstrap answers "are these two checkpoints different
  on this test set"; it does not cover training noise. So "interval off 0" in the contrast
  tables of §4, §4.1 and §4.2 is not "the recipe is different". Any mAP effect of about 0.025
  or less (the resolution term at fov90's pixels, −0.027; `fov25px122` − fov50, +0.022) is
  within one seed's reach, and fov50's −0.035 against the control is comparable to it: larger
  than both measured seed moves, but two pairs do not bound the next one. The two pairs put
  seed-to-seed spread at 0.006–0.025 mAP; with n = 2 that is a range, not a standard deviation.
- **The per-tag attributions of §4.2 are within one seed, except *missing tactile warning*.**
  That tag's seed move is about −0.004 against a −0.036 resolution effect at 57 px (−0.014 at
  122 px). Every other per-tag difference in the resolution contrasts is about the size of a
  seed's move on the same tag, so those tag lists say where the loss fell for these
  checkpoints, not which tags the pixels or the street cost. The largest per-tag losses of §4,
  fov90 against the control (*pooled water* −0.208, *surface problem* −0.171), are bigger than
  any per-tag seed move measured on all rows (at most −0.097), and *surface problem* loses at
  both seeds (next bullet).
- **The fov90 deficit replicates across seeds**: −0.069 [−0.092, −0.046] at seed 86 (0.280 vs
  0.349) and −0.050 [−0.074, −0.028] at seed 87 (0.274 vs 0.324). Per tag, what replicates is
  *missing tactile warning* and *surface problem*, off 0 at both seeds (−0.047 / −0.048 and
  −0.139 / −0.098); *narrow* and *pooled water* lose at both seeds but are off 0 only at seed 86
  (−0.088, −0.205; at seed 87 −0.054 and −0.095, through 0). At seed 87 a street-dependent tag,
  *not enough landing space*, also loses off 0 (−0.047), about fov90's own seed move on that
  tag. The wider-crop term (`fov25px57` − fov90, +0.042) is larger than either seed move on mAP
  but is itself one seed of `fov25px57` against one of fov90. The split "0.03 with the pixels +
  0.04 with the wider crop" is a point estimate with about ±0.025 of seed noise on each term.
  The reading that survives is the qualitative one, which holds in every arm, on both row
  subsets and at both seeds: widening the crop at this input costs mAP, most reliably on
  *missing tactile warning* and *surface problem*, and no street-dependent tag gains from it.

**How the five klone jobs were submitted** (2026-09-25, from the checkout
`/gscratch/makelab/jonf/context_fov_86/RampNet`; the lines are sacct's `SubmitLine`, with
`WORK=/gscratch/makelab/jonf/context_fov_86`, `PY=/gscratch/makelab/jonf/envs/tagger/bin/python`
and `REPO` the checkout):

```bash
sbatch /gscratch/makelab/jonf/context_fov_86/downsample.slurm      # 40599892, now scripts/analysis/context_fov_86_downsample.slurm
sbatch --dependency=afterok:40599892 --job-name=ctx_fov25px57 \
  --export=ALL,ARM=fov25px57,WORK=$WORK,PY=$PY,REPO=$REPO scripts/analysis/context_fov_86.slurm    # 40599914
sbatch --dependency=afterok:40599892 --job-name=ctx_fov25px122 \
  --export=ALL,ARM=fov25px122,WORK=$WORK,PY=$PY,REPO=$REPO scripts/analysis/context_fov_86.slurm   # 40599915
sbatch --nice=200 --dependency=afterany:40599915 --job-name=ctx_fov25_s87 \
  --export=ALL,ARM=fov25,SEED=87,WORK=$WORK,PY=$PY,REPO=$REPO scripts/analysis/context_fov_86.slurm  # 40599943
sbatch --nice=200 --dependency=afterany:40599915 --job-name=ctx_fov90_s87 \
  --export=ALL,ARM=fov90,SEED=87,WORK=$WORK,PY=$PY,REPO=$REPO scripts/analysis/context_fov_86.slurm  # 40599944
```

The same through the stage script: `sbatch scripts/analysis/context_fov_86_downsample.slurm`, then
`SBATCH_ARGS="--dependency=afterok:<that job>" ARMS="fov25px57 fov25px122" ... context_fov_86.sh train`
and `SEED=87 SBATCH_ARGS="--nice=200 --dependency=afterany:<the fov25px122 job>" ARMS="fov25 fov90" ... context_fov_86.sh train`
(the `train` stage names the jobs as above and passes `SEED` through). The CPU stages that
follow, from committed files only: `res-contrast res-report seed-contrast`. Re-running
`seed-contrast` on 2026-09-26 reproduced the six seed files committed with the runs byte for
byte apart from `ts` (those files are kept as run), and wrote the two seed-86 fov90 − fov25
files new.

## 5. Cost

All of it free: makelab2 (the lab's A40 box) and klone's `gpu-l40s-makelab` allocation. Every
number here is a committed ledger row: `analysis_out/usage_log.jsonl` (`train-<arm>`,
`infer-control-ep49-interim`, keyed by `run_id`) and `analysis_out/compute_log.jsonl` (the ten
Slurm jobs: five from `docs/data/compute/sacct_klone_2026-09-24.txt` and five from
`docs/data/compute/sacct_klone_2026-09-26.txt`, see
[`compute_cost.md`](compute_cost.md)).

| step | where | wall-clock | GPU-hours | $ |
|---|---|---:|---:|---:|
| cut, 3 fov arms (32,544 crops) + viewport (10,848) | makelab2, 12 CPU workers | 1,767 s + 1,122 s | 0 | 0 |
| env build (job 40485927) | klone `ckpt-all`, CPU | 144 s | 0 | 0 |
| `viewport` (job 40486691, g3120) | klone, 1x L40S | 4.04 h (train 3.15 h) | 4.04 | 0 |
| `fov25` (job 40486692, g3100) | klone, 1x L40S | 4.95 h (train 3.10 h, prep 0.73 h) | 4.95 | 0 |
| `fov50` (job 40486693, g3100) | klone, 1x L40S | 3.98 h (train 3.14 h, prep 0.45 h) | 3.98 | 0 |
| `fov90` (job 40486694, g3104) | klone, 1x L40S | 3.57 h (train 3.13 h) | 3.57 | 0 |
| interim control inference (epoch-49 snapshot, 10,857 crops) | makelab2 A40, shared 4 ways (`gpu_share` 0.25) | 476 s | 0.03 | 0 |
| downsample, both resolution arms (job 40599892, n3194, §4.2) | klone `ckpt-all`, 4 CPUs | 11.9 min | 0 | 0 |
| `fov25px57` (job 40599914, g3108) | klone, 1x L40S | 3.33 h (train 3.11 h, prep 103 s) | 3.33 | 0 |
| `fov25px122` (job 40599915, g3108) | klone, 1x L40S | 3.35 h (train 3.11 h, prep 204 s) | 3.35 | 0 |
| fov25 seed 87 (job 40599943, g3112, §4.2) | klone, 1x L40S | 3.53 h (train 3.13 h) | 3.53 | 0 |
| fov90 seed 87 (job 40599944, g3100, §4.2) | klone, 1x L40S | 3.40 h (train 3.05 h, prep 400 s) | 3.40 | 0 |
| **total** | | | **30.18** (klone 30.15 + makelab2 0.03) | **0** |

GPU-hours on the shared A40 are `elapsed_s × gpu_share` ([`tag_benchmark_86.md`](tag_benchmark_86.md)
§7), so the interim inference is a quarter of its 476 s. The final control costs this
experiment nothing further: its training and inference are #178's (`train-control` and
`infer-train-control-final` in `usage_log.jsonl`, counted in
[`tag_benchmark_86.md`](tag_benchmark_86.md) §7), and scoring it here is CPU. The klone figure is the sum of the eight
L40S rows in `compute_log.jsonl`: the four of 2026-09-24 (`tests/test_slurm_usage.py` pins
16.54) and the four of 2026-09-25/26 (two resolution arms, two seed-87 runs; the same test pins
their 13.61 together with the downsample job, from
`docs/data/compute/sacct_klone_2026-09-26.txt`, [`compute_cost.md`](compute_cost.md)).
Training itself is the same 3.1 h on every arm (100 epochs at 111–116 s on a dedicated
L40S, `train_<arm>_train_meta.json`); the spread in job wall-clock is the one-off crop
preparation on g3100 (fov25 2,634 s, fov50 1,603 s, against 349–369 s on the other two nodes;
first read of the crops from `/gscratch`), then inference and scoring. One L40S was free on the
allocation, so the arms ran one after another (14:20 UTC to 06:53 UTC the next day); on the
#178 recipe's makelab2 A40 the same four arms would have shared one GPU at ~590 s/epoch.

## 6. Not run / caveats

- **The first control was interim (superseded; the record of what was read first, §4.1).**
  The 100-epoch #178 control (`train_control`) died at epoch
  index 66 with the makelab2 reboot of 2026-09-23 15:15 UTC (recorded on #178). The interim
  control (§4.1) was that dead run's epoch-index-49 checkpoint (makelab2
  `/homes/gws/jonf/nobackup/tagbench86/dead_2026-09-23/train_control/best_after_ep49.pth`,
  sha256 `f22a2f0954d866a2aea06b0b2d06f3e8660dddc4911a8a43a5edc69e9b4bf036`), inferred on
  2026-09-24 over all 10,857 HF crops and filtered to the common test rows. The final control
  scores 0.3545 on these rows against the interim's 0.3552, with larger per-tag moves (§4.1).
  It comes from the relaunch, a different run, so the difference is not a clean read of what
  the last 50 epochs of one run do. The interim files were
  made by these commands, in order (the first on makelab2 with the #178 runbook's `TB_WORK`
  and tagger venv, the rest on CPU in any checkout; the `interim-infer` stage is reconstructed
  from the committed meta and ledger row, since its as-run shell line was not saved, and its
  output `control_interim_ep49_predictions_all.csv` stays on makelab2):

  ```bash
  # makelab2, GPU: -> $TB_WORK/control_interim_ep49_predictions_all.csv (+ .meta.json) and the ledger row
  CONTROL=interim PY=/homes/gws/jonf/envs/tagger/bin/python bash scripts/analysis/context_fov_86.sh interim-infer
  # CPU: `control` reads that makelab2 file; `contrast` and `report` read only committed files
  CONTROL=interim PY=python bash scripts/analysis/context_fov_86.sh control contrast report
  ```

  The second writes `control_interim_ep49_{test_predictions.csv,scores.json,per_label.csv}`,
  `contrast_vs_control_interim_ep49{,_leak_free}.json`, `contrast_vs_viewport.json` and
  `summary_interim_ep49.{json,md}`. From a clean clone, `CONTROL=interim ... contrast report`
  and `tag_benchmark_86.py score` over the committed `control_interim_ep49_test_predictions.csv`
  (with `--split-csv split_common.csv`) re-derive every interim number. Re-running them on the
  committed inputs reproduced the committed files except for their `ts` stamps (and the `subset`
  key that `contrast` now writes): checked 2026-09-24 for the score file, both contrast files'
  rows, and the summary table.

  **The swap to the final control** (done 2026-09-24, after the #178 re-run ended; kept as the
  exact sequence that produced §4). Everything after step 1 is CPU-only; nothing here touches
  makelab2.

  1. On #178 (`bench/tag-benchmark-86`), on makelab2: `tag_benchmark_86.sh finish`, then commit
     and push its outputs, as that doc's runbook says. That writes
     `analysis_out/tag_benchmark_86/train_control_final_test_predictions.csv`.
  2. Here:

     ```bash
     git fetch origin
     git merge origin/bench/tag-benchmark-86        # #178's final control predictions
     test -f analysis_out/tag_benchmark_86/train_control_final_test_predictions.csv
     PY=D:/Git/RampNet/.venv/Scripts/python.exe bash scripts/analysis/context_fov_86.sh control contrast report
     git checkout -- analysis_out/context_fov_86/contrast_vs_viewport.json   # it does not read the control; keep the as-run file
     git status --short analysis_out/context_fov_86
     ```

     `CONTROL=final` is the default. `test-only` must print 2182 rows. The new files are
     `control_test_predictions.csv` (+ `.meta.json`), `control_scores.json`,
     `control_per_label.csv`, `contrast_vs_control.json`, `contrast_vs_control_leak_free.json`,
     `summary.json` and `summary.md`; nothing interim is overwritten. Each `contrast` call
     (1,000 paired draws) took 12–18 minutes on the Windows desktop's CPU on 2026-09-24, with
     other jobs running, so the stage takes about 45 minutes.
  3. Replace every interim number. `grep -n -i interim docs/context_fov_86.md` lists every
     place: the status paragraph, §3's pointer to the control score file, §4's opening
     sentence, headline paragraph and report command (now `summary.md`, no `CONTROL=`), the
     control row of the results table, the four "− control" rows of the contrast table, the
     leak-free table, readings 1–3 and the conclusion that cites them. Re-read every reading
     against the new numbers rather than only swapping them: a reading that held against the
     interim control may not hold against the final one. Keep this bullet, re-titled as the
     record of what was read first; the interim files stay committed.
  4. PR #180's body: the "Result (interim control)" table and its reading, and the status
     checklist. Then commit, push, post the final numbers on #180, and mark it ready.
  The merge in step 2 will conflict in `analysis_out/usage_log.jsonl` (`finish` appends #178's
  rows there, as this branch did); keep both blocks: rows supersede by `run_id`, and no test
  reads them by position. #185's ledger conflict is already resolved on this branch.
  As run on 2026-09-24: `usage_log.jsonl` merged without a conflict (both blocks kept); the only
  conflict was `.gitignore`, where both branches carried the same `tag_benchmark_86` block, kept
  once with #178's `dead_run_2026-09-23/` exception. `test-only` printed 2182 rows, and the
  stage took about 45 minutes, as estimated. The final review added a fourth `contrast` call to
  the stage, `contrast_vs_viewport_leak_free.json` (§4); it reads no control and was run once,
  on its own, with the exact line now in `context_fov_86.sh`.

- **Not reproducible from a clean clone: the pano store, the crops and the checkpoints are
  lab-local.** The crops are cut from the makelab2 pano store, which is unpublished
  ([`replication.md`](replication.md)). The crops as trained are on klone,
  `/gscratch/makelab/jonf/context_fov_86/crops/`, and in the transfer tar beside them
  (`context_fov_86_crops.tar`, 6,284,554,240 bytes, sha256 in `crops_tar.sha256`). The 21,696
  crops of the two resolution arms (§4.2) are in the same `crops/` directory on klone but not
  in the tar; they re-derive from the tar's fov25 crops with the `downsample` stage
  (`context_fov_86_downsample.slurm`), and `crops_as_trained_fov25px122.sha256` and
  `crops_as_trained_fov25px57.sha256` hold the sha256 of each as trained (written by
  `downsample` on klone on 2026-09-25; a re-derivation on another Pillow or libjpeg build is not
  guaranteed to match them byte for byte). The eight `best.pth` (the four §4 arms, the two
  resolution arms and the two seed-87 runs) are in
  `/gscratch/makelab/jonf/context_fov_86/train_<run>/`, `<run>` being the arm name or
  `<arm>_s87` (sha256 in `train_<run>_best_pth.sha256`); the interim checkpoint and its
  all-crop predictions are on makelab2 (above). None is published. What a clean clone can
  re-derive: every number in §4 and §4.2, from the committed predictions, on CPU: `contrast`,
  `report`, `res-contrast`, `res-report` and `seed-contrast`, the first two in either mode, and `control`
  only with `CONTROL=final` (it reads #178's committed `train_control_final_test_predictions.csv`;
  in interim mode it reads the all-crop predictions on makelab2, so there
  `tag_benchmark_86.py score` over the committed `control_interim_ep49_test_predictions.csv` is
  the clean-clone route). What it can check
  without re-running: any copy of the crops against `crops_as_trained.sha256` (the sha256 of
  every one of the 43,392 JPEGs the arms trained on, written on klone on 2026-09-24 from the
  unpacked crops, none of whose mtime or ctime is later than the first training job's start; the
  tar's own sha256 was taken the same day, and `tests/test_context_fov_86.py` checks the listing
  against the label tables and the manifests, and the two resolution-arm listings against their
  label tables), and a re-cut against
  the cutter's manifests, `manifest_fov.jsonl.gz` and `manifest_viewport.jsonl.gz` (committed
  gzipped, one row per crop with its geometry and sha256 as cut). **The viewport crops were
  rewritten in place** by `crop640` after the manifest was written, so the manifest's sha256 for a
  viewport crop is of the 1440×960 cut, not of what trained; for those, only
  `crops_as_trained.sha256` describes the trained-on file. Re-training needs the crops, which
  means the store or the tar; publishing the tar (6.3 GB) and the eight checkpoints (about
  350 MB each) to Hugging Face would close that, and is not done here.
- **One seed per arm, two for fov25 and fov90; seed noise is measured at 0.006–0.025 mAP and
  up to 0.140 AP on a single tag (§4.2, two pairs).** A seed change alone produced a paired
  interval off 0 (fov25, −0.025 [−0.047, −0.000] on mAP; *surface problem* −0.050, −0.140 on
  the leak-free rows), so every "off 0" in this document is a statement about two checkpoints
  on this test set, not about the recipe; a difference between arms smaller than a seed's worth
  is not a result, and per tag only *missing tactile warning* (seed move about −0.004) is
  resolved. The paired contrasts in §4 are the right interval for "A vs B on these test rows"
  (both arms scored on the same pano resample), not for "A's recipe vs B's recipe" (that needs
  seeds). The checkpoint rule (best training exact-match accuracy) picked epoch index 81 / 77 /
  41 / 93 for viewport / fov25 / fov50 / fov90, 79 / 99 for `fov25px57` / `fov25px122`, and
  83 / 47 for the seed-87 fov25 / fov90, so the arms are not compared at one epoch either.
  `fov25px122`'s 99 is the last epoch: its best training accuracy (0.9998) came at the end of a
  noisy plateau at 0.997–0.9995, so a longer run could have picked a later checkpoint.
- **Field of view and resolution are confounded** (§2, reading 3) in the four arms of §4. The
  resolution arms (§4.2, run 2026-09-25) split the wider arms' loss, as point estimates with
  about ±0.025 mAP of seed noise on each term: about 0.03 mAP goes with the lost pixels at
  fov90's pixels per degree, none measurably at fov50's, and the rest goes with the wider crop.
  That rest is two things at once, more street and a smaller ramp at the model's input, because
  the trainer upsamples a resolution arm's 57 or 122 px crop to 256 px, so its ramp fills the
  frame (§4.2). **Not run: the arm that would separate those two**, fov90 crops with everything
  outside the central 25° set to flat mean grey. It keeps fov90's ramp size at the input and
  removes the street; it is a CPU-only edit of the existing fov90 crops plus one 3 h L40S run.
  **Also not run: the converse arm**, fov90 at a larger input so its centre resolution matches
  fov25's (about 1,150 px); it needs a different backbone input size and far more GPU time.
- **A second JPEG encode on the viewport arm and on the resolution arms.** The cutter writes
  JPEG at quality 92. On `viewport` the 640 px box is then re-saved at quality 92, and the
  trainer shrinks it 2.5× to 256 px. On `fov25px122` and `fov25px57` the downsampled crop is
  re-saved at quality 92 at 122 or 57 px, and the trainer then *enlarges* it 2.1× or 4.5×, so
  the second encode's 8×8 blocks reach the model about 17 or 36 px across. fov25, fov50 and
  fov90 are encoded once. The HF control is PNG throughout. At quality 92 the effect is probably
  small, but its direction is known: it penalises the resolution arms, which would inflate the
  resolution term of §4.2 and deflate the wider-crop term. Saving the resolution arms' crops as
  PNG would have cost nothing; do that if they are ever re-cut.
- **Store coverage.** Labels whose pano is not in the store (5 of the 10,853 cut,
  `cut_summary_viewport.json`: 10,848 ok, 5 `missing_pano`) drop from every arm, and from the
  control's re-score, but not from the control's training (§1).
- **The cut cost the #178 arms their run.** Cutting with 12 workers beside three trainers
  drove makelab2's load to 60 on 48 cores (12:30 UTC); the box stopped answering SSH and was
  rebooted at 15:15 UTC, which killed the three #178 arms at epoch 62–66 of 100 (no resume in
  the recipe). Cut with `WORKERS=6` or fewer when anything is training on that box; the cut
  took 48 minutes at 12 (both passes); expect roughly double at 6, and nothing else dies.
