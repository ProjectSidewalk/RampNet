# Context experiment: field of view vs curb-ramp tag accuracy (#86, RampNet 2.0 plan item 4)

**Status: four arms scored against the final #178 control (2026-09-24).** Crops cut on makelab2
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
Per-tag AP differences whose interval excludes 0 are marked.

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
3. **At a fixed 256 px input, widening the crop costs resolution faster than any tag gains from
   context.** Field of view and angular resolution move together in these arms (§2): at 50° the
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
   one seed per arm cannot resolve gains that size.

So the plan-item-4 question, "do the street-dependent tags fail for lack of context or lack of
vision", is **not answered by this design**. What it shows is narrower: delivering context as a
wider single crop at the tagger's 256 px input does not produce a detectable gain on any
street-dependent tag and costs the ramp-surface tags, and that loss may be resolution rather than context, because the
two are confounded here. Three readings survive. The street-dependent tags may be limited by the
labels (§2.2 of the plan: positive-unlabeled, rater-dependent), which no crop can fix and item 5
addresses; they may need context *and* resolution together (a larger input or a two-crop model);
or context may not help them at all. **The arm that separates resolution from context** is
fov25 crops downsampled to fov90's pixels per degree (2.2 px/deg at the centre: a 25° view
rendered at about 57 px, then resized to 256), trained with the same recipe: the same scene as
fov25 at fov90's resolution. If it loses what fov90 loses, the loss is resolution; if it keeps
fov25's scores, the loss is the added street. That is a CPU-only cut (or a resize of the
existing fov25 crops) plus one 3 h L40S run. The converse, fov90 at a larger input so that its
centre resolution matches fov25's (about 1,150 px), tests whether context helps once resolution
is restored, but needs a different backbone input size and far more GPU time.

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

## 5. Cost

All of it free: makelab2 (the lab's A40 box) and klone's `gpu-l40s-makelab` allocation. Every
number here is a committed ledger row: `analysis_out/usage_log.jsonl` (`train-<arm>`,
`infer-control-ep49-interim`, keyed by `run_id`) and `analysis_out/compute_log.jsonl` (the five
Slurm jobs, from `docs/data/compute/sacct_klone_2026-09-24.txt`, see
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
| **total** | | | **16.57** (klone 16.54 + makelab2 0.03) | **0** |

GPU-hours on the shared A40 are `elapsed_s × gpu_share` ([`tag_benchmark_86.md`](tag_benchmark_86.md)
§7), so the interim inference is a quarter of its 476 s. The final control costs this
experiment nothing further: its training and inference are #178's (`train-control` and
`infer-train-control-final` in `usage_log.jsonl`, counted in
[`tag_benchmark_86.md`](tag_benchmark_86.md) §7), and scoring it here is CPU. The klone figure is the sum of the four
L40S rows in `compute_log.jsonl` (`tests/test_slurm_usage.py` pins 16.54).
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
  (`context_fov_86_crops.tar`, 6,284,554,240 bytes, sha256 in `crops_tar.sha256`); the four
  `best.pth` are in `/gscratch/makelab/jonf/context_fov_86/train_<arm>/` (sha256 in
  `train_<arm>_best_pth.sha256`); the interim checkpoint and its all-crop predictions are on
  makelab2 (above). None is published. What a clean clone can re-derive: every number in §4,
  from the committed predictions, on CPU: `contrast` and `report` in either mode, and `control`
  only with `CONTROL=final` (it reads #178's committed `train_control_final_test_predictions.csv`;
  in interim mode it reads the all-crop predictions on makelab2, so there
  `tag_benchmark_86.py score` over the committed `control_interim_ep49_test_predictions.csv` is
  the clean-clone route). What it can check
  without re-running: any copy of the crops against `crops_as_trained.sha256` (the sha256 of
  every one of the 43,392 JPEGs the arms trained on, written on klone on 2026-09-24 from the
  unpacked crops, none of whose mtime or ctime is later than the first training job's start; the
  tar's own sha256 was taken the same day, and `tests/test_context_fov_86.py` checks the listing
  against the label tables and the manifests), and a re-cut against
  the cutter's manifests, `manifest_fov.jsonl.gz` and `manifest_viewport.jsonl.gz` (committed
  gzipped, one row per crop with its geometry and sha256 as cut). **The viewport crops were
  rewritten in place** by `crop640` after the manifest was written, so the manifest's sha256 for a
  viewport crop is of the 1440×960 cut, not of what trained; for those, only
  `crops_as_trained.sha256` describes the trained-on file. Re-training needs the crops, which
  means the store or the tar; publishing the tar (6.3 GB) and the four checkpoints (about
  350 MB each) to Hugging Face would close that, and is not done here.
- **One seed per arm.** The benchmark's seed-variance is unmeasured; a difference between
  arms smaller than a seed's worth is not a result. The paired contrasts in §4 are the right
  interval for "A vs B on these test rows" (both arms scored on the same pano resample), not
  for "A's recipe vs B's recipe" (that needs seeds). The checkpoint rule (best training
  exact-match accuracy) picked epoch 81 / 77 / 41 / 93 for viewport / fov25 / fov50 / fov90,
  so the arms are not compared at one epoch either.
- **Field of view and resolution are confounded** (§2, reading 3). Every arm is resized to
  256×256, so a wider arm is also a coarser one. The separating arm is named in §4 and not run.
- **A second JPEG encode on the viewport arm.** The cutter writes JPEG at quality 92; the
  640 px box is then re-saved at quality 92. The fov arms are encoded once. The HF control is
  PNG throughout.
- **Store coverage.** Labels whose pano is not in the store (5 of the 10,853 cut,
  `cut_summary_viewport.json`: 10,848 ok, 5 `missing_pano`) drop from every arm, and from the
  control's re-score, but not from the control's training (§1).
- **The cut cost the #178 arms their run.** Cutting with 12 workers beside three trainers
  drove makelab2's load to 60 on 48 cores (12:30 UTC); the box stopped answering SSH and was
  rebooted at 15:15 UTC, which killed the three #178 arms at epoch 62–66 of 100 (no resume in
  the recipe). Cut with `WORKERS=6` or fewer when anything is training on that box; the cut
  took 48 minutes at 12 (both passes); expect roughly double at 6, and nothing else dies.
