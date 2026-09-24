# Context experiment: field of view vs curb-ramp tag accuracy (#86, RampNet 2.0 plan item 4)

**Status: four arms scored (2026-09-24); the control column is INTERIM.** Crops cut on makelab2
(12 workers, 48 min; 32,544 of 32,559 label-centred crops and 10,848 of 10,853 viewport crops,
the rest `missing_pano`, `cut_summary_*.json`); the four arms trained on klone's `gpu-l40s`
allocation 2026-09-23/24 (§5). The 100-epoch #178 control died with a makelab2 reboot and is
being re-run; §4 reads its epoch-49 snapshot in its place until then (§6, first bullet).
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
(labels, recipe, split, test rows, backbone) and varies only how much of the scene the crop shows.

## 2. What the control actually sees

The HF `sidewalk-tagger-ai-validated` crops are 1440×960 screenshots of the labeling viewport,
and the tagger's `crop.py` takes a 640 px box around the label point before training. The
viewport's horizontal field of view depends on the labeler's zoom (`rampnet/crops.py`
`get_3d_fov`: 89.75° at zoom 1, 53° at zoom 2, 27.68° at zoom 3), so the box's field of view is
about **48° / 25° / 12.5°** at zoom 1 / 2 / 3 (`2·atan(320 / f)` with `f = 720 / tan(fov/2)`).
The benchmark's labels are 54% zoom 1, 28% zoom 2, 18% zoom 3 (`docs/crop_cutter.md`), so the
control is a mixture, not one field of view, and its framing is the labeler's, with the ramp
off-centre.

## 3. Arms

All arms are cut by `scripts/crop_cutter.py` (plan item 2b, PR #177) from the makelab2 pano
store, gnomonic, rendered with the viewer's tilt (the cutter's default `--tilt mm`), on the
same 10,853 labels (the benchmark's 10,857 minus 4 with no rawLabels geometry,
`docs/data/crop_cutter/coverage_input.csv`), then restricted to the labels every arm has.

| arm | framing | crop | what it isolates |
|---|---|---|---|
| control | HF screenshot, labeler's zoom, then the 640 px box | 640×640 at 12.5°–48° | the published benchmark (#178, `train_control`, 100 epochs) |
| `viewport` | the labeler's own view re-cut from the store (1440×960), then the same 640 px box (`crop640`, the tagger's arithmetic, saved at JPEG quality 92) | 640×640 at 12.5°–48° | re-cut from the archive vs the screenshot, at the control's field of view |
| `fov25` | label-centred, 25° horizontal | 640×640 | about the zoom-2 control, centred |
| `fov50` | label-centred, 50° | 640×640 | about the zoom-1 control, centred |
| `fov90` | label-centred, 90° | 640×640 | wider than any control |

Every arm trains the benchmark's recipe unchanged (`tag_benchmark_86.py train`: DINOv2-B/14
with registers, full fine-tune, Adam 1e-6, batch 4, 100 epochs, seed 86, checkpoint by best
training exact-match accuracy) on the **published HF split** restricted to the common labels
(`split_common.csv`), and is scored on the common test rows. The #178 control is re-scored on
those same rows (`control_scores.json`), so the comparison is paired on the test set. The
pano-grouped re-split is not used here: #178 found no measurable inflation from the pano leak.

## 4. Results

**Wider is worse.** On the 2,182 common test rows, mAP falls monotonically with field of view
past the control's: the label-centred 25° arm matches the control and the re-cut viewport arm,
50° loses 0.03, and 90° loses 0.07, with the loss concentrated on the tags that are visible on
the ramp itself. The street-dependent tags the experiment was designed for (*points into
traffic*, *not level with street*, *not enough landing space*) do not gain from any of the
wider crops. `analysis_out/context_fov_86/summary.md`, produced by `context_fov_86.py report
--control-scores control_interim_ep49_scores.json --control-label "…INTERIM"`:

| arm | n test | mAP | 95% CI | micro-F1 | macro-F1 | leak-free mAP | missing-tactile-warning | narrow | not-enough-landing-space | not-level-with-street | points-into-traffic | pooled-water | steep | surface-problem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control (#178 epoch-49 snapshot, INTERIM) | 2182 | 0.355 | [0.335, 0.391] | 0.659 | 0.312 | 0.381 | 0.970 | 0.259 | 0.164 | 0.171 | 0.296 | 0.294 | 0.078 | 0.610 |
| viewport | 2182 | 0.346 | [0.325, 0.381] | 0.650 | 0.330 | 0.346 | 0.968 | 0.232 | 0.168 | 0.146 | 0.310 | 0.284 | 0.099 | 0.556 |
| fov25 | 2182 | 0.349 | [0.329, 0.381] | 0.659 | 0.317 | 0.361 | 0.976 | 0.265 | 0.149 | 0.126 | 0.341 | 0.320 | 0.053 | 0.565 |
| fov50 | 2182 | 0.319 | [0.302, 0.350] | 0.636 | 0.293 | 0.331 | 0.957 | 0.236 | 0.195 | 0.110 | 0.333 | 0.198 | 0.032 | 0.494 |
| fov90 | 2182 | 0.280 | [0.266, 0.310] | 0.595 | 0.250 | 0.279 | 0.929 | 0.177 | 0.153 | 0.099 | 0.296 | 0.115 | 0.048 | 0.426 |

n = 2,182 test rows on 1,868 panoramas in every row; the CI is the pano-clustered bootstrap
(1,000 draws) on the arm's own mAP; leak-free = the 957 rows whose panorama has no train label.
*parallel-lines* and *tactile-warning* have too few positives to score and are outside the
fixed eight-tag average, as in #178.

**Paired contrasts** (`contrast_vs_control_interim_ep49.json`, `contrast_vs_viewport.json`;
`context_fov_86.py contrast`): each arm minus the reference on the same test rows *and the same
1,000 pano resamples*, which is the interval that answers "is this arm different on this test
set". Separate CIs overlap for a reason unrelated to the arms (they share every panorama).
Per-tag AP differences whose interval excludes 0 are marked.

| arm minus reference | mAP | micro-F1 | macro-F1 | per-tag AP with an interval off 0 |
|---|---:|---:|---:|---|
| viewport − control (interim) | −0.010 [−0.035, +0.017] | −0.009 [−0.023, +0.006] | +0.019 [−0.015, +0.054] | surface-problem −0.054 [−0.102, −0.009] |
| fov25 − control (interim) | −0.006 [−0.035, +0.020] | +0.001 [−0.015, +0.015] | +0.005 [−0.030, +0.037] | none (points-into-traffic +0.045 [−0.003, +0.089]) |
| fov50 − control (interim) | **−0.036 [−0.064, −0.009]** | **−0.023 [−0.039, −0.007]** | −0.019 [−0.053, +0.014] | missing-tactile −0.013, steep −0.045, surface-problem −0.116 |
| fov90 − control (interim) | **−0.075 [−0.105, −0.048]** | **−0.064 [−0.082, −0.046]** | **−0.061 [−0.097, −0.030]** | missing-tactile −0.041, narrow −0.082, pooled-water −0.179, surface-problem −0.184 |
| fov25 − viewport | +0.004 [−0.024, +0.025] | +0.009 [−0.005, +0.023] | −0.013 [−0.045, +0.018] | none |
| fov50 − viewport | **−0.026 [−0.053, −0.002]** | −0.014 [−0.030, +0.003] | **−0.038 [−0.073, −0.003]** | steep −0.067, surface-problem −0.062 |
| fov90 − viewport | **−0.065 [−0.093, −0.042]** | **−0.055 [−0.072, −0.038]** | **−0.080 [−0.114, −0.045]** | missing-tactile −0.040, narrow −0.055, pooled-water −0.169, surface-problem −0.130 |

Reading, in the order the arms were built to be read:

1. **Re-cutting from the archive costs nothing measurable.** `viewport` (the labeler's view
   rendered from the pano store, same 640 px box) is −0.010 [−0.035, +0.017] against the HF
   screenshot control on mAP, with one tag off 0 (*surface problem*, −0.054): the JPEG
   re-encode and the renderer's tilt model are not what limits the tagger. That is the
   permission plan item 5 needed: crops cut by `crop_cutter.py` for the ~354k labels without a
   production crop are interchangeable with the screenshots the benchmark was built on.
2. **Centring the ramp at the zoom-2 field of view changes nothing.** `fov25` is within
   ±0.03 of both the control and `viewport` on every headline metric, and no tag moves off 0.
   The one hint of a gain is *points into traffic*, +0.045 [−0.003, +0.089] against the
   control and +0.030 [−0.017, +0.076] against `viewport`; it needs seeds before it is a
   result.
3. **Context hurts, and it hurts the ramp-surface tags first.** At 50° the ramp occupies a
   quarter of the pixels it had at 25°, and *surface problem* (−0.116), *steep* (−0.045) and
   *missing tactile warning* (−0.013) drop with intervals off 0; at 90° every ramp-surface tag
   drops and mAP is −0.075 [−0.105, −0.048]. These are the tags that need pixels on the ramp,
   and a fixed 640 px crop trades those pixels for street. The street-dependent tags do not pay
   for the trade: *points into traffic* is flat at every field of view (+0.045 / +0.037 /
   +0.000 against the control), *not level with street* only gets worse (−0.045 / −0.061 /
   −0.072, intervals through 0), *not enough landing space* is flat (−0.015 / +0.031 /
   −0.011).

So the plan-item-4 question, "do the street-dependent tags fail for lack of context or lack of
vision", answers on the vision side, at least for context delivered this way: showing the
tagger more street at 640 px does not help any tag and costs the ones it was good at. Two
readings survive for the next experiment. Either the street-dependent tags are limited by the
labels (§2.2 of the plan: positive-unlabeled, rater-dependent), which a wider crop cannot fix
and item 5 addresses; or they need context *and* resolution together, which means a larger
input or a two-crop model, not a wider single crop. The experiment does not separate those.

## 5. Cost

All of it free: makelab2 (the lab's A40 box) and klone's `gpu-l40s-makelab` allocation. Every
number here is a committed ledger row: `analysis_out/usage_log.jsonl` (`train-<arm>`,
`infer-control-ep49-interim`, keyed by `run_id`) and `analysis_out/compute_log.jsonl` (the five
Slurm jobs, from `docs/data/compute/sacct_klone_2026-09-24.txt`, see
[`compute_cost.md`](compute_cost.md)).

| step | where | wall-clock | GPU-hours | $ |
|---|---|---:|---:|---:|
| cut, 3 fov arms (32,544 crops) + viewport (10,848) | makelab2, 12 CPU workers | 1,767 s + 1,122 s | 0 | 0 |
| env build (job 40485927) | klone, CPU | 144 s | 0 | 0 |
| `viewport` (job 40486691, g3120) | klone, 1x L40S | 4.04 h (train 3.15 h) | 4.04 | 0 |
| `fov25` (job 40486692, g3100) | klone, 1x L40S | 4.95 h (train 3.10 h, prep 0.73 h) | 4.95 | 0 |
| `fov50` (job 40486693, g3100) | klone, 1x L40S | 3.98 h (train 3.14 h, prep 0.45 h) | 3.98 | 0 |
| `fov90` (job 40486694, g3104) | klone, 1x L40S | 3.57 h (train 3.13 h) | 3.57 | 0 |
| interim control inference (epoch-49 snapshot, 10,857 crops) | makelab2 A40, shared 4 ways | 476 s | 0.13 | 0 |
| **total** | | | **16.7** | **0** |

Training itself is the same 3.1 h on every arm (100 epochs at 111–116 s on a dedicated
L40S, `train_<arm>_train_meta.json`); the spread in job wall-clock is the one-off crop
preparation on g3100 (fov25 2,634 s, fov50 1,603 s, against 349–369 s on the other two nodes;
first read of the crops from `/gscratch`), then inference and scoring. One L40S was free on the
allocation, so the arms ran one after another (14:20 UTC to 06:53 UTC the next day); on the
#178 recipe's makelab2 A40 the same four arms would have shared one GPU at ~590 s/epoch.

## 6. Not run / caveats

- **The control row is interim.** The 100-epoch #178 control (`train_control`) died at epoch
  index 66 with the makelab2 reboot of 2026-09-23 15:15 UTC (recorded on #178). The row in §4
  is that dead run's epoch-index-49 checkpoint (`best_after_ep49.pth`, sha256 `f22a2f09…`;
  `control_interim_ep49_*`), inferred on 2026-09-24 and filtered to the common test rows.
  Whether the last 50 epochs of the recipe move the control's test mAP is exactly what the
  re-run will show; until then the control column is a lower-epoch read, not the benchmark of
  record. The re-run started 2026-09-24 04:09 UTC and finishes about 21:00 UTC; then
  `tag_benchmark_86.sh finish` on #178, and here:

  ```bash
  bash scripts/analysis/context_fov_86.sh control report
  python scripts/analysis/context_fov_86.py contrast --reference control \
      --reference-pred analysis_out/context_fov_86/control_test_predictions.csv \
      --out analysis_out/context_fov_86/contrast_vs_control.json
  ```

  (`control` writes `control_scores.json`; the `contrast` needs the control's common test
  rows, `tag_benchmark_86.py test-only` over #178's `train_control_final_test_predictions.csv`
  with `--split-csv split_common.csv`, the same filter the arms went through, written to
  `control_test_predictions.csv`.) Replace the §4 table and the
  contrast column, keep the interim files as the record of what was read first.
- **One seed per arm.** The benchmark's seed-variance is unmeasured; a difference between
  arms smaller than a seed's worth is not a result. The paired contrasts in §4 are the right
  interval for "A vs B on these test rows" (both arms scored on the same pano resample), not
  for "A's recipe vs B's recipe" (that needs seeds). The checkpoint rule (best training
  exact-match accuracy) picked epoch 81 / 77 / 41 / 93 for viewport / fov25 / fov50 / fov90,
  so the arms are not compared at one epoch either.
- **A second JPEG encode on the viewport arm.** The cutter writes JPEG at quality 92; the
  640 px box is then re-saved at quality 92. The fov arms are encoded once. The HF control is
  PNG throughout.
- **Store coverage.** Labels whose pano is not in the store (5 of 10,848 in the coverage
  probe) drop from every arm, including the control's re-score.
- **The cut cost the #178 arms their run.** Cutting with 12 workers beside three trainers
  drove makelab2's load to 60 on 48 cores (12:30 UTC); the box stopped answering SSH and was
  rebooted at 15:15 UTC, which killed the three #178 arms at epoch 62–66 of 100 (no resume in
  the recipe). Cut with `WORKERS=6` or fewer when anything is training on that box; the cut
  took 48 minutes at 12 (both passes); expect roughly double at 6, and nothing else dies.
