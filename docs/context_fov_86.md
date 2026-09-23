# Context experiment: field of view vs curb-ramp tag accuracy (#86, RampNet 2.0 plan item 4)

**Status: in flight (started 2026-09-23).** Crops cut on makelab2 (12 workers, 60 min; 32,544 of
32,559 label-centred crops and 10,848 of 10,853 viewport crops, the rest `missing_pano`,
`cut_summary_*.json`); training runs on klone's `gpu-l40s` allocation once they land. Numbers below are filled in as each arm finishes.
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

*(to be filled in from `analysis_out/context_fov_86/summary.md` as the arms finish)*

## 5. Cost

Cutting: makelab2, CPU only. Training: klone `gpu-l40s`, one L40S per arm; each arm writes a
`paid: false` row to `analysis_out/usage_log.jsonl` (`train-<arm>`, host and GPU from the
job), and `scripts/analysis/slurm_usage.py` adds the `sacct` record to
`analysis_out/compute_log.jsonl`.

## 6. Not run / caveats

- **One seed per arm.** The benchmark's seed-variance is unmeasured; a difference between
  arms smaller than a seed's worth is not a result. The pano-clustered 95% CI on each arm's
  mAP is in its score file.
- **A second JPEG encode on the viewport arm.** The cutter writes JPEG at quality 92; the
  640 px box is then re-saved at quality 92. The fov arms are encoded once. The HF control is
  PNG throughout.
- **Store coverage.** Labels whose pano is not in the store (5 of 10,848 in the coverage
  probe) drop from every arm, including the control's re-score.
