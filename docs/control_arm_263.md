# Control arm: released weights plus 2,000 GSV steps, three seeds (#263, Track 3.4)

**Status:** scored 2026-10-22. Plan: Track 3 of
[#263](https://github.com/ProjectSidewalk/RampNet/issues/263). Run and written up by an Opus 5.5
agent.

Scripts: `stage_two/run_finetune_control.slurm` (training), `scripts/analysis/control263_score.slurm`
(scoring; a wrapper around the #82 scorer `scripts/analysis/aug82_score_ckpts.sh`),
`scripts/analysis/control_arm_263.py` (the read). Test: `tests/test_control_arm_263.py`. Every
number below is in `analysis_out/control_arm_263/results.json` and `results.md`, which
`control_arm_263.py` writes from the committed caches in `analysis_out/control_arm_263/`.

## What this arm is for

Every later #263 arm (Lyon mixes, masked loss) starts from the released RampNet weights and trains
on GSV replay plus something else. This arm trains on the GSV replay alone, with the same budget:
2,000 optimizer steps at global batch 16 (32,000 panos, about a fifth of an epoch), the recipe's
constant LR 1e-5, horizontal flip, no augmentation, no `--loss-mask`. It answers two questions
before any other arm is read: what do the extra GSV steps alone do to the released model, and how
far apart do three seeds of the same recipe land.

## Answer

**The extra GSV steps do not improve the released model off GSV, and they cost about 0.01 F1 on
the US splits at 0.30.** Three-seed mean minus released, paired pano bootstrap, 95% intervals,
with each seed's own contrast where the seeds disagree. **The seed-mean interval resamples panos
only and leaves out seed-to-seed variance**, so where the seeds disagree in sign it says nothing
about the recipe.

- **US pooled basis** (`miss_decomposition.US_SPLITS`, 9 splits, 1,078 panos): F1 −0.011
  [−0.017, −0.006] at 0.30. This one is seed-consistent: s1 −0.013 [−0.021, −0.006], s2 −0.009
  [−0.015, −0.002], s3 −0.012 [−0.018, −0.006]. max-F1 −0.007 [−0.012, −0.002] (s1 −0.006, s2
  −0.004 [−0.010, +0.002], s3 −0.011). At 0.55 the seeds disagree in sign (seed mean −0.018
  [−0.024, −0.013]; s1 +0.009 [+0.002, +0.017], s2 −0.034 [−0.042, −0.025], s3 −0.030
  [−0.039, −0.023]), so there is no 0.55 result for the recipe here.
- **The four open-rig splits** (laurens_mapillary, bayonne, juneau, budapest_district5; 469
  panos): F1 −0.007 [−0.020, +0.005] at 0.30 and max-F1 −0.004 [−0.015, +0.004], neither
  resolved. At 0.55 the seeds disagree in sign: seed mean −0.032 [−0.045, −0.018], but s1
  +0.020 [+0.004, +0.037], s2 −0.048 [−0.066, −0.029], s3 −0.066 [−0.085, −0.049]. The
  seed-to-seed F1 range at 0.55 (0.086) is about six times the seed-mean interval's half-width,
  so **the 0.55 seed mean is not a loss of the recipe**; it is where three operating points
  happened to average.
- **manual_gold** (in-distribution, 1,000 panos): F1 +0.003 [+0.000, +0.007] at 0.30, max-F1
  +0.005 [+0.002, +0.008] (s1 +0.005, s2 +0.007, s3 +0.004 [−0.001, +0.007]). The only place the
  extra steps help, and only slightly.
- **laurens_mapillary**: max-F1 0.710 for the released model. All three controls are lower
  (0.664, 0.653, 0.664), each with an interval below zero, and so are #82's two controls
  (0.668, 0.638); seed mean −0.050 [−0.073, −0.025]. This is seed-consistent.
- **The Laurens rig gap widens, and it widens from the GoPro side.** On the 47 paired corners,
  GSV-minus-GoPro F1 grows by +0.044 [+0.010, +0.079] at 0.30. At max-F1 the change is +0.022
  [−0.002, +0.051], and the interval includes zero. At max-F1 the GSV arm does not move
  (laurens_gsv +0.004 [−0.013, +0.025]), while the GoPro arm drops (laurens_mapillary −0.050
  [−0.073, −0.025]). laurens_gsv's +0.062 [+0.030, +0.094] at 0.55 is an operating-point effect:
  +0.139 for seed 1, +0.001 for seed 2, +0.045 for seed 3, and +0.008 [−0.018, +0.034] at 0.30.

**What replicates #82, and what does not.** These are three new *runs*, not three new seeds:
seeds 1 and 2 are #82's seeds again and seed 3 is new. They used a re-staged dataset and add two
splits. The laurens_mapillary max-F1 drop replicates (−0.050 here, −0.057 [−0.085, −0.034] in
#82). So does the loss of a little F1 on the US splits from the extra training alone
([`aug_transfer_82.md`](aug_transfer_82.md), item 3 of Step 3). The rig-gap widening replicates
at 0.30 (+0.044 here, +0.046 [+0.009, +0.084] in #82) but only weakly at max-F1, #82's headline
read: +0.028 [+0.003, +0.064] in #82 and +0.022 [−0.002, +0.051] here, an interval that now
includes zero.

**Seeds mostly move the operating point; they also differ a little in ranking.** At max-F1 the
three seeds are within 0.017 of each other on the open-rig pool and 0.007 on the US pool. But one
pair is resolved even there: seed 3 minus seed 2 is −0.017 [−0.028, −0.004] on open-rig 4 and
−0.007 [−0.012, −0.000] on the US pool. At a fixed threshold the seeds differ much more. Seed 1
sits at higher recall and lower precision than seeds 2 and 3 (US pool recall at 0.55: 0.722,
against 0.636 and 0.643), so the seed-to-seed F1 range at 0.55 is 0.086 on the open-rig pool and
0.043 on the US pool. #82's seed 1 sat at the same high-recall operating point. The rerun section
below suggests that the operating point follows the seed more than run-to-run noise, but that
read rests on one pair of reruns that also differ in dataset copy, commit and GPU.

**Reading rule for the later arms (proposed, not decided).** Read each arm against the
**same-seed** control, paired, and report max-F1 and each seed's own contrast next to every
fixed-threshold number. Two seeds of the unchanged recipe already differ by up to 0.09 F1 at 0.55
on the open-rig pool.

## Inputs

| what | where | sha256 |
|---|---|---|
| released weights (init) | `/gscratch/scrubbed/jfroehli/aug82/released_rampnet_state_dict.pth`, converted from the Hub `projectsidewalk/rampnet-model` revision `606a119` `model.safetensors` (blob `f2119e3b…`) | `024a987c…` (prefix printed by each training job) |
| GSV training data | `/gscratch/scrubbed/jfroehli/rampnet_dataset_v2`, 150,063 train / 42,875 val pairs, from Hub `projectsidewalk/rampnet-dataset` revision `ee882e3f` ([`restage_263.md`](restage_263.md)) | per-file, see that doc |
| control_s1 | `/gscratch/makelab/jonf/control263/s1/final_step_2000.pth` (job 42108716, 4 × L40S) | `7c6f4920a5e34f32e9d50742a00a2d5cdff575fc74f155e077dfdd64b9494eb5` |
| control_s2 | `/gscratch/makelab/jonf/control263/s2/final_step_2000.pth` (job 42108717, 4 × L40S) | `bbb353f39bd65c095a52e2f19c643d3e6118753c800bb9e01dba443b0158b03b` |
| control_s3 | `/gscratch/makelab/jonf/control263/s3/final_step_2000.pth` (job 42108718, 4 × A40) | `cabd583033f4438c0636f06deb5997ffd5088dc0f9b126f305410ce15f99b523` |
| #82 controls (rerun read) | `/gscratch/makelab/jonf/aug82/control_s{1,2}.pth` | `d027a41d…`, `2b8ed5c5…` (full hashes in `/gscratch/makelab/jonf/aug82/SHA256SUMS`) |
| benchmark panos | `/gscratch/makelab/jonf/rampnet_benchmark/<split>/panos` on klone | per split, `benchmark/<split>/imagery_manifest.json` |

All three training jobs ran from `/gscratch/makelab/jonf/RampNet263` at commit `20ea575`
(`main`; the job log reports `dirty files: 1`, and `git status` in that checkout shows `logs/` as
its only untracked path), with `restarts: 0`, so no preemption and no resume. The six training
logs are committed under `analysis_out/control_arm_263/trainlogs/`, following #82's layout: each
job's full `.out` (seed, commit, data root, init hash, GPUs, node, the copy line with the
checkpoint sha256), the first 60 lines of its `.err` (for the three g3110 jobs, the whole ECC
traceback), and one progress line every 200 micro-batches. `ckpt_SHA256SUMS.txt` holds the three
checkpoint hashes. The final weights,
their `.sha256` and the TensorBoard events were copied to `/gscratch/makelab/jonf/control263/s<seed>/`.
`/gscratch/makelab/jonf/control263/control_s<seed>.pth` are symlinks to those files, with a
`SHA256SUMS` beside them that the scorer checks before it loads anything.

## Scoring

The scorer is the #82 screen's, unchanged apart from a `LABEL_PREFIX` variable that names the
cache's `meta.model`: `operating_point_curve.py extract [--checkpoint …] --score-floor 0.05
--min-distance 10`, no TTA, fp32 (no cache fell back to fp16), one cache directory per
checkpoint. The released checkpoint was scored again in the same pass (no `--checkpoint`, so the
Hub weights), and every contrast is checkpoint against checkpoint through that one code path.

**Splits: all 14 reviewed splits.** annapolis, bayonne, bend, budapest_district5, clovis,
gainesville, juneau, laurens_gsv, laurens_mapillary, manual_gold, morgantown, paterson, richmond,
sao_paulo. Every cache holds exactly the pano set of the committed `analysis_out/op_cache/<split>.json`
(tested). **Not scored: `benchmark/lyon`.** It has no reviewed ground truth yet; its review is
open on [#159](https://github.com/ProjectSidewalk/RampNet/issues/159). It is a training-only
candidate city, so it was never one of the 14.

**Statistics** are `aug_finetune_82.py`'s, imported rather than re-implemented: P / R / F1 at 0.30
and 0.55, AP, max-F1 (best F1 over every threshold), pano-level paired cluster bootstrap
stratified by split, 2,000 draws, rng seed 82. The seed-mean contrast applies the same pano
resample to all three controls and the released checkpoint. Pools are micro-averaged over their
splits, resampled within each split.

**Determinism.** The released re-score is bit-identical to #82's klone caches of the released
checkpoint on all 12 splits both scored (the same peaks, to the bit, on every pano; tested).
Against the shared `analysis_out/op_cache/`, which was made on makelab2 and mostly before the
f4c71c8 extractor fix, the peak count differs on some panos. On the 13 splits other than
manual_gold, peak values differ by at most 7e-5 on panos with the same peak count. On manual_gold
some peak sets differ outright (largest difference 0.48 on a pano with the same count). Nothing
here reads `op_cache` for a number; it is used only to check pano coverage.

## Pooled results, all panos

Three-seed mean minus released, 95% interval (panos resampled, seed variance not included; the
**dF1 @0.55 and dR @0.55 columns are not seed-consistent**, see the per-seed table below). Pools:
**open-rig 4** is the four open-rig splits
#263 names. **open-rig 3** leaves out budapest_district5, which `miss_decomposition.HELD_OUT` and
[`open_rig_misses_258.md`](open_rig_misses_258.md) never pool because its GT is single-rater at
low confidence. **GSV tier** is bend, paterson and gainesville. **US pooled basis** is
`miss_decomposition.US_SPLITS`, six Mapillary and three GSV splits, juneau and laurens_mapillary
included.

| pool | panos | dF1 @0.30 | dF1 @0.55 | d max-F1 | dR @0.30 | dR @0.55 |
|---|---:|---|---|---|---|---|
| open-rig 4 | 469 | -0.007 [-0.020, +0.005] | -0.032 [-0.045, -0.018] | -0.004 [-0.015, +0.004] | -0.002 [-0.015, +0.011] | -0.034 [-0.048, -0.019] |
| open-rig 3 (no budapest) | 344 | -0.005 [-0.020, +0.010] | -0.023 [-0.039, -0.007] | -0.012 [-0.021, +0.001] | +0.005 [-0.011, +0.023] | -0.028 [-0.045, -0.011] |
| GSV tier | 360 | -0.013 [-0.021, -0.007] | -0.014 [-0.023, -0.006] | -0.008 [-0.016, -0.003] | -0.008 [-0.017, +0.001] | -0.010 [-0.022, +0.001] |
| US pooled basis (9 splits) | 1078 | -0.011 [-0.017, -0.006] | -0.018 [-0.024, -0.013] | -0.007 [-0.012, -0.002] | -0.007 [-0.014, -0.000] | -0.021 [-0.029, -0.013] |
| manual_gold | 1000 | +0.003 [+0.000, +0.007] | +0.003 [+0.000, +0.007] | +0.005 [+0.002, +0.008] | +0.004 [-0.000, +0.008] | +0.007 [+0.002, +0.012] |

Released pooled F1 at 0.30 / max-F1, for scale: open-rig 4 0.691 / 0.696, open-rig 3 0.703 /
0.727, GSV tier 0.835 / 0.835, US pool 0.825 / 0.826, manual_gold 0.903 / 0.907.

**Each seed against released.** The seed-mean intervals above leave out seed-to-seed variance.
This table puts each seed's own paired contrast next to the seed mean. A read is seed-consistent
only where the three seeds agree in sign. **At 0.55 they do not.** Seed 1 is above released on
every pool, resolved on open-rig 4, open-rig 3 and the US pool. Seeds 2 and 3 are below it on
every pool except manual_gold.

| pool | read | seed mean | s1 | s2 | s3 |
|---|---|---|---|---|---|
| open-rig 4 | F1 @0.30 | -0.007 [-0.020, +0.005] | -0.005 [-0.021, +0.012] | -0.001 [-0.016, +0.015] | -0.017 [-0.030, -0.004] |
| open-rig 4 | F1 @0.55 | -0.032 [-0.045, -0.018] | +0.020 [+0.004, +0.037] | -0.048 [-0.066, -0.029] | -0.066 [-0.085, -0.049] |
| open-rig 4 | max-F1 | -0.004 [-0.015, +0.004] | -0.005 [-0.018, +0.006] | +0.004 [-0.009, +0.014] | -0.012 [-0.025, -0.001] |
| open-rig 3 (no budapest) | F1 @0.30 | -0.005 [-0.020, +0.010] | +0.006 [-0.013, +0.026] | -0.004 [-0.022, +0.015] | -0.017 [-0.033, -0.001] |
| open-rig 3 (no budapest) | F1 @0.55 | -0.023 [-0.039, -0.007] | +0.030 [+0.011, +0.049] | -0.045 [-0.068, -0.024] | -0.054 [-0.074, -0.034] |
| open-rig 3 (no budapest) | max-F1 | -0.012 [-0.021, +0.001] | -0.011 [-0.023, +0.003] | -0.008 [-0.019, +0.007] | -0.016 [-0.028, -0.001] |
| GSV tier | F1 @0.30 | -0.013 [-0.021, -0.007] | -0.017 [-0.026, -0.006] | -0.011 [-0.020, -0.002] | -0.013 [-0.023, -0.004] |
| GSV tier | F1 @0.55 | -0.014 [-0.023, -0.006] | +0.004 [-0.006, +0.015] | -0.016 [-0.027, -0.006] | -0.030 [-0.043, -0.019] |
| GSV tier | max-F1 | -0.008 [-0.016, -0.003] | -0.005 [-0.015, +0.001] | -0.007 [-0.015, -0.000] | -0.013 [-0.022, -0.004] |
| US pooled basis (9 splits) | F1 @0.30 | -0.011 [-0.017, -0.006] | -0.013 [-0.021, -0.006] | -0.009 [-0.015, -0.002] | -0.012 [-0.018, -0.006] |
| US pooled basis (9 splits) | F1 @0.55 | -0.018 [-0.024, -0.013] | +0.009 [+0.002, +0.017] | -0.034 [-0.042, -0.025] | -0.030 [-0.039, -0.023] |
| US pooled basis (9 splits) | max-F1 | -0.007 [-0.012, -0.002] | -0.006 [-0.012, -0.001] | -0.004 [-0.010, +0.002] | -0.011 [-0.016, -0.004] |
| manual_gold | F1 @0.30 | +0.003 [+0.000, +0.007] | -0.002 [-0.006, +0.002] | +0.009 [+0.005, +0.013] | +0.003 [-0.001, +0.008] |
| manual_gold | F1 @0.55 | +0.003 [+0.000, +0.007] | +0.009 [+0.005, +0.014] | -0.000 [-0.004, +0.004] | +0.001 [-0.004, +0.004] |
| manual_gold | max-F1 | +0.005 [+0.002, +0.008] | +0.005 [+0.001, +0.009] | +0.007 [+0.003, +0.010] | +0.004 [-0.001, +0.007] |

Seed-consistent reads: GSV tier and US pool at 0.30 (all three seeds below released, each interval
below zero). At max-F1 on the same two pools all three point estimates are below zero, but some
intervals include it.
The open-rig pools at 0.30 and max-F1 are unresolved for two of three seeds. The unbiased subset
gives the same signs (`results.json`, `pooled.unbiased.*.contrasts`).

**Unbiased subset.** The same table without each split's five hand-picked `top` panos
(`verdicts.json` `group == "top"`), the convention `score_validation.py` and #258 use. No reading
changes:

| pool | panos | dF1 @0.30 | dF1 @0.55 | d max-F1 |
|---|---:|---|---|---|
| open-rig 4 | 449 | -0.011 [-0.025, +0.003] | -0.032 [-0.048, -0.017] | -0.006 [-0.018, +0.004] |
| open-rig 3 (no budapest) | 329 | -0.009 [-0.025, +0.008] | -0.025 [-0.043, -0.006] | -0.013 [-0.024, +0.000] |
| GSV tier | 345 | -0.015 [-0.024, -0.008] | -0.015 [-0.025, -0.006] | -0.010 [-0.017, -0.003] |
| US pooled basis (9 splits) | 1033 | -0.014 [-0.020, -0.008] | -0.019 [-0.026, -0.013] | -0.008 [-0.013, -0.002] |
| manual_gold | 1000 | +0.003 [+0.000, +0.007] | +0.003 [+0.000, +0.007] | +0.005 [+0.002, +0.008] |

## Per split, all panos

| split | panos | released P / R / F1 @0.30 | dF1 @0.30 | dF1 @0.55 | released max-F1 | d max-F1 |
|---|---:|---|---|---|---:|---|
| annapolis | 125 | 0.895 / 0.809 / 0.850 | +0.011 [-0.003, +0.025] | -0.019 [-0.036, -0.003] | 0.853 | +0.017 [+0.001, +0.026] |
| bayonne | 125 | 0.754 / 0.498 / 0.600 | +0.021 [-0.008, +0.050] | -0.023 [-0.057, +0.011] | 0.649 | +0.005 [-0.008, +0.026] |
| bend | 110 | 0.915 / 0.829 / 0.870 | -0.008 [-0.019, +0.003] | -0.007 [-0.020, +0.005] | 0.873 | -0.001 [-0.013, +0.009] |
| budapest_district5 | 125 | 0.680 / 0.643 / 0.661 | -0.014 [-0.033, +0.005] | -0.055 [-0.082, -0.030] | 0.676 | -0.021 [-0.041, -0.003] |
| clovis | 125 | 0.838 / 0.821 / 0.829 | -0.011 [-0.034, +0.010] | -0.021 [-0.047, +0.007] | 0.837 | +0.003 [-0.014, +0.019] |
| gainesville | 125 | 0.854 / 0.772 / 0.811 | -0.015 [-0.032, +0.001] | -0.021 [-0.042, -0.002] | 0.821 | -0.014 [-0.033, +0.000] |
| juneau | 125 | 0.911 / 0.761 / 0.829 | -0.010 [-0.026, +0.007] | -0.038 [-0.061, -0.019] | 0.831 | +0.001 [-0.013, +0.011] |
| laurens_gsv | 86 | 0.923 / 0.654 / 0.766 | +0.008 [-0.018, +0.034] | +0.062 [+0.030, +0.094] | 0.812 | +0.004 [-0.013, +0.025] |
| laurens_mapillary | 94 | 0.885 / 0.526 / 0.660 | -0.024 [-0.056, +0.005] | -0.002 [-0.032, +0.027] | 0.710 | -0.050 [-0.073, -0.025] |
| manual_gold | 1000 | 0.909 / 0.897 / 0.903 | +0.003 [+0.000, +0.007] | +0.003 [+0.000, +0.007] | 0.907 | +0.005 [+0.002, +0.008] |
| morgantown | 125 | 0.878 / 0.813 / 0.844 | -0.019 [-0.035, -0.004] | -0.044 [-0.067, -0.024] | 0.854 | -0.015 [-0.030, -0.003] |
| paterson | 125 | 0.947 / 0.724 / 0.821 | -0.017 [-0.028, -0.007] | -0.016 [-0.028, -0.004] | 0.823 | -0.015 [-0.024, -0.005] |
| richmond | 124 | 0.893 / 0.832 / 0.861 | -0.007 [-0.020, +0.006] | -0.003 [-0.014, +0.010] | 0.870 | -0.000 [-0.010, +0.011] |
| sao_paulo | 125 | 0.803 / 0.797 / 0.800 | -0.004 [-0.019, +0.013] | +0.010 [-0.012, +0.034] | 0.804 | +0.002 [-0.009, +0.021] |

Single splits of 86 to 125 panos carry intervals of ±0.02 to ±0.06; read the pools first. The
dF1 @0.55 column is a seed mean. At 0.55 the seeds disagree in sign on 8 of the 14 splits
(laurens_mapillary: s1 +0.045, s2 −0.030, s3 −0.020), and where they agree the size varies
(laurens_gsv: s1 +0.139, s2 +0.001, s3 +0.045). Per-seed
P / R / F1 and per-seed contrasts for every split, both thresholds and both subsets, are in
`results.md` / `results.json`.

## Seed spread

Each run's pooled value on the observed sample (no resample). "#82 s1/s2" are the #82 screen's
two controls (below).

| pool | read | released | s1 | s2 | s3 | #82 s1 | #82 s2 |
|---|---|---:|---:|---:|---:|---:|---:|
| open-rig 4 | F1 @0.30 | 0.691 | 0.686 | 0.691 | 0.674 | 0.691 | 0.681 |
| open-rig 4 | F1 @0.55 | 0.638 | 0.658 | 0.590 | 0.571 | 0.652 | 0.565 |
| open-rig 4 | R @0.55 | 0.495 | 0.537 | 0.432 | 0.415 | 0.529 | 0.406 |
| open-rig 4 | max-F1 | 0.696 | 0.691 | 0.700 | 0.684 | 0.693 | 0.687 |
| US pooled basis | F1 @0.30 | 0.825 | 0.812 | 0.817 | 0.813 | 0.813 | 0.816 |
| US pooled basis | F1 @0.55 | 0.801 | 0.810 | 0.767 | 0.770 | 0.810 | 0.753 |
| US pooled basis | R @0.55 | 0.688 | 0.722 | 0.636 | 0.643 | 0.723 | 0.617 |
| US pooled basis | max-F1 | 0.826 | 0.820 | 0.822 | 0.815 | 0.820 | 0.818 |
| manual_gold | F1 @0.30 | 0.903 | 0.901 | 0.912 | 0.907 | 0.902 | 0.911 |
| manual_gold | max-F1 | 0.907 | 0.912 | 0.913 | 0.910 | 0.912 | 0.912 |

The three pairwise seed contrasts (`spread: control_s<b> - control_s<a>`) with paired intervals,
for every pool at both thresholds, are in `results.md`. At 0.55 on open-rig 4, seed 3 minus seed 1
is −0.086 [−0.108, −0.067] F1; at max-F1 the same pair is −0.007 [−0.021, +0.005]. One
max-F1 pair is resolved: seed 3 minus seed 2 is −0.017 [−0.028, −0.004] on open-rig 4 and −0.007
[−0.012, −0.000] on the US pool, so the seeds also differ a little in ranking, not only in
operating point.

**Same seed, run again (confounded).** #82 trained two controls with this recipe and seeds 1 and
2 on 2026-10-02. For this issue, those two checkpoints were scored on bayonne and juneau, the two
splits that did not exist then, so they cover all 14 splits. Comparing them with this issue's
seed-1 and seed-2 runs is **not a pure determinism test**. The pairs differ in more than the run:

- **Dataset copy.** #82 trained on the earlier scrubbed `rampnet_dataset`, whose byte provenance
  was never checked ([`aug_transfer_82.md`](aug_transfer_82.md), Reproducing). This arm trained on
  `rampnet_dataset_v2`, the Hub parquet's own jpg bytes ([`restage_263.md`](restage_263.md)). Pano
  order is the same, because `train.py` sorts the file list.
- **Commit.** #82's control_s1 ran at `6e05943` and its control_s2 at `81d3afa`. This arm ran at
  `20ea575`.
- **GPU.** #82's control_s1 ran on 4 × A40 (job 41123606); this seed 1 ran on 4 × L40S. #82's
  control_s2 ran on L40S in its first incarnation and on L40 in its last (job 41138715); this seed
  2 ran on 4 × L40S (the #82 logs are in `analysis_out/aug_transfer_82/finetune/trainlogs/`).
- **Resume.** #82's control_s2 was preempted and resumed at micro-batch 4,400 of 8,000 (three
  incarnations). A resume re-seeds the DataLoader workers, so the horizontal flips in its second
  half differ from those of a run that never stopped. This issue's seed-2 run was not preempted.

With those differences, seed 1 still lands close to #82's seed 1. On the US pool, this
control_s1 minus #82's is −0.002 [−0.004, +0.001] F1 at 0.30 and +0.000 [−0.002, +0.002]
max-F1. On open-rig 4 it is −0.005 [−0.010, +0.000] at 0.30, but at 0.55 the interval excludes
zero: +0.006 [+0.001, +0.011]. Seed 2 differs more. On the US pool the gap is +0.014 [+0.007,
+0.022] F1 at 0.55. On open-rig 4 it is +0.025 [+0.009, +0.041] at 0.55 and +0.013 [+0.002,
+0.021] at max-F1, about a third of the seed-to-seed range at 0.55. Any of the four differences
could cause the seed-2 gap; none was tested. Two pairs of reruns *suggest* that the seed sets the
operating point more than run-to-run variation does. They do not show that run-to-run variation
is negligible. The reruns are not bit-identical, which is expected: training is not deterministic
(AMP, no deterministic algorithms).

## Laurens rig effect (47 paired corners, GSV minus GoPro)

| checkpoint | F1 gap @0.30 | F1 gap @0.55 | max-F1 gap |
|---|---|---|---|
| released | +0.095 [+0.038, +0.151] | +0.076 [-0.005, +0.160] | +0.115 [+0.047, +0.163] |
| control_s1 | +0.149 [+0.092, +0.204] | +0.154 [+0.084, +0.231] | +0.133 [+0.073, +0.181] |
| control_s2 | +0.134 [+0.073, +0.200] | +0.110 [+0.021, +0.207] | +0.148 [+0.079, +0.198] |
| control_s3 | +0.137 [+0.076, +0.191] | +0.145 [+0.041, +0.248] | +0.128 [+0.066, +0.191] |

Change in the gap, three-seed mean minus released (difference in differences, same pair
resample): +0.044 [+0.010, +0.079] at 0.30, +0.061 [+0.015, +0.104] at 0.55, and
+0.022 [-0.002, +0.051] at max-F1 (the interval includes zero). Positive means a wider gap.
At max-F1 the widening comes from the GoPro arm: laurens_mapillary −0.050 [−0.073, −0.025],
laurens_gsv +0.004 [−0.013, +0.025]. #82's seed-mean change at max-F1 was +0.028 [+0.003,
+0.064].

## Caveats

- **The intervals resample panos, not training seeds.** The seed-mean contrast averages three
  seeds under one pano resample. Its interval does not include seed-to-seed variance, and at 0.55
  that changes the reading: the seed mean excludes zero while seed 1 is significantly on the
  other side (per-seed table). The spread rows estimate seed variance from three runs, or five
  counting #82's, and #82's two are confounded reruns.
- **Hardware differs by seed.** Seeds 1 and 2 trained on L40S and seed 3 on A40, and AMP numerics
  differ between them. With one run per seed this cannot be separated from the seed. Scoring used
  A40 for released, s1 and s2, and L40S for s3. **The cross-GPU-type scoring difference was not
  measured.** Both determinism references (#82's klone released caches and makelab2's op_cache)
  were scored on A40s, like this released re-score, and no comparison here crosses GPU types. An
  arm scored on a different GPU type than its control could flip detections that sit near a
  threshold. Scoring both on the same type, or measuring the difference on one split, would close
  this.
- **GT quality varies across the open-rig splits.** budapest_district5 is single-rater at low
  confidence, and bayonne is single-rater at medium confidence. open-rig 3 leaves budapest out and
  gives the same reading.
- **The `top` stratum** in each benchmark split is hand-picked dense panos. The unbiased-subset
  table drops them and gives the same reading.
- **2,000 steps is one point.** Nothing here says what 5,000 or 20,000 GSV steps would do. The
  Lyon arms in #263 use the same 2,000-step budget, so this is the point they are read against.

## Not run, and why

- **lyon**: no reviewed GT (review open on #159). It is not one of the 14 splits.
- **The #258 miss buckets on the controls**: not part of this arm. The caches have peaks down to
  0.05, so `miss_taxonomy.py` can read them later on CPU.
- **TTA, distance-binned recall, and the seed-variance campaign's AP bootstrap** were not run.
  The plan asks for the fixed-threshold read, and #82's protocol is the precedent.
- **Training curves**: the TensorBoard events are in `/gscratch/makelab/jonf/control263/s*/runs`
  on klone and are not committed or read here. The committed `trainlogs/*.progress.txt` hold the
  running loss every 200 micro-batches.

## Compute

All free (klone `ckpt-all`): **37.29 GPU-hours**. Training was 34.20 GPU-h. That includes 3.48
GPU-h for the first three submissions, which failed on node `g3110` with an uncorrectable ECC
error; seed 1 had first spent 50 min building the env tarball. Scoring was 3.10 GPU-h: four
14-split jobs of 32 to 49 min on one GPU, plus 9 min for the #82 controls' two splits. Rows are in
`analysis_out/compute_log.jsonl` from the committed `sacct -D` dump
`docs/data/compute/sacct_klone_2026-10-22_control263.txt`. The per-job table and the exact pull
command are in [`compute_cost.md`](compute_cost.md) (klone, 2026-10-22).

## Reproducing

**Not published, stated beside the numbers.** The three checkpoints exist only at
`/gscratch/makelab/jonf/control263/` on klone (3 × 344 MB, sha256 above). Every number in this doc
re-derives on CPU from the committed caches with `control_arm_263.py --check`. Re-scoring the
checkpoints, or scoring them on a new split, needs those files. Pushing the three files to a
Hugging Face repo would unblock that. Re-training gives different checkpoints (the
same-seed reruns' hashes differ from #82's), so the hashes and caches are the durable record.

```bash
# --- training (klone, from a checkout of main at 20ea575 or later; see the launcher header) ---
mkdir -p logs
for s in 1 2 3; do
    RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2 SEED=$s HF_HUB_OFFLINE=1 \
        sbatch --exclude=g3110 stage_two/run_finetune_control.slurm
done

# --- scoring (klone, from a checkout of this branch with logs/ and the pano links) ---
for c in annapolis bayonne bend budapest_district5 clovis gainesville juneau laurens_gsv \
         laurens_mapillary manual_gold morgantown paterson richmond sao_paulo; do
    ln -s /gscratch/makelab/jonf/rampnet_benchmark/$c/panos benchmark/$c/panos
done
cd /gscratch/makelab/jonf/control263
for s in 1 2 3; do ln -sfn s$s/final_step_2000.pth control_s$s.pth; done
for s in 1 2 3; do echo "$(cut -d' ' -f1 s$s/final_step_2000.pth.sha256)  control_s$s.pth"; done > SHA256SUMS
cd -
for L in released control_s1 control_s2 control_s3; do
    RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2 LABELS=$L \
        sbatch scripts/analysis/control263_score.slurm
done
RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2 LABELS="control_s1 control_s2" \
    CKPTS=/gscratch/makelab/jonf/aug82 LABEL_PREFIX=aug82_ SPLITS=bayonne,juneau \
    OUT=$PWD/analysis_out/control_arm_263/caches_aug82_controls \
    sbatch scripts/analysis/control263_score.slurm

# --- copy analysis_out/control_arm_263/caches*/ back, then on CPU (about 5 min) ---
python scripts/analysis/control_arm_263.py            # -> results.json / results.md
python scripts/analysis/control_arm_263.py --check
pytest -q tests/test_control_arm_263.py
```

The training launcher was submitted with `HF_HUB_OFFLINE=1` because the klone home directory is
over its 10 GB quota and the backbone is read from the home HF cache. Offline mode keeps the hub
client from writing there. The scoring launcher sets it itself.
