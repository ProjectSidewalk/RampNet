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
the US splits.** Three-seed mean minus released, paired pano bootstrap, 95% intervals:

- **US pooled basis** (`miss_decomposition.US_SPLITS`, 9 splits, 1,078 panos): F1 −0.011
  [−0.017, −0.006] at 0.30, −0.018 [−0.024, −0.013] at 0.55, max-F1 −0.007 [−0.012, −0.002].
- **The four open-rig splits** (laurens_mapillary, bayonne, juneau, budapest_district5; 469
  panos): F1 −0.007 [−0.020, +0.005] at 0.30, −0.032 [−0.045, −0.018] at 0.55, max-F1 −0.004
  [−0.015, +0.004]. The 0.55 loss is recall: −0.034 [−0.048, −0.019].
- **manual_gold** (in-distribution, 1,000 panos): F1 +0.003 [+0.000, +0.007] at 0.30, max-F1
  +0.005 [+0.002, +0.008]. The only place the extra steps help, and only slightly.
- **laurens_mapillary**: max-F1 0.710 for the released model. All three controls are lower
  (0.664, 0.653, 0.664), and so are #82's two controls (0.668, 0.638); seed mean −0.050
  [−0.073, −0.025].
- **The Laurens rig gap widens.** On the 47 paired corners, GSV-minus-GoPro F1 grows by +0.044
  [+0.010, +0.079] at 0.30 and +0.022 [−0.002, +0.051] at max-F1. GSV training helps the GSV arm
  of Laurens (laurens_gsv +0.062 [+0.030, +0.094] F1 at 0.55) and not the GoPro arm.

This repeats the #82 finding with three new seeds, a re-staged dataset and two more splits:
extra training on GSV alone does not move RampNet toward the open rigs
([`aug_transfer_82.md`](aug_transfer_82.md), item 3 of Step 3).

**Seeds mostly move the operating point, not the ranking.** At max-F1, the three seeds agree
within 0.017 on the open-rig pool and within 0.007 on the US pool. At a fixed threshold they do
not: seed 1 sits at higher recall and lower precision than seeds 2 and 3 (US pool recall at 0.55:
0.722 against 0.636 and 0.643), so the seed-to-seed F1 range at 0.55 is 0.086 on the open-rig
pool and 0.043 on the US pool. #82 saw the same thing for its seed 1, and the rerun below shows
that this comes from the seed, not from run-to-run noise.

**What this means for the later arms.** Read each arm against the **same-seed** control,
paired, and report max-F1 next to every fixed-threshold number. A fixed-threshold difference at
0.55 between two different seeds can be as large as 0.09 F1 on the open-rig pool with no change
in recipe.

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
(`main`; the one untracked path was `logs/`), with no preemption and no resume. The final weights,
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

Three-seed mean minus released, 95% interval. Pools: **open-rig 4** is the four open-rig splits
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

Single splits of 86 to 125 panos carry intervals of ±0.02 to ±0.06; read the pools first. Per-seed
P / R / F1 for every split, both thresholds and both subsets, are in `results.md`.

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
is −0.086 [−0.108, −0.067] F1; at max-F1 the same pair is −0.007 [−0.021, +0.005].

**Same seed, run again.** #82 trained two controls with this recipe and these seeds (1 and 2) on
2026-10-02, from the earlier scrubbed copy of the dataset at an earlier commit. For this issue,
those two checkpoints were scored on bayonne and juneau, the two splits that did not exist then,
so they cover all 14 splits. Seed 1 reproduces closely: this issue's control_s1 minus #82's
control_s1 is −0.002 [−0.004, +0.001] F1 at 0.30 and +0.000 [−0.002, +0.002] max-F1 on the US
pool, and −0.005 [−0.010, +0.000] F1 at 0.30 on open-rig 4. Seed 2 reproduces less closely: +0.014
[+0.007, +0.022] F1 at 0.55 on the US pool, and +0.025 [+0.009, +0.041] on open-rig 4. #82's
control_s2 was preempted and resumed at micro-batch 4,400 of 8,000. A resume re-seeds the
DataLoader workers, so the horizontal flips in its second half differ from those of a run that
never stopped; the data order does not (aug_transfer_82.md, Step 3). This issue's seed-2 run was
not preempted. That is a plausible cause of the gap, not a tested one. The two reruns weren't bit-identical, which is expected:
training is not deterministic (AMP, no deterministic algorithms).

## Laurens rig effect (47 paired corners, GSV minus GoPro)

| checkpoint | F1 gap @0.30 | F1 gap @0.55 | max-F1 gap |
|---|---|---|---|
| released | +0.095 [+0.038, +0.151] | +0.076 [-0.005, +0.160] | +0.115 [+0.047, +0.163] |
| control_s1 | +0.149 [+0.092, +0.204] | +0.154 [+0.084, +0.231] | +0.133 [+0.073, +0.181] |
| control_s2 | +0.134 [+0.073, +0.200] | +0.110 [+0.021, +0.207] | +0.148 [+0.079, +0.198] |
| control_s3 | +0.137 [+0.076, +0.191] | +0.145 [+0.041, +0.248] | +0.128 [+0.066, +0.191] |

Change in the gap, three-seed mean minus released (difference in differences, same pair
resample): +0.044 [+0.010, +0.079] at 0.30, +0.061 [+0.015, +0.104] at 0.55, and
+0.022 [-0.002, +0.051] at max-F1. Positive means a wider gap.

## Caveats

- **The intervals resample panos, not training seeds.** The seed-mean contrast averages three
  seeds under one pano resample. Its interval does not include seed-to-seed variance. The spread
  rows estimate that variance from three runs, or five counting #82's.
- **Hardware differs by seed.** Seeds 1 and 2 trained on L40S and seed 3 on A40, and AMP numerics
  differ between them. With one run per seed this cannot be separated from the seed. Scoring used
  A40 for released, s1 and s2, and L40S for s3. The determinism check bounds cross-GPU scoring
  differences at about 1e-4 in a peak score.
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
  on klone and are not committed or read here.

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
Hugging Face repo would unblock that. Re-training gives different checkpoints, as the rerun
section shows, so the hashes and caches are the durable record.

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
