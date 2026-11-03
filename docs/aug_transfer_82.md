# Augmentation as a rig-transfer lever: frozen probe and paired fine-tune screen (#82)

**Status:** complete (PR #235); the answer is no. Plan: the 2026-10-02 comment on
[#82](https://github.com/ProjectSidewalk/RampNet/issues/82#issuecomment-5956602885). Implemented by
an Opus 5.5 agent on 2026-10-02.

Scripts: `scripts/analysis/aug_probe_82.py` (+ `aug_probe_82.sh`), `scripts/analysis/aug_finetune_82.py`,
`scripts/analysis/aug82_score_ckpts.sh` (+ `.slurm`), `stage_two/run_finetune_aug82.slurm`; the
transforms are `rampnet/augment.py`. Tests: `tests/test_train_augment_82.py`,
`tests/test_aug_probe_82.py`, `tests/test_aug_finetune_82.py`. Every table below is pasted from
`analysis_out/aug_transfer_82/probe_results.md` or `finetune_results.md`, which the scripts write
from the committed caches in the same directory.

**Answer, 2026-10-03.** No detectable effect. Training-time augmentation of resolution, blur,
compression and colour produced no detectable narrowing of RampNet's GSV-to-GoPro gap in this
screen: at most about 0.04 of a 0.115 gap, by the pano-level intervals. Against same-seed controls,
the change in the Laurens paired-corner rig effect is within seed noise for every arm (seed mean at
max-F1: res −0.015 [−0.034, +0.012], photo −0.020 [−0.039, +0.004], both −0.008 [−0.032, +0.013],
against a control-to-control spread of +0.029). No fine-tuned checkpoint beats the released one on
laurens_mapillary, and the transfer pool as a whole drops with the extra training (−0.015 max-F1);
on clovis and richmond some fine-tunes do edge above the released checkpoint (Step 3). The frozen
probe (Step 1) found that no single pixel-statistics axis moved to its GoPro value costs the
released model much recall, and that none of the repairs tested recovers any; neither step tested
a gamma lift or gamma-heavy training, the one axis the probe found the model sensitive to. Details
and caveats (two seeds per arm; a fifth of an epoch; same-seed flip pairing lost after a resume in
three runs) are in Step 3.

## The question

On the 47 Laurens corners both rigs saw, RampNet loses 0.112 F1 [0.033, 0.199] going from GSV to
GoPro Max, almost all of it recall (`docs/laurens_paired_151.md`). This issue asks whether
training-time augmentation of pixel statistics (resolution, blur, compression, exposure, colour)
narrows that gap without new labels. Three steps, cheapest first:

1. **Frozen probe (no training).** Degrade GSV panos toward the measured GoPro statistics one axis
   at a time, and repair GoPro panos toward GSV, and score the released model. An axis the frozen
   model does not react to is not worth an augmentation arm.
2. **Flags in `stage_two/train.py`**, off by default.
3. **Paired fine-tune screen** from the released checkpoint: control / resolution / photometric /
   both, two seeds each, about a fifth of an epoch, scored on all 12 bundles.

## Step 1 result: what the frozen model reacts to

**Short answer.** The released checkpoint is barely affected by any single pixel-statistics axis
moved to its measured GoPro value. Pooled over the four GSV splits at 0.30, the largest single-axis
recall cost is −0.018 (JPEG at quality 75). The Laurens recall gap at 0.30 is −0.128. The model does
react to darkening applied as a gamma curve, and to combinations of axes. On laurens_gsv, every
measured axis applied at once with a brightness-scale darkening costs −0.064 recall [−0.115, −0.014],
about half the Laurens gap at 0.30. That combination uses the clovis-level softening, though, and
laurens_mapillary is not actually softer than laurens_gsv. Restricted to the axes where the paired
Laurens arms really differ (exposure and colour), the share is 7% if the darkening is a brightness
scale and 64% if it is a gamma curve. The measurements here cannot say which of the two is closer to
the real GoPro darkening. **None of the repair transforms (colour-statistics match to GSV, unsharp
mask, CLAHE) recovers recall on any GoPro split. Every one costs recall or does nothing.** So the
repairs tested do not recover recall. None of them is a gamma lift, the inverse of the one axis the
degrade side found the model sensitive to (gamma darkening; `photo_gamma@gopro` reproduces 0.64 of
the Laurens gap at 0.30), so the probe does not rule out GoPro darkening as part of the gap. Step 3
caps gamma at 1.5 and does not test it either. A frozen model that loses recall under a GoPro-like
combination is sensitive in a way that augmentation could reduce, which is what Step 3 tests.

### Rig statistics

Per split, median over the bundle's panos, measured at the model's 2048×4096 input
(`analysis_out/aug_transfer_82/stats.json`, `aug_probe_82.py stats`). Sharpness (Laplacian variance,
and the high-frequency fraction: spectral power above 0.25 cycles/px over power above 0.02) and noise
(Immerkær's estimator) are measured on the ground band, rows 1024–1791. JPEG quality is the IJG
quality whose luminance table matches the native file's. manual_gold is missing because the
makelab2 checkout had no panos for it when `stats` ran; it is GSV imagery from the training
distribution and not one of the rigs being compared.

| split | native w | JPEG q (native) | lum mean | lum std | sat mean | log2 R/B | hf_frac | lap var | noise sigma |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| annapolis | 8000 | 75 | 117.6 | 91.6 | 35.6 | 0.001 | 0.0343 | 93 | 0.55 |
| bend | 16384 | 95 | 139.2 | 49.0 | 59.0 | -0.179 | 0.0755 | 275 | 1.39 |
| budapest_district5 | 5760 | 75 | 121.0 | 73.8 | 49.2 | -0.101 | 0.0781 | 300 | 1.09 |
| clovis | 5760 | 99 | 120.9 | 45.8 | 80.3 | -0.345 | 0.0293 | 76 | 0.62 |
| gainesville | 16384 | 95 | 147.6 | 48.8 | 56.0 | -0.137 | 0.0831 | 341 | 1.29 |
| laurens_gsv | 16384 | 95 | 169.0 | 49.8 | 63.4 | -0.100 | 0.0816 | 349 | 1.79 |
| laurens_mapillary | 5760 | 75 | 114.3 | 57.5 | 94.3 | -0.436 | 0.0883 | 370 | 1.35 |
| morgantown | 4096 | 75 | 126.0 | 66.2 | 47.1 | -0.121 | 0.0691 | 283 | 0.73 |
| paterson | 16384 | 95 | 148.6 | 52.6 | 47.1 | -0.118 | 0.0743 | 308 | 1.38 |
| richmond | 11000 | 75 | 133.2 | 87.2 | 30.3 | -0.018 | 0.0529 | 204 | 0.68 |
| sao_paulo | 16384 | 95 | 147.2 | 56.8 | 43.5 | -0.085 | 0.0599 | 294 | 1.30 |

What this shows:
- **The paired Laurens GoPro arm is not softer than its GSV arm at the model's input.** Its
  Laplacian variance is 370 against 349. GoPro Max native is 5,760 px wide, and the model sees it
  downsampled to 4,096, as it does GSV. The soft rigs are clovis (2018 GoPro Fusion, 76),
  annapolis (93) and richmond (204).
- **On Laurens, what differs is exposure and colour.** laurens_mapillary has 0.68× the mean
  luminance (0.48× in the ground band: 81 against 168), 1.49× the saturation, 1.15× the contrast,
  and a bluer balance (log2 R/B lower by 0.34). Its files are JPEG quality ~75 against ~95 for GSV,
  and its noise is not higher.

### Levels

Each degradation has three levels: half, gopro (the measured GoPro value) and beyond. The ratio axes
are placed on the paired Laurens medians. Sharpness is placed on clovis (see Deviations), by inverting
the Laplacian-variance calibration curves measured on 24 laurens_gsv panos. Noise has no measured
difference, so its levels are fixed (σ 2 / 4 / 8). Repairs: colour-statistics match to laurens_gsv's
median channel mean and std at α 0.5 and 1.0, unsharp mask at 50% and 100% (radius 2 px), and CLAHE
on luma at clip 0.01 and 0.02.

| axis | levels (half, gopro, beyond) | how placed |
|---|---|---|
| downscale | 0.929, 0.854, 0.504 | inverts lap_var to clovis (76) from laurens_gsv (349) |
| blur | 0.51, 0.719, 1.153 | inverts lap_var to clovis |
| brightness | 0.823, 0.677, 0.557 | ratio of median luminance |
| contrast | 1.074, 1.154, 1.239 | ratio of median luminance std |
| saturation | 1.219, 1.486, 1.811 | ratio of median saturation |
| gamma | 1.396, 1.949, 2.72 | exponent mapping median GSV luminance to GoPro |
| wb | -0.168, -0.337, -0.505 | difference in median log2(R/B) |
| noise | 2.0, 4.0, 8.0 | fixed: GoPro is not noisier (1.35 vs 1.79) |
| jpeg | 89.8, 74.8, 49.8 | estimated native quality of laurens_mapillary |

The gopro level runs on all four GSV splits, the beyond level on laurens_gsv and bend, and the half
level on laurens_gsv only (trimmed to fit the shared A40). The full tables, at both thresholds and
every level, are in `analysis_out/aug_transfer_82/probe_results.md`.

### The untransformed run reproduces the instrument

The `none` arm (native → `Resize` → `PRE`, no transform) matches the committed #25 `r2048` caches
**exactly** on all seven probe splits: the same peaks on every pano, max score difference 0
(`analysis_out/aug_transfer_82/instrument_check.json`). Every delta below is against that run, with a
paired pano bootstrap (2,000 draws, seed 82).

### Degrade GSV toward GoPro: pooled over laurens_gsv, bend, gainesville, paterson, gopro level

| arm | ΔR @0.30 | ΔF1 @0.30 | ΔR @0.55 | ΔF1 @0.55 |
|---|---|---|---|---|
| downscale@gopro | -0.012 [-0.023, -0.002] | -0.008 [-0.017, -0.001] | -0.040 [-0.055, -0.027] | -0.029 [-0.041, -0.019] |
| blur@gopro | -0.017 [-0.027, -0.007] | -0.008 [-0.016, -0.001] | -0.036 [-0.050, -0.024] | -0.025 [-0.035, -0.015] |
| brightness@gopro | -0.004 [-0.011, +0.003] | -0.001 [-0.007, +0.004] | -0.021 [-0.032, -0.011] | -0.014 [-0.022, -0.006] |
| contrast@gopro | +0.000 [-0.005, +0.005] | +0.002 [-0.003, +0.006] | -0.007 [-0.015, +0.001] | -0.005 [-0.012, +0.001] |
| saturation@gopro | -0.003 [-0.008, +0.003] | -0.001 [-0.006, +0.003] | -0.002 [-0.008, +0.005] | -0.000 [-0.005, +0.005] |
| gamma@gopro | -0.010 [-0.019, -0.001] | -0.000 [-0.007, +0.006] | -0.036 [-0.049, -0.024] | -0.024 [-0.033, -0.014] |
| wb@gopro | +0.002 [-0.005, +0.009] | +0.001 [-0.004, +0.007] | -0.009 [-0.020, +0.002] | -0.005 [-0.013, +0.003] |
| noise@gopro | -0.003 [-0.013, +0.007] | -0.003 [-0.011, +0.005] | -0.021 [-0.034, -0.009] | -0.016 [-0.025, -0.006] |
| jpeg@gopro | -0.018 [-0.029, -0.007] | -0.017 [-0.025, -0.008] | -0.035 [-0.051, -0.020] | -0.029 [-0.041, -0.018] |
| all@gopro | -0.209 [-0.234, -0.185] | -0.133 [-0.154, -0.112] | -0.261 [-0.288, -0.232] | -0.215 [-0.241, -0.189] |

### The combinations, on laurens_gsv and bend (added after `all@gopro` was read; see Deviations)

| split | arm | ΔR @0.30 | ΔF1 @0.30 | ΔR @0.55 | ΔF1 @0.55 |
|---|---|---|---|---|---|
| laurens_gsv | all@gopro | -0.277 [-0.354, -0.205] | -0.225 [-0.296, -0.155] | -0.236 [-0.311, -0.167] | -0.254 [-0.341, -0.174] |
| laurens_gsv | all_brightness@gopro | -0.064 [-0.115, -0.014] | -0.050 [-0.090, -0.014] | -0.064 [-0.126, -0.009] | -0.055 [-0.116, -0.001] |
| laurens_gsv | res_all@gopro | -0.036 [-0.081, +0.008] | -0.035 [-0.069, -0.004] | -0.032 [-0.093, +0.022] | -0.026 [-0.086, +0.025] |
| laurens_gsv | photo_brightness@gopro | -0.009 [-0.040, +0.022] | -0.009 [-0.031, +0.015] | -0.009 [-0.043, +0.026] | -0.005 [-0.037, +0.030] |
| laurens_gsv | photo_gamma@gopro | -0.082 [-0.140, -0.034] | -0.052 [-0.097, -0.012] | -0.145 [-0.221, -0.071] | -0.143 [-0.224, -0.067] |
| bend | all@gopro | -0.248 [-0.304, -0.198] | -0.143 [-0.195, -0.103] | -0.330 [-0.386, -0.277] | -0.256 [-0.311, -0.206] |
| bend | all_brightness@gopro | -0.104 [-0.143, -0.069] | -0.051 [-0.083, -0.025] | -0.183 [-0.234, -0.140] | -0.129 [-0.170, -0.095] |
| bend | res_all@gopro | -0.092 [-0.124, -0.060] | -0.046 [-0.073, -0.021] | -0.153 [-0.199, -0.111] | -0.106 [-0.145, -0.074] |
| bend | photo_brightness@gopro | +0.003 [-0.010, +0.017] | +0.007 [-0.005, +0.020] | -0.034 [-0.064, -0.007] | -0.024 [-0.044, -0.006] |
| bend | photo_gamma@gopro | -0.012 [-0.030, +0.003] | +0.001 [-0.013, +0.015] | -0.049 [-0.084, -0.018] | -0.032 [-0.058, -0.010] |

`all@gopro` overshoots: it applies brightness *and* gamma, each placed to explain the whole
luminance difference on its own. `all_brightness` is the same set with one darkening op and without
the unmeasured noise. `res_all` is downscale + blur + JPEG. `photo_brightness` and `photo_gamma` are
contrast + saturation + white balance plus one darkening op.

### Share of the Laurens recall gap reproduced on laurens_gsv

The gap is laurens_mapillary minus laurens_gsv over the whole arms, through the same instrument:
recall 0.5261 − 0.6545 = −0.1284 at 0.30 and 0.3896 − 0.4591 = −0.0695 at 0.55. The share is
an arm's ΔR on laurens_gsv divided by that gap. It is a point ratio with no interval, and the
denominator compares two different pano sets (the paired-corner gap is in `laurens_paired_151.md`).

| arm | share @0.30 | share @0.55 |
|---|---:|---:|
| downscale@gopro | -0.32 | -0.13 |
| blur@gopro | -0.11 | -0.06 |
| brightness@gopro | -0.04 | 0.26 |
| contrast@gopro | -0.07 | -0.06 |
| saturation@gopro | -0.07 | -0.26 |
| gamma@gopro | 0.07 | 0.92 |
| wb@gopro | -0.11 | 0.33 |
| noise@gopro | -0.11 | 0.06 |
| jpeg@gopro | 0.07 | -0.00 |
| all@gopro | 2.16 | 3.40 |
| res_all@gopro | 0.28 | 0.46 |
| photo_brightness@gopro | 0.07 | 0.13 |
| photo_gamma@gopro | 0.64 | 2.09 |
| all_brightness@gopro | 0.50 | 0.92 |

At 0.55, darkening by gamma alone (`gamma@gopro`) reproduces 0.92 of the gap. At 0.30 it reproduces
0.07. Darkening moves peaks from above 0.55 to between 0.30 and 0.55, so it mostly changes scores
rather than what the model finds.

### Repair GoPro toward GSV: pooled over laurens_mapillary, clovis, richmond

| arm | ΔR @0.30 | ΔF1 @0.30 | ΔR @0.55 | ΔF1 @0.55 |
|---|---|---|---|---|
| colour_match@0.5 | -0.009 [-0.020, +0.001] | -0.012 [-0.021, -0.004] | -0.015 [-0.025, -0.005] | -0.008 [-0.017, +0.000] |
| colour_match@1 | -0.016 [-0.030, -0.001] | -0.010 [-0.021, +0.001] | -0.053 [-0.073, -0.034] | -0.037 [-0.055, -0.021] |
| unsharp@50 | -0.005 [-0.015, +0.005] | -0.005 [-0.014, +0.004] | -0.016 [-0.026, -0.006] | -0.010 [-0.018, -0.002] |
| unsharp@100 | -0.024 [-0.039, -0.009] | -0.018 [-0.030, -0.006] | -0.036 [-0.051, -0.022] | -0.022 [-0.034, -0.011] |
| clahe@0.01 | -0.032 [-0.046, -0.018] | -0.021 [-0.032, -0.009] | -0.048 [-0.067, -0.030] | -0.029 [-0.045, -0.014] |
| clahe@0.02 | -0.052 [-0.072, -0.032] | -0.034 [-0.052, -0.018] | -0.089 [-0.111, -0.067] | -0.061 [-0.081, -0.042] |
| repair_all | -0.032 [-0.047, -0.015] | -0.025 [-0.038, -0.011] | -0.085 [-0.105, -0.066] | -0.059 [-0.077, -0.041] |

No repair has a recall interval above zero on any split or pool, at either threshold.


## Step 2: augmentation flags in `stage_two/train.py`

Three new flags, all off by default:

- `--aug OP=P:LO:HI`, repeatable. It applies `OP` with probability `P` at a level drawn uniformly
  from `[LO, HI]`. The ops are `downscale`, `blur`, `brightness`, `contrast`, `saturation`, `gamma`,
  `wb`, `hue`, `noise` and `jpeg`, from `rampnet/augment.py`, the same functions the probe uses. They
  are applied in that fixed order (optics, colour, sensor noise, compression last) to the PIL image
  after the horizontal flip and before `Resize`/`ToTensor`/`Normalize`. Training panos are stored at
  2048×4096, so the resize is a no-op and levels are in model-input pixels, as in the probe. The ops
  are pixel-wise, so labels are untouched. Only the train split is augmented.
- `--max-steps N` stops after N optimizer steps. It writes `checkpoints/final_step_N.pth` (a bare
  state_dict, the format `best_model.pth` has) and then `latest_checkpoint.pth`, and skips
  validation (42,875 val panos would take longer than the screen itself).
- `--grad-accum K` gives a global batch of world size × K. It is only allowed for a `--max-steps`
  run that ends inside the first epoch, because accumulation groups are aligned on the global step
  counter and a group could otherwise straddle an epoch boundary.

**Off means off.** With no flag, `EquiHeatmapDataset` returns bit-identical tensors to the published
recipe's: `tests/test_train_augment_82.py` pins the sha256 of its output on synthetic PNGs, computed
from `train.py` at 459ea9e before any of this was added. The step sequence at `--grad-accum 1`
is the published one (zero_grad, forward, backward on the unscaled loss, step, update).
`stage_two/run_train.slurm` is not touched (a test asserts it).

**Arms are paired by `--seed`.** Augmentation draws come from their own generator,
`numpy.random.default_rng(SeedSequence([82, seed, epoch, sample_index]))`, never from the global
`random`/`numpy`/`torch` streams. The flip uses `random.random()` inside the DataLoader worker, so if
an augmentation shared that stream, an augmented arm would see different flips from its same-seed
control. A test runs the dataset with and without augmentation from the same `random.seed` and checks
three things: the heatmaps (which depend only on the points and the flip) are identical, the global
stream ends in the same state, and the pixels did change.

**Requeue/resume.** On a resume from `latest_checkpoint.pth`:
- The data order is unchanged (`DistributedSampler` is a function of seed and epoch, and
  `ResumeSkipSampler` drops the batches already done).
- The augmentation draws are unchanged, because they are keyed on the sample index rather than a
  running counter. A test checks this.
- The flip draws are not reproduced. The DataLoader re-seeds its workers on every start, so after a
  resume a sample can get a different flip than it would have in an uninterrupted run. The published
  recipe has the same property.

## Step 3: the paired fine-tune screen

**Setup.** Every run starts from the released weights: Hub revision 606a119, `model.safetensors`
sha256 `f2119e3b…`, converted once to a bare state_dict, `released_rampnet_state_dict.pth`, sha256
`024a987c…`. Each run then trains 2,000 optimizer steps at the recipe's constant LR 1e-5 with Adam
and AMP, global batch 16 (4 GPUs × batch 1 × accumulation 4), on the full Stage 2 train split
(`/gscratch/scrubbed/jfroehli/rampnet_dataset/train`, 150,063 panos; it was intact on 2026-10-02, so no
subset was staged). That is the train split of the published `projectsidewalk/rampnet-dataset`
(same 150,063-pano count; see Reproducing for the clean-clone route and what was not checked). 2,000 steps is 32,000 panos, about a fifth of an epoch, and it took 2–3 h per
run on 4 A40/L40/L40S (1.0–1.4 s per micro-batch). The arms (`stage_two/run_finetune_aug82.slurm`):

| arm | `--aug` added to the recipe's flip | why these ranges |
|---|---|---|
| control | none | the same extra training, nothing new |
| res | `downscale=0.5:0.5:1.0` `blur=0.3:0.0:1.15` `jpeg=0.5:50:95` | from the GSV value through the probe's beyond level (downscale 0.50, blur 1.15 px, JPEG 50) |
| photo | `brightness=0.5:0.6:1.2` `contrast=0.5:0.85:1.25` `saturation=0.5:0.7:1.6` `gamma=0.5:0.75:1.5` `wb=0.5:-0.5:0.3` `hue=0.3:-6:6` | covers the measured Laurens GoPro offsets (brightness 0.68, contrast 1.15, saturation 1.49, WB −0.34) in both directions; gamma is capped at 1.5 because the measured 1.95 already overshoots in the probe |
| both | res + photo | |

The levels were set from the rig statistics before the probe's contrasts were read. Training had to
start early in the day to finish, so the probe's results did not feed back into the ranges. Each arm
ran at seeds 1 and 2. Arms at the same seed share data order and augmentation draws, and they share
flip draws until either run resumes from a checkpoint, so every augmented arm is compared with its
same-seed control. The two controls differ in seed only, and their difference (`spread`) is the
noise floor printed beside every contrast.

**Same-seed flip pairing broke in three runs.** A resume re-seeds the DataLoader workers, so flips
after it are re-drawn (see Step 2, Requeue/resume). From the committed logs
(`analysis_out/aug_transfer_82/finetune/trainlogs/`): photo_s1 resumed at micro-batch 4000 (job
41138714), and control_s2 and both_s2 resumed at 4400 (jobs 41138715 and 41123607, three
incarnations each). control_s1, res_s1, both_s1, res_s2 and photo_s2 ran without a resume. So for
about the second half of the 8,000 micro-batches, flips are not paired in photo_s1 vs control_s1
(only the arm resumed) or in res_s2 and photo_s2 vs control_s2 (only the control resumed). both_s2
vs control_s2 both resumed at 4400 but after different earlier incarnations, so their flips are not
known to match either. Data order and augmentation draws are unaffected. This does not bias a
contrast; it makes the pairing looser than designed, and it matters for one reading flagged below.

All eight fine-tunes trained to 2,000 steps and every one was scored on all 12 bundles (scoring jobs
41103600 for the released checkpoint, then 41145630, 41145631, 41159296, 41160386, 41161884,
41163344, 41168242 and 41170148; checkpoint hashes in
`analysis_out/aug_transfer_82/finetune/trainlogs/ckpt_SHA256SUMS.txt`). Every number below is
pasted from `analysis_out/aug_transfer_82/finetune_results.md`. Intervals are 95% paired
bootstraps (2,000 draws, seed 82) over panos, stratified by split, or over the 47 Laurens corner
pairs for the rig effect. **None of these intervals includes training-seed variance.** With two
seeds per arm, the only estimate of seed variance is the gap between the two controls (the
`spread` row), which is one draw.

### Short answer

**No augmentation arm narrows the GSV-to-GoPro gap by more than the two controls differ from each
other (at most about 0.04 of a 0.115 gap, by the pano-level intervals), and no fine-tuned
checkpoint beats the released one on laurens_mapillary.** Negative result. On the other two GoPro
splits the picture is mixed rather than a drop: on clovis (released max-F1 0.837) control_s1 0.844,
control_s2 0.845, photo_s1 0.838, photo_s2 0.842 and both_s2 0.850 are above it, and on richmond
(0.870) control_s1 0.878 and both_s2 0.871 are. The transfer pool as a whole drops (item 3).

1. **On the Laurens paired corners, the change in the rig effect against the control is within
   seed noise for every arm.** Seed mean (both seeds of an arm against both controls), the change
   in the max-F1 rig effect is −0.015 [−0.034, +0.012] for res, −0.020 [−0.039, +0.004] for photo
   and −0.008 [−0.032, +0.013] for both (negative means a smaller gap). The two controls alone
   differ by +0.029 [−0.007, +0.060]. The released checkpoint's rig effect is +0.115 [+0.047,
   +0.163] at max-F1, so even the best point estimate would close about a sixth of it.
2. **On the pooled transfer splits, augmentation does nothing measurable.** Seed-mean max-F1
   against the control on laurens_mapillary + clovis + richmond: res +0.003 [−0.007, +0.010],
   photo +0.005 [−0.002, +0.013], both +0.004 [−0.005, +0.011]. In-domain (manual_gold + bend)
   it is +0.001 to +0.003.
3. **The extra training itself hurts transfer.** Seed-mean control minus released: max-F1 on the
   transfer pool −0.015 [−0.026, −0.003], and on laurens_mapillary −0.057 [−0.085, −0.034]. All
   eight fine-tunes are below the released checkpoint on laurens_mapillary at max-F1 (0.637 to
   0.684, against 0.710), and the GSV-minus-GoPro gap on the paired corners widens by +0.028
   [+0.003, +0.064] at max-F1. In-domain the extra steps help slightly (+0.005 [+0.002, +0.008]).
4. **The preliminary read, that res and both raise recall and lower precision at 0.30, was a
   seed effect, not an augmentation effect.** On 11 of the 12 splits at 0.30 (all but
   morgantown), every seed-1 checkpoint, control included, sits at higher recall and lower
   precision than every seed-2 checkpoint
   (laurens_gsv recall at 0.30: 0.727 to 0.754 for seed 1, 0.605 to 0.627 for seed 2). Seed 1's
   arms were the ones read first, against the released checkpoint. At max-F1 the two controls
   are close to each other (transfer pool −0.007 [−0.022, +0.005]). Same-seed controls are what
   made this visible.

### Per-checkpoint max-F1 (AP in parentheses), selected splits

| split | released | control s1 / s2 | res s1 / s2 | photo s1 / s2 | both s1 / s2 |
|---|---|---|---|---|---|
| laurens_gsv | 0.812 (0.768) | 0.806 / 0.819 | 0.810 / 0.812 | 0.813 / 0.822 | 0.804 / 0.824 |
| laurens_mapillary | 0.710 (0.691) | 0.668 / 0.637 | 0.668 / 0.660 | 0.684 / 0.662 | 0.651 / 0.668 |
| clovis | 0.837 (0.868) | 0.844 / 0.845 | 0.827 / 0.833 | 0.838 / 0.842 | 0.830 / 0.850 |
| richmond | 0.870 (0.876) | 0.878 / 0.870 | 0.869 / 0.867 | 0.870 / 0.860 | 0.866 / 0.871 |
| manual_gold | 0.907 (0.909) | 0.912 / 0.912 | 0.912 / 0.915 | 0.912 / 0.914 | 0.915 / 0.916 |
| bend | 0.873 (0.871) | 0.874 / 0.873 | 0.876 / 0.874 | 0.878 / 0.869 | 0.874 / 0.875 |

All 12 splits, with AP for every checkpoint, are in the first table of `finetune_results.md`.

### Laurens rig effect (GSV minus GoPro) on the 47 paired corners

| checkpoint | ΔR @0.30 | ΔF1 @0.30 | ΔF1 max-F1 |
|---|---|---|---|
| released | +0.128 [+0.054, +0.201] | +0.095 [+0.038, +0.151] | +0.115 [+0.047, +0.163] |
| control_s1 | +0.196 [+0.121, +0.269] | +0.145 [+0.094, +0.197] | +0.129 [+0.075, +0.176] |
| control_s2 | +0.137 [+0.043, +0.231] | +0.137 [+0.061, +0.221] | +0.158 [+0.093, +0.209] |
| res_s1 | +0.190 [+0.113, +0.267] | +0.142 [+0.081, +0.204] | +0.116 [+0.058, +0.170] |
| res_s2 | +0.080 [+0.018, +0.145] | +0.094 [+0.043, +0.147] | +0.140 [+0.077, +0.196] |
| photo_s1 | +0.164 [+0.084, +0.241] | +0.125 [+0.069, +0.177] | +0.101 [+0.050, +0.165] |
| photo_s2 | +0.123 [+0.053, +0.191] | +0.119 [+0.064, +0.176] | +0.146 [+0.078, +0.194] |
| both_s1 | +0.158 [+0.077, +0.236] | +0.129 [+0.067, +0.187] | +0.128 [+0.066, +0.185] |
| both_s2 | +0.127 [+0.046, +0.217] | +0.127 [+0.062, +0.202] | +0.142 [+0.079, +0.195] |

max-F1 here is each rig's best F1 over thresholds 0.05 to 0.95 in steps of 0.01, chosen
separately for each rig. That makes the max-F1 column blind by construction to an arm that fixes
GoPro score calibration, which is what the probe says darkening breaks: a calibration repair would
show up only at a fixed threshold, as a smaller gap at 0.30 or 0.55. Those rows are the ones to read
for it, and they are null too (next table). The released row at 0.30 is the same number as the `rampnet_r2048`
same-input re-run in `laurens_paired_151.md` (+0.095 [+0.040, +0.152]); the CI differs in the
third decimal because the resample is drawn here with seed 82.

### Difference in differences: change in the rig effect against the control

Same pair resample applied to both checkpoints, so corner difficulty cancels. Negative means the
arm has a smaller GSV-minus-GoPro gap than its control.

| contrast | Δ(rig effect) R @0.30 | Δ(rig effect) F1 @0.30 | Δ(rig effect) max-F1 |
|---|---|---|---|
| spread: control_s2 − control_s1 | −0.060 [−0.128, +0.009] | −0.008 [−0.065, +0.052] | +0.029 [−0.007, +0.060] |
| res_s1 − control_s1 | −0.006 [−0.045, +0.036] | −0.003 [−0.036, +0.033] | −0.013 [−0.045, +0.023] |
| res_s2 − control_s2 | −0.057 [−0.115, +0.001] | −0.043 [−0.093, +0.007] | −0.018 [−0.044, +0.019] |
| photo_s1 − control_s1 | −0.032 [−0.069, +0.005] | −0.020 [−0.050, +0.009] | −0.028 [−0.048, +0.008] |
| photo_s2 − control_s2 | −0.014 [−0.075, +0.047] | −0.017 [−0.071, +0.034] | −0.012 [−0.046, +0.018] |
| both_s1 − control_s1 | −0.038 [−0.092, +0.018] | −0.016 [−0.060, +0.028] | −0.001 [−0.032, +0.027] |
| both_s2 − control_s2 | −0.010 [−0.053, +0.029] | −0.009 [−0.047, +0.025] | −0.016 [−0.040, +0.010] |
| **seed mean:** control − released | +0.038 [−0.007, +0.085] | +0.046 [+0.009, +0.084] | +0.028 [+0.003, +0.064] |
| **seed mean:** res − control | −0.032 [−0.068, +0.003] | −0.023 [−0.055, +0.007] | −0.015 [−0.034, +0.012] |
| **seed mean:** photo − control | −0.023 [−0.060, +0.012] | −0.018 [−0.050, +0.010] | −0.020 [−0.039, +0.004] |
| **seed mean:** both − control | −0.024 [−0.059, +0.013] | −0.013 [−0.041, +0.015] | −0.008 [−0.032, +0.013] |

What the table does and does not show:
- **Every per-seed interval includes zero**, at every read.
- **All six per-seed max-F1 point estimates are negative** (−0.001 to −0.028). They are not six
  independent results: the three arms at a seed share one control, so this is two draws of
  control noise. The control spread (+0.029) is larger than any of them.
- **The best case is small.** Even taking photo's seed mean (−0.020) at face value, augmentation
  would recover the +0.028 the extra training cost, not the +0.115 gap the released model
  already has.

### Pooled pano-level contrasts, seed mean (arm s1 and s2 against control s1 and s2)

| pool | contrast | ΔR @0.30 | ΔP @0.30 | ΔF1 @0.30 | Δ max-F1 |
|---|---|---|---|---|---|
| transfer | control − released | +0.007 [−0.009, +0.022] | −0.038 [−0.057, −0.019] | −0.014 [−0.028, −0.001] | −0.015 [−0.026, −0.003] |
| transfer | res − control | +0.020 [+0.008, +0.033] | −0.027 [−0.039, −0.015] | +0.000 [−0.010, +0.010] | +0.003 [−0.007, +0.010] |
| transfer | photo − control | +0.014 [+0.005, +0.024] | −0.006 [−0.017, +0.007] | +0.006 [−0.002, +0.015] | +0.005 [−0.002, +0.013] |
| transfer | both − control | +0.011 [+0.001, +0.021] | −0.014 [−0.026, −0.003] | −0.001 [−0.008, +0.008] | +0.004 [−0.005, +0.011] |
| in-domain | control − released | +0.008 [+0.004, +0.012] | −0.003 [−0.007, +0.003] | +0.002 [−0.001, +0.006] | +0.005 [+0.002, +0.008] |
| in-domain | res − control | +0.004 [+0.001, +0.006] | −0.001 [−0.004, +0.002] | +0.001 [−0.001, +0.003] | +0.001 [−0.001, +0.003] |
| in-domain | photo − control | +0.000 [−0.002, +0.003] | +0.001 [−0.002, +0.004] | +0.001 [−0.001, +0.003] | +0.001 [−0.001, +0.003] |
| in-domain | both − control | +0.001 [−0.002, +0.004] | +0.006 [+0.004, +0.009] | +0.004 [+0.002, +0.006] | +0.003 [+0.001, +0.005] |
| US7 | control − released | +0.002 [−0.006, +0.010] | −0.025 [−0.034, −0.015] | −0.011 [−0.017, −0.005] | −0.007 [−0.013, −0.002] |
| US7 | res − control | +0.013 [+0.008, +0.018] | −0.017 [−0.023, −0.010] | +0.000 [−0.004, +0.005] | +0.001 [−0.003, +0.005] |
| US7 | photo − control | +0.005 [−0.000, +0.010] | −0.001 [−0.008, +0.005] | +0.003 [−0.002, +0.007] | +0.002 [−0.002, +0.006] |
| US7 | both − control | +0.006 [+0.001, +0.011] | −0.006 [−0.012, +0.000] | +0.001 [−0.004, +0.005] | +0.001 [−0.002, +0.006] |

transfer = laurens_mapillary + clovis + richmond; in-domain = manual_gold + bend; US7 = the eight
US splits of `miss_decomposition.US_SPLITS`. The res arm does trade precision for recall at 0.30
(+0.020 recall, −0.027 precision on the transfer pool), which is a score shift: at max-F1 it is
+0.003. **That reading comes entirely from seed 2,** against control_s2, the control whose flips
were re-drawn after its resume: on the transfer pool at 0.30 all three seed-2 contrasts have recall
intervals above zero (res_s2 +0.039 [+0.020, +0.057], photo_s2 +0.028 [+0.012, +0.044], both_s2
+0.019 [+0.005, +0.032]), while all three seed-1 contrasts are about zero (+0.001, +0.000, +0.003).
It is as consistent with control_s2 being a low-recall draw as with an effect of the arms. The one positive interval at max-F1, both on in-domain (+0.003 [+0.001, +0.005]), is
in-domain and about a third of a point. Per-seed contrasts for every split are in
`finetune_results.md`.

### Caveats

- **n = 2 seeds per arm.** The intervals are over panos or corner pairs. They say how sure we are
  about these particular checkpoints, not about what another seed would do. The control spread is
  one draw of seed-to-seed variance, and on several reads it is as large as any arm effect. The
  seed variance campaign in `seed_variance_51_135.md` found seed-to-seed spread to be the binding
  limit on full training runs too.
- **A fifth of an epoch, at constant LR 1e-5, from the released checkpoint.** This is a screen,
  not the recipe. An effect that needs a full epoch of augmented training, or training from
  scratch, would not show up here. What this screen does show is that the extra steps alone move
  transfer by more (−0.015 max-F1 on the transfer pool) than any augmentation moves it.
- **The ranges were set before the probe was read** (see Setup), and the probe later found that
  the axes the frozen model reacts to most are darkening by gamma and combinations of axes. The
  photo arm covers gamma only up to 1.5. A gamma-heavy arm was not run.
- **The paired-corner set is small** (47 pairs), which is why its intervals are about ±0.03 to
  ±0.06 on a difference in differences.
- **max-F1 picks its threshold on the evaluation data itself**, so it is an optimistic number for
  every checkpoint alike. It is used here to compare checkpoints whose scores are calibrated
  differently, not as a deployable operating point.


## Deviations from the plan, and why

- **Where the probe applies its transforms.** The brief asked for the transform before the model's
  standard resize to 2048×4096. It is applied right after it: native → `Resize((2048, 4096))`
  bilinear (the first step of the scorer's `threshold_sweep.PRE`) → transform → `PRE` (whose resize
  is then a no-op; PIL returns a copy at equal size). Native widths run from 5,760 to 16,384, so a
  native-resolution blur sigma or JPEG quality would mean a different thing on every split. Training
  panos are stored at 2048×4096, so training augmentation can only happen at that size. With this
  choice, probe levels and training ranges are in the same units. The untransformed arm through this
  path reproduces the committed instrument exactly (below).
- **Sharpness levels come from clovis, not laurens_mapillary.** At the model's input the paired GoPro
  split is not softer than GSV (Laplacian variance 370 against 349), because its 5,760 px native is
  downsampled too. "GoPro-like" sharpness is therefore placed on clovis (76), the softest GoPro split
  and the one the issue was filed about.
- **Probe levels were trimmed to fit the A40.** The GPU was shared with another job at 100%
  utilisation, which made a forward pass ~1.7 s instead of ~0.6 s. The GoPro-measured level runs on
  all four GSV splits, the beyond level on laurens_gsv and bend, and the half level on laurens_gsv
  only (`LEVEL_SPLITS`).
- **Decomposition arms were added after the first read.** `all@gopro` applies brightness and gamma,
  each placed to account for the whole luminance difference, so it is darker than the GoPro imagery.
  Four arms (`res_all`, `photo_brightness`, `photo_gamma`, `all_brightness`, each with at most one
  luminance op, on laurens_gsv and bend) were added after that was seen.
- **4 GPUs × accumulation 4, not 16 GPUs.** ckpt-all was full of other jobs; one node with 4 GPUs
  schedules far sooner than four. The global batch is the recipe's 16.
- **Fine-tuned checkpoints were scored on klone, not makelab2.** The A40 was busy with the probe.
  The released checkpoint scored on klone reproduces the committed #25 `r2048` caches on all 11
  splits that have them: same peaks, max score difference 8.2e-5
  (`tests/test_aug_finetune_82.py`). manual_gold's panos on klone had been purged and were copied
  from makelab2, then verified against `benchmark/manual_gold/imagery_manifest.json`.

## Infrastructure notes (what went wrong on klone, so the next run does not repeat it)

- `/gscratch/scrubbed/jfroehli/hf` had been purged: the Hub blob symlink was dangling, and jobs
  41103263–70 died seconds after starting. The launcher now reads the home HF cache (read only) and
  checks the blob's sha256.
- On 2026-10-02 every training job sat 25–30 min in uninterruptible GPFS I/O inside `import torch`
  (all ranks in D state, reading the conda env under `/gscratch/makelab`). A single `cat` of
  `libtorch_cuda.so` (1.4 GB) on a compute node took 47 s. The launcher first gained a
  library pre-read, then the option to unpack a tarball of the env onto node-local NVMe
  (`/scr`, 2.8 TB), which job 41123104 built.

## Compute

All of it was free. **Step 1:** makelab2's A40, 3.97 h of extraction plus a 624 s CPU stats pass,
recorded as four `paid: false` rows (`aug-probe-82:*`) in `analysis_out/usage_log.jsonl`. **Step 3:**
klone `ckpt-all`, 43 allocations, **108.66 GPU-hours**, recorded in `analysis_out/compute_log.jsonl`
from the committed dump `docs/data/compute/sacct_klone_2026-10-03_aug82.txt` (pulled with `sacct
-D`, so requeued incarnations are counted). Of that, 79.51 GPU-h is the eight finishing
incarnations (11.7 to 14.2 GPU-h for the five that ran in one go; 4.3, 5.4 and 6.0 GPU-h for
control_s2, both_s2 and photo_s1, which finished from a resume), 23.13 GPU-h went to incarnations that were preempted or requeued
(resume from `latest_checkpoint.pth` kept part of that work), 0.15 GPU-h to the eight jobs that
failed at start, and 5.87 GPU-h to scoring nine checkpoints on all 12 bundles (about 35 min each on
one A40/L40/L40S). The breakdown and the exact commands are in `docs/compute_cost.md` (klone,
2026-10-03).

## Reproducing

All CPU steps run from a clean clone with `requirements-dev.txt`; the GPU steps need the benchmark
panos (`benchmark/<split>/panos/`, not committed; fetched per `benchmark/README.md`, or on makelab2 at
`/homes/gws/jonf/RampNet/benchmark/<split>/panos`, on klone at
`/gscratch/makelab/jonf/rampnet_benchmark/<split>/panos` and, for manual_gold,
`/gscratch/scrubbed/jfroehli/manual_gold/panos`).

**Not published, stated beside the numbers.**
- **The eight fine-tuned checkpoints exist only at `/gscratch/makelab/jonf/aug82/`** on klone
  (8 x ~340 MB, sha256 in `analysis_out/aug_transfer_82/finetune/trainlogs/ckpt_SHA256SUMS.txt`).
  Every number in this doc re-derives on CPU from the committed caches
  (`aug_finetune_82.py --check`), but re-scoring them, or any new analysis (another split, recall
  by distance), needs those files. What would unblock it: a push of the eight files to a Hugging
  Face repo. Re-training is not a substitute: training is not bit-reproducible (no deterministic
  algorithms, AMP, and flips re-drawn on every resume), so a re-run gives different checkpoints
  with, at best, similar numbers. The committed caches and hashes are the durable record.
- **The train split.** Training read `/gscratch/scrubbed/jfroehli/rampnet_dataset/train`. From a
  clean clone the route is `python download_dataset.py` (repo root), which writes
  `./dataset/{train,val,test}` from `projectsidewalk/rampnet-dataset`; its train split has the same
  150,063 panos. Not checked: whether the klone files are byte-identical to that script's output
  (the script re-encodes each image as JPEG quality 95), or which Hub revision the klone copy was
  made from. `download_dataset.py` does not pin a revision.

```bash
# --- Step 1 (makelab2; ~10 min CPU for stats, ~4.3 h on a shared A40 for the probe) ---
python scripts/analysis/aug_probe_82.py stats --panos-root /homes/gws/jonf/RampNet --workers 6
python scripts/analysis/aug_probe_82.py arms             # the arm table derived from stats.json
bash scripts/analysis/aug_probe_82.sh                    # none -> check (stops on mismatch) -> all arms -> report
# re-derive on CPU from the committed caches:
python scripts/analysis/aug_probe_82.py check            # instrument: none == #25 r2048, exit 1 otherwise
python scripts/analysis/aug_probe_82.py report --check   # probe_results.json is up to date

# --- Step 3 (klone, from a checkout of this branch at /gscratch/scrubbed/$USER/RampNet_aug82) ---
mkdir -p logs
# optional but strongly recommended on klone: a tarball of the env, unpacked per job onto /scr
sbatch -p ckpt-all -c 4 --mem=8G --time=3:00:00 --wrap "tar cf /gscratch/scrubbed/$USER/aug82/sidewalkcv2_env.tar -C /gscratch/makelab/jonf/envs sidewalkcv2"
for s in 1 2; do for a in control both res photo; do
  RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2 ARM=$a SEED=$s sbatch --job-name=aug82_${a}_s$s stage_two/run_finetune_aug82.slurm
done; done
# released checkpoint, scored once with the same scorer (benchmark/<split>/panos symlinked as above)
RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2 LABELS=released sbatch scripts/analysis/aug82_score_ckpts.slurm
# after each job finishes: copy to /gscratch/makelab/jonf/aug82/<arm>_s<seed>.pth, hash, score
bash scripts/analysis/aug82_finish_klone.sh control_s1 both_s1 ...
# copy analysis_out/aug_transfer_82/finetune/<label>/*.json back, then on CPU:
python scripts/analysis/aug_finetune_82.py               # -> finetune_results.json / .md
python scripts/analysis/aug_finetune_82.py --check
```

