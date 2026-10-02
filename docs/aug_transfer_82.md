# Augmentation as a rig-transfer lever: frozen probe and paired fine-tune screen (#82)

**Status:** in progress (PR #235). Plan: the 2026-10-02 comment on
[#82](https://github.com/ProjectSidewalk/RampNet/issues/82#issuecomment-5956602885). Implemented by
an Opus 5.5 agent on 2026-10-02.

Scripts: `scripts/analysis/aug_probe_82.py` (+ `aug_probe_82.sh`), `scripts/analysis/aug_finetune_82.py`,
`scripts/analysis/aug82_score_ckpts.sh` (+ `.slurm`), `stage_two/run_finetune_aug82.slurm`; the
transforms are `rampnet/augment.py`. Tests: `tests/test_train_augment_82.py`,
`tests/test_aug_probe_82.py`, `tests/test_aug_finetune_82.py`. Every table below is pasted from
`analysis_out/aug_transfer_82/probe_results.md` or `finetune_results.md`, which the scripts write
from the committed caches in the same directory.

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

STEP1_PLACEHOLDER

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

STEP3_PLACEHOLDER

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

COST_PLACEHOLDER

## Reproducing

REPRO_PLACEHOLDER
