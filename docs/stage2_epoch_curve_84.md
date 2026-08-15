# Run A: the Stage 2 epoch curve (#84)

**Status: launched 2026-08-15. Results section is empty on purpose — it is filled from the run.**

## What this run is

One question: **does `manual_gold` F1 peak before, at, or after epoch 5?**

The paper run's auto-label validation loss — the signal `train.py` selects on — bottoms at epoch 5
and rises monotonically through epoch 11 (measured from the rescued TensorBoard events, #104). So we
already have the machine-label half of the curve. What we do not have is the human half: the
epoch-N weights were deleted in a 2025-07-11 cleanup, so `manual_gold` cannot be scored across the
curve from the original run. Run A regenerates exactly those checkpoints.

The three readings were pre-registered in [#84](https://github.com/ProjectSidewalk/RampNet/issues/84)
before launch:

| if `manual_gold` peaks | then |
| :--- | :--- |
| **earlier than 5** | the Stage 1 label ceiling is real and binds early; the auto-val gain from epochs 1→5 is the model fitting Stage 1's errors. A #59 result: quality binds before quantity. |
| **at ~5** | the selection rule is fine and the released model is simply undertrained — a free ~12% val-loss improvement was left on the table. |
| **later than 5** | auto-val is the wrong selection signal outright. |

Two things the pre-registration fixes, and which the analysis must honour:

- **Tie bar 0.01 F1.** `manual_labels/` holds 3,919 instances over 1,000 panos, so counting noise
  alone is ≈0.008 s.e. on recall. A peak must clear the bar *and* be supported by its neighbours; a
  one-epoch spike between two ties is noise.
- **Record max-F1 per checkpoint, not only F1@0.30.** Holding conf 0.30 fixed across the curve
  confounds calibration with capability. The protocol headline stays F1@0.30 single-pass (#54) for
  cross-model comparability, but the epoch-peak question is answered on the calibration-free curve.
  If the two curves peak at different epochs, that is itself a finding.

The Run B gate is also evaluated on the calibration-free curve: **Run B runs unless the curve
degrades** — every epoch ≥ 2 below epoch 1 by more than the tie bar. Flat is not a cancellation.

## What is held fixed, and the one thing that is not

Everything in the released recipe is unchanged: 4 nodes × 4 GPUs ⇒ world size 16, and `train.py`
uses `batch_size=1` per rank, so the global batch is **16** exactly as in the paper run. Constant
lr 1e-5 (`--preset scratch`), no scheduler, seed 42, same data, same splits, ImageNet-initialised
backbone.

**The only change is `--epochs 8` instead of 1.**

The world size is load-bearing rather than a speed knob. Because the per-rank batch is 1, changing
the node or GPU count changes the global batch and therefore the optimisation regime — a 1-GPU copy
of this run trains at global batch 1 and is not a slower replicate, it is a different experiment.
This is why the run is on a 16-GPU cluster allocation and not on a single lab A40.

One honest limit on the word "replicate": **the code that produced the 2025 run is not in git.** The
public history starts from a squashed commit already carrying `num_epochs = 1`, so June-2025
run-time values are unrecoverable. Run A replicates the *committed* recipe, and its epoch-1
checkpoint is comparable to the released model modulo seed and dataloader order — the released
`best_model.pth` is byte-identical to that run's `epoch_1_step_9378.pth`.

The *environment*, at least, is close to exact. The env is built from the committed
`environment.yml`, which asks for python 3.10 / pytorch 2.6 / torchvision 0.21 / cuda-version 12.6 —
and `environment.lock.yml`, the linux-64 lock from the paper machine, pins
`pytorch=2.6.0=cuda126_mkl_py310_...`, `torchvision=0.21.0=cuda126_py310_...`,
`python=3.10.13`, `timm=1.0.15`. They agree on every version that could plausibly move a number, so
the delta is patch-level and transitive at most. The build records `conda list --explicit` into its
job log precisely so that delta can be diffed rather than assumed.

(In passing: `CLAUDE.md` describes this env as "Linux + CUDA 11.8". That is stale — the lock file
says `cuda-version=12.9` with cuda126 builds of torch. Worth fixing there separately.)

## Where it runs, and why

**Training on klone `ckpt-all`, 16 GPUs, free.** The 8.24 h preemption slice ceiling that gave the
#51 YOLO tiles arms zero completed epochs does not bite here: `train.py` writes
`latest_checkpoint.pth` every 1,000 steps (~22 min of state at risk) and resume takes precedence
over `--init-weights`, so preemption costs the measured ×1.67 calendar factor and nothing else.

At the measured 1.341 s/step (rank 0 median, n = 119,902) and 9,378 steps/epoch: **3.49 h/epoch,
~56 GPU-h/epoch ⇒ ~28 h compute, ~47 h calendar for 8 epochs.**

Tillicum was ruled out. The step time is I/O-bound, not compute-bound — the p25–p75 spread is 6 ms
across 119,902 samples, roughly **3% MFU** — so paying $0.90/GPU-h × 16 for H200s that spend 97% of
their cycles waiting on the dataloader buys nothing.

**Evaluation on makelab2.** Scoring the 8 checkpoints has no batch-regime constraint, the A40
handles it, and it keeps the sweep off the queue-contended cluster.

## Exact commands, in order

Prerequisites verified before launch on 2026-08-15:

- Staged dataset at `/gscratch/scrubbed/jfroehli/rampnet_dataset` is **intact**, 465 GB —
  150,063 train / 42,875 val / 21,438 test panorama+JSON pairs. 150,063 with `drop_last=True` at
  world size 16 is 9,378 steps/epoch, matching the paper run's step count exactly. The directory
  mtime (2026-07-24) is older than `/gscratch/scrubbed`'s ~21-day purge window, so this was checked
  by counting and reading files, not by stat'ing the directory.
- Per-epoch checkpoints go to `/gscratch/makelab` — **purchased and never purged** — not to
  `scrubbed`. They are the entire product of ~28 GPU-h and a silent quota-exhausted write on a
  shared pool is what corrupted `y26_pano`'s `last.pt` in the #51 grid.

```bash
# 0. Sync the repo to klone (code only; the dataset already lives on scratch).
rsync -av --exclude .venv --exclude .model_cache --exclude 'benchmark/*/panos' \
      --exclude 'benchmark/*/gallery' --exclude view_dump --exclude dataset \
      --exclude runs --exclude '*.pt' RampNet/ klone:~/RampNet/

# 1. One-time: build the sidewalkcv2 conda env from the committed environment.yml.
#    As a batch job, not on a login node -- klone reaps heavy login processes and
#    that reap also kills the SSH control master.
cd ~/RampNet && mkdir -p logs
export RAMPNET_REPO=$HOME/RampNet
export RAMPNET_ENV=/gscratch/scrubbed/jfroehli/envs/sidewalkcv2
export RAMPNET_CONDA_PKGS=/gscratch/scrubbed/jfroehli/conda_pkgs   # NOT ~/.conda/pkgs: klone
                                                                   # homes are capped at 10 GB and
                                                                   # pytorch+CUDA alone is ~5 GB
sbatch --export=ALL,RAMPNET_REPO,RAMPNET_ENV,RAMPNET_CONDA_PKGS \
       stage_two/run_build_env.slurm

# 1b. Pre-warm the timm backbone cache. --preset scratch builds the model with
#     pretrained_backbone=True, so without this all 16 ranks race to download
#     convnextv2_base.fcmae_ft_in22k_in1k_384 into a cold shared cache at once.
#     Compute nodes do have outbound network, so it works either way -- this just
#     removes the race. Takes about a minute.
"$RAMPNET_ENV/bin/python" -c "
import timm
timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', pretrained=True, num_classes=0)
print('backbone cached')"

# 2. Launch Run A. --chdir puts every working-directory artefact train.py writes
#    (latest_checkpoint.pth, best_model.pth, runs/, peek_training/, logs/) on the
#    durable volume alongside the checkpoints.
export RUN_DIR=/gscratch/makelab/jonf/rampnet_run_a_84
mkdir -p "$RUN_DIR"/{checkpoints,logs}
export RAMPNET_DATA=/gscratch/scrubbed/jfroehli/rampnet_dataset
export RAMPNET_EPOCHS=8
sbatch --chdir="$RUN_DIR" \
       --export=ALL,RAMPNET_REPO,RAMPNET_ENV,RAMPNET_DATA,RAMPNET_EPOCHS \
       $HOME/RampNet/stage_two/run_train_epoch_curve.slurm
```

Both launchers are committed: `stage_two/run_build_env.slurm` and
`stage_two/run_train_epoch_curve.slurm`. Neither hardcodes a path — repo, env, data root and epoch
count all come from the environment, with defaults documented in the file headers.

### Artefacts

```
/gscratch/makelab/jonf/rampnet_run_a_84/
├── checkpoints/epoch_N_step_S.pth   # the product: 8 files, ~1.07 GB each
├── latest_checkpoint.pth            # resume state, rewritten every 1,000 steps
├── best_model.pth                   # auto-val-selected, i.e. what the committed rule would ship
├── runs/experiment_1/               # tensorboard: train step loss + per-epoch val loss
├── peek_training/                   # per-epoch heatmap overlay on a random val pano
└── logs/run_a_84_<jobid>.{out,err}
```

Each checkpoint carries `model_state_dict` + Adam state + `current_val_loss`, so the auto-label arm
of the curve is readable from the checkpoints themselves as well as from TensorBoard.

Budget note: ~10 GB total. `/gscratch/makelab` was at 88% (905 GB / 1 TB, 119 GB free) at launch.
That is fine for Run A but **Run B at 30–60 epochs would want 32–64 GB**, which needs checking
against the quota before it is submitted, not after.

## Scoring the curve

Evaluation runs on makelab2, not on the cluster: it has no batch-regime constraint, the A40 handles
it, and it keeps the sweep off the queue-contended partition.

`stage_two/evaluate.py` already does everything the pre-registration asks for, with **no code
change**. One run per checkpoint produces both required numbers:

```bash
python evaluate.py --checkpoint <run_dir>/checkpoints/epoch_N_step_S.pth \
                   --dataset manual --threshold 0.0 --no-tta \
                   --results-dir evaluation_results/run_a_84/epoch_N
```

- `--threshold 0.0` keeps every peak, so the full curve is swept.
  `pr_rc_vs_c_data_manual_r0.022_pt0.0.csv` gives precision and recall at every unique confidence —
  **F1@0.30 and max-F1 both fall out of that one file**, which is exactly the pairing the amendment
  requires.
- `--no-tta` is deliberate and is **not** the script default. The default is flip-TTA "as in the
  paper", but #78 measured it as not worth 2× GPU (+0.9 R / −2.4 P after the operating-point drop,
  F1 down on 4 of 5 US splits) and the #54 protocol headline is single-pass. Leaving the default on
  would silently score a different protocol than every number it will be compared against.

**Gotcha to respect: use a separate `--results-dir` per epoch.** The heatmap cache is keyed by
checkpoint fingerprint (the #24 fix), so caching is safe — but the *output* filenames are keyed by
dataset and params only, not by checkpoint. Eight epochs into one results dir silently overwrite
each other and leave one epoch's numbers wearing the whole run's name.

## Results

*Not yet available — the run was launched 2026-08-15.*

| epoch | auto-label val loss | `manual_gold` F1@0.30 | `manual_gold` max-F1 |
| ---: | ---: | ---: | ---: |
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |

For reference, the paper run's auto-label val loss over the same range: .000520, .000478, .000463,
.000466, **.000458**, .000468, .000470, .000473. If Run A's auto-val arm tracks that within noise,
it is evidence the replicate landed; if it does not, the June-2025 code difference is larger than
assumed and every reading below has to be qualified.

One consequence worth stating before the numbers arrive rather than after: **if the outcome is
"select on human F1", then selecting on `manual_gold` stops it being a clean benchmark.** The fix
needs a held-out human split for selection. Noting the constraint here so it is not discovered
downstream.
