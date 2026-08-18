# The 8-epoch cosine rung: does annealing help RampNet at all? (#135)

**Status: PRE-REGISTERED 2026-08-18, before launch. Nothing below has been run.** The decision
rule, the comparison, and what each outcome means are fixed here so they cannot be chosen after
the numbers arrive. Results will be appended to this file, not substituted into it.

## Why this run, and why now

The #84 amendment recorded an 8-epoch cosine arm as an option, "recorded not scheduled". Two
measurements since have made it the right next step rather than a nice-to-have.

**1. Constant LR does not just flatten — it declines.** `docs/stage2_run_b_power_135.md` re-read
Run A's curve with a paired instrument and found epochs 7 and 8 measurably *below* the plateau
(3 vs 7 at z = 3.9, 3 vs 8 at 3.0, 6 vs 7 at 2.8). The shape is:

> epoch 1 clearly low → epochs 2–6 a genuine plateau → epochs 7–8 measurably declining.

Drifting back up after settling is the classic signature of a learning rate left too high to
settle with. That is precisely what a decay fixes — and it means **there is already something
for an 8-epoch anneal to act on.** Before this measurement the curve looked flat, and an
8-epoch cosine would have had nothing to demonstrate against.

**2. The benchmark can read the answer.** The same document measured the paired minimum
detectable effect on `manual_gold` at **0.0063 max-F1** for the epoch separations involved, from
28 real checkpoint pairs. An effect the size of #51's annealed tail (+0.024 on this benchmark)
would be unmissable; even a third of it is readable.

**3. It is the only comparison that attributes.** Run B as specified is 30–60 epochs *and*
cosine, so B − A confounds length with schedule — the #84 amendment says so explicitly. This rung
holds the budget fixed at Run A's 8 epochs and changes only the schedule, so its difference is
attributable to the decay and nothing else.

It is **not** a substitute for Run B. An 8-epoch cosine decays fast and cannot show a benefit
that only appears with length. It answers the narrower question — *does annealing do anything
for this model at all* — for about a quarter of the cost.

## What is held fixed, and the one thing that is not

Everything in Run A's recipe: 4 nodes × 4 GPUs ⇒ world size 16, `batch_size=1` per rank ⇒ global
batch 16, **peak** lr 1e-5 (`--preset scratch`), seed 42, same data, same splits,
ImageNet-initialised backbone, **8 epochs**.

**The only change is `--lr-schedule cosine`:** the learning rate decays from 1e-5 to 0 over the
run's 75,024 steps, as `0.5 · lr · (1 + cos(π · step / total))`.

**No warmup.** Adding one would be a second change, and the rung exists to isolate the decay.
Run A had none and was stable from step 0, so there is no reason to introduce one here.

Because the seed and data order are identical to Run A's, the two runs share their
initialisation and their batch sequence exactly; they diverge only through the optimiser step
size. This is the cleanest isolation available.

### The schedule is stateless, and that is the design

`stage_two/train.py`'s `lr_at_step` computes the rate from the **absolute step index** rather
than from a scheduler object. This matters because Stage 2 runs on klone's preemptible
`ckpt-all` partition and resumes from `latest_checkpoint.pth` — Run A was requeued twice.

A stateful scheduler (`CosineAnnealingLR` and friends keep `last_epoch` internally) restarts its
decay from the peak on every requeue unless its state is *also* saved and restored. That turns a
smooth cosine into a sawtooth, and **the failure is silent**: the job completes, the loss curve
looks plausible, and the schedule under test was never applied. Reading the rate from
`global_step` — already checkpointed, already broadcast to every rank, already correct across
resume — makes that impossible by construction rather than by remembering to serialize one more
field.

`train.py` now also logs the per-step learning rate to TensorBoard. **Read that scalar before
trusting any result from this run**; a sawtooth is invisible in the loss curve and obvious there.
`tests/test_train_lr_schedule.py` pins the property directly by simulating a run chopped into
segments by requeues and asserting the LR sequence is identical to an uninterrupted one.

The default remains `--lr-schedule constant`, so every existing invocation, including a re-run of
Run A or the paper recipe, is unaffected. A test asserts that default.

## Pre-registered decision rule

All comparisons on `manual_gold`, **single-pass** (#54 protocol), on the **calibration-free
max-F1** column, **paired** against Run A's checkpoints via
`scripts/analysis/benchmark_power_135.py`, with each pair read against its own bootstrap standard
error (B = 5,000, seed 42). Significance is |Δ| / s.e. ≥ 1.96.

**Primary test — does the schedule help at matched budget?**

> Δ = max-F1(cosine, epoch 8) − max-F1(Run A, epoch 8), paired.

| outcome | reading |
| :--- | :--- |
| Δ > 0 and significant | **Annealing helps.** The released recipe leaves accuracy on the table for want of a schedule, not for want of data or length. |
| not significant | **No effect at 8 epochs.** Either annealing does not help this model, or 8 epochs is too short a horizon for it to show. The secondary test discriminates. |
| Δ < 0 and significant | **Annealing hurts at this budget.** A real and publishable negative result. |

**Secondary test — is the decline arrested?** This is the mechanism, and it can fire even when
the primary is a tie.

> Run A declines from its own epoch 3 to epoch 8 by −0.0066 max-F1 (z = 3.0). Compute the
> cosine arm's own epoch 3 → epoch 8 difference.

If the cosine arm's epoch 3 → 8 difference is ≥ 0, or its decline is significantly smaller than
Run A's, the decay is doing what the mechanism predicts even if the endpoint has not yet moved.

**Gate on Run B (30 epochs), decided now:**

- **Proceed** if the primary is positive-and-significant, **or** the primary is a tie but the
  secondary confirms the decline is arrested.
- **Do not proceed** if the primary is significantly negative **and** the secondary shows no
  arrest.
- A tie on both is **not** an automatic cancellation; it is a judgment call about spending
  1,800 GPU-hours on a mechanism that showed nothing at a quarter of the cost, and it will be
  recorded as a judgment call rather than dressed as a rule.

**Reported alongside, not as the decision:** F1@0.30 and AP per epoch, the auto-label validation
loss curve (does the schedule move the signal `train.py` selects on?), and the full 8-epoch
paired matrix against Run A so the trajectories can be compared epoch by epoch rather than only
at the endpoint.

### What this run cannot settle

**Seed variance is unmeasured and this is n = 1.** Identical seed and data order mean the two
runs differ only in schedule *as a trajectory*, but a single pair cannot say how much of any
observed difference a different seed would have produced anyway. A difference below ~0.01 max-F1
is therefore **measured but not attributable** without a seed control, which doubles the cost.
This constraint is inherited from `docs/stage2_run_b_power_135.md` and is not resolved here.

## Cost

| | GPU-h | compute | calendar (measured factor) |
| :--- | ---: | ---: | ---: |
| **This rung (8 epochs, cosine)** | **~480** | **~30 h** | **~34 h** |
| Run B (30 epochs, cosine) | ~1,800 | ~4.7 days | ~5–7 days |
| Run B (60 epochs, cosine) | ~3,600 | ~9.4 days | ~11–15 days |

Free in money: `ckpt-all` is klone's scavenger partition. The calendar figure is Run A's
*measured* wall-clock for the identical shape (34 h including two requeues), not the ×1.67
preemption estimate, which was pessimistic for that run.

## Exact commands, in order

The env and the staged dataset already exist from Run A; only the run directory is new. **It
must not be Run A's** — `train.py` writes `latest_checkpoint.pth`, `best_model.pth`,
`runs/experiment_1` and `checkpoints/` relative to the working directory, so pointing this at
Run A's directory would resume Run A's weights and overwrite its artefacts.

```bash
# 0. Sync the repo to klone (code only; the dataset already lives on scratch).
rsync -av --exclude .venv --exclude .model_cache --exclude 'benchmark/*/panos' \
      --exclude 'benchmark/*/gallery' --exclude view_dump --exclude dataset \
      --exclude runs --exclude '*.pt' RampNet/ klone:~/RampNet/

# 1. Launch. Same env, same data, new run dir.
export RAMPNET_REPO=$HOME/RampNet
export RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2
export RAMPNET_DATA=/gscratch/scrubbed/jfroehli/rampnet_dataset
export RAMPNET_EPOCHS=8
export RUN_DIR=/gscratch/makelab/jonf/rampnet_cosine_rung_135
mkdir -p "$RUN_DIR"/{checkpoints,logs}
sbatch --chdir="$RUN_DIR" \
       --export=ALL,RAMPNET_REPO,RAMPNET_ENV,RAMPNET_DATA,RAMPNET_EPOCHS \
       $HOME/RampNet/stage_two/run_train_cosine_rung.slurm
```

Scoring, once the eight checkpoints exist — identical to Run A's path, so the numbers are
comparable by construction:

```bash
# On makelab2 (A40), one eval per checkpoint, ~12.5 min each.
python evaluate.py --checkpoint <run_dir>/checkpoints/epoch_N_step_S.pth \
                   --dataset manual --threshold 0.0 --no-tta \
                   --data-root <staged test split> --manual-labels ../manual_labels \
                   --results-dir evaluation_results/cosine_rung_135/epoch_N

# Then recover per-pano detections from the heatmap cache the eval leaves behind
# (CPU only, no GPU) and run the paired comparison against Run A.
python scripts/analysis/dump_peaks_from_cache.py \
    --cache-dir <run_dir>/evaluate_cache --verify
python scripts/analysis/benchmark_power_135.py --splits manual_gold
```

Storage: ~10 GB, the same as Run A. `/gscratch/makelab` had ~348 GB free and 823,623 of
1,000,000 inodes as of 2026-08-18, both measured with `du`/`find` rather than read from
`hyakstorage`, whose report is cached and lags by hours.

## Launch record

| | job | outcome |
| :--- | :--- | :--- |
| cosine rung | *(pending)* | |

## Results

*Not yet run. This section will be appended to, and the pre-registration above left as written.*
