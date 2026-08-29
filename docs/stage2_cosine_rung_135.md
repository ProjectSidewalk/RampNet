# The 8-epoch cosine rung: does annealing help RampNet at all? (#135)

**Status: PRE-REGISTERED 2026-08-18 before launch; training COMPLETE 2026-08-21; `manual_gold` scored 2026-08-29 — TIED at every epoch.** Everything above Results is as written on 2026-08-18 and has not been edited. The decision
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
`ckpt-all` partition and resumes from `latest_checkpoint.pth` — Run A was requeued five
times across its two job ids, measured with `sacct -D` (plain `sacct` shows only the last
incarnation, which is why the earlier count here said twice).

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
| cosine rung | **`38640313`** | submitted 2026-08-18 16:51 PDT, `ckpt-all`, 4 nodes × 4 GPUs, world size 16, fresh start (no resume file present at submit). PENDING on `(Resources)` at submission. |

Pre-flight, verified rather than assumed:

- klone's `~/RampNet/stage_two/train.py` was **byte-identical to `main`** before the upload
  (sha256 `ccce228b…`), so the copy that runs is `main` plus the scheduler and nothing else. Its
  hash on klone after upload is `051e3256…`, matching this branch exactly.
- The launcher transferred as `Bourne-Again shell script, ASCII text executable` — no CRLF, which
  would have failed with a bad-interpreter error at launch.
- Dataset intact at `/gscratch/scrubbed/jfroehli/rampnet_dataset`: 300,126 entries under `train/`
  = 150,063 panorama+JSON pairs, matching Run A's 9,378 steps/epoch at world size 16. Counted,
  not stat'ed — the directory is older than `scrubbed`'s purge window.
- `/gscratch/makelab` had 270 GB free; this run needs ~10 GB.
- Env reused from Run A at `/gscratch/makelab/jonf/envs/sidewalkcv2` (the durable copy, job
  `38566410`), so the environment is identical to the arm being compared against.

**First check once it starts, before it is left to run:** read the `LR` scalar in
`runs/experiment_1`. It must start at 1e-5 and fall smoothly. A sawtooth means a requeue reset
the schedule, which would invalidate the run and is invisible in the loss curve.

## Results

**Status: COMPLETE.** Training finished 2026-08-21, the applied LR schedule is verified over 100%
of the run, and `manual_gold` was scored 2026-08-29: **the two arms are tied at every epoch**,
largest delta 0.0042 against #138's measured paired MDE of 0.0063.

### The run completed, after 21 restarts, without the livelock fix

`38640313` finished at **2026-08-21T19:07:16**. It was never cancelled and never resubmitted: the
requeue livelock diagnosed on 2026-08-20 **broke on its own** when `ckpt-all` capacity freed up and
incarnations went from minutes to eight-hour slices.

This is recorded as a caveat, not a success. The fix in `fa02b37` was **never applied to this job** —
the `train.py` that ran carries no `ResumeSkipSampler` and still has `checkpoint_interval_steps = 1000`
hardcoded. The rung finished on capacity weather. That does not retire the fix; it says the livelock
is a property of the partition regime, which will recur, and that a run whose completion depends on
luck is not one to schedule deliberately.

| | value |
| :--- | ---: |
| restarts | 21 |
| aggregate wall-clock | 35.06 h |
| **GPU-hours (× 16)** | **560.9** (free, `ckpt-all`) |
| epoch checkpoints written | 8 of 8 |
| final validation loss | 0.0005 |

Checkpoints at `/gscratch/makelab/jonf/rampnet_cosine_rung_135/checkpoints/`,
`epoch_1_step_9378.pth` … `epoch_8_step_75024.pth`.

### The pre-registered LR check passes over the whole run

The pre-registration above names one check as the thing that licenses reading any of this: the `LR`
scalar must fall smoothly, because a requeue that reset the schedule would be invisible in the loss
curve. Run against the complete event set with the **as-run** `train.py` as the lift source (not
this branch's copy, which `fa02b37` changed):

```
11 incarnation(s); 10 resume boundaries
Merged: 75024 unique steps, 1-75024 of 75024 (100.00% of the run)
Non-decreasing violations: 0
Max |logged - lr_at_step(step-1)|: 4.547e-13 at step 15339  (tolerance 1.0e-11)
PASS
```

Every boundary lands at the rate its step index predicts — `0.993003`, `0.964910`, `0.693931`,
`0.654700`, `0.165709`, `0.000460` × peak — and never at `1.000000`, which is the sawtooth
signature. The cosine reaches `4.38e-15` at step 75,024, the pre-registered floor of zero.
**The schedule under test is the schedule that was applied, over the entire run rather than a
sample.** (Eleven incarnations against 21 restarts: ten allocations logged no training step at all.)

### Auto-label validation loss, budget-matched against Run A

Run A (`docs/stage2_epoch_curve_84.md`) and this rung differ in exactly one thing. Both are seed 42,
world size 16, global batch 16, same data, same splits, same ImageNet-initialised backbone.

| epoch | Run A (constant 1e-5) | cosine rung | cosine better by |
| ---: | ---: | ---: | ---: |
| 1 | 0.00052194 | 0.00052057 | +0.263% |
| 2 | 0.00048067 | 0.00048145 | −0.162% |
| 3 | 0.00046341 | 0.00046519 | −0.383% |
| 4 | 0.00046485 | 0.00046191 | +0.633% |
| **5** | **0.00045976** | **0.00045534** | **+0.962%** |
| 6 | 0.00046855 | 0.00045752 | +2.355% |
| 7 | 0.00047676 | 0.00046077 | +3.354% |
| 8 | 0.00048100 | 0.00046186 | +3.980% |

Committed at `docs/data/stage2_cosine_rung_135.csv`, and re-derivable from a clean clone with
no cluster access — both runs' event files are in the repo:

```bash
python scripts/analysis/stage2_epoch_curve.py \
    --events-dir stage_two/cosine_rung_135_events \
    --paper-events-dir stage_two/run_a_84_events \
    --curve-label cosine_rung --reference-label run_a \
    --out-csv docs/data/stage2_cosine_rung_135.csv
```

The percentages are computed from the raw float32 scalars in the events, not from the
8-decimal values in the table, so hand-arithmetic off the printed column can differ in the
third decimal. That is the same trap #84 recorded when a typed delta read 1.69% against a
true 1.755%, which is why these are script-derived.

Two readings:

1. **Annealing does not move the optimum.** Both arms bottom at **epoch 5**. The rung does not
   relocate the minimum the #84 amendment identified, and does not turn epochs 7–8 into the best
   checkpoints.
2. **Annealing damps the post-minimum decline.** Run A rises **+4.62%** from its own minimum to
   epoch 8; the rung rises **+1.43%**. About 70% of the late-epoch degradation is schedule rather
   than overfitting.

The reason to believe (2) is the shape, not the size: the arms are indistinguishable through
epochs 1–3, where cosine is still between 1.00× and 0.69× of peak, and separate monotonically from
epoch 4 as the anneal bites. That is a schedule effect, not seed noise — though with n=1 per arm,
seed variance remains the unmeasured term #138 identified as the binding limit.

### What is not answered yet, and a prediction on the record

**Auto-label val loss is the signal `train.py` selects on, not the deciding metric.** The pre-registered
question is `manual_gold`. That sweep is running on makelab2 with the protocol copied verbatim from
`run_a_84/run_evals.sh` — `--threshold 0.0 --no-tta`, a separate `--results-dir` per epoch, and the
eval host's repo pinned at `dc7450e`, the commit Run A was scored under, so the comparison isolates
the schedule rather than confounding it with the #140 matcher change (#148).

Recorded before the numbers arrive, so it can be wrong: #84 measured a 13.5% auto-val improvement
buying ~0.009 F1 on `manual_gold`. Scaled, this rung's 0.961% advantage at the shared optimum maps
to well under #138's measured paired MDE of **0.0063**, and its 3.979% advantage at epoch 8 to about
0.005. **The expected result is "tied at every epoch."** If that holds, the finding is that annealing
buys a real, reproducible, mechanistically-consistent improvement in the *selection signal* that does
not survive translation to human-labelled F1 — which closes the Run B gate on a measurement rather
than an extrapolation.

### `manual_gold`: tied at every epoch, as predicted

Scored 2026-08-29 on makelab2, one `evaluate.py` run per checkpoint, protocol copied verbatim
from `run_a_84/run_evals.sh` and the repo pinned at `dc7450e` — the commit Run A was scored
under — so the comparison isolates the schedule rather than confounding it with the #140 matcher
change (#148). All 8 checkpoints were sha256-verified against the klone originals before
transfer, and `evaluate.py` stamps each checkpoint's fingerprint into its own metrics file: the
eight `checkpoint_fingerprint` values in `summary.csv` match the eight source hashes, so every
row below ties to specific weights rather than to a directory name.

| epoch | Run A F1@0.30 | cosine F1@0.30 | Δ | Run A max-F1 | cosine max-F1 | Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9052 | 0.9054 | +0.0002 | 0.9064 | 0.9069 | +0.0005 |
| 2 | 0.9120 | 0.9119 | −0.0001 | 0.9165 | 0.9169 | +0.0003 |
| 3 | 0.9128 | 0.9126 | −0.0002 | **0.9191** | **0.9206** | +0.0015 |
| 4 | 0.9131 | **0.9163** | +0.0032 | 0.9171 | 0.9183 | +0.0013 |
| 5 | 0.9143 | 0.9124 | −0.0018 | 0.9179 | 0.9164 | −0.0015 |
| 6 | **0.9161** | 0.9144 | −0.0017 | 0.9165 | 0.9173 | +0.0007 |
| 7 | 0.9103 | 0.9113 | +0.0010 | 0.9110 | 0.9153 | +0.0042 |
| 8 | 0.9088 | 0.9124 | +0.0036 | 0.9124 | 0.9154 | +0.0030 |

**Largest |Δ| on either pre-registered metric: 0.0042, against #138's measured paired MDE of
0.0063. No epoch separates the two arms.** The prediction recorded above, before the numbers
existed, was that the benchmark answer would be "tied at every epoch". It is.

The two arms' F1@0.30 peaks are **0.9161 (Run A, epoch 6)** and **0.9163 (cosine, epoch 4)** —
a difference of 0.0002, which is to say the same number at a different epoch. Both replicate
#84's finding of no resolvable human peak: every epoch from 2 on sits inside the tie bar.

### The finding: a real gain in the selection signal that does not translate

This is the result, and it is a negative one about the *signal*, not about the schedule:

- **Annealing measurably improves auto-label validation loss.** Up to 3.980% at epoch 8,
  monotone in the anneal, and mechanistically consistent — the arms are indistinguishable while
  the cosine is still near peak and separate as it bites. That is not noise.
- **None of it reaches human-labelled F1.** The same checkpoints, scored against 3,919
  human-placed instances, are tied everywhere.

So the ~1% auto-val improvement at the shared optimum buys nothing measurable on the benchmark,
which is consistent with #84's exchange rate (a 13.5% auto-val gain bought ~0.009 F1) and is
what the prediction was scaled from. **Auto-label validation loss is a real optimisation signal
that is only loosely coupled to the thing we care about.**

One directional hint, stated as a hint and not a finding: from its own F1@0.30 peak to epoch 8,
Run A declines **−0.0073** while the cosine arm declines **−0.0039**. That is the late-epoch
damping the auto-val curve shows, surviving into F1 at about half the size — but both numbers
straddle the tie bar, so this is a thing to test at length, not a thing to claim at n=1.

### AP moves where F1 does not, and that is worth recording

AP is **not** a pre-registered metric here and nothing above rests on it, but the two disagree
in a way that would mislead anyone reading only one:

| epoch | Run A AP | cosine AP | Δ AP | Δ max-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.9066 | 0.9158 | **+0.0092** | +0.0042 |
| 8 | 0.9085 | 0.9154 | **+0.0069** | +0.0030 |

At epoch 7 the AP gap is more than twice the max-F1 gap. AP integrates the whole
precision/recall curve while F1 is read at a point, so the natural reading is that annealing
improves the **low-confidence tail** more than the operating region — where a fixed-threshold
metric cannot see it. Under this project's recall-first framing that tail is not worthless, but
it is also not what #54's operating point scores, and quoting the AP delta as the headline would
overstate the result by a factor of two. Recorded so the next person does not have to rediscover
the discrepancy.

### What this does and does not settle for Run B

- **It settles**: at 8 epochs, budget-matched, seed-matched, the LR schedule does not change
  `manual_gold` performance. The mechanism-based argument for the annealed arm — "annealing
  helps, so a longer annealed run should help more" — has no benchmark support at this length.
- **It does not settle**: whether a 30-epoch annealed run helps. That changes length and
  schedule together, which is precisely the confound the #84 amendment flagged, and this rung
  was deliberately built not to answer it.

What the rung does supply is a measured prior for that decision: the schedule is worth ~0.000
F1 at 8 epochs and possibly ~0.003 in late-epoch damping, against #138's 0.0063 detection floor.
A 30-epoch arm would need the effect to grow substantially with length to be resolvable at all.

### Reproducing

The downsampled PR-vs-confidence curves are committed at
`docs/data/cosine_rung_135_manual_gold/` (a few KB each; the full curves are ~4 MB × 8 and the
checkpoints 8.6 GB, neither committable). They re-derive F1 at any threshold to three decimals,
well inside the tie bar, so the table above is checkable without cluster access:

```bash
python scripts/analysis/stage2_manual_gold_curve.py \
    --results-root docs/data/cosine_rung_135_manual_gold --downsampled
```

The checkpoints remain at `/gscratch/makelab/jonf/rampnet_cosine_rung_135/checkpoints/` and on
makelab2 at `/homes/gws/jonf/RampNet/cosine_rung_135/checkpoints/`. Publishing them is a
decision about HF storage, not a technical blocker — the same gap Run A records.
