# What Stage 2 training actually costs

**One epoch is 3.5 hours on 16 GPUs, not 36.** Measured 2026-08-03 from the paper run's own
TensorBoard events, which survived on a lab scratch volume and are now committed here.

This replaces the estimate in [`curb_ramp_data_sourcing.md` §7](curb_ramp_data_sourcing.md#7-retrain-cost),
which was ~10x too high. That estimate extrapolated the README's *"will take a very long time
(> 24 hours)"* as a per-epoch figure; the ">24 hours" in fact describes the **whole ~12-epoch,
preemption-riddled run**.

Everything below is reproduced by one committed command:

```bash
python scripts/analysis/stage2_train_cost.py --records 500000
```

## The measurement

| | |
| :--- | ---: |
| Median step time (rank 0) | **1.341 s** (p25 1.339 / p75 1.345, n = 119,902) |
| Global batch | 16 panoramas (16 GPUs x batch 1) |
| Throughput | **11.93 panos/s** |
| Steps per epoch | **9,378** (150,063 train panos / 16) |
| **Wall-clock per epoch** | **3.49 h** (+ ~0.3 h validation) |
| **GPU-hours per epoch** | **~56** |

The interquartile band is 6 ms wide across 119,902 samples, which is the signature of a hard
**I/O bound**, not a compute bound: ~7.7 TFLOPs of forward+backward per panorama against an A100
is roughly **3% MFU**. The cost is 8.4 MP JPEG decode + resize on ~3 CPU cores per rank
(`--cpus-per-task=12`, 4 ranks per node, `num_workers=4`). Two consequences:

- **cost scales with panoramas, not with label count** — which is what makes the projection below
  meaningful, and what makes "number of labels" the wrong unit for a training-time question;
- **it is fixable**. Pre-resized panoramas, more dataloader CPU, or a shard format that avoids
  per-file decode could cut it several-fold. Nothing here is a floor imposed by the model.

## The run was ~12 epochs, and the released model is epoch 1

Both statements are true and the repo previously recorded only the second.

Validation scalars land on exact multiples of 9,378, so the epoch length is confirmed
independently of the step-rate arithmetic. The run reached step 112,434 = **11.99 epochs** before
being cancelled, over **44.7 h of active compute** and **74.6 h of calendar time** across 15
preemptions on the `ckpt-all` scavenger partition — a **x1.67 overhead**, each restart rewinding to
the last 1000-step checkpoint.

| epoch | 1 | 2 | 3 | 4 | **5** | 6 | 7 | 8 | 9 | 10 | 11 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| auto-label val loss | .000520 | .000478 | .000463 | .000466 | **.000458** | .000468 | .000470 | .000473 | .000484 | .000487 | .000487 |

**Auto-label validation loss bottoms at epoch 5 (-12% against epoch 1), then rises monotonically
through epoch 11.** So at 1 epoch the model is *not* converged — but the available headroom is
small and it reverses.

Nevertheless `best_model.pth` is **byte-identical** (`cmp`) to `checkpoints/epoch_1_step_9378.pth`,
copied back by hand on 2025-06-21 13:02 over the val-best file the training script had saved
automatically. **The released RampNet model is the epoch-1 checkpoint**, exactly as the README
describes. What the README omits is that 11 further epochs were run and discarded.

This is **half of issue #84's epoch curve, already run**: the auto-label half exists above at zero
cost. The half that is still missing is the **human** one — evaluating epochs 1/3/5/8 against
`manual_gold`. If auto-label val loss keeps improving while human F1 does not, that is the Stage 1
label ceiling, and it is the whole question #84 was filed to answer. **The epoch-N weights no
longer exist** (a 2025-07-11 cleanup left only epoch 1), so that half needs a re-run.

## Projection to a 500,000-record corpus

**"500k" counts government ramp records, not labels.** RampNet 1.0 is 278,544 records ->
214,376 panoramas carrying **849,895 point labels** (~3 views per ramp, ~4 labels per panorama).
500k records is **1.80x** the records and would imply ~1.5M labels.

| | RampNet 1.0 | at 500k records |
| :--- | ---: | ---: |
| Government ramp records | 278,544 | 500,000 (**1.80x**) |
| Panoramas (x0.770 per record) | 214,376 | ~384,800 |
| Train split (70.0%) | 150,063 | ~269,400 |
| Steps per epoch (/16 GPUs) | 9,378 | ~16,840 |
| **Wall-clock per epoch** | 3.49 h | **6.27 h** (~100 GPU-h) |

| Schedule | Compute (16 GPUs) | Calendar on preemptible `ckpt` |
| :--- | ---: | ---: |
| 1 epoch (the released recipe) | **6.3 h** | ~10.5 h |
| 5 epochs (1.0's auto-val optimum) | 31.4 h | ~2.2 d |
| 12 epochs (what 1.0 actually ran) | 75.2 h | ~5.2 d |

**Stage 2 retraining is an overnight job.** The epoch count is worth ~12x what the data doubling
is worth, so the schedule decision dominates the corpus decision for wall-clock purposes.

**Stage 1 generation remains the long pole and is still unmeasured**: ~170k *new* panoramas at 32
tiles each is ~5.5M tile requests against Google's undocumented endpoints, rate-limit bound, with
100 h allocated in `run_download_dataset.slurm`, plus ~800 GB of storage. §7's "about a week of
wall-clock" stands — but it is essentially all Stage 1. The crop model needs no retrain.

Two caveats on the projection: the 0.770 panoramas/record ratio and the 70.0% train share are
RampNet 1.0's and will shift with city density (denser, NYC-like inventories put more ramps in each
panorama); and at a fixed 1 epoch, 1.8x the data silently becomes a 1.8x longer schedule, which is
a change in training regime, not only in cost.

## Provenance, and what is not replicable

The inputs are **18 TensorBoard event files, committed at `docs/data/rampnet1_stage2_run/`** with a
`SHA256SUMS` manifest. They were rescued 2026-08-03 from
`/gscratch/makelab/jsomeara/RampNet/stage_two/runs/experiment_1` on the UW Hyak cluster, where they
were the only copy. `scripts/analysis/stage2_train_cost.py` parses the TFRecord and protobuf
framing directly, so replication needs **no TensorFlow, no TensorBoard, no GPU and no network**:

```bash
python scripts/analysis/stage2_train_cost.py                  # the measurement
python scripts/analysis/stage2_train_cost.py --records 500000 # the projection
python scripts/analysis/stage2_train_cost.py --verify         # re-check SHA256SUMS
```

Stated gaps, deliberately:

- **The code that produced this run is not in git.** The public history begins with a squashed
  "Initial Commit" dated 2025-07-15, a month after the run, already carrying `num_epochs = 1`. The
  run-time value was >= 12 and is unrecoverable.
- **The Slurm record is incomplete.** `sacct -u jsomeara -S 2025-06-16 -E 2025-06-22` shows only
  three `train_curb_ramp_detector` jobs of 2-3 h each, because `--requeue` overwrites the accounting
  record on each requeue; the event files are the only evidence of the other 12 segments. (The
  event-file hostnames — g3050, g3060, g3082 — are what tie the logs to those 4-node jobs.)
- **The hardware claim is looser than the README's.** README says 16x L40s; the surviving Slurm
  records show those incarnations holding **A100s**, under `--constraint='l40s|l40|a40|a100'`.
  Because the run is I/O-bound, GPU type barely moves the step rate — consistent with the observed
  1.34-1.53 s/step across segments on mixed hardware.
- **Per-epoch weights are gone**, so the human half of the epoch curve cannot be recovered from
  this run; it needs a re-run (#84).
