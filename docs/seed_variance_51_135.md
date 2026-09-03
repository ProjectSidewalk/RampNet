# Seed variance: the number both #51 and #135 are now blocked on

**Status: PRE-REGISTERED 2026-09-03, before any replicate finished. The reading rules
below were written first, so the interpretation cannot be chosen after seeing the
numbers — the same discipline as the #71 checkpoint-selection protocol and Tillicum
measurement 3.**

## Why this is the binding number

Two independent lines of work arrived at the same wall from opposite sides.

**#51.** [`operating_point_parity_51.md`](operating_point_parity_51.md) found that the
published RampNet-vs-YOLO gap was mostly an operating-point artifact: at matched
operating points the residual is **0.039 F1**, not 0.252 or 0.160. Every arm in that
comparison is **one seed** — `seed: 0`, the Ultralytics default, in all eight
`args.yaml` files. #51's own rule is that differences under ~0.02 should not be read,
and 0.039 is close enough to that floor that the architecture claim cannot be
adjudicated at n=1.

**#135.** The power analysis measured a paired MDE of 0.0063 on `manual_gold` and then
said plainly that the binding limit is **unmeasured seed variance, n=1** — Stage 2's
`train.py` hardcoded `torch.manual_seed(42)` with no flag.

So the same missing measurement gates both. Everything downstream inherits it: with no
noise floor, no future RampNet 2.0 improvement can be called real either.

## What is being run

Two campaigns, launched 2026-09-03.

### Campaign A — YOLO seed variance (#51), on Tillicum

Three fresh replicates of **`y11x_tiles`**, the strongest YOLO leg and the one carrying
the 0.039. Config identical to the pre-registered #51 protocol and to the existing arm's
`args.yaml` in every respect except the seed:

| | value | source |
|---|---|---|
| base | `yolo11x.pt` | `runs/y11x_tiles/args.yaml` |
| data | tiles, 557,413 train / 161,002 val | verified on Tillicum, exact match to the klone record |
| `imgsz` / `batch` | 1024 / 12 | as-run |
| `epochs` / `patience` | 60 / 20 | as-run — see "read at a matched epoch" below |
| `optimizer` | `auto` (resolves to `MuSGD`) | as-run |
| `seed` | **1, 2, 3** | the only variable |
| `save_period` | **1** | deviation, stated below |

**Why Tillicum and not free klone.** The tiles arm is storage-bound on klone — it
consumes 8.5 MB/s against a filesystem measured at 8.3–11.8 MB/s, i.e. it sits *on* the
ceiling ([`tillicum.md`](tillicum.md)). Three concurrent replicates there would contend
for the wall itself and each would run slower than the 7.06–7.38 h/epoch a single arm
saw. On Tillicum the same arm uses 4% of available bandwidth and is genuinely GPU-bound
at 3.0 h/epoch, so three replicates run concurrently without interfering.

**The one deliberate deviation: `save_period=1`.** The #51 arms ran `save_period: -1`,
which is exactly why the epoch-curve follow-up had to be retracted — no per-epoch
weights exist for any arm and they cannot be recovered. Keeping every epoch costs ~150
MB × 44 × 3 ≈ 20 GB against a 1 TB allocation and buys back the budget analysis that is
currently foreclosed. It does not affect training.

### Campaign B — RampNet seed variance (#135), on klone

Three replicates of the committed Stage 2 recipe (1 epoch / 9,378 steps, constant lr
1e-5, global batch 16, selection on auto-label val loss) at seeds **1, 2, 3**, via
`stage_two/run_train_seed.slurm`. Free, preemptable, resumed by `--requeue` plus
`train.py`'s own `latest_checkpoint.pth`.

**These replicates are not compared against the released checkpoint.** The released
model is a hand-copied epoch-1 checkpoint chosen by neither the paper's rule nor #84's
(see [`stage2_epoch_curve_84.md`](stage2_epoch_curve_84.md)), so its provenance differs
from a clean run of the committed recipe. The SD is computed over the three replicates
alone.

## The reading, fixed in advance

The statistic for both campaigns is the **macro-mean F1 over the seven pooled US
splits**, each replicate read at its own uniform threshold selected on the `sao_paulo`
dev split — i.e. exactly the parity protocol, applied per replicate. Selection never
touches a reported split.

**Read at a matched epoch.** The existing `y11x_tiles` arm's `best.pt` is from ~ep44,
the point it reached before it ran out of free GPU. Each replicate is therefore read at
its **best-val epoch among epochs ≤ 44**, which is what "best.pt as-saved" meant for
that arm. Replicates run the full `epochs=60` schedule so the LR curve is identical — a
44-epoch schedule is *not* the first 44 epochs of a 60-epoch one, and truncating the
schedule instead of the run would have confounded the comparison.

**Decision rule for #51**, on the sample SD `s` of the three Campaign A replicates:

| `s` | reading |
|---|---|
| **≤ 0.010** | 0.039 is ≈4σ. The architecture advantage is small but real; #51 closes with the gap restated at 0.039 ± the measured spread. |
| **≥ 0.020** | 0.039 is inside 2σ. #51 closes with **"a supervised YOLO baseline is statistically indistinguishable from RampNet at matched operating points"** — and the `manual_gold` loss stands as the sharper statement. |
| **0.010–0.020** | Ambiguous. Report the interval, claim neither, and combine with Campaign B's SD for a two-sample test rather than asserting from A alone. |

**This can make our own headline smaller, and that outcome is accepted in advance.**
The ≥0.020 branch is a live possibility, not a formality.

## Stated limitations

- **n=3 gives a wide interval on the SD itself** — with 2 degrees of freedom the 95% CI
  on σ spans roughly 0.5σ̂ to 3.7σ̂. A fourth and fifth replicate are a cheap extension
  (~$119 each) and are the pre-registered response if the result lands in the ambiguous
  band. They are not being run up front.
- **Campaign A's replicates are Tillicum H200; the existing `seed: 0` arm is klone
  L40S.** The SD is computed over the three same-hardware replicates only. The seed-0
  arm is a separate cross-hardware check, not a fourth sample — mixing them would
  conflate seed with kernel and hardware differences.
- **Only `y11x_tiles`.** `y11l_*`, `y26_*` and the pano arms are not replicated; the SD
  measured here is not automatically theirs.
- **Campaign B's calendar is unbounded.** `ckpt-all`'s duty cycle was 3.9% in 2026-08
  (#135), so a replicate can sit pending for days. It costs nothing, but it may not
  finish alongside Campaign A. If it stalls, the fallback is Tillicum at ~$50/replicate;
  that would be recorded here as a venue change, with the hardware caveat above.
- **Seed variance is not the same as training-run variance.** Preemption, requeue
  boundaries and non-deterministic kernels also move a result. Ultralytics sets
  `deterministic=True` by default, which removes some of that for Campaign A; Campaign B
  has no such guarantee and its spread is therefore an upper bound on seed effect alone.

## Reproducing

Both launchers take the seed as an environment variable and record it in the job log.

```bash
# Campaign A, per replicate (Tillicum). CHAIN covers the 24 h normal-QoS ceiling.
SEED=1 NAME=y11x_tiles_s1 YOLO_CKPT=yolo11x.pt \
  YOLO_DATA=/gpfs/scrubbed/$USER/yolo/tiles/data.yaml \
  YOLO_IMGSZ=1024 BATCH=12 EPOCHS=60 SAVE_PERIOD=1 CHAIN=5 \
  sbatch scripts/model_comparison/run_yolo_train_tillicum.slurm

# Campaign B, per replicate (klone).
SEED=1 sbatch stage_two/run_train_seed.slurm
```

The seed plumbing itself is unit-tested in `tests/test_seeding.py` — including that the
default reproduces published runs exactly (`sampler_seed_for(42) == 0`) and that the
Tillicum launcher's positional argument list still lines up with its unpack, an
off-by-one that would silently train at the wrong seed.
