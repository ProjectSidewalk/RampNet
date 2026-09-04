# Seed variance: the number both #51 and #135 are now blocked on

**Status: PRE-REGISTERED 2026-09-03, before any replicate finished. The reading rules
below were written first, so the interpretation cannot be chosen after seeing the
numbers — the same discipline as the #71 checkpoint-selection protocol and Tillicum
measurement 3.**

**Amended 2026-09-04 in code review, before any replicate was scored** — see
[Amendment 1](#amendment-1-2026-09-04). The original rule divided 0.039 by Campaign A's
SD alone, and gave Campaign B no reading at all. Both are corrected below; the original
text is kept verbatim so the amendment is auditable rather than a silent rewrite.

## Where the inputs live

**`docs/operating_point_parity_51.md` is not on `main` yet.** It, the script that
produces the pre-registered statistic (`scripts/analysis/operating_point_parity_51.py`)
and its artifact (`docs/data/operating_point_parity_51.json`) are all on branch
`fix/yolo-label-cache-rescue-51`, which is open as PR #154 against `main` and is **not**
a parent of this branch. So if this document merges first, `main` carries a
pre-registration whose headline 0.039 and whose scoring tool cannot be found from a
clean clone. Every link below marked † resolves only once #154 lands. Stated here rather
than left to be discovered.

## Why this is the binding number

Two independent lines of work arrived at the same wall from opposite sides.

**#51.** [`operating_point_parity_51.md`](operating_point_parity_51.md)† found that the
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
that arm. "Best-val" is **`metrics/mAP50-95(B)` from the run's own `results.csv`** — this
Ultralytics build selects on mAP50-95 alone, not the 0.1/0.9 fitness blend, measured in
[`tillicum.md`](tillicum.md) (the retarget dry run reproduced the klone arm's ep21
`best_fitness` to five decimals). Naming the column matters: the blend and mAP50-95 do
not always peak at the same epoch.

Replicates are configured `epochs=60` so the LR curve is identical — a 44-epoch schedule
is *not* the first 44 epochs of a 60-epoch one, and truncating the schedule instead of
the run would have confounded the comparison. **They will not actually reach epoch 60**:
`CHAIN=5` buys six 24 h slices = 144 GPU-h, which at 3.0 h/epoch is about 48 epochs
before per-slice restart overhead. That is deliberate and costs nothing — the read is at
≤ 44 — but the runs stop around ep48, not ep60, and the LR curve they follow up to that
point is the 60-epoch one, which is the whole requirement.

**Decision rule for #51**, on the sample SD `s` of the three Campaign A replicates.
*(Kept verbatim as the pre-registration of record. The σ it divides by is corrected, and
its bands made disjoint, in [Amendment 1](#amendment-1-2026-09-04) — apply that version.)*

| `s` | reading |
|---|---|
| **≤ 0.010** | 0.039 is ≈4σ. The architecture advantage is small but real; #51 closes with the gap restated at 0.039 ± the measured spread. |
| **≥ 0.020** | 0.039 is inside 2σ. #51 closes with **"a supervised YOLO baseline is statistically indistinguishable from RampNet at matched operating points"** — and the `manual_gold` loss stands as the sharper statement. |
| **0.010–0.020** | Ambiguous. Report the interval, claim neither, and combine with Campaign B's SD for a two-sample test rather than asserting from A alone. |

**This can make our own headline smaller, and that outcome is accepted in advance.**
The ≥0.020 branch is a live possibility, not a formality.

## Amendment 1 (2026-09-04)

**Ratified by Jon Froehlich, 2026-09-04, before any replicate was scored.** Raised in code review rather than by looking at a result: the correction below is arithmetic, the cut points 0.010 and 0.020 are unchanged from the original table, and Campaign B's rule was written with no number from either campaign in hand.

Raised in code review of PR #155, **before any replicate had finished** — the three
klone replicates had not started (their launcher died at submit time on 2026-09-03) and
the Tillicum replicates were mid-schedule with no epoch scored. No number from either
campaign had been looked at when this was written. The table above is left in place
because it is the pre-registration of record; this section says how it is applied.

### A1.1 The σ is the SD of the *gap*, not of Campaign A alone

The table divides 0.039 by `s`, the SD over Campaign A's three replicates. But 0.039 is
a **difference between two single-seed runs** — RampNet 0.843 minus `y11x_tiles` 0.804
(the parity table†). The standard error of a difference of two independent single draws
is

    s_gap = sqrt(s_A² + s_B²)

not `s_A`. Using `s_A` alone assumes the RampNet side is noiseless at n=1, which is the
assumption #135 explicitly said was unjustified — and is the reason Campaign B is being
run at all. It is also anti-conservative in exactly the direction that flatters us: at
`s_A = 0.010` the table reads "≈4σ", but with `s_B = 0.010` the gap is 2.8σ, with
`s_B = 0.015` it is 2.2σ, and with `s_B = 0.020` it is 1.7σ — inside the band the table
itself calls indistinguishable. The published conclusion could invert with no change to
any input the original rule looked at.

**So the bands are applied to `s_gap`, with the same cut points, and made disjoint:**

| `s_gap` | reading |
|---|---|
| `s_gap < 0.010` | 0.039 is ≳4σ. The architecture advantage is small but real; #51 closes with the gap restated at 0.039 ± the measured spread. |
| `0.010 ≤ s_gap < 0.020` | Ambiguous. Report the interval and the two component SDs, and claim neither. A fourth and fifth Campaign A replicate are the pre-registered response. |
| `s_gap ≥ 0.020` | 0.039 is inside 2σ. #51 closes with **"a supervised YOLO baseline is statistically indistinguishable from RampNet at matched operating points"** — and the `manual_gold` loss stands as the sharper statement. |

The endpoints now belong to exactly one row; in the original all three rows contained
0.010 and 0.020.

**If Campaign B does not deliver `s_B` in time** — its calendar is unbounded, see the
limitations — the table is applied to `s_A` alone and the result is reported as an
**upper bound on significance / lower bound on σ**, in those words, never as the
finding. It is not permitted to quietly become the finding because B was slow.

### A1.2 Campaign B's own reading

The original document gave Campaign B no decision rule, which left half the campaign
free to be interpreted after the fact. Fixed here.

Campaign B's statistic is `s_B`, the sample SD of the macro-mean US7 F1 over its three
replicates, read the same way as Campaign A's. It is read against **#135's measured
paired MDE of 0.0063** on `manual_gold`:

| `s_B` | reading |
|---|---|
| `s_B < 0.0063` | Seed variance is smaller than the paired epoch-to-epoch MDE. #135's MDE stands as the binding limit on Run-A-style comparisons, and #135 closes on that. |
| `s_B ≥ 0.0063` | Seed variance dominates the paired MDE. **Every unpaired single-seed comparison in this repo, including #84's epoch curve read across runs, is limited by `s_B`, not by the MDE** — #135 closes with the noise floor restated at `s_B` and the MDE demoted to the paired case only. |

Either way `s_B` feeds `s_gap` above, and either way it is recorded here with its three
per-replicate numbers, so a fourth replicate can be added to the same sample later.

`s_B` is an **upper** bound on seed effect alone, for the reason already in the
limitations: klone `ckpt-all` preempts, and a requeued replicate resumes from
`latest_checkpoint.pth` without restoring the augmentation RNG stream, so requeue
boundaries contribute to the spread. If the campaign lands in a band whose edge is
within that uncertainty, say so rather than picking the side.

### A1.3 How each campaign gets scored

Neither campaign's path from artifact to statistic was written down. It is not a
one-liner and the gaps are real:

**Campaign A.** For each replicate: pick the epoch checkpoint with the highest
`metrics/mAP50-95(B)` at epoch ≤ 44 from `runs/<name>/results.csv`, run the YOLO
detector over the eight splits (US7 + `sao_paulo`) at the 0.05 score floor, and write a
sweep file in the format of `docs/data/yolo_geometry_51/*.txt`. Then
`scripts/analysis/operating_point_parity_51.py`† must be pointed at the three new legs
— **its leg list is hardcoded**, so this needs a code change, not just new inputs. That
change is not in this PR.

**Campaign B.** For each replicate's `best_model.pth`: regenerate `analysis_out/op_cache/`
over the same eight splits at the 0.05 floor (the committed op_cache is the *published*
checkpoint's and cannot be reused), then run the same parity script with the replicate
in place of RampNet. Also not in this PR.

Until both exist as committed, argument-configured scripts, this campaign is not
replicable from a clean clone. That is a stated gap, not an oversight.

## Stated limitations

- **n=3 gives a wide interval on the SD itself** — with 2 degrees of freedom the 95% CI
  on σ spans roughly 0.5σ̂ to 3.7σ̂. A fourth and fifth replicate are the pre-registered
  response if the result lands in the ambiguous band. They are not being run up front.
  Cost per extra replicate at $0.90/GPU-hour and 3.0 h/epoch: **~$130** as launched
  (`CHAIN=5` caps the run at 144 GPU-h ≈ ep48), or ~$119 if stopped at ep44, the last
  epoch the reading uses. An earlier draft quoted $119 while also saying the replicates
  run all 60 epochs; those two are inconsistent — 60 epochs would be 180 GPU-h ≈ $162,
  and the chain does not buy that much wall-clock.
  Extend the campaign with seeds **4, 5, …, never 0**: `sampler_seed_for(0)` collides
  with the published run's data order (`rampnet/seeding.py`).
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
- **Campaign B's artifact lands on a volume that purges, and the two facts interact.**
  `train.py` writes `best_model.pth` to the working directory, so it lands in `RUNDIR`,
  which defaults under `/gscratch/scrubbed` — purged on a ~21-day idle window
  ([`stage2_epoch_curve_84.md`](stage2_epoch_curve_84.md); #84 put Run A's checkpoints
  on the purchased `/gscratch/makelab` for exactly this reason). Combined with the
  unbounded calendar above, a replicate can finish and then age out while its siblings
  are still pending. **The copy-out in "Reproducing" is part of the protocol, not
  housekeeping.** The default is left on scrubbed so it matches the replicates already
  queued; moving it is a decision for a later campaign, not a mid-flight edit.
- **Seed variance is not the same as training-run variance.** Preemption, requeue
  boundaries and non-deterministic kernels also move a result. Ultralytics sets
  `deterministic=True` by default, which removes some of that for Campaign A; Campaign B
  has no such guarantee and its spread is therefore an upper bound on seed effect alone.

## Reproducing

Both launchers take the seed as an environment variable and record it in the job log.

```bash
# Campaign A, per replicate (Tillicum). CHAIN covers the 24 h normal-QoS ceiling.
# PYTHON is REQUIRED: the launcher defaults to `python`, which on Tillicum has no
# ultralytics -- our environment.yml pins CUDA 11.8 and does not transfer to Rocky 9 /
# H200. This is the interpreter docs/tillicum.md's as-run invocation uses.
cd ~/RampNet && mkdir -p logs
PYTHON=/gpfs/projects/makelab/$USER/envs/rampnet-yolo/bin/python \
SEED=1 NAME=y11x_tiles_s1 YOLO_CKPT=yolo11x.pt \
  YOLO_DATA=/gpfs/scrubbed/$USER/yolo/tiles/data.yaml \
  YOLO_IMGSZ=1024 BATCH=12 EPOCHS=60 SAVE_PERIOD=1 CHAIN=5 \
  sbatch scripts/model_comparison/run_yolo_train_tillicum.slurm

# Campaign B, per replicate (klone). logs/ must exist BEFORE submit -- Slurm opens
# --output against the submit directory, so a missing logs/ fails the job at start.
cd ~/RampNet && mkdir -p logs
SEED=1 sbatch stage_two/run_train_seed.slurm

# ... and copy Campaign B's artifact off scrubbed as soon as a replicate completes.
# RUNDIR is /gscratch/scrubbed/$USER/seedvar/rampnet_s<SEED>, which purges on a ~21-day
# idle window; /gscratch/makelab is purchased and never purged.
cp /gscratch/scrubbed/$USER/seedvar/rampnet_s1/best_model.pth \
   /gscratch/makelab/$USER/seedvar/rampnet_s1_best.pth
```

The seed plumbing itself is unit-tested in `tests/test_seeding.py` — including that the
default preserves the published pairing (`sampler_seed_for(42) == 0`; same seeds and
same data order, not bit-identical, since cuDNN autotuning and AMP are not seeded), that
the `DistributedSampler` actually receives that seed (asserted on the parsed statement,
because the same expression appears in a log line and a substring check passed after the
kwarg was deleted), and that the Tillicum launcher's positional argument list still lines
up with its unpack — an off-by-one that would silently shift `data` into `imgsz` and
train the whole arm at the wrong settings.
