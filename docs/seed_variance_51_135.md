# Seed variance: the number both #51 and #135 are now blocked on

**Status: PRE-REGISTERED 2026-09-03, before any replicate finished; SCORED 2026-09-15
at n=3 + 3; the three YOLO primary legs RE-SCORED 2026-09-17 on the pre-registered
checkpoints (an off-by-one in the copy, below); Campaign B EXTENDED to n=9 under
Amendment 2 and read 2026-09-18 (results below). The reading rules were written first,
so the interpretation cannot be chosen after seeing the numbers — the same discipline
as the #71 checkpoint-selection protocol and Tillicum measurement 3.**

**Amended 2026-09-04 in code review, before any replicate was scored** — see
[Amendment 1](#amendment-1-2026-09-04). The original rule divided 0.039 by Campaign A's
SD alone, and gave Campaign B no reading at all. Both are corrected below; the original
text is kept verbatim so the amendment is auditable rather than a silent rewrite.

**Amended 2026-09-17, after the n=3 result was seen and before any further replicate was
trained** — see
[Amendment 2](#amendment-2-2026-09-17-campaign-b-is-extended-to-nine-replicates). The
n=3 read landed in the ambiguous band with 96% of the noise on the RampNet side; the
pre-registered response (two more YOLO seeds) could not move the number, six more
RampNet seeds could, and they were free. The n=3 reading stays in the record; the n=9
reading sits beside it.

## Results (n=3 read 2026-09-15, corrected 2026-09-17; n=9 read 2026-09-18)

Every number here is `docs/data/seed_variance_51_135.json`, produced by
`scripts/analysis/seed_variance_read_51_135.py` from the inputs in
`docs/data/seed_variance_51_135/` (CPU, no panos), and pinned by
`tests/test_seed_variance_read_51_135.py`. The inputs came from three makelab2 runs,
all on the same A40 with the same two environments (ultralytics 8.4.120 / torch 2.13
for YOLO, torch 2.6.0 / timm 1.0.28 for RampNet), each with its `env_*.txt` recording
the checkpoint sha256s and its `driver_*.log` the wall-clock:

| run | what it scored | record |
|---|---|---|
| 2026-09-15 | YOLO s1–s3 (six legs) + RampNet s1–s3; 5 h 44 min | `env_2026-09-15.txt`, `driver_2026-09-15.log`; the three `_ep` legs it scored are archived under `yolo_mislabelled_ep45_45_43/` |
| 2026-09-17 | the same six YOLO legs on the right checkpoints; 2 h 20 min | `env.txt`, `driver.log` |
| 2026-09-17 | RampNet s4–s9; 3 h 32 min | `env_amend2_2026-09-17.txt`, `driver_amend2_2026-09-17.log` |

**The correction.** The 09-15 run read the three Campaign A primary legs at
`results.csv` epochs 45/45/43, not the pre-registered 44/44/42: Ultralytics names
`epochN.pt` from 0 but `results.csv` from 1, and the copy took the file whose name
matched the picked number. Found 2026-09-17 by comparing the checkpoints' own
`train_metrics` against the CSV; the correct files were re-scored the same day. The
three `best.pt` control legs reproduced the 09-15 files on every one of the 278 sweep
cells, so the harness is deterministic and the checkpoint swap is the only change. A
guard (`check_epoch_ckpt.py`) now refuses to score a checkpoint whose `epoch` field
does not match its label, and two tests pin the corrected read against the archived
one. `s_B` was never affected.

### The pre-registered read

Macro-mean F1 over the seven pooled US splits, each replicate at its own uniform
threshold selected on `sao_paulo`. Campaign A read at its best `metrics/mAP50-95(B)`
epoch ≤ 44 (s1 ep44, s2 ep44, s3 ep42 — on s2 the fitness blend would have said ep41,
by 0.00006; the named column wins). Campaign B is each replicate's `best_model.pth`;
its restart count (klone requeues, resumed from the last 1,000-step checkpoint) is from
`sacct -D` and is shown because A2.2 says it travels with the number.

| leg | thr | richmond | bend | clovis | morgantown | annapolis | paterson | gainesville | sao_paulo (dev) | manual_gold† | **US7 macro F1** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `y11x_tiles_s1_ep44` | 0.10 | 0.807 | 0.874 | 0.775 | 0.858 | 0.784 | 0.769 | 0.776 | 0.791 | 0.912 | **0.8061** |
| `y11x_tiles_s2_ep44` | 0.10 | 0.808 | 0.877 | 0.766 | 0.858 | 0.782 | 0.756 | 0.787 | 0.782 | 0.910 | **0.8049** |
| `y11x_tiles_s3_ep42` | 0.10 | 0.808 | 0.883 | 0.783 | 0.847 | 0.795 | 0.766 | 0.786 | 0.789 | 0.912 | **0.8097** |
| `rampnet_s1` (0 restarts) | 0.45 | 0.865 | 0.855 | 0.743 | 0.803 | 0.853 | 0.818 | 0.793 | 0.773 | 0.904 | **0.8188** |
| `rampnet_s2` (0) | 0.40 | 0.861 | 0.873 | 0.781 | 0.834 | 0.867 | 0.818 | 0.805 | 0.790 | 0.909 | **0.8340** |
| `rampnet_s3` (0) | 0.40 | 0.843 | 0.838 | 0.810 | 0.803 | 0.841 | 0.802 | 0.749 | 0.783 | 0.900 | **0.8124** |
| `rampnet_s4` (A2, 0) | 0.30 | 0.850 | 0.862 | 0.760 | 0.824 | 0.879 | 0.813 | 0.785 | 0.788 | 0.904 | **0.8247** |
| `rampnet_s5` (A2, 3) | 0.40 | 0.862 | 0.849 | 0.833 | 0.823 | 0.870 | 0.803 | 0.763 | 0.803 | 0.909 | **0.8290** |
| `rampnet_s6` (A2, 0) | 0.25 | 0.841 | 0.853 | 0.791 | 0.828 | 0.870 | 0.809 | 0.787 | 0.801 | 0.901 | **0.8254** |
| `rampnet_s7` (A2, 3) | 0.35 | 0.851 | 0.841 | 0.752 | 0.830 | 0.837 | 0.788 | 0.735 | 0.785 | 0.903 | **0.8050** |
| `rampnet_s8` (A2, 1) | 0.40 | 0.860 | 0.858 | 0.829 | 0.815 | 0.877 | 0.810 | 0.777 | 0.796 | 0.909 | **0.8322** |
| `rampnet_s9` (A2, 1) | 0.35 | 0.848 | 0.861 | 0.784 | 0.807 | 0.863 | 0.817 | 0.784 | 0.771 | 0.903 | **0.8232** |
| `y11x_tiles` (the n=1 arm the 0.039 was measured on) | 0.10 | 0.803 | 0.870 | 0.770 | 0.862 | 0.783 | 0.761 | 0.778 | 0.776 | 0.911 | **0.8039** |
| `RampNet` (published checkpoint, n=1) | 0.30 | 0.864 | 0.871 | 0.836 | 0.845 | 0.853 | 0.818 | 0.812 | 0.800 | 0.902 | **0.8427** |

† not pooled; shown because it is the split the "sharper statement" lives on.

| statistic | pre-registered, n_A=3 + n_B=3 | Amendment 2, n_A=3 + n_B=9 |
|---|---|---|
| mean US7 F1, Campaign A / Campaign B | 0.8069 / 0.8217 | 0.8069 / 0.8227 |
| `s_A` (YOLO) | **0.0025** | 0.0025 (not recomputed; A2.2) |
| `s_B` (RampNet) | **0.0111** | **0.0094** |
| `s_gap = sqrt(s_A² + s_B²)` | **0.0114** | **0.0097** |
| published gap (n=1 vs n=1) | 0.0388 | 0.0388 |
| gap of replicate means (B − A) | 0.0148 | 0.0158 |
| Welch 95% CI on the gap of means (A2.3 item 3) | [−0.011, 0.041], t 2.25 on 2.2 df, p 0.14 | **[0.008, 0.024]**, t 4.58 on 10.0 df, p 0.001 |
| published gap / `s_gap` | 3.4 σ | 4.0 σ |
| **A1.1 band** | **ambiguous** (`0.010 ≤ s_gap < 0.020`) | **real** (`s_gap < 0.010`) |
| **A1.2** | `s_B ≥ 0.0063`: seed variance dominates the paired MDE | same |

**#51, the n=3 read (as scored, corrected checkpoints).** The band is ambiguous, so per
A1.1 this reading claims neither "real" nor "indistinguishable" and reports the
interval: the published 0.039 sits at 3.4 σ of a σ that is itself measured from three
draws per arm, and 0.039 ± 2 `s_gap` is **[0.016, 0.062]**. `s_gap` is 96% `s_B`. The
pre-registered response to this band — a fourth and fifth Campaign A replicate — cannot
narrow it: two more YOLO seeds at `s_A` = 0.0025 move `s_gap` by nothing. That is what
Amendment 2 acted on.

**#51, the n=9 read (Amendment 2).** Six more RampNet replicates move `s_B` from 0.0111
to 0.0094 and `s_gap` from 0.0114 to 0.0097, which is the **real** band by the
pre-registered cut. Two things have to be said in the same breath:

- **The band call sits 0.0003 under the edge.** On 8 df the 95% CI on σ_B is
  0.68–1.92× the estimate, i.e. [0.0064, 0.0180], which puts `s_gap` anywhere in
  [0.0068, 0.0182] — the real and ambiguous bands both. The cut points were fixed in
  advance and the rule is applied as written, but the point estimate does not clear
  the edge by more than its own uncertainty, and A1.2's instruction ("say so rather
  than pick the side") applies. What the extension did settle is the *upper* end:
  at n=3 the σ_B interval reached 0.070, spanning all three bands; at n=9 it reaches
  0.018, and "indistinguishable" (`s_gap ≥ 0.020`) is excluded.
- **A2.3 item 5 governs what "real" means here.** The finding is *small but real, at
  the gap of replicate means* — **0.016 F1, Welch 95% CI [0.008, 0.024]** — not that
  the published 0.039 is the size of the effect. The CI excludes zero (t 4.58 on 10 df,
  p 0.001) and excludes 0.039. The two questions A2.3 item 3 separates give the same
  answer: RampNet's recipe beats the `y11x_tiles` recipe at matched operating points
  on the seven US splits, by about 0.016 F1, and the published 0.039 overstates it by
  about 2.5×.

The 0.039 decomposes the way the n=3 read already suggested: the published RampNet
checkpoint (0.8427) sits **0.020 above its recipe's nine-replicate mean, 2.1 `s_B`**;
the published `y11x_tiles` arm (0.8039) sits 0.003 *below* its replicate mean, 1.2
`s_A`. The n=1 comparison paired a favourable RampNet draw with a slightly unfavourable
YOLO one. (The released checkpoint is the paper run's epoch 1, comparable to these
replicates "modulo seed and dataloader order" per `stage2_epoch_curve_84.md` — but the
code that produced it predates git, and this document pre-registered that the released
checkpoint is compared, not pooled. It stays out of `s_B`.)

Two more descriptive facts the rule does not use:

- **RampNet's seed-selected thresholds span 0.25–0.45 (median 0.40).** The n=3 read
  said "0.40–0.45, not the published 0.30"; with nine seeds, two select at or below
  0.30 (s4 at 0.30, s6 at 0.25). The published 0.30 is inside the seed spread, at its
  low end.
- **Restarts do not explain the spread.** The two replicates requeued three times
  landed at opposite ends (s5 0.8290, s7 0.8050); the five never requeued span
  0.8124–0.8340. A1.2's upper-bound caveat stands, but nothing in the data points at
  requeue boundaries as the mechanism.

**#135.** `s_B` = 0.0094 on nine replicates is 1.5× the paired epoch-to-epoch MDE of
0.0063 (1.8× at n=3). Per A1.2, **every unpaired single-seed comparison in this repo
is limited by `s_B`, not by the MDE** — that includes #84's epoch curve read across
runs, the cosine rung's tie, and any single-checkpoint number quoted against another.
The MDE is demoted to the paired (same-run, epoch-vs-epoch) case only. The n=9
estimate is the one to carry: it is tighter, it includes replicates that were
requeued (the production regime on `ckpt-all`), and its CI [0.0064, 0.0180] does not
reach below the MDE.

### Secondary reads (post-hoc, descriptive only)

Added 2026-09-15 before any number was seen, fenced from the decision above; the
RampNet rows re-reported on n=9 per A2.3 item 4, with the seeds-1–3 subset kept.

**YOLO at as-saved `best.pt` (≤ 60 epochs).** 0.8133 / 0.8127 / 0.8130, mean 0.8130,
SD **0.0003** (unchanged by the correction: these legs reproduced cell-for-cell).
Sixteen more epochs move the tiles arm by +0.006 pooled over the corrected ≤ 44 read
and shrink its spread eight-fold. The n=1 "more training hurt out-of-distribution"
finding was on the *pano* arm (`yolo_geometry_51.md`); on the tiles arm, with three
seeds, it does not appear.

**manual_gold, both arms.** At each leg's own `sao_paulo`-selected threshold (the same
threshold as the primary read), YOLO scores 0.910–0.912 and RampNet 0.900–0.909 —
the "RampNet loses `manual_gold` at matched operating points" statement from
`operating_point_parity_51.md` replicates on all 27 pairings, now with spread:
YOLO 0.911 ± 0.001 (n=3), RampNet 0.905 ± 0.003 (n=9). Over the full sweep:

| leg | F1 at protocol thr | max F1 | at thr |
|---|---|---|---|
| `y11x_tiles_s{1,2,3}_ep≤44` | 0.828 / 0.825 / 0.821 (@0.25) | 0.912 / 0.910 / 0.912 | 0.10 |
| `y11x_tiles_s{1,2,3}_best` | 0.831 / 0.834 / 0.830 (@0.25) | 0.912 / 0.905 / 0.912 | 0.10 |
| `rampnet_s{1,2,3}` | 0.900 / 0.907 / 0.894 (@0.30) | 0.905 / 0.909 / 0.901 | 0.40 / 0.40 / 0.50 |
| `rampnet_s{4..9}` (A2) | 0.904 / 0.905 / 0.903 / 0.905 / 0.905 / 0.900 (@0.30) | 0.905 / 0.909 / 0.904 / 0.905 / 0.909 / 0.903 | 0.35 / 0.40 / 0.35 / 0.30 / 0.40 / 0.35 |
| `run_a_epoch_1` (seed 42, reference, same 1-epoch recipe) | 0.906 (@0.30) | 0.906 | 0.35 |

RampNet's max-F1 seed SD on `manual_gold` is **0.0030 on n=9** (0.0042 on the
seeds-1–3 subset). `stage2_cosine_rung_135.md` set "seed SD ≤ ~0.002 max-F1" as the
condition for reopening Run B; it is not met at either n. (YOLO's protocol column is
at 0.25, far from its 0.10 optimum, which is why it reads 0.82 there and 0.91 at max —
the published-point comparison on this split was never matched, which is exactly what
the parity protocol fixed.)

### Caveats that travel with these numbers

- **`s_A` is a sample SD on n=3** and was not recomputed under Amendment 2. Its 95% CI
  on 2 df is 0.52–6.3× the estimate, [0.0013, 0.0159]; at the top of that range
  `s_gap` would be 0.018 and the band ambiguous. `s_A²` is 7% of `s_gap²` at the point
  estimates, so this is the smaller of the two uncertainties, but it is not zero.
- **The n=9 band call is 0.0003 from the cut.** Stated above; repeated here because the
  band word ("real") will be quoted without the interval, and the interval is the
  finding.
- **Seeds 4–9 were declared after the n=3 result was seen.** Amendment 2 is not a blind
  amendment and the document says so. The statistic, script and cut points were not
  changed; the sample was extended on the noisier arm, which is the direction that
  *shrinks* `s_gap` regardless of where the new draws land. A reader who wants the
  blind reading has it: the n=3 column.
- The YOLO sweep rows carry 3-decimal F1 (they are parsed from `compare.py` reports,
  the parity script's path), so each per-split value has ±0.0005 quantisation and the
  macro mean ±0.0002. RampNet's values are re-scored from the op_cache at full precision.
- Campaign A's three replicates ran on Tillicum H200s; the n=1 arm ran on klone L40S.
  Hardware is a stated confound for the *offset* between the seed-0 arm and the
  replicate mean, not for `s_A`.
- Four of the six Amendment 2 replicates were requeued (s5 ×3, s7 ×3, s8 ×1, s9 ×1),
  resuming from a 1,000-step checkpoint each time; seeds 1–3 ran one incarnation each.
  `s_B` at n=9 is therefore training-run variance under `ckpt-all`'s preemption regime,
  an upper bound on the seed effect alone (Stated limitations). It is also the regime
  every future `ckpt-all` replicate will run in.
- The pooled statistic is macro F1 over seven US splits at one uniform threshold.
  Per-split, the replicates disagree with each other by far more than the macro does —
  RampNet's nine-seed SD is 0.032 on clovis and 0.022 on gainesville against 0.009 on
  the macro — which the macro averages away.

## Where the inputs live

The pre-registered statistic is produced by `scripts/analysis/operating_point_parity_51.py`
and read from its artifact `docs/data/operating_point_parity_51.json`, documented in
[`operating_point_parity_51.md`](operating_point_parity_51.md). When this document was
written those three files were on PR #154, not on `main`; #154 merged on 2026-09-04, so
every link below marked † now resolves from a clean clone. The marks are left in place
so the history of the dependency is visible.

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

**Closed 2026-09-15**, before any replicate was scored, by three committed scripts —
see [Scoring, as run](#scoring-as-run) below:
`scripts/model_comparison/yolo_baseline/run_seedvar_eval.sh` produces both campaigns'
inputs on makelab2; `scripts/analysis/operating_point_curve.py extract --checkpoint`
is the Campaign B half; `scripts/analysis/seed_variance_read_51_135.py` is this
document's reading rule as code, importing `select_threshold` and `macro_at` from the
parity script rather than re-implementing them. The parity script itself was not
re-pointed — its hardcoded legs and control gate still guard the published 0.039 — it
only gained an `op_cache` argument on the two functions the read imports.

## Amendment 2 (2026-09-17): Campaign B is extended to nine replicates

**Declared 2026-09-17, after the n=3 result above was scored and before any further
replicate was trained.** Amendment 1 was made blind; this one was not. It is a
deviation from the pre-registered response to the ambiguous band, and it is recorded
as one — the n=3 reading above stays in the record exactly as scored, and the n=9
reading is reported beside it, never in place of it.

### A2.1 Why the pre-registered response is replaced

A1.1's response to the ambiguous band was a fourth and fifth Campaign A replicate.
That was written on the assumption that the paid arm would be the noisy one. The
measurement says the opposite: `s_A` = 0.0023, `s_B` = 0.0111, so `s_B²` is 96% of
`s_gap²`. Two consequences, both arithmetic on the numbers already in this document:

- **The standard error of the gap of replicate means** is `sqrt(s_A²/n_A + s_B²/n_B)`:
  0.0065 at 3 + 3. Two more YOLO seeds (5 + 3) give 0.0065; six more RampNet seeds
  (3 + 9) give 0.0039, if `s_B` holds. The pre-registered response cannot move the
  number; the alternative can, and costs nothing (klone `ckpt-all`).
- **The SD estimate itself.** On 2 degrees of freedom the 95% CI on σ is 0.52–6.3×
  the sample SD, so `s_B` = 0.0111 is consistent with a true σ anywhere in
  **[0.006, 0.070]** — spanning all three A1.1 bands. On 8 df it narrows to
  0.68–1.92×, i.e. [0.0075, 0.021] if the point estimate holds. (The "0.5σ̂ to 3.7σ̂"
  in Stated limitations below understates the upper end; 6.3× is the 95% figure on
  2 df. The original text is left as written.)

The decision is driven by that arithmetic, not by the direction of any result: the
gap of replicate means (0.0115) cannot be raised toward the published 0.039 by any
number of RampNet seeds, and the reading rules below are the ones already ratified.

### A2.2 What is run

Six more replicates of the committed Stage 2 recipe, **seeds 4, 5, 6, 7, 8, 9**,
launched from the same checkout and launcher as seeds 1–3, with the same
`REPO=`/`RAMPNET_ENV=` invocation (Reproducing, below), on klone `ckpt-all`. Free.
Campaign A is **not** extended; `n_A` stays 3 and `s_A` is not recomputed.

Seeds 1–3 remain in the sample. No replicate is dropped for any reason other than
never producing a `best_model.pth`. A requeued replicate is kept, and its restart
count is recorded beside its number, per A1.2's upper-bound note.

### A2.3 How it is read

1. **One read, no interim look.** The read runs once, when all six have a
   `best_model.pth`, or on **2026-10-15** with however many have completed by then —
   whichever comes first — and `n_B` is reported as whatever it is. Nothing is scored
   before that.
2. **Same statistic, same script, same bands.** `s_B` is recomputed on `n_B` = 9,
   `s_gap` with the unchanged `s_A`, and A1.1's bands applied with the cut points
   unchanged (0.010 / 0.020). A1.2 is re-applied to the n=9 `s_B`. The n=3 band call
   above is not overwritten; the table gains a column.
3. **The gap of replicate means is reported with a Welch 95% CI on 3 + 9.** This is
   a report, not a decision rule. It answers a different question from A1.1: A1.1 asks
   whether the published n=1 gap clears the noise, the CI says what the
   recipe-vs-recipe gap is. If they disagree, both are stated; neither is suppressed.
4. **The secondary reads** (`manual_gold` at matched thresholds, as-saved thresholds)
   are re-reported on n=9 and stay fenced from the statistic, as in Results.
5. **What cannot come out of this.** If the n=9 `s_gap` lands under 0.010, the finding
   is "small but real, at the gap of replicate means (~0.01)" — not that the published
   0.039 is the size of the effect. The published checkpoint is compared, not pooled
   (A1.2), and that does not change here.

### A2.4 Plumbing gaps, named up front (as A1.3 did)

- `scripts/analysis/seed_variance_read_51_135.py` has one `SEEDS = (1, 2, 3)` tuple
  serving both arms. It needs a separate RampNet seed list before it can read 3 + 9;
  the test suite pins the n=3 artifact and will need the n=9 artifact pinned beside it.
- Scoring is the same makelab2 driver (`run_seedvar_eval.sh`, RampNet half only):
  ~35 min per replicate ⇒ ~3.5 h for six, plus copying six checkpoints klone → desktop
  → makelab2 with the hash at every hop.
- The copy-out from `/gscratch/scrubbed` to `/gscratch/makelab` after each completion
  is part of the protocol (Reproducing); with six replicates on a 3.9% duty cycle the
  21-day purge window is more likely to bite than it was with three.

**Closed 2026-09-18.** The script has `SEEDS_A` / `SEEDS_B` / `SEEDS_PREREG`, and the
artifact carries `statistics` (n=3, unchanged) and `statistics_a2` (n=9) side by side,
both pinned. All six replicates completed within 5 h 15 min of submission — the duty
cycle was nothing like 3.9% on 2026-09-17 — and were copied to `/gscratch/makelab` with
hashes the same afternoon; the purge window never came into it. Scoring took 3 h 32 min.

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

### Scoring, as run

Both campaigns are scored by one driver on makelab2 (the host every committed
`manual_gold` and YOLO-sweep number came from — all nine bundles' panos are staged
there, and its `.venv-eval` / `.venv` are the ultralytics and Stage 2 environments
recorded in `docs/data/seed_variance_51_135/env.txt`). The checkpoints are copied in
under names whose **stem is the leg label** — `compare.py` has no `--yolo-label`, it
uses the weights file's stem — and hashed at every hop
(`seedvar_ckpts/SHA256SUMS` on makelab2 equals the klone manifests).

```bash
# 1. Which epoch each Campaign A replicate is read at. The column is
#    metrics/mAP50-95(B); on s2 the fitness blend picks ep41 instead, by 0.00006.
python -c "from scripts.analysis.seed_variance_read_51_135 import pick_epoch as p;   print([p(f'docs/data/seed_variance_51_135/y11x_tiles_s{s}/results.csv')[0] for s in (1,2,3)])"
# -> [44, 44, 42]

# 2. On makelab2, with seedvar_ckpts/{y11x_tiles_s1_ep44,y11x_tiles_s2_ep44,
#    y11x_tiles_s3_ep42,y11x_tiles_s{1,2,3}_best}.pt and rampnet_s{1,2,3}_best.pth:
nohup scripts/model_comparison/yolo_baseline/run_seedvar_eval.sh > seedvar_driver.out 2>&1 &
#    YOLO: one compare.py call per split, six legs, perspective tiling, imgsz 1024,
#    cache floor 0.05, full sweep -- the seed-0 arm's flags exactly.
#    RampNet: operating_point_curve.py extract --checkpoint ... --cache <own dir>,
#    floor 0.05, min_distance 10, no TTA -- the committed op_cache arm.
#    Splits: the seven pooled US splits, sao_paulo (dev), manual_gold (secondary).

# 2b. Amendment 2 (seeds 4-9), same driver, RampNet half only, after
#     rampnet_s{4..9}_best.pth are in seedvar_ckpts/ with their hashes checked:
RN_SEEDS="4 5 6 7 8 9" YOLO_LEGS="" OUT=seedvar_eval_51_135_amend2   nohup scripts/model_comparison/yolo_baseline/run_seedvar_eval.sh > seedvar_amend2.out 2>&1 &
#     Restart counts for the table come from klone, not from the .out logs (a requeue
#     overwrites them): sacct -D -X --parsable2 -j <the nine job ids in CAMPAIGN_B_JOBS>
#     -> docs/data/seed_variance_51_135/klone_sacct_D.txt

# 3. Copy yolo/*.txt, rampnet_s*/*.json, env.txt, driver.log into
#    docs/data/seed_variance_51_135/ and apply the read (CPU, no panos needed):
python scripts/analysis/seed_variance_read_51_135.py            # writes docs/data/seed_variance_51_135.json
python scripts/analysis/seed_variance_read_51_135.py --check    # exit 1 if the artifact is stale
python scripts/analysis/seed_variance_read_51_135.py --markdown # the tables in the Results section
```

`tests/test_seed_variance_read_51_135.py` pins the epoch picks against the committed
`results.csv`, the disjoint bands, `s_gap`'s arithmetic, that every primary threshold
was selected on `sao_paulo` and no pooled split, that the secondary reads cannot enter
`s_gap`, and that the committed artifact reproduces from the committed inputs.

The seed plumbing itself is unit-tested in `tests/test_seeding.py` — including that the
default preserves the published pairing (`sampler_seed_for(42) == 0`; same seeds and
same data order, not bit-identical, since cuDNN autotuning and AMP are not seeded), that
the `DistributedSampler` actually receives that seed (asserted on the parsed statement,
because the same expression appears in a log line and a substring check passed after the
kwarg was deleted), and that the Tillicum launcher's positional argument list still lines
up with its unpack — an off-by-one that would silently shift `data` into `imgsz` and
train the whole arm at the wrong settings.
