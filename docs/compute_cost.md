# What our cluster compute has cost

**2,684.4 GPU-hours across 3,990 job allocations on klone since 2026-07-02, at $0.** That is
the compute side of every RampNet experiment run on Jon's klone account, and until now it was
recorded nowhere — the figures in [`tillicum.md`](tillicum.md) and
[`stage2_training_cost.md`](stage2_training_cost.md) were transcribed by hand, per job, when
someone remembered (#143). It is a snapshot as of the 2026-08-19 pull: **158.0 of those
GPU-hours are elapsed-so-far from 3 jobs that were still RUNNING** (`38304087` alone, 157.5 h),
and the next pull will re-record them finished and move the total.

**Plus 674.7 GPU-hours across 38 allocations on Tillicum since 2026-07-30, at $607.24** — the
only billed compute in the project, back-filled 2026-09-21 from a second committed dump
(see [Tillicum](#tillicum-6747-gpu-hours-60724-the-only-billed-compute) below). Of that,
$540.61 is the three Campaign A seed replicates (#51), which the seed-variance doc had
projected at ~$130 each and which cost ~$180 each.

The ledger is `analysis_out/compute_log.jsonl`, one row per job allocation, written by:

```bash
python scripts/analysis/slurm_usage.py --cluster klone --user jfroehli \
    --from-file docs/data/compute/sacct_klone_2026-08-19.txt \
    --by-name
```

The raw `sacct` dump that command parses is committed at
`docs/data/compute/sacct_klone_2026-08-19.txt` (765,754 bytes, sha256
`d6597d97e3ccca4324763195c66f268f6eaf76bc07b4e34d3c565ee4c4e8c69d`), so the numbers below are
re-derivable from a clean clone **with no cluster account** — the same reason `usage_log.jsonl`
is committed while `vertex_usage.py` needs cloud credentials. The ledger itself can never be
byte-identical on regeneration (every row carries a `recorded_at` stamp), but the dump can, and
`--from-file` prints its sha256 so a re-run can be checked against this line rather than
assumed to match; `tests/test_slurm_usage.py` asserts the committed ledger is exactly this
dump's parse. Regenerate the dump with `--print-command`.

## `sacct -D` is worth 4.3x, and it is the whole finding

Slurm reports only the **last incarnation** of a requeued job unless you pass `-D`. Our klone
work lives on the preemptable `ckpt` partition, where **95% of allocations end in `PREEMPTED`**
(3,780 of 3,990; 96% counting the 59 `REQUEUED` incarnations). So the default view does not
undercount slightly — it discards nearly everything:

| the #51 YOLO baseline (`yolo_curb_ramp_train`) | rows | GPU-hours |
| :--- | ---: | ---: |
| `sacct -D` (every incarnation) | 3,857 | **2,046.9** |
| `sacct` without `-D` (last incarnation per job id) | 27 | 470.5 |
| | | **4.35x** |

Any hand tally taken from a default `sacct` is short by this factor, and nothing about the
output says so. It is also why the ledger keys rows on **(cluster, job id, start)** rather than
the job id: 3,857 rows collapse to 27 keys otherwise.

**This validates rather than contradicts the number already in the repo.**
[`tillicum.md`](tillicum.md) records *"496.5 GPU-hours consumed on the baseline since
2026-07-24 (`sacct`, all arms)"*, written 2026-07-30 at 07:07. That figure was a **snapshot**
of a live `sacct -S 2026-07-24` on the baseline's job name: every incarnation alive after
07-24, with a still-running one counting its elapsed so far. Re-read that way from this dump
(`scripts/analysis/gpu_hours_as_of.py`, each incarnation's elapsed truncated at the query
instant):

```bash
python scripts/analysis/gpu_hours_as_of.py \
    --from-file docs/data/compute/sacct_klone_2026-08-19.txt \
    --since 2026-07-24 --at 2026-07-30T07:00 --job-name yolo_curb_ramp_train
# 497.5 GPU-hours as of 2026-07-30T07:00:00: 400 incarnation(s) ...
```

The baseline passes 496.5 at about 06:50 that morning, seventeen minutes before the line was
written (author time 07:07; the commit landed at 08:15, by which point the snapshot reads
503.9). It also confirms the original query was duplicate-inclusive: the same snapshot over
only the last incarnation per job id (what `sacct` shows without `-D`) is 85.8 GPU-hours.
Two things this check is *not*: it is not account-wide — every job name at that instant gives
553.2 — and it is not a running sum over jobs by their end time, which crosses 496.5 only at
2026-07-30T11:03 on the baseline, hours after the line was written, because it waits for each
incarnation to finish before counting any of it. An earlier version of this section used that
by-end sum over every job name and reported a crossing at 2026-07-29T21:17; that number was a
coincidence of two mistakes, not a validation.

## What the 2,684 hours went on

| job name | jobs | GPU-h | what it is |
| :--- | ---: | ---: | :--- |
| `yolo_curb_ramp_train` | 3,857 | 2,046.9 | the supervised YOLO baseline (#51) |
| `rampnet_run_a_84` | 6 | 528.6 | Run A, the epoch curve (#84) |
| `rampnet_cosine_rung_135` | 1 | 25.1 | the 8-epoch cosine rung (#135), one preempted incarnation; still PENDING its requeue at the 2026-08-19 pull, which the ledger does not record as an allocation |
| `qwen_curb_ramp_compare` | 20 | 19.3 | Qwen legs of the model comparison |
| `open_curb_ramp_compare` | 16 | 8.9 | OWLv2 / Grounding DINO legs |
| `eval_run_a_84` | 16 | 3.1 | Run A evaluation |
| everything else (46 names) | 74 | 52.5 | env builds, smoke tests, data prep, eval |

GPUs drawn, by allocation: `l40` 2,194, `l40s` 1,285, `h200` 456 (klone's `ckpt-g2` nodes),
`a40` 23, `p100` 6, `2080ti` 1. Consistent with [#135](https://github.com/ProjectSidewalk/RampNet/issues/135)'s
finding that the scavenger queue hands out whatever is free and the type is not worth waiting
for.

**The ledger records every job on the account, not only RampNet's.** 28 allocations named
`sal-*` / `arch_*` (34.3 GPU-h, **1.3%**) belong to other projects. That is deliberate: the
artifact is a complete measurement and attribution is left to the reader, rather than baking a
filter chosen once into a durable file. Use `--by-name`, and exclude those two prefixes for a
RampNet-only total of **2,650.0 GPU-h**.

## Cost

**$0.** Every one of these hours was on klone, which is free at the point of use (condo model,
and `ckpt` is scavenger). The price is paid in preemption instead: 95% of allocations were
preempted, and `stage2_training_cost.md` measures the resulting overhead at **1.67x** on the
paper's Stage 2 run — 44.7 h of compute stretched over 74.6 h of calendar across 15 restarts.

Rates live in `COMPUTE_PRICING` in `scripts/model_comparison/pricing.py`, verified-only with an
`as_of` date and a source, same discipline as the token table. A cluster with no entry prices
to `None`, not `$0` — "we checked and it is free" and "we have no rate" are different
statements and only one is safe to put in a paper.

## Tillicum: 674.7 GPU-hours, $607.24, the only billed compute

Back-filled 2026-09-21, five weeks after the klone pull, from
`docs/data/compute/sacct_tillicum_2026-09-21.txt` (7,282 bytes, sha256
`420642438d3b36b1bebb0dfd659bde759d5767882a3f22bceb9a847826e022fb`), pulled on Tillicum with
the exact command `--print-command` prints and parsed with:

```bash
python scripts/analysis/slurm_usage.py --cluster tillicum --user jfroehli \
    --from-file docs/data/compute/sacct_tillicum_2026-09-21.txt \
    --by-name
```

On a clean clone the ledger is rebuilt in append order: the klone command above first, then
this one — `tests/test_slurm_usage.py` compares the ledger to the two dumps' parses in that
order. Run against the committed ledger, either command appends nothing, because every row is
already there.

Every training allocation is named `yolo_curb_ramp_train`, so `--by-name` cannot separate the
runs. The attribution below comes from the `out:` line the launcher echoes at the top of each
job's log (`logs/yolo_train_till_<jobid>.out` on Tillicum), read 2026-09-21; it is not
derivable from the dump alone.

| run | jobs | GPU-h | $ | what it is |
| :--- | ---: | ---: | ---: | :--- |
| `y11x_tiles_s1` | 10 | 225.18 | 202.66 | Campaign A seed 1 (#51), 2026-09-03 → 09-14, ran all 60 epochs |
| `y11x_tiles_s2` | 10 | 195.45 | 175.91 | seed 2, 2026-09-03 → 09-13, 60 epochs |
| `y11x_tiles_s3` | 10 | 180.05 | 162.05 | seed 3, 2026-09-03 → 09-12, 60 epochs |
| `y11x_pano_h200` | 4 | 62.36 | 56.12 | the pano arm resumed from its klone checkpoint and run to epoch 60, 2026-08-04 → 08-06; it is a published arm in `model_comparison.md`, but its compute was recorded nowhere until now |
| `y11x_tiles_h200_probe` | 1 | 7.01 | 6.30 | measurement 3 in [`tillicum.md`](tillicum.md), bounded at 7 h and hit the wall |
| `yolo_data_prep` | 1 | 4.67 | 4.20 | the tiles dataset regeneration; 16,794 s = 4.665 GPU-h, $4.1985, i.e. the hand-transcribed $4.20 at cent precision |
| `tillicum_smoke`, `chain_env_probe` | 2 | 0.03 | 0.00 | `debug` QoS, UsageFactor 0 in Slurm |
| **total** | **38** | **674.74** | **607.24** | totals are from unrounded values; the rows above are rounded individually and do not sum to them exactly (the three replicate dollar rows add to $540.62 against a true $540.61) |

**Reconciled to the bill, to the cent.** `hyakusage` on 2026-09-21, saved as
`docs/data/compute/hyakusage_tillicum_2026-09-21.txt` (2,528 bytes, sha256
`3205d2dd8c4d0e4b6ea39ed323d86d1d8be5f7b23c308e215d5817977977d60f`), reports the current
billing cycle, 2026-08-26 to 2026-09-21, at **600.68 GPU hours, $540.61, 30 jobs**. The
ledger's 30 rows that ended in that window sum to 600.68 GPU-h and $540.61;
`tests/test_slurm_usage.py` parses those figures out of the committed `hyakusage` file and
asserts the ledger matches them. The August cycle (the other 8 rows) was drawn against the
100-hour demo credit: `hyakusage` shows **$23.35** of credit remaining, and $90.00 − $23.35 =
$66.65, which is this ledger's $66.62 plus the $0.03 that `hyakusage` charges for the `debug`
smoke job and Slurm's UsageFactor 0 does not (the disagreement documented in
[`tillicum.md`](tillicum.md)). So the ledger's lifetime figure is $607.24 and `hyakusage`'s
equivalent is $607.27; the 3-cent difference is that one job and nothing else. Campaign A was
the first spend against the $1,500/month budget, and it used 36% of one month. `hyakusage`
also states the credit has no expiration, which was unverified in `tillicum.md` until now.

**Three things this corrects.** (1) [`seed_variance_51_135.md`](seed_variance_51_135.md) said
the replicates would stop near epoch 48 because `CHAIN=5` buys six slices; each chain in fact
ran ten slices (24 of the 30 allocations hit the 24 h wall; the last link of each chain was
partial, and three links ran under two minutes and logged no epoch) and all three reached
epoch 60, so the per-replicate cost is 180 to 225 GPU-h and $162 to $203, not 144 GPU-h and
~$130. **The spread between replicates is mostly not restart overhead.** From the committed
`docs/data/seed_variance_51_135/y11x_tiles_s{1,2,3}/results.csv` `time` column, logged
epoch time is 212.5 / 188.0 / 173.5 h against 225.2 / 195.5 / 180.1 h billed, so restarts
account for 12.7 / 7.5 / 6.6 h — 6.1 h of the 45.1 h between s1 and s3. The median epoch is
the same in all three (≈10,400 s, 2.9 h); the rest is contiguous blocks of epochs at about
2.4× that — s1 epochs 1–3 and 44–49 (≈39 h excess), s2 epochs 41–43 (≈14 h), s3 none —
i.e. throughput stalls on the node or the filesystem, cause not established. Same 60
epochs on the same schedule, so the read is unaffected; but a fourth replicate should be
budgeted at the modal ~180 GPU-h / $162 with a stall tail up to ~$203, not at a mean.
(2) The `y11x_pano_h200` arm's 62.4 GPU-h were unrecorded anywhere. (3) The two
hand-transcribed figures in `tillicum.md` ($4.20 prep, $0.03 smoke) are now measured: the
prep job is 4.665 GPU-h at `normal` and prices to $4.20 at cent precision; the smoke job is
0.03 GPU-h at `debug`, which this ledger prices at $0 per Slurm's UsageFactor while
`hyakusage` charges $0.03 — still unresolved and still capped at $0.90 per job.

## klone, 2026-09-24: the #131 Phase 1 replication, pulled by job id

Three jobs, 0.22 GPU-hours, $0, from `docs/data/compute/sacct_klone_2026-09-24_sa131.txt`
(519 bytes, sha256 `5a08c98d9885bb789b6cc7657f1489e490bff11ffdb18ff57401999efe7a6c36`), pulled
on klone with the `--print-command` invocation plus `-j 40546626,40546727,40549843` so it holds
these three allocations and nothing else, and parsed with:

```bash
python scripts/analysis/slurm_usage.py --cluster klone --user jfroehli \
    --from-file docs/data/compute/sacct_klone_2026-09-24_sa131.txt --out analysis_out/compute_log.jsonl
```

| job | what | partition | elapsed | GPU-h |
| :--- | :--- | :--- | ---: | ---: |
| 40546626 `sa131_unpack` | the seven US splits from the Hub onto scratch (CPU) | ckpt-all | 218 s | 0 |
| 40549843 `sa131_phase1` | `silent_activation.py`, ckpt copy | ckpt-all, L40S on g3124 | 527 s | 0.146 |
| 40546727 `sa131_phase1` | `silent_activation.py`, lab allocation | gpu-l40s, L40S on g3104 | 271 s | 0.075 |

The two GPU jobs are the same run twice; both replicas are byte-identical
([`curb_ramp_data_sourcing.md` §0c](curb_ramp_data_sourcing.md), #131). A pull by job id is the
right shape for a small experiment: it adds exactly its own rows, so the ledger stays the sum of
its committed dumps and two experiments pulled the same day cannot re-record each other's jobs.
(A pull by date range would also have re-recorded the plan-item-4 arms running that day, which
#180 records from its own dump.)

## Gaps, stated

- **Tillicum is a snapshot as of 2026-09-21.** Nothing after that date is in the ledger; the
  next Tillicum campaign needs another pull, with the same command and a new dated dump.
  Nothing was RUNNING at the pull, so no row will be re-recorded.
- **The window starts 2026-07-02.** The klone dump was pulled with `-S 2026-07-01` and its first
  record started 2026-07-02, so nothing here says whether older records exist on the account;
  it is only where this pull begins, not where RampNet's compute begins.
- **The paper's own Stage 1 and Stage 2 runs are not here at all.** They ran on a different
  user's account (see [`stage1_generation_cost.md`](stage1_generation_cost.md)), so they are
  outside this `sacct` query and are not recoverable through it. What survives of them is the
  TensorBoard-derived measurement in [`stage2_training_cost.md`](stage2_training_cost.md):
  ~56 GPU-hours per epoch, ~12 epochs.
- **Retention is finite.** `sacct` keeps job records for a while and then does not. This
  back-fill was possible in August 2026 (klone) and September 2026 (Tillicum); the same
  queries in 2027 will return less.
- **GPU time on a host without Slurm is not in this ledger, by design.** makelab2 (the lab's
  A40) has no `sacct`, and `compute_log.jsonl` is asserted equal to the `sacct` dump's parse, so
  a row for it cannot go here. It goes in `analysis_out/usage_log.jsonl` instead, the same
  append-only ledger the paid API legs use, as a `paid: false` row with the host, GPU and
  `elapsed_s`. The two #163 Vistas re-runs are the example: 237.153 s and 191.318 s on
  makelab2, recorded there and quoted, with the per-stage breakdown, in the cost paragraph of
  [`model_comparison.md` §Resolution parity](model_comparison.md). At $0 it changes no total,
  but a reader who starts here should not conclude the time went unrecorded.
