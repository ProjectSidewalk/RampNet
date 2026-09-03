# What our cluster compute has cost

**2,684.4 GPU-hours across 3,991 job allocations on klone since 2026-07-02, at $0.** That is
the compute side of every RampNet experiment run on Jon's klone account, and until now it was
recorded nowhere — the figures in [`tillicum.md`](tillicum.md) and
[`stage2_training_cost.md`](stage2_training_cost.md) were transcribed by hand, per job, when
someone remembered (#143).

The ledger is `analysis_out/compute_log.jsonl`, one row per job allocation, written by:

```bash
python scripts/analysis/slurm_usage.py --cluster klone --user jfroehli \
    --from-file docs/data/compute/sacct_klone_2026-08-19.txt \
    --by-name
```

The raw `sacct` dump that command parses is committed at
`docs/data/compute/sacct_klone_2026-08-19.txt`, so the numbers below are re-derivable from a
clean clone **with no cluster account** — the same reason `usage_log.jsonl` is committed while
`vertex_usage.py` needs cloud credentials. Regenerate the dump with `--print-command`.

## `sacct -D` is worth 4.3x, and it is the whole finding

Slurm reports only the **last incarnation** of a requeued job unless you pass `-D`. Our klone
work lives on the preemptable `ckpt` partition, where **96% of allocations end in `PREEMPTED`**
(3,780 of 3,991). So the default view does not undercount slightly — it discards nearly
everything:

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
2026-07-24 (`sacct`, all arms)"*, measured 2026-07-30. Summing this ledger over jobs **ending**
from 2026-07-24 onward, the running total crosses 496.5 GPU-hours at **2026-07-29T21:17** —
i.e. the evening before that figure was written. It reproduces to the hour, which also
confirms the original query was duplicate-inclusive.

## What the 2,684 hours went on

| job name | jobs | GPU-h | what it is |
| :--- | ---: | ---: | :--- |
| `yolo_curb_ramp_train` | 3,857 | 2,046.9 | the supervised YOLO baseline (#51) |
| `rampnet_run_a_84` | 6 | 528.6 | Run A, the epoch curve (#84) |
| `rampnet_cosine_rung_135` | 2 | 25.1 | the 8-epoch cosine rung (#135), still in flight as of the 2026-08-19 pull |
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
and `ckpt` is scavenger). The price is paid in preemption instead: 96% of allocations were
preempted, and `stage2_training_cost.md` measures the resulting overhead at **1.67x** on the
paper's Stage 2 run — 44.7 h of compute stretched over 74.6 h of calendar across 15 restarts.

Rates live in `COMPUTE_PRICING` in `scripts/model_comparison/pricing.py`, verified-only with an
`as_of` date and a source, same discipline as the token table. A cluster with no entry prices
to `None`, not `$0` — "we checked and it is free" and "we have no rate" are different
statements and only one is safe to put in a paper.

## Gaps, stated

- **Tillicum is not in the ledger.** Its jobs bill at $0.90/GPU-hour and are the only compute
  spend we have. The back-fill needs a `sacct` pull from Tillicum, which is Duo-gated; the
  control master was down when this was written. One command once it is up:
  `python scripts/analysis/slurm_usage.py --cluster tillicum --since 2026-07-01 --save-raw docs/data/compute/sacct_tillicum_<date>.txt`.
  Until then the $4.20 and $0.03 figures in [`tillicum.md`](tillicum.md) remain hand-transcribed.
- **The window starts 2026-07-02**, which is where this account's retained records begin, not
  where RampNet's compute begins.
- **The paper's own Stage 1 and Stage 2 runs are not here at all.** They ran on a different
  user's account (see [`stage1_generation_cost.md`](stage1_generation_cost.md)), so they are
  outside this `sacct` query and are not recoverable through it. What survives of them is the
  TensorBoard-derived measurement in [`stage2_training_cost.md`](stage2_training_cost.md):
  ~56 GPU-hours per epoch, ~12 epochs.
- **Retention is finite.** `sacct` keeps job records for a while and then does not. This
  back-fill was possible in August 2026; the same query in 2027 will return less.
