# What Stage 1 generation cost, and what it lost

**97.91% yield, and the entire 2.09% loss is panoramas Google refused to serve.** Measured
2026-08-03 from the paper run's own logs, which survived on a lab scratch volume and are now
committed at `docs/data/rampnet1_stage1_run/`.

[`curb_ramp_data_sourcing.md` §7](curb_ramp_data_sourcing.md#7-retrain-cost) calls Stage 1 "the long
pole and **entirely unmeasured**". It is the long pole — that part holds, and is now the *only*
reason a retrain takes days rather than hours (see [`stage2_training_cost.md`](stage2_training_cost.md)).
It is no longer unmeasured.

```bash
python scripts/analysis/stage1_yield.py
```

## Yield

| | |
| :--- | ---: |
| Intended (`finaldataset.jsonl` lines) | 219,170 |
| Written (`progress.txt`, unique) | **214,599** |
| Never written | 4,571 (**2.09%**) |
| ...of which Google refused to serve | **4,570 of 4,571 (99.98%)** |
| ...unexplained | 1 |

`download_dataset.py` appends an index to `progress.txt` only after **both** the `.jpg` and the
`.json` are written (`mark_done(idx)` is the last statement of `process_line`), and re-reads that
file on restart, so it is a clean completion ledger with no duplicates. A fetch failure returns
early without marking, so it is simply retried on the next pass — which is why the residual is
almost exactly the set of panos that never became fetchable.

**Fetch failure is the whole loss mechanism.** For planning purposes that is the useful form:
budget ~2% shrinkage between "records that project onto a panorama" and "panoramas you actually
get", and expect it to be irreducible by retrying.

### The disk-quota scare cost nothing

The last Slurm incarnation's log ends in a wall of `[Errno 122] Disk quota exceeded` — 11,438 of
them — which reads like a truncated dataset. It was not:

| | |
| :--- | ---: |
| Quota-failed line indices in the log | 11,438 |
| ...still missing from `progress.txt` | **0** |

Every one completed on a later pass. Worth recording because the surface impression is alarming and
wrong, and because **the run did hit a storage wall at ~214k panoramas**. A 2.0-scale corpus is
roughly double that; storage headroom is a real prerequisite, not an afterthought.

### An unexplained 223

`progress.txt` records 214,599 written, and the published dataset has **214,376** — a gap of **223
panoramas (0.10%)**, unaccounted for. It plausibly arises downstream in `split_dataset.py` or at
publication time, but nothing in the rescued evidence establishes that, so it is recorded as open
rather than explained.

## Timing — lower bounds

From Slurm accounting (`sacct -u jsomeara -S 2025-03-01 -E 2025-07-01`), summed across every
incarnation:

| Step | Wall-clock | Jobs |
| :--- | ---: | ---: |
| `run_download_dataset.slurm` (fetch + crop-model inference) | **≥49.1 h** | 26 |
| `run_generate_meta.slurm` | ≥13.5 h | 7 |
| `run_generate_negatives.slurm` | ≥1.5 h | 2 |
| *(separate: `download_data.slurm`, the crop-model training data)* | ≥45.3 h | 7 |

**These are lower bounds, for two reasons**: `--requeue` overwrites the accounting record, so
requeued incarnations are invisible (the same artefact that hid the Stage 2 run); and the download
was finished off in `interactive` jobs on 2025-06-16/17 whose output never reached
`download_dataset.out`. The evidence for the second is direct — `progress.txt` was modified
**2025-06-17**, two days after the last Slurm log write on 2025-06-15, and Stage 2 training began
2025-06-17 08:39, hours after the last panorama landed.

So: **≤4,370 panoramas/hour** end-to-end, including crop-model inference on every panorama, against
Google's undocumented tile endpoints at 32 tiles each.

### What that implies for a rebuild

At the 500k-record target (~385k panoramas, see `stage2_training_cost.md`):

| | Panoramas | At ≤4,370/h |
| :--- | ---: | ---: |
| Incremental (new panoramas only) | ~170,000 | **≥39 h** |
| Full regeneration from scratch | ~385,000 | **≥88 h (≥3.7 d)** |

Add `generate_dataset_meta` at ~1.8x its measured 13.5 h (~24 h). **§7's "about a week of
wall-clock" is confirmed as the right order of magnitude for a full rebuild** — and now with a
measured basis rather than an allocation guess. The incremental path is ~2.5-3 days.

Two caveats. The rate is a *lower bound on duration* built from a *lower bound on time*, so treat
≥39 h as optimistic. And rate limiting remains the dominant risk: nothing here establishes that
Google's endpoints will sustain this rate in 2026, only that they did in 2025.

## A rescued artefact: the paper's Stage 1 accuracy, and half of #18's correction

`docs/data/rampnet1_stage1_run/stage1_evaluation_results.txt` is the output that produced the
paper's Stage 1 dataset-agreement numbers — precision **0.9403**, recall **0.9245** on the
1,000-panorama gold set, which the README quotes as 94.0%.

It also records the quantity the README says has **not** been re-measured: **119 predicted points
"ignored (matched already-claimed GT)"** out of 3,972. Counting those as false positives — the
redundant-detection half of the [#18](https://github.com/ProjectSidewalk/RampNet/issues/18)
correction — is arithmetic on the committed file:

| Stage 1 dataset agreement | As published | Redundant points counted as FP |
| :--- | ---: | ---: |
| Precision | 0.9403 | **0.9121** (−2.8 pts) |
| Recall | 0.9245 | 0.9245 (unaffected) |

**This is not the corrected Stage 1 number.** The README is explicit that *two* changes move it:
redundant points counted as FP, **and** matching through `rampnet/metrics.py` (nearest unclaimed
ground truth) rather than first-in-list order. Only the first is computable from this artefact; the
second still requires re-running `stage_one/dataset_evaluation/evaluate.py`. 0.9121 is therefore an
**upper bound** on the corrected precision, and a useful one: it says the Stage 1 correction is of
the same size as the Stage 2 one (−1.0 precision), not an order larger.

## Provenance and replication

Committed evidence in `docs/data/rampnet1_stage1_run/` (2.6 MB, `SHA256SUMS` alongside):

| File | What it is |
| :--- | :--- |
| `progress.txt` | 214,599 completed line indices — the completion ledger |
| `download_dataset.out` | the final Slurm incarnation's log (fetch failures + quota errors) |
| `missing_panos.csv` | the 4,571 never-written `(line_index, pano_id)` pairs, derived |
| `stage1_evaluation_results.txt` | the paper's Stage 1 gold-set evaluation output |

Rescued from `/gscratch/makelab/jsomeara/RampNet/stage_one/dataset_generation/`, where they were
the only copy; the full 19 GB tree is now mirrored to
`/gscratch/makelab/jonf/rescue_jsomeara_rampnet/`.

Stated gaps:

- **`finaldataset.jsonl` (64 MB, 219,170 lines) is not committed** — it is the manifest the run
  consumed, and it lives in the rescued tree above. `stage1_yield.py` reproduces every number
  without it, because `missing_panos.csv` is committed; pass `--finaldataset` to regenerate that
  CSV from the manifest and assert it identical.
- **`download_dataset.out` is only the last incarnation.** `#SBATCH --output=download_dataset.out`
  carries no `%j`, so each of the 26 jobs overwrote its predecessor. Per-run failure taxonomy for
  the earlier passes is gone.
- **The source inventories are also in the rescued tree but not committed** — `location_data/`
  (nyc.csv, portland.geojson, bend.geojson; 71 MB) and `street_data/` (801 MB). These are the
  **paper-era** files, which matters because §9 measures drift against today's endpoints (Bend
  +8.7%). Publishing them is the single largest remaining replication win for Stage 1 and is not
  done here.
