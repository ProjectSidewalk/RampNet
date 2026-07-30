# Tillicum — access, migration, and what it costs

Working notes for running RampNet jobs on **Tillicum**, UW-IT's usage-billed GPU
cluster. The makelab group was provisioned on **2026-07-29**.

This doc exists because Tillicum is **not** klone-with-different-hostnames — the
scheduler model, the walltime ceiling, the CPU:GPU ratio, and the cost model all
differ in ways that make our existing `.slurm` scripts **non-submittable as written**.
See [Migrating our Slurm scripts](#migrating-our-slurm-scripts) for the diff.

> **Sourcing.** Everything in the tables below is from the [Tillicum
> docs](https://hyak.uw.edu/docs/systems/tillicum/scheduling-jobs) and the provisioning
> email (2026-07-29), read on 2026-07-30. Sections marked **UNVERIFIED** are our
> inference or are undocumented — we have not run a single job on Tillicum yet, so
> nothing here is confirmed by experience. Correct this file from measurement, not
> from re-reading the docs.

## Why we care: the klone problem this solves

The supervised-YOLO baseline (#51) ran on klone's `ckpt` scavenger partition, which is
free but **preemptable**. As of 2026-07-30 that had stopped working as a compute
strategy:

- **496.5 GPU-hours consumed** on the baseline since 2026-07-24 (`sacct`, all arms).
- **170 job restarts in one night** across five jobs (39 / 37 / 33 / 33 / 28).
- **Zero completed epochs** on five of six arms over that same night; four of the six
  arms are still "one-epoch models" holding an ep1 `best.pt`.

The binding constraint is not the nominal 8.24 h checkpoint slice — it is the
**effective** contiguous run the partition hands out, which collapsed to minutes. Any
arm whose epoch exceeds that interval can never complete one. See #51 and #70.

**Tillicum does not preempt.** The docs state that even the `urgent` QoS will "not
cancel or preempt jobs that are already running." That is the entire reason to pay.

## Access

```bash
ssh <UWNetID>@tillicum.hyak.uw.edu
```

Duo 2FA, same as klone — **Claude cannot authenticate this.** The `wsl-ssh.ps1` helper
in `dotfiles` has no `tillicum` target yet; adding one (with a `ControlPersist` master,
mirroring the `klone` block in `ssh_config.wsl`) is a prerequisite for driving Tillicum
from an agent session. Until then, Tillicum is Jon-only.

> The get-started page writes the hostname two ways — `tillicum.hyak.uw.edu` in the
> hostname field and `tillicum.hyak.edu` in the `ssh` example. The former is almost
> certainly correct; **UNVERIFIED** until someone logs in.

## The scheduler model: QoS, not partitions

klone is a condo: you get partitions tied to hardware your group owns or scavenges
(`-p ckpt-g2 -q ckpt-gpu`). **Tillicum has no partitions at all.** You pick a QoS, and
billing handles the rest.

| QoS | Max time | Max GPUs/job | Concurrent | Notes |
|---|---|---|---|---|
| `normal` | **24 h** | 16 | 48 GPUs | default |
| `debug` | 1 h | 1 | 1 job | smoke tests |
| `interactive` | 8 h | 2 | 2 jobs | `salloc` work |
| `long` | **7 days** | 16 | 96 GPUs (shared) | **pre-approval required** |
| `wide` | 24 h | unlimited | 96 GPUs (shared) | **pre-approval required** |
| `urgent` | 3 days | 64 | 96 GPUs (shared) | **pre-approval required** |

**The 24 h ceiling on `normal` is a real constraint for us.** Our tiles epochs ran
4.1–9.5 h on klone; a 60-epoch schedule does not fit in 24 h. Two options:

1. **Request `long` QoS** (7 days) via the Tillicum Special QoS Access Request Form.
   This is the clean answer and should be requested now — it gates the #70 rerun.
2. **Chain 24 h `normal` jobs**, resuming from `weights/last.pt`. Our
   `run_yolo_train.slurm` already has the resume block for this, and without
   preemption a resume boundary is predictable rather than random.

Option 2 works today and needs no approval; option 1 is less operationally fiddly.
Do 1, fall back to 2.

## Hardware

192 × NVIDIA **H200** (141 GB HBM3e, NVLink 4.0), 8 GPUs per node, 24 × Dell XE9680,
1,536 Intel Emerald Rapids cores, 400 Gbps NDR InfiniBand, ~3 PB flash, Rocky 9.

Per-GPU binding: **200 GB system RAM and 8 CPUs**. This ratio is not negotiable —
the docs are explicit that exceeding 8 CPUs per GPU requires requesting *additional
GPUs*. **CPU-only jobs are prohibited**; every job must request ≥1 GPU.

> Operational consequence: our CPU-side data prep (`run_yolo_data.slurm`,
> `run_yolo_prep.slurm`) has no business on Tillicum. Keep prep on klone, train on
> Tillicum.

## Cost model

**$0.90 per GPU-hour**, where **GPU-hour = elapsed wall-clock × N GPUs**. Billed
monthly via ITBill to the lab's UW worktag. (**The worktag and subscription IDs are
deliberately not recorded in this repo** — it is public. Ask Jon, or see the Tillicum
provisioning email.)

- **100 free GPU demo hours** on the new account (≈ $90). Whether these expire is
  **UNVERIFIED**.
- Budget requested 2026-07-30: **$1,500/month** ≈ **1,666 GPU-hours/month**.

At that ceiling:

| job shape | wall-clock hours/month |
|---|---|
| 1 GPU | ~1,666 |
| 2 GPUs | ~833 |
| 8 GPUs (full node) | ~208 |

Monitor with `hyakusage`.

### The 2-GPU trap for I/O-bound arms

This is the cost decision specific to our workload. Our tiles arm is **I/O-bound, not
GPU-bound** — the same epoch took 4.1–9.5 h on klone depending only on which node it
landed on. The obvious fix is more dataloader workers, but on Tillicum **more CPUs
means more GPUs**, and billing multiplies by GPU count:

- 1 GPU → 8 dataloader CPUs → 1× billing rate
- 2 GPUs → 16 dataloader CPUs → **2× billing rate**

So 2 GPUs is only cheaper *per epoch* if 16 CPUs makes the tiles epoch **more than
twice** as fast. That is an empirical question, and it is the single best use of the
free demo hours. See [First runs](#first-runs-what-to-measure).

## Storage

`/gpfs/projects/makelab` — **1 TB**, backed up daily, "purged at the end of the
project." Note this is *not* archival: `/gscratch/makelab` on klone remains the system
of record for durable weights (see `scripts/model_comparison/yolo_baseline/README.md`).
Keep the existing pattern: **train on Tillicum, `rsync` `best.pt` back to
`/gscratch/makelab`.**

Undocumented and pending a reply from UW-IT: the quota-increase path, what defines
"end of project," and guidance for **many-small-file** access patterns. That last one
matters to us more than bandwidth — our tiles dataset is hundreds of thousands of small
JPEGs, and `du -sh` over it on klone's `/gscratch/scrubbed` exceeds a 2-minute timeout,
which is a metadata-throughput symptom rather than a size one.

> The docs describe demo accounts as receiving "100 GB of dedicated project storage,"
> while our provisioning email says 1 TB. Assume 1 TB (the email is specific to us) but
> confirm.

## Software environment

Tillicum's documented path is **Apptainer containers**, not conda modules —
see [UWrc/tillicum-containers](https://github.com/UWrc/tillicum-containers), which
ships a `pytorch_timm.def` example (we use `timm` for the ConvNeXt-V2 backbone) plus
`run_inference.slurm` and `array_inference.slurm` templates. Recommended image sources
are DockerHub and the NVIDIA NGC catalog.

**Our `environment.yml` should not be assumed to transfer.** It pins linux-64 packages
against **CUDA 11.8**, and Tillicum is H200 (sm_90) on Rocky 9. CUDA 11.8 nominally
covers sm_90, but a stack built for klone's OS and driver is not a safe bet on a
different distro and a newer card. **UNVERIFIED — do not plan around either outcome
until someone tries it.** The low-risk path is an NGC PyTorch container.

## Migrating our Slurm scripts

Our current header (`scripts/model_comparison/run_yolo_train.slurm`) against what
Tillicum accepts:

| klone (current) | Tillicum | why |
|---|---|---|
| `#SBATCH -p ckpt-g2` | **delete** | no partitions exist |
| `#SBATCH -q ckpt-gpu` | `#SBATCH --qos=normal` | or `long` once approved |
| `#SBATCH --requeue` | **delete** | nothing preempts |
| `#SBATCH --time=72:00:00` | `24:00:00` | 72 h needs `long` QoS |
| `#SBATCH --gpus-per-node=1` | `#SBATCH --gres=gpu:1` | also `--gpus=1` / `-G 1` |
| `#SBATCH --cpus-per-task=12` | **`8`** (1 GPU) | 12 CPUs on 1 GPU is rejected |
| `#SBATCH --mem=64G` | keep, or up to `200G` | no default; must be explicit |
| `#SBATCH --nodes=1 --ntasks=1` | unchanged | |

**`--cpus-per-task=12` with one GPU is the blocking incompatibility** — it is not a
tuning preference, it exceeds the hard 8:1 ratio and the job will not run.

Also drop the inherited mail settings. Our klone jobs carry
`MailType=END,FAIL,TIME_LIMIT` from a site default or the submit environment (it is
*not* in the script), which on a preemptable partition produced ~170 emails in one
night. On Tillicum there is no requeue storm to amplify it, but set `--mail-type=NONE`
explicitly so the behavior is the script's decision rather than the site's.

## First runs: what to measure

Spend the free demo hours on measurement, not on a production run. Three jobs, ~15
GPU-hours total, answering the questions the #70 budget depends on:

1. **`debug` QoS smoke test** (1 h, ~1 GPU-h) — does our environment run at all?
   This is where the conda-vs-container question gets settled.
2. **One pano epoch, 1 GPU** (~1–3 h). klone L40S baseline: 2.6–2.8 h/epoch. An H200
   should be meaningfully faster, but *how much* is the number the budget needs.
3. **One tiles epoch, 1 GPU (8 CPU) vs 2 GPUs (16 CPU)** (~10 GPU-h). Settles the
   2-GPU trap above, and — more importantly — tells us whether tiles-vs-pano in the
   #51 results is an **architecture finding or a storage artifact**. That question is
   currently unresolved and it materially affects what the baseline means.

Record the measured epoch times back into this file and into #70.

## Admin

- **Group:** `u_hyak_tillicum_makelab` (UW Groups Service). Jon is a member manager and
  can add users via the Membership tab; UW-IT can grant member-manager rights to others
  on request.
- **Maintenance:** second Tuesday monthly. Join the mailing list; put it on the run
  calendar so long jobs are not scheduled across it.
- **Support:** `help@uw.edu` with "Tillicum" in the subject; twice-weekly office hours.
- **Budget changes:** contact UW-IT; enforcement semantics (blocks new submissions vs.
  cancels running jobs) are **UNVERIFIED** and were asked about on 2026-07-30.

## Open questions

Sent to UW-IT 2026-07-30, unanswered at time of writing:

1. Does budget enforcement block *new submissions* or *cancel running jobs*?
2. Is there any way to get more CPUs per job without the GPU billing multiplier?
3. Guidance for many-small-file datasets on `/gpfs`; is packing into an archive or
   container format recommended?
4. Storage: quota-increase path, and what "purged at the end of the project" means.
5. Do the 100 free demo hours expire?
6. How to request `long` QoS access.

## Related

- #51 — supervised YOLO baseline (the runs that motivated this)
- #70 — stabilized rerun; **this is its unblocking dependency**
- `scripts/model_comparison/yolo_baseline/README.md` — preserved klone training record
- `hyak_yolo_runbook.sh`, `hyak_qwen_runbook.sh` — klone runbooks (repo root)
