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

### `debug` is free — but the two cost tools disagree (measured 2026-07-31)

The full QoS table, from `sacctmgr show qos`:

| QoS | Priority | UsageFactor | MaxWall | per-user cap |
|---|---:|---:|---|---|
| `debug` | 50 | **0.000000** | 1 h | 1 GPU, 8 CPU, 200 G, 1 node |
| `normal` | 25 | 1.000000 | 1 day | 48 GPU |
| `interactive` | 35 | 1.000000 | 8 h | — |
| `urgent` | 200 | 1.000000 | 3 days | 64 GPU |
| `long` | 25 | 1.000000 | 7 days | — |
| `wide` | 25 | 1.000000 | 1 day | 96 GPU |

`debug` bills at **UsageFactor 0** and carries *higher* priority than `normal` (50 vs
25), so every environment and throughput probe belongs there. That is the basis for the
"costs nothing" claim in `scripts/tillicum_smoke.slurm`.

**Caveat, unresolved:** `hyakusage` does not agree. The smoke job (`198638`, 2 min on
`debug`) shows up in its QoS breakdown as **0.03 GPU-hours, $0.03** — i.e. raw
wall-clock × $0.90 with the 0.0 UsageFactor apparently *not* applied, even though
`hyakusage`'s own header says "billable GPU hours = raw GPU hours × QOS multiplier."
Slurm's accounting config and the reporting tool are stating different things, and we do
not know which one ITBill actually follows. Worth asking UW-IT, but not urgent: `debug`
is capped at 1 h × 1 GPU, so the exposure is **at most $0.90 per job** even if
`hyakusage` turns out to be the honest one. Do not quote a "free" figure from this
without saying which tool it came from.

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
matters to us more than bandwidth.

### What actually has to move (measured 2026-07-30 on klone)

| dataset | size | train files | val files |
|---|---|---|---|
| `yolo/tiles` | **210 GB** | 557,413 | 161,002 |
| `yolo/pano` | **76 GB** | 150,063 | 42,875 |
| **total** | **286 GB** | **~911,000 files**, ~300 KB average | |

Two conclusions:

- **286 GB fits the 1 TB allocation** with room for checkpoints and run dirs. Storage
  quota is not a blocker for the #70 rerun.
- **~911,000 files is the blocker.** This is the many-small-file pathology in its pure
  form, and it is not hypothetical: `du -sh` over the tiles tree exceeded a 2-minute
  timeout three separate times before completing on a 550 s budget, and even
  `ls -lU | head -201` on `tiles/images/train` timed out. That is metadata throughput,
  not bandwidth.

**So pack into SquashFS, never copy the tree** — see
`scripts/model_comparison/pack_yolo_dataset.slurm`, and UW-IT's own recommendation in the
answers below. A per-file `rsync`/`scp` pays ~911,000 round trips; `tar` fixes the
transfer but not the destination, since untarring recreates all 911k files on `/gpfs` and
pays the metadata cost permanently. A SquashFS image is mounted read-only, so the
destination only ever holds **one file**.

Run the pack inside a Slurm job, **not on a klone login node** — that is exactly the
heavy-login-process reap that kills the SSH master.

**The subtle part is the Ultralytics label cache.** Ultralytics writes
`labels/<split>.cache` and validates it against a hash of the label+image **absolute
paths**. So (a) the `.cache` files must be *inside* the image — they exist on klone
already and a read-only mount cannot create them — and (b) mounting at a different path
than they were built under silently invalidates them and forces a full ~911k-file
rescan. Budget one slow first epoch on Tillicum to regenerate them, and do not mistake
it for the steady-state cost. Getting this wrong is invisible: training still works, it
is just permanently slow.

Staging order: **`pano` first (76 GB, 193k files)** — it unblocks the smoke test and the
pano epoch-time measurement while the much larger `tiles` image builds behind it.

> The docs describe demo accounts as receiving "100 GB of dedicated project storage,"
> while our provisioning email says 1 TB. Assume 1 TB (the email is specific to us) but
> confirm.

### How the data actually got there: regenerate, don't transfer (resolved 2026-07-31)

The SquashFS plan above is sound and it **worked** — klone job `37940649` packed
`pano` into a single 76 GB `pano.sqfs` in 2 h 38 m (sha256 `75be5150…aea1dd0a`, still on
klone at `/gscratch/scrubbed/jfroehli/yolo_squashfs/`). It solved the destination
problem exactly as designed: one file instead of 193k.

**It did not solve the transfer, and the transfer is the real blocker.** There is no
automated klone → Tillicum path: no shared filesystem (`/gscratch` and `/mmfs1` do not
exist on Tillicum), and neither end can authenticate to the other non-interactively —
both are Duo/keyboard-interactive with no `publickey`, so `BatchMode` fails in both
directions. Globus CLI is installed on neither side. Verified 2026-07-30.

So we **regenerated the dataset on Tillicum from Hugging Face** instead
(`scripts/model_comparison/run_yolo_data_prep_tillicum.slurm`, job `198910`, 4 h 40 m).
This is safe rather than merely convenient, because the prep is deterministic by
construction: `prepare_yolo_dataset.py` thins background tiles with an md5 of the file
stem specifically so the choice is stable across processes and runs (see
`_keep_background` and its comment about salted `hash()`).

**Verified equivalent to klone, 2026-07-31.** Counts match the #51 record in all eight
directories — and, because klone's tree is still live, we could go further than counts.
The md5 of the sorted filename list is **identical on both clusters for all eight**, so
the two datasets contain the same files under the same train/val split, not merely the
same number of them:

| directory | files | md5 of sorted filename list (klone == Tillicum) |
|---|---:|---|
| `tiles/images/train` | 557,413 | `f5e664fbd64651be0ff89045d217ff50` |
| `tiles/images/val`   | 161,002 | `f48f91878d34e72a7b0b2dfd48f6c90a` |
| `pano/images/train`  | 150,063 | `a8f3af23a61952ef1209435ca0e295ea` |
| `pano/images/val`    |  42,875 | `83f5371483f3dbfa5a6aece939b86901` |
| `tiles/labels/train` | 557,413 | `19c5c38ae148a907042d213c84002cc6` |
| `tiles/labels/val`   | 161,002 | `711d5a59a2d6019ac1857133b167266e` |
| `pano/labels/train`  | 150,063 | `015fc700faee211e37af3f8edeca0dfc` |
| `pano/labels/val`    |  42,875 | `2fc039583a9b0cc08ca57b11739f5a5a` |

Reproduce on either cluster with, per directory:
`ls -U <root>/yolo/<d> | sort | md5sum` — klone root `/gscratch/scrubbed/jfroehli`,
Tillicum root `/gpfs/scrubbed/jfroehli`. The prep run itself reported 767,840 boxes over
192,938 panos with **0 read errors**. Anything trained on Tillicum is therefore
comparable to the #51 klone arms on the data axis.

Two caveats that travel with this:

- **It cost $4.20** — 4.67 GPU-hours at `normal` QoS. Tillicum rejects CPU-only jobs, so
  a data-prep job must hold an H200 it never uses. That is structural, not an error, but
  it is the argument for keeping prep on klone (free) whenever a dataset already exists
  there and *can* be reached. Here it could not be.
- **It landed on `/gpfs/scrubbed`, not the 1 TB project quota** — 1.61 TB across 2.28 M
  files, counting the ~462 GB Hugging Face source cache. `scrubbed` is purged on an
  inactivity timer, so this copy is not durable: it is fine while an arm is actively
  reading it, and must not be assumed to survive a gap between arms. Durable artifacts
  still belong on klone's `/gscratch/makelab`.

Keep the SquashFS path documented anyway: it is the right answer the moment a transfer
route exists (a Globus endpoint, or an intermediate host either end can reach), and the
`pano.sqfs` image is already built.

## Software environment

Tillicum's documented path is **Apptainer containers**, not conda modules —
see [UWrc/tillicum-containers](https://github.com/UWrc/tillicum-containers), which
ships a `pytorch_timm.def` example (we use `timm` for the ConvNeXt-V2 backbone) plus
`run_inference.slurm` and `array_inference.slurm` templates. Recommended image sources
are DockerHub and the NVIDIA NGC catalog.

**Our `environment.yml` should not be assumed to transfer.** It pins linux-64 packages
against **CUDA 11.8**, and Tillicum is H200 (sm_90) on Rocky 9. CUDA 11.8 nominally
covers sm_90, but a stack built for klone's OS and driver is not a safe bet on a
different distro and a newer card. **Still UNVERIFIED** — nobody has tried to solve
`environment.yml` here, because the YOLO baseline does not need it. The low-risk path
remains an NGC PyTorch container.

### The YOLO stack, however, is VERIFIED (2026-07-31)

`scripts/tillicum_setup_env.sh` builds it, and it reproduces the klone `#51` toolchain
**exactly** — confirmed by running the interpreter, not by reading a lockfile:

```
3.11.15   torch 2.13.0+cu126   ultralytics 8.4.105
```

which is character-for-character what klone's training logs report
(`Ultralytics 8.4.105 · Python-3.11.15 · torch-2.13.0+cu126`). It lives at
`/gpfs/projects/makelab/$USER/envs/rampnet-yolo` — the backed-up 1 TB allocation rather
than `scrubbed`, because the environment is small and annoying to rebuild while the
dataset is huge and reproducible. It has since driven a full 4 h 40 m production job
(`198910`) without incident.

Two details in that script worth keeping if it is ever edited:

- **Conda is not optional here.** Tillicum's *system* python is 3.9.25, and the cu126
  wheel index tops out at torch 2.8.0 for 3.9 — so a naive `pip install torch
  ultralytics` silently yields 2.8.0 + 8.4.113 and a baseline nobody can publish. The
  conda module is the only way to get 3.11 on this cluster.
- **The version check is a gate, not a report.** The script refuses to install a
  substitute torch and exits non-zero on any mismatch. That matters because installing
  `ultralytics` last can pull its own torch over the pinned one; the post-install assert
  is what catches it. A mismatched toolchain produces plausible numbers and an
  unpublishable comparison, which is the failure mode `#71`'s protocol exists to
  prevent.

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

### Measurement 3, pre-registered in full (2026-08-09)

Written **before the job was submitted**, so the reading cannot be chosen after seeing the
number. Same discipline as the #71 checkpoint-selection protocol.

**What runs.** One epoch of `y11x_tiles` on **1 H200 / 8 CPU**, resumed from a *copy* of its
epoch-21 checkpoint, with `batch=12`, `imgsz=1024`, `workers=8` — i.e. training math
identical to the klone arm, so throughput is the only variable. `--time=07:00:00` bounds the
cost at **7 GPU-h = $6.30**, inside the remaining $29.65 credit.

**Why 7 h is the wall.** It is deliberately the falsification threshold, not a round number:
klone's dedicated L40S does this epoch in **7.06–7.38 h**. A job that cannot complete one
epoch inside a 7 h wall has ruled out the migration by itself, at a cost of $6.30.

**Why this arm and not another.** `docs/tillicum.md` says above that our tiles arm is
I/O-bound, which would make a faster GPU worthless. That is true of `y26_tiles` and
`y11l_tiles`; it is **not** true of `y11x_tiles`, and the distinction is what makes this
measurement worth paying for. On the same dedicated node and the same storage,
`y26_tiles_l40s` sustains **34.0 img/s** while `y11x_tiles` sustains **25.2 img/s** — the
larger model is slower against storage that demonstrably serves the smaller one faster. The
July batch probe agrees: batch 3 and batch 12 both sustained ~15.6 img/s, so throughput is
not batch- or queue-limited. `y11x_tiles` is the one tiles arm where H200 silicon should
convert into wall-clock.

**Baseline to beat** (measured on klone `gpu-l40s`, job `38063498`, 21 epochs):

| quantity | klone L40S |
|---|---|
| epoch wall-clock | 7.06–7.38 h (dead flat) |
| sustained rate | **2.1 it/s** at batch 12 (46,452 iters/epoch) = 25.2 img/s |
| Ultralytics storage probe | `read: 8.3–11.8 MB/s, size: 252–338 KB` |

**Primary metric: sustained it/s**, because it is readable within ~30 minutes of the job
reaching steady state and is directly comparable at identical batch and imgsz. Secondary:
the full epoch wall from `results.csv`, and the `Slow image access` probe line, which is what
tells us whether Tillicum's `/gpfs/scrubbed` is better or worse than klone's for ~300 KB
files.

**Pre-registered readings.** Speedup `S` = (measured it/s) / 2.1:

| S | reading | consequence |
|---|---|---|
| **≥ 1.7×** | the pano arm's 1.74–2.05× transfers to tiles | finishing the arm costs **~$121–138** and lands ~5 days earlier than klone. Worth putting to Jon as a funded decision. |
| **1.2–1.7×** | real but modest | not worth ~$100+ on an arm whose curve is flat at +0.00037/epoch. Stay on klone, free. |
| **< 1.2×** | the H200 buys nothing here | migration is dead, **and** the compute-bound reading of the klone evidence above is wrong — the arm is storage-bound on both clusters and the img/s gap between `y11x` and `y26` needs another explanation. |

**What this does NOT answer, correcting item 3 above.** That item claims this measurement
"tells us whether tiles-vs-pano in the #51 results is an architecture finding or a storage
artifact." It does not, and the overstatement is worth fixing before it is quoted. Timing one
epoch establishes whether *tiles training throughput* is storage-limited on a given
filesystem. Whether the tiles arms' **accuracy** trails pano because they were starved of
epochs is a different question, and it is answered by running a tiles arm to ep60 — not by a
stopwatch. This probe prices that run; it does not substitute for it.

**Contamination controls.**

- The source is the **durable snapshot** `/gscratch/makelab/jonf/rampnet_yolo_baseline_51/y11x_tiles/weights/last.pt`,
  not the live file, which the running klone job rewrites at every epoch boundary. sha256
  `52cf5013e696…c064`, verified at each hop.
- It writes to its **own run directory**, `y11x_tiles_h200_probe`. The klone arm keeps
  running, untouched. Paths are rewritten with the committed
  `retarget_yolo_checkpoint.py` — without it, `resume=True` restores the checkpoint's
  `save_dir` and writes back into the original run directory.
- `epochs=60` is **left alone**. Shortening it would change the LR-decay denominator, and
  leaving it means the probe run is a legitimate continuation that can simply keep going if
  the answer is favourable.
- **The probe's mAP is not a reportable result.** It is a third lineage of this arm, run to
  measure time. Report its throughput; report accuracy only from the klone arm.

#### As run — job `217298`, 2026-08-09

Submitted 16:5x PDT, node **`g016`** (H200, 143,771 MiB). Exact commands, in order, from a
clean starting point. They assume the tiles dataset is already on Tillicum, which it is —
`/gpfs/scrubbed/jfroehli/yolo/tiles`, regenerated from HF and md5-verified against klone per
the table above.

```bash
# 1. Stage the checkpoint from the DURABLE SNAPSHOT, not the live run dir.
#    sha256 52cf5013e696885b6c30cb0bf783256bf06d4b756c123277e6482e9f453fc064
#    verified identical at all three hops: klone -> workstation -> Tillicum.
RUN=/gpfs/projects/makelab/jfroehli/yolo_runs/y11x_tiles_h200_probe
mkdir -p $RUN/weights
# scp klone:/gscratch/makelab/jonf/rampnet_yolo_baseline_51/y11x_tiles/weights/last.pt $RUN/weights/
# scp klone:/gscratch/makelab/jonf/rampnet_yolo_baseline_51/y11x_tiles/results.csv      $RUN/
sha256sum $RUN/weights/last.pt $RUN/results.csv

# 2. Rewrite the six cluster-absolute paths. Without this, resume=True restores the
#    checkpoint's save_dir and writes back into the klone run directory.
/gpfs/projects/makelab/jfroehli/envs/rampnet-yolo/bin/python \
  ~/RampNet/scripts/model_comparison/retarget_yolo_checkpoint.py \
  $RUN/weights/last.pt \
  --data /gpfs/scrubbed/jfroehli/yolo/tiles/data.yaml \
  --project /gpfs/projects/makelab/jfroehli/yolo_runs \
  --name y11x_tiles_h200_probe --apply
# -> backup at weights/last.pt.preretarget; out sha256 b2c6210c0b5f...762b;
#    "verified: all six path keys persisted on reload"

# 3. Submit, bounded at 7 h.
cd ~/RampNet && mkdir -p logs
PYTHON=/gpfs/projects/makelab/jfroehli/envs/rampnet-yolo/bin/python \
YOLO_DATA=/gpfs/scrubbed/jfroehli/yolo/tiles/data.yaml \
YOLO_IMGSZ=1024 BATCH=12 EPOCHS=60 NAME=y11x_tiles_h200_probe \
  sbatch --time=07:00:00 --job-name=y11x_tiles_h200_probe \
         scripts/model_comparison/run_yolo_train_tillicum.slurm
```

Confirmed at startup, from the job's own echoed args: `batch=12, imgsz=1024, workers=8,
epochs=60, patience=20, seed=0, lr0=0.01, close_mosaic=10, optimizer=auto`, and
`save_dir=…/y11x_tiles_h200_probe` — identical training math to the klone arm, writing to its
own directory. The retarget dry run also reported `epochs done: 21 of 60,
best_fitness: 0.47072`, which equals the klone arm's ep21 `mAP50-95` to five decimals —
an independent confirmation that this Ultralytics build selects on **mAP50-95 alone**, not the
0.1/0.9 blend (see `scripts/model_comparison/yolo_baseline/README.md`).

**Cosmetic wart, same one klone had.** The launcher's header echoes `base: yolo11l.pt` on a
resume, because `YOLO_CKPT` is unset and `resume=True` ignores it anyway. The klone script was
fixed to print `MODE: RESUME` / `MODE: FRESH`; this one has not been. The `[resume]` line
below it is the trustworthy signal. Do not read the `base:` line as the architecture — confirm
from the `YOLO11x summary: … parameters` line instead.

#### Result: 2.38×, and the reason is storage, not the GPU

Read at iteration ~880 of 46,452, i.e. early in the epoch and well clear of warmup. **The
confirmed full-epoch time is still pending** and will be differenced from two consecutive
`results.csv` rows once the job reaches the 7 h wall.

| | klone L40S (`38063498`) | Tillicum H200 (`217298`) | ratio |
|---|---:|---:|---:|
| sustained | 2.1 it/s | **5.0 it/s** | **2.38×** |
| images/s at batch 12 | 25.2 | 60.0 | 2.38× |
| Ultralytics storage verdict | `Slow image access detected` | **`Fast image access ✅`** | |
| measured read | 8.3 ± 3.3 / 11.8 ± 4.8 MB/s | **542.9 ± 78.3 MB/s** | **~46–65×** |
| ping | 0.7 ± 0.1 / 1.3 ± 1.6 ms | 0.5 ± 0.1 ms | |
| file size probed | 252–338 KB | 338.1 KB | identical |

`S = 2.38 ≥ 1.7`, so by the pre-registered table the migration is **worth putting to Jon as a
funded decision**. Revised cost, using ~3.0 h/epoch (2.58 h of training at 5.0 it/s plus
validation over 161,002 tiles): **39 epochs ≈ 117 GPU-h ≈ $105**, about 4.9 days of compute
against klone's 11.5 days. That is *cheaper* than the $121–138 the pre-registration projected,
because the speedup beat the pano arm's 1.74–2.05×.

**The pre-registered rationale for choosing this arm was wrong, and the data says so plainly.**
The section above argued `y11x_tiles` was worth moving because it is *compute*-bound on klone,
inferring that from `y26_tiles_l40s` sustaining 34.0 img/s against `y11x_tiles`' 25.2 on the
same storage. Multiply those through by the file size and the inference collapses:

- `y11x_tiles` consumed 25.2 img/s × 338 KB = **8.5 MB/s**
- `y26_tiles_l40s` consumed 34.0 img/s × 338 KB = **11.5 MB/s**
- klone's own probe measured that filesystem at **8.3–11.8 MB/s**

Both arms were sitting *on* the storage ceiling, and the img/s gap I read as an architecture
difference is inside the measured variance of a contended shared filesystem. On Tillicum the
same arm consumes 20.3 MB/s against 542.9 MB/s available — **4% of capacity** — so there it is
genuinely GPU-bound, and 2.38× is the H200-vs-L40S compute ratio showing through once storage
stops being the wall.

The pre-registration anticipated this reading being falsified only by a result *below* 1.2×.
It was falsified by a result well above it, via a channel the threshold table did not
contemplate — the storage probe line, which was listed only as a secondary metric. Worth
remembering that the secondary metric carried the finding.

**What this implies beyond the arm we probed, and did not pay to learn.** If klone's tiles
throughput is a storage ceiling rather than a per-model property, it applies to `y11l_tiles`
and `y26_tiles` too — the two arms currently at ep9 after ~23 and ~35 days of projected
runway. They are smaller models than `y11x`, so unbinding storage should help them at least as
much. That reframes the whole tiles column of #51 from "these arms are slow" to "these arms
are on the wrong filesystem", and it is a much better explanation of the ckpt slice-ceiling
history recorded above than anything model-specific. **It is an inference, not a measurement**
— no tiles arm other than `y11x` has been timed on Tillicum.

## Admin

- **Group:** `u_hyak_tillicum_makelab` (UW Groups Service). Jon is a member manager and
  can add users via the Membership tab; UW-IT can grant member-manager rights to others
  on request.
- **Maintenance:** second Tuesday monthly. Join the mailing list; put it on the run
  calendar so long jobs are not scheduled across it.
- **Support:** `help@uw.edu` with "Tillicum" in the subject; twice-weekly office hours.
- **Budget changes:** contact UW-IT; enforcement semantics (blocks new submissions vs.
  cancels running jobs) are **UNVERIFIED** and were asked about on 2026-07-30.

## Answers from UW-IT (Sumaiya Sathar, 2026-07-30)

All six questions answered. Several change the plan, so they are recorded here rather
than left in a mailbox.

1. **Budget enforcement is safe.** Set to **$1,500/month, enforcement active**. Enforced
   budgets **do not cancel running jobs** — a job already running continues to
   completion; enforcement only **blocks new submissions** until the cap is raised or
   the period rolls over. So the cap cannot destroy an in-flight experiment, which was
   the only reason to have preferred warn-only.
2. **No CPUs without GPUs — confirmed, and it is a rate-model constraint, not a
   scheduling one.** There is no QoS granting extra cores without the matching GPU
   allocation; UW-IT is considering it, but the UW-approved rate model cannot charge for
   CPUs independently, and unbalanced nodes are undesirable. **So the 2-GPU dataloader
   trick is the only lever**, and its 2× billing is unavoidable — see the trap above.
3. **Many small files → `/gpfs/scrubbed`, and pack into SquashFS.** This is the big one.
   Active many-small-file datasets belong on **`/gpfs/scrubbed`** (larger capacity, has
   an automatic cleanup policy) rather than the 1 TB project quota, and UW-IT explicitly
   recommends **SquashFS** for this shape. Data Commons is an option if the dataset can
   be public.
4. **Storage grows in 1 TB increments** on request to `help@uw.edu` (subject "Tillicum")
   with a workflow justification — it is shared active-compute storage, not archival.
5. **No project end date.** "Purged at the end of the project" simply means that if we
   stop using Tillicum we must copy data off so the space can be reclaimed. Nothing
   expires on a timer.
6. **1 TB is correct for us** — the 100 GB figure in the public docs applies to *demo*
   accounts. Providing a worktag makes it a regular account, which is why we got 1 TB.
7. **The 100 free GPU hours do not expire.** They remain until used.
8. **`long` QoS requires the Special QoS Access Request Form** — still to be submitted,
   then reviewed. Not a blocker for the measurement phase (everything fits in 24 h), but
   it gates comfortable production runs.

### What this changes

- **Dataset goes to `/gpfs/scrubbed`, not `/gpfs/projects/makelab`.** The 1 TB project
  quota is for artifacts we want backed up (checkpoints, results), not the 286 GB of
  training data. This also removes storage pressure from the plan entirely.
- **Pack with SquashFS, not tar.** See
  `scripts/model_comparison/pack_yolo_dataset.slurm`. tar fixes the *transfer* but not
  the *destination* — untarring recreates all ~911k files on `/gpfs` and pays the
  metadata cost permanently. A SquashFS image is mounted read-only and the destination
  only ever holds one file.
- **Enforced budget is safe to leave on**, so the runaway-spend guard costs us nothing.
- **The `DEVICE=0` / 2-GPU design in the Tillicum launcher is now confirmed necessary**
  rather than a workaround pending a better answer.

## Related

- #51 — supervised YOLO baseline (the runs that motivated this)
- #70 — stabilized rerun; **this is its unblocking dependency**
- `scripts/model_comparison/yolo_baseline/README.md` — preserved klone training record
- `hyak_yolo_runbook.sh`, `hyak_qwen_runbook.sh` — klone runbooks (repo root)
