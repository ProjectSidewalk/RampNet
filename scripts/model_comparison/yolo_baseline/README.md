# Supervised YOLO baseline — training record (issue #51)

This directory is the **preserved record** of the supervised-YOLO baseline runs
trained on the RampNet dataset on Hyak (klone). It exists so the experiment survives
`/gscratch/scrubbed`'s ~21-day auto-purge and is reproducible/defensible for the paper.

**What's here (small, text, committed):**

- `run_yolo_train.slurm` — the per-config training launcher (one config per job).
- `hyak_yolo_runbook.sh` — the operator runbook: env → data → prep → train → collect.
- `runs/<config>/results.csv` — per-epoch training curves (the primary evidence).
- `runs/<config>/args.yaml` — the exact resolved config for each run.

**What's deliberately NOT here** (see the repo `.gitignore` philosophy — curated record
in git, bulk/binary/regenerable out):

- **Model weights** (`best.pt`/`last.pt`) — binaries; they belong in durable external
  storage, not git history. See "Where the weights live" below.
- **Slurm logs** (`*.out`/`*.err`) — 30 MB of progress-bar spam, already gitignored;
  the few relevant lines are quoted below.
- **Benchmark-city eval numbers** — these `results.csv` values are the **internal YOLO
  val-split proxy** used to judge *training health*, **not** the reported baseline. The
  reported numbers come from local `compare.py` benchmark eval on each `best.pt`.

## Config grid

Six configs = {YOLO11l, YOLO11x, YOLO26l} × {tiles @1024, pano @1280}. One dropped.

| Config       | Model    | Input        | Base ckpt     | Batch |
|--------------|----------|--------------|---------------|-------|
| `y11l_tiles` | YOLO11l  | tiles @1024  | `yolo11l.pt`  | 6     |
| `y11x_tiles` | YOLO11x  | tiles @1024  | `yolo11x.pt`  | 3     |
| `y26_tiles`  | YOLO26l  | tiles @1024  | `yolo26l.pt`  | —     |
| `y11l_pano`  | YOLO11l  | pano @1280   | `yolo11l.pt`  | 4     |
| `y11x_pano`  | YOLO11x  | pano @1280   | `yolo11x.pt`  | 2     |
| `y26_pano`   | YOLO26l  | pano @1280   | `yolo26l.pt`  | 4     |

## Status & findings (snapshot: 2026-07-28, training in progress)

Val-split proxy mAP50 from `results.csv`. **All configs show an inflated epoch-1 value
(~0.65–0.78): that is the COCO-pretrained backbone's val *before* fine-tuning + mosaic
augmentation engage — a pretrained artifact, not a real score.** The meaningful signal is
whether a run *recovers* from the standard post-epoch-1 dip and climbs.

| Config       | Epochs | ep1 (artifact) | Current mAP50 | Assessment |
|--------------|--------|----------------|---------------|------------|
| `y26_pano`   | 12     | 0.738          | **0.624 ↑**   | ✅ recovering steadily, climbing back toward/past ep1 |
| `y11l_tiles` | 2      | 0.655          | 0.309         | 🟡 early, post-dip, still volatile |
| `y26_tiles`  | 3      | 0.647          | 0.280         | 🟡 early, post-dip, still volatile |
| `y11x_pano`  | 6      | 0.777          | **0.000**     | ❌ collapsed — 4 straight epochs at literal 0 |
| `y11l_pano`  | 7      | 0.779          | **~0.024**    | ❌ collapsed — flickering near 0 (P≈0.94 / R≈0.03) |
| `y11x_tiles` | 0      | —              | —             | ⏹ dropped 2026-07-27 (GPU-saturated: epoch ~10 h > ckpt slice) |

### The collapse (`y11x_pano`, `y11l_pano`)

Both **YOLO11 pano** runs failed to recover from the post-epoch-1 dip: val mAP fell to
~0 and stayed there for 4+ epochs. Crucially:

- **Training loss stayed healthy and decreasing** (box 1.5→1.4, cls 1.3→1.2). Not a
  gradient blow-up.
- **No NaN/Inf, no AMP failure** anywhere in the logs. A *validation-side* collapse.
- **Every other config was unaffected** — `y26_pano` (same input, different arch) and
  both tiles runs recovered from the same dip.

**Leading hypothesis:** BatchNorm/EMA instability from a small *physical* batch at high
input resolution — `y11x_pano` ran `batch=2`, `y11l_pano` `batch=4`, both at
`imgsz=1280`; tiles ran `imgsz=1024`/`batch=6` and YOLO26 (NMS-/DFL-free) tolerated the
regime. Gradient accumulation (`nbs=64`, active) fixes the optimizer step but **not** BN
statistics, which are computed on the 2–4 physical samples. Fix + confirmation is the
subject of the stabilized-rerun issue. **These runs are not reportable** — their `best.pt`
holds only the epoch-1 pretrained artifact.

## Provenance

- **Training code:** `run_yolo_train.slurm` + `hyak_yolo_runbook.sh`, committed in this
  directory. (The Hyak runs used an rsync of the working tree; this commit anchors it.)
- **Dataset:** `projectsidewalk/rampnet-dataset` (HF), pulled onto the cluster via
  `download_dataset.py`. Prep: `prepare_yolo_dataset.py --box-size fixed:0.03
  --bg-keep-frac 0.15`, tiles `--geometry tiles` @1024, pano `--geometry pano` @1280.
- **Toolchain:** Ultralytics 8.4.105 · Python 3.11.15 · torch 2.13.0+cu126.
- **Hardware:** NVIDIA L40 (45 GB), 1 GPU/job, Hyak `ckpt-g2` (preemptable/requeue).
- **Hyperparameters (resolved):** `epochs=60`, `patience=20`, `optimizer=auto`,
  `lr0=0.01`, `lrf=0.01`, `momentum=0.937`, `weight_decay=0.0005`, `warmup_epochs=3.0`,
  `close_mosaic=10`, `amp=true`, `seed=0`. Per-run detail in each `runs/<config>/args.yaml`.
- **Slurm job IDs:** y11l_tiles 37745358 · y26_tiles 37745360 · y11l_pano 37745361 ·
  y11x_pano 37745362 · y26_pano 37745363.
- **Run dates:** 2026-07-26 → in progress as of 2026-07-28.

## Where the weights live

`best.pt` files are **not** in git. Durable homes:

- **Healthy runs (the reported baseline):** _TODO — stage to Hugging Face
  (`projectsidewalk/…`) or lab storage and record the URL/path here._
- **Collapsed runs (`y11x_pano`, `y11l_pano`):** not worth keeping — the epoch-1 artifact
  is non-reportable; the `results.csv` curve here is the evidence. Left to expire with
  scratch.

## Reproducing

On klone, via the runbook (each stage is idempotent):

```bash
bash hyak_yolo_runbook.sh env      # lean venv on scratch (ultralytics + cu126 torch)
bash hyak_yolo_runbook.sh data     # pull RampNet dataset from HF (hours)
bash hyak_yolo_runbook.sh prep     # build tiles/ and pano/ YOLO datasets
bash hyak_yolo_runbook.sh train    # sbatch the 6 configs concurrently
bash hyak_yolo_runbook.sh collect  # rsync best.pt back; eval locally with compare.py
```
