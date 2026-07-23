# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for the RampNet paper (ICCV'25 workshop): a two-stage pipeline that (1) auto-generates a large curb ramp detection dataset by translating open government curb-ramp GPS locations into pixel coordinates on Google Street View panoramas, and (2) trains a curb ramp detection model on that dataset. Published artifacts: `projectsidewalk/rampnet-dataset` and `projectsidewalk/rampnet-model` on Hugging Face.

## Environment & commands

- Conda env (Linux + CUDA 11.8; env file pins linux-64 packages — this does not run natively on Windows):
  ```bash
  conda env create -f environment.yml
  conda activate sidewalkcv2
  ```
- There is no test suite, build step, or lint config. Scripts are configured by editing constants at the top of each file (e.g. `MODEL_CHECKPOINT_PATH`, `EVALUATE_ON_MANUAL_DATASET`, `CONSIDER_MANUAL`), not CLI args.
- Every long-running script has a matching `.slurm` launcher (the paper's runs used a Slurm cluster; Stage 2 training used 16x L40s). Stage 2 training is DDP — locally it's launched via `torchrun` (see `stage_two/run_train.slurm`); single-process fallback works since `setup_distributed()` defaults to world_size 1.
- `download_dataset.py` (repo root) downloads the pre-generated Stage 1 dataset from Hugging Face into `./dataset/{train,val,test}` — the shortcut that skips all of Stage 1.

## Pipeline order (Stage 1 → Stage 2)

Stage 1 has two halves — the crop model, then dataset generation, which must run in this order:

1. **Crop model, round 1** (`stage_one/crop_model/ps_model/`): `data/download_data.py`, then `./splititup.sh dataset_1` to split, then `model/train.py` → `best_model.pth`. Trains on Project Sidewalk crops.
2. **Crop model, round 2** (`stage_one/crop_model/ps_and_manual_model/`): copy round-1 `best_model.pth` here **renamed to `ps_model.pth`**, put the manual crop dataset in `dataset_1/`, run `train.py` → `best_model.pth`. This final model is what dataset generation loads (hardcoded path in `dataset_generation/inference_isolator.py`).
3. **Dataset generation** (`stage_one/dataset_generation/`), in order: `combine_location_data.py` (→ `all_locations.csv`) → `generate_dataset_meta.py` (→ `dataset.jsonl`) → `generate_negative_panos.py` (→ `negativepanos.jsonl`) → manually merge into `finaldataset.jsonl` (paper used ~20% negatives) → `download_dataset.py` (fetches GSV tiles, runs the crop model to place points; → `../../dataset/`) → `split_dataset.py` (→ `dataset_split`, then delete `dataset` and rename). Requires city location/street geojson files in `location_data/` and `street_data/` (see README for sources). Set `CONSIDER_MANUAL = True` in `split_dataset.py` if the generated dataset will be evaluated against `manual_labels` — otherwise the random split leaks eval panos into train/val.
4. **Stage 1 evaluation**: `stage_one/dataset_evaluation/evaluate.py` compares generated labels against `manual_labels/`.
5. **Stage 2** (`stage_two/`): `train.py` (1 epoch default, saves `best_model.pth`), `evaluate.py` (prints metrics and writes PR curves to `evaluation_results/`), `demo.py`. Toggle `EVALUATE_ON_MANUAL_DATASET` in `evaluate.py` to pick the benchmark; **delete `evaluate_cache/` whenever you change eval settings** — stale caches silently corrupt results. Note `evaluation_results/` is committed from past runs, so its presence doesn't mean an eval succeeded.

## Architecture notes

- **One model architecture, now consolidated in `rampnet/model.py`**: `KeypointModel` — a timm `convnextv2_base.fcmae_ft_in22k_in1k_384` backbone with a small conv + bilinear-upsample head producing a single-channel keypoint heatmap. It is **defined once** in `rampnet/model.py` and imported everywhere else (`stage_two/{train,evaluate,demo}.py`, `stage_one/crop_model/*/train.py`, `ps_and_manual_model/evaluate.py`, `dataset_generation/inference_isolator.py`, `scripts/export_hf_model.py`). The crop-vs-pano difference is a **constructor argument**, not a copy: `heatmap_size`, with named constants `CROP_INPUT_SIZE`/`CROP_HEATMAP_SIZE` (1024×352 input → 256×88 heatmap) and `PANO_INPUT_SIZE`/`PANO_HEATMAP_SIZE` (2048×4096 input → 512×1024 heatmap). A change to the architecture only needs to touch `rampnet/model.py` — **except** the HF package's `scripts/hf_package/modeling_rampnet.py`, a verbatim copy synced from `rampnet/model.py` by the exporter, which must be kept in step or published-model checkpoints won't load. (Historical note: this class used to be copy-pasted across ~7 files; it was consolidated, so older "propagate to all copies" guidance no longer applies.)
- **Labels are points, not boxes**: JSON metadata alongside each pano jpg holds normalized (x, y) curb ramp coordinates; training targets are Gaussian heatmaps (sigma 10 on 512×1024). Detections are extracted from predicted heatmaps with `skimage.feature.peak_local_max` (min_distance=10, threshold 0.5). Evaluation matches predictions to ground truth within a normalized radius (0.022 for panos).
- **Dataset layout on disk**: `dataset/{train,val,test}/<pano_id>.jpg` + `<pano_id>.json`. `manual_labels/*.txt` are YOLO-format (`class cx cy w h`, normalized) gold-standard labels for 1,000 panos; the images themselves live in the HF dataset, not this repo.
- `stage_one/dataset_generation/search_panos.py` and the tile-fetching code in its `download_dataset.py` talk directly to unofficial Google Street View endpoints (tile server + `streetlevel`-style search), so they're network-heavy and can break if Google changes the API.
