# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for the RampNet paper (ICCV'25 workshop): a two-stage pipeline that (1) auto-generates a large curb ramp detection dataset by translating open government curb-ramp GPS locations into pixel coordinates on Google Street View panoramas, and (2) trains a curb ramp detection model on that dataset. Published artifacts: `projectsidewalk/rampnet-dataset` and `projectsidewalk/rampnet-model` on Hugging Face.

## Record scientific progress in GitHub, as it happens

**This is a research repo, and the notes are part of the result.** Every experiment, negative result, caveat, and methodological decision gets written into GitHub — a committed doc, a PR body, or an issue comment — *at the time it is produced*, not batched up at the end. The paper is written from these notes; a finding nobody can find later did not happen.

**Why this is a standing rule and not a preference:** **uncommitted** results are one cleanup away from gone. Scratch worktrees under `%TEMP%` get removed when the session that owned them ends, and a removed worktree is also deregistered from `.git/worktrees` — so anything not committed at that moment is unrecoverable, and `git` cannot even tell you it existed. (Committed-and-pushed work in such a worktree is perfectly safe; the branch outlives the directory. Before concluding a branch is gone, check `git ls-remote --heads origin <branch>` — a missing local worktree proves nothing.)

In practice:

- **Commit and push early.** A WIP commit on a branch is free. Never let hours of analysis sit uncommitted, and never leave the *only* copy of uncommitted work in a directory something else controls.
- **Findings go in `docs/*.md`, committed**, with numbers reproducible from a committed script plus the committed bundles / `.model_cache`. If a number can't be re-derived from the repo, say where it came from (which split, which script, which run date).
- **Run status and in-flight experiments go in issue comments** as they happen — what was submitted, where, and what it's expected to show. A run whose result you are waiting on should be discoverable by someone who isn't you.
- **Negative results are results.** "We tried X and it didn't work" is the most commonly lost and among the most valuable — it's why nobody re-proposes it. See `scripts/analysis/peak_nms_check.py` (#62) for the shape: the answer was "no", and the script that proves it is committed.
- **Gaps are content.** If a split, model, or analysis was *not* run, say so explicitly and say why. An omission is indistinguishable from a withheld result — that ambiguity is exactly what the coverage matrix in `docs/model_comparison.md` exists to remove.
- **Caveats travel with the numbers**, in the same document, not in a separate "limitations" note nobody reads alongside the table.

## Every experiment must be replicable by someone else, from a clean clone

**This is not a nice-to-have and it is not "later".** The test for whether an experiment is
finished: *could a new student clone this repo, follow written instructions, obtain every input,
re-run it, and get our numbers?* If the answer is no, the experiment is not done — it is a result
that happens to live on one machine.

Before an experiment counts as complete:

- **The script is committed and configured from CLI args or committed constants** — never from
  edits made during a session. A number produced by a script that no longer exists in that form is
  not reproducible, it is a memory.
- **Every input is retrievable by someone else**: committed in the repo, or published to Hugging
  Face with the exact identifier written down. `.model_cache/`, `benchmark/*/panos/`, `~/Downloads`
  and "it's on the lab box" are **local conveniences, not inputs**. If an input cannot be published
  (size, licence), say so **explicitly, in the doc, next to the number**, and name what would
  unblock it.
- **Human judgments are committed per-rater**, and the **rubric they were made under travels in the
  same file**. A verdict whose scheme is unknown cannot be reused, compared, or trusted — and a
  rubric that changed mid-pass silently averages two definitions into one rate.
- **Derived artifacts carry a content hash**, so a regenerated copy can be *proven* identical
  rather than assumed to be. Regeneration that silently drifts is worse than no regeneration.
- **The run instructions live in the repo**, as exact commands in order, not in a chat log or a PR
  comment. Someone who was not in the room is the audience.
- **Deliberate omissions are content.** "Requires the native-res panos from HF #21, not yet
  published" belongs beside the result, not in someone's head. An unstated gap is indistinguishable
  from a hidden one.

When a task involves human raters, assume from the start that **a second rater will repeat it and
the two will be compared** — per-rater files, a stable item list, and an agreement script, not a
single blob that has to be reverse-engineered later.

## Environment & commands

- Conda env (Linux + CUDA 11.8; env file pins linux-64 packages — this does not run natively on Windows):
  ```bash
  conda env create -f environment.yml
  conda activate sidewalkcv2
  ```
- There is a pytest suite in `tests/` — run it with `pytest -q` (about 30 seconds). It is CPU-only, needs no network, and reads only committed fixtures (`manual_labels/`, `benchmark/*/records.jsonl`, `benchmark/*/verdicts.json`), because models are built with `pretrained_backbone=False`. **Keep it that way**: a test that needs a GPU, a checkpoint, or a network call belongs behind a skip, not in the default run. `requirements-dev.txt` is the minimal dependency set for it, and `.github/workflows/tests.yml` runs it on Python 3.10 and 3.12 for every PR. CI installs CPU-only pip wheels, so a green run verifies the code, **not** that `environment.yml` still solves.
- There is no lint or formatting config, and no build step beyond `pip install -e .` for the `rampnet` package. Older scripts are configured by editing constants at the top of the file (e.g. `MODEL_CHECKPOINT_PATH`, `CONSIDER_MANUAL`); newer ones (`stage_two/evaluate.py`, most of `scripts/`) take CLI args instead — check the file before assuming.
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
