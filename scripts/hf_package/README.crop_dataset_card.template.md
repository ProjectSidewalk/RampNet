---
license: mit
task_categories:
- keypoint-detection
tags:
- curb-ramp
- accessibility
- streetscape
- project-sidewalk
configs:
{configs_yaml}
---

# RampNet Crop-Model Dataset — Round 1 (Project Sidewalk crops)

The training data behind **round 1** of the RampNet Stage 1 crop model — {n_crops} crops,
{total_gb} GB — from **RampNet: A Two-Stage Pipeline for Bootstrapping Curb Ramp Detection in
Streetscape Images from Open Government Metadata** (O'Meara et al., ICCV'25 CV4A11y workshop,
[arXiv:2508.09415](https://arxiv.org/abs/2508.09415)).

The crop model is what turns a government curb ramp GPS coordinate into a pixel keypoint on a
panorama; every label in
[`rampnet-dataset`](https://huggingface.co/datasets/projectsidewalk/rampnet-dataset) was placed by
it. It is trained in two rounds:

| round | data | where |
| :--- | :--- | :--- |
| **1** | Project Sidewalk crops | **this repo** |
| 2 | manually labeled crops | [`rampnet-crop-model-dataset`](https://huggingface.co/datasets/projectsidewalk/rampnet-crop-model-dataset) |

The resulting checkpoints are at
[`rampnet-crop-model`](https://huggingface.co/projectsidewalk/rampnet-crop-model).

## Why this needed publishing

**It cannot be regenerated.** `stage_one/crop_model/ps_model/data/download_data.py` fetches from
the live Project Sidewalk servers with no snapshot pinning, and those databases keep growing — so
re-running it today builds a *different* training set, not this one. The same is true of the
train/val/test partition: `splititup.sh` shuffles with `shuf` and **no seed**, so the 70/15/15 split
here is not reproducible either. Both are reasons to ship the artifact rather than instructions.

## Contents

| column | meaning |
| :--- | :--- |
| `crop_id` | the original filename stem |
| `pano_id` | Project Sidewalk panorama id the crop came from |
| `image` | the crop, stored as the **exact source bytes** — 683×2048 JPEG, not re-encoded |
| `keypoints` | list of `{{x, y}}` curb ramp locations |
| `n_keypoints` | how many |
| `width`, `height` | pixel dimensions as stored |
| `sha256` | hash of the exact bytes |

{n_crops} crops carry {n_keypoints} keypoints between them. {keypoint_summary}

### The keypoints were hiding in the filenames

In the original dataset the labels are encoded in the filename and parsed at load time:
`007mz25c_-_118_596_-_478_611.jpg` is panorama `007mz25c` with keypoints (118, 596) and
(478, 611). Here they are a real column.

**Coordinates are stored verbatim, in the pixel space of the stored crop (683×2048), and are
deliberately not normalised.** The training loader multiplies them by exactly `0.5`, while the
image itself is resized 683 → 352 on the x axis (a factor of 0.515). That ~2% discrepancy lives in
the original code; normalising here would silently commit to one reading of it. If you are
reproducing the paper, mirror
[`train.py`](https://github.com/ProjectSidewalk/RampNet/blob/main/stage_one/crop_model/ps_model/model/train.py);
if you are training something new, decide for yourself.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
row = ds[0]
print(row["crop_id"], row["image"].size, row["keypoints"])
```

## Provenance

| Field | Value |
| :--- | :--- |
| Pipeline code | https://github.com/ProjectSidewalk/RampNet @ `{git_commit}` |
| Exported | {export_date} by `scripts/export_crop_dataset.py` |
| Replication ledger | [`docs/replication.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/replication.md) |

Crops derive from Project Sidewalk labels over Google Street View panoramas; the contributing
cities are listed in the training-data contamination registry in
[`docs/data_provenance.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/data_provenance.md),
which you should read before evaluating any RampNet-derived model in those cities.

## Citation

```bibtex
@inproceedings{{omeara2025rampnet,
  author    = {{John S. O'Meara and Jared Hwang and Zeyu Wang and Michael Saugstad and Jon E. Froehlich}},
  title     = {{{{RampNet: A Two-Stage Pipeline for Bootstrapping Curb Ramp Detection in Streetscape Images from Open Government Metadata}}}},
  booktitle = {{{{ICCV'25 Workshop on Vision Foundation Models and Generative AI for Accessibility: Challenges and Opportunities (ICCV 2025 Workshop)}}}},
  year      = {{2025}},
  doi       = {{https://doi.org/10.48550/arXiv.2508.09415}},
}}
```
