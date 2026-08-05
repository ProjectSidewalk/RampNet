---
license: mit
pretty_name: "RampNet Crop-Model Dataset — Round 1 (Project Sidewalk crops)"
size_categories:
- 10K<n<100K
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
| 2 | manually labeled crops | [`rampnet-crop-model-dataset-round2`](https://huggingface.co/datasets/projectsidewalk/rampnet-crop-model-dataset-round2) |

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
| `crop_uid` | the opaque 8-character token the filename starts with — **not a panorama id** |
| `image` | the crop, stored as the **exact source bytes** — 683×2048 JPEG, not re-encoded |
| `keypoints` | list of `{{x, y}}` curb ramp locations |
| `n_keypoints` | how many |
| `width`, `height` | pixel dimensions as stored |
| `sha256` | hash of the exact bytes |

{n_crops} crops carry {n_keypoints} keypoints between them. {keypoint_summary}

### The keypoints were hiding in the filenames

In the original dataset the labels are encoded in the filename and parsed at load time:
`007mz25c_-_118_596_-_478_611.jpg` carries keypoints (118, 596) and (478, 611). Here they are a
real column.

### `crop_uid` is not a panorama id, and the panorama is not recoverable

The leading token looks like an id you could join on. It is not one.
[`download_data.py:277`](https://github.com/ProjectSidewalk/RampNet/blob/main/stage_one/crop_model/ps_model/data/download_data.py)
builds each filename with `random.choices(alphabet, k=8)`, so the token is freshly random per crop
and the Project Sidewalk panorama it was cut from is **not stored anywhere in this artifact**.
Every one of the {n_crops} crops has a distinct token, so nothing collided — but that is a property
of this draw, not a guarantee. If you need crop → panorama provenance, it is not here.

### The x axis carries a 3.1% label/image mismatch

**Coordinates are stored verbatim, in the pixel space of the stored crop (683×2048), and are
deliberately not normalised** — because normalising would force a choice about the following, and
that choice belongs to you.

`train.py` resizes every crop with `transforms.Resize((1024, 352))` and scales **both** keypoint
axes by `0.5`:

| axis | image scale | keypoint scale | agree? |
| :--- | :--- | :--- | :--- |
| y | 2048 → 1024 = **0.5** | 0.5 | yes |
| x | 683 → 352 = **0.5154** | 0.5 | **no — off by 3.1%** |

So y is consistent and x is under-scaled: in the resized 352-px-wide crop a label drifts left of
its ramp in proportion to x, by up to ~10.5 px at the right edge. This is in the original code and
the paper's model was trained through it. If you are reproducing the paper, mirror
[`train.py`](https://github.com/ProjectSidewalk/RampNet/blob/main/stage_one/crop_model/ps_model/model/train.py);
if you are training something new, scale x by 352/683 instead.

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
