---
license: mit
tags:
  - curb-ramp-detection
  - accessibility
  - street-view
  - keypoint-heatmap
  - dataset-generation
datasets:
  - projectsidewalk/rampnet-crop-model-dataset
base_model:
  - timm/convnextv2_base.fcmae_ft_in22k_in1k_384
---

# RampNet Stage 1 Crop Model

The **Stage 1** crop model from **RampNet: A Two-Stage Pipeline for Bootstrapping Curb Ramp
Detection in Streetscape Images from Open Government Metadata** (O'Meara et al., ICCV'25 CV4A11y
workshop, [arXiv:2508.09415](https://arxiv.org/abs/2508.09415)).

This is **not** the curb ramp detector — that is
[`projectsidewalk/rampnet-model`](https://huggingface.co/projectsidewalk/rampnet-model). This is the
model that makes the *training data* for it: given a government-published curb ramp GPS coordinate
and the street-view panorama nearest it, it predicts where in that panorama the ramp actually
appears. Every one of the 849,895 keypoint labels in
[`rampnet-dataset`](https://huggingface.co/datasets/projectsidewalk/rampnet-dataset) was placed by
this model.

**Stage 1 cannot be reproduced without it.** `stage_one/dataset_generation/inference_isolator.py`
loads the round-2 checkpoint by a hardcoded relative path; the government inventories and street
data in the training repo are inert without it.

## The two rounds

Training is two-stage, and both checkpoints are published because round 1 is the initialisation for
round 2 — without it, the second round cannot be reproduced either.

| round | file | trained on | role |
| :--- | :--- | :--- | :--- |
| 1 | `round1_ps_best_model.pth` | Project Sidewalk crops | pre-training |
| 2 | `round2_ps_and_manual_best_model.pth` | + manually labeled crops ([`rampnet-crop-model-dataset`](https://huggingface.co/datasets/projectsidewalk/rampnet-crop-model-dataset)) | **the one Stage 1 loads** |

## Provenance

| Field | Value |
| :--- | :--- |
| Training code | https://github.com/ProjectSidewalk/RampNet @ `{git_commit}` |
| Round 1 sha256 | `{round1_sha256}` |
| Round 2 sha256 | `{round2_sha256}` |
| Exported | {export_date} by `scripts/export_crop_model.py` |

These are the **paper-era checkpoints**, recovered from cluster storage — the artifacts that
produced the published dataset, not a retrain.

A note recorded because it is easy to get wrong when reproducing the pipeline: in the original run,
round 1's `best_model.pth` was copied into the round-2 directory renamed to `ps_model.pth`. Those
two files are **byte-identical** (verified by sha256), so `round1_ps_best_model.pth` here serves
both purposes.

## Architecture

A `timm` `convnextv2_base.fcmae_ft_in22k_in1k_384` backbone with a small conv + bilinear-upsample
head producing a single-channel keypoint heatmap — the same `KeypointModel` class as the Stage 2
detector, differing only in `heatmap_size`.

| | crop model (this) | Stage 2 detector |
| :--- | :--- | :--- |
| input | 1024 x 352 | 2048 x 4096 |
| heatmap | 256 x 88 | 512 x 1024 |

## Usage

These are raw PyTorch state dicts, not a `transformers` package — the consumer is the Stage 1
pipeline, which builds the model directly. To reproduce Stage 1, put round 2 where
`inference_isolator.py` expects it:

```bash
hf download {repo_id} round2_ps_and_manual_best_model.pth --local-dir .
mv round2_ps_and_manual_best_model.pth \
   RampNet/stage_one/crop_model/ps_and_manual_model/best_model.pth
```

To load it directly:

```python
import torch
from rampnet.model import KeypointModel, CROP_HEATMAP_SIZE

model = KeypointModel(heatmap_size=CROP_HEATMAP_SIZE)   # (256, 88)
state = torch.load("round2_ps_and_manual_best_model.pth", map_location="cpu")
model.load_state_dict(state.get("model_state_dict", state))
model.eval()
```

## Limitations

- **The round-1 training set is not reproducible.** `stage_one/crop_model/ps_model/data/download_data.py`
  reads live from Project Sidewalk servers with no snapshot pinning, and those databases keep
  growing, so re-running it builds a different crop set than the paper's.
- Trained on Project Sidewalk cities and used on NYC / Portland / Bend panoramas; see the
  contamination registry in
  [`docs/data_provenance.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/data_provenance.md)
  before evaluating any RampNet-derived model in those cities.

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
