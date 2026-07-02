---
license: mit
library_name: transformers
pipeline_tag: keypoint-detection
tags:
  - curb-ramp-detection
  - accessibility
  - street-view
  - keypoint-heatmap
---

# RampNet Curb Ramp Detection Model

Stage-2 model from **RampNet: A Two-Stage Pipeline for Bootstrapping Curb Ramp Detection in
Streetscape Images from Open Government Metadata** (O'Meara et al., ICCV'25 CV4A11y workshop,
[arXiv:2508.09415](https://arxiv.org/abs/2508.09415)).

Takes a 2048x4096 equirectangular street-view panorama (ImageNet-normalized) and predicts a
512x1024 heatmap of curb ramp locations. Extract detections with `skimage.feature.peak_local_max`.

## Provenance

| Field | Value |
| :--- | :--- |
| Training code | https://github.com/ProjectSidewalk/RampNet @ `{git_commit}` |
| Source checkpoint | `{checkpoint_name}` (sha256 prefix `{checkpoint_fingerprint}`) |
| Training dataset | [projectsidewalk/rampnet-dataset](https://huggingface.co/datasets/projectsidewalk/rampnet-dataset) revision `{dataset_revision}` |
| Exported | {export_date} by `scripts/export_hf_model.py` |

## Evaluation (1,000-panorama manually labeled gold set)

{eval_section}

**Important:** these numbers were measured **with horizontal-flip test-time augmentation**
(evaluate the original and mirrored panorama, combine heatmaps with elementwise max). Single-pass
inference will land somewhat below them; derive your own threshold curve without TTA before
choosing an operating point.

## Choosing a detection threshold

- `{recommended_threshold}` is the recommended default operating point (see evaluation above).
- Sweeping thresholds: run `stage_two/evaluate.py` in the training repo, which emits full
  precision/recall-vs-confidence curves as CSV.
- Per-city deployments should calibrate on ~100 locally labeled panoramas; see the repo README's
  "Choosing a Detection Threshold" section.

## Usage

```python
import torch
from transformers import AutoModel
from PIL import Image
import numpy as np
from torchvision import transforms
from skimage.feature import peak_local_max

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained("{repo_id}", trust_remote_code=True).to(DEVICE).eval()

preprocess = transforms.Compose([
    transforms.Resize((2048, 4096), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open("panorama.jpg").convert("RGB")
with torch.no_grad():
    heatmap = model(preprocess(img).unsqueeze(0).to(DEVICE)).squeeze().cpu().numpy()

peaks = peak_local_max(np.clip(heatmap, 0, 1), min_distance=10, threshold_abs={recommended_threshold})
scale_w, scale_h = img.width / heatmap.shape[1], img.height / heatmap.shape[0]
print([(int(c * scale_w), int(r * scale_h)) for r, c in peaks])
```

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
