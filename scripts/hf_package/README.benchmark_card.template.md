---
license: mit
task_categories:
- object-detection
tags:
- curb-ramp
- accessibility
- streetscape
- benchmark
- evaluation
configs:
{configs_yaml}
---

# RampNet Benchmark Imagery

The panoramas behind the RampNet detection benchmark — {n_cities} city splits, {total_gb} GB — from
**RampNet: A Two-Stage Pipeline for Bootstrapping Curb Ramp Detection in Streetscape Images from
Open Government Metadata** (O'Meara et al., ICCV'25 CV4A11y workshop,
[arXiv:2508.09415](https://arxiv.org/abs/2508.09415)).

This is the **imagery only**. The labels — per-split detections, human ground-truth verdicts, and
the rubrics they were made under — live in git at
[`benchmark/`](https://github.com/ProjectSidewalk/RampNet/tree/main/benchmark). That split is
deliberate: verdicts get revised, imagery does not, so this repo only ever grows.

## Configs

| config | what it is | when you want it |
| :--- | :--- | :--- |
| `native` | the panoramas exactly as fetched — 4096 to 16384 px wide, depending on city and imagery source | the resolution experiment; any re-render at higher fidelity |
| `4096x2048` | the same panoramas at the model's input size | **what ground-truth reviewers actually saw** — `gt_gallery.py` renders at 4096×2048 and never native, so this is the config a second rater needs |
| `galleries` | the incremental false-positive crops shown in the operating-point A/B pass | redoing that A/B |

Configs are named by **resolution, not by consumer**. "Model resolution" is a relative label that
becomes wrong the moment the model's input size changes, and a published path cannot be corrected
later without replacing large blobs.

Note that `4096x2048` is not uniformly smaller: for splits whose native imagery is already at or
near model resolution it can be *larger*, because it carries an extra JPEG generation. It is a
fidelity artifact — it reproduces what a reviewer's eyes were on — not a compression trick.

## Usage

```python
from datasets import load_dataset

# every city at review resolution
ds = load_dataset("{repo_id}", "4096x2048")

# one split, native pixels
gainesville = load_dataset("{repo_id}", "native", split="gainesville")
print(gainesville[0]["pano_id"], gainesville[0]["image"].size)
```

Each row carries:

| column | meaning |
| :--- | :--- |
| `pano_id` | panorama id — the join key to `benchmark/<city>/records.jsonl` in git |
| `city` | split name |
| `image` | the image, stored as the **exact source bytes**, not re-encoded on write |
| `width`, `height` | pixel dimensions as stored |
| `sha256` | hash of those exact bytes |

## Verifying you have the pixels the reviewers judged

Same pano id and filename is **not** evidence the bytes are the ones a reviewer saw — a re-fetch
from Google Street View or Mapillary can return re-stitched or re-compressed imagery. Two
independent checks exist:

1. **Per row.** Every row carries the `sha256` of its own embedded bytes.
   `scripts/export_benchmark.py verify` re-hashes every image straight out of the Parquet, so the
   round trip through Parquet is checked rather than assumed.
2. **Against the review.** `benchmark/<city>/imagery_manifest.json` in git records a sha256 and
   pixel size per panorama, pinned at the time each split was reviewed — 206 KB describing the
   whole archive. That is what ties these bytes to the verdicts.

```bash
python scripts/analysis/imagery_manifest.py --verify
```

## Provenance

| Field | Value |
| :--- | :--- |
| Benchmark code and labels | https://github.com/ProjectSidewalk/RampNet @ `{git_commit}` |
| Exported | {export_date} by `scripts/export_benchmark.py` |
| Replication ledger | [`docs/replication.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/replication.md) |
| How a split is built and reviewed | [`docs/adding_a_benchmark_city.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/adding_a_benchmark_city.md) |

Imagery sources differ by city (Google Street View and Mapillary); per-split provenance, ground
truth precision/recall, and reviewer confidence are documented in
[`benchmark/README.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/benchmark/README.md).

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
