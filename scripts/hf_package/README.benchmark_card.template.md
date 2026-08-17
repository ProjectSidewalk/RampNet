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

> ### ⚠️ This benchmark is **not** part of the RampNet paper
>
> It did not exist when RampNet was published. The paper's tag,
> [`v1.0-iccv2025`](https://github.com/ProjectSidewalk/RampNet/tree/v1.0-iccv2025) (August 2025),
> contains **no `benchmark/` directory at all** — its evaluation was a **1,000-panorama manually
> labeled gold set** (`manual_labels/`, imagery in
> [`rampnet-dataset`](https://huggingface.co/datasets/projectsidewalk/rampnet-dataset)), drawn from
> the same three training cities.
>
> These {n_cities} city splits were built **eleven months later**, {split_date_range}, as
> post-publication work: to test the published model on cities and imagery sources it was never
> trained on, and to compare it against VLM detectors.
>
> **Use this to evaluate RampNet. Do not cite it as the paper's evaluation** — the ground truth,
> the cities, and the matching protocol all differ, so numbers measured here are not comparable
> with the ones in the paper.

The panoramas behind that benchmark — {n_cities} city splits, {total_gb} GB. Unlike the paper's
gold set, the splits deliberately include **non-US cities and a second imagery source** (Mapillary
as well as Google Street View), which is the whole point: the paper's own evaluation was in-domain.

It is self-contained: the `records` config carries the ground truth, so you can score a model
against this benchmark without cloning anything. The **rubrics** those verdicts were made under,
the per-split reviewer confidence, and the review notes stay in git at
[`benchmark/`](https://github.com/ProjectSidewalk/RampNet/tree/main/benchmark) — read
[`benchmark/RUBRICS.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/benchmark/RUBRICS.md)
before treating a verdict as self-explanatory, and `benchmark/README.md` before quoting a
precision figure, because several splits carry caveats the numbers alone do not show.

## Configs

| config | what it is | when you want it |
| :--- | :--- | :--- |
| **`records`** | **the ground truth** — per-panorama metadata, model detections with their human verdict, and reviewer-marked missed ramps | scoring any model against this benchmark |
| `native` | the panoramas exactly as fetched — 4096 to 16384 px wide, depending on city and imagery source | the resolution experiment; any re-render at higher fidelity |
| `4096x2048` | the same panoramas at the model's input size | **what ground-truth reviewers actually saw** — `gt_gallery.py` renders at 4096×2048 and never native, so this is the config a second rater needs |
| `galleries` | the incremental false-positive crops shown in the operating-point A/B pass — 8 splits, not 9 | redoing that A/B |

### The `records` config

One row per reviewed panorama, joinable to any imagery config on `pano_id`:

| column | meaning |
| :--- | :--- |
| `source` | `gsv` or `mapillary` |
| `capture_date`, `lat`, `lng`, `camera_heading`, `width`, `height` | panorama metadata as fetched |
| **`copyright`** | per-record source attribution, e.g. `© <contributor> / Mapillary (CC BY-SA 4.0)` |
| `detections` | model detections: `x_normalized`, `y_normalized`, `confidence`, and **`verdict`** |
| `missed` | ramps the reviewer marked that the model did not find, each with an `unsure` flag |
| `no_missed` | reviewer confirmed they checked the whole panorama and found nothing missed |
| `model_id`, `model_training_date`, `label_type` | which model produced the detections |

`verdict` is one of **`correct`**, **`incorrect`**, **`unsure`**, **`duplicate`**. `unsure` is an
abstention and `duplicate` marks a second detection of an already-matched ramp — both carry the
meaning the scorer gives them, so the labels mean exactly what the published precision/recall were
computed against. Panoramas that were never reviewed are not included.

**Labels are derived, not original.** `benchmark/<city>/records.jsonl` and `verdicts.json` in git
are the source of truth; this config is regenerated from them by `scripts/export_benchmark.py`.
Verdicts get revised, imagery does not — keeping them in separate configs means a label correction
never rewrites an image blob.

Configs are named by **resolution, not by consumer**. "Model resolution" is a relative label that
becomes wrong the moment the model's input size changes, and a published path cannot be corrected
later without replacing large blobs.

Note that `4096x2048` is not uniformly smaller: for splits whose native imagery is already at or
near model resolution it can be *larger*, because it carries an extra JPEG generation. It is a
fidelity artifact — it reproduces what a reviewer's eyes were on — not a compression trick.

## Usage

```python
from datasets import load_dataset

# the ground truth for one split
gt = load_dataset("{repo_id}", "records", split="gainesville")
print(gt[0]["source"], gt[0]["capture_date"], gt[0]["copyright"])
print(gt[0]["detections"])      # each with x_normalized, y_normalized, confidence, verdict
print(gt[0]["missed"])          # ramps the model did not find

# the matching pixels, at the resolution reviewers saw
px = load_dataset("{repo_id}", "4096x2048", split="gainesville")
by_id = {{r["pano_id"]: r["image"] for r in px}}
image = by_id[gt[0]["pano_id"]]
```

Each row of the **`native`** and **`4096x2048`** configs carries:

| column | meaning |
| :--- | :--- |
| `pano_id` | panorama id — the join key to `records`, and to `benchmark/<city>/records.jsonl` in git |
| `city` | split name |
| `image` | the image, stored as the **exact source bytes**, not re-encoded on write |
| `width`, `height` | pixel dimensions as stored |
| `sha256` | hash of those exact bytes |

**`galleries` rows are crops of a single detection, not panoramas**, so they carry both ids:

| column | meaning |
| :--- | :--- |
| `crop_id` | the crop's own tag, `<pano_id>_<x>_<y>` — unique per detection, **not** a panorama id |
| `pano_id` | the panorama the crop came from, parsed out of `crop_id`; this is the join key |
| `x_normalized`, `y_normalized` | where in that panorama the detection sits, matching the `detections` entry in `records` |
| `city`, `image`, `width`, `height`, `sha256` | as above |

```python
# every A/B crop that came from one panorama
crops = load_dataset("{repo_id}", "galleries", split="paterson")
mine = [c for c in crops if c["pano_id"] == gt[0]["pano_id"]]
```

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

There is no separate publication for this benchmark. Cite the paper for **the pipeline and model
being evaluated**, and please make clear that the evaluation set is post-publication rather than
the paper's own:

```bibtex
@inproceedings{{omeara2025rampnet,
  author    = {{John S. O'Meara and Jared Hwang and Zeyu Wang and Michael Saugstad and Jon E. Froehlich}},
  title     = {{{{RampNet: A Two-Stage Pipeline for Bootstrapping Curb Ramp Detection in Streetscape Images from Open Government Metadata}}}},
  booktitle = {{{{ICCV'25 Workshop on Vision Foundation Models and Generative AI for Accessibility: Challenges and Opportunities (ICCV 2025 Workshop)}}}},
  year      = {{2025}},
  doi       = {{https://doi.org/10.48550/arXiv.2508.09415}},
}}
```
