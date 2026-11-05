---
license: mit
pretty_name: "RampNet Benchmark Imagery"
task_categories:
- object-detection
tags:
- curb-ramp
- accessibility
- streetscape
- benchmark
- evaluation
- out-of-distribution
size_categories:
- {size_category}
language:
- en
annotations_creators:
- expert-generated
source_datasets:
- original
configs:
{configs_yaml}
---

# Dataset Card for RampNet Benchmark Imagery

> ### ⚠️ This benchmark is **not** part of the RampNet paper
>
> It did not exist when RampNet was published. The paper's tag,
> [`v1.0-iccv2025`](https://github.com/ProjectSidewalk/RampNet/tree/v1.0-iccv2025) (August 2025),
> contains **no `benchmark/` directory at all** — its evaluation was a **1,000-panorama manually
> labeled gold set** (`manual_labels/`, imagery in
> [`rampnet-dataset`](https://huggingface.co/datasets/projectsidewalk/rampnet-dataset)), drawn from
> the same three training cities.
>
> These {n_cities} splits were built **after publication**, {split_date_range}, as
> post-publication work: to test the published model mostly on cities and imagery sources it was
> never trained on, and to compare it against VLM detectors. The exception is `bend`, one of the
> paper's training cities; its panoramas that are also in the training data are flagged (see
> "Training overlap").
>
> **Use this to evaluate RampNet. Do not cite it as the paper's evaluation** — the ground truth,
> the cities, and the matching protocol all differ, so numbers measured here are not comparable
> with the ones in the paper.

The panoramas behind that benchmark — {n_cities} splits of ground truth, imagery for {n_imagery_splits}
of them, {total_gb} GB. Unlike the paper's
gold set, the splits deliberately include **non-US cities and a second imagery source** (Mapillary
as well as Google Street View), which is the whole point: the paper's own evaluation was in-domain.

It is self-contained: the `records` config carries the ground truth, so you can score a model
against this benchmark without cloning anything. It also carries, on every row, the reviewer's
self-rated confidence and caveats for that split, any note the reviewer left on the panorama, and
whether the panorama is also in RampNet's training data — see the two sections below. The
**rubrics** the verdicts were made under stay in git at
[`benchmark/`](https://github.com/ProjectSidewalk/RampNet/tree/main/benchmark) — read
[`benchmark/RUBRICS.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/benchmark/RUBRICS.md)
before treating a verdict as self-explanatory, and `benchmark/README.md` before quoting a
precision figure, because several splits carry caveats the numbers alone do not show.

## Dataset Details

### Dataset Description

A curb ramp detection benchmark in which every detection of the published RampNet model on a
sample of panoramas was judged by a human reviewer, and every ramp the model missed was marked, so
that precision, recall and AP can be measured for any point detector. Each split is one city (or
district) on one imagery provider. Curb ramps are labelled as points on equirectangular
panoramas.

- **Curated by:** Project Sidewalk (Makeability Lab, University of Washington). The reviewer of a
  split, where recorded, is in that split's `reviewer` column. TODO(Jon): which people to list as
  the benchmark's creators (decided with the DOI record; `docs/fair_metadata_150.md` §3).
- **Funded by:** TODO(Jon): the funding acknowledgement, if the card should carry one.
- **Shared by:** Project Sidewalk ([huggingface.co/projectsidewalk](https://huggingface.co/projectsidewalk))
- **Language(s) (NLP):** English, in the reviewer's free-text fields (`note`, `review_summary`,
  `review_caveats`). Everything else is images, coordinates and fixed labels.
- **License:** MIT for the benchmark (`license: mit`). Each record's imagery attribution is in its
  `copyright` field; Mapillary records name the contributor and the CC BY-SA 4.0 licence.
  TODO(Jon): decide whether the card should also state the terms Google Street View imagery
  carries; no committed document records that decision.
- **Version:** 1.0.0 (proposed; no tag exists on the Hub yet, see "Maintenance and versioning").

### Dataset Sources

- **Repository:** https://github.com/ProjectSidewalk/RampNet, ground truth under
  [`benchmark/`](https://github.com/ProjectSidewalk/RampNet/tree/main/benchmark)
- **Paper (the model being evaluated, not this benchmark):** RampNet: A Two-Stage Pipeline for
  Bootstrapping Curb Ramp Detection in Streetscape Images from Open Government Metadata,
  ICCV'25 workshop, [doi:10.48550/arXiv.2508.09415](https://doi.org/10.48550/arXiv.2508.09415)
- **Erratum:** [`v1.1-corrected-eval`](https://github.com/ProjectSidewalk/RampNet/releases/tag/v1.1-corrected-eval),
  which defines the one-to-one matching rule this benchmark is scored with
- **Machine-readable metadata:** [`croissant/rampnet-benchmark.json`](https://github.com/ProjectSidewalk/RampNet/blob/main/croissant/rampnet-benchmark.json)
  (Croissant 1.1, GeoCroissant 1.0 and RAI, with every file's size and sha256)
- **Demo:** none.

## Uses

### Direct Use

- **Out-of-domain evaluation** of curb ramp detectors: precision, recall and AP per city and per
  imagery provider, on cities RampNet was not trained on (Bend excepted).
- **Comparing detectors** under one protocol. In a clone of the GitHub repository,
  `python -m rampnet.eval score --bundle benchmark/<split> --predictions <file>` scores a
  prediction file with the same matcher and recall gating behind every committed benchmark number
  ([`docs/eval_protocol_150.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/eval_protocol_150.md)).
- **Resolution and imagery-fidelity experiments**, with the `native` and `4096x2048` configs.

#### Supported tasks and leaderboards

- **Curb ramp point detection** (`object-detection`). Predictions are (x, y) points, normalized to
  the panorama, with an optional confidence. They match ground truth one-to-one within 0.022 of the
  panorama width (in a 1024 x 512 space), with x wrapping across the seam; `unsure` points are
  ignore regions; recall counts only panoramas whose reviewer confirmed a complete scan.
- **Leaderboard:** planned in [#150](https://github.com/ProjectSidewalk/RampNet/issues/150), not
  yet live. The committed scoreboard of every model run so far is
  [`docs/model_scoreboard.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/model_scoreboard.md).

### Out-of-Scope Use

- **Training.** The splits are small, and training on them spends the only held-out cities.
- **Quoting a number here as the RampNet paper's evaluation** (see the warning above).
- **Pooling `budapest_district5` with the US splits** without reading its caveats: it was reviewed
  at low self-rated confidence with a US-derived rubric.
- **Measuring ramp condition or ADA compliance**, or identifying people or property in the imagery.

## Dataset Structure

### Splits and their rationale

Every split is a test split. As of this export (reviewed panoramas only):

{split_table}

**City-held-out by design.** Each split is a city, or a district of one, that RampNet's Stage 1
data does not contain, so a score on it measures transfer to a new place; Bend is the one
exception, kept as an in-distribution reference. Imagery provider and camera are fixed within a
split and differ between splits, so a split measures a place and a capture rig together.

**Sampling within a split.** Each split samples a run of the auto-labeler over the city in three
strata, recorded per row in `review_group`: `top` (the 5 panoramas with the most detections,
always included), `random`, and `empty` (panoramas with no detection). The **unbiased subset** is
`random` + `empty`; quote it when you want a number that generalises to the city. The usual design
is 5 + 95 + 25 = 125; Bend has 10 empty panoramas and Richmond 94 random ones, and splits added
later record their own sizes in git.

### Configs

| config | what it is | when you want it |
| :--- | :--- | :--- |
| **`records`** | **the ground truth** — per-panorama metadata, model detections with their human verdict, and reviewer-marked missed ramps | scoring any model against this benchmark |
| `native` | the panoramas exactly as fetched — 4096 to 16384 px wide, depending on city and imagery source | the resolution experiment; any re-render at higher fidelity |
| `4096x2048` | the same panoramas at the model's input size | **what ground-truth reviewers actually saw** — `gt_gallery.py` renders at 4096×2048 and never native, so this is the config a second rater needs |
| `galleries` | the incremental false-positive crops shown in the operating-point A/B pass — 314 crops over 9 splits | redoing that A/B |
{records_only_note}
### The `records` config

One row per reviewed panorama, joinable to any imagery config on `pano_id`:

| column | meaning |
| :--- | :--- |
| `source` | `launch` (Google Street View) or `mapillary`, as the fetcher recorded it |
| `capture_date`, `lat`, `lng`, `camera_heading`, `width`, `height` | panorama metadata as fetched |
| **`copyright`** | per-record source attribution, e.g. `© <contributor> / Mapillary (CC BY-SA 4.0)` |
| `detections` | model detections: `x_normalized`, `y_normalized`, `confidence`, and **`verdict`** |
| `missed` | ramps the reviewer marked that the model did not find, each with an `unsure` flag |
| `no_missed` | reviewer confirmed they checked the whole panorama and found nothing missed |
| `model_id`, `model_training_date`, `label_type` | which model produced the detections |
| `review_group` | the sampling stratum: `top` (densest panoramas, always included), `random`, or `empty` |
| **`train_overlap`** | `true` if this panorama is also in `rampnet-dataset`'s train or validation split (see "Training overlap"); never null |
| `note` | the reviewer's note on this one panorama, if they left one; otherwise null |
| `reviewer`, `reviewed_at` | who reviewed the split, and when |
| **`review_confidence`** | the reviewer's own rating of the split's review, copied verbatim (`high`, `low`, or a note that it was not recorded) |
| `review_summary` | the reviewer's summary of the split |
| **`review_caveats`** | the reviewer's caveats for the split, one string each |

The last five columns come from the split's review notes and repeat on every row of that split.
They are **null when no review notes were recorded for the split** — which means "not recorded",
not "no caveats". A split whose reviewer recorded notes but no caveats has `review_caveats == []`.

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

### Reviewer caveats travel with the rows

Each split's reviewer notes are copied onto its rows, so the caveat arrives with the number. As of
this export:

{confidence_table}

```python
from datasets import load_dataset

bp = load_dataset("{repo_id}", "records", split="budapest_district5")
print(bp[0]["review_confidence"])        # "{budapest_confidence}"
for caveat in bp[0]["review_caveats"]:
    print("-", caveat)
```

### Training overlap

Reviewed panoramas that are also in `rampnet-dataset`'s train/validation splits, by split (Bend
is one of the paper's three training cities):

{overlap_listing}

They are kept and flagged (`train_overlap == true`), not dropped, so the published numbers are
unchanged. The exact-id check was first run on 2026-07-22 and re-run on {overlap_checked_at} for
every split (`scripts/analysis/train_overlap_check.py`, result in
[`benchmark/train_overlap.json`](https://github.com/ProjectSidewalk/RampNet/blob/main/benchmark/train_overlap.json)).

Dropping Bend's flagged panoramas moves its numbers inside their Wilson intervals, in both
directions:

| Bend subset | panoramas | precision | recall |
| :--- | ---: | ---: | ---: |
| all reviewed | 110 | 0.954 | 0.758 |
| all reviewed, overlap dropped | 106 | 0.956 | 0.753 |
| unbiased (random + empty) | 105 | 0.972 | 0.738 |
| unbiased, overlap dropped | 101 | 0.976 | 0.731 |

```python
bend = load_dataset("{repo_id}", "records", split="bend")
unseen = bend.filter(lambda r: not r["train_overlap"])     # {bend_unseen_rows} rows
```

### Usage

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

### Verifying you have the pixels the reviewers judged

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

## Dataset Creation

### Curation Rationale

RampNet's paper evaluated on a gold set drawn from its own three training cities, all imaged by
Google Street View, so it could not say how the model does anywhere else. This benchmark measures
that: new cities, including two outside the US, and a second imagery provider (Mapillary) with
its own cameras. It also gives other detectors, including vision-language models, a common
ground truth to be compared on.

### Source Data

#### Data Collection and Processing

For each city, ProjectSidewalk/sidewalk-auto-labeler enumerated the panoramas in an area, ran the
published RampNet model on each, and exported a native-resolution bundle. A stratified sample of
those panoramas (see "Sampling within a split") was reviewed, and only reviewed panoramas are
published. The imagery is stored as the exact bytes fetched (`native`) and at the model's input
size (`4096x2048`); each panorama's sha256 was pinned at review time in
`benchmark/<city>/imagery_manifest.json`. How a split is built and reviewed, step by step:
[`docs/adding_a_benchmark_city.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/adding_a_benchmark_city.md).

#### Who are the source data producers?

- **Google Street View** panoramas (`source` = `launch`), captured by Google.
- **Mapillary** panoramas (`source` = `mapillary`), captured by Mapillary contributors on a range
  of 360-degree cameras; each record's `copyright` names the contributor. The camera per panorama
  is in `benchmark/<city>/records.jsonl` (`camera_model`) and is summarised per split in the
  Croissant record.

No demographic information about the imagery contributors is collected.

### Annotations

#### Annotation process

One reviewer per split, in RampNet's ground-truth gallery
([`scripts/gt_gallery.py`](https://github.com/ProjectSidewalk/RampNet/blob/main/scripts/gt_gallery.py),
a static HTML page) at 4096x2048, under the rubric in
[`benchmark/RUBRICS.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/benchmark/RUBRICS.md)
§1:

- every model detection is judged `correct`, `incorrect`, `unsure` or `duplicate`;
- the whole panorama is scanned for ramps the model missed, each marked, optionally as `unsure`;
- `no_missed` records the attestation that the scan was complete. Recall counts only panoramas
  whose scan was confirmed this way, or that have at least one missed mark.

**The ground truth is model-anchored.** Reviewers judge RampNet's own detections and then add what
it missed, so the ground truth is not independent of the model being evaluated, and a detector
that finds ramps in places RampNet never looked has only the reviewer's missed-ramp scan to be
credited against. The paper's 1,000-panorama gold set in `rampnet-dataset` is the un-anchored one.

**No second rater yet.** Every split has had one review pass; inter-rater agreement has not been
measured (a Budapest second rater is [#74](https://github.com/ProjectSidewalk/RampNet/issues/74)).
Each split's reviewer notes, confidence and caveats are on its rows (see above).

#### Who are the annotators?

Members of the Project Sidewalk research team. Where a split recorded review notes, the reviewer
is in its `reviewer` column; splits without review notes did not record a reviewer. TODO(Jon):
confirm and record the reviewer of the splits whose `reviewer` is null.

#### Personal and Sensitive Information

Street-level panoramas can show people, faces, vehicles, licence plates and private buildings. The
images are stored as the exact source bytes, with no redaction added by this release. The
`copyright` field of Mapillary records names the contributing user account, as the imagery
licence requires.

## Bias, Risks, and Limitations

- **Not the paper's evaluation.** Different ground truth, cities and matching protocol.
- **Model-anchored ground truth** (above): a fair comparison between RampNet and a different
  detector is harder than one between two versions of RampNet.
- **Provider, camera and city are confounded.** Each split has one provider and usually one camera
  family, so a difference between two splits cannot be attributed to the place alone.
- **Capture years differ by split** (the table above), and within a split they can span more than
  a decade of Street View captures, so a split mixes old and new streetscapes and cameras.
- **Small splits.** About 110 to 125 panoramas each, so per-split intervals are wide (about ±5
  points on precision and recall at Bend's counts). Pool, or quote the interval.
- **Reviewer confidence varies.** Budapest was reviewed at low self-rated confidence with a
  US-derived rubric; see "Reviewer caveats travel with the rows".
- **Training overlap.** Bend is a training city, and four of its panoramas are in the training
  data (flagged, see "Training overlap").
- **Points only.** No ramp extents, slope or condition.

### Recommendations

Quote the unbiased subset (`random` + `empty`) with a Wilson interval, per split. Read
`review_caveats` before quoting a split, and do not pool `budapest_district5` with the US splits.
Drop `train_overlap` rows, or report Bend separately, when you want strictly unseen panoramas.
Score with the erratum's one-to-one rule (`python -m rampnet.eval`), never with the paper's
original matcher, and say in any write-up that the benchmark is post-publication.

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
  doi       = {{10.48550/arXiv.2508.09415}},
  url       = {{https://cv4a11y.github.io/ICCV2025/index.html}},
}}
```

To cite the benchmark itself (the DOI is not yet minted):

```bibtex
@misc{{projectsidewalk2026rampnetbenchmark,
  author       = {{{{Project Sidewalk}}}},
  title        = {{{{rampnet-benchmark: a post-publication, human-verified curb ramp detection benchmark}}}},
  year         = {{2026}},
  publisher    = {{Hugging Face}},
  howpublished = {{\url{{https://huggingface.co/datasets/{repo_id}}}}},
  note         = {{DOI: forthcoming}}
}}
```

**APA:**

Project Sidewalk. (2026). *rampnet-benchmark: A post-publication, human-verified curb ramp
detection benchmark* [Data set]. Hugging Face. https://huggingface.co/datasets/{repo_id}

When you quote a number, cite the
[`v1.1-corrected-eval`](https://github.com/ProjectSidewalk/RampNet/releases/tag/v1.1-corrected-eval)
erratum for the matching rule.

## Glossary

- **Verdict:** a reviewer's judgment of one model detection: `correct`, `incorrect`, `unsure`
  (cannot tell from this imagery; an ignore point for scoring) or `duplicate` (a second detection
  of a ramp already counted; a false positive).
- **Missed mark:** a ramp the reviewer added where the model found nothing; a ground-truth point
  for recall, or an ignore point if marked `unsure`.
- **Recall-confirmed panorama:** one whose reviewer attested a complete scan (`no_missed`) or
  marked at least one miss; only these count toward recall.
- **Stratum (`review_group`):** `top`, `random` or `empty`; the **unbiased subset** is `random` +
  `empty`.
- **Model-anchored:** ground truth built by judging one model's detections, as here.
- **`launch`:** the imagery fetcher's name for Google Street View.

## More Information

### Provenance

| Field | Value |
| :--- | :--- |
| Benchmark code and labels | https://github.com/ProjectSidewalk/RampNet @ `{git_commit}` |
| Exported | {export_date} by `scripts/export_benchmark.py` |
| Replication ledger | [`docs/replication.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/replication.md) |
| How a split is built and reviewed | [`docs/adding_a_benchmark_city.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/adding_a_benchmark_city.md) |

Imagery sources differ by city (Google Street View and Mapillary); per-split provenance, ground
truth precision/recall, and reviewer confidence are documented in
[`benchmark/README.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/benchmark/README.md).

### Licensing information

MIT for the benchmark; per-record imagery attribution in `copyright` (see "Dataset Description"
for the open question about imagery terms).

### Maintenance and versioning

- **Static release per revision.** Verdicts are revised in git and re-exported to the `records`
  config without touching the image files.
- **Versioning policy** (proposed in [#150](https://github.com/ProjectSidewalk/RampNet/issues/150)):
  **a major version on any change to a data file** (a revised verdict included, since it changes
  a `records` file); **a minor version on a metadata-only change** (this card, the Croissant
  record). The version label is settled together with the DOI.
- **Identifier.** A DataCite DOI will be minted per Hub revision; the Croissant record carries a
  placeholder until then.
- **New splits** are added only with their training-overlap check committed
  (`benchmark/train_overlap.json`); the exporter refuses a split without one.

### Contributions

Report a problem with a verdict, or propose a new city, at
https://github.com/ProjectSidewalk/RampNet/issues. A second rater for any split is especially
welcome: the per-rater files and the rubric are in git, and the gallery renders the same view the
first reviewer saw.

## Dataset Card Authors

Project Sidewalk, Makeability Lab, University of Washington. TODO(Jon): name the individual card
authors, if any should be listed.

## Dataset Card Contact

https://github.com/ProjectSidewalk/RampNet/issues
