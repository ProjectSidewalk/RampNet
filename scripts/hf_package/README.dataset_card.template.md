---
license: {license}
pretty_name: "{pretty_name}"
task_categories:
- object-detection
- keypoint-detection
tags:
- curb-ramp
- accessibility
- streetscape
- street-view
- panorama
- open-government-data
size_categories:
- {size_category}
annotations_creators:
- machine-generated
source_datasets:
- original
---

# Dataset Card for {pretty_name}

**{usable_total} Google Street View panoramas with {card_labels} curb ramp locations, labelled
automatically from open-government curb ramp records in three US cities.** It is the training data
for the RampNet curb ramp detector ([`rampnet-model`](https://huggingface.co/projectsidewalk/rampnet-model)),
from **{paper_title}** (O'Meara et al., ICCV'25 workshop,
[arXiv:2508.09415](https://arxiv.org/abs/2508.09415)).

> **Use the corrected numbers.** The paper's evaluation protocol was corrected after publication
> (the [`v1.1-corrected-eval`](https://github.com/ProjectSidewalk/RampNet/releases/tag/v1.1-corrected-eval)
> erratum). The corrected Stage 1 label agreement on the {gold_panos}-panorama gold set is
> precision {stage1_precision} / recall {stage1_recall}, not the published {stage1_precision_published} /
> {stage1_recall_published}. This dataset's files did not change; only how it was scored.

The data files this card describes are those at Hub revision `{revision}` ({date_modified}), the
revision [`croissant/rampnet-dataset.json`](https://github.com/ProjectSidewalk/RampNet/blob/main/croissant/rampnet-dataset.json)
pins. That file is the machine-readable record (Croissant 1.1, GeoCroissant 1.0 and RAI) and this
card is rendered from it; where they could disagree, the Croissant file is the record of truth.

## Dataset Details

### Dataset Description

Stage 1 of RampNet turns open-government curb ramp inventories (GPS points) into pixel labels on
street-level panoramas, with no human labelling of the training labels. For each government
record it finds nearby Google Street View panoramas, fetches each at 4096x2048, renders a
perspective crop toward the record, and lets a crop-level keypoint model place the ramp in the
crop. The crop heatmaps are projected back onto the panorama and their peaks become the labels.
Each row is one panorama with zero or more curb ramp points.

The dataset exists so that a curb ramp detector can be trained at scale without a large
hand-labelled set of panoramas: {usable_total} panoramas and {card_labels} points, bootstrapped
from a crop model that was itself trained on {crop_pretrain_crops} crowd-labelled and
{crop_manual_crops} hand-labelled crops (see [Source Data](#source-data)).

- **Curated by:** {creators} ({publisher}, Makeability Lab, University of Washington)
- **Funded by:** {funding}
- **Shared by:** {publisher} ([huggingface.co/projectsidewalk](https://huggingface.co/projectsidewalk))
- **Language(s) (NLP):** not applicable. The data are images and coordinates; there is no text
  field.
- **License:** MIT, as declared by this card since publication (`license: {license}`). The
  panoramas are Google Street View imagery. TODO(Jon): decide whether the card should also state
  the terms the imagery itself carries; no committed document records that decision.
- **Version:** {version} (proposed; no tag exists on the Hub yet, see
  [Maintenance and versioning](#maintenance-and-versioning)).
- **Created / first published / last modified on the Hub:** {date_created} / {date_published} / {date_modified}

### Dataset Sources

- **Repository:** {code_url} (Stage 1 code: `stage_one/dataset_generation/`)
- **Paper:** {paper_title}, {paper_venue}, [doi:{paper_doi}](https://doi.org/{paper_doi})
- **Erratum:** [`v1.1-corrected-eval`](https://github.com/ProjectSidewalk/RampNet/releases/tag/v1.1-corrected-eval)
  and the README section [Erratum: Evaluation Metrics (July 2026)](https://github.com/ProjectSidewalk/RampNet/blob/v1.1-corrected-eval/README.md#erratum-evaluation-metrics-july-2026)
- **Demo:** none for the dataset. The trained model's card has a usage example:
  [`projectsidewalk/rampnet-model`](https://huggingface.co/projectsidewalk/rampnet-model).
- **Out-of-distribution benchmark:** [`projectsidewalk/rampnet-benchmark`](https://huggingface.co/datasets/projectsidewalk/rampnet-benchmark)
  (human-verified, other cities and a second imagery provider; not part of the paper)

## Uses

### Direct Use

- Training curb ramp detectors (point or keypoint detection) on equirectangular street-level
  panoramas. This is what the dataset was built for, and what `rampnet-model` was trained on.
- In-distribution evaluation, on the `test` split, for models trained on the `train` split.
- Research on bootstrapping accessibility datasets from open government data: how label quality
  depends on record precision, install dates, and the crop model that places each point.

#### Supported tasks and leaderboards

- **Curb ramp point detection** (`object-detection` / `keypoint-detection`). A prediction is a set
  of (x, y) points per panorama; scoring matches them one-to-one to ground truth within a radius
  of 0.022 of the panorama width (the erratum's rule, implemented in `rampnet/metrics.py`).
- **Leaderboard:** none yet. A leaderboard over `rampnet-benchmark` is planned in
  [#150](https://github.com/ProjectSidewalk/RampNet/issues/150); this dataset's own splits are not
  a held-out benchmark (see [Splits](#splits-and-their-rationale)).

### Out-of-Scope Use

- **Out-of-domain evaluation.** Every panorama comes from New York City, Portland OR or Bend OR,
  the three cities the model was trained on, so a score on any split here is in-distribution. Use
  [`rampnet-benchmark`](https://huggingface.co/datasets/projectsidewalk/rampnet-benchmark) for
  other cities and imagery providers.
- **Treating labels as human ground truth.** The labels are machine-generated (see
  [Annotations](#annotations)). The human gold set is the {gold_panos}-panorama `manual_labels/` set
  in the GitHub repository.
- **Measuring ramp condition or ADA compliance.** A label says a curb ramp is there, not whether it
  is usable, and it carries no extent, slope or condition.
- **Identifying people, vehicles or property** visible in the imagery.

## Dataset Structure

### Data instances

One row per panorama: the image, its id and location, its heading, the machine-generated curb
ramp points, and the government records that produced them. An abridged row:

```python
{{
  "image": <PIL.JpegImageFile 4096x2048>,
  "pano_id": "<Google Street View panorama id>",
  "record_creation_time": 1748000000,             # when the generator wrote the row, not capture
  "curb_ramp_points_normalized": [[0.412, 0.611], [0.587, 0.604]],
  "pano_coord": [40.7, -73.9],
  "curb_ramp_coords": [[40.70001, -73.90002], ...],   # NOT index-aligned with the points
  "pano_azimuth": 123.4,
}}
```

The values above are illustrative, not a real row.

### Data fields

From the `panoramas` record set of the Croissant file:

{fields_table}

To get pixel coordinates, multiply x by the image width and y by its height.

### Splits and their rationale

| split | Hub directory | `datasets` name | Parquet shards | rows on the Hub | usable panoramas |
| :--- | :--- | :--- | ---: | ---: | ---: |
| train | `train/` | `train` | {shards_per_split} | {rows_train} | {usable_train} ({pct_train}) |
| validation | `val/` | `validation` | {shards_per_split} | {rows_val} | {usable_val} ({pct_val}) |
| test | `test/` | `test` | {shards_per_split} | {rows_test} | {usable_test} ({pct_test}) |
| **total** | | | **{n_shards}** | **{rows_total}** | **{usable_total}** |

{n_shards} Parquet shards, {total_gb} GB in all.

**The split is random at the panorama level, not city-held-out.** Panoramas of New York City,
Portland and Bend are shuffled together before splitting, so `test` measures generalisation to
unseen panoramas of seen cities, not to unseen cities. That is the split the paper trained on; its
headline evaluation was the gold set described next. City-held-out evaluation lives in a separate
dataset, `rampnet-benchmark`, whose ground truth is human-reviewed rather than generated.

**The Hub copy is the split of record.** The paper-era split was made without a seed, so re-running
`split_dataset.py` does not reproduce it
([`docs/data_provenance.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/data_provenance.md)
§4). The {gold_panos}-panorama manual gold set was drawn from the same three cities and its
panoramas sit in this dataset's `test` split; it is a label-quality check, not a geographically
held-out one.

**Rows on the Hub vs. usable panoramas.** Each split holds {placeholder_rows_per_split} placeholder
rows (`pano_id` `file_1`, `file_2`, `file_3`) whose 19-byte image blob no decoder opens
([`docs/restage_263.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/restage_263.md)).
Skip them: they are why the Hub's row count ({rows_total}) exceeds the panorama count every RampNet
document quotes ({usable_total}) by {placeholder_total}. A Parquet scan of all shards counts
{scan_labels} labels against the {card_labels} this card has always quoted. That 9-label gap has
the same size as the placeholder-row gap but has not been checked to be the same rows. The
generation log wrote {panos_written_log} panoramas; the gap between that and the Hub is open
([`docs/rampnet1_findings.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/rampnet1_findings.md)
§1).

**Key.** Whether `pano_id` is unique across rows has not been checked, so neither this card nor the
Croissant record declares a key.

### Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="validation", streaming=True)

for row in ds:
    if row["pano_id"].startswith("file_"):        # the placeholder rows; see above
        continue
    w, h = row["image"].size
    points_px = [(x * w, y * h) for x, y in row["curb_ramp_points_normalized"]]
    print(row["pano_id"], points_px)
    break
```

To pin the exact files this card describes, pass `revision="{revision}"`.

## Dataset Creation

### Curation Rationale

Curb ramps are what make a sidewalk crossing usable for people who use wheelchairs, walkers and
strollers. The paper's motivation is that curb ramp detectors for street-level imagery were held
back by scarce training data: hand-labelling ramps on full panoramas is slow, and Project
Sidewalk's crowdsourced labels, the largest source the team had, are tied to small crops. Some
cities, however, publish curb ramp inventories as GPS points. RampNet's premise is that those records, projected onto panoramas, can
bootstrap a large training set automatically, and that a detector trained on it can then find
ramps in cities that have no inventory.

### Source Data

#### Data Collection and Processing

1. **Government records.** Open-government curb ramp inventories for three US cities, filtered to
   records with "Good" location precision: {gov_records_committed} records in the committed files
   (the paper's Table 1 gives {gov_records_table1}; the 544-record difference is reconciled in
   [`docs/data_provenance.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/data_provenance.md)
   §3.3). New York City is {nyc_share_pct} of them.

{city_table}

2. **Panorama selection.** For each record, Street View panoramas within {discovery_radius} were
   discovered, and every record within {inclusion_radius} of a panorama was attached to it. A
   record was used only if its install date predates the panorama's capture month (the paper's
   run also admitted undated records). Negative panoramas were sampled along street centrelines;
   the paper used {negative_share} negatives.
3. **Imagery.** Each panorama was fetched from the Google Street View tile server at zoom level 3
   (32 tiles) and stitched to 4096x2048.
4. **Label placement.** For each attached record, a 90-degree perspective crop toward the record's
   bearing is rendered and the crop model predicts a heatmap; the heatmaps are projected back to
   the panorama and max-combined, and the labels are the peaks
   (`peak_local_max` with {peak_rule}, the threshold scaled to the 0-255 map).
5. **Split** into train/validation/test (see [Splits](#splits-and-their-rationale)).

Generation finished on {generation_finished}; no committed record gives the start date.

**What did not make it.** {unlabelled_records} of the {gov_records_committed} committed records
({unlabelled_pct}) never became a label, mostly because no panorama resolved for them or the
install date failed the check. {panos_not_written} of the {panos_intended} intended panoramas
({panos_not_written_pct}) were never written, almost all because the imagery server did not serve
them.

Code: [`stage_one/dataset_generation/`](https://github.com/ProjectSidewalk/RampNet/tree/main/stage_one/dataset_generation),
run in the order given in the repository's `CLAUDE.md` and README.

#### Who are the source data producers?

- **Curb ramp records:** the open-data programmes of New York City, Portland OR and Bend OR. The
  exact files, with sha256 and retrieval notes, are in
  [`docs/data_provenance.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/data_provenance.md)
  §3.
- **Imagery:** Google Street View, captured by Google and served through its tile endpoints.
  Capture dates are not stored in the rows.
- **The crop model's training data:** round 1, {crop_pretrain_crops} crops with
  {crop_pretrain_labels} labels from Project Sidewalk, crowdsourced and crowd-validated in
  {crop_model_cities} other cities
  ([`rampnet-crop-model-dataset-round1`](https://huggingface.co/datasets/projectsidewalk/rampnet-crop-model-dataset-round1));
  round 2, {crop_manual_crops} crops with {crop_manual_labels} manual labels
  ([`rampnet-crop-model-dataset-round2`](https://huggingface.co/datasets/projectsidewalk/rampnet-crop-model-dataset-round2)).
  Counts as in this card's original summary table.
  No demographic information about Project Sidewalk contributors is collected for this dataset.

### Annotations

#### Annotation process

There is no human annotation of the training labels. Each label is a peak of the crop model's
projected heatmaps, so the labels are as good as the government record (is a ramp there, and
where), the panorama (does it show the corner, and was the ramp built by then) and the crop model
(does it put the point on the ramp).

**Measured label quality.** Against the {gold_panos}-panorama manual gold set ({gold_marks} human
marks in `manual_labels/`), the Stage 1 labels agree at **precision {stage1_precision} / recall
{stage1_recall}** under the erratum's one-to-one matching. The paper's {stage1_precision_published} /
{stage1_recall_published} used the withdrawn matching rule. No per-label uncertainty is stored.

**The bootstrap.** The crop model was trained in two rounds: first on Project Sidewalk crowd labels
from {crop_model_cities} cities, then on the small manual crop set. That second-round model places
every point in this dataset, so its errors are in every label.

#### Who are the annotators?

- **Training labels:** machine-generated by the round-2 crop model
  ([`projectsidewalk/rampnet-crop-model`](https://huggingface.co/projectsidewalk/rampnet-crop-model)).
- **Crop-model labels:** Project Sidewalk crowd workers (round 1) and the research team (round 2).
- **The {gold_panos}-panorama gold set** used to measure label quality: labelled manually by the
  research team. TODO(Jon): name who labelled the manual crop set and the gold set, if the card
  should; no committed document records it.

#### Personal and Sensitive Information

Street-level panoramas can show people, faces, vehicles, licence plates and private buildings. The
images are as served by Google Street View; this release adds no redaction of its own. The dataset adds no personal attributes and labels only curb
ramps. Panorama coordinates are public-street locations.

## Bias, Risks, and Limitations

### Social impact

Built to scale the mapping of curb ramps that wheelchair, walker and stroller users depend on, and
to support Project Sidewalk's sidewalk accessibility audits. Errors cut both ways: a missed ramp
understates a neighbourhood's accessibility, and a false one overstates it, which is worse for
someone planning a route.

### Known biases

- **City mix.** Three US cities, dominated by one: New York City supplies {nyc_share_pct} of the
  government records. No rural area and no non-US city.
- **Imagery provider.** Google Street View only, from Google's own car-mounted rigs, which vary
  by capture year.
- **Capture years.** The rows carry no capture date, so the year mix is unknown. The install-date
  filter admits only ramps built before capture, so recently built ramps are under-represented.
  TODO(Jon): a capture-year histogram would need a metadata lookup for every panorama id; none has
  been run.
- **Selection.** {unlabelled_pct} of government records produced no label, so coverage within a
  city follows where records were precise and panoramas resolved, not where ramps are.
- **Model-anchored labels.** The crop model was trained on Project Sidewalk labels from
  {crop_model_cities} other cities, and every point inherits its errors.

### Limitations

- **Seam duplicates.** {seam_pairs} label pairs ({seam_label_pct} of labels, on {seam_panos}
  panoramas, {seam_pano_pct}) straddle the 360-degree seam at x = 0 / x = 1 and label one ramp
  twice, because the peak finder cannot suppress across the wrap. The 1.0 files are deliberately
  not being changed; the fix is targeted at RampNet 2.0
  ([`docs/seam.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/seam.md) §3). Merge
  points across the seam before training or scoring if that matters to you. The gold set has its
  own {seam_gold_duplicates} adjudicated seam double-marks (same document, §2).
- **Points and records are not paired.** `curb_ramp_points_normalized` and `curb_ramp_coords` are
  not parallel arrays; do not match point *i* to record *i*. The generator max-combines the
  heatmaps of all records before extracting peaks, which destroys that correspondence.
- **Label position is coarse.** The crop model shares the panorama model's head: its 256x88
  heatmap is a bilinear x8 upsample of a 32x11 map, so its peaks sit on an 8-pixel grid. On the
  crop model's own test split, a sub-cell decode cuts the mean detection-to-label distance from
  {crop_decode_argmax_px} to {crop_decode_gaussian_px} heatmap pixels, but Stage 1 does not use
  it, and whether the grid survives the cubic resize and reprojection into these labels was not
  measured ([`docs/crop_decode_221.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/crop_decode_221.md)
  §6).
- **Missing positives.** A ramp with no government record, or whose record failed the install-date
  check, is unlabelled in every panorama that shows it. A low recall measured against these labels
  can be the labels' fault.
- **Placeholder rows and count gaps.** See [Splits](#splits-and-their-rationale).
- **The paper's numbers.** The published evaluation over-counted matches; quote the corrected
  numbers (the erratum above).
- **Contamination.** Any evaluation in New York City, Portland or Bend is optimistically biased,
  and so is one in the {crop_model_cities} Project Sidewalk cities the crop model was trained on
  ([`docs/data_provenance.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/data_provenance.md)
  §1 lists them).

### Recommendations

Skip the `file_*` placeholder rows. Treat the labels as noisy, machine-generated supervision, and
measure a model on human ground truth: the gold set for in-distribution label quality, and
`rampnet-benchmark` for other cities and imagery providers. Score with the erratum's one-to-one
rule (`python -m rampnet.eval` in the GitHub repository). Merge seam-straddling pairs if seam
behaviour matters to your use. Do not report a score on this dataset as out-of-domain performance.

## Citation

Cite the paper for the method and the dataset. Please also cite the erratum when you quote any
evaluation number.

**BibTeX** (paper):

```bibtex
{cite_as}
```

The `doi` field is the arXiv preprint's DOI; `note = {{DOI: forthcoming}}` refers to the
workshop proceedings DOI, which has not been issued yet.

**BibTeX** (this dataset; the DOI is not yet minted):

```bibtex
@misc{{projectsidewalk2025rampnetdataset,
  author       = {{{creators_bibtex}}},
  title        = {{{{rampnet-dataset: the RampNet Stage 1 curb ramp dataset}}}},
  year         = {{2025}},
  publisher    = {{Hugging Face}},
  howpublished = {{\url{{{repo_url}}}}},
  note         = {{Version {version}, revision {revision}. DOI: forthcoming}}
}}
```

**APA:**

{apa_authors} (2025). {paper_title}. In *{paper_venue}*. https://doi.org/{paper_doi}

## Glossary

- **Curb ramp:** the sloped transition between a sidewalk and the street at a crossing.
- **Equirectangular panorama:** a 360 x 180 degree image; x wraps, so x = 0 and x = 1 are the same
  column (the **seam**).
- **Normalized point:** (x, y) in [0, 1] as a fraction of image width and height. x = 0.5 is the
  panorama's heading (`pano_azimuth`).
- **Stage 1 / Stage 2:** RampNet's dataset generation (this dataset) and detector training
  (`rampnet-model`).
- **Crop model:** the keypoint model that places a ramp inside a perspective crop rendered toward a
  government record; it places every label here.
- **Gold set:** the {gold_panos} panoramas labelled by hand (`manual_labels/`), the human ground
  truth the paper evaluated against.
- **Negative panorama:** a panorama sampled along a street with no government record attached; its
  label list is empty.

## More Information

### Licensing information

MIT (see [Dataset Description](#dataset-description) for the open question about imagery terms).

### Maintenance and versioning

- **Static release.** The {version} files at revision `{revision}` will not be changed. Known
  defects (the seam duplicates, the placeholder rows) are documented here instead, and fixes go
  into a RampNet 2.0 dataset.
- **Versioning policy** (proposed in [#150](https://github.com/ProjectSidewalk/RampNet/issues/150)):
  **a major version on any change to a data file; a minor version on a metadata-only change** (this
  card, the Croissant record). The version label is settled together with the DOI.
- **Identifier.** A DataCite DOI will be minted per Hub revision; until then the Croissant
  `identifier` reads `{identifier}`.
- **Machine-readable metadata.** [`croissant/rampnet-dataset.json`](https://github.com/ProjectSidewalk/RampNet/blob/main/croissant/rampnet-dataset.json),
  validated by `scripts/validate_croissant.py`, with every Parquet shard's size and sha256.
- **This card** is rendered from that file by `scripts/export_dataset_cards.py`; a change to either
  shows up in its `--check`.

### Contributions

Report a problem or propose a fix at {contact}. The provenance of every number on this card is in
[`docs/dataset_cards_150.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/dataset_cards_150.md).

### Related datasets and models

| repository | what it is |
| :--- | :--- |
| [`rampnet-model`](https://huggingface.co/projectsidewalk/rampnet-model) | the Stage 2 detector trained on this dataset |
| [`rampnet-benchmark`](https://huggingface.co/datasets/projectsidewalk/rampnet-benchmark) | human-verified, post-publication benchmark in other cities and on Mapillary |
| [`rampnet-crop-model`](https://huggingface.co/projectsidewalk/rampnet-crop-model) | the crop model that placed every label here |
| [`rampnet-crop-model-dataset-round1`](https://huggingface.co/datasets/projectsidewalk/rampnet-crop-model-dataset-round1) | the crop model's round-1 training data (Project Sidewalk crops) |
| [`rampnet-crop-model-dataset-round2`](https://huggingface.co/datasets/projectsidewalk/rampnet-crop-model-dataset-round2) | the crop model's round-2 training data (manually labelled crops) |
| [`rampnet-stage1-inputs`](https://huggingface.co/datasets/projectsidewalk/rampnet-stage1-inputs) | Stage 1's intermediate manifests (`all_locations.csv`, `dataset.jsonl`, `finaldataset.jsonl`, the negative-panorama lists) |

## Dataset Card Authors

{card_authors}

## Dataset Card Contact

{contact}
