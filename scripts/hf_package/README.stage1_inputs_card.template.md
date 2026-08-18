---
license: mit
task_categories:
- object-detection
tags:
- curb-ramp
- accessibility
- streetscape
- open-government-data
- reproducibility
---

# RampNet Stage 1 Inputs

The **inputs** to the RampNet Stage 1 pipeline, from **RampNet: A Two-Stage Pipeline for
Bootstrapping Curb Ramp Detection in Streetscape Images from Open Government Metadata**
(O'Meara et al., ICCV'25 CV4A11y workshop, [arXiv:2508.09415](https://arxiv.org/abs/2508.09415)).

[`projectsidewalk/rampnet-dataset`](https://huggingface.co/datasets/projectsidewalk/rampnet-dataset)
is what Stage 1 *produced* — 214k annotated panoramas. **This is what went in.**

`v1.0-iccv2025` shipped the Stage 1 code without these files. That is not a gap a re-download can
close: the city open-data portals serve *current* inventories and they drift, so a file fetched
today is a different experiment rather than a copy of this one.

## Contents

{contents_table}

Every file above is byte-identical to the artifacts recovered from the cluster storage that held
the paper's run. The sha256 values are also recorded in
[`docs/data_provenance.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/data_provenance.md).

**How much of that is enforced, and how much is provenance.** `scripts/export_stage1_inputs.py`
refuses to publish unless the three `location_data/` inventories and `all_locations.csv` hash to
the values pinned in `data_provenance.md` §3 and §3.1. Those four are the ones a present-day
checkout will silently *re-create differently* — `combine_location_data.py` now seeds its shuffle
and changed its unknown-date handling, and the portals have drifted — so an unverified copy would
look identical and not be. The remaining manifests and the raw `street_data/` downloads have no
published hash to check against yet; they rest on the recovery provenance above, and their sha256
is recorded in the table so a future copy can be *proved* identical to this one.

### `manifests/` — the reproduction path, and the most important part here

| file | what it is |
| :--- | :--- |
| `finaldataset.jsonl` | the exact **219,170-panorama** manifest `download_dataset.py` consumed |
| `dataset.jsonl` | the 175,336 positive panoramas, with the ramp coordinates that landed in each |
| `negativepanosSHORTENED.jsonl` | the **43,834 negatives actually used** (exactly 20.0% of the final set) |
| `negativepanos.jsonl` | the full 88,125-candidate negative pool it was drawn from |
| `all_locations.csv` | the three inventories merged to `(latitude, longitude, date)` |

`finaldataset.jsonl` = `dataset.jsonl` + `negativepanosSHORTENED.jsonl`, which is worth stating
because the repo describes that merge as a manual step.

### `location_data/` and `street_data/` — the sources

`location_data/` holds the three government curb ramp inventories. `street_data/` holds the street
centrelines used to sample negative panorama locations; these are the **full downloads**, whereas
the training repo commits an 18.7 MB derivative carrying only the geometry and name field the
pipeline reads (proven to yield an identical sampling network by
`scripts/build_street_derivative.py verify`).

## Three things these files cannot give you

Published beside the data rather than discovered later:

1. **The paper's row order is unreproducible.** `combine_location_data.py` shuffled
   `all_locations.csv` with **no seed** — `random.seed(42)` was added afterwards. The row
   *contents* are intact, which is why provenance can still be recovered by coordinate join
   (`scripts/analysis/gov_provenance.py`), but the ordering is gone.
2. **The negatives cannot be regenerated, only downloaded.** `generate_negative_panos.py` samples
   street locations with an unseeded RNG. `negativepanosSHORTENED.jsonl` is the *only* record of
   which negatives the paper used — that is why it is here.
3. **Date semantics changed after the paper.** The paper-era `convert_date` mapped an unknown
   install date to `"2000-01-01"`, so every undated ramp trivially passed the
   "installed before the panorama was captured" check; the current code returns `""`. Measured,
   **23,088 records (8.36%)** have no install date, which bounds how differently a re-run from
   current `main` would select.

## Which government records became training labels

`scripts/analysis/gov_provenance.py` in the training repo rebuilds the mapping from each
`all_locations.csv` row back to its source file and government ID, and verifies it —
**276,071 / 276,071 rows resolved, 0 unmatched**.

| | Bend | Portland | NYC | total |
| :--- | ---: | ---: | ---: | ---: |
| government records | 13,357 | 45,035 | 217,679 | 276,071 |
| consumed by a generated panorama | 5,110 | 21,075 | 130,527 | **156,712** |
| consumption rate | 38.26% | 46.80% | 59.96% | **56.77%** |

So **43.23% never became a training label**, mostly because no panorama resolved for the location
or the install date failed the predates-capture check.

### Why this says 276,071 and the paper says 276,615

Table 1 of the paper totals **276,615** across these three cities (NYC 217,680, Portland 45,324,
Bend 13,611) — **544 more than the files published here**, as Bend −254, Portland −289, NYC −1.
If you are comparing the paper against this download, that is the difference you will find, and it
is between Table 1 and the files rather than a loss anywhere in the pipeline. Three things are
checkable from this repository:

- **Nothing is dropped on read.** Every Bend and Portland feature is a `Point`, and every NYC row
  parses as `POINT(...)`; all 276,071 records survive parsing.
- **These are the files the paper's run consumed.** Every row of `manifests/all_locations.csv` —
  the run's own output, recovered rather than regenerated — resolves to one of them: 276,071 /
  276,071, none unmatched, with per-city attribution equal to the per-file counts exactly.
- **They are an earlier state of the same inventories, not a re-download.** Both drifting cities
  have *grown* since the paper (measured 2026-07-31: Bend 14,805, Portland 46,101), so a fresh
  download today is **larger** than Table 1 while these are smaller. NYC is effectively frozen and
  differs by a single record.

What is **not** established is where Table 1's exact figures came from. The likeliest reading is
counts read from each city's portal at survey time rather than from the downloaded files, but that
is a hypothesis: the two deltas correspond to roughly three and five months of each city's own
measured growth, which is the right order of magnitude but not one consistent date. We state it as
unresolved rather than quietly reconciled. **Quote 276,071 for anything derived from these files.**

## Usage

This is a set of source documents, not a row-iterable dataset — `load_dataset()` will not work.
Download it directly:

```bash
hf download {repo_id} --repo-type dataset --local-dir rampnet-stage1-inputs
```

Then place the pieces where the pipeline expects them, under
`stage_one/dataset_generation/` in [the training repo](https://github.com/ProjectSidewalk/RampNet):

```
location_data/          -> stage_one/dataset_generation/location_data/     (also already in git)
street_data/            -> stage_one/dataset_generation/street_data/
manifests/*.jsonl,csv   -> stage_one/dataset_generation/
```

You will also need the crop model,
[`projectsidewalk/rampnet-crop-model`](https://huggingface.co/projectsidewalk/rampnet-crop-model) —
Stage 1 does not run without it.

## Provenance

| Field | Value |
| :--- | :--- |
| Pipeline code | https://github.com/ProjectSidewalk/RampNet @ `{git_commit}` |
| Exported | {export_date} by `scripts/export_stage1_inputs.py` |
| Replication ledger | [`docs/replication.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/replication.md) |

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
