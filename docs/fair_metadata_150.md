# Croissant 1.1 + GeoCroissant 1.0 + RAI metadata for the two RampNet datasets

**Status: written 2026-09-25 for the third checkbox of
[#150](https://github.com/ProjectSidewalk/RampNet/issues/150); revised the same day after the
review on [#190](https://github.com/ProjectSidewalk/RampNet/pull/190).** Two files, written by hand
and validated with `mlcroissant`:

| file | describes | Hub revision it pins |
| :--- | :--- | :--- |
| [`croissant/rampnet-dataset.json`](../croissant/rampnet-dataset.json) | [`projectsidewalk/rampnet-dataset`](https://huggingface.co/datasets/projectsidewalk/rampnet-dataset), the Stage 1 training data | `ee882e3f3c779dc13182f307bca616e50d9b8c5c` (2025-08-15) |
| [`croissant/rampnet-benchmark.json`](../croissant/rampnet-benchmark.json) | [`projectsidewalk/rampnet-benchmark`](https://huggingface.co/datasets/projectsidewalk/rampnet-benchmark), the post-publication city benchmark | `63d5ffd0e4b6795702db40be89ea4d9672d91ab5` (2026-08-17) |

**Nothing has been uploaded to Hugging Face and no DOI has been minted.** These files are in the
repo only. Uploading them to the Hub records and minting the DOI are separate steps (§6).

## 1. What the files contain

Both files declare `conformsTo` Croissant 1.1, Croissant RAI 1.0 and GeoCroissant 1.0.

**`@context`.** It is Appendix 1 of the Croissant 1.1 spec term for term, with two differences,
both checked by `scripts/validate_croissant.py` against its `CONTEXT` constant:

- **schema.org is `https://schema.org/`.** Appendix 1 writes `http://schema.org/`, but
  `mlcroissant` 1.1.0 rejects that form ("The current JSON-LD doesn't extend
  https://schema.org/Dataset"), and the spec's own later examples use https.
- **One extra term, `geocr`** (`http://mlcommons.org/croissant/geo/`), which GeoCroissant 1.0
  requires.

`mlcroissant`'s built-in context also has the Croissant 1.0 terms `path`, `repeated`, `replace` and
`samplingRate`. The files use none of them, so they are left out, and `mlcroissant` logs "The JSON-LD
`@context` is not standard" naming those four. That warning is expected and is not an error.

**Other properties.** Both files carry `license`, `citeAs`, `citation`, `version`,
`isLiveDataset: false`, `datePublished`, `dateModified`, creator and publisher, and `isBasedOn`
with a `sc:SoftwareSourceCode` node whose `codeRepository` is the GitHub repo. The Croissant spec
has no property for a code repository, and schema.org's `codeRepository` belongs to
`SoftwareSourceCode`, not `Dataset`; the first version of these files used `sameAs`, which says the
dataset *is* that page, and that is not true.

RAI properties: `rai:dataCollection`, `rai:dataCollectionType`, `rai:dataCollectionTimeframe`,
`rai:dataCollectionRawData`, `rai:dataCollectionMissingData`, `rai:dataAnnotationProtocol`,
`rai:annotationsPerItem`, `rai:dataBiases`, `rai:dataLimitations`, `rai:dataUseCases`,
`rai:dataSocialImpact`, `rai:personalSensitiveInformation`, `rai:dataReleaseMaintenancePlan`, plus
`rai:machineAnnotationTools` and `rai:dataAnnotationAnalysis` for the dataset and
`rai:dataAnnotationPlatform` for the benchmark.

**Citation.**

- **rampnet-dataset.** The dataset card asks readers to cite the paper, so `citeAs` is the card's
  BibTeX with two changes. `doi` is the bare DOI `10.48550/arXiv.2508.09415`, because a BibTeX `doi`
  field takes a DOI, not a URL. `note = {DOI: forthcoming}` is copied verbatim from the card.
  `citation` lists the paper and the
  [`v1.1-corrected-eval`](https://github.com/ProjectSidewalk/RampNet/releases/tag/v1.1-corrected-eval)
  erratum (the release, and the README section at that tag), as #150 asks.
- **rampnet-benchmark.** Its card says there is no separate publication for the benchmark and that
  it is not the paper's evaluation. So `citeAs` cites the dataset itself: a `@misc` entry for
  `projectsidewalk/rampnet-benchmark` at the pinned revision, with the DOI placeholder in `note` and
  no `doi` field until one is minted. The paper and the erratum are in `citation`.

**GeoCroissant.** [Spec 1.0, published 2026-01-29](https://github.com/mlcommons/croissant/blob/main/docs/croissant-geo-spec.md),
read at mlcommons/croissant commit `828034a`. It is used for `geocr:coordinateReferenceSystem`
(`EPSG:4326`), `geocr:samplingStrategy` and `geocr:spatialBias`, with `spatialCoverage` as one
`GeoShape` box per city, as the spec's own example does. GeoCroissant 1.0 has no property for a
camera heading; the heading fields are described in their field `description`, with the
convention stated there. `geocr:spatialResolution` and the band properties do not apply to
street-level panoramas and are left out.

**Links.** Every reference in the JSON to a file in this repo is a GitHub URL pinned to commit
`8a59c1572a474eeef67b8d681fd6b36d3afe7d84` (main when these files were written) or to a release
tag, so it resolves on the Hub and does not drift. The one exception is
`scripts/validate_croissant.py`, which is added by the same PR and so has no earlier commit to pin
to. The validator rejects unpinned links and bare repo-relative paths, and, where git has the
commit, checks that each linked path exists at it.

**rampnet-dataset** record sets:

| record set | key | what it is |
| :--- | :--- | :--- |
| `splits` | `name` | train / val / test (`cr:TrainingSplit` / `cr:ValidationSplit` / `cr:TestSplit`) with row counts |
| `panoramas` | none (see below) | one row per panorama: `image`, `pano_id`, `record_creation_time`, `curb_ramp_points_normalized`, `pano_coord`, `curb_ramp_coords`, `pano_azimuth`, and `split` parsed from the shard path |
| `file_manifest` | `path` | all 384 Parquet shards with size and sha256 |

**`panoramas` has no key.** No committed source shows that `pano_id` is unique across the 214,385
rows: `analysis_out/stage1_seam_scan.json` counts rows, not distinct ids, a Hub listing shows only
files, and checking it would mean reading the `pano_id` column out of the 463 GB of Parquet. So the
key is stated as unverified in the JSON rather than asserted. The 9-row difference between the
Hub's 214,385 rows and the card's 214,376 is recorded in `docs/rampnet1_findings.md` §1 and not
explained in any committed document; duplicate panoramas are one candidate, not an established one.

**rampnet-benchmark** record sets:

| record set | key | what it is |
| :--- | :--- | :--- |
| `splits` | `name` | the 9 published city splits, each a `cr:TestSplit` |
| `records` | (`split`, `pano_id`) | the ground truth: per-panorama metadata (`lat`/`lng` mapped to `sc:latitude`/`sc:longitude`, `camera_heading`, `capture_date`, `copyright`), `detections` with `verdict`, `missed`, `no_missed`, `review_group` |
| `native`, `px4096x2048` | (`split`, `pano_id`) | two imagery configs (`px4096x2048` is the `4096x2048` config, renamed so the `@id` does not begin with a digit) |
| `galleries` | (`split`, `crop_id`) | the incremental false-positive crops of single detections from the operating-point A/B pass |
| `split_extents` | `split` | per split: place, imagery source, reviewed panoramas, bounding box, capture-month range |
| `file_manifest` | `path` | all 36 Parquet files with size and sha256 |

All four per-panorama or per-crop record sets use the same rule: key on the split plus the id,
because GSV and Mapillary ids are different id spaces. The validator checks that `pano_id` is unique
within each split of the committed bundles (all 1,109 are unique across splits today too).
`split_extents` uses custom fields (`min_lat` … `max_lng`, `capture_date_min/max`); it is not a
GeoCroissant construct, and the same boxes appear in `spatialCoverage` as GeoShape boxes.
`review_group` holds the literal strings `top`, `random` and `empty`.

**Pinning.** The `repo` FileObject's `contentUrl` is the Hub tree at the pinned revision
(`https://huggingface.co/datasets/projectsidewalk/<repo>/tree/<40-hex sha>`), the same `tree/<ref>`
form the Hub's own Croissant export uses, so a consumer following it gets that revision and not
whatever `main` is later. A Hugging Face repository is a git repository and has no single content
hash, so `repo` carries the same placeholder `sha256` the Hub's own export uses (a link to
[mlcommons/croissant#80](https://github.com/mlcommons/croissant/issues/80)); `mlcroissant` does not
hash directories. The content pin is the `file_manifest` record set: every Parquet file's size and
sha256 at that revision, from the Hub's tree listing (the Git LFS object id is the sha256 of the
file's content). Neither Hub repository has a tag (`/refs` lists none); tagging the pinned revisions
would give them a readable name but is not required.

Besides the Parquet files, the dataset tree holds `README.md` and `.gitattributes`, and the
benchmark tree also holds `build_manifest.json`. `scripts/export_benchmark.py` writes that file to
record the configs and splits it built, each config's total bytes, the git commit it ran from and
the date; the copy at `63d5ffd` names commit `d7aeda4-dirty` and 2026-08-17. The `repo`
descriptions say so.

## 2. Where every number came from

| number | value | source |
| :--- | :--- | :--- |
| dataset rows per split | 150,066 / 42,878 / 21,441 (214,385) | Hub dataset viewer, `datasets-server.huggingface.co/info`, read 2026-09-25 |
| dataset card totals | 214,376 panoramas / 849,895 labels | the dataset card; the 9-row difference is recorded, not explained, in `docs/rampnet1_findings.md` §1 and `docs/seam.md` §3 |
| dataset files | 384 shards, 128 per split, sizes and sha256 | Hub tree listing at `ee882e3` |
| government records | NYC 217,679, Portland 45,035, Bend 13,357 (276,071); Table 1's 276,615 | `docs/data_provenance.md` §3, §3.3; counts **re-derived** by the validator |
| NYC share of records | 78.8% of the 276,071 committed records | 217,679 / 276,071, **re-derived** by the validator, which fails if either figure in the JSON drifts. The 78.2% quoted in `docs/rampnet1_findings.md` §1 comes from `docs/curb_ramp_data_sourcing.md` §1, which divides by the live 2026-07-30 counts (278,544); it is not used here |
| records never labelled | 43.23% (119,359) | `docs/data_provenance.md` §3.1 |
| panoramas never written | 4,571 of 219,170 (2.09%), 4,570 not served | `docs/stage1_generation_cost.md` |
| dataset city boxes | e.g. NYC 40.49902 −74.25478 40.91256 −73.70028 | **re-derived** by the validator from the committed `stage_one/dataset_generation/location_data/` with the parsers in `combine_location_data.py`, rounded to 5 decimals, after checking each file's sha256 against `docs/data_provenance.md` §3 |
| discovery / inclusion radius | 10 m / 35 m | `stage_one/dataset_generation/generate_dataset_meta.py` constants |
| label peak rule | min_distance 40 px, threshold 0.4 | `stage_one/dataset_generation/download_dataset.py` |
| imagery fetch | tile server, zoom level 3, 32 tiles, 4096x2048 | `fetch_panorama` in `rampnet/gsv.py`, which `download_dataset.py` calls; 32 tiles from `docs/stage1_generation_cost.md` |
| Stage 1 agreement | corrected P 0.9152 / R 0.9275; published P 0.9403 / R 0.9245 | `docs/rampnet1_findings.md` §1 |
| seam duplicates | 8,361 pairs, 0.98% of labels, 3.7% of panoramas | `docs/seam.md`, `docs/rampnet1_findings.md` §2 |
| dataset `rai:dataCollectionTimeframe` | 2025-06-17 (completion only) | `docs/stage1_generation_cost.md`; no committed record gives a start date |
| crop-model cities | 12 Project Sidewalk cities | `docs/data_provenance.md` §1 |
| benchmark rows per split, 1,109 total | 125 each, bend 110, richmond 124 | Hub dataset viewer, matched by the committed bundles (`split_extents`) |
| benchmark files | 36 Parquet files, sizes and sha256 | Hub tree listing at `63d5ffd` |
| per-split box, capture months, source | `split_extents` rows; overall 2007-10 to 2026-07 | derived from committed `benchmark/<city>/records.jsonl` + `verdicts.json`, reviewed panoramas only; **re-derived** by the validator |
| camera models | annapolis Trimble MX7; budapest_district5 GoPro Max (124) and LG-R105 (1); clovis GoPro Fusion; morgantown GoPro Max; richmond iSTAR Pulsar (77), GoPro Max (33), `none` (14); GSV splits record none | `pano.camera_model` in the committed `records.jsonl`, reviewed panoramas only |
| strata | `top` 5, `random` 95 (richmond 94), `empty` 25 (bend 10) | `verdicts.json` `group`, which the exporter writes as `review_group` |
| gallery crops | 314 | the benchmark card, and the Hub viewer's per-split counts (27+24+89+23+34+30+10+29+48) |
| review dates, benchmark `rai:dataCollectionTimeframe` | 2026-07-22 to 2026-08-01 | the benchmark card, from `verdicts.json` `exported_at` |
| Bend overlap | 4 panoramas; P/R 0.954/0.758 → 0.956/0.753 when dropped | `benchmark/README.md` |

Dates: `datePublished` is the day the data files were first uploaded (2025-07-15 for the dataset,
2026-08-04 for the benchmark), `dateCreated` for the dataset is the Hub repository's creation date
(2025-07-12), and `dateModified` is the last commit on the Hub (2025-08-15 and 2026-08-17), all
from the Hub's commit history.

**The published `records` config matches git, and the check is re-runnable.**
`python scripts/validate_croissant.py --rebuild-records` rebuilds the config from the committed
bundles with `scripts/export_benchmark.py`'s own `build_records`, drops the two Laurens splits
(in git, not on the Hub), and compares each file's sha256 with the `file_manifest` rows, which are
the Hub's own hashes at `63d5ffd`. On 2026-09-25 all nine matched byte for byte **with pyarrow
25.0.0**. With pyarrow 25.0.1 all nine differ, because Parquet bytes depend on the writer version,
so a mismatch under another pyarrow says nothing about the ground truth. The exporter does not
commit sha256s of its own; `file_manifest` is the only committed record of them.

## 3. What is left blank, and why

- **DOI.** Both files carry `identifier` = `DOI-NOT-YET-MINTED (issue #150): replace with the
  DataCite DOI once minted`. The offline check accepts only that exact string or a
  `https://doi.org/10.…` URL, so the placeholder cannot be half-edited. **`--release` fails on it**,
  and on any "forthcoming" or "NOT-YET" text anywhere in a file, so the dataset's
  `note = {DOI: forthcoming}` and the benchmark's `note` both have to be replaced before an upload.
- **Version.** Both files say `version: 1.0.0`. Neither Hub repository has a version tag; the
  label is proposed and follows the policy in #150 (major on any data-file change, minor on
  metadata-only). It should be fixed at the same time as the DOI. `--release` fails if it is empty.
- **`rai:annotatorDemographics`.** Not stated in any committed document, so it is omitted.
- **Benchmark creators.** The benchmark card names no individuals, so `creator` is the Project
  Sidewalk organization. Who to list is a decision for the DOI record.
- **Dataset capture dates.** `rampnet-dataset` rows carry no capture date (`record_creation_time`
  is when the generator wrote the row), so the dataset file has no `temporalCoverage`.
- **Dataset panorama extents.** Panorama coordinates for the dataset are only in the 463 GB Parquet
  files, which were not downloaded. The city boxes are the extent of the committed government
  records instead, and each box says so. Negative panoramas were sampled along street centrelines
  and can fall outside those boxes.
- **Dataset `panoramas` key.** Unverified, so not declared (§1).
- **Laurens.** `laurens_gsv` and `laurens_mapillary` are in git but not on the Hub
  (`benchmark/README.md`). The files describe the Hub, so they are not included; a test fails if
  the split list changes without the manifest.
- **Licence.** `license` is the MIT licence each dataset card declares; the benchmark's per-record
  `copyright` attribution is described as a field.

## 4. How to validate

```bash
python scripts/validate_croissant.py                      # offline: structure, links, re-derived numbers
python scripts/validate_croissant.py --mlcroissant        # + MLCommons reference validator
python scripts/validate_croissant.py --hub                # + pin == Hub main, file sizes and sha256 (network)
python scripts/validate_croissant.py --rebuild-records    # + records config rebuilt byte for byte (pyarrow 25.0.0)
python scripts/validate_croissant.py --load               # + load data through the files (mlcroissant)
python scripts/validate_croissant.py --release --mlcroissant --hub   # the gate before any upload
mlcroissant validate --jsonld croissant/rampnet-benchmark.json
```

`pytest -q` runs `tests/test_croissant.py`, which needs no network. `requirements-dev.txt` carries
`mlcroissant` (about 6 MB of pure-Python wheels beyond what the file already installs), so CI runs
the reference validator on every PR. The `--hub` drift and pagination logic is tested against
canned responses; the live `--hub` call is not part of the suite.

The run for this revision used `mlcroissant` 1.1.0 and pyarrow 25.0.0 in a scratch venv
(`python -m venv <tmp>`, then `<tmp>/Scripts/pip install mlcroissant pyarrow==25.0.0`):

```text
$ mlcroissant validate --jsonld croissant/rampnet-dataset.json
W0925 08:38:30 rdf.py:89] WARNING: The JSON-LD `@context` is not standard. ... {'repeated', 'samplingRate', 'replace', 'path'}
I0925 08:38:30 validate.py:53] Done.
$ python scripts/validate_croissant.py --mlcroissant --hub --rebuild-records --load
rampnet-dataset      ok   3 record sets, 384 files in manifest, revision ee882e3, identifier: placeholder (DOI not minted)
    loaded a synthetic one-row shard
rampnet-benchmark    ok   7 record sets, 36 files in manifest, revision 63d5ffd, identifier: placeholder (DOI not minted)
    loaded 1,109 records rows, 2,061 detections, 1,191 missed
```

**`--hub` checks two things.** It resolves the Hub's current `main` and fails if it is not the
revision in `contentUrl`, with a message to re-pin; a commit is immutable, so comparing hashes at
the pinned revision alone would pass forever after the Hub moved. It then fetches the tree listing
at the pinned revision, following the listing's `Link: rel="next"` pagination, and compares every
Parquet file's size and sha256 with `file_manifest`.

**`--load` reads data, which validation does not.** It loads the benchmark `records` record set
from the `--rebuild-records` output (the `repo` FileObject mapped to the local rebuild), and checks
the rows per split, detections and missed marks against the committed bundles. It loads the
dataset `panoramas` record set from a one-row synthetic shard written with the Hub's schema, and
checks that all eight fields parse, including the nested point arrays and the image. Loading the
real dataset through `mlcroissant` clones the Hub repository, which for `rampnet-dataset` is 463 GB;
that has not been done.

`mlcroissant` 1.1.0 on Windows builds FileSet-relative paths with backslashes, so an `includes`
pattern such as `data/records/*.parquet` matches nothing and loading fails with "No objects to
concatenate". Validation is unaffected. `--load` patches the one function involved to return POSIX
paths when `os.sep` is not `/`; on Linux or macOS it changes nothing. During the synthetic-shard load
`mlcroissant` also logs "Could not match ... in train", matching the `split` regex against the
already-extracted value `train`. The value it returns is correct, and `--load` checks it.

## 5. Notes for whoever edits these next

- **The Hub's automatic Croissant export already declares `conformsTo` 1.1** (read 2026-09-25),
  not 1.0 as #150 says. What it lacks is what these files add: RAI and GeoCroissant properties,
  field descriptions, content hashes, and the actual file layout. It describes the Hub's
  auto-converted `refs/convert/parquet` branch rather than the files on `main`.
- On the dataset, the split directories are `train/`, `val/`, `test/`; `datasets` calls `val`
  `validation`. The Croissant `splits` record set uses the directory names.
- If any Hub file changes, the revision in `contentUrl` and in the `repo` and `file_manifest`
  descriptions, the `file_manifest` rows and `dateModified` must change together. `--hub` fails as
  soon as the Hub's `main` moves past the pin, and the offline check fails if the three revision
  mentions disagree.
- If a benchmark verdict is revised and re-exported, `split_extents` still holds unless reviewed
  panoramas were added or removed; the offline check catches that. The `records` sha256s in
  `file_manifest` change, and `--hub` catches that.
- A new reference to a file in this repo must be a GitHub URL pinned to a commit or tag; the
  offline check fails on a bare path.

## 6. What remains for the other #150 checkboxes

- **Upload** both files to their Hub repositories, after the DOI and version are settled, so the
  record and the metadata agree. **The upload requires
  `python scripts/validate_croissant.py --release --mlcroissant --hub` to pass**; today it fails on
  the DOI placeholder and the "forthcoming" notes, by design.
- **Persistent identifier:** mint the DOIs, replace the placeholder in both files and in both
  `citeAs` notes (and add a `doi` field to the benchmark's), set the version, and bump
  `dateModified`.
- **Dataset cards:** the RAI text here can seed the cards' bias, limitation and use sections. Two
  card fixes surfaced by this work, for Jon to decide (the cards were not edited here):
  - The published benchmark card's `records` table says `source` is `gsv` or `mapillary`; the data
    holds `launch` or `mapillary` (the template is
    `scripts/hf_package/README.benchmark_card.template.md`, line 61).
  - The card's BibTeX has `doi = {https://doi.org/…}`, a URL in a field that takes a bare DOI, and
    the benchmark card's entry has no `url`.
- **Evaluation protocol as code and the leaderboard:** not started in this PR.
- **#127:** the Bend overlap flag and review notes are in `rai:dataLimitations` here, but not yet
  in the published rows or card.
- **Laurens:** once pushed, add the two splits to `splits`, `split_extents`, `spatialCoverage`,
  `temporalCoverage`, the FileSet split regexes and `file_manifest`.
