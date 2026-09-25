# Croissant 1.1 + GeoCroissant 1.0 + RAI metadata for the two RampNet datasets

**Status: written 2026-09-25 for the third checkbox of
[#150](https://github.com/ProjectSidewalk/RampNet/issues/150).** Two files, written by hand and
validated with `mlcroissant`:

| file | describes | Hub revision it matches |
| :--- | :--- | :--- |
| [`croissant/rampnet-dataset.json`](../croissant/rampnet-dataset.json) | [`projectsidewalk/rampnet-dataset`](https://huggingface.co/datasets/projectsidewalk/rampnet-dataset), the Stage 1 training data | `ee882e3f3c779dc13182f307bca616e50d9b8c5c` (2025-08-15) |
| [`croissant/rampnet-benchmark.json`](../croissant/rampnet-benchmark.json) | [`projectsidewalk/rampnet-benchmark`](https://huggingface.co/datasets/projectsidewalk/rampnet-benchmark), the post-publication city benchmark | `63d5ffd0e4b6795702db40be89ea4d9672d91ab5` (2026-08-17) |

**Nothing has been uploaded to Hugging Face and no DOI has been minted.** These files are in the
repo only. Uploading them to the Hub records and minting the DOI are separate steps (§6).

## 1. What the files contain

Both files declare `conformsTo` Croissant 1.1, Croissant RAI 1.0 and GeoCroissant 1.0, and use the
Croissant 1.1 JSON-LD context (Appendix 1 of the spec) plus the `geocr:` prefix. They carry
`license`, `citeAs` (the paper's BibTeX), `version`, `isLiveDataset: false`, `datePublished`,
`dateModified`, creator and publisher, and these RAI properties: `rai:dataCollection`,
`rai:dataCollectionType`, `rai:dataCollectionRawData`, `rai:dataAnnotationProtocol`,
`rai:annotationsPerItem`, `rai:dataBiases`, `rai:dataLimitations`, `rai:dataUseCases`,
`rai:dataSocialImpact`, `rai:personalSensitiveInformation`, `rai:dataReleaseMaintenancePlan`
(plus `rai:machineAnnotationTools` for the dataset and `rai:dataAnnotationPlatform` for the
benchmark).

GeoCroissant ([spec 1.0, published 2026-01-29](https://github.com/mlcommons/croissant/blob/main/docs/croissant-geo-spec.md),
read at mlcommons/croissant commit `828034a`) is used for `geocr:coordinateReferenceSystem`
(`EPSG:4326`), `geocr:samplingStrategy` and `geocr:spatialBias`, with `spatialCoverage` as one
`GeoShape` box per city, as the spec's own example does. GeoCroissant 1.0 has no property for a
camera heading; the heading fields are described in their field `description`, with the
convention stated there. `geocr:spatialResolution` and the band properties do not apply to
street-level panoramas and are left out.

**rampnet-dataset** record sets:

| record set | what it is |
| :--- | :--- |
| `splits` | train / val / test (`cr:TrainingSplit` / `cr:ValidationSplit` / `cr:TestSplit`) with row counts |
| `panoramas` | one row per panorama: `image`, `pano_id`, `record_creation_time`, `curb_ramp_points_normalized`, `pano_coord`, `curb_ramp_coords`, `pano_azimuth`, and `split` parsed from the shard path |
| `file_manifest` | all 384 Parquet shards with size and sha256 |

**rampnet-benchmark** record sets:

| record set | what it is |
| :--- | :--- |
| `splits` | the 9 published city splits, each a `cr:TestSplit` |
| `records` | the ground truth: per-panorama metadata (`lat`/`lng` mapped to `sc:latitude`/`sc:longitude`, `camera_heading`, `capture_date`, `copyright`), `detections` with `verdict`, `missed`, `no_missed`, `review_group` |
| `native`, `px4096x2048`, `galleries` | the three imagery configs (`px4096x2048` is the `4096x2048` config, renamed so the `@id` does not begin with a digit) |
| `split_extents` | per split: place, imagery source, reviewed panoramas, bounding box, capture-month range |
| `file_manifest` | all 36 Parquet files with size and sha256 |

**Pinning.** A Hugging Face repository is a git repository and has no single content hash, so the
`repo` FileObject carries the same placeholder `sha256` the Hub's own Croissant export uses (a link
to [mlcommons/croissant#80](https://github.com/mlcommons/croissant/issues/80)); `mlcroissant` does
not hash directories. The actual pin is the `file_manifest` record set: every Parquet file's size
and sha256 at the revision named in the `repo` description, taken from the Hub's tree listing (the
Git LFS object id is the sha256 of the file's content). `--hub` (§4) re-checks it.

## 2. Where every number came from

| number | value | source |
| :--- | :--- | :--- |
| dataset rows per split | 150,066 / 42,878 / 21,441 (214,385) | Hub dataset viewer, `datasets-server.huggingface.co/info`, read 2026-09-25 |
| dataset card totals | 214,376 panoramas / 849,895 labels | the dataset card; the 214,385 vs 214,376 difference is `docs/rampnet1_findings.md` §1 and `docs/seam.md` §3 |
| dataset files | 384 shards, 128 per split, sizes and sha256 | Hub tree listing at `ee882e3` |
| government records | NYC 217,679, Portland 45,035, Bend 13,357 (276,071); Table 1's 276,615 | `docs/data_provenance.md` §3, §3.3 |
| NYC share of records | 78.2% | `docs/rampnet1_findings.md` §1 |
| records never labelled | 43.23% | `docs/data_provenance.md` §3.1 |
| dataset city boxes | e.g. NYC 40.49902 −74.25478 40.91256 −73.70028 | computed from the committed `stage_one/dataset_generation/location_data/` files with the parsers in `combine_location_data.py`, rounded to 5 decimals |
| discovery / inclusion radius | 10 m / 35 m | `stage_one/dataset_generation/generate_dataset_meta.py` constants |
| label peak rule | min_distance 40 px, threshold 0.4 | `stage_one/dataset_generation/download_dataset.py` |
| Stage 1 agreement | corrected P 0.9152 / R 0.9275; published P 0.9403 / R 0.9245 | `docs/rampnet1_findings.md` §1 |
| seam duplicates | 8,361 pairs, 0.98% of labels, 3.7% of panoramas | `docs/seam.md`, `docs/rampnet1_findings.md` §2 |
| generation finished | 2025-06-17 | `docs/stage1_generation_cost.md` |
| crop-model cities | 12 Project Sidewalk cities | `docs/data_provenance.md` §1 |
| benchmark rows per split, 1,109 total | 125 each, bend 110, richmond 124 | Hub dataset viewer, matched by the committed bundles (`split_extents`) |
| benchmark files | 36 Parquet files, sizes and sha256 | Hub tree listing at `63d5ffd` |
| per-split box, capture months, source | `split_extents` rows; overall 2007-10 to 2026-07 | derived from committed `benchmark/<city>/records.jsonl` + `verdicts.json`, reviewed panoramas only; re-derived by `scripts/validate_croissant.py` |
| gallery crops | 314 | the benchmark card, and the Hub viewer's per-split counts (27+24+89+23+34+30+10+29+48) |
| review dates | 2026-07-22 to 2026-08-01 | the benchmark card, from `verdicts.json` `exported_at` |
| Bend overlap | 4 panoramas; P/R 0.954/0.758 → 0.956/0.753 when dropped | `benchmark/README.md` |
| sampling strata | 5 top / N random / M empty; bend 10 empty vs 25 | `benchmark/README.md` |

Dates: `datePublished` is the day the data files were first uploaded (2025-07-15 for the dataset,
2026-08-04 for the benchmark), `dateCreated` for the dataset is the Hub repository's creation date
(2025-07-12), and `dateModified` is the last commit on the Hub (2025-08-15 and 2026-08-17), all
from the Hub's commit history.

**The published `records` config matches git.** Rebuilding it from the committed bundles with
`scripts/export_benchmark.py records --out <tmp>` (no `--push`) on 2026-09-25 produced nine Parquet
files whose sha256 equal the Hub's at `63d5ffd`, byte for byte. So the ground truth in git and on
the Hub are the same today, and the `split_extents` numbers describe the published rows.

## 3. What is left blank, and why

- **DOI.** Both files carry `identifier` = `DOI-NOT-YET-MINTED (issue #150): replace with the
  DataCite DOI once minted`. The validator accepts only that exact string or a
  `https://doi.org/10.…` URL, so the placeholder cannot be half-edited. The BibTeX in `citeAs`
  carries `note = {Dataset DOI: forthcoming}`, as the dataset card does.
- **Version.** Both files say `version: 1.0.0`. Neither Hub repository has a version tag; the
  label is proposed and follows the policy in #150 (major on any data-file change, minor on
  metadata-only). It should be fixed at the same time as the DOI.
- **`rai:annotatorDemographics`.** Not stated in any committed document, so it is omitted.
- **Benchmark creators.** The benchmark card names no individuals, so `creator` is the Project
  Sidewalk organization. Who to list is a decision for the DOI record.
- **Dataset capture dates.** `rampnet-dataset` rows carry no capture date (`record_creation_time`
  is when the generator wrote the row), so the dataset file has no `temporalCoverage`.
- **Dataset panorama extents.** Panorama coordinates for the dataset are only in the 463 GB Parquet
  files, which were not downloaded. The city boxes are the extent of the committed government
  records instead, and each box says so. Negative panoramas were sampled along street centrelines
  and can fall outside those boxes.
- **Laurens.** `laurens_gsv` and `laurens_mapillary` are in git but not on the Hub
  (`benchmark/README.md`). The files describe the Hub, so they are not included; a test fails if
  the split list changes without the manifest.
- **Licence.** `license` is the MIT licence each dataset card declares; the benchmark's per-record
  `copyright` attribution is described as a field.

## 4. How to validate

```bash
python scripts/validate_croissant.py                  # offline: structure + re-derived extents
python scripts/validate_croissant.py --mlcroissant    # + MLCommons reference validator
python scripts/validate_croissant.py --hub            # + Hub file sizes and sha256 (network)
mlcroissant validate --jsonld croissant/rampnet-benchmark.json
```

`pytest -q` runs `tests/test_croissant.py`, which needs no network; its `mlcroissant` test is
skipped when the package is not installed. The run for this PR used `mlcroissant` 1.1.0 in a
scratch venv (`python -m venv <tmp>` then `<tmp>/Scripts/pip install mlcroissant`):

```text
$ mlcroissant validate --jsonld croissant/rampnet-dataset.json
I0925 08:12:36.684304 41004 validate.py:53] Done.
$ mlcroissant validate --jsonld croissant/rampnet-benchmark.json
I0925 08:12:38.829938 63916 validate.py:53] Done.
$ python scripts/validate_croissant.py --mlcroissant --hub
rampnet-dataset      ok   3 record sets, 384 files in manifest, identifier: placeholder (DOI not minted)
rampnet-benchmark    ok   7 record sets, 36 files in manifest, identifier: placeholder (DOI not minted)
```

**Loading data through the files was checked too, not only validation.** `mlcroissant validate`
does not read any data. Two load tests were run on 2026-09-25, without downloading the Hub data:

- **benchmark `records`**, with the `repo` FileObject mapped to a local rebuild of the records
  config (the byte-identical one above): 1,109 rows, the nine splits with the viewer's counts,
  2,061 detections and 1,191 missed marks, matching the exporter's totals.
- **dataset `panoramas`**, on a one-row synthetic shard with the Hub's schema: all eight fields
  parse, including the nested point arrays and the image.

`mlcroissant` 1.1.0 on Windows builds file paths with backslashes, so a FileSet's `includes`
pattern (`data/records/*.parquet`) matches nothing and loading fails with "No objects to
concatenate". Validation is unaffected. The load tests above patched `get_fullpath` to return
POSIX paths; on Linux or macOS no patch is needed. Loading the full data with `mlcroissant` clones
the Hub repository, which for `rampnet-dataset` is 463 GB.

## 5. Notes for whoever edits these next

- **The Hub's automatic Croissant export already declares `conformsTo` 1.1** (read 2026-09-25),
  not 1.0 as #150 says. What it lacks is what these files add: RAI and GeoCroissant properties,
  field descriptions, content hashes, and the actual file layout. It describes the Hub's
  auto-converted `refs/convert/parquet` branch rather than the files on `main`.
- On the dataset, the split directories are `train/`, `val/`, `test/`; `datasets` calls `val`
  `validation`. The Croissant `splits` record set uses the directory names.
- If any Hub file changes, the `repo` description's revision, the `file_manifest` rows and
  `dateModified` must change together; `--hub` fails until they do.
- If a benchmark verdict is revised and re-exported, `split_extents` still holds unless reviewed
  panoramas were added or removed; the offline check catches that.

## 6. What remains for the other #150 checkboxes

- **Upload** both files to their Hub repositories, after the DOI and version are settled, so the
  record and the metadata agree.
- **Persistent identifier:** mint the DOIs, replace the placeholder in both files, set the
  version, and bump `dateModified`.
- **Dataset cards:** the RAI text here can seed the cards' bias, limitation and use sections.
- **Evaluation protocol as code and the leaderboard:** not started in this PR.
- **#127:** the Bend overlap flag and review notes are in `rai:dataLimitations` here, but not yet
  in the published rows or card.
- **Laurens:** once pushed, add the two splits to `splits`, `split_extents`, `spatialCoverage`,
  `temporalCoverage`, the FileSet split regexes and `file_manifest`.
