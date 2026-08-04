# Data Provenance and Training-Data Contamination Registry

This document records where every piece of RampNet training data came from, which external
services the regeneration pipeline depends on, and — critically — **which cities' data entered
training**, so that future model evaluations do not accidentally use contaminated ground truth.

## 1. Training-data contamination registry

If you evaluate a RampNet-derived model in any of these cities, the evaluation is **optimistically
biased** — the model may have seen imagery or labels from there during training. Use held-out
cities (or freshly collected post-deployment validation data, e.g. Project Sidewalk agree-rates)
for unbiased measurements.

| City | Entered training via | Pipeline stage |
| :--- | :--- | :--- |
| New York City, NY | Open-government curb ramp locations (`nyc.csv`) | Stage 1 dataset → Stage 2 model |
| Portland, OR | Open-government curb ramp locations (`portland.geojson`) | Stage 1 dataset → Stage 2 model |
| Bend, OR | Open-government curb ramp locations (`bend.geojson`) | Stage 1 dataset → Stage 2 model |
| Blackhawk Hills, IL | Project Sidewalk labels | Crop-model pre-training |
| Chicago, IL | Project Sidewalk labels | Crop-model pre-training |
| Cliffside Park, NJ | Project Sidewalk labels | Crop-model pre-training |
| Columbus, OH | Project Sidewalk labels | Crop-model pre-training |
| Knoxville, TN (knox) | Project Sidewalk labels | Crop-model pre-training |
| Mendota, IL | Project Sidewalk labels | Crop-model pre-training |
| Newberg, OR | Project Sidewalk labels | Crop-model pre-training |
| Oradell, NJ | Project Sidewalk labels | Crop-model pre-training |
| Pittsburgh, PA | Project Sidewalk labels | Crop-model pre-training |
| Seattle, WA (sea) | Project Sidewalk labels | Crop-model pre-training |
| St. Louis, MO | Project Sidewalk labels | Crop-model pre-training |
| Teaneck, NJ | Project Sidewalk labels | Crop-model pre-training |

The Project Sidewalk city list is the set of `https://sidewalk-<city>.cs.washington.edu` servers
queried in `stage_one/crop_model/ps_model/data/download_data.py`. Note the crop model's influence
flows into the Stage 1 dataset (it places every keypoint), so crop-model cities are transitively
contaminated for the full pipeline too. The 1,000-panorama manual gold set (`manual_labels/`) is
sampled from the NYC/Portland/Bend Stage 1 dataset — it is a *label-quality* gold standard, not a
geographically held-out one.

This registry also governs **city selection for any future training-corpus expansion**: a city
already listed here is already disqualified as clean evaluation ground, so training on it costs
nothing we still hold, while a registry-clean city is one we permanently give up as a benchmark
split. Candidate cities, verified inventory sizes, and that selection rule are worked through in
[`curb_ramp_data_sourcing.md`](curb_ramp_data_sourcing.md) (issue #59).

Additionally, the Project Sidewalk label CSVs are fetched **live** at pipeline run time with no
snapshot pinning: re-running `download_data.py` today produces a different crop-model training set
than the paper's, because those databases keep growing.

## 2. Undocumented Google endpoints (regeneration brittleness)

Stage 1 regeneration depends entirely on internal, unversioned Google Street View endpoints.
None of these are covered by any API contract; Google can change or remove them at any time.
As of this writing the parses are guarded by validating helpers in
`stage_one/dataset_generation/search_panos.py` that raise `GoogleEndpointSchemaError` on schema
drift instead of silently producing garbage.

### 2.1 Panorama search
- **Endpoint:** `https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch?pb=...`
  (hand-crafted protobuf-in-URL, built in `make_search_url`)
- **Parse:** JSONP payload; panorama list lives at `data[1][5][0][3][0]` (reversed), with per-pano
  fields `pano[0][1]`=id, `pano[2][0][2..3]`=lat/lon, `pano[2][2][0..2]`=heading/pitch/roll,
  `pano[3][0]`=elevation (`extract_panoramas`).
- **Failure mode:** schema drift here raises from pydantic validation or index errors.

### 2.2 Panorama metadata (capture date + heading)
- **Endpoint:** `https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/GetMetadata`
  (JSON+protobuf POST body)
- **Parses:** capture date at `[1][0][6][7]` = `[year, month]`; heading at `[1][0][5][0][1][2][0]`
  (degrees). Both feed directly into label placement — the heading determines where on the
  panorama each curb ramp keypoint lands — which is why implausible values now raise.

### 2.3 Panorama tiles
- **Endpoint:** `https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid=...&x=..&y=..&zoom=..`
- Used by both `stage_one/dataset_generation/download_dataset.py` (zoom 3, 4096x2048 target) and
  `stage_one/crop_model/ps_model/data/download_data.py` (zoom 4, 8192x4096 target). Panorama
  dimensions are discovered heuristically by probing for all-black tiles; non-standard panoramas
  can be misdetected.

## 3. Open-government source data

**These files are now committed.** `v1.0-iccv2025` shipped without them — they were excluded by a
`location_data/*` line in `stage_one/dataset_generation/.gitignore` — which left the one Stage 1
input nobody else could reconstruct sitting only on a cluster scratch directory. The live portal
links in the README serve *current* versions, and those drift (§9 measures Bend at +8.7%), so a
download made today is a different file, not a copy of this one.

| file | bytes | records | date field | sha256 |
| :--- | ---: | ---: | :--- | :--- |
| `location_data/bend.geojson` | 14,434,722 | 13,357 | `InstallDate` | `a0da4e016474c2c8fddcc6f77a7dd4a3aa5caaea455c839fad762d66a7af948e` |
| `location_data/portland.geojson` | 15,326,478 | 45,035 | `InstallDate` | `d5366a7e0d18f09f9ba49f1cbf7a26b99ee90633689dbe94cbde2a21bd395dbe` |
| `location_data/nyc.csv` | 42,057,860 | 217,679 | `GeoCyclora` | `beea2b323d00d82192dd18ace3f257cef30ce3b579544d4e607fe7abe5e57f8c` |

They are marked `binary` in `.gitattributes` so line-ending normalisation cannot alter those
hashes. **`street_data/` (801 MB) is still not committed** — `New York - Streets.geojson` alone is
669 MB, past GitHub's 100 MB hard limit. It belongs on Hugging Face; tracked in issue #21.

Install-date semantics differ per city, and many records have **no install date**; see
`TREAT_UNDATED_AS_PREDATING` in `generate_dataset_meta.py` for how those are handled.

### 3.1 Which government records are in the paper's training set

`combine_location_data.py` writes `all_locations.csv`
(sha256 `06fec4e9a8077582deac12c3c303b89c8a2396ce3d78e7e923b0960a2c091a3b`) with three columns —
`latitude`, `longitude`, `date` — then shuffles. That destroys NYC's `RampID`/`CornerID`, the
geojson `OBJECTID`/`FacilityID`/`NonAssetID`, the 24 NYC attribute columns, and **any column
saying which city a row came from**. The published file cannot, on its own, answer "is this
government ramp in the training set?"

`scripts/analysis/gov_provenance.py` rebuilds that mapping and **verifies it**:

```bash
python scripts/analysis/gov_provenance.py \
    --location-data stage_one/dataset_generation/location_data \
    --all-locations <all_locations.csv> \
    --dataset-jsonl <dataset.jsonl> \
    --out analysis_out/gov_provenance.csv
```

Output: 276,071 rows keyed by `all_locations_row`, carrying `source_file`, `source_row`, every
government ID, and `in_dataset` / `pano_id`
(sha256 `26e44cb13b5ef32ce435bf07bea057b2804bb8a99075a78cc69fd5c10a771437`; 29,438,458 bytes — not
committed, it regenerates in about a minute, and that hash is what proves a regenerated copy is
the same one). **All 276,071 rows resolved to a source record; none were unmatched.**

| | Bend | Portland | NYC | total |
| :--- | ---: | ---: | ---: | ---: |
| government records | 13,357 | 45,035 | 217,679 | 276,071 |
| dropped by the combine step | 0 | 0 | 0 | **0** |
| consumed by a generated panorama | 5,110 | 21,075 | 130,527 | **156,712** |
| consumption rate | 38.26% | 46.80% | 59.96% | **56.77%** |

So **43.23% of the government records — 119,359 of them — never became a training label**, mostly
because no panorama resolved for them or the install date failed the predates-capture check. That
number belongs next to any claim about how much open-government data the pipeline converts.
`dataset.jsonl` holds 175,336 panoramas and 959,442 ramp instances, i.e. **6.12 panorama-instances
per consumed record** — a ramp is normally visible from several panoramas.

### 3.2 Two caveats that limit this reconstruction

**The paper's row order is unreproducible.** The paper-era `combine_location_data.py` called
`random.shuffle(all_data)` with **no seed**; `random.seed(42)` was added afterwards. So the
permutation that produced the published `all_locations.csv` cannot be replayed — this is the
Stage 1 counterpart of the split-seeding caveat in §4. `gov_provenance.py` therefore joins on the
(latitude, longitude) pair instead of replaying the shuffle, which is why it works at all.

**8 coordinates are shared by two government records** (16 of the 276,071 rows). For those, the
join cannot tell the twins apart and takes the first; the script reports the count rather than
hiding it. This is also the whole of the residual date disagreement — 276,066 of 276,071 dates
reproduce exactly, and the 5 that do not are necessarily among those 16, since every unambiguous
row is derived by the same function from the same source value.

**Date semantics changed after the paper, and the affected share is measurable.** The paper-era
`convert_date` mapped an unknown date to `"2000-01-01"`, which made every undated ramp trivially
pass the "installed before the panorama was captured" check; today's returns `""` and lets
`generate_dataset_meta.py` decide. Running `gov_provenance.py` under each gives the size of the
difference directly:

| `convert_date` from | dates reproduced |
| :--- | ---: |
| paper-era checkout | 276,066 / 276,071 (100.00%) |
| current `main` | 252,983 / 276,071 (**91.64%**) |

The 23,088-row gap is exactly the set of government records **with no install date** — 8.36% of the
corpus, every one of which the paper's run silently dated to 2000-01-01 and admitted. Re-running
Stage 1 from current `main` therefore selects a **different** set of records than the paper did, and
that 8.36% is the upper bound on how much. Pass `--repo-root` to point `gov_provenance.py` at
whichever checkout's logic you mean to reproduce.

## 4. Split of record

The train/val/test split of the released dataset is frozen in
[projectsidewalk/rampnet-dataset](https://huggingface.co/datasets/projectsidewalk/rampnet-dataset).
The split scripts are seeded now, but the paper split predates the seeding — treat the HuggingFace
dataset, not a re-run of `split_dataset.py`, as authoritative for reproducing paper experiments.
