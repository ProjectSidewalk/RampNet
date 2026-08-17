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

**How `download_data.py` turns a label into a crop** (the constants live in the script, and the
published round-1 set — `rampnet-crop-model-dataset-round1` — carries the same summary on its
card): keep `CurbRamp` labels with crowd validation **Agree − Disagree ≥ 2**; fetch the label's
panorama as zoom-4 GSV tiles and resize the assembled equirectangular to 8192×4096; turn the
label's panorama x into a yaw **snapped to the nearest 30°** (12 possible headings per pano) and
render a perspective view at **FOV 90°, pitch −30°**, 2048×2048; keep the **central horizontal
third** → 683×2048 (≈37° effective hFOV). The anchor label plus every other validated label on the
same panorama is projected into the view with the matching point transform; those inside the strip
are encoded into the filename. Keypoints are projections of the **stored** Project Sidewalk
panorama coordinates — no re-annotation happened at crop time. The 30° snap keeps the anchor
within ~±15° of the view axis, guaranteeing it lands in-frame, which is why the round-1 set
contains no empty crops.

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
hashes.

**These three files hold 276,071 records; the paper's Table 1 says 276,615. Read §3.3 before
quoting either number** — the 544-record gap is real, it is between the *files* and *Table 1*
rather than anywhere in the pipeline, and it is not fully explained.

**Street centrelines (`street_data/`) are committed as a derivative.** The raw downloads are
801.6 MB — `New York - Streets.geojson` alone is 669 MB, past GitHub's 100 MB hard limit — but
their only consumer, `generate_negative_panos.py`, reads just the LineString geometry plus **one**
name field, used solely as an emptiness test (`FULLNAME` Bend, `FULL_NAME` Portland, `Street` NY).
Route numbers, ZIP, MSAG, ESN, one-way direction, road class and county are never read.
`scripts/build_street_derivative.py` strips them:

| file | as downloaded | committed derivative | ratio | sha256 |
| :--- | ---: | ---: | ---: | :--- |
| `Bend - Streets` | 8,934,761 | 553,314 | 16.1× | `2d72baada118d1e2…` |
| `Portland - Streets` | 123,616,787 | 8,238,233 | 15.0× | `ba5cd74eb6b11509…` |
| `New York - Streets` | 669,049,016 | 9,908,856 | **67.5×** | `a6864eb57b1d5913…` |
| **total** | **801,600,564** | **18,700,403** | **42.9×** | |

The name field is *kept*, not dropped: Portland has 4,192 features whose `FULL_NAME` is empty and
which `load_city_streets` skips, so a pure-geometry file would silently re-admit them.

Equivalence is proved rather than asserted. `build_street_derivative.py verify` computes a
**consumer fingerprint** — sha256 over the ordered (name, geometry) pairs of every feature
surviving the filter, which is exactly what the length-weighted sampling index is built from:

```bash
python scripts/build_street_derivative.py verify --src <raw street_data/> --out stage_one/dataset_generation/street_data
#   Bend       MATCH 7e93bb99a3ff7a00     7,179 features kept
#   New York   MATCH 81de37602a47b4be   241,206 features kept
#   Portland   MATCH c93982e64cfa7b8d   107,233 features kept
```

`generate_negative_panos.py` prefers a full download when present and falls back to the
derivative, so existing checkouts are unaffected. The pristine originals still belong on Hugging
Face (issue #21) — the derivative is lossy with respect to them, just not in any way this pipeline
can observe.

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

### 3.3 These files hold 276,071 records; the paper's Table 1 says 276,615

Both numbers are now published in this repo, so the 544-record gap between them has to be stated
rather than left for a reader to trip over. `README.md` footnote ¹ gives the paper's Table 1 total
as 276,615 — NYC 217,680 + Portland 45,324 + Bend 13,611. The committed inventories do not sum to
that:

| | Bend | Portland | NYC | total |
| :--- | ---: | ---: | ---: | ---: |
| paper Table 1 | 13,611 | 45,324 | 217,680 | **276,615** |
| committed file | 13,357 | 45,035 | 217,679 | **276,071** |
| difference | −254 | −289 | −1 | **−544** |

**It is not a pipeline drop.** Counted straight from the committed blobs, every geojson feature
carries ≥2 coordinates and every NYC row parses as a `POINT`, so the combine step discards
nothing — which is what the "dropped by the combine step: 0" row in §3.1 is reporting, and it is
correct as far as it goes. The gap is between the files and Table 1, not between the files and
`all_locations.csv`.

**What ties the files to the paper's run** is `all_locations.csv`, the paper's own combine output,
recovered from the run's scratch directory rather than regenerated (§3.2 explains why regenerating
it is impossible — the shuffle was unseeded). It has **276,071 rows, and `gov_provenance.py`
resolves all 276,071 of them against these three files with none unmatched**. A later, larger
snapshot would still match as a superset; what a later snapshot could not do is have exactly the
same row count. So these are the files that produced the paper's `all_locations.csv`.

**What is still unexplained is where Table 1's 276,615 came from.** The most likely reading is that
Table 1 quotes the counts the portals *advertised* when the survey was compiled, which is a
different measurement on a different date from the counts of the files that were downloaded and
consumed — but that is a hypothesis, not something recovered from the paper's artifacts, and the
person who compiled Table 1 is the one who can settle it.

One inference to **not** draw: the committed `nyc.csv` count (217,679) equals the 2026-07-30
re-download count in `curb_ramp_data_sourcing.md` §9, which looks at first like evidence the file
is a present-day download. It is not evidence either way — that same section measures NYC's drift
over eleven months at **one record** (−0.0005%), so matching today's portal costs NYC nothing.
The informative cities point the other way: Bend and Portland have *grown* since the paper
(+8.7% and +1.6%), so a file downloaded today would be **larger** than Table 1, and the committed
ones are smaller.

**Until that is settled, quote 276,071 for anything derived from the committed inputs** — the
consumption rates in §3.1, the provenance CSV, and any re-run of Stage 1 — and quote 276,615 only
when citing the paper's Table 1 as published.

## 4. Split of record

The train/val/test split of the released dataset is frozen in
[projectsidewalk/rampnet-dataset](https://huggingface.co/datasets/projectsidewalk/rampnet-dataset).
The split scripts are seeded now, but the paper split predates the seeding — treat the HuggingFace
dataset, not a re-run of `split_dataset.py`, as authoritative for reproducing paper experiments.
