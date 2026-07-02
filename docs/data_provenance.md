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

The exact NYC/Portland/Bend curb ramp files used for the paper are archived in the paper's
supplemental material; the live portal links in the README serve current (drifting) versions.
Install-date semantics differ per city, and many records have **no install date**; see
`TREAT_UNDATED_AS_PREDATING` in `stage_one/dataset_generation/generate_dataset_meta.py` for how
those are handled during regeneration.

## 4. Split of record

The train/val/test split of the released dataset is frozen in
[projectsidewalk/rampnet-dataset](https://huggingface.co/datasets/projectsidewalk/rampnet-dataset).
The split scripts are seeded now, but the paper split predates the seeding — treat the HuggingFace
dataset, not a re-run of `split_dataset.py`, as authoritative for reproducing paper experiments.
