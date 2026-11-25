# Stage 1 `label_type` axis (#274 Phase 1 item 3): what exists, the design fork, what was built

> **Update 2026-11-03:** fork A is decided as **A1** (run-level). Fork B is being tested, B1 against B2, on a
> Seattle pilot that takes its locations from SDOT's inventory rather than OSM; see
> [`stage1_crosswalk_sdot_274.md`](stage1_crosswalk_sdot_274.md). That pilot also found that Google has
> decommissioned `GeoPhotoService.SingleImageSearch`, which `search_panos.search_panoramas` calls. Its answer is
> "decommissioned and turned down", so `generate_dataset_meta.py` can no longer discover panos as written.

Phase 1 item 3 of [#274](https://github.com/ProjectSidewalk/RampNet/issues/274): add a `label_type` axis to
Stage 1 with an OSM crossing source, keep the bearing rule shared, produce candidate crossings per pano, and
show with a parity test that CurbRamp output is unchanged. Written 2026-11-02. **Status: stopped at a design
fork, by the plan's rule** ("if the design has a real fork, write the options in the doc and stop"). The one
piece that does not depend on the fork, an OSM crossing source reader, is built and tested; nothing in the
Stage 1 pipeline scripts is changed, so CurbRamp output is unchanged by construction (there is no edit to
hold a parity test against yet; see "What a parity test will pin").

## Stage 1 as it stands (read 2026-11-02)

| step | script | in | out |
|---|---|---|---|
| locations | `stage_one/dataset_generation/combine_location_data.py` | Bend / Portland / NYC inventories | `all_locations.csv`: `latitude, longitude, date` |
| pano search + inclusion | `generate_dataset_meta.py` | `all_locations.csv` | `dataset.jsonl`: `{pano_id, pano_coords, curb_ramps_coords}`; panos within 10 m of a location are discovered, every location within 35 m is included, a pano is dropped if any included ramp's install date is not before the capture month |
| crops + placement | `download_dataset.py` | `finaldataset.jsonl` | per coord: forward azimuth pano → ramp (`pyproj.Geod.inv`) minus pano azimuth, a 90° view at −30° pitch, the centre 341 px strip (±18.37° of bearing), the crop model's heatmap, back-projected; peaks over the merged heatmap become `curb_ramp_points_normalized` |
| negatives | `generate_negative_panos.py` | street centrelines | panos ≥ 60 m from any ramp |

Three facts set the fork. The type is implicit everywhere: the location file has no type column, the jsonl
key is `curb_ramps_coords`, the output key is `curb_ramp_points_normalized`. The **inventory point is used
for bearing only**; the crop model places the ramp inside the strip. And the placer is a curb ramp model:
there is no crosswalk crop model.

## The fork

**A. Where `label_type` lives in the data.**

- **A1, a run-level parameter.** One run per label type: `all_locations_<type>.csv`, `dataset_<type>.jsonl`,
  output keys unchanged, `label_type` recorded once in the run's metadata. Smallest change; CurbRamp parity
  is trivial (a CurbRamp run is today's run); mirrors the PS server, which takes one label type per
  `AiLabelsSubmission`. Cost: a pano near both a ramp and a crossing is fetched twice and its two outputs
  must be joined later; the negative-pano rule (≥ 60 m from any ramp) has to be decided per type.
- **A2, a per-point field.** `all_locations.csv` gains `label_type`; `dataset.jsonl` becomes
  `{pano_id, pano_coords, points: [{lat, lng, label_type}]}`; the output gains points per type. One fetch per
  pano and one record per pano for every type. Cost: a schema migration of the published dataset's record
  shape, or a compatibility shim that writes `curb_ramps_coords` too; every consumer of `curb_ramps_coords`
  (`scripts/analysis/gov_provenance.py`, `stage1_bearing_residual.py`, `generate_negative_panos.py`) must be
  checked, and the inclusion rule (35 m) may not suit crossings, which are longer and seen from farther.
- **A3, parallel keys.** Keep `curb_ramps_coords`, add `crossings_coords` beside it. No migration for
  CurbRamp readers; the type list is hard-coded in key names, so a third type means a third key.

**B. What places a crossing.** The bearing rule can be shared as is (an OSM node is the crossing's centre,
Phase 0 item 2: median 1.2 m along / 1.0 m across), but the strip then needs a placer:

- **B1, the node itself, projected.** Use the OSM node's position through the pano's ground raycast as the
  label point. No model; inherits the node's offset from the paint (Phase 0: crowd labels sit a median
  1–1.5 m from the node) and the pano's position error.
- **B2, the Vistas crosswalk mask inside the strip** (Phase 0 item 3: a usable extent filter, 0.82 recall
  at 0.05 of the pano width): the component centroid nearest the bearing. A filter, not a placer, by Phase
  0's own reading, so this would be a new claim needing its own GT.
- **B3, a crosswalk crop model**, trained on crops cut at OSM bearings with B1 or B2 points as weak labels:
  what Stage 1 does for ramps, and the point of the detector, but it needs the Phase 1 GT first.

**C. The positive rule per city.** Phase 0 says marked ∪ untagged-with-crowd-label in Seattle, `unmarked`
excluded both ways. The Phase 1 GT round (`docs/crosswalk_gt_phase1_274.md`) measures it in four more
cities and an `osm_only` stratum; the reader below takes the rule as arguments so it can follow the GT.

**Recommendation (for Jon to decide, not implemented):** A1 + B1 for the first crossing run. A1 changes no
CurbRamp byte and matches the server's one-type-per-POST; B1 needs no model and its error is the one Phase 0
already measured. A2 is the right shape once a third type arrives, and is a migration best done once.
B is decided after the Phase 1 GT is scored, not before.

## What was built

`stage_one/dataset_generation/crossing_source.py` (stdlib only): Overpass `highway=crossing` nodes → Stage 1
location rows. The first three columns are exactly `combine_location_data.py`'s (`latitude, longitude,
date`), `date` is empty (OSM has no install date, so `TREAT_UNDATED_AS_PREDATING` decides, as for undated
ramps), and three columns are added: `label_type` (`Crosswalk`), `source_id` (`osm:node/<id>`), `osm_class`.
Positive rule as arguments, defaulting to Phase 0's Seattle reading (marked; untagged only with a crowd label
within 10 m; `unmarked` excluded). Output is sorted by node id and de-duplicated across overlapping tiles.
Tests: `tests/test_crossing_source_274.py` (class rule against the census's, positive rule, schema prefix read
from `combine_location_data.py`'s source, de-duplication).

Not built, because each depends on A or B: wiring the rows into `generate_dataset_meta.py`, a `label_type`
column or key in the dataset, candidate crossings per pano, and any change to `download_dataset.py`.

## What a parity test will pin

Once A is chosen: run `generate_dataset_meta.process_pano` and `download_dataset`'s azimuth step on a fixed
set of synthetic panos and ramp points (no network: `search_panoramas` and the heading lookup stubbed) before
and after the change, and assert the CurbRamp records are identical, key for key and float for float. Under
A1 that test is nearly free; under A2 it must also assert the compatibility key, if one is kept.

## Known gaps

- OSM `highway=crossing` nodes have no date, and paint is removed and repainted in a way ramps are not; a
  crossing painted after the pano was captured, or removed before it, becomes a wrong positive. Overpass
  `out meta` gives the node's last edit time, which bounds nothing about the paint.
- The 35 m inclusion and 10 m discovery radii were tuned for ramps; a crossing is 8–20 m long.
- Junction-tagged cities (Santiago, Rancagua, São Paulo; Phase 0 item 2) give bearing only; B1 there places
  the junction, not the crossing.

## Replication

| what | command | needs |
|---|---|---|
| tests | `python -m pytest tests/test_crossing_source_274.py` and `python -m doctest stage_one/dataset_generation/crossing_source.py` | nothing |
| crossing locations for a city | `crossing_source.crossing_locations(json.load(open(tile))["elements"], crowd_points=...)` over the Phase 0 census tiles (`crossing_census.py --out OUT`, `OUT/osm/<city>/Crosswalk/*.json`) | the census cache |
| figures | none: no results here | — |

Where each number lives: Phase 0 placement and class numbers, `docs/crosswalk_phase0_274.md`; the Stage 1
radii and the ±18.37° strip, `stage_one/dataset_generation/generate_dataset_meta.py` and `download_dataset.py`.
