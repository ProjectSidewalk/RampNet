# Frozen government curb-ramp inventories

Dated snapshots of the open-government point inventories this project reads, with the endpoint and
query that produced each one. Written by `scripts/analysis/fetch_inventory.py`; see
`docs/curb_ramp_data_sourcing.md` §9 for why they exist.

**The short version:** these inventories are live services that drift. Bend has grown **+8.8%**
since the paper, Portland **+1.7%**. `stage_one/dataset_generation/location_data/` is neither
present nor tracked, and the README tells you to download from portal links that serve *current*
data — so anyone re-running Stage 1 builds a measurably different dataset from ours and has no way
to detect the difference. RampNet 1.0 is replicable only from the paper's supplemental material,
not from this repository. That is the error this directory exists to stop repeating.

## Layout

Each snapshot is two files:

| File | What it is |
| :--- | :--- |
| `<city>-<YYYY-MM-DD>.jsonl.gz` | one JSON record per line, each flattened to `{...attributes, "lon": …, "lat": …}` in **EPSG:4326** — or `{...attributes, "paths": [[[lon, lat], …]]}` for a centreline snapshot |
| `<city>-<YYYY-MM-DD>.manifest.json` | endpoint, exact query, fetch date, geometry type, declared-vs-retained count, page count, and a **sha256 of the payload** |

Read one with:

```python
import gzip, json
with gzip.open("data/inventories/denver-co-2026-07-31.jsonl.gz", "rt") as fh:
    records = [json.loads(line) for line in fh if line.strip()]
```

The payload digest is stable: gzip's `mtime` and embedded `filename` are pinned, so identical
records hash identically regardless of when or where the file was written. A changed digest means
changed data, which is the only thing it should mean.

## What is here

### Curb-ramp inventories

| Snapshot | Records | Role |
| :--- | ---: | :--- |
| `nyc-ny` | 217,679 | In training. Paper Tab. 1: 217,680 — **−0.0005%, effectively frozen**. Carries `rampid` **and** `cornerid`, which makes it the per-ramp/per-corner reference for every other city (§5d) |
| `portland-or` | 46,101 | In training. Paper Tab. 1: 45,324 — **drifted +1.7%** |
| `bend-or` | 14,805 | In training. Paper Tab. 1: 13,611 — **drifted +8.8%** |
| `denver-co` | 72,770 | Candidate, first assessed under #96 — **Good** (§5f) |
| `seattle-wa` | 38,364 | Rated **Poor** by the paper; partially re-reviewed under #96 (§5h, §5i). SDOT's own active filter; `Curb_Ramps_CDL` carries 46,431 including retired |
| `sf-ca` | 50,096 | ⚠️ **Disqualified** — only 7,553 distinct coordinates, 1:1 with the intersection node id (§5d) |
| `charlotte-nc` | 40,600 | Candidate. 5,505 rows are `RP_Type = NoRamp`; its coordinates disagree with its own corner key |
| `boston-ma` | 24,022 | Candidate, temporal ❌ (~12-yr gap) |
| `sioux-falls-sd` | 19,991 | Candidate. Publishes ramp type, which decomposes the per-corner ratio (§5d) |
| `minneapolis-mn` | 18,453 | Candidate. Own corner key; written with `--allow-partial` (4 null geometries) |
| `arlington-va` | 10,342 | Candidate |

### Street centrelines — reference geometry, not supply

Fetched with `--geometry polyline`, so records carry `paths` (lists of lon/lat vertices) instead of
`lon`/`lat`. **These are not curb ramps and must never be counted as supply.** They are the
independent reference `inventory_centerline_offset.py` measures ramp coordinates against (§5i).

| Snapshot | Segments | Role |
| :--- | ---: | :--- |
| `seattle-wa-centerlines` | 34,484 | SDOT Street Network Database (SND) |
| `denver-co-centerlines` | 7,866 | Denver's control — its reviewer-measured offset is known random, so the check must read ~0 on it |

**A city's centrelines must come from the same publisher and CRS as its ramp layer.** Seattle's SND
and its curb ramps are both EPSG:2926 from ArcGIS org `ZOyb2t4B0UYuYNYH`, reprojected to 4326 by the
same server — so a datum or reprojection fault moves both together and cancels, while a defect in
the ramp layer alone shows at full size. Substituting a national basemap's roads would destroy that
discrimination.

## Two limits, stated rather than implied

1. **These are not the paper's files.** Portland and Bend have drifted, so a Stage 1 re-run from
   here reproduces *today's* dataset, not the ICCV one. The paper-exact NYC/Portland/Bend files
   exist in exactly one place — the paper's supplemental material — and recovering them is still
   open work.
2. **Record count ≠ ramp count.** An inventory may hold multiple records per ramp, or one per
   corner. `scripts/analysis/inventory_geometry.py` measures which.

## Adding a city

```bash
python scripts/analysis/fetch_inventory.py \
    --city <slug> --fetched YYYY-MM-DD \
    --arcgis <layer-url>            # or --socrata <resource-url> [--point-field the_geom]

# its street centrelines, for the §5i registration check
python scripts/analysis/fetch_inventory.py \
    --city <slug>-centerlines --geometry polyline --fetched YYYY-MM-DD \
    --arcgis <centreline-layer-url>
```

`--fetched` is required and explicit rather than read from the clock, so the snapshot's name is not
a function of the machine that made it. Commit the pair, then analyse from the file — never from
the live endpoint.
