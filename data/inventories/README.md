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
| `<city>-<YYYY-MM-DD>.jsonl.gz` | one JSON record per line, each flattened to `{...attributes, "lon": …, "lat": …}` in **EPSG:4326** |
| `<city>-<YYYY-MM-DD>.manifest.json` | endpoint, exact query, fetch date, declared-vs-retained count, page count, and a **sha256 of the payload** |

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

| Snapshot | Records | Role |
| :--- | ---: | :--- |
| `nyc-ny` | 217,679 | In training. Paper Tab. 1: 217,680 — **−0.0005%, effectively frozen**. Carries `rampid` **and** `cornerid`, which makes it the per-ramp/per-corner reference for every other city (§5d) |
| `portland-or` | 46,101 | In training. Paper Tab. 1: 45,324 — **drifted +1.7%** |
| `bend-or` | 14,805 | In training. Paper Tab. 1: 13,611 — **drifted +8.8%** |
| `denver-co` | 72,770 | Candidate, first assessed under #96 |

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
```

`--fetched` is required and explicit rather than read from the clock, so the snapshot's name is not
a function of the machine that made it. Commit the pair, then analyse from the file — never from
the live endpoint.
