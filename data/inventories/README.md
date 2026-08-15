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

## Verifying a checkout

Every manifest carries `sha256` of the payload it was written beside, so a checkout can be checked
without the network:

```bash
# bash / WSL — compare each payload against its own manifest
for f in data/inventories/*.jsonl.gz; do
    want=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['sha256'])" "${f%.jsonl.gz}.manifest.json")
    got=$(sha256sum "$f" | cut -d' ' -f1)
    [ "$want" = "$got" ] && echo "OK   $(basename "$f")" || echo "FAIL $(basename "$f")"
done
```

```powershell
# PowerShell — same check, since this repo is Windows-primary
Get-ChildItem data/inventories/*.jsonl.gz | ForEach-Object {
    $want = (Get-Content ($_.FullName -replace '\.jsonl\.gz$','.manifest.json') | ConvertFrom-Json).sha256
    $got  = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower()
    "{0} {1}" -f $(if ($want -eq $got) {"OK  "} else {"FAIL"}), $_.Name
}
```

A `FAIL` means the bytes differ from the ones every §5 number was computed on — re-fetching is not
the fix, because a re-fetch returns *today's* inventory. Restore the file from git.

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

**A new city also needs a basemap before it can be reviewed**, and that is a source-code step, not
a CLI flag: `TILE_SOURCES` in `scripts/analysis/inventory_review_sheet.py` is a registry of named
services. It stays in the source deliberately — each entry carries the vegetation cover, m/px and
"what this basemap can and cannot grade" note that the verdicts have to be read against, and those
belong next to the URL they qualify, not in a config file that can drift away from it.

```bash
# 1. Find out what the service actually serves. Never read its metadata and believe it:
#    three cities in a row advertised a deeper cache than they had, and probing is
#    what caught it. --at takes a dense point where imagery must exist.
python scripts/analysis/probe_basemap.py \
    --url 'https://<host>/arcgis/rest/services/<service>/MapServer' \
    --at <lat> <lon>

# 2. Add the entry to TILE_SOURCES with the measured max_zoom, an attribution, and a
#    note stating the resolution and what it is adequate to judge.
# 3. Then build the sheet, which now offers the new source by name:
python scripts/analysis/inventory_review_sheet.py \
    --city <slug> --inventory data/inventories/<slug>-YYYY-MM-DD.jsonl.gz \
    --seed 96 --tile-source <your-new-source>
```

## Re-fetching a snapshot

Each manifest records the endpoint and the exact query that produced its payload, so the command
that made any committed snapshot is recoverable from the snapshot itself:

```bash
python -c "import json;m=json.load(open('data/inventories/denver-co-2026-07-31.manifest.json'));print(m['endpoint']);print(m['first_query'])"
```

Re-running it reproduces the *fetch*, not the *file*: these inventories drift (Bend by +8.8% since
the paper), so a re-fetch today is a new snapshot under a new `--fetched` date, and the committed
one stays the input every published number was computed on.
