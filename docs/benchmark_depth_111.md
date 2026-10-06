# Google depth for the benchmark panoramas outside a Project Sidewalk store (#111)

**Status:** measured 2026-10-06 (UTC). Plan: the 2026-10-05 comment on
[#111](https://github.com/ProjectSidewalk/RampNet/issues/111#issuecomment-6009504321). Script
`scripts/analysis/harvest_depth_111.py`; committed record `benchmark/<split>/depth_manifest.json`
for manual_gold, bend and laurens_gsv, plus `benchmark/{bend,laurens_gsv}/depth_labeler_compare.json`;
tests `tests/test_harvest_depth_111.py`. Related: #112 (depth-binned recall, which used the
labeler's archive), #151 (Laurens), #101.

## What this answers

Google serves a metric depth payload (a plane list plus a per-pixel plane index) next to each GSV
panorama's metadata. Project Sidewalk's pano store now holds it for its own cities, and the
auto-labeler archived it for bend and laurens_gsv, but nobody held it for the 1,000 `manual_gold`
panoramas. This archives the payload verbatim for every GSV benchmark panorama outside a Project
Sidewalk store, pins each payload by sha256 in a committed manifest, and records which panoramas
Google still serves.

| question | answer |
|---|---|
| How many manual_gold panoramas still have Google depth? | **649 of 1,000** (64.9%). The other **351** are gone: Google answers the id with a not-found code. **No** served panorama lacked a depth payload (0 no_depth), and no request errored. |
| Does availability differ by source city? | NYC 385 of 571 (67.4%), Portland 209 of 348 (60.1%), Bend 55 of 81 (67.9%). |
| How old is the manual_gold imagery that survives? | Capture years run from 2007 to 2024; 2019 (115), 2024 (99) and 2022 (73) are the largest years, and 51 panoramas are from 2007. The capture month comes from the same response, so it is known only for the 649 served panoramas. |
| How much of manual_gold's depth is a real measurement? | Of 649 payloads, **271 (41.8%)** have Google's exactly-level stand-in ground plane at 2.500 m, and 4 are degenerate (two planes or fewer). The **375** with a measured ground plane give a camera height median **2.377 m** (p10 2.158, p90 2.461) and a ground tilt median 1.59 deg (p90 4.04). One measured height (0.301 m) is outside the labeler's plausible 0.8–3.5 m window. |
| Are the payloads stable over time? | **No, not on a two-month scale.** The labeler fetched bend's payloads on 2026-08-05; tonight **106 of 110** came back with different bytes. laurens_gsv's 86, fetched by the labeler on 2026-09-27, came back **byte-identical (86 of 86)**. |
| How big are the bend revisions? | Same capture, same 512x256 grid. Only 11 of the 106 keep their plane count. The plane-index agreement is a median 0.941 (min 0.477). The camera height is unchanged (under 1 mm) on 73 of 106, and the largest change is 0.114 m. The stand-in status flipped on 2 panoramas. |
| Does the decoder here reproduce the labeler's? | Yes. On every byte-identical payload (86 laurens_gsv + 4 bend) the camera height matches the labeler's `index.csv` to 1e-3 (`--check` asserts it). Before any fetch it also matched on 5,137 archived labeler payloads (3,000 bend, 2,137 laurens_gsv run panos): plane count, camera height to 1e-3 and tilt to 2e-3, 0 mismatches (an ad hoc check, not committed). |

## Coverage, all splits

From `harvest_depth_111.py summarize`. "This archive" counts are saved / gone / no_depth / error /
not_fetched.

| split | panos | source | Google depth held by | this archive |
|---|---:|---|---|---|
| manual_gold | 1000 | GSV (no source key) | none before #111 | 649 / 351 / 0 / 0 / 0 |
| bend | 110 | launch | labeler runs/bend/depth (unpublished) | 110 / 0 / 0 / 0 / 0 |
| laurens_gsv | 86 | launch | labeler runs/laurens_gsv/depth (unpublished, single copy) | 86 / 0 / 0 / 0 / 0 |
| paterson | 125 | launch | labeler archive + Project Sidewalk pano store | not fetched |
| gainesville | 125 | launch | labeler archive + Project Sidewalk pano store | not fetched |
| sao_paulo | 125 | launch | labeler archive + Project Sidewalk pano store | not fetched |
| annapolis | 125 | mapillary | no Google depth (not GSV imagery) | — |
| budapest_district5 | 125 | mapillary | no Google depth (not GSV imagery) | — |
| clovis | 125 | mapillary | no Google depth (not GSV imagery) | — |
| morgantown | 125 | mapillary | no Google depth (not GSV imagery) | — |
| richmond | 124 | mapillary | no Google depth (not GSV imagery) | — |
| laurens_mapillary | 94 | mapillary | no Google depth (not GSV imagery) | — |
| richmond_neighbourhood | 2867 | mapillary | no Google depth (not GSV imagery) | — |
| bayonne | 125 | panoramax | no Google depth (not GSV imagery) | — |

paterson, gainesville and sao_paulo were not re-fetched: the plan scoped them out because two
copies already exist. Given the bend result below, their labeler payloads are probably not what
Google serves today either. That is untested.

## manual_gold availability

| source city (coordinate box) | panos | saved | gone | no_depth |
|---|---:|---:|---:|---:|
| NYC | 571 | 385 | 186 | 0 |
| Portland | 348 | 209 | 139 | 0 |
| Bend | 81 | 55 | 26 | 0 |
| **total** | **1000** | **649** | **351** | **0** |

manual_gold's records carry no city key, so the city comes from `pano_coord` against three coarse
boxes (the cities are hundreds of km apart, so the boxes are unambiguous). A saved panorama's
heading in the response matches `records.jsonl`'s `pano_azimuth` to within 0.208 deg on all 649,
which confirms the response describes the same panorama the benchmark image is.

A gone panorama gets no capture date. So whether the losses concentrate in old captures cannot be
read from this response. The 351 gone panoramas have no Google depth from this endpoint, and a
manual_gold depth analysis covers at most the 649. They are not a random sample: the gone share
differs by city, so any depth-binned manual_gold number carries that selection.

The stand-in share (41.8%) is far above the labeler's bend (18.2% of these 110 panos here, 16% over
the whole bend run) and laurens_gsv (30.2% here). The likely reason is that manual_gold's imagery is
older on average, but that is not measured here. A camera height for those 271 panoramas has to come
from elsewhere: the plane is Google's default, not a measurement.

Capture years of the 649 saved: 2007 51, 2008 1, 2009 27, 2011 29, 2012 30, 2013 4, 2014 17,
2015 9, 2016 14, 2017 20, 2018 59, 2019 115, 2020 40, 2021 49, 2022 73, 2023 12, 2024 99.

## bend and laurens_gsv: re-fetch against the labeler's archive

Both splits were re-archived into RampNet's own store, and each payload was compared with the
labeler's copy by the sha256 of the base64 payload string.

| split | labeler fetched | re-fetched | identical | revised |
|---|---|---|---:|---:|
| laurens_gsv | 2026-09-27 | 2026-10-06 | 86 | 0 |
| bend | 2026-08-05 | 2026-10-06 | 4 | 106 |

The bend revisions are real changes to the reconstruction, not a re-encoding: plane counts change on
95 of 106. They are not a different capture: the response's capture month equals `records.jsonl`'s
`capture_date` on every bend panorama compared. In bend, the measured ground height survives the
revision on 73 of 106 panoramas (under 1 mm), and moves at most 0.114 m on the rest.

What this means for #112: `detection_recall_analysis.md` §0 was computed from the 2026-08-05 bend
payloads. Those are pinned per file in `analysis_out/recall_by_depth_112.json` and remain the inputs
of record for those tables. A re-run of #112 against today's payloads would read different planes on
106 of bend's 110 panoramas. Whether that moves any binned number has not been checked. The ground
heights mostly hold, so the expectation is a small shift, not a large one.

The general point: a depth number is a function of the fetch date as well as the panorama id. The
committed manifests pin the bytes each analysis reads, and a re-fetch can be compared byte for byte.
The bend result shows that comparison is needed.

## How it was built

- **Pano ids.** From `benchmark/<split>/records.jsonl`, asserted equal to `imagery_manifest.json`'s
  set. Ids leading with `-` (14 in manual_gold) are only ever joined into paths.
- **Fetch.** One by-id photometa request per panorama with depth requested, built with streetlevel
  0.12.10's `build_find_panorama_by_id_request_url` and streetlevel's own headers and user agent,
  sent through `requests` so the HTTP status and final URL can be inspected. Status mapping is the
  labeler's: response code `resp[1][0][0][0]` in (1, 3) is served, anything else is gone; a served
  panorama with no payload at `resp[1][0][5][0][5][1][2]` is no_depth.
- **Pacing.** Strictly serial. Sleep uniform(interval, 2 x interval) before each request, starting
  at 1.0 s, x0.8 after 200 consecutive clean requests (floor 0.25 s), doubled on 429/5xx (ceiling
  30 s); these are sidewalk-panorama-tools' `DepthPacer` numbers. A 403, a `/sorry/` or
  `consent.google.com` redirect, or a body without the JSON prefix stops the run with zero retries.
  So do 25 consecutive failures, or a no_depth share above 5% after 100 panoramas. **None fired.**
  There were 1,196 requests for 1,196 panoramas, with no retries, no push-backs and no blocks. The
  interval reached 0.41 s by the end of manual_gold.
- **Storage.** `benchmark/<split>/depth/<pano_id>.json.gz` (gitignored), in the labeler's shape
  `{"pano_id", "depth_b64", "fetched_at"}` plus a `meta` dict (heading, pitch, roll, capture
  month). `recall_by_depth_112.load_payload` therefore reads it unchanged. The manifest's `sha256`
  is of the base64 string, not the gzip file, so it does not depend on `fetched_at` or gzip
  metadata.
- **Decoder.** A stdlib port of the labeler's `depth.parse` and `depth.ground_plane`: the 8-byte
  header `<BHHHB`, uint8 plane indices, `<ffff` planes. The ground plane is the plane with the most
  pixels among those within 18 deg of horizontal and with at least 90% of their pixels below the
  horizon. Camera height is that plane's distance and tilt is its normal's angle from vertical.
  "Exactly level" is a normal of exactly (0, 0, ±1), Google's stand-in.

Archive sizes on disk: manual_gold 3.80 MB (649 files), bend 0.68 MB (110), laurens_gsv 0.26 MB (86).
Manifest digests: manual_gold `db827337fd09a5b2`, bend `e16aa8f10098df3b`, laurens_gsv
`09263279acf11f31`.

## Cost

| run | requests | wall-clock | dollars |
|---|---:|---:|---:|
| manual_gold (20-pano smoke, then the rest) | 1,000 | 1,101 s | $0 |
| bend | 110 | 178 s | $0 |
| laurens_gsv | 86 | 139 s | $0 |

All on the desktop CPU, metadata endpoint only, with no imagery and no model. Rows
`harvest-depth-111:<split>` in `analysis_out/usage_log.jsonl`, `paid: false`. The 20-pano smoke
run used a first pacer that sped up after every clean request rather than after 200. It was
corrected before the main run, and those 20 requests went out at gaps between about 0.25 and 2 s.

## What is not done

- **No upload.** The payloads are a local archive on the desktop (this worktree), the same standing
  as the labeler's archive. Whether and where to publish them is Jon's call. Until then a clean
  clone has the manifests, and `--check` verifies them without the payloads. Re-running `harvest`
  rebuilds the archive while the endpoint serves it. Given the bend result, a rebuild will not
  reproduce these exact bytes, and `verify` will report the drift per panorama.
- **No depth analysis on manual_gold.** `recall_by_depth_112.py --only manual_gold` is the next step.
  It needs a path option pointing at this archive, and it has to handle the 351 gone and 271
  stand-in panoramas explicitly.
- **paterson, gainesville, sao_paulo** were not re-fetched (out of the plan's scope). Their
  stability against the labeler's copies is untested.
- **Mapillary and Panoramax splits** have no Google depth. There was nothing to fetch.
- `scripts/analysis/harvest_depth_launch_151.py` is left as is. It is now redundant: the labeler
  fixed the `launch`/`gsv` source check in `87acfc0` ("stop refusing GSV runs by their raw source string"), merged via sidewalk-auto-labeler PR #97.
- The 351 gone manual_gold panoramas were not re-sought by coordinate (a different panorama is a
  different image).

## Reproduce

```powershell
$env:PYTHONPATH = (Get-Location).Path
pip install streetlevel==0.12.10        # harvest only; the tests and --check do not need it
python scripts/analysis/harvest_depth_111.py harvest --split manual_gold --limit 20
python scripts/analysis/harvest_depth_111.py harvest --split manual_gold --resume
python scripts/analysis/harvest_depth_111.py harvest --split bend --resume
python scripts/analysis/harvest_depth_111.py harvest --split laurens_gsv --resume
python scripts/analysis/harvest_depth_111.py compare-labeler --split bend --labeler-root D:/Git/sidewalk-auto-labeler
python scripts/analysis/harvest_depth_111.py compare-labeler --split laurens_gsv --labeler-root D:/Git/sidewalk-auto-labeler
python scripts/analysis/harvest_depth_111.py summarize
python scripts/analysis/harvest_depth_111.py --check
```

`compare-labeler` needs the labeler checkout's `runs/<split>/depth`, which is unpublished. Its
committed output carries everything `summarize` and `--check` read. Installing streetlevel 0.12.10
pulls `pyequilib`, which pulls the newest CPU `torch` from PyPI over whatever torch is installed.
It also installs `pyproj`, which turns on a latent failure in
`tests/test_stage1_bearing_residual.py::test_great_circle_matches_the_geodesic_used_by_stage_1`
(spherical vs WGS84 azimuth, 0.08 deg against a 0.01 deg bar; the test skips without pyproj). So
install streetlevel into a separate venv for `harvest`, not the project venv. This happened on the
desktop venv on 2026-10-05; torch was reinstalled and the streetlevel-only packages removed
afterwards.
