# Juneau (#254): a second Trimble MX7 city, staged

**Status (2026-10-11): staged, NOT reviewed.** `benchmark/juneau/` holds 125 panoramas and their
detections; there is no `verdicts.json`, nothing is scored, and the split is not in
`rampnet/eval.py::PINNED_SPLITS`. Issue
[#254](https://github.com/ProjectSidewalk/RampNet/issues/254); bundle in PR
[#250](https://github.com/ProjectSidewalk/RampNet/pull/250). The labeler run is
[sidewalk-auto-labeler PR 164](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/164).

## 1. What the split is for (Phase 0)

- **The rig is Annapolis's rig.** 120 of 125 panos are 8000x4000 Trimble MX7 captures by the
  Mapillary contributor `rking`, who shot all 125 Annapolis panos (Annapolis 2020 and 2023,
  Juneau 2024-05 to 2025-09). The other five are GoPro Max (four at 4096x2048 from
  `tdbowthorpe_AKDOT`, one at 5760x2880 from `kinneyeng`). Juneau is therefore **not a new
  imagery provider**; what it can be is a replication of Annapolis's far-field distance finding
  on different streets, a different climate and later capture dates, at the highest latitude on
  the board (58.3 N).
- **Deployment target.** Juneau is a Project Sidewalk city (SidewalkWebpage#5737, downtown minus
  Douglas Island, about 4.8 km2), so like Bayonne this is its ground-truth gate.
- **Imagery tier:** survey-grade, as for `annapolis` (the records carry `Trimble` / `MX7`).
- **Pooling and tier are not decided.** Whether a second city from the same contributor pools
  with the US recommendation, or is held out as a replication, is Jon's call and is open on
  #254. Writing it down before the review is the point of this section.

## 2. The bundle

| | |
|---|---|
| source run | `runs/juneau/results.jsonl` in the labeler: 6,098 Mapillary panos, 4,999 after 5 m thinning; detection code `c30fbff`; model `606a1195`; klone job 41691449 |
| sample | 5 top-detection / 95 random / 25 empty, seed 0, min confidence 0.55, 30 m spacing (`sample.json`) |
| detections | 217 over the 100 non-empty panos, min confidence 0.5501 |
| imagery | 125 native panos, sha256 pinned in `imagery_manifest.json` (digest `f37ab52d9d1165cf`); `panos/`, `index.csv` and `gallery/` git-ignored as for every split |
| provenance | `bundle_provenance.json`: run, export, resample and staging, with the hashes |

## 3. The spacing bug and the re-sample

The first export (2026-10-09, RampNet `d6fee0f`) declared 30 m spacing but had **7 pano pairs
under 30 m, the closest 15.8 m** (two panos of one Mapillary sequence). The Opus 5.5 review of
#250 traced it to the labeler's `geo.LatLngSpacingIndex._key`, which sized the longitude grid
cell with each point's own cos(lat): at 58.3 N, 134.4 W the longitude index is about 264,000
cells from the meridian, and a 30 m change in latitude moved it by about two cells, past the
3x3 neighbourhood the index searches, so those pairs were never compared. The error grows with
|longitude| x tan(latitude), which is why Bayonne (1.5 W) is clean and Juneau is the worst case
on the board. The same bug left `laurens_mapillary` with 6 pairs under 30 m (min 26.2 m); that
split has verdicts and stands, with the cause now noted in `docs/laurens_paired_151.md`.

Fixed in [sidewalk-auto-labeler PR 165](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/165)
(one longitude cell size per index, from a reference latitude, with a 2% margin; a brute-force
test at Juneau's latitude). Re-exported 2026-10-11 from a worktree of that fix with the same
`results.jsonl`, seed and strata: **119 of 125 panos unchanged, 6 dropped, 6 added, 0 pairs
under 30 m, closest 30.7 m.** The ids are in `bundle_provenance.json`. No verdicts existed, so
nothing was discarded.

## 4. Not done yet

Everything Bayonne did before and after its review (`docs/bayonne_split_159.md` sections 4-11)
is still to do here: the GT review itself (`scripts/gt_gallery.py`), the `--unreviewed` low-floor
op_cache and the GT-free reads, challenger detections, a `juneau_254.py` with `verify`, a
`tests/test_juneau_254.py`, scoring, and registration in `PINNED_SPLITS` and
`low_floor_sweep.ALL_SPLITS`. The run archive (results.jsonl and the 125 sampled panos) is on
makelab2; `run.archive` in the provenance has the path. The labeler PR 164 is unmerged, so `c30fbff` is reachable only from its
branch until it merges.
