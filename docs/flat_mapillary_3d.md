# Flat Mapillary imagery around curb ramps: census and per-corner 3D ([#214](https://github.com/ProjectSidewalk/RampNet/issues/214), part of [#48](https://github.com/ProjectSidewalk/RampNet/issues/48))

Code:
- `scripts/analysis/flat_mapillary_48.py`: `ramps`, `census`, `select`, `fetch`, `score`;
- `scripts/analysis/flat3d/reconstruct.py`: the per-corner SfM, GS and MVS;
- `scripts/analysis/crossview_arms/flat3d.py`: the harness arms;
- `scripts/analysis/flat3d/viewer_bundle.py`.

Outputs are in `analysis_out/flat_mapillary_3d/`. The harness is `docs/crossview_align_48.md`
([PR #210](https://github.com/ProjectSidewalk/RampNet/pull/210)). Run 2026-09-28 on free
compute only (desktop, makelab2 A40).

<!-- SUMMARY -->

## 1. Why

Richmond, VA is our Mapillary city. The assumption going in was that most of its imagery is
ordinary phone and dashcam frames, taken every few metres along a drive, rather than 360°
panoramas.

Cross-view matching between our panoramas fails mostly on wide baselines (fallback 29%
under 10 m, 74% at 10–20 m, 100% beyond 20 m; `docs/crossview_align_48/matching.md` on
`analysis/crossview-matching-48`). Dense same-drive sequences are the opposite regime, and
Mapillary already runs SfM on them. The question has two parts:

1. How much of that imagery is there around our ramps?
2. Does a per-corner 3D reconstruction built from it place a GT ramp point in another
   pano better than what we have?

## 2. Phase 0: census (metadata only)

**Ramp positions.** A pool ramp's world position is eval_sites' merged GT point
(`multiview_evidence_48.world_gt`); its lat/lng is not committed. The committed
`captures_R25.csv` does have, for every capture within 25 m, the horizontal distance and
the equirect column at which the labeler projects the ramp. With each capture's SfM
position and heading (the labeler's richmond `results.jsonl`), each capture gives an
estimate of the ramp position. Across the captures of a ramp these agree to within 0.9 cm
at most (`census/ramps.csv`, 253 ramps).

**Query.** Graph API bbox search within 30 m of each ramp. No box hit the 2,000-result
page limit (the script splits a box that does). `sfm_cluster` was fetched separately, one
image per call, for the harness ramps' flat images: asking for it in a bbox query, or in id
batches, answers HTTP 500.

**Result.** Within 30 m of the 253 ramps there are 12,150 unique images. Only **1,422
(12%) are flat**; 10,728 are 360 panos.

| per ramp, median [p10, p90] | flat 15 m | flat 25 m | flat 30 m | flat sequences 30 m | Mapillary panos 25 m | labeler-run panos 25 m |
|---|---|---|---|---|---|---|
| all 253 ramps | 2 [0, 10] | 6 [0, 22] | 8 [0, 27] | 2 [0, 5] | 60 [17, 175] | 17 |
| 31 harness ramps | 7 [0, 10] | 15 [0, 23] | 19 [0, 26] | 3 [0, 5] | 64 [23, 149] | – |

- **90 of 253 ramps (36%) have no flat image within 30 m**, 7 of them among the 31 harness
  ramps.
- **Flat images that face the corner** have the corner inside their FOV (from their SfM
  compass angle and focal length, +15° margin; fisheye frames excluded). There is a median
  of **7 per harness corner**, range 0–34 (`manifest.json` → `n_flat_facing`).
- **Poses: all present.** Every flat image has `computed_geometry`,
  `computed_compass_angle`, `computed_rotation` and `camera_parameters` (1,422 / 1,422).
  `sfm_cluster` exists for all 490 of the harness ramps' flat images the API answered
  (33 of 523 went unanswered).
- **Cameras.** Garmin VIRB 271, unnamed 248, GoPro HERO11 194, moto x4 176, GoPro Max
  (single-lens) 171, then assorted iPhones. 1,353 are perspective and 69 fisheye.
- **Capture years.** 2018: 412, 2021: 271, 2024: 207, 2025: 437 (`summary.json`).

**Reading.** Near our ramps, the premise does not hold: 88% of the imagery is 360°. The
larger untapped source is the **un-thinned panos**. The labeler run keeps one pano per grid
cell, a median of 17 within 25 m of a ramp, while Mapillary has 60. Phase 1 therefore
has a third variant that adds up to 30 of those panos per corner (`mlypano`).

**Caveats.**
- The census counts images by position only. A flat image 20 m away that faces away from
  the corner is counted in the 30 m column.
- Positions are Mapillary's SfM `computed_geometry`, which can sit metres off for a whole
  sequence (labeler `sources/mapillary.py`, the Laurens case).

<!-- PHASE1 -->
