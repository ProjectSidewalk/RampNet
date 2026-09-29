# Flat Mapillary imagery around curb ramps: census and per-corner 3D ([#214](https://github.com/ProjectSidewalk/RampNet/issues/214), part of [#48](https://github.com/ProjectSidewalk/RampNet/issues/48))

Code:
- `scripts/analysis/flat_mapillary_48.py`: `ramps`, `census`, `select`, `fetch`, `score`;
- `scripts/analysis/flat3d/reconstruct.py`: the per-corner SfM, GS and MVS;
- `scripts/analysis/crossview_arms/flat3d.py`: the harness arms;
- `scripts/analysis/flat3d/viewer_bundle.py`.

Outputs are in `analysis_out/flat_mapillary_3d/`. The harness is `docs/crossview_align_48.md`
([PR #210](https://github.com/ProjectSidewalk/RampNet/pull/210)). Run 2026-09-28 on free
compute only (desktop, makelab2 A40).

## Summary

- **Richmond has little flat imagery near its ramps.** Within 30 m of the 253 pool ramps,
  12% of the Mapillary images are flat (1,422 of 12,150). The median ramp has 8 flat
  images, and 36% have none. The median harness corner has 7 flat images that actually
  face it. Every flat image does have Mapillary's SfM pose (§2).
- **The larger untapped source is un-thinned 360 panos:** a median of 60 within 25 m, against
  the 17 our labeler run keeps (§2).
- **Every corner reconstructs.** Per-corner SfM from the pano views plus the flat images
  registered the source and other view of all 60 Richmond pairs, in all 93 reconstructions
  (31 corners × 3 image sets):
  - 95% of flat images registered (205 / 216);
  - median reprojection error 0.99 px (§4).
- **The 3D lift halves the projection's error, and the flat images add nothing to it.**
  - The sparse lift of the click onto the reconstructed ground is 2.53° [1.79, 3.97] with
    flat images, against 4.56° for the projection.
  - Without the flat images it is 2.62°; with 30 more Mapillary panos, 2.70°.
  - Paired, flat vs no flat: +0.00° [−0.01, 0.05]; 60 pairs, 31 ramps.
  - It ties the 360-only SfM on `crossview-sfm-48` (`sfm_colmap`, 2.52°), is slightly
    behind RoMa (2.40°), and is behind MASt3R pair (1.93°; paired −0.56° [−0.77, −0.16]) (§5).
- **The dense lifts are negative results.** Reading the click's depth from the Gaussian splat
  (10.5°), from the splat's median depth (5.9°, post hoc), or from MVS (4.0°) is worse than
  the sparse ground-plane lift on the same model (§6).
- **Verdict (proposed, not decided).** Flat Mapillary imagery does not improve cross-view
  placement in Richmond. There is too little of it at our corners, and where there is
  plenty (34 images at `richmond:150`) the 360-only model already has the geometry. The
  per-corner 3D models themselves work and are reusable for inspection: viewer bundles
  for 6 corners are in §7.
  - Do not pursue flat imagery for #48 placement.
  - If more views are wanted, the un-thinned panos are the resource, but they did not
    help placement here either.

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

## 3. Phase 1: per-corner reconstruction

**Corners.** There is one corner per Richmond ramp in the frozen harness pairs: 31 corners,
60 pairs. Only Richmond pairs are eligible, because the GSV cities have no flat imagery.
Every GSV pair falls back to the projection in these arms, so read only the Richmond
table (§5).

**Corner centre.** The source click raycast onto flat ground at 2.6 m, the harness's own
aim, never the reference. The per-corner image list is `manifest.json`
(`flat_mapillary_48.py select`).

**Images** (`reconstruct.py`):
- **Harness views.** `<pair>_src.jpg`, centred on the GT click, and `<pair>_oth.jpg`,
  centred on today's projection. Both are 1024 × 768 with a 75° FOV, cut from the
  native-res pano.
- **Other run panos within 25 m.** One view of each, same size, aimed at the corner.
- **Flat images.** The Mapillary 2048-px thumbnails of every perspective flat image within
  30 m that faces the corner, resized to a 1600-px long side.
- **`mlypano` only.** Up to 30 un-thinned Mapillary panos within 25 m, full resolution,
  cut the same way as the run panos.

**Three image sets per corner:**

| variant | images | median input images per corner |
|---|---|---|
| `noflat` (control) | harness views + run pano views | 20 |
| `flat` | + flat images (0–34 per corner; 7 corners have none, so there `flat` = `noflat`) | 24 |
| `mlypano` | + up to 30 un-thinned Mapillary panos | 54 |

**SfM.**
- Features: ALIKED (top 4,096 by score, never a raster slice) + LightGlue (kornia 0.8.3),
  over every image pair.
- Mapping: pycolmap 4.2.0 incremental mapping.
- Cameras: pano views are exact pinholes. Flat images are COLMAP RADIAL from Mapillary's
  own SfM-refined `camera_parameters`. All intrinsics are fixed.
- Position priors: each image's Mapillary SfM position in the corner's ENU frame, 3 m
  horizontal sigma, robust loss. Heights are not measured, so the z prior is 2.6 m (pano)
  or 1.5 m (flat) with a 2 m sigma. The model comes out metric and approximately
  east-north-up.

**Lifting the source click to 3D.** Each lift is a separate arm:

| lift | how | arm suffix |
|---|---|---|
| sparse | The click ray meets a plane perpendicular to gravity. Gravity comes from the level source view, and the plane sits at the modal level of the reconstructed points near the click. This is `sfm.py`'s lift on `crossview-sfm-48`, same constants | `_sfm` |
| gs | gsplat 1.5.3 expected depth at the click. Trained on the undistorted registered images: 7,000 iterations, SH degree 0, L1 + 0.2 D-SSIM, default densification | `_gs` |
| gs median | **Post hoc, added after the pilot.** The depth where the click ray's transmittance through the saved splat falls to 0.5 (`gs_median_depth.py`) | `_gsmed` |
| mvs | COLMAP 3.13.0 CUDA patch-match photometric depth at the click, source view only, 20 source images | `_mvs` |

**Transfer.** The 3D point is projected into the `_oth` view's reconstructed camera, then
onto that pano's equirect. Nothing reads the reference: the manifest and the reconstruction
see the pairs with their answer columns removed.

**Pilot first.** 5 corners (the ones with the most flat images), all three variants,
reported in [#214](https://github.com/ProjectSidewalk/RampNet/issues/214#issuecomment-5883690769).
The pilot changed one thing before scaling. MVS over every view with geometric
consistency took 1,750 s for one corner, so MVS now runs photometric, source view only
(about 11 s). Pilot JSON is in git history (`a30b6d2`). All numbers below are from the
full run, which re-ran the pilot corners with the final code.

## 4. Reconstruction results (`summary.json` → `reconstruction`)

| | noflat | flat | mlypano |
|---|---|---|---|
| corners reconstructed | 31 / 31 | 31 / 31 | 31 / 31 |
| pairs with source and other view in one model | 60 / 60 | 60 / 60 | 60 / 60 |
| registered images, median [range] | 19 [5, 31] | 23 [7, 53] | 52 [12, 84] |
| flat images registered | – | 205 / 216 | 211 / 216 |
| run pano views registered | 467 / 480 | 467 / 480 | 470 / 480 |
| Mapillary panos registered | – | – | 801 / 803 |
| 3D points, median | 4,475 | 5,357 | 9,138 |
| mean reprojection error, median [range] | 1.00 [0.78, 1.16] px | 0.99 [0.84, 1.14] px | 0.96 [0.71, 1.08] px |
| camera vs its Mapillary position, median of per-corner medians | 0.37 m | 2.01 m | 0.76 m |
| GS training PSNR, median (training views, not held out) | 29.7 | 27.7 | 27.1 |
| Gaussians / .ply size, median | 151k / 10.3 MB | 165k / 11.2 MB | 184k / 12.5 MB |

**Caveats beside this table:**
- **"Registered" is not "correct."** A view can register with a wrong pose, and nothing
  here checks poses against an independent answer. The harness error in §5 is the check.
- **Flat cameras disagree with their Mapillary positions.** With flat images in the model,
  the median camera-to-prior disagreement grows from 0.37 m to 2.01 m, and a few corners
  have p90 disagreements of tens of metres (`corners/*.json` →
  `model.prior_residual_h_m_p90`). Either Mapillary's positions for the older flat drives
  are off, or some flat images register wrongly under the robust prior. Which it is was
  not resolved.
- **PSNR is on the training views.** There are too few views per corner to hold any out,
  so it measures fit, not novel-view quality.

## 5. Harness arms on the 60 Richmond pairs (`results_richmond.json`)

Scored with the harness's own functions (`flat_mapillary_48.py score`), on Richmond pairs
only. CIs resample ramps (31).
- `lg` is on the harness branch.
- `roma*` come from `analysis/crossview-matching-48`, and `sfm_colmap*`, `mast3r_pair`,
  `vggt_corner` and `mv3d_consensus` from `analysis/crossview-sfm-48`.
- Those files were first read in place with `git show` from their branches. Since all the
  family branches were merged into `analysis/crossview-align-48`, they are committed beside
  this one and the score reads them locally (`predict_all.sh`). The re-score was identical
  in every number; only `config.sources` changed.

| arm | median ° [CI] | p90 ° | within 2° | fallback | paired gain vs projection [CI] | closer than projection |
|---|---|---|---|---|---|---|
| projection (= `proj_height_auto` on Mapillary) | 4.56 [3.87, 5.57] | 11.5 | 0.10 | – | – | – |
| **flat_sfm** | **2.53 [1.79, 3.97]** | 12.3 | 0.38 | 0 | 1.74 [0.62, 2.71] | 0.72 |
| noflat_sfm (control) | 2.62 [2.18, 3.52] | 13.9 | 0.35 | 0 | 1.69 [0.54, 2.55] | 0.68 |
| mlypano_sfm | 2.70 [2.18, 3.30] | 11.3 | 0.33 | 0 | 1.65 [0.55, 2.63] | 0.70 |
| flat_mvs | 3.95 [2.65, 7.26] | 28.1 | 0.32 | 0.05 | 0.66 [−2.02, 2.22] | 0.53 |
| noflat_mvs | 4.56 [2.68, 7.20] | 28.7 | 0.27 | 0.10 | 0.80 [−2.09, 2.25] | 0.57 |
| mlypano_mvs | 4.52 [2.97, 8.02] | 26.0 | 0.23 | 0.12 | 0.57 [−2.84, 1.46] | 0.55 |
| flat_gsmed (post hoc) | 5.87 [2.63, 16.85] | 52.7 | 0.25 | 0 | −1.16 [−12.26, 0.91] | 0.40 |
| noflat_gsmed (post hoc) | 6.07 [2.70, 9.64] | 39.8 | 0.28 | 0.03 | −1.06 [−5.58, 1.86] | 0.43 |
| mlypano_gsmed (post hoc) | 4.39 [2.50, 17.60] | 42.3 | 0.28 | 0.02 | −0.14 [−14.02, 2.28] | 0.47 |
| flat_gs | 10.48 [5.22, 21.21] | 45.6 | 0.12 | 0.02 | −7.39 [−13.70, −0.77] | 0.31 |
| noflat_gs | 10.08 [4.63, 13.52] | 38.3 | 0.18 | 0.02 | −5.39 [−9.23, −0.87] | 0.32 |
| mlypano_gs | 12.56 [8.08, 22.37] | 42.0 | 0.07 | 0.02 | −8.49 [−18.25, −2.13] | 0.22 |
| *references from other branches* | | | | | | |
| lg (ALIKED + LightGlue homography) | 3.05 [2.39, 4.33] | 9.4 | 0.30 | 0.48 | 2.25 [0.88, 3.64] on 31 aligned | 0.77 |
| roma_magsac | 2.40 [2.03, 2.93] | 11.9 | 0.35 | 0.02 | 1.79 [0.71, 2.75] | 0.76 |
| sfm_colmap (360-only per-corner SfM) | 2.52 [2.06, 3.72] | 12.4 | 0.35 | 0 | 1.70 [0.25, 2.81] | 0.65 |
| vggt_corner | 2.35 [1.70, 3.47] | 10.5 | 0.42 | 0.02 | 1.55 [0.68, 3.11] | 0.78 |
| mv3d_consensus | 1.98 [1.45, 2.82] | 7.8 | 0.50 | 0.15 | 2.61 [1.24, 3.67] on 51 | 0.82 |
| mast3r_pair | 1.93 [1.57, 3.19] | 9.2 | 0.50 | 0.03 | 2.35 [1.13, 3.33] | 0.78 |

**Paired comparisons** (`summary.json` → `paired`; median per-pair difference, > 0 means the
first arm is closer; ramp-bootstrap CI):

| comparison | median ° [CI] | first arm closer |
|---|---|---|
| flat_sfm vs noflat_sfm | +0.00 [−0.01, 0.05] | 0.52 |
| mlypano_sfm vs noflat_sfm | +0.08 [−0.03, 0.20] | 0.60 |
| mlypano_sfm vs flat_sfm | +0.03 [−0.07, 0.18] | 0.55 |
| flat_sfm vs sfm_colmap | +0.01 [−0.01, 0.08] | 0.58 |
| flat_sfm vs roma_magsac | −0.13 [−0.28, −0.01] | 0.33 |
| flat_sfm vs mast3r_pair | −0.56 [−0.77, −0.16] | 0.35 |
| flat_mvs vs flat_sfm | −0.16 [−0.45, 0.20] | 0.45 |
| flat_gsmed vs flat_sfm | −1.31 [−12.89, −0.41] | 0.30 |
| flat_gsmed vs flat_gs | +0.56 [0.13, 1.74] | 0.67 |

**By oth range** (median °, projection / flat_sfm / noflat_sfm / mlypano_sfm / mast3r_pair):
- 0–6 m (6 pairs): 11.09 / 5.03 / 5.21 / 5.53 / 4.48;
- 6–12 m (23): 5.97 / 4.65 / 3.83 / 3.32 / 2.21;
- 12–18 m (31): 3.48 / 1.84 / 2.17 / 2.10 / 1.66.

Strata this small give direction, not size.

## 6. Reading

- **The 3D lift is what helps, not the flat images.** Every variant of the sparse
  ground-plane lift gets about 2.5–2.7°, against 4.56° for the projection. That is the
  same as the 360-only per-corner SfM on the other branch (2.52°). Adding flat images
  moves the paired median by 0.00°.
- **Why the flat images do not help:**
  - At most corners there are only a handful of them (median 7 facing the corner).
  - They are mostly older drives (2018, 2021) than the 2024–25 pano drives.
  - The pano views already see the ground at the click from 5–18 m, which is all the
    sparse lift needs.
- **Where flat images did matter, it went both ways:**
  - `richmond:191`: 0.47° vs 4.03° without them on p011;
  - `richmond:150`: 4.9° vs 17.1° on p046;
  - `richmond:236`: 5.38° vs 2.95° on p018.
  - These are individual pairs, and the paired median is zero.
- **More 360 views do not help either.** `mlypano`, with a median of 52 registered images
  against 19, is not measurably better than `noflat` (+0.08° [−0.03, 0.20]). The
  per-corner models are not short of views; the remaining error is elsewhere.
- **What the remaining error could be.** The residual ~2.5° sits close to the
  reference-noise scale (about 1.5° per peak, `docs/crossview_align_48.md` §2).
  - Every point here is a detection peak at each end: a peak on the source side, lifted,
    compared with a peak on the other side.
  - The worst pairs are outliers where the lift lands on the wrong surface or the other
    camera registers wrongly: p020 54°, p035 39°, p048 27°. Both viewer bundles for
    failures (§7) are such cases.
  - MASt3R pair (1.93°) does better. It predicts dense geometry for the pair directly,
    rather than a ground plane at the click.
- **Dense depth at a single pixel is fragile on these scenes.**
  - The splat's expected depth (`_gs`) is pulled by floaters in front of and behind the
    click; median depth (`_gsmed`) fixes part of that (+0.56° [0.13, 1.74] over `_gs`).
  - Both are still worse than the ground plane, with p90 around 40–50°.
  - Photometric MVS at the click has no depth in 5–12% of pairs, and its p90 is 26–29°.
  - A ground plane fitted to many points near the click is robust to a wrong single
    depth; a single-pixel read is not.
  - Untested: depth from the splat or MVS fed into the same ground-plane fit instead of
    read at one pixel.

## 7. Viewer bundles (`analysis_out/flat_mapillary_3d/scenes/<ramp_uid>/`)

Six corners, from the `flat` variant:
- best reconstructions: `richmond:191`, `richmond:96`, `richmond:186` and `richmond:150`.
  These have the most registered flat images or the flat images' clearest effect.
- failures: `richmond:204` (p020, 54°) and `richmond:38` (p035, 39°), where the lift or the
  other camera went wrong.

Each bundle has:
- `cameras.json`: every registered camera's camera-to-world rotation (OpenCV axes), centre,
  RADIAL intrinsics, image id, flat or pano, capture date, sequence, and its ENU prior;
- `points.json`: the GT click in 3D for each lift, and per harness pair every arm's
  prediction and the reference, each as a ray from the other camera. It also has the
  projection's own 3D point at 2.6 m. **This file reads the reference**; it is built after
  scoring, for inspection only.
- `points.ply`: sparse SfM points;
- `README.md`: the frame.

**The frame** is the SfM model's frame: metric and approximately east-north-up (x east,
y north, z up, metres) about the corner origin.

**The splats** (`splat.ply`, standard 3DGS format, SH degree 0, 9–13 MB each, 67 MB for
the six) are not committed. They are on makelab2 at
`/homes/gws/jonf/flat3d/scenes/<ramp_uid>/`, and their sha256 values are in `cameras.json`
→ `files`. Side-by-side renders of each harness view (real | splat) are in
`/homes/gws/jonf/flat3d/corners/<ramp_uid>/renders/` for all 93 models. Those renders are
the "render views toward the ramp" check; they are not committed.

## 8. What this does not show

- **Richmond only, 60 pairs, 31 ramps.** No GSV city has flat imagery in this pipeline.
- **The known-answer set is the harness's.** Its caveats apply unchanged
  (`docs/crossview_align_48.md` §2 and §7): pairs were admitted by a 2.6 m world test, the
  reference is a detection peak, and there are no negative pairs.
- **The flat images are only those facing the corner, and no fisheye frames.** A looser
  FOV filter or a fisheye camera model would add a few images per corner. The census
  says there would not be many more.
- **Mapillary's own poses are used only as position priors.** A variant that fixes every
  image to Mapillary's full SfM pose (`computed_rotation` + position) and only
  triangulates was not run.
- **The splat was trained once per model with one recipe.** It had no depth supervision
  and no sky or vehicle masks. A splat trained with depth priors might read depth better.
  That is untested.
- **The `_gsmed` lift is post hoc,** chosen after seeing `_gs` fail. It is reported as its
  own arm.

## 9. Reproduction

`score`, `summarize.py` and the arms' `predict` read only committed files. The rest needs
the labeler's richmond run (`results.jsonl`), the harness views, the native-res run panos
on makelab2, and a Mapillary token.

```bash
# desktop, CPU
python scripts/analysis/flat_mapillary_48.py ramps  --results LABELER/runs/richmond/results.jsonl
python scripts/analysis/flat_mapillary_48.py census --env LABELER/.env --raw-cache SCRATCH/mly_raw
python scripts/analysis/flat_mapillary_48.py select --results LABELER/runs/richmond/results.jsonl
python scripts/analysis/flat_mapillary_48.py fetch  --env LABELER/.env --out FLAT
python scripts/analysis/flat_mapillary_48.py fetch  --env LABELER/.env --out MLY --kind mly_pano
# copy FLAT and MLY to makelab2 ($B/flat, $B/mly_panos_all), then on makelab2:
bash scripts/analysis/flat3d/setup_makelab2.sh      # venv: torch 2.6 cu124, pycolmap 4.2.0, kornia 0.8.3, gsplat 1.5.3
bash scripts/analysis/flat3d/setup_colmap_cuda.sh   # micromamba env: COLMAP 3.13.0 CUDA
bash scripts/analysis/flat3d/full_makelab2.sh       # 93 reconstructions, 3.6 h on an A40
python scripts/analysis/flat3d/gs_median_depth.py --corners-root $B/corners
# copy $B/corners/<tag>/result.json to analysis_out/flat_mapillary_3d/corners/<tag>.json, then:
git fetch origin && bash scripts/analysis/flat3d/predict_all.sh
python scripts/analysis/flat3d/summarize.py
python scripts/analysis/flat3d/viewer_bundle.py --corners-root SCENES --corner richmond:191 ... \
    --out analysis_out/flat_mapillary_3d/scenes --max-mb 1
pytest -q tests/test_flat_mapillary_3d.py
```

**Inputs that are not published, and what would unblock them.**
- **Images.** They are not redistributed here. `fetched_images.csv` lists every image id with
  its sha256, so a re-fetch can be checked byte for byte. Mapillary may re-encode or remove
  images, so a later re-fetch can differ; the census tables record what existed on
  2026-09-28.
- **The harness views and native-res panos,** as in `docs/crossview_align_48.md` §9.
- **The reconstruction is not bit-reproducible.** pycolmap seeds are fixed (48), but GPU
  feature extraction, multithreaded mapping and splat training are not deterministic. A
  re-run will give slightly different numbers.

## 10. Cost and time

All on free compute. There are `paid: false` rows in `analysis_out/usage_log.jsonl`.

| step | where | wall-clock | GPU-h | API calls |
|---|---|---|---|---|
| census (metadata) | desktop | 962 s over the final runs | 0 | ~1,000 (250 bbox + sfm_cluster) |
| fetch: 193 flat thumbnails (79 MB), 686 panos (2.4 GB, plus 150 fetched twice) | desktop | 574 s | 0 | 104 |
| pilot: 15 reconstructions | makelab2 A40 | 4,535 s (1,750 s of it the all-view MVS) | ≤ 1.26 | 0 |
| full run: 93 reconstructions | makelab2 A40 | 12,877 s | ≤ 3.58 | 0 |
| `gs_median_depth`, predict, score, tests | makelab2 / desktop CPU | ~5 min | 0 | 0 |

GPU-hours are wall-clock × 1 GPU. That is an upper bound, because mapping and patch-match
setup run on the CPU. Transfers to makelab2 took about 15 min in total. No klone, no
Tillicum, no paid API.
