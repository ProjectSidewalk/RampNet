# Cross-view placement ([#48](https://github.com/ProjectSidewalk/RampNet/issues/48)): multi-view 3D arms (SfM, feed-forward 3D, splatting)

One family of arms for the cross-view harness in `docs/crossview_align_48.md`
([PR #210](https://github.com/ProjectSidewalk/RampNet/pull/210)). The task: place a
source-view GT ramp point in another capture of the same ramp by reconstructing the scene in
3D. Everything uses the harness's frozen 300 pairs (`pairs.csv` sha256 `a85a11bc…`), its
reference (the other view's own RampNet detection) and its metrics. Run 2026-09-28 on
makelab2's A40, free compute only.

Code, all in `scripts/analysis/crossview_arms/`:

- `_mv3d.py`: corner manifest, corner views, camera model, and the pilot / report /
  agreement / predict-many tools;
- `sfm.py` (COLMAP) and `ff3d.py` (MASt3R, DUSt3R, VGGT, MapAnything);
- `mv3d_consensus.py`, a post hoc composite;
- `splat.py` and `_splat_run.py` (InstantSplat);
- `_scenes.py`, the 3D viewer bundles.

Tests are in `tests/test_crossview_mv3d.py`. Tables come from
`analysis_out/crossview_align_48/mv3d_results.json` (`_mv3d.py report`, scored against
both baselines). `mv3d_scores.json` is the harness's own `score` over these arms, with city,
range and date strata. The shared `results.json` has since been regenerated over all 83
arms after the family merges (`docs/crossview_align_48.md` §9); see §11.

## Summary

> **Multiplicity (added 2026-09-29, review of
> [#210](https://github.com/ProjectSidewalk/RampNet/pull/210)).** "CI-clear" in this doc
> means the arm's own uncorrected 95% ramp-bootstrap CI excludes zero. 83 arms were scored
> on the same 300 pairs, so some CI-clear gains are expected by selection alone. A
> one-sided Bonferroni screen over all 83 arms (`docs/crossview_align_48.md`, "Combined
> comparison") keeps only `mapa_posed_pair` and `mapa_posed_corner` on GSV, and those two
> plus `mapa_k_pair` and `mapa_posed_poseonly` over all 300 pairs. The Mapillary stratum was
> not screened. `mapa_posed_pair` is itself post hoc.

- **The baseline to beat is `proj_height_auto`**, not today's projection. It puts flat
  ground at the labeler's per-rig 'auto' height: median 4.06° over all 300 pairs, against
  5.62° for the projection.
- **Feed-forward 3D beats it and does not fall back.** MapAnything run on just the source and
  the other view. **`mapa_posed_pair` is post hoc:** it (and `mapa_posed_poseonly`) was
  registered in commit `0428bb7`, after `mapa_posed_corner`'s 300-pair result had been seen,
  and neither had a pilot. `mapa_mono_depthonly` came later still (`c84dc74`). Their numbers
  are in-sample; they need a re-test on fresh pairs with settings fixed in advance:
  - given the pose priors and intrinsics (`mapa_posed_pair`): median **2.80° [2.49, 3.07]**,
    0% fallback. Paired gain over auto is **0.72° [0.42, 1.01]**, closer on 68% of pairs;
  - given only the intrinsics (`mapa_k_pair`, no GPS, no heading): 2.99° [2.42, 3.67],
    1% fallback, paired gain over auto 0.73° [0.30, 1.09].
- **Other models also beat auto.** Paired gains over auto, where each arm answers:
  - CI-clear before correction: MASt3R 0.57° [0.18, 1.02] (11% fallback) and DUSt3R 0.47°
    [0.17, 0.83] (2%); neither survives the 83-arm Bonferroni screen;
  - not CI-clear: VGGT 0.34–0.41°.
- **Where the ramp is in 3D is the lever. The camera poses are not.** Each model run was read
  two more ways:
  - **pose only** (the model's relative pose, today's flat ground): MASt3R, VGGT and COLMAP
    lose to auto on GSV (−0.30° to −0.43°, all CI-clear). The learned relative poses
    disagree with the GPS / compass priors by a median 2–3°. On 11–38% of GSV pairs they
    disagree by more than 10°.
  - **depth only** (MapAnything's 3D point at the click, prior cameras): +0.38° [0.05, 0.68].
  - **The same model on the source view alone** loses to auto: −0.50° [−1.03, −0.04]. The
    gain comes from seeing the second view, not from a monocular depth prior.
- **Mapillary (Richmond) is where 3D matters most.** Every 3D arm roughly halves the error
  there: MASt3R 1.93°, DUSt3R 2.11°, MapAnything 2.12–2.68°, against 4.56° for both
  baselines (auto leaves Mapillary at 2.6 m). Paired gains are 0.9–2.6°, CI-clear. On GSV
  the gains are 0.0–0.6° and CI-clear only for MapAnything.
- **Per-corner SfM (COLMAP, learned features, with or without position priors) fails on
  GSV,** as the literature predicts:
  - it puts the source and other view in one model on 88–90% of GSV pairs, but only 55% of
    those models have a relative pose within 10° of the prior;
  - the click has no reconstructed ground on a quarter of GSV pairs (58 of 240);
  - where it answers on GSV it loses to auto: −0.92° [−2.62, −0.25] without priors,
    −0.73° with them.
  - On Mapillary's dense sequences it works (2.52°).
- **Gaussian splatting (InstantSplat) was stopped at a 5-corner pilot: it overfits.**
  Training-view PSNR rises from 14–18 dB to 25–33 dB, while the click's rendered depth moves
  by up to 2× and ends up no better than pairwise MASt3R (§7).
- **Two independent models agreeing is a free confidence signal** (post hoc). On 57% of
  pairs MapAnything and MASt3R land within 1.5° of each other. There their midpoint is 2.11°
  from the reference, against auto's 3.48° on the same pairs. That is the reference-noise
  floor. It is a candidate association signal for
  [labeler#56](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/56), untested
  against negatives (§9).
- **A 30-pair pilot was misleading, and is recorded as such.** It put every feed-forward arm
  *behind* auto by 0.2–0.8°. The same predictions beat auto on all 300 (§3).
- **Verdict (proposed, not decided):** MapAnything on the pair with the pose priors
  (`mapa_posed_pair`) is this family's best single arm. It never falls back, its gain over
  auto is CI-clear on both imagery sources (and survives the 83-arm Bonferroni screen on GSV
  and over all 300 pairs), and it takes about 0.35 s per pair on an A40. It is post hoc (see
  above), so these are in-sample numbers; `mapa_posed_corner` is the pre-specified arm
  closest to it.
  Keep the GPS / compass poses; drop per-corner SfM and splatting for GSV.

## 1. What was tested

The harness's `proj_height_auto` raycasts the click onto flat ground at a per-rig camera
height, then projects that ground point into the other view. Three things can move the
point off the ramp: the height / ground model, the source camera's pose and the other
camera's pose. This family asks whether reconstructing the scene helps, and which of those
it fixes.

1. **Per-corner SfM with pose priors** (COLMAP) uses every capture of the corner, not just the
   pair. This is the regime the literature says fails: sparse, wide-baseline and multi-date
   (`docs/multiview_48.md` §2). "Maps from Motion" reports COLMAP failing on 80% of sparse
   sequences; see also "When Wider Views Fail".
2. **Feed-forward 3D**: MASt3R / DUSt3R (pairwise), and VGGT-1B and MapAnything
   (arXiv 2509.13414) on the pair and on the corner. MapAnything is also run conditioned on
   the known intrinsics and pose priors.
3. **Pose only vs ground only.** The same model runs are read twice: once keeping only the
   model's relative pose, once keeping only its 3D point.
4. **Sparse-view Gaussian splatting** (InstantSplat, which initialises from MASt3R) on top of
   a corner reconstruction. This was requested after the first results.

## 2. Setup

**Corners.** There is one corner per ramp in the pair list (174). A corner's captures are
the run panos within 25 m of the ramp's pool position
(`analysis_out/multiview_48/captures_R25.csv`). Only the membership is used; its detection
columns are never read. That gives 1,708 captures, a median of 8 per corner (Richmond median
20, max 31). `_mv3d.py manifest` writes `mv3d_corners.json` (committed), with each capture's
pose prior and the view to cut from it.

- **Views.** The source and the pair's other view are the harness's own views (1024 × 768,
  75°). Every other capture gets a view of the same size, aimed where the harness aims the
  other view: the source click raycast onto flat ground at 2.6 m and projected into that
  capture. `_mv3d.py render` cut the 1,234 extra views on makelab2 (252 s, 6 workers).
- **Pose prior.** Position, heading and 'auto' height come from the labeler's own loader
  (`fuse_sites.load_at_height(..., 'auto')`, `pano_pose(p, 'off')`). Cameras are flat. GSV
  uses its metadata; Mapillary uses SfM `computed_geometry` / `computed_compass_angle`.
- **Instrument check.** In the manifest's frame, the flat 2.6 m transfer reproduces
  `proj_x` / `proj_y` to 0.0036° on all 300 pairs. The heights equal `proj_height_auto`'s
  on every pair pano. Both are tested.
- **No answers.** The manifest is built from the pair list with `ref_*` dropped (tested). No
  arm reads a RampNet detection in any view.

**Lift and project.** Every 3D arm lifts the source click (the centre of the source view) to
a 3D point. It then projects that point into the other view with a camera for that view:

| arm | model run | lift | other camera |
|---|---|---|---|
| `sfm_colmap` | COLMAP 4.2 incremental mapping over all the corner's views; ALIKED + LightGlue matches for every view pair; intrinsics fixed | ray meets a gravity-level ground plane at the mode of the reconstructed points' level near the click (§6) | reconstructed |
| `sfm_colmap_prior` | the same, with COLMAP's position priors (3 m / 1 m sigma, robust) | same | reconstructed |
| `mast3r_pair`, `dust3r_pair` | pairwise pointmaps, 512 × 384 | the source pointmap at the click | PnP-RANSAC on the other pointmap, known intrinsics |
| `vggt_pair`, `vggt_corner` | VGGT-1B, 518 × 392, pair or up to 12 corner views | depth head at the click, VGGT's source camera | VGGT's |
| `mapa_k_pair`, `mapa_k_corner` | MapAnything given the intrinsics | pointmap at the click | MapAnything's |
| `mapa_posed_pair`, `mapa_posed_corner` | MapAnything given intrinsics AND pose priors (metric) | pointmap at the click | MapAnything's (it keeps the priors to ~0.3°) |
| `splat_instantsplat` (+ `_init`) | InstantSplat on up to 12 corner views (§7) | rendered depth at the click | InstantSplat's optimised one |

Other readings of the same runs (`ff3d._read`, `_mv3d.poseonly_transfer`):

- **`*_poseonly`** keeps only the model's source → other relative rotation and baseline
  direction. The source camera keeps its prior, the baseline length comes from the priors,
  and the click goes through today's flat-ground transfer at the 'auto' height. A model that
  agrees exactly with the prior reproduces `proj_height_auto`; this is tested in an
  arbitrary similarity frame.
- **`mapa_posed_depthonly`** keeps only MapAnything's 3D point relative to the source camera,
  projected with the other view's prior camera.
- **`mapa_mono_depthonly`** is the monocular control: MapAnything sees the source view alone.

A corner run uses up to 12 views: the source, the other view, then the captures nearest the
ramp. An arm falls back (the harness scores it at the projection) when:

- a view is not registered;
- the click has no support;
- the point lands behind the other camera.

## 3. Pilot, and why it was not used to drop arms

Each arm first ran on 30 pairs: the first 6 per city by pair id, which is 15 ramps
(`_mv3d.py pilot`). Predictions are in `mv3d_pilot_<arm>.jsonl`; the table comes from
`_mv3d.py pilot-report`. There are no CIs, since n = 30. The splat rows cover only the 5
splat pilot corners (7 of these pairs).

| pilot | fallback | median, 30 pairs | n used | used: arm vs auto vs projection | used: gain vs projection | used: gain vs auto | used: closer than auto |
|---|---|---|---|---|---|---|---|
| dust3r_pair | 0.00 | 2.85 | 30 | 2.85 vs 3.63 vs 4.69 | +0.38 | −0.84 | 0.37 |
| mapa_k_corner | 0.03 | 3.68 | 29 | 3.97 vs 3.68 vs 4.77 | +0.23 | −0.47 | 0.38 |
| mapa_k_pair | 0.00 | 4.01 | 30 | 4.01 vs 3.63 vs 4.69 | +0.14 | −0.71 | 0.40 |
| mapa_posed_corner | 0.00 | 2.92 | 30 | 2.92 vs 3.63 vs 4.69 | +0.14 | −0.70 | 0.40 |
| mapa_posed_depthonly__frame_bug | 0.33 | 35.19 | 20 | 40.64 vs 1.48 vs 3.75 | −37.62 | −39.37 | 0.00 |
| mast3r_pair | 0.13 | 2.37 | 26 | 2.34 vs 3.63 vs 4.69 | +0.53 | −0.18 | 0.46 |
| sfm_colmap | 0.40 | 5.56 | 18 | 5.56 vs 4.36 vs 4.69 | −0.27 | −1.56 | 0.33 |
| sfm_colmap__planelift | 0.30 | 6.65 | 21 | 5.36 vs 4.28 vs 4.61 | −2.42 | −1.26 | 0.33 |
| sfm_colmap_prior | 0.37 | 5.88 | 19 | 7.09 vs 4.61 vs 4.77 | −1.15 | −1.34 | 0.21 |
| sfm_colmap_prior__planelift | 0.37 | 7.02 | 19 | 7.97 vs 4.57 vs 4.77 | −2.47 | −2.47 | 0.26 |
| sfm_poseonly | 0.07 | 5.06 | 28 | 5.06 vs 3.91 vs 4.69 | +0.40 | −0.17 | 0.46 |
| splat_instantsplat | 0.77 | 4.32 | 7 | 4.40 vs 4.77 vs 6.52 | +1.13 | −0.33 | 0.43 |
| splat_instantsplat_init | 0.77 | 4.13 | 7 | 4.04 vs 4.77 vs 6.52 | +1.76 | +0.09 | 0.57 |
| vggt_corner | 0.10 | 2.94 | 27 | 3.49 vs 3.68 vs 5.77 | +1.01 | −0.46 | 0.48 |
| vggt_pair | 0.17 | 2.99 | 25 | 3.24 vs 3.57 vs 5.77 | +0.66 | −0.79 | 0.40 |

On the same 30 pairs: projection 4.69°, `proj_height_auto` 3.63°.

What the pilot decided, and what it got wrong:

- **It caught two bugs and one bad design.**
  - The first SfM lift was a free RANSAC plane through the points near the click. It locked
    onto background structure and put the click about 20% too far
    (`sfm_colmap__planelift`, `sfm_colmap_prior__planelift`). It was replaced by the
    gravity-level ground lift, post hoc. The pilot's 30 pairs are a subset of the 300, so
    `sfm_colmap` as committed is a post hoc arm (marked so in the combined table).
  - The first `mapa_posed_depthonly` read the point in MapAnything's output frame. That
    frame is re-centred on the first view (the source's "pose shift" was exactly its camera
    height), not the prior's frame (`__frame_bug`).
  - All three are kept as records.
- **It was not representative.** On its 30 pairs, every feed-forward arm lost to auto by
  0.2–0.8°. On all 300, the same arms beat auto by 0.3–0.7°.
  - The pilot pairs' own predictions did not change. For `mapa_posed_corner` the median
    shift between the pilot and the full run on those 30 pairs is 0.02°, and the paired
    gain on them is −0.70 in both runs.
  - Auto is simply unusually good on those 15 ramps: 3.63° against 4.06° overall.
  - A pilot this small should decide only whether an arm runs at all, never which arm is
    better.
- **So every arm that ran without crashing was run on all 300.** That includes the ones the
  pilot made look dominated (`vggt_pair`, `mapa_k_pair`, `dust3r_pair`). Only the three
  superseded variants above stop at the pilot, plus splatting, which was stopped for a
  different reason (§7).

## 4. Results, all 300 pairs

Errors are the median angular error to the reference. CIs are 2.5–97.5 percentiles over
2,000 resamples of ramps. "Used" means the pairs the arm placed (did not fall back on). Gains
are paired medians (baseline minus arm), so positive means the arm is closer.

| arm | fallback | median ° [CI], all 300 | within 2° | used n | used: arm vs auto vs projection ° | used: gain over projection [CI] | used: gain over auto [CI] | used: closer than auto |
|---|---|---|---|---|---|---|---|---|
| projection | 0 | 5.62 [4.53, 6.56] | 0.19 | 300 | – | – | – | – |
| **proj_height_auto** (baseline) | 0 | 4.06 [3.66, 4.56] | 0.24 | 300 | – | +0.00 [+0.00, +0.43] | – | – |
| `mapa_posed_pair` | 0.00 | **2.80 [2.49, 3.07]** | 0.32 | 300 | 2.80 vs 4.06 vs 5.62 | +1.35 [+0.83, +2.36] | **+0.72 [+0.42, +1.01]** | 0.68 |
| `mapa_posed_corner` | 0.00 | 2.86 [2.56, 3.31] | 0.35 | 300 | 2.86 vs 4.06 vs 5.62 | +1.42 [+0.67, +2.26] | +0.61 [+0.36, +0.88] | 0.65 |
| `mapa_k_pair` | 0.01 | 2.99 [2.42, 3.67] | 0.34 | 297 | 2.96 vs 4.05 vs 5.57 | +1.24 [+0.86, +2.53] | +0.73 [+0.30, +1.09] | 0.64 |
| `mapa_k_corner` | 0.06 | 3.16 [2.77, 3.83] | 0.31 | 283 | 3.03 vs 4.05 vs 5.56 | +0.98 [+0.58, +2.26] | +0.62 [+0.21, +1.01] | 0.61 |
| `mast3r_pair` | 0.11 | 3.20 [2.66, 3.78] | 0.34 | 268 | 2.96 vs 4.00 vs 5.52 | +1.20 [+0.55, +2.32] | +0.57 [+0.18, +1.02] | 0.60 |
| `dust3r_pair` | 0.02 | 3.08 [2.59, 3.59] | 0.34 | 294 | 3.07 vs 4.04 vs 5.57 | +1.39 [+0.76, +2.41] | +0.47 [+0.17, +0.83] | 0.60 |
| `vggt_pair` | 0.15 | 3.51 [2.92, 4.73] | 0.31 | 256 | 3.35 vs 4.01 vs 5.56 | +1.03 [+0.52, +1.69] | +0.41 [−0.16, +0.88] | 0.56 |
| `vggt_corner` | 0.07 | 3.55 [2.98, 4.45] | 0.29 | 278 | 3.55 vs 4.13 vs 5.88 | +0.79 [+0.28, +1.41] | +0.34 [+0.00, +0.66] | 0.57 |
| `sfm_colmap` | 0.29 | 4.90 [3.81, 6.22] | 0.24 | 212 | 4.99 vs 4.33 vs 5.88 | +0.01 [−0.59, +0.88] | −0.30 [−0.84, +0.07] | 0.44 |
| `sfm_colmap_prior` | 0.34 | 4.44 [3.81, 5.90] | 0.24 | 199 | 4.14 vs 4.23 vs 5.57 | +0.48 [−0.30, +1.31] | −0.16 [−0.73, +0.38] | 0.47 |
| `mapa_posed_depthonly` | 0.00 | 3.09 [2.73, 3.50] | 0.31 | 300 | 3.09 vs 4.06 vs 5.62 | +0.94 [+0.36, +2.12] | +0.38 [+0.05, +0.68] | 0.57 |
| `mapa_mono_depthonly` | 0.00 | 4.94 [4.37, 5.53] | 0.17 | 300 | 4.94 vs 4.06 vs 5.62 | +0.07 [−0.46, +0.53] | −0.50 [−1.03, −0.04] | 0.42 |
| `mapa_posed_poseonly` | 0.00 | 3.90 [3.45, 4.40] | 0.25 | 300 | 3.90 vs 4.06 vs 5.62 | +0.39 [+0.22, +0.63] | +0.07 [+0.03, +0.10] | 0.63 |
| `mast3r_poseonly` | 0.08 | 4.43 [3.69, 5.71] | 0.23 | 277 | 4.36 vs 4.05 vs 5.55 | +0.45 [+0.02, +0.84] | −0.19 [−0.44, +0.01] | 0.44 |
| `vggt_corner_poseonly` | 0.08 | 4.93 [4.15, 6.29] | 0.20 | 277 | 4.92 vs 4.09 vs 5.77 | +0.02 [−0.25, +0.57] | −0.18 [−0.46, −0.01] | 0.43 |
| `sfm_poseonly` | 0.16 | 6.00 [4.91, 6.91] | 0.18 | 253 | 5.68 vs 4.09 vs 5.49 | −0.10 [−0.50, +0.43] | −0.18 [−0.41, +0.02] | 0.43 |
| `mv3d_consensus` (post hoc) | 0.43 | 3.11 [2.69, 4.16] | 0.33 | 172 | 2.11 vs 3.48 vs 4.19 | +1.25 [+0.85, +2.47] | +0.84 [+0.42, +1.24] | 0.67 |
| `mv3d_consensus_else_auto` (post hoc) | 0.00 | 3.03 [2.58, 3.72] | 0.34 | 300 | 3.03 vs 4.06 vs 5.62 | +1.09 [+0.61, +1.69] | +0.00 [+0.00, +0.00] | 0.39 |

- **All-pairs medians.** A fallback counts at the projection's error, so a high-fallback
  arm's all-pairs median is pulled toward 5.62°. The "used" columns show what the arm does
  when it answers; the fallback column shows how often it does.
- **`mv3d_consensus_else_auto`.** Its median paired gain over auto is 0.00 by construction,
  because 43% of its pairs *are* auto's point. Its median, 3.03° against 4.06°, is the fair
  summary.
- **By range** (`mv3d_scores.json`; median °, projection / auto / `mapa_posed_pair` /
  `mapa_k_pair`):
  - 0–6 m: 12.67 / 9.54 / 7.87 / 6.82;
  - 6–12 m: 6.87 / 4.68 / 3.05 / 3.42;
  - 12–18 m: 3.40 / 2.90 / 2.11 / 2.13.

  The gain is largest close in, where the flat-ground depression angle is most sensitive.
- **By date** (same month / different month):
  - `mapa_posed_pair`: 2.83° / 2.74°;
  - `mast3r_pair`: 2.70° / 3.78°.

  MASt3R's gain over the projection vanishes across capture dates: its median paired gain on
  the 109 different-month pairs is 0.00° (fallbacks count as zero). MapAnything given the
  priors does not have that problem.

## 5. By imagery

| arm | GSV (240): fallback, median ° all pairs [CI] | GSV used: gain over auto [CI] | Mapillary (60): fallback, median ° all pairs [CI] | Mapillary used: gain over auto (= over projection) [CI] |
|---|---|---|---|---|
| **proj_height_auto** | 0.00, 3.92 [3.53, 4.36] | – | 0.00, 4.56 [3.87, 5.57] | – |
| `mapa_posed_pair` | 0.00, 2.85 [2.58, 3.39] | +0.63 [+0.28, +0.92] | 0.00, 2.26 [1.89, 3.01] | +1.73 [+0.83, +2.77] |
| `mapa_posed_corner` | 0.00, 3.00 [2.58, 3.39] | +0.51 [+0.23, +0.84] | 0.00, 2.68 [1.95, 3.72] | +0.94 [+0.42, +2.44] |
| `mapa_k_pair` | 0.01, 3.28 [2.61, 3.96] | +0.35 [+0.13, +0.77] | 0.00, 2.16 [1.47, 2.90] | +2.60 [+1.09, +3.23] |
| `mapa_k_corner` | 0.07, 3.59 [3.01, 4.30] | +0.33 [−0.02, +0.72] | 0.00, 2.12 [1.60, 2.97] | +2.44 [+0.80, +2.79] |
| `mast3r_pair` | 0.12, 3.66 [2.88, 4.29] | +0.33 [−0.06, +0.61] | 0.03, 1.93 [1.57, 3.19] | +2.35 [+1.13, +3.33] |
| `dust3r_pair` | 0.03, 3.30 [2.93, 4.04] | +0.24 [−0.06, +0.50] | 0.00, 2.11 [1.55, 2.75] | +2.60 [+1.28, +3.07] |
| `vggt_pair` | 0.18, 4.44 [3.34, 5.67] | +0.03 [−0.85, +0.41] | 0.02, 2.06 [1.52, 2.96] | +2.11 [+1.30, +3.34] |
| `vggt_corner` | 0.09, 4.17 [3.19, 5.27] | +0.02 [−0.52, +0.33] | 0.02, 2.35 [1.70, 3.47] | +1.55 [+0.68, +3.11] |
| `sfm_colmap` | 0.37, 5.86 [4.39, 7.38] | −0.92 [−2.62, −0.25] | 0.00, 2.52 [2.06, 3.72] | +1.70 [+0.25, +2.81] |
| `sfm_colmap_prior` | 0.42, 5.57 [4.14, 6.87] | −0.73 [−2.06, −0.09] | 0.02, 2.54 [2.11, 3.73] | +1.56 [+0.23, +2.73] |
| `mapa_posed_depthonly` | 0.00, 3.09 [2.68, 3.60] | +0.38 [−0.03, +0.67] | 0.00, 3.16 [2.40, 4.20] | +0.41 [+0.04, +2.10] |
| `mapa_mono_depthonly` | 0.00, 4.95 [4.31, 5.68] | −0.54 [−1.39, −0.04] | 0.00, 4.81 [3.39, 6.36] | −0.42 [−1.12, +0.66] |
| `mapa_posed_poseonly` | 0.00, 3.80 [3.27, 4.37] | +0.05 [+0.02, +0.09] | 0.00, 4.14 [3.72, 5.45] | +0.17 [+0.06, +0.25] |
| `mast3r_poseonly` | 0.09, 4.81 [3.70, 5.96] | −0.43 [−0.63, −0.14] | 0.03, 4.08 [3.15, 5.08] | +0.67 [+0.30, +1.27] |
| `vggt_corner_poseonly` | 0.09, 5.09 [4.18, 6.67] | −0.42 [−0.55, −0.10] | 0.02, 4.45 [3.21, 5.91] | +0.24 [−0.05, +0.85] |
| `sfm_poseonly` | 0.19, 6.45 [5.24, 8.20] | −0.30 [−0.70, −0.07] | 0.02, 4.29 [3.62, 5.61] | +0.19 [−0.27, +0.65] |
| `mv3d_consensus` | 0.50, 3.93 [2.95, 5.27] | +0.40 [+0.05, +0.68] | 0.15, 1.98 [1.45, 2.82] | +2.61 [+1.24, +3.67] |
| `mv3d_consensus_else_auto` | 0.00, 3.54 [2.86, 4.13] | +0.00 [+0.00, +0.00] | 0.00, 1.98 [1.45, 2.82] | +1.77 [+0.87, +2.97] |

- **Mapillary.** Gains are large for every 3D arm, including SfM and the pose-only readings
  (0.2–0.7°). Auto does not change Mapillary, which stays at 2.6 m, so there the gain over
  auto equals the gain over the projection.
  - The positive pose-only gains say Richmond's poses are worth correcting.
  - Image matching (`lg`) found the same (`docs/crossview_align_48.md` §5).
  - For comparison, `lg` on Mapillary is 3.05° over all pairs, with 48% fallback.
- **GSV.** Gains are 0.0–0.6° and CI-clear only for the MapAnything pair and posed arms.
  Every *learned* relative pose (MASt3R, VGGT, COLMAP) makes the pose-only transfer worse
  than auto. MapAnything given the priors keeps them, so its pose-only reading equals auto
  plus 0.05°.

## 6. Pose or ground geometry?

| reading (same model run) | GSV: paired gain over auto [CI] | Mapillary: paired gain over auto [CI] |
|---|---|---|
| MapAnything posed, full (`mapa_posed_corner`) | 0.51 [0.23, 0.84] | 0.94 [0.42, 2.44] |
| MapAnything posed, depth only (`mapa_posed_depthonly`) | 0.38 [−0.03, 0.67] | 0.41 [0.04, 2.10] |
| MapAnything posed, pose only (`mapa_posed_poseonly`) | 0.05 [0.02, 0.09] | 0.17 [0.06, 0.25] |
| MapAnything on the source view alone, depth only (`mapa_mono_depthonly`) | −0.54 [−1.39, −0.04] | −0.42 [−1.12, 0.66] |
| MASt3R, full / pose only | 0.33 [−0.06, 0.61] / −0.43 [−0.63, −0.14] | 2.35 [1.13, 3.33] / 0.67 [0.30, 1.27] |
| VGGT corner, full / pose only | 0.02 [−0.52, 0.33] / −0.42 [−0.55, −0.10] | 1.55 [0.68, 3.11] / 0.24 [−0.05, 0.85] |
| COLMAP, full / pose only | −0.92 [−2.62, −0.25] / −0.30 [−0.70, −0.07] | 1.70 [0.25, 2.81] / 0.19 [−0.27, 0.65] |

- **Where the ramp is in 3D is the lever; the camera poses are not.** On GSV, swapping in any
  learned relative pose makes the flat-ground transfer worse.
  - The models' relative rotations disagree with the prior by a median 2.0–3.3° on GSV and
    2.5–3.0° on Mapillary.
  - A gross failure (> 10°) hits 11% (MapAnything, intrinsics only), 16% (MASt3R), 20%
    (VGGT corner) and 38% (COLMAP) of GSV pairs.
  - MapAnything given the priors keeps them, to a median 0.3°.
  - Every prediction row carries these diagnostics as `rel_rot_vs_prior_deg` and
    `baseline_dir_vs_prior_deg`.
  - The posed MapAnything rows also carry `src_pose_shift_m` and `oth_pose_shift_m`. **They
    are misnamed and are not pose shifts.** They compare the model's output camera centre,
    in an output frame re-centred on the first view, with the ENU prior centre, in a
    different frame. `src_pose_shift_m` is therefore about the camera height (1.99–2.61 m)
    and `oth_pose_shift_m` runs 1.7–52.5 m. The names are kept because they are in the
    committed rows; do not read them as pose error.
- **The depth comes from the second view.** The same model, given only the source view, is
  worse than flat ground. Its depth only helps when it has the other view to triangulate
  against. So a monocular metric-depth model is not a substitute here; the monocular-depth
  family's own arms test that directly.
- **The full readings beat depth-only by about 0.1–0.5°.** The model's own camera is
  consistent with its own depth, and a small shared error partly cancels.
- **Multi-view context did not help.**
  - The pair beat the corner for MapAnything: `mapa_posed_pair` 2.80° against
    `mapa_posed_corner` 2.86°, and `mapa_k_pair` 2.99° against `mapa_k_corner` 3.16°.
  - VGGT's corner beat its pair only slightly on GSV once fallbacks are counted.
  - Up to 12 views spanning 25 m and several capture dates is the wide-angle regime "When
    Wider Views Fail" describes.

### SfM, in detail

| | GSV (240) | Mapillary (60) |
|---|---|---|
| source and other view in one model (no priors / priors) | 90% / 88% | 100% / 100% |
| of all pairs: relative pose within 10°, baseline within 20° of the prior | 55% / 55% | 92% / 88% |
| click placed (≥ 3 ground points near it, in front of the other camera) | 63% / 58% | 100% / 98% |
| median views; share registered; COLMAP time per pair | 8; 100%; 2.9 s | 20; 100%; 15.9 s |
| placed: arm vs auto; paired gain over auto (no priors) | 6.30° vs 4.28°; −0.92 [−2.62, −0.25] | 2.52° vs 4.56°; 1.70 [0.25, 2.81] |

- **"Registered" is not "right".** COLMAP registers nearly every view. On GSV it often does
  so into a wrong model, because repetitive street furniture and opposite-side views give
  consistent-looking but wrong two-view geometry. 38% of GSV co-registrations are more than
  10° off the prior.
- **Position priors did not fix it.** COLMAP's priors constrain camera centres, not rotation,
  and a wrong model's rotation stays wrong (36% > 10° with priors).
- **The ground at the click is usually texture-poor asphalt or concrete.** 58 pairs (no
  priors) have fewer than 3 reconstructed points below the horizon within 120 px (9°) of the
  click.
- **On Mapillary the sequences are dense** (median 20 captures per corner, seconds apart),
  and there SfM behaves: every pair is placed, and it helps.

## 7. Sparse-view Gaussian splatting: stopped at the pilot

InstantSplat (NVlabs, `b951567`, unmodified) ran with the settings of its
`scripts/run_infer.sh`, on the corners of the pilot pairs (5 corners, one per city, 8 pairs).
`_splat_run.py` did the following:

- fed it the source view, every pair's other view and the nearest captures (5–9 views);
- ran its MASt3R initialisation (all view pairs, global alignment, shared focal,
  co-visibility downsampling);
- jointly trained 3D Gaussians and camera poses for 1,000 iterations;
- rendered expected depth at the source click, with each Gaussian coloured by its
  camera-space depth and normalised by accumulated opacity.

`splat_instantsplat` projects the click with the optimised cameras. `splat_instantsplat_init`
does the same from the Gaussians and poses saved after the first iteration, which is
essentially MASt3R's multi-view global alignment. The per-corner outputs are committed as
`mv3d_splat_pilot_<ramp>.json`.

| corner | views | train-view PSNR, init → trained | click depth, init → trained | focal (true 667 px) | time |
|---|---|---|---|---|---|
| bend:110 | 9 | 13.7 → 25.3 dB | ×1.09 | 636 | 202 s |
| gainesville:215 | 8 | 15.5 → 25.5 dB | ×1.23 | 646 | 147 s |
| paterson:114 | 6 | 17.2 → 27.7 dB | ×0.94 | 648 | 116 s |
| richmond:180 | 5 | 18.1 → 32.6 dB | ×0.96 | 663 | 119 s |
| sao_paulo:175 | 8 | 13.5 → 25.8 dB | ×2.02 | 653 | 169 s |

Per pair, error to the reference in degrees:

| pair | trained splat | splat init | `mast3r_pair` | `mapa_posed_pair` | auto |
|---|---|---|---|---|---|
| p000 | 5.10 | 3.01 | 1.16 | 1.66 | 4.77 |
| p001 | 5.79 | 4.04 | 1.00 | 0.73 | 3.50 |
| p060 | 4.40 | 5.04 | 9.37 | 8.59 | 6.25 |
| p061 | 3.34 | 3.76 | fallback | 8.88 | 5.39 |
| p120 | 3.32 | 6.44 | fallback | 3.30 | 2.76 |
| p180 | 0.81 | 1.44 | 1.38 | 0.77 | 1.53 |
| p240 | fallback | fallback | 5.18 | 8.93 | 24.33 |
| p241 | 46.36 | 61.17 | 2.40 | 3.39 | 9.22 |

- **It overfits.** Training-view PSNR roughly doubles in dB, while the click's rendered
  depth moves by as much as 2× (sao_paulo:175). Five to nine views spanning 10–30 m are
  far too few for photometric optimisation to constrain the ground at one pixel.
- **Training helps relative to its own initialisation on 5 of 7 pairs** (median +0.6°), but
  the result is not better than plain pairwise MASt3R. Where both answer, the splat is closer
  on 2 of 5 pairs, with a median paired gain of −3.9°. On these pairs MapAnything is at 3.30°
  median against the splat's 4.40°.
- **It also costs far more:** about 2–3.5 minutes per corner against 0.35 s per pair.
- **Caveats.**
  - InstantSplat estimates its own focal, 0.6–4.6% short of the known one. The known-K
    variant was not tried, to keep the method unmodified.
  - 8 pairs is a pilot, not a measurement.
- Per the brief ("if it only overfits the input views, stop and write that up"), FSGS and
  DNGaussian were **not run**.

## 8. The reference, again

Where the arms land far from the reference, they often land close to *each other*.
`_mv3d.py agreement` uses 1.5° for "agree" and 5° for "far", both set before looking:

| models (independent) | both placed | agree ≤ 1.5° | of those, both > 5° from the reference | reference error when they agree / disagree | auto on the same pairs |
|---|---|---|---|---|---|
| MapAnything (pair, K) and MASt3R | 267 | 64% | 15% | 1.86° / 4.08° | 3.48° / 5.44° |
| MapAnything (posed pair) and VGGT (corner) | 278 | 39% | 16% | 1.92° / 2.78° | 2.71° / 5.52° |
| MapAnything (posed corner) and LightGlue (`lg`) | 89 | 47% | 14% | 1.35° / 1.78° | 2.35° / 4.05° |

- **When two independent reconstructions agree, the error to the reference is 1.35–1.92°.**
  That is at the reference-noise floor: manual_gold peaks sit a median 1.51° from the human
  box centre (`reference_noise.json`). Agreement between an image-matching arm and a 3D arm,
  two unrelated techniques, gives the same picture.
- **About 15% of agreeing pairs sit more than 5° from the reference.** A handful of these
  were rendered locally and not committed (p004, p181–p183, p242). In them, the source click
  is on one part of an extended ramp or corner (e.g. at a sign pole's base beside a ramp),
  while the other view's detection peak is on another part 1–3 m away.
  - This is reference noise that grows with baseline, and no transfer can remove it.
  - It bounds what any arm can show on this set.
  - It is not evidence that the 3D point is right.

## 9. What this does not show

- **No negatives.** As for every arm on this harness (`docs/crossview_align_48.md` §7), every
  pair is a true match. The agreement signal in §8 is promising for association
  ([labeler#56](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/56)), but it
  is untested on "a different ramp nearby".
- **Only pairs the 2.6 m world test admits.** The worst projection failures are not scored.
  Gains over the projection are therefore conservative; gains over auto are too, less so.
- **The consensus arms are post hoc.** The 1.5° threshold was chosen before the agreement
  table was computed, but the composite itself was defined after both inputs were scored.
- **Pilot sample.** §3: 30 pairs from 15 ramps misranked every feed-forward arm. The splat
  verdict (§7) rests on 8 pairs and on the overfitting signature, not on a significance test.
- **One resolution per model, no tuning.** Models ran at their default input size on the
  harness's 75° views, with no confidence threshold, no masking and no test-time
  augmentation. Wider views (more context) and native-resolution inputs were not tried.
- **Not run:** π³, MUSt3R, VGGT-Omega, Depth Anything 3's multi-view mode, PanoVGGT (the
  equirect-native model), FSGS and DNGaussian. MapAnything on all of a corner's views (it
  was capped at 12) was not run either; the pair already beat the 12-view corner.
- **The views and panos are not published**, which is the same blocker as the harness's
  matching arms (§11).

## 10. 3D viewer bundles

Six corners are exported: the 4 where `mapa_posed_pair` is most accurate *and* beats auto on
both of the corner's pairs (richmond:99, richmond:191, bend:191, richmond:180), and the 2
where it is worst (paterson:92, bend:7). They were chosen by `_scenes.py pick`, which reads
the reference because it is choosing what to look at.

The JSON is committed under `analysis_out/crossview_align_48/scenes/<ramp_uid with : → _>/`.
The scene files are on makelab2 under `/homes/gws/jonf/crossview48/scenes/`, pinned by
sha256 in `scenes/scenes_manifest.json`.

- **`scene.ply`**: MapAnything's metric point cloud of the corner, from the
  `mapa_posed_corner` run (up to 12 views, given intrinsics and pose priors). The lowest 30%
  confidence is dropped, then 400,000 points are sampled. Binary, xyz float32 + rgb uint8,
  6 MB each.
- **`splat.ply`** (richmond:180 only): the InstantSplat pilot splat.
  - It keeps the 100,000 most opaque of 971,414 Gaussians, in the standard 3DGS layout
    (25 MB).
  - It is carried into the corner frame by a similarity fitted to the cameras: rotation
    from their orientations, then scale and translation from their centres. The camera-centre
    residuals are 0.14–0.43 m.
  - It is the overfit splat of §7, and its SH bands beyond DC are not rotated.
- **`cameras.json`**: every capture of the corner, with its pano id, role
  (`src` / `oth` / `extra`), `is_source`, the view file, the 1024 × 768 intrinsics, and the
  camera-to-world **prior** pose. For the views MapAnything saw, its output pose is included
  too.
- **`points.json`** contains:
  - the GT click: its ray, MapAnything's 3D point and the flat-ground point;
  - per pair, the reference (the other view's RampNet detection) as a ray from the other
    camera;
  - per pair, the projection and every committed arm's predicted point, each as a ray from
    the other camera with its error in degrees.
- **Frame.** East-north-up metres (x east, y north, z up). The origin is on the ground below
  the source camera (the labeler's LocalFrame at the source pano). z = 0 is flat ground, and
  each camera sits at its 'auto' height. Poses are camera-to-world 4 × 4 with OpenCV camera
  axes (x right, y down, z forward). Each JSON carries this line as `frame`.
- **Checked.** A top-down render of `bend_191` and `richmond_99` puts the cameras, the curb
  line and the click where the views show them. The median point height within 1 m of the
  click is 0.2 m and 0.8 m.

## 11. Cost, versions and reproduction

**Cost.** Everything ran on makelab2's A40 or its CPUs. The GPU is shared; free memory was
checked before each launch, and these runs used ≤ 12 GB beside other users' 9–16 GB.

- **No klone.** The family's GPU time is about 1.1 hours, and the environments and weights
  (about 20 GB) were already on makelab2. Duplicating them on klone's gscratch was not worth
  it.
- **No Tillicum and no paid API.**
- `paid: false` rows are in `analysis_out/usage_log.jsonl`.

| step | where | wall-clock | GPU-h (A40) | $ |
|---|---|---|---|---|
| manifest (`_mv3d.py manifest`) | desktop CPU | ~1 min | 0 | 0 |
| corner views (`render`, twice: 4- then 8-decimal manifest) | makelab2 CPU | 136 s + 252 s | 0 | 0 |
| pilots, 30 pairs (feed-forward arms, incl. first-use downloads) | makelab2 A40 | 16 min | 0.27 | 0 |
| pilots, 30 pairs (SfM arms, incl. superseded plane lift) | makelab2 CPU + A40 for LightGlue | ~16 min | (≤ 0.27, mostly idle) | 0 |
| full runs, feed-forward (10 model runs, 13 arms) | makelab2 A40 | 36.5 min | 0.61 | 0 |
| full runs, SfM (2 COLMAP runs, 3 arms) | makelab2 CPU + A40 for LightGlue | 54.6 min | (≤ 0.91, mostly idle) | 0 |
| InstantSplat pilot (5 corners) | makelab2 A40 | 12.6 min | 0.21 | 0 |
| viewer scenes (6 MapAnything runs) + splat export | makelab2 A40 / CPU | ~3 min | 0.03 | 0 |
| consensus arms, report, score, tests | desktop CPU | ~10 min | 0 | 0 |

The GPU-bound total is about 1.1 A40-hours. The SfM runs held the GPU only for LightGlue and
were CPU-bound, so they are listed separately rather than counted as GPU-hours.

**Versions.** Both environments are scratch venvs on makelab2 (Python 3.12.13), not the
repo's env.

- Main env (`/homes/gws/jonf/crossview48_sfm/venv`):
  - torch 2.6.0+cu124, torchvision 0.21.0, numpy 2.5.2;
  - pycolmap 4.2.0, kornia 0.8.3, OpenCV **5.0.0** (`cv2.__version__` as recorded in every
    multi-view `predictions/*.meta.json`; an earlier version of this list said
    opencv-python-headless 4.10.0.84, which is not what ran; the exact wheel build was not
    recorded);
  - VGGT (facebookresearch/vggt `a288dd0`);
  - MapAnything (facebookresearch/map-anything `3d10cf7`, uniception 0.1.7);
  - MASt3R (`Nik-V9/mast3r@6b9f163`, the MapAnything-packaged fork);
  - DUSt3R (`naver/dust3r@bb9f9f5`), CroCo (`naver/croco@87244aa`).
- Weights, pinned in `ff3d.MODEL_REVISIONS`:
  - `facebook/VGGT-1B@860abec`;
  - `facebook/map-anything@a1d87e9`;
  - `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric@06e7259`;
  - `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt@61c5744`.
- Splat env (`venv_splat`): the same torch, plus InstantSplat `b951567` with its
  `simple-knn`, `diff-gaussian-rasterization` and `fused-ssim` built with CUDA 12.8 for
  sm_86. Its MASt3R checkpoint was loaded with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. Only
  a truncated sha256 was written down (`e28f91b4…6eb2`); the full hash is not recorded in
  any committed file, so it cannot verify a checkpoint. Record the full hash on the next
  build.
- None of these packages are in `requirements.txt` or `environment.yml`.
- **Setup script: `scripts/analysis/crossview_mv3d_setup.sh`** (`main`, `splat` or `both`).
  It was written after the runs from the versions above and the meta files. **It has not
  been rebuilt or verified:** the original envs were built by hand and their install
  commands were not recorded, so the install order and editable/`--no-deps` choices are a
  reconstruction. **Follow-up, not run:** rebuild from it on makelab2 and re-predict one
  arm (e.g. `mapa_posed_pair`) to check it reproduces the committed rows.

**Reproduction.** Every prediction's `.meta.json` records the host, GPU, wall-clock, config
and pairs hash. The `predict-many` metas also record the manifest hash.

```bash
# 1. corner manifest (desktop CPU; labeler checkout 39afcd4 and its runs/, as for `pairs`)
python scripts/analysis/crossview_arms/_mv3d.py manifest \
    --labeler-root ../sidewalk-auto-labeler --runs-root ../sidewalk-auto-labeler/runs
# 2. corner views (makelab2 CPU; the native-res archive)
python scripts/analysis/crossview_arms/_mv3d.py render \
    --archive-root /projects/makeabilitylab/sidewalk-auto-labeler/runs --out corner_views
# 3. arms (GPU; comma groups share one model run). VIEWS = the harness's cut-views output
for g in mapa_posed_corner,mapa_posed_depthonly vggt_corner,vggt_corner_poseonly \
         mast3r_pair,mast3r_poseonly mapa_k_corner mapa_posed_pair mapa_posed_poseonly \
         vggt_pair mapa_k_pair dust3r_pair mapa_mono_depthonly \
         sfm_colmap,sfm_poseonly sfm_colmap_prior; do
  python scripts/analysis/crossview_arms/_mv3d.py predict-many --arms $g \
      --views VIEWS --extra corner_views=corner_views; done
python scripts/analysis/crossview_align_48.py predict --arm mv3d_consensus
python scripts/analysis/crossview_align_48.py predict --arm mv3d_consensus_else_auto
# 4. tables (CPU, committed files only)
python scripts/analysis/crossview_arms/_mv3d.py report --arms ARMS      # -> mv3d_results.json
python scripts/analysis/crossview_align_48.py score --arms proj_height_auto,ARMS \
    --out analysis_out/crossview_align_48/mv3d_scores.json
python scripts/analysis/crossview_arms/_mv3d.py pilot-report
python scripts/analysis/crossview_arms/_mv3d.py agreement --arms mapa_k_pair,mast3r_pair
# 5. splat pilot (InstantSplat env), then its arms read the outputs
$SPLAT_PY scripts/analysis/crossview_arms/_splat_run.py --instantsplat INSTANTSPLAT \
    --views VIEWS --corner-views corner_views --work /tmp/splat --out splat_out \
    --ramps bend:110,richmond:180,paterson:114,gainesville:215,sao_paulo:175
python scripts/analysis/crossview_arms/_mv3d.py pilot --arm splat_instantsplat \
    --views VIEWS --extra splat_dir=splat_out
# 6. viewer bundles
python scripts/analysis/crossview_arms/_scenes.py pick --arm mapa_posed_pair
python scripts/analysis/crossview_arms/_scenes.py export --ramps RAMPS --views VIEWS \
    --extra corner_views=corner_views --out scenes
$SPLAT_PY scripts/analysis/crossview_arms/_scenes.py splat --splat-json splat_out/richmond_180.json \
    --splat-ply WORK/richmond_180/model/point_cloud/iteration_1000/point_cloud.ply --out scenes
python scripts/analysis/crossview_arms/_scenes.py points --scenes scenes
```

**Unpublished inputs and what would unblock them.**

- **The labeler runs** are needed to rebuild the manifest. The manifest itself is committed.
- **The harness views and the corner views** are cut from the labeler's native-res archive on
  makelab2. Publishing the 1,834 views (600 harness + 1,234 corner) with a sha256 manifest
  beside `projectsidewalk/rampnet-benchmark` would make every arm here replicable from a
  clean clone.
- **The viewer scene files** (36 MB of point clouds plus one 25 MB splat) could go in the same
  place.
- **The first pilots ran from an older manifest.** They used a 4-decimal manifest and views
  cut from it (0.036° centre rounding). Every full run used the committed 8-decimal manifest
  and views re-cut from it.
- **`results.json` has been regenerated.** On this family's own branch it was not, and the
  rederive test failed there. After the five families were merged,
  `crossview_align_48.py score` was re-run over all 83 arms (2026-09-29), and
  `tests/test_crossview_align_48.py::test_committed_results_rederive_from_committed_predictions`
  now passes.

**Content hashes** (sha256):

- `mv3d_corners.json`: `92d25502f7d8bdeed83c4ea727a2ee3250e3485a22c74435ffeb5fdd9e53e5fc`;
- `mv3d_results.json`: `3ed69ba3172bc3b8cbee9d6e7cc1b8b1351288a9b0d6383b6180a5278f8dae9d`;
- `mv3d_scores.json`: `2e1183e97d5ff640a7c7c6b717ebcfc50230d443784373985a87e9eca45896ca`.
