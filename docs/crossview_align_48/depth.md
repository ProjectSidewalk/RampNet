# Cross-view placement with monocular metric depth ([#48](https://github.com/ProjectSidewalk/RampNet/issues/48))

One family of arms for the cross-view harness in [`docs/crossview_align_48.md`](../crossview_align_48.md):
the source click's **range** comes from a monocular metric depth model instead of flat ground.
Everything else is `proj_height_auto`: same bearing, and the other view is placed on flat
ground at its 'auto' height with the labeler's inverse. So each arm here differs from
`proj_height_auto` in the source range only, and the paired comparison against it isolates what
depth adds. Scored on the frozen 300 pairs (`pairs.csv` sha256 `a85a11bc…`). Run 2026-09-28/29
on free compute (makelab2 A40 for the models, desktop CPU for the arms).

Code: `scripts/analysis/crossview_depth_48.py` (GPU extract, summary, instrument check),
`scripts/analysis/crossview_arms/depth_mono.py` (the 17 arms). Outputs:
`analysis_out/crossview_align_48/depth/` (per-click depth rows per model, `summary.json`,
`results_depth.json`) and `analysis_out/crossview_align_48/predictions/mono_*.jsonl`.

## Summary

- **On GSV, no depth arm beats `proj_height_auto`.** The best is DA3 rescaled to the known
  camera height (`mono_da3_hcal`): paired gain vs auto **+0.22° [−0.03, 0.61]** over the 269 pairs
  it applies to, and +0.09° [−0.25, 0.48] on GSV alone. Every other GSV gain vs auto is within
  ±0.2° of zero or negative. The labeler's per-rig 'auto' height already does what depth would.
- **On Mapillary (Richmond, 60 pairs, 31 ramps), raw depth from UniDepth v2 or DA3 beats the
  projection with CIs clear of zero.** 'auto' keeps Mapillary at 2.6 m, so there auto = projection
  (4.56°). `mono_unidepth_point`: **3.44° [2.72, 4.56]**, paired gain **0.86° [0.47, 1.31]**, closer
  than the projection on 72% of pairs, within 2° 0.25 vs 0.10, never falls back.
  `mono_da3_point`: 4.05°, gain 0.50° [0.09, 1.15]. **One city, and chosen from 17 arms after the
  fact** — read it as a lead to replicate, not a result to ship (see Caveats).
- **Depth Pro and Metric3D read range short on these views and lose badly raw.** At the GSV
  clicks, raw range / Google's depth-map range is **0.54** (Depth Pro) and **0.90** (Metric3D),
  against 1.12 (DA3), 1.18 (UniDepth) and 1.07 (flat ground at 'auto'). Raw Depth Pro is 19.2° from
  the reference, raw Metric3D 7.8°. Rescaling to the known camera height (c) repairs Metric3D to a
  tie with auto (−0.09° [−0.35, 0.09]) but not Depth Pro (−2.85° [−4.35, −1.35]).
- **The local ground plane (b) is the same measurement as the point depth (a) here.** Plane range
  / point range has median 1.00 and p10–p90 within ±2.5% for all four models; the neighbourhood of
  a ramp click is locally planar and the model's depth at the click already sits on it. (b) is not
  a stabiliser on this set.
- **Matching Google's depth frame makes placement worse.** DA3 divided by #101's k_point = 1.106
  (`mono_da3_point_k101`) agrees best with Google at the click (ratio 1.008, 69% within 10%) and
  places worst of the DA3 arms (4.94°, gain vs auto −0.73° [−1.28, −0.34]). The same pattern as
  the harness's `proj_gsv_depth`. The arms that place well read range 7–18% longer than Google,
  which is the size of the labeler's per-city Google depth-frame factors (1.06–1.16,
  `docs/da3_calibration_101.md` §2.2). Consistent with Google's frame running short; not proof.
- **Verdict (proposed, not decided):** keep `proj_height_auto` as the GSV base. For Mapillary,
  UniDepth-v2 range is the one depth candidate worth a replication on a second Mapillary city
  before it is used; it costs ~3 s of A40 per pano. Drop Depth Pro and Metric3D for this job, and
  drop the local-plane variant.

## 1. Method

**Extract (GPU, `crossview_depth_48.py extract`).** Per unique source click (174 clicks on 144
source panos; answer columns dropped before anything else):

- The source pano is downscaled to 4096 px wide and rendered into #101's six perspective views
  (`equirect_tiling.default_views()`: 90° FOV, pitch −30°, 1024 px, yaws every 60°, focal 512 px).
  Each model runs once per view with the exact intrinsics where it accepts them. Its output is read
  as planar (z) depth and resized to 504 × 504 so every model is sampled on the same grid.
- **(a) point:** 7×7 median at the click, in the view where it is most central, converted to
  horizontal range (#101's `best_view` / `_sample` / `ray_from_value` / `horizontal_range`).
- **(b) local plane:** the view's 3-D points within 2.5 m (horizontal) of the click's own 3-D point
  and ≥ 2° below the horizon; #101's RANSAC plane (normal within 20° of vertical, 0.10 m inliers,
  least-squares refit); passes at ≥ 50 points with ≥ 30% on the plane; click ray intersected with it.
- **(c) camera height:** #101's ring ground fit (road band 20–45° below the horizon, lowest plane
  holding ≥ 15%, passing at ≥ 25% of ≥ 200 points). The arm rescales (a) or (b) by
  (labeler 'auto' height of the source pano) / (model's fitted height).

All constants were fixed before any arm was scored. **Arms** (`depth_mono.py`): per model
`mono_<m>_point` (a), `_plane` (b), `_hcal` (a×c), `_plane_hcal` (b×c), plus
`mono_da3_point_k101`. The range sets an effective height `d·tan(depression)` for the labeler's
raycast (the `proj_gsv_depth` construction); the world point is placed in the other view by
`geometry.inverse` at the 'auto' height. A range outside 0.5–60 m, a failed plane fit (b), or a
failed ring fit (c) falls back to the projection, as the harness requires.

**Models** (code commits and weights revisions are in each `depth/<m>.meta.json` and in the usage
rows):

| model | code | weights | notes |
|---|---|---|---|
| Depth Anything 3 | ByteDance-Seed/Depth-Anything-3 `3d835ec` | `depth-anything/DA3METRIC-LARGE` rev `4010e39f` | #101's pins; its extract code reused |
| Depth Pro | apple/ml-depth-pro `9e65e4d` | `apple/DepthPro` `depth_pro.pt` (rev `ccd1350a`) | fp16, `f_px = 512` passed. Its own focal estimate on the first 12 views is 497–667 px, so the short range is not a focal problem |
| UniDepth v2 | lpiccinelli-eth/UniDepth `8d8cfe4` | `lpiccinelli/unidepth-v2-vitl14` (rev `52b349b5`) | intrinsics passed as a `Pinhole` camera |
| Metric3D v2 | YvanYin/Metric3D `eb5b6fa` | `JUGGHM/Metric3D` `metric_depth_vit_large_800k.pth` (rev `80d2d141`) | 616 × 1064 input, canonical focal 1000 rescaled; `mmcv` stubbed with `mmengine.Config` (Metric3D's own fallback) |

All four ran on the equirect's perspective views, not the equirect directly: none of the four
takes an equirect input in its released inference code.

**Instrument checks.**

- `crossview_depth_48.py flatcheck` feeds the arms a synthetic depth row whose range is the
  flat-ground range at the 'auto' height. The point and hcal arms then reproduce the committed
  `proj_height_auto` predictions to **≤ 0.0002°** over all 600 placements, with no fallback. So
  any difference from `proj_height_auto` below is the depth model's range.
- DA3 reproduces #101 on these clicks: raw DA3 / Google at the 143 GSV clicks is **1.115**
  against #101's pooled 1.106, and DA3's fitted GSV camera height median is 2.28 m against
  #101's 2.24 m.

## 2. Results

`proj_height_auto` itself: all pairs 4.06° [3.66, 4.56], within 2° 0.24. Paired gains are the
median per-pair reduction in error, with ramp-bootstrap CIs, exactly as the harness computes them
(`crossview_depth_48.py summarize`, `depth/summary.json`). "Applied" = pairs where the arm did not
fall back.

### All 300 pairs

| arm | fallback | median ° [CI] | applied n: arm vs auto °, paired gain vs auto [CI] | applied: paired gain vs projection [CI] | range / Google at the click, median (share within 10%) |
|---|---|---|---|---|---|
| proj_height_auto | 0 | 4.06 [3.66, 4.56] | – | 0.00 [0.00, 0.43] (all pairs) | 1.074 (0.58) |
| mono_da3_point | 0 | 4.11 [3.61, 4.64] | 300: 4.11 vs 4.06, +0.07 [−0.14, 0.29] | 0.32 [0.04, 0.82] | 1.115 (0.40) |
| mono_da3_plane | 0.00 | 4.10 [3.60, 4.71] | 299: 4.10 vs 4.06, +0.10 [−0.21, 0.34] | 0.40 [0.10, 0.94] | 1.120 (0.40) |
| mono_da3_hcal | 0.10 | 3.75 [3.24, 4.57] | 269: 4.04 vs 4.19, **+0.22 [−0.03, 0.61]** | 0.84 [0.26, 1.37] | 1.137 (0.29) |
| mono_da3_plane_hcal | 0.11 | 3.91 [3.12, 4.46] | 268: 4.04 vs 4.21, +0.12 [−0.05, 0.59] | 0.84 [0.30, 1.32] | 1.133 (0.28) |
| mono_da3_point_k101 | 0 | 4.94 [4.29, 5.75] | 300: 4.94 vs 4.06, −0.73 [−1.28, −0.34] | −0.42 [−0.68, 0.29] | **1.008 (0.69)** |
| mono_unidepth_point | 0 | 4.04 [3.48, 4.73] | 300: 4.04 vs 4.06, +0.18 [−0.15, 0.47] | 0.54 [0.32, 1.04] | 1.179 (0.17) |
| mono_unidepth_plane | 0.01 | 3.94 [3.38, 4.59] | 298: 4.00 vs 4.08, +0.29 [−0.06, 0.56] | 0.64 [0.39, 1.12] | 1.180 (0.17) |
| mono_unidepth_hcal | 0.14 | 4.01 [3.49, 4.54] | 258: 3.77 vs 3.95, −0.01 [−0.25, 0.14] | 0.29 [−0.06, 0.61] | 1.072 (0.60) |
| mono_unidepth_plane_hcal | 0.15 | 4.02 [3.48, 4.47] | 256: 3.71 vs 4.00, −0.02 [−0.22, 0.10] | 0.30 [−0.06, 0.59] | 1.068 (0.63) |
| mono_metric3d_point | 0 | 7.81 [7.01, 8.99] | 300: 7.81 vs 4.06, −3.19 [−3.70, −2.67] | −2.52 [−3.66, −2.04] | 0.899 (0.45) |
| mono_metric3d_plane | 0.00 | 8.02 [6.89, 9.20] | 299: 8.03 vs 4.06, −3.26 [−3.85, −2.85] | −2.90 [−3.68, −2.15] | 0.891 (0.42) |
| mono_metric3d_hcal | 0.15 | 4.24 [3.62, 5.12] | 256: 4.02 vs 3.97, −0.09 [−0.35, 0.09] | 0.24 [−0.24, 0.84] | 1.090 (0.50) |
| mono_metric3d_plane_hcal | 0.15 | 4.32 [3.70, 4.99] | 255: 4.16 vs 3.97, −0.11 [−0.38, 0.11] | 0.27 [−0.30, 0.65] | 1.081 (0.54) |
| mono_depthpro_point | 0 | 19.23 [17.57, 22.51] | 300: 19.23 vs 4.06, −14.43 [−16.76, −12.72] | −13.21 [−15.13, −11.31] | 0.541 (0.05) |
| mono_depthpro_plane | 0 | 19.30 [17.42, 22.41] | 300: 19.30 vs 4.06, −14.49 [−16.74, −12.56] | −13.02 [−15.36, −10.84] | 0.537 (0.05) |
| mono_depthpro_hcal | 0.31 | 7.17 [6.01, 8.63] | 208: 8.12 vs 4.17, −2.85 [−4.35, −1.35] | −2.22 [−3.64, −0.66] | 0.843 (0.29) |
| mono_depthpro_plane_hcal | 0.31 | 7.37 [6.03, 8.59] | 208: 7.93 vs 4.17, −2.81 [−4.70, −1.64] | −2.11 [−3.63, −0.78] | 0.844 (0.28) |

- **The "median ° [CI]" column counts fallbacks at the 2.6 m projection**, as the harness does.
  That is why the hcal arms can show a lower all-pairs median than their applied-subset median. A
  composite that falls back to `proj_height_auto` instead gains nothing measurable anywhere: the
  best, `mono_unidepth_plane`, is +0.27° [−0.05, 0.56] vs auto (`summary.json` →
  `composite_else_auto_vs_auto`).
- **The range check** is per unique GSV source click (up to 143 of 144 with a Google range),
  against `depth.ground_range_at` on the source pano's harvested Google depth payload. It is
  independent of the reference, but it is not ground truth: Google's frame is itself thought to
  run 6–16% short by city (§ Summary).
- **Fallbacks are ring-fit failures (c).** The ring ground fit passes on 129 / 102 / 122 / 121 of
  144 panos for DA3 / Depth Pro / UniDepth / Metric3D. The local plane (b) passes on 172–174 of 174
  clicks. The fitted camera heights (median, passing panos) are DA3 2.33 m, Depth Pro 1.45 m,
  UniDepth 2.61 m, Metric3D 1.94 m: Depth Pro reads the whole scene ~0.6× short, and its fit fails
  most often.

### By imagery

| arm | GSV (240): median ° [CI] | GSV applied: gain vs auto [CI] | Mapillary (60, 31 ramps): median ° [CI] | Mapillary: gain vs auto (= vs projection) [CI] | Mapillary within 2° |
|---|---|---|---|---|---|
| projection | 6.09 | – | 4.56 [3.87, 5.57] | – | 0.10 |
| proj_height_auto | 3.92 [3.53, 4.36] | (vs projection 0.67 [0.25, 1.20]) | 4.56 (unchanged) | 0 | 0.10 |
| mono_da3_point | 4.11 [3.57, 4.78] | −0.03 [−0.46, 0.16] | 4.05 [3.15, 5.20] | 0.50 [0.09, 1.15] | 0.22 |
| mono_da3_plane | 4.13 [3.53, 4.88] | −0.06 [−0.57, 0.18] | 4.05 [3.26, 5.14] | 0.52 [0.17, 1.04] | 0.22 |
| mono_da3_hcal | 3.58 [3.08, 4.56] | 0.09 [−0.25, 0.48] | 4.19 [3.24, 5.23] | 0.62 [0.14, 1.11] (55 applied) | 0.20 |
| mono_da3_point_k101 | 5.17 [4.42, 6.05] | −0.99 [−1.68, −0.47] | 3.93 [2.54, 5.84] | −0.04 [−0.69, 1.13] | 0.25 |
| mono_unidepth_point | 4.18 [3.59, 4.98] | −0.01 [−0.46, 0.37] | **3.44 [2.72, 4.56]** | **0.86 [0.47, 1.31]** | 0.25 |
| mono_unidepth_plane | 4.19 [3.34, 4.79] | 0.05 [−0.40, 0.46] | 3.41 [2.69, 4.52] | 0.78 [0.37, 1.27] | 0.25 |
| mono_unidepth_hcal | 3.97 [3.45, 4.61] | −0.07 [−0.29, 0.08] | 4.24 [2.88, 4.67] | 0.44 [−0.42, 1.20] (52) | 0.18 |
| mono_metric3d_plane_hcal | 4.44 [3.81, 5.03] | −0.19 [−0.60, 0.04] | 3.64 [2.48, 5.37] | 0.62 [−0.31, 1.41] (54) | 0.17 |
| mono_metric3d_point | 7.90 [6.53, 9.28] | −3.30 [−4.29, −2.66] | 7.39 [5.94, 9.73] | −2.76 [−4.91, −1.83] | 0.00 |
| mono_depthpro_hcal | 8.47 [6.96, 9.61] | −4.38 [−6.61, −2.96] | 3.93 [2.82, 4.58] | 0.13 [−0.53, 1.29] (43) | 0.18 |
| mono_depthpro_point | 20.81 [18.65, 24.66] | −15.39 [−18.54, −13.25] | 16.55 [13.87, 19.50] | −10.78 [−14.70, −8.42] | 0.00 |

Every arm's full strata (city, range bin, date) are in `depth/results_depth.json` (harness
`score`) and `depth/summary.json` (paired vs auto). By range, the only stratum where any depth arm
points at a gain vs auto is 0–6 m (`mono_da3_hcal` +1.61° [−0.33, 4.92], 18 pairs); the CI spans
zero and rests on 21 pairs.

For comparison, the harness's `lg` (ALIKED + LightGlue) reaches 3.05° on all 60 Mapillary pairs
and 2.10° on the 31 it aligns. Depth arms never fall back, so on Mapillary they would be the base
under `lg`, not a replacement for it; that combination is not tested here.

## 3. Reading

- **Why depth ties auto on GSV.** On GSV the 'auto' height is a good per-rig constant, and a flat
  road within 18 m is what the pairs contain. A monocular model's range at a single pixel has its
  own ~10–20% scatter (p10–p90 of range / Google spans ~0.98–1.29 for DA3), which is at least as
  large as the scatter flat ground at the right height has. Rescaling the model to the known height
  (c) removes its global scale error but inherits the per-pano ground-fit noise #101 already
  measured (DA3 heights do not track panos within a rig).
- **Why it helps on Mapillary.** There 'auto' has no height and no pose: Richmond is flat ground at
  2.6 m with pitch/roll left out. A depth model's range does not depend on the camera's height or
  tilt, so it can correct both at once. That is one plausible reading; it was not tested (for
  example, by checking the gain against the SfM pitch per pano).
- **The plane variant is redundant, not stabilising.** Point and plane ranges agree to ±2.5% at
  p10–p90 for every model, so the ground neighbourhood of a detected ramp is not where monocular
  depth is noisy on this set.
- **Scale conventions differ by a factor of two across models on these views.** DA3 and UniDepth
  land near the imagery's own geometry; Metric3D ~10% short and Depth Pro ~45% short of Google.
  Depth Pro's own focal estimate is close to the true 512 px, so this is not an intrinsics error
  on our side. It is a reason not to take any "metric" model's scale on trust on pitched-down
  wide-FOV views.

## 4. Caveats (beside the numbers above)

- **Richmond is one city and one rig** (NCTech iStar Pulsar per #101), 60 pairs on 31 ramps. The
  UniDepth and DA3 Mapillary gains were found by looking across 17 arms and two imagery strata;
  with that many looks, one CI-clear stratum could be chance. Against that, the direction agrees
  across DA3, UniDepth and (hcal) Metric3D. Replicate on a second Mapillary city before use.
- **The set is selected at 2.6 m** (harness §2): the reference detection must raycast within 5 m of
  the source point, both at 2.6 m. That leans toward pairs where 2.6 m geometry roughly works,
  which favours the projection on Mapillary and may favour ranges near flat-ground ranges generally.
- **Reference noise.** A median 1.5° per detection peak (harness §2); gains under ~1° are near that
  floor.
- **The range check is against Google, which is not truth** (see §2). The finding that
  Google-matched DA3 places worse is consistent with Google's frame running short, and equally
  consistent with a selection effect of the 2.6 m test.
- **Other view unchanged.** Only the source range moves; the other view is still flat ground at
  'auto'. Depth on the other view (a 3-D point matched to a 3-D point) is not tested.

## 5. Not run, and why

- **Equirect-native inference:** none of the four models' released inference takes an equirect;
  all ran on six perspective views, the #101 rendering.
- **Other depth models** (e.g. MoGe, Depth Anything v2 metric): not run; the brief named four.
  #101 notes Depth-Anything-V2 metric reads ~3× long on these views.
- **A click-centred view** (pitch toward the click rather than the fixed −30° ring): not run.
  The click sits 7–24° below the horizon (p10–p90; median 12°), inside the ring views, which span
  +15° to −75° of elevation at the view centre; a near-horizon click sits in the upper third.

## 6. Cost

| step | where | wall-clock | GPU-h | $ |
|---|---|---|---|---|
| extract DA3 (144 panos) | makelab2 A40 (shared, ~9 GB used by another job) | 357 s + 8 s load | 0.10 | 0 |
| extract Depth Pro | makelab2 A40 | 547 s + 154 s load (first weights download) | 0.19 | 0 |
| extract UniDepth v2 | makelab2 A40 | 419 s + 18 s load | 0.12 | 0 |
| extract Metric3D v2 | makelab2 A40 | 948 s + 26 s load | 0.27 | 0 |
| smoke tests (2 panos per model, twice) | makelab2 A40 | ~2 min | ≤ 0.04 | 0 |
| 17 arms `predict` + `summarize` + `score` | desktop CPU | ~2 min + ~1 min + ~4 min | 0 | 0 |

Total ≈ 0.72 GPU-hours, $0. Wall-clock includes decoding the native-res JPEGs (the model-only
seconds are 138 / 354 / 233 / 778 s in each `depth/<m>.meta.json`). The four extract runs are
`paid: false` rows in `analysis_out/usage_log.jsonl` (labels `crossview-depth-48:extract-<m>`);
`gpu_hours` there is wall-clock including model load, on a GPU shared with another job.

## 7. Reproduction

```bash
# makelab2: scratch env (uv venv over .venv-eval's site-packages; never touches .venv-eval),
# model code clones, pinned DA3 commit, the mmcv stub. Package versions: depth/freeze_scratch.txt
bash scripts/analysis/crossview_depth_48_setup.sh
# GPU extract, all four models (~40 min on an A40); writes <R>/out/<model>.jsonl + meta + usage rows
bash scripts/analysis/crossview_depth_48_run.sh "da3 depthpro unidepth metric3d"
# copy out/*.jsonl, *.meta.json into analysis_out/crossview_align_48/depth/

# desktop CPU: the arms (labeler inputs as for the harness's geometry arms, §9 there)
for a in $(python scripts/analysis/crossview_align_48.py arms | awk '/^mono_/{print $1}'); do
  python scripts/analysis/crossview_align_48.py predict --arm $a \
      --labeler-root ../sidewalk-auto-labeler --runs-root ../sidewalk-auto-labeler/runs \
      --results-root runs_archive; done
python scripts/analysis/crossview_depth_48.py flatcheck --labeler-root ../sidewalk-auto-labeler \
    --runs-root ../sidewalk-auto-labeler/runs --results-root runs_archive
python scripts/analysis/crossview_depth_48.py summarize        # committed inputs only
python scripts/analysis/crossview_align_48.py score --arms <mono_*>,proj_height_auto \
    --out analysis_out/crossview_align_48/depth/results_depth.json
```

**Unpublished inputs and what would unblock them.** The GPU extract needs the 144 native-res
source panos, which live only in the labeler archive on makelab2 (the same gap as the harness's
views; publishing the panos in `pairs.csv` with a sha256 manifest would close both; each depth
row records its pano's sha256). The arms need the labeler runs and, for the range check, the GSV
depth payloads (harness §9). `summarize` and `score` read only committed files. The depth rows
(`depth/<m>.jsonl`) are committed, so every table here re-derives on CPU from the repo plus the
labeler inputs.
