# Placing a GT ramp point in other panoramas ([#48](https://github.com/ProjectSidewalk/RampNet/issues/48)): alignment harness and first arms

A pilot, and the shared harness for comparing techniques. Code:
`scripts/analysis/crossview_align_48.py` (harness) and `scripts/analysis/crossview_arms/`
(the techniques, called "arms"). Outputs: `analysis_out/crossview_align_48/`. Tests:
`tests/test_crossview_align_48.py`. Run 2026-09-28 on free compute only.

## Summary

- **Question.** A GT ramp point is carried from its source pano into another capture of the
  same ramp. Can we place it closer to the ramp than today's flat-ground projection? That
  projection raycasts at 2.6 m and places the point with the labeler's
  `geo.ground_point_to_pano`.
- **Known-answer set:** 300 frozen (ramp, other view) pairs, 60 per city, 174 ramps (§2).
  The reference is the other view's own RampNet detection of the ramp.
  `pairs.csv` sha256 `a85a11bc…`.
- **Today's projection** is a median **5.62° [4.53, 6.56]** from the reference (66 px on a
  4096-wide pano). Only 19% of pairs are within 2°.
- **Best free fix: the labeler's own 'auto' camera height.** This is GSV per capture-year rig
  heights of 2.0 / 2.5 m instead of 2.6 (`proj_height_auto`). On GSV pairs it cuts the median
  from 6.09° to **3.92° [3.53, 4.36]**, with a paired median gain of 0.67° [0.25, 1.20]; the
  share within 2° goes from 0.21 to 0.28. It never falls back and costs nothing. Richmond
  (Mapillary) is unchanged, because 'auto' keeps Mapillary at 2.6 m.
- **Image alignment (ALIKED + LightGlue, `lg`) wins where it works, and it mostly does not
  work.** It aligns 30% of pairs. On those it cuts the error from 4.40° to **2.41°
  [1.91, 2.87]** (paired gain 0.88° [0.35, 2.12]). The other 70% fall back to the projection.
  It is the only arm that helps Richmond: 3.05° vs 4.56° over all 60 Mapillary pairs, and
  2.10° vs 4.54° on the 31 it aligns.
- **These did not help:**
  - Mapillary SfM pitch/roll (`proj_mly_gravity`, `proj_mly_road`): no gain.
  - Per-pano measured GSV heights (`proj_height_perpano`) and GSV depth-map range
    (`proj_gsv_depth`): lower medians, but more pairs got worse than better. Their paired
    gains are ≤ 0, and the share within 2° drops.
  - SIFT and NCC baselines: worse than projection.
- **The Richmond run already uses SfM.** Its positions are Mapillary's SfM
  `computed_geometry` and its headings are `computed_compass_angle`. Raw GPS makes it worse
  (`proj_mly_rawgps`, paired gain −0.52° [−1.36, −0.06]).
- **Verdict (proposed, not decided):**
  - For gallery rings, use `proj_height_auto` everywhere. Where LightGlue aligns, use its
    point instead, and mark which ring is which.
  - As an association signal for the labeler's clustering
    ([labeler#56](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/56)),
    image alignment is too often silent to be the primary signal, and it is untested as a
    discriminator (§7).
- **The harness is pluggable.** A new technique is one function plus one `@register` line
  in a new file under `crossview_arms/`. It is scored on the same frozen pairs with the same
  metrics (§5).

## 1. Why

`multiview_evidence_48.py` projects a world GT point into other captures. Three things move
the projected point:

- GT placement error, p50 1.9 m / p90 4.4 m (labeler `docs/reprojection-residual.md`);
- camera height: the 2.6 m constant is high for the 2025–26 GSV rig and for Mapillary rigs;
- pose error, which has not been measured.

At 12–18 m the projected point can land beside the ramp. Jon saw this on richmond:3 in the
residual gallery, and he proposed aligning the images themselves. The geometry-prior arms
were added to test whether a better camera model gets the same result for free.

## 2. Known-answer set (frozen)

Ramps where two views **each** have a verdict-true detection of one merged ramp barely exist:
1 ramp, in bend. The reference is therefore a RampNet detection.

- **Source point:** a verdict-true operational (≥ 0.55) detection that is a GT point of a
  world pool ramp (eval_sites' pool, same inputs as `multiview_evidence_48.py run`). The
  world position is that one view's own raycast at 2.6 m.
- **Other view:** a non-source run pano with its camera within 18 m of that position, and in
  which a ≥ 0.55 detection claims the ramp by the world test: raycast within 5 m, one-to-one
  claims in confidence order, as in `capture_table`. **That detection's pixel is the
  reference.**
- **Ambiguity filters:** drop a ramp with another pool ramp within 6 m. Drop a pair whose
  other view has a second ≥ 0.55 detection landing within 8 m.
- **Sample:** 1,723 eligible pairs, 60 sampled per city with seed 48, at most 2 other views
  per ramp. `eligible_pairs.csv` re-derives `pairs.csv` (tested).
- **Frozen:** `PAIRS_SHA256` in the script pins `pairs.csv`. `pairs` refuses to overwrite it
  without `--force`, and `predict` / `score` refuse any other bytes.

| city | imagery | ramps considered | other views within 18 m | claimed | dropped ambiguous | eligible |
|---|---|---|---|---|---|---|
| richmond | Mapillary | 172 | 1,744 | 956 | 172 | 784 |
| paterson | GSV | 106 | 459 | 268 | 42 | 226 |
| gainesville | GSV | 144 | 599 | 225 | 28 | 197 |
| bend | GSV | 176 | 678 | 487 | 154 | 333 |
| sao_paulo | GSV | 91 | 416 | 231 | 48 | 183 |

The sampled pairs:

- **by range:** 21 / 158 / 121 pairs at 0–6 / 6–12 / 12–18 m;
- **by capture date:** 191 same-month and 109 different-month;
- **camera baseline:** p10 7.8 m, median 13.0 m, p90 26.2 m.

**Caveats beside the set:**

- **The world test truncates the projection's error.** A pair qualifies only if the
  reference detection raycasts within 5 m of the source point: median gap 2.2 m, p90 4.2 m.
  That test raycasts at the same flat 2.6 m. So the set leans toward pairs where 2.6 m
  geometry already roughly works, and the projection numbers are optimistic. That makes
  every arm's gain over projection conservative, and more so for the geometry arms.
- **Reference noise.** The reference and the source point are both detection peaks. On
  manual_gold, peaks at ≥ 0.55 sit a median **1.51°** (p90 3.04°, n = 3,420) from the human
  box centre they match (`reference_noise.json`). That is a rough scale, not a bound. With a
  peak at each end of a pair, errors of about 2° are at the floor this set can resolve.
- **A claimed detection can be a different, unlabeled ramp within 5 m.** The ambiguity
  filters reduce this but cannot rule it out.

## 3. Metrics

For each pair, the arm's point is compared with the reference:

- **angular error:** the great-circle angle, in degrees;
- **pixel error:** the seam-wrapped distance on a 4096 × 2048 equirect;
- **hit rates:** the share within 2° and within the benchmark's 0.022 radius.

An arm that returns nothing falls back to the projection, and that counts toward its
fallback rate. The **paired gain** is the median per-pair reduction in error against the
projection. CIs are 2.5–97.5 percentiles over 2,000 bootstrap resamples of **ramps**. Results
are stratified by imagery, city, range bin and same- vs different-month.

A ring in the residual gallery is about 1.4° in radius (14 px in a 36° crop), so errors of
several degrees are visible.

## 4. Arms and results

### The arms

**Geometry priors** (`crossview_arms/geometry.py`). These re-do both halves of the projection
with the labeler's own code; only the camera model changes.

| arm | what changes | applies to |
|---|---|---|
| `proj_flat_check` | nothing: the instrument check. It reproduces `proj_x`/`proj_y` to ≤ 0.004° | all |
| `proj_height_auto` | camera height from the labeler's default `auto` resolver: GSV per capture-year rig 2.0 or 2.5 m, Mapillary 2.6 m (`fuse_sites.load_at_height`, labeler #79) | GSV (Mapillary stays 2.6) |
| `proj_height_perpano` | each pano's measured height from Google's depth planes (`per-pano`, through `depth.believe_height`), else 2.6 | GSV |
| `proj_gsv_depth` | source range read from the depth planes at the GT pixel (`depth.ground_range_at`); the other view is flat at its measured height | GSV |
| `proj_mly_gravity` | adds the SfM rotation's pitch/roll (`computed_rotation`) on both sides; the posed inverse is solved numerically against `detection_ground_point` | Mapillary |
| `proj_mly_road` | the same, relative to the sequence's SfM road grade (labeler `road` mode) | Mapillary |
| `proj_mly_rawgps` | contrast: raw GPS `geometry` in place of SfM `computed_geometry` | Mapillary |

**Image matching** (`crossview_arms/matching.py`). Each matches a rectilinear source view
centred on the GT point against the other view centred on the projection. Views are
1024×768 with a 75° horizontal FOV, cut from the native-res archive. Matches are kept only on
the ground, a RANSAC homography is fitted, and the GT point is mapped through it. The arm
falls back if there are fewer than 15 inliers or the mapped point leaves the view.

| arm | what |
|---|---|
| `lg` | ALIKED + LightGlue (kornia 0.8.3); ground band ≥ 5° below the horizon (post hoc, see below) |
| `lg_local` | the same, with the homography fitted only to matches within 160 px of the GT point |
| `lg_band0.5` | ground band 0.5°, the pre-specified setting |
| `sift` | OpenCV SIFT + ratio test, with the same band, RANSAC and fallback |
| `ncc` | a range-scaled 64 px template, normalized cross-correlation within ±200 px of the projection, falling back below a peak of 0.5 |

### All 300 pairs

| arm | median ° [CI] | median px | p90 ° | within 2° | fallback | where it did not fall back: n, arm vs projection °, paired gain [CI] |
|---|---|---|---|---|---|---|
| projection | 5.62 [4.53, 6.56] | 66 | 15.7 | 0.19 | 0 | - |
| proj_height_auto | **4.06 [3.66, 4.56]** | 47 | 13.9 | 0.24 | 0 | 300: 4.06 vs 5.62; 0.00 [0.00, 0.43] (Mapillary pairs are unchanged) |
| proj_height_perpano | 4.95 [4.48, 5.65] | 57 | 17.5 | 0.14 | 0 | 300: 4.95 vs 5.62; −0.00 [−0.00, 0.00] |
| proj_gsv_depth | 4.82 [4.50, 5.65] | 56 | 15.0 | 0.13 | 0.20 (Mapillary n/a) | 240: 4.96 vs 6.09; −0.51 [−1.14, 0.28] |
| proj_mly_gravity | 5.72 [4.53, 6.67] | 67 | 17.0 | 0.19 | 0.80 (GSV n/a) | 60: 4.68 vs 4.56; −0.31 [−1.26, 0.69] |
| proj_mly_road | 5.88 [4.53, 6.87] | 68 | 17.1 | 0.22 | 0.80 (GSV n/a) | 60: 4.81 vs 4.56; −0.21 [−1.45, 0.98] |
| proj_mly_rawgps | 6.03 [4.74, 6.95] | 69 | 17.5 | 0.21 | 0.80 (GSV n/a) | 60: 5.24 vs 4.56; −0.52 [−1.36, −0.06] |
| lg | 4.79 [3.51, 6.01] | 55 | 15.7 | 0.25 | 0.70 | 89: **2.41 vs 4.40; 0.88 [0.35, 2.12]** |
| lg_local | 4.84 [3.97, 6.15] | 57 | 15.7 | 0.21 | 0.78 | 67: 2.66 vs 4.45; 0.82 [0.30, 2.05] |
| lg_band0.5 | 5.57 [4.61, 6.78] | 65 | 17.6 | 0.22 | 0.63 | 111: 4.26 vs 4.51; −0.08 [−1.13, 0.63] |
| sift | 5.88 [4.56, 6.78] | 68 | 17.0 | 0.19 | 0.94 | 18: 9.91 vs 3.92; −3.79 [−16.66, 0.53] |
| ncc | 10.87 [9.74, 12.50] | 125 | 22.9 | 0.09 | 0.24 | 228: 12.42 vs 4.76; −6.05 [−7.48, −4.91] |

### By imagery

| arm | GSV (240) median ° [CI] | GSV paired gain [CI], within 2° | Mapillary (60) median ° [CI] | Mapillary paired gain [CI], within 2° |
|---|---|---|---|---|
| projection | 6.09 [4.53, 7.03] | –, 0.21 | 4.56 [3.87, 5.57] | –, 0.10 |
| proj_height_auto | **3.92 [3.53, 4.36]** | **0.67 [0.25, 1.20]**, 0.28 | unchanged | – |
| proj_height_perpano | 5.15 [4.45, 5.79] | −0.44 [−1.07, 0.00], 0.15 | unchanged | – |
| proj_gsv_depth | 4.96 [4.55, 5.89] | −0.51 [−1.14, 0.28], 0.14 | n/a | – |
| proj_mly_gravity | n/a | – | 4.68 [3.30, 6.12] | −0.31 [−1.26, 0.69], 0.12 |
| proj_mly_road | n/a | – | 4.81 [2.90, 7.39] | −0.21 [−1.45, 0.98], 0.25 |
| proj_mly_rawgps | n/a | – | 5.24 [4.10, 8.77] | −0.52 [−1.36, −0.06], 0.20 |
| lg (fallback 0.76 / 0.48) | 5.91 [4.15, 6.69] | aligned 58: 0.47 [0.01, 1.65], 0.23 | **3.05 [2.39, 4.33]** | aligned 31: **2.25 [0.88, 3.64]**, 0.30 |
| lg_local | 5.97 [4.42, 6.89] | aligned 44: 0.33 [−0.08, 0.88] | 3.38 [2.92, 4.04] | aligned 23: 2.96 [1.03, 3.65] |

By range and date, for the three arms that matter (median °; paired gain where not fallen
back):

| stratum | projection | proj_height_auto | proj_gsv_depth | lg (aligned n: aligned vs projection) |
|---|---|---|---|---|
| 0–6 m (21) | 12.67 | 9.54, gain 1.44 [−0.00, 4.42] | 9.30 (15 applied) | 5: 3.63 vs 10.64 |
| 6–12 m (158) | 6.87 | 4.68, gain 0.33 [0.00, 1.15] | 5.84 | 41: 2.62 vs 4.56 |
| 12–18 m (121) | 3.40 | 2.90, gain 0.00 [0.00, 0.20] | 3.60, gain −0.72 [−1.14, −0.11] | 43: 1.91 vs 3.68 |
| same month (191) | 5.97 | 3.90 | 4.72 | 69: 2.36 vs 4.45 |
| different month (109) | 4.84 | 4.19 | 5.16 | 20: 3.05 vs 3.50 |

Stratum CIs rest on as few as 5 pairs; read them as direction, not size. Every number is in
`results.json` → `arms.<arm>.<stratum>`.

## 5. Reading

- **Camera height is the cheapest lever, and the labeler already has it.** Moving GSV from
  2.6 m to its per-year rig height (2.0 or 2.5 m) lowers the projected point toward where
  the ramp really is. Most of the gain is at 0–12 m, where the depression angle is most
  sensitive to height. It is conservative here, because the pair set was selected at 2.6 m.
- **Per-pano measured heights and the depth map help the bad pairs and hurt the good ones.**
  Their medians fall (5.15° and 4.96° vs 6.09° on GSV), but more pairs got worse than
  better, and the share within 2° drops from 0.21 to 0.15 / 0.14. Per-pano heights are
  noisier than a per-rig constant.
  - `proj_gsv_depth` reads range at the exact GT pixel. There a detection peak sits on a
    ramp, curb or kerb face rather than on the road plane, and nearby occluders are
    common. It is worst at 12–18 m (gain −0.72° [−1.14, −0.11]).
  - A version that samples the ground plane around the pixel, or uses the other view's
    depth too, is not tested.
- **Mapillary pitch/roll does not help this placement.** Neither the gravity frame nor the
  road-relative frame moves the median or the paired gain; both CIs span zero. The SfM
  position is doing its job: raw GPS is measurably worse.
- **Image alignment is the only thing that helps Mapillary,** and it is where alignment
  succeeds most often (fallback 48% vs 76% on GSV). On the pairs it aligns it is near the
  reference-noise floor (§2). Most GSV pairs have no usable ground correspondence. The
  median pair has 13 m of baseline, and ramps are often seen from opposite sides of an
  intersection, across parked cars, or on bare asphalt.
- **Cheap image methods fail.** SIFT rarely finds a ground homography at these baselines,
  and NCC picks the wrong texture inside a ±16° window most of the time.

### Two things fixed during the run (recorded, not reported as results)

- **Keypoint truncation.** The first matching runs cut ALIKED's keypoints to 2,048. ALIKED
  returns them in raster order, so the slice dropped the lower half of busy views: the
  ground. Every committed number post-dates the fix.
- **Ground band.** The pre-specified ground band (0.5° below the horizon) let far-field
  points dominate the homography. That homography is close to a pure rotation and does not
  transfer to a ramp 5–18 m away. The 5° band was chosen after seeing those failures, so it
  is post hoc. Both are committed: `lg_band0.5` has no gain.
- **NCC window.** The NCC search was first run over the whole view (23° median) and was then
  windowed to the projection's uncertainty.
- **Informal probe, not a result:** on 30 pairs, DISK + LightGlue and LoFTR gave no more
  ground inliers than ALIKED + LightGlue. That probe is not committed and should not be
  quoted.

## 6. Adding an arm

1. Create a new file under `scripts/analysis/crossview_arms/`. A new file per technique
   keeps parallel work free of merge conflicts. In it, write one function and register it:

   ```python
   from crossview_arms._registry import register
   import crossview_align_48 as H          # view_to_pano, pano_to_view, geometry helpers

   @register("my_arm", needs=("views",), description="one line for the results table",
             config={"threshold": 0.5})
   def my_arm(pair, ctx):
       src = ctx.view(pair, "src")          # BGR 1024x768 view centred on the GT point
       oth = ctx.view(pair, "oth")          # ... and on today's projection
       ...
       return {"x": x_norm, "y": y_norm, "score": s}   # equirect coords in the OTHER pano
       # or return None to fall back to the projection
   ```

2. What an arm gets:
   - `pair` is a row of the frozen `pairs.csv` **with the answer columns (`ref_*`) removed**.
   - `ctx` (`crossview_align_48.Context`) lazily provides:
     - `view(pair, "src"|"oth")` and `view_centre(pair, which)`;
     - `labeler()`, the labeler's `geo` / `fuse_sites` / `eval_sites`;
     - `pano(city, id)`, the raw results.jsonl pano block, including `source_metadata`;
     - `slim(city, id)`, the labeler's SlimPano;
     - `cache` for models, and `args`, including repeatable `--extra KEY=VALUE`.
   - `crossview_arms/geometry.py` has `forward` / `inverse` (the labeler's raycast and
     its inverse, with a numeric solve for posed cameras) and `at_height` (the labeler's
     height resolver).
3. Run it, then score every arm:

   ```bash
   python scripts/analysis/crossview_align_48.py arms                       # it is listed
   python scripts/analysis/crossview_align_48.py predict --arm my_arm --views VIEWS \
       [--labeler-root ../sidewalk-auto-labeler --runs-root ../sidewalk-auto-labeler/runs \
        --results-root RUNS_ARCHIVE]
   python scripts/analysis/crossview_align_48.py score        # every predictions/*.jsonl
   pytest -q tests/test_crossview_align_48.py                 # results.json re-derives
   ```

   `predict` writes `predictions/my_arm.jsonl` (one row per pair) and `my_arm.meta.json`
   (wall-clock, host, visible GPU, versions, config, labeler provenance, the pairs hash).
   Commit both and `results.json`. If it used a GPU, add a `paid: false` row to
   `analysis_out/usage_log.jsonl` as below. `score` takes about 3 minutes for 13 arms.
4. `VIEWS` is the directory `cut-views` wrote. On makelab2 it is
   `/homes/gws/jonf/crossview48/views` (600 JPEGs, 162 MB). A different view size or FOV
   means cutting new views, keyed by their own centre and FOV; say so in the arm's config.

## 7. What this does not show

- **No negatives.** Every pair is a true match, so this does not test whether an arm can tell
  "same ramp" from "a different ramp nearby". That is what an association signal for
  [labeler#56](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/56) has to
  do. A negative pair list (the same source ramp against a different nearby ramp's detection
  in the same other view) is the natural next addition to this harness.
- **Only pairs the 2.6 m world test admits,** so the worst projection failures (e.g.
  richmond:3) are not scored.
- **Other views within 18 m only;** the pool itself stops at 25 m.
- **No Mapillary height arm.** Labeler #89 found no validated Mapillary height estimator, and
  its `camera_heights.json` has no applied group, so the labeler resolves Richmond to 2.6 m.
  A fixed lower height for Mapillary was not tried: it would be a tuned constant, and this
  set is too small to tune on and test on at once.

## 8. Verdict (proposed, not decided)

- **Gallery rings:**
  - Use `proj_height_auto` as the base placement everywhere. It is free, never falls back,
    and on GSV it cuts the median error by about a third with a CI-clear paired gain.
  - Where `lg` aligns, prefer its point, and style the two ring types differently.
  - Leave the Mapillary pose arms and the per-pano / depth arms out.
  - The gallery is being rebuilt separately; nothing here touches it.
- **Association signal for labeler clustering:**
  - `proj_height_auto` is a better prior for the geometry the clustering already uses.
  - Image alignment is too often silent (70% of pairs) to be the primary signal. It is
    worth a discrimination test with negatives before anyone builds on it.

## 9. Reproduction

`score` and `noise` read only committed files. `predict` for the committed arms needs
unpublished inputs.

```bash
# 1. pair list (desktop CPU, ~45 s): same inputs as multiview_48.md section 10 (a labeler
#    checkout read-only, its runs/, and the archived results.jsonl copies). Frozen.
python scripts/analysis/crossview_align_48.py pairs --labeler-root labeler_main \
    --runs-root ../sidewalk-auto-labeler/runs --results-root runs_archive

# 2. views (makelab2 CPU, 116 s with 8 workers; 600 views from 428 panos, 162 MB)
python scripts/analysis/crossview_align_48.py cut-views \
    --archive-root /projects/makeabilitylab/sidewalk-auto-labeler/runs --out views

# 3. matching arms (needs `pip install kornia==0.8.3`; first run downloads the ALIKED and
#    LightGlue weights from GitHub)
for a in lg lg_local lg_band0.5 sift ncc; do
  python scripts/analysis/crossview_align_48.py predict --arm $a --views views; done

# 4. geometry arms (labeler checkout at 39afcd4 or later main: needs fuse_sites.load_at_height
#    and depth.ground_range_at; runs/<city>/depth/ for the GSV cities)
for a in proj_flat_check proj_height_auto proj_height_perpano proj_gsv_depth \
         proj_mly_gravity proj_mly_road proj_mly_rawgps; do
  python scripts/analysis/crossview_align_48.py predict --arm $a \
      --labeler-root ../sidewalk-auto-labeler --runs-root ../sidewalk-auto-labeler/runs \
      --results-root runs_archive; done

# 5. score and the reference-noise estimate (CPU, committed inputs only)
python scripts/analysis/crossview_align_48.py score
python scripts/analysis/crossview_align_48.py noise
```

**Unpublished inputs and what would unblock them.**

- **The labeler runs.** `pairs` and the geometry arms need them, exactly as
  `multiview_evidence_48.py run` does (`docs/multiview_48.md` §10). The geometry arms also
  need the labeler's `runs/<city>/depth/` payloads for the four GSV cities. Those are
  harvested from Google and are not published.
- **The native-res panos.** `cut-views` needs the 428 panos in `pairs.csv`, which are only in
  the labeler's archive on makelab2. The views are not committed. Publishing the panos, or
  the 600 views with a sha256 manifest, to Hugging Face beside
  `projectsidewalk/rampnet-benchmark` would make the matching arms replicable from a clean
  clone.
- **`kornia`** is not in `requirements.txt` or `environment.yml`. It was installed into a
  scratch directory, so the conda env was not changed.

**Content hashes** (sha256):

- `pairs.csv`: `a85a11bceb57e7d4db4bc5914c35cb17b5fdaf9bc8c9189957574a13260db388`
  (frozen, `PAIRS_SHA256`);
- `eligible_pairs.csv`: `9ec142e3…abbd134`;
- `results.json`: `035fb29c529e19dbdece828512010d995c3fc3ac2b2c86a8c781ac9427407ece`;
- `reference_noise.json`: `0b090de8…29bf8c`.

Every prediction file's meta records the pairs hash. The tests check that each prediction
file covers the frozen pairs, and that every arm's headline median and fallback rate in
`results.json` re-derive from its predictions.

## 10. Cost and time

| step | where | wall-clock | GPU-h | $ |
|---|---|---|---|---|
| `pairs` | desktop CPU | 44 s | 0 | 0 |
| `cut-views` (600 views) | makelab2 CPU, 8 workers | 116 s (+ ~1 min tar and copy) | 0 | 0 |
| matching arms, committed runs: lg 50 s, lg_local 44 s, lg_band0.5 57 s, sift 85 s (CPU), ncc 6 s | desktop RTX 3070 | 242 s | 0.04 (LightGlue arms) | 0 |
| matching, pre-harness runs (two committed then superseded, five superseded by the fixes in §5) | desktop RTX 3070 | 269 s + 737 s | ≤ 0.28 | 0 |
| geometry arms (7) | desktop CPU | 22 s | 0 | 0 |
| `score`, `noise` | desktop CPU | ~3 min + seconds | 0 | 0 |

No makelab2 GPU, no klone, no Tillicum, no paid API. GPU runs have `paid: false` rows in
`analysis_out/usage_log.jsonl`: the two pre-harness `match` runs and the three LightGlue
`predict` runs.
