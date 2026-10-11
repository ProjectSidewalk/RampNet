# Cross-view placement ([#48](https://github.com/ProjectSidewalk/RampNet/issues/48)): semantic and structural arms

One family of arms for the cross-view harness in
[`docs/crossview_align_48.md`](../crossview_align_48.md): align the **scene's structure**
(curb lines, the sidewalk/road boundary, painted markings, ground line segments) instead of
its texture. Code: `scripts/analysis/crossview_arms/semantic.py`. Scored on the same frozen
300 pairs (`pairs.csv` sha256 `a85a11bc…`) with the harness's metrics (§3 there). Run
2026-09-28 on free compute only.

## Summary

> **Multiplicity (added 2026-09-29, review of
> [#210](https://github.com/ProjectSidewalk/RampNet/pull/210)).** "CI-clear" in this doc
> means the arm's own uncorrected 95% ramp-bootstrap CI excludes zero. 83 arms were scored
> on the same 300 pairs, so some CI-clear gains are expected by selection alone. A
> one-sided Bonferroni screen over all 83 arms (`docs/crossview_align_48.md`, "Combined
> comparison") keeps only `mapa_posed_pair` and `mapa_posed_corner` on GSV, and those two
> plus `mapa_k_pair` and `mapa_posed_poseonly` over all 300 pairs. The Mapillary stratum was
> not screened. `mapa_posed_pair` is itself post hoc.

- **Best arm: `sem_chamfer_auto`.** Vistas curb and marking edges from both views are put
  on flat ground at the labeler's 'auto' camera heights. A translation is then fitted that
  lays the other view's structure over the source's, and the GT point is moved by it.
  - **Median 3.30° [2.95, 4.11]** from the reference, against 4.06° [3.66, 4.56] for
    `proj_height_auto` and 5.62° for today's projection. p90 is 10.3° against 13.9°.
  - **Paired gain over `proj_height_auto`: +0.42° [0.14, 0.71]**, closer on 60% of pairs.
    Over the projection: +0.85° [0.54, 1.50].
  - It **falls back on 1% of pairs** (3 of 300), against 70% for `lg`.
- **It moves bad pairs, not good ones into the noise floor.** The share within 2° is 0.25
  against 0.24 for `proj_height_auto`. The gain is in the middle and tail of the
  distribution (p90 −3.6°), not at the ~2° floor the reference noise sets.
- **It helps on both imageries.** GSV: +0.34° [0.02, 0.56] over `proj_height_auto`.
  Mapillary, where 'auto' is 2.6 m: +0.75° [0.38, 1.68] over the projection (60 pairs,
  median 3.84° vs 4.56°). `lg` is still better on Mapillary (3.05° over all 60 pairs).
- **Semantics are what make it work.** The same chamfer on raw LSD line segments
  (`lsd_chamfer`) is worse than `proj_height_auto` (−0.45° [−0.74, −0.13]). Kept to
  segments on Vistas curb and marking edges (`lsd_sem_chamfer`), it recovers half the
  semantic arm's gain.
- **Snapping to the nearest curb does not help** beyond the height fix (`sem_snap_auto`
  −0.00°, `sem_curb_shift_auto` −0.01° vs `proj_height_auto`).
- **Verdict (proposed, not decided):** `sem_chamfer_auto` improves on `proj_height_auto`
  with an uncorrected CI clear of zero while almost never falling back. That gain does
  **not** survive the 83-arm Bonferroni screen on GSV (lower bound −0.12°), and a leakage
  path through the relabelled Curb Cut pixels is disclosed but not measured (§4). It is a
  candidate base placement for gallery rings, pending the ignore-mask re-run (§4), with `lg` preferred where `lg` aligns (that
  combination is not scored here). The gain is modest (0.4° median, about 5 px on a
  4096-wide pano), and it needs a GPU segmentation pass per view.

## 1. What the arms do

Every arm has the same three steps.

1. **Structure in each view.** The views are the harness's 1024×768, 75° views: the source
   centred on the GT point, the other view on today's projection.
   - **`sem_*` arms: Mask2Former Swin-L trained on Mapillary Vistas v1.2**
     (`facebook/mask2former-swin-large-mapillary-vistas-semantic`, revision `4772b6bf`, the
     checkpoint of [#126](https://github.com/ProjectSidewalk/RampNet/issues/126)). It runs
     at the views' native 1024×768, not the processor's default 384×384.
     - **The "Curb Cut" class (id 9) is suppressed before the argmax**, so those pixels go
       to their next-best class (see §4).
     - Two edge channels are kept. The *curb edge* is Curb / Sidewalk / Pedestrian Area
       pixels touching road-like pixels. The road-like classes are road, bike lane, parking,
       service lane, crosswalk and lane markings, catch basin, manhole and pothole. The
       *marking edge* is the outline of Lane Marking – Crosswalk / – General.
   - **`lsd_*` arms: OpenCV LSD** line segments of at least 20 px on the grey view, with no
     model. `lsd_sem_chamfer` keeps only segment pixels within 4 px of a Vistas curb or
     marking edge.
2. **Onto the ground.** Edge pixels at least 5° below the pano horizon (the `lg` band) and
   above the rig are raycast onto flat ground, using each pano's heading, position and
   camera height from the labeler. This is the labeler's flat path, done analytically so a
   whole view projects in one numpy call.
   - `sem_geom_check` is the instrument check. With zero shift it reproduces the committed
     `proj_x` / `proj_y` to ≤ 0.0035° on all 300 pairs.
   - Only points within 30 m of their camera and 10 m of W are kept, where W is the GT
     point's world position.
3. **Align, then re-project W.**
   - **`*_chamfer`** fits a translation *t* so that the other view's structure plus *t*
     overlays the source's. The fit uses a truncated chamfer (0.5 m) on a 5 cm raster, over
     ±3 m. The chosen *t* is the posterior mean of exp(−ΔJ / 0.02 m) × N(0, 1.5 m²), so a
     single straight curb pins the point across the curb and leaves it where it was along
     the curb. W − *t* is then placed in the other pano. The arm falls back if either view
     has fewer than 40 structure cells, or if the data do not shrink the posterior sd below
     0.8× the prior along the best-constrained direction.
   - **`sem_snap`** moves W to the other view's nearest curb-edge point within 2.5 m.
   - **`sem_curb_shift`** moves W by the difference between the source's and the other
     view's nearest curb points. It is a one-point curb registration.

The camera height is 2.6 m, as in today's projection, unless the arm ends in `_auto`. Those
arms use the labeler's 'auto' per-rig heights, as `proj_height_auto` does. A fallback goes
to today's 2.6 m projection, as the harness defines it, including for the `_auto` arms.

**What was fixed before scoring, and what was not.** Every constant in `semantic.py`, and
the arms `sem_chamfer`, `sem_chamfer_auto`, `sem_chamfer_curb`, `lsd_chamfer`,
`lsd_sem_chamfer`, `sem_snap` and `sem_curb_shift`, were written before any arm was scored,
and nothing was re-tuned after. Before scoring, only answer-free diagnostics were read:
shift sizes, posterior sd, cell counts, and one edge overlay. **`sem_snap_auto` and
`sem_curb_shift_auto` were added after seeing the 2.6 m snap scores** (post hoc). Neither
changes the verdict.

## 2. Results

Scored with `crossview_align_48.py score --arms …` into
`analysis_out/crossview_align_48/results_semantic.json`. Paired comparisons against both
baselines are from `semantic.py compare`, in `compare_semantic.json`. CIs are 2.5–97.5
percentiles over 2,000 bootstrap resamples of ramps, seed 48.

### All 300 pairs

| arm | median ° [CI] | px | p90 ° | within 2° | fallback |
|---|---|---|---|---|---|
| projection | 5.62 [4.53, 6.56] | 66 | 15.7 | 0.19 | 0 |
| proj_height_auto | 4.06 [3.66, 4.56] | 47 | 13.9 | 0.24 | 0 |
| sem_geom_check (instrument) | 5.62 [4.52, 6.56] | 66 | 15.7 | 0.19 | 0 |
| sem_chamfer | 4.08 [3.45, 4.50] | 47 | 12.2 | 0.22 | 0.02 |
| **sem_chamfer_auto** | **3.30 [2.95, 4.11]** | **38** | **10.3** | 0.25 | 0.01 |
| sem_chamfer_curb | 4.07 [3.37, 4.74] | 48 | 11.9 | 0.25 | 0.02 |
| lsd_chamfer | 4.92 [4.28, 6.04] | 57 | 14.8 | 0.18 | 0.04 |
| lsd_sem_chamfer | 4.49 [3.86, 5.03] | 52 | 13.2 | 0.20 | 0.04 |
| sem_snap | 4.45 [3.82, 5.33] | 52 | 16.5 | 0.17 | 0.06 |
| sem_snap_auto (post hoc) | 3.83 [3.44, 4.36] | 45 | 11.9 | 0.21 | 0.05 |
| sem_curb_shift | 4.83 [4.08, 5.99] | 56 | 17.0 | 0.21 | 0.08 |
| sem_curb_shift_auto (post hoc) | 3.88 [3.43, 4.88] | 46 | 13.2 | 0.25 | 0.07 |

### Paired gains (positive = the arm is closer to the reference)

Each cell gives the median per-pair gain [CI] and the share of pairs where the arm is
strictly closer. "Aligned" means the pairs where the arm did not fall back.

| arm | vs projection, all 300 | vs proj_height_auto, all 300 | aligned n | aligned: vs projection | aligned: vs proj_height_auto |
|---|---|---|---|---|---|
| sem_chamfer | +0.70 [0.40, 1.07], 0.64 | +0.07 [−0.19, 0.41], 0.51 | 294 | +0.74 [0.42, 1.23] | +0.12 [−0.18, 0.41] |
| **sem_chamfer_auto** | **+0.85 [0.54, 1.50], 0.68** | **+0.42 [0.14, 0.71], 0.60** | 297 | +0.85 [0.59, 1.57] | **+0.43 [0.15, 0.71]** |
| sem_chamfer_curb | +0.85 [0.33, 1.28], 0.64 | +0.06 [−0.23, 0.54], 0.51 | 293 | +0.95 [0.43, 1.42] | +0.14 [−0.23, 0.61] |
| lsd_chamfer | +0.08 [−0.05, 0.31], 0.51 | −0.45 [−0.74, −0.13], 0.41 | 289 | +0.18 [−0.16, 0.38] | −0.46 [−0.74, −0.12] |
| lsd_sem_chamfer | +0.49 [0.17, 0.88], 0.59 | −0.24 [−0.55, 0.04], 0.45 | 289 | +0.58 [0.23, 0.96] | −0.24 [−0.55, 0.04] |
| sem_snap | +0.09 [0.00, 0.20], 0.54 | −0.26 [−0.63, 0.08], 0.45 | 282 | +0.15 [0.03, 0.30] | −0.26 [−0.63, 0.18] |
| sem_snap_auto | +0.22 [0.00, 0.63], 0.55 | −0.00 [−0.23, 0.16], 0.49 | 284 | +0.44 [0.10, 0.93] | −0.08 [−0.27, 0.19] |
| sem_curb_shift | +0.00 [−0.02, 0.15], 0.47 | −0.39 [−0.81, 0.00], 0.43 | 277 | +0.05 [−0.17, 0.28] | −0.34 [−0.72, 0.05] |
| sem_curb_shift_auto | +0.43 [0.11, 0.75], 0.57 | −0.01 [−0.40, 0.18], 0.49 | 279 | +0.58 [0.32, 0.93] | −0.04 [−0.43, 0.20] |

### By imagery

| arm | GSV (240): median °, gain vs proj_height_auto [CI], fallback | Mapillary (60): median °, gain vs projection (= auto there) [CI], fallback |
|---|---|---|
| projection | 6.09 | 4.56 |
| proj_height_auto | 3.92 | 4.56 (unchanged) |
| sem_chamfer | 4.15, −0.13 [−0.44, 0.25], 0.03 | 3.84, +0.75 [0.38, 1.68], 0 |
| **sem_chamfer_auto** | **3.23, +0.34 [0.02, 0.56]**, 0.01 | 3.84, +0.75 [0.38, 1.68], 0 |
| sem_chamfer_curb | 4.25, −0.14 [−0.77, 0.19], 0.02 | **3.28, +1.38 [0.44, 2.30]**, 0.03 |
| lsd_chamfer | 5.31, −0.70 [−1.51, −0.32], 0.03 | 3.71, +0.31 [0.00, 1.10], 0.05 |
| lsd_sem_chamfer | 4.52, −0.49 [−0.94, −0.20], 0.03 | 4.21, +0.51 [0.00, 1.90], 0.05 |
| sem_snap_auto | 3.86, −0.18 [−0.40, 0.12], 0.05 | 3.60, +0.15 [0.00, 0.45], 0.08 |
| sem_curb_shift_auto | 3.72, −0.05 [−0.40, 0.19], 0.07 | 4.82, +0.00 [−1.15, 0.39], 0.08 |
| `lg` (from the main doc) | 5.91, fallback 0.76 | 3.05, fallback 0.48 |

`sem_chamfer` and `sem_chamfer_auto` are identical on Mapillary, because 'auto' resolves
Richmond to 2.6 m.

### By range and city

Median °, with the paired gain over the projection [CI] in brackets.

| arm | 0–6 m (21) | 6–12 m (158) | 12–18 m (121) | paterson | gainesville | bend | sao_paulo | richmond |
|---|---|---|---|---|---|---|---|---|
| projection | 12.67 | 6.87 | 3.40 | 6.60 | 9.54 | 3.54 | 4.52 | 4.56 |
| proj_height_auto | 9.54 (+1.44 [−0.00, 4.42]) | 4.68 (+0.33 [0.00, 1.15]) | 2.90 (+0.00 [0.00, 0.20]) | 4.26 | 3.70 | 3.30 | 4.09 | 4.56 |
| sem_chamfer | 8.00 (+3.37 [0.65, 7.24]) | 4.96 (+1.05 [0.52, 1.81]) | 2.89 (+0.36 [0.00, 0.75]) | 4.03 | 5.22 | 3.46 | 3.51 | 3.84 |
| sem_chamfer_auto | 8.09 (+3.32 [0.65, 7.34]) | 4.20 (+1.53 [0.75, 2.25]) | **2.25** (+0.44 [0.03, 0.80]) | 3.19 | 3.06 | 3.58 | 3.15 | 3.84 |
| sem_chamfer_curb | 8.16 (+3.37 [0.18, 7.24]) | 5.14 (+1.21 [0.56, 2.22]) | 2.39 (+0.32 [0.06, 0.92]) | 3.99 | 5.05 | 4.17 | 4.21 | 3.28 |

Stratum CIs rest on as few as 21 pairs (0–6 m) and 60 per city; read them as direction.
Every number is in `results_semantic.json` → `arms.<arm>.<stratum>`.

## 3. Reading

- **Structure registration and camera height fix different errors, and they add.** The
  height fix mostly corrects range (the along-ray error). The chamfer corrects what is
  left:
  - pose error: position and heading;
  - GT placement error across the curb;
  - on Mapillary, the height error that 'auto' does not model there.

  On GSV the chamfer at 2.6 m (`sem_chamfer`) only matches `proj_height_auto`. On top of
  'auto' it beats it. The median shift it applies is 0.81 m (p90 1.75 m).
- **Bend is the exception.** Its projection is already good (3.54°), and no arm here moves
  it. Gainesville gains most: the chamfer on top of 'auto' goes from 3.70° to 3.06°.
- **Curb-only vs curb + markings.** Dropping the marking channel (`sem_chamfer_curb`) helps
  Richmond (3.28° vs 3.84°) and hurts São Paulo (4.21° vs 3.51°). Across all pairs it is a
  wash. It was not run on 'auto'. The per-city differences rest on 60 pairs each, so no
  channel choice is recommended from them.
- **Why raw lines fail.** LSD fires on shadows, cracks, parked-car edges and pavement
  seams. Those do not repeat across views taken minutes to years apart, so the chamfer
  locks onto noise. Restricting the lines to semantic edges removes most of that, but the
  lines are then a subset of the edges `sem_chamfer` already uses, and it adds nothing.
- **Why snapping fails.** Snapping assumes the error is across the curb. Much of it is
  along the curb, especially at 6–18 m where the curb runs roughly radially. There a snap
  moves the point sideways by the wrong amount or not at all. `sem_curb_shift` also
  assumes the source point's offset from its curb transfers. That offset is measured
  through the same noisy flat-ground raycast, so errors at both ends add.
- **Not run: line matching (GlueStick / DeepLSD) and a vanishing-point frame.** Once both
  views are on a common ground plane, a homography fitted from matched lines has little
  left to recover beyond the translation the chamfer already fits. Measuring that would
  need a line matcher that is not installed here. A vanishing-point estimate of each view's
  pitch/roll would be a separate pose-prior arm. On GSV, whose equirects are already
  gravity-rectified, the Mapillary pose arms suggest it would not help.
- **Not run: a curb-corner (curb-line intersection) snap.** It was dropped after the two
  nearest-curb snaps showed no gain. A corner snap is a special case of the same
  across-curb assumption.

## 4. Circularity, and what these arms do not read

- **Reference.** The known answer is the other view's own RampNet detection peak. No arm
  here reads any detector's output in either view: not RampNet, and not Vistas' "Curb Cut"
  class.
- **Curb Cut suppressed.** A segmenter that marks curb cuts in the other view is a ramp
  detector, and scoring it against a ramp detector's peak would be circular. The class-9
  channel is removed before the argmax, so the label maps contain no curb-cut class. The
  suppressed pixel count per view is in `semantic_seg_manifest.json`.
  - **One residual path remains, and it is disclosed but not measured.** The model's
    internal features were trained with that class. The pixels it would have called curb
    cut are relabelled to their next-best class, mostly sidewalk or road. Either way the
    ramp's outline then enters the curb-edge channel: as a notch in the sidewalk/road
    boundary, or as a shift of it. So the chamfer can register the ramp's shape in one view
    to its shape in the other, and the reference is a ramp detection in the other view.
    This is a weaker form of the circularity the suppression was meant to remove.
  - **It is not rare.** Per `semantic_seg_manifest.json`, Curb Cut would have won pixels in
    **298 of 300 other views and 296 of 300 source views**: a median of about 2,000 px per
    view, up to 51,697 px in an other view and 122,859 px (16% of the view) in a source view.
    The chamfer uses every structure cell within 10 m of W, of which the ramp is a small
    part, so the effect may be small. That is a guess, not a measurement.
  - **This matters for the headline of this family.** `sem_chamfer_auto` is one of five arms
    with an uncorrected CI-clear GSV gain over auto (+0.34° [0.02, 0.56]), and its lower
    bound is close to zero. It does **not** survive the Bonferroni screen over all 83 arms
    in `docs/crossview_align_48.md` (lower bound −0.12°). Quote it as a candidate, with this
    leakage caveat beside it.
  - **Follow-up, not run:** have `Segmenter` also write the unsuppressed Curb Cut mask, map
    those pixels (dilated a few px) to an ignore label in neither `RAISED` nor `ROADLIKE`,
    and re-run as a new arm `sem_chamfer_auto_ccmask`. About 3 min on the A40 plus CPU time.
    That turns this disclosure into a measurement.
  - `label_map` now refuses a map whose sha256 does not match the **committed** manifest
    (`semantic_seg_manifest.json`), or the one named by `--extra seg_manifest=PATH` for a new
    segmentation such as `_ccmask`. It never trusts `seg_dir`'s own `manifest.json`, which a
    foreign map could ship with. It also refuses, with an explicit error rather than an
    `assert`, any map that contains the Curb Cut class. So a stale, foreign or unsuppressed
    `seg_dir` cannot be read silently. A map regenerated on a miss is listed in
    `seg_dir/regenerated.json`, and a later run that finds it says it was regenerated rather
    than calling it foreign. Added 2026-09-29; the committed runs predate the check.
- **A detector-guided arm** would snap to a ramp detection in the other view: RampNet's,
  Vistas' Curb Cut blob, or an open-vocabulary detector's. It would need a reference that
  is not a detector: human clicks on the ramp in both views of each pair. The natural
  design:
  - a second-rater-ready click file per pair, with the 300 pairs as the item list;
  - both the detector-guided arm and every arm here scored against the clicks;
  - the current reference kept as a third "arm", so its own noise is measured too.

  That also replaces the reference-noise floor (§2 of the main doc) with a measured one.
- **Aerial anchors** (a curb or sidewalk map from overhead imagery, used to register the
  ground) are out of scope for this family. They belong to
  [sidewalk-auto-labeler#104](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/104).

## 5. Limits

- **The main doc's §7 applies unchanged.** There are no negatives, so this does not show
  whether structure alignment can tell "same ramp" from "a nearby ramp". Only pairs the
  2.6 m world test admits are scored, which makes the projection's numbers optimistic and
  any gain conservative.
- **Reference noise.** With a peak at each end, errors of about 2° are at the floor this
  set resolves. The within-2° share barely moves for any arm here, so no claim is made
  about placement below that floor.
- **Flat ground.** Every arm assumes flat ground over 10 m around W. The curb face (about
  15 cm) and street crossfall are ignored.
- **The harness falls back to the 2.6 m projection, even for `_auto` arms.** At 1%
  fallback this barely matters for `sem_chamfer_auto`. For `sem_snap_auto` (5%) and
  `sem_curb_shift_auto` (7%) the all-pairs numbers slightly understate the arm, so read the
  aligned columns too.
- **One segmenter.** OneFormer and the Cityscapes classes were not tried. Cityscapes has no
  curb class, so it could give only a sidewalk/road boundary.

## 6. Reproduction

```bash
# 1. segmentation (makelab2 A40, 169 s for 600 views; any CUDA GPU; CPU works, slowly)
python scripts/analysis/crossview_arms/semantic.py segment --views VIEWS --out SEG_DIR --device cuda
# 2. arms (desktop CPU; labeler inputs as for the geometry arms)
for a in sem_geom_check sem_chamfer sem_chamfer_auto sem_chamfer_curb lsd_chamfer \
         lsd_sem_chamfer sem_snap sem_snap_auto sem_curb_shift sem_curb_shift_auto; do
  python scripts/analysis/crossview_align_48.py predict --arm $a --views VIEWS --cpu \
      --extra seg_dir=SEG_DIR --labeler-root LABELER --runs-root LABELER/runs \
      --results-root RUNS_ARCHIVE; done
# 3. tables (CPU, committed inputs only)
python scripts/analysis/crossview_align_48.py score --arms proj_height_auto,sem_geom_check,sem_chamfer,sem_chamfer_auto,sem_chamfer_curb,lsd_chamfer,lsd_sem_chamfer,sem_snap,sem_snap_auto,sem_curb_shift,sem_curb_shift_auto \
    --out analysis_out/crossview_align_48/results_semantic.json
python scripts/analysis/crossview_arms/semantic.py compare --arms sem_chamfer,sem_chamfer_auto,sem_chamfer_curb,lsd_chamfer,lsd_sem_chamfer,sem_snap,sem_snap_auto,sem_curb_shift,sem_curb_shift_auto \
    --out analysis_out/crossview_align_48/compare_semantic.json
```

- **Inputs.** The views and labeler inputs are those of the main doc (§9 there), with the
  same blockers: native-res panos and labeler runs are not published.
- **Label maps.** The 600 maps (10 MB) are not committed. Their sha256s are in
  `semantic_seg_manifest.json`, so a regenerated set can be checked against them. They are
  deterministic given the views, the pinned revision and the GPU; CPU float32 output was
  not compared.
- **Environment.** Segmentation ran with transformers 5.15.0 and torch 2.13.0+cu130
  (makelab2 `.venv-eval`). The arms ran with torch 2.6.0 and OpenCV 5.0.0, the desktop
  `.venv`. transformers reports the Swin backbone's final `layernorm` as missing from the
  checkpoint. That layer is not on Mask2Former's feature path. #126 checked that
  this transformers 5.x load reproduces the 4.x published detections to within one
  detection.
- **The shared `results.json` was not regenerated on this branch.** Once these prediction
  files are merged, `score` must be re-run for
  `tests/test_crossview_align_48.py::test_committed_results_rederive_from_committed_predictions`
  to pass.

## 7. Cost

| step | where | wall-clock | GPU-h | $ |
|---|---|---|---|---|
| Vistas segmentation, 600 views | makelab2 A40 | 169 s | 0.047 | 0 |
| chamfer arms (5): `lsd_chamfer` alone, the other four in parallel, 2 threads each | desktop CPU | 121–354 s each | 0 | 0 |
| snap arms (4) and the geometry check | desktop CPU | 5–26 s each | 0 | 0 |
| `score`, `compare` | desktop CPU | ~3 min | 0 | 0 |

Each arm's `.meta.json` has its own wall-clock. The segmentation has a `paid: false` row
in `analysis_out/usage_log.jsonl`. No klone, no Tillicum, and no paid API were used.
