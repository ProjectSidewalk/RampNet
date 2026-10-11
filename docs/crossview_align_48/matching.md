# Cross-view placement ([#48](https://github.com/ProjectSidewalk/RampNet/issues/48)): the pairwise image-matching family

This is one of the arm families run on the cross-view harness
(`docs/crossview_align_48.md`, frozen pairs `a85a11bc…`). The question is whether a
better matcher, estimator or prior lowers the pilot `lg` arm's 70% fallback rate without
losing its accuracy. The pilot's `lg` is ALIKED + LightGlue with a ground homography.

- Code: `scripts/analysis/crossview_arms/pairwise_matching.py` (the arms) and
  `scripts/analysis/crossview_matching_48.py` (the tables).
- Outputs: `analysis_out/crossview_align_48/predictions/<arm>.jsonl` and
  `.meta.json`, and `analysis_out/crossview_align_48/matching/report.json`.
- Test: `tests/test_crossview_matching_48.py`.
- Run 2026-09-28 on the desktop RTX 3070 only.

## Summary

> **Multiplicity (added 2026-09-29, review of
> [#210](https://github.com/ProjectSidewalk/RampNet/pull/210)).** "CI-clear" in this doc
> means the arm's own uncorrected 95% ramp-bootstrap CI excludes zero. 83 arms were scored
> on the same 300 pairs, so some CI-clear gains are expected by selection alone. A
> one-sided Bonferroni screen over all 83 arms (`docs/crossview_align_48.md`, "Combined
> comparison") keeps only `mapa_posed_pair` and `mapa_posed_corner` on GSV, and those two
> plus `mapa_k_pair` and `mapa_posed_poseonly` over all 300 pairs. The Mapillary stratum was
> not screened. `mapa_posed_pair` is itself post hoc.

- **The fallback problem is a matcher problem, and RoMa solves it.** RoMa (dense, outdoor
  weights) with the pilot's own estimator (`roma`: RANSAC ground homography, ≥ 15 inliers,
  mapped point inside the view) falls back on **7%** of pairs. `lg` falls back on 70%. On
  GSV the rates are 8% vs 76%.
  - The other sparse matchers fall back more than ALIKED: SuperPoint 74%, DISK 84%, SIFT
    85%, each with LightGlue.
  - LoFTR falls back on 86%.
- **On Mapillary (Richmond) RoMa is a clear win on every pair.** The median error is
  **2.58° vs 4.56°** for the projection over all 60 pairs, with 3% fallback. The paired
  gain is 1.75° [0.83, 2.77]. `roma_local`, which fits the homography only to matches near
  the GT point, gives 2.37° [1.83, 2.94], a gain of 1.87° [0.98, 2.88], and 40% within 2°
  (auto height: 10%).
- **On GSV no matching arm beats the free `proj_height_auto` prior.** That prior is 3.92°
  over the 240 GSV pairs.
  - `roma` gives 4.41° on GSV, a paired gain vs auto of −0.19° [−0.91, 0.12].
  - This includes the pilot's `lg`. On the 58 GSV pairs it aligns it gives 2.64°, while
    auto height gives **2.40°** on those same pairs (gain −0.15° [−0.98, 0.45]). So `lg`'s
    GSV "win where it works" is really a selection of easy pairs. The pilot compared it
    only with the 2.6 m projection (4.30° there).
- **Reading RoMa's dense warp at the GT point beats reading its homography.** This arm
  (`roma_warp`) assumes no plane.
  - It aligns 58% of pairs and falls back where RoMa's certainty at the point is below
    0.05 (romatch's default `sample_thresh`, borrowed as a cutoff; §1).
  - Where it aligns: **2.11° vs 3.48°** for auto height, a paired gain of **0.58°
    [0.35, 1.21]**. On GSV alone: 2.17° vs 2.97°, 0.37° [−0.03, 0.57], which spans zero;
    the pooled CI-clear gain comes from Mapillary (2.85° [1.44, 3.50]).
  - The aligned subset is selected by RoMa's own certainty, the same kind of easy-pair
    selection this doc points out for `lg`: on those pairs auto is 3.48°, against 4.06°
    overall. The paired gain mostly absorbs this, but not entirely.
  - With auto height as the fallback (`roma_warp_hyb`) it never falls back. Results: all
    pairs 3.11° [2.75, 3.93] vs auto 4.06° [3.66, 4.56]; GSV 3.58° [2.82, 4.28] vs 3.92°;
    Mapillary 2.58°. Its harm rate (10% of pairs more than 2° worse than auto, 20% more
    than 2° better) is lower than `roma_warp`'s only because of dilution: it is the same 29
    harmed and 59 helped pairs, divided by all 300 instead of the 174 the warp moved. The
    126 pairs copied from auto can never be harmed. On the pairs it moves it is exactly
    `roma_warp`.
  - It reports 0% fallback, but 126 of its 300 answers (42%) are the auto prior. The
    combined table's common-rule columns score every arm that way; there `roma_warp` and
    `roma_warp_hyb` are identical (3.11°).
  - Caveat: `roma_local` and `roma_warp` were added **after** seeing `roma`'s scores (§4).
- **Why `lg` falls back** (211 pairs, §3):
  - In 122 the matches exist but are not on the ground.
  - In 77 there are ground matches but no plane with 15 inliers.
  - In 9 the views barely match at all.
  - The strongest covariate is camera baseline. `lg` falls back on 29% of pairs under 10 m,
    74% at 10–20 m and **100% at ≥ 20 m**. A capture-month change is second: 82% vs 64%.
- **Negative results** (§5):
  - MAGSAC++ in place of RANSAC changes nothing.
  - The epipolar snap (an essential matrix from all matches, moving the auto-height prior
    onto the GT point's epipolar line) aligns 44–98% of pairs, but its paired gain against
    auto height is 0.00–0.03° overall. It is a no-op except on Mapillary (§5).
  - LoFTR is worse than the projection where it aligns.
- **Verdict (proposed, not decided):**
  - Mapillary: use `roma_local` (or `roma`) in place of the projection. This is the one
    place image matching clearly pays. But `roma_local` is post hoc (§4), and the Mapillary
    numbers are not Bonferroni-screened (the screen covers GSV and all 300 pairs only).
  - GSV: keep `proj_height_auto` as the base. **On GSV no matching arm has a CI-clear gain
    over auto height**, not even where it aligns: `roma_warp`'s aligned GSV gain is 0.37°
    [−0.03, 0.57]. (An earlier version of this bullet said `roma_warp_hyb` was CI-clear on
    GSV; the 0.58° [0.35, 1.21] it quoted is the pooled aligned gain, driven by Mapillary.
    Corrected after the review of [#210](https://github.com/ProjectSidewalk/RampNet/pull/210).)
    `roma_warp` / `roma_warp_hyb` are post hoc (§4) and remain hypotheses for fresh pairs.
  - Drop the sparse-matcher and LoFTR arms.

## 1. Arms

Every arm uses the harness's existing views (`cut-views`: 1024×768, 75°; the source view is
centred on the GT point, the other view on the 2.6 m projection). They also share the
pilot's ground band: 5° below the horizon and above the rig at −70°. Arm names are
`<matcher>[_<estimator>]`.

| matcher | what |
|---|---|
| `lg_*` | the pilot's ALIKED + LightGlue (kornia 0.8.3), re-run for the estimator variants |
| `sp_lg` | SuperPoint + LightGlue (cvg/LightGlue `eb42fee`), top 4,096 keypoints by score |
| `disk_lg` | DISK + LightGlue, as above |
| `siftlg` | SIFT + LightGlue, as above (not the pilot's `sift`, which uses a ratio test) |
| `loftr` | LoFTR outdoor (kornia 0.8.3), confidence ≥ 0.5 |
| `roma` | RoMa outdoor (romatch 0.1.2, default 560/864 resolution), 5,000 samples from its own certainty-balanced sampler, seed 48 |

| estimator | what | fallback |
|---|---|---|
| (none) | the pilot's: RANSAC ground homography, 4 px | < 15 inliers, or the mapped point leaves the view |
| `_magsac` | MAGSAC++ ground homography, 4 px | same |
| `_epi` | Essential matrix from **all** matches (known pinhole focal length, MAGSAC, 1 px). The GT point's epipolar line in the other view; the `proj_height_auto` point moved to its nearest point on that line | < 15 E-inliers, or the snapped point leaves the view |
| `_hyb` | homography, else `_epi`, else the `proj_height_auto` point | never |
| `roma_local` | RoMa ground homography fitted only to matches within 160 px of the GT point (`lg_local`'s radius and ≥ 12 minimum) | fewer than 12 near matches, or as above |
| `roma_warp` | RoMa's dense A→B warp, bilinearly read at the GT point; no planar model | certainty there < 0.05. This is romatch's default `sample_thresh`, borrowed as a cutoff. In romatch 0.1.2 it is a saturation threshold (`sample()` sets certainty above it to 1, and lower-certainty pixels can still be sampled), not romatch's definition of a usable match |
| `roma_warp_hyb` | `roma_warp`, else the `proj_height_auto` point | never |

**The auto-height prior** is the committed `proj_height_auto` prediction: GSV at the
labeler's per-year rig height, Mapillary unchanged at 2.6 m. It is camera geometry only
and never reads the reference. The views were *not* re-cut around it. It is a median ~2°
from the 2.6 m point, well inside a 75° view, so it serves as a prior point and a fallback,
not as a view centre.

**Fallback rows carry a reason.** Each family arm writes `why` (`no_ground_matches`,
`few_inliers`, `mapped_outside_view`, `low_certainty`, `epi_*`) and its match counts, even
when it falls back. The harness keeps these as diagnostics.

## 2. Results

`scripts/analysis/crossview_matching_48.py` adds three things to the harness's `score`:

- the paired gain against `proj_height_auto` as well as against the 2.6 m projection;
- the same on the subset each arm aligns;
- **harm / help**: the share of aligned pairs the arm moved > 2° further from / closer to
  the reference than auto height.

As in the harness, CIs are 2.5–97.5 percentiles over 2,000 resamples of ramps, and a median
paired gain over all pairs is 0 whenever most pairs fell back (a fallback has zero gain vs
projection; a `_hyb` fallback has zero gain vs auto). Every number below is in
`matching/report.json`.

Reference rows:

| arm | all 300 | GSV 240 | Mapillary 60 |
|---|---|---|---|
| projection (2.6 m) | 5.62 [4.53, 6.56] | 6.09 | 4.56 |
| `proj_height_auto` | 4.06 [3.66, 4.56] | 3.92 [3.53, 4.36] | 4.56 (unchanged) |

### All pairs

| arm | median ° [CI] | within 2° | fallback | gain vs auto, all pairs [CI] | aligned n | aligned: arm / proj / auto ° | aligned gain vs auto [CI] | harm / help |
|---|---|---|---|---|---|---|---|---|
| lg (pilot) | 4.79 [3.51, 6.01] | 0.25 | 0.70 | −0.00 [−0.44, 0.00] | 89 | 2.41 / 4.40 / 2.99 | 0.45 [−0.26, 0.93] | 0.28 / 0.31 |
| sp_lg | 4.98 [3.81, 6.18] | 0.23 | 0.74 | −0.07 [−0.46, 0.00] | 78 | 2.56 / 4.23 / 2.90 | 0.15 [−0.21, 0.94] | 0.23 / 0.27 |
| disk_lg | 5.52 [4.42, 6.52] | 0.22 | 0.84 | −0.20 [−0.47, 0.00] | 47 | 3.00 / 4.45 / 3.07 | −0.24 [−0.72, 0.49] | 0.32 / 0.26 |
| siftlg | 5.42 [4.47, 6.54] | 0.21 | 0.85 | −0.00 [−0.52, 0.00] | 45 | 2.99 / 4.20 / 3.07 | 0.09 [−1.00, 0.62] | 0.31 / 0.24 |
| loftr | 6.03 [4.98, 6.96] | 0.18 | 0.86 | −0.33 [−0.79, −0.00] | 42 | 6.30 / 4.23 / 3.58 | **−2.70 [−5.05, −0.19]** | 0.55 / 0.19 |
| roma | 3.51 [2.92, 4.70] | 0.32 | **0.07** | 0.03 [−0.22, 0.43] | 279 | 3.51 / 5.57 / 4.05 | 0.12 [−0.19, 0.48] | 0.29 / 0.28 |
| roma_magsac | 3.89 [2.95, 4.83] | 0.29 | 0.06 | 0.09 [−0.29, 0.46] | 283 | 3.70 / 5.56 / 4.03 | 0.11 [−0.20, 0.46] | 0.29 / 0.28 |
| roma_local † | 3.29 [2.92, 4.21] | 0.31 | 0.06 | 0.30 [−0.10, 0.66] | 283 | 3.23 / 5.57 / 4.03 | 0.34 [−0.03, 0.67] | 0.27 / 0.32 |
| roma_epi | 4.34 [3.77, 5.38] | 0.23 | 0.02 | 0.02 [0.00, 0.08] | 295 | 4.27 / 5.55 / 4.03 | 0.03 [0.00, 0.08] | 0.10 / 0.04 |
| roma_hyb | 3.79 [3.16, 4.85] | 0.32 | 0 | −0.00 [−0.33, 0.30] | 300 | 3.79 / 5.62 / 4.06 | – | 0.30 / 0.26 |
| roma_warp † | 3.25 [2.73, 4.43] | 0.34 | 0.42 | 0.12 [−0.00, 0.54] | 174 | **2.11** / 4.41 / 3.48 | **0.58 [0.35, 1.21]** | 0.17 / 0.34 |
| roma_warp_hyb † | **3.11 [2.75, 3.93]** | **0.35** | 0 | 0.00 [0.00, 0.00] | 300 | 3.11 / 5.62 / 4.06 | – | **0.10** / 0.20 |
| lg_epi | 4.89 [3.80, 6.12] | 0.23 | 0.35 | −0.01 [−0.13, 0.01] | 195 | 3.80 / 4.61 / 3.78 | −0.00 [−0.08, 0.05] | 0.11 / 0.03 |
| lg_hyb | 4.08 [3.55, 5.07] | 0.25 | 0 | 0.00 [0.00, 0.00] | 300 | 4.08 / 5.62 / 4.06 | – | 0.16 / 0.10 |

† added after `roma` was scored (§4). The `_magsac`, `_epi` and `_hyb` variants of every
matcher are in `report.json`; none changes the reading below.

### By imagery

| arm | GSV median ° [CI], fallback | GSV gain vs auto [CI] (all / aligned) | Mapillary median ° [CI], fallback | Mapillary gain vs projection = auto [CI] (all / aligned) |
|---|---|---|---|---|
| lg | 5.91 [4.15, 6.68], 0.76 | −0.52 [−1.20, −0.11] / −0.15 [−0.98, 0.45] (58) | 3.05 [2.39, 4.33], 0.48 | 0.00 [0.00, 0.47] / 2.25 [0.88, 3.63] (31) |
| sp_lg | 6.02 [4.55, 6.91], 0.80 | −0.62 [−1.20, −0.22] / −0.21 [−0.77, 0.10] | 3.41 [2.86, 3.92], 0.50 | 0.00 / 3.54 [0.71, 4.49] |
| loftr | 6.53 [5.35, 7.67], 0.92 | −0.79 [−1.63, −0.35] / −4.51 [−11.90, −2.60] | 4.65 [3.50, 5.87], 0.63 | 0.00 / −0.27 [−4.74, 3.21] |
| roma | 4.41 [3.26, 5.79], 0.08 | −0.19 [−0.91, 0.12] / −0.14 [−0.97, 0.11] | **2.58 [2.16, 3.38]**, 0.03 | **1.75 [0.83, 2.77]** / 1.92 [0.97, 2.79] |
| roma_magsac | 4.49 [3.43, 5.79], 0.07 | −0.24 [−0.92, 0.17] | 2.40 [2.03, 2.93], 0.02 | 1.76 [0.70, 2.65] |
| roma_local † | 4.06 [3.19, 5.18], 0.06 | −0.07 [−0.48, 0.34] / 0.02 [−0.39, 0.34] | **2.37 [1.83, 2.94]**, 0.03 | **1.87 [0.98, 2.88]** / 2.18 [1.10, 3.03] |
| roma_epi | 4.51 [3.79, 5.80], 0.02 | 0.00 [−0.05, 0.02] | 3.66 [2.89, 4.64], 0 | 0.34 [0.13, 0.60] |
| roma_hyb | 4.83 [3.45, 5.84], 0 | −0.29 [−0.97, 0.00] | 2.65 [2.29, 3.38], 0 | 1.75 [0.83, 2.77] |
| roma_warp † | 3.99 [2.79, 5.70], 0.45 | 0.03 [−0.36, 0.43] / 0.37 [−0.03, 0.57] (131) | 2.58 [1.71, 3.33], 0.28 | 1.22 [0.00, 2.74] / 2.85 [1.44, 3.50] (43) |
| roma_warp_hyb † | **3.58 [2.82, 4.28]**, 0 | 0.00 [0.00, 0.00] | 2.58 [1.71, 3.33], 0 | 1.22 [0.00, 2.74] |

Within 2°, on the pairs each arm aligns, against auto height on those pairs: `roma_warp`
GSV 0.46 vs 0.36, Mapillary 0.53 vs 0.09. `roma` GSV 0.33 vs 0.28, Mapillary 0.34 vs 0.09.

**Caveats that travel with these numbers:**

- **Reference noise floor.** The reference and the source are both detection peaks, with
  about 1.5° of peak noise each (`reference_noise.json`). Errors near 2° are at the floor
  this set can resolve, so `roma_warp`'s 2.11° aligned median is about as good as the
  instrument can show.
- **Selection at 2.6 m.** The pair set was admitted by a 2.6 m world test, which flatters
  the projection and auto height. Gains over them are conservative.
- **Mapillary is one city,** 60 pairs from 31 ramps. The CIs resample ramps, but they
  cannot speak to other Mapillary cities or rigs.
- **The `lg_*` variants re-ran ALIKED + LightGlue.** Their RANSAC stage aligns 90 pairs,
  where the committed `lg` aligns 89. That is GPU non-determinism in the matcher, and it
  is why `lg` itself is not re-registered here.

## 3. Why pairs fall back

**The failure taxonomy of the pilot's `lg`.** It is read from `lg_hyb`'s first stage,
which is the same RANSAC ground homography on the same kind of matches.

| cause | pairs |
|---|---|
| aligned | 90 |
| matches exist, but < 15 on the ground | **122** |
| ≥ 15 ground matches, but no plane with 15 inliers | 77 |
| < 15 matches anywhere in the view | 9 |
| plane maps the GT point outside the view | 2 |

So the views usually *do* match, on buildings, poles and trees. The ground between them
does not.

**Fallback rate by covariate** (`lg` → `roma`):

| covariate | stratum (n) | lg | roma |
|---|---|---|---|
| baseline | < 10 m (72) | 0.29 | 0.01 |
| | 10–20 m (146) | 0.74 | 0.03 |
| | ≥ 20 m (82) | **1.00** | 0.20 |
| capture month | same (191) | 0.64 | 0.06 |
| | different (109) | 0.82 | 0.08 |
| imagery | GSV (240) | 0.76 | 0.08 |
| | Mapillary (60) | 0.48 | 0.03 |
| range to ramp | 0–6 / 6–12 / 12–18 m | 0.76 / 0.74 / 0.64 | 0.05 / 0.08 / 0.07 |

Baseline dominates. Beyond 20 m, `lg` never aligns. Mapillary's lower fallback matches
its shorter baselines between consecutive frames of a sequence. RoMa's remaining fallbacks
are almost all at ≥ 20 m. In all 21 of them it found a plane, but that plane maps the GT
point outside the other view.

**What the failures look like.** This was an informal visual pass on 18 pairs, 6 in each
group, sampled with seed 48 in scratch. The imagery is not committed. The pass drew the
reference too, so it is post hoc and is description, not measurement.

- **`lg` fallbacks with matches off the ground** (p062, p105, p162, p167, p168, p221):
  baselines of 21–27 m. In most, the two cameras look at the corner from different
  streets, so the shared ground is a narrow strip seen at grazing angles from opposite
  sides. What is shared is plain asphalt, and parked cars cover part of it. Only one
  (p105, 2020 → 2024) also spans years.
- **`lg` fallbacks with no plane** (p092, p097, p126, p169, p236, p265): the ground band
  holds several surfaces (road, raised sidewalk, curb faces, grass), plus cars and people.
  No one plane explains 15 of them.
- **Where `roma` lands > 2° further than auto height on GSV** (74 of 240 pairs; sampled
  p095, p138, p144, p267, p271, p287):
  - Three of the six are **crosswalk stripes**. A repetitive pattern lets a plausible
    homography slip by one stripe.
  - Two are corners where road, curb and raised brick sidewalk are different planes.
  - One has the reference on a different part of the corner than the source point.
  - In all six the homography is fitted over the whole ground band, so it averages over
    surfaces the ramp is not on. That is the case for reading the dense warp at the
    point itself (`roma_warp`) or fitting locally (`roma_local`).

## 4. Pre-specified vs post hoc

**Pre-specified within this family.** These were fixed before any arm in this family was
scored:

- the matchers and their native settings (LoFTR 0.5, RoMa 5,000 samples);
- the four estimators;
- all thresholds, which are inherited from the pilot and not re-tuned: the 5° band, 4 px,
  15 inliers, and the 1 px essential-matrix threshold (OpenCV's default).

**But the 5° band is post hoc at the pilot level,** and every arm in this family inherits
it. It was chosen for `lg` after the pre-specified 0.5° band failed (`lg_band0.5` is that
row), on pilot pairs that are a subset of these 300. `lg` is marked post hoc for it in the
combined table and in its `meta.json`. The other matching arms are not marked one by one,
but read every number in this file as conditional on that one post hoc choice.

**Post hoc: `roma_local`, `roma_warp` and `roma_warp_hyb`.** They were added after `roma`
was scored and found not to beat auto height on GSV. Their parameters were not tuned: they
are `lg_local`'s radius and minimum, and romatch's default 0.05 `sample_thresh`, borrowed
as a certainty cutoff. The
choice to try them, however, followed a look at the results. Treat their gains as
hypotheses for the next pair set, not as established. No constant in this file was
changed after scoring, so there is no separate "tuned" row.

Their committed `.meta.json` files recorded `"pre_specified": true`, inherited from the
family's shared `BASE_CONFIG` / `WARP_CONFIG`. That flag was wrong. The as-run config is
left unchanged, and each file now carries a top-level `provenance_correction` marking the
arm post hoc; the code records `false` for any future run.

## 5. Negative and null results

- **Other sparse features do not help.** SuperPoint, DISK and SIFT with LightGlue all fall
  back more often than ALIKED (74 / 84 / 85% vs 70%). Where they align they are no better
  than auto height. The pilot's informal 30-pair DISK probe pointed the same way; this is
  now the full-set result.
- **LoFTR is harmful where it aligns.** On its 42 aligned pairs it gives 6.30° vs 3.58°
  for auto (−2.70° [−5.05, −0.19]), and on GSV −4.51°. Its coarse matches on low-texture
  asphalt produce confident wrong planes.
- **MAGSAC++ vs RANSAC:** no difference in fallback or accuracy for any matcher.
- **The epipolar snap is a no-op.** It uses the other ~50–90% of matches (buildings and
  trees), so it aligns up to 98% of pairs (`roma_epi`). The snap moves the auto-height
  point a median 16–20 px (~1.5°; p90 60–100 px), yet the paired gain against auto height
  on the aligned pairs is 0.00–0.03° overall (e.g. `roma_epi` 0.03° [0.00, 0.08]). The
  harm rate is 10–16% and the help rate 0–4%: a move of more than 2° is more often wrong than right.
  - The ambiguity that matters is *along* the line, which is exactly the range/height
    error, and the line cannot resolve it.
  - Mapillary is the one exception: `roma_epi` gives 0.34° [0.13, 0.60] there, a real but
    small gain.
- **The hybrids inherit their first stage.** `lg_hyb` and `sp_lg_hyb` end up at auto
  height (4.08° / 4.13° vs 4.06°). `roma_hyb` is worse than `roma` on GSV (4.83°), because
  the pairs where RoMa's plane leaves the view get the epipolar snap instead of auto.

## 6. Not run, and why

- **eLoFTR, DKM:**
  - DKM is RoMa's predecessor from the same group, so RoMa supersedes it.
  - eLoFTR needs its own repository and checkpoint.
  - LoFTR's result (§5) gave no reason to expect a semi-dense LoFTR variant to fix
    low-texture ground.
- **MASt3R matching head:** not run. It needs its repository, a ~2.7 GB checkpoint and more
  than the 3070's 8 GB at its native resolution, so it would have to run on makelab2. It is
  the natural next matcher to try on GSV, because it regresses 3D pointmaps rather than
  fitting a plane.
- **Wider, tilted-down or bird's-eye views; multi-scale:** not run. The fallback these were
  meant to fix is already gone with RoMa at the existing views (7%). The GSV error that
  remains comes from wrong planes, not missing coverage (§3). A wider view adds *more*
  non-ground-plane surface to the band. A bird's-eye rendering would also need each pano's
  heading, which the view-based harness does not carry.
- **Auto height as the view centre:** not run, for the reason in §1. It is used as a prior
  and a fallback instead.

## 7. Reproduction

Inputs: the harness views (`cut-views` output, 600 JPEGs, on makelab2 at
`/homes/gws/jonf/crossview48/views`; not published — see `docs/crossview_align_48.md` §9
for what would unblock that) and the committed `proj_height_auto` predictions.

Packages: these are not in `requirements.txt`. They were installed with `--no-deps` into a
scratch directory, so the shared venv was not changed:

- `lightglue` from `git+https://github.com/cvg/LightGlue.git@eb42fee2d71449efb0aa5c10549752b5d75384d8`;
- `romatch==0.1.2` with `loguru==0.7.3`, `win32_setctime` on Windows, and `kornia==0.8.3`
  (as for the pilot).

The venv already had torch 2.6.0+cu126, torchvision 0.21, timm 1.0.28, einops 0.8.2 and
opencv 5.0.0. Weights download on first use: LightGlue's GitHub releases, romatch's
releases and DINOv2.

```bash
export PYTHONPATH=<scratch pkgs>
for a in lg_magsac lg_epi lg_hyb sp_lg sp_lg_magsac sp_lg_epi sp_lg_hyb \
         disk_lg disk_lg_magsac disk_lg_epi disk_lg_hyb siftlg siftlg_magsac siftlg_epi siftlg_hyb \
         loftr loftr_magsac loftr_epi loftr_hyb roma roma_magsac roma_epi roma_hyb roma_local \
         roma_warp roma_warp_hyb; do
  python scripts/analysis/crossview_align_48.py predict --arm $a --views VIEWS \
      --extra match_cache=CACHE_DIR
done
python scripts/analysis/crossview_matching_48.py       # -> matching/report.json, tables
pytest -q tests/test_crossview_matching_48.py
```

`--extra match_cache` stores each matcher's raw matches per pair, so the estimator variants
reuse one network run.

- **Without the cache,** each variant re-runs its matcher. Results can then differ by a
  pair or two, from GPU non-determinism (§2).
- **Order matters.** The first arm of each matcher group fills the cache: `lg_magsac`,
  `sp_lg`, `disk_lg`, `siftlg`, `loftr`, `roma` and `roma_warp`. The table above lists them
  in that order.

The shared `results.json` was **not** regenerated on this branch, per the family-branch
rule. So `tests/test_crossview_align_48.py::test_committed_results_rederive_from_committed_predictions`
fails here until `score` is re-run on the merged set of predictions.

## 8. Cost

Everything ran on the desktop RTX 3070. There was no makelab2 GPU, no klone, no Tillicum
and no paid API.

| network run (fills the cache for) | wall-clock |
|---|---|
| ALIKED + LG (`lg_magsac`, `lg_epi`, `lg_hyb`) | 45 s |
| SuperPoint + LG (`sp_lg*`) | 52 s |
| DISK + LG (`disk_lg*`) | 72 s |
| SIFT + LG (`siftlg*`) | 126 s |
| LoFTR (`loftr*`) | 92 s |
| RoMa samples (`roma*`, `roma_local`) | 466 s |
| RoMa dense warp (`roma_warp*`) | 311 s |
| estimator-only arms on cached matches (19 arms) | 139 s total, CPU-bound |

The total is 1,303 s ≈ **0.36 GPU-hours** (an upper bound: it counts the CPU estimation
time as GPU time), at $0. The seven network runs have `paid: false` rows in
`analysis_out/usage_log.jsonl`. The per-arm wall-clock is in each `.meta.json`.
