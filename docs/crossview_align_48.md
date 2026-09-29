# Placing a GT ramp point in other panoramas ([#48](https://github.com/ProjectSidewalk/RampNet/issues/48)): alignment harness and first arms

A pilot, and the shared harness for comparing techniques. Code:
`scripts/analysis/crossview_align_48.py` (harness) and `scripts/analysis/crossview_arms/`
(the techniques, called "arms"). Outputs: `analysis_out/crossview_align_48/`. Tests:
`tests/test_crossview_align_48.py`. Run 2026-09-28 on free compute only.

## Summary

- **All five arm families, combined (2026-09-29):** the best single arm is MapAnything on the
  pair with the pose priors (`mapa_posed_pair`), median 2.80° [2.49, 3.07] with no fallback
  and a CI-clear gain over `proj_height_auto` on GSV and Mapillary alike. It is one of only
  two arms whose GSV gain survives a Bonferroni correction over all 83 arms.
  **It is a post hoc arm.** It was registered after `mapa_posed_corner`'s 300-pair result
  had been seen, and it never had a pilot. So 2.80° is an in-sample result on the pairs that
  suggested it. It needs confirmation, with its settings fixed in advance, on fresh pairs
  drawn from the 1,423 eligible pairs not in the 300 before anyone adopts it. That re-test
  has not been run. See
  [Combined comparison across all arm families](#combined-comparison-across-all-arm-families)
  for the table and its caveats.
- **The rest of this Summary, and §4–§8, are the pilot (13 arms, 2026-09-28).** Where the
  pilot's readings are superseded by the combined table, the bullet says so.
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
  Among the pilot's 13 arms it was the only one that helped Richmond: 3.05° vs 4.56° over all
  60 Mapillary pairs, and 2.10° vs 4.54° on the 31 it aligns. *Superseded:* in the combined
  table 18 arms have a lower Mapillary median than `lg`, from 1.93° (`mast3r_pair`) to 2.68°.
- **These did not help:**
  - Mapillary SfM pitch/roll (`proj_mly_gravity`, `proj_mly_road`): no gain.
  - Per-pano measured GSV heights (`proj_height_perpano`) and GSV depth-map range
    (`proj_gsv_depth`): lower medians, but more pairs got worse than better. Their paired
    gains are ≤ 0, and the share within 2° drops.
  - SIFT and NCC baselines: worse than projection.
- **The Richmond run already uses SfM.** Its positions are Mapillary's SfM
  `computed_geometry` and its headings are `computed_compass_angle`. Raw GPS makes it worse
  (`proj_mly_rawgps`, paired gain −0.52° [−1.36, −0.06]).
- **Pilot verdict (proposed, not decided; superseded by the combined takeaways):**
  - For gallery rings, use `proj_height_auto` everywhere. Where LightGlue aligns, use its
    point instead, and mark which ring is which. The combined comparison replaces this with
    `mapa_posed_pair` as the candidate, pending its fresh-pair re-test.
  - As an association signal for the labeler's clustering
    ([labeler#56](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/56)),
    image alignment is too often silent to be the primary signal, and it is untested as a
    discriminator (§7).
- **The harness is pluggable.** A new technique is one function plus one `@register` line
  in a new file under `crossview_arms/`. It is scored on the same frozen pairs with the same
  metrics (§5).

## Combined comparison across all arm families

Added 2026-09-29, when the five family branches were merged into this one
([#211](https://github.com/ProjectSidewalk/RampNet/pull/211),
[#212](https://github.com/ProjectSidewalk/RampNet/pull/212),
[#213](https://github.com/ProjectSidewalk/RampNet/pull/213),
[#215](https://github.com/ProjectSidewalk/RampNet/pull/215),
[#216](https://github.com/ProjectSidewalk/RampNet/pull/216)). Each family has its own write-up
with the full arm list, method and caveats:

- semantic / structural: [`crossview_align_48/semantic.md`](crossview_align_48/semantic.md)
- pairwise image matching: [`crossview_align_48/matching.md`](crossview_align_48/matching.md)
- monocular metric depth: [`crossview_align_48/depth.md`](crossview_align_48/depth.md)
- multi-view 3D (SfM, feed-forward 3D, splatting): [`crossview_align_48/multiview_3d.md`](crossview_align_48/multiview_3d.md)
- flat Mapillary imagery, Richmond only ([#214](https://github.com/ProjectSidewalk/RampNet/issues/214)): [`flat_mapillary_3d.md`](flat_mapillary_3d.md)

**Takeaways (proposed, not decided).**

- **Best single arm: MapAnything on the pair, given the pose priors (`mapa_posed_pair`), and
  it is post hoc.** Median 2.80° [2.49, 3.07] over all 300 pairs, against 4.06° for
  `proj_height_auto` and 5.62° for today's projection. It never falls back. Its paired gain
  over auto is CI-clear on both imageries: +0.63° [0.28, 0.93] on GSV and +1.73°
  [0.83, 2.77] on Mapillary, and the GSV and all-pairs gains survive the Bonferroni screen
  below (the Mapillary stratum was not screened). It was added after its sibling `mapa_posed_corner` (2.86°, not
  post hoc) had been scored on these pairs (commit `0428bb7`), so its numbers are in-sample.
  `mapa_posed_corner` is the pre-specified arm closest to it. Both need the fresh-pair
  re-test before adoption.
- **On GSV, only the two posed MapAnything arms beat auto once the 83-arm multiplicity is
  applied.** Without correction, five of the 83 shared arms have a GSV gain whose 95% CI
  clears zero (`combined_table.json` → `gsv_ci_clear_vs_auto`): `mapa_posed_pair` +0.63°,
  `mapa_posed_corner` +0.51°, `mapa_k_pair` +0.33°, `sem_chamfer_auto` +0.34°, and
  `mapa_posed_poseonly` by a negligible +0.05°. A one-sided Bonferroni screen over all 83
  arms (20,000 ramp resamples, seed 48; `combined_table.json` → `multiplicity`) keeps only
  `mapa_posed_pair` (lower bound +0.16°) and `mapa_posed_corner` (+0.08°). `mapa_k_pair`
  (−0.13°), `sem_chamfer_auto` (−0.12°) and `mapa_posed_poseonly` (−0.01°) do not survive:
  gains of their size are what selection alone would produce. Bonferroni ignores the strong
  correlation between arms, so it is conservative, and a non-survivor is not shown to be
  null. Over all 300 pairs the survivors are those two plus `mapa_k_pair` (+0.14°) and
  `mapa_posed_poseonly` (+0.02°). Every matching arm, every depth arm and per-corner COLMAP
  ties or loses to the free per-rig height on GSV.
- **`sem_chamfer_auto` is demoted to a candidate, and it carries a leakage path that has not
  been measured.** Vistas' Curb Cut class is suppressed before the argmax, but the pixels it
  would have won are relabelled to the next-best class, mostly sidewalk or road. So the
  ramp's outline still shapes the curb-edge channel the chamfer aligns. Curb Cut would have
  won pixels in 298 of 300 other views and 296 of 300 source views (median about 2,000 px per
  view, up to 16% of a view; `semantic_seg_manifest.json`). The reference is a ramp detection,
  so this is a weaker form of the circularity the suppression was meant to remove.
  **Follow-up, not run:** re-run as `sem_chamfer_auto_ccmask`, with the unsuppressed Curb
  Cut mask (dilated a few px) mapped to an ignore label in neither `RAISED` nor `ROADLIKE`
  (about 3 min on the A40). See `crossview_align_48/semantic.md` §4.
- **On Mapillary (Richmond), the methods that reconstruct or densely match both views roughly
  halve the error:** 1.9–2.7° for the multi-view 3D arms, RoMa and per-corner SfM, against
  4.56° for the projection (auto is the projection there). Most sparse pairwise matchers do
  not: `lg` 3.05°, `sp_lg` 3.41°, `disk_lg` and `siftlg` 3.97°, `lg_epi` 4.30°, `sift`
  4.56°, `loftr` 4.65°, `ncc` 11.54°. The flat Mapillary images add nothing to per-corner
  SfM: `flat_sfm` 2.53° vs the no-flat control `noflat_sfm` 2.62°, paired +0.00°
  [−0.01, 0.05] over all 60 pairs, and +0.02° [−0.04, 0.13] over the 47 pairs whose corner
  has flat images (`flat_mapillary_3d.md` §5).
- **Negatives that hold across families:** the learned relative pose alone
  (`mast3r_poseonly`), one view alone (`mapa_mono_depthonly`), raw Depth Pro range, raw line
  segments without semantics (`lsd_chamfer`), LoFTR, and dense depth read from a splat
  (`flat_gs`) are all worse than auto.

**How to read the table.**

- "gain vs auto" is the median per-pair reduction in error against `proj_height_auto`, with
  a CI over 2,000 resamples of ramps. On Mapillary auto equals the projection, so there it
  is also the gain over today's projection.
- An arm that returns nothing falls back to the 2.6 m projection (harness §3), so a high
  fallback rate dilutes its all-pairs numbers. The "where it answers" column shows the arm on
  the pairs where it answered.
- **Two gain definitions are in use.** The gain columns here are over **all** pairs of the
  stratum, with a fallback scored as the projection. The family docs (e.g.
  `multiview_3d.md`) usually quote the gain on the pairs the arm answered, which is this
  table's "where it answers" column. So `mapa_k_pair` is +0.72 [0.28, 1.07] here and
  +0.73 [0.30, 1.09] in its family doc, and `mast3r_pair` +0.45 here and +0.57 there.
- **One common fallback rule.** The `_hyb` and `_else_auto` composites use auto's point
  instead of falling back, so they report 0% fallback while 42–43% of their answers are
  auto. The last two columns put every arm on one rule: a fallback is scored at
  `proj_height_auto`'s point, and the last column is the share of pairs scored at the auto
  prior (fallbacks plus a hybrid's own auto answers). Under that rule `roma_warp` and
  `roma_warp_hyb` are identical (3.11°, share 0.42), and `lg` goes from 4.79° to 3.79°.
- The composites' median gain over auto is +0.00 because the pairs where they used auto
  (a gain of exactly zero) straddle the median. That is an empirical outcome, not a matter
  of construction: they used auto on 42% (`roma_warp_hyb`) and 43%
  (`mv3d_consensus_else_auto`) of pairs, under half, but neither the positive nor the
  negative share exceeds 50%. Read their component (`roma_warp`, `mv3d_consensus`) in the
  "where it answers" column.
- **Subset arms are not ranked on 300 pairs.** The flat-Mapillary arms run only on the 60
  Richmond pairs and return null elsewhere. The harness would count those 240 as fallbacks
  and report an all-pairs median as if they had run everywhere. They are therefore scored on
  the Richmond stratum only (`flat_mapillary_48.py score` → `results_richmond.json`) and
  their all-pairs and GSV cells are empty. They are kept out of the shared `results.json`,
  which reads only `crossview_align_48/predictions/`.
- "post hoc" marks an arm, or a setting of it, that was added, or picked for this table,
  after scores on these same 300 pairs (or a subset of them, such as a 30-pair pilot) had
  been seen. Corrected 2026-09-29 by checking each arm against git history (review of
  [#210](https://github.com/ProjectSidewalk/RampNet/pull/210)):
  - `lg`'s 5° ground band, which every matching-family arm inherits (`matching.md` §4);
    `sem_snap_auto`;
  - `roma_local`, `roma_warp`, `roma_warp_hyb`: added after `roma` did not beat auto on GSV.
    Their committed `meta.json` recorded `pre_specified: true` (inherited from the family's
    shared config); a top-level `provenance_correction` now supersedes it;
  - `mono_da3_hcal` and `mono_unidepth_point`: each picked from the 17 depth arms;
  - **`mapa_posed_pair`** and `mapa_posed_poseonly`: registered in `0428bb7`, whose commit
    message quotes `mapa_posed_corner`'s 300-pair result; neither had a pilot;
  - `mapa_mono_depthonly`: registered later still (`c84dc74`);
  - `sfm_colmap`: its gravity-level ground lift replaced the plane lift after the 30-pair
    pilot, whose pairs are a subset of the 300;
  - `mast3r_poseonly`: registered in `bb6cd3c` at 18:18, three minutes after `0214816`
    quoted the pilot's scores. That the pilot was seen first is **inferred from commit
    times**; no commit message says so;
  - the two `mv3d_consensus` composites.

  Post hoc by the same rule but not rows of this table (final re-review of #210, N4):
  `sfm_colmap_prior` (the same lift change as `sfm_colmap`, `0214816`); `sfm_poseonly`
  (registered in `0214816`, whose message quotes the pilot's scores); `vggt_corner_poseonly`
  (registered in `6695b4f`, whose message quotes the pilot's `mapa_posed_depthonly`);
  `sem_curb_shift_auto` (`semantic.md` §1); and the flat family's three `_gsmed` arms. None
  survives either screen, and all are negatives or non-headline, so no conclusion changes.

  `combined_table.json` → `rows[].post_hoc_why` gives the reason for each table row, and
  `crossview_combined_48.POST_HOC_NOT_IN_TABLE` for the rest. Every one of these arms'
  `meta.json` carries a top-level `provenance_correction` with the same reason (a test checks
  this). No prediction `.jsonl` changed.

Built by `scripts/analysis/crossview_combined_48.py` (CPU, committed inputs only, about
2.5 minutes with the 20,000-resample Bonferroni screen) →
`analysis_out/crossview_align_48/combined_table.json`. Every all-pairs median and
fallback rate matches `results.json`; the Richmond cells of the flat arms match
`results_richmond.json`.

| arm | family | post hoc | all 300: median ° [CI] | fallback | paired gain vs auto, all [CI] | GSV (240): median ° [CI] | GSV gain vs auto [CI] | Mapillary (60): median ° [CI] | Mapillary gain vs auto [CI] | where it answers: n, arm vs auto °, gain vs auto [CI] | common rule (fallback → auto), all 300: median °, gain vs auto [CI] | share scored at the auto prior |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `projection` | baseline |  | 5.62 [4.53, 6.56] | 0.00 | -0.00 [-0.43, 0.00] | 6.09 [4.53, 7.03] | -0.67 [-1.20, -0.25] | 4.56 [3.87, 5.57] | +0.00 [0.00, 0.00] | (never falls back) | 5.62, -0.00 [-0.43, 0.00] | 0.00 |
| `proj_height_auto` | baseline |  | 4.06 [3.66, 4.56] | 0.00 | – | 3.92 [3.53, 4.36] | – | 4.56 [3.87, 5.57] | – | – | – | – |
| `proj_gsv_depth` | geometry (pilot) |  | 4.82 [4.50, 5.65] | 0.20 | -0.13 [-0.68, -0.00] | 4.96 [4.55, 5.89] | -0.84 [-1.26, -0.36] | 4.56 [3.87, 5.57] | +0.00 [0.00, 0.00] | 240: 4.96 vs 3.92, -0.84 [-1.26, -0.36] | 4.82, -0.13 [-0.68, 0.00] | 0.20 |
| `lg` | matching (pilot) | yes | 4.79 [3.51, 6.01] | 0.70 | -0.00 [-0.44, 0.00] | 5.91 [4.15, 6.68] | -0.52 [-1.20, -0.11] | 3.05 [2.39, 4.33] | +0.00 [0.00, 0.47] | 89: 2.41 vs 2.99, +0.45 [-0.26, 0.93] | 3.79, +0.00 [0.00, 0.00] | 0.70 |
| `sem_chamfer_auto` | semantic |  | 3.30 [2.95, 4.11] | 0.01 | +0.42 [0.14, 0.71] | 3.23 [2.81, 4.12] | +0.34 [0.02, 0.56] | 3.84 [2.73, 4.24] | +0.75 [0.38, 1.68] | 297: 3.28 vs 4.05, +0.43 [0.15, 0.71] | 3.30, +0.40 [0.14, 0.70] | 0.01 |
| `lsd_chamfer` | semantic |  | 4.92 [4.28, 6.04] | 0.04 | -0.45 [-0.74, -0.13] | 5.31 [4.38, 6.69] | -0.70 [-1.51, -0.32] | 3.71 [2.93, 5.44] | +0.31 [0.00, 1.10] | 289: 4.89 vs 4.06, -0.46 [-0.74, -0.12] | 4.88, -0.30 [-0.69, -0.01] | 0.04 |
| `sem_snap_auto` | semantic | yes | 3.83 [3.44, 4.36] | 0.05 | -0.00 [-0.23, 0.16] | 3.86 [3.44, 4.41] | -0.18 [-0.40, 0.12] | 3.60 [2.92, 5.24] | +0.15 [-0.00, 0.45] | 284: 3.78 vs 3.96, -0.08 [-0.27, 0.19] | 3.82, +0.00 [-0.22, 0.09] | 0.05 |
| `roma` | matching |  | 3.51 [2.92, 4.70] | 0.07 | +0.03 [-0.22, 0.43] | 4.41 [3.26, 5.79] | -0.19 [-0.91, 0.12] | 2.58 [2.16, 3.38] | +1.75 [0.83, 2.77] | 279: 3.51 vs 4.05, +0.12 [-0.19, 0.48] | 3.54, +0.00 [-0.07, 0.26] | 0.07 |
| `roma_local` | matching | yes | 3.29 [2.92, 4.21] | 0.06 | +0.30 [-0.10, 0.66] | 4.06 [3.19, 5.18] | -0.07 [-0.48, 0.34] | 2.37 [1.83, 2.94] | +1.87 [0.98, 2.88] | 283: 3.23 vs 4.03, +0.34 [-0.03, 0.67] | 3.29, +0.21 [0.00, 0.47] | 0.06 |
| `roma_warp` | matching | yes | 3.25 [2.73, 4.43] | 0.42 | +0.12 [-0.00, 0.54] | 3.99 [2.79, 5.70] | +0.03 [-0.36, 0.43] | 2.58 [1.71, 3.33] | +1.22 [0.00, 2.74] | 174: 2.11 vs 3.48, +0.58 [0.35, 1.21] | 3.11, +0.00 [0.00, 0.00] | 0.42 |
| `roma_warp_hyb` | matching | yes | 3.11 [2.75, 3.93] | 0.00 | +0.00 [0.00, 0.00] | 3.58 [2.82, 4.28] | +0.00 [0.00, 0.00] | 2.58 [1.71, 3.33] | +1.22 [0.00, 2.74] | (never falls back) | 3.11, +0.00 [0.00, 0.00] | 0.42 |
| `sp_lg` | matching |  | 4.98 [3.81, 6.18] | 0.74 | -0.07 [-0.46, 0.00] | 6.02 [4.55, 6.91] | -0.62 [-1.20, -0.22] | 3.41 [2.86, 3.92] | +0.00 [0.00, 0.00] | 78: 2.56 vs 2.90, +0.15 [-0.21, 0.94] | 3.79, +0.00 [0.00, 0.00] | 0.74 |
| `loftr` | matching |  | 6.03 [4.98, 6.96] | 0.86 | -0.33 [-0.79, -0.00] | 6.53 [5.35, 7.67] | -0.79 [-1.63, -0.35] | 4.65 [3.50, 5.87] | +0.00 [-0.00, 0.00] | 42: 6.30 vs 3.58, -2.70 [-5.05, -0.19] | 4.28, +0.00 [0.00, 0.00] | 0.86 |
| `mono_da3_hcal` | depth | yes | 3.75 [3.24, 4.57] | 0.10 | +0.16 [-0.03, 0.47] | 3.58 [3.08, 4.57] | +0.08 [-0.25, 0.40] | 4.19 [3.24, 5.23] | +0.39 [0.00, 1.04] | 269: 4.04 vs 4.19, +0.22 [-0.04, 0.61] | 3.75, +0.03 [0.00, 0.29] | 0.10 |
| `mono_unidepth_point` | depth | yes | 4.04 [3.48, 4.73] | 0.00 | +0.18 [-0.15, 0.47] | 4.18 [3.59, 4.98] | -0.01 [-0.46, 0.37] | 3.44 [2.71, 4.56] | +0.86 [0.47, 1.31] | (never falls back) | 4.04, +0.18 [-0.15, 0.47] | 0.00 |
| `mono_depthpro_point` | depth |  | 19.23 [17.57, 22.51] | 0.00 | -14.43 [-16.76, -12.72] | 20.81 [18.65, 24.66] | -15.39 [-18.54, -13.24] | 16.55 [13.87, 19.50] | -10.78 [-14.70, -8.44] | (never falls back) | 19.23, -14.43 [-16.76, -12.72] | 0.00 |
| `mapa_posed_pair` | multi-view 3D | yes | 2.80 [2.49, 3.07] | 0.00 | +0.72 [0.42, 1.01] | 2.85 [2.58, 3.39] | +0.63 [0.28, 0.93] | 2.26 [1.89, 3.01] | +1.73 [0.83, 2.77] | (never falls back) | 2.80, +0.72 [0.42, 1.01] | 0.00 |
| `mapa_posed_corner` | multi-view 3D |  | 2.86 [2.56, 3.31] | 0.00 | +0.61 [0.36, 0.88] | 3.00 [2.58, 3.39] | +0.51 [0.23, 0.84] | 2.68 [1.95, 3.71] | +0.94 [0.42, 2.44] | (never falls back) | 2.86, +0.61 [0.36, 0.88] | 0.00 |
| `mapa_k_pair` | multi-view 3D |  | 2.99 [2.42, 3.67] | 0.01 | +0.72 [0.28, 1.07] | 3.28 [2.61, 3.96] | +0.33 [0.12, 0.77] | 2.16 [1.47, 2.90] | +2.60 [1.09, 3.23] | 297: 2.96 vs 4.05, +0.73 [0.30, 1.09] | 2.99, +0.72 [0.26, 1.07] | 0.01 |
| `mast3r_pair` | multi-view 3D |  | 3.20 [2.66, 3.78] | 0.11 | +0.45 [0.06, 0.74] | 3.66 [2.88, 4.29] | +0.12 [-0.25, 0.47] | 1.93 [1.57, 3.19] | +2.22 [1.07, 3.06] | 268: 2.96 vs 4.00, +0.57 [0.18, 1.02] | 3.09, +0.32 [0.00, 0.61] | 0.11 |
| `dust3r_pair` | multi-view 3D |  | 3.08 [2.59, 3.59] | 0.02 | +0.47 [0.16, 0.79] | 3.30 [2.93, 4.04] | +0.24 [-0.07, 0.48] | 2.11 [1.55, 2.75] | +2.60 [1.28, 3.07] | 294: 3.07 vs 4.04, +0.47 [0.17, 0.83] | 3.07, +0.44 [0.12, 0.79] | 0.02 |
| `vggt_pair` | multi-view 3D |  | 3.51 [2.92, 4.73] | 0.15 | +0.41 [-0.06, 0.77] | 4.44 [3.34, 5.67] | +0.10 [-0.55, 0.44] | 2.06 [1.52, 2.96] | +1.97 [1.30, 3.26] | 256: 3.35 vs 4.01, +0.41 [-0.16, 0.88] | 3.61, +0.00 [0.00, 0.26] | 0.15 |
| `mv3d_consensus` | multi-view 3D | yes | 3.11 [2.69, 4.16] | 0.43 | +0.32 [-0.00, 0.64] | 3.93 [2.95, 5.27] | +0.01 [-0.49, 0.41] | 1.98 [1.45, 2.82] | +1.77 [0.87, 2.97] | 172: 2.11 vs 3.48, +0.84 [0.42, 1.24] | 3.03, +0.00 [0.00, 0.00] | 0.43 |
| `mv3d_consensus_else_auto` | multi-view 3D | yes | 3.03 [2.58, 3.72] | 0.00 | +0.00 [0.00, 0.00] | 3.54 [2.86, 4.13] | +0.00 [0.00, 0.00] | 1.98 [1.45, 2.82] | +1.77 [0.87, 2.97] | (never falls back) | 3.03, +0.00 [0.00, 0.00] | 0.43 |
| `mapa_posed_poseonly` | multi-view 3D | yes | 3.90 [3.45, 4.40] | 0.00 | +0.07 [0.03, 0.10] | 3.80 [3.27, 4.37] | +0.05 [0.02, 0.09] | 4.14 [3.72, 5.45] | +0.17 [0.07, 0.25] | (never falls back) | 3.90, +0.07 [0.03, 0.10] | 0.00 |
| `mast3r_poseonly` | multi-view 3D | yes | 4.43 [3.69, 5.71] | 0.08 | -0.24 [-0.47, -0.02] | 4.81 [3.69, 5.96] | -0.45 [-0.70, -0.15] | 4.08 [3.15, 5.08] | +0.59 [0.28, 1.26] | 277: 4.36 vs 4.05, -0.19 [-0.44, 0.01] | 4.38, -0.07 [-0.31, 0.00] | 0.08 |
| `mapa_mono_depthonly` | multi-view 3D | yes | 4.94 [4.37, 5.53] | 0.00 | -0.50 [-1.03, -0.04] | 4.95 [4.31, 5.68] | -0.54 [-1.39, -0.04] | 4.81 [3.39, 6.36] | -0.42 [-1.11, 0.66] | (never falls back) | 4.94, -0.50 [-1.03, -0.04] | 0.00 |
| `sfm_colmap` | multi-view 3D | yes | 4.90 [3.81, 6.22] | 0.29 | -0.30 [-0.76, 0.02] | 5.86 [4.39, 7.38] | -0.67 [-1.51, -0.24] | 2.52 [2.06, 3.73] | +1.70 [0.25, 2.81] | 212: 4.99 vs 4.33, -0.30 [-0.84, 0.07] | 4.14, +0.00 [0.00, 0.00] | 0.29 |
| `flat_sfm` | flat Mapillary (Richmond only) |  | not run (Richmond only) | 0.00 (Richmond) | – | n/a | n/a | 2.53 [1.79, 3.97] | +1.74 [0.62, 2.71] | (never falls back) | 2.53, +1.74 [0.62, 2.71] (Richmond) | 0.00 |
| `noflat_sfm` | flat Mapillary (Richmond only) |  | not run (Richmond only) | 0.00 (Richmond) | – | n/a | n/a | 2.62 [2.18, 3.52] | +1.69 [0.54, 2.55] | (never falls back) | 2.62, +1.69 [0.54, 2.55] (Richmond) | 0.00 |
| `mlypano_sfm` | flat Mapillary (Richmond only) |  | not run (Richmond only) | 0.00 (Richmond) | – | n/a | n/a | 2.70 [2.18, 3.30] | +1.65 [0.55, 2.63] | (never falls back) | 2.70, +1.65 [0.55, 2.63] (Richmond) | 0.00 |
| `flat_mvs` | flat Mapillary (Richmond only) |  | not run (Richmond only) | 0.05 (Richmond) | – | n/a | n/a | 3.95 [2.65, 7.26] | +0.26 [-1.40, 2.07] | 57: 4.44 vs 4.61, +0.66 [-2.02, 2.22] | 3.95, +0.26 [-1.40, 2.07] (Richmond) | 0.05 |
| `flat_gs` | flat Mapillary (Richmond only) |  | not run (Richmond only) | 0.02 (Richmond) | – | n/a | n/a | 10.48 [5.22, 21.21] | -7.33 [-12.57, -0.33] | 59: 10.71 vs 4.57, -7.39 [-13.70, -0.77] | 10.48, -7.33 [-12.57, -0.33] (Richmond) | 0.02 |

**Caveats that travel with this table.**

- **The reference is a detection, not a ground truth.** Both the source point and the
  reference are RampNet detection peaks. On manual_gold a peak sits a median 1.51° from the
  human box centre (§2), so errors of about 2° are at the floor this set can resolve. The
  Mapillary medians near 2° are at that floor, and differences among them are not resolvable
  here.
- **Pair selection truncates the projection's error.** A pair qualifies only if the reference
  raycasts within 5 m of the source point at a flat 2.6 m (§2). The worst projection failures
  are not scored, so gains over the projection, and less so over auto, are *likely*
  conservative. That is an inference, not a measurement: the excluded pairs have larger
  projection errors, but no arm's error on them is known, and an arm may also fail more
  often on geometrically hard pairs.
- **No negative pairs.** Every pair is a true match. Nothing here tests whether an arm can
  tell this ramp from a different ramp nearby, which is what an association signal for
  [labeler#56](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/56) needs (§7).
- **Mapillary is one city:** Richmond, 60 pairs on 31 ramps. Every Mapillary conclusion,
  including the flat-imagery negative, is a Richmond result.
- **Many arms were scored on the same 300 pairs, and some were added post hoc, including the
  headline arm.** Across 83 shared arms, a few CI-clear gains are expected by selection
  alone; the Bonferroni screen above says which survive it. Neither the correction nor the
  CIs account for arms being *added* after results were seen. **Follow-up, not run:** re-test
  `mapa_posed_pair`, `mapa_posed_corner` and `mapa_k_pair`, with settings fixed in advance,
  on fresh pairs from the 1,423 eligible pairs (`eligible_pairs.csv`) not in the 300,
  preferring ramps not among the 174 already used.

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
  other view has a second ≥ 0.55 detection landing within 8 m. That 8 m is measured from
  the **pool ramp's** position, while the 5 m world test (`ref_world_gap_m`) and the 18 m
  range are measured from the source view's own 2.6 m raycast.
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
  geometry already roughly works, and the projection numbers are optimistic. That likely
  makes every arm's gain over projection conservative, and more so for the geometry arms.
  This is an inference: no arm's error on the excluded pairs is known.
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

## 5. Reading (pilot, 13 arms)

This is the pilot's reading, written before the four other arm families ran. Where the
combined table changes it, the bullet says so.

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
- **Among the pilot's arms, image alignment is the only thing that helps Mapillary,** and it
  is where alignment succeeds most often (fallback 48% vs 76% on GSV). *Superseded:* in the
  combined table 18 arms beat `lg` on Mapillary, led by the multi-view 3D arms and RoMa. On the pairs it aligns it is near the
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
     What is enforced at run time, and what is not:
     - **Enforced for everything the harness hands an arm.** `ctx.pairs` holds the same
       stripped rows. The SlimPanos from `ctx.slim`, `ctx.at_height` and
       `geometry.at_height` have their `detections` emptied, because the reference is one
       of the other view's detections. `ctx.labeler()` exposes `geo` and only
       `HEIGHT_AUTO` / `pano_pose` of `fuse_sites` (`ARM_FS_NAMES`): its loaders
       (`load_results`, `load_at_height`, which return detections) and `eval_sites` are
       withheld. While an arm runs, `read_frozen_pairs()` and `read_rows()` raise, so an
       arm cannot fetch the pair list through the harness and build `"ref_" + "x"` at run
       time. Planted-arm tests cover each of these routes.
     - **Not enforced at run time, grep guard only.** An arm can still `open()`
       `pairs.csv`, `eligible_pairs.csv` or a `results.jsonl` itself (the paths follow from
       `H.OUT` and `ctx.args`), import `fuse_sites` directly, or reach private harness
       state. `test_arm_modules_never_reach_the_answer` fails on any of those names in an
       arm module, per definition rather than per file: only the CLI subcommands that run
       before or after prediction (`cmd_*`, `_mv3d.build_manifest`) are exempt, and no arm
       may call one. A name assembled at run time (for example `"pairs" + ".csv"`) would
       evade the grep, so this half is a guard against accidents, not against a determined
       cheat.
     - History: until 2026-09-29 `ctx.pairs` still held `ref_*`, and `ctx.labeler()` was the
       full labeler. A grep of every arm found no read of either. The 39 arms that run on
       CPU from committed inputs (the 7 geometry arms, `sem_geom_check`, the 17 `mono_*`
       arms, the 12 `flat_*` / `noflat_*` / `mlypano_*` arms and both `mv3d_consensus`
       arms) were re-predicted through the changed Context, with the labeler at `39afcd4`,
       and every one is byte-identical to its committed `.jsonl`. The GPU and imagery arms
       were not re-run.
   - `ctx` (`crossview_align_48.Context`) lazily provides:
     - `view(pair, "src"|"oth")` and `view_centre(pair, which)`;
     - `labeler()`, the labeler's `geo`, and `pano_pose` / `HEIGHT_AUTO` of `fuse_sites`;
     - `pano(city, id)`, the raw results.jsonl pano block, including `source_metadata`;
     - `slim(city, id)`, the labeler's SlimPano, and `at_height(city, height)`, the pair
       list's SlimPanos through the labeler's height resolver (both without detections);
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
   python scripts/analysis/crossview_combined_48.py           # the combined table + screens
   pytest -q tests/test_crossview_align_48.py                 # results.json re-derives
   ```

   `predict` writes `predictions/my_arm.jsonl` (one row per pair) and `my_arm.meta.json`
   (wall-clock, host, visible GPU, versions, config, labeler provenance, the pairs hash).
   Commit both and `results.json`. If it used a GPU, add a `paid: false` row to
   `analysis_out/usage_log.jsonl` as below. `score` took about 3 minutes for 13 arms and 20 minutes for
   the 83 arms after the family merges (desktop CPU, 2026-09-29).
   **`score` with no `--arms` takes every `predictions/*.jsonl`.** An exploratory file left
   there joins `results.json` and the multiplicity screen, which silently grows the number
   of arms being compared. `score` names any arm that is new since the committed
   `results.json`, and the rederive test fails until `results.json` lists it. Delete
   exploratory predictions rather than leaving them there.
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

## 8. Pilot verdict (proposed, not decided; superseded)

**Superseded by the combined takeaways**
([Combined comparison](#combined-comparison-across-all-arm-families)), which propose
`mapa_posed_pair` as the candidate placement, pending its fresh-pair re-test. The pilot's
verdict is kept below as the record of what the 13 pilot arms supported.

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
- `results.json`: `9b7edc77c3894687bcce59fd7c240691a6610520a6e80d11c02dfee547c6ae9b` (re-scored
  2026-09-29 over all 83 arms after the family merges; the 13 pilot arms' entries are
  byte-identical to the pilot's `035fb29c…`);
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
| `score` over all 83 arms after the family merges; `crossview_combined_48.py`; Richmond re-score (`flat_mapillary_48.py score`) | desktop CPU | 20 min 20 s; ~1 min; 2 min 40 s | 0 | 0 |
| `crossview_combined_48.py` with the Bonferroni screen and common-rule column (review fixes) | desktop CPU | 2 min 21 s | 0 | 0 |

No makelab2 GPU, no klone, no Tillicum, no paid API. GPU runs have `paid: false` rows in
`analysis_out/usage_log.jsonl`: the two pre-harness `match` runs and the three LightGlue
`predict` runs.
