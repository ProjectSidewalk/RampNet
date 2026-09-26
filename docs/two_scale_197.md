# Two-scale inference for the frozen checkpoint (#197)

Issue [#197](https://github.com/ProjectSidewalk/RampNet/issues/197), a follow-up to
[#25](https://github.com/ProjectSidewalk/RampNet/issues/25) /
[PR #196](https://github.com/ProjectSidewalk/RampNet/pull/196)
([`input_res_sweep_25.md`](input_res_sweep_25.md)). #196 found that feeding the released
checkpoint a 2× input (r4096, or the no-new-pixels upsample u4096) raises far-field recall and
lowers near-field recall. This document tests the obvious combination: take near-field detections
from the 1× pass (`r2048`) and far-field detections from the 2× upsampled pass (`u4096`), with no
retraining.

**Status: plan only (this section was committed before any fusion number was computed).**

## Inputs

Everything is CPU-only and committed: the #196 peak caches
`analysis_out/input_res_sweep_25/cache/{r2048,u4096}/<split>.json` (every `peak_local_max` peak
down to 0.05 on the 512×1024 heatmap, extracted with `exclude_border=False`, so the seam strip is
present in both arms), and the GT each cache carries (from the committed bundles). No GPU, no
panoramas, no network, no paid API. u4096 is used rather than r4096 because it needs only the
2048×4096 image, which is what a deployment already has, and #196 found the two statistically
indistinguishable.

## Fusion rules, fixed before scoring

Each rule produces one list of (x, y, score) per panorama, each peak keeping the score of the pass
it came from. The merged list is scored by the benchmark scorer (`score_pano`, radius 0.022,
seam-wrapped, `unsure` points ignored) through `benchmark_power_135.score_model`, exactly as #196
scored single arms.

"Dedupe" below means score-ordered non-maximum suppression across the two passes at the scorer's
own match radius and metric (`rampnet.geometry.dist_sq` with the pano scales, wrapped at the seam):
walk the merged list by descending score and drop any peak within the radius of one already kept
from the *other* pass. Peaks from the same pass are never suppressed against each other (each
pass's list is already `peak_local_max` output with `min_distance` 10).

- **R1, range split at 18 m (primary, fixed a priori).** Keep r2048 peaks with score ≥ 0.30 whose
  flat-ground range is under 18 m, and u4096 peaks with score ≥ 0.30 whose range is 18 m or more
  or which lie at or above the horizon; then dedupe across the boundary. Range is
  `recall_by_depth_112.flat_range(y, CAM_H = 2.5 m)`, so the split is a single image row:
  y ≤ 0.54393 is far. 18 m is `FAR_M`, the far band of
  [`detection_recall_analysis.md`](detection_recall_analysis.md) and #46 ("57.8% of misses at
  0.30 are beyond 18 m"), defined before #196 existed. It is not tuned. **Disclosure:** #196's
  pooled band table, on these same splits, shows u4096 ahead of r2048 only from 25 m on
  (18–25 m: r2048 0.771, u4096 0.754), so 18 m may be too near. That table was read before this
  plan; the cutoff is not moved because of it, and R3 exists to let the data choose without
  scoring on the split it chose on.
- **R2, union with dedupe (= max over the two passes).** Every r2048 peak ≥ 0.30 and every u4096
  peak ≥ 0.30, then dedupe. Where both passes fire on one spot the higher score survives, so this
  is the peak-level max of the two passes. No tunable.
- **R3, range split, leave-one-split-out.** As R1, but the cutoff D and the u4096 threshold t_u are
  chosen from D ∈ {12, 18, 25, 40} m × t_u ∈ {0.20, 0.30, 0.40}, plus "no fusion" (r2048 alone),
  by the best F1 at 0.30-equivalent (r2048 at 0.30, u4096 at t_u) pooled over the *other ten*
  splits. The held-out split is scored at the setting chosen without it; pools are built from the
  held-out outputs. r2048 stays at 0.30 in every rule.

## What is reported

For each rule against r2048 at 0.30, per split (all 11 #196 splits) and on #196's three pools
(headline = annapolis + richmond + laurens_mapillary; US pool = `miss_decomposition.US_SPLITS`;
GSV pool = bend + paterson + gainesville + sao_paulo + laurens_gsv):

1. P / R / F1 and paired ΔP / ΔR / ΔF1 with 95% CIs (pano-level paired cluster bootstrap
   stratified by split, 2,000 replicates, seed 197, `benchmark_power_135.observed_and_se`).
2. Duplicates removed by the dedupe, and far-band (≥ 18 m) recall of the fused list against
   r2048 and u4096 alone (how much of u4096's far gain survives the fusion). A naive union with no
   dedupe is reported as a reference row.
3. ΔFP / ΔTP (FP per recovered ramp) against r2048 at 0.30, and the **matched-recall
   threshold-only baseline**: r2048 alone at the highest threshold on a 0.05–0.95 grid (step 0.01)
   whose recall reaches the fused rule's recall, chosen in sample on the same split or pool (which
   favours the baseline). Paired ΔF1 CI, fused minus that baseline.
4. **Wrong-pano control.** Cyclic shift of the u4096 peaks over the sorted pano list within each
   split (pano i gets pano i + k's u4096 peaks, 20 evenly spaced k), fused with the correct r2048
   peaks by the same rule. Attributable ΔR = real ΔR − mean shifted ΔR, pooled.
5. Recall by flat-ground band (US pool) for r2048, u4096 and each rule, with paired CIs for the
   near (< 18 m) and far (≥ 18 m) bands.
6. Inference cost (Q4) estimated from the #196 rows in `analysis_out/usage_log.jsonl`; no new GPU
   work.

## Verdict rule, fixed before scoring

Applied to R1 on the US pool for the headline, and to every rule × split / pool in the tables.
All bounds unrounded. The labels are checked in the order listed; the first that applies is
the verdict.

- **HELPS**: ΔR CI lower bound > 0, AND ΔF1 CI lower bound > 0, AND the paired ΔF1 CI against the
  matched-recall threshold-only baseline has lower bound > 0.
- **RECALL LEVER**: ΔR CI lower bound > 0, ΔF1 CI not above 0, but the rule beats the
  matched-recall threshold-only baseline on F1 (paired CI lower bound > 0). It buys recall more
  cheaply than lowering the threshold, without raising F1 at 0.30.
- **NO BETTER THAN A THRESHOLD**: ΔR CI lower bound > 0, and the ΔF1 CI against the matched-recall
  baseline reaches 0 or below.
- **HURTS**: the ΔF1 CI upper bound < 0.
- **NULL**: anything else.

The precision caveat from #196 carries over unchanged and is stated beside every table: the GT is
anchored to reviewed 2048-input detections, so a real ramp that only u4096 finds scores as a false
positive. Every precision and F1 number for a fused rule is a lower bound in that sense.
