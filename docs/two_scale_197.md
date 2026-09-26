# Two-scale inference for the frozen checkpoint (#197)

Issue [#197](https://github.com/ProjectSidewalk/RampNet/issues/197), a follow-up to
[#25](https://github.com/ProjectSidewalk/RampNet/issues/25) /
[PR #196](https://github.com/ProjectSidewalk/RampNet/pull/196)
([`input_res_sweep_25.md`](input_res_sweep_25.md)). #196 found that feeding the released
checkpoint a 2× input (r4096, or the no-new-pixels upsample u4096) raises far-field recall and
lowers near-field recall. This document tests the obvious combination: take near-field detections
from the 1× pass (`r2048`) and far-field detections from the 2× upsampled pass (`u4096`), with no
retraining.

**Run date:** 2026-09-26. **Script:** `scripts/analysis/two_scale_197.py` (tests:
`tests/test_two_scale_197.py`). **Outputs:** `analysis_out/two_scale_197/results.json` and
`results.md`; every table below is pasted from `results.md`. **Compute:** CPU only, about 55 s on
`jonfhome` (Windows desktop); no GPU, no panoramas, no network, no spend, so no ledger row.

**Revised 2026-09-26 after review** ([PR #199 review](https://github.com/ProjectSidewalk/RampNet/pull/199#issuecomment-5850348171)):
the GSV pool's R2 "RECALL LEVER" is now reported beside a post hoc leave-paterson-out pool, where
it does not hold; the best single r2048 threshold is added as context; the seed-variance
citation is corrected. No pre-stated verdict changed.

The sections "Inputs" through "Verdict rule" were committed in `a5c4a8f` before any fusion number
was computed, and are unchanged below except where marked. The results start at
[Result](#result).

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
  [`detection_recall_analysis.md`](detection_recall_analysis.md) and #46, defined before #196
  existed ("57.8% of misses at 0.30 are beyond 18 m" is from
  [`data_scaling_59.md`](data_scaling_59.md), the far-field row of its miss table; the plan cited
  the wrong file, corrected after review). It is not tuned. **Disclosure:** #196's
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
   *Added after review, context only:* this is not #194's matched-recall definition, which takes
   the **best-F1** threshold among those that reach the recall (a lower threshold can have a
   higher F1). #194's version favours the baseline at least as much. Both it and #194's stronger
   control, the best single r2048 threshold on the grid, are reported beside the verdicts; neither
   enters them.
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

**Added after scoring (not in the plan): one gap in the verdict rule.** The rule does not say what
happens when no r2048 threshold on the 0.05–0.95 grid reaches the fused recall, so there is no
matched-recall baseline to beat. The script treats that as "not beaten", which is the conservative
reading. It happens on paterson only (R1 and R2; the table marks it "unreachable"). Under the
opposite reading paterson's R2 would read HELPS and its R1 RECALL LEVER. No pooled verdict is
affected: every pool has a reachable matched threshold for R1, R2 and R3.

## Result

**Headline, the pre-stated primary (R1, range split at 18 m, US pool): HURTS.** Against r2048 at
0.30 over the eight US splits (953 panos, 2,309 GT ramps): ΔR +0.006 [−0.006, +0.018], ΔP −0.048
[−0.059, −0.037], ΔF1 −0.018 [−0.027, −0.009]. It recovers 13 ramps for 115 extra false positives.
A threshold alone (r2048 at 0.28) reaches the same recall for 29 extra FPs, and beats R1 on F1 by
0.017 [0.007, 0.025].

**Why R1 fails: the far field is a few image rows, and a hard row cut throws away r2048's own
far-field hits.** At a 2.5 m camera, 18–25 m is y 0.532–0.544 of the image height, about 6 rows of
the 512-row heatmap, while the match radius is about 22 rows. A post hoc diagnostic (added after
the first results, no verdict) counts the 85 US-pool ramps that r2048 finds and R1 loses: **all
85** had their r2048 peak on the far side of the cut (so R1 discarded it); for 31 u4096 had no
peak ≥ 0.30 within the match radius, for 46 u4096's only peak was on the near side, which R1
also discards, and for the remaining 8 u4096 did have a far-side peak within the radius and the
ramp was still lost (that peak went to another GT point or was removed by the dedupe). The 18–25 m band carries the loss: recall 0.771 (r2048) →
0.663 (R1), below u4096 alone (0.754). #196's band table already showed u4096 ahead of r2048 only
from 25 m on, and the plan disclosed this before scoring; 18 m was kept because it was fixed a
priori.

**The union (R2, the peak-level max of the two passes) is the only rule that moves recall by a
useful amount, and on the pre-stated rule it is no better than lowering the threshold on the US
pool.** US pool: R 0.767 → 0.812, ΔR +0.045 [+0.037, +0.055], ΔP −0.077 [−0.088, −0.066], ΔF1
−0.011 [−0.019, −0.003]; +105 ramps for +212 FPs, 2.0 FP per recovered ramp. The matched-recall
threshold (r2048 at 0.16) pays +262 FPs for the same recall. R2 beats it on F1 by +0.009 [−0.000,
+0.018]; the unrounded lower bound is −0.00008, so the rule reads NO BETTER THAN A THRESHOLD by a
margin that is nothing. On the **GSV pool** the same comparison clears: ΔR +0.057 [+0.045,
+0.069], ΔF1 −0.001 [−0.011, +0.009], +85 ramps for +126 FPs (1.5 per ramp), against +209 FPs for
the matched threshold (0.13), fused − matched ΔF1 +0.022 [+0.010, +0.033]: **RECALL LEVER, driven
by paterson.** paterson supplies 30 of the 85 recovered ramps, and it is the split where no r2048
threshold on the grid reaches R2's recall and where u4096 alone raises F1. With paterson left out
(post hoc, added after review; bend + gainesville + sao_paulo + laurens_gsv), R2 reads NO BETTER
THAN A THRESHOLD: ΔR +0.050 [+0.038, +0.063], ΔF1 −0.012 [−0.024, −0.001], +55 ramps for +113 FPs
(2.1 per ramp) against +95 FPs for the matched threshold (0.18), fused − matched −0.009 [−0.022,
+0.004]. So the GSV-pool reading is a paterson result, not a GSV one. On the headline Mapillary
pool R2 does not clear either (fused − matched −0.014 [−0.029, +0.000]).

**The leave-one-split-out rule (R3) chose the most conservative setting on the grid for every
held-out split (D = 40 m, t_u = 0.40)**, and it buys little: US pool ΔR +0.010 [+0.006, +0.015],
ΔF1 +0.003 [−0.000, +0.007], +24 ramps for +17 FPs, NO BETTER THAN A THRESHOLD (fused − matched
+0.004 [−0.001, +0.009]). On the other ten splits it beat "no fusion" by 0.0012–0.0026 F1 for
ten of the eleven held-out splits (with richmond held out, 0.7966 against 0.7954), and by
0.000002 with paterson held out (0.7996894 against 0.7996873, in `results.md`'s LOSO table). It
sits on the edge of the grid, so the F1-optimal far-field cutoff for this checkpoint may be beyond
40 m or at no fusion at all. It also ties the best single r2048 threshold on the headline pool
(F1 0.8107 against 0.8105 at 0.26) and is within noise of it on the US pool (0.8281 against 0.8256
at 0.32, ΔF1 +0.003 [−0.001, +0.007]).

**No rule reads HELPS on any pool.** Per split, one rule × split reads HELPS: paterson under R3
(ΔR +0.023 [+0.009, +0.040], ΔF1 +0.014 [+0.004, +0.026], beats its matched threshold by +0.018
[+0.005, +0.033]). Two things have to be said beside it. **It rests on a near-tie in the LOSO
choice:** with paterson held out, (40 m, 0.40) beat "no fusion" on the other ten splits by 0.000002
F1; had "no fusion" been chosen, paterson's R3 would equal r2048 and read NULL. **And u4096 alone
does better on paterson** (F1 0.8575 against R3's 0.8345; R2 0.8541). paterson is the one split
where #196 found u4096 alone raises F1, so this is the same scale effect, not a new one.
richmond is the only split where R2 reads RECALL LEVER; morgantown reads HURTS under all three
rules. On both Laurens arms the best single r2048 threshold (0.19 on laurens_gsv, 0.16 on
laurens_mapillary) beats every rule on F1 (context table in `results.md`).

**Answers to the issue's four questions:**

1. **Does a stated-in-advance rule beat 1× at 0.30 on recall and F1?** No. The a priori rule
   (R1) loses F1 on the US pool and does not raise recall. The union (R2) raises recall on every
   pool (CI above 0) and lowers or holds F1 (never with a CI above 0). The LOSO rule (R3) raises
   recall by a point and holds F1, and ties the best single r2048 threshold.
2. **How much of the far-field gain survives de-duplication?** For R2, all of u4096's far-band
   gain and more: US-pool far-band (≥ 18 m, n = 728) ΔR +0.119 [+0.096, +0.142] against +0.077
   [+0.047, +0.108] for u4096 alone, because R2 keeps r2048's own far hits too, and near-band
   recall does not drop (+0.011 [+0.006, +0.016], n = 1,576). The dedupe removes 1,328 u4096 peaks
   and 388 r2048 peaks on the US pool. Without it (the naive union) recall is 0.858 but precision
   0.507 (+1,712 FPs). Most of the naive union's extra recall over R2 (106 ramps) is not
   recovered by a looser dedupe: R2s (below) recovers 18 of them. The rest need a u4096 or r2048
   peak within a Euclidean 10 heatmap px of a higher-scoring peak from the other pass. Whether
   those are duplicate detections of one ramp that the greedy matcher hands to a second nearby GT
   point, or genuine detections of an adjacent ramp, was **not measured**; either way they cost
   about 8 FPs per ramp in the naive union.
3. **FP per recovered ramp, and against the matched-recall baseline?** R2: 2.0 FP per recovered
   ramp on the US pool, 1.5 on GSV, 2.1 on the headline pool; the matched threshold costs 2.5
   (262 FPs for 105 ramps), 2.4 (209 for 86) and 1.3 (50 for 38) respectively. Without
   paterson (post hoc) the GSV figures are 2.1 for R2 against 1.7 for the matched threshold (95
   FPs for 55 ramps). R1: 8.9 (US). R3: 0.7 (US). The matched threshold is chosen in sample on the
   pool it is compared on, which favours it. Against the best single r2048 threshold (also in
   sample), R2 loses F1 on the US pool (−0.012 [−0.020, −0.004]) and ties on GSV (−0.001
   [−0.011, +0.009]).
4. **Inference cost:** about **4.8× the 1× pass** in GPU time (below).

**So: two-scale inference is a recall dial for the far field that costs about 2 FPs per ramp and
about 4.8× the GPU time, and on the US pool it is not measurably better than lowering the
threshold, which costs no extra compute.** It reads as a better dial than the threshold on the
GSV pool, but that reading is driven by paterson: without paterson (post hoc) it is no better
than a threshold there either. Per split it is a better dial only on richmond (and on paterson,
where no threshold reaches its recall at all). It does not raise F1 at 0.30 anywhere pooled.

## Tables

Paired pano-level cluster bootstrap, stratified by split, 2,000 replicates, seed 197,
`benchmark_power_135.observed_and_se`; the same weight matrix is applied to both lists. Δ is
rule − r2048 at 0.30. Every fused list is final before scoring (thresholds and dedupe already
applied) and is scored at threshold 0, which for r2048 alone reproduces #196's 0.30 row exactly
(tested on richmond).

### Pooled (from `results.md`)

| pool | rule | P | R | F1 | ΔP | ΔR | ΔF1 | verdict |
|---|---|---|---|---|---|---|---|---|
| headline (3 Mapillary) | r2048 | 0.892 | 0.735 | 0.806 | — | — | — | |
| headline | R1 | 0.840 | 0.740 | 0.787 | −0.052 [−0.068, −0.035] | +0.005 [−0.014, +0.023] | −0.019 [−0.033, −0.005] | HURTS |
| headline | R2 | 0.814 | 0.776 | 0.795 | −0.078 [−0.097, −0.059] | +0.041 [+0.027, +0.057] | −0.011 [−0.025, +0.003] | NO BETTER THAN A THRESHOLD |
| headline | R3 | 0.888 | 0.746 | 0.811 | −0.004 [−0.010, +0.002] | +0.011 [+0.001, +0.021] | +0.005 [−0.002, +0.012] | NO BETTER THAN A THRESHOLD |
| headline | u4096 | 0.817 | 0.644 | 0.720 | −0.075 [−0.105, −0.046] | −0.091 [−0.125, −0.058] | −0.086 [−0.113, −0.060] | |
| headline | naive union | 0.500 | 0.798 | 0.615 | −0.392 [−0.411, −0.371] | +0.063 [+0.045, +0.084] | −0.191 [−0.213, −0.169] | |
| **US pool** (8) | r2048 | 0.892 | 0.767 | 0.825 | — | — | — | |
| US pool | **R1** | 0.844 | 0.773 | 0.807 | −0.048 [−0.059, −0.037] | +0.006 [−0.006, +0.018] | −0.018 [−0.027, −0.009] | **HURTS** |
| US pool | R2 | 0.815 | 0.812 | 0.814 | −0.077 [−0.088, −0.066] | +0.045 [+0.037, +0.055] | −0.011 [−0.019, −0.003] | NO BETTER THAN A THRESHOLD |
| US pool | R3 | 0.886 | 0.777 | 0.828 | −0.006 [−0.011, −0.003] | +0.010 [+0.006, +0.015] | +0.003 [−0.000, +0.007] | NO BETTER THAN A THRESHOLD |
| US pool | u4096 | 0.817 | 0.692 | 0.749 | −0.075 [−0.093, −0.059] | −0.075 [−0.095, −0.056] | −0.076 [−0.091, −0.061] | |
| US pool | naive union | 0.507 | 0.858 | 0.638 | −0.385 [−0.397, −0.373] | +0.091 [+0.079, +0.105] | −0.187 [−0.202, −0.173] | |
| **GSV pool** (5) | r2048 | 0.887 | 0.759 | 0.818 | — | — | — | |
| GSV pool | R1 | 0.848 | 0.771 | 0.808 | −0.040 [−0.053, −0.027] | +0.012 [−0.004, +0.027] | −0.011 [−0.022, −0.000] | HURTS |
| GSV pool | R2 | 0.819 | 0.816 | 0.817 | −0.069 [−0.082, −0.056] | +0.057 [+0.045, +0.069] | −0.001 [−0.011, +0.009] | RECALL LEVER |
| GSV pool | R3 | 0.881 | 0.769 | 0.821 | −0.006 [−0.011, −0.002] | +0.010 [+0.005, +0.015] | +0.003 [−0.001, +0.007] | NO BETTER THAN A THRESHOLD |
| GSV pool | u4096 | 0.817 | 0.698 | 0.752 | −0.071 [−0.090, −0.050] | −0.061 [−0.087, −0.035] | −0.066 [−0.084, −0.046] | |
| GSV pool | naive union | 0.522 | 0.884 | 0.657 | −0.365 [−0.380, −0.349] | +0.125 [+0.107, +0.145] | −0.162 [−0.181, −0.141] | |
| *GSV pool minus paterson* (4, **post hoc**) | r2048 | 0.869 | 0.772 | 0.818 | — | — | — | |
| *GSV minus paterson* | R1 | 0.825 | 0.772 | 0.798 | −0.044 [−0.061, −0.027] | +0.000 [−0.017, +0.016] | −0.020 [−0.032, −0.007] | *HURTS (post hoc)* |
| *GSV minus paterson* | R2 | 0.789 | 0.822 | 0.805 | −0.080 [−0.095, −0.064] | +0.050 [+0.038, +0.063] | −0.012 [−0.024, −0.001] | *NO BETTER THAN A THRESHOLD (post hoc)* |
| *GSV minus paterson* | R3 | 0.861 | 0.777 | 0.817 | −0.008 [−0.014, −0.003] | +0.005 [+0.002, +0.010] | −0.001 [−0.004, +0.003] | *NO BETTER THAN A THRESHOLD (post hoc)* |

The r2048 and u4096 rows equal #196's (`input_res_sweep_25.md`) to 3 dp. The GSV pool's R1 "HURTS"
rests on an upper bound of −0.00004. The pools overlap: paterson is in both the US and GSV pools,
bend and gainesville too. **The GSV pool's R2 RECALL LEVER is driven by paterson:** the
leave-paterson-out rows (added after review, not in the plan, so their labels are context and not
pre-stated verdicts) read NO BETTER THAN A THRESHOLD, and paterson supplies 30 of the GSV pool's
85 recovered ramps.

Caveats beside this table:

- **Precision (and so F1) of every fused rule is a lower bound.** The GT is anchored to reviewed
  2048-input detections plus the reviewer's missed marks, so a real ramp that only u4096 finds, and
  that the reviewer did not mark, scores as an FP. This is the same caveat #196 carries, and it
  bears harder here because every rule adds u4096 peaks. No gallery of the added FPs was reviewed.
  If enough of R2's added FPs are real ramps, R2's ΔF1 moves up; the recall columns do not depend
  on this.
- **The matched-recall threshold is chosen in sample** on the split or pool it is compared on
  (the highest grid threshold whose recall reaches the fused recall, the plan's definition). This
  favours the baseline. #194 instead takes the best-F1 threshold among those that reach the
  recall; that favours the baseline at least as much, and on these data it changes no verdict
  (`results.md`, context table; tested).
- **One checkpoint, deterministic inference.** The bootstrap interval is the whole uncertainty for
  this checkpoint. It says nothing about another checkpoint of the same recipe: the seed-to-seed
  SD of the recipe's macro-mean US7 F1 is `s_B` = 0.0094 over nine retrained replicates
  ([`seed_variance_51_135.md`](seed_variance_51_135.md), Amendment 2), comparable to or larger
  than most ΔF1s here.

### Cost of the recovered ramps (pooled, from `results.md`)

| pool | rule | ramps recovered | extra FP | FP / ramp | dedupe dropped (r2048 / u4096) | matched r2048 threshold | its extra FP | its F1 | fused − matched ΔF1 |
|---|---|---|---|---|---|---|---|---|---|
| headline | R1 | +4 | +44 | 11.00 | 1 / 2 | 0.29 | +6 | 0.806 | −0.019 [−0.034, −0.005] |
| headline | R2 | +35 | +75 | 2.14 | 144 / 456 | 0.20 | +50 | 0.809 | −0.014 [−0.029, +0.000] |
| headline | R3 | +9 | +4 | 0.44 | 11 / 7 | 0.27 | +16 | 0.808 | +0.002 [−0.006, +0.012] |
| US pool | R1 | +13 | +115 | 8.85 | 2 / 14 | 0.28 | +29 | 0.823 | −0.017 [−0.025, −0.007] |
| US pool | R2 | +105 | +212 | 2.02 | 388 / 1328 | 0.16 | +262 | 0.805 | +0.009 [−0.000, +0.018] |
| US pool | R3 | +24 | +17 | 0.71 | 21 / 10 | 0.26 | +49 | 0.825 | +0.004 [−0.001, +0.009] |
| GSV pool | R1 | +18 | +63 | 3.50 | 1 / 13 | 0.26 | +29 | 0.818 | −0.011 [−0.022, +0.001] |
| GSV pool | R2 | +85 | +126 | 1.48 | 259 / 868 | 0.13 | +209 | 0.796 | +0.022 [+0.010, +0.033] |
| GSV pool | R3 | +15 | +11 | 0.73 | 12 / 0 | 0.26 | +29 | 0.818 | +0.003 [−0.002, +0.009] |
| *GSV minus paterson* (post hoc) | R1 | +0 | +52 | — | 1 / 9 | 0.30 | +0 | 0.818 | −0.020 [−0.032, −0.007] |
| *GSV minus paterson* (post hoc) | R2 | +55 | +113 | 2.05 | 172 / 641 | 0.18 | +95 | 0.814 | −0.009 [−0.022, +0.004] |
| *GSV minus paterson* (post hoc) | R3 | +6 | +10 | 1.67 | 6 / 0 | 0.28 | +15 | 0.815 | +0.002 [−0.003, +0.007] |

"Ramps recovered" is the net change in true positives on recall-confirmed panos; "extra FP" is
over all panos, as `aggregate` counts them. FP per ramp is a ceiling for the same reason precision
is a floor.

### Context: the best single r2048 threshold (added after review, not a verdict input)

#194's stronger control: the r2048 threshold on the 0.05–0.95 grid with the best F1 on the same
split or pool, chosen in sample. Paired ΔF1, fused − that threshold (same bootstrap).

| pool | best single thr | its F1 | R1: fused − best | R2: fused − best | R3: fused − best |
|---|---|---|---|---|---|
| headline | 0.26 | 0.810 | −0.024 [−0.040, −0.008] | −0.016 [−0.030, −0.002] | +0.000 [−0.009, +0.010] |
| US pool | 0.32 | 0.826 | −0.019 [−0.028, −0.009] | −0.012 [−0.020, −0.004] | +0.003 [−0.001, +0.007] |
| GSV pool | 0.30 | 0.818 | −0.011 [−0.022, −0.000] | −0.001 [−0.011, +0.009] | +0.003 [−0.001, +0.007] |
| *GSV minus paterson* (post hoc) | 0.30 | 0.818 | −0.020 [−0.032, −0.007] | −0.012 [−0.024, −0.001] | −0.001 [−0.004, +0.003] |

No rule beats the best single threshold on any pool; R2 loses to it on the US and headline pools.
Per split (`results.md`), only paterson's R2 (+0.031 [+0.013, +0.049]) and R3 (+0.011 [+0.002,
+0.022]) beat it, and on both Laurens arms it beats every rule.

### Per split, the three rules

| split | r2048 P / R / F1 | rule | P / R / F1 | ΔR | ΔF1 | ramps / FP vs r2048 | matched thr: ΔF1 fused − matched | verdict |
|---|---|---|---|---|---|---|---|---|
| annapolis | 0.895 / 0.809 / 0.850 | R1 | 0.814 / 0.803 / 0.808 | −0.007 [−0.047, +0.032] | −0.042 [−0.070, −0.013] | −2 / +26 | 0.32: −0.044 [−0.071, −0.018] | HURTS |
|  |  | R2 | 0.791 / 0.850 / 0.820 | +0.041 [+0.018, +0.067] | −0.030 [−0.055, −0.005] | +12 / +38 | 0.15: −0.016 [−0.040, +0.009] | NO BETTER THAN A THRESHOLD |
|  |  | R3 | 0.893 / 0.823 / 0.857 | +0.014 [−0.004, +0.034] | +0.007 [−0.006, +0.020] | +4 / +1 | 0.27: +0.006 [−0.009, +0.022] | NULL |
| bend | 0.915 / 0.829 / 0.870 | R1 | 0.858 / 0.829 / 0.843 | +0.000 [−0.032, +0.033] | −0.027 [−0.050, −0.004] | +0 / +20 | 0.30: −0.027 [−0.050, −0.004] | HURTS |
|  |  | R2 | 0.836 / 0.872 / 0.853 | +0.043 [+0.021, +0.069] | −0.017 [−0.034, +0.001] | +14 / +31 | 0.08: +0.019 [−0.002, +0.042] | NO BETTER THAN A THRESHOLD |
|  |  | R3 | 0.907 / 0.835 / 0.869 | +0.006 [+0.000, +0.016] | −0.001 [−0.007, +0.007] | +2 / +3 | 0.27: +0.001 [−0.007, +0.010] | NULL |
| budapest_district5 | 0.680 / 0.643 / 0.661 | R1 | 0.622 / 0.613 / 0.617 | −0.030 [−0.067, +0.006] | −0.043 [−0.072, −0.017] | −9 / +21 | 0.37: −0.059 [−0.089, −0.029] | HURTS |
|  |  | R2 | 0.590 / 0.680 / 0.632 | +0.037 [+0.017, +0.061] | −0.029 [−0.053, −0.007] | +11 / +51 | 0.19: +0.016 [−0.009, +0.040] | NO BETTER THAN A THRESHOLD |
|  |  | R3 | 0.676 / 0.640 / 0.657 | −0.003 [−0.018, +0.010] | −0.003 [−0.016, +0.008] | −1 / +1 | 0.31: −0.007 [−0.020, +0.005] | NULL |
| clovis | 0.838 / 0.821 / 0.829 | R1 | 0.816 / 0.821 / 0.818 | +0.000 [−0.034, +0.036] | −0.011 [−0.035, +0.015] | +0 / +5 | 0.30: −0.011 [−0.035, +0.015] | NULL |
|  |  | R2 | 0.777 / 0.841 / 0.808 | +0.021 [+0.000, +0.048] | −0.021 [−0.045, +0.004] | +4 / +16 | 0.25: −0.014 [−0.041, +0.016] | NULL |
|  |  | R3 | 0.833 / 0.821 / 0.827 | +0.000 [+0.000, +0.000] | −0.002 [−0.007, +0.000] | +0 / +1 | 0.30: −0.002 [−0.007, +0.000] | NULL |
| gainesville | 0.854 / 0.772 / 0.811 | R1 | 0.810 / 0.768 / 0.789 | −0.004 [−0.043, +0.038] | −0.022 [−0.049, +0.008] | −1 / +13 | 0.32: −0.025 [−0.052, +0.007] | NULL |
|  |  | R2 | 0.766 / 0.842 / 0.802 | +0.070 [+0.043, +0.100] | −0.009 [−0.034, +0.019] | +19 / +34 | 0.16: +0.004 [−0.022, +0.034] | NO BETTER THAN A THRESHOLD |
|  |  | R3 | 0.846 / 0.787 / 0.815 | +0.015 [+0.003, +0.030] | +0.004 [−0.006, +0.016] | +4 / +3 | 0.26: +0.004 [−0.012, +0.019] | NO BETTER THAN A THRESHOLD |
| laurens_gsv | 0.923 / 0.654 / 0.766 | R1 | 0.922 / 0.645 / 0.759 | −0.009 [−0.044, +0.027] | −0.007 [−0.032, +0.018] | −2 / +0 | 0.33: −0.002 [−0.029, +0.025] | NULL |
|  |  | R2 | 0.905 / 0.691 / 0.783 | +0.036 [+0.013, +0.065] | +0.018 [+0.002, +0.037] | +8 / +4 | 0.25: −0.001 [−0.026, +0.025] | NO BETTER THAN A THRESHOLD |
|  |  | R3 | 0.923 / 0.654 / 0.766 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0 / +0 | 0.30: +0.000 [+0.000, +0.000] | NULL |
| laurens_mapillary | 0.885 / 0.526 / 0.660 | R1 | 0.883 / 0.514 / 0.650 | −0.012 [−0.034, +0.006] | −0.010 [−0.030, +0.006] | −3 / +0 | 0.33: −0.003 [−0.027, +0.019] | NULL |
|  |  | R2 | 0.883 / 0.546 / 0.675 | +0.020 [+0.004, +0.039] | +0.015 [+0.002, +0.030] | +5 / +1 | 0.27: −0.006 [−0.023, +0.011] | NO BETTER THAN A THRESHOLD |
|  |  | R3 | 0.885 / 0.526 / 0.660 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0 / +0 | 0.30: +0.000 [+0.000, +0.000] | NULL |
| morgantown | 0.878 / 0.813 / 0.844 | R1 | 0.801 / 0.783 / 0.792 | −0.030 [−0.054, −0.007] | −0.053 [−0.075, −0.032] | −8 / +22 | 0.40: −0.060 [−0.089, −0.034] | HURTS |
|  |  | R2 | 0.751 / 0.824 / 0.786 | +0.011 [+0.000, +0.026] | −0.059 [−0.080, −0.038] | +3 / +43 | 0.22: −0.037 [−0.056, −0.017] | HURTS |
|  |  | R3 | 0.861 / 0.813 / 0.836 | +0.000 [+0.000, +0.000] | −0.008 [−0.018, −0.002] | +0 / +5 | 0.32: −0.015 [−0.026, −0.005] | HURTS |
| paterson | 0.947 / 0.724 / 0.821 | R1 | 0.918 / 0.770 / 0.838 | +0.046 [+0.012, +0.079] | +0.017 [−0.007, +0.040] | +18 / +11 | unreachable | NO BETTER THAN A THRESHOLD |
|  |  | R2 | 0.916 / 0.800 / 0.854 | +0.076 [+0.050, +0.103] | +0.033 [+0.015, +0.053] | +30 / +13 | unreachable | NO BETTER THAN A THRESHOLD |
|  |  | R3 | 0.946 / 0.747 / 0.835 | +0.023 [+0.009, +0.040] | +0.014 [+0.004, +0.026] | +9 / +1 | 0.15: +0.018 [+0.005, +0.033] | HELPS |
| richmond | 0.893 / 0.832 / 0.861 | R1 | 0.845 / 0.861 / 0.853 | +0.029 [+0.000, +0.060] | −0.008 [−0.031, +0.014] | +9 / +18 | 0.20: +0.008 [−0.016, +0.032] | NULL |
|  |  | R2 | 0.805 / 0.890 / 0.845 | +0.058 [+0.030, +0.091] | −0.016 [−0.043, +0.010] | +18 / +36 | 0.08: +0.091 [+0.059, +0.126] | RECALL LEVER |
|  |  | R3 | 0.885 / 0.848 / 0.867 | +0.016 [+0.000, +0.037] | +0.005 [−0.007, +0.019] | +5 / +3 | 0.24: +0.014 [−0.003, +0.033] | NULL |
| sao_paulo | 0.803 / 0.797 / 0.800 | R1 | 0.754 / 0.808 / 0.780 | +0.011 [−0.015, +0.037] | −0.020 [−0.043, +0.004] | +3 / +19 | 0.25: −0.007 [−0.029, +0.017] | NULL |
|  |  | R2 | 0.706 / 0.847 / 0.770 | +0.050 [+0.025, +0.079] | −0.030 [−0.055, −0.004] | +14 / +44 | 0.16: +0.009 [−0.019, +0.035] | NO BETTER THAN A THRESHOLD |
|  |  | R3 | 0.791 / 0.797 / 0.794 | +0.000 [+0.000, +0.000] | −0.006 [−0.012, −0.001] | +0 / +4 | 0.30: −0.006 [−0.012, −0.001] | HURTS |

`results.md` also carries each split's u4096, naive-union and R2s rows.

### Wrong-pano control (R2)

Cyclic shift of the u4096 peaks over the sorted pano list within each split (20 evenly spaced
shifts), fused with the correct r2048 peaks by the same rule and scored the same way.

| pool | real ΔR | null ΔR mean | null ΔR max | attributable ΔR | real ΔFP | null ΔFP mean |
|---|---|---|---|---|---|---|
| headline | +0.0410 | +0.0135 | +0.0246 | +0.0275 | +75 | +665.9 |
| US pool | +0.0455 | +0.0118 | +0.0165 | +0.0337 | +212 | +1879.7 |
| GSV pool | +0.0569 | +0.0127 | +0.0167 | +0.0441 | +126 | +1217.7 |

A fifth to a third (22–33%) of R2's raw recall gain on each pool is what the shifted peaks buy by
chance; the attributable gain on the US pool is +0.034, about 78 ramps. **The null is not
density-matched**, so the attributable ΔR is a lower bound: after the dedupe, a shifted u4096
pass leaves about nine times as many surviving peaks as the real one (null ΔFP +1,880 against
+212 on the US pool), and each surviving peak is a chance to land on a ramp, so the null overstates
what chance would buy at the real pass's density. On no split does the real ΔR fall
inside the shifted range except laurens_mapillary (real +0.020, null max +0.036) and morgantown
(real +0.011, null max +0.019), whose R2 gains are indistinguishable from chance (even by this
conservative null). The wrong-pano copy's extra FPs are high because the real pass's extra peaks
mostly land on ramps r2048 already found and are removed by the dedupe, while shifted peaks
rarely coincide with an r2048 peak. Without paterson (post hoc) the attributable GSV ΔR is
+0.038 (real +0.050, null mean +0.012).

**The same control is not informative for R1 and R3**, and their rows in `results.md` should not be
read as evidence. Those rules *replace* r2048's far field with u4096's, so a wrong-pano u4096 far
field loses r2048's far hits and its ΔR is strongly negative (US pool −0.184 for R1); the
"attributable" figure then measures how much the far field contributes at all, not the gain.

### Recall by flat-ground range, US pool, at 0.30 (from `results.md`)

| band | n | r2048 | u4096 | naive union | R1 | R2 | R3 |
|---|---|---|---|---|---|---|---|
| 0-8 m | 516 | 0.841 | 0.620 | 0.845 | 0.841 | 0.841 | 0.841 |
| 8-12 m | 594 | 0.827 | 0.680 | 0.879 | 0.827 | 0.837 | 0.827 |
| 12-18 m | 466 | 0.805 | 0.740 | 0.878 | 0.794 | 0.828 | 0.805 |
| 18-25 m | 419 | 0.771 | 0.754 | 0.890 | 0.663 | 0.821 | 0.768 |
| 25-40 m | 232 | 0.543 | 0.720 | 0.823 | 0.716 | 0.728 | 0.578 |
| 40 m+ | 77 | 0.273 | 0.558 | 0.636 | 0.558 | 0.571 | 0.480 |
| above horizon | 5 | 0.200 | 0.400 | 0.400 | 0.400 | 0.400 | 0.400 |

| rule | near < 18 m (n 1,576) | far ≥ 18 m (n 728) |
|---|---|---|
| u4096 | −0.147 [−0.169, −0.124] | +0.077 [+0.047, +0.108] |
| R1 | −0.003 [−0.007, +0.000] | +0.023 [−0.012, +0.058] |
| R2 | +0.011 [+0.006, +0.016] | +0.119 [+0.096, +0.142] |
| R3 | +0.000 [+0.000, +0.000] | +0.032 [+0.017, +0.047] |

The five above-horizon GT points are in neither band. The distance axis is flat-ground geometry at
an assumed 2.5 m camera, the committed axis for these splits; band edges are approximate
([`detection_recall_analysis.md` §0](detection_recall_analysis.md)). R1's 12–18 m loss (0.805 →
0.794) is the same boundary effect as its 18–25 m loss, seen from the near side. The headline and
GSV pools show the same pattern (`results.md`).

### Post hoc: dedupe radius (R2s, not in the plan, no verdict)

After the first results showed the naive union's recall well above R2's, R2 was re-run with the
dedupe radius at a Euclidean 10 heatmap px instead of the scorer's match radius (about 22.5 px),
to see whether the dedupe was discarding real neighbouring ramps. 10 px is the extractor's
`min_distance`, but `peak_local_max` applies it as a square (Chebyshev) window, so two peaks of
one pass can sit 11–14 px apart on a diagonal; the Euclidean disc is slightly looser than the
extractor's own rule. This is post hoc with no verdict, so the difference changes no reading. On the US
pool R2s gains 18 more ramps than R2 (R 0.820 against 0.812) for 71 more FPs, and F1 falls (0.806
against 0.814; ΔF1 vs r2048 −0.019 [−0.028, −0.010]). The looser dedupe does not close the gap to
the naive union, and is worse than R2 on F1 on all three pools. It does not change any
reading above.

## Inference cost (Q4)

Estimated from three #196 rows in `analysis_out/usage_log.jsonl`, pinned by label and `ts` in
`COST_ROWS` (`input-res-25:r2048` at 2026-09-26T14:18:26Z; `input-res-25:u4096` at
2026-09-26T14:44:19Z, 125 panos, and 2026-09-26T15:09:24Z, 1,164 panos; makelab2, one NVIDIA A40,
fp32). `inference_cost()` re-derives these, refuses a ledger that lacks one of them, and ignores
any later `input-res-25:*` row (tested). No new GPU work was run.

| pass | panos | GPU-side s / pano |
|---|---|---|
| r2048 (1×, 2048×4096) | 1,289 | 0.610 |
| u4096 (2× upsample, 4096×8192) | 1,289 | 2.344 |
| **two-scale (both)** | | **2.953, 4.84× the 1× pass** |

"GPU-side" is host-to-device copy, forward and peak extraction on the main thread, per #196's
ledger convention. Caveats beside the number:

- **The CPU cost of the upsample is not in it.** The bicubic resize to 4096×8192 ran in #196's
  prefetch worker together with the JPEG decode and every other arm's resize, and the ledger's
  `cpu-wait` rows are not split per arm, so the u4096 share of CPU time cannot be separated from
  the committed rows. A deployment that upsamples on the GPU would move it there.
- **One GPU, fp32, batch size 1.** The ratio is close to the 4× pixel ratio, as a convolutional
  backbone's cost should be; it was not measured under fp16, batching, or on other hardware.
- **The two passes were timed in different invocations.** r2048 was timed in its own run (with
  only the CPU-wait row beside it); u4096 in two multi-arm runs that interleaved r3072, r4096,
  rnative and r4096_hm1024 on the same card, and the A40 is shared with other lab users (#196
  records 8.8 GB held by another process during its smoke test). The ratio compares GPU-side
  seconds from those different runs, not a paired timing of both passes on one pano.
- For scale: R2 on the US pool recovers 105 ramps over 953 panos at 2.34 extra GPU-seconds per
  pano (about 37 extra GPU-minutes on this card); lowering the threshold to 0.16 recovers the same
  number for no extra compute and 50 more FPs.

## What this does not answer

- **Soft or wider boundaries.** R1 fails at a hard row cut. A rule that takes the far field from
  u4096 only beyond 40 m, or that keeps r2048 everywhere and adds only u4096 peaks with no r2048
  peak nearby (which is what R2 does), are the two ends; nothing between them was scored, and
  R3's grid ended at 40 m.
- **Whether R2's added FPs are real ramps.** A reviewer pass over them would turn the precision
  lower bound into a number. Until then R2's ΔF1 is a floor.
- **r4096 instead of u4096.** #196 found them indistinguishable at 2×; u4096 was used because a
  deployment has the 2048 image. The native pixels were not tried as the far pass.
- **A retrained model.** A model trained with scale augmentation could make the second pass
  unnecessary; that is #25's retrain arm and is still open.

## Seam

All inputs are #196's caches, extracted with `exclude_border=False` after the seam fix `f4c71c8`,
so the peaks beside the 360° seam are present in both passes. No `analysis_out/op_cache` file is
used here, so the pre-seam-fix caveat that #196 and #194 carry for op_cache numbers does not apply.
The dedupe measures distance wrapped at the seam, as the scorer does.

## Reproduction, from a clean clone

```bash
pip install -e . && pip install -r requirements-dev.txt

# every table in this doc, from the committed #196 caches (~55 s, CPU)
python scripts/analysis/two_scale_197.py report

# prove the committed results.json and results.md reproduce byte for byte (writes nothing)
python scripts/analysis/two_scale_197.py report --check

# tests (~1 min, most of it the report --check drift guard): fusion helpers, verdict
# branches, LOSO selection, richmond rows vs #196, the Q4 cost and the full report re-derived
python -m pytest -q tests/test_two_scale_197.py
```

The inputs are `analysis_out/input_res_sweep_25/cache/{r2048,u4096}/*.json` (hashed in that
directory's `SHA256SUMS`; `python scripts/analysis/input_res_sweep_25.py sums` verifies them) and
`analysis_out/usage_log.jsonl`. `results.json` sha256 `4e2153adba06…`, `results.md` sha256
`cd1702c6b901…` as committed; `--check` compares the full bytes. `--cities` and `--n-shifts`
change the inputs and the control and therefore the numbers; the committed files use the defaults
(all 11 splits, 20 shifts).
