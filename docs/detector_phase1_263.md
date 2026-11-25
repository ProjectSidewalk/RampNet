# Detector Phase 1 (#263): recipe arms and an upscaled horizon-band pass

Issue [#263](https://github.com/ProjectSidewalk/RampNet/issues/263).
**Status: proposed.** Stage 0 is under way on [PR #295](https://github.com/ProjectSidewalk/RampNet/pull/295)
(not merged): 0a and 0b have run on the CPU, and 0c is written and not run. Nothing past Stage 0
has run. Every choice in this document is proposed and open until Jon decides it. The only
exceptions are the items under **Decided** in §11, each with who decided it and when.

**What this document is.** It is the plan of 2026-11-15, "Detector Phase 1 arms and an upscaled
horizon-band pass", revised after a review the same day. PR #295 executes its Stage 0 and quotes its
sections 3, 4 and 7 verbatim (`horizon_band_263.md`, Appendix A, on PR #295's branch). It replaces the
Phase 1 design of 2026-10-22, which lived at this same path on the unmerged branch of
[PR #270](https://github.com/ProjectSidewalk/RampNet/pull/270) (commit `c3dcb25e`). Committing it on
2026-11-25 changed three things:

1. Content carried over from #270's design: the defect, ceiling and warm-start columns of the arm
   table, the zoom op spec, the cost table with its sources, the durability items, and the list of
   what this document does not decide.
2. Corrections from the review of #270
   ([PR #270 comment](https://github.com/ProjectSidewalk/RampNet/pull/270#issuecomment-6935046651)),
   findings 1 to 12, where they apply to this plan. The History section lists each one.
3. Numbers re-read from committed files. PR #295's committed Stage 0 results replace the plan's
   scratch numbers. Where the plan and a committed file disagreed, the committed file wins, and the
   History section lists each change.

Numbers cited to PR #295 are on branch `analysis/horizon-band-263` (`docs/horizon_band_263.md`,
`analysis_out/horizon_band_263/`) and reach `main` only when that PR merges. Pixels are heatmap px on
the 512×1024 grid unless stated. S7 and US7 are the same seven splits: richmond, bend, clovis,
morgantown, annapolis, paterson, gainesville (`seed_variance_51_135.md`).

## 1. Goal, and what Phase 1 is for

Jon decides (a) whether any Phase 1 recipe arm yields a checkpoint that replaces the shipped one
(`projectsidewalk/rampnet-model` revision `606a119`), and (b) whether a second, upscaled pass over a
horizon band enters production.

- **In domain, most misses are recipe defects.** On the seven US splits at 0.30, 427 misses split
  merged 29.0%, sub_threshold 38.9%, localization 2.1%, silent 30.0% (`analysis_out/miss_taxonomy.json`,
  [`data_scaling_59.md`](data_scaling_59.md)).
- **On the open rigs, the deficit is mostly calibration and the near field.** On the GoPro Max pool
  (laurens_mapillary + bayonne) at 0.30, sub_threshold is 52.6% of misses against 38.5% silent. In
  recall points the excess over the #46 reference is +0.178 against +0.127, a difference of +0.051
  [−0.008, +0.109]. Strict-silent near-field misses are 51 of 407 near ramps against 16 of 1,360 on
  the reference ([`open_rig_misses_258.md`](open_rig_misses_258.md)). On Lyon, 464 of 677 sure misses
  (69%) are within about 24 m, 523 (77%) tilt-corrected, and recall is 0.38 even under about 7 m
  ([`lyon_review_159.md`](lyon_review_159.md) §10).

Phase 1 attacks the first reading. No GSV-only arm adds open-rig positives, so for the open rigs
Phase 1 must not regress them. That is not automatic: 2,000 fine-tune steps at LR 1e-5 from the
released weights cost −0.057 max-F1 [−0.085, −0.034] on laurens_mapillary while gaining +0.005 in
domain ([`aug_transfer_82.md`](aug_transfer_82.md)), and the control arm replicated the loss
(§2).

## 2. What exists

- **Control arm** (#271, merged 2026-11-01, [`control_arm_263.md`](control_arm_263.md)): the released
  weights plus 2,000 GSV steps at LR 1e-5, three seeds, scored on all 14 reviewed splits. On the
  nine-split US pool, F1 at 0.30 is −0.011 [−0.017, −0.006], every seed below zero.
  laurens_mapillary max-F1 is −0.050 [−0.073, −0.025], every seed. Open-rig 4 max-F1 is −0.004
  [−0.015, +0.004]; the three seeds' max-F1 are 0.691 / 0.700 / 0.684 (released 0.696 plus each
  seed's contrast), an SD of 0.008. On open-rig 3 (budapest_district5 left out), the pool this
  plan's rules read, the control looks worse: max-F1 −0.012 [−0.021, +0.001] and F1 at 0.30 −0.005
  [−0.020, +0.010]. That max-F1 point estimate is below the ship rule's −0.010 floor (§6), while its
  interval is not wholly below zero; so the unchanged recipe's own 2,000-step fine-tune would fail
  that floor against released. Seeds mostly move the operating point: the seed-to-seed F1 range
  at 0.55 is 0.086 on the open-rig pool. The doc proposes, without a decision, that every later arm
  be read against its same-seed control, with max-F1 beside every fixed threshold.
- **Seed variance** ([`seed_variance_51_135.md`](seed_variance_51_135.md)): macro US7 F1, the mean of
  the seven per-split F1s with each replicate at its own sao_paulo-selected threshold. Campaign B's
  nine replicates: mean 0.8227, s_B 0.0094 [0.0064, 0.0180]. The shipped checkpoint scores 0.8427,
  2.1 s_B above the mean, or +1.1 to +3.1 across the interval on s_B. manual_gold was not pooled
  there.
- **Resolution** ([`input_res_sweep_25.md`](input_res_sweep_25.md),
  [`two_scale_197.md`](two_scale_197.md)): a full 2× pass costs 2.35 s/pano GPU-side against 0.61 at
  1× (1.5×: 1.33). r4096 and u4096 are indistinguishable, so upsampling to 2× acts like real 2× pixels.
  At 2×, recall at 25–40 m rises from 0.543 to 0.750 and at 40 m+ from 0.273 to 0.558, while 0–8 m
  falls from 0.841 to 0.611. The two-scale union recovers +105 ramps for +212 FPs on the US pool at
  about 4.8× the GPU time, and reads NO BETTER THAN A THRESHOLD by an unrounded lower bound of
  −0.00008.
- **Decode and seam: no recall lever.** The Gaussian sub-cell decode moves positions by −0.727 px
  [−0.783, −0.673] on manual_gold and no detection metric
  ([`subcell_decode_221.md`](subcell_decode_221.md)). Seam-wrapped NMS removes 7 of 3,351 city peaks
  and 8 of 3,981 manual_gold peaks at 0.30, +0.00066 [+0.00006, +0.00135] F1 under GT-B, and stays
  opt-in (Jon, 2026-11-16, [`seam_nms_default_248.md`](seam_nms_default_248.md)). Production recall on
  seam-band ramps is 0.96, 24 of 25 ([`rampnet1_findings.md`](rampnet1_findings.md)).
- **Stage 0a and 0b** (PR #295): results in §4.
- **Argon teacher** (PR #294, open): stopped at its Stage 0 gates, so no Argon pseudo-labels exist
  for arm B.
- **Training data**: `rampnet_dataset_v2` was purged on 2026-11-12 and needs a re-stage before any
  rung (§8).

## 3. Split roles

| | selection | held out, read once |
|---|---|---|
| band | S7 (already read at 2× in #196/#197); sao_paulo for thresholds | manual_gold, juneau, bayonne (never run upscaled) |
| arms | sao_paulo (threshold, checkpoint); manual_gold (rung 1 gate) | macro US7; open-rig 3 (laurens_mapillary, bayonne, juneau), with budapest_district5 and laurens_gsv reported beside |

- **Rung 1 reads manual_gold only** (and sao_paulo for thresholds), as the 2026-11-15 plan has it.
  Every rung 1 checkpoint is scored on all 14 splits in one pass, so that one scorer and one cache
  serve both rungs, but the US7, open-rig and laurens_gsv caches are written and not read until the
  rung 2 / ship read. That read is their one read. The alternative, open-rig 3 as a named rung 1
  guard (which takes it out of "read once"), is listed in §11 as Jon's call.
- laurens_mapillary was read at 2× in #197 and again in the plan's 2026-11-15 scratch run. It enters
  no band rule and is reported only.
- budapest_district5 is in `miss_decomposition.HELD_OUT` (low-confidence single-rater GT), so the
  no-regression pool is **open-rig 3**, with open-rig 4 (adding budapest_district5) shown beside it,
  as in `control_arm_263.md`. laurens_gsv is the same-town control.
- **Caveat.** S7 chose the band's core with the shipped weights, and US7 is the same seven splits. A
  US7 read of arm B *with* the band inherits that choice, so it is not held out for the band's
  configuration. B's single-pass US7 read is.

## 4. Stage 0 (no training)

**0a. Merged pairs (CPU). Run in PR #295.** On eight splits (`US_SPLITS` minus juneau, which is held
out until 0c): 136 pairs, 25 under 8 px Chebyshev (18%), so arm C is not triggered (the bar is more
than one third). 38 pairs outside paterson are 12 px or more apart, against a bar of 30, so arm A
runs. Both metrics give the same verdicts. On S7 alone the arm A margin is thin: 31 pairs. S7's 24
of 124 pairs under 8 px were already in `open_rig_misses_258.md`. juneau adds 10 merged pairs, none
under 8 px. Paterson holds 80 of the 136 pairs.

**0b. Band simulation (CPU, exploratory, binds nothing). Run in PR #295.** The in-core peaks of the
committed full-pano upscaled caches are merged into the 1× peaks at 0.30. On S7 at 0.30, 1× alone
scores F1 0.8417. The rows below use the standard scorer (unmatched detections beyond 25 m count as
FPs), with paired pano bootstrap intervals:

| config (core rows) | ramps gained | FPs added | ΔF1 against 1× | band minus matched-recall threshold, ΔF1 |
|---|---|---|---|---|
| 2x-wide (0.45–0.60) | +100 | +174 | −0.007 [−0.015, +0.001] | +0.035 [+0.025, +0.044] |
| 2x-narrow (0.47–0.56) | +89 | +146 | −0.005 [−0.013, +0.003] | +0.029 [+0.020, +0.039] |
| 1.5x-wide | +106 | +146 | −0.000 [−0.008, +0.008] | +0.049 [+0.039, +0.058] |
| 1.5x-narrow | +95 | +120 | +0.002 [−0.005, +0.010] | +0.040 [+0.031, +0.050] |

- On S7 pooled, every configuration beats the matched-recall threshold, and none gains F1 against
  the 1× pass. The pooled reading leans on the GSV splits. Mapillary4 (richmond, annapolis, clovis,
  morgantown) gains 37 ramps for 107 FPs at 2x-wide, ΔF1 −0.022 [−0.033, −0.011]. GSV3 (bend,
  paterson, gainesville) gains 63 for 67.
- The 1.5× cache is real pixels where the native image has them. Production and 0c upsample, which
  is untested at 1.5×, so the 1.5× rows are optimistic.
- Leave-one-split-out picks the narrow core on 7 of 7 splits at both scales, but narrow minus wide is
  not separated from zero on F1. Wide stays the binding configuration.
- Treating unmatched far detections as ignore points instead (the upper bound), three of the four
  configurations gain F1 against 1×. The two bounds read F1 against base differently.

**0c. Real strips with the shipped weights (makelab2 A40, about 3 to 4 GPU-h). The binding gate.
Written in PR #295, not run.** Jon approves and starts it.

- **Strip.** 1.5x-wide is tested first. The plan's 768×6144 (5.0 MP) is off the backbone's stride-32
  grid, so PR #295 uses the smallest grid-aligned window that keeps at least 0.05 of context:
  864×6528, 5.64 MP. The window includes 192 px of circular wrap each side, and the head is set to
  input/4. Only core peaks are kept, merged by #197's seam-wrapped `dedupe`. The narrow core and the
  2× strip also run. The backbone's GlobalResponseNorm layers normalise over the whole feature map,
  so a strip's features can differ from the full pass's however much context it carries. The size
  of that effect on the shipped weights is unmeasured.
- **Context rule (S7).** The reference is every full-pass peak at 0.30 or above in the core, from a
  full-pano pass at the same scale with the same input/4 head and scaled `min_distance` (15 at 1.5×,
  20 at 2×). The strip must reproduce at least 95% of these, with |median score shift| under 0.05.
  **Decided (Jon, 2026-11-25): the lenient reading binds.** A reference peak counts as reproduced when
  a strip core peak lies within the match radius at any score at or above the floor. The strict
  reading, which also requires the strip peak to reach 0.30, is reported beside it. Jon chose the
  plan's (lenient) reading on the reasons below, which Claude set out for him and he accepted: the
  median-shift clause bounds a systematic drop in score; the sao_paulo calibration exists to move
  the band threshold when strip scores shift; and a strip that loses ramps fails the binding
  manual_gold gate as a fail, never as a false win. PR #295 records the decision and these reasons in
  `horizon_band_263.md`, "Decided before any 0c data exists (Jon, 2026-11-25)" (commit `04d06c88`).
- **Calibration (sao_paulo).** A band threshold is set, to equal the 1× in-core precision at 0.30,
  if the median paired score shift exceeds 0.03 or in-core precision at 0.30 differs by more than
  0.05.
- **Gate (manual_gold), binding for 1.5x-wide.** The recall gain's interval is above zero, band minus
  matched-recall threshold F1 is above zero, and F1 against the 1× pass is not wholly below zero.
  All three must hold under both bounds: unmatched detections beyond 25 m (or at or above the horizon)
  scored as FPs, and dropped.
- **Open-rig claim (juneau + bayonne pooled).** The recall gain is at least +0.02 with an interval
  above zero; otherwise the band is not offered for open sources. juneau's and bayonne's native panos
  are not on the Hub, so a clean clone cannot re-run this part.
- **On failure:** the band is dropped for the shipped weights and kept only as a read for arm B.

## 5. Training arms

Each arm is one change against its control. "Ceiling" is the most recall its target bucket could
return if the arm fixed that bucket entirely; none will. "Warm start" says whether the arm can be
screened as a fine-tune from the released weights (same architecture, same parameter names) or
needs the from-scratch rung.

| order | arm | change | defect it targets | ceiling | warm start | seed-1 kill | status |
|---|---|---|---|---|---|---|---|
| 1 | **F. fine-tune LR** | `--preset finetune` (LR 3e-6) instead of control263's 1e-5 | fine-tuning itself costs transfer (#82, #271) | none; it is the control | yes | n/a; 3 seeds | proposed |
| 2 | **B. zoom augmentation** | a geometric `zoom` op (spec below), range [0.5, 2.0] | scale: far-field sub_threshold and silent misses | on S7 at 0.30, 153 misses at 25 m or beyond (of 300 GT); the frozen full 2× pass recovers 70 of them under the scorer's 1:1 match (59 of 112 under the plan's approximate count; PR #295, `simulate.md`) | yes | manual_gold recall beyond 25 m up less than +0.03 at the control's FP count | proposed; the op does not exist |
| 3 | **A. target σ** | Gaussian target σ 10 → 6 (`generate_heatmap_from_points`); `min_distance` stays 10 | merged pairs (#46) | S7: merged is 29.0% of misses, 124 pairs, 0.060 recall points; with `min_distance` 10 kept, at most the 78 pairs above it (0.038), 35 of them outside paterson; 80 of the 124 are paterson's. manual_gold: 185 of 423 misses (0.047 recall points); its separations are not computed | yes | manual_gold merged misses fall less than 10% | proposed; 0a permits it |
| 4 | **rung 2** | the passing arms as one bundle, from scratch | as its parts | as its parts | no | n/a; 3 seeds plus one tie-in seed | proposed |
| — | **C. stride-16 head** | FPN-style head on the stride-16 feature, or a learned upsample (#221) | the 8-px peak lattice | localization is 2.1% of misses (9); the Gaussian decode moves positions by −0.727 px but no detection metric | no (new parameters) | — | not scheduled (proposed): 0a did not trigger it |
| — | **D. circular padding** | `padding_mode="circular"` on the horizontal axis | seam response | about 0.001 F1 is at stake at the seam (`rampnet1_findings.md`, `seam_nms_default_248.md`) | yes | — | dropped (proposed) |
| — | **E. flat-ramp supervision** | none in Phase 1 | flush and minimal-reveal ramps (#151) | — | — | — | deferred: needs labels, not a recipe change |

Out of scope in the #263 body (each item there returns after its decision point), and not proposed
here: the backbone swap (#107), Richmond or Bayonne panos in training, any frontier-model spend. `--mask-norm pixels|weight` ([`sparse_supervision.md`](sparse_supervision.md))
belongs to Phase 2, where sparse panos exist. If 0a had triggered C, it would run after rung 2.

**The zoom op.** A geometric op in `rampnet/augment.py`: a random scale in [0.5, 2.0] of the
equirectangular canvas (downscale and pad, or crop a window and upscale), with the points transformed
the same way, applied before the Resize. The existing `downscale` op is resolution degradation, not
object scale, so it is not this. Equirectangular zoom is not camera zoom, and how the op treats the
horizontal wrap is part of the implementation and untested.

**Why σ = 6 and not a sweep.** The target is a per-pixel max of Gaussians (`stage_two/train.py`), so
it already has two modes for any pair. A smaller σ deepens the dip between them: at the median
separation (12 px), from 0.84 of the peak at σ=10 to 0.61 at σ=6. What merges a pair is the output, a
bilinear ×8 upsample of the stride-32 map whose peaks sit on an 8-px lattice
(`subcell_decode_221.md`), so two peaks that survive `min_distance` 10 are about 16 px apart. Whether a
deeper target dip moves the output is what the merged-bucket readout tests. σ=4 would put the target
under the 8-px cell. One value, and a second only if A passes and the bucket shows headroom. (The crop
model trains at σ=12 on a 256×88 map, `crop_decode_221.md`; that is a different scale and not
evidence either way.)

**Why zoom has the largest forecast and the least evidence.** The frozen sweep showed the effect is
object scale, not pixels, and that the two scales trade near against far. A model trained across
scales is the only arm with a forecast on record: +10 recall points for a resolution retrain,
saturating near 0.88 ([`detection_recall_analysis.md`](detection_recall_analysis.md)), never tested.
On the open rigs the deficit is mostly near-field (§1), the band where the frozen 2× pass lost
recall, so zoom is not expected to close the open-rig gap. The open pool is read as no-regression.

**Rung 1, the fine-tune screen.** Each arm runs 2,000 optimizer steps at global batch 16 from the
shipped weights, with the control launcher's other settings (`stage_two/run_finetune_control.slurm`:
`ckpt-all`, 4 GPUs, about 12 GPU-h and 3 h per seed). Seed 1 runs first; seeds 2 and 3 run only if
the seed-1 kill is not met. Scoring uses the #82 scorer unchanged (`operating_point_curve.py extract
--score-floor 0.05 --min-distance 10`, no TTA, one cache per checkpoint).

- **LR rule.** A and B use 3e-6 with F as their control, unless F is more than 0.005 F1 below
  control263 on manual_gold; then they use 1e-5 with control263. One fresh 1e-5 control seed (about
  12 GPU-h) ties the two dataset copies together.
- **Pass.** The same-seed mean contrast on manual_gold has an interval above zero, and 2 of 3 seeds
  agree.
- **No open-rig rule at rung 1.** Rung 1 reads manual_gold only (§3); the open rigs are read once,
  at the ship read (§6). The alternative is in §11: a rung 1 guard that rejects an arm whose
  same-seed contrast on open-rig 3 is below zero by more than control263's seed SD on the same read
  (from the per-seed rows of `control_arm_263.md`, about 0.004 at max-F1 and about 0.012 at F1
  @0.30). It would catch a transfer loss before rung 2 is spent, at the cost of the open rigs' one
  read.
- **A null on B is inconclusive**, since 2,000 steps may not teach scale. B then leaves the default
  bundle, and Jon decides whether to run a from-scratch seed.
- **B replaces the band** if, on manual_gold, its single-pass recall beyond 25 m is within 0.02 of
  shipped-plus-band and its 0–8 m recall is within 0.01 of shipped. The split is proposed here (the
  2026-11-15 plan names none): manual_gold is where 0c measures shipped-plus-band, it is readable at
  rung 1, and it avoids the S7 overlap in §3's caveat. Jon's call (§11).
- **GPU type.** control263 was scored on A40 for released, s1 and s2, and on L40S for s3
  (`control_arm_263.md`, Caveats); the cross-type difference was never measured. Score each arm on
  its same-seed control's GPU type, or measure the difference on one split first.

**Rung 2, from scratch.** The passing bundle trains with the full recipe (1 epoch, 9,378 steps, LR
1e-5 constant, global batch 16, 16 GPUs, selection on auto-label val loss; `run_train_seed.slurm`), 3
seeds, plus one tie-in seed of the unchanged recipe on the re-staged data, because the commit the
Campaign B s1–s3 replicates trained at is not recorded. The baseline is Campaign B's nine
replicates. Campaign B was scored on makelab2, so its nine checkpoints are re-scored on klone in the
same pass as the arms, through one scorer on one GPU type, as #271 did.

**Fix before any submission.**

- `scripts/analysis/aug_finetune_82.py`'s pool labelled `"US7 (miss_decomposition.US_SPLITS)"` holds
  eight splits (it includes laurens_mapillary), and `US_SPLITS` now has nine (juneau). Rename the pool
  and define US7 as the seven splits of `seed_variance_51_135.md`.
- `stage_two/run_train_seed.slurm` defaults `DATA_ROOT` to the deleted `rampnet_dataset`. Pointing it
  at `rampnet_dataset_v2` is not enough, because that copy is purged too (§8).

## 6. Readout and ship rule

- **Rung 1 reads manual_gold only:** same-seed arm minus control, paired pano bootstrap, 2,000 draws,
  with the pass rule and seed-1 kills of §5. US7 and the open rigs are not read at rung 1 (§3).
- **Primary, at the rung 2 / ship read: macro US7 F1**, each checkpoint at its own
  sao_paulo-selected threshold, with F1 at 0.30 and 0.55, max-F1, and each seed's own contrast
  beside it. Welch over seed means, 3 arm seeds against Campaign B's nine, with the paired pano
  bootstrap per seed (stratified by split, 2,000 draws, macro mean per draw) beside it. manual_gold
  is a secondary read there, not pooled in.
- **Minimum detectable effect.** With s_B = 0.0094 and n = 3 against n = 9, a seed-mean Welch test
  (3.5 df) has 80% power at about 0.025 F1; at 0.015 its power is about 0.42. Rung 1's paired reads
  have no stated MDE. Arm A's whole reachable ceiling (0.038 recall points) is about the size of this
  MDE in F1 or below it, so A rests on the per-bucket and paired-pano reads, not on the Welch test.
- **Per-bucket readout.** `miss_taxonomy.py` on each arm (on manual_gold at rung 1, on US7 at the
  ship read), so that A is judged on the merged bucket and B on the far-field buckets, not only on
  F1. This is how a negative F1 result still says whether an
  arm did what it was built to do.
- **Ship rule, one named checkpoint** (seed chosen on sao_paulo): the macro US7 interval against
  **shipped** is above zero; open-rig 3 F1 and max-F1 are not wholly below zero and their point
  estimates are at least −0.010 (open-rig 4 and laurens_gsv shown beside); manual_gold is not wholly
  below zero. Beating the replicate mean is reported and is never sufficient. For scale: the
  unchanged recipe's 2,000-step fine-tune (control263) sits at −0.012 max-F1 on open-rig 3, below
  this floor (§2).
- **Unjudged detections.** New far-field true positives that nobody judged count as FPs, which
  biases against B and the band. Every F1 for B and the band is reported under both bounds: far
  unmatched detections as FPs, and as ignore points. No rating is planned. Only if the two bounds give
  different verdicts: one audit of 100 far-field detections, last. Whether manual_gold marks every
  ramp at 30–40 m is not checked.
- **Operating points.** 0.30 ([`operating_point.md`](operating_point.md)), with 0.55 for continuity.
  budapest_district5 stays out of every pool (it is in `miss_decomposition.HELD_OUT`, which also
  lists sao_paulo, bayonne and laurens_gsv; this plan uses those three as the roles in §3 say).

## 7. Production cost of the band

Base 0.61 s/pano GPU-side for the 2048×4096 pass (8.39 MP). Pixel-linear estimates with PR #295's
grid-aligned strips: 1.5x-wide 5.64 MP, about 1.67×; 1.5x-narrow 4.39 MP, about 1.52×; 2x-wide 9.19 MP,
about 2.10×. (The plan's 1.6×, 1.4× and 2.1× used unaligned windows.) 0c measures the real time and
peak memory.

- **Limit: 1.75× GPU-side**, because the forward pass is the serialized step on GPU-bound GSV runs;
  rate-limited open sources are download-bound. The limit is Jon's to change. PR #295's `report`
  names it in the decision line and does not enforce it.
- The 2× strip needs the limit lifted and must gain at least 1.25 times the 1.5× strip's ramps on
  manual_gold.
- **Labeler contract, only on a ship decision:** a per-detection pass tag; a manifest binding that
  refuses mixes; the top-50 cap and the 0.1 storage floor applied after the merge; and a mapping of
  any new threshold, band or checkpoint, onto the 0.30 and 0.55 tiers. Without these, the 0.30 tier
  silently applies to band detections.

## 8. Compute, cost and durability

All training and scoring is on klone `ckpt-all` at $0 (`paid: false`); each job's rows go into
`analysis_out/compute_log.jsonl` via `scripts/analysis/slurm_usage.py` after it runs. 0c runs on
makelab2 and writes its own `paid: false` rows to `analysis_out/usage_log.jsonl` (PR #295).

| step | GPU-h | calendar | source |
|---|--:|---|---|
| Stage 0a, 0b | 0 | done | CPU, about 1 minute (PR #295) |
| Stage 0c | about 3 to 4 | hours | PR #295, from the #196 timings |
| rung 1, F, 3 seeds + scoring | about 38 (49 at 1.3×) | about 3 h per seed, seeds in parallel | training about 12 GPU-h per seed: control263 9.38–11.94 (`control_arm_263.md`), #82 11.7–14.2 per uninterrupted run, 1.3× with preemption ((79.51 + 23.13) / 79.51, `aug_transfer_82.md`); scoring about 0.75 GPU-h per checkpoint on 14 splits (#271: 0.53–0.81, `compute_log.jsonl` jobs 42152649–52) |
| rung 1, B, 1 to 3 seeds + scoring | about 13 to 38 (16 to 49) | | as above |
| rung 1, A, 1 to 3 seeds + scoring | about 13 to 38 (16 to 49) | | as above |
| rung 1, fresh 1e-5 control seed | about 13 (16) | | as above |
| rung 2, one bundle, 3 seeds | about 170 (280 at 1.67×) | Campaign B ran 6 replicates within 5 h 15 min | about 56 GPU-h per epoch and 1.67× preemption overhead (`stage2_training_cost.md`); 5 h 15 min from `seed_variance_51_135.md` |
| rung 2 tie-in seed | about 56 (93) | | as above |
| rung 2 scoring, 4 checkpoints, plus Campaign B's nine re-scored | about 10 | hours | 0.75 GPU-h per checkpoint |
| **Phase 1 through rung 2, from here** | **about 340 to 550** | **about a week if klone is free** | lower: no preemption, and the fewest seeds that can reach rung 2 (one of A or B at three seeds, since rung 1's pass needs 2 of 3 seeds to agree, the other killed at seed 1): 3 + 38 + 38 + 13 + 13 + 170 + 56 + 10 = 341; upper: all seeds, 1.3× on rung 1, 1.67× on rung 2 |

Rung 1 scoring writes caches for all 14 splits so that the ship read needs no second scoring pass;
only manual_gold and sao_paulo are read at rung 1 (§3). If no arm passes rung 1, rung 2 does not run
and Phase 1 stops at about 80 to 170 GPU-h. C, if it
were ever triggered, would add about 170 to 280. The plan's own total (4 + 140 + 235–345 =
380–490) did not follow from its rows: its rung 1 rows sum to 72–120 before scoring, not 140. This
table replaces it.

**Durability, on the critical path.**

- **The training data is gone.** `rampnet_dataset_v2` was purged on 2026-11-12: on 2026-11-16 its
  `train/`, `val/` and `test/` were empty and only `_quarantine/` remained (checked on klone and
  reported in the PR #270 review comment linked above; not in a committed file). Every rung needs a
  re-stage first. `stage_two/run_restage_parquet_263.slurm` did it last time as job 42090420, 4 h 14 m
  plus 37 m after a requeue on CPU, $0. It needs about 431 GiB of shard cache beside 462 GiB of output
  on scrubbed ([`restage_263.md`](restage_263.md)), and the first training job has to read the copy
  within about 21 days of the re-stage.
- **Checkpoints.** As of the same 2026-11-16 check, Campaign B's checkpoints (nine `best` plus nine
  `epoch_1_step_9378`, `/gscratch/makelab/jonf/seedvar/`) and control263's three
  (`/gscratch/makelab/jonf/control263/`) were still there.
- **Space.** `/gscratch/makelab` was at 968 of 1,024 GB and 969,056 of 1,000,000 files on 2026-11-16
  (`hyakstorage`, same source); the file count is now the tighter limit. A `KeypointModel`
  checkpoint is about 344 MiB: `control_arm_263.md` lists control263's three at 344 MB each, and the
  two Stage 1 crop-model checkpoints in [`replication.md`](replication.md), the same architecture, are
  360,358,458 bytes each. Rung 2 keeps two per seed.
- **Ledger.** control263 has its 11 rows (37.29 GPU-h). Campaign B's training jobs (39880702,
  39880703, 39880706 and the six seed 4–9 jobs) have no row in `compute_log.jsonl`. `sacct`
  retention is finite, so they should be recovered now with `slurm_usage.py` from a `sacct -D` pull.

## 9. Files and branches

- `analysis/horizon-band-263` (PR #295): `merged_separation_263.py`, `horizon_band_263.py`
  (`simulate`, `extract`, `calibrate`, `report`, `run0c`, `figures`, `--check`), tests,
  `docs/horizon_band_263.md`, `check_all` entries.
- `feat/phase1-arms-263` (not started): the `zoom` op with its point transform, `--target-sigma`, the
  launcher, the `DATA_ROOT` and pool fixes (§5), and a results doc with a replication section, a
  number-location table and `--check`.
- The labeler, only on a ship decision: the contract in §7.

## 10. Schedule and Jon's part

The plan's dates (0c from 11-16 to 11-18, the gate on 11-19, rung 1 from 11-20 to 11-24, rung 2 to
about 12-03) have passed without 0c running. No new dates are set. Jon's part: approve and start 0c on
makelab2; `klone open`; the re-stage and space on `/gscratch/makelab` (§8).

The plan shared its seeds and control with the Argon-teacher plan (control263 as that plan's C0, and
the fresh control seed serving both). That plan stopped at its Stage 0 (PR #294), so the sharing is
moot unless it resumes. If it does, its arms should use whatever LR this plan's LR rule picks.

## 11. Decisions

**Decided.**

- **0c's context rule binds on the lenient (plan) reading** (Jon, 2026-11-25). The rule and the
  reasons he accepted are in §4; PR #295 records both (commit `04d06c88`). The recommendation in
  PR #295's review thread had been the strict reading.
- **PR #295 merges as reviewed, as Stage 0 with 0c not run** (Jon, 2026-11-25). The two 0c code
  fixes still open there (a null band threshold under `calibrate --smoke`, and the calibration sha256
  check moved into `guard_extract`) go into a follow-up PR before 0c can run. PR #295's
  `horizon_band_263.md` records this ("to be fixed in a follow-up PR before 0c can run", commit
  `04d06c88`).

**Open, and Jon's.**

- Whether rung 1 opens before #263's decision point. The #263 body puts head, σ and resolution
  changes after that point. Both inputs it waited on are in (the Lyon review, #277; the control arm,
  #271), and no decision has been posted on #263 since 2026-10-22. The case for opening now: it is $0,
  needs no ratings, and its arms do not touch the open-imagery question. The case against: klone time
  that the verifier (#258) and Phase 2 also want.
- Starting 0c on makelab2, once the follow-up PR with the two 0c code fixes has merged.
- Whether to store all of `simulate.json`'s contrasts at six decimals (raised in PR #295's review
  thread; not a before-0c item).
- σ (6 proposed) and the zoom range ([0.5, 2.0] proposed, matching the frozen sweep's 2×).
- The 1.75× cost limit.
- The primary statistic, the no-regression pool (open-rig 3) and the ship rule (§6).
- Whether rung 1 reads manual_gold only (proposed, §3) or also applies a rung 1 guard on open-rig 3
  (§5), which would take open-rig 3 out of "held out, read once".
- The split for the "B replaces the band" rule (manual_gold proposed, §5).
- Arm C not scheduled and arm D dropped (both proposed, §5).
- The reading rule proposed in `control_arm_263.md` (same-seed control, max-F1 beside every fixed
  threshold), which this plan adopts.
- The LR rule (§5).
- A from-scratch seed for B after an inconclusive rung 1 null.
- The re-stage, space on `/gscratch/makelab`, and recovering Campaign B's ledger rows (§8).
- Publishing juneau's and bayonne's native panos to the Hub, which a clean-clone re-run of 0c's
  open-rig claim needs.
- Any cloud fallback and its spend.

## 12. Risks, and what is not verified

**Risks.** The shipped checkpoint may be a lucky seed (2.1 s_B above Campaign B's mean); fine-tuning
costs transfer; equirectangular zoom is not camera zoom; upsampled 1.5× input is untested; the strip's
GlobalResponseNorm statistics differ from the full pass's; a new checkpoint shifts confidences, so its
threshold has to be remapped onto the labeler's tiers; arm A's ceiling is about the size of the MDE.

**Not verified.** klone state since 2026-11-16 (the dataset, control263 and Campaign B checkpoints);
strip time, memory and context effect; upscaled input on manual_gold, juneau and bayonne; whether
manual_gold marks every ramp at 30–40 m; the +10-point retrain forecast; labeler end-to-end
throughput; cloud prices; the GPU-type scoring difference; how many of manual_gold's 185 merged misses
are the 10 seam double-marks in its GT ([`seam.md`](seam.md)).

## History

- **2026-10-22.** First Phase 1 design, [PR #270](https://github.com/ProjectSidewalk/RampNet/pull/270)
  (`c3dcb25e`, branch `docs/detector-phase1-263`, never merged). Superseded by this document.
- **2026-11-15.** The plan this document commits, written by Claude Fable 5.2 (`claude-fable-5-2`) and
  reviewed the same day by Claude Opus 5.5 (`claude-opus-5-5`). The review is not committed. Changes
  after it: held-out overlap removed (band rules read S7 only, rung 1 gates on manual_gold, US7 held
  for the ship read); 0b made exploratory and 0c the binding gate; a 1.5× strip tested first and the
  cost limit set at 1.75× with its reason; "GSV lever" withdrawn for per-split figures; the statistic
  named macro, with the s_B interval, an MDE, an LR rule and a matching control; the two-bound F1;
  F first, seed-1 kills, C conditional, a B null marked inconclusive; numbers for the calibration and
  B-replaces-band rules; the strip compared at the same head and `min_distance`; the labeler
  contract; misses by range, bucket ceilings and replication items; D dropped.
- **2026-11-25.** Committed here by Claude Opus 5.5 (`claude-opus-5-5`), with the review of #270
  applied:
  1. The purged training data (§8).
  2. Arm A's ceiling with `min_distance` kept: 78 pairs and 0.038 recall points, not 124 and 0.060;
     35 outside paterson; S7's under-8-px count was already in `open_rig_misses_258.md` (§4, §5).
  3. The MDE: about 0.025 F1 at 80% power, not 0.015 (§6). The plan had repeated #270's error.
  4. The primary statistic as macro US7, with the rung 2 readout (§6). The finding's rung 1 US7
     read was written for #270's design, where US7 was not held out; this plan keeps rung 1 on
     manual_gold only, as the 2026-11-15 plan has it.
  5. The no-regression pool as open-rig 3, budapest_district5 held out, and Campaign B re-scored on
     klone with the arms (§3, §5, §6). The ship rule's "open-rig four" became open-rig 3. The
     finding's rung 1 rejection rule on that pool is not adopted, for the same reason as item 4; it
     is listed in §11 as an alternative for Jon.
  6. The σ rationale, from the per-pixel max in `train.py` (§5).
  7. The cost table, rebuilt from measured sources (§8).
  8. The control arm reported as done, with its result (§2).
  9. D dropped, with the seam numbers (§5).
  10. The zoom arm's open-rig sentence, after the Lyon audit (§5).
  11. C's ceiling cell, without the claim that the decode took most of it (§5).
  12. Stale text: storage figures of 2026-11-16; control263's ledger rows, with only Campaign B's
      missing; US7 defined explicitly; the scoring GPU-type caveat; the near-field counts cited to a
      committed doc; "every choice is proposed". The review's line that `compute_log.jsonl` "holds no
      klone job submitted in September 2026" is too broad: 27 klone rows from September exist, from
      other studies. What is missing is Campaign B's own jobs.

  Numbers changed against committed sources, besides those findings: B's far-field ceiling is quoted
  under the scorer's 1:1 match (70 of 153), with the plan's approximate count (59 of 112) beside it;
  the 1.5× strip is 5.64 MP and about 1.67× (PR #295's grid-aligned window), not 5.0 MP and 1.6×;
  arm A's manual_gold bucket is 185 of 423 misses, which the plan gave without its denominator. The
  plan's schedule and cloud cutoff dates are dropped as stale.
- **2026-11-25, review 1 of PR #306.** Rung 1 reads manual_gold only, and US7 and open-rig 3 stay
  held out for the ship read (the first commit of PR #306, `30ab22e6`, read both at rung 1); the cost floor
  for a Phase 1 that reaches rung 2 is about 340 GPU-h, not 320; the control arm on open-rig 3 is
  quoted beside open-rig 4; the "B replaces the band" rule names manual_gold (proposed); the
  decisions are dated and sourced to PR #295's commit `04d06c88`; smaller wording fixes.
