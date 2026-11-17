# Detector campaign, Phase 1: recipe arms on the GSV dataset (#263)

**Issue:** [#263](https://github.com/ProjectSidewalk/RampNet/issues/263).
**Status: proposed, 2026-10-22. Nothing in this document has been run or approved.** The #263 body
puts head, sigma and resolution changes out of scope until its decision point, which waits on the
Lyon review (#159) and the control-arm scores. This document is written now so that the arms, the
readout and the cost are settled before that point, not after it. Every choice marked *proposed* is
open until Jon decides it.

## 1. What Phase 1 is for

Phase 0 delivered the plumbing (restage, `--loss-mask`, the control arm) and two readings that
bound what a retrain can buy on the GSV data alone:

- **In domain, the misses are mostly recipe defects.** On the pooled seven US splits at 0.30, 427
  misses split merged 29.0%, sub_threshold 38.9%, localization 2.1%, silent 30.0%
  (`rampnet1_report.md`, `data_scaling_59.md`). The merged bucket is 124 adjacent pairs that the
  model emits as one peak; 87% of them sit within 2σ of the σ=10 training target and 78 of 124 are
  farther apart than the extractor's `min_distance`, so the target, not the extractor, is the lever
  (`data_scaling_59.md`). Their median separation is 12.2 heatmap px Euclidean, 11.4 Chebyshev
  (`analysis_out/miss_taxonomy.json`, `merged_separation_px`).
- **On the open rigs, the misses are mostly calibration, with a near-field silent excess.** On the
  GoPro Max pool at 0.30, sub_threshold is 52.6% of misses against 38.5% silent; the excess over the
  reference is +0.178 against +0.127, a difference of +0.051 [−0.008, +0.109] (`open_rig_misses_258.md`).
  The strict-silent excess is concentrated in the near field: 51 of 407 ramps against 16 of 1,360
  on the reference (#263, status comment of 2026-10-20).

Phase 1 attacks the first reading. It cannot fix the second: no GSV-only arm adds open-rig
positives, and the two arms that target the open rigs (sparse Lyon labels, the #258 verifier) are
Phase 2 and #258. What Phase 1 must do for the open rigs is **not regress them**, and #82 showed
that is not automatic: 2,000 fine-tune steps at LR 1e-5 from the released weights cost −0.057
max-F1 [−0.085, −0.034] on laurens_mapillary while gaining +0.005 in domain (`aug_transfer_82.md`).

## 2. Two things to read before any arm is submitted

Both are cheap and both change the design.

**Pre-check A: the control-arm scores (running, #263).** Jobs 42108716/7/8 finished; scoring is
submitted on branch `analysis/control-arm-263`. They give the one number Phase 1 cannot do without:
the **seed SD of a 2,000-step fine-tune** at n=3, which #82 had only as a single pair (control spread
+0.029 on the Laurens rig effect, larger than any arm effect it measured). They also say whether the
#82 transfer loss replicates. If the fine-tune control is below the released checkpoint on the open
pool by more than its seed SD, the fine-tune rung in §4 is not a usable screen at LR 1e-5 and arm F
(the LR arm) moves first.

**Pre-check B: the merged-pair separation histogram (CPU, minutes).** `miss_taxonomy.py` records
only the median and the count above `min_distance`. Extend it to dump the 124 per-pair separations
and plot them against three lines: the 8-px cell of the stride-32 head (#221), 2σ for σ = 10, and 2σ
for the candidate σ in arm A. The fraction of pairs **under 8 px** is the fraction no sigma change
can recover without the head change (arm C); the fraction between 8 px and 2σ_new is what arm A can
reach. If more than about a third of the pairs are under 8 px, arm C stops being optional. This is
the only new analysis Phase 1 needs before training, and it should be committed with a `--check`.

## 3. The arms

Each arm is one change against the control. "Ceiling" is the most recall the targeted bucket could
return if the arm fixed it entirely; none will. "Warm start" says whether the arm can be screened as
a fine-tune from the released weights (same architecture, same parameter names) or needs the
from-scratch rung.

| arm | change | defect it targets | ceiling (pooled US7 at 0.30) | warm start | status |
|---|---|---|---|---|---|
| **A. target σ** | Gaussian target σ 10 → 6 heatmap px (`generate_heatmap_from_points`); `min_distance` stays 10 | merged pairs (#46) | merged = 29.0% of misses, 124 pairs; reachable share set by pre-check B | yes | proposed |
| **B. zoom augmentation** | a geometric `zoom` op in `rampnet/augment.py`: random scale in [0.5, 2.0] of the equirectangular canvas (downscale and pad, or crop a window and upscale), points transformed with it, before the Resize | scale sensitivity (#25): at 2× input, far recall 25–40 m 0.543 → 0.750 and 40 m+ 0.273 → 0.558, near 0–8 m 0.841 → 0.611 (`input_res_sweep_25.md`); the higher-resolution retrain is forecast at +10 recall points, saturating near 0.88, with the warning not to add it to the operating-point gain (`detection_recall_analysis.md`, untested) | sub_threshold and silent in the far field | yes | proposed; the op does not exist (the `downscale` op is resolution degradation, not object scale) |
| **C. stride-16 head** | FPN-style head reading the stride-16 feature, or a learned upsample (#221 item 3) | 8-cell peak grid; needed only to let σ < 8 px resolve pairs | localization is 2.1% of misses and the Gaussian decode already took most of it (−0.727 px, +1 TP; `subcell_decode_221.md`) | no (new parameters) | proposed; gated on pre-check B |
| **D. circular padding** | `padding_mode="circular"` on the horizontal axis of every `Conv2d` in head and backbone (#82 body) | seam response (#132, #248): rolling the seam moves 12 of 25 seam-band ramps by > 0.05 against 0 of 77 controls | about 0.2% of peaks straddle the seam (72 pairs at 0.30 on 8,000 pano-checkpoints, `seam_nms_248.md`) | yes (no new parameters) | proposed as a rider on the from-scratch rung only; **cannot be scored until the manual_gold seam GT is merged** (8 of 10 adjudicated straddles are GT double-marks) |
| **E. flat-ramp supervision** | none in Phase 1 | flush / minimal-reveal ramps (#151) | — | — | deferred: needs labels, not a recipe change |
| **F. fine-tune LR** | `--preset finetune` (3e-6) instead of the control's 1e-5 | the #82 finding that fine-tuning itself costs transfer | none; it is the control for the screen | yes | proposed; runs only if pre-check A shows the 1e-5 control regressing the open pool |

Not in Phase 1, by the #263 body: the backbone swap (#107), Richmond or Bayonne panos in training,
any frontier-model spend. The `--mask-norm pixels|weight` choice (`sparse_supervision.md`) belongs to
Phase 2, where sparse panos exist.

**Why σ = 6 and not a sweep.** Pairs at the median separation (12 px) are 1.2σ apart under σ=10 and
2σ apart under σ=6, which is the separation at which two Gaussians of equal height first show two
modes with a clear dip. σ=4 would resolve more pairs on paper but puts the target under the 8-px
head cell, where pre-check B says the head cannot follow. One value, and a second only if A passes
and B shows headroom. (The crop model trains at σ=12 on a 256×88 map, `crop_decode_221.md`; that is a
different scale and not evidence either way.)

**Why zoom is the arm with the largest forecast and the least evidence.** The frozen sweep showed
the effect is object scale, not pixels (r4096 and u4096 indistinguishable), and that the two scales
trade near against far. A model trained across scales is the only arm with a forecast on record (+10
recall points for a resolution retrain), and that forecast has never been tested
(`two_scale_197.md` says so). It is also the arm
most likely to move the open rigs, where the GoPro Max optics put ramps at a different apparent
scale; that is a hypothesis, not a result, and the open pool is read as no-regression, not as the
target.

## 4. Two rungs, and the readout for each

**Rung 1, the fine-tune screen.** Each arm: 2,000 optimizer steps at global batch 16 from the
released weights, 3 seeds, the control launcher's settings otherwise (`stage_two/run_finetune_control.slurm`:
`ckpt-all`, 4 GPUs, about 12 GPU-h per seed, about 3 h calendar). Scored with the #82 scorer
unchanged (`operating_point_curve.py extract … --score-floor 0.05 --min-distance 10`, one cache per
checkpoint, no TTA) so that the three control263 seeds are the paired baseline. Read with
`aug_finetune_82.py`'s contrasts: same-seed arm-minus-control, seed-mean contrast, paired pano
bootstrap stratified by split (2,000 draws). **Fix first:** the script's pool labelled "US7" holds
eight splits (it includes laurens_mapillary); Phase 1 reads need the seven-split pool so the
Campaign B seed SD applies.

Arms on rung 1, in order: F if pre-check A demands it; then A and B at whichever LR rung 1 settled
on. C is not warm-startable and skips to rung 2. D rides along on rung 2 only.

**Rung 2, from scratch.** The best bundle from rung 1, plus C if pre-check B demands it and D as a
rider, trained with the full recipe (1 epoch, 9,378 steps, LR 1e-5 constant, global batch 16, 16
GPUs, selection on auto-label val loss; `run_train_seed.slurm`), 3 seeds. Baseline: Campaign B's nine
replicates (`seed_variance_51_135.md`): US7 macro F1 mean 0.8227, s_B 0.0094 [0.0064, 0.0180], each
replicate at its own threshold selected on sao_paulo. The released checkpoint (0.8427, +2.1 s_B) is
compared, not pooled, as that document says. Two fixes before submission: `run_train_seed.slurm`
still defaults `DATA_ROOT` to the deleted `rampnet_dataset` (point it at `rampnet_dataset_v2`), and
the commit the s1–s3 replicates trained at is not recorded, so the rung 2 control should include
one fresh seed of the unchanged recipe on the restaged data to tie the two datasets together.

**Primary readout (proposed):** paired ΔF1 at the seed-selected threshold on the seven US splits
plus manual_gold, pooled over panos, 3 arm seeds against 3 control seeds, interval from the paired
pano bootstrap and a Welch interval over seed means. An arm passes if the pooled interval excludes
zero in its favour. **No-regression readout:** the four non-GSV splits (laurens_mapillary, bayonne,
budapest_district5, juneau) pooled, with laurens_gsv shown beside laurens_mapillary as the same-town
control; an arm is rejected if that pool drops by more than the control seed SD regardless of the
primary. **Per-bucket readout:** the miss taxonomy (`miss_taxonomy.py`) on the arm, so that arm A is
judged on the merged bucket and arm B on the far-field buckets, not only on F1. This is how a
negative F1 result still says whether the arm did what it was built to do.

Operating point: 0.30 (`operating_point.md`), with 0.55 in the tables for continuity with the
committed scoreboard. Thresholds selected on sao_paulo as in Campaign B; `HELD_OUT` stays held out.

**Minimum detectable effect.** With s_B = 0.0094 and n = 3 against n = 9, a seed-mean Welch test
resolves about 0.015 F1 at 80% power; the pano-level paired bootstrap is tighter for arms that
move specific panos (A and B both do). The merged bucket alone is worth up to 0.060 recall points
pooled (124 of 2,060 GT at pooled recall 0.793, `analysis_out/miss_taxonomy.json`); the detectable
effect is a fraction of that. Arm B's forecast is well above the MDE if it is real.

## 5. Cost and calendar, klone `ckpt` ($0)

| step | GPU-h | calendar | notes |
|---|--:|---|---|
| pre-check A (control scoring, running) | about 2.3 | done today | 4 checkpoints × about 35 min on one GPU |
| pre-check B | 0 | an hour | CPU |
| rung 1, one arm, 3 seeds | about 36 | about 3 h per seed, seeds in parallel | #82 measured 11.7–14.2 GPU-h per uninterrupted run, 1.3× with preemption |
| rung 1, arms F, A, B + scoring | about 125 | 1–2 days | scoring about 1.75 GPU-h per checkpoint |
| rung 2, one bundle, 3 seeds | about 170 (280 with the paper run's 1.67× preemption) | Campaign B saw 6 replicates finish within 5 h 15 min | plus the recipe tie-in seed, about 56 |
| **Phase 1 total** | **about 400–460** | **about a week** | all `paid: false`; ledger rows per job via `slurm_usage.py` |

Two durability items are on the critical path. `rampnet_dataset_v2` is on scrubbed storage and
purges after about 21 idle days (`restage_263.md`); last touched 2026-10-21, so it is gone around
2026-11-11 unless a job reads it or the launcher re-stages. `/gscratch/makelab` is at 1019 of 1024
GB; rung 2 checkpoints (360 MB each, 3 seeds × 2) and caches need room, which means deleting from the
list given to Jon on 2026-10-21 or re-pointing `RUNDIR` at scrubbed. Also: neither Campaign B nor the
control263 jobs have ledger rows on main yet (`compute_log.jsonl`); the control rows are owed by the
#263 scoring branch, the Campaign B rows are a gap to close while `sacct` still has them.

## 6. What this document does not decide

- Whether Phase 1 opens before the #263 decision point. The body says sigma and head return
  "after the decision point". The argument for opening rung 1 now is that it is $0, needs no
  ratings, and its arms do not touch the open-imagery question the decision point is about; the
  argument against is GPU time on klone that the verifier and Phase 2 will also want. Jon's call.
- The σ value (6 proposed) and the zoom range ([0.5, 2.0] proposed, matching the frozen sweep's 2×).
- The primary pool. US7 + manual_gold is proposed because the seed SD exists for it; the #263
  comment of 2026-10-22 names "open-rig splits + manual_gold" for the control, which is the
  no-regression pool here, not the primary.
- Whether circular padding is worth a from-scratch seed before the seam GT is merged. Proposed: no;
  it rides on rung 2 and is scored when the GT is ready.

## Reproduce

Nothing to reproduce yet. When the pre-check B script lands it will be
`scripts/analysis/merged_separation_263.py` with `--check`, reading the committed
`analysis_out/miss_taxonomy.json` inputs; the rung launchers extend
`stage_two/run_finetune_control.slurm` with the arm flags and are named in the issue when submitted.
