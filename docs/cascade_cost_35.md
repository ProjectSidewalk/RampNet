# Gated cascade: what it costs (#35, the blank #126 left)

**Issues:** [#35](https://github.com/ProjectSidewalk/RampNet/issues/35) (complementarity / cascade),
[#126](https://github.com/ProjectSidewalk/RampNet/issues/126) (the cascade gate, which measured the
ceiling and left the false-positive cost blank).
**Run date:** 2026-09-26. **Script:** `scripts/analysis/cascade_cost_35.py`
(tests: `tests/test_cascade_cost_35.py`). **Outputs:** `analysis_out/cascade_cost_35/`
(one JSON per split × challenger, plus `summary.json`).
**Compute:** CPU only, every input committed; no GPU, no panoramas, no `.model_cache`, no network,
no spend. The primary pair (full grid, all 123 null shifts, both 2,000-resample bootstraps, the
calibration) takes about 10 s on `jonfhome` (Windows desktop); the whole cross-challenger,
cross-split sweep took 13 minutes there (773 s, 2026-09-26). No ledger row: `usage_log.jsonl` is for model runs and
`compute_log.jsonl` for cluster jobs, and neither applies (same convention as `complementarity.py`
and `cascade_gate.py`). Wall-clock and host per file are in `analysis_out/cascade_cost_35/runs.json`,
kept out of the per-pair JSON so that a re-run reproduces each committed file byte for byte
(`--check` regenerates every file from the `args` it records and compares bytes).

## Result, in one paragraph

**The gated cascade is VIABLE under the pre-stated rule, and what it buys is recall, not F1.** On richmond at the recommended 0.30 point, promoting RampNet's sub-threshold floor peaks (score ≥ 0.05) that sit within R/2 of a Vistas parity-arm box scoring at least that arm's median lifts recall **0.829 → 0.868 (+12 ramps, +0.036 attributable after the chance null, i.e. ~11 ramps)** for **10 extra FPs (0.9 FP per attributable ramp)**, F1 **0.8639 → 0.8720**. All 12 gained ramps are among the 19 #126 called promotable, so the cascade realises 63% of the ceiling it was bounded by. The naive union at the same 0.30 point pays **11.6** FP per recovered ramp under `complementarity.py`'s convention ((470 − 28) FP for 295 − 257 ramps) and **14.8** under `aggregate`'s (664 FP for 45 ramps); the ~8.2 quoted in #126 is the same union at the shipped 0.55 point, on raw rather than attributable ramps. But the F1 gain is **not** distinguishable from re-tuning the threshold: the best single threshold on the same split (0.33) already scores F1 0.8712, and the pano bootstrap, holding the chosen setting fixed, puts the cascade's ΔF1 at +0.008 [−0.008, +0.026]. The honest comparison is **matched recall**: reaching R 0.868 with a threshold alone costs 48 extra FPs and F1 0.821; the gate costs 10. The setting was chosen in sample from 60 on one split with no held-out split for this challenger, and the same setting does not transfer to the other richmond challengers. The rule itself is not easily fooled here: run on 123 wrong-pano copies of the same challenger (same boxes, shifted to other panos), it reads VIABLE on none. Across 137 (split, leg) pairs the rule reads 46 VIABLE / 36 PARTIAL / 55 NOT VIABLE, against about 1.1 VIABLE expected from wrong-pano challengers, with the only large gains on `laurens_mapillary`, the rig-shifted split, and none on `manual_gold`. Individual VIABLE verdicts for the dense detectors are weaker than that total suggests (OWLv2's wrong-pano copies read VIABLE 15–30% of the time on three splits). So: the cascade is a cheap recall dial with a measured price of about one FP per ramp at its best setting, not a free F1 improvement, and every number here inherits the pre-seam-fix op_cache caveat below.

## The decision rule, stated before running, and where the data landed

Primary pair: **richmond × `mask2former-vistas-curb-cut-1024x1024`** (the Vistas parity arm, the one
#126 costed), RampNet at **T_hi = 0.30** (the recommended operating point). The rule was written in the
plan before any cascade number existed, and is applied here as written:

- **VIABLE**: some setting (T_lo, r_gate, c_min) gives attributable ΔR ≥ +0.020 (about 6 of 310 ramps,
  after the null) with F1 ≥ the baseline 0.8639, and that setting's F1 exceeds threshold-only at the
  same T_lo.
- **NOT VIABLE**: no setting clears both bars, or every setting that raises recall is matched by
  threshold-only.
- **PARTIAL**: recall rises but F1 drops below baseline.

The first two sentences of NOT VIABLE and PARTIAL overlap as written (a setting that raises recall with
F1 below baseline "clears neither"). The script (`verdict_of`) resolves it this way, and the tests pin
each branch: no row with attributable ΔR ≥ 0.020 → NOT VIABLE; rows that reach it but never beat
threshold-only at their T_lo → NOT VIABLE; rows that reach it and beat threshold-only but only with F1
below baseline → PARTIAL.

**The primary pair lands on VIABLE.** 4 of the 60 grid settings clear all three bars; three of them clear the 0.020 bar by under 0.002. The four: (0.05, 0.011, 0.641) F1 0.8720, attributable ΔR +0.0357, threshold-only@T_lo 0.7099; (0.1, 0.011, 0.641) F1 0.8713, attributable ΔR +0.0206, threshold-only@T_lo 0.7810; (0.15, 0.011, 0.520) F1 0.8684, attributable ΔR +0.0205, threshold-only@T_lo 0.8214; (0.15, 0.022, 0.641) F1 0.8660, attributable ΔR +0.0221, threshold-only@T_lo 0.8214. The best by F1 overall (T_lo 0.15, R/2, median; F1 0.8723) is **not** one of them: it gains 6 ramps, attributable ΔR +0.0179, just under the 0.020 bar. That is recorded as the rule applied, not moved.

**Where the rule is weaker than it looks, stated beside the verdict rather than instead of it:**

1. **The control named in the rule is weak.** "Threshold-only at the same T_lo" at T_lo = 0.05 is F1
   0.710; almost anything beats it. The stronger no-challenger control is the best single threshold
   anywhere, and on richmond that is **T = 0.33, F1 0.8712** — the cascade's best viable F1 (0.8720)
   ties it. What the cascade buys over a re-tuned threshold is **recall at the same F1**, not F1.
   The fairest single comparison is matched recall: to reach the cascade's R 0.868 with a threshold
   alone takes T = 0.15, which pays **48** extra FPs (76 against 28) for the same 12 ramps; the gate
   pays **10**.
2. **The winning setting is chosen in sample**, from 60 settings, on the same 310 ramps it is scored on.
   Only 4 of 60 settings are viable and all use c_min = the challenger's median box score (or its lower
   quartile at T_lo 0.15) with a tight gate (r_gate = R/2, one at R). No held-out split exists for this
   challenger (the parity detections are richmond-only), so the choice cannot be validated
   out of sample here.
3. **ΔF1 is not resolved from zero, and ΔR is only once the setting is fixed.** Pano-resampled 95%
   interval (2,000 resamples, paired), *conditional on the in-sample selection* and on raw ΔR:
   ΔF1 **+0.008 [−0.008, +0.026]**, ΔR **+0.039 [+0.018, +0.063]**. Repeating the selection inside
   every resample, the rule reads VIABLE in 90.4% of resamples and picks the same setting in 48%; in
   the other 9.6% no setting survives and there is no cascade to deploy, so the selection-aware ΔR
   interval reaches 0 ([0.000, +0.063]).
4. **Calibration: the rule does not fire on wrong-pano challengers here.** The 123 shifted copies of
   the parity arm, each run through the full rule with its own null, read 0 VIABLE / 3 PARTIAL /
   120 NOT VIABLE. Their largest attributable ΔR anywhere on the grid is +0.023 (median +0.006),
   against +0.046 for the real challenger, and none reaches it. A wrong-pano challenger can clear
   the 0.020 recall bar; what stops it is the F1 bar.
5. **The fixed setting does not transfer to other challengers** on richmond (table (d)): the same
   (0.05, R/2, median) lowers F1 for 12 of the 14 other legs; the other two gain +0.004 (gemini-3.7-flash) and +0.001 (y11l_pano).

## Method

Per pano and per setting:

```
kept     = RampNet floor peaks with score >= T_hi
cands    = challenger boxes with score >= c_min (a box with no score always passes)
promoted = floor peaks with T_lo <= score < T_hi within r_gate of any cand (wrapped at the seam)
preds    = kept + promoted            each keeps its own RampNet score
```

scored by the benchmark's scorer — `score_pano` at radius 0.022, wrapped, highest score first,
`unsure` points ignored — and `aggregate` (precision over every pano, recall over the `fn_confirmed`
panos; on richmond that is all 124 panos for both, 92 of which hold at least one of the 310 ramps). Because every kept peak outscores every promoted peak and the matcher
is greedy in score order, a promoted peak can never take a ramp from a kept one, so `promoted_tp` /
`promoted_fp` (the net change against baseline) are exactly the promoted peaks' own outcomes;
`promoted_ignored` are promoted peaks that landed on a reviewer's `unsure` mark.

**Inputs.** RampNet's floor peaks come from `analysis_out/op_cache/<split>.json` (every
`peak_local_max` peak down to 0.05), not from the bundle records, which are the shipped point and
contain nothing below 0.5519 to promote. The challenger's boxes come from
`benchmark/model_detections/<leg>__<split>.json`. GT is `benchmark/<split>/verdicts.json`, or the
YOLO labels for `manual_gold`.

**Grid.** T_lo ∈ {0.05, 0.10, 0.15, 0.20, 0.25}; r_gate ∈ {0.011, 0.022, 0.044} (R/2, R, 2R in
normalized x); c_min ∈ {0, q25, q50, q75} of that challenger's box scores on that split, or {0} for a
challenger that emits no scores (the chat VLMs). 60 settings for a scored challenger, 15 for a
score-less one.

**Controls, same scorer, same panos, same run.** `baseline` = kept only; `threshold_only[T_lo]` = every
peak ≥ T_lo; `threshold_only_best` = the best single threshold on a 0.05–0.95 sweep in 0.01 steps;
`threshold_only_at_matched_recall` = the best-F1 threshold reaching a row's recall; `naive_union` =
kept + every challenger box.

**Null.** Cyclic shift over the sorted pano list: for k = 1..n−1, pano i gets the boxes of pano
(i + k) mod n — same boxes, same density and clustering, wrong pano — and the cascade is re-scored at
every setting under every shift. `attributable_dR` = real ΔR − mean shifted ΔR. This is the
construction in `complementarity.complementary_null` and `null_recall.py`. The primary pair uses all
123 shifts; the cross-split sweep uses 20 evenly spaced shifts (`--null-shifts 20`) to keep it to
minutes. `--null random` draws a seeded random subset of the same cyclic shifts instead of evenly
spaced ones (it is still a shift null, not random box positions); it was not used.

**Calibration.** The shifted challengers are also run through the whole rule as if each were real:
the challenger at shift k takes the true alignment plus the other evaluated shifts as its null, gets
its own attributable ΔR on every setting, and gets a verdict. The fraction reading VIABLE is the
rule's false-VIABLE rate on that pair, with the max-over-grid selection included. With all shifts
(richmond) this is exact, because the shifts of a shifted challenger are the original's shifts; with
20 it is the same construction over those 20 (`calibration` in each JSON).

**Bootstraps.** `bootstrap_vs_baseline` resamples panos (2,000, paired) with the setting held fixed,
so it is *conditional on the in-sample selection* and on raw, not attributable, ΔR.
`bootstrap_selection_aware` repeats the whole rule inside every resample (null mean, controls,
viability, best by F1 then attributable ΔR); a resample with no viable setting deploys no cascade and
contributes Δ = 0. Because a viable row has F1 ≥ baseline by definition, its ΔF1 interval cannot go
below zero; what it adds is how often the rule still says VIABLE and how often it picks the same
setting.

**Instrument check (passed before any cascade number was read).** Baseline on richmond at 0.30:
**257 / 28 / 53, P 0.9018 / R 0.8290 / F1 0.8639**, identical to
`analysis_out/op/corrected_at_0.3.csv`; at 0.5519 it is 238 / 9 / 72, the published row. The naive
union reproduces the published **0.549** under `complementarity.py`'s convention: each model's list
is scored on its own, TP is the oracle union of the ramps either matched, and the two FP bills are
added with no dedup (295 TP / 470 FP = 28 + 442). Under `aggregate`'s convention, one merged list
goes through `score_pano`, and the same union is **0.463** (302 / 692). The two conventions see the
same panos (all 124 are `fn_confirmed`); the gap is how the merged list scores. When a RampNet peak
and a Vistas box land on the same ramp (236 ramps are hit by both), the second hit is an FP: of the
531 hits that are TPs when each list is scored alone, 229 stop being TPs in the merged list, 222
becoming FPs and 7 landing on `unsure` marks (+222 FP). Greedy matching on the merged list also hands
7 boxes to a neighbouring ramp neither list matched alone (+7 TP, 302 against the oracle's 295). The
tables below use `aggregate`'s convention throughout, because it is the one every cascade row and
control is scored under; the 0.549 is quoted only as the check.

## (a) Richmond × Vistas parity arm, T_hi 0.30 — the primary table

Ceiling beside every row (from #126, `analysis_out/cascade_gate_op030.json`): of the 38 ramps the
parity arm recovers and RampNet misses, **19 are promotable** (a floor peak 0.05–0.30 in radius;
+0.061 R if all were gained), 4 have a peak ≥ 0.30 that the matcher gave to an adjacent ramp (#130),
15 have no peak in radius. The last two groups are unreachable by any threshold rule. Complementarity's
attributable-after-null figure is ~30.

| row | TP | FP | FN | P | R | F1 | ΔR (ramps) | attributable ΔR | promoted FP | FP / attributable ramp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline, RampNet ≥ 0.30 | 257 | 28 | 53 | 0.9018 | 0.8290 | 0.8639 | +0.0000 (+0) | — | +0 FP vs baseline | — |
| threshold-only ≥ 0.05 | 279 | 197 | 31 | 0.5861 | 0.9000 | 0.7099 | +0.0710 (+22) | — | +169 FP vs baseline | — |
| threshold-only ≥ 0.1 | 271 | 113 | 39 | 0.7057 | 0.8742 | 0.7810 | +0.0452 (+14) | — | +85 FP vs baseline | — |
| threshold-only ≥ 0.15 | 269 | 76 | 41 | 0.7797 | 0.8677 | 0.8214 | +0.0387 (+12) | — | +48 FP vs baseline | — |
| threshold-only ≥ 0.2 | 266 | 51 | 44 | 0.8391 | 0.8581 | 0.8485 | +0.0290 (+9) | — | +23 FP vs baseline | — |
| threshold-only ≥ 0.25 | 260 | 38 | 50 | 0.8725 | 0.8387 | 0.8553 | +0.0097 (+3) | — | +10 FP vs baseline | — |
| threshold-only best (≥ 0.33, 0.05–0.95 sweep) | 257 | 23 | 53 | 0.9179 | 0.8290 | 0.8712 | +0.0000 (+0) | — | -5 FP vs baseline | — |
| naive union (aggregate convention) | 302 | 692 | 8 | 0.3038 | 0.9742 | 0.4632 | +0.1452 (+45) | — | +664 FP vs baseline | — |
| **cascade, best viable** (T_lo 0.05, r_gate 0.011, c_min 0.641 = q50) | 269 | 38 | 41 | 0.8762 | 0.8677 | 0.8720 | +0.0387 (+12) | +0.0357 | 10 | 0.90 |
| cascade, best by F1 (T_lo 0.15, r_gate 0.011, c_min 0.641 = q50; not viable: attributable < 0.020) | 263 | 30 | 47 | 0.8976 | 0.8484 | 0.8723 | +0.0194 (+6) | +0.0179 | 2 | 0.36 |
| cascade, best R at P ≥ baseline (T_lo 0.25, r_gate 0.011, c_min 0.520 = q25) | 259 | 28 | 51 | 0.9024 | 0.8355 | 0.8677 | +0.0065 (+2) | +0.0058 | 0 | 0.00 |

The best viable row gains **12 ramps, all 12 of them among #126's 19 promotable sites** — 63% of the reachable ceiling — for 10 promoted FPs (7 further promoted peaks land on reviewer `unsure` marks and score as neither). A threshold alone reaching the same recall (≥ 0.15) pays 48 extra FPs and scores F1 0.8214. Pano-resampled 95% interval against baseline, conditional on the in-sample selection (2,000 paired resamples, setting held fixed): ΔF1 [-0.008, +0.026], raw ΔR [+0.018, +0.063]. With the selection repeated in every resample, the rule still reads VIABLE in 90.4% of resamples (PARTIAL in 187 and NOT VIABLE in 5 of the 2,000) and picks this same setting in 48%; ΔF1 [0.000, +0.025], ΔR [0.000, +0.063] with Δ = 0 where nothing is viable, or ΔR [+0.022, +0.063] over the viable resamples only. c_min quartiles of the parity arm's richmond box scores: 0.520, 0.641, 0.753.

**The FP columns understate the cascade, on every split except `manual_gold`.** Richmond's GT is anchored on RampNet's shipped detections plus the reviewer's missed marks, so a promoted peak on a real ramp nobody marked scores as an FP. `manual_gold` is the one split whose GT was labelled independently of any model.

## (b) The full grid at r_gate = R (0.022)

| T_lo | c_min | TP | FP | FN | P | R | F1 | promoted | promoted TP / FP / ignored | ΔR | attributable ΔR | thr-only F1 @T_lo | viable |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| 0.05 | 0.000 | 275 | 97 | 35 | 0.7392 | 0.8871 | 0.8065 | 104 | 18 / 69 / 17 | +0.0581 | +0.0458 | 0.7099 |  |
| 0.05 | 0.520 | 273 | 81 | 37 | 0.7712 | 0.8806 | 0.8223 | 85 | 16 / 53 / 16 | +0.0516 | +0.0412 | 0.7099 |  |
| 0.05 | 0.641 | 271 | 56 | 39 | 0.8287 | 0.8742 | 0.8509 | 50 | 14 / 28 / 8 | +0.0452 | +0.0380 | 0.7099 |  |
| 0.05 | 0.753 | 260 | 36 | 50 | 0.8784 | 0.8387 | 0.8581 | 13 | 3 / 8 / 2 | +0.0097 | +0.0058 | 0.7099 |  |
| 0.1 | 0.000 | 267 | 63 | 43 | 0.8091 | 0.8613 | 0.8344 | 58 | 10 / 35 / 13 | +0.0323 | +0.0238 | 0.7810 |  |
| 0.1 | 0.520 | 267 | 55 | 43 | 0.8292 | 0.8613 | 0.8449 | 49 | 10 / 27 / 12 | +0.0323 | +0.0250 | 0.7810 |  |
| 0.1 | 0.641 | 266 | 43 | 44 | 0.8608 | 0.8581 | 0.8595 | 30 | 9 / 15 / 6 | +0.0290 | +0.0242 | 0.7810 |  |
| 0.1 | 0.753 | 259 | 32 | 51 | 0.8900 | 0.8355 | 0.8619 | 7 | 2 / 4 / 1 | +0.0065 | +0.0039 | 0.7810 |  |
| 0.15 | 0.000 | 266 | 48 | 44 | 0.8471 | 0.8581 | 0.8526 | 38 | 9 / 20 / 9 | +0.0290 | +0.0226 | 0.8214 |  |
| 0.15 | 0.520 | 266 | 43 | 44 | 0.8608 | 0.8581 | 0.8595 | 32 | 9 / 15 / 8 | +0.0290 | +0.0236 | 0.8214 |  |
| 0.15 | 0.641 | 265 | 37 | 45 | 0.8775 | 0.8548 | 0.8660 | 21 | 8 / 9 / 4 | +0.0258 | +0.0221 | 0.8214 | yes |
| 0.15 | 0.753 | 259 | 30 | 51 | 0.8962 | 0.8355 | 0.8648 | 5 | 2 / 2 / 1 | +0.0065 | +0.0046 | 0.8214 |  |
| 0.2 | 0.000 | 264 | 39 | 46 | 0.8713 | 0.8516 | 0.8613 | 25 | 7 / 11 / 7 | +0.0226 | +0.0183 | 0.8485 |  |
| 0.2 | 0.520 | 264 | 37 | 46 | 0.8771 | 0.8516 | 0.8642 | 23 | 7 / 9 / 7 | +0.0226 | +0.0191 | 0.8485 |  |
| 0.2 | 0.641 | 263 | 35 | 47 | 0.8825 | 0.8484 | 0.8651 | 17 | 6 / 7 / 4 | +0.0194 | +0.0170 | 0.8485 |  |
| 0.2 | 0.753 | 259 | 29 | 51 | 0.8993 | 0.8355 | 0.8662 | 4 | 2 / 1 / 1 | +0.0065 | +0.0052 | 0.8485 |  |
| 0.25 | 0.000 | 260 | 30 | 50 | 0.8966 | 0.8387 | 0.8667 | 10 | 3 / 2 / 5 | +0.0097 | +0.0077 | 0.8553 |  |
| 0.25 | 0.520 | 260 | 29 | 50 | 0.8997 | 0.8387 | 0.8681 | 9 | 3 / 1 / 5 | +0.0097 | +0.0080 | 0.8553 |  |
| 0.25 | 0.641 | 260 | 29 | 50 | 0.8997 | 0.8387 | 0.8681 | 7 | 3 / 1 / 3 | +0.0097 | +0.0086 | 0.8553 |  |
| 0.25 | 0.753 | 258 | 28 | 52 | 0.9021 | 0.8323 | 0.8658 | 2 | 1 / 0 / 1 | +0.0032 | +0.0025 | 0.8553 |  |

1 of these 20 rows is viable; the other three viable rows are at r_gate = R/2 (0.011). Widening the gate to 2R (0.044) never produces a viable row: the null's share of the gain grows with the gate radius (a wider gate catches more peaks by chance), which is the expected shape. The complete 60-row grid is in the JSON.

## (c) The null, for the viable rows and the best-F1 row

| setting | real ΔR | null ΔR mean | null ΔR max | attributable ΔR | real ΔFP | null ΔFP mean |
|---|---:|---:|---:|---:|---:|---:|
| T_lo 0.05, r_gate 0.011, c_min 0.641 (viable) | +0.0387 | +0.0030 | +0.0129 | +0.0357 | +10 | +3.47 |
| T_lo 0.1, r_gate 0.011, c_min 0.641 (viable) | +0.0226 | +0.0020 | +0.0097 | +0.0206 | +4 | +1.79 |
| T_lo 0.15, r_gate 0.011, c_min 0.520 (viable) | +0.0226 | +0.0021 | +0.0129 | +0.0205 | +6 | +1.48 |
| T_lo 0.15, r_gate 0.022, c_min 0.641 (viable) | +0.0258 | +0.0037 | +0.0161 | +0.0221 | +9 | +3.38 |
| T_lo 0.15, r_gate 0.011, c_min 0.641 (best by F1) | +0.0194 | +0.0014 | +0.0097 | +0.0179 | +2 | +1.01 |

The null is small at these settings because the gate is tight: a wrong-pano box rarely lands within 11 px of a sub-threshold peak. For the headline row the worst of the 123 shifts gains 4 ramps against the real 12.

## (d) Every published challenger on richmond, T_hi 0.30, all 123 shifts

| challenger | verdict | best viable (T_lo, r_gate, c_min) | F1 | attributable ΔR | FP / attr ramp | fixed setting F1 | fixed ΔF1 [95% CI] | fixed attributable ΔR | union F1 |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| gemini-3.6-flash | PARTIAL | — | — | — | — | 0.8637 | -0.0002 [-0.011, +0.012] | +0.0179 | 0.5971 |
| gemini-3.1-pro-preview | VIABLE | 0.2, 0.044, 0 | 0.8660 | +0.0201 | 1.45 | 0.8590 | -0.0048 [-0.017, +0.007] | +0.0139 | 0.6030 |
| Qwen/Qwen3-VL-8B-Instruct | NOT VIABLE | — | — | — | — | 0.8529 | -0.0110 [-0.022, -0.002] | +0.0013 | 0.5273 |
| Qwen/Qwen3-VL-32B-Instruct | NOT VIABLE | — | — | — | — | 0.8629 | -0.0010 [-0.006, +0.005] | +0.0029 | 0.7346 |
| allenai/Molmo2-8B | PARTIAL | — | — | — | — | 0.8567 | -0.0072 [-0.018, +0.003] | +0.0078 | 0.5490 |
| google/owlv2-large-patch14-ensemble | PARTIAL | — | — | — | — | 0.7845 | -0.0793 [-0.110, -0.051] | +0.0092 | 0.0640 |
| IDEA-Research/grounding-dino-base | NOT VIABLE | — | — | — | — | 0.8175 | -0.0464 [-0.067, -0.027] | +0.0078 | 0.0586 |
| mask2former-vistas-curb-cut | VIABLE | 0.2, 0.022, 0 | 0.8717 | +0.0230 | 0.70 | 0.8502 | -0.0137 [-0.029, +0.001] | +0.0111 | 0.4960 |
| mask2former-vistas-curb-cut+curb | PARTIAL | — | — | — | — | 0.8341 | -0.0297 [-0.044, -0.017] | +0.0037 | 0.2626 |
| mask2former-vistas-curb-cut-1024x1024 | VIABLE | 0.05, 0.011, 0.641 | 0.8720 | +0.0357 | 0.90 | 0.8720 | +0.0081 [-0.008, +0.026] | +0.0357 | 0.4632 |
| gemini-3.7-flash | NOT VIABLE | — | — | — | — | 0.8677 | +0.0038 [+0.000, +0.010] | +0.0055 | 0.6342 |
| y11l_pano | NOT VIABLE | — | — | — | — | 0.8652 | +0.0014 [-0.007, +0.010] | +0.0082 | 0.5252 |
| y11x_pano_h200 | NOT VIABLE | — | — | — | — | 0.8638 | -0.0001 [-0.009, +0.009] | +0.0082 | 0.5596 |
| y26_pano | PARTIAL | — | — | — | — | 0.8633 | -0.0006 [-0.010, +0.008] | +0.0142 | 0.3703 |
| claude-opus-5-effort-low | PARTIAL | — | — | — | — | 0.8506 | -0.0132 [-0.028, +0.001] | +0.0139 | 0.5496 |

Baseline F1 0.8639; best threshold-only F1 0.8712 at 0.33. Verdicts: 3 VIABLE, 6 PARTIAL, 6 NOT VIABLE. The *fixed setting* is (T_lo 0.05, r_gate 0.011, c_min = that challenger's median box score, or all boxes for a score-less one); it was chosen after the primary run, as the primary pair's best viable row, and is read here on legs it was not chosen on. **Only the parity arm itself gains F1 with it**; gemini-3.7-flash (+0.004) and y11l_pano (+0.001) are the only other non-negative ones, and the dense detectors (OWLv2 −0.079, Grounding DINO −0.046) lose heavily, as a detector that boxes everything must: at 74 boxes per pano the gate is open almost everywhere and the cascade degrades toward threshold-only at T_lo. Both Vistas arms at the published 384 input are weaker than the parity arm: the curb-cut 384 arm is VIABLE at a different setting (T_lo 0.2, R, all boxes), the +curb arm is not.

## (e) Across splits (T_hi 0.30, 20 null shifts)

| split | legs | VIABLE / PARTIAL / NOT | baseline F1 @0.30 | best thr-only F1 (t) | best viable leg | its F1 | attributable ΔR | FP / attr ramp | viable legs above best thr-only F1 |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| annapolis | 17 | 5 / 6 / 6 | 0.8530 | 0.8556 (0.31) | gemini-3.1-pro-preview | 0.8606 | +0.0247 | 0.97 | 4 of 5 |
| bend | 12 | 3 / 2 / 7 | 0.8706 | 0.8727 (0.5) | y26_pano | 0.8766 | +0.0220 | 0.83 | 3 of 3 |
| budapest_district5 | 12 | 1 / 8 / 3 | 0.6736 | 0.6839 (0.37) | y11l_pano | 0.6768 | +0.0233 | 1.86 | 0 of 1 |
| clovis | 12 | 5 / 2 / 5 | 0.8355 | 0.8418 (0.35) | google/owlv2-large-patch14-ensemble | 0.8434 | +0.0238 | 1.29 | 3 of 5 |
| gainesville | 12 | 11 / 0 / 1 | 0.8124 | 0.8224 (0.38) | y11x_pano_h200 | 0.8287 | +0.0498 | 0.81 | 4 of 11 |
| laurens_mapillary | 12 | 10 / 1 / 1 | 0.6600 | 0.7082 (0.16) | y26_pano | 0.8018 | +0.1821 | 0.13 | 10 of 10 |
| manual_gold | 9 | 0 / 1 / 8 | 0.9018 | 0.9047 (0.36) | — | — | — | — | 0 of 0 |
| morgantown | 12 | 4 / 6 / 2 | 0.8448 | 0.8515 (0.32) | y26_pano | 0.8549 | +0.0202 | 0.37 | 1 of 4 |
| paterson | 12 | 1 / 0 / 11 | 0.8184 | 0.8212 (0.26) | y11x_pano_h200 | 0.8229 | +0.0214 | 1.42 | 1 of 1 |
| richmond | 15 | 3 / 6 / 6 | 0.8639 | 0.8712 (0.33) | mask2former-vistas-curb-cut-1024x1024 | 0.8720 | +0.0357 | 0.90 | 2 of 3 |
| sao_paulo | 12 | 3 / 4 / 5 | 0.8000 | 0.8000 (0.3) | y11l_pano | 0.8139 | +0.0347 | 0.51 | 3 of 3 |

All 137 (split, leg) pairs with published detections and an op_cache (richmond with all 123 shifts, the rest with 20): **46 VIABLE, 36 PARTIAL, 55 NOT VIABLE** under the pre-stated rule applied per pair. Against the stronger control (the best single threshold on that split), 31 of the 46 viable pairs still come out ahead. Per-pair rows, including the fixed setting, both bootstrap intervals and the calibration rate, are in `summary.json` and printed by `--summary`.

**The per-pair verdict is not family-wise calibrated, and 46 of 137 has to be read with that.** Each pair's verdict is the max over 15–60 settings, judged against a 20-shift null on every split but richmond. The calibration (each pair's shifted challengers run through the whole rule) puts the VIABLE count expected with no challenger aligned to its panos at **1.1** summed over the 137 pairs (`summary.json` → `calibration`), so the total is far above chance. The rate is not flat across legs, though: 13 pairs have a nonzero false-VIABLE rate, and the highest are the dense OWLv2 arm on `laurens_mapillary` (0.30), `clovis` (0.15) and `gainesville` (0.15), each of which is itself VIABLE. Those three verdicts, and `clovis` × y26_pano (0.05), are the ones a chance alignment could produce; with 20 wrong-pano challengers per pair these rates are coarse (one in 20 is 0.05). Under the selection-aware bootstrap every one of the 46 viable pairs stays VIABLE in at least half its resamples, and 18 in at least 90%.

Three readings, each with its caveat beside it:

- **`manual_gold` (the one independently labelled split): 0 of 9 viable.** RampNet is in-distribution there (baseline 0.902) and nothing any challenger adds survives the rule. The Vistas arms have no files on `manual_gold`.
- **`laurens_mapillary` is the outlier: 10 of 12 viable, and the three YOLO pano arms lift F1 from 0.660 to 0.78–0.80 (attributable ΔR +0.15 to +0.18) against a best single threshold of 0.708; the viable chat VLMs and OWLv2 reach 0.71–0.75.** All ten best viable settings sit on the grid's T_lo edge (0.05, the op_cache floor), and the seven VLM and OWLv2 ones also on its widest gate (r_gate 0.044), so the optimum there may lie outside the grid and these F1s are lower bounds on what a wider search would find. This is the split where RampNet's deficit was traced to the capture rig rather than the town, and RampNet was measured as 3–5× more rig-sensitive than the YOLO arms ([`rampnet1_findings.md`](rampnet1_findings.md), [#151](https://github.com/ProjectSidewalk/RampNet/issues/151)). The cascade here is recovering RampNet's rig-shifted sub-threshold response with a less rig-sensitive detector; it is not evidence about GSV splits. The same op_cache seam caveat applies.
- **Elsewhere, where a pair is viable, the F1 margin over the best single threshold is small** (annapolis +0.005, bend +0.004, clovis +0.002, richmond +0.001, paterson +0.002), each chosen in sample from 15–60 settings with 20 null shifts. On `budapest_district5` the one viable pair does not beat the best threshold. `sao_paulo` is the exception to the small-margin pattern (0.814 against a best threshold of 0.800, y11l_pano).

## At the shipped point (T_hi 0.5519), richmond × parity arm

#35's original text was written at the shipped threshold, so the same grid was run there with T_lo
extended to {0.05, …, 0.25, 0.30, 0.40}. Baseline 238 / 9 / 72, F1 0.8546 (the published row). The rule reads **VIABLE** with 47 of 84 settings viable — unsurprising, since at 0.5519 the threshold itself is mis-set and almost any recovery of sub-threshold peaks helps. Best: (0.2, 0.011, 0.520) → 256 / 17 / 54, P 0.9377 / R 0.8258 / F1 **0.8782**, attributable ΔR +0.0548, 8 promoted FPs (0.47 per attributable ramp), bootstrap ΔF1 [+0.006, +0.044] (setting held fixed). Its recall (0.826) is *below* the 0.30 baseline's, so this is mostly the threshold correction of #54/#55 done a different way, not a gain over the recommended point. It does edge the best single threshold (F1 0.8712 at 0.33) by 0.007 while keeping precision at 0.938 against 0.918, which is the one place in this read where the cascade is ahead on both axes of a re-tuned threshold. At this point the rule is also easier to satisfy by chance: 4 of the 123 wrong-pano copies of the parity arm read VIABLE (against 0 of 123 at 0.30), because promoting almost any sub-threshold peak corrects a mis-set threshold. It is in `richmond__mask2former-vistas-curb-cut-1024x1024__thi0.5519.json`.

## Caveats (each applies to every number above)

- **The op_caches predate the seam fix.** They were written at `c7098be` (2026-07-28), before
  `f4c71c8` (2026-08-18), and carry the ~3.5° blind strip beside the seam described in
  [`seam.md`](seam.md). The cascade can only promote peaks the cache lists, so any seam-side gain is
  under-stated, not inflated. On richmond one `challenger_only` site (`723487737079243`, x = 0.0069)
  is known to have a heatmap peak the cache dropped. Regenerating the op_caches needs a GPU and the
  native-resolution panoramas; it was not done here.
- **One RampNet checkpoint, no seeds.** Seed variance is the binding limit on RampNet-vs-YOLO
  comparisons elsewhere in this repo; nothing here measures it for the cascade.
- **The Vistas parity arm is itself one run on one split** (richmond, 124 panos). There is no second
  split for it, so the primary result has no out-of-sample check.
- **Promoted peaks keep RampNet's own scores**, so the cascade's output has a score and an AP could be
  computed, but that AP would rank promoted peaks by a score the gate has already overridden. No AP is
  reported.
- **`laurens_gsv` has no op_cache** and is not in the sweep (listed under `gaps` in `summary.json`).
- **The ceiling artifact exists for richmond × parity arm at 0.30 only**; other pairs carry
  `ceiling: null`.
- **The 4 hand-off and 15 peakless ramps** in the richmond ceiling are out of reach of any threshold
  rule and are not counted against the cascade.

## What would change the answer

- **A regenerated op_cache** (post-`f4c71c8`) for every split: removes the seam strip, and is needed
  before any cross-split cascade number is quoted as final.
- **A second split for the parity arm**, so the chosen setting can be scored on data it was not
  chosen on.
- **A different second stage.** This measures the cheapest cascade: the challenger only moves
  RampNet's threshold. The last comment on #35 argues for an arbiter that re-examines a crop around
  each candidate; that is a different design and this measurement does not test it.

## Reproduction

From a clean clone, CPU only:

```bash
# primary pair, T_hi 0.30, full grid, all shifts, both bootstraps, calibration (~10 s)
python scripts/analysis/cascade_cost_35.py --split richmond --challenger mask2former-vistas-curb-cut-1024x1024

# primary pair at the shipped point
python scripts/analysis/cascade_cost_35.py --split richmond --challenger mask2former-vistas-curb-cut-1024x1024 --t-hi 0.5519 --t-lo 0.05 0.10 0.15 0.20 0.25 0.30 0.40

# every other published leg on richmond, all shifts (~3 min)
python scripts/analysis/cascade_cost_35.py --all-published --splits richmond

# every other split with an op_cache, 20 shifts (~10 min)
python scripts/analysis/cascade_cost_35.py --all-published --null-shifts 20 --splits annapolis bend budapest_district5 clovis gainesville laurens_gsv laurens_mapillary manual_gold morgantown paterson sao_paulo

# summary.json and the tables in (d) and (e)
python scripts/analysis/cascade_cost_35.py --summary

# prove every committed per-pair file reproduces byte for byte (writes nothing; ~13 min)
python scripts/analysis/cascade_cost_35.py --check

# tests (~12 s; re-derive the primary pair byte for byte and summary.json)
python -m pytest -q tests/test_cascade_cost_35.py
```

`RAMPNET_ANALYSIS_OUT` or `--out` redirects the outputs. Each run also updates `runs.json` (wall-clock,
host, Python version per file); that file is the only one a re-run changes.
