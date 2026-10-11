# Gated cascade: what it costs (#35, the blank #126 left)

**Issues:** [#35](https://github.com/ProjectSidewalk/RampNet/issues/35) (complementarity / cascade),
[#126](https://github.com/ProjectSidewalk/RampNet/issues/126) (the cascade gate, which measured the
ceiling and left the false-positive cost blank).
**Run date:** 2026-09-26. **Script:** `scripts/analysis/cascade_cost_35.py`
(tests: `tests/test_cascade_cost_35.py`). **Outputs:** `analysis_out/cascade_cost_35/`
(one JSON per split × challenger, plus `summary.json`, plus `runs.json` with the wall-clock and host of each file).
**Compute:** CPU only, every input committed; no GPU, no panoramas, no `.model_cache`, no network,
no spend. The primary pair (full grid, all 123 null shifts, both 2,000-resample bootstraps, the
calibration) takes about 10 s on `jonfhome` (Windows desktop); the whole cross-challenger,
cross-split sweep took 13 minutes there (773 s, 2026-09-26). No ledger row: `usage_log.jsonl` is for model runs and
`compute_log.jsonl` for cluster jobs, and neither applies (same convention as `complementarity.py`
and `cascade_gate.py`). Wall-clock and host per file are in `analysis_out/cascade_cost_35/runs.json`,
kept out of the per-pair JSON so that a re-run reproduces each committed file byte for byte
(`--check` regenerates every file from the `args` it records and compares bytes).

## Result, in one paragraph

**Update 2026-09-27 (§Transfer below): the richmond setting does not transfer.** Read unchanged on
three GSV splits under a rule stated before scoring, it passes on gainesville only (bend and paterson
fail; annapolis, a second Mapillary split, passes but is not counted), and on post-seam-fix peaks
(a sensitivity read added after scoring) it passes on none. gainesville's pass and its post-seam
failure each turn on **one false positive** in criterion 3, which carries no interval, so the per-split
outcome there is fragile; the overall verdict is not (at most 1 of 3 in every read). On the GSV splits
a lower threshold buys the same recall for 1.2–1.4 FP per ramp, against 4.0 on richmond: on bend and
paterson the gate costs more per ramp than that, on gainesville slightly less (1.11 against 1.20).
Running Mask2Former at 1024 costs about 1.8× RampNet's own per-panorama wall-clock on GSV imagery
(same L40S node). The paragraph below is the richmond
result as first written.

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

## (e) Across splits (T_hi 0.30, 20 null shifts; richmond 123)

| split | legs | VIABLE / PARTIAL / NOT | baseline F1 @0.30 | best thr-only F1 (t) | best viable leg | its F1 | attributable ΔR | FP / attr ramp | viable legs above best thr-only F1 |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| annapolis ‡ | 18 | 6 / 6 / 6 | 0.8530 | 0.8556 (0.31) | mask2former-vistas-curb-cut-1024x1024 | 0.8681 | +0.0380 | 0.54 | 5 of 6 |
| bend ‡ | 13 | 4 / 2 / 7 | 0.8706 | 0.8727 (0.5) | y26_pano | 0.8766 | +0.0220 | 0.83 | 4 of 4 |
| budapest_district5 | 12 | 1 / 8 / 3 | 0.6736 | 0.6839 (0.37) | y11l_pano | 0.6768 | +0.0233 | 1.86 | 0 of 1 |
| clovis | 12 | 5 / 2 / 5 | 0.8355 | 0.8418 (0.35) | google/owlv2-large-patch14-ensemble † | 0.8434 | +0.0238 | 1.29 | 3 of 5 |
| gainesville ‡ | 13 | 12 / 0 / 1 | 0.8124 | 0.8224 (0.38) | y11x_pano_h200 | 0.8287 | +0.0498 | 0.81 | 5 of 12 |
| laurens_mapillary | 12 | 10 / 1 / 1 | 0.6600 | 0.7082 (0.16) | y26_pano | 0.8018 | +0.1821 | 0.13 | 10 of 10 |
| manual_gold | 9 | 0 / 1 / 8 | 0.9018 | 0.9047 (0.36) | — | — | — | — | 0 of 0 |
| morgantown | 12 | 4 / 6 / 2 | 0.8448 | 0.8515 (0.32) | y26_pano | 0.8549 | +0.0202 | 0.37 | 1 of 4 |
| paterson ‡ | 13 | 2 / 0 / 11 | 0.8184 | 0.8212 (0.26) | mask2former-vistas-curb-cut-1024x1024 | 0.8280 | +0.0229 | 1.00 | 2 of 2 |
| richmond | 15 | 3 / 6 / 6 | 0.8639 | 0.8712 (0.33) | mask2former-vistas-curb-cut-1024x1024 | 0.8720 | +0.0357 | 0.90 | 2 of 3 |
| sao_paulo | 12 | 3 / 4 / 5 | 0.8000 | 0.8000 (0.3) | y11l_pano | 0.8139 | +0.0347 | 0.51 | 3 of 3 |

‡ **Added 2026-09-27 by the transfer run (§Transfer below).** The parity arm now has detections on annapolis, bend, gainesville and paterson, and its four in-sample grids were run with every shift (not 20), so `summary.json` now covers **141 pairs: 50 VIABLE / 36 PARTIAL / 55 NOT VIABLE**, with the expected false-VIABLE count 1.14 and 14 pairs with a nonzero rate (the new one is gainesville × parity arm, 2 of 124 wrong-pano copies, 0.016). The four rows above were updated for them. On annapolis and paterson the parity arm's in-sample best row is the split's best viable pair and beats the best single threshold by +0.0125 and +0.0068, more than the +0.006 ceiling stated for "elsewhere" below; on bend and gainesville its margins are +0.0035 and +0.0057. Those grid rows are chosen in sample, like every row in this table; the out-of-sample read is in §Transfer. The text from here to the end of (e) describes the 137 pairs of 2026-09-26 and is left as written.

All 137 (split, leg) pairs with published detections and an op_cache (richmond with all 123 shifts, the rest with 20): **46 VIABLE, 36 PARTIAL, 55 NOT VIABLE** under the pre-stated rule applied per pair. Against the stronger control (the best single threshold on that split), 31 of the 46 viable pairs still come out ahead. Per-pair rows, including the fixed setting, both bootstrap intervals and the calibration rate, are in `summary.json` and printed by `--summary`.

**The per-pair verdict is not family-wise calibrated, and 46 of 137 has to be read with that.** Each pair's verdict is the max over 15–60 settings, judged against a 20-shift null on every split but richmond. The calibration (each pair's shifted challengers run through the whole rule) puts the VIABLE count expected with no challenger aligned to its panos at **1.1** summed over the 137 pairs (`summary.json` → `calibration`), so the total is far above chance. The rate is not flat across legs, though: 13 pairs have a nonzero false-VIABLE rate, and the highest are the dense OWLv2 arm on `laurens_mapillary` (0.30), `clovis` (0.15) and `gainesville` (0.15), each of which is itself VIABLE. Those three verdicts, and `clovis` × y26_pano (0.05), are the ones a chance alignment could produce; with 20 wrong-pano challengers per pair these rates are coarse (one in 20 is 0.05), and 0 of 20 still allows a true rate up to about 0.14 (one-sided 95%). They are also biased low: with 20 shifts a wrong challenger's null pool includes the true alignment at weight 1/20 rather than 1/123, which for a strong challenger lowers its attributable ΔR by roughly 0.002, the same size as the margins by which several verdicts clear 0.020. So the 1.1 is a point estimate, not a bound. † in table (e): OWLv2's wrong-pano copies read VIABLE on clovis at a rate of 0.15. Under the selection-aware bootstrap every one of the 46 viable pairs stays VIABLE in at least half its resamples, and 18 in at least 90%.

Three readings, each with its caveat beside it:

- **`manual_gold` (the one independently labelled split): 0 of 9 viable.** RampNet is in-distribution there (baseline 0.902) and nothing any challenger adds survives the rule. The Vistas arms have no files on `manual_gold`.
- **`laurens_mapillary` is the outlier: 10 of 12 viable, and the three YOLO pano arms lift F1 from 0.660 to 0.78–0.80 (attributable ΔR +0.15 to +0.18) against a best single threshold of 0.708; the viable chat VLMs and OWLv2 reach 0.71–0.75.** All ten best viable settings sit on the grid's T_lo edge (0.05, the op_cache floor), and the seven VLM and OWLv2 ones also on its widest gate (r_gate 0.044), so the optimum there may lie outside the grid and these F1s are lower bounds on what a wider search would find. This is the split where RampNet's deficit was traced to the capture rig rather than the town, and RampNet was measured as 3–5× more rig-sensitive than the YOLO arms ([`rampnet1_findings.md`](rampnet1_findings.md), [#151](https://github.com/ProjectSidewalk/RampNet/issues/151)). OWLv2's verdict here is the weakest of the ten: its wrong-pano copies read VIABLE at a rate of 0.30 on this split. The cascade here is recovering RampNet's rig-shifted sub-threshold response with a less rig-sensitive detector; it is not evidence about GSV splits. The same op_cache seam caveat applies.
- **Elsewhere, where a pair is viable, the F1 margin over the best single threshold is at most +0.006** (largest per split: gainesville +0.006, annapolis +0.005, bend +0.004, morgantown +0.003, clovis +0.002, paterson +0.002, richmond +0.001), each chosen in sample from 15–60 settings with 20 null shifts (123 on richmond). On `budapest_district5` the one viable pair does not beat the best threshold. `sao_paulo` is the exception to the small-margin pattern, on all three YOLO pano legs against a best threshold of 0.800: y11l_pano 0.814 (+0.014), y11x_pano_h200 0.813 (+0.013), y26_pano 0.808 (+0.008).

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
  split for it, so the primary result has no out-of-sample check. *(2026-09-27: it now has four more
  splits, and the out-of-sample check is §Transfer: the setting does not transfer.)*
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
  chosen on. *(Done 2026-09-27, §Transfer: it does not transfer.)*
- **A different second stage.** This measures the cheapest cascade: the challenger only moves
  RampNet's threshold. The last comment on #35 argues for an arbiter that re-examines a crop around
  each candidate; that is a different design and this measurement does not test it.

## Transfer: the richmond setting on four more splits (2026-09-27)

**Plan:** [#35 comment of 2026-09-27](https://github.com/ProjectSidewalk/RampNet/issues/35#issuecomment-5858166216)
(plan by Fable 5.1; implementation by Opus 5.5). **Script:** `scripts/analysis/cascade_transfer_35.py`
(tests: `tests/test_cascade_transfer_35.py`). **Output:** `analysis_out/cascade_cost_35/transfer/transfer.json`.
Everything above was measured on richmond, the only split the parity arm had been run on, at a setting
chosen there in sample. This section runs the parity arm on four more splits and reads that setting
on them unchanged.

### The rule, stated before any transfer number existed

The fixed setting is the richmond best viable row: **T_hi 0.30, T_lo 0.05, r_gate 0.011 (R/2), c_min =
the parity arm's median box score on the split being scored** (a rank, as in table (d)). On each split
the cascade *transfers* when all three hold:

1. `verdict_of` applied to the fixed row alone reads **VIABLE** (attributable ΔR ≥ 0.020 after the
   wrong-pano null with every cyclic shift, F1 ≥ the 0.30 baseline, F1 above threshold-only at 0.05);
2. the pano-bootstrap 95% interval of the **attributable ΔR** lies above 0 (2,000 resamples, setting
   held fixed; the null mean is recomputed inside each resample);
3. it pays **fewer FPs per attributable ramp than the matched-recall threshold** pays per ramp (the
   best-F1 single threshold reaching the cascade's recall; its price is extra FP over extra recall
   ramps against the baseline). If no single threshold reaches that recall, criterion 3 holds.
   *As stated, the two prices have different denominators: the cascade's FPs are divided by
   attributable (null-subtracted) ramps, the threshold's by raw ramps, since a threshold has no
   challenger and so no null. That is conservative against the cascade. Noted after review; the
   raw/raw comparison is recorded in `transfer.json` (`criterion_3_raw_vs_raw`) and flips no
   primary verdict: bend 1.50 against 1.43, paterson 4.00 against 1.25, gainesville 1.00 against
   1.20, annapolis 0.89 against 2.00. It does flip one sensitivity cell (below).*

**The cascade transfers if it transfers on at least two of the three GSV splits** (bend, paterson,
gainesville). annapolis (a second Mapillary rig) is scored the same way and reported, but not counted.
The per-split 60-setting grid (`cascade_cost_35.py`, all shifts) is reported second, as an in-sample
read. This rule and the script that applies it were committed (313e1e2) before any of the four splits
had parity-arm detections.

### Result: the setting does not transfer (1 of 3 GSV splits)

**Under the rule above, the cascade does not transfer.** The fixed richmond setting passes all three
criteria on gainesville only. On bend the attributable gain is real (interval above 0) but falls
under the 0.020 bar (+0.0167) and costs more per ramp than lowering the threshold (1.64 against 1.43
FP per ramp). On paterson it fails all three. It does pass on annapolis, the second Mapillary split,
which the rule reports but does not count.

A descriptive reading of the matched-recall column (each price rests on 4 to 12 ramps and has no
interval): on richmond the same recall bought with a lower threshold cost 4.0 FP per ramp, so a gate
that paid 0.9 was a large saving. On the three GSV splits a lower threshold costs only 1.2–1.4 FP per
ramp. On bend and paterson the gate costs more than that (1.64, 4.90); on gainesville it costs slightly
less (1.11 against 1.20), and that margin is **one false positive**: an 11th promoted FP would make it
11 / 9.03 = 1.22 and fail. bend's criterion-3 miss is two FPs (9 against at most 7). On every GSV
split the cascade's F1 at the fixed setting is below the best single threshold on that split.

**(t1) The fixed setting on each split** (T_hi 0.30, T_lo 0.05, r_gate 0.011, c_min = the parity
arm's median box score on that split; every cyclic shift as the null; 2,000 pano resamples).

| split | c_min (median) | baseline TP/FP/FN, F1 | cascade TP/FP/FN | P | R | F1 | ΔR (ramps) | null ΔR mean / max | attributable ΔR [95% CI] | promoted FP | FP / attributable ramp | matched-recall threshold: t, +FP / +ramps = FP per ramp, F1 | best single threshold (t, F1) | verdict (fixed row) | CI > 0 | cheaper | transfers |
|---|---:|---|---|---:|---:|---:|---:|---|---|---:|---:|---|---|---|---|---|---|
| bend (GSV) | 0.675 | 269/22/58, 0.8706 | 275/31/52 | 0.8987 | 0.8410 | 0.8689 | +0.0183 (+6) | +0.0016 / +0.0092 | +0.0167 [+0.0044, +0.0317] | 9 | 1.64 | 0.21, +10 / +7 = 1.43, 0.8693 | 0.50, 0.8727 | NOT VIABLE | yes | no | **no** |
| paterson (GSV) | 0.687 | 284/15/111, 0.8184 | 288/31/107 | 0.9028 | 0.7291 | 0.8067 | +0.0101 (+4) | +0.0019 / +0.0101 | +0.0083 [−0.0012, +0.0218] | 16 | 4.90 | 0.23, +5 / +4 = 1.25, 0.8193 | 0.26, 0.8212 | NOT VIABLE | no | no | **no** |
| gainesville (GSV) | 0.625 | 210/35/62, 0.8124 | 220/45/52 | 0.8302 | 0.8088 | 0.8194 | +0.0368 (+10) | +0.0036 / +0.0147 | +0.0332 [+0.0135, +0.0558] | 10 | 1.11 | 0.23, +12 / +10 = 1.20, 0.8163 | 0.38, 0.8224 | VIABLE | yes | yes | **yes** |
| annapolis (Mapillary; not counted) | 0.710 | 238/26/56, 0.8530 | 247/34/47 | 0.8790 | 0.8401 | 0.8591 | +0.0306 (+9) | +0.0022 / +0.0170 | +0.0284 [+0.0090, +0.0512] | 8 | 0.96 | 0.18, +18 / +9 = 2.00, 0.8444 | 0.31, 0.8556 | VIABLE | yes | yes | yes |
| richmond (in-sample reference) | 0.641 | 257/28/53, 0.8639 | 269/38/41 | 0.8762 | 0.8677 | 0.8720 | +0.0387 (+12) | +0.0030 / +0.0129 | +0.0357 [+0.0169, +0.0596] | 10 | 0.90 | 0.15, +48 / +12 = 4.00, 0.8214 | 0.33, 0.8712 | VIABLE | yes | yes | yes |

The richmond row reproduces table (a)'s best viable row exactly (the test pins it). Threshold-only
at T_lo 0.05, the control named in criterion 1: bend 0.8095, paterson 0.7817, gainesville 0.7066,
annapolis 0.7593. Pano-bootstrap ΔF1 against baseline, setting fixed: bend [−0.011, +0.008],
paterson [−0.025, +0.001], gainesville [−0.010, +0.025], annapolis [−0.010, +0.023]; no split
resolves an F1 change from zero. Parity-arm density on these splits: 7.0 boxes/pano on bend, 5.7 on
paterson, 5.0 on gainesville, 5.0 on annapolis (6.2 on richmond).

**The rule itself does not fire on wrong-pano challengers.** Each wrong-pano copy of the parity arm
run through criterion 1 at the fixed setting reads VIABLE 0 times on every split (109 copies on
bend, 124 on each of the others, 123 on richmond).

**Two sensitivity reads, not part of the rule, and neither changes the overall verdict.** The first
was in the script committed before scoring; the second was **added after the primary result was
known** (00b09e6, three minutes after d64dcec scored the transfer).

- **richmond's absolute c_min (0.641) instead of each split's median:** bend NOT VIABLE (attributable
  +0.0195, 1.73 FP per ramp against the threshold's 1.43), paterson NOT VIABLE (+0.0104, CI reaches
  −0.0001), gainesville transfers (+0.0334, 1.10 against 1.20), annapolis transfers (+0.0380, 1.34
  against 2.08). Still 1 of 3 GSV splits.
- **Post-seam-fix floor peaks (added after scoring).** The op_caches predate the seam fix f4c71c8. The #25 input-size
  sweep committed a post-fix extraction of the same checkpoint at the same 2048×4096 input
  (`analysis_out/input_res_sweep_25/cache/r2048/`, which reproduces the op_cache exactly once the
  border band is dropped). Scored on those peaks, the fixed setting transfers on **0** of 3 GSV
  splits: gainesville loses criterion 3 (12 FP over 9.03 attributable ramps = 1.33, against the
  threshold's 13 / 10 = 1.30), again by **one FP** (an 11th instead of a 12th would pass), and on the
  raw/raw comparison it would pass (12 / 10 = 1.20 against 1.30), so this cell turns on the
  denominator convention too;
  bend (+0.0167, 1.83 against 1.43) and paterson (+0.0108, 3.76 against 1.88) still fail. annapolis
  still transfers (0.96 against 2.00), and so does richmond (0.99 against 4.17).

**(t2) The in-sample grid per split (secondary; chosen in sample from 60 settings, all shifts).**
These are the per-pair files `analysis_out/cascade_cost_35/<split>__mask2former-vistas-curb-cut-1024x1024.json`,
made by the same command as table (a).

| split | grid verdict | viable rows | best viable (T_lo, r_gate, c_min) | F1 | attributable ΔR | FP / attr ramp | best single threshold F1 | selection-aware: VIABLE in | wrong-pano false-VIABLE rate |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| bend | VIABLE | 2 of 60 | 0.15, 0.011, 0 | 0.8762 | +0.0202 | 0.76 | 0.8727 | 69% | 0 of 109 |
| paterson | VIABLE | 2 of 60 | 0.10, 0.022, 0 | 0.8280 | +0.0229 | 1.00 | 0.8212 | 66% | 0 of 124 |
| gainesville | VIABLE | 28 of 60 | 0.15, 0.022, 0.509 (q25) | 0.8281 | +0.0428 | 0.86 | 0.8224 | 99% | 2 of 124 |
| annapolis | VIABLE | 18 of 60 | 0.10, 0.011, 0 | 0.8681 | +0.0380 | 0.54 | 0.8556 | 99% | 0 of 124 |
| richmond | VIABLE | 4 of 60 | 0.05, 0.011, 0.641 (q50) | 0.8720 | +0.0357 | 0.90 | 0.8712 | 90% | 0 of 123 |

Every split has *some* viable setting in sample, and the best one differs on every split (T_lo 0.05
to 0.15, gate R/2 or R, c_min 0 on three of the four). That is the pattern the rule was written to
guard against: a grid of 60 settings finds a winner on each split, but the winner does not carry
from one split to the next. On bend and paterson the grid verdict rests on 2 of 60 settings and
survives only about two thirds of the selection-aware resamples. gainesville and annapolis are
different: many settings are viable and the verdict survives 99% of resamples. So the evidence
reads as a split-dependent effect, clearly present on gainesville, annapolis and richmond and weak
or absent on bend and paterson, rather than a setting that can be fixed once and deployed.

### What running the second model costs (klone job 40774146, one L40S)

Measured in one Slurm job on one GPU (NVIDIA L40S, node g3102, AMD EPYC 9534 host, 8 CPUs
allocated), the two models one after the other on the same 485 panoramas. Rows are in
`analysis_out/usage_log.jsonl` (`paid: false`, host `g3102`, 2026-09-27).

| | bend (GSV) | paterson (GSV) | gainesville (GSV) | annapolis (Mapillary) | all 485 |
|---|---:|---:|---:|---:|---:|
| panorama size | 16384×8192 | 16384×8192 | 16384×8192 | 8000×4000 | |
| Mask2Former 1024, s/pano (decode + 6-view reprojection + 6 forwards fp16 + masks → points, one thread) | 1.65 | 1.56 | 1.57 | 0.94 | 1.42 |
| Mask2Former model load, s | 46.1 (cold) | 8.3 | 12.1 | 9.0 | |
| RampNet 1× (2048×4096, fp32), wall s/pano (decode in a parallel thread) | 0.91 | 0.88 | 0.90 | 0.40 | 0.77 |
| RampNet 1×, GPU-side s/pano (copy + forward + peaks), fp32 | 0.43 | 0.40 | 0.40 | 0.40 | 0.41 |
| RampNet 1×, GPU-side s/pano, fp16 autocast | | | | | 0.27 |

**Per panorama, the second model costs about 1.8× RampNet's own pass on GSV imagery and 2.3× on
annapolis's smaller panoramas, so running the cascade costs about 2.8–3.3× running RampNet alone.**
On GSV imagery (1.56–1.65 s/pano) a million panoramas is about 430–460 extra **node-hours** on an
L40S node (wall-clock of a GPU node, not GPU compute: most of that time is CPU work; about 400 at the
1.42 s pooled over all four splits, which includes the smaller Mapillary panoramas). Two things make
the ratio approximate. RampNet's timing overlaps JPEG decoding with the GPU in a second thread, and
`compare.py` does not, so the ratio is somewhat unfavourable to Mask2Former; in a deployment the two
would also share one decode. And Mask2Former's GPU forward alone was not isolated here. On the A40
in August it was 0.092 s per view, about 0.55 s per panorama; if that carries over to the L40S, most
of the 1.4–1.6 s is JPEG decode of a 16384×8192 panorama, reprojection and post-processing on the
CPU, which could be optimised. RampNet's GPU work is 0.40–0.43 s/pano. The fp16 RampNet pass is for the like-for-like precision only: its peaks differ from the
op_cache by up to 3×10⁻³ in score and move at most one detection per split at 0.30
(`docs/data/cascade_transfer_35/rampnet_r2048_fp16_instrument_check.json`), so every cascade number
uses the fp32 op_cache.

The whole job took 1,780 s (0.494 GPU-hours), $0 (the lab's own L40S allocation;
`analysis_out/compute_log.jsonl`, `docs/compute_cost.md`). The CPU scoring took under a minute on
`jonfhome`.

### How it was run, and what was checked first

- **Imagery.** Panoramas from the klone mirror `/gscratch/makelab/jonf/rampnet_benchmark/<split>/panos`,
  every file checked against the committed `imagery_manifest.json` sha256 before any GPU work
  (`imagery_manifest.py --verify`: all four OK; `docs/data/cascade_transfer_35/imagery_verify_40774146.log`).
- **The parity arm, with the richmond flags.** `compare.py benchmark/<split> --models
  rampnet,vistas:curb-cut --vistas-input-size 1024 1024` (fp16, `min_area_px` 16, no revision pin,
  as on richmond), fresh cache. Solo at the arm's own points (P / R / F1, AP): bend 0.367 / 0.832 /
  0.509, 0.425; paterson 0.401 / 0.701 / 0.510, 0.430; gainesville 0.383 / 0.846 / 0.527, 0.521;
  annapolis 0.436 / 0.898 / 0.587, 0.526 (richmond: 0.383 / 0.884 / 0.534, 0.649). (RampNet's row in
  those runs is read from the committed bundle, as in every `compare.py` run.)
- **Environment, and how it differs from richmond's run.** `/gscratch/makelab/jonf/envs/sidewalkcv2`:
  Python 3.10.20, torch 2.6.0 (CUDA 12.6), transformers 5.15.0, timm 1.0.28; checkpoint snapshot
  `4772b6bf101d91f2534c106dc524d906aeb3c68a` of `facebook/mask2former-swin-large-mapillary-vistas-semantic`
  and `606a11956743f7eb328d9207769034752f6191f4` of `projectsidewalk/rampnet-model`, recorded in
  `docs/data/cascade_transfer_35/checkpoints.json` and appended to that directory's `env.txt`
  (read from the job's offline HF cache after the run; the launcher now writes them into `env.txt` itself).
  The richmond parity run used transformers 5.15.0 with torch 2.13.0 on an A40 and did not record a
  checkpoint revision. The transformers version is the same; the torch version and the GPU are not.
  On richmond a transformers major-version change moved the 384 arm by one detection in 523, so this
  is expected to be immaterial, but the four new splits have no same-environment control.
- **RampNet's side is unchanged.** The same job re-extracted RampNet's floor peaks on the L40S
  (`input_res_sweep_25.py extract --arms r2048`, fp32) and `check` compared them to the committed
  op_caches: **PASS** on all four splits, identical TP/FP/FN at 0.30 and 0.55, max score difference
  1.1×10⁻⁴ (`rampnet_r2048_fp32_instrument_check.json`). So the committed op_caches are what this GPU
  produces, and the cascade scoring uses them unchanged.
- **Detections published** with `export_model_cache.py` (below) and `--verify` reported all four
  files scoring identically to the cache (output in `docs/data/cascade_transfer_35/export_verify.log`,
  re-run 2026-09-27 after review against the same cache copied from the job).

### Caveats (each applies to every number in this section)

- **Fused precision is a lower bound.** GT on these splits is anchored to RampNet's shipped
  detections and the reviewer's missed marks, so a promoted peak on a real ramp nobody marked scores
  as an FP. The same scorer is applied to the cascade and to the threshold controls; whether
  unmarked ramps are hit equally often by gated peaks and by threshold-added peaks is **not
  measured**. Gated peaks sit on Mask2Former curb-cut segments, so an unmarked real ramp is plausibly
  more likely under a gated peak, which would bias criterion 3 against the cascade. At one-FP margins
  that matters; the absolute precision is not reliable either.
- **The op_caches predate the seam fix f4c71c8**, as for richmond. The post-fix sensitivity above
  measures this directly on these splits (added after scoring): it lowers the count of transferring
  GSV splits from 1 to 0, by one FP on gainesville.
- **One challenger, one RampNet checkpoint, no seeds.** The rule is about this parity arm; nothing
  here measures seed variance of either model.
- **Three GSV splits.** "Transfers on 2 of 3" is the rule as stated; with three splits the rule has
  little resolution, and gainesville's pass is by **one FP** on criterion 3 (1.11 against 1.20),
  which carries no interval.
- **No held-out read of the in-sample grids** (t2): each grid's winner is picked on the split it is
  scored on.

### Reproduction (transfer)

**GPU part.** Any Slurm host with a GPU and network on the login node; the exact steps below are
for klone. Nothing here assumes jfroehli's home or scratch: the scratch root is `$W`, the
interpreter is `$PY`, and the launcher's defaults are documented at its top.

```bash
# 0. a clone at the commit you want, and an env with torch, torchvision, transformers, timm,
#    scikit-image and huggingface_hub (the run used Python 3.10, torch 2.6.0+cu126,
#    transformers 5.15.0, timm 1.0.28; docs/data/cascade_transfer_35/env.txt)
git clone https://github.com/ProjectSidewalk/RampNet.git && cd RampNet
export W=/gscratch/scrubbed/$USER/cascade_35 PY=/path/to/env/bin/python
mkdir -p "$W" logs

# 1. the native-resolution panoramas, into this checkout's benchmark/<split>/panos. All four
#    splits are on projectsidewalk/rampnet-benchmark (config "native"); the unpacker checks every
#    file against the committed imagery_manifest.json and refuses a mismatch. About 6 GB to download (5.86 GB of Parquet), 6.5 GB of JPEGs on disk.
#    (Of the benchmark's splits, laurens_gsv and laurens_mapillary are NOT on the Hub yet and
#    manual_gold is excluded by design -- see scripts/unpack_benchmark_panos.py; none of the
#    three is needed here.)
$PY scripts/unpack_benchmark_panos.py --out "$PWD" --cities bend,paterson,gainesville,annapolis

# 2. the two checkpoints into $W/hf, on the login node (the job runs offline by default).
#    The run resolved facebook/mask2former-swin-large-mapillary-vistas-semantic to
#    4772b6bf101d91f2534c106dc524d906aeb3c68a and projectsidewalk/rampnet-model to
#    606a11956743f7eb328d9207769034752f6191f4; the launcher writes what it resolved to $W/env.txt,
#    so a drift of `main` since then shows up there rather than silently.
HF_HOME=$W/hf $PY -c 'from huggingface_hub import snapshot_download as d
d("facebook/mask2former-swin-large-mapillary-vistas-semantic"); d("projectsidewalk/rampnet-model")'

# 3. one L40S, ~30 min. Writes $W/model_cache, $W/usage_log.jsonl, $W/env.txt, $W/logs/ and the
#    RampNet re-extraction; the Slurm log goes to ./logs/. Account/partition are klone's lab
#    allocation -- change the #SBATCH lines elsewhere. HF_OFFLINE=0 skips step 2 if the compute
#    node has network.
sbatch --export=ALL scripts/analysis/cascade_transfer_35.slurm

# 4. publish the detections from that cache, and prove the files score identically to it
$PY scripts/analysis/export_model_cache.py --cache-dir "$W/model_cache" --models vistas:curb-cut \
    --splits bend,paterson,gainesville,annapolis --vistas-input-size 1024 1024
$PY scripts/analysis/export_model_cache.py --verify --cache-dir "$W/model_cache" --models vistas:curb-cut \
    --splits bend,paterson,gainesville,annapolis --vistas-input-size 1024 1024

# 5. the ledgers: append $W/usage_log.jsonl's rows to analysis_out/usage_log.jsonl, and pull the
#    job's sacct row (docs/compute_cost.md, klone 2026-09-27)
```

`--vistas-revision` is deliberately **not** passed: it enters the detection signature when set,
so it would make a different leg from the published richmond file. The snapshot is pinned by
record instead (`env.txt`). The committed run (job 40774146) used the launcher as it stood at
84db34e, which hardcoded `W`, `PY` and the log path to jfroehli's scratch and had no snapshot
lines in `env.txt`; its snapshot hashes were read from that scratch HF cache after the run and
appended to the committed `env.txt`, marked as such. The steps it ran are the ones above.

**CPU part**, from a clean clone (every input committed; under a minute):

```bash
for s in bend paterson gainesville annapolis; do
  python scripts/analysis/cascade_cost_35.py --split $s --challenger mask2former-vistas-curb-cut-1024x1024
done
python scripts/analysis/cascade_cost_35.py --summary
python scripts/analysis/cascade_transfer_35.py            # transfer.json + table (t1)
python scripts/analysis/cascade_transfer_35.py --check    # byte-compare, writes nothing
python -m pytest -q tests/test_cascade_transfer_35.py
```

The job's own logs, the environment record and both instrument checks are in
`docs/data/cascade_transfer_35/`.

## Reproduction

From a clean clone, CPU only:

```bash
# primary pair, T_hi 0.30, full grid, all shifts, both bootstraps, calibration (~10 s)
python scripts/analysis/cascade_cost_35.py --split richmond --challenger mask2former-vistas-curb-cut-1024x1024

# primary pair at the shipped point
python scripts/analysis/cascade_cost_35.py --split richmond --challenger mask2former-vistas-curb-cut-1024x1024 --t-hi 0.5519 --t-lo 0.05 0.10 0.15 0.20 0.25 0.30 0.40

# every other published leg on richmond, all shifts (~3 min)
python scripts/analysis/cascade_cost_35.py --all-published --splits richmond

# every other split with an op_cache, 20 shifts (~10 min). Since 2026-09-27 this would also
# re-run the four parity-arm pairs of §Transfer with 20 shifts, where the committed files used
# every shift; re-run those four with the per-split commands in §Transfer (or use --check,
# which regenerates each file from the args it records)
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
