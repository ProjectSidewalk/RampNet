# Gated cascade with Gemini 4 Argon as the second arm (#35)

**Issue:** [#35](https://github.com/ProjectSidewalk/RampNet/issues/35). **Run date:** 2026-11-15.
**Script:** `scripts/analysis/cascade_argon_35.py` (tests: `tests/test_cascade_argon_35.py`).
**Outputs:** `analysis_out/cascade_cost_35/argon/` (37 per-pair files in `cascade_cost_35.py`'s
format, `argon_pairs.json`, `runs.json`).
**Compute:** CPU only, committed inputs only, no model call, no network, no spend. About 2 min for the
pair files and 91 s for `argon_pairs.json` on `jonfhome`; `--check` takes 164 s there. No ledger
row, as for `cascade_cost_35.md`.
**Status:** exploratory. No rule was stated before these numbers were read.

## Why

[`cascade_cost_35.md`](cascade_cost_35.md) never ran the Gemini 4 Argon legs; its `summary.json`
lists them under `gaps`. The question here is whether agreement with Argon tells us more about a
RampNet sub-threshold peak than RampNet's own score does, and whether it tells us more than an
open-weight detector does. It bears on whether to pay for an "Argon as teacher" distillation run.

## Result

**On the open-imagery splits, gating on Argon is not better than gating on a YOLO pano arm.**
Pooled over the four open-rig splits the two are tied to within 0.005 F1 under either read, and
on `laurens_mapillary`, the split with the largest cascade gain, the best YOLO arm is ahead of
both Argon legs. Argon is ahead on `juneau` only, by about +0.01, with an interval that includes 0.

Argon's agreement does carry information beyond RampNet's score on two splits: on
`laurens_mapillary` and `bayonne` both Argon legs read VIABLE with every resample of the
selection-aware bootstrap viable, and on `laurens_mapillary` the cascade beats the best single
threshold by +0.069 [+0.034, +0.110] (zero-shot) and +0.080 [+0.044, +0.120] (confidence leg).
The YOLO arms do the same thing there, and slightly more of it.

## Method

`cascade_cost_35.run_pair`, unchanged, with the study's arguments: T_hi 0.30, T_lo ∈ {0.05 … 0.25},
r_gate ∈ {R/2, R, 2R}, 20 evenly spaced null shifts (all 123 on richmond), 2,000 resamples, seed 0.
Same GT, same op_cache floor peaks, same scorer (`score_pano`, radius 0.022, wrapped).

- **`gemini-4-argon`** (standing zero-shot leg) emits no score: 15 settings, c_min = 0 only.
- **`gemini-4-argon-confidence-on`** (the #276 leg) has scores: 60 settings, c_min ∈ {0, q25, q50, q75}.
- **Open-weight arms:** `y11l_pano`, `y11x_pano_h200`, `y26_pano`. Their committed files in
  `analysis_out/cascade_cost_35/` are read as they are. The study had not run them on `juneau`,
  `bayonne` or `laurens_gsv`; those nine pairs are run here.

Three reads of each pair, all the study's: `best_viable`, `best_by_f1` (both chosen **in sample**,
from 15 settings for the score-less leg and 60 for a scored arm, which favours the scored arms) and
`fixed` (T_lo 0.05, r_gate R/2, c_min = the arm's median score or all boxes; chosen on richmond ×
the Vistas arm, so selected on none of these splits or arms).

**Intervals** are `cascade_cost_35.bootstrap_delta` (pano resample, paired, setting held fixed).
The study points it at a cascade row and the 0.30 baseline. Here the same function is also pointed
at the best single threshold, the threshold at matched recall, and the best open-weight arm's row.
That is the only thing added to the study's method.

**Pools** are count sums (TP, FP, recall TP, recall GT added across splits), one arm on every split:
`open_276` = `laurens_mapillary` + `bayonne` (`prompt_276_heldout.OPEN_POOL`), `open_rig` = those
plus `juneau` and `budapest_district5`. The cascade study has no pool and no pooled bootstrap, so
the pooled rows carry no interval.

## Open-rig splits (F1)

Baseline is RampNet's op_cache peaks at ≥ 0.30. "Best thr" is the best single threshold on a
0.05–0.95 sweep. All four splits are Mapillary or Panoramax imagery.

**`best_by_f1` (in-sample best grid row)**

| split | n GT | baseline @0.30 | best thr (t) | best YOLO arm | Argon zero-shot | Argon + confidence |
|---|---:|---:|---:|---|---:|---:|
| laurens_mapillary | 249 | 0.6600 | 0.7082 (0.16) | 0.8018 (y26_pano) | 0.7775 | 0.7877 |
| juneau | 297 | 0.8294 | 0.8303 (0.26) | 0.8354 (y26_pano) | 0.8449 | 0.8401 |
| bayonne | 301 | 0.6000 | 0.6467 (0.19) | 0.6774 (y11x_pano_h200) | 0.6533 | 0.6643 |
| budapest_district5 | 300 | 0.6736 | 0.6839 (0.37) | 0.6814 (y11l_pano) | 0.6779 | 0.6767 |
| **pool open_276** | 550 | 0.6265 | 0.6739 (per split) | 0.7231 (y11x_pano_h200) | 0.7090 | 0.7201 |
| **pool open_rig** | 1147 | 0.6948 | 0.7175 (per split) | 0.7384 (y11x_pano_h200) | 0.7358 | 0.7395 |

**`fixed` (not selected on these data)**

| split | baseline @0.30 | best thr (t) | best YOLO arm | Argon zero-shot | Argon + confidence |
|---|---:|---:|---|---:|---:|
| laurens_mapillary | 0.6600 | 0.7082 (0.16) | 0.7564 (y26_pano) | 0.7423 | 0.7133 |
| juneau | 0.8294 | 0.8303 (0.26) | 0.8270 (y11x_pano_h200) | 0.8404 | 0.8399 |
| bayonne | 0.6000 | 0.6467 (0.19) | 0.6426 (y11x_pano_h200) | 0.6492 | 0.6452 |
| budapest_district5 | 0.6736 | 0.6839 (0.37) | 0.6678 (y11l_pano) | 0.6678 | 0.6735 |
| **pool open_276** | 0.6265 | 0.6739 (per split) | 0.6910 (y11l_pano) | 0.6904 | 0.6752 |
| **pool open_rig** | 0.6948 | 0.7175 (per split) | 0.7192 (y11l_pano) | 0.7243 | 0.7191 |

`best_viable` equals `best_by_f1` on these four splits for both Argon legs. Verdicts under the
study's rule: both Argon legs VIABLE on all four. YOLO: all three VIABLE on `laurens_mapillary` and
`bayonne`; on `budapest_district5` only `y11l_pano`; on `juneau` only `y26_pano`. One calibration
flag: the confidence leg's wrong-pano copies read VIABLE 1 time in 20 on `laurens_mapillary`.

Attributable ΔR of the best viable row (after the wrong-pano null), Argon zero-shot / confidence /
best YOLO: `laurens_mapillary` +0.134 / +0.150 / +0.182; `bayonne` +0.088 / +0.103 / +0.121;
`juneau` +0.035 / +0.036 / +0.028; `budapest_district5` +0.028 / +0.029 / +0.023.

## Paired intervals (ΔF1, Argon cascade minus the comparator, 95%)

| split | leg | read | vs best threshold | vs threshold at matched recall | vs best YOLO arm |
|---|---|---|---|---|---|
| laurens_mapillary | zero-shot | best_by_f1 | +0.069 [+0.034, +0.110] | +0.088 [+0.052, +0.130] | −0.024 [−0.050, +0.000] (y26) |
| laurens_mapillary | zero-shot | fixed | +0.034 [−0.003, +0.076] | +0.034 [−0.003, +0.076] | −0.014 [−0.047, +0.018] (y26) |
| laurens_mapillary | confidence | best_by_f1 | +0.080 [+0.044, +0.120] | +0.112 [+0.076, +0.154] | −0.014 [−0.040, +0.012] (y26) |
| laurens_mapillary | confidence | fixed | +0.005 [−0.028, +0.042] | +0.005 [−0.028, +0.042] | **−0.043 [−0.069, −0.017]** (y26) |
| juneau | zero-shot | best_by_f1 | +0.015 [−0.001, +0.030] | +0.022 [+0.002, +0.042] | +0.010 [−0.007, +0.027] (y26) |
| juneau | zero-shot | fixed | +0.010 [−0.007, +0.026] | +0.017 [−0.003, +0.038] | +0.014 [−0.000, +0.028] (y11x) |
| juneau | confidence | best_by_f1 | +0.010 [−0.007, +0.026] | +0.017 [−0.005, +0.038] | +0.005 [−0.013, +0.023] (y26) |
| juneau | confidence | fixed | +0.010 [−0.006, +0.025] | +0.017 [−0.004, +0.039] | +0.013 [−0.002, +0.028] (y11x) |
| bayonne | zero-shot | best_by_f1 | +0.007 [−0.024, +0.039] | same threshold | −0.024 [−0.057, +0.010] (y11x) |
| bayonne | zero-shot | fixed | +0.002 [−0.027, +0.034] | same threshold | +0.007 [−0.022, +0.038] (y11x) |
| bayonne | confidence | best_by_f1 | +0.018 [−0.004, +0.042] | same threshold | −0.013 [−0.040, +0.014] (y11x) |
| bayonne | confidence | fixed | −0.002 [−0.032, +0.033] | same threshold | +0.003 [−0.025, +0.033] (y11x) |
| budapest_district5 | zero-shot | best_by_f1 | −0.006 [−0.031, +0.021] | +0.041 [+0.022, +0.060] | −0.004 [−0.021, +0.014] (y11l) |
| budapest_district5 | zero-shot | fixed | −0.016 [−0.035, +0.005] | +0.025 [+0.008, +0.043] | +0.000 [−0.012, +0.011] (y11l) |
| budapest_district5 | confidence | best_by_f1 | −0.007 [−0.034, +0.018] | +0.046 [+0.028, +0.063] | −0.005 [−0.023, +0.014] (y11l) |
| budapest_district5 | confidence | fixed | −0.010 [−0.031, +0.010] | +0.031 [+0.013, +0.049] | +0.006 [−0.008, +0.019] (y11l) |

Of the sixteen Argon-minus-YOLO intervals, one excludes 0, and it favours YOLO; a second ends at exactly 0.000 on the same side. Argon beats the
best single threshold with an interval above 0 on `laurens_mapillary` only (in-sample rows); on
`budapest_district5` neither Argon leg beats it. All 14 splits are in `argon_pairs.json` and
`--tables`; on the ten other splits every Argon-minus-YOLO interval includes 0 except `laurens_gsv`
zero-shot `fixed`, −0.032 [−0.060, −0.003].

## Argon-only points

An Argon box with no RampNet floor peak within the scoring radius, at the op_cache's 0.05 floor and
at the labeler's 0.10 storage floor. Scored alone by `score_pano` on the recall-confirmed panos.

| split | leg | floor | boxes | TP / FP / ignored | precision |
|---|---|---:|---:|---|---:|
| laurens_mapillary | zero-shot | 0.05 | 36 | 17 / 18 / 1 | 0.486 |
| laurens_mapillary | zero-shot | 0.10 | 46 | 27 / 18 / 1 | 0.600 |
| laurens_mapillary | confidence | 0.10 | 82 | 34 / 45 / 3 | 0.430 |
| bayonne | zero-shot | 0.05 | 85 | 28 / 55 / 2 | 0.337 |
| bayonne | zero-shot | 0.10 | 99 | 33 / 63 / 3 | 0.344 |
| bayonne | confidence | 0.10 | 192 | 42 / 141 / 9 | 0.230 |
| juneau | zero-shot | 0.05 | 31 | 6 / 23 / 2 | 0.207 |
| juneau | zero-shot | 0.10 | 37 | 8 / 27 / 2 | 0.229 |
| juneau | confidence | 0.10 | 109 | 17 / 87 / 5 | 0.163 |
| budapest_district5 | zero-shot | 0.05 | 84 | 13 / 68 / 3 | 0.160 |
| budapest_district5 | zero-shot | 0.10 | 85 | 13 / 69 / 3 | 0.159 |
| budapest_district5 | confidence | 0.10 | 152 | 27 / 120 / 5 | 0.184 |

**Mostly false positives against the ground truth on three of the four splits**, and about even
on `laurens_mapillary`. Every one of these TPs is a ramp the 0.30 baseline misses. On the other splits
the precision of Argon-only points is 0.00–0.25 at the 0.05 floor.

## Caveats

- **Argon-only FPs are upper bounds.** GT on every split but `manual_gold` is RampNet's reviewed
  detections plus the reviewer's missed marks. A real ramp that neither RampNet nor the reviewer
  marked scores as an FP. On `manual_gold` (independent labels) Argon-only precision is 0.14.
- **The cascade cannot use an Argon-only point at all.** It only promotes RampNet's own peaks, so
  the cascade numbers say nothing about what Argon's extra boxes would add as training labels.
  The table above is the only evidence here on that, and it is a count, not an experiment.
- **`best_by_f1` and `best_viable` are in-sample**, and the scored arms pick from four times as
  many settings as the zero-shot leg.
- **The pooled rows have no interval**, and the per-split intervals are conditional on the setting.
- **20 null shifts** on every split but richmond, as in the study; **pre-seam-fix op_caches**, as in
  the study (`cascade_cost_35.md`, Caveats).
- **One Argon run per leg.** #276 Step 7 measured run-to-run variation for the prompt arms; nothing
  here measures it for the cascade.
- **Not run:** the two `definition-*` Argon legs, the Gemini 3.8 Flash legs, the `y11x_tiles` legs
  and the Vistas arm on the splits the study skipped. `summary.json` still lists every pair in this
  note under `gaps`, because it reads only the files directly under `analysis_out/cascade_cost_35/`.
  It was left unchanged on purpose: adding these files there would change `summary.json`, the 141-pair
  counts quoted in `cascade_cost_35.md`, and the tests that pin them.

## Where each number lives

| number | file | key |
|---|---|---|
| per-split baseline, best threshold | `analysis_out/cascade_cost_35/argon/argon_pairs.json` | `splits[].baseline`, `.threshold_only_best` |
| each arm's three reads, verdict | same | `splits[].arms.<leg>.{best_viable,best_by_f1,fixed,verdict}` |
| paired intervals | same | `splits[].paired.<leg>.<read>.minus_*` |
| Argon-only counts | same | `splits[].argon_only.<leg>[]` (`none` = no floor peak) |
| pools | same | `pools.{open_276,open_rig}` |
| full grid, null, calibration per pair | `analysis_out/cascade_cost_35/argon/<split>__<leg>.json`, and `analysis_out/cascade_cost_35/<split>__<yolo leg>.json` for the pairs the study ran | as in `cascade_cost_35.md` |

## Reproduction

```bash
# from a clean clone, CPU only; every input is committed
python scripts/analysis/cascade_argon_35.py            # runs missing pairs, writes argon_pairs.json (~4 min)
python scripts/analysis/cascade_argon_35.py --tables   # every split, all three reads
python scripts/analysis/cascade_argon_35.py --check    # regenerate all 37 pair files + argon_pairs.json, byte-compare (~3 min)
pytest -q tests/test_cascade_argon_35.py
```
