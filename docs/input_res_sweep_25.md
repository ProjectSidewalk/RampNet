# Frozen-model input-size sweep (#25, arm 1)

Issue [#25](https://github.com/ProjectSidewalk/RampNet/issues/25) asks whether RampNet should see
more pixels. RampNet was trained, and is deployed, at a fixed 2048×4096 input; most benchmark
panoramas are stored larger (annapolis 8000 px wide, richmond mostly 11000–12288, the GSV splits
13312–16384). This document answers the cheaper of the issue's two questions: **does the released
checkpoint, unchanged, tolerate or benefit from a larger input?** It does not answer whether a
model *retrained* at a higher resolution would gain. The +10-recall-point forecast in
[`detection_recall_analysis.md` §4](detection_recall_analysis.md) is a forecast for that retrain
arm, and nothing here tests it.

Script: `scripts/analysis/input_res_sweep_25.py` (launcher `input_res_sweep_25.sh`, tests
`tests/test_input_res_sweep_25.py`). Every number below re-derives on CPU from the committed
caches in `analysis_out/input_res_sweep_25/cache/<arm>/<split>.json`; the tables are pasted from
`analysis_out/input_res_sweep_25/results.md`, which `report` writes.

## The verdict rule, stated before the tables

Written in the plan before any arm was run, and applied here as written:

> "Resolution helps the frozen model" on a split iff recall at 0.30 rises with the paired 95% CI
> excluding 0 AND precision at 0.30 does not fall by more than the recall gain (F1 non-negative
> within CI) AND the gain exceeds `u4096`'s (i.e. native pixels, not object scale, carry it).
> Anything else is "tolerates" (|ΔF1| CI covers 0) or "hurts" (ΔF1 CI < 0). Far-band recall
> (≥18 m) is reported as the mechanism check, not as the verdict.

How the clauses were made operational (`verdict()` in the script): "recall rises" is the lower
bound of the paired ΔR CI above 0; "precision does not fall by more than the recall gain" is
ΔP ≥ −ΔR on the point estimates; "F1 non-negative within CI" is the upper bound of the ΔF1 CI at
or above 0; "the gain exceeds u4096's" is the lower bound of the paired CI of R(arm) − R(u4096)
above 0. The verdict is taken on the unrounded bounds.

**The rule has one gap, and paterson falls in it.** It names no outcome for a split where F1 rises
with its CI above 0 and recall leads, but the upsample control u4096 rises just as much. The
script labels that case "gains, object scale only". It is not "helps" under the rule, because
the third clause fails. That clause is a clean pixel control only at 4096, where u4096 has the
same input size; at 1.5× the label overstates what is known (see the paterson caveat).

## Result

**The frozen model does not benefit from more input pixels on any split. At the issue's
forecast point, 2× (r4096), the rule reads "hurts" on 10 of 11 splits.** On 7 of those 10 the
recall CI is below 0 as well. On bend, richmond and sao_paulo the ΔR CI covers 0, so their
"hurts" rests on the precision drop, which is a lower bound (marked † in the tables; see the
precision caveat under the headline table). The one exception, paterson, gains F1, and u4096
gains as much at the same input size with no new pixels, so the gain is object scale, not
resolution. Pooled over the eight US splits, r4096 against the 2048×4096 control at 0.30: ΔR
−0.068 [−0.087, −0.046], ΔP −0.075 [−0.092, −0.057], ΔF1 −0.071 [−0.086, −0.055]. 1.5× (r3072)
comes closest to harmless: "tolerates" on annapolis, bend and richmond and on the GSV pool; F1
up on paterson (a gain the rule cannot attribute at 1.5×, see the GSV caveats); ΔR +0.005
[−0.009, +0.019] on the US pool. But precision still falls there (ΔP −0.047 [−0.061, −0.034])
and pooled F1 with it (−0.018 [−0.028, −0.007]), so the US-pool r3072 "hurts" is
precision-driven too.

**r4096 and u4096 are statistically indistinguishable.** Pooled US, R(r4096) − R(u4096) =
+0.008 [−0.000, +0.016] and ΔF1 +0.005 [−0.002, +0.012]; on the headline Mapillary pool it is
−0.001 [−0.013, +0.011]. At 2× the frozen model gets essentially nothing from the real pixels
that it would not get from a bicubic upsample of the 2048 image. What changes its output is the
apparent size of the objects.

**That scale shift moves recall in both directions.** Far ramps gain and near ramps lose. US
pool at 0.30, r4096 against r2048, by flat-ground range: recall in the 25–40 m band goes from
0.543 to 0.750 (n = 232) and in the 40 m+ band from 0.273 to 0.558 (n = 77), so over all 309 GT
ramps beyond 25 m from 0.476 to 0.702 (147 → 217 hits, from those two rows). In the 0–8 m band
(n = 516) it goes from 0.841 to 0.611. The paired CI is for the ≥18 m far band (n = 728): +0.098
[+0.065, +0.131]; none was computed for the 25 m+ band alone. By apparent size, measured at the
control's 2048×4096 input, ramps under 32 px gain and ramps of 80 px or more fall from 0.843 to
0.624. A model trained at one object scale finds the ramps that
the resize brings into its trained size range and loses the ones it pushes out. The forecast
assumed the same thing, but only for a retrained model. The frozen model pays for the far-field
gain with its near field, and the near field holds more ramps.

## Instrument check

The r2048 arm is the committed instrument (`threshold_sweep.PRE` → `heatmap_for` →
`peaks_to_dets`, floor 0.05, `min_distance` 10). It has to reproduce the committed
`analysis_out/op_cache` before any other arm's number is read. `check` ran on makelab2 before the
other five arms were extracted, and the launcher stops if it fails.

**The first check failed, and the failure is a known defect of the reference, not of the
instrument.** On 10 of 10 splits the r2048 arm had more peaks than the op_cache: 173 in all, 0
missing. **All 173 are in the 360° seam strip**, within `min_distance` (10 heatmap columns,
about 3.5°) of x = 0 or x = 1. Two of them, both budapest_district5 and both below 0.30, are
corner peaks, also within 10 rows of the bottom edge; no extra peak is in the top or bottom rows
away from the seam. Nine of the ten op_caches predate the seam fix `f4c71c8` (2026-08-18): six
were written at `c7098be` (2026-07-28: annapolis, bend, budapest_district5, clovis, morgantown,
richmond), paterson at `2e67bcf` (07-29), gainesville at `f024570` (07-30) and sao_paulo at
`bdd7d55` (08-01). That fix made
`exclude_border=False` load-bearing in `threshold_sweep.peaks_to_dets`
([`seam.md`](seam.md)), and skimage's default drops exactly that strip. The tenth,
laurens_mapillary's, was added on 2026-08-31, after the fix, and matches with no adjustment.
`check` therefore compares each split in the mode its op_cache was made in: with the seam strip
set aside for the nine pre-fix caches, and in full for laurens_mapillary. It passes a split only
if (a) that comparison matches every peak position exactly and every score within 2e-4, (b) no
op_cache peak is missing, (c) every extra peak is in the seam strip (asserted as
`extra_at_seam`, so a peak in the top or bottom rows away from the seam would fail), and (d)
tp/fp/fn at 0.30 and 0.55 are identical. `analysis_out/op_cache` was not regenerated or modified here.

| split | op_cache extracted with | seam-strip peaks recovered | peaks missing | max \|score diff\| (matched) | tp/fp/fn @0.30: op_cache / r2048 | tp/fp/fn @0.55: op_cache / r2048 |
|---|---|---|---|---|---|---|
| annapolis | exclude_border=True | 7 | 0 | 4.1e-5 | 238/26/56 / 238/28/56 | 217/6/77 / 217/7/77 |
| bend | exclude_border=True | 10 | 0 | 6.8e-5 | 269/22/58 / 271/25/56 | 247/5/80 / 249/7/78 |
| budapest_district5 | exclude_border=True | 58 | 0 | 6.6e-5 | 193/80/107 / 193/91/107 | 153/22/147 / 153/23/147 |
| clovis | exclude_border=True | 18 | 0 | 5.4e-5 | 160/28/35 / 160/31/35 | 139/13/56 / 139/14/56 |
| gainesville | exclude_border=True | 15 | 0 | 5.3e-5 | 210/35/62 / 210/36/62 | 183/10/89 / 183/11/89 |
| laurens_mapillary | exclude_border=False | 0 | 0 | 3.0e-5 | 131/17/118 / 131/17/118 | 97/11/152 / 97/11/152 |
| morgantown | exclude_border=True | 24 | 0 | 5.0e-5 | 215/27/52 / 217/30/50 | 195/5/72 / 196/5/71 |
| paterson | exclude_border=True | 13 | 0 | 1.04e-4 | 284/15/111 / 286/16/109 | 269/8/126 / 270/9/125 |
| richmond | exclude_border=True | 12 | 0 | 5.4e-5 | 257/28/53 / 258/31/52 | 238/9/72 / 239/11/71 |
| sao_paulo | exclude_border=True | 16 | 0 | 5.1e-5 | 224/55/57 / 224/55/57 | 183/19/98 / 183/19/98 |
| laurens_gsv | — (no op_cache) | — | — | — | not checked | not checked |

Once the seam strip is set aside, the tp/fp/fn counts equal the op_cache's on every checked
split, at both thresholds (asserted by `check`, and for richmond by
`test_r2048_richmond_reproduces_op_cache_modulo_the_seam_strip`). The r2048 column above
*includes* the recovered seam peaks, because that is the corrected extractor and the control
every other arm is compared against.

Caveats on this table:

- **Nine of the ten op_caches predate the seam fix** (six at `c7098be`, the rest 07-29 to
  08-01; `f4c71c8` is 08-18). They lack the floor peaks in the ~3.5° strip beside the seam, so
  op_cache-based numbers in other docs slightly understate both recall and false positives near
  the seam. The cascade-cost write-up in [PR #194](https://github.com/ProjectSidewalk/RampNet/pull/194)
  (open, not yet merged) carries the same caveat.
  **Richmond moves by +1 tp / +3 fp at 0.30 and +1 tp / +2 fp at 0.55.** The +1 tp is pano
  `723487737079243`, the seam site `model_comparison.md` predicted a regenerated cache would
  list: a 0.946 peak at x = 0.0 is a hit. The same ramp also produces a 0.731 peak at
  x = 0.996 on the other side of the seam, which scores as a false positive. Pano
  `1130501775077894` gains a 0.818 seam FP and `1073049581231056` a 0.335 one, which is below
  0.55 and so counts only at 0.30. So this sweep's r2048 row differs slightly from the committed
  richmond row at the recommended point: P/R/F1 0.893/0.832/0.861 here against 0.902/0.829/0.864.
- **Score tolerance is 2e-4, not the plan's 1e-4.** One matched peak out of about 5,000
  (paterson `0Drku25sOlOlWGiVf7uetw`) differs by 1.04e-4, at an identical position and with
  identical counts. Every other peak differs by at most 6.8e-5. This is fp32 convolution noise
  between two machines: the op_cache was extracted on a different GPU and software stack. It is
  the only matched peak above 1e-4.
- **laurens_gsv has no op_cache** and is not checked. Its r2048 arm is the same code path as the
  ten that are.
- The committed native panoramas were verified against each split's `imagery_manifest.json`
  (`imagery_manifest.py --verify`, all 11 splits OK) before the run.

## Arms

Every arm resizes the native jpg with torchvision `Resize` on the PIL image, the call `PRE` makes.

| arm | input (H×W) | answers |
|---|---|---|
| `r2048` (control) | native → 2048×4096 BILINEAR (through `threshold_sweep.PRE` itself) | must equal the op_cache |
| `r3072` | native → 3072×6144 BILINEAR | 1.5× |
| `r4096` | native → 4096×8192 BILINEAR | 2×, the issue's forecast point |
| `rnative` | native size, floored at 2048×4096, **capped at 5500×11000** | what the stored pixels hold, up to the cap |
| `u4096` (upsample control) | native → 2048×4096 BILINEAR → 4096×8192 BICUBIC | object scale with no new information; `r4096 − u4096` is the value of the native pixels |
| `r4096_hm1024` (sensitivity) | as r4096; head upsamples to 1024×2048, peaks at `min_distance` 20 | extraction-parameter sensitivity |

The model resamples its /32 feature map to a fixed 512×1024 heatmap whatever the input size, so
the five main arms extract peaks on the same grid with the same `min_distance` in normalized
units. The sensitivity arm loads the released weights strictly into a 1024×2048 head (Upsample
has no parameters) and reuses r4096's backbone pass, since `model(x)` is exactly
`head(feature_extractor(x))` (tested).

**The rnative cap is 5500×11000, not the planned 6144×12288.** In the smoke test, 6144×12288 ran
out of memory on the A40 in fp32 and again under fp16 autocast. Another process holds 8.8 GB of
the card. 5500×11000 is richmond's dominant native size and peaks at 28.5 GiB in fp32. So the
rnative arm is true native for annapolis, richmond's 11000-wide panos, and every 5760-wide split.
It is downsampled to 5500×11000 for richmond's 14 panos at 12288 and for every GSV pano (13312 and
16384 wide), so for GSV "rnative" means about 2.7× the control, not native. Every pano in every
arm ran in fp32; no cache records an fp16 fallback. rnative is **not** a single scale factor
across splits: it is 1× for morgantown (122 of 125 panos are 4096 wide), about 1.4× for the
5760-wide Mapillary splits, about 2× for annapolis, and about 2.7× for richmond and GSV.

## Headline: the Mapillary splits with headroom, at 0.30

Paired pano-level cluster bootstrap, 2000 replicates, seed 25, the same weight matrix applied to
both arms (`benchmark_power_135.observed_and_se`). Δ is arm − r2048.

| split | arm | P | R | F1 | ΔP | ΔR | ΔF1 | verdict |
|---|---|---|---|---|---|---|---|---|
| annapolis | r2048 | 0.895 | 0.809 | 0.850 | — | — | — | |
| annapolis | r3072 | 0.849 | 0.823 | 0.836 | −0.046 [−0.081, −0.010] | +0.014 [−0.025, +0.049] | −0.014 [−0.043, +0.011] | tolerates |
| annapolis | r4096 | 0.792 | 0.738 | 0.764 | −0.103 [−0.149, −0.058] | −0.071 [−0.124, −0.020] | −0.086 [−0.125, −0.048] | hurts |
| annapolis | rnative | 0.799 | 0.758 | 0.778 | −0.096 [−0.138, −0.053] | −0.051 [−0.104, +0.003] | −0.072 [−0.111, −0.032] | hurts† |
| annapolis | u4096 | 0.790 | 0.731 | 0.760 | −0.104 [−0.152, −0.058] | −0.078 [−0.132, −0.024] | −0.090 [−0.131, −0.049] | hurts |
| richmond | r2048 | 0.893 | 0.832 | 0.861 | — | — | — | |
| richmond | r3072 | 0.808 | 0.868 | 0.837 | −0.085 [−0.131, −0.044] | +0.035 [+0.000, +0.073] | −0.025 [−0.057, +0.007] | tolerates |
| richmond | r4096 | 0.806 | 0.845 | 0.825 | −0.087 [−0.141, −0.040] | +0.013 [−0.030, +0.058] | −0.036 [−0.076, −0.000] | hurts† |
| richmond | rnative | 0.821 | 0.755 | 0.787 | −0.072 [−0.134, −0.017] | −0.077 [−0.134, −0.023] | −0.075 [−0.124, −0.030] | hurts |
| richmond | u4096 | 0.811 | 0.829 | 0.820 | −0.082 [−0.133, −0.038] | −0.003 [−0.047, +0.042] | −0.042 [−0.080, −0.006] | hurts† |
| laurens_mapillary | r2048 | 0.885 | 0.526 | 0.660 | — | — | — | |
| laurens_mapillary | r3072 | 0.913 | 0.462 | 0.613 | +0.028 [−0.011, +0.070] | −0.064 [−0.112, −0.019] | −0.047 [−0.089, −0.006] | hurts |
| laurens_mapillary | r4096 | 0.885 | 0.277 | 0.422 | −0.001 [−0.082, +0.075] | −0.249 [−0.309, −0.189] | −0.238 [−0.300, −0.173] | hurts |
| laurens_mapillary | rnative | 0.895 | 0.478 | 0.623 | +0.010 [−0.024, +0.044] | −0.048 [−0.089, −0.009] | −0.037 [−0.073, −0.002] | hurts |
| laurens_mapillary | u4096 | 0.928 | 0.309 | 0.464 | +0.043 [−0.014, +0.105] | −0.217 [−0.281, −0.151] | −0.196 [−0.258, −0.129] | hurts |
| **pooled (3 splits)** | r2048 | 0.892 | 0.735 | 0.806 | — | — | — | |
| pooled | r3072 | 0.841 | 0.734 | 0.784 | −0.051 [−0.075, −0.027] | −0.001 [−0.025, +0.022] | −0.022 [−0.041, −0.004] | hurts† |
| pooled | r4096 | 0.809 | 0.642 | 0.716 | −0.082 [−0.115, −0.053] | −0.093 [−0.124, −0.061] | −0.090 [−0.115, −0.064] | hurts |
| pooled | rnative | 0.826 | 0.675 | 0.743 | −0.066 [−0.096, −0.037] | −0.060 [−0.089, −0.031] | −0.063 [−0.087, −0.039] | hurts |
| pooled | u4096 | 0.817 | 0.644 | 0.720 | −0.075 [−0.107, −0.046] | −0.091 [−0.125, −0.057] | −0.086 [−0.113, −0.059] | hurts |

† "hurts" with a ΔR CI that covers 0: the verdict is carried by the precision drop, which is a
lower bound (next caveat). The rule is applied as written; the mark is only a reading aid.

AP over the 0.05 floor (headline pool): r2048 0.823, r3072 0.804, r4096 0.739, rnative 0.758,
u4096 0.745. The same comparison at the shipped 0.55 point is in `results.md`; every arm is
worse there than at 0.30 (pooled US ΔR at 0.55: r4096 −0.132 [−0.154, −0.108]), which is
consistent with larger inputs lowering peak heights, not only moving them. Peak heights were not
analysed directly.

Caveats beside this table:

- **Richmond's r4096 "hurts" is on the edge.** The upper bound of its ΔF1 CI is −0.0000454
  (−0.00005 at 5 dp), which rounds to −0.000 in the table. The verdict is taken on the unrounded
  value (tested). A first version of the report rounded before judging and read it as
  "tolerates".
- **Precision at large inputs is a lower bound, and the verdict depends on it wherever the ΔR CI
  covers 0.** The GT is anchored to the reviewed 2048-input detections, so a real ramp that only
  a larger input finds scores as a false positive. No spot-check gallery of the new FPs was made.
  Quote ΔP here as "at least this large an apparent drop". The rule decides on the ΔF1 CI, which
  includes precision, so a "hurts" whose ΔR CI covers 0 (the † rows) is precision-driven: if
  enough of its new FPs are real ramps, its ΔF1 CI would cover 0 and it would read "tolerates".
  richmond r4096, with a ΔF1 upper bound of −0.0000454 and ΔR +0.013 [−0.030, +0.058], would flip
  on very few. Only where recall itself falls with its CI below 0 does "hurts" not rest on
  precision: r4096, rnative and u4096 on all three pools, and the unmarked "hurts" rows. The
  recall columns do not depend on precision. A reviewer pass over the r3072 and r4096 incremental
  FPs is the obvious follow-up before any precision number, or any † verdict, from this sweep is
  quoted on its own.
- **One checkpoint.** Inference is deterministic, so the bootstrap band is the whole uncertainty
  for *this* checkpoint. It says nothing about another checkpoint of the same recipe; seed-to-seed
  movement of about 0.025 F1 was measured in #187.
- **laurens_mapillary is the split most sensitive to scale** (ΔR −0.249 at r4096), and it is
  also the split whose deficit was traced to the capture rig ([#151](https://github.com/ProjectSidewalk/RampNet/issues/151)).
  Its rnative is only 1.4× the control, which is why rnative hurts it much less than r4096 does.

## GSV z5 splits and the rest, at 0.30

| split | r2048 P / R / F1 | arm | ΔR | ΔF1 | verdict |
|---|---|---|---|---|---|
| bend | 0.915 / 0.829 / 0.870 | r3072 | +0.015 [−0.025, +0.056] | −0.007 [−0.033, +0.019] | tolerates |
| | | r4096 | −0.058 [−0.121, +0.003] | −0.069 [−0.110, −0.029] | hurts† |
| | | rnative | −0.116 [−0.184, −0.052] | −0.111 [−0.156, −0.069] | hurts |
| paterson | 0.947 / 0.724 / 0.821 | r3072 | +0.081 [+0.050, +0.115] | +0.043 [+0.023, +0.066] | gains; pixel-vs-scale untested at 1.5×‡ |
| | | r4096 | +0.091 [+0.054, +0.126] | +0.036 [+0.009, +0.063] | gains, object scale only |
| | | rnative | +0.051 [+0.010, +0.092] | +0.011 [−0.021, +0.042] | tolerates |
| | | u4096 | +0.084 [+0.049, +0.119] | +0.037 [+0.012, +0.062] | gains, object scale only |
| gainesville | 0.854 / 0.772 / 0.811 | r3072 | −0.007 [−0.051, +0.035] | −0.033 [−0.066, −0.003] | hurts† |
| | | r4096 | −0.092 [−0.152, −0.027] | −0.085 [−0.129, −0.039] | hurts |
| | | rnative | −0.232 [−0.304, −0.154] | −0.172 [−0.226, −0.116] | hurts |
| sao_paulo | 0.803 / 0.797 / 0.800 | r3072 | +0.025 [−0.011, +0.059] | −0.027 [−0.057, −0.001] | hurts† |
| | | r4096 | −0.036 [−0.082, +0.011] | −0.050 [−0.089, −0.011] | hurts† |
| | | rnative | −0.174 [−0.236, −0.112] | −0.127 [−0.176, −0.079] | hurts |
| laurens_gsv | 0.923 / 0.654 / 0.766 | r3072 | −0.096 [−0.159, −0.032] | −0.073 [−0.120, −0.022] | hurts |
| | | r4096 | −0.264 [−0.364, −0.166] | −0.220 [−0.308, −0.139] | hurts |
| | | rnative | −0.400 [−0.515, −0.291] | −0.370 [−0.487, −0.261] | hurts |
| clovis | 0.838 / 0.821 / 0.829 | r3072 | −0.046 [−0.101, +0.006] | −0.043 [−0.084, −0.002] | hurts† |
| | | r4096 | −0.154 [−0.227, −0.081] | −0.107 [−0.169, −0.050] | hurts |
| | | rnative | +0.000 [−0.058, +0.051] | −0.027 [−0.070, +0.012] | tolerates |
| morgantown | 0.878 / 0.813 / 0.844 | r3072 | −0.049 [−0.091, −0.012] | −0.070 [−0.106, −0.037] | hurts |
| | | r4096 | −0.146 [−0.198, −0.096] | −0.138 [−0.181, −0.096] | hurts |
| | | rnative | +0.000 [+0.000, +0.000] | −0.002 [−0.005, +0.000] | tolerates |
| budapest_district5 | 0.680 / 0.643 / 0.661 | r3072 | −0.080 [−0.126, −0.037] | −0.074 [−0.112, −0.039] | hurts |
| | | r4096 | −0.193 [−0.262, −0.130] | −0.162 [−0.222, −0.106] | hurts |
| | | rnative | −0.040 [−0.083, +0.000] | −0.047 [−0.084, −0.013] | hurts† |
| **US pool** (8 splits) | 0.892 / 0.767 / 0.825 | r3072 | +0.005 [−0.009, +0.019] | −0.018 [−0.028, −0.007] | hurts† |
| | | r4096 | −0.068 [−0.087, −0.046] | −0.071 [−0.086, −0.055] | hurts |
| | | rnative | −0.057 [−0.076, −0.037] | −0.058 [−0.072, −0.043] | hurts |
| | | u4096 | −0.075 [−0.094, −0.055] | −0.076 [−0.090, −0.060] | hurts |
| **GSV pool** (5 splits) | 0.887 / 0.759 / 0.818 | r3072 | +0.014 [−0.005, +0.033] | −0.011 [−0.024, +0.002] | tolerates |
| | | r4096 | −0.051 [−0.079, −0.025] | −0.056 [−0.076, −0.036] | hurts |
| | | rnative | −0.146 [−0.179, −0.114] | −0.116 [−0.142, −0.093] | hurts |

The US pool is `miss_decomposition.US_SPLITS`: richmond, bend, clovis, morgantown, annapolis,
paterson, gainesville and laurens_mapillary. Older docs call it "the seven US splits"; it became
eight when laurens_mapillary joined. budapest_district5 and laurens_gsv are held out of it, as
elsewhere. Every per-arm ΔP, and each split's u4096 row, is in `results.md`.

† as in the headline table: "hurts" with a ΔR CI covering 0, so precision-driven (a lower
bound). budapest_district5 rnative's ΔR upper bound is exactly 0.0 unrounded, so its CI covers 0.
‡ `results.md` and `results.json` carry the script's label, "gains, object scale only"; see the
paterson caveat below for why the doc does not use it at r3072.

Caveats beside this table:

- **The GSV "4× headroom" is mostly not used here.** rnative on GSV is the 5500×11000 cap, about
  2.7× the control. It does worse than r4096 (2×) on every GSV split, the reverse of what more
  real pixels would predict and the same direction as a larger scale mismatch.
- **morgantown's and clovis's rnative rows are near-null controls, not resolution results.**
  122 of morgantown's 125 panos are 4096 wide, so their rnative input *is* the control's input.
  Its ΔR of exactly 0 is the null behaving as it should, and the −0.002 F1 comes from the other 3
  panos. clovis is 5760 wide, only 1.4×.
- **paterson is the one split where a larger input raises F1.** At r4096 the gain is a scale
  effect: u4096, the same 4096×8192 input with no new pixels, does as well, and R(r4096) −
  R(u4096) is +0.008 [−0.013, +0.028]. **The rule's third clause is a valid pixel control only
  at 4096.** For r3072 (and rnative) it compares against u4096, a different input size, and
  there is no u3072 arm, so paterson's r3072 − u4096 (ΔR −0.003 [−0.029, +0.024]) measures scale
  as much as pixels. The script labels paterson r3072 "gains, object scale only"; the honest
  reading is "gains; pixel-vs-scale untested at 1.5×". Why paterson's ramps sit on the side of
  the scale trade that gains was not investigated.

## u4096: is it the pixels or the object scale?

Paired R(arm) − R(u4096) at 0.30. u4096 is the same 4096×8192 input as r4096, made from the
2048 derivative, so it adds no information.

| pool | r3072 − u4096: ΔR | r4096 − u4096: ΔR | r4096 − u4096: ΔF1 | rnative − u4096: ΔR |
|---|---|---|---|---|
| headline (3 Mapillary) | +0.090 [+0.068, +0.116] | −0.001 [−0.013, +0.011] | −0.004 [−0.014, +0.008] | +0.032 [+0.007, +0.056] |
| US pool | +0.081 [+0.066, +0.095] | +0.008 [−0.000, +0.016] | +0.005 [−0.002, +0.012] | +0.018 [+0.003, +0.033] |
| GSV pool | +0.076 [+0.059, +0.094] | +0.011 [−0.002, +0.025] | +0.010 [−0.001, +0.021] | −0.084 [−0.104, −0.064] |

**At a matched 4096×8192 input, native pixels are worth at most about 1.5–2 recall points
(US-pool CI upper bound +0.016), and the point estimate is within noise on every pool.** Per
split, richmond, the split with the most real headroom, is the closest to a pixel gain: R(r4096)
− R(u4096) = +0.016 [+0.000, +0.034], with an unrounded lower bound of exactly 0.0 (so not above
0, and the rule's third clause is not met). The
r3072 − u4096 and rnative − u4096 columns compare *different* input sizes, so they measure scale
as much as pixels: r3072 beats u4096 because 1.5× distorts object scale less than 2× does, not
because of its pixels. This is the reading the issue's caveat anticipated ("upscaling adds no
information"). The frozen model cannot use the extra information either.

## Headroom class

Pooled over all 11 splits, arm vs r2048 at 0.30. Per pano and per arm: **full** means the native
width is at least the arm's input width, so the arm only downsamples. **Partial** means the
native width is above 4096 but below the arm's width, so the arm sees more real detail than the
control and is then upsampled the last stretch; that is annapolis at r4096. **None** means the
native width is 4096 or less, so nothing is new.

| arm | class | panos | ΔP | ΔR | ΔF1 |
|---|---|---|---|---|---|
| r3072 | full | 785 | −0.049 [−0.063, −0.035] | +0.016 [+0.001, +0.033] | −0.013 [−0.024, −0.002] |
| r3072 | partial | 345 | −0.049 [−0.079, −0.021] | −0.064 [−0.092, −0.037] | −0.058 [−0.081, −0.036] |
| r3072 | none | 159 | −0.076 [−0.121, −0.035] | −0.034 [−0.071, −0.003] | −0.054 [−0.085, −0.026] |
| r4096 | full | 660 | −0.064 [−0.085, −0.043] | −0.043 [−0.068, −0.018] | −0.052 [−0.070, −0.034] |
| r4096 | partial | 470 | −0.093 [−0.122, −0.061] | −0.161 [−0.191, −0.129] | −0.136 [−0.161, −0.110] |
| r4096 | none | 159 | −0.102 [−0.157, −0.052] | −0.125 [−0.173, −0.081] | −0.115 [−0.156, −0.077] |
| u4096 | full | 660 | −0.071 [−0.090, −0.053] | −0.054 [−0.077, −0.031] | −0.062 [−0.080, −0.045] |
| u4096 | partial | 470 | −0.080 [−0.107, −0.051] | −0.160 [−0.190, −0.128] | −0.130 [−0.156, −0.104] |
| u4096 | none | 159 | −0.102 [−0.156, −0.052] | −0.125 [−0.170, −0.086] | −0.115 [−0.155, −0.079] |
| rnative | full | 1130 | −0.068 [−0.084, −0.052] | −0.101 [−0.122, −0.081] | −0.088 [−0.104, −0.073] |
| rnative | partial | 0 | — | — | — |
| rnative | none | 159 | +0.000 | +0.000 | +0.000 |

u4096 has no real pixels beyond the control's in any class. Its class is the pano's r4096 class
(both feed an 8192-wide input), so each u4096 row is the no-new-pixels counterpart of the r4096
row on the same panos.

The **none** class is the built-in null: morgantown's 122, richmond's 20, budapest's 15 and
paterson's 2 panos at 4096 wide or less. At r4096 it loses exactly as much as u4096 does
(ΔR −0.125 in both), which is what an upsample with no new pixels should do. **The full class,
where every input pixel is real, still loses at r4096** (ΔR −0.043 [−0.068, −0.018]), and is
only slightly less bad than u4096 on the same panos (−0.054 [−0.077, −0.031]). **In the partial
class the two lose the same recall: r4096 −0.161 [−0.191, −0.129], u4096 −0.160 [−0.190,
−0.128], on the same 470 panos.** Those panos have real detail beyond 4096 px (annapolis's 8000,
the 5760-wide splits), and the frozen model gets nothing from it at 2×. The plan asked for a two-way split (native
wider than the arm, or not). The three-way split is used so that annapolis's 8000-wide panos,
which are upsampled only by 2% at r4096, are not counted with the true null. The class split is
not a controlled comparison: the classes are different cities.

## Recall by distance and apparent size (the mechanism check)

US pool, recall at 0.30, by flat-ground range (`recall_by_depth_112.flat_range`, camera 2.5 m)
and by apparent size at the control's 2048×4096 input (`apparent_px`, 1.2 m ramp, 4096 px per
360°). n is the number of GT ramps on recall-confirmed panos. Five GT points lie at or above the
horizon, where the flat-ground model gives no range; they form the "above horizon" row and are
outside the far band.

| range | n | r2048 | r3072 | r4096 | rnative | u4096 |
|---|---|---|---|---|---|---|
| 0–8 m | 516 | 0.841 | 0.754 | 0.611 | 0.620 | 0.620 |
| 8–12 m | 594 | 0.827 | 0.774 | 0.682 | 0.739 | 0.680 |
| 12–18 m | 466 | 0.805 | 0.807 | 0.755 | 0.745 | 0.740 |
| 18–25 m | 419 | 0.771 | 0.823 | 0.773 | 0.752 | 0.754 |
| 25–40 m | 232 | 0.543 | 0.724 | 0.750 | 0.746 | 0.720 |
| 40 m+ | 77 | 0.273 | 0.545 | 0.558 | 0.558 | 0.558 |
| above horizon | 5 | 0.200 | 0.600 | 0.400 | 0.400 | 0.400 |

| apparent size | n | r2048 | r3072 | r4096 | rnative | u4096 |
|---|---|---|---|---|---|---|
| 0–12 px | 29 | 0.448 | 0.586 | 0.621 | 0.621 | 0.655 |
| 12–20 px | 52 | 0.154 | 0.519 | 0.519 | 0.519 | 0.500 |
| 20–32 px | 240 | 0.537 | 0.713 | 0.738 | 0.742 | 0.708 |
| 32–50 px | 457 | 0.727 | 0.786 | 0.737 | 0.716 | 0.716 |
| 50–80 px | 774 | 0.845 | 0.824 | 0.756 | 0.775 | 0.748 |
| 80 px+ | 752 | 0.843 | 0.755 | 0.624 | 0.648 | 0.630 |
| above horizon | 5 | 0.200 | 0.600 | 0.400 | 0.400 | 0.400 |

Far-band (≥18 m) recall change vs r2048 at 0.30, paired (728 far GT ramps in the US pool): r3072
+0.117 [+0.091, +0.142], r4096 +0.098 [+0.065, +0.131], rnative +0.084 [+0.050, +0.119],
**u4096 +0.077 [+0.044, +0.111]**.

**The hypothesis was far-band gains, and the far band does gain.** But u4096, with no new
information, recovers most of r4096's far-band gain (+0.077 against +0.098), and near ramps
lose more than far ramps gain. That reading rests on two separate CIs against r2048, which
overlap; no paired r4096 − u4096 CI was computed for the far band. The per-split, headline-pool and GSV-pool band
tables are in `results.md`; the GSV pool shows the same crossover. Caveat: the distance axis is
flat-ground geometry at an assumed 2.5 m camera height, the committed axis for all these splits.
[`detection_recall_analysis.md` §0](detection_recall_analysis.md) measures it against GSV's own
depth on the four harvested GSV splits. Band boundaries are approximate, and the crossover, from
a loss near to a gain far, is far larger than that error.

## Sensitivity: the heatmap grid (r4096_hm1024)

Heatmap upsampled to 1024×2048 with `min_distance` 20, so the normalized suppression radius is
unchanged. Against r4096 at 0.30 the two are nearly the same: US pool ΔR −0.066 vs −0.068,
ΔF1 −0.071 vs −0.071, identical on annapolis to 3 dp. The largest difference is bend at 0.55, 0.012 in
recall. Bilinear-then-1×1-conv is linear, so the two
heatmaps differ only by where the peaks are localized. **The conclusion does not depend on the
heatmap grid or `min_distance`.**

## What this does not answer

- **Whether a model retrained at a higher input resolution gains.** That is issue #25's second arm
  and the arm the +10-point forecast describes. This sweep only shows that the released weights
  are tuned to one object scale; a retrain is how that would change. The retrain arm remains open.
- Whether a multi-scale inference (union or max of r2048 and r3072 peaks) would keep the near
  field and add the far field. The far-band gains and near-band losses above suggest it is worth
  measuring: it is a CPU-only merge of two committed caches. It was not in the plan and is not
  done here.
- Whether the new r3072 and r4096 FPs are real ramps the GT missed (see the precision caveat
  above). Enough of them would turn the † "hurts" rows into "tolerates".

## Cost

makelab2, one NVIDIA A40 (another process held 8.8 GB throughout, idle), fp32, `paid: false`,
**$0**. Rows are in `analysis_out/usage_log.jsonl` (run ids `input-res-sweep-25:*`). There is no
`compute_log.jsonl` row, by the rule in [`compute_cost.md`](compute_cost.md): makelab2 has no
Slurm.

| run | wall-clock | what |
|---|---|---|
| r2048, all 11 splits, 1289 panos | 1,104 s | 0.61 s/pano GPU-side |
| first grid attempt | 1,055 s | finished annapolis (kept), died on bend's first rnative pano (OOM, see below) |
| grid, the other 10 splits, 1164 panos | 10,993 s | GPU-side s/pano: r3072 1.33, r4096 2.35, rnative 3.30, u4096 2.34, r4096_hm1024 0.06 (head only) |
| two smoke tests (3 panos × 2 splits) | ~180 s | one row set each |
| **total** | **about 13,330 s (3.7 GPU-hours)** | |

About two minutes are not in the ledger: the first smoke test, which died at the 6144×12288 cap
before the script wrote rows on failure, and a one-pano timing profile. The script now also
writes a `panos_scored: 0` row carrying the elapsed time when a run dies before its first scored
pano (review fix, PR #196). The first grid attempt's six rows name `bundle: "annapolis,bend"`:
it finished annapolis and died on bend's first rnative pano, and the r3072, r4096 and cpu-wait
rows' `panos_scored: 126` include that bend pano (only `bundle` was edited after the run).
The first *logged* smoke test's rows (14:11:06Z) come from an earlier script version (a `decode` row, no
`run_wall_s`). The first grid attempt
died because a retry after an out-of-memory error ran inside the `except` block, where the
traceback still held the failed forward's activations. Recovery now happens outside it (commit
`0016d2f`), and no pano needed a retry in the final run.

## Reproduction, from a clean clone

```bash
pip install -e . && pip install -r requirements-dev.txt

# CPU only, from the committed caches: every table in this doc
python scripts/analysis/input_res_sweep_25.py check    # -> instrument_check.json, exit 1 on failure
python scripts/analysis/input_res_sweep_25.py report   # -> results.json + results.md
python scripts/analysis/input_res_sweep_25.py sums     # outputs + 66 caches vs SHA256SUMS

# the same two into a scratch dir, compared byte for byte with the committed files
python scripts/analysis/input_res_sweep_25.py report --out /tmp/res25/results.json
python scripts/analysis/input_res_sweep_25.py check --out /tmp/res25/instrument_check.json
python scripts/analysis/input_res_sweep_25.py sums --root /tmp/res25 --partial

# Full GPU re-run (~48 GB card; here an A40) of the 9 splits on the Hub, into fresh caches
H9=annapolis,bend,budapest_district5,clovis,gainesville,morgantown,paterson,richmond,sao_paulo
python scripts/unpack_benchmark_panos.py --out /scratch/bench --cities $H9
bash scripts/analysis/input_res_sweep_25.sh /scratch/bench /path/to/venv/bin/activate \
    --cache-root /scratch/res25/cache --cities $H9
```

The launcher runs `extract --arms r2048`, then `check`, then the other five arms, then `report`,
and resumes per arm and split. **Every cache is committed, so without `--cache-root` (or
`--force`, which re-extracts over the committed caches in place) it skips every arm and split
and only runs check and report.** With `--cache-root DIR` the caches go under DIR and
`results.json`, `results.md` and `instrument_check.json` beside it, never over the committed
files. `--limit N` (a smoke test) is refused unless `--cache-root` is a scratch dir.

**laurens_gsv and laurens_mapillary are not on the Hub yet** (`unpack_benchmark_panos.py`
docstring), which is why the command above passes the other nine. Re-running those two needs
the local bundle imagery at `<panos-root>/benchmark/<split>/panos/`; add them to `--cities` when
it is there. Their caches are committed, so the CPU steps cover them. A nine-split report pools
differently (the US pool loses laurens_mapillary, the GSV pool laurens_gsv), so compare per-split
rows, not pooled ones. A re-extraction on another GPU is not expected to be byte-identical (the
instrument check's score tolerance is 2e-4 for that reason), so SHA256SUMS pins the committed
bytes and the CPU steps, not a GPU re-run. A GPU with less free memory than about 30 GB needs
`--native-cap` lowered, which changes the rnative arm; state the cap used. `results.json` from a
re-run of `report` was byte-identical on makelab2 (numpy 2.0.2, Linux) and on Windows (numpy
2.5.1), and again after the review fixes.
