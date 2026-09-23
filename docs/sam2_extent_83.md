# SAM2 extent vs the whole-apron gold (#83 path 1, RampNet 2.0 plan item 7)

**Issue:** #83 (path 1, the first experiment), plan item 7 in [`rampnet2_plan.md`](rampnet2_plan.md).
**Run date:** 2026-09-23 (UTC). **Script:** `scripts/analysis/sam2_extent_83.py`
(tests: `tests/test_sam2_extent_83.py`; exact commands: `scripts/analysis/sam2_extent_83_runbook.sh`).
**Results:** `analysis_out/sam2_extent_83/` (per-item rows, run provenance, summaries).

## Result, in one paragraph

**Path 1 as designed, with a point prompt alone, is not production-grade, and the projection does not
matter.** On Richmond's complete gold (299 ramps), SAM2.1 Hiera-L prompted with one point at the
*gold box center* on a gnomonic view has a median IoU of **0.260** against the whole-apron box; 23%
of ramps reach IoU ≥ 0.5 and 8% reach ≥ 0.75. Prompted from the **RampNet detection** (the
end-to-end arm, 227 `det:` ramps) it is **0.195** (16% ≥ 0.5). The mask is the wrong *object*,
not a slightly wrong outline: its median size is right (size ratio 1.01) but the p10–p90 spread
is 0.45–4.2×, because SAM2 returns the tactile pad or a slab of the ramp on one side of that and
the whole corner, the road or a car on the other (galleries below). **The gnomonic view does not
fix this:** gnomonic minus equirect at 90° is **+0.013 IoU [−0.002, +0.028]** (pano-clustered
95% CI), and **+0.001 [−0.010, +0.012]** at matched magnification (gnomonic 76° vs equirect
90°); across all 72 projection comparisons on Richmond (2 prompts × 4 variants × 9 FOV pairings)
the largest mean delta is 0.026 IoU, none of the 24 matched-FOV CIs excludes zero, and the 9
cross-FOV ones that do split 6 for gnomonic, 3 for equirect. Adding a **box prior built from
flat-ground geometry alone** lifts the median IoU to 0.574 (box center) and 0.436 (detection),
but most of that is the prior: the prior box by itself, no SAM2, scores 0.469 and 0.356. SAM2 adds
+0.098 IoU [+0.075, +0.121] on top of it. On **box width** (the box's horizontal angular span, a
proxy for the ramp width #86 wants, not a measurement of it), SAM2 plus the prior puts 43% of
Richmond's detected ramps within [0.8, 1.25]× the gold width against 44% for the prior alone,
**−0.018 [−0.077, +0.038]**; pooled over all four cities it is 42% vs 39%, **+0.023 [−0.025,
+0.069]**. Per city it helps on São Paulo, hurts on Annapolis, and is unresolved elsewhere (below).

No acceptance bar for extent was ever set, so "not production-grade" is a judgment, stated here
so it can be argued with: at the detection prompt, 16% of ramps reach IoU 0.5 and 17% get a
box width within [0.8, 1.25]×; with the prior, 37% and 43%. (Those are detected ramps only; over
all boxed ramps, IoU ≥ 0.5 falls to about 12% and 28%, see "Who arm 3 covers".) A label that is
wrong on well over half the ramps cannot seed a measurement pipeline.

**The decision this supports for #83 (proposed; Jon's call):** do not mint extent labels with
point-prompted SAM2 on the 850k points.
Path 1 needs work before it can be the extent source, and the work is on *what SAM2 is asked*
(a better prompt or a ramp-specific decoder), not on *what SAM2 sees*: the equirect crop is as good
as the gnomonic view, so the input path for any follow-up can be whichever is cheaper (the
equirect crop needs no reprojection, and the reprojection was two-thirds of this run's
wall-clock). It moves #83's weight toward path 2 (a size head on the keypoint model), with the gold here as its
test set, and makes the geometry prior the baseline any extent method has to beat, on box width
in particular. The three partial-gold cities (two GSV, one Mapillary) reproduce the IoU findings
within their smaller samples; the width comparison does not replicate from city to city (next
section).

## Richmond (primary: complete gold, Mapillary)

299 boxed ramps on 86 panos (the other 6 of `boxes.json`'s 92 panos hold only "can't determine
extent" items; 11 of those are excluded, 299 + 11 = all 310 adjudicated). Headline variant is
`pt_multi` (one point, multimask, best predicted IoU); the `+ prior` rows are `ptbox_multi`.

| arm @ 90° | n | IoU median | mean | ≥0.5 | ≥0.75 | size ratio p10 / p50 / p90 | box width in [0.8, 1.25] | gold covered (med) | mask in gold (med) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. box center, gnomonic, point only | 299 | 0.260 | 0.321 | 0.231 | 0.077 | 0.45 / 1.01 / 4.23 | 0.234 | 0.75 | 0.91 |
| 2. box center, equirect, point only | 299 | 0.239 | 0.308 | 0.231 | 0.064 | 0.46 / 0.99 / 4.29 | 0.194 | 0.78 | 0.91 |
| 3. detection point, gnomonic, point only (`det:` items) | 227 | 0.195 | 0.265 | 0.159 | 0.053 | 0.35 / 0.82 / 4.57 | 0.172 | 0.53 | 0.99 |
| 1 + prior | 299 | 0.574 | 0.553 | 0.625 | 0.171 | 0.66 / 1.04 / 1.63 | 0.458 | 0.86 | 0.95 |
| 2 + prior | 299 | 0.587 | 0.556 | 0.632 | 0.157 | 0.65 / 1.02 / 1.60 | 0.462 | 0.84 | 0.96 |
| 3 + prior (`det:` items) | 227 | 0.436 | 0.439 | 0.366 | 0.110 | 0.61 / 1.01 / 1.63 | 0.427 | 0.72 | 0.85 |
| control: prior box alone, at the box center | 299 | 0.469 | 0.455 | 0.431 | 0.060 | 0.43 / 0.88 / 1.43 | 0.381 | 0.64 | — |
| control: prior box alone, at the detection (`det:` items) | 227 | 0.356 | 0.343 | 0.167 | 0.004 | 0.42 / 0.89 / 1.42 | 0.445 | 0.50 | — |

- *Size ratio* is √(area SAM2 box / area gold box); *box width in [0.8, 1.25]* is the share whose
  SAM2 box width is within [0.8, 1.25]× the gold box width (an empty mask counts as a miss; there
  were ≤ 1 per cell); *gold covered* is the share of the gold box inside the SAM2 box; *mask in
  gold* the share of the mask's pixels inside the gold box.
- **Box width is not ramp width.** It is the axis-aligned equirect box's horizontal angular span
  (longitude), so it is not in metres, and for a ramp seen diagonally it mixes the ramp's width with
  its length. It is the only width-like quantity a box score can give, so read it as a proxy for
  #86's measurement. It is also a **post-hoc** column: it was added in `93c478f`, after the
  Richmond numbers had been seen. The headline variant (`HEADLINE_VARIANT = "pt_multi"`) and the
  prior's constants, by contrast, were committed in `a258256`, before any run. The band is
  symmetric in log ratio (1/1.25 = 0.8); the symmetric-in-ratio [0.8, 1.2] barely moves Richmond's
  detection comparison (0.396 vs 0.401).
- **Who arm 3 covers.** Arm 3 and the detection controls are on the 227 `det:` ramps: RampNet
  detections that a reviewer judged true, at the shipped 0.55 threshold (`records.jsonl` holds
  nothing below it; Richmond's lowest is 0.5519). Production's recommended operating point is 0.30
  ([`operating_point.md`](operating_point.md)). So arm 3 is conditioned on a confident, correct
  detection. The 72 `missed:` ramps are the ones the shipped threshold never prompts at all
  (RampNet still misses 53 of Richmond's ramps at 0.30, `model_comparison.md`, so 0.30 would add
  about 19 low-confidence prompts). On `missed:` items the recorded point is a reviewer's click,
  not a detection; those rows are in the CSV and the summary (`subsets`), not in this table.
  **End to end**, over all 299 boxed ramps, the 16% (point only) and 37% (with the prior) at
  IoU ≥ 0.5 become about 12% and 28%, because a ramp with no detection gets no extent. That is
  before false detections, which would get an extent too.
- The box center is an oracle prompt: production has no gold box. It bounds what a perfectly
  centered prompt buys; the detection rows are the operating number.

**Prompt variants** (90°, gnomonic; the equirect arm ranks them the same way):

| variant | n | IoU median | mean | ≥0.5 | ≥0.75 | size ratio p10 / p50 / p90 | box width in [0.8, 1.25] | gold covered (med) | mask in gold (med) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| box center, gnomonic, `pt_multi` | 299 | 0.260 | 0.321 | 0.231 | 0.077 | 0.45 / 1.01 / 4.23 | 0.234 | 0.75 | 0.91 |
| box center, gnomonic, `pt_single` | 299 | 0.281 | 0.335 | 0.258 | 0.067 | 0.45 / 1.00 / 4.11 | 0.237 | 0.74 | 0.92 |
| box center, gnomonic, `ptbox_multi` | 299 | 0.574 | 0.553 | 0.625 | 0.171 | 0.66 / 1.04 / 1.63 | 0.458 | 0.86 | 0.95 |
| box center, gnomonic, `ptbox_single` | 299 | 0.571 | 0.535 | 0.599 | 0.157 | 0.56 / 0.94 / 1.48 | 0.448 | 0.79 | 0.97 |
| detection, gnomonic, `pt_multi` (`det:`) | 227 | 0.195 | 0.265 | 0.159 | 0.053 | 0.35 / 0.82 / 4.57 | 0.172 | 0.53 | 0.99 |
| detection, gnomonic, `pt_single` (`det:`) | 227 | 0.222 | 0.276 | 0.167 | 0.049 | 0.34 / 0.80 / 4.49 | 0.181 | 0.47 | 0.99 |
| detection, gnomonic, `ptbox_multi` (`det:`) | 227 | 0.436 | 0.439 | 0.366 | 0.110 | 0.61 / 1.01 / 1.63 | 0.427 | 0.72 | 0.85 |
| detection, gnomonic, `ptbox_single` (`det:`) | 227 | 0.405 | 0.404 | 0.339 | 0.049 | 0.49 / 0.89 / 1.49 | 0.401 | 0.59 | 0.86 |

Single-mask vs multimask is a wash for a point (within 0.03 median IoU); a box prior dominates either.
Multimask with the prior picks bigger masks (size p50 1.04 vs 0.94) and wins by a little.

**By distance band** (box center, gnomonic, 90°; cells are IoU median / share ≥ 0.5; bands are
depression strata from the gold box's center row at the 2.5 m convention; the delta columns are
gnomonic − equirect, mean IoU with the pano-clustered 95% CI, 2,000 resamples per band):

| band | n | point only | point + prior | prior alone | Δ proj, point only | Δ proj, + prior |
|---|---:|---:|---:|---:|---:|---:|
| >36 m / horizon | 23 | 0.205 / 0.09 | 0.430 / 0.35 | 0.070 / 0.00 | -0.032 [-0.101, +0.028] | +0.015 [-0.018, +0.050] |
| 18-36 m | 89 | 0.260 / 0.16 | 0.562 / 0.66 | 0.390 / 0.24 | +0.008 [-0.024, +0.038] | +0.008 [-0.009, +0.026] |
| 9-18 m | 121 | 0.274 / 0.29 | 0.587 / 0.62 | 0.511 / 0.55 | +0.029 [+0.007, +0.057] | -0.012 [-0.032, +0.004] |
| 5-9 m | 57 | 0.249 / 0.26 | 0.583 / 0.65 | 0.597 / 0.63 | +0.004 [-0.007, +0.014] | -0.007 [-0.030, +0.021] |
| <5 m | 9 | 0.202 / 0.33 | 0.707 / 0.89 | 0.586 / 0.67 | +0.013 [-0.013, +0.041] | -0.008 [-0.037, +0.021] |

The projection delta is flat across bands, including the near field (<5 m, 5–9 m) where the
1/cos(latitude) stretch is largest. The one band here whose CI excludes zero (9–18 m, point only,
+0.029) does not repeat with the prior. Over every per-band projection comparison in the summary
(120: both prompts, all variants, the three matched FOVs, five bands), 16 CIs exclude zero against 6
expected by chance at 5%, and 13 of the 16 favour the gnomonic view. **6 of the 16 are in the
<5 m band**, which holds only 9 ramps on a handful of panos (2,000 resamples), and they split 4
for gnomonic, 2 for equirect (−0.019 and −0.039, both box center at 60°); the third negative is
at >36 m (−0.032, `ptbox_multi` at 60°). All 8 in 5–18 m favour gnomonic: 5 in 9–18 m (where
17 of 24 comparisons are positive, largest +0.035) and 3 in 5–9 m (largest +0.026). The <5 m
band also holds the largest delta, −0.080, with a CI spanning zero. So there may be a small
gnomonic edge of about +0.03 IoU at 5–18 m; it is an order of magnitude smaller than the gap to
a usable extent, and the matched-FOV headline deltas do not show it. (The tally is
`band_projection_ci_exclusions` in `richmond_summary.json`; `tests/test_sam2_extent_83.py`
re-derives it from the rows.)

The far field (>36 m) is where everything fails: those gold boxes
are a median 142 native px wide (30–277), about a third of that after SAM2's resize to 1024. The
prior alone collapses there (median 0.070: near the horizon a small depression error is a large
distance error), SAM2 with a point stays at its usual 0.2, and the two together reach 0.43. At the
other end, 5–9 m, the prior alone (0.597) already matches SAM2 with the prior (0.583).

**Paired deltas** (mean over the same ramps; 95% CI from 10,000 pano-clustered resamples, seed
83; the last column counts ramps where the first arm won / lost):

| comparison | n | mean Δ IoU | 95% CI | median Δ | wins / losses |
|---|---:|---:|---:|---:|---:|
| `gnomonic-equirect\|boxcenter\|90\|pt_multi` | 299 | +0.0128 | [-0.0024, +0.0279] | +0.0019 | 159 / 140 |
| `gnomonic-equirect\|boxcenter\|90\|ptbox_multi` | 299 | -0.0031 | [-0.0141, +0.0078] | -0.0008 | 147 / 151 |
| `gnomonic@76-equirect@90\|boxcenter\|pt_multi` | 299 | +0.0010 | [-0.0096, +0.0118] | +0.0003 | 159 / 140 |
| `gnomonic@76-equirect@90\|boxcenter\|ptbox_multi` | 299 | -0.0010 | [-0.0075, +0.0054] | -0.0019 | 139 / 158 |
| `gnomonic-equirect\|boxcenter\|60\|pt_multi` | 299 | -0.0082 | [-0.0168, +0.0009] | -0.0008 | 134 / 164 |
| `gnomonic-equirect\|boxcenter\|60\|ptbox_multi` | 299 | -0.0001 | [-0.0074, +0.0078] | -0.0011 | 148 / 151 |
| `gnomonic-equirect\|point\|90\|pt_multi` | 299 | +0.0129 | [-0.0024, +0.0281] | +0.0008 | 157 / 141 |
| `gnomonic-equirect\|point\|90\|ptbox_multi` | 299 | +0.0039 | [-0.0070, +0.0144] | +0.0000 | 147 / 147 |
| `gnomonic@76-equirect@90\|point\|pt_multi` | 299 | +0.0054 | [-0.0041, +0.0156] | +0.0017 | 181 / 117 |
| `gnomonic@76-equirect@90\|point\|ptbox_multi` | 299 | +0.0057 | [-0.0023, +0.0137] | +0.0010 | 158 / 131 |
| `fov90-fov60\|boxcenter_gnomonic\|pt_multi` | 299 | +0.0077 | [-0.0104, +0.0250] | +0.0007 | 153 / 146 |
| `fov90-fov60\|boxcenter_gnomonic\|ptbox_multi` | 299 | -0.0021 | [-0.0163, +0.0111] | +0.0061 | 158 / 141 |
| `detpoint-boxcenter\|det\|gnomonic\|90\|pt_multi` | 227 | -0.0782 | [-0.1179, -0.0419] | -0.0106 | 97 / 130 |
| `detpoint-boxcenter\|det\|gnomonic\|90\|ptbox_multi` | 227 | -0.1339 | [-0.1570, -0.1107] | -0.0782 | 55 / 172 |
| `sam-prior\|boxcenter_gnomonic\|90\|ptbox_multi` | 299 | +0.0977 | [+0.0750, +0.1210] | +0.0903 | 219 / 79 |
| `sam-prior\|point_gnomonic\|90\|ptbox_multi` | 299 | +0.0888 | [+0.0689, +0.1088] | +0.0549 | 203 / 91 |

Reading the keys: `gnomonic-equirect|<prompt>|<fov>|<variant>` is the matched-FOV projection
delta; `gnomonic@76-equirect@90` the matched-magnification one; `fov90-fov60` the FOV sensitivity
within one projection; `detpoint-boxcenter` the cost of prompting from the detection rather than
the gold center (`det:` ramps); `sam-prior` SAM2 + prior minus the prior box alone.

**Why the detection prompt costs so much.** 91% of the detection points fall *inside* the gold
box (median offset 0.09 box-widths horizontally, 0.19 box-heights vertically), yet the paired
cost is −0.078 IoU point-only and −0.134 with the prior. With a point alone, SAM2's choice of
segment is sensitive to where on the ramp the point lands (a point on the pad returns the pad).
With the prior, the prior box moves with the point, so its own error adds.

**Failure gallery** (box center, gnomonic, 90°, `pt_multi`): green = gold whole-apron box,
magenta = SAM2 box from the gnomonic view, orange = SAM2 box from the equirect crop of the same
prompt, white dot = prompt. Captions: IoU gnomonic, IoU equirect, band, item.

Worst 8 (all IoU ≤ 0.02): the mask leaks into the road, a building face, a car or the whole
crosswalk; three are `>36 m` ramps and the rest 9–36 m. The equirect arm fails on the same items
(orange boxes, IoU ≤ 0.09).

![worst SAM2 point-prompt cases](assets/sam2_extent_83_richmond_pt_multi_worst.jpg)

Median 8 (IoU ≈ 0.26): the typical miss is *part* of the ramp (the tactile pad, or the sloped
apron without its flares) or the ramp plus the adjoining sidewalk corner.

![median SAM2 point-prompt cases](assets/sam2_extent_83_richmond_pt_multi_median.jpg)

With the geometry prior (`ptbox_multi`), median 8 (IoU ≈ 0.57) and worst 8:

![median SAM2 point + prior cases](assets/sam2_extent_83_richmond_ptbox_multi_median.jpg)

![worst SAM2 point + prior cases](assets/sam2_extent_83_richmond_ptbox_multi_worst.jpg)

## Secondary: Annapolis, São Paulo, Paterson (partial gold)

The same pipeline ran unchanged on the three partial whole-apron sets (random-pano prefixes; see
[`crop_window_eval.md`](crop_window_eval.md) Round 3). They add a second provider (GSV: São Paulo,
Paterson) and a higher-resolution Mapillary city (Annapolis, 8000 px). Cells are IoU median (share
≥ 0.5), all at 90°, gnomonic, `pt_multi` / `ptbox_multi`; detection columns are `det:` ramps only;
the last column is the paired SAM2-plus-prior minus prior-alone delta with the recorded-point
prompt (all items, so it includes the reviewer clicks on `missed:` ramps).

| split (imagery) | n box / det | box center: point only | + prior | prior alone | detection: point only | + prior | prior alone | box width in [0.8, 1.25], detection: + prior / prior alone | SAM2 + prior − prior alone (recorded-point prompt, all items) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Richmond (Mapillary, complete) | 299 / 227 | 0.260 (0.23) | 0.574 (0.63) | 0.469 (0.43) | 0.195 (0.16) | 0.436 (0.37) | 0.356 (0.17) | 0.43 / 0.44 | +0.089 [+0.069, +0.109] |
| Annapolis (Mapillary, partial) | 131 / 94 | 0.183 (0.14) | 0.516 (0.53) | 0.428 (0.40) | 0.160 (0.09) | 0.462 (0.38) | 0.352 (0.16) | 0.38 / 0.49 | +0.085 [+0.061, +0.113] |
| São Paulo (GSV, partial) | 119 / 75 | 0.295 (0.24) | 0.589 (0.63) | 0.523 (0.55) | 0.289 (0.21) | 0.471 (0.43) | 0.371 (0.25) | 0.40 / 0.21 | +0.054 [+0.029, +0.078] |
| Paterson (GSV, partial) | 109 / 86 | 0.264 (0.14) | 0.527 (0.53) | 0.509 (0.52) | 0.191 (0.07) | 0.416 (0.33) | 0.373 (0.17) | 0.44 / 0.31 | +0.072 [+0.039, +0.106] |
| 3 partial, pooled | 359 / 255 | 0.249 (0.17) | 0.537 (0.56) | 0.495 (0.49) | 0.200 (0.12) | 0.441 (0.38) | 0.360 (0.19) | 0.41 / 0.35 | +0.070 [+0.055, +0.087] |
| all 4, pooled | 658 / 482 | 0.255 (0.20) | 0.555 (0.59) | 0.487 (0.46) | 0.197 (0.14) | 0.439 (0.37) | 0.358 (0.18) | 0.42 / 0.39 | +0.079 [+0.066, +0.092] |

Projection deltas (gnomonic − equirect, box-center prompt, mean IoU, pano-clustered 95% CI):

| split | Δ proj @90, point only | Δ proj @90, + prior | Δ gnomonic@76 − equirect@90, point only | Δ gnomonic@76 − equirect@90, + prior |
|---|---:|---:|---:|---:|
| Richmond | +0.013 [-0.002, +0.028] | -0.003 [-0.014, +0.008] | +0.001 [-0.010, +0.012] | -0.001 [-0.007, +0.005] |
| Annapolis | +0.011 [-0.019, +0.039] | +0.000 [-0.015, +0.014] | -0.001 [-0.013, +0.012] | +0.000 [-0.011, +0.011] |
| São Paulo | +0.010 [-0.021, +0.036] | -0.013 [-0.034, +0.006] | +0.010 [-0.017, +0.040] | -0.012 [-0.026, +0.000] |
| Paterson | +0.016 [-0.009, +0.039] | -0.014 [-0.030, +0.004] | +0.015 [-0.005, +0.038] | -0.006 [-0.019, +0.008] |
| 3 partial, pooled | +0.012 [-0.004, +0.028] | -0.009 [-0.019, +0.001] | +0.008 [-0.004, +0.020] | -0.006 [-0.013, +0.002] |
| all 4, pooled | +0.012 [+0.001, +0.023] | -0.006 [-0.014, +0.001] | +0.005 [-0.003, +0.013] | -0.004 [-0.009, +0.001] |

What replicates: point-only SAM2 sits at a median IoU of 0.18–0.30 in every city, with the
detection prompt at or below the box center; the prior alone accounts for most of the lift the
prior gives (SAM2 adds +0.05 to +0.09 over it); and no single city shows a projection delta whose
CI excludes zero. Pooled over all 658 ramps, the matched-FOV point-only delta is **+0.012
[+0.001, +0.023]**, the only projection CI in these headline rows to clear zero; it disappears at
matched magnification (+0.005 [−0.003, +0.013]) and turns negative with the prior (−0.006
[−0.014, +0.001]). Read it as "the projection is worth about 0.01 IoU at most", not as a reason
to reproject.

**What does not replicate: box width.** The paired comparison is the share of detected ramps
whose box width is within [0.8, 1.25]× the gold, SAM2 + prior (`point_gnomonic`, 90°,
`ptbox_multi`) minus the prior box alone, on `det:` ramps, with the same pano-clustered bootstrap
(10,000 resamples, seed 83; key `width-sam-prior|det|point_gnomonic|90|ptbox_multi` in each
summary, and the same pairing over all items under `width-sam-prior|all|…`):

| split | n (`det:`) | SAM2 + prior | prior alone | Δ | 95% CI |
|---|---:|---:|---:|---:|---:|
| Richmond (Mapillary, complete) | 227 | 0.427 | 0.445 | −0.018 | [−0.077, +0.038] |
| Annapolis (Mapillary) | 94 | 0.383 | 0.489 | −0.106 | [−0.194, −0.020] |
| São Paulo (GSV) | 75 | 0.400 | 0.213 | +0.187 | [+0.078, +0.315] |
| Paterson (GSV) | 86 | 0.442 | 0.314 | +0.128 | [−0.011, +0.259] |
| 3 partial, pooled | 255 | 0.408 | 0.349 | +0.059 | [−0.011, +0.132] |
| all 4, pooled | 482 | 0.417 | 0.394 | +0.023 | [−0.025, +0.069] |

SAM2 helps on São Paulo and hurts on Annapolis; Richmond, Paterson and both pools are unresolved.
The point estimates look like a provider split (GSV up, Mapillary down), but the CIs do not
support one: Paterson's spans zero, and one resolved city per provider is not a pattern. The
prior's own size bias also differs by rig (its 2.5 m camera is closer to GSV's ~2.2 m than to
Mapillary's ~1.7 m, `crop_window_eval.md`), so the prior-alone column moves with the city as much
as SAM2 does. Either way, no arm gets half of the detected ramps' box widths within [0.8, 1.25]×.

## Method

**Question.** RampNet emits points. Given a point, can a point-prompted SAM2 recover the ramp's
whole-apron extent well enough to measure it (width first, #86)? This is #83's path 1, the one
that needs no Stage 2 retrain.

**Gold.** `benchmark/<city>/boxes.json`, drawn with `scripts/box_gallery.py` under BOX_RULE v2
(whole constructed ramp surface: apron + pad + flares; tight; axis-aligned), #116. Richmond is
complete (299 boxed + 11 "can't determine extent" = all 310 adjudicated ramps, 92 panos,
Mapillary). Annapolis (131 + 11 of 294, Mapillary), São Paulo (119 + 15 of 281, GSV) and Paterson
(109 + 10 of 395, GSV) are partial random-pano samples ([`crop_window_eval.md`](crop_window_eval.md)
Round 3). "Can't determine extent" items are excluded and counted; nothing else is dropped.
`manual_labels/` boxes are not used: they are tactile-pad marks, not aprons (#114).

**Arms.** Every boxed item is segmented from two prompt sources, each through two projections,
at three fields of view:

| name | prompt | what SAM2 sees |
|---|---|---|
| `boxcenter_gnomonic` | gold box center (oracle, "GT-center") | rectilinear (gnomonic) view centered on the prompt |
| `boxcenter_equirect` | gold box center | plain seam-wrapped equirect crop centered on the prompt (`box_gallery.py`'s cut) |
| `point_gnomonic` | the item's recorded point | gnomonic view |
| `point_equirect` | the item's recorded point | equirect crop |

The recorded point is, on `det:<i>` items, the RampNet detection the reviewer judged true
(checked against `records.jsonl` at load, so it is exactly the model's output); on `missed:<i>`
items it is the reviewer's click. So **plan arm 1** is `boxcenter_gnomonic`, **arm 2** is
`boxcenter_equirect`, and **arm 3 (end-to-end)** is `point_gnomonic` on the `det:` items.

FOVs: **90°** (the annotation view, `crop_fov_deg` in every `boxes.json`), **60°** (narrower,
the sensitivity row), and **76°**. The 76° row exists because equal FOV is not equal
magnification: at FOV f a gnomonic view's center carries `(side/2)/tan(f/2)` pixels per radian
against the equirect crop's `side/f`, so at 90° the gnomonic view shows the prompted ramp at
0.785× the linear size. SAM2 resizes both to 1024 px, so this is what it sees. Gnomonic at
2·atan(π/4) = 76.3° matches the equirect crop at 90° in center magnification, so
**gnomonic@76 vs equirect@90 is the matched-magnification comparison**, and gnomonic@90 vs
equirect@90 is the matched-FOV one. Both are reported. The match is exact vertically and at the
horizon; at the ramp's own row the equirect crop is still 1/cos(depression) wider horizontally
(×1.035 at 15° below the horizon, ×1.10 at 25°).

Both images are cut at the same pixel side, `box_gallery.crop_side(W, H, fov)` (3072 px at 90°
on a 12288-wide pano), from the native pano; the gnomonic view is rendered bilinearly (the in-repo
`equirect_to_perspective` is nearest-neighbour, fine for a detector, not for edges), using the
camera math of `scripts/model_comparison/equirect_tiling.py` unchanged.

**Prompt variants.** One image embedding per view; four decodes from it. `pt_multi`: one
positive point, multimask output, keep the mask with the highest predicted IoU. **This is the
headline, fixed in the script before any number was seen** (`HEADLINE_VARIANT`), because a lone
point is the ambiguous case multimask exists for. `pt_single`: one point, single mask.
`ptbox_multi` / `ptbox_single`: the point plus a box prior computed from production-available
geometry only: flat ground, 2.5 m camera, the prompt's depression gives a ground distance, and
the prior is the image of a 3 m × 3 m ground square (2 × a 1.5 m nominal apron) centered there.
It never sees the gold.

**Scoring.** Each mask is reduced to its tight bounding box in pano-normalized equirect
coordinates. For a gnomonic mask, the four corners of every boundary pixel are mapped back
through the inverse projection and the box is taken with x unwrapped around the view center, so
a mask crossing the 0/1 seam gets a narrow box. For an equirect crop, pixel edges are offset by
the crop origin, modulo the pano width. Per item: **IoU** with the gold box (seam-aware);
**gold coverage** (share of the gold box inside the SAM2 box); **mask in gold** (share of the
mask's pixels inside the gold box); the **size ratio** √(area_SAM2 / area_gold); and SAM2's own
predicted IoU. Bands are crop_window_eval's depression strata, from the gold box's center row
(flat ground, 2.5 m camera; read them as bands, not distances). The projection delta is paired
by item, and its 95% CI is a **pano-clustered** percentile bootstrap (10,000 resamples, seed 83),
since ramps in one pano share imagery and rig. Items are keyed on (city, pano, item) and clustered
on (city, pano), so a pooled summary never merges two cities' items (no pano id collides today;
the change from (pano, item) moved pooled CI endpoints by ≤ 0.001, through resampling order
only). The width comparison uses the same bootstrap on a 0/1 in-band indicator.

## Inputs and provenance

| input | identifier |
|---|---|
| SAM2 code | `facebookresearch/sam2` at `2b90b9f5ceec907a1c18123530e92e794ad901a4` (main, 2024-12-15), installed editable, dist `SAM-2` 1.0, CUDA extension not built (`SAM2_BUILD_CUDA=0`) |
| checkpoint | `sam2.1_hiera_large.pt` from `dl.fbaipublicfiles.com/segment_anything_2/092824/`, sha256 `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`; config `configs/sam2.1/sam2.1_hiera_l.yaml` |
| environment | Python 3.11.15, torch 2.5.1+cu124, bf16 autocast; full freeze in [`data/sam2_extent_83_env_freeze.txt`](data/sam2_extent_83_env_freeze.txt) |
| gold | `benchmark/<city>/boxes.json` (BOX_RULE v2); Richmond sha256 `f6b7f73c…6293b`, every city's in its `<city>_run.json` |
| detections | `benchmark/<city>/records.jsonl`; `det:<i>` points are asserted equal to the record's detection at load |
| panos | `benchmark/<city>/panos/`, git-ignored; `run` verifies every file against `imagery_manifest.json` (Richmond digest `ec8ef3678aa9bf84`) and refuses to start on a mismatch. The imagery is published on HF as [`projectsidewalk/rampnet-benchmark`](https://huggingface.co/datasets/projectsidewalk/rampnet-benchmark), config `native`, but **no committed command unpacks it into `panos/`** (see Reproduce) |
| host | makelab2 (1× NVIDIA A40, shared with one unrelated process holding ~8.8 GB) |

Committed outputs, `analysis_out/sam2_extent_83/` (LF-pinned, floats rounded to 5 places):

| file | sha256 |
|---|---|
| `all4_summary.json` | `ccf2f3f7e789e54de0d442e6441d2ec574068fe6c8fc788cca0f77de6b1ed8ed` |
| `annapolis_rows.csv` | `ae02c87380aad01faae7154e27370d3bea196fe40909b373cb9d8c3362d6c37f` |
| `annapolis_run.json` | `3c1ed9641a3bceb3f8718021b0a1539574eea42550c3f2bafd8686a1d8c23f13` |
| `annapolis_summary.json` | `393cf075d7629a372c2658cc7aee7ca1e4a03e829dd08c42f766efce1c2287d7` |
| `partial3_summary.json` | `58ebcd32f2e075fb0c5e25cb4cc501a3ec3a4adb3354efedd3b01b3bc40c2409` |
| `paterson_rows.csv` | `3f556a04e58f64668663a0215292d6074dc27d2dfb74d4172ed19cc182aa377d` |
| `paterson_run.json` | `a214b970ec1f9edd7a12f0539fb752dbee20c6ba837ebed2f6772afa286c8c6b` |
| `paterson_summary.json` | `95a9b58acc391c9f954690995a22bac8b967c28874eaf4d069d00ad7c3a12fd3` |
| `richmond_rows.csv` | `af33a8699344b409df0392dc051252ccfbc1a6ad652881e5ff9b41276b3ec871` |
| `richmond_run.json` | `b3d2f8a97ae4f164a40f809296a0a4650f0538c7a24f9650522744cf01fb25a1` |
| `richmond_summary.json` | `8d469713e92a80debcb924e0a02496a56deca11d3eb84100c8457fea26a64db0` |
| `sao_paulo_rows.csv` | `77609e21da3da7cfa4fe272287c8bdd0ad8365b3e6d8b0bf7a30053513dd6954` |
| `sao_paulo_run.json` | `90db88d955b6dec75d2db997b15b9261492a8340f7223a26ede9f544a5824653` |
| `sao_paulo_summary.json` | `146fe9b3cdda0cc4a994ec250184541e8180dbf17c618cb04effc29014a6c431` |

Each `<city>_run.json` records the rows-CSV sha256 as written on makelab2; the committed CSVs
match it byte for byte. The summaries regenerate on CPU from the CSVs
(`summarize`, deterministic: seeded bootstrap).

## What it cost

makelab2 has no Slurm, so the time is recorded as `paid: false` rows in
`analysis_out/usage_log.jsonl` (provider `sam2`), per [`compute_cost.md`](compute_cost.md).

| run | panos | ramps | views | wall-clock | render | in SAM2 calls: `set_image` / `predict` | in no timer |
|---|---:|---:|---:|---:|---:|---:|---:|
| smoke (Richmond, 2 panos) | 2 | 7 | 84 | 90 s | not split | not split | not split |
| Richmond | 86 | 299 | 3,588 | 3,289 s (54.8 min) | 2,294 s | 202 / 372 s | 421 s |
| Annapolis | 42 | 131 | 1,572 | 876 s (14.6 min) | 571 s | 65 / 124 s | 116 s |
| São Paulo | 40 | 119 | 1,428 | 4,252 s (70.9 min) | 2,802 s | 274 / 740 s | 436 s |
| Paterson | 30 | 109 | 1,308 | 3,172 s (52.9 min) | 2,023 s | 217 / 566 s | 365 s |
| **four full runs** | | **658** | **7,980** | **11,589 s (3.22 h)** | 7,691 s | 758 / 1,801 s | 1,338 s |
| **with the smoke run** | | | | **11,679 s (3.24 h)** | | | |

Dollar cost: $0 (lab hardware). **The wall-clock is the reliable number.** The split is timer
arithmetic (`time.time()` around each step, no `torch.cuda.synchronize()`), so how the SAM2 time
divides between `set_image` and `predict` is not reliable. Their sum, about **0.71 h**, is closer,
because `predict` returns numpy and so waits for the GPU, but it also includes upsampling the
masks to the 3–4k px view and the GPU→CPU copy. Whether `set_image` synchronizes internally was
not checked. The last column is the rest: 1,301 s inside the per-view loop but outside any timer
(JPEG-decoding each pano, back-projecting and scoring every mask, building rows) and 37 s before
and after it (the sha256 pre-check of every pano, model load, CSV write). The A40 was shared
throughout with one unrelated process (~8.8 GB). São Paulo and Paterson are slower per view
because nearly all their panos are 16384 px wide, so their views are 4096 px square before SAM2
downsamples them.

Most of the wall-clock is not the GPU. On Richmond, the numpy gnomonic render took 2,294 s of
3,289 (70%). Anyone scaling this up should render on the GPU or at 1024 px directly; the equirect
arm needs no render at all.

## What was not run, and why

- **No second rater on the gold.** All four `boxes.json` were drawn by one annotator (jonf). The
  IoUs here are against one person's reading of BOX_RULE v2; the gap between SAM2 and the gold is
  large enough (median 0.2–0.6) that rater noise of the size seen elsewhere in this repo does not
  change the decision, but no inter-rater IoU exists to prove it.
- **No manual_gold arm.** `manual_labels/` boxes are pad marks (#114) and the GSV re-annotation
  (`box_gallery.py --from-manual-labels`) has not been drawn.
- **No SAM2 fine-tune, no automatic-mask-generator, no text-prompted variant** (e.g. Grounded
  SAM). Out of scope for path 1's first experiment; the decision above is what would justify
  them.
- **Only one box prior** (2.5 m camera, 1.5 m apron, scale 2). It was not tuned, deliberately: a
  prior tuned on this gold would be scored on the gold it was tuned on. Its measured size bias
  (median size ratio 0.88) says a per-rig camera height would move it; that belongs to whichever
  follow-up adopts the prior.
- **Masks are not kept**, only their boxes, so the committed rows cannot be re-scored against a
  polygon gold if one is ever drawn. A full re-run of the four cities took 3.22 h wall-clock on
  makelab2's A40 (Richmond alone 0.91 h), of which about 0.71 h was inside SAM2 calls
  (unsynchronized timers, above). Two-thirds of the wall-clock was the gnomonic render, so a
  masks-only re-run of the equirect arms alone would skip it.
- **SAM2 small/base/tiny were not run.** Large is the ceiling for the family; a smaller model
  cannot rescue a prompt problem.

## Caveats

- **The gold is a box, and so is the score.** BOX_RULE v2 boxes oblique ramps axis-aligned with
  empty corners, and the SAM2 box is the same kind of box, so IoU compares like with like; it
  says nothing about mask shape inside the box.
- **The box center is a biased oracle for oblique ramps**: the center of an axis-aligned box
  around a diagonal apron can sit off the ramp surface. The detection prompt does not have this
  problem and does worse anyway (see the detection-offset paragraph).
- **Bands** are depression strata at a 2.5 m flat-ground convention (Richmond's Mapillary rig is
  nearer 1.7 m, `crop_window_eval.md`), so read them as bands, not distances; the `<5 m` band has
  9 ramps.
- **Partial cities are pano-sampled prefixes** (random panos, every ramp in each sampled pano),
  so per-ramp rates are unbiased for the covered panos but the partial files each carry a
  `completeness_warning`, repeated in their run JSON.
- **The resolution SAM2 sees is fixed by SAM2**, not by the pano: every view is resized to 1024 × 1024
  px, so a 90° view gives SAM2 about 11 px per degree as long as the source crop is at least
  1024 px. Nearly every pano here is 5760–16384 px wide; the exceptions are Richmond's 9 panos at
  4096 px (a 1024 px crop, no resize) and one Paterson gen-1 pano at 3328 px, whose 832 px crop is
  upsampled. The FOV rows are the only resolution lever tested.

## Reproduce

Every step is in `scripts/analysis/sam2_extent_83_runbook.sh`, in order. Two inputs are not in a
clean clone:

- **The SAM2.1 checkpoint**: public, fetched by the runbook, sha256 pinned.
- **The panos, `benchmark/<city>/panos/<pano_id>.jpg`. Gap: no committed command produces them.**
  They are published in [`projectsidewalk/rampnet-benchmark`](https://huggingface.co/datasets/projectsidewalk/rampnet-benchmark),
  config `native`, one Parquet per city at `data/native/<city>.parquet`; each row carries
  `pano_id`, the exact source bytes (`image.bytes`, not re-encoded) and their `sha256`
  (`benchmark/README.md`, `docs/replication.md` §3). `scripts/export_benchmark.py` builds, verifies
  and pushes that Parquet, but none of its modes (build / verify / records / card / push / adopt)
  writes it back out as files, and `imagery_manifest.json` pins each file's sha256, size and pixel
  dimensions but not how it was fetched. The step itself is small (write each row's bytes to
  `benchmark/<city>/panos/<pano_id>.jpg`), and a wrong unpack cannot produce numbers silently,
  because `run` re-hashes every pano against `imagery_manifest.json` and refuses to start on a
  mismatch. What closes the gap is a tested `extract` mode on `scripts/export_benchmark.py` (or
  a small fetch script) that ends with `python scripts/analysis/imagery_manifest.py --verify`.

**What "reproduced" means.** SAM2 inference was deterministic on the as-run setup: the smoke run
(`--limit 2`, the first two Richmond panos in sorted order) wrote 336 rows that are
**byte-identical** to the first 336 data rows of the committed `richmond_rows.csv` from the full
run (sha256 of both: `ddc4da3b…edb94a20`). So on an A40 with the pinned environment the target is
exact: `cmp` your rows CSV against the committed one. On other GPUs, bf16 kernels can differ in
the last bits, so expect row-level IoUs to move slightly and compare the summaries' medians and
CIs instead. The runbook runs the smoke comparison first (step 3), then the full run into your
own `$OUT`, then summarizes **your** rows (step 4), not the committed ones.

```bash
# GPU (environment + checkpoint setup is in the runbook). Smoke first: must equal the committed
# rows' first 337 lines (header + 336) on an A40.
python scripts/analysis/sam2_extent_83.py run --city richmond --limit 2 \
    --arm boxcenter_gnomonic,boxcenter_equirect,point_gnomonic,point_equirect --fov 90,76,60 \
    --checkpoint /path/sam2.1_hiera_large.pt --out "$WORK/runs/smoke" --usage-log "$WORK/runs/usage_rows.jsonl"
head -n 337 analysis_out/sam2_extent_83/richmond_rows.csv | cmp - "$WORK/runs/smoke/richmond_rows.csv"

# then every city, into your own $OUT
python scripts/analysis/sam2_extent_83.py run --city richmond \
    --arm boxcenter_gnomonic,boxcenter_equirect,point_gnomonic,point_equirect --fov 90,76,60 \
    --checkpoint /path/sam2.1_hiera_large.pt --out "$OUT" --usage-log "$WORK/runs/usage_rows.jsonl"

# CPU: summarize YOUR rows, then compare with the committed ones
for c in richmond annapolis sao_paulo paterson; do
  python scripts/analysis/sam2_extent_83.py summarize --city $c --out "$OUT"
  cmp "$OUT/${c}_rows.csv" analysis_out/sam2_extent_83/${c}_rows.csv
done
python scripts/analysis/sam2_extent_83.py summarize --city annapolis,sao_paulo,paterson --name partial3 --out "$OUT"
python scripts/analysis/sam2_extent_83.py summarize --city richmond,annapolis,sao_paulo,paterson --name all4 --out "$OUT"

# CPU, no GPU at all: regenerate the committed summaries byte-for-byte from the committed rows
python scripts/analysis/sam2_extent_83.py summarize --city richmond --out analysis_out/sam2_extent_83
git diff --exit-code analysis_out/sam2_extent_83/
python scripts/analysis/sam2_extent_83.py gallery --city richmond --gallery-variant pt_multi \
    --out analysis_out/sam2_extent_83 --assets docs/assets      # needs Richmond's panos
pytest -q tests/test_sam2_extent_83.py
```
