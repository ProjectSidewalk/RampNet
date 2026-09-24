# Where RampNet's recall goes: a depth-quantified error analysis

Analysis of every curb ramp RampNet **fails to detect** on the deployment benchmark
(`benchmark/{richmond,bend}` — 637 reviewer-confirmed ramps across 234 panos, 150 missed at the
deployed operating point). Scripts: [`scripts/analysis/`](../scripts/analysis/README.md).

**Bottom line.** Recall is *distance-limited*: RampNet is reliable to ~18 m and effectively blind
past 25 m. Three levers close the gap, in increasing cost — a free operating-point change
(+7–10 pts), higher-resolution retraining (+10 pts, saturating ~0.88), and denser pano sampling
(unmeasured, plausibly the largest). Precision is **not** the problem and culling distant
detections actively hurts.

> **The metre labels were re-measured on GSV's own depth (§0, #112).** On bend, the GSV half
> of this population, the flat-ground axis below runs **~8% long** (per-point median), so
> "18 m / 25 m" reads **16.7 m / 23.2 m** there; with the labeler's depth-frame correction it is
> within 2% of what is printed. The ~25–35% stretch the issue measured is a property of Google's
> **2025–26 rig** (paterson, gainesville), whose camera sits at ~1.8–1.9 m instead of ~2.4 m, and
> on that rig the same thresholds are **12.9 m / 17.9 m**. Richmond is Mapillary, which serves no
> depth, so its half of every table stays on the flat-ground axis. Every table in this document
> is kept as published; the depth-axis versions are in §0.

## Why this needed depth

The benchmark labels are **points, not boxes**, so a ramp's apparent size is unknown — yet "the
misses are small and far away" was the leading hypothesis. Two independent distance estimates were
used:

1. **Flat-ground geometry.** Ramps sit on the ground, so `d = camera_height / tan(depression)`,
   where depression comes from the point's latitude in the equirectangular pano.
2. **Depth Anything 3 (metric)** run on the perspective-reprojected views from
   `scripts/model_comparison/equirect_tiling.py`, with our exactly-known intrinsics.

They agree to within **6.5–8.5%** (Spearman ρ = 0.95 Bend / 0.81 Richmond). Depth additionally
rescues 4 Richmond ramps that geometry placed *above the horizon* — geometrically impossible for a
ground ramp, and a direct symptom of unleveled consumer rigs / hills. That ρ gap is itself
informative: geometry degrades exactly where the camera rig varies (Mapillary), which is the OOD
imagery we care most about.

Apparent size then follows from distance: a ramp of real width `W` at distance `d` subtends
`W/d × (4096 / 2π)` px in RampNet's 4096-px-wide input space.

## 0. The distance axis, re-measured on GSV depth (#112)

Every distance below is flat-ground geometry (or DA3, which agreed with it to 6.5–8.5%) at an
*assumed* 2.5 m camera height. GSV's depth payload is a list of planes plus a per-pixel plane
index, and the dominant ground plane's distance is the camera height — a per-panorama
measurement, not a constant. The sidewalk-auto-labeler archived that payload for every panorama
of its bend, paterson, gainesville and sao_paulo runs; the 485 benchmark panoramas of those four
splits are all in it, sha256-verified against its `index.csv`. laurens_gsv was never harvested
and richmond is Mapillary, so neither has a depth axis.

`scripts/analysis/recall_by_depth_112.py` re-derives the axis with the labeler's own parser
(`depth.py` at labeler commit `86bb909`, branch `camera-height-40`; `origin/main` was `c4bebf1`
and lacks only the stand-in-ground classification used here) and re-issues this document's tables
on the flat axis and the depth axis side by side, per split and pooled over the four GSV splits.
The committed `analysis_out/recall_by_depth_112.json` carries every GT point and detection with
both distances, and every table re-derives from those rows on CPU
(`recall_by_depth_112.py --check`, pinned by `tests/test_recall_by_depth_112.py`), so nothing
here needs the payloads to check. The full tables, per split, are in
`analysis_out/recall_by_depth_112.md`.

**The rule.** Depth distance = horizontal range along the exact ray through the point to the
payload plane under its pixel (the plane index is per pixel of a 512×256 grid; the intersection
is continuous, never snapped to a pixel row). If that plane is not ground-like (a wall or a car
in front of the ramp: 39 of 1,100 GT points) the point falls back to the measured camera height
over level ground, and the row says so. Apparent size uses the Euclidean ray distance. Ground
truth and hits are exactly this document's (verdict review, deployed 0.55, the same greedy
matcher); the richmond + bend population reproduces at **637 / 0.765** on the way in.

**Which panoramas count.** 14% of GSV payloads carry Google's stand-in ground — an exactly
level plane at exactly 2.500 m, a default rather than a measurement — and a few are 2-plane
fallbacks or implausible. Those panoramas are **excluded from the depth axis**, never backfilled,
and their GT points stay on the flat axis only:

| split | panos | measured | stand-in ground | degenerate / implausible | camera height median (min–max), measured |
|---|---:|---:|---:|---:|---|
| bend | 110 | 90 | 20 | 0 | 2.37 m (1.66–2.49) |
| paterson | 125 | 113 | 10 | 2 / 0 | 2.06 m (1.46–2.50) |
| gainesville | 125 | 112 | 12 | 0 / 1 | 1.83 m (1.12–2.48) |
| sao_paulo | 125 | 101 | 24 | 0 | 2.25 m (1.22–2.48) |

The excluded panoramas skew old (bend's stand-ins are 2012–2018 imagery plus 11 from 2024) and
their GT recall on the flat axis is lower than the included panoramas' on bend (0.699 vs 0.779,
73 vs 254 points) and higher on the other three; the depth-axis tables are therefore a subset,
not the whole split, and the `all` row of each table says which.

**What the camera height is.** It tracks the rig, not the city: bend's 2024 imagery reads
2.37 m (83 panos), paterson's pre-2025 imagery 2.34–2.46 m, and Google's **2025–26 rig 1.86 m
(paterson, 52 panos) and 1.79 m (gainesville, 87 panos)**. The labeler's own study
(`docs/camera-height-study.md` on its `camera-height-40` branch) finds by bearing-only
triangulation that the depth frame runs **6–16% short** of the height the imagery implies, by
city (bend ~1.06, paterson ~1.08, gainesville ~1.10, sao_paulo ~1.16), and cannot yet say
whether that is a scale or an offset. Both readings are reported: the raw depth axis, and the
depth axis multiplied by that per-city factor ("depth × scale").

### 0.1 How stretched the flat axis is, and what the thresholds become

Per-point median of flat(2.5 m) / depth over the measured-ground GT points; the ratio of
medians is beside it because the issue tabulated that form, and the two differ:

| population | n | median flat / median depth | per-point median ratio (p10–p90) | 18 m / 25 m become | depth × scale: ratio, thresholds |
|---|---:|---:|---|---|---|
| **bend** (this document's GSV city) | 254 | 14.41 / 12.08 = 1.19 | **1.079** (1.00–1.28) | **16.7 m / 23.2 m** | 1.018, 17.7 m / 24.6 m |
| paterson | 360 | 14.95 / 13.41 = 1.12 | 1.191 (1.00–1.55) | 15.1 m / 21.0 m | 1.102, 16.3 m / 22.7 m |
| gainesville | 249 | 14.95 / 11.08 = 1.35 | **1.393** (1.07–1.93) | **12.9 m / 17.9 m** | 1.272, 14.1 m / 19.6 m |
| sao_paulo | 237 | 11.46 / 10.65 = 1.08 | 1.083 (1.00–1.33) | 16.6 m / 23.1 m | 0.934, 19.3 m / 26.8 m |
| GSV pooled | 1,100 | 14.41 / 11.74 = 1.23 | 1.145 (1.00–1.58) | 15.7 m / 21.8 m | 1.050, 17.1 m / 23.8 m |

The issue's own check reproduces on operational detections at the labeler's 2.6 m: the ratio of
medians is 1.30 on paterson (issue: 1.29) and 1.46 on gainesville (issue: 1.35; the per-detection
median is 1.42), and correcting only the camera height leaves 1.000–1.011 (issue: 1.02–1.03) —
the cotangent form is right and only the constant was wrong, as stated. What the issue's
extrapolation got wrong is the population: paterson and gainesville are half-to-mostly the
2025–26 rig, and bend is not. **On this document's own GSV city the axis is 8% long, not
25–30%**, and after the labeler's depth-frame correction the printed thresholds are within 2% of
the measured ones. The 25–35% stretch is real and matters for any split — or deployment — on the
new rig.

**Cross-check against a depth-free measurement.** The labeler measures the same range scale
without any depth: a leave-one-view-out reprojection residual over every multi-view site
([sidewalk-auto-labeler PR #76](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/76),
its `docs/reprojection-residual.md` at commit `e221b03` on branch `reprojection-residual-36`)
gives a corrected flat-ground scale at 2.6 m of **k = 1.19 for paterson's 2025 rig and 1.18 for
gainesville's 2026 rig**, against 1.01–1.07 for 2019–24 vintages; under per-pano depth heights
the same fit reads 0.99 on the new rig and 0.94–1.00 on older imagery, i.e. depth ranges 1–4%
short — the same direction as the camera-height study's "depth frame runs short", at a smaller
magnitude. Put on the same footing (flat at 2.6 m over depth, per capture year, GT points), this
document's rows agree with that on the older vintages and not on the new rig: bend 2024 1.12
(n 236) vs k 1.06, paterson 2019–24 1.07–1.14 vs 1.04–1.06, sao_paulo 2022–25 1.07–1.13 vs
1.01–1.02, all within a few percent and on the side a 1–4% short depth range predicts; but
**paterson 2025 reads 1.45 (n 166) and gainesville 2026 1.51 (n 200) here against 1.19 and 1.18
there.** Two independent methods agree that the new rig is the outlier and disagree on its size
by a factor the older vintages do not show. That gap is an open question, not resolved here;
the candidates are the pano population (benchmark panos vs every multi-view site), the
stand-in-ground exclusion, and the 2026 rig's depth frame. The same labeler document reports
that about half of #101's 0.07–0.13 along-ray slope is regression bias (a naive along-ray-vs-range
fit returns 0.047–0.054 on simulated data with no scale error) — reported there, not re-derived
here ([#101](https://github.com/ProjectSidewalk/RampNet/issues/101)).

### 0.2 Recall by distance, flat vs depth

bend, measured-ground panoramas (254 of the 327 GT points in §1's bend half):

| distance | n (flat 2.5 m) | recall (flat) | n (depth) | recall (depth) | n (depth × scale) | recall |
|---|---:|---:|---:|---:|---:|---:|
| 0–8 m | 59 | 0.898 | 67 | 0.895 | 60 | 0.883 |
| 8–12 m | 62 | 0.903 | 60 | 0.883 | 62 | 0.919 |
| 12–18 m | 60 | 0.817 | 77 | 0.753 | 74 | 0.757 |
| 18–25 m | 56 | 0.625 | 41 | 0.610 | 47 | 0.638 |
| 25–40 m | 14 | 0.357 | 9 | 0.222 | 11 | 0.182 |
| 40 m+ | 3 | 0.000 | – | – | – | – |
| all | 254 | 0.779 | 254 | 0.779 | 254 | 0.779 |

Four GSV splits pooled, measured-ground panoramas (1,100 GT points):

| distance | n (flat 2.5 m) | recall (flat) | n (depth) | recall (depth) |
|---|---:|---:|---:|---:|
| 0–8 m | 214 | 0.836 | 287 | 0.808 |
| 8–12 m | 245 | 0.792 | 273 | 0.762 |
| 12–18 m | 238 | 0.719 | 267 | 0.712 |
| 18–25 m | 234 | 0.705 | 186 | 0.613 |
| 25–40 m | 134 | 0.448 | 84 | 0.333 |
| 40 m+ | 35 | 0.114 | 3 | 0.333 |
| all | 1,100 | 0.703 | 1,100 | 0.703 |

The shape is the same on both axes; what moves is the population under each label. On the
depth axis 130 of the 403 ramps the flat axis put beyond 18 m are inside it (bend 23 of 73,
gainesville 50 of 111), the "far-field" band shrinks, and its recall falls (pooled 18–25 m
0.705 → 0.613, 25–40 m 0.448 → 0.333): the far ramps that remain far are missed more often than
the flat axis made it look, because the flat axis had been diluting that band with nearer ramps.
The collapse is at **18–20 m on the new rig and ~23 m on the 2.4 m rig**, not at 25 m.

### 0.3 Apparent size and the resolution forecast

bend, measured-ground (recall by apparent size of a 1.2 m ramp):

| apparent size | n (flat) | recall (flat) | n (depth) | recall (depth) |
|---|---:|---:|---:|---:|
| 12–20 px | 3 | 0.000 | – | – |
| 20–32 px | 15 | 0.333 | 10 | 0.200 |
| 32–50 px | 59 | 0.593 | 57 | 0.579 |
| 50–80 px | 90 | 0.856 | 93 | 0.828 |
| 80 px+ | 87 | 0.931 | 94 | 0.915 |

Four GSV splits pooled:

| apparent size | n (flat) | recall (flat) | n (depth) | recall (depth) |
|---|---:|---:|---:|---:|
| 0–12 px | 6 | 0.333 | – | – |
| 12–20 px | 31 | 0.065 | 4 | 0.250 |
| 20–32 px | 138 | 0.435 | 88 | 0.330 |
| 32–50 px | 248 | 0.665 | 257 | 0.591 |
| 50–80 px | 361 | 0.778 | 340 | 0.768 |
| 80 px+ | 316 | 0.832 | 411 | 0.803 |

The issue's second point holds — the missed ramps are **larger in pixels** than the flat axis
said (pooled, the sub-32 px population drops from 175 to 92 points) — but recall *within* each
size band is lower, not higher, so the diagnosis "not enough signal in the pixels" does not
weaken; it just has fewer ramps in its smallest band. The §4 forecast, re-run by the same method
on each axis:

| factor | bend, flat | bend, depth | bend, depth × scale | pooled, flat | pooled, depth |
|---|---|---|---|---|---|
| 1.5× | +0.096 | +0.096 | +0.090 | +0.072 | +0.069 |
| 2× | +0.117 | **+0.120** | +0.111 | +0.093 | **+0.089** |
| 3× | +0.145 | +0.134 | +0.120 | +0.119 | +0.099 |

**The 2× forecast is unchanged** (+0.10–0.12 on bend, +0.09 pooled). What the depth axis takes
away is the tail: at 3× the pooled gain drops from +0.119 to +0.099, because the ramps the flat
axis said were tiny and far are not, so doubling their size buys less than the curve suggested.
The retraining-resolution argument in §4 survives with a lower ceiling; the ranking of the three
levers does not change.

### 0.4 Precision by distance is still flat

Pooled over the four GSV splits on the depth axis (TP + FP, measured-ground panoramas):
0–8 m 0.955 (n 243), 8–12 m 0.937 (222), 12–18 m 0.941 (204), 18–25 m 0.950 (121), 25–40 m 0.963
(27). §2's conclusion — do not cull by distance — is unchanged on either axis.

### 0.5 Not done, and caveats that travel with these numbers

- **The payloads are an unpublished input.** They are Google-derived and mirrored from the
  labeler's archive (`makelab2:/projects/makeabilitylab/sidewalk-auto-labeler/runs/<city>/depth/`,
  local mirror `D:\Git\sidewalk-auto-labeler\runs\<city>\depth\`); every file's sha256 and
  each `index.csv`'s sha256 are in `analysis_out/recall_by_depth_112.json`, and publication is
  pending Jon's decision. The per-point rows are committed so every table here re-derives
  without them; only *new* points need the archive.
- **The parser revision is a branch, not main.** `depth.py` at labeler `86bb909`
  (`camera-height-40`, its PR #68, unmerged at the time of writing). Its range functions are
  identical to `origin/main` (`c4bebf1`); the branch adds only the stand-in-ground
  classification (`SYNTHETIC_GROUND`) the exclusion rule needs.
- **The depth frame's absolute scale is open.** The labeler's triangulation says the depth
  heights are 6–16% short; the "depth × scale" columns apply that as a multiplicative factor
  per city, which is one of the two hypotheses its study cannot yet separate. Read the raw
  depth column as a lower bound on range and the scaled one as the current best estimate.
- **Richmond and laurens_gsv keep the flat axis.** Mapillary serves no depth (richmond,
  and everything in the Mapillary tier); laurens_gsv was not harvested. Mapillary is exactly
  where the flat axis is worst (§ *Why this needed depth*, ρ 0.81), and this measurement says
  nothing about it.
- **DA3 was not regressed against GSV depth.** `gt_depth_da3.json` is not committed and needs a
  GPU to regenerate, so the issue's "calibrate DA3 and carry it to Mapillary" item is untouched.
  The 6.5–8.5% DA3/flat agreement on bend is consistent with both sharing bend's ~8% bias, but
  that is an inference, not a measurement.
- **Occlusion was not partitioned.** The plane index does identify a non-ground surface under
  39 GT points (`depth_source = fallback_wall` in the rows), which is the raw material for the
  issue's occluded-vs-under-resolved split, but no such table was produced.
- **One seed, one operating point (0.55), verdict-review GT** — the same limits as every other
  table in this document.

Other documents that quote the published metre labels, left as they are and pointing here:
`curb_ramp_data_sourcing.md` §0a (the 18 m far/near boundary at 0.30 over seven splits, five of
them Mapillary), `operating_point.md` (the near/mid/far bands, already stated as a rank
statement), `crop_window_eval.md` (flat-ground strata at 2.5 m), `rampnet1_findings.md` and
`rampnet1_report.md` §6.6 (the recall-by-distance row; its stretch figure is corrected).

## 1. Recall collapses with distance

| distance | n | recall |
|---|---|---|
| 0–8 m | 133 | 0.842 |
| 8–12 m | 173 | 0.879 |
| 12–18 m | 197 | 0.812 |
| 18–25 m | 101 | **0.564** |
| 25–40 m | 33 | **0.182** |
| **all** | 637 | 0.765 |

By apparent size: 20–32 px → 0.189, 32–50 px → 0.671, 50–80 px → 0.825, 80 px+ → 0.876. There is
simply not enough signal left in the pixels.

> **Recommendation:** report benchmark recall **stratified by distance**. "Reliable to 18 m, blind
> past 25 m" is far more actionable than a scalar 0.765. (On GSV's own depth the labels are
> ~17 m / 23 m for the 2.4 m rig this table's bend half was captured with, and ~13 m / 18 m on
> Google's 2025–26 rig — §0.1.)

## 2. Precision is flat with distance — do not cull

| distance | detections | precision |
|---|---|---|
| 0–8 m | 122 | 0.943 |
| 8–12 m | 132 | 0.970 |
| 12–18 m | 116 | 0.966 |
| 18–25 m | 104 | 0.962 |
| 25 m+ | 32 | **1.000** |

Culling detections beyond 18 m would **lose 132 true ramps to remove 4 false ones**, and would
*lower* precision (0.962 → 0.959). **When RampNet sees a distant ramp it is almost always right; it
just usually doesn't see it.** Far-field is a *sensitivity* problem, not a *reliability* one — which
also means lowering the threshold at range is safe.

## 3. Lever A — the operating point (free)

Inference was re-run on all 234 benchmark panos, byte-faithful to the deployment path
(resize 2048×4096, no TTA); at `(0.55, 10)` it reproduces the committed `records.jsonl` exactly.

| threshold | richmond P / R / F1 | bend P / R / F1 |
|---|---|---|
| **0.55** (deployed) | 0.964 / 0.768 / .855 | 0.980 / 0.755 / .853 |
| 0.35 | 0.921 / 0.823 / **.869** | 0.929 / 0.804 / .862 |
| 0.25 | 0.872 / 0.839 / .855 | 0.904 / 0.835 / **.868** |
| 0.15 | 0.780 / 0.868 / .821 | 0.853 / 0.850 / .851 |

- **+7–10 recall points for one changed constant.** No retraining.
- **0.55 was not even F1-optimal** — F1 peaks at 0.35/0.25, so this is an improvement even under a
  symmetric metric, before invoking any recall-first argument.
- `min_distance` 10 → 3 gains ~0.5 pt and costs precision. **Keep 10.**
- **~44% of "misses" were sub-threshold, not blind.** The model saw them and was under-confident —
  a large share of the recall gap is *calibration*, not vision.

## 4. Lever B — resolution (forecast)

Mapping each size bucket to the recall observed at double its apparent size:

| resolution | forecast recall |
|---|---|
| 1.5× | 0.765 → 0.846 (+0.081) |
| **2×** | 0.765 → **0.867 (+0.103)** |
| 3× | 0.765 → 0.875 (+0.111) |

**+10 points at 2×, saturating ~0.875** — even large, near ramps only reach 0.876, so the residual
~12% is occlusion / motion blur / odd geometry, not size. The detail genuinely exists: Richmond
panos are natively ~11000 px wide and Bend 16384 px, against a 4096 px model input.

> **Caveat:** upscaling adds no information. An honest gain requires the **retrain-at-higher-res**
> arm, not a frozen-model input-size sweep.

## 5. The levers partially overlap

| distance | thr 0.55 | thr 0.25 | thr 0.15 | gain |
|---|---|---|---|---|
| 0–8 m | 0.827 | 0.902 | 0.932 | +0.105 |
| 12–18 m | 0.817 | 0.883 | 0.914 | +0.096 |
| 18–25 m | 0.554 | 0.673 | 0.693 | **+0.139** |
| 25 m+ | 0.152 | 0.303 | 0.364 | **+0.212** |

The threshold helps at *all* distances but most at range, so it competes with resolution for the
same ramps. It does **not** solve far-field (25 m+ tops out at 0.364), and ramps still missed at
0.25 are 54% beyond 18 m. **Do not budget +10 and +10 as +20** — combined, expect roughly **0.90**.

## 6. Lever C — sampling density (unmeasured, possibly largest)

This benchmark measures **per-pano recall** — "did RampNet find this ramp *in this image*." The
deployment product needs **per-ramp recall across the run** — "did we find this ramp *anywhere*."
Those differ enormously, because a ramp invisible at 30 m is at 8 m two panos later.

Using the measured curve and 5 m pano spacing, the three nearest views alone give
`0.158 × 0.158 × 0.121 ≈ 0.3%` chance of missing a ramp in all of them.

> **This is an upper bound and assumes independent failures, which is false.** Occlusion, motion
> blur and unusual geometry persist across neighbouring views — the ramps *no* model found are
> exactly that correlated-failure population, and they set the real floor.

Two consequences: deployment recall is almost certainly far above 0.765 and **has never been
measured**; and sampling density (the auto-labeler's 5 m thinning — a *config*, not a model change)
may be the cheapest lever of all.

## Qualitative check

![Hard misses](figures/recall_hard_misses_richmond.png)

Thirty Richmond ramps missed by **both** RampNet and Gemini-3.1-Pro, sorted by estimated apparent
size; fixed 12° angular crops so distant ramps genuinely look small. These are unmistakably **real
ramps** — many show clear tactile domes — so the ground truth is sound. Alongside small size, the
visible secondary factors are **occlusion** (a parked pickup, hedges, poles), **motion blur** on
consumer-rig Mapillary frames, and pano stitching artifacts.

## Recommended order

1. **Change the operating point** to ~0.25–0.35, keep `min_distance=10`. Free, validated, F1-positive.
2. **Measure per-ramp deployment recall** and revisit sampling density. Cheap; possibly the biggest win.
3. **Higher-resolution retrain.** Forecast +10 pts, saturating ~0.88.
4. Beyond ~0.90: occlusion/blur robustness, or multi-view aggregation.

## Caveats

- Ground truth is anchored to the original review, so a real ramp surfaced by a lower threshold that
  the reviewer never marked scores as a **false positive**. The precision drops in §3 are therefore
  slight *over*-estimates, and low-threshold precision deserves a fresh spot-check before being quoted.
- Apparent size assumes a ~1.2 m ramp width; distance assumes ~2.5 m camera height where geometry is
  used. Both were cross-checked against DA3 metric depth — and, for the GSV half, against GSV's own
  depth in §0: the 2.5 m assumption is ~8% high on bend and ~30% high on the 2025–26 rig.
- Two cities (one GSV/in-distribution, one Mapillary/OOD). Patterns are consistent across both, but
  this is not yet broad geographic evidence.
