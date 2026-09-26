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
> of this population, the flat-ground axis below runs **~6–8% long** near the thresholds, so
> "18 m / 25 m" reads **16.9 m / 23.3 m** there. With the labeler's depth-frame correction it is
> within 2% of what is printed. The larger stretch the issue measured is a property of Google's
> **2025–26 rig** (paterson 2025, gainesville 2026 capture), whose camera sits at ~1.8–1.9 m
> instead of ~2.4 m. That rig's axis is ~1.4× long, and the same thresholds are
> **13.1 m / 18.7 m** on it. Richmond is Mapillary, which serves no depth, so its half of every
> table stays on the flat-ground axis. Every table in this document is kept as published; the
> depth-axis versions are in §0. (§0 was corrected on 2026-09-24: its first version read the
> depth payload azimuth-mirrored; see §0 and [#112](https://github.com/ProjectSidewalk/RampNet/issues/112).)

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

`scripts/analysis/recall_by_depth_112.py` re-derives the axis. It parses the payloads and
classifies the ground plane with the labeler's own `depth.py` (labeler commit `86bb909`, branch
`camera-height-40`; `origin/main` was `c4bebf1` and lacks only the stand-in-ground
classification used here), and does the per-point plane lookup and ray itself, for the reason
in the next paragraph. It re-issues this document's tables on the flat axis and the depth axis
side by side, per split and pooled over the four GSV splits. The committed
`analysis_out/recall_by_depth_112.json` carries every GT point and detection with both
distances, and every table re-derives from those rows on CPU (`recall_by_depth_112.py --check`).
`tests/test_recall_by_depth_112.py` pins those tables and checks that every table in this
section is the committed one, verbatim (`--check --doc-tables` prints them). Nothing here needs
the payloads to check. The full tables, per split, are in `analysis_out/recall_by_depth_112.md`.

**Which way round the payload is (checked against the imagery, not assumed).** The payload
stores its plane index in *raw* column order. The labeler's `depth.py` maps an image column `c`
to raw column `511 − c`. That is right for streetlevel's rastered depth map, which is mirrored,
but **it is mirrored against the RampNet benchmark JPEGs**, which are the frame the GT points and
detections live in. The first version of this section (PR #184 before review) used the labeler's
lookup and so read every point's depth at the azimuth-mirrored position. The mapping used now is
**image column `c` = raw column `c`**, with the labeler's raw-column ray
(`phi = (1 − x)·2π + π/2`). `scripts/analysis/depth_image_alignment_112.py` measures it four
independent ways, and commits the result to `analysis_out/depth_image_alignment_112.json`:

| check | image column = raw column | image column = raw 511 − c (the labeler's lookup) |
|---|---|---|
| A. Sky: best of both hypotheses × 512 column shifts, correlating the payload's sky mask with an image sky score (404 panos with sky) | **189** peak within ±2 columns of zero shift; 333 win at zero shift | 1 |
| B. The plane under each GT point is ground-like (1,101 points, measured-ground panos) | **1,096** (99.5%) | 1,061 (96.4%) |
| B. The same for true-positive detections (774) | **770** (99.5%) | 745 (96.3%) |
| C. Plane-index boundaries vs vertical image edges near the horizon (483 panos) | **348** win at zero shift; 40 best within ±2 | 0 best within ±2 |
| D. Raw-space ray formula: where ground meets a wall across a column boundary, the two planes give the same range (47,019 boundaries, all 485 payloads; no image involved) | **median \|log ratio\| 0.023** (labeler's `_direction`) | 0.424 (mirrored azimuth) |

A–C decide the image↔raw column mapping; D confirms the labeler's ray formula in raw space. The
composition is the lookup the script uses. The independent reviewer's own checks
([PR #184 review](https://github.com/ProjectSidewalk/RampNet/pull/184#pullrequestreview-5305326643)) agreed (sky vs
brightness 189 vs 26; ground hit 1,096 vs 1,061; seam 0.031 vs 0.344 on paterson). Sky and edges
peak off zero for the remaining panos, but not near the mirrored mapping's zero either: only
1 (sky) and 0 (edge) panos put the mirrored peak within ±2 columns. Over all shifts the mirrored
mapping scores best on 94 of 404 sky panos and 202 of 483 edge panos, at large shifts, which reads
as noise rather than support for the mirror. Why those panos peak off zero was not measured.

**The labeler: checked there, not here.** The labeler's own `ground_range_at` uses the same
stored ↔ raw convention as the lookup this section stopped using. Whether the labeler's imagery
shares these JPEGs' orientation or streetlevel's raster is a question for the labeler, and
nothing in this document tests it. It was checked in the labeler on 2026-09-25 and confirmed:
the labeler's `ground_range_at` is mirrored for image-frame coordinates, and production outputs
are unaffected today. Filed as
[sidewalk-auto-labeler#80](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/80).

**The rule.** Depth distance = horizontal range along the exact ray through the point to the
payload plane under its pixel (the plane index is per pixel of a 512×256 grid; the intersection
is continuous, never snapped to a pixel row). If that plane is not ground-like (tilt > 18°) the
point falls back to level ground at the measured camera height (the ground plane's tilt is not
applied there), and the row says so. With the aligned lookup that happens for **5 of 1,101 GT
points** (4 `fallback_wall`, 1 above the horizon; the mirrored lookup had put 39 there). Apparent
size uses the Euclidean ray distance. Ground truth and hits are exactly this document's (verdict
review, deployed 0.55, the same greedy matcher). The richmond + bend population reproduces at
**637 / 0.765** on the way in.

**Which panoramas count.** 14% of GSV payloads carry Google's stand-in ground — an exactly
level plane at exactly 2.500 m, a default rather than a measurement — and a few are 2-plane
fallbacks or implausible. Those panoramas are **excluded from the depth axis**, never backfilled,
and their GT points stay on the flat axis only:

| split | panos | measured | stand-in ground | degenerate / implausible | camera height median (min–max), measured |
|---|---:|---:|---:|---:|---|
| bend | 110 | 90 | 20 | 0 / 0 | 2.37 m (1.66–2.49) |
| paterson | 125 | 113 | 10 | 2 / 0 | 2.06 m (1.46–2.50) |
| gainesville | 125 | 112 | 12 | 0 / 1 | 1.83 m (1.12–2.48) |
| sao_paulo | 125 | 101 | 24 | 0 / 0 | 2.25 m (1.22–2.48) |

The excluded panoramas skew old (bend's stand-ins are 2012–2018 imagery plus 11 from 2024).
Their GT recall on the flat axis is lower than the included panoramas' on bend (0.699 vs 0.779,
73 vs 254 points) and higher on the other three. The depth-axis tables are therefore a subset,
not the whole split, and the `all` row of each table says which.

**What the camera height is.** It tracks the rig, not the city. bend's 2024 imagery reads
2.37 m (74 panos with GT points), paterson's pre-2025 imagery 2.34–2.46 m, and Google's
**2025–26 rig 1.86 m (paterson 2025) and 1.80 m (gainesville 2026)**. The labeler's own study
(`docs/camera-height-study.md` on its `camera-height-40` branch) uses bearing-only
triangulation. It finds that the depth frame runs **6–16% short** of the height the imagery
implies, by city (bend ~1.06, paterson ~1.08, gainesville ~1.10, sao_paulo ~1.16), and cannot
yet say whether that is a scale or an offset. Both readings are reported: the raw depth axis,
and the depth axis multiplied by that per-city factor ("depth × scale").

### 0.1 How stretched the flat axis is, and what the thresholds become

The ratio flat(2.5 m) / depth grows with distance, so the stretch at the *median point* is not
the stretch at 18 m or 25 m. Each threshold is therefore deflated by the median ratio of the
points whose flat distance lies within ±20% of it: 14.4–21.6 m for 18 m, 20–30 m for 25 m. The
window's n is in brackets. The per-point median over all points, and the ratio of medians the
issue tabulated, are beside it:

| population | n | median flat / median depth | median-point ratio (p10–p90) | 18 m becomes (window n) | 25 m becomes (window n) | depth × scale: 18 m / 25 m become |
|---|---:|---:|---|---|---|---|
| **bend** (this document's GSV city) | 254 | 14.41 / 12.33 = 1.17 | 1.062 (1.00–1.23) | 16.9 m (107) | 23.3 m (49) | 17.9 m / 24.6 m |
| paterson | 360 | 14.95 / 13.70 = 1.09 | 1.166 (1.00–1.56) | 15.9 m (127) | 21.7 m (90) | 17.2 m / 23.4 m |
| gainesville | 249 | 14.95 / 12.08 = 1.24 | 1.372 (1.05–1.93) | 13.8 m (103) | 19.7 m (66) | 15.2 m / 21.6 m |
| sao_paulo | 237 | 11.46 / 11.13 = 1.03 | 1.083 (1.00–1.31) | 16.9 m (77) | 22.8 m (48) | 19.6 m / 26.4 m |
| GSV pooled | 1,100 | 14.41 / 12.24 = 1.18 | 1.133 (1.00–1.61) | 16.2 m (414) | 21.7 m (253) | 17.4 m / 23.5 m |

**By capture year, which is the rig, and not by split.** A split mixes vintages, and paterson is
under half new-rig imagery (166 of its 360 points). Rows are (split, capture year) with at least
20 measured-ground GT points, then the 2025–26 rig pooled and the older US vintages pooled:

| capture vintage | GT points (panos) | camera height, median | flat 2.5 m / depth, median point (p10–p90) | flat 2.6 m / depth | 18 m becomes (window n) | 25 m becomes (window n) |
|---|---:|---:|---|---:|---|---|
| bend 2024 | 236 (74) | 2.37 m | 1.062 (1.00–1.21) | 1.10 | 16.9 m (100) | 23.4 m (45) |
| paterson 2019 | 20 (5) | 2.46 m | 1.044 (1.00–1.18) | 1.09 | – | – |
| paterson 2020 | 31 (9) | 2.37 m | 1.102 (1.02–1.28) | 1.15 | – | – |
| paterson 2021 | 69 (11) | 2.34 m | 1.066 (1.00–1.27) | 1.11 | 18.0 m (24) | 23.5 m (14) |
| paterson 2024 | 66 (19) | 2.35 m | 1.069 (1.00–1.19) | 1.11 | 16.8 m (28) | 22.8 m (14) |
| paterson 2025 | 166 (44) | 1.86 m | 1.374 (1.15–1.69) | 1.43 | 13.6 m (55) | 18.7 m (44) |
| gainesville 2024 | 32 (10) | 2.22 m | 1.050 (1.00–1.19) | 1.09 | 17.8 m (13) | – |
| gainesville 2026 | 200 (68) | 1.80 m | 1.441 (1.19–1.97) | 1.50 | 13.0 m (84) | 18.7 m (53) |
| sao_paulo 2023 | 20 (5) | 2.20 m | 1.033 (1.00–1.30) | 1.07 | – | – |
| sao_paulo 2024 | 96 (27) | 2.31 m | 1.071 (1.00–1.21) | 1.11 | 16.6 m (32) | 22.8 m (19) |
| sao_paulo 2025 | 91 (23) | 2.25 m | 1.105 (1.00–1.32) | 1.15 | 17.7 m (34) | 23.5 m (18) |
| 2025-26 rig (paterson 2025 + gainesville 2026) | 366 (112) | 1.83 m | 1.399 (1.16–1.90) | 1.45 | 13.1 m (139) | 18.7 m (97) |
| older US vintages (bend, paterson, gainesville; the rest) | 497 (144) | 2.36 m | 1.063 (1.00–1.24) | 1.11 | 17.1 m (198) | 23.2 m (108) |

The issue's own check reproduces on operational detections at the labeler's 2.6 m. The ratio of
medians is 1.26 on paterson (issue: 1.29) and 1.38 on gainesville (issue: 1.35; the
per-detection median is 1.40). Correcting only the camera height leaves 0.998–1.003 (issue:
1.02–1.03), so the cotangent form is right and only the constant was wrong, as stated. What the
issue's extrapolation got wrong is the population. **On the 2025–26 rig the flat axis is ~1.4×
long** (40% at the median point, 37% and 34% at the two thresholds), so 18 m / 25 m read
**13.1 m / 18.7 m**. **On the older US vintages pooled it is ~1.05–1.08× long**, so they read **17.1 m / 23.2 m**
(per vintage 16.8–18.0 m / 22.8–23.5 m, on windows of 13–100 points). That includes bend, this
document's GSV city, at 16.9 m / 23.3 m.
With the labeler's depth-frame correction, bend's thresholds are 17.9 m / 24.6 m, within 2% of
what is printed at both. Without it they are 6–7% short.

**Cross-check against a depth-free measurement.** The labeler measures the same range scale
without any depth: a leave-one-view-out reprojection residual over every multi-view site
([sidewalk-auto-labeler PR #76](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/76),
its `docs/reprojection-residual.md` at commit `e221b03` on branch `reprojection-residual-36`).
That gives a corrected flat-ground scale at 2.6 m of **k = 1.19 for paterson's 2025 rig and 1.18
for gainesville's 2026 rig**, against 1.01–1.07 for 2019–24 vintages. Under per-pano depth
heights the same fit reads 0.99 on the new rig and 0.94–1.00 on older imagery, i.e. depth ranges
1–4% short. That is the same direction as the camera-height study's "depth frame runs short", at
a smaller magnitude. The "flat 2.6 m / depth" column above puts this document's rows on the same
footing. They agree with it on the older vintages and not on the new rig. Bend 2024 reads 1.10
vs k 1.06, paterson 2019–24 1.09–1.15 vs 1.04–1.06, and sao_paulo 2023–25 1.07–1.15 vs
1.01–1.02. All are higher than k by 4–13%, the side a 1–4% short depth range predicts, if by
more than that. But
**paterson 2025 reads 1.43 (n 166) and gainesville 2026 1.50 (n 200) here against 1.19 and 1.18
there.** Two independent methods agree that the new rig is the outlier, and disagree on its size
by a factor the older vintages do not show. That gap is an open question, not resolved here. The
candidates are the pano population (benchmark panos vs every multi-view site), the
stand-in-ground exclusion, and the 2026 rig's depth frame. The same labeler document reports that
about half of #101's 0.07–0.13 along-ray slope is regression bias (a naive along-ray-vs-range
fit returns 0.047–0.054 on simulated data with no scale error). That is reported there, not
re-derived here ([#101](https://github.com/ProjectSidewalk/RampNet/issues/101)).

### 0.2 Recall by distance, flat vs depth

bend, measured-ground panoramas (254 of the 327 GT points in §1's bend half):

| distance | n (flat 2.5 m) | recall (flat) | n (depth) | recall (depth) | n (depth × scale) | recall |
|---|---:|---:|---:|---:|---:|---:|
| 0–8 m | 59 | 0.898 | 61 | 0.885 | 58 | 0.897 |
| 8–12 m | 62 | 0.903 | 64 | 0.922 | 62 | 0.903 |
| 12–18 m | 60 | 0.817 | 72 | 0.778 | 71 | 0.803 |
| 18–25 m | 56 | 0.625 | 46 | 0.565 | 50 | 0.600 |
| 25–40 m | 14 | 0.357 | 11 | 0.273 | 13 | 0.231 |
| 40 m+ | 3 | 0.000 | – | – | – | – |
| all | 254 | 0.779 | 254 | 0.779 | 254 | 0.779 |

Four GSV splits pooled, measured-ground panoramas (1,100 GT points):

| distance | n (flat 2.5 m) | recall (flat) | n (depth) | recall (depth) |
|---|---:|---:|---:|---:|
| 0–8 m | 214 | 0.836 | 281 | 0.804 |
| 8–12 m | 245 | 0.792 | 254 | 0.791 |
| 12–18 m | 238 | 0.719 | 286 | 0.717 |
| 18–25 m | 234 | 0.705 | 194 | 0.598 |
| 25–40 m | 134 | 0.448 | 83 | 0.301 |
| 40 m+ | 35 | 0.114 | 2 | 0.000 |
| all | 1,100 | 0.703 | 1,100 | 0.703 |

The shape is the same on both axes; what moves is the population under each label. On the depth
axis, 124 of the 403 ramps the flat axis put at or beyond 18 m are inside it (bend 16 of 73,
gainesville 51 of 111), and the "far-field" band shrinks. Its recall falls: pooled 18–25 m goes
from 0.705 to 0.598, and 25–40 m from 0.448 to 0.301. The far ramps that remain far are missed
more often than the flat axis made it look, because the flat axis had been diluting that band
with nearer ramps. Read through §0.1's thresholds, the published "reliable to 18 m, blind past
25 m" is **~13 m / ~19 m on the 2025–26 rig and ~17 m / ~23 m on the older 2.3–2.4 m rigs**.

### 0.3 Apparent size and the resolution forecast

bend, measured-ground (recall by apparent size of a 1.2 m ramp):

| apparent size | n (flat 2.5 m) | recall (flat) | n (depth) | recall (depth) |
|---|---:|---:|---:|---:|
| 12–20 px | 3 | 0.000 | – | – |
| 20–32 px | 15 | 0.333 | 11 | 0.273 |
| 32–50 px | 59 | 0.593 | 60 | 0.600 |
| 50–80 px | 90 | 0.856 | 87 | 0.828 |
| 80 px+ | 87 | 0.931 | 96 | 0.906 |
| all | 254 | 0.779 | 254 | 0.779 |

Four GSV splits pooled:

| apparent size | n (flat 2.5 m) | recall (flat) | n (depth) | recall (depth) |
|---|---:|---:|---:|---:|
| 0–12 px | 6 | 0.333 | – | – |
| 12–20 px | 31 | 0.065 | 2 | 0.000 |
| 20–32 px | 138 | 0.435 | 89 | 0.292 |
| 32–50 px | 248 | 0.665 | 266 | 0.609 |
| 50–80 px | 361 | 0.778 | 338 | 0.772 |
| 80 px+ | 316 | 0.832 | 405 | 0.800 |
| all | 1,100 | 0.703 | 1,100 | 0.703 |

The issue's second point holds: the missed ramps are **larger in pixels** than the flat axis
said. Pooled, the sub-32 px population drops from 175 to 91 points. But recall *within* each
size band is mostly lower, not higher, so the diagnosis "not enough signal in the pixels" does
not weaken; it just has fewer ramps in its smallest band. The §4 forecast (the gain in recall),
re-run by the same method on each axis:

| factor | bend, flat | bend, depth | bend, depth × scale | pooled, flat | pooled, depth |
|---|---|---|---|---|---|
| 1.5× | +0.096 | +0.087 | +0.092 | +0.072 | +0.071 |
| 2× | +0.117 | +0.111 | +0.116 | +0.092 | +0.088 |
| 3× | +0.145 | +0.125 | +0.126 | +0.119 | +0.096 |

**The 2× forecast barely moves** (+0.11–0.12 on bend, +0.09 pooled). What the depth axis takes
away is the tail: at 3× the pooled gain drops from +0.119 to +0.096, because the ramps the flat
axis said were tiny and far are not, so tripling their size buys less than the curve suggested.
The retraining-resolution argument in §4 survives with a lower ceiling; the ranking of the three
levers does not change.

### 0.4 Precision by distance is still flat

Pooled over the four GSV splits on the depth axis (TP + FP, measured-ground panoramas):

| distance (depth) | detections (TP + FP) | precision |
|---|---:|---:|
| 0–8 m | 238 | 0.950 |
| 8–12 m | 216 | 0.935 |
| 12–18 m | 217 | 0.945 |
| 18–25 m | 123 | 0.951 |
| 25–40 m | 23 | 1.000 |
| all | 817 | 0.946 |

§2's conclusion (do not cull by distance) is unchanged on either axis.

### 0.5 Not done, and caveats that travel with these numbers

- **The payloads are an unpublished input.** They are Google-derived and mirrored from the
  labeler's archive (`makelab2:/projects/makeabilitylab/sidewalk-auto-labeler/runs/<city>/depth/`,
  local mirror `D:\Git\sidewalk-auto-labeler\runs\<city>\depth\`). Every file's sha256 and
  each `index.csv`'s sha256 are in `analysis_out/recall_by_depth_112.json`, and publication is
  pending Jon's decision. The per-point rows are committed so every table here re-derives
  without them; only *new* points, and the alignment check, need the archive.
- **The depth axis covers only measured-ground panoramas.** bend 254 of 327 GT points, paterson
  360 of 395, gainesville 249 of 272, sao_paulo 237 of 280. On the flat axis, the excluded points'
  recall differs from the included ones' by up to ~10 points (bend 0.699 vs 0.779; the other
  three are higher, 0.767–0.783 vs 0.671–0.691).
- **The image↔payload mapping is this document's, not the labeler's.** It is measured above
  against the benchmark JPEGs. `tests/test_recall_by_depth_112.py` pins it with a tilted-plane
  known answer that runs in CI without payloads. It has not been checked against any other
  imagery.
- **The parser revision is a branch, not main.** `depth.py` at labeler `86bb909`
  (`camera-height-40`, its PR #68, unmerged at the time of writing). Only its parser, ground-plane
  pick and stand-in classification are used here.
- **The depth frame's absolute scale is open.** The labeler's triangulation says the depth
  heights are 6–16% short. The "depth × scale" columns apply that as a multiplicative factor per
  city, which is one of the two hypotheses its study cannot yet separate. Read the raw depth
  column as a lower bound on range and the scaled one as the current best estimate.
- **Richmond and laurens_gsv keep the flat axis.** Mapillary serves no depth (richmond,
  and everything in the Mapillary tier), and laurens_gsv was not harvested. Mapillary is exactly
  where the flat axis is worst (§ *Why this needed depth*, ρ 0.81), and this measurement says
  nothing about it.
- **DA3 was not regressed against GSV depth.** `gt_depth_da3.json` is not committed and needs a
  GPU to regenerate, so the issue's "calibrate DA3 and carry it to Mapillary" item is untouched.
  The 6.5–8.5% DA3/flat agreement on bend is consistent with both sharing bend's ~6–7% bias, but
  that is an inference, not a measurement.
- **Occlusion was not partitioned, and the depth payload is not the instrument for it.** With the
  aligned lookup only 5 of 1,101 GT points sit under a non-ground plane. The 39 the first version
  reported were almost all the azimuth mirror, so the payload does not supply raw material for
  the issue's occluded-vs-under-resolved split.
- **One seed, one operating point (0.55), verdict-review GT**: the same limits as every other
  table in this document.

Other documents that quote the published metre labels, left as they are and pointing here:
`data_scaling_59.md` §0a (the 18 m far/near boundary at 0.30 over seven splits, five of
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

**Where these numbers come from, and why §5 differs (#171).** A hit here is a committed
deployment detection (`benchmark/{richmond,bend}/records.jsonl`, peak threshold 0.55) matched
to a GT ramp: 487 of 637 (richmond 238 of 310, bend 249 of 327). That total re-derives from the
repo: it is `published_reproduction` in `analysis_out/recall_by_depth_112.json`. The distance
bands are Depth Anything 3 metric depth, binned by `scripts/analysis/depth_analysis.py` from
`gt_depth_da3.json`. That file is not committed, and it was not found on this workstation or in
the makelab2 home directory on 2026-09-25, so the per-band n and recall in this table **cannot
be re-derived from the repo**. §5's "thr 0.55" column uses the same bands but a different
detection source (a re-run of inference), and this table is the one to quote for recall at the
deployed operating point. Neither table is newer: both were added in the same commit
(`846378c`, #37).

By apparent size: 20–32 px → 0.189, 32–50 px → 0.671, 50–80 px → 0.825, 80 px+ → 0.876. There is
simply not enough signal left in the pixels.

> **Recommendation:** report benchmark recall **stratified by distance**. "Reliable to 18 m, blind
> past 25 m" is far more actionable than a scalar 0.765. (On GSV's own depth the labels are
> ~17 m / 23 m for the 2.4 m rig this table's bend half was captured with, and ~13 m / 19 m on
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

Inference was re-run on all 234 benchmark panos (resize 2048×4096, no TTA). At `(0.55, 10)` it
reproduces the committed `records.jsonl` recall on richmond exactly: the same 238 of 310 GT
ramps are hit. On bend it does not. The re-run hits 247 of 327 and the committed records 249,
and 10 GT ramps change state (6 lost, 4 gained). Bend is the one GSV split, and its production
path fed the model a 4096×2048 intermediate rather than the native-res bundle pano
(`scripts/analysis/README.md`, `low_floor_sweep.py parity`). So bend's recall at 0.55 below
(0.755) is 0.006 under the committed 0.761, and the pooled re-run recall at 0.55 is 485 / 637 =
0.761, against §1's 0.765. An earlier version of this paragraph said the re-run reproduced the
records exactly; that was true of richmond only.

The recall in every row below re-derives from `analysis_out/overlap.json`, the per-GT-ramp hits
at all four thresholds written by `scripts/analysis/overlap_test.py` (same inference and peak
extraction as `threshold_sweep.py`). It was committed on 2026-09-25 as a byte copy of the
gitignored `analysis_out/overlap.json` in the main checkout on Jon's Windows workstation
(`jonfhome`), sha256
`d0d8b4b0449d9f2f8008e9ce22770fd6d8620c4f419c3e43c92199d63a88dace`, which
`tests/test_detection_recall_provenance.py` pins. When it was run rests only on that local
file's mtime, 2026-07-26 18:33 −0700; no log of the run was kept. Precision and F1 do not
re-derive: that file holds hits per GT ramp, not false positives, and `threshold_sweep.py`'s
own output was not kept.

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

**The 0.55 column here is not §1's table (#171).** It comes from the §3 re-run
(`overlap_test.py`, `analysis_out/overlap.json`), not from the committed records §1 reads, so
it differs from §1 exactly where bend's re-run differs: 10 ramps, net −2 (485 against 487 hits).
The bands are consistent with §1's DA3 bands and n: that is inferred from the arithmetic
below, not read from code, because the code that made this table is not in the repo (see the
end of this note). Read that way, 25 m+ is §1's 25–40 m row, since §1's rows sum to 637 and so
hold no ramp beyond 40 m, and the 8–12 m row is not shown. Per band, the hit counts
implied by n and the printed recall differ by one or two ramps: 0–8 m 110 against 112 of 133,
12–18 m 161 against 160 of 197, 18–25 m 56 against 57 of 101, 25 m+ 5 against 6 of 33 (the
omitted 8–12 m row is +1 by subtraction). The column is kept because the gain column needs all
three thresholds from one inference path. For recall at the deployed operating point, quote §1.

**This table cannot be re-derived from the repo, for two reasons.** The per-ramp hits behind
it can (`analysis_out/overlap.json`), but (1) the DA3 depths, `gt_depth_da3.json`, are not
committed (see §1), and (2) no committed script joins `overlap.json` to depth. Despite its
docstring, `overlap_test.py` stops at writing `overlap.json`; it never reads the depth file.
The only committed readers of `gt_depth_da3.json` are `depth_extract_da3.py`, which writes it,
and `depth_analysis.py`, which bins the committed-records hit, not the re-run. So the 0.35,
0.25 and 0.15 columns, the gain column, and "54% beyond 18 m" below all come from a join that
was run and not kept. Re-deriving them needs both the depth file and that join.

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
  depth in §0: the 2.5 m assumption is ~6–8% high on bend and ~40% high on the 2025–26 rig
  (the flat axis is 1.40× the depth axis there at the median point; 2.5 m / 1.83 m is 1.37).
- Two cities (one GSV/in-distribution, one Mapillary/OOD). Patterns are consistent across both, but
  this is not yet broad geographic evidence.
