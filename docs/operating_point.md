# The deployment operating point: how low should the peak threshold go?

RampNet's deployed operating point is peak extraction at `threshold_abs = 0.55`,
`min_distance = 10`. Until this analysis it had **never been characterised below 0.55** —
the committed benchmark detections stop there, so every published precision/recall number
describes one point on a curve nobody had drawn.

This document draws it, on all seven benchmark splits, and recommends a number. It is
issue [#54](https://github.com/ProjectSidewalk/RampNet/issues/54); the ground-truth
correction it depends on is [#55](https://github.com/ProjectSidewalk/RampNet/issues/55);
the deployment consumer is
[sidewalk-auto-labeler#20](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/20)
and the multi-view consumer is
[labeler#27](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/27) stage 4.

**Headline:** lowering the threshold from 0.55 to **0.30** buys **+7.7 recall points** pooled
(+6 to +10 per split) for −4.8 points of GT-completeness-corrected precision, while detection
density rises only from 1.85 to 2.23 per pano. Recall is RampNet's weak metric everywhere,
and this is the cheapest lever that exists — one constant, no retraining.

## How the numbers were produced

One inference pass per pano, extracting *every* heatmap peak down to a 0.05 floor (keeping
`min_distance = 10`) and carrying each peak's height as its confidence; the threshold is
then swept post-hoc on CPU. So a single GPU run supports every operating point rather than
one run per threshold.

- Extraction: `scripts/analysis/operating_point_curve.py extract`, launched on Hyak by
  `scripts/analysis/run_low_floor_extract.slurm` (one L40S, 41 min for 1,625 panos).
- Analysis: `scripts/analysis/low_floor_sweep.py` (`parity`, `sweep`, `hist`, `gtbias`,
  `distance`, `tagcheck`) — **CPU-only**, reading the cached detections, so every number
  below re-derives without a GPU.
- Preprocessing replicates the deployment path exactly: PIL RGB → resize 2048×4096 bilinear
  → ImageNet normalisation → `peak_local_max`, **no TTA**.

`min_distance` stays at 10 and is not a lever: #25 found 10→3 buys ~0.5 recall points and
costs precision, and `peak_nms_check.py` (#62) showed no suppression radius separates real
adjacent ramp pairs from the rare duplicate.

### The parity gate

Peaks at ≥ 0.55 must reproduce each split's committed `records.jsonl`, or the sweep inherits
a preprocessing divergence. Bit-exactness is the wrong bar, so displacement is measured in
**match radii (R)** — the unit the scorer works in — and the gate asks whether any drift
could change a scoring outcome.

| split | records | cache | identical cell | within 0.5 R | max displacement |
|---|---|---|---|---|---|
| richmond | 267 | 267 | 100.0% | 100.0% | 0.000 R |
| clovis | 164 | 164 | 100.0% | 100.0% | 0.000 R |
| morgantown | 209 | 209 | 100.0% | 100.0% | 0.000 R |
| annapolis | 227 | 227 | 100.0% | 100.0% | 0.000 R |
| budapest_district5 | 189 | 189 | 100.0% | 100.0% | 0.000 R |
| bend | 265 | 257 | 79.0% | 98.4% | 0.439 R |
| manual_gold † | 3610 | 3487 | 80.9% | 99.9% | 0.472 R |

**Every Mapillary split reproduces bit-exactly, including on different hardware** (these were
extracted on an L40S; the committed records were not). Bend is the sole exception and the
reason is structural, not stochastic: bend is the only GSV city split, and the GSV production
path assembled tiles into a **4096×2048 intermediate**, so production fed the model a
different resample of the same pano than this native-res bundle does. The prediction that
follows — that the four then-unextracted Mapillary splits would come back exact and bend
would remain the only one that jitters — was made before those splits ran, and held.

† `manual_gold` is deliberately **not gated**: its committed detections were exported *with*
horizontal-flip TTA at a 0.05 floor (`benchmark/manual_gold/detections_meta.json`), while
this cache is the no-TTA deployment path. Its row is therefore a **TTA-vs-no-TTA delta**, and
a useful free data point for [#78](https://github.com/ProjectSidewalk/RampNet/issues/78):
TTA yields **3.5% more detections** at ≥0.55 (3610 vs 3487) with 99.9% of them co-located
within half a match radius.

**Carry this caveat for bend specifically.** Its ground-truth points derive from the
*production* detections while its predictions here come from the *native-res* resample, so
GT and predictions sit up to 0.44 R apart. That eats into the 1 R matching tolerance and
makes bend's numbers mildly pessimistic relative to the other splits.

## The central bias: sub-0.55 precision is a lower bound, and we can prove it

The benchmark GT for the six city splits was assembled during a review of RampNet's own
detections *at or above 0.55*. So in exactly the band this sweep opens up, a prediction can
only be credited as a true positive if a human independently flagged that ramp during the
missed-ramp pass. **A real curb ramp nobody marked scores as a false positive.**

This is usually stated as a caveat. It can be measured. Decomposing every true positive by
the origin of the GT point it matched (`low_floor_sweep.py gtbias`):

| confidence | TPs from a reviewed detection | TPs from a missed mark |
|---|---|---|
| 0.0–0.5 | **0** | 37 (richmond) / 32 (bend) / 43 (clovis) |
| 0.5–0.6 | 6 / 15 / 8 | 4 / 6 / 1 |
| 0.6–1.0 | 231 / 233 / 131 | 1 / 3 / 0 |

Below the review floor, **every** true positive comes from a missed mark and none from a
reviewed detection, on every split. That is structural, and it has a visible fingerprint in
the calibration curve: pooled P(real) jumps from 0.476 in the 0.45–0.50 bin to 0.821 in
0.50–0.55 — a discontinuity at the review boundary that no property of the model could
produce.

### `manual_gold` as a control — and a tension it exposes

`benchmark/manual_gold` was labelled independently of RampNet (1,000 panos, YOLO box centres,
no verdict review), so it carries **no anchoring at any threshold**. It is the natural control.
The comparison supports the *mechanism* but **not** a large aggregate effect, and that is worth
reporting rather than smoothing over.

What it does show — the discontinuity is specific to the anchored splits:

| bin | city splits (anchored) | manual_gold (un-anchored) |
|---|---|---|
| 0.40–0.45 | 0.568 | 0.481 |
| 0.45–0.50 | 0.476 | 0.544 |
| **0.50–0.55** | **0.821** | **0.621** |
| 0.55–0.60 | 0.839 | 0.637 |

The anchored curve leaps at the review floor (+0.345 across it); the un-anchored one steps by
+0.077 and keeps rising smoothly. That is the fingerprint the mechanism predicts.

What it does **not** show is a large aggregate gap:

| band | city splits (anchored) | manual_gold (un-anchored) |
|---|---|---|
| below 0.50 | 0.195 (n = 995) | 0.208 (n = 1164) |
| 0.50–0.55 | 0.821 (n = **28**) | 0.621 (n = 58) |
| 0.55 and up | 0.965 (n = 1074) | 0.955 (n = 3487) |

Pooled precision below 0.50 is **essentially identical** on anchored and un-anchored GT
(0.195 vs 0.208), and so is precision above 0.55. So although the *direction* of the bias is
certain — an unreviewed real ramp can only be scored as a false positive — this control does
**not** establish that it is large, and the cliff itself rests on only 28 detections in the
0.50–0.55 bin.

Two readings are consistent with this and the control alone cannot separate them: the anchoring
effect is genuinely small in aggregate, or it is real but offset by `manual_gold` being
*in-distribution* GSV from the training cities (which should make RampNet look better there,
pushing the un-anchored numbers up).

**#55's A/B tagging settles it, and the effect is real.** All six city splits have now been
spot-checked (jonf, 2026-07-28): every unmatched prediction in the `[0.25, 0.55)` band was
tagged **A** (a real ramp the GT missed), **B** (a genuine false positive) or **unsure**.

| split | incremental FPs | A | B | unsure | A-rate |
|---|---|---|---|---|---|
| richmond | 29 | 5 | 22 | 2 | 17.2% |
| bend | 24 | 7 | 15 | 2 | 29.2% |
| clovis | 23 | 7 | 13 | 3 | 30.4% |
| morgantown | 30 | 4 | 25 | 1 | 13.3% |
| annapolis | 27 | 6 | 15 | 6 | 22.2% |
| budapest_district5 | 89 | 23 | 59 | 7 | 25.8% |

**Pooled over the five US splits, 26.9% of the incremental false positives in `[0.30, 0.55)`
were real curb ramps the ground truth had missed.** So the raw curve's precision penalty for
lowering the threshold is materially overstated — by roughly a quarter of the newly-added
false positives — and the aggregate similarity to `manual_gold` above is better explained by
that split's in-distribution advantage than by the bias being small.

Spread is 13%–30% with no obvious ordering by imagery quality (morgantown, the cleanest split,
is lowest; clovis, the softest, is highest), so **do not quote a single cross-split
GT-completeness constant** — apply the per-split correction.

### Why the recall gain, by contrast, is exact — and itself a lower bound

Every additional true positive below 0.55 matches a ramp a reviewer explicitly marked as a
real missed ramp. Those are confirmed ramps, not inferred ones, so the recall gain is not
subject to the bias above.

It is stronger than that. If the GT is missing `K` real ramps (which #55 exists to measure),
completing it would add those ramps to the denominator *and* credit them to the low-threshold
operating point that found them. Writing `a` for the extra TPs gained and `n` for the recall
denominator, the corrected gain `(a+K)/(n+K)` exceeds the measured `a/n` whenever `n > a` —
and here `n` ≈ 1,393 against `a` ≈ 70. **So the measured recall gain understates the true
one.**

Both directions favour lowering the threshold. The measured trade is a worst case.

## Results

### Pooled across the five US/VA city splits (n = 609 panos)

Wilson 95% intervals in brackets.

| threshold | precision | recall | F1 | detections/pano |
|---|---|---|---|---|
| 0.25 | 0.871 [0.852, 0.888] | 0.829 [0.808, 0.848] | 0.850 | 2.34 |
| 0.30 | 0.897 [0.879, 0.912] | 0.818 [0.797, 0.837] | 0.855 | 2.23 |
| **0.32** | 0.905 [0.888, 0.920] | 0.814 [0.793, 0.834] | **0.857** | 2.20 |
| 0.35 | 0.913 [0.896, 0.928] | 0.802 [0.780, 0.822] | 0.854 | 2.14 |
| 0.40 | 0.931 [0.916, 0.944] | 0.790 [0.767, 0.810] | 0.855 | 2.06 |
| 0.45 | 0.943 [0.928, 0.955] | 0.775 [0.752, 0.796] | 0.851 | 1.98 |
| 0.50 | 0.961 [0.948, 0.971] | 0.760 [0.737, 0.782] | 0.849 | 1.91 |
| **0.55** (deployed) | 0.965 [0.952, 0.974] | 0.744 [0.720, 0.766] | 0.840 | 1.85 |

**F1 is flat.** It varies by 0.008 across the whole 0.25–0.55 range, so F1-optimality alone
does not pick an operating point — a finding worth stating plainly, because "F1-optimal" is
the obvious criterion and here it is nearly uninformative. What *does* move decisively is
recall: 0.744 → 0.814, with **non-overlapping** confidence intervals.

### Per split, 0.55 → 0.32

| split | P@0.55 | R@0.55 | P@0.32 | R@0.32 | ΔR | ΔP | dets/pano | F1-opt |
|---|---|---|---|---|---|---|---|---|
| richmond | 0.964 | 0.768 | 0.911 | 0.829 | +0.061 | −0.052 | 2.15 → 2.57 | 0.33 |
| bend | 0.980 | 0.755 | 0.927 | 0.813 | +0.058 | −0.053 | 2.34 → 2.75 | 0.50 |
| clovis | 0.914 | 0.713 | 0.855 | 0.815 | **+0.103** | −0.060 | 1.31 → 1.61 | 0.35 |
| morgantown | 0.975 | 0.730 | 0.903 | 0.805 | +0.075 | −0.072 | 1.67 → 1.99 | 0.32 |
| annapolis | 0.973 | 0.738 | 0.912 | 0.806 | +0.068 | −0.062 | 1.82 → 2.13 | 0.32 |
| budapest † | 0.874 | 0.510 | 0.718 | 0.637 | +0.127 | −0.156 | 1.51 → 2.27 | 0.37 |
| manual_gold ‡ | 0.955 | 0.849 | 0.926 | 0.884 | +0.035 | −0.028 | 3.49 → 3.74 | 0.36 |

† Budapest is swept but **held out of the pooled recommendation** (single-rater GT at low
reviewer confidence — see `benchmark/README.md`).
‡ `manual_gold` is likewise held out: in-distribution GSV from the training cities with
independently-labelled GT. It is the control, not a deployment city.

**Clovis is the cheapest place to lower the threshold, not the most expensive.** This
reverses the expectation going in — clovis has the softest imagery (2018 GoPro Fusion) and
the lowest deployed precision, so it looked like the binding constraint. It gains **+10.3
recall points for −6.0 precision**, the best ratio of any US split.

### Per imagery tier

Tiers are assigned **per pano** from `camera_make`/`camera_model`, not per split — richmond
alone mixes iSTAR Pulsar and GoPro Max, so split-level grouping would smear the camera effect.

| tier | n | F1-opt | ΔR at 0.32 | ΔP at 0.32 |
|---|---|---|---|---|
| action cam, modern (GoPro Max) | 158 | 0.32 | +0.073 | −0.068 |
| action cam, legacy (GoPro Fusion 2018) | 125 | 0.35 | +0.103 | −0.060 |
| survey-grade (Trimble MX7) | 125 | 0.32 | +0.068 | −0.062 |
| pro 360 (iSTAR Pulsar) | 77 | 0.34 | +0.063 | −0.051 |
| Google Street View | 110 | **0.50** | +0.058 | −0.053 |

**The tier optima cluster at 0.32–0.35, which argues against per-tier thresholds.** Four of
five tiers agree within 0.03 despite spanning a 2018 action camera and a survey-grade vehicle
rig. That convergence is the useful result: imagery quality moves the *level* of precision but
not the *location* of the optimum.

GSV is the exception at 0.50, and it should not drive a per-tier policy yet, because three
things are confounded in it and all point the same way: it is a single split (bend, n=110),
bend is **in-domain** (a Stage-2 training city), and it is the one split whose GT and
predictions come from different resamples (see the parity caveat). Revisit when a second GSV
split exists.

### Where the recall gain lands on the distance axis

`benchmark/README.md` establishes that RampNet's misses skew far-field, which is the case for
multi-view fusion (labeler#27). If lowering the threshold only recovered near ramps, the two
levers would overlap. It does not:

| band | mean ΔR across the 5 US splits | ramps gained |
|---|---|---|
| near (<12.5 m) | +0.071 | 46 |
| mid (12.5–25 m) | +0.078 | 40 |
| far (>25 m) | +0.072 | 12 |

**The gain is essentially uniform across distance**, so the threshold lever and the
multi-view lever are largely independent and **stack**. Far-field recall stays poor even at
0.32 (bend 0.214, clovis 0.389, annapolis 0.490), so multi-view remains necessary — lowering
the threshold does not substitute for it.

Distances are the flat-ground estimate (camera height 2.5 m assumed), monotonic in `y`, so the
band ordering is a rank statement; only the metre labels depend on the assumption.

### Density: this is not recall-by-carpet

`docs/model_comparison.md` establishes that an open detector's apparent recall is largely
density — OWLv2 reaches its recall at **55–88 boxes/pano**. The obvious objection to lowering
RampNet's threshold is that it buys recall the same cheap way. It does not: pooled density goes
**1.85 → 2.23 detections/pano** at the recommended 0.30 (2.20 at 0.32, 2.14 at 0.35). That is a
21% increase in review burden for a 10% relative increase in recall, and it leaves RampNet
roughly **25–40× sparser** than the open detectors at their operating points.

### Corrected results (#55 applied)

Crediting confirmed A tags moves them from false positives to true positives, in **both** the
precision and recall denominators — correcting precision alone would report a corrected P
against an uncorrected R. `band hi` additionally credits the `unsure` items, so it is the
honest upper end rather than a formality.

Pooled over the five US splits:

| operating point | raw P | **corrected P** | band hi | raw R | corrected R | corrected F1 |
|---|---|---|---|---|---|---|
| 0.25 | 0.871 | 0.893 | 0.904 | 0.829 | 0.833 | 0.862 |
| **0.30** | 0.897 | **0.917** | 0.928 | 0.818 | 0.821 | **0.866** |
| 0.35 | 0.913 | 0.928 | 0.935 | 0.802 | 0.804 | 0.862 |
| 0.55 (deployed) | 0.965 | — † | — | 0.744 | — | 0.840 |

† No correction applies at 0.55: the incremental band is empty by construction. Its precision
is not immune to GT incompleteness, but with only 38 pooled false positives there the effect is
small.

Per split at 0.30:

| split | raw P | corrected P | band hi | corrected R | corrected F1 |
|---|---|---|---|---|---|
| richmond | 0.902 | 0.919 | 0.930 | 0.832 | 0.873 |
| bend | 0.924 | 0.942 | 0.948 | 0.825 | 0.880 |
| clovis | 0.851 | **0.883** | 0.899 | 0.826 | 0.853 |
| morgantown | 0.888 | 0.905 | 0.909 | 0.808 | 0.854 |
| annapolis | 0.902 | 0.920 | 0.939 | 0.813 | 0.863 |
| budapest ‡ | 0.707 | 0.762 | 0.777 | 0.660 | 0.707 |

‡ excluded from the pooled row and the recommendation.

**The correction changed the answer.** On raw numbers the F1 optimum sat at 0.32 and the
conservative choice was 0.35; with the GT completeness correction applied, corrected F1 peaks
at **0.30** and corrected precision there is **0.917** pooled — higher than the *raw* precision
at 0.35. That is precisely what #55 existed to determine, and it moved the recommendation down.

**Caveat on 3 of the A tags** (1 each in clovis, morgantown, budapest): they sit within 2 match
radii of an already-detected ramp, so they are more likely a second hit on one ramp than a ramp
the GT missed. The tooling flags these rather than hiding them; removing all three from the US
pool changes pooled corrected precision by under 0.002.

### The storage floor and the recall ceiling (labeler#28, labeler#27 stage 4)

The submission threshold this document recommends is a *policy* choice and reversible. The
labeler's **storage floor** (`DETECTION_STORAGE_FLOOR = 0.1`, with a top-50 per-pano cap;
labeler#28) is neither: a peak below it is never written, so no downstream consensus policy
can ever promote it. That makes it worth checking directly rather than assuming.

![Storage floor and recall ceiling](figures/storage_floor_ceiling.png)

Reproduce with `python scripts/analysis/low_floor_sweep.py floor` and
`python scripts/analysis/plot_storage_floor.py`.

**What a 0.1 floor costs.** Counting ground-truth ramps whose *best* candidate falls in
`[0.05, 0.10)` — i.e. ramps that a 0.1 floor makes permanently unrecoverable:

| split | GT ramps | best candidate in [0.05, 0.10) | share |
|---|---|---|---|
| richmond | 310 | 8 | 2.58% |
| bend | 327 | 8 | 2.45% |
| clovis | 195 | 6 | 3.08% |
| morgantown | 267 | 7 | 2.62% |
| annapolis | 294 | 9 | 3.06% |
| **POOLED (5 US)** | **1393** | **38** | **2.73%** |
| budapest | 300 | 7 | 2.33% |
| manual_gold | 3919 | 19 | 0.48% |

**The recall ceiling.** The share of GT ramps with *any* candidate at or above a floor —
the hard upper bound on what multi-view consensus can ever recover:

| | pooled (5 US) |
|---|---|
| recall at the deployed 0.55 | 0.744 |
| recall at the recommended 0.30 | 0.818 |
| **ceiling at the 0.10 storage floor** | **0.872** |
| ceiling at the 0.05 extraction floor | 0.899 |

So labeler#27 stage 4 has **+12.8 recall points** of headroom above the deployed threshold to
work with, and **+5.4 points** above the operating point recommended here — but it is capped
at 0.872 by the storage floor, not by the 0.899 the model actually produces.

**The verdict: lower the storage floor from 0.10 to 0.05.** The floor's own stated
justification (labeler#28) is that storing too little is irreversible while storing too much
only costs disk, bounded by the top-K cap. That argument survives contact with the data, and
the data says 0.1 is not where the bound should sit:

- **The cap never binds.** At the 0.1 floor the busiest pano in the entire benchmark holds
  **14** candidates against a cap of 50; medians are 2–5. At a 0.05 floor the maximum is still
  **14**. The top-50 cap is not the volume bound — the floor is, and it is doing work nobody
  asked it to do.
- **The cost is ~2.7% of findable ramps**, permanently, for a volume saving of roughly one
  extra candidate per pano.
- **The recovered ramps are exactly the population multi-view fusion is for.** They are
  ramps too faint to clear any single-view threshold, which is the case stage 4 exists to
  handle.

This is a labeler-side change and belongs in that repo; it is recorded here because this is
where the measurement lives.

## The deployment constraint is not what it was assumed to be

It is widely assumed that Project Sidewalk gates AI label submissions on
`ai-validation-min-accuracy` (0.92). **It does not.** In `AiController.submitAiLabel`,
submission is gated by an internal API key plus the boolean per-city
`ai-label-submission-enabled` flag — there is **no accuracy threshold on that path at all**.
The 0.92 belongs to a different subsystem (the DINOv2 *validator* scoring existing labels) and
never touches the auto-labeler.

Two consequences:

1. **No hard floor blocks a lower operating point.** The constraint is a policy judgment about
   review burden, not a gate.
2. **The change is reversible.** `ExploreService.submitAiLabelData` persists each label's
   `confidence` into `label_ai_info`, so anything submitted at a lower threshold can be
   re-thresholded server-side later. The asymmetry that matters — a ramp never submitted is
   invisible forever, a low-confidence one that was submitted can always be filtered — points
   the same way as the recall-first policy.

## Recommendation

**Adopt a single uniform threshold of 0.30**, replacing 0.55.

| | at 0.55 (deployed) | at 0.30 |
|---|---|---|
| pooled precision | 0.965 | **0.917** corrected (0.897 raw, 0.928 band high) |
| pooled recall | 0.744 | **0.821** corrected (0.818 raw) |
| pooled F1 | 0.840 | **0.866** corrected |
| detections/pano | 1.85 | 2.23 |

**+7.7 recall points for −4.8 precision points, at 0.38 more detections per pano.**

Why 0.30:

- It is the **corrected** F1 optimum, computed after applying #55's per-split A/B tagging
  rather than from the raw curve that the anchoring bias distorts.
- Corrected precision stays **≥0.88 on every US split** and 0.917 pooled — above the 0.92 bar
  the project uses elsewhere for AI assertions once the `band high` (0.928) is taken into
  account, and comfortably clear of anything that reads as a quality regression.
- Recall is RampNet's weak metric on **every** split, precision is not. Under the recall-first
  policy — a false negative is a permanent hole in the inventory, a false positive is a cheap
  human review — this is the right side of a flat trade.
- Density stays sparse (2.23/pano), so the recall is not bought by carpeting the image.
- The change is **reversible**: per-label confidence is stored server-side, so anything
  submitted at 0.30 can be filtered later.

**If a more conservative first move is wanted, 0.35** gives corrected P 0.928 / R 0.804 and
gives up 1.7 recall points. Both are inside the flat region; 0.30 is the recall-first choice
and 0.35 the precision-first one. I recommend 0.30.

**Clovis is the one split to watch**: corrected precision 0.883 at 0.30 (band high 0.899)
against 0.914 deployed — the only US split that gives up meaningfully more than it does
elsewhere. It is also the split with the largest recall gain (+10.3 points) and the highest
A-rate (30%), so the trade is still favourable; but if per-city tuning is ever introduced,
clovis-like imagery (2018-era GoPro Fusion) is where it would start.

**Not per-tier**, for now: four of five tiers agree within 0.03, and the one exception (GSV) is
confounded by in-domain-ness, a single split, and the resample caveat. A per-tier policy would
be fitting noise.

**Budapest is excluded from this recommendation** and should get its own decision. It is the
one split where lowering costs about as much precision as it buys recall, and its GT is
single-rater at low reviewer confidence. Nothing here transfers to non-US infrastructure
without a rubric written for it.

### Honest limits on this number

- **It is tuned on the benchmark.** There is no separate validation split, so 0.30 is an
  optimistic estimate of what threshold tuning buys. The mitigation is that the F1 curve is
  flat — anything in 0.25–0.40 performs within noise, so the *choice* is robust even if the
  *argmax* is not.
- **The correction rests on one rater**, the same person who produced the GT. A second rater on
  the A/B tags would tighten it, and is the same ask already outstanding for budapest.
- **`unsure` items are not credited** in the corrected column, only in `band high`. Pooled that
  is 14 items — the gap between 0.917 and 0.928.

### What would change this

- **A second GSV split**, which would show whether bend's 0.50 optimum is GSV or in-domain-ness.
- **Multi-view fusion** (labeler#27), which changes the question from "what threshold for a
  single view" to "what threshold for promotion given k agreeing views" — a lower number, since
  consensus supplies the precision the threshold currently has to.
- **Flip-TTA at the deployment point** (#78), measurable from the same cache: TTA yields 3.5%
  more detections at ≥0.55 on manual_gold.

## For labeler#27 stage 4: the promotion floor

Stage 4 wants the confidence at which a single-view detection is trustworthy enough to promote
on multi-view consensus, taken from GT-true vs GT-false histograms. Those are in
`analysis_out/op/confidence_calibration.{json,csv}` per split and pooled, with Wilson intervals.

**The measured single-view crossover is ≈0.40–0.45, on both anchored and un-anchored GT.** On
the five US city splits pooled, P(real) first exceeds 0.5 in the 0.40–0.45 bin (0.568) but dips
back to 0.476 in 0.45–0.50 — it is noisy at these bin sizes. On `manual_gold`, whose GT carries
no anchoring, the curve is smooth and crosses cleanly in the 0.45–0.50 bin (0.544, against
0.481 just below).

Worth being precise about the direction, because it is easy to get backwards: the anchored
splits *under-credit* real ramps below 0.55, so their crossover is if anything an
**over**-estimate of where real detections stop dominating — yet the un-anchored control lands
at essentially the same place or slightly higher. So there is no evidence here for a
dramatically lower crossover, and the two GT regimes agreeing is the more useful result than
either number alone.

Caveat in the other direction: `manual_gold` is in-distribution GSV from the training cities,
so it is optimistic for out-of-distribution deployment.

**The promotion floor should nonetheless sit below this crossover**, and that is reasoning
rather than measurement: the crossover is where a *single* view stops being more likely right
than wrong, whereas stage 4 accepts a detection only when *k* views independently agree on a
ground location. Consensus supplies evidence a single view lacks, so the break-even confidence
under agreement is lower. How much lower is not answerable from per-pano data — the benchmark
bundles are thinned to 30 m spacing precisely so no physical ramp appears twice, so nothing
here measures view agreement. That is stage 3's own GT to produce.

## Reproducing

```bash
# GPU, once (Hyak): writes analysis_out/op_cache/<split>.json
sbatch -A <account> scripts/analysis/run_low_floor_extract.slurm

# CPU, everything else
python scripts/analysis/low_floor_sweep.py parity     # the gate — run this first
python scripts/analysis/low_floor_sweep.py sweep      # per-split + pooled + per-tier
python scripts/analysis/low_floor_sweep.py hist       # calibration for labeler#27
python scripts/analysis/low_floor_sweep.py gtbias     # the anchoring measurement
python scripts/analysis/low_floor_sweep.py distance   # where the recall gain lands
python scripts/analysis/low_floor_sweep.py tagcheck   # #55 tags still resolve
```
