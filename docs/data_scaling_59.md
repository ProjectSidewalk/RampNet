# Data scaling (#59): would more training data buy recall? E1 and the miss buckets

§0–§0c of [`curb_ramp_data_sourcing.md`](curb_ramp_data_sourcing.md), moved verbatim under #145
with their section numbers kept, so a reference to "§0c" still means the same section once its
file name is updated. They answer #59's first question, whether more training data would buy
recall at all; the sourcing question (which cities, how much each would buy, what a retrain
costs) stays in that document, and a bare § reference to §1–§10 below means it.

## 0. E1 result: the harm hypothesis is NOT supported

Everything below prices out *sourcing*. #59 raised a prior objection — that scaling the same
pipeline could make the worst failure mode **worse** — and that had to be settled before any of it
mattered. It is now measured. Script: `scripts/analysis/stage1_label_recall.py`
(17 tests in `tests/test_stage1_label_recall.py`); result JSON in
`analysis_out/e1_stage1_label_recall.json`.

**The hypothesis.** Stage-1 agreement against the manual gold set is P .9403 / R .9245, so ~7.5% of
gold-visible ramps are unlabeled inside training panoramas. In heatmap regression that is not
neutral — the target is **zero** there, so the loss actively pushes activations down. If those
misses cluster in the far/small regime (plausible: projection error and the crop model both degrade
with distance), we have been training the model to suppress detections exactly where it is blind,
and scaling bakes that in harder.

**The measurement.** Two recall curves over the same **3,919 gold ramps in 793 panoramas**, on the
#25 bins with the identical `geom()` estimator `size_analysis.py` uses. Model detections at 0.55.

Both columns are recall against the same human ground truth, but they measure different things, and
the asymmetry is the point:

- **"Stage-1"** = *was a training label present at this ramp?* Stage 1 is **not a detector** — it is
  handed the ramp's location by the government inventory, projects that coordinate onto the
  panorama, and uses the crop model only to refine where the point lands. So this column measures
  **whether the supervision existed**, which is precisely the ceiling the hypothesis is about.
- **"model"** = *did the trained detector fire there?* It gets no positional hint and must locate
  ramps from pixels alone.

Stage-1 should therefore be the higher curve everywhere; the question is only whether it *falls off
with distance the same way the model does*.

*Caveat:* these Stage-1 labels come from the **test** split, so the model never trained on these
particular panoramas. Same pipeline and same cities, so they are a fair proxy for the label quality
the train split received — but a proxy, not a direct measurement of it.

| distance | n | Stage-1 labels | model | gap |
| :--- | ---: | ---: | ---: | ---: |
| 0–8 m | 1,374 | 0.959 | 0.943 | +0.016 |
| 8–12 m | 1,065 | 0.918 | 0.894 | +0.024 |
| 12–18 m | 865 | 0.924 | 0.842 | +0.082 |
| 18–25 m | 498 | 0.900 | 0.779 | +0.120 |
| 25–40 m | 113 | **0.779** | **0.487** | **+0.292** |
| **drop-off** | | **0.180** | **0.457** | |

Apparent size tells the same story: Stage-1 falls 0.951 → 0.797 across 80+ px down to 20–32 px
(−0.154); the model falls 0.938 → 0.541 (−0.397).

**Verdict: FLAT.** Stage-1 label recall does decline with distance, but **the model's cliff is ~2.5×
steeper**. At 25–40 m a training label was present at 78% of gold ramps while the model detected 49%
— the far ramps *were* labeled; the model is not reaching the ceiling that supervision set. The gap widens
monotonically with distance and with shrinking apparent size, which is the signature of a
resolution/model limit, not an inherited label limit. Consistent with #25's forecast (+0.103 recall
at 2× linear resolution).

**Instrument check.** Model overall recall comes out at **0.873**, reproducing the published
gold-set figure at 0.55 exactly; Stage-1 overall lands at 0.928 against the documented .9245. Both
curves reproduce known numbers, so the comparison is not an artifact of this script.

**What this does and does not license:**

- ✅ **Naive scaling is not contraindicated by this mechanism.** The #59 objection that motivated
  caution does not hold up. Sourcing work below is worth doing.
- ❌ **It does not show that scaling helps.** That is E2 (#84's epoch curve — are we even
  data-limited at 1 epoch?) and E3. This closes an objection; it does not make a case.
- ⚠️ **The implicit-hard-negative mechanism is real, just not binding.** 22% of ramps at 25–40 m
  are still unlabeled and still train as zeros. It is a second-order effect here, not a
  first-order one.
- ⚠️ **This is an in-distribution result.** The gold set is drawn from the NYC/Portland/Bend test
  split, so it says nothing about the out-of-distribution failures (Paterson's paired TSIs,
  Gainesville's diagonal ramps) that motivate the diversity argument in §1.

One analysis defect worth recording, since it nearly decided the experiment: the drop-off was
initially computed between the nearest and farthest *populated* buckets, and the gold set has **4**
ramps beyond 40 m. At n=4 a single ramp moves recall by 0.25, and that bucket inverted the sign of
Stage-1's drop-off (to −0.041). Buckets now require n ≥ 30, and a regression test pins it.

## 0a. How much of the missing recall can more data even reach?

E1 closed the objection to scaling but also implied something sharper: the far-field cliff is a
**pixel-count** problem, so more cities cannot fix it — while the *vocabulary* failures the
benchmark keeps surfacing (Paterson's paired tactile surfaces, Gainesville's diagonal arterial
ramps) plausibly are fixable that way. Those are two different populations with two different
programmes attached, and nobody had sized them.

> **Since qualified by §0c.** The split below stands as a measurement, but its "fixable by"
> column's hard binary does not: the model detects other far-field ramps of the *same apparent
> size* as its silent misses at a median 57% rate, so far-field failure is graded sensitivity,
> not a floor. Read the far/near boundary as a difficulty gradient, not a reachability partition.

Script: `scripts/analysis/miss_decomposition.py` (15 tests). Reads the committed low-floor caches,
so no GPU, no network, no imagery. Threshold 0.30 (the #79 recommendation); boundary 18 m, the last
distance at which the model still has adequate signal.

**Pooled across the seven US splits — 2,060 GT ramps, 427 misses, recall 0.793:**

| population | misses | share | recall points | fixable by |
| :--- | ---: | ---: | ---: | :--- |
| **Far-field** (≥ 18 m) | 247 | **57.8%** | 0.120 | multi-view (#48/#38), resolution (#25) — **not** more cities |
| **Near-field** (< 18 m) | 180 | **42.2%** | 0.087 | broader/more diverse training corpus |

**Neither dominates.** Roughly three-fifths of the missing recall is pixel-starved and two-fifths is
not. Both programmes have a real target, and the sourcing work below is aimed at a population worth
about **8.7 recall points** pooled.

| split | tier | GT | recall | miss | far | near | far % | MV ceiling |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| richmond | mapillary | 310 | 0.829 | 53 | 32 | 21 | 60.4% | 0.889 |
| bend | gsv | 327 | 0.823 | 58 | 36 | 22 | 62.1% | 0.904 |
| clovis | mapillary | 195 | 0.821 | 35 | 21 | 14 | 60.0% | 0.907 |
| morgantown | mapillary | 267 | 0.805 | 52 | 15 | 37 | **28.8%** | 0.829 |
| annapolis | mapillary | 294 | 0.810 | 56 | 37 | 19 | 66.1% | 0.902 |
| paterson | gsv | 395 | 0.719 | 111 | 64 | 47 | 57.7% | 0.792 |
| gainesville | gsv | 272 | 0.772 | 62 | 42 | 20 | 67.7% | 0.872 |
| *budapest* † | – | 300 | 0.643 | 107 | 40 | 67 | 37.4% | 0.667 |
| *manual_gold* † | – | 3,919 | 0.892 | 423 | 141 | 282 | **33.3%** | 0.915 |

† swept, not pooled.

**The geometry caveat was checked, and it does not drive the result.** Flat-ground distance is
`camera_height / tan(depression)`, which fails on unleveled rigs and hills —
`docs/detection_recall_analysis.md` reports it agreeing with DA3 metric depth at Spearman **0.95 on
GSV but only 0.81 on Mapillary**, and four of the seven pooled splits are Mapillary. A tilted rig
pushes a near ramp toward the horizon, so geometry would call it *far* and **overstate** the
far-field share. Two checks say that is not what happened:

- **Above-horizon GT ramps** — geometrically impossible for a ground ramp, hence a direct tell —
  number **5 across 1,066 Mapillary ramps (0.5%) and 0 across 994 GSV ramps.**
- The **GSV** tier shows a *higher* far share (61.5%) than **Mapillary** (53.6%) — the opposite
  direction to the bias.

(Budapest, held out, has **12 of 300 (4%)** above the horizon — eight times the US rate, consistent
with its consumer rig and its low reviewer confidence. A useful independent corroboration of why it
is held out.)

**Apparent size is the starker cut**, and is what makes the pixel-limit reading concrete:

| apparent size | n GT | recall |
| :--- | ---: | ---: |
| 12–20 px | 49 | **0.163** |
| 20–32 px | 230 | 0.543 |
| 32–50 px | 427 | 0.745 |
| 50–80 px | 677 | 0.885 |
| 80+ px | 643 | 0.885 |

Below ~32 px the model is largely blind, and recall saturates around 0.885 above 50 px — so extra
pixels stop helping well before perfect, which is itself a caution against expecting resolution
alone to close the gap.

**Optimistic multi-view ceiling: 0.868 (+0.075 recall).** That assumes a closer capture exists for
every far ramp and that re-observation succeeds at the measured near-field rate; it ignores fusion
cost and the extra false positives more looks would generate. It is "what is on the table", not a
forecast — but +7.5 points with **no new data collection** is a serious number next to a
multi-month sourcing campaign.

**Caveat on the near-field population — now measured, see §0b.** Calling all 42.2% "vocabulary" was
an inference, not a measurement. A near-field miss can equally be occlusion (a parked car), deep
shadow, or surface debris — Gainesville's reviewer flagged debris explicitly — or a GT disagreement.
The near-field figure bounds the sourcing-addressable population **from above**, and §0b tightens
that bound by a factor of 3.8.

## 0b. Bucketing the misses: most of the near-field population is not a data problem

§0a's near-field figure was an upper bound with an explicit caveat attached. #46 measured what the
"appearance/vocabulary" label was standing in for, and **two-thirds of it turns out to be something
more data cannot fix.**

Script: `scripts/analysis/miss_taxonomy.py` (29 tests). Same committed low-floor caches, same
threshold, same boundary, same `geom()` and matcher as §0a — so the two partition an identical
population and the bucket counts sum to §0a's totals. No GPU, no network, no imagery.

The caches hold every peak down to a **0.05 score floor**, well below the 0.30 operating point, so
for each missed ramp we can ask what the model actually did there.

**Pooled, seven US splits — 427 misses:**

| bucket | misses | share | recall points | what it actually is |
| :--- | ---: | ---: | ---: | :--- |
| **merged** | 124 | 29.0% | 0.060 | one peak emitted for a pair of adjacent ramps |
| **sub_threshold** | 166 | 38.9% | 0.081 | localized, scored in [0.05, 0.30) |
| **localization** | 9 | 2.1% | 0.004 | fired just outside the match radius |
| **silent** | 128 | 30.0% | 0.062 | nothing there at all, even at the floor |

**The near-field split is the number that moves.** Of §0a's 0.087 recall points:

| bucket | misses | recall points | addressable by more cities? |
| :--- | ---: | ---: | :--- |
| merged | 48 | 0.023 | **No** — heatmap representation |
| sub_threshold | 84 | 0.041 | **No** — confidence, already priced by #54/#55 |
| localization | 3 | 0.001 | marginal |
| **silent** | **45** | **0.022** | **Yes — this is the sourcing programme's target** |

So the population a broader corpus can reach is about **0.023 recall points, not 0.087** — §0a's
near-field figure **over-states it by 3.8×**. That does not kill the sourcing case, but it resizes
it: the honest headline is "worth ~2 recall points", not ~9.

### Two confounds checked, both negative

**The matcher is not manufacturing misses.** #46 lists this as a suspect — a correct-but-loose
detection scored as an FP *and* its ramp as an FN, one error counted twice. Rescoring every pano
with maximum-cardinality bipartite matching instead of the deployed greedy matcher is a **wash**:
10 ramps are hit only under optimal, 10 only under greedy, **net zero**. The difference is a
permutation, not lost recall.

**"A peak was there" is not density.** `docs/model_comparison.md`'s null-recall correction found
open-detector recall was largely density (OWLv2 at 55–88 boxes/pano). RampNet emits **4.2**
floor-level peaks per pano, and against a null that holds each ramp's elevation and randomizes its
azimuth, near-field `sub_threshold` is **46.7% real vs 4.7% chance**. The bucket survives; the null
rate is printed beside every bucket rather than argued away.

### `merged` is a target problem, not an extractor one

`peak_local_max` suppresses on a maximum filter, i.e. **Chebyshev** distance ≤ `min_distance=10`.
**78 of 124 merged pairs (63%) sit above that** — the extractor was free to emit two peaks and did
not, so the heatmap itself had one mode. And **87% sit within 2σ of the σ=10 training target**,
which is what cannot represent an adjacent pair as two modes in the first place.

That retires the untested "`min_distance=3`" idea for at least 63% of the bucket, on top of #62
finding NMS at the match radius actively harmful. **If this bucket is worth attacking, the lever is
the training target's σ, not the peak extractor.**

### Paterson's anomaly now has a mechanism

| split | misses | merged | share |
| :--- | ---: | ---: | ---: |
| **paterson** | 111 | **80** | **72%** |
| bend | 58 | 14 | 24% |
| richmond | 53 | 10 | 19% |
| morgantown | 52 | 9 | 17% |
| annapolis | 56 | 5 | 9% |
| gainesville | 62 | 5 | 8% |
| clovis | 35 | 1 | 3% |

Paterson's paired tactile surfaces are **not** a vocabulary failure — they are two ramps the
heatmap cannot separate. That is why it is the narrowest RampNet lead in the benchmark, and it is
fixed by σ, not by Newark.

### Bracketing `silent`: did any other model see these ramps?

`silent` means *RampNet* saw nothing. It does not mean nothing is there. Script:
`scripts/analysis/silent_witness.py` (17 tests), reading `.model_cache` — no GPU, no imagery.

For each silent miss, did any challenger put a detection within the match radius? If one did, the
imagery demonstrably contains a recognizable ramp, so RampNet's failure is **specific to RampNet** —
which is the strongest evidence for a genuine appearance/vocabulary failure obtainable without a
human, and exactly what more training data targets.

The density correction is mandatory here, for the third time in this analysis: OWLv2 witnesses
121 of 128 silent misses, but chance alone accounts for 76.9 of them.

**Every number in this section is against one fixed witness pool** —
`rampnet.roster.WITNESS_POOL_46`, the roster as it stood on 2026-07-31, which is the pool the
#46 tagging pass below was rated under. The pool is frozen and `silent_witness.json` records
it, because a further witness can only shrink the unwitnessed set and that set is the pass's
item list. This is not hypothetical: adding the already-published `gemini-3.7-flash` leg takes
the unwitnessed count from 59 to 58 and the lower bound from 0.0092 to 0.0088, and the item it
removes is one of the 50 already rated. See #122.

| witness | raw | by chance | **excess** |
| :--- | ---: | ---: | ---: |
| gemini-3.1-pro-preview | 46 | 9.4 | +36.6 |
| gemini-3.6-flash | 33 | 8.1 | +24.9 |
| molmo2-8B | 26 | 8.5 | +17.5 |
| Qwen3-VL-8B | 22 | 6.1 | +15.9 |
| Qwen3-VL-32B | 16 | 2.7 | +13.3 |
| **union, the pool's sparse models** | **69** | 30.0 | **+39.0** |
| *union, the pool's dense detectors* | *127* | *102.2* | *+24.8* |

**Near-field: 32 of 45 witnessed raw (71.1%), chance 13.0, so ~19 corrected (42%).**

That brackets the sourcing-addressable population against the 2,060 pooled GT ramps:

| | recall points | ramps | what it is |
| :--- | ---: | ---: | :--- |
| #59's original bound | 0.087 | 180 | the whole near-field population |
| §0b's bound | 0.022 | 45 | near-field `silent` only |
| **lower bound** | **0.009** | **~19** | **confirmed** visible to another model, and missed |
| **upper bound** | **0.022** | **45** | all near-field `silent` |

**So the sourcing programme's target is between ~1 and ~2 recall points.** The gap is the
unwitnessed remainder — *not* shown to be unaddressable, only unproven either way. Closing it is
what the gallery is for.

### Caveats, travelling with the numbers

- **`silent` is still an upper bound**, now with a floor under it. It means the cached detections
  witness nothing there. Occlusion, deep shadow, debris and GT disagreement all still live inside
  the unwitnessed remainder, and separating them needs the imagery — that is #46's gallery half.
  **The reviewer pass is now done** (one rater, no second — `docs/replication.md` §"What the first
  pass produced"), and its far-field verdicts raised their own question: **§0c**.
- **The witness test is one-directional.** A witnessed ramp is confirmed recognizable; an
  unwitnessed one is not confirmed *un*recognizable, since every challenger is weaker than RampNet
  on this task and may simply have missed it too.

### The work that closes the bracket, and its exact size

The gap between 0.009 and 0.022 is the **59 unwitnessed** silent misses. Those, and only those,
need a reviewer:

```bash
python scripts/analysis/silent_witness.py --json-out analysis_out/silent_witness.json
python scripts/analysis/miss_gallery.py --bucket silent \
    --queue analysis_out/silent_witness.json --render analysis_out/gallery46_silent
python scripts/analysis/make_tagger.py analysis_out/gallery46_silent
# open analysis_out/gallery46_silent/tagger.html
```

That yields **50 crops** — 59 unwitnessed, less 9 below the 30-source-pixel floor, which are
excluded from any rate rather than labelled. The verdict scheme is built so exactly one answer is
sourcing-addressable. It has eight verdicts, and the authoritative copy — the one that travels
inside every per-rater file — is `benchmark/RUBRICS.md` §3:

| verdict | what it means | programme |
| :--- | :--- | :--- |
| `visible` | the ramp itself is resolvable | **vocabulary — this is the sourcing target** |
| `context-only` | ramp not resolvable; crosswalk / apron / curb-cut cues imply one | learnable, from scene layout |
| `occluded` | something physically in the way (even if still identifiable) | capture |
| `lighting` | exposure **destroyed** it — clipped white or crushed black | capture |
| `surface` | debris, snow, leaves, construction covering it | environment |
| `not-a-ramp` | nothing ramp-like here | GT error |
| `definition` | imagery clear; whether this **class** counts is the question | rubric question |
| `unclear` | cannot tell even with context | excluded from every rate |

**The `visible` rate over those 50, applied to the 59, is what converts the bracket into a point
estimate.** It has been run (2026-07-31, one rater): near-field `visible` 7 of 13, which puts the
sourcing-addressable population at **~0.013 recall points** (~19 chance-corrected witnessed + 7
visible, against 2,060 pooled GT — `docs/replication.md`). Single-rater caveat applies, and the
**far-field** verdicts from the same pass raised the question §0c takes up.
- **Some of `merged` may be double-marked GT.** 24 of 124 pairs sit below 8 px (~25 cm at 10 m),
  which is not a physical spacing for two ramps; on the verdict splits that is plausibly one ramp
  marked twice. If so they are *spurious GT* and leave the population entirely rather than changing
  bucket: merged 100, recall 0.802 (from 0.793), **silent unchanged**. `manual_gold` is the control
  — its GT is independent manual labeling with no RampNet review in the loop, and it shows the same
  mechanism at **44%** of misses.
- **`sub_threshold` is not free recall.** Those ramps are recoverable by lowering the threshold,
  which #54/#55 already evaluated and priced in precision; 0.30 was chosen knowing it.

## 0c. The far-field `visible` anomaly: the pixel floor does not survive its own hits

The reviewer pass produced a result §0a's framing did not predict. Of the **37 far-field**
silent-miss crops: **34 `visible`, 2 `context-only`, 1 `unclear`** — a 94% visible rate over
rateable crops, with **zero** `occluded` and **zero** `lighting` verdicts. Three facts sharpen it:

- the rubric licenses `visible` only on the **model-resolution panel** (`benchmark/RUBRICS.md`),
  so this is not the reviewer spending the 4× stored pixels the model never received;
- every rated crop is **unwitnessed** — none of the 8 challenger models put anything in radius
  either;
- the deepest crops (40–150 m, down to **10.5 model px**) were rated visible **9 of 9**.

At face value: ramps resolvable at the model's own pixel budget, invisible to all eight models —
against the reading that far-field misses are pixel-starved and unreachable by any training-side
fix. The four-hypothesis study design is on #46 (2026-07-31); this section is **Phase 0**: check
the *sample* (the rated 37 passed two selection filters) and check the framing against the model's
own far-field behaviour, before the verdicts are allowed to mean anything.

Script: `scripts/analysis/farfield_forensics.py` (41 tests); result JSON
`analysis_out/farfield_forensics.json`. Committed inputs only — the low-floor caches, the witness
list, the gallery manifest and verdicts, and the imagery manifests' `width` fields. No GPU, no
network, no imagery. Both phases read one named rater's pass (`--rater`, default `jonf`,
resolving to `benchmark/miss_taxonomy_46/silent__<rater>.json`) and the rater is recorded in the
result JSON, so the second pass this section keeps asking for is a flag rather than a patch.

### The sample: survivorship is real, mild, and now quantified

The 83 far-field silent misses reduce to 37 rated through two filters — **witnessed** (37,
already explained by another model's detection) and the **30-source-pixel judgeability floor**
(9). The floor is not one floor: stored panoramas run 4096–16384 px wide while `geom()` sizes
ramps at the model's 4096-px input, so 30 source px is a different model-pixel cut per split:

| split | tier | stored px | floor (model px) | far-silent | unwitnessed | rated |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| richmond | mapillary | 4096–12288 | 10.0–30.0 | 13 | 5 | 4 |
| bend | gsv | 13312–16384 | 7.5–9.2 | 14 | 9 | 9 |
| clovis | mapillary | 5760 | 21.3 | 9 | 4 | 3 |
| morgantown | mapillary | 4096 | 30.0 | 6 | 2 | 1 |
| annapolis | mapillary | 8000 | 15.4 | 14 | 8 | 2 |
| paterson | gsv | 16384 | 7.5 | 7 | 4 | 4 |
| gainesville | gsv | 16384 | 7.5 | 20 | 14 | 14 |

Which split a miss happened in decides whether a reviewer ever saw it — the 16384-px GSV splits
admit far misses down to 7.5 model px while morgantown stops at 30, and the deck comes out
**27 GSV / 10 Mapillary**.

| population | n | dist q1/med/q3 (m) | px q1/med/q3 |
| :--- | ---: | :---: | :---: |
| rated (reached the deck) | 37 | 23.6 / 27.6 / 39.4 | 19.9 / 28.3 / 33.2 |
| below the floor (excluded) | 9 | 55.4 / 116.7 / 150.0 | 5.2 / 6.7 / 14.1 |
| witnessed (never queued) | 37 | 23.4 / 29.5 / 36.3 | 21.5 / 26.5 / 33.5 |
| **all far-field silent misses** | **83** | 23.6 / 31.2 / 42.6 | 18.4 / 25.0 / 33.1 |
| far-field hits, for contrast | 453 | 20.3 / 21.3 / 33.9 | 23.1 / 36.6 / 38.6 |

- **AUC(rated px vs unrated far-silent px) = 0.600**; the rated median sits at the **62nd
  percentile** of the far-silent size distribution. A bias toward bigger/closer exists and is mild.
- The 9 excluded items are the **extreme tail** — median 117 m, three of them above-horizon clamps
  (i.e. not distances at all). So the 94% generalizes to the far-silent *core* (~18–50 m); it says
  nothing about the deep tail, which is exactly where pixel starvation is most plausible.
- Mis-binning guards: **zero** above-horizon clamps among the rated 37, and the deck is majority
  GSV — the tier where flat-ground distance is trustworthy (Spearman 0.95 vs 0.81).

### The framing: the model's own hits refute a hard pixel floor

The decisive check needs no reviewer at all. If far-field silence were pixel-starvation, the model
should not be detecting *other* ramps at the same apparent size. It is:

| band | GT | recall | silent misses | rated | rated `visible` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 18–25 m | 395 | 0.777 | 25 | 14 | 12 |
| 25–40 m | 226 | 0.549 | 32 | 14 | 13 |
| 40–150 m | 74 | 0.284 | 23 | 9 | 9 |
| clamp ≥ 150 m | 5 | 0.200 | 3 | 0 | 0 |
| **total** | **700** | | **83** | **37** | **34** |

The totals row is not decoration: the bands have to sum to the far field or the table is
describing a smaller population than the section around it. An earlier version did not.
`geom()` reaches 150 m by two routes — the above-horizon branch, and `min(d, 150.0)` on a row
that is *below* the horizon and saturates anyway — and only the first is a `y` tell, so a
half-open top band dropped the second kind out of every row of this table while `y > 0.5` also
kept it out of `clamp`. Two far-field GT rows, one of them a silent miss. The top band is now
closed at its upper edge and the partition is asserted at runtime and in
`tests/test_farfield_forensics.py::test_the_bands_partition_the_far_field`.

- **Matched-size detection rate**: for each `visible` miss, the model's recall over all far-field
  GT within ±20% of that miss's apparent size is **median 0.57** (q1 0.31, q3 0.74). A hard pixel
  floor would put these near zero.
- **AUC(far-hit px vs far-silent px) = 0.718** — size matters, but it is far from deciding.
- Recall declines **0.777 → 0.549 → 0.284** across the bands. Even at 40–150 m the model finds
  roughly 3 in 10.
- **Against E1's gold-set bins, one band agrees and one does not, and both belong here.** The
  pooled 25–40 m rate (0.549) sits close to E1's 0.49; the pooled 18–25 m rate (0.777) is
  12 points *below* E1's 0.90. That is the expected direction and not a contradiction — the gold
  set is `manual_gold`, which is in-distribution, while these seven are deployment cities where
  RampNet's F1 runs 0.12–0.37 lower. Quoting only the band that agrees would misrepresent the
  comparison; the shape (a steep decline with distance) is what replicates, not the levels.

**Far-field failure is graded sensitivity, not a cliff.** A silent far-field miss is not a ramp
below a physical detection floor — it is the unlucky tail of a process that succeeds on most
same-sized ramps. That is consistent with `docs/detection_recall_analysis.md`'s sensitivity
finding and with the human verdicts, and inconsistent with reading "more examples do not add
pixels" as a claim about *reachability*. (As a claim about pixels it remains true; the error was
inferring unreachability from it.)

### Phase 1: attenuated or absent? Almost never absent

`silent` is a statement about **peaks** — no `peak_local_max` peak ≥ 0.05 within the match radius.
Phase 1 makes the statement about the **heatmap**: `scripts/analysis/silent_activation.py`
(41 tests) loads the published checkpoint (`projectsidewalk/rampnet-model` — the weights every
committed cache came from), runs one pass per panorama holding a silent miss (single-pass fp32,
matching `op_cache`), and reads the max heatmap value inside the match radius. The scaled matcher
space *is* the 512×1024 heatmap grid, so the grid and the radius are the matcher's — **with one
deliberate divergence: this window wraps at the 360° seam and the matcher's does not**
(`greedy_match` takes a plain x difference). Wrapping is the right geometry for an equirectangular
panorama, so the divergence is flagged per row (`seam`) rather than removed. **9 of the 128 misses
are seam rows**; in every one the nearest floor peak is ≥ 23.4 px against a 22.5 px radius, so
none would change bucket under a wrapping matcher and no number below moves — but that is a
property of this population, not a guarantee, which is why the flag ships in the JSON.

Result JSON: `analysis_out/silent_activation.json`; run on the local RTX 3070, all 128 pooled
silent misses across 108 panoramas. **Unlike Phase 0 this needs pixels** — the native-resolution
panoramas at `benchmark/<city>/panos/`, which are git-ignored and published as the Hugging Face
dataset `projectsidewalk/rampnet-benchmark` (the bundle #94's imagery manifests pin by content
hash). `--panos-root` points at whichever checkout holds them. Everything else it reads is
committed.

**Replicated on a second machine (#131, 2026-09-24).** The same script, on klone's L40S, from
the published inputs only, reproduces every number in the tables below. Result:
`analysis_out/silent_activation_replica.json` (run record beside it, `.run.json`: host, GPU,
driver, library versions, model and dataset commits). Against the committed file it is the
issue's **outcome 3, at the noise floor**: no row changed class (10 / 39 / 79 both), no row
flipped `above_own_null_p95` (31 both), `null_pct` identical to its three decimals in all 128
rows, and the only movement is in the raw activations — `act` differs in 50 of 128 rows by at
most **7 × 10⁻⁵**, `null_p95` in 61 by ≤ 4 × 10⁻⁵, `null_med` in 19 by ≤ 2 × 10⁻⁵,
`act_at_site` in 37 by ≤ 1 × 10⁻⁵ (fifth-decimal float noise, on values rounded to 5 places).
`scripts/analysis/compare_silent_activation.py` produces that report from the two committed
files, and `tests/test_compare_silent_activation.py` pins it. So the size of GPU / driver / OS
nondeterminism on this pipeline, single-pass fp32, is **below 10⁻⁴ on the heatmap**, and no
statement this section makes depends on the fourth decimal. Two further things the run
established:

- **It is deterministic on the same hardware class.** The job ran twice, on the lab's
  allocation (g3104) and as a copy on the scavenger partition (g3124), both L40S; the two
  replicas are byte-identical (`silent_activation_replica_ckpt.json`). The drift is between the
  RTX 3070 / Windows original (cuDNN 9.5 is what that venv reports today, not a record of the
  2026-07-31 run) and the L40S / Rocky 8 / cuDNN 9.10 replica, not between runs.
- **The header's `cities` list is in a different order** in the two files (alphabetical in the
  original, which took `US_SPLITS` from `miss_decomposition`; the frozen tuple the script carries
  since the #99 review starts at richmond). It is the run's scope, not an input to any number:
  results are sorted by (city, pano) and the null RNG is consumed in that order. The comparison
  treats it as a set.

What differed, for the record: the original was the Windows venv (the PyTorch pip wheel
`torch 2.6.0+cu126`, cuDNN 9.5, `timm 1.0.28`, RTX 3070 — what the venv reports today; its
exact versions on 2026-07-31 were not written down, which is itself a finding — the run record
now travels with the artifact); the replica is the repo's `environment.yml` env on klone
(Python 3.10.20, torch 2.6.0 as the conda-forge CUDA 12.6 build
`pytorch-2.6.0-cuda126_mkl_py310_h5ee0071_304`, cuDNN 91002, timm 1.0.28, numpy 2.2.6,
Pillow 12.0.0, scikit-image 0.25.2, driver 580.178.04, Rocky Linux 8.10), `rampnet-model` at
`606a1195`, `rampnet-benchmark` at `63d5ffd0`. So the torch *build* differs as well as the GPU,
OS, driver and cuDNN; the torch and timm version numbers and the repo code do not.

Which RampNet commit produced each replica (the committed `.run.json` files predate the
`code` field the launcher now writes, so this is recovered, not recorded): both jobs ran the
Python in the klone checkout at `c803120` (cloned at `1b6e463` at 21:59:25 PDT on 2026-09-23,
fast-forwarded to `c803120` at 22:13:29, per its reflog; both jobs started after that). The
launcher is the copy Slurm stored at submit time: the lab job 40546727 was submitted at
21:59:40 with `1b6e463`'s launcher, the ckpt copy 40549843 at 22:13:41 with `c803120`'s — each
stored script (`sacct --batch-script`) is byte-identical to that commit's file. Neither
difference reaches a number: `scripts/analysis/silent_activation.py` and `rampnet/` are the same
blobs at both commits, and the launchers differ only in the `OUT` / `TAG` overrides, which is
why the lab job's ledger row carries the older `silent-activation-131:` run id. The checkout's
tracked code was clean when checked on 2026-09-24 (only `analysis_out/` modified); whether it was
clean at the moment each job ran was not recorded. Cost: 0.22 GPU-hours on klone,
$0, three rows in `analysis_out/compute_log.jsonl` and two `paid: false` rows in
`analysis_out/usage_log.jsonl` (265 s and 495 s of wall-clock; the ckpt copy shared its node).

To re-run it from a clean clone (klone; any Linux box with one CUDA GPU works the same way
without the `sbatch`):

```bash
# 1. the seven pooled splits' native panoramas from the Hub into a checkout-shaped root, every
#    image checked against the Parquet's sha256 and the committed imagery_manifest.json
python scripts/unpack_benchmark_panos.py --out $WORK/benchmark_root \
    --cities richmond,bend,clovis,morgantown,annapolis,paterson,gainesville
# 2. the study, written beside the committed result, never over it
python scripts/analysis/silent_activation.py --panos-root $WORK/benchmark_root \
    --json-out analysis_out/silent_activation_replica.json
# 3. the three-outcome comparison (exit 0 bytes / 1 values / 2 moved / 3 different population)
python scripts/analysis/compare_silent_activation.py \
    analysis_out/silent_activation.json analysis_out/silent_activation_replica.json
```

On klone steps 1 and 2 are `scripts/analysis/silent_activation_131_unpack.slurm` (CPU, ckpt) and
`scripts/analysis/silent_activation_131.slurm` (one L40S), in that order; the usage comments in
each carry the exact `sbatch` lines, and the second writes the run record and the ledger row
itself. Run `mkdir -p logs` in the checkout before the first `sbatch`: both jobs write their
Slurm log to `logs/` relative to the submit directory, `logs/` is not tracked, and without it
the job fails at launch with no log at all.

| population | n | act q1 / med / q3 | act ≥ 0.01 |
| :--- | ---: | :---: | ---: |
| near / rated | 13 | 0.009 / 0.099 / 0.197 | 9 |
| near / witnessed | 32 | 0.032 / 0.211 / 0.592 | 30 |
| far / rated | 37 | 0.022 / 0.076 / 0.409 | 34 |
| far / below-floor | 9 | 0.042 / 0.194 / 0.381 | 8 |
| far / witnessed | 37 | 0.045 / 0.188 / 0.615 | 37 |
| **all silent misses** | **128** | 0.032 / 0.136 / 0.548 | **118** |

What that in-window mass *is* (classes are act ranges; the offset and nearest-peak columns
confirm the intended reading rather than define it):

| class | definition | n | near / far | `visible` (all fields) | argmax offset med | nearest floor peak med | null pct q1/med/q3 |
| :--- | :--- | ---: | :---: | ---: | ---: | ---: | :---: |
| **absent** | act < 0.01 | **10** | 6 / 4 | 5 | 22.0 px | 77.5 px (3.4 R) | 0.445 / **0.495** / 0.650 |
| **faint local** | 0.01 ≤ act < 0.05 | 39 | 12 / 27 | 13 | **10.2 px** | 85.6 px (3.8 R) | 0.600 / **0.780** / 0.855 |
| **tail** | act ≥ 0.05 | 79 | 27 / 52 | 23 | 22.3 px | **31.1 px (1.4 R)** | 0.860 / **0.915** / 0.990 |

- **Only 10 of 128 silent misses (8%) have a genuinely flat heatmap.** "Silent = the model saw
  nothing" is wrong for 92% of the bucket; `silent` was peak bookkeeping, not absence of response.
- **62% are a neighbouring mode's tail.** The argmax sits in the window's outer quarter in 75 of
  79, and the nearest cached floor peak is ~1.4 R away with **median score 0.685** — a *confident*
  adjacent detection (70/79 within 2 R). That mode is a neighbour ramp's TP, an FP, or plausibly
  this very ramp localized just outside the radius — the `localization` bucket only inspects
  *kept* (≥ 0.30) annulus peaks, so a floor-level one leaves a miss "silent". Whichever it is,
  this is the σ/representation family again (`merged`'s mechanism), not vocabulary.
- **30% are a faint local response at the site itself** (mass on-site in 30 of 39, nothing else
  within ~3.8 R) — the `sub_threshold` continuum extending below the floor. Attenuation, not
  blindness.
- For the far-field rated-`visible` population — the anomaly itself — the split is **3 absent /
  12 faint-local / 19 tail**: the model is responding at or next to ~91% of the far ramps a human
  called resolvable. Consistent with Phase 0's graded-sensitivity reading; squarely against a
  vocabulary hole.
- **The null separates the three classes cleanly, and it is the check the classes needed.** The
  cutoffs are raw activation, so on their own they assert rather than demonstrate that
  `faint local` is a *response*. Against each site's own panorama (azimuth-randomized at the
  site's elevation, self-excluding within 2 R), `absent` sits at chance — **median percentile
  0.495**, which is what a flat heatmap should read — while `faint local` is at **0.780** and
  `tail` at **0.915**. A sub-floor bump that a human would dismiss as nothing does not land two
  thirds of the way up its own band's distribution.
- **Read the percentile, not the p95 count.** Only 31/128 clear their own p95, and just 2 of 39
  faint-local sites do, which at a glance looks like it undercuts the paragraph above. It does
  not, because the p95 is not a noise floor: with a 22.5 px radius on a 1024-wide grid there are
  only ~23 non-overlapping windows per elevation band, so the 95th percentile of 200 draws is
  effectively the band *maximum* — pooled, median `null_p95` is 0.595 against a median `null_med`
  of 0.003. The flag therefore asks "is this site the strongest thing on its horizon row", which
  nothing sub-floor can pass by construction. It is conservative twice over, since a draw may also
  land on *another GT ramp* in the same band; only the site's own 2 R zone is excluded. Both
  statistics are in the JSON (`null_pct`, `above_own_null_p95`) — the percentile is the one to
  quote, and the p95 count is recorded so nobody has to rediscover why it is low.
- **Half the `absent` sites were rated `visible` by the reviewer** (5 of 10 — 3 far, 2 near).
  That is the sharpest cell in the table: a human calling the ramp resolvable at the model's own
  pixel budget while the heatmap is flat. Ten cases is too few to carry a claim, but they are the
  cleanest targets Phase 2's scale counterfactual has, and they should be run individually rather
  than only in aggregate.

### What changes, what does not, and what is still open

- **§0a's measured split stands** (247 far / 180 near at 18 m). What falls is the hard binary in
  its "fixable by" column: the far field is *harder*, not *unreachable*.
- **The sourcing bracket (§0b) excluded all 83 far-field silent misses from the addressable
  population because of that binary.** That exclusion is no longer safe — but Phase 1 cuts the
  other way too: of the 45 *near-field* silent misses the 0.013 estimate rests on, only **6 are
  heatmap-absent**; the rest are faint-local (12) or an adjacent confident mode (27), i.e. the
  calibration and σ families §0b already prices separately. The 0.013 point estimate is
  deliberately **not revised** in either direction until Phase 2 (the scale counterfactual, whose
  primary target is now the 10 absent sites plus whether scale lifts faint-local over the floor)
  and Phase 3 (the decoy control on the verdicts) run. Quote 0.013 with this section attached.
- **Multi-view's remedy logic is untouched** — a ramp invisible at 30 m is at 8 m two panoramas
  later whatever the failure mechanism — but §0a's "MV ceiling" column shares the binary
  assumption and will move with the same phases.
- **The human-side caveat is live.** One rater; and the 9-of-9 `visible` rate in the deepest band
  (down to 10.5 model px) is where pointed-verification bias would show most strongly. Phase 3's
  decoy deck should therefore be **stratified by distance band**, oversampling 40–150 m.

**The takeaway.** "Are far ramps harder?" — yes, threefold (recall 0.777 → 0.292 across the
bands), but Phases 0–1 show distance acting as a **stressor on failure families this taxonomy
already prices, not as a new category of failure**: 62% the σ/representation family, 30% the
`sub_threshold` continuum, 8% genuine absence. The implied lever is therefore decoder- and
representation-side — target σ, peak spacing, threshold calibration, and Phase 2's scale question
for the residual — **not far-field training vocabulary**; and multi-view remains the one remedy
that sidesteps all three mechanisms at once, by re-presenting the same ramp near-field.

