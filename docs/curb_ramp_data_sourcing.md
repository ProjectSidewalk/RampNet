# Curb-ramp data sourcing: candidate cities for a larger Stage 1 corpus

Working notes for [#59](https://github.com/ProjectSidewalk/RampNet/issues/59) — "would more
training data buy recall?" This document answers the narrower, prior question: **which cities could
we source from, how much would each buy, and what would it cost to retrain?**

Live counts were verified on **2026-07-30** by querying each publishing agency's own API; the
endpoint is recorded per row so every number is re-derivable. Counts drift (several of these refresh
weekly), so treat them as a snapshot.

**Nothing here has been acted on.** No city has been sourced, no location-precision assessment has
been run on any new city, and no retrain has been attempted. This is pre-work.

> **Read §2 first.** The paper already assessed location precision for eight cities, and that
> assessment — not inventory size — is the binding constraint on everything below.

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

| witness | raw | by chance | **excess** |
| :--- | ---: | ---: | ---: |
| gemini-3.1-pro-preview | 46 | 9.4 | +36.6 |
| gemini-3.6-flash | 33 | 8.1 | +24.9 |
| molmo2-8B | 26 | 8.5 | +17.5 |
| Qwen3-VL-8B | 22 | 6.1 | +15.9 |
| Qwen3-VL-32B | 16 | 2.7 | +13.3 |
| **union, 5 sparse models** | **69** | 30.0 | **+39.0** |
| *union, 2 dense detectors* | *127* | *102.2* | *+24.8* |

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

```
python scripts/analysis/silent_witness.py --json-out analysis_out/silent_witness.json
python scripts/analysis/miss_gallery.py --bucket silent \
    --queue analysis_out/silent_witness.json --render analysis_out/gallery46_silent
python scripts/analysis/make_tagger.py analysis_out/gallery46_silent
# open analysis_out/gallery46_silent/tagger.html
```

That yields **50 crops** — 59 unwitnessed, less 9 below the 30-source-pixel floor, which are
excluded from any rate rather than labelled. The verdict scheme is built so exactly one answer is
sourcing-addressable:

| verdict | what it means | programme |
| :--- | :--- | :--- |
| `visible` | clear ramp, unobstructed | **vocabulary — this is the sourcing target** |
| `occluded` | vehicle, pole, vegetation, person | capture |
| `lighting` | deep shadow or blown highlight | capture |
| `surface` | debris, snow, leaves, construction | environment |
| `not-a-ramp` | no ramp / flush or blended transition | GT disagreement |
| `unclear` | cannot tell from this imagery | excluded from every rate |

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

Script: `scripts/analysis/farfield_forensics.py` (21 tests); result JSON
`analysis_out/farfield_forensics.json`. Committed inputs only — the low-floor caches, the witness
list, the gallery manifest and verdicts, and the imagery manifests' `width` fields. No GPU, no
network, no imagery.

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
| 40–150 m | 72 | 0.292 | 22 | 9 | 9 |
| clamp ≥ 150 m | 5 | 0.200 | 3 | 0 | 0 |

- **Matched-size detection rate**: for each `visible` miss, the model's recall over all far-field
  GT within ±20% of that miss's apparent size is **median 0.57** (q1 0.31, q3 0.74). A hard pixel
  floor would put these near zero.
- **AUC(far-hit px vs far-silent px) = 0.718** — size matters, but it is far from deciding.
- Recall declines **0.777 → 0.549 → 0.292** across the bands. Even at 40–150 m the model finds
  roughly 3 in 10 (and the pooled 25–40 m rate agrees with E1's gold-set 0.49 at the same range).

**Far-field failure is graded sensitivity, not a cliff.** A silent far-field miss is not a ramp
below a physical detection floor — it is the unlucky tail of a process that succeeds on most
same-sized ramps. That is consistent with `docs/detection_recall_analysis.md`'s sensitivity
finding and with the human verdicts, and inconsistent with reading "more examples do not add
pixels" as a claim about *reachability*. (As a claim about pixels it remains true; the error was
inferring unreachability from it.)

### Phase 1: attenuated or absent? Almost never absent

`silent` is a statement about **peaks** — no `peak_local_max` peak ≥ 0.05 within the match radius.
Phase 1 makes the statement about the **heatmap**: `scripts/analysis/silent_activation.py`
(14 tests) loads the published checkpoint (`projectsidewalk/rampnet-model` — the weights every
committed cache came from), runs one pass per panorama holding a silent miss (single-pass fp32,
matching `op_cache`), and reads the max heatmap value inside the match radius. The scaled matcher
space *is* the 512×1024 heatmap grid, so the window is exactly the matcher's. Result JSON:
`analysis_out/silent_activation.json`; run on the local RTX 3070, all 128 pooled silent misses.

| population | n | act q1 / med / q3 | act ≥ 0.01 |
| :--- | ---: | :---: | ---: |
| near / rated | 13 | 0.009 / 0.099 / 0.197 | 9 |
| near / witnessed | 32 | 0.033 / 0.211 / 0.592 | 30 |
| far / rated | 37 | 0.022 / 0.076 / 0.409 | 34 |
| far / below-floor | 9 | 0.042 / 0.194 / 0.381 | 8 |
| far / witnessed | 37 | 0.045 / 0.188 / 0.615 | 37 |
| **all silent misses** | **128** | 0.032 / 0.136 / 0.548 | **118** |

What that in-window mass *is* (classes are act ranges; the offset and nearest-peak columns
confirm the intended reading rather than define it):

| class | definition | n | near / far | rated `visible` | argmax offset med | nearest floor peak med |
| :--- | :--- | ---: | :---: | ---: | ---: | ---: |
| **absent** | act < 0.01 | **10** | 6 / 4 | 5 | 22.0 px | 77.5 px (3.4 R) |
| **faint local** | 0.01 ≤ act < 0.05 | 39 | 12 / 27 | 13 | **10.2 px** | 85.6 px (3.8 R) |
| **tail** | act ≥ 0.05 | 79 | 27 / 52 | 23 | 22.3 px | **31.1 px (1.4 R)** |

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
- The strict per-pano null (azimuth-randomized at the site's elevation, self-excluding within 2 R)
  passes 31/128 at its p95 — a deliberately hard bar, since the p95 is set by the pano's strongest
  modes; the decomposition above is the sharper lens.

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

## 1. The current training corpus is mostly one city

Stage 1 is built from three cities' open-government inventories (`docs/data_provenance.md` §1).
Counting source inventories rather than derived panos, at today's live counts:

| Training city | Ramp records (live) | Paper Table 1 | Share |
| :--- | ---: | ---: | ---: |
| New York City, NY | 217,679 | 217,680 | **78.2%** |
| Portland, OR | 46,065 | 45,324 | 16.5% |
| Bend, OR | 14,800 | 13,611 | 5.3% |
| **Total** | **278,544** | 276,615 | |

Yielding **214,376 panoramas / 849,895 labels**, split 70/20/10 with a **150,063-panorama train
split** (paper §4.2).

**NYC is 78.2% of the training ramps.** In inventory terms RampNet is largely an NYC curb-ramp
detector with some Portland and a little Bend. That bears on the failure modes the benchmark keeps
surfacing — Paterson's paired tactile surface indicators, Gainesville's large diagonal ramps into
wide arterials (`docs/model_comparison.md`) — neither of which is NYC design vocabulary. The lever
this corpus is missing is plausibly **composition, not volume**.

The live counts validate cleanly against the paper's Table 1 (NYC differs by 1; DC below is
*identical*), which is a useful cross-check on the query method used throughout this document.

### Ramps → panoramas calibration

278,544 ramp records produced 214,376 panoramas:

> **≈ 0.77 panoramas per government ramp record** (inclusive of the ~20% negative panoramas)

The ratio runs **higher in sprawl cities** (ramps sparser, fewer share a panorama) and **lower in
dense grids**. A planning heuristic from three cities, not a law.

## 2. Location precision is the binding constraint, and the paper already measured it

**Paper §3.1, Table 1.** Eight government datasets were assessed by **manually overlaying curb-ramp
locations on an aerial imagery base map** and judging whether coordinates land on the physical ramp
(Fig. 2). One city rated Poor, four OK, three Good.

> *"For our purposes, we use all data from the good category: New York City, NY; Portland, OR; and
> Bend, OR — all which offer precise and diverse curb ramp styles."*

**The selection rule was "Good only."** The current corpus is not three cities because three were
available — it is three cities because **only three passed**.

| City | Paper count | **Precision** | Status |
| :--- | ---: | :--- | :--- |
| New York City, NY | 217,680 | **Good** | in training |
| Portland, OR | 45,324 | **Good** | in training |
| Bend, OR | 13,611 | **Good** | in training |
| Los Angeles, CA | 91,759 | OK | assessed, unused |
| Austin, TX | 48,995 | OK | assessed, unused |
| Washington, DC | 34,859 | OK | assessed, unused |
| Nashville, TN | 18,285 | OK | assessed, unused |
| **Seattle, WA** | 45,653 | **Poor** | **assessed, rejected** |

### ⚠️ Seattle is rated Poor — do not treat it as the obvious first add

Seattle looks like the ideal candidate on every other axis: the highest scorer in Deitz's 178-city
sample (13/14), a weekly refresh, ~46k records, rich attributes (condition, width, install date),
and it is **already contamination-burned** via the crop model, so training on it would cost no
evaluation ground. An earlier draft of this document recommended it first on exactly that reasoning.

**Table 1 rates its location precision Poor**, which is the one criterion that disqualifies a
dataset for Stage 1 — the pipeline projects the government coordinate onto the panorama, so a
misplaced coordinate produces a misplaced label. Rich attributes do not compensate.

Seattle may still matter for [#86](https://github.com/ProjectSidewalk/RampNet/issues/86) (condition
and width supervision does not need the point to be pixel-accurate), and the rating predates several
years of weekly updates so a re-assessment is defensible. But **it cannot be the default first add**,
and any argument for it has to start by contesting Table 1.

### What this reframes

The candidate pool splits three ways, and only the third has real upside:

- **Good (276,615)** — already fully used. There is no unexploited good data in Table 1.
- **OK (193,898)** — LA, Austin, DC, Nashville. Available, assessed, deliberately not used. Taking
  them is a **quality step-down the paper chose not to make**, and doing so should be an explicit,
  recorded decision rather than a side effect of chasing a number.
- **Unassessed** — every city in §3 that is not in Table 1. This is where the upside is, and none of
  it is known-good yet.

## 3. Candidate inventories not in Table 1 (unassessed)

Live counts, 2026-07-30. **None has had its location precision assessed.** Ordered by size.

| Jurisdiction | Ramp records | Endpoint | Notes |
| :--- | ---: | :--- | :--- |
| **VDOT** (Virginia, statewide) | **83,000** | `services.arcgis.com/p5v98VHDX9Atv3l7/…/ADA_Curb_Ramp_Condition_FS_9_View` | State highway ROW only. ⚠️ **includes Richmond** — see §6 |
| **Denver, CO** | **72,770** | `services1.arcgis.com/zdB7qR0BtYrg0Xpl/…/ODC_TRANS_CURBRAMPS_P` (layer **228**) | Largest city inventory found. ⚠️ *"delineated from 2022 aerial imagery"* — derived; recent, but verify it is per-ramp |
| **San Francisco, CA** | **50,096** | `services.arcgis.com/Zs2aNLFN00jrS4gG/…/Curb_Ramps_from_DataSF_pulled_weekly_` | Mirrors DataSF, **pulled weekly** |
| **WisDOT** (statewide) | ~49,000 | `data-wisdot.opendata.arcgis.com` | *Not queried.* ⚠️ 2014/15 **desktop** inventory from photo log + satellite |
| **NYSDOT** (statewide) | **42,297** | Socrata `data.ny.gov/resource/hmbc-hni2` | State ROW. NYC already in training — de-dup needed |
| **Charlotte, NC** | **40,601** | `gis.charlottenc.gov/…/CDOT_ADA/ADA_Curb_Ramps/MapServer/0` | From Charlotte's ADA self-evaluation |
| **CDOT** (Colorado, statewide) | **24,549** | Socrata `data.colorado.gov/resource/sb9m-ecvv` | State ROW only |
| **Boston, MA** | **24,022** | `gisportal.boston.gov/…/Infrastructure/OpenData/MapServer/3` | *Pedestrian Ramp Inventory*, BostonGIS |
| **Sioux Falls, SD** | **19,977** | `gis.siouxfalls.gov/…/Data/Transportation/MapServer/15` | Remarkable for a 200k city — matches Deitz's account of a post-FHWA-complaint data programme |
| **Minneapolis, MN** | **18,447** | `services.arcgis.com/afSMGVsC7QlRK1kZ/…/Minneapolis_ADA_Ped_Ramps_-_View_Layer_` | Consistent with the 17,800 in their 2020 ADA Transition Plan |
| **Arlington, VA** | **10,342** | `arlgis.arlingtonva.us/…/Open_Data/od_Sidewalk_ADA_Ramps` | |
| **Raleigh, NC** | ~14,550 | City pedestrian facility assessment (PDF) | *Reported in a study document, not queried* |
| Spokane WA · Tacoma WA · Dallas TX | exists, not pulled | ArcGIS Hub / city portals | Confirmed to publish a layer |
| **Phoenix, AZ** | gated | CurbPHX | Comprehensive, rich attributes (width, curb height, surface condition) — **behind a city login**. Request-access candidate |
| **Chicago, IL** | none published | — | See below |
| **Atlanta, GA** | 4,517 (wrong polarity) | `services2.arcgis.com/zLeajbicrDRLQcny/…/MAF_Missing_ADA_Ramps_Draft` | See below |
| Columbus OH · Houston TX · Des Moines IA · Pittsburgh PA · Oakland CA | none found | — | See below |

**Unassessed city total: ~236,000 ramps** (Denver, SF, Charlotte, Boston, Sioux Falls, Minneapolis,
Arlington). Plus ~199,000 across four state DOTs.

### Negative results

- **Chicago** would have been the highest-value candidate — CDOT states **137,000+ ramps installed
  since 2006**, and Chicago is **already contamination-burned**, so it would have been free. But that
  is a program statistic on a department page; the Socrata portal returns only crash data. **The
  ramps exist, the count is public, the geolocated inventory is not.**
- **Pittsburgh** — also contamination-burned, also unavailable. WPRDC carries steps, signalized
  intersections and centerlines but no ramps; the regional SPC layer's host refused connection; the
  two ArcGIS hits are vendor consulting deliverables (913 features). Worth one email to WPRDC.
- **Columbus** (score 9, contamination-burned) — only a *UIRF Planned Projects* layer.
- **Houston** — sidewalk permits, a sidewalk asset layer, service areas. No ramp inventory.
- **Des Moines** — sidewalk centerlines only. **Oakland** — Socrata dataset returns 4 records.

### Atlanta publishes *missing* ramps — wrong polarity for Stage 1, right one for #86

Atlanta's *Moving Atlanta Forward Draft ADA Ramp Installation Locations* holds **4,517 features**,
but they are locations where a ramp is **planned because none exists**. Useless for Stage 1.

Potentially valuable elsewhere: **absence ≠ negative** is named in #86 as one of RampNet 2.0's hard
parts, and it is the mechanism behind this issue's label-ceiling argument — we cannot distinguish
"no ramp here" from "a ramp we failed to record." A municipally-attested list of corners that
**lack** a ramp is **confirmed-absence supervision**: true negatives with provenance. Opportunistic
find; not part of any route below, and "Draft" should be taken seriously.

### On search method: terminology is the trap

Naming is wildly inconsistent — *Curb Ramps* (Seattle, Portland), *Pedestrian Ramp Locations* (NYC),
*ADA Ped Ramps* (Minneapolis), *Pedestrian Ramp Inventory* (Boston), *Access Ramps* (LA), *Sidewalk
ADA Ramps* (Arlington), *sCurbRamps* (Bend).

An earlier pass of this document searched titles for "curb ramp" only, concluded supply was thin,
and recommended verifying rather than searching further. **That was wrong** — a title search for
"curb ramp" does not match NYC's own dataset. Re-running across *pedestrian ramp*, *ADA ramp*, *ped
ramp*, *access ramp* and *curb cut* surfaced Denver, San Francisco, Boston, Sioux Falls, Nashville
and Arlington in a single pass, adding ~195k ramps. Anyone extending this list should search the
synonym set, not the phrase.

## 4. Deitz et al. (2021) — already cited by the paper as [10]

[Deitz, Lobben & Alferez 2021](https://doi.org/10.1177/20539517211047735) scored 178 US
municipalities on 14 accessibility data features. **The RampNet paper already draws on it** (§3.1:
"of the 178 US cities studied in [10], 90% published open street data but only 34% had sidewalk data
and far fewer (10%) included curb ramps").

Useful additional structure for candidate generation:

- Curb-ramp data appears **only in municipalities scoring ≥7** of 14 — sole exception Los Angeles
  (score 4, flagged incomplete). Their Table 9 (cities ≥6) is effectively the candidate pool, and it
  is how Denver, Boston, SF, Sioux Falls, Madison, Nashville and Arlington were located here.
- **Seattle is the highest scorer in the sample (13/14)** — a reminder that open-data maturity and
  *coordinate precision* are different things, since Table 1 rates it Poor.

Three limits: the portal review ran **June 2019 – March 2020**; the sample is large municipalities
only, so **Bend is not in it at all**; and portal-based discovery misses the ~1,469 ArcGIS items,
including state DOTs.

A **1-hit-in-6 test of cities below the cutoff** (Minneapolis, Des Moines, Houston, Phoenix, Atlanta,
Pittsburgh) found only Minneapolis — which itself scores 7, i.e. above the line. Weak evidence
(n=6), but it points the same way as the paper: hunting below the cutoff has poor yield, while the
score-≥7 pool is productive.

## 5. The location-precision assessment (the critical path)

The paper's method is **qualitative**: overlay coordinates on aerial imagery, judge alignment
visually, bucket Good / OK / Poor. No thresholds are published. Replicating it for a new city is
cheap — no GPU, no pipeline run, no sourcing commitment — and it is the **gate every unassessed city
in §3 must pass**.

Worth quantifying while extending it, so the tiers stop being a judgment call:

| Check | Why it matters | Test |
| :--- | :--- | :--- |
| **Positional offset** | The Stage 1 mechanism. Fig. 2b is the failure. | Sample ~50 points/city, measure metres from the true ramp on aerial imagery; report a distribution, not a bucket |
| **Per-ramp vs per-corner** | If a city records one point per *corner*, paired ramps collapse to one label — **the supervision gap behind Paterson's failure** | Records per intersection; inspect a paired corner. NYC's ~1.8/intersection implies per-ramp |
| **Staleness** | DC's live count is *identical* to the paper's, corroborating a static 2016 capture. Ramps built since are missing (recall loss); removed ramps are phantom labels (precision loss) | Compare capture date to GSV capture date |
| **Completeness** | The label-recall term that sets the ceiling. Stage 1 agreement is **P .9403 / R .9245** | Spot-check N intersections in GSV for ramps absent from the inventory |
| **CRS / datum** | A wrong projection silently shifts a whole city | Round-trip known points |
| **Active vs retired** | Seattle publishes 46,386 total but 38,468 active | Prefer the publisher's active filter |

A quantitative offset distribution would also let the OK tier be re-examined: "OK" may mean 2 m or
8 m, and that difference plausibly decides whether LA's 91,759 records are usable.

**One discrepancy to resolve.** Table 1 rates **LA "OK"** at 91,759, but LA's published Access Ramps
layer is documented as *"the geographic center of corner polygon features, with attributes indicating
the presence or absence of a wheelchair access ramp"*, derived from 2014 aerial imagery — i.e.
corner-centroid, not ramp-located. Either they assessed a different layer or "OK" tolerates
corner-level placement. Worth settling before counting LA's 91,759.

## 5a. Temporal distance is a second, independent gate — and it is not the #11 check

Positional precision asks *is the coordinate on the ramp?* Temporal distance asks *did the ramp and
the pixels exist at the same time?* They are independent, and a city can pass one and fail the other.

**This is not covered by the existing temporal-consistency filter.** [#11](https://github.com/ProjectSidewalk/RampNet/issues/11)
(closed) fixed two real bugs in that filter — a `2000-01-01` sentinel that let undated records bypass
the check, and a month comparison that ignored the year. What it now does correctly is an
**ordering** test: discard a panorama unless the ramp was installed strictly before the month of
capture (`generate_dataset_meta.py`, `(year, month)` tuple compare). Ordering is not distance, and
three failure modes survive it:

1. **Inventory older than the imagery → unlabeled positives.** Ramps built after the inventory
   snapshot are visible in the panorama but absent from the government data, so no ordering check can
   see them. The target heatmap is **zero** at a real ramp — the label-ceiling mechanism this issue
   is built around. **DC is the worst case in hand:** its live count is *identical* to the paper's
   Table 1 figure, corroborating a static 2016 capture, against GSV imagery a decade newer.
2. **Undated records → phantom labels.** `TREAT_UNDATED_AS_PREDATING = True` is a deliberate,
   documented choice, but it means every dateless record is *assumed* to predate the panorama. Where
   a city's undated fraction is high the filter is effectively off for those records, and ramps not
   yet built when the pano was captured become labels at empty pixels. **The undated fraction is
   therefore a per-city measure of how much of the filter actually functions**, and it has never been
   reported for any city.
3. **Rebuilt or removed ramps** change appearance or vanish between the two dates, and carry no
   signal either way.

Note the two errors have **opposite signs**: (1) suppresses detections at real ramps, (2) trains
detections at nothing. Both degrade the corpus, so a city's *net* temporal exposure is not captured
by any single number — report both.

**This is cheap to measure and needs no human and no imagery** — install-date distribution, undated
fraction, inventory snapshot date, and the GSV/Mapillary capture-date distribution are all metadata.
It should therefore run *before* the visual precision assessment (§5), because it can disqualify a
city for a few minutes of compute rather than a few hours of review.

### The imagery half is already collected: Streetscape Tracker

[Streetscape Tracker](https://github.com/jonfroehlich/streetscape-tracker) (`D:\Git\gsv-tracker`,
dashboard at `makeabilitylab.cs.washington.edu/public/streetscape-tracker/`) samples a frozen
geographic grid per city and records **`capture_date` per panorama** for both GSV and Mapillary, in
immutable dated snapshots. Its catalogue holds **1,204 cities / 1,863 snapshots** on makelab1
(`/projects/makeabilitylab/streetscape-tracker/data/`).

Coverage of this document's candidates is good, and critically **includes all three training
cities**, so the assessment can be calibrated against known-Good imagery:

- **Tracked** — Austin, Bend, Boston, Denver, Los Angeles, Minneapolis, Nashville, New York,
  Portland OR, Seattle, Sioux Falls, Washington DC. Also the Gainesville, Paterson and Richmond
  benchmark splits.
- **Not tracked — worth queueing**: **Charlotte NC, San Francisco CA, Arlington VA** (only Arlington
  TN and Arlington WA are in the catalogue).
- Several tracked cities have stale runs (DC 2024-04-11, Seattle 2024-12-19, Portland OR
  2024-12-21) and would benefit from a re-run.

**First measurement (2026-07-30), GSV capture-year distribution vs inventory date:**

| City | Precision (Table 1) | Inventory vintage | Modal imagery years | Gap |
| :--- | :--- | :--- | :--- | :--- |
| Bend, OR | **Good** (training) | rolling / live | **2024** (83% of 189k) | ~0 yr |
| Austin, TX | OK | live API | **2024–25** (73% of 2.0M) | ~0 yr |
| Denver, CO | unassessed | 2022 aerial delineation | **2022–24** (70% of 2.3M) | ~0–2 yr |
| **Washington, DC** | OK | **2016 static capture** | **2022–23** (63% of 159k) | **~6 yr** ⚠️ |

**DC is confirmed as the worst temporal case in the pool** — its inventory predates the modal
imagery by roughly six years, so every ramp DC built between 2016 and 2023 is in the pixels and
absent from the labels. **Denver reads better than its "aerial-derived" flag suggested**: a 2022
delineation against 2022–24 imagery is nearly contemporaneous, which is the best temporal alignment
of any unassessed candidate.

**Caveat on using the tracker this way.** Its GSV series is a *grid sample* (nearest pano per grid
point) spread uniformly over the city, whereas Stage 1 selects panoramas within 10 m of curb-ramp
locations — i.e. concentrated at intersections. The distribution above is therefore a **screening**
signal, good for ranking cities, not a substitute for the per-ramp Δt the pipeline would actually
see.

### The gate is implemented: `scripts/analysis/temporal_gap.py`

Reports the two exposures **separately** — they have opposite signs, so the script deliberately
emits no net "gap" number. It also reproduces the pipeline's ordering check (including the #11
regression) so the filter's data cost is measured rather than assumed, handles the format zoo
(ArcGIS epoch-milliseconds, Socrata ISO strings, bare years) and treats the `2000-01-01` sentinel as
undated while reporting it separately. Pure core, unit-tested in `tests/test_temporal_gap.py`
(21 tests); no GPU, no network, no imagery.

```
python scripts/analysis/temporal_gap.py --city "Washington, DC" \
    --inventory dc.json --inventory-date-field INSTALLDATE \
    --tracker-snapshot washington--district-of-columbia--...2024-04-11.csv.gz \
    --snapshot-date 2016-01
```

### First city assessed: DC is disqualified, and worse than "stale"

Querying DC's layer schema and a `groupBy` over its inspection year settles it:

- **There is no install-date field.** The layer carries `CONDITION`, `YEAR_INSPECTED`,
  `ESTIMATED_YEAR_OF_IMPROVEMENT`, `CREATED_DATE` and `LAST_EDITED_DATE` — record-keeping timestamps
  and a *future* improvement year, but nothing recording when a ramp was built. **DC's undated
  fraction is 100%**, so `TREAT_UNDATED_AS_PREDATING = True` waves all 34,859 records through and
  **the ordering filter is entirely inert for this city.**
- **`YEAR_INSPECTED` is 2016 for all 34,859 records, with no other value.** A single one-shot survey,
  never updated — which confirms the 2016 snapshot empirically rather than inferring it from the
  record count matching the paper's Table 1.

So DC maximises both exposures at once: a decade of unrecorded construction (unlabeled positives)
against a filter that cannot reject anything (phantom labels). **Drop it from every route in §6.**
It was carrying 34,859 ramps in the "+ all OK" tier, so that tier's 470,513 becomes **435,654** —
pushing 500k further out of reach on assessed data.

Worth noting for [#86](https://github.com/ProjectSidewalk/RampNet/issues/86) even so: DC publishes a
`CONDITION` field, which is condition supervision we currently discard at ingest. Being unusable for
Stage 1 localisation does not make a dataset unusable for everything.

**Correction to an earlier commit on this branch.** That commit said DC "maximises both exposures at
once". Only one of the two. DC's `YEAR_INSPECTED = 2016` is an **existence bound** — every record
demonstrably existed in 2016, and the imagery is 2022-23, so no DC record can be un-built at capture
time and **phantoms are structurally impossible**. DC's exposure is entirely one-sided: unlabeled
positives. The generalisation is in §5b, and the tool now models it.

### 5b. What bounds phantoms is the existence date, not the install date

A ramp audited in 2016 demonstrably existed in 2016, whatever its install field says. So for any
panorama captured after a record's **existence bound** — an audit date, an inspection year, or the
vintage of the aerial imagery a layer was delineated from — a phantom label is *structurally
impossible*, however many records are undated.

This matters far more than it first appears, because **install-date coverage turns out to be poor
almost everywhere** (§5c). Three of the seven cities assessed have no install-date field at all.
Were the install date the operative mechanism, most of the candidate pool would be unusable. It is
not: every one of these inventories carries *some* existence evidence, and that is the quantity the
gate should be built on.

It also reframes `TREAT_UNDATED_AS_PREDATING = True`. As a blanket assumption it is unjustified. As
a *derived consequence* of "the whole inventory was surveyed before this imagery was captured", it
is exactly right — and checkable per city.

### 5c. Six cities assessed (2026-07-30)

Schema and histogram queries against each publisher's API; GSV capture years from Streetscape
Tracker. Nothing here required a GPU, a pipeline run, or a human reviewer.

| City | Ramps | Install-date field | **Undated** | Existence bound | Median capture | Verdict |
| :--- | ---: | :--- | ---: | :--- | :--- | :--- |
| **Denver** | 72,770 | *none* | 100% | **2022** aerial delineation | 2022 | ✅ near-contemporaneous |
| **Sioux Falls** | 19,977 | `INSTALLDATE` / `InstallYear` | **37.6%** | `INSPECTEDDATE` | 2024 | ✅ best date coverage found |
| **Minneapolis** | 18,447 | `YearBuilt` | 67.9% | `stamp_date` | 2022 | ✅ usable |
| **Austin** | 49,796 | `YEAR_BUILT` | **84.6%** | `ASSESSMENT_DATE` | 2024 | ⚠️ large but date-poor |
| **Nashville** | 18,388 | *none* | 100% | `DateAudited`, spanning **1998–2025** | 2022 | ⚠️ bound spans 26 yr |
| **Boston** | 24,022 | `CONST_DATE`, **all `18991230`** | **100%** | `INSP_DATE`, **2007–2010** | 2022 | ❌ ~12-year gap |
| *Washington, DC* | 34,859 | *none* | 100% | `YEAR_INSPECTED` 2016 | 2022–23 | ❌ ~6-year gap |

**Four findings, in order of how much they change the plan:**

1. **Install-date coverage is poor across the entire pool — this is systemic, not a DC quirk.** The
   *best* city found is 37.6% undated. Three of seven have no install-date field at all, and two
   more exceed 67%. `TREAT_UNDATED_AS_PREDATING` is therefore load-bearing for every candidate we
   would add, which makes §5b's existence bound the primary mechanism rather than a refinement.
2. **Boston is the worst candidate, not DC.** Its `CONST_DATE` column is uniformly the string
   `18991230` (the spreadsheet zero date) — the field exists and carries nothing — and its
   inspection survey ran **2007–2010** against median 2022 imagery. That is a **~12-year**
   one-directional gap, during which Boston has been actively building ramps.
3. **Denver is the strongest large candidate, and for the reason I had flagged against it.** Being
   *"delineated from 2022 aerial imagery"* is a positional-precision concern, but it is a temporal
   **strength**: it fixes a hard existence bound for the entire layer at 2022, against median 2022
   GSV capture. 72,770 ramps at near-zero temporal gap. The two gates genuinely pull in opposite
   directions here, which is the clearest argument for keeping them separate.
4. **Austin is the awkward one.** 49,796 ramps — the second-largest city inventory — but 84.6% of
   `YEAR_BUILT` is empty, so it rests entirely on `ASSESSMENT_DATE` as its bound. Worth resolving,
   because it is also the richest #86 source found (`RATING` A–F, `CURB_RAMP_TYPE`,
   `DETECTABLE_WARNING`, `DATE_CONSTRUCTION_COMPLETED`).

**Two parser defects that only real data exposed**, both of which would have silently corrupted
results rather than failing loudly:

- **Compact `YYYYMMDD` read as epoch milliseconds.** Boston's `"18991230"` parsed as 1970 — turning
  a null placeholder into a plausible install date, and making the city look temporally fine.
- **Typo years defining the snapshot.** Minneapolis carries a single `2926` among 18k records; the
  default snapshot used `max()`, so one row would have set the city's snapshot to the 30th century
  and zeroed its exposure. Now a 99th-percentile quantile.

Both are fixed and regression-tested. Boston's sentinel also generalised the placeholder handling:
the tool now knows `1899-12` alongside #11's `2000-01`, and flags any *unrecognised* implausibly-old
value that dominates an inventory, since every source invents its own null date.

### For #86: attribute richness is inversely related to nothing useful

Assessed as a side effect, since these schemas had to be read anyway. **Minneapolis is the richest
source found** — per-ramp running slope, cross slope, landing dimensions, ramp width, gutter
condition rating, plus a `Retired` flag. **Sioux Falls** carries `WIDTH`, `SLOPE`, `CROSSSLOPE`,
`COMPLIANCERATING`, `Condition` and a `PHOTO` reference. **Austin** carries an A–F `RATING`.
**Nashville** and **Boston** both carry extensive slope measurements.

This is measurement-and-condition supervision at a scale we do not currently collect, and it is
uncorrelated with Stage 1 usability — Boston is disqualified for #59 and still interesting for #86.

## 6. Routes to a 500,000-ramp corpus

**Be explicit about which 500k is meant:**

- **500k ramp *records*** → need **+221,456** on today's 278,544.
- **500k *panoramas*** → needs ~650k ramps at the 0.77 ratio. A much larger programme.

For the ramp target, tiering by Table 1 gives the decisive result:

| Pool | Ramps | Cumulative |
| :--- | ---: | ---: |
| **Good** (NYC + Portland + Bend) — already used | 276,615 | 276,615 |
| **+ all OK** (LA + Austin + DC + Nashville) | 193,898 | **470,513** ❌ |
| **+ unassessed cities** (Denver, SF, Charlotte, Boston, Sioux Falls, Minneapolis, Arlington) | ~236,000 | ~706,500 |
| **+ state DOTs** (VDOT, WisDOT, NYSDOT, CDOT) | ~198,800 | ~905,300 |

> **500,000 is not reachable on assessed data alone.** Good + *every* OK city reaches **470,513** —
> about 30k short — and that is already after accepting a quality tier the paper deliberately
> rejected. **Every route to 500k depends on cities whose location precision nobody has checked.**

That makes §5 the critical path, not discovery. Concretely: assessing the seven unassessed cities is
a few days of visual work with no compute, and it determines whether 500k is a real target or an
arithmetic one. If roughly 60% of that ~236k passes at Good, 500k clears comfortably on
city-inventory data alone, with no state-DOT tail.

### ⚠️ VDOT would burn the Richmond benchmark split

VDOT is the largest single source (83,000) and the most tempting one-step fix. It is **statewide
Virginia, which includes Richmond** — one of the nine benchmark splits and the Mapillary
out-of-distribution flagship. Training on it unclipped destroys that split's held-out status.

Mitigation: clip to exclude the Richmond deployment footprint (the same dissolved PS-regions polygon
the benchmark uses) before ingest. Feasible, but the failure mode is silent. **NYSDOT** needs the
same treatment for NYC overlap.

## 7. Retrain cost

Scenario: **+ DC, Austin, Charlotte, Minneapolis** (a mix of OK-tier and unassessed, sized to be
illustrative rather than recommended).

| | Ramps | Panos (× 0.77) | Train split | Steps (÷16) |
| :--- | ---: | ---: | ---: | ---: |
| Current | 278,544 | 214,376 | 150,063 | 9,379 |
| + four cities | ~422,000 | ~325,000 | ~227,500 | ~14,200 |
| Growth | +52% | **1.52×** | | |

- **Stage 2: ≳36 h on 16 L40s** for one epoch (≳580 GPU-h) — the paper trained 1 epoch on 16 L40s,
  batch 1 per GPU (VRAM-bound), and the README says it *"will take a very long time (> 24 hours)"*.
  **>24 h is a floor, not a measurement**; no exact wall-clock is recorded anywhere.
- **Stage 1 generation is the long pole and is entirely unmeasured** — ~111k new panoramas at 32
  tiles each ≈ **3.5M tile requests** against Google's undocumented endpoints, fetched 26 panoramas
  at a time. `run_download_dataset.slurm` allocates 100 h. Rate limiting is the dominant risk.
- **The crop model needs no retrain** — reusing it is also the cleaner experiment, since only the
  data changes.

Order of magnitude: **about a week of wall-clock**, wide error bars on the Stage 1 half. Add #84's
epoch curve and multiply.

## 8. Selection rule

**Train on cities you would never want as a benchmark split.** Every city added to training is
permanently disqualified as clean evaluation ground.

- **Already contamination-burned, so free** — Seattle, Columbus, Chicago, Pittsburgh, St. Louis,
  Knoxville and the other crop-model cities (`docs/data_provenance.md` §1). Painfully, **Seattle is
  rated Poor and the other three publish no inventory**, so this category currently yields nothing.
- **Registry-clean, so costly** — Austin, Charlotte, DC, Denver, SF, Boston, Sioux Falls,
  Minneapolis, Arlington. Training on them forecloses them as future splits. Acceptable, since none
  is among the current nine, but it should be deliberate.
- **Never add Paterson or Gainesville** — two of only three GSV benchmark splits, i.e. nearly all
  in-domain-imagery evaluation.

### Diversity, if precision permits

The composition argument in §1 favours cities that are *not* NYC-like and not Pacific-Northwest-like:
Austin, Charlotte and Nashville attack the Sunbelt/Southeast vocabulary gap behind the Paterson and
Gainesville failures. But **precision gates diversity** — a Good-rated bland city beats a Poor-rated
diverse one, because the Poor city's labels are wrong wherever they land.

## 9. Archive the inventories, because they drift

**The source inventories are not in this repo.** `stage_one/dataset_generation/location_data/` and
`street_data/` are neither present nor tracked — the README tells you to download them from live
portal links. `docs/data_provenance.md` §3 records that the exact NYC/Portland/Bend files used for
the paper are archived in the **paper's supplemental material**, so RampNet 1.0 is replicable, but
**not from this repository**, and not by following the README, which serves current data.

The counts collected here quantify how much that matters. Comparing paper Table 1 against the same
endpoints today:

| City | Paper (Tab. 1) | 2026-07-30 | Drift |
| :--- | ---: | ---: | ---: |
| **Bend, OR** | 13,611 | **14,800** | **+8.7%** |
| Portland, OR | 45,324 | 46,065 | +1.6% |
| Seattle, WA | 45,653 | 46,386 | +1.6% |
| Austin, TX | 48,995 | 49,796 | +1.6% |
| Nashville, TN | 18,285 | 18,388 | +0.6% |
| New York City, NY | 217,680 | 217,679 | −0.0005% |

Bend has grown **8.7%** — it is the smallest training city, so it drifts fastest in relative terms,
and it is 5.3% of the corpus. Anyone re-running Stage 1 from the README links today builds a
measurably different dataset from the paper's and has no way to detect the difference. (NYC's
one-record change is its own signal: that inventory is effectively frozen.)

**Recommendation: commit a dated snapshot of every inventory at ingest**, alongside the fetch URL
and query, the way `benchmark/*/records.jsonl` already pins benchmark inputs. These are point files
— tens of thousands of rows, a few MB gzipped — so the cost is trivial next to losing
reproducibility. That also makes the §5c numbers re-derivable later, since every count in this
document is a snapshot of a moving target.

Doing this for the RampNet 2.0 corpus is straightforward. Doing it retroactively for 1.0 means
recovering the three files from the paper's supplemental material and committing them, which is
worth doing while it is still easy: they are the only artifacts that make the published dataset
reproducible from source, and they exist in exactly one place.

## 10. Caveats

- **Counts are a 2026-07-30 snapshot**; several refresh weekly.
- **Only Table 1's eight cities have any precision assessment.** Everything in §3 is unassessed, and
  size says nothing about usability.
- **Record count ≠ ramp count.** Inventories may hold multiple records per ramp, or one per corner.
- **Two counts are secondhand** — WisDOT ~49,000, Raleigh ~14,550 — from documents, not APIs.
- **The 0.77 panoramas-per-ramp ratio comes from three cities**, two of them dense grids. It is the
  weakest link in §7.
- **Nothing here measures whether more data helps.** That is #59's E1–E3. This document only
  establishes what sourcing would cost if the answer is yes.
