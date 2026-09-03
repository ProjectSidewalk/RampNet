# Can the benchmark resolve Run B? (#135)

**Status: complete 2026-08-18. No GPU time was spent.** The short version: `manual_gold` can
resolve the effect Run B would plausibly produce, but only if the comparison is read **paired**;
**pooling the benchmark splits does not help**; and Run A's curve, re-read paired, turns out to
**decline measurably after epoch 6** rather than staying flat to epoch 8. The recommendation is
at the bottom.

## The question

Run A (#84) came back with a plateau: `manual_gold` max-F1 for epochs 2–8 spans 0.008,
inside the pre-registered 0.01 tie bar, so no epoch in that range separates from any other.
Run B is 30–60 epochs with cosine decay — 1,675–3,350 GPU-hours. The obvious risk is that it
returns *another* unreadable plateau, in which case the hours buy nothing either way.

So before submitting anything: **what effect size can this benchmark actually detect, and
does pooling the ten splits raise it?** Both are answerable from what is already in the repo.

## The answer, up front

| | s.e. on `manual_gold` max-F1 | detectable at 80% power |
| :--- | ---: | ---: |
| **Unpaired**, as the 0.01 tie bar assumes | 0.0042 | **0.0117** |
| **Paired, MEASURED** — Run A epochs, median | 0.0021 | **0.0059** |
| **Paired, MEASURED** — Run A epochs, range over 28 pairs | 0.0016–0.0029 | **0.0045–0.0081** |
| Pooling all ten splits, unpaired | 0.0039 | 0.0109 |

Three things follow.

1. **Pairing is worth 2–3×, and it is free.** Both checkpoints are scored on the same 1,000
   panoramas against the same ground truth, so the panorama-to-panorama difficulty that
   dominates the unpaired noise is common to both and cancels. Measured across all 28 pairs of
   Run A checkpoints, the gain is **2.0× on the median pair**.
2. **Pooling is worth 7%.** Ten splits together hold 6,560 instances against `manual_gold`'s
   3,919, and the MDE moves 0.0117 → 0.0109. That is not a lever.
3. **The tie bar is the wrong instrument for an epoch-vs-epoch comparison**, and using it cost
   Run A a result. Read paired, Run A's curve is not "a step then a plateau" — it is a
   plateau with a **measurable decline in its tail**. See "Run A's plateau, re-read" below.

An earlier draft of this document could only *bracket* the paired standard error between two
stand-in pairs, at [0.0043, 0.0135], because Run A's committed artifacts are aggregate PR
curves and carry nothing per-panorama. That bracket is now closed: the 2026-08-17 scoring's
heatmap cache survived on makelab2, so the per-panorama detections were recovered for all
eight epochs with **no GPU, no panorama images and no network**, and the standard error is
measured rather than bounded. The stand-in pairs are kept below as the sanity check they
turned out to be — they bracketed the answer correctly, and the upper one was conservative by
about 1.7×.

## What the benchmark actually holds

| split | panos | recall panos | GT instances | ramps/pano | GT source |
| :--- | ---: | ---: | ---: | ---: | :--- |
| annapolis | 125 | 125 | 294 | 2.35 | anchored |
| bend | 110 | 110 | 327 | 2.97 | anchored |
| budapest_district5 | 125 | 125 | 300 | 2.40 | anchored |
| clovis | 125 | 125 | 195 | 1.56 | anchored |
| gainesville | 125 | 125 | 272 | 2.18 | anchored |
| **manual_gold** | **1000** | **1000** | **3919** | **3.92** | **manual** |
| morgantown | 125 | 125 | 267 | 2.14 | anchored |
| paterson | 125 | 125 | 395 | 3.16 | anchored |
| richmond | 124 | 124 | 310 | 2.50 | anchored |
| sao_paulo | 125 | 125 | 281 | 2.25 | anchored |
| pooled cities (9) | 1109 | 1109 | 2641 | | anchored |
| pooled all (10) | 2109 | 2109 | 6560 | | mixed |

`manual_gold` is 60% of the instances on its own, and it is the only split whose ground truth
was labeled independently of RampNet. The nine city splits derive their GT from a human review
of RampNet's own detections, so they are RampNet-anchored — the caveat `rampnet/detection_eval.py`
and `docs/model_comparison.md` already carry.

## The unpaired noise floor, and where 0.01 came from

Cluster bootstrap, resampling **panoramas** (B = 20,000, seed 42), RampNet at the #54 operating
point of 0.30:

| split | F1 | s.e.(F1) | s.e.(R) | naive binomial s.e.(R) | design effect | MDE 80% |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| manual_gold | 0.9041 | 0.0042 | 0.0055 | 0.0045 | 1.46 | 0.0117 |
| paterson | 0.8053 | 0.0175 | 0.0259 | 0.0234 | 1.23 | 0.0492 |
| richmond | 0.8546 | 0.0182 | 0.0291 | 0.0240 | 1.47 | 0.0511 |
| annapolis | 0.8395 | 0.0218 | 0.0331 | 0.0256 | 1.67 | 0.0612 |
| sao_paulo | 0.7767 | 0.0281 | 0.0377 | 0.0277 | 1.86 | 0.0787 |
| budapest_district5 | 0.6442 | 0.0324 | 0.0383 | 0.0289 | 1.76 | 0.0908 |
| **pooled cities (9)** | 0.8036 | 0.0078 | 0.0113 | 0.0089 | 1.60 | **0.0219** |
| **pooled all (10)** | 0.8671 | 0.0039 | 0.0058 | 0.0047 | 1.56 | **0.0109** |

(The four splits omitted for width — bend, clovis, gainesville, morgantown — are in
`docs/data/benchmark_power_135.json`; none is an outlier.)

**Resampling panoramas rather than instances matters.** Ramps cluster within a panorama, so an
instance-level binomial understates the spread by the design effect, measured here at **1.23 to
1.88**, 1.46 on `manual_gold`.

The pre-registration derived the 0.01 bar from "≈0.008 s.e. on recall". That number is
`sqrt(0.25/3919) = 0.0080`, the binomial worst case at p = 0.5. At the recall RampNet actually
achieves it is `sqrt(0.9 × 0.1 / 3919) = 0.0048`, and the measured clustered value is **0.0055**.
So the bar was set conservatively, by about 45% — which was the right instinct given that the
clustering had not been measured, and it happens to land close to the right answer for the wrong
reason: 0.01 is very nearly the correct *unpaired* MDE of 0.0117.

## Pooling does not help

`manual_gold` alone gives an MDE of 0.0117. All ten splits pooled give 0.0109 — a **7%**
improvement — and the nine city splits *on their own* are worse than `manual_gold` alone
(0.0219), despite holding 1,109 panoramas to its 1,000. Two reasons: they carry 2,641 instances
to `manual_gold`'s 3,919, and F1 is lower there (0.80 vs 0.90), which puts more variance in
every count.

So pooling buys 7% of an MDE in exchange for mixing an independently-labeled ground truth with
nine RampNet-anchored ones. It is not worth it, and the #135 spec's instruction to evaluate
**per-split rather than pooled** is the right call for reasons beyond the ones it gave.

## The paired noise floor, and the mechanism

The quantity that governs a paired comparison is **discordance**: how many ramps one model finds
and the other misses. If `b` is the count A misses and B finds and `c` the reverse, the paired
difference in recall is `(b − c)/n` with variance `(b + c)/n²` — it does not depend on
`n·p·(1−p)` at all. That is why two similar checkpoints can be separated far more finely than
the unpaired bar suggests, and it is measurable without running anything.

On `manual_gold`, for the three real detector pairs the repo can build from committed data:

| pair | ΔF1 | s.e.(ΔF1) | unpaired s.e. | gain | ΔmaxF1 | s.e. | b | c | discordance |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RampNet TTA − RampNet single-pass | +0.0022 | 0.0017 | 0.0060 | 3.6× | +0.0049 | 0.0015 | 78 | 3 | 2.1% |
| y11x_pano_h200 − y11l_pano | +0.0120 | 0.0034 | 0.0086 | 2.6× | +0.0244 | 0.0029 | 165 | 152 | 8.1% |
| y11x_pano_h200 − y26_pano | +0.1119 | 0.0043 | 0.0079 | 1.8× | +0.1492 | 0.0048 | 233 | 123 | 9.1% |

Read the first row as the near-identical end of the range and the third as the far end. These
three pairs were originally the whole answer: the comparison Run B would actually make — two
Stage 2 checkpoints of one lineage — had no per-panorama data, so the paired s.e. could only be
bracketed by pairs chosen to straddle it. That is no longer necessary, and the measurement is
below. The brackets are kept because they turned out to be correct, which is worth knowing the
next time only a bracket is available.

### The real thing: every pair of Run A checkpoints

`dump_peaks_from_cache.py` recovers per-panorama detections from `evaluate.py`'s heatmap cache,
so all 28 pairs of Run A checkpoints can be compared on the same 1,000 panoramas directly.

| | s.e.(Δ max-F1) | MDE 80% | discordance |
| :--- | ---: | ---: | ---: |
| adjacent epochs (gap 1) | 0.0016–0.0023 | 0.0045–0.0064 | 2.5–4.2% |
| **gap ≥ 3 — the Run B analogue** | **median 0.0022** | **0.0063** | 3.3–6.6% |
| across all 28 pairs | 0.0016–0.0029 | 0.0045–0.0081 | 2.5–6.6% |

**s.e.(Δ max-F1) for two Stage 2 checkpoints is 0.0016 to 0.0029 — measured, not assumed — so
the MDE is 0.0045 to 0.0081.** The earlier bracket [0.0043, 0.0135] contained it; the working
assumption of "4–6% discordance ⇒ 0.006–0.009" was almost exactly right.

**The standard error grows with epoch separation, and that matters for planning.** Discordance
runs 2.5–4.2% for adjacent epochs and 6.5–6.6% for the widest pair available (1 vs 7, 1 vs 8),
with s.e. tracking it from 0.0016 to 0.0029. A Run B checkpoint at epoch 30 under cosine decay
is further from any Run A checkpoint than any pair measured here, so **plan the Run B comparison
at s.e. ≈ 0.003 and an MDE of ≈ 0.008**, not at the median. That is an extrapolation of a
measured trend rather than a measurement, and it is flagged as such.

The middle proxy row is the one worth dwelling on. `y11x_pano_h200` versus `y11l_pano` is #51's
annealed-tail result — the measurement cited in #135 as the argument *for* running Run B. On
`manual_gold` that difference is **+0.0244 max-F1 at s.e. 0.0029, an 8σ effect**. Whatever else
is uncertain, an effect of *that* class is comfortably readable here. (Two caveats travel with
it: the two arms differ in architecture as well as budget, so the margin is not attributable to
the anneal alone; and the sign flips on 4 of the 9 city splits, which is a #51 finding in its own
right and not one this analysis pursues.)

### Instrument check: one of those pairs disagrees for the wrong reason

#132 established that the committed `analysis_out/op_cache` detections were extracted with
`skimage.feature.peak_local_max`'s default `exclude_border`, so they hold **no** detections at the
panorama seam. `benchmark/*/records.jsonl` does. The RampNet-vs-RampNet row above straddles those
two sources, so part of its disagreement is that artifact rather than a model difference. Measured:

| pair | disagreements within 0.03 of the seam | baseline |
| :--- | ---: | ---: |
| RampNet TTA − RampNet single-pass | **24 of 81 (30%)** | 2.1% |
| y11x_pano_h200 − y11l_pano | 7 of 317 (2%) | 2.1% |
| y11x_pano_h200 − y26_pano | 9 of 356 (3%) | 2.1% |

A 30% seam share against a 2.1% baseline is a 14× enrichment; the two pairs drawn from a single
source read the baseline exactly, which is the control that says the diagnostic is not firing
spuriously. Net: the RampNet self-pair's true discordance is nearer **1.5%** than 2.1%, so its
s.e. is an *over*estimate and the lower end of the bracket is, if anything, conservative. The
check is part of the committed script rather than a note, so it re-runs on every future pair.

Two smaller confirmations that the matcher is behaving: the self-pair's `b = 78, c = 3` is
almost perfectly one-sided, which is exactly what flip-TTA's max-combine mechanism predicts
(pointwise ≥, so it can only add detections) and what #78 documented; and the +0.0022 F1 it
buys on `manual_gold` reproduces #78's finding that this is the one split where TTA helps.

### Re-scored after #140 sealed the seam, and what that moved

Everything above was first computed against the **pre-#140** matcher: this analysis branched
before the seam wrap landed. Merging `main` in surfaced that immediately, and by the intended
route — `score_model` asserts on every panorama that its own `match_detail` agrees with
`rampnet.detection_eval.score_pano`, and the assertion fired, because `score_pano` had started
wrapping and `match_detail` had not. Two matchers exist here precisely so that a divergence is
an error rather than a slightly different number, and that is what happened.

`match_detail` now wraps, and both the greedy match and the ignore-point fallback go through
`rampnet.geometry.dist_sq` rather than an inline distance — the same defect #132 §4 found in
`score_pano` itself.

**Measured, the seam is worth almost nothing to this analysis.** Re-scoring all eight committed
epoch dumps with the wrapping matcher moves **one** number:

| epoch | max-F1, pre-#140 | max-F1, post-#140 | Δ |
| :--- | ---: | ---: | ---: |
| 7 | 0.911009 | 0.910745 | **−0.000264** |
| 1–6, 8 | — | — | **0.000000** |

One prediction on one panorama now claims a ground-truth ramp across the seam instead of
counting as a false positive on one side and a miss on the other. That is 4% of the 0.0063
paired MDE the recommendation rests on, and it moves epoch 7 **further below** the plateau, so
every verdict below holds and the 7-and-8 decline is marginally sharper, not softer. Both
matchers are pinned in `tests/test_benchmark_power_135.py`: the historical curve is checked
against the historical matcher at 1e-5, and the single post-#140 difference is asserted rather
than absorbed by a loosened tolerance, so a *second* epoch starting to move fails the build.

One consequence for how the max-F1 table below is built. Its point estimates used to be read
from `docs/data/run_a_84_manual_gold/summary.csv` while its standard errors were bootstrapped
from re-scored detections — which after #140 meant one column was pre-fix and the one beside it
post-fix. They now both come from the epoch dumps under the same matcher. `summary.csv` remains
the provenance record of the #84 run and the regression test's target; it is **not** an input to
this analysis, and it is deliberately left as written, because only four of its ten columns are
exactly re-derivable from the committed detections (the max-F1 block is; the operating-point and
AP columns come from `evaluate.py` and use different conventions).

**This does not generalise to the rest of the repo.** Re-scoring the whole roster while checking
the above turned up three *committed* YOLO baseline cells that #140 did move and that were never
regenerated — see **#148**. RampNet's own numbers are unaffected on every split.

## Run A's plateau, re-read as a paired comparison

Applying the paired instrument to Run A's own committed numbers. "Required discordance" is how
far apart two checkpoints would have to be for the observed gap to be *unreadable* at 95%;
real pairs on this benchmark run 2.1–9.1%, so anything above that is resolvable for any
plausible pair. Design effect 1.15, measured.

**Recall at the fixed 0.30 operating point:**

| epochs | recall A | recall B | Δ | required b+c | required rate | verdict |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 vs 2 | 0.8961 | 0.9138 | +0.0176 | 1081 | 27.6% | resolvable |
| 1 vs 3 | 0.8961 | 0.9158 | +0.0196 | 1346 | 34.4% | resolvable |
| 2 vs 6 | 0.9138 | 0.8987 | −0.0151 | 790 | 20.2% | resolvable |
| 3 vs 6 | 0.9158 | 0.8987 | −0.0171 | 1019 | 26.0% | resolvable |
| 3 vs 7 | 0.9158 | 0.8788 | −0.0370 | 4774 | >100% | resolvable |
| 5 vs 8 | 0.9061 | 0.9005 | −0.0056 | 110 | 2.8% | borderline |

**max-F1, the calibration-free gate column.** Each pair is read against **its own measured paired
standard error**, not a global bracket — which matters, because the s.e. grows with epoch
separation:

| epochs | max-F1 A | max-F1 B | Δ | s.e. | z | verdict |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 vs 2 | 0.9064 | 0.9165 | +0.0101 | 0.0023 | 4.4 | resolvable |
| 1 vs 3 | 0.9064 | 0.9191 | +0.0126 | 0.0025 | 5.1 | resolvable |
| 2 vs 6 | 0.9165 | 0.9165 | +0.0000 | 0.0022 | 0.0 | **not resolvable** |
| 3 vs 6 | 0.9191 | 0.9165 | −0.0025 | 0.0018 | 1.4 | **not resolvable** |
| 6 vs 7 | 0.9165 | 0.9107 | −0.0058 | 0.0020 | 3.0 | resolvable |
| 3 vs 7 | 0.9191 | 0.9107 | −0.0083 | 0.0021 | 4.0 | resolvable |
| 3 vs 8 | 0.9191 | 0.9124 | −0.0066 | 0.0022 | 3.0 | resolvable |
| 5 vs 8 | 0.9179 | 0.9124 | −0.0055 | 0.0020 | 2.8 | resolvable |
| 1 vs 8 | 0.9064 | 0.9124 | +0.0060 | 0.0028 | 2.1 | resolvable |

The two tables say different things, and together they sharpen Run A's conclusion — and in one
respect correct it.

**The plateau is real, but narrower than "epochs 2–8".** Epochs 2 and 6 are identical on max-F1
to four decimals and 3 vs 6 is unreadable, so the *core* of Run A's finding survives a sharper
instrument intact. But **epochs 7 and 8 are measurably below the plateau** — 3 vs 7 at z = 4.0,
3 vs 8 at 3.0, 6 vs 7 at 3.0 — and the unpaired 0.01 bar could not see it.

So the curve is not "steps up once from epoch 1 to 2 and is then flat", which is what #84
recorded. Measured paired, it is:

> **epoch 1 clearly low → epochs 2–6 a genuine plateau → epochs 7–8 measurably declining, though
> still above epoch 1.**

A shallow inverted U with a flat top. This corrects the *shape*, not the headline: the
pre-registered question was where `manual_gold` F1 peaks, and there is still no resolvable peak —
2, 3, 4, 5 and 6 remain mutually indistinguishable. What changes is that the tail is no longer
part of the plateau, and **at constant learning rate the model begins to lose capability after
about epoch 6.** That is a measured fact with a direct bearing on Run B, taken up below.

**On the operating point, the "plateau" contains differences the tie bar hid.** The recall gaps
at fixed 0.30 need 20–34% discordance to be unreadable, against 2.1–9.1% observed. These are real
movements, and the doc's own reading — *"later epochs buy F1 with precision and pay for it in
recall… epoch 3 is the better checkpoint than epoch 6 despite the lower F1@0.30"* — was correct
but had to be hedged as "inside the tie bar" because the bar being applied was unpaired. It is
not inside a paired bar. **Under the recall-first stance, the epoch-3 preference is a measured
result, not a judgment call.**

That is a #84 correction produced at zero cost, and it is the concrete demonstration of the
methodological point: the same numbers, read paired, support a conclusion the unpaired bar
refused.

## What this means for Run B

**The benchmark is not a reason to cancel.** At a measured paired MDE of **0.0063** for the
epoch separations available, and ≈0.008 extrapolated to Run B's larger separation, `manual_gold`
resolves an effect of the size #51's annealed tail produced (+0.024 here, 8σ) with room to spare,
and resolves anything down to about **0.8 points of F1**. The "returns another unreadable
plateau" risk is real only if Run B's anneal buys less than ~0.008 max-F1 — a claim about the
anneal, not about the instrument.

**What the measurement changed about the question.** Run A's curve does not merely flatten after
epoch 2; it flattens and then **measurably declines by epochs 7–8**. So Run B is not testing
"does a longer budget add anything to a flat curve" — it is testing whether **cosine decay
arrests and reverses a decline that is now measured rather than hypothesised.** That is a
sharper hypothesis than the one #135 was filed under, and it cuts both ways: the decline is the
classic signature of a learning rate left too high late in training, which is exactly what an
anneal fixes and is the mechanism #51's arms displayed — but it also means a 30-epoch run whose
decay does *not* bite could plausibly land **below** Run A's epoch 3.

**Recommendation: run the 30-epoch arm. Do not run 60 on spec.** ⚠️ **SUPERSEDED 2026-09-03 — Run B was decided AGAINST.** This paragraph was written before the 8-epoch cosine rung had results. The rung then tested the same mechanism at matched budget and moved `manual_gold` by nothing measurable, and the n=1 objection below turned out to be binding rather than a footnote. The decision and its reasoning are in [`stage2_cosine_rung_135.md`](stage2_cosine_rung_135.md); the rest of this paragraph is kept as written. The plateau ends at epoch 6, so
the extra 30 epochs buy schedule shape rather than useful steps, and a 60-epoch cosine spends
proportionally longer at the high learning rates that the epoch 7–8 decline is evidence against.
If a tail shows at 30, going to 60 becomes an informed follow-up instead of a speculative
doubling.

**One cheaper rung is now more attractive than it was.** The #84 amendment recorded an
**8-epoch cosine** arm as "recorded, not scheduled" — ~28 h and ~450 GPU-h, budget-matched to
Run A. Since constant-LR is now known to peak around epoch 3 and decline by 7, that rung tests
"does decay beat constant at the *same* budget" cleanly, isolating schedule from length, for
about a quarter of the 30-epoch arm's cost. It is not a substitute for Run B — it cannot show an
annealed tail that needs length — but it would make Run B's result attributable in a way that
B − A on its own is not, which is the confound the amendment already flagged. Raising it because
the measurement moved it, not to relitigate the spec.

**Three amendments to the read, which cost nothing and are worth pre-registering now:**

1. **Read Run B against Run A paired, on max-F1**, with the 0.01 tie bar replaced by the measured
   paired MDE — **0.008** at Run B's expected separation. Reading a 1,675-GPU-hour result with an
   instrument 2× blunter than necessary is the cheapest mistake available here.
2. **Report per-split, not pooled.** Pooling buys 7% and costs GT homogeneity.
3. **Attribute cautiously below ~0.01.** See the limit below.

### The limit that actually binds, and it is not the benchmark

Everything above is **panorama sampling variance only**. It says how precisely a difference
between two specific checkpoints can be measured on this benchmark. It says nothing about
**training-seed variance** — how much two runs of the identical recipe with different seeds
would differ — and Run B is n = 1.

Nothing in this repo measures that. Run A's free noise floor (the requeue that recomputed epoch 5
on two nodes, agreeing to 0.0090%) bounds resume-path nondeterminism on *validation loss*, not
seed-to-seed variation in `manual_gold` F1. So if Run B comes back +0.006 max-F1, the benchmark
can see it, but nothing available says whether the anneal or the seed produced it. A seed
control doubles the bill.

The practical form: **a Run B gain below ~0.01 max-F1 should be reported as measured but
unattributed** unless a seed control is run. A gain at the #51 scale (~0.02+) is large enough
that seed variance is an implausible explanation and can be attributed to the schedule directly.

### How the bracket was closed, for no GPU at all

The first draft of this document costed this at "about a GPU-hour" for two checkpoints, on the
assumption that the per-panorama detections would have to be regenerated by running the model.
They did not. **`evaluate.py`'s heatmap cache from the 2026-08-17 scoring survived on makelab2**
— 13 GB at `run_a_84/evaluate_cache/heatmaps/`, all eight epochs single-pass, 1,000 panoramas
each — and extracting peaks from a cached heatmap is CPU-only numpy. So the measurement needed
**no model, no panorama images, no GPU and no network**, took 4 minutes, and covered the whole
curve instead of the two checkpoints originally proposed.

The panorama images are in fact *not* on that host, which is what makes the cache load-bearing
rather than merely convenient.

`scripts/analysis/dump_peaks_from_cache.py` does the recovery. It is a separate script rather
than a flag on `evaluate.py` deliberately: that evaluator produced every committed Stage 2
number and its heatmap cache key is `<fingerprint>_<dataset>_<tta>` and nothing else, so the
cheapest way to guarantee this analysis could not perturb either was not to touch it. What it
*does* share is the part that must not diverge — `extract_peaks_from_heatmap`,
`PEAK_MIN_DISTANCE` and `MODEL_HEATMAP_SIZE` are imported from `evaluate.py`, not copied.

**Verified against the committed curve, which is the load-bearing check.** Re-scoring each dump
against `manual_labels/` reproduces `docs/data/run_a_84_manual_gold/summary.csv`:

| | agreement, all 8 epochs |
| :--- | :--- |
| max-F1 | **5×10⁻⁹ to 4×10⁻⁷** |
| F1@0.30 | 1.2×10⁻⁴ to 1.4×10⁻⁴ |

max-F1 is a property of the entire PR curve, so agreement at 10⁻⁷ says the peak extraction is
the same operation, not a similar one. The uniform 10⁻⁴ offset on F1@0.30 is the committed
table's own 0.005-confidence-grid downsampling, which its provenance note already states
re-derives F1 "to ~3 decimals".

One deliberate truncation travels with these files: they carry a **0.05 peak floor**, matching
`analysis_out/op_cache`, where Run A's scoring used `--threshold 0.0`. That keeps them ~200 KB
each instead of ~40 MB (threshold 0.0 retains ~511,000 predictions over 1,000 panoramas, nearly
all noise floor) and sits far below everything they are used for — the protocol point is 0.30
and Run A's max-F1 lands between 0.268 and 0.582. **AP is therefore not recoverable from them**
and must be read from `docs/data/run_a_84_manual_gold/`.

## Reproduce, from a clean clone

No cluster access, no `.model_cache`, no GPU, no network — every input is committed, including
the eight Run A epoch dumps at `docs/data/run_a_84_detections/`.

```bash
python scripts/analysis/benchmark_power_135.py \
    --bootstrap 20000 --matrix-bootstrap 5000 \
    --out-json docs/data/benchmark_power_135.json
```

Roughly 20 minutes on a laptop. Inputs: `manual_labels/` and `benchmark/*/records.jsonl` +
`verdicts.json` for ground truth, `benchmark/model_detections/*.json` for the YOLO arms,
`docs/data/run_a_84_detections/*.json` for the Run A epoch dumps,
`analysis_out/op_cache/*.json` for the single-pass RampNet arm, and
`docs/data/run_a_84_manual_gold/summary.csv` for Run A's own curve. Every derived number in this
document is in `docs/data/benchmark_power_135.json`.

Regenerating the epoch dumps themselves needs the heatmap cache, which is 13 GB and lives on
makelab2 rather than in the repo — **that is the one input here a clean clone cannot obtain**,
and it is why the dumps are committed rather than left to be rebuilt. With the cache in hand:

```bash
python scripts/analysis/dump_peaks_from_cache.py \
    --cache-dir <...>/run_a_84/evaluate_cache --verify
```

They land in `docs/data/run_a_84_detections/`, beside the rest of the #84 data, and **not** in
`benchmark/model_detections/` — `rampnet/roster.py` asserts every file in that directory belongs
to a registered challenger leg (#122), and Run A's epochs are internal checkpoints of one
experiment rather than entries in the RampNet-vs-VLM comparison. The test suite caught the first
attempt to put them there, which is the registry working as intended.

The bootstrap is seeded (`--seed 42`), so the run is deterministic; changing `--bootstrap` moves
the third decimal of the standard errors and none of the conclusions.

**No figure.** The tables carry the result and a plot of six standard errors would not add to it.
Noted here so the omission is visible rather than assumed.
