# Can the benchmark resolve Run B? (#135)

**Status: complete 2026-08-18. No cluster time was spent — every number below comes from
committed data.** The recommendation is at the bottom; the short version is that
`manual_gold` can resolve the effect Run B would plausibly produce, but only if the
comparison is read **paired**, and that **pooling the benchmark splits does not help**.

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
| **Paired**, two near-identical checkpoints | 0.0015 | **0.0043** |
| **Paired**, two quite different detectors | 0.0048 | **0.0135** |
| Pooling all ten splits, unpaired | 0.0039 | 0.0109 |

Three things follow.

1. **Pairing is worth 2–3×, and it is free.** Both checkpoints are scored on the same 1,000
   panoramas against the same ground truth, so the panorama-to-panorama difficulty that
   dominates the unpaired noise is common to both and cancels. The measured gain is 1.8× to
   3.6× on the standard error, depending on the pair.
2. **Pooling is worth 7%.** Ten splits together hold 6,560 instances against `manual_gold`'s
   3,919, and the MDE moves 0.0117 → 0.0109. That is not a lever.
3. **The tie bar is the wrong instrument for an epoch-vs-epoch comparison**, and using it cost
   Run A a result — see "Run A's plateau, re-read" below.

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

Read the first row as the near-identical end of the range and the third as the far end. **The
comparison Run B would actually make — two Stage 2 checkpoints of one lineage — has no committed
per-pano data**, because Run A's committed artifacts are downsampled PR curves and are aggregate
only. So the paired s.e. is bracketed rather than point-estimated, and the bracket is built so it
holds by construction: one checkpoint under two inference protocols is *more* correlated than two
checkpoints, and two different YOLO architectures trained to different budgets are *less*.

**s.e.(Δ max-F1) for two Stage 2 checkpoints lies between 0.0015 and 0.0048, so the MDE lies
between 0.0043 and 0.0135.** The standard error tracks the discordance, and the discordance
tracks how different the two detectors are, so a Run B checkpoint compared against a Run A
checkpoint — same recipe, same data, same seed, differing in schedule — sits nearer the low end.
Taking 4–6% discordance as the working assumption puts the MDE at roughly **0.006–0.009 max-F1**.

The middle row is the one worth dwelling on. `y11x_pano_h200` versus `y11l_pano` is #51's
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

**max-F1, the calibration-free gate column**, against the paired s.e. bracket (0.0015–0.0048):

| epochs | max-F1 A | max-F1 B | Δ | z (best case) | z (worst case) | verdict |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 vs 2 | 0.9064 | 0.9165 | +0.0101 | 6.5 | 2.1 | resolvable |
| 1 vs 3 | 0.9064 | 0.9191 | +0.0126 | 8.1 | 2.6 | resolvable |
| 2 vs 6 | 0.9165 | 0.9165 | +0.0000 | 0.0 | 0.0 | not resolvable |
| 3 vs 6 | 0.9191 | 0.9165 | −0.0025 | 1.6 | 0.5 | not resolvable |
| 3 vs 7 | 0.9191 | 0.9110 | −0.0080 | 5.2 | 1.7 | borderline |
| 5 vs 8 | 0.9179 | 0.9124 | −0.0055 | 3.5 | 1.1 | borderline |

The two tables say different things, and together they sharpen Run A's conclusion rather than
overturn it.

**On capability, the plateau is real.** Epochs 2 and 6 are identical on max-F1 to four decimals,
and 3 vs 6 is unreadable even at the most favourable end of the bracket. Run A's finding that
nothing in 2–8 separates is not an artifact of a blunt instrument — a sharper instrument still
cannot separate them.

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

**The benchmark is not a reason to cancel.** At a working paired MDE of 0.006–0.009 max-F1,
`manual_gold` resolves an effect of the size #51's annealed tail produced (+0.024 on this
benchmark, 8σ) with room to spare, and resolves anything down to about half a point of F1. The
"returns another unreadable plateau" risk is real only if Run B's anneal buys less than
~0.005 max-F1 — which is possible, but it is a claim about the anneal, not about the instrument.

**Recommendation: run the 30-epoch arm. Do not run 60 on spec.** Run A showed the constant-LR
curve is flat from epoch 2, so the extra 30 epochs buy a longer anneal ramp rather than more
useful steps, and 30 epochs is the half of the decision that tests the hypothesis. If a tail
shows up at 30, going to 60 becomes an informed follow-up instead of a speculative doubling.

**Three amendments to the read, which cost nothing and are worth pre-registering now:**

1. **Read Run B against Run A paired, on max-F1**, with the tie bar replaced by the measured
   paired MDE. Reading a 1,675-GPU-hour result with an instrument 2.6× blunter than necessary
   is the cheapest mistake available here.
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

### One measurement that would close the bracket first, for about a GPU-hour

The bracket [0.0043, 0.0135] is wide because the epoch-to-epoch discordance of two RampNet
checkpoints has never been measured — only bounded. It is measurable now: **the eight Run A
checkpoints already exist** at `/gscratch/makelab/jonf/rampnet_run_a_84/checkpoints/`, and
scoring two of them (epoch 3 and epoch 6, the two ends of the plateau) with per-panorama output
gives the discordance directly, pinning the MDE to a number before 1,675 hours are committed.

It needs a small addition, because neither existing path emits per-panorama detections:
`stage_two/evaluate.py` writes aggregate PR curves only, and `compare.py --models rampnet` reads
detections from the bundle rather than running a checkpoint (`BundleRampNetDetector`). A
`--dump-detections` flag on `evaluate.py`, writing the same
`{pano_id: [[x, y, conf], …]}` shape as `benchmark/model_detections/`, is enough;
`benchmark_power_135.py` reads that shape already.

Cost: ~12.5 min per checkpoint on makelab2's A40, per the #84 scoring record — and possibly less,
since the heatmap cache from the 2026-08-17 scoring run may still be on that host, in which case
no inference is needed at all. That has not been checked from here and should not be assumed.

## Reproduce, from a clean clone

No cluster access, no `.model_cache`, no GPU, no network — every input is committed.

```bash
python scripts/analysis/benchmark_power_135.py \
    --bootstrap 20000 --out-json docs/data/benchmark_power_135.json
```

Roughly 8 minutes on a laptop. Inputs: `manual_labels/` and `benchmark/*/records.jsonl` +
`verdicts.json` for ground truth, `benchmark/model_detections/*.json` for the YOLO arms,
`analysis_out/op_cache/*.json` for the single-pass RampNet arm, and
`docs/data/run_a_84_manual_gold/summary.csv` for Run A's own curve. Every derived number in this
document is in `docs/data/benchmark_power_135.json`.

The bootstrap is seeded (`--seed 42`), so the run is deterministic; changing `--bootstrap` moves
the third decimal of the standard errors and none of the conclusions.

**No figure.** The tables carry the result and a plot of six standard errors would not add to it.
Noted here so the omission is visible rather than assumed.
