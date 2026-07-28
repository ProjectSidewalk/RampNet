# Model comparison: RampNet vs. general-purpose models

Uses the standardized curb-ramp benchmark (`benchmark/{bend,richmond,clovis,annapolis}/`, plus
the 1k in-distribution `manual_gold` split — see its section below) to compare
RampNet against off-the-shelf models. The question: does a general model match or beat the
purpose-trained RampNet on real deployment imagery (GSV + Mapillary 360)? The harness is
model-agnostic, so new models (issues #20, #39) plug in the same way.

## What has been run where

Every split in `benchmark/` appears here, including the ones with gaps — an omission below is
a run that hasn't happened, not a result being withheld.

| split | RampNet | challengers | null-recall | GT-completeness corrected (#55) | notes |
|---|---|---|---|---|---|
| richmond | ✅ | ✅ all 8 | ✅ | ✅ 17% (5/29) | OOD deployment, Mapillary 360 |
| bend | ✅ | ✅ all 8 | ✅ | ✅ 29% (7/24) | in-domain GSV reference |
| clovis | ✅ | ✅ all 8 | ✅ | ❌ | hardest split — 2018 GoPro Fusion |
| annapolis | ✅ | ✅ all 8 | ✅ | ❌ | survey-grade Trimble MX7; far-field finding |
| morgantown | ✅ | ❌ **not run** | RampNet only | ❌ | harness check only so far |
| budapest_district5 | ✅ | ❌ **not run** | RampNet only | ❌ | **GT itself is low-confidence** — see its section |
| manual_gold | ✅ | ✅ all 8 | ❌ too slow | ❌ | 1k panos, un-anchored GT |

"All 8" is the roster in the class table below: RampNet, 2 Geminis, 2 Qwens, Molmo, OWLv2,
Grounding DINO. The two ❌ challenger cells are a compute gap, not a judgment about the
splits — `.model_cache` holds no detections for either, so putting them in the table means
~1.5 h of GPU on Hyak plus a couple of desktop-hours of Vertex calls per city. The
`null_recall.py` cells follow from that: it re-scores cached detections, so it can only
cover models that have been run.

Three classes of challenger, which fail differently and are worth keeping distinct:

| class | models | output | tunable? |
|---|---|---|---|
| **chat VLMs** | `gemini-3.6-flash`, `gemini-3.1-pro-preview`, `Qwen/Qwen3-VL-*` | boxes, no score | no — one operating point |
| **open-vocab detectors** | `google/owlv2-large-patch14-ensemble`, `IDEA-Research/grounding-dino-base` | boxes **with calibrated scores** | yes — AP, PR curve, threshold sweep |
| **pointing models** | `allenai/Molmo2-8B`, `allenai/MolmoPoint-8B` | **points** (RampNet's native format) | no score, but no box→point reduction |

The chat VLMs are all doing localization as a side skill, and they lose the same way: they
are false-positive-heavy (119–293 FP against RampNet's 9). The other two classes exist in
this harness to test whether that is a property of *general models* or of *chat models*.

**It looks like a general-model property, not a chat-specific one:** see the results below —
the off-the-shelf open-vocabulary detectors do far *worse* than the chat VLMs on this task,
not better. (Caveat up front: these are general open-vocab detectors given a short text
query — *not* detectors trained for curb ramps. The only model here purpose-built for the
class is RampNet itself, and every challenger is zero-shot with an untuned prompt. See
"Scope of the claim" below.)

## Results

Perspective tiling, match radius 0.022, all models scored against the same derived GT.
Open detectors are shown at their 0.05 cache floor; their tuned operating points are in the
sweep below. Run on Hyak (L40S); RampNet and Gemini rows are cache-scored.

**richmond** (124 reviewed panos, 310 GT ramps)

| model | P | R | F1 | AP | tp/fp/fn |
|---|---|---|---|---|---|
| **rampnet** | **0.964** | 0.768 | **0.855** | 0.763 | 238/9/72 |
| gemini-3.1-pro-preview | 0.631 | 0.700 | 0.664 | – | 217/127/93 |
| gemini-3.6-flash | 0.626 | 0.642 | 0.634 | – | 199/119/111 |
| **molmo2-8B** (points) | 0.410 | 0.516 | **0.457** | – | 160/230/150 |
| Qwen3-VL-32B-Instruct | 0.760 | 0.297 | 0.427 | – | 92/29/218 |
| Qwen3-VL-8B-Instruct | 0.323 | 0.452 | 0.377 | – | 140/293/170 |
| owlv2-large-patch14-ensemble | 0.033 | **0.971** | 0.064 | 0.104 | 301/8799/9 |
| grounding-dino-base | 0.028 | 0.852 | 0.053 | 0.032 | 264/9321/46 |

**bend** (110 reviewed panos, 327 GT ramps)

| model | P | R | F1 | AP | tp/fp/fn |
|---|---|---|---|---|---|
| **rampnet** | **0.961** | 0.761 | **0.850** | 0.754 | 249/10/78 |
| gemini-3.1-pro-preview | 0.706 | 0.581 | 0.638 | – | 190/79/137 |
| gemini-3.6-flash | 0.608 | 0.587 | 0.597 | – | 192/124/135 |
| **molmo2-8B** (points) | 0.510 | 0.401 | **0.449** | – | 131/126/196 |
| Qwen3-VL-32B-Instruct | 0.706 | 0.294 | 0.415 | – | 96/40/231 |
| Qwen3-VL-8B-Instruct | 0.379 | 0.336 | 0.357 | – | 110/180/217 |
| owlv2-large-patch14-ensemble | 0.037 | 0.951 | 0.070 | 0.093 | 311/8187/16 |
| grounding-dino-base | 0.038 | 0.850 | 0.073 | 0.049 | 278/6969/49 |

**clovis** (125 reviewed panos, 195 GT ramps) — Mapillary GoPro Fusion 360s, the hardest of the
three deployment cities

| model | P | R | F1 | AP | tp/fp/fn |
|---|---|---|---|---|---|
| **rampnet** | **0.914** | 0.713 | **0.801** | 0.688 | 139/13/56 |
| gemini-3.1-pro-preview | 0.531 | 0.477 | 0.503 | – | 93/82/102 |
| gemini-3.6-flash | 0.460 | 0.497 | 0.478 | – | 97/114/98 |
| **molmo2-8B** (points) | 0.331 | 0.436 | **0.376** | – | 85/172/110 |
| Qwen3-VL-32B-Instruct | 0.696 | 0.200 | 0.311 | – | 39/17/156 |
| Qwen3-VL-8B-Instruct | 0.222 | 0.292 | 0.252 | – | 57/200/138 |
| owlv2-large-patch14-ensemble | 0.025 | **0.908** | 0.049 | 0.067 | 177/6911/18 |
| grounding-dino-base | 0.018 | 0.867 | 0.035 | 0.026 | 169/9433/26 |

Clovis is 100% soft, 2018-era GoPro Fusion 360 imagery, so every model degrades relative to
richmond/bend — RampNet's own P/R slips to 0.914/0.713 (from richmond's 0.960/0.765). But the
**ranking is identical across all three cities**, and RampNet's lead *widens*: the gap to the best
challenger grows from ~0.19 (richmond) to **~0.30** here. These are the all-125 numbers, so every
model is scored on the same panos; clovis's ground-truth quality against the 120-pano *unbiased*
subset (P 0.889 / R 0.650) is in `benchmark/README.md`. (`gemini-2.5-flash`, not run on richmond,
scores F1 0.278 on clovis — between Qwen-32B and Qwen-8B, tracking its 0.252 on bend.)

Molmo is the strongest **open-weight** model here — best F1 of the four, and the only
challenger with a *balanced* profile (P≈R) rather than an FP flood or extreme caution. It
also does it natively in points, no box→center reduction. But it is still ~0.4 F1 behind
RampNet and clearly behind both Geminis, and it is *sparse*: on the verified overlay pano it
proposed 2 points where 4 ramps were visible, which shows up as its 150/196 false negatives.

**annapolis** (125 reviewed panos, 294 GT ramps) — Trimble MX7 vehicle survey rig, the first
survey-grade camera in the benchmark

| model | P | R | F1 | AP | tp/fp/fn |
|---|---|---|---|---|---|
| **rampnet** | **0.973** | 0.738 | **0.839** | 0.734 | 217/6/77 |
| gemini-3.1-pro-preview | 0.613 | 0.527 | 0.567 | – | 155/98/139 |
| gemini-3.6-flash | 0.637 | 0.490 | 0.554 | – | 144/82/150 |
| **molmo2-8B** (points) | 0.434 | 0.415 | **0.424** | – | 122/159/172 |
| Qwen3-VL-32B-Instruct | 0.608 | 0.296 | 0.398 | – | 87/56/207 |
| Qwen3-VL-8B-Instruct | 0.304 | 0.354 | 0.327 | – | 104/238/190 |
| owlv2-large-patch14-ensemble | 0.032 | **0.959** | 0.063 | 0.126 | 282/8444/12 |
| grounding-dino-base | 0.029 | 0.898 | 0.055 | 0.042 | 264/8992/30 |

Best sweep F1 for the open detectors: OWLv2 **0.208** (thr 0.25), Grounding DINO **0.091**
(thr 0.15). The **ranking is identical for the fourth city running**, and every per-model
signature recurs: Molmo best open-weight with the only balanced profile, Qwen-32B cautious
(challenger-best precision 0.608 at the worst recall 0.296), Qwen-8B FP-leaky, open-vocab
detectors trading a huge nominal recall for ~3% precision. RampNet's lead over the best
challenger is 0.27 F1, between richmond's ~0.19 and clovis's ~0.30.

Annapolis is also where the open detectors' recall column stops being believable — see
"How much of a detector's recall is real?" below, which was measured here first and then
reproduced on every split that has challenger detections.

### Scope of the claim

The headline — RampNet beats every off-the-shelf model tested — is real and wide, but it
carries qualifiers that must travel with it: the challengers are **zero-shot**, run with a
single **untuned prompt/query**, scored by **box-center** at a tight radius, against a
**RampNet-anchored** ground truth (see Caveats — though the `manual_gold` split below now
tests exactly that qualifier and reproduces the ranking on un-anchored GT). The honest one-liner is *"an in-domain model
trained for curb ramps beats zero-shot general models — chat VLMs and open-vocab detectors
alike — under a reasonable but untuned prompt,"* not *"purpose-built detection loses,"* which
no experiment here tested (OWLv2 / Grounding DINO are general open-vocab models, not
curb-ramp detectors). How much of the gap survives a tuned prompt (#45), a failure-artifact
audit (#46), and a nadir/hood mask (#47) is exactly what those follow-ups measure.

### What the numbers say

1. **RampNet still wins by a wide margin**, and nothing tested comes close on F1. The best
   challenger (Gemini-3.1-pro, F1 0.664) trails it by ~0.19; the best open-weight model
   (Molmo, F1 0.457) by ~0.40.
2. **Off-the-shelf open-vocab detectors did worse than chat models, not better.** OWLv2's
   best F1 over the whole threshold sweep is **0.184** (thr 0.25: P 0.130 / R 0.310);
   Grounding DINO's is **0.073**. Both are far below Gemini-3.6-flash's 0.634. So the
   *chat-specific* version of the issue-#39 hypothesis — that the VLMs' weakness was a *chat*
   problem an open-vocab *detector* would sidestep — does not hold: a text-queried open-vocab
   detector is simply not selective enough for an object that looks like a slightly different
   patch of concrete. Note the narrower scope, though: these are general open-vocab models
   with an untuned query (#45), not detectors trained for curb ramps — the only purpose-built
   model here is RampNet.
3. **Capacity isn't the chat VLMs' problem either.** Qwen-32B moved to the *precise* end
   (P 0.760 / R 0.297) versus 8B's FP flood (P 0.323 / R 0.452) — the operating point moved,
   F1 barely did (0.427 vs 0.377).
4. **But OWLv2 has an extraordinary recall ceiling** — with a large asterisk. At its floor it
   finds **97.1%** of richmond's ramps, against RampNet's 76.8%. Most of that gap is **not
   detection**: at ~74 boxes per pano a 0.022 match radius hands out most of the recall for
   free. Quantified in "How much of a detector's recall is real?" below — read it before
   quoting the 97.1%.

### The recall-first angle: OWLv2 as a candidate generator

Recall matters more than precision here (a false negative is a permanently missing ramp; a
false positive is a cheap review). So the question is not OWLv2's F1 but whether it *sees*
what RampNet misses. By the table it appears to — nearly all of it. **The next section shows
most of this is a density artifact, so read the two together**; "recovered by OWLv2" below
means "a ramp fell within the match radius of some OWLv2 box," which at 74 boxes per pano is
a much weaker statement than it looks:

| OWLv2 thr | RampNet FN | recovered by OWLv2 | union recall | OWLv2 FP | FP per recovered ramp |
|---|---|---|---|---|---|
| 0.05 | 72 | **69** | **0.990** | 8799 | 128 |
| 0.10 | 72 | 60 | 0.961 | 5167 | 86 |
| 0.15 | 72 | 46 | 0.916 | 2923 | 64 |
| 0.20 | 72 | 34 | 0.877 | 1507 | 44 |
| 0.25 | 72 | 18 | 0.826 | 640 | 36 |

A RampNet ∪ OWLv2 oracle would reach **0.990** recall on richmond. The cost is the story: at
128 false positives per recovered ramp, versus **~6** for Gemini-3.6-flash (which recovered
20 of the 72 at 119 FP total — see issue #35), OWLv2 is a **6–20× less efficient**
complement. It is a recall oracle, not a usable candidate generator, unless a downstream
verifier can reject ~128 proposals per find more cheaply than Gemini can propose 6.

### How much of a detector's recall is real? (`scripts/analysis/null_recall.py`)

A model that carpets the pano earns recall for free. At a fixed match radius, enough
scattered boxes land within radius of most GT points whether or not the model saw anything —
and the open detectors emit **~57–78 boxes per pano** against RampNet's ~2. So "recall" is
not measuring the same thing down the column.

The null model: score each pano's ground truth against **another pano's** predictions. That
keeps the detector's exact detection count and spatial distribution — including systematic
clustering like the hood/nadir boxes — and destroys every true correspondence, so whatever
recall survives is what the radius hands out for free at that density. Averaged over all
non-identity cyclic shifts (deterministic, no seed); `null max` is the worst single shift,
which is close to the mean, so the null is a property of the density and not of one
pathological pairing.

| split | model | boxes/pano | recall | **null** | null max | above chance |
|---|---|---|---|---|---|---|
| richmond | rampnet | 2.2 | 0.768 | 0.055 | 0.132 | 0.754 |
| richmond | gemini-3.1-pro-preview | 3.1 | 0.700 | 0.083 | 0.142 | 0.673 |
| richmond | gemini-3.6-flash | 2.7 | 0.642 | 0.063 | 0.113 | 0.618 |
| richmond | molmo2-8B | 3.3 | 0.516 | 0.076 | 0.110 | 0.476 |
| richmond | Qwen3-VL-32B | 1.0 | 0.297 | 0.023 | 0.065 | 0.280 |
| richmond | Qwen3-VL-8B | 3.6 | 0.452 | 0.080 | 0.116 | 0.404 |
| richmond | **owlv2-large** | **74.3** | 0.971 | **0.733** | 0.829 | 0.891 |
| richmond | **grounding-dino-base** | **78.0** | 0.852 | **0.619** | 0.690 | 0.610 |
| bend | rampnet | 2.4 | 0.761 | 0.063 | 0.144 | 0.745 |
| bend | gemini-3.1-pro-preview | 2.6 | 0.581 | 0.064 | 0.119 | 0.553 |
| bend | gemini-3.6-flash | 3.0 | 0.587 | 0.070 | 0.122 | 0.556 |
| bend | molmo2-8B | 2.4 | 0.401 | 0.060 | 0.116 | 0.362 |
| bend | Qwen3-VL-32B | 1.2 | 0.294 | 0.024 | 0.070 | 0.276 |
| bend | Qwen3-VL-8B | 2.7 | 0.336 | 0.048 | 0.086 | 0.303 |
| bend | **owlv2-large** | **78.0** | 0.951 | **0.738** | 0.780 | 0.813 |
| bend | **grounding-dino-base** | **66.4** | 0.850 | **0.648** | 0.706 | 0.574 |
| clovis | rampnet | 1.3 | 0.713 | 0.044 | 0.118 | 0.700 |
| clovis | gemini-3.1-pro-preview | 1.4 | 0.477 | 0.042 | 0.108 | 0.454 |
| clovis | gemini-3.6-flash | 1.7 | 0.497 | 0.049 | 0.092 | 0.472 |
| clovis | molmo2-8B | 2.1 | 0.436 | 0.055 | 0.108 | 0.403 |
| clovis | Qwen3-VL-32B | 0.4 | 0.200 | 0.012 | 0.036 | 0.190 |
| clovis | Qwen3-VL-8B | 2.1 | 0.292 | 0.043 | 0.087 | 0.261 |
| clovis | **owlv2-large** | **57.0** | 0.908 | **0.646** | 0.738 | 0.739 |
| clovis | **grounding-dino-base** | **77.1** | 0.867 | **0.627** | 0.718 | 0.642 |
| annapolis | rampnet | 1.8 | 0.738 | 0.051 | 0.085 | 0.724 |
| annapolis | gemini-3.1-pro-preview | 2.1 | 0.527 | 0.060 | 0.102 | 0.497 |
| annapolis | gemini-3.6-flash | 1.9 | 0.490 | 0.049 | 0.082 | 0.463 |
| annapolis | molmo2-8B | 2.3 | 0.415 | 0.064 | 0.116 | 0.375 |
| annapolis | Qwen3-VL-32B | 1.2 | 0.296 | 0.029 | 0.054 | 0.275 |
| annapolis | Qwen3-VL-8B | 2.8 | 0.354 | 0.068 | 0.109 | 0.306 |
| annapolis | **owlv2-large** | **70.2** | 0.959 | **0.769** | 0.847 | 0.823 |
| annapolis | **grounding-dino-base** | **74.3** | 0.898 | **0.641** | 0.738 | 0.715 |

RampNet on the two splits without challenger runs, for completeness: morgantown 1.7
boxes/pano, recall 0.730, null 0.037; budapest 1.5 boxes/pano, recall 0.510, null 0.029.
Both sit in the same 0.03–0.07 band as every other sparse model.

"Above chance" is headroom-normalized: of the recall a perfect detector could add over the
null, how much this model captured. It is a *generous* framing when the null is high — read
it alongside the raw gap, not instead of it.

What this changes:

1. **The open detectors' recall column is mostly density, not detection.** Scoring richmond's
   GT against *unrelated* OWLv2 boxes still recovers **73.3%** of the ramps. Of its headline
   0.971, only ~24 points are attributable to localization at all. The effect reproduces on
   all four splits (nulls 0.62–0.77) and tracks box density, not city difficulty.
2. **The recall-oracle table above is weaker than it reads.** "OWLv2 recovered 69 of
   RampNet's 72 misses" is the raw count; chance alone would cover ~0.733 × 72 ≈ **53** of
   them, so the above-chance recovery is closer to **16 ramps than 69**. (First-order: it
   applies the split-wide null rate to the miss subset, which assumes the null rate doesn't
   vary between the ramps RampNet misses and the ones it finds. Worth checking directly
   before leaning on the number — the misses skew far-field, and box density is not uniform
   in a reprojected view.)
3. **Every model with a normal detection budget is unaffected.** RampNet, both Geminis, Molmo
   and both Qwens sit on nulls of 0.01–0.08 across all four splits, so their recall is
   essentially all signal, and the F1 ranking — which the FP flood already penalizes — does
   not move.
4. **It does not say the open detectors are noise.** Their above-chance figures are clearly
   positive (0.57–0.89); it says the *headline recall number* is not comparable to a sparse
   model's, and any "recall ceiling" or union-oracle claim built on it must be discounted
   first.

This also settles a tempting misreading of the annapolis far-field result: OWLv2 reaching
R 0.959 where RampNet gets 0.738 is **not** evidence that the far-field ramps are resolvable
in the imagery. At 70 boxes per pano most of that coverage is free, so it neither supports
nor refutes the viewpoint-limit finding in `benchmark/README.md`, which rests on its own
rank statistics. The test that *would* settle it is a distance-stratified version of this
table — real vs null recall for near and far ramps separately.

Not run on `manual_gold`: the null is O(n²) in panos, so 1,000 panos is ~64× a city split per
model. A subsampled version would work and hasn't been done.

## Why the benchmark verdicts can't be reused directly

The bundle's `verdicts.json` holds a human judgment for **each RampNet detection** (aligned
positionally to `records.jsonl`). Those verdicts describe RampNet's points and can't score a
different model, which produces different boxes. So we derive a **model-agnostic ground
truth** from the same review and score every model against it identically.

## Methodology

Per pano (`rampnet/detection_eval.py`):

- **GT ramp points** = detections the reviewer confirmed real (`True`) **∪** ramps they
  marked as missed (non-`unsure`). This is the reviewer's complete enumeration of the real
  curb ramps in the pano.
- **Ignore points** = `unsure` detections **∪** `unsure` missed marks. A prediction landing
  here is scored as **neither** TP nor FP (mirrors `validation.collect`'s `unsure`
  abstention) — the reviewer couldn't tell from the imagery, so no model is rewarded or
  penalized there.
- `False` / `duplicate` detections join neither set. A duplicate is a second hit on a ramp
  already in GT, so it becomes a false positive naturally under greedy 1:1 matching.

Scoring is uniform across models: every detector's output is reduced to center points
`(x, y[, confidence])` — a VLM box → its center — and greedily matched to GT within the
normalized radius **0.022** (the pano value in `rampnet/metrics.py`), with the same
anisotropic 1024/512 scaling. **Precision** counts detections on every pano; **recall** counts
only panos whose missed-ramp check is confirmed (`no_missed` set, or a missed mark exists), so
un-scanned panos can't bias it — the same gate `validation.collect` uses. Each of P/R/F1
carries a **Wilson 95% CI** (`rampnet.validation.wilson_interval`).

## Harness self-validation

Scoring RampNet's **own** bundle detections against this derived GT reproduces the published
verdict-based numbers within a small tolerance, which validates the harness before any VLM
spend:

| City | Harness (derived GT) | Published (verdict-based) |
|------|----------------------|---------------------------|
| richmond | P 0.964 / R 0.768 | P 0.960 / R 0.765 |
| bend | P 0.961 / R 0.761 | P 0.954 / R 0.758 |
| clovis | P 0.914 / R 0.713 | P 0.914 / R 0.713 |
| morgantown | P 0.975 / R 0.730 | P 0.975 / R 0.730 |
| annapolis | P 0.973 / R 0.738 | P 0.964 / R 0.728 |
| budapest_district5 | P 0.874 / R 0.510 | P 0.873 / R 0.503 |

The ~0.005–0.010 upward drift on richmond/bend/annapolis/budapest is expected: a RampNet
`False` detection occasionally falls within radius of a real GT point, which the per-detection
human verdict scored differently (on clovis and morgantown no `False` detection does, so the
two columns coincide exactly). The `compare.py` CLI prints both side by side.

### The two splits with no challenger results yet

Both are documented here rather than omitted; the gap is compute, not a finding.

**morgantown** (split added 2026-07-25) is a harness check only so far. RampNet's own row is
P 0.975 / R 0.730 / F1 0.835 / AP 0.728 (195/5/72, 9 ignored), the AP truncated at the bundle's
0.55 peak floor exactly as the other cities' RampNet AP is. It is the *best* precision in the
benchmark and its imagery is 2024 GoPro Max, so it is the natural control for the "precision
tracks the camera" story in `benchmark/README.md` — which is exactly why a challenger roster on
it would be worth having.

**budapest_district5** (split added 2026-07-24) is the first non-US split, and RampNet's row is
P 0.874 / R 0.510 — a recall unlike anything else in the benchmark. **Read
`benchmark/README.md`'s Budapest section before quoting it.** The short version: the reviewer
rated their own pass *low confidence* and the curb-ramp rubric does not transfer cleanly to
Hungarian street furniture (the diagonal-apron question alone is worth ~4 precision points),
so the split's ground truth is itself uncertain in a way no other split's is. That makes it the
most interesting split to run challengers on and the most dangerous one to quote:

- A challenger table here would be a **rubric-robustness** test, not a difficulty test. If the
  ranking holds on GT the reviewer distrusts, that is a stronger statement than any US split
  can make. If it doesn't, the first suspect is the rubric, not the models.
- Any number from it must travel with the low-confidence flag. Do **not** pool it with the US
  splits or average it into a headline.
- It needs a second rater more than it needs more models. That is the higher-value next step;
  the model runs are cheap and can happen either way.

## Caveats (read before quoting numbers)

- **RampNet-anchored GT.** The GT was assembled during a RampNet review. A reviewer scanning
  fresh for another model might catch a few more ramps; the complete-scan attestation
  (`no_missed`) mitigates this, but it is a known asymmetry. **It has now been measured on
  two cities, and it is not small.** Re-reviewing the detections RampNet only surfaces below
  its deployed 0.55 threshold — a confidence band the GT never fully audited — found real,
  unlabelled curb ramps at **17% (5/29, richmond)** and **29% (7/24, bend)** of those
  detections (issue #55; tags in `benchmark/<city>/incremental_fp_tags.json`, reproduce with
  `operating_point_curve.py gallery --tags`). Two consequences worth carrying:
  - Precision below ~0.55 is **understated** for RampNet, and by an amount that varies by
    city — so the anchoring asymmetry is not a constant that cancels in a ranking.
  - The A-rate is measured *only* on RampNet's own low-confidence detections. It says the
    GT is incomplete; it does **not** say by how much a challenger is penalised, since a
    challenger's misses are a different population. Do not subtract it from anyone's score.

  Only richmond and bend have been corrected — clovis, morgantown, annapolis,
  budapest_district5 and manual_gold have not (see the coverage matrix at the top), so no
  cross-city GT-completeness constant should be quoted yet. Two cities is also too few to
  tell whether the 17%/29% spread is imagery, reviewer, or noise.
- **Box → point reduction.** Box models are scored by their box centers, at the same radius
  as RampNet's point detections. Localization differences finer than the radius aren't
  measured. Molmo is the exception — it emits points natively, so nothing is reduced.
- **Equirectangular projection disadvantages the challengers.** RampNet was trained on
  2048×4096 equirect panos; the others were not, and ramps are tiny in a warped 4k+ pano.
  The fair input is **perspective reprojection** (default, below): the pano is reprojected
  into overlapping rectilinear views. Whole-pano (`--tiling none`) remains available as a
  lower bound.
- **No AP for *chat* VLMs.** Gemini/Qwen box detection carries no calibrated per-box
  confidence, so those rows are a single **operating point** and their AP column reads `-`.
  AP / PR curves (via `rampnet.metrics`) are reported wherever confidences exist: RampNet,
  OWLv2, and Grounding DINO.
- **AP is measured on a slightly different slice than P/R.** AP needs one consistent recall
  denominator, so it is computed over the **recall-confirmed panos** only (the `no_missed`
  gate), while the precision column counts detections on every pano. On the current bundles
  nearly every pano is recall-confirmed, so the two slices are close — but they are not the
  same set. Note also that `--op-threshold` truncates the curve it is computed from: the AP
  printed alongside a thresholded row is the AP *of that row's operating range*, so quote
  full-range AP from a run without `--op-threshold`.
- **AP is not comparable across models at different floors — and RampNet's is truncated.**
  Every model's curve stops where its detections stop. RampNet's bundle detections were
  extracted at a **0.5** peak threshold, so its curve has no low-confidence tail at all
  (visible in `--sweep`: every row below 0.5 is identical) and its AP — 0.763 richmond /
  0.754 bend — is a **lower bound**, close to its recall ceiling of 0.768 / 0.761 times a
  near-1.0 precision envelope. The open detectors are cached down to 0.05, so their curves
  extend into a region RampNet's simply doesn't cover. Compare AP between OWLv2 and
  Grounding DINO freely; against RampNet, compare operating points, or re-extract RampNet's
  detections at a lower peak threshold first.
- **A swept threshold is tuned on the test set.** The `--sweep` table's best-F1 row is
  chosen on the benchmark itself. There is no separate val split, so quote it as an
  optimistic upper bound on what threshold tuning buys, not as a held-out result.

## VLM input: perspective reprojection (fair) vs whole-pano (lower bound)

`--tiling perspective` (default) reprojects the pano into a ring of overlapping rectilinear
views (`equirect_tiling.default_views`: 90° FOV, −30° pitch toward the ground, 6 yaws 60°
apart → 30° overlap), runs the detector per view, maps each detection's center back to pano
coordinates (`perspective_point_to_equirect`), and merges detections across the overlaps
(`dedup_points`, with 0/1 seam wrap). This is what a VLM expects — undistorted photos —
so it's the fair comparison. `--tiling none` sends one downscaled whole-pano call as a
lower bound.

**Seams.** Neighboring views overlap by 30°, so a ramp near a tile boundary is seen whole
in at least one view; the duplicate detection from the adjacent view is merged by
`dedup_points` (within the match radius). Residual nuance: a ramp truncated at a tile edge
can yield a box whose center is offset enough to escape the dedup radius and double-count as
a false positive — the mitigation is enough overlap that each ramp is near-centered in some
view, which is a rig-tuning question (`fov_h_deg` / `n_yaw` / `pitch_deg`) to calibrate
empirically once the live VLM runs.

## Validating the reprojection

- **Numerical** (`tests/test_equirect_tiling.py`): round-trip identity (view point → pano →
  view recovers the original), the view center looks at its (yaw, pitch), and the
  gnomonic-correctness invariants — the horizon renders as a straight horizontal and
  meridians as straight verticals (a warped projection would bend them). Plus renderer/scalar
  agreement.
- **Visual** (`scripts/model_comparison/dump_views.py`): renders a real pano's views with a
  graticule overlay. Great-circle meridians + the equator must render as **straight lines**;
  buildings/poles should look like normal photos. `python scripts/model_comparison/dump_views.py
  benchmark/richmond --out <dir>`.

## Validating the box mapping

Reprojection is only half the pipeline; the other half is turning a provider's boxes back into
pano points, and that half has a silent failure mode — **box coordinate conventions differ by
provider and even between Qwen generations**. `scripts/model_comparison/dump_detections.py`
overlays a detector's raw boxes (red) on each view together with the pano's ground-truth ramps
(green) and ignore points (amber), so a mapping error shows up as boxes sitting consistently
off the ramps:

```bash
python scripts/model_comparison/dump_detections.py benchmark/richmond \
    --model qwen:Qwen/Qwen3-VL-8B-Instruct --out view_dump/qwen
```

### Qwen box coordinates are normalized 0–1000

`gemini_boxes_to_points` divides by 1000; `qwen_boxes_to_points` takes an explicit
`coord_space` because the family changed convention:

- **Qwen3-VL** (`norm1000`, the default): `bbox_2d = [x1, y1, x2, y2]` normalized to **0–1000**,
  as in the upstream 2D-grounding cookbook (`bbox_2d[0] / 1000 * width`). Being
  resolution-independent, the processor's smart-resize (which rounds to multiples of 28)
  **cannot** shift them — this retires the earlier "normalize by the processed size" caveat.
- **Qwen2/2.5-VL** (`pixels`): absolute pixels of the image the processor actually fed the model.

`infer_qwen_coord_space` picks by model id; `--qwen-coord-space` overrides. The two are *not*
auto-detected, because at a 1024px view they differ by only 2.4% — a wrong choice does not
crash, it introduces a small systematic localization bias. Verified empirically by rendering
one view at 512 / 1024 / 1400 px: the returned coordinates stayed in the same ~0–1000 band
instead of scaling with the image, and the overlay put boxes squarely on tactile ramps.

`dump_detections.py` draws all three prediction shapes: plain boxes (Gemini, Qwen), **scored**
boxes (OWLv2, Grounding DINO — the score is printed next to each box, since that is the
number the threshold sweep tunes), and **points** (Molmo, drawn as a red crosshair-in-circle
with the same visual weight as a box, so a scale error is equally obvious).

## What each model class buys you

### Open-vocabulary detectors: real confidences, so a real curve

`OwlV2Detector` and `GroundingDinoDetector` are text-prompted *detectors*, not chat models:
the "prompt" is a short query (`"a photo of a curb ramp"` for OWLv2, which is CLIP-based;
`"curb ramp."` — lowercase, period-terminated — for Grounding DINO), and every box comes
back with a **calibrated score**. The harness threads that score all the way through
(`pixel_boxes_to_points` → `dedup_points` keeps the highest-scoring copy of a cross-view
duplicate → `score_pano` matches greedily in score order), which unlocks three things no
chat VLM in this harness can offer:

- **AP** in the main table,
- **PR curves** (`--pr-out DIR` → one JSON per model plus a combined PNG),
- a **threshold sweep** (`--sweep`) — P/R/F1 at each cutoff, best-F1 row flagged.

That last one matters directly for the recall-first direction: a detector you can *tune*
toward recall is worth more than a chat model pinned at one operating point.

**`--score-threshold` is a cache floor, not the operating point.** Detections are computed
once down to a low score (default **0.05**) and cached; every higher operating point is then
a free local re-score (`--op-threshold`, `--sweep`) with no second model run. The floor is
part of the detector signature, so *lowering* it invalidates the cache and re-runs the model
— raising the reported threshold never does. The sweep only prints rows at or above the
floor: below it the cache holds no detections, so those rows would just repeat the floor row
while reading as real measurements.

**OWLv2's boxes are relative to a padded square.** Its image processor pads to
`max(h, w)` (bottom/right) before resizing, so boxes live in that square's frame with the
image in the top-left corner; `owlv2_target_size` states that frame explicitly and
`pixel_boxes_to_points` normalizes by the *original* width/height, dropping centers that
land in the pad. Current transformers already scales OWLv2 boxes by `max(h, w)` internally
(`_scale_boxes`: *"for owlv2 image is padded to max size"*), so on this version passing the
square and passing the image's own `(h, w)` agree — verified on a 2:1 crop, where both put
the top box at y 0.815 against a true position of 0.817. Square views (the default rig) are
unaffected either way; whole-pano mode (`--tiling none`) is the only place the distinction
could bite, and passing the square is also correct under the older per-axis scaling.

### Molmo: points, not boxes

Molmo is the one challenger whose native output is a **point**, which is RampNet's own
output format — so it is the only apples-to-apples comparison in the table, with no
box→center reduction. There is no per-point score, so Molmo gets an operating point but no
PR curve.

**Its coordinate convention changed between generations**, and unlike Qwen's two box
conventions the two are distinguishable by *syntax*, so `molmo_points_from_text` infers the
scale per tag (override with `--molmo-coord-scale`):

- **Molmo 1** — `<point x="35.4" y="61.2" alt="...">` / `<points x1=… y1=… x2=… y2=…>`:
  coordinates are **percentages (0–100)**.
- **Molmo 2** — `<points coords="0 354 612; 1 700 480"/>`, triplets of `id x y`:
  coordinates are **scaled by 1000**, per the model card's own regex. (Issue #39 expected
  0–100 for all of Molmo; that holds for Molmo 1 only.)

A wrong scale here fails loudly rather than silently: points outside `[0,1]` after scaling
are dropped (as the model card's reference implementation does), so mis-scaled 0–1000
numbers divided by 100 land out of frame and the model appears to detect nothing.

`MolmoPoint-8B` is different again — it emits points as **special tokens** that only the
model can decode (`extract_image_points`, with metadata from the processor and a
constrained-decoding logits processor). `infer_molmo_mode` picks that path by model id;
`molmo_token_points_to_items` reads only the last two values of each returned row, because
the model card documents the leading ids two different ways.

**The `transformers==4.57.1` pin on the Molmo cards is real, not cautionary.** Under 5.14.1
Molmo's own Hub code fails at import:

```
TypeError: Unexpected keyword argument image_use_col_tokens.
  ... transformers_modules/allenai/Molmo2_hyphen_8B/.../processing_molmo2.py line 93
```

It also needs `einops` **and `requests`**, which the lean cluster env lacked. The fix is a
**dedicated env at the pin**, not a downgrade of the env the other models use — the harness
itself imports fine on 4.57.1, so only Molmo's interpreter differs:

```bash
conda create -p /gscratch/scrubbed/$USER/envs/molmo python=3.11 -y
MOLMOPY=/gscratch/scrubbed/$USER/envs/molmo/bin/python
$MOLMOPY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
$MOLMOPY -m pip install "transformers==4.57.1" accelerate pillow numpy einops requests
PYTHON=$MOLMOPY MODELS=rampnet,molmo:allenai/Molmo2-8B \
    sbatch -A <account> scripts/model_comparison/run_open_models.slurm
```

**Status: run and scored** (richmond/bend/clovis/annapolis/manual_gold, above). Verifying the
overlay first was not
optional — the first run put every point on the left edge because Molmo's `coords` list
opens with an **image index** the parser mistook for a point id (fixed; see the parser note
below). After the fix the crosshairs sit on ramps and the numbers are the F1-0.45 rows above.
The lesson stands for the next pointing model: `dump_detections.py` on one pano first, and if
nothing is detected the scale is wrong (try `--molmo-coord-scale`).

## Status

- **Shipped:** the model-agnostic scorer (`rampnet/detection_eval.py`), the comparison CLI
  (`scripts/model_comparison/compare.py`), the RampNet-from-bundle baseline
  (`BundleRampNetDetector`), the **perspective reprojection + dedup** (`equirect_tiling.py`),
  the **live `GeminiDetector`** (google-genai; API key or Vertex+ADC), the **live
  `QwenDetector`** (transformers; Qwen3-VL on a cluster GPU), the **live `OwlV2Detector` /
  `GroundingDinoDetector`** (with AP, PR curves and a threshold sweep), and the
  **`MolmoDetector`** (points; `Molmo2-8B` verified by overlay and run on all four scored
  cities — see the Molmo section; `MolmoPoint-8B`'s special-token path is wired but has not
  met real weights). Tested (`test_detection_eval.py`, `test_model_comparison.py`,
  `test_equirect_tiling.py`).
- **Open, and tracked in the coverage matrix at the top:** morgantown and
  budapest_district5 have RampNet rows but no challenger runs, and `manual_gold` has no
  null-recall pass. Those are the only gaps between `benchmark/` and this document.
- **Smoke-tested locally** on `Qwen/Qwen3-VL-2B-Instruct` (the largest that fits an 8 GB dev
  GPU) to validate wiring, JSON parsing, and box mapping before spending cluster time. 2B is
  far too weak to benchmark — the real runs are 8B and 32B on Hyak.
- **Where runs happen:** benchmark numbers come from **Hyak** (or makelab2), never the dev
  box. The desktop is for de-risking a cluster job — a 1–2 pano wiring probe and a
  `dump_detections.py` overlay — and those results are smoke tests, not results.
- **Desktop and cluster agree exactly.** The 2-pano smoke on an RTX 3070 and on an L40S
  produced *identical* numbers (OWLv2 18/156/1/5, AP 0.356; Grounding DINO 18/160/1/3, AP
  0.247), and the overlay job reproduced the same 94 OWLv2 boxes across the same six views.
  So a desktop probe is a faithful rehearsal of the cluster job — worth knowing before
  spending an allocation on a wiring bug.

## Gemini credentials

The `GeminiDetector` reads credentials from the environment; `compare.py` auto-loads a
git-ignored repo-root `.env` (so nothing lands in the shell or transcript). Two options:

- **Vertex AI + ADC** (for orgs that disallow API keys). In `.env`:
  ```
  GOOGLE_GENAI_USE_VERTEXAI=true
  GOOGLE_CLOUD_PROJECT=your-project-id
  GOOGLE_CLOUD_LOCATION=global
  ```
  and once, in your own terminal:
  `gcloud auth application-default login && gcloud auth application-default set-quota-project <project>`
  (the SDK finds the ADC file automatically at runtime; gcloud itself isn't needed after login).
- **API key** (if allowed): `GOOGLE_API_KEY=...` in `.env`.

**Location matters for model availability.** The newest Gemini flash ids
(`gemini-3.6-flash`, `gemini-3.5-flash`) are served only on the `global` Vertex location;
regional endpoints (e.g. `us-west1`) lag — there they cap at `gemini-2.5-flash`. Use
`global` unless an org data-residency policy requires a region (the benchmark imagery is
public GSV/Mapillary, so residency is not a concern here). Vertex model ids differ from the
AI-Studio aliases (`gemini-flash-latest` only resolves on `global`); pin them explicitly with
`gemini:<model-id>` in `--models`.

## Running it

```bash
# RampNet baseline (no GPU, no keys — reads detections from the bundle):
python scripts/model_comparison/compare.py benchmark/richmond --models rampnet

# RampNet vs Gemini variants (needs credentials above). Each --models token is a
# provider or provider:model_id; variants of one provider become separate rows:
python scripts/model_comparison/compare.py benchmark/richmond \
    --models rampnet,gemini:gemini-2.5-flash,gemini:gemini-3.6-flash

# Cost control / smoke: cap panos; whole-pano lower bound instead of tiling:
python scripts/model_comparison/compare.py benchmark/richmond --models rampnet,gemini --limit 20
python scripts/model_comparison/compare.py benchmark/richmond --models gemini --tiling none

# Qwen3-VL (open weights, needs a GPU — see the Hyak runbook below):
python scripts/model_comparison/compare.py benchmark/richmond \
    --models rampnet,qwen:Qwen/Qwen3-VL-8B-Instruct

# Open-vocabulary detectors: AP in the table, plus the curve and the sweep.
python scripts/model_comparison/compare.py benchmark/richmond \
    --models rampnet,owlv2,gdino --sweep --pr-out evaluation_results/pr_richmond

# Scoring only (no GPU, no model load) once .model_cache holds the detections.
# Every operating point is a free re-score of that cache:
python scripts/model_comparison/compare.py benchmark/richmond \
    --models rampnet,owlv2,gdino --op-threshold 0.2
```

A model that can't run (missing credentials, missing client lib, remote code that won't load
on this transformers version) is skipped with a clear note rather than crashing the run — so
one broken model can't cost you the models that already ran.

## Running the open-weight models on Hyak

Benchmark runs go on the cluster, not the dev box — Qwen3-VL-8B is ~16 GB in bf16 (32B
~64 GB) and Molmo-8B ~16 GB, and even the small detectors should produce their reported
numbers where every other model's came from. Two launchers:

- `scripts/model_comparison/run_qwen.slurm` — the Qwen leg.
- `scripts/model_comparison/run_open_models.slurm` — OWLv2 + Grounding DINO (default), or
  Molmo via `MODELS=`. OWLv2-large and Grounding DINO-base are ~1–2 GB and finish in
  minutes on one card; Molmo-8B takes hours because it generates text per view.

**The results come back through the detection cache.** `cache_key` hashes only
`(label, detector signature, city, pano id)` — nothing machine-specific — so detections computed
on Hyak drop straight into a local `.model_cache/`. And when every pano of a model is already
cached, `score_model` skips `detector.prepare()` entirely, so the final table can be produced on
a laptop that cannot load Qwen at all.

```bash
# 1. Stage the repo plus the (git-ignored) bundle imagery. Send the NATIVE panos:
#    the harness downscales in-process, and pre-resizing re-encodes the JPEG,
#    which is not free (a past gold-set re-eval moved P +2.2 / R -1.8 on
#    re-encoding alone).
rsync -av --exclude .venv --exclude .model_cache --exclude 'benchmark/*/panos' \
      RampNet/ klone:~/RampNet/
rsync -av benchmark/richmond/panos/ klone:~/RampNet/benchmark/richmond/panos/

# 2. On a login node: build an env. The full environment.yml works (remember
#    CONDA_OVERRIDE_CUDA=12.6, or conda-forge silently installs CPU-only torch),
#    but this leg needs only numpy/PIL/torch/torchvision/transformers -- the
#    RampNet baseline reads detections from the bundle, so no timm, no model load.
#    A lean env off the CUDA wheel index is faster and has no CPU-fallback trap:
module load conda/Miniforge3-25.9.1-0
conda create -p /gscratch/scrubbed/$USER/envs/qwenvl python=3.11 -y
ENVPY=/gscratch/scrubbed/$USER/envs/qwenvl/bin/python
$ENVPY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
$ENVPY -m pip install "transformers>=4.57" accelerate pillow numpy

# 3. Pre-download the weights so the GPU job isn't billed for the transfer.
#    (~17 GB for Qwen-8B; OWLv2-large + Grounding DINO-base are ~2 GB together.)
export HF_HOME=/gscratch/scrubbed/$USER/hf
$ENVPY -c 'from huggingface_hub import snapshot_download as d; [d(m) for m in [
    "Qwen/Qwen3-VL-8B-Instruct",
    "google/owlv2-large-patch14-ensemble",
    "IDEA-Research/grounding-dino-base"]]'

# 4. Submit. -A is required (find yours: sacctmgr -nP show assoc user=$USER
#    format=Account,QOS). 8B fits one L40S; 32B needs two (device_map shards it).
#
#    Partition: the launchers default to `-p gpu-l40s` (non-preemptible, lab-capped
#    at 2 GPUs, so legs queue behind each other). The ckpt scavenger queue is faster
#    when it's free and preemption only costs ~one pano of cache -- but CHECK
#    `squeue -u $USER` FIRST. If a long run is already parked there (the issue #51
#    YOLO baseline lives on ckpt-g2 for ~1-2 weeks), stay on gpu-l40s: these legs
#    are 12-30 min each and are not worth any risk to a multi-day job.
mkdir -p logs
export PYTHON=$ENVPY
sbatch -A <account> scripts/model_comparison/run_qwen.slurm
BUNDLE=benchmark/bend sbatch -A <account> scripts/model_comparison/run_qwen.slurm
# 32B needs two cards. Use --gpus-per-node, NOT --gpus: the launcher already
# sets --nodes=1 --gpus-per-node=1, and `--gpus=2` against that is rejected
# ("required nodes (2) doesn't fall between min_nodes (1) and max_nodes (1)").
QWEN_MODEL=Qwen/Qwen3-VL-32B-Instruct sbatch -A <account> --gpus-per-node=2 \
    scripts/model_comparison/run_qwen.slurm

# 4b. The open-vocabulary detectors: minutes, one card, both cities.
sbatch -A <account> scripts/model_comparison/run_open_models.slurm
BUNDLE=benchmark/bend sbatch -A <account> scripts/model_comparison/run_open_models.slurm

# 4c. Molmo (hours — it generates text per view). Verify the point mapping on one
#     pano FIRST — this overlay is what caught the image-index parsing bug on the
#     first real run (see the Molmo section), and any new checkpoint can pull the
#     same kind of trick:
$ENVPY scripts/model_comparison/dump_detections.py benchmark/richmond \
    --model molmo:allenai/Molmo2-8B --out view_dump/molmo
MODELS=rampnet,molmo:allenai/Molmo2-8B \
    sbatch -A <account> scripts/model_comparison/run_open_models.slurm

# 5. Bring the detections home and score every model side by side, no GPU needed.
rsync -av klone:~/RampNet/.model_cache/ .model_cache/
python scripts/model_comparison/compare.py benchmark/richmond --sweep \
    --pr-out evaluation_results/pr_richmond \
    --models rampnet,gemini:gemini-3.6-flash,qwen:Qwen/Qwen3-VL-8B-Instruct,owlv2,gdino
```

Runs are resumable: a job that is preempted or times out has already cached everything it
finished, so re-submitting picks up where it stopped.

## The manual_gold split: bigger, un-anchored, in-distribution (issue #58)

`benchmark/manual_gold` scores the same model roster against the repo's 1,000-pano manually
labeled gold set (`manual_labels/*.txt` — 3,919 curb ramps, 207 negative panos; imagery =
the GSV test split of `projectsidewalk/rampnet-dataset`). It exists to answer the question
the city splits structurally cannot: **does the ranking hold when the ground truth was
never anchored to a RampNet review?** The city GT was assembled while verifying RampNet's
detections; the gold set was labeled from scratch, with no model in the loop. If challenger
recall jumps here relative to the cities, that quantifies the anchoring-bias caveat that
travels with every number above — and if it doesn't, that caveat can finally be retired
with evidence. At 4× the size of the largest city split, it also tightens every Wilson CI.

It **complements** the deployment cities; it does not replace them and its numbers must
not be pooled with theirs:

- **In-distribution, home-field.** GSV imagery from the training cities (NYC / Portland /
  Bend), held out of Stage-2 training but squarely inside RampNet's training distribution.
  Frame it exactly like the `bend` split: an in-domain reference, not a generalization test.
  richmond/clovis remain the OOD deployment stories.
- **No ignore points.** The labels carry no `unsure` class, so no model gets the abstention
  the city splits grant — uniformly, but scores are slightly harsher here by construction.
- **Same box→center reduction, both sides of the match.** GT is YOLO box centers; VLM boxes
  reduce to centers as everywhere else. `fn_confirmed` is True on every pano (full manual
  labeling is a complete scan), so all 1,000 panos — including the 207 negatives, where VLM
  hallucination shows up as pure FPs — are in both the precision and recall pools.
- **Zero pano overlap** with bend/richmond/clovis (`scripts/fetch_manual_gold.py --audit`
  verifies this plus HF split membership).
- **Lower-res source imagery than the city bundles.** Gold panos are stored at the HF
  dataset's resolution, not the native 16k GSV the `bend` bundle carries; the perspective
  views the VLMs see are rendered from that.

### Building it

```bash
python scripts/fetch_manual_gold.py --audit    # id-only membership/overlap audit, no download
python scripts/fetch_manual_gold.py            # HF test split (~44 GB) -> panos/ + records.jsonl
python scripts/export_gold_records.py --checkpoint <stage2.pth>   # RampNet detections + gate

# or, on Hyak, both steps as one resumable Slurm job (fetch on CPU, export on the GPU):
CHECKPOINT=/path/to/stage2.pth sbatch -A <account> scripts/run_gold_bundle.slurm

python scripts/model_comparison/compare.py benchmark/manual_gold \
    --models rampnet --op-threshold 0.55 --sweep
```

The fetch writes the parquet's **raw image bytes** (a past gold-set re-eval moved
P +2.2 / R -1.8 on JPEG re-encoding alone — do not fetch through `download_dataset.py`,
which re-saves at quality 95). The exporter reuses `stage_two/evaluate.py`'s exact
inference path (TTA, peak extraction) and ends with a **reproduction gate**: scored through
this harness at conf >= 0.55, the exported detections must land on the published gold-set
numbers (P 0.949 / R 0.873). That single check validates the fetch, the inference config,
the YOLO→point conversion, and the scorer — run it before spending anything on challengers.
Unlike the city bundles (extracted at 0.5, truncating RampNet's PR curve — see the AP
caveat above), gold detections are exported down to a 0.05 floor, so RampNet gets an
untruncated AP on this split.

There are no verdicts here, so the RampNet verdict-based cross-check is skipped
(`score_validation.py` / `gt_gallery.py` don't apply); the reproduction gate is this
split's equivalent.

### Cost and runtime (plan before submitting)

1,000 panos ≈ **8× clovis**: ~6,000 tiled views per VLM. Gemini runs are ~8× the clovis
API spend per model — smoke with `--limit 20` first; the detection cache makes every re-run
free. Actuals from the 2026-07-25 runs (all cheaper than budgeted): OWLv2+GDINO pair 1h33m
on one L40S; Qwen-8B 1h57m (one L40S); Qwen-32B 3h46m (two A40s); Molmo2-8B 3h49m of GPU
total across two jobs (the cache carried the first job's detections into the second);
Gemini needs no GPU at all — both models ran from a desktop against Vertex, pro-preview in
~4 h, flash in ~9 h. The GPU legs all ran happily on the **ckpt scavenger queue**
(preemption costs ~one pano of cache).

### Results (all 8 model groups)

Same protocol as the city tables: perspective tiling, match radius 0.022, box centers both
sides. RampNet is shown at its published 0.55 operating point (the same detections score
P 0.723 / R 0.935 / F1 0.815 at the 0.05 export floor); open detectors are shown at their
0.05 cache floor with tuned operating points noted below.

**manual_gold** (1,000 panos, 3,919 GT ramps, 207 negative panos — GT labeled with no model
in the loop)

| model | P | R | F1 | AP | tp/fp/fn |
|---|---|---|---|---|---|
| **rampnet** @0.55 | **0.947** | 0.873 | **0.908** | 0.917 | 3420/190/499 |
| gemini-3.1-pro-preview | 0.653 | 0.503 | 0.568 | – | 1972/1047/1947 |
| gemini-3.6-flash | 0.609 | 0.485 | 0.540 | – | 1899/1221/2020 |
| **molmo2-8B** (points) | 0.511 | 0.360 | **0.422** | – | 1409/1346/2510 |
| Qwen3-VL-8B-Instruct | 0.445 | 0.341 | 0.386 | – | 1338/1667/2581 |
| Qwen3-VL-32B-Instruct | 0.739 | 0.177 | 0.285 | – | 693/245/3226 |
| owlv2-large-patch14-ensemble | 0.046 | **0.906** | 0.088 | 0.097 | 3551/73444/368 |
| grounding-dino-base | 0.043 | 0.855 | 0.082 | 0.067 | 3351/74953/568 |

Best sweep F1 for the open detectors: OWLv2 **0.180** (thr 0.20), Grounding DINO **0.140**
(thr 0.20) — the FP flood is not a threshold artifact.

What this split adds to the story:

1. **The ranking is identical to all three deployment cities, on ground truth that never
   saw a RampNet review** — and RampNet's lead is the widest measured anywhere: **0.34 F1**
   over the best challenger (vs ~0.19 on richmond, ~0.30 on clovis), at CIs roughly 4×
   tighter than any city split (3,919 GT points vs 195–327).
2. **The anchoring-bias caveat can now be retired with evidence.** If the city GT had been
   tilted toward what RampNet finds, challengers would gain here — real-but-unlabeled ramps
   that scored as challenger FPs in the cities would become TPs. Neither signature appears:
   Gemini-3.1-pro's recall is 0.503 here vs 0.581–0.700 on the GSV cities, and its precision
   (0.653) sits inside its city range (0.531–0.706). The same holds down the roster. The
   city numbers were not an artifact of RampNet-anchored ground truth.
3. **RampNet's in-distribution advantage is now quantified, not asserted**: recall 0.873 on
   home-field GSV imagery vs 0.713–0.768 deployed OOD, at essentially the same precision.
   Frame it exactly like `bend`: an in-domain reference, not a generalization claim.
4. **Every city-level pattern recurs**: Qwen-32B is again the cautious one (challenger-best
   P 0.739 at R 0.177 — capacity moves the operating point, not F1), Molmo is again the best
   open-weight model with the most balanced profile, and the open-vocab detectors again pair
   a real recall ceiling (0.85–0.91 at the floor) with unusable precision (~73–75k FPs
   across the split — and with 207 negative panos in every pool, hallucination costs are
   fully counted here).
5. **RampNet's row doubles as the reproduction gate**: scored through this harness, the
   exported detections land on the published gold-set numbers (P 0.947 / R 0.873 vs
   published 0.949 / 0.873), and the 0.05 export floor gives it the one untruncated AP in
   the comparison (0.917).

## Next increments

1. **Calibrate the reprojection rig — now measurable, and demonstrably costly.** With
   `pitch_deg=-30` the bottom ~40% of every view is the capture vehicle's hood and the black
   nadir cap, so roughly a third of every paid call is spent on pixels that cannot contain a
   curb ramp. The open-detector overlays show this is not merely wasteful: Grounding DINO's
   **highest-scoring box in a view (0.40) is the hood itself**, outranking its correct 0.22
   box on a real tactile pad. Because AP ranks by score, hood detections at the top of the
   ranking depress AP directly — which is a plausible part of why Grounding DINO's AP (0.032)
   trails OWLv2's (0.104) despite similar operating points. Masking the nadir/hood region is
   now a change whose benefit can be *measured* (ΔAP), not just argued. Report perspective vs
   `--tiling none` side by side.
2. **Run `MolmoPoint-8B`** (its special-token path is wired but has never met real
   weights — overlay first) and decide whether it or `Molmo2-8B`'s XML path, already run
   and scored above, is the one to report.
3. **Prompt-sweep the open detectors before writing them off entirely.** `--owlv2-query` /
   `--gdino-query` are free hyperparameters and these models are cheap to run (43
   detections/min on one L40S; a full city is ~15 min). The current queries are a single
   untuned phrase each; "curb cut", "wheelchair ramp at a crosswalk", or a multi-query
   ensemble might move them. Given F1 0.184 vs RampNet's 0.855 this will not change the
   verdict, but it would tell us whether the ceiling is the *query* or the *model class*.
4. **If a recall-first candidate generator is wanted, compare complements on FP-per-find,
   not F1.** That metric ranks Gemini-3.6-flash (~6) far above OWLv2 (36–128) despite
   OWLv2's much higher recall ceiling — see the table above. Discount both against the null
   first: at OWLv2's density most "finds" are free.
5. **Close the two coverage gaps.** morgantown and budapest_district5 have RampNet rows and
   no challengers. Neither is expensive — 125 panos each is ~1.5 h of one L40S for the four
   open-weight models plus a couple of desktop-hours of Vertex calls for the two Geminis, and
   the imagery is already in `benchmark/*/panos/`. budapest is the higher-value of the two
   (a rubric-robustness test no US split can run), with the standing caveat that its GT is
   low-confidence; morgantown is the cleanest imagery in the benchmark and the natural
   control. Neither should be started expecting the ranking to change.
6. **Distance-stratify the null-recall table.** The open detectors' recall is mostly a density
   artifact, and RampNet's misses are mostly far-field (`benchmark/README.md`, annapolis).
   Whether the challengers see anything real in the far field is currently *unanswered* —
   the headline recall can't settle it and neither can the null as computed. Real-vs-null
   recall for near and far ramps separately would.
7. **Subsample the null for `manual_gold`.** It is the one split with challenger runs and no
   null-recall pass, because the shift-average is O(n²) in panos. A fixed random subset of
   shifts would give the same estimate at a fraction of the cost.

## Files

- `rampnet/detection_eval.py` — model-agnostic GT + scorer, AP/PR curve (pure, torch-free);
  includes the YOLO manual-label loader for `manual_gold`.
- `scripts/fetch_manual_gold.py` — `manual_gold` imagery + records from the HF test split
  (plus the `--audit` id checks).
- `scripts/export_gold_records.py` — RampNet detections for `manual_gold` + the
  reproduction gate (GPU).
- `scripts/model_comparison/detectors.py` — `Detector` protocol, RampNet baseline, VLM /
  open-vocabulary / pointing detectors.
- `scripts/model_comparison/equirect_tiling.py` — perspective reprojection + point mapping + dedup.
- `scripts/model_comparison/compare.py` — comparison CLI (table, sweep, PR curves).
- `scripts/model_comparison/dump_views.py` — visual de-distortion QA (graticule overlay).
- `scripts/model_comparison/dump_detections.py` — visual mapping QA (boxes/points vs ground truth).
- `scripts/model_comparison/run_qwen.slurm` — Hyak launcher for the Qwen leg.
- `scripts/model_comparison/run_open_models.slurm` — Hyak launcher for OWLv2 / Grounding DINO / Molmo.
- `scripts/analysis/null_recall.py` — real vs chance recall at a model's box density (the
  "how much of a detector's recall is real?" table). Cache-only; no GPU, no keys.
- `requirements-vlm.txt` — optional VLM deps.
- `tests/test_detection_eval.py`, `tests/test_model_comparison.py`,
  `tests/test_equirect_tiling.py` — guards.
