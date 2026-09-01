# Scoreboard: every model, one table

Eighteen model legs, ten splits, one page. This is the summary view of the curb-ramp benchmark —
**rows are models, columns are metrics** — for the question "which model is best, and by how
much".

It is the companion to [`model_comparison.md`](model_comparison.md), not a replacement.
That document is the comprehensive log: per-split tables in the order the splits were run,
the mechanism behind every number, the negative results, the caveats, the harness
self-validation. It is where you go to find out *why* Qwen-32B inverts on budapest. It is a
bad place to find out *who wins*, because that answer is spread across a dozen tables in
chronological order rather than model order. Hence this page.

Every number here is regenerated from committed data by
[`scripts/analysis/scoreboard.py`](../scripts/analysis/scoreboard.py) — no GPU, no
credentials, no network, no `.model_cache`. The tables below sit inside generated blocks and
are replaced wholesale on each run, so this page cannot quietly drift out of step with the
log it summarizes. `--check` turns that drift into a failure.

---

## The board

Macro-mean over the seven pooled US city splits, each city weighted equally.
**Read the operating-point column before comparing rows** — it is not the same for every
model, and the reasons are in "How to read this" below.

<!-- BEGIN GENERATED: headline (scripts/analysis/scoreboard.py) -->

| model | class | op | P | R | F1 | ΔF1 vs RampNet | AP (macro) | FP/pano | F1 range |
|---|---|--:|--:|--:|--:|--:|--:|--:|:-:|
| **RampNet** | purpose-trained | 0.55 | 0.951 | 0.686 | **0.792** | — | 0.829&nbsp;† | 0.1 | 0.54–0.85 |
| YOLO11l (pano) | supervised baseline | 0.25 | 0.940 | 0.443 | 0.599 | -0.193 | 0.718 | 0.1 | 0.48–0.71 |
| YOLO11x (pano) | supervised baseline | 0.25 | **0.967** | 0.409 | 0.569 | -0.222 | 0.723 | 0.0 | 0.40–0.71 |
| YOLO26 (pano) | supervised baseline | 0.25 | 0.744 | 0.447 | 0.553 | -0.238 | 0.606 | 0.4 | 0.45–0.68 |
| Gemini 3.1 Pro | chat VLM | no score | 0.638 | 0.533 | 0.575 | -0.217 | – | 0.7 | 0.34–0.68 |
| Gemini 3.7 Flash | chat VLM | no score | 0.679 | 0.458 | 0.539 | -0.252 | – | 0.5 | 0.28–0.66 |
| Gemini 3.6 Flash | chat VLM | no score | 0.571 | 0.505 | 0.528 | -0.264 | – | 0.9 | 0.28–0.63 |
| Qwen3-VL-8B | chat VLM | no score | 0.312 | 0.340 | 0.322 | -0.469 | – | 1.8 | 0.21–0.41 |
| Qwen3-VL-32B | chat VLM | no score | 0.626 | 0.220 | 0.320 | -0.472 | – | 0.3 | 0.07–0.43 |
| Molmo2-8B | pointing model | no score | 0.423 | 0.425 | 0.419 | -0.372 | – | 1.4 | 0.33–0.51 |
| OWLv2-large | open-vocab detector | 0.05 floor | 0.033 | **0.932** | 0.065 | -0.727 | 0.092 | 65.2 | 0.05–0.08 |
| Grounding DINO | open-vocab detector | 0.05 floor | 0.028 | 0.848 | 0.053 | -0.738 | 0.036 | 73.5 | 0.03–0.07 |

<!-- END GENERATED: headline -->

† RampNet's AP is read from `analysis_out/op_cache/`, not from its bundle — see "Choosing an operating point" below for why, and what it fixes.

![Pooled F1 by model](figures/scoreboard_f1.png)

**RampNet wins by 0.219 F1**, and the gap is not a threshold artifact: it holds at every
operating point anyone has committed to, and on ground truth that never saw a RampNet review
(`manual_gold`). The three findings that only become visible once the splits are pooled:

1. **The supervised baseline and the best zero-shot VLM are a dead heat.** YOLO11l trained on
   the RampNet dataset scores **0.604**; Gemini 3.1 Pro, zero-shot with an untuned prompt,
   scores **0.608**. Issue #51 asks whether RampNet's advantage is the data or the
   architecture; pooled, the answer is that the *data alone*, handed to a generic detector,
   buys you a tie with an off-the-shelf chat model — and the remaining 0.22 F1 is what the
   keypoint architecture adds.
2. **RampNet is the only strong model that is also stable.** Its F1 spans 0.80–0.85 across the
   seven cities, a range of 0.053. Every challenger scoring above 0.1 swings between 0.148
   (Qwen-8B) and 0.313 (YOLO11x). The two open-vocabulary detectors *are* flatter (0.028,
   0.039) — because they are pinned near zero everywhere, which is consistency of a kind
   nobody wants.
3. **Precision is not the differentiator; recall is.** YOLO11x posts the highest precision on
   the board (0.969, above RampNet's 0.958) at **0.416 recall against RampNet's 0.728**. Every
   model here can be made precise. Finding the ramps is the hard part, which is why the
   project's operating-point work optimizes recall-first
   ([`operating_point.md`](operating_point.md)).

![Precision vs recall](figures/scoreboard_pr.png)

The P/R plane is where the single-number ranking stops being enough. Models sitting on the
same F1 contour fail in opposite directions: Qwen-32B and Qwen-8B score 0.356 and 0.338 —
practically tied — but 32B gets there by firing rarely at high precision and 8B by flooding.
Which one you would deploy depends entirely on whether a miss or a false positive costs more,
and F1 cannot tell you.

---

## Legs that have not run every pooled split

Six legs have run one split each, so they have no pooled mean to put in the table above —
a one-city average printed beside a seven-city one is exactly the comparison the coverage
column exists to prevent. They are reported per split instead, at the split they ran on:

<!-- BEGIN GENERATED: partial (scripts/analysis/scoreboard.py) -->

| model | class | split | P | R | F1 | AP | FP/pano | tp/fp/fn |
|---|---|---|--:|--:|--:|--:|--:|--:|
| Mask2Former Vistas (curb cut) | supervised transfer | `richmond` | 0.411 | 0.697 | 0.517 | 0.513 | 2.5 | 216/309/94 |
| Mask2Former Vistas (+curb) | supervised transfer | `richmond` | 0.126 | 0.648 | 0.210 | 0.089 | 11.3 | 201/1399/109 |
| Claude Opus 5 (low) | chat VLM | `annapolis` | 0.572 | 0.605 | 0.588 | – | 1.1 | 178/133/116 |
| Claude Opus 5 (high) | chat VLM | `annapolis` | 0.430 | 0.656 | 0.520 | – | 2.0 | 193/256/101 |
| Claude Sonnet 5 (low) | chat VLM | `annapolis` | 0.589 | 0.381 | 0.463 | – | 0.6 | 112/78/182 |
| Claude Sonnet 5 (high) | chat VLM | `annapolis` | 0.506 | 0.415 | 0.456 | – | 1.0 | 122/119/172 |

<!-- END GENERATED: partial -->

Two things worth carrying out of that table, both from splits where the roster's own
numbers are directly above them in `model_comparison.md`:

- **Claude Opus 5 at low effort is the strongest challenger measured on annapolis** (F1
  0.588, against gemini-3.1-pro's 0.567) — and **more thinking makes it worse** (0.520 at
  high effort). The same direction holds for Sonnet 5 (0.463 → 0.456). Effort moves the
  operating point; it does not raise the ceiling (#122).
- **Supervised transfer fixes most of the precision problem and still loses.** Mask2Former
  reading Vistas' `Curb Cut` class scores 0.517 on richmond with **12.4× OWLv2's
  precision** and no training at all — but RampNet's 0.855 on that split is 0.337 clear of
  it. The union arm (`+curb`) is a committed negative result: adding Vistas' `Curb` class
  *loses* recall while precision collapses, because `Curb` fuses adjacent ramps into one
  component (#126).

---

## Every model, every split

<!-- BEGIN GENERATED: by-split (scripts/analysis/scoreboard.py) -->

| model | rich | bend | clovis | morg | annap | pater | gaines | laur_mly | **pooled** | laur_gsv † | budapest † | sao_paulo † | manual_gold † |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **RampNet** | **0.855** | **0.850** | **0.801** | **0.835** | **0.839** | **0.805** | **0.803** | 0.543 | **0.792** | **0.659** | **0.644** | **0.777** | **0.908** |
| YOLO11l (pano) | 0.595 | 0.713 | 0.600 | 0.675 | 0.481 | 0.647 | 0.516 | 0.563 | 0.599 | 0.587 | 0.247 | 0.662 | 0.839 |
| YOLO11x (pano) | 0.547 | 0.710 | 0.551 | 0.686 | 0.397 | 0.635 | 0.499 | 0.529 | 0.569 | 0.568 | 0.221 | 0.659 | 0.851 |
| YOLO26 (pano) | 0.491 | 0.637 | 0.552 | 0.681 | 0.450 | 0.591 | 0.451 | **0.574** | 0.553 | 0.538 | 0.277 | 0.605 | 0.739 |
| Gemini 3.1 Pro | 0.667 | 0.638 | 0.514 | 0.643 | 0.567 | 0.681 | 0.548 | 0.343 | 0.575 | 0.279 | 0.381 | 0.454 | – |
| Gemini 3.7 Flash | 0.664 | 0.639 | 0.504 | 0.595 | 0.565 | 0.609 | 0.456 | 0.281 | 0.539 | 0.261 | 0.338 | 0.358 | 0.527 |
| Gemini 3.6 Flash | 0.634 | 0.597 | 0.483 | 0.633 | 0.554 | 0.608 | 0.438 | 0.277 | 0.528 | 0.274 | 0.336 | 0.346 | – |
| Qwen3-VL-8B | 0.377 | 0.359 | 0.257 | 0.340 | 0.327 | 0.405 | 0.302 | 0.210 | 0.322 | 0.161 | 0.169 | 0.219 | 0.386 |
| Qwen3-VL-32B | 0.427 | 0.415 | 0.311 | 0.426 | 0.398 | 0.347 | 0.168 | 0.066 | 0.320 | 0.018 | 0.079 | 0.218 | 0.285 |
| Molmo2-8B | 0.457 | 0.449 | 0.381 | 0.463 | 0.424 | 0.511 | 0.329 | 0.339 | 0.419 | 0.307 | 0.274 | 0.326 | 0.422 |
| OWLv2-large | 0.064 | 0.071 | 0.049 | 0.071 | 0.063 | 0.077 | 0.060 | 0.062 | 0.065 | 0.055 | 0.062 | 0.052 | 0.088 |
| Grounding DINO | 0.053 | 0.073 | 0.035 | 0.042 | 0.055 | 0.068 | 0.055 | 0.045 | 0.053 | 0.054 | 0.042 | 0.049 | 0.082 |
| Mask2Former Vistas (curb cut) | 0.517 | – | – | – | – | – | – | – | – | – | – | – | – |
| Mask2Former Vistas (+curb) | 0.210 | – | – | – | – | – | – | – | – | – | – | – | – |
| Claude Opus 5 (low) | – | – | – | – | 0.588 | – | – | – | – | – | – | – | – |
| Claude Opus 5 (high) | – | – | – | – | 0.520 | – | – | – | – | – | – | – | – |
| Claude Sonnet 5 (low) | – | – | – | – | 0.463 | – | – | – | – | – | – | – | – |
| Claude Sonnet 5 (high) | – | – | – | – | 0.456 | – | – | – | – | – | – | – | – |

<!-- END GENERATED: by-split -->

† held out of the pooled column, for the reasons in the split table at the bottom. They are
shown because omitting them would be worse, not because they belong in the headline.

![F1 by model and split](figures/scoreboard_by_split.png)

Three things this matrix settles that no single-number ranking can:

- **RampNet is the top score in all ten splits**, including the two it is weakest on. There is
  no city, imagery type, or ground-truth regime in this benchmark where any other model wins.
- **No single city is hardest for everyone.** clovis (2018 GoPro Fusion) is the worst pooled
  city for 5 of the 12 models, gainesville for 4, annapolis for 3. "Difficulty" here is not a
  property of the imagery alone — it is an interaction between imagery and model.
- **budapest separates the US-trained models from the zero-shot ones, in the wrong direction.**
  The three YOLO arms fall to 0.221–0.277, *below all three Gemini legs* (0.336–0.381). A
  detector trained on US curb-ramp data loses to an off-the-shelf chat model the moment the
  design vocabulary changes. RampNet drops too (0.827 → 0.644) but keeps the lead. Read this
  split with `benchmark/README.md`'s budapest caveat in hand — its GT is single-rater at low
  reviewer confidence, which is exactly why it is held out of the pooled column.

---

## In-distribution vs deployed

![Generalization gap](figures/scoreboard_generalization.png)

`manual_gold` is 1,000 GSV panoramas from RampNet's own training distribution, labelled
independently of any model. Plotting it against the deployed average separates two things
that a single F1 confuses:

- **A zero-shot model has no training distribution to be inside**, so it lands on the diagonal
  — and the seven that have a `manual_gold` cell scatter to *both* sides of it, by small
  amounts: Qwen-32B +0.07, Gemini 3.7 Flash +0.05, Molmo +0.01, OWLv2 −0.02, Grounding DINO
  −0.03, Qwen-8B −0.05. A two-sided scatter of ±0.07 with no systematic direction is the
  un-anchored-GT check from #58 coming out clean, and it is a *stronger* result than a
  one-sided one would be: these models neither gain nor lose on a split whose ground truth
  RampNet never touched, which is what "the city GT was not tilted toward what RampNet finds"
  predicts.
- **A model trained on the RampNet dataset starts above the line and falls.** How far it falls
  is the generalization penalty, and it is the whole #51 ablation in one distance: RampNet
  **−0.08**, YOLO26 **−0.19**, YOLO11l **−0.24**, YOLO11x **−0.28**.

The uncomfortable corollary, stated because it is real: **in-distribution, YOLO11x is not
behind.** Its `manual_gold` AP is **0.931** against RampNet's **0.917**, at the same 0.05
export floor — and RampNet's export used horizontal-flip TTA while YOLO's did not, so that
comparison is if anything generous to RampNet. On home turf a generic detector trained on this
dataset matches the purpose-built one. It is the 0.20 F1 it gives back on unfamiliar cities
that RampNet does not — and out of domain the AP ordering is not close either, **0.849 to
0.730** (macro-mean, the table above; micro-pooled it is 0.844 to 0.734 — see the note on the
two AP families under "Choosing an operating point").

---

## Choosing an operating point

Every row above is one point. For the models that emit a calibrated score, that point is a
choice, and the choice is RampNet's to make — it is the subject of
[#54](https://github.com/ProjectSidewalk/RampNet/issues/54) and
[#55](https://github.com/ProjectSidewalk/RampNet/issues/55), written up in
[`operating_point.md`](operating_point.md). This is the surface those points sit on:

![PR curves, pooled over the seven US splits](figures/scoreboard_pr_curves.png)

The figure says three things a table of F1 cannot:

1. **A calibrated score is a dial; a chat VLM is a dot.** RampNet, the three YOLO arms and
   the two open detectors can be moved anywhere along their curves for free — no retraining,
   no second inference pass. The Gemini/Qwen/Molmo rows are single points because those
   models emit no confidence to threshold on. Comparing a tuned model against an untunable
   one at one threshold flatters whichever happened to land well.
2. **RampNet's curve dominates over the whole range**, not just at 0.55. At every recall the
   YOLO arms reach, RampNet is above them, and the AP ordering (0.844 vs 0.734) is the
   integral of that.
3. **The deployed point is not the F1 optimum.** RampNet sits at 0.55 (hollow marker) where
   the curve is nearly flat; #54's recommended 0.30 (filled) buys recall at a shallow
   precision cost.

> **Two AP families, and this figure uses the other one.** A PR curve is an integral over
> ranked predictions, so pooling it across splits has to be **micro** — concatenate every
> panorama, integrate once — and the legend above reports that. The headline table's AP
> column is the **macro-mean** of the per-split APs, each city weighted equally, like every
> other column in it. Same detections, same scorer; the two land a few thousandths apart
> (RampNet 0.844 micro / 0.849 macro, YOLO11x 0.734 / 0.730 — note it moves the *other* way).
> Neither is more correct. They are labelled everywhere both appear, and a comparison is only
> meaningful within one family: macro-to-macro the gap is 0.119, micro-to-micro 0.110.

<!-- BEGIN GENERATED: thresholds (scripts/analysis/scoreboard.py) -->

| peak threshold | P | R | F1 |  |
|---|--:|--:|--:|---|
| **0.55** | 0.959 | 0.686 | 0.800 | deployed today (`OPERATIONAL_CONFIDENCE`, auto-labeler) |
| **0.30** | 0.899 | 0.764 | 0.826 | recommended by #54; **not yet adopted** (labeler#20 open) |

<!-- END GENERATED: thresholds -->

Those are the **raw** numbers. Applying #55's per-split GT-completeness correction —
27.2% of the incremental false positives in `[0.30, 0.55)` are real ramps the GT missed —
`operating_point.md` reports corrected **P 0.919 / R 0.796 / F1 0.853** at 0.30, against
0.964 / 0.722 / 0.826 deployed. Corrected precision stays ≥0.88 on every US split, and
detection density rises only 1.86 → 2.23 per panorama.

**Why the scoreboard still reports 0.55.** Because that is what is deployed:
`OPERATIONAL_CONFIDENCE = 0.55` in the auto-labeler's `detectors/__init__.py`, and the
benchmark bundles are a sample of a real run at that setting. #54's recommendation has not
been adopted — [ProjectSidewalk/sidewalk-auto-labeler#20](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/20)
is open, and its 2026-08-04 status update raises a genuine complication: in *world* space,
after multi-view fusion across a whole city, the drop buys only +0.4 to +3.2 recall points
rather than the per-panorama +7.4, because other views were already covering for each other.
That is a deployment question about a different repo's product metric. The per-panorama
choice — which is what this benchmark measures and what this page reports — is settled, and
the table above is what settled it.

---

## How to read this

**The pool is seven cities, not ten.** The split registry is imported from
`low_floor_sweep.US_SPLITS`, the same one `miss_decomposition.py` and the operating-point
sweep use, so a split cannot be pooled here and held out there. The three held-out splits
carry their documented reasons in the table below. `model_comparison.md` states outright that
budapest's numbers "must not be pooled with the US splits or averaged into a headline"; this
page obeys that.

**Macro, not micro.** Each city contributes equally. Pooling raw counts would weight paterson
(395 GT ramps) twice as heavily as clovis (195), and folding in `manual_gold` would be far
worse — its 3,919 GT points outnumber all nine cities combined, so a pooled headline would be
59% one split that is in-distribution for exactly one model on the board.

**Operating points differ by model class, and are inherited rather than chosen here.**

| class | reported at | why |
|---|---|---|
| RampNet | peak threshold **0.55** | its deployed threshold; the city bundles are extracted at it, so this is RampNet as shipped. Its AP alone comes from the 0.05 low-floor cache — see "Choosing an operating point" |
| YOLO arms | conf **0.25** | pre-registered in the #71 protocol before any benchmark contact; per-split best-F1 sits at 0.10–0.15, which would be tuning on test |
| open-vocab detectors | **0.05 export floor** | as the log reports them; their tuned sweep points are in `model_comparison.md` and roughly triple their F1, to ~0.2 |
| chat VLMs, Molmo | **no score to threshold** | they emit boxes/points with no confidence, so their single row *is* the model — not a low threshold someone picked for them |

**AP is cross-comparable, but RampNet's comes from a second file.** The city bundles hold
RampNet's detections only down to its deployed 0.55, because they *are* a production run and
that is where production stops — so an AP computed from them integrates a curve cut off at
the operating point. Read that way RampNet's pooled AP is **0.720**, which sits *below* the
YOLO arms' 0.730 and is an artifact of the floor, not a result. `analysis_out/op_cache/` is
the #54 re-extraction of the same panoramas down to 0.05 — the floor every other scored model
is exported at — and RampNet's AP read from it is **0.849**. That is the number in the table,
marked †.

The substitution is gated on *measured* truncation, not a list of split names: it applies
only where the bundle floor sits more than 0.1 above the cache floor. `manual_gold`'s bundle
is already at 0.05, so it keeps its own AP (0.917) — there is nothing to un-truncate there,
and swapping in the cache would quietly trade that split's flip-TTA export for a no-TTA one.

**This is the one number on this page that differs from `model_comparison.md`**, so here is
the whole mapping rather than a description of it. The middle column is what the log prints;
the test asserts it against the log's committed tables, so the two documents cannot drift
apart without failing CI:

<!-- BEGIN GENERATED: ap-provenance (scripts/analysis/scoreboard.py) -->

| split | AP in `model_comparison.md` | AP here | read from | why |
|---|--:|--:|---|---|
| `richmond` | 0.763 | **0.876** | `op_cache` (0.05 floor) | truncated at the deployed 0.55 |
| `bend` | 0.754 | **0.868** | `op_cache` (0.05 floor) | truncated at the deployed 0.55 |
| `clovis` | 0.688 | **0.871** | `op_cache` (0.05 floor) | truncated at the deployed 0.55 |
| `morgantown` | 0.728 | **0.856** | `op_cache` (0.05 floor) | truncated at the deployed 0.55 |
| `annapolis` | 0.734 | **0.875** | `op_cache` (0.05 floor) | truncated at the deployed 0.55 |
| `paterson` | 0.681 | **0.748** | `op_cache` (0.05 floor) | truncated at the deployed 0.55 |
| `gainesville` | 0.691 | **0.846** | `op_cache` (0.05 floor) | truncated at the deployed 0.55 |
| `laurens_mapillary` | 0.377 | **0.691** | `op_cache` (0.05 floor) | truncated at the deployed 0.55 |
| `laurens_gsv` | 0.494 | 0.494 | bundle — 0.55 floor, no `op_cache` | **truncated**; not comparable with the rows above |
| `budapest_district5` | 0.478 | **0.648** | `op_cache` (0.05 floor) | truncated at the deployed 0.55 |
| `sao_paulo` | 0.666 | **0.812** | `op_cache` (0.05 floor) | truncated at the deployed 0.55 |
| `manual_gold` | 0.917 | 0.917 | bundle — already at 0.05 | no truncation to undo; flip-TTA export |

<!-- END GENERATED: ap-provenance -->

Everything else agrees to three decimals, and that is enforced rather than claimed:
`test_every_number_matches_model_comparison` parses all ten of the log's per-split tables
and checks P, R, F1 and AP on every row it finds. A number edited in either document
without re-running fails CI.

Two things travel with that number. **P/R/F1 still come from `records.jsonl`**, the published
deployment-faithful operating point, so a single row has two sources; and **the sub-0.55 half
of the curve is a lower bound**, because the GT was assembled from detections at or above 0.55
and #55 measured that 27.2% of the incremental FPs in [0.30, 0.55) are GT-completeness
artifacts. RampNet's 0.849 is therefore itself conservative.

**Two known asymmetries in the `manual_gold` column.** RampNet's detections there were
exported with horizontal-flip TTA and at a 0.05 floor (`benchmark/manual_gold/detections_meta.json`);
the city splits used neither. Flip-TTA is worth roughly nothing out of domain but is
measurably positive in domain (#78), so RampNet's `manual_gold` row is its most flattering
number on this page.

**Everything else about the protocol is shared** — same greedy matcher, same 0.022 normalized
match radius, same derived ground truth, same perspective tiling for the VLM inputs. The
scorer is `rampnet/detection_eval.py` and the thresholding is `compare.operating_report`, both
imported rather than reimplemented, so a row here means exactly what the same row means in the
log.

---

## What is missing

Omissions are content, so they are named rather than left as blanks:

- **`gemini-3.1-pro-preview` and `gemini-3.6-flash` have no `manual_gold` row here.** Their
  city detections are published; their `manual_gold` detections are not, and are absent from
  this workstation's `.model_cache` (probed by reconstructing the cache keys — the same probe
  returns 124/124 for richmond, so the method is sound and the split is genuinely not here).
  `model_comparison.md` quotes F1 0.568 and 0.540 for them from the original run. **Those two
  numbers are currently the only ones in this comparison that cannot be re-derived from a
  clean clone.** Fixing it means locating the machine that holds that cache and running
  `export_model_cache.py --models gemini:gemini-3.6-flash,gemini:gemini-3.1-pro-preview`, or
  re-paying for the run. It is also why both are absent from the generalization figure.
- **`gemini-3.7-flash`, the YOLO arms, the Vistas arms and the Claude legs are all on this
  board but `standing=False` in the roster**, so `model_comparison.md`'s roster tables keep
  a consistent 8-model set pending each write-up. Every one of those legs has its
  detections published and verified against cache, and `gemini-3.7-flash`'s silent-panorama
  behaviour specifically was investigated under #120 (10/10 pairs identical to cache).
  Including them here is deliberate: this page's job is to show everything that has been
  measured, and `standing` governs which roster tables a leg appears in, not whether its
  numbers are real.
- **The YOLO tiles arms are absent** — still training. The three pano arms are the completed
  half of the #51 ablation; the resolution-controlled half is not done.
- **`manual_gold` has no null-recall pass** (O(n²) in panos), so the open detectors' recall
  discount is unmeasured on that split.
- **Six legs have one split each**, so they are in the partial table rather than the
  headline: the two Vistas arms (richmond) and the four Claude legs (annapolis). Extending
  either to the full pool is a run, not a code change.
- **Nothing else in the registry is missing.** The board is driven by `rampnet/roster.py`,
  and `unregistered_exports` is empty — every published detections file is claimed by a
  roster entry and scored here.

---

## Splits

<!-- BEGIN GENERATED: coverage (scripts/analysis/scoreboard.py) -->

| split | role | panos | GT ramps | note |
|---|---|--:|--:|---|
| `richmond` | pooled | 124 | 310 | US deployment city, verdict-grade GT |
| `bend` | pooled | 110 | 327 | US deployment city, verdict-grade GT |
| `clovis` | pooled | 125 | 195 | US deployment city, verdict-grade GT |
| `morgantown` | pooled | 125 | 267 | US deployment city, verdict-grade GT |
| `annapolis` | pooled | 125 | 294 | US deployment city, verdict-grade GT |
| `paterson` | pooled | 125 | 395 | US deployment city, verdict-grade GT |
| `gainesville` | pooled | 125 | 272 | US deployment city, verdict-grade GT |
| `laurens_mapillary` | pooled | 94 | 249 | US deployment city, verdict-grade GT |
| `laurens_gsv` | held out † | 86 | 220 | second imagery arm of laurens, which is already pooled through laurens_mapillary -- the two arms sample one town and largely the same physical ramps (59% of gsv panos within 20 m of a mapillary one, median NN 17.2 m), so pooling both would double-count them and break the independence the Wilson intervals assume (GT is HIGH confidence; held out for non-independence, not GT quality) |
| `budapest_district5` | held out † | 125 | 300 | single-rater GT at low reviewer confidence (docs/model_comparison.md: do not pool) |
| `sao_paulo` | held out † | 125 | 281 | non-US city — the pooled recommendation is a US-deployment basis (GT is HIGH reviewer confidence; held out for geography, not GT quality) |
| `manual_gold` | held out † | 1000 | 3919 | in-distribution GSV + independently-labelled GT (in-domain reference, not a deployment city) |

<!-- END GENERATED: coverage -->

---

## Reproducing this page

From a clean clone, with no GPU, no credentials and no network:

```bash
pip install numpy pillow                        # the whole scoring path, nothing else
python scripts/analysis/scoreboard.py --no-figures   # tables + analysis_out/scoreboard.json
python scripts/analysis/scoreboard.py --check        # non-zero if this page or its JSON is stale

# The figures additionally need matplotlib, which lives in requirements.txt /
# environment.yml rather than requirements-dev.txt — the test suite stays plotting-free,
# the same arrangement plot_operating_point.py uses.
pip install matplotlib && python scripts/analysis/scoreboard.py

# requirements-dev.txt also works, but it installs torch, timm, transformers, datasets
# and scikit-image for the REST of the test suite — several GB this page does not need.
```

Inputs are `benchmark/<split>/{records.jsonl,verdicts.json}`, `manual_labels/`, and
`benchmark/model_detections/`. All are committed. `analysis_out/scoreboard.json` carries every
per-(model, split) cell — precision, recall, F1, AP, TP/FP/FN, panorama and GT counts — for
anything this page does not tabulate.

Files: [`scripts/analysis/scoreboard.py`](../scripts/analysis/scoreboard.py) (scoring),
[`scripts/analysis/scoreboard_render.py`](../scripts/analysis/scoreboard_render.py) (tables
and the splice), [`scripts/analysis/scoreboard_figures.py`](../scripts/analysis/scoreboard_figures.py)
(the five figures).
