# Scoreboard: every model, one table

Eighteen model legs, ten splits, one page. This is the summary view of the curb-ramp benchmark —
**rows are models, columns are metrics** — for the question "which model is best, and by how
much".

It is the companion to [`model_comparison.md`](model_comparison.md), not a replacement.
That document is the comprehensive log: per-split tables in the order the splits were run,
the mechanism behind every number, the negative results, the caveats, the harness
self-validation. It is where you go to find out *why* Qwen-32B inverts on budapest. It is a
bad place to find out *who wins*, because that answer is spread across ten tables in
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

| model | class | op | P | R | F1 | ΔF1 vs RampNet | AP | FP/pano | F1 range |
|---|---|--:|--:|--:|--:|--:|--:|--:|:-:|
| **RampNet** | purpose-trained | 0.55 | 0.958 | 0.728 | **0.827** | — | 0.720&nbsp;† | 0.1 | 0.80–0.85 |
| YOLO11l (pano) | supervised baseline | 0.25 | 0.939 | 0.449 | 0.604 | -0.223 | 0.722 | 0.1 | 0.48–0.71 |
| YOLO11x (pano) | supervised baseline | 0.25 | **0.969** | 0.416 | 0.575 | -0.252 | 0.730 | 0.0 | 0.40–0.71 |
| YOLO26 (pano) | supervised baseline | 0.25 | 0.736 | 0.446 | 0.550 | -0.277 | 0.602 | 0.4 | 0.45–0.68 |
| Gemini 3.1 Pro | chat VLM | no score | 0.653 | 0.570 | 0.606 | -0.221 | – | 0.7 | 0.50–0.68 |
| Gemini 3.7 Flash | chat VLM | no score | 0.697 | 0.496 | 0.575 | -0.252 | – | 0.5 | 0.46–0.66 |
| Gemini 3.6 Flash | chat VLM | no score | 0.587 | 0.548 | 0.562 | -0.265 | – | 0.9 | 0.44–0.63 |
| Qwen3-VL-32B | chat VLM | no score | 0.663 | 0.246 | 0.355 | -0.472 | – | 0.3 | 0.17–0.43 |
| Qwen3-VL-8B | chat VLM | no score | 0.324 | 0.358 | 0.337 | -0.490 | – | 1.8 | 0.25–0.41 |
| Molmo2-8B | pointing model | no score | 0.431 | 0.439 | 0.429 | -0.398 | – | 1.4 | 0.33–0.51 |
| OWLv2-large | open-vocab detector | 0.05 floor | 0.034 | **0.942** | 0.065 | -0.762 | 0.097 | 64.8 | 0.05–0.08 |
| Grounding DINO | open-vocab detector | 0.05 floor | 0.028 | 0.856 | 0.055 | -0.772 | 0.037 | 71.6 | 0.03–0.07 |

<!-- END GENERATED: headline -->

† RampNet's AP is **truncated and not comparable** to the others — see the AP caveat below.

![Pooled F1 by model](figures/scoreboard_f1.png)

**RampNet wins by 0.221 F1**, and the gap is not a threshold artifact: it holds at every
operating point anyone has committed to, and on ground truth that never saw a RampNet review
(`manual_gold`). The three findings that only become visible once the splits are pooled:

1. **The supervised baseline and the best zero-shot VLM are a dead heat.** YOLO11l trained on
   the RampNet dataset scores **0.604**; Gemini 3.1 Pro, zero-shot with an untuned prompt,
   scores **0.606**. Issue #51 asks whether RampNet's advantage is the data or the
   architecture; pooled, the answer is that the *data alone*, handed to a generic detector,
   buys you a tie with an off-the-shelf chat model — and the remaining 0.22 F1 is what the
   keypoint architecture adds.
2. **RampNet is the only strong model that is also stable.** Its F1 spans 0.80–0.85 across the
   seven cities, a range of 0.053. Every challenger scoring above 0.1 swings between 0.153
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
same F1 contour fail in opposite directions: Qwen-32B and Qwen-8B score 0.355 and 0.337 —
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
  precision** and no training at all — but RampNet's 0.855 on that split is 0.338 clear of
  it. The union arm (`+curb`) is a committed negative result: adding Vistas' `Curb` class
  *loses* recall while precision collapses, because `Curb` fuses adjacent ramps into one
  component (#126).

---

## Every model, every split

<!-- BEGIN GENERATED: by-split (scripts/analysis/scoreboard.py) -->

| model | rich | bend | clovis | morg | annap | pater | gaines | **pooled** | budapest † | sao_paulo † | manual_gold † |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **RampNet** | **0.855** | **0.850** | **0.801** | **0.835** | **0.839** | **0.805** | **0.803** | **0.827** | **0.644** | **0.777** | **0.908** |
| YOLO11l (pano) | 0.595 | 0.713 | 0.600 | 0.675 | 0.481 | 0.647 | 0.516 | 0.604 | 0.247 | 0.662 | 0.839 |
| YOLO11x (pano) | 0.547 | 0.710 | 0.551 | 0.686 | 0.397 | 0.635 | 0.499 | 0.575 | 0.221 | 0.659 | 0.851 |
| YOLO26 (pano) | 0.491 | 0.637 | 0.552 | 0.681 | 0.450 | 0.591 | 0.451 | 0.550 | 0.277 | 0.605 | 0.739 |
| Gemini 3.1 Pro | 0.664 | 0.638 | 0.503 | 0.639 | 0.567 | 0.681 | 0.548 | 0.606 | 0.381 | 0.454 | – |
| Gemini 3.7 Flash | 0.664 | 0.639 | 0.499 | 0.595 | 0.565 | 0.609 | 0.456 | 0.575 | 0.338 | 0.358 | 0.527 |
| Gemini 3.6 Flash | 0.634 | 0.597 | 0.478 | 0.629 | 0.554 | 0.608 | 0.438 | 0.562 | 0.336 | 0.346 | – |
| Qwen3-VL-32B | 0.427 | 0.415 | 0.311 | 0.421 | 0.398 | 0.347 | 0.168 | 0.355 | 0.079 | 0.218 | 0.285 |
| Qwen3-VL-8B | 0.377 | 0.357 | 0.252 | 0.337 | 0.327 | 0.405 | 0.302 | 0.337 | 0.169 | 0.219 | 0.386 |
| Molmo2-8B | 0.457 | 0.449 | 0.376 | 0.460 | 0.424 | 0.511 | 0.329 | 0.429 | 0.274 | 0.326 | 0.422 |
| OWLv2-large | 0.064 | 0.070 | 0.049 | 0.071 | 0.063 | 0.077 | 0.060 | 0.065 | 0.062 | 0.052 | 0.088 |
| Grounding DINO | 0.053 | 0.073 | 0.035 | 0.042 | 0.055 | 0.068 | 0.055 | 0.055 | 0.042 | 0.049 | 0.082 |
| Mask2Former Vistas (curb cut) | 0.517 | – | – | – | – | – | – | – | – | – | – |
| Mask2Former Vistas (+curb) | 0.210 | – | – | – | – | – | – | – | – | – | – |
| Claude Opus 5 (low) | – | – | – | – | 0.588 | – | – | – | – | – | – |
| Claude Opus 5 (high) | – | – | – | – | 0.520 | – | – | – | – | – | – |
| Claude Sonnet 5 (low) | – | – | – | – | 0.463 | – | – | – | – | – | – |
| Claude Sonnet 5 (high) | – | – | – | – | 0.456 | – | – | – | – | – | – |

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
  — Molmo +0.01, Qwen-32B +0.07, Gemini 3.7 Flash +0.05. That they sit slightly *above* the
  line is the un-anchored-GT check from #58 coming out clean: they do not gain on a split
  whose ground truth RampNet never touched.
- **A model trained on the RampNet dataset starts above the line and falls.** How far it falls
  is the generalization penalty, and it is the whole #51 ablation in one distance: RampNet
  **−0.08**, YOLO26 **−0.19**, YOLO11l **−0.24**, YOLO11x **−0.28**.

The uncomfortable corollary, stated because it is real: **in-distribution, YOLO11x is not
behind.** Its `manual_gold` AP is **0.931** against RampNet's **0.917**, at the same 0.05
export floor — and RampNet's export used horizontal-flip TTA while YOLO's did not, so that
comparison is if anything generous to RampNet. On home turf a generic detector trained on this
dataset matches the purpose-built one. It is the 0.20 F1 it gives back on unfamiliar cities
that RampNet does not.

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
| RampNet | peak threshold **0.55** | its deployed threshold; the city bundles are extracted at it, so this is RampNet as shipped ([`operating_point.md`](operating_point.md) recommends moving to 0.30, which is a separate change) |
| YOLO arms | conf **0.25** | pre-registered in the #71 protocol before any benchmark contact; per-split best-F1 sits at 0.10–0.15, which would be tuning on test |
| open-vocab detectors | **0.05 export floor** | as the log reports them; their tuned sweep points are in `model_comparison.md` and roughly triple their F1, to ~0.2 |
| chat VLMs, Molmo | **no score to threshold** | they emit boxes/points with no confidence, so their single row *is* the model — not a low threshold someone picked for them |

**AP is the one column that is not cross-comparable, and RampNet's is the reason.** AP
integrates the whole confidence sweep, but the city bundles only contain RampNet detections
above 0.55 — so its curve is cut off at the operating point and its pooled AP (0.720) is a
**lower bound**, not a score. Do not read YOLO11x's 0.730 as beating it. The AP column is
meaningful *between* the models exported at the 0.05 floor (the YOLO arms and the two open
detectors), and on `manual_gold`, where RampNet is also exported at 0.05 and its 0.917 is a
real number.

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
  a consistent 8-model set pending each write-up. Its detections are
  published and verified (10/10 pairs identical to cache) and its silent-panorama behaviour
  was investigated under #120. Including it here is deliberate: this page's job is to show
  everything that has been measured.
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
| `budapest_district5` | held out † | 125 | 300 | single-rater GT at low reviewer confidence (docs/model_comparison.md: do not pool) |
| `sao_paulo` | held out † | 125 | 281 | non-US city — the pooled recommendation is a US-deployment basis (GT is HIGH reviewer confidence; held out for geography, not GT quality) |
| `manual_gold` | held out † | 1000 | 3919 | in-distribution GSV + independently-labelled GT (in-domain reference, not a deployment city) |

<!-- END GENERATED: coverage -->

---

## Reproducing this page

From a clean clone, with no GPU, no credentials and no network:

```bash
pip install -r requirements-dev.txt             # the scoring path: numpy + pillow
python scripts/analysis/scoreboard.py --no-figures   # tables + analysis_out/scoreboard.json
python scripts/analysis/scoreboard.py --check        # non-zero if this page is stale

# The figures additionally need matplotlib, which lives in requirements.txt /
# environment.yml rather than requirements-dev.txt — the test suite stays plotting-free,
# the same arrangement plot_operating_point.py uses.
pip install matplotlib && python scripts/analysis/scoreboard.py
```

Inputs are `benchmark/<split>/{records.jsonl,verdicts.json}`, `manual_labels/`, and
`benchmark/model_detections/`. All are committed. `analysis_out/scoreboard.json` carries every
per-(model, split) cell — precision, recall, F1, AP, TP/FP/FN, panorama and GT counts — for
anything this page does not tabulate.

Files: [`scripts/analysis/scoreboard.py`](../scripts/analysis/scoreboard.py) (scoring),
[`scripts/analysis/scoreboard_render.py`](../scripts/analysis/scoreboard_render.py) (tables
and the splice), [`scripts/analysis/scoreboard_figures.py`](../scripts/analysis/scoreboard_figures.py)
(the four figures).
