# The RampNet-vs-YOLO gap at matched operating points (#51)

**Status: measured 2026-09-01, CPU only, from already-committed detections. The published
gap is mostly an operating-point artifact. At parity it is 0.039 F1, not 0.160 — and the
best YOLO leg matches or beats RampNet on 3 of the 10 splits.**

## What was wrong with the comparison

[`model_scoreboard.md`](model_scoreboard.md) and [`yolo_geometry_51.md`](yolo_geometry_51.md)
compare F1 across rows whose operating points were chosen by *different procedures*:

| | operating point | how it was chosen |
|---|---|---|
| RampNet | 0.55 | its shipped deployment threshold |
| every YOLO leg | 0.25 | the Ultralytics `predict()` default |

Nobody selected 0.25 for YOLO. It is what you get when you do not pass `conf`. So the
published gap mixes *which model is better* with *whose default happened to suit F1 on
this benchmark*. The scoreboard already tells the reader to check the op column before
comparing rows; this measures what that warning was worth. It turns out to be worth a
lot.

## What parity means here

One threshold per model, chosen the same way for every model, on data the headline is
never reported over:

1. **Select** each model's single uniform threshold on a **dev split**, by F1.
2. **Report** every model at that threshold, macro-meaned over the seven pooled US splits.

The dev split is `sao_paulo` — already outside the pooled seven, and unremarkable, unlike
`budapest_district5` (the benchmark's only ranking inversion, low reviewer confidence) or
`manual_gold` (the only independently-labelled split). The threshold is **uniform across
splits**; a per-split best would be tune-on-test and the script does not offer it.

Selection never touches a reported split, and
`test_no_candidate_dev_split_is_ever_reported_over` asserts it rather than trusting review.

## Results

Macro-mean over the seven pooled US splits:

| model | sel thr | P | R | F1 | ΔF1 vs RampNet | at published op | gain from parity |
|---|--:|--:|--:|--:|--:|--:|--:|
| **RampNet** | 0.30 | 0.896 | 0.797 | **0.843** | — | 0.824 | +0.018 |
| `y11x_tiles` (ep44) | 0.10 | 0.875 | 0.749 | **0.804** | **−0.039** | 0.667 | **+0.137** |
| `y11x_pano` (ep38) | 0.10 | 0.787 | 0.726 | 0.752 | −0.091 | 0.623 | +0.129 |
| `y11x_pano_h200` (ep60) | 0.10 | 0.839 | 0.683 | 0.751 | −0.092 | 0.575 | +0.176 |

**The asymmetry in the last column is the whole finding.** Parity moves RampNet by 0.018
because 0.55 was nearly right for it. It moves the tiles arm by 0.137 because 0.25 was
nowhere near right. The published comparison was scoring one model at its operating point
and the other 0.137 F1 away from its own.

Per split, F1 at each model's selected threshold:

| split | RampNet | `y11x_tiles` | `y11x_pano` | `y11x_pano_h200` |
|---|--:|--:|--:|--:|
| richmond | **0.864** | 0.803 | 0.791 | 0.777 |
| bend | **0.871** | 0.870 | 0.789 | 0.799 |
| clovis | **0.836** | 0.770 | 0.704 | 0.734 |
| morgantown | 0.845 | **0.862** | 0.791 | 0.813 |
| annapolis | **0.853** | 0.783 | 0.723 | 0.689 |
| paterson | **0.818** | 0.761 | 0.794 | 0.776 |
| gainesville | **0.812** | 0.778 | 0.673 | 0.668 |
| *budapest_district5* | **0.674** | 0.551 | 0.528 | 0.510 |
| *sao_paulo* (dev) | **0.800** | 0.776 | 0.744 | 0.754 |
| *manual_gold* | 0.902 | **0.911** | 0.844 | 0.872 |

### The dev split does not carry the result

| dev split | RampNet | `y11x_tiles` | gap |
|---|--:|--:|--:|
| `budapest_district5` | 0.842 @0.35 | 0.810 @0.05 | 0.032 |
| `sao_paulo` | 0.843 @0.30 | 0.804 @0.10 | 0.039 |
| `manual_gold` | 0.842 @0.35 | 0.804 @0.10 | 0.038 |

The gap is 0.032–0.039 whichever of the three is used.

## What it means

### 1. The headline shrinks by a factor of four, and it is not architecture that moved

Published residual **0.160**; at parity **0.039**. Nothing about either model changed —
only the procedure for picking where to read them. #51's claim that the keypoint
architecture is what RampNet contributes still points the right way, but 0.039 is a very
different claim from 0.252 or 0.160, and it is close enough to the noise floor to need
the seed work below before it carries weight.

### 2. RampNet's lead is not uniform, and it loses the split that matters most

The tiles arm **beats** RampNet on `morgantown` (0.862 vs 0.845) and on **`manual_gold`
(0.911 vs 0.902)**, and ties on `bend` (0.870 vs 0.871). `manual_gold` is the only split
whose ground truth was labelled independently of RampNet's own outputs — the one place
the comparison is furthest from circular — and the supervised YOLO baseline wins it.

That is a materially different picture from "RampNet wins everywhere", and it is the
first result in this benchmark where a non-RampNet model takes a split outright at a
defensible operating point.

### 3. What does NOT change: AP

AP is threshold-free, so parity cannot move it, and it still favours RampNet:

| | AP (macro, US7) |
|---|--:|
| RampNet | **0.849** |
| `y11x_tiles` | 0.773 |
| `y11x_pano` | 0.752 |
| `y11x_pano_h200` | 0.730 |

**Both sides are floored at 0.05** (`op_cache` `meta.score_floor` and
`YoloDetector.score_threshold`), so this comparison was already like-for-like — an
earlier reading of this analysis claimed the AP columns were asymmetric and that was
wrong. RampNet's PR curve genuinely dominates by **0.076 AP**. The honest summary is that
RampNet has the better curve, while at each model's own best point on that curve the F1s
are much closer than published.

## Gaps, stated

- **The tiles arm's parity F1 is a lower bound.** With `budapest_district5` as dev its
  selected threshold is 0.05, the cache floor — the edge of what was ever measured. Its
  true optimum may be lower and is unmeasured. Both models would have to be re-dumped at
  a lower floor to close this, and doing it for only one of them would reintroduce
  exactly the asymmetry this document removes.
- **One seed per cell.** This is now the binding limitation, not a footnote. At a 0.160
  gap seed variance was irrelevant; at 0.039 it is the entire question, and #51's own
  rule is that differences under ~0.02 should not be read. Three seeds of `y11x_tiles`
  is the experiment that would settle this.
- **`y11x_tiles` is at ep44 of a 60-epoch schedule**, and the pano lineage suggests more
  epochs would *hurt* it out of distribution ([`yolo_geometry_51.md`](yolo_geometry_51.md)),
  so this is not simply "the undertrained arm".
- **Only the `x` architecture.** `y11l_*` and `y26_*` legs are not swept here; their
  committed reports do not carry a sweep.
- **The VLM and pointing challengers cannot be included at all** — they emit no calibrated
  confidence, so they have one operating point rather than a curve. Their rows in
  `model_comparison.md` are unaffected by this document, in either direction.

## Reproducing

CPU only, no GPU, no network — everything it reads is committed:

```bash
python scripts/analysis/operating_point_parity_51.py --sensitivity
python scripts/analysis/operating_point_parity_51.py --check   # fails if the artifact drifted
pytest tests/test_operating_point_parity_51.py -q
```

Artifact: [`docs/data/operating_point_parity_51.json`](data/operating_point_parity_51.json).
Inputs: `analysis_out/op_cache/*.json` (RampNet, re-scored at each threshold) and
`docs/data/yolo_geometry_51/*.txt` (the YOLO legs' committed sweeps).

**The control.** RampNet at 0.55 re-scores to 0.824 from `op_cache` against a published,
bundle-derived 0.827. That −0.0025 is peak extraction, not a bug: `peak_local_max` at a
0.05 floor finds a different peak set than at a 0.55 floor, because newly admitted low
peaks change which maxima survive `min_distance` suppression — the same split-brain
`scoreboard.uses_low_floor_cache` exists to manage. The script asserts the two stay within
0.005 of each other and fails loudly if that ever stops being true.
