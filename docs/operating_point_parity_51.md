# The RampNet-vs-YOLO gap at matched operating points (#51)

**Status: measured 2026-09-01, CPU only, from already-committed detections. The published
gap is mostly an operating-point artifact. At parity it is 0.039 F1, not 0.160 — and on
3 of the 10 splits the best YOLO leg is level with RampNet or ahead of it, though every
one of those three margins (0.001–0.017) is inside this benchmark's own noise rule.**

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
`test_no_candidate_dev_split_is_ever_reported_over` asserts it rather than trusting
review — on the output of `build()` for *every* candidate dev split, not on the
definition of the candidate list, which would be a tautology.

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

### 2. RampNet's lead is not uniform — but the splits it does not win are all inside the noise

The tiles arm is **ahead** on `morgantown` (0.862 vs 0.845, +0.017) and on `manual_gold`
(0.911 vs 0.902, +0.009), and level on `bend` (0.870 vs 0.871, −0.001). That is a
materially different picture from "RampNet wins everywhere".

**Read all three as ties, not as wins.** Every margin is below the ~0.02 F1 this document's
own noise rule says must not be read (see *Gaps*), and one of them has a second, measured
reason to be discounted:

- **`manual_gold`'s margin is the size of a known instrument difference on that split.**
  RampNet's committed `manual_gold` detections were exported *with* horizontal-flip TTA
  (`benchmark/manual_gold/detections_meta.json`); the `op_cache` this document sweeps is
  the no-TTA deployment path, and at 0.55 the two differ by 0.009 F1 — the same size as
  the 0.009 margin. With TTA at 0.30 RampNet would plausibly be level.
- **A paired test would settle it and needs no GPU** — both models' per-pano detections
  exist. It cannot be run from a clean clone, because the tiles arm's detections are not
  published (see *Gaps*).

So the defensible claim is the weaker one: at a defensible operating point the supervised
baseline is **no longer distinguishable from RampNet** on three of ten splits, including
`manual_gold` — the only split whose ground truth was labelled independently of RampNet's
own outputs, and so the one place the comparison is furthest from circular. That is
already a first for this benchmark. Calling it an outright win needs the seed work below.

### 3. What does NOT change: AP

AP is threshold-free, so parity cannot move it, and it still favours RampNet:

| | AP (macro, US7) | source |
|---|--:|---|
| RampNet | **0.849** | [`model_scoreboard.md`](model_scoreboard.md) AP column, `op_cache`-derived |
| `y11x_tiles` | 0.773 | [`yolo_geometry_51.json`](data/yolo_geometry_51.json) `pooled.*.macro_ap` |
| `y11x_pano` | 0.752 | as above |
| `y11x_pano_h200` | 0.730 | as above |

RampNet's 0.849 is the **only number on this page that this document's own script does not
compute** — it is carried from the scoreboard, whose AP column is `op_cache`-derived like
everything here, so it is on the same footing as the three YOLO rows.

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
- **The tiles arm's detections are not published**, so the paired test that would settle
  the three near-tied splits in §2 cannot be run from a clean clone. `y11x_tiles` and
  `y11x_pano` are not in `rampnet/roster.py` and have no files under
  `benchmark/model_detections/`; their checkpoints live only on cluster storage
  (`/gscratch/makelab/jonf/rampnet_yolo_baseline_51/<arm>/weights/best.pt`, sha256s in
  [`env.txt`](data/yolo_geometry_51/env.txt)). Registering them as roster legs and
  exporting their detections is what unblocks it.
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
bundle-derived 0.827.

That −0.0025 is **not** peak extraction. An earlier version of this section said it was —
that lowering `peak_local_max`'s floor to 0.05 changes which maxima survive `min_distance`
suppression — and that is wrong: `peak_local_max` suppresses on a maximum filter, so a
pixel that is the maximum of its neighbourhood at 0.55 is still the maximum at 0.05.
Lowering the floor can only *add* peaks; the ≥ 0.55 subset of a 0.05-floor extraction is
exactly a 0.55-floor extraction.

The two sources differ because they are **two different heatmaps**, and
[`operating_point.md`](operating_point.md) already measured which and why:

| split | op_cache @0.55 | published bundle | Δ | why |
|---|--:|--:|--:|---|
| richmond | 0.8546 | 0.855 | −0.0004 | same computation — Mapillary, bit-exact |
| clovis | 0.8012 | 0.801 | +0.0002 | same computation |
| morgantown | 0.8351 | 0.835 | +0.0001 | same computation |
| annapolis | 0.8395 | 0.839 | +0.0005 | same computation |
| *budapest_district5* | 0.6442 | 0.644 | +0.0002 | same computation |
| bend | 0.8532 | 0.850 | **+0.0032** | GSV: production used a 4096×2048 resample |
| paterson | 0.8006 | 0.805 | **−0.0044** | GSV resample |
| gainesville | 0.7871 | 0.803 | **−0.0159** | GSV resample |
| *sao_paulo* (dev) | 0.7578 | 0.777 | **−0.0192** | GSV resample |
| *manual_gold* | 0.8990 | 0.908 | **−0.0090** | committed detections carry flip-TTA; `op_cache` does not |

Three things follow, and they qualify every number above:

1. **The pooled macro clears its 0.005 tolerance partly by cancellation** (+0.003 on bend
   against −0.004 and −0.016 on paterson and gainesville). A macro-only control could not
   see a split-level divergence, so the script now asserts the control **per split** too:
   the five same-computation splits to 0.002, the five with a documented reason to differ
   to 0.025. The five exact splits are the real regression guard.
2. **The dev split has the largest discrepancy of the ten** (−0.019). RampNet's threshold
   is selected on the split where its `op_cache` curve is least like its shipped
   detections. That is what the sensitivity table answers: the selection lands on
   0.30–0.35 whichever of the three candidates is used.
3. **Three of the seven pooled splits are GSV, so RampNet's parity F1 is about 0.002 low**
   relative to a bundle-derived path. The reported 0.039 gap is very slightly conservative,
   not flattering.

## Cost

CPU only, seconds, $0 — this document needed no GPU. The YOLO sweeps it reads came from
the 2026-08-30 geometry run: 67 min 39 s on one A40 on makelab2, ≈ 1.13 GPU-hours at no
cost (lab hardware). See [`yolo_geometry_51.md`](yolo_geometry_51.md) for that run's
provenance. A ledger row follows once [#147](https://github.com/ProjectSidewalk/RampNet/pull/147)
lands; until then this paragraph is the record.
