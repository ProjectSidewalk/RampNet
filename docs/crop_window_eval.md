# Crop-window eval against the manual_gold boxes — and what those boxes turn out to be

**Issue:** #114. **Run date:** 2026-08-14. **Script:** `scripts/analysis/crop_window_eval.py`
(tests: `tests/test_crop_window_eval.py`). **Inputs:** `manual_labels/*.txt` (3,919 boxes on
1,000 panos) + `benchmark/manual_gold/records.jsonl` (pano dims + RampNet detections). No
imagery is needed for any number below; the committed summary with a content hash of the
per-box table is `analysis_out/crop_window_eval.json`.

```
python scripts/analysis/crop_window_eval.py                 # numbers + JSON + per-box CSV
python scripts/analysis/crop_window_eval.py --fetch-sample 40 --gallery 80   # + overlay gallery
python scripts/analysis/crop_window_eval.py --bundle benchmark/richmond      # real extent gold
python scripts/analysis/crop_window_eval.py --bundle benchmark/richmond --cam-height 1.7
```

This was set up to score three crop-window sizing rules (below) against "3,919 human-drawn
curb-ramp bounding boxes." The most important result is about the boxes, not the rules.

---

## Finding 1 — `manual_labels/` w/h is NOT object-extent gold

The repo has always consumed these files as center points (`rampnet/detection_eval.py`
drops w/h at parse). Issue #83 proposed the w/h as the eval set for extent work. **Measured,
they cannot serve that role.** Three independent lines of evidence:

**(a) Box sizes are far below any physical ramp's apparent size, at every distance.**
Median box max-side is 12.8 px on a 4096×2048 equirect. Against a flat-ground model
(2.5 m camera height, 1.5 m nominal ramp width — the same model `size_analysis.py` and the
YOLO pseudo-box prep use):

| depression | flat-ground dist | n | box side p50 | geometric prediction | ratio |
|---|---|---:|---:|---:|---:|
| 3–5° | ~36 m | 45 | 8.3 px | 27 px | 0.31 |
| 5–8° | ~22 m | 593 | 8.7 px | 45 px | 0.20 |
| 8–12° | ~14 m | 888 | 10.9 px | 69 px | 0.16 |
| 12–18° | ~9 m | 1,149 | 13.2 px | 105 px | 0.13 |
| 18–25° | ~6 m | 841 | 16.2 px | 154 px | 0.11 |
| 25–40° | ~4 m | 377 | 20.7 px | 249 px | 0.08 |

A 13 px object at 9 m subtends ~19 cm. Box size *does* grow with proximity — there is some
extent signal — but ~7× slower than a real ramp's apparent size.

**(b) The tightness ratio is uniform across deployments** (p50 ≈ 0.134 for the NYC,
Portland-area, and Bend clusters alike), so this is not one deployment's annotator quirk.
But it splits **by pano**: thresholding at ratio ≥ 0.4, 659 panos are all-small, 52 are
all-big, 82 mixed. Only **9.8 % of boxes (384)** are plausibly full-extent.

**(c) Visual confirmation** (fresh GSV fetches of live gold panos, boxes overlaid — the
imagery is view-only, not the bundle's canonical HF bytes):

| near-point marks (the typical case) | full-extent boxes (the ~10 % case) |
|---|---|
| ![near-point gold mark, NYC corner](assets/crop_window_eval_nearpoint_nyc.jpg) | ![full-extent gold boxes, NYC](assets/crop_window_eval_fullextent.jpg) |
| ![near-point gold mark on a detectable-warning pad](assets/crop_window_eval_nearpoint_dwp.jpg) | |

In the typical case the green gold box is a tight mark on the detectable-warning pad / curb
tip; the visibly larger ramp apron around it is unboxed. In the minority case the box
covers the full apron. (Overlay colors: green = gold box; orange = v1-norm window;
red = v1-raw; blue = geo-v1.5.)

**Consequences.**

- The center points stay exactly as trustworthy as they always were — nothing here touches
  the benchmark's point semantics.
- **#83's plan to use these boxes as the SAM2/extent eval set needs a convention audit
  first**; as-is they would score a correct full-apron mask as a huge false inflation on
  ~90 % of items.
- The ecosystem currently has **no usable curb-ramp extent gold at all.** Producing it —
  with an explicit whole-apron box rule, on manual_gold (GSV) and on richmond's benchmark
  panos (Mapillary) — is the prerequisite for validating any sizing rule or extent model.
  The eval harness here re-runs unchanged against a proper box set.

## Finding 2 — what the rules comparison can still say

Scored rules (windows placed per CropRunner's geometry: x wraps the seam, y clamps by
shifting, size capped at pano dims):

- **v1-raw** — `predict_crop_size` exactly as sidewalk-panorama-tools runs it (pixel-linear
  distance fit on native `pano_y`; constants calibrated on 6656-px-high panos).
- **v1-norm** — the resolution-normalized port planned for SidewalkWebpage's CropService
  (compute in 6656-height reference space, scale back; bit-identical at 6656).
- **geo-v1.5** — depression → flat-ground distance → 1.5 m footprint → pad so the object
  would sit at 12.5 % of the window side.

**Containment of the marked point-region is a non-problem for every rule** (≥ 97.5 %
everywhere; v1-raw and geo-v1.5 at 100 %, v1-norm's misses concentrate at 18–36 m where its
windows are smallest: 94.7 % in that stratum, detection-prompted). Given Finding 1 this
means "the crop keeps the marked spot," **not** "the crop contains the whole ramp."

**The v1 resolution defect is confirmed and quantified at this height:** at 2048-px panos
the raw formula's median window is **294 px vs 149 px normalized — 1.97×** oversized
relative to its own calibration geometry (median over detection-prompted boxes). The
inflation direction flips with pano height (manual_gold sits below the 6656 calibration
height; an 8192-height pano sits above it), which is exactly why the CropService port
normalizes.

**Context-ratio numbers against these boxes are not interpretable as object-context ratios**
(Finding 1) and are deliberately not headlined here; they live in the JSON/CSV for
re-analysis once real extent gold exists.

**Detection coverage** (production floor 0.55): 3,420 of 3,919 gold points (87.3 %) are
covered by an operational detection — those are the labels that would exist and get crops.
190 operational detections match no gold point: would-be crops of non-ramps.

## Finding 3 — the 10–15 % "context band" is an ML-consumer number, not a Gallery number

The overlay tiles make a product tension visible. Pad-to-band sizing on a *full-extent*
object model (geo-v1.5) produces enormous windows (~800 px at 4096-width — a ~70° FOV),
because "object at 12.5 % of side" around a ~100 px ramp *implies* that much context.
Meanwhile v1-norm's ~150 px window frames the ramp tightly — closer to what a training-crop
consumer wants than to a human-facing Gallery card, which (per SidewalkWebpage's own canvas
captures) shows a whole 3:2 viewport of context. The 10–15 % band came from a survey of
**ML pipelines** (sidewalk-panorama-tools' consumer-requirements report — a survey, not a
measurement); the human-facing crop's right context level has never been specified anywhere.
The acceptance test for the production CropService should set the band per consumer class,
not inherit the ML number.

---

## Round 2 — REAL extent gold (richmond, COMPLETE: all 310 adjudicated ramps, 2026-08-14)

`scripts/box_gallery.py` (#116) produced the first whole-apron extent gold, now
complete: **299 boxed + 11 can't-determine = all 310 adjudicated Richmond ramps on all
92 benchmark panos** (jonf, one day, box rule v1→v2 — v2 added two clarifying bullets
mid-annotation, no convention change). Committed as `benchmark/richmond/boxes.json`;
scored with `--bundle benchmark/richmond`; summary committed as
`analysis_out/crop_window_eval_richmond.json`. New elements vs Round 1: containment is
finally *whole-apron* containment; the **directional road-context margin** — how far the
window extends below the box bottom (≈ the ramp–street junction), in box heights — is
reported per rule ("road p50"; "edgecut" = share of windows that clip the street edge
off); and the **size ratio + constant-rescale sweep**, which is what separates a rule's
accuracy from its scale constant (see Finding 4 — the first pass at this section drew the
wrong conclusion for want of it). Sanity check on the gold itself: per-band box-size
medians run ~1.2–1.3× the 1.5 m flat-ground nominal, uniformly across distance — true
full-apron extents (manual_labels sat at ~0.13×).

**Detection-prompted (production-realistic), n=227:**

| rule | containment (95% CI) | ctx p50 | road p50 | side p50 | **size ratio p10 / p50 / p90** | **p90/p10** |
|---|---|---:|---:|---:|---:|---:|
| v1-raw | 0.449 [0.39, 0.51] | 0.87 | 1.4 | 340 px | 0.55 / 0.95 / 1.79 | **3.24** |
| v1-norm | 0.260 [0.21, 0.32] | 1.02 | 1.1 | 304 px | 0.46 / 0.80 / 1.40 | **3.02** |
| geo-v1.5 | 0.996 [0.98, 1.00] | 0.23 | 6.6 | 1458 px | 2.03 / 3.59 / 6.26 | **3.08** |

**Finding 4 — the v1 formula is object-sized, not window-sized; but the *ranking* those
containment numbers imply is a scale artifact.** Both halves matter and only the first
survived the first pass at this.

The v1 half is real. Both variants produce windows about the size of the ramp itself
(ctx p50 0.87–1.02), so containment collapses monotonically with proximity: v1-raw goes
0.86 → 0.70 → 0.39 → 0.22 → 0.00 across the distance strata (far → <5 m); v1-norm never
exceeds 0.43 in any band. A 2013 formula fitted to a near-point labelling convention is
simply too small to hold a full apron at close/mid range.

The ranking half does not. **Containment and context ratio are both monotone in window
size**, so a rule can win either one by being uniformly bigger — and geo-v1.5 is
uniformly bigger, by 4.5×. The scale-free comparison is the *size ratio*, predicted side
over the side that box and prompt actually require (`required_side`, a property of the
gold, not of any rule). Its **p90/p10 spread is 3.0–3.2 for all three rules**. They are
equally accurate; they differ in one constant. Sweeping that constant (`--rescale-sweep`,
re-scored with the real window geometry including the pano-dimension cap):

| rule | k for ≥99.5 % containment | ctx p50 there | side p50 there |
|---|---:|---:|---:|
| v1-raw | ×3.0 | 0.291 | 1020 px |
| v1-norm | ×3.5 | 0.291 | 1063 px |
| geo-v1.5 | ×1.0 | 0.229 | 1458 px |

At matched containment the v1 rules cut a **~30 % smaller median window** for the same
guarantee; geo-v1.5's extra size buys extra context, which is a per-consumer preference
(Finding 3), not accuracy. So the conclusion for
[SW#4865](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/4865) is **not** "ship
geometry instead" — it is:

- **The resolution normalization is correct and should ship.** Upstream's own docstring
  says `old_pano_y` "converts pano_y and pano_height to the OLD version of pano_y that we
  had when this alg was written", and that conversion is exactly what is missing; v1-norm
  is the faithful port, not a variant. It is also the *best-calibrated* of the three here
  (spread 3.02).
- **Its constant is ~3.5× too small for whole-apron containment.** One multiplier, chosen
  per consumer class from the sweep table — a one-line change to the planned port.
- **The number that no rule beats is the spread.** Nothing is within ~1.75× of the right
  size at either tail, at any scale. *That* residual — not the constant — is the case for
  an extent-aware rule (SAM2 box, #83), and it is invisible in a containment column.

**geo-v1.5's target ratio is not calibrated at 2.5 m.** It measures ctx p50 0.229 against
its own 0.125 target — 1.83×, uniformly across bands, and not a clamp artifact (2 of 227
windows hit the pano cap). The gold's own apron-size check says extents run ~1.2–1.3× the
1.5 m nominal, which predicts 0.156, not 0.229. The gap is the camera height: richmond is
Mapillary, and `--cam-height 1.7` puts the measured ratio at **0.156**, reconciling the
two exactly. 2.5 m over-estimates distance, so the apparent ramp is under-estimated and
the window comes out small — geo's one free parameter was absorbing the rig height. Per-pano
heights are now measurable on GSV ([sidewalk-auto-labeler#40](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/40),
median 2.21 m; #101), which is the way to remove the parameter rather than retune it.

**Aprons are 3.3:1, and every candidate cuts a square.** Median box aspect (width/height
in equirect pixels) is 3.29, p10–p90 2.0–5.9. Containment is therefore decided by
horizontal extent alone: at geo's ctx p50 of 0.229 the *vertical* context ratio is 0.068,
about 15 box-heights of sky and sidewalk. Most of the "enormous windows" complaint in
Finding 3 is the square-window assumption, not any sizing rule — the report now splits
`ctx_h`/`ctx_v` so a non-square candidate can be scored against the same gold.

Its **single containment miss** (of 227) is still diagnostic: a ~28 m ramp whose 231 px
window is only slightly larger than the 187 px box, contained at gold-center but clipped
at the detection prompt — far-field windows leave too little *placement slack* for
detection offset. A padding floor (or explicit placement-error allowance) in the far field
is the obvious tuning for whichever rule ships.

**The verdict is stable, not sample-limited:** it was already outside the CIs at the
112-box interim cut and did not move from 246 → 299 (v1-norm 0.295 → 0.263 → 0.260;
the interim exports behind those checkpoints are archived with provenance notes in
`benchmark/richmond/box_annotation_log/`, so the trajectory is regenerable).

**What the GSV arm can and cannot settle.** Re-annotating manual_gold via
`box_gallery.py --from-manual-labels` is still the right next step — a second provider,
a second labelling population, and the only check on whether the ~3.5× constant
transfers. But two things it will *not* do, both worth knowing before the session:

- **It will not probe the >6656 regime.** All 1,000 `benchmark/manual_gold` records are
  4096×2048, so the arm scores at pano_h = 2048 — below the calibration height, the same
  regime as richmond. Testing where the raw formula *over*-sizes needs records carrying
  native GSV dimensions (8192/16384).
- **It annotates at model resolution.** Those same records mean 1024 px crops, versus
  richmond's 2750–3072 px, so "tight at native zoom" is a coarser instrument there.
  (9 of richmond's own 92 panos are 4096×2048 too — `crop_px_by_pano_dims` in both the
  gold and the summary now records this rather than leaving it to be assumed.)

Caveats: single annotator (jonf); 4 pano-height groups {5500: 59, 6144: 10, 2880: 8,
2048: 9 panos} so richmond alone cannot separate height effects; `missed:*` items (72)
score only in gold-center mode (production cuts no crop without a detection); 11 ramps
are "can't determine extent" and excluded — `boxes.json` is cross-checked against
`verdicts.json` (299 + 11 = 310 adjudicated ✅) so a partial pass cannot silently shrink
the denominator. `required_side` ignores the near-pole clamp-by-shift, which can only
help containment, so the size ratio is a mild lower bound (6 of 227 geo rows shift, 0 of
227 for either v1 rule). The size-ratio spread on manual_labels (4.3–5.4) is *not*
comparable — those are near-point marks, so the denominator is not an extent.

## Caveats

- Flat-ground distance from depression (2.5 m camera height) is a proxy; per-pano heights
  vary (sidewalk-auto-labeler#40 measured median 2.21 m on GSV) and terrain is not flat.
  None of Finding 1 depends on the proxy's precision — a 2× distance error cannot explain a
  7× size gap that is uniform across deployments. Finding 4's *geo-v1.5 context ratio*
  does depend on it, and measurably so: see the `--cam-height` result there. The depression
  strata labels are printed at the 2.5 m convention regardless of `--cam-height`, so they
  stay comparable across runs — read them as bands, not distances.
- 1.5 m nominal ramp width is a convention; real aprons vary ~1–3 m. It enters geo-v1.5
  multiplicatively alongside camera height, so richmond alone cannot separate "aprons are
  1.25× nominal" from "the rig sits at 2.0 m" — only their product is identified.
- The visual checks use fresh GSV fetches of gold pano ids (some ids have decayed); the
  bundle's canonical imagery remains the HF test split via `scripts/fetch_manual_gold.py`.
- Depression strata use the *box center's* y; detection-prompted rows use the detection's
  position for the window but the box's stratum.
