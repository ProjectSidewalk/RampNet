# São Paulo box-annotation log

Progress snapshot from the annotation session (jonf, 2026-08-18) that produced
`../boxes.json` — whole-apron extent gold over São Paulo's adjudicated ramps (#116),
drawn under box rule **v2**, the same rule as Richmond's, Paterson's and Annapolis' gold.

## This gold is a deliberate PARTIAL sample — read this before scoring it

Like `benchmark/paterson/` and `benchmark/annapolis/`, and unlike `benchmark/richmond/`
(all 310 adjudicated ramps), this set covers a **subset**: 127 of São Paulo's 281. That
is by design. Richmond established that the within-city answer converges by ~100 boxes,
so the effort is spread at ~100 each across four cities; the open questions are all
*between*-city.

**The subset is an unbiased random sample of panos.** `scripts/box_gallery.py` orders the
gallery by `sha1(pano_id)` with a pano's ramps kept consecutive, and that hash is
independent of every pano property, so any prefix is a random sample and stopping partway
is statistically clean. Per-ramp rates are estimated fine; anything needing *complete*
coverage of a pano set is not (every pano in this file is complete — the sampling unit is
the pano, not the ramp).

## Why this city mattered

Three reasons, all between-city:

1. **A non-US design standard** (NBR 9050). If the per-city scale constant tracks how big
   ramps actually are, a different design code should move it. It does — São Paulo needs
   the *smallest* constant of the four cities (v1-norm ×2.0 for ≥98.5% containment,
   against Paterson ×2.5, Richmond ×3.5, Annapolis ×4.0).
2. **A second GSV city.** The tighter depression-only fit replicates: R² 0.653 here and
   0.600 on Paterson, against 0.478 (Richmond) and 0.431 (Annapolis), both Mapillary.
   With resolution already excluded by Annapolis, provider is what is left standing.
3. **Hills.** Every candidate rule assumes flat ground, and this is the hilly city. No
   penalty is visible: São Paulo has the *tightest* scale-free spread of the four
   (2.06 / 2.08 / 1.79 for v1-raw / v1-norm / geo-v1.5).

The annotation-zoom rival explanation stays ruled out: median box width in the drawing
view is 470 px here (Paterson 398, Annapolis 336, Richmond 278), so a 1 px hand-jitter is
0.21% of a box. São Paulo is the *finest* drawing view and Paterson the second finest,
yet Annapolis (336 px) sits between Paterson and Richmond on zoom while matching
Richmond on spread — zoom does not order the spreads, provider does.

## Snapshots

| file | items | box rule | notes |
|---|---|---|---|
| `2026-08-18_127of281.json` | 112 boxed + 15 can't | v2 | Single session, the whole pass. 39 panos (38 × 16384×8192 at 4096 px crops, 1 × 13312×6656 at 3328 px); 71 of the 112 det-prompted; none edge-flagged. |

## Annotation notes

- **São Paulo's ramps are the smallest of the four cities** by implied cross-range extent
  (median 2.1 m at a nominal 2.5 m camera height, against Paterson 2.3 m, Richmond 2.6 m,
  Annapolis 3.4 m). Metric figures are camera-height-dependent and should be read as
  suggestive; the angular numbers in #114 are assumption-free.
- **They also sit closer to the camera** — median depression 14.0° over the det-prompted
  boxes, against 10.4–12.0° in the other three. That shifted distribution is what breaks
  the "k / apparent-width ratio is constant to 3%" claim from the Annapolis pass (#114).
- **One item to re-check if this gold is ever reused for placement work:**
  `fto2w3ZBO7XYUyIPzctfxw det:0`'s stored prompt point sits 23.4° away from the box drawn
  for it — more than two box widths — while `det:1` on the same pano (marked
  can't-determine) sits *inside* that box. It looks like a box attached to the wrong
  detection. It is the single item that keeps every v1 variant off 100% containment at
  any scale; excluding it, v1-norm reaches 1.000 at ×2.0.

Same standing rule as the other cities: if a re-annotation by a different annotator ever
happens, render **without** prefill from this file or from `../boxes.json`.
