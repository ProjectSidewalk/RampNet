# São Paulo box-annotation log

Progress snapshots from the annotation session (jonf, 2026-08-18) that produced
`../boxes.json` — whole-apron extent gold over São Paulo's adjudicated ramps (#116),
drawn under box rule **v2**, the same rule as Richmond's, Paterson's and Annapolis' gold.

## This gold is a deliberate PARTIAL sample — read this before scoring it

Like `benchmark/paterson/` and `benchmark/annapolis/`, and unlike `benchmark/richmond/`
(all 310 adjudicated ramps), this set covers a **subset**: 134 of São Paulo's 281. That
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
   the *smallest* constant of the four cities (v1-norm ×2.0 for ≥98.5% det-mode
   containment, against Paterson ×2.5, Richmond ×3.5, Annapolis ×4.0), and its ramps are
   the smallest: apparent width 10.1° at a matched 12 m range, against 11.1° / 12.8° /
   15.7°.
2. **A second GSV city.** The tighter depression-only fit replicates, and then some:
   R² 0.738 here and 0.600 on Paterson, against 0.478 (Richmond) and 0.431 (Annapolis),
   both Mapillary.
3. **Hills.** Every candidate rule assumes flat ground, and this is the hilly city. No
   penalty is visible: São Paulo has the *tightest* scale-free spread of the four
   (2.07 / 2.11 / 1.81 for v1-raw / v1-norm / geo-v1.5) and the tightest size-vs-range
   relation (log-log slope −0.90, r = −0.91, against −0.80 to −0.87 and r −0.62 to −0.86
   elsewhere; −1.00 is the geometric ideal for a fixed physical size).

The annotation-zoom rival explanation stays ruled out: median box width in the drawing
view is 440 px here (Paterson 398, Annapolis 336, Richmond 278), so a 1 px hand-jitter is
0.23% of a box. São Paulo is the *finest* drawing view and Paterson the second finest,
yet Annapolis (336 px) sits between Paterson and Richmond on zoom while matching
Richmond on spread — zoom does not order the spreads, provider does.

## Snapshots

| file | items | box rule | notes |
|---|---|---|---|
| `2026-08-18_127of281.json` | 112 boxed + 15 can't | v2 | First export of the session. Superseded — see the correction below. |
| `2026-08-18_134of281.json` | 119 boxed + 15 can't | v2 | The gold. 40 panos carry boxes (39 × 16384×8192 at 4096 px crops, 1 × 13312×6656 at 3328 px); 75 of the 119 det-prompted; none edge-flagged. |

### The correction between the two snapshots

Scoring the first export surfaced one item that no crop scale could contain:
`fto2w3ZBO7XYUyIPzctfxw det:0` carried a box **23.4° away from its own prompt point**,
while `det:1` on the same pano — marked can't-determine — sat *inside* that box. It was a
box drawn around the wrong ramp (gallery item 122/281). The second export fixes it
(det:0 boxed at its own ramp, det:1 boxed at the corner apron) and adds six further boxes
on other panos.

Everything published from the first export is superseded by the second: with the fix,
every rule reaches 100% containment at its own constant, São Paulo's depression-only R²
rises 0.653 → 0.738, and the earlier explanation for its lower constant — "its ramps sit
closer to the camera" — is **withdrawn**. The four cities' depression distributions are in
fact matched (median 10.4–12.0°, median range 11.8–13.6 m); what actually differs is
apparent ramp size at matched range, plus how tightly a city's ramps obey the range law.

## Annotation notes

- **São Paulo's ramps are the smallest of the four cities**: implied physical width 2.1 m
  at 12 m range, against Paterson 2.3 m, Richmond 2.7 m, Annapolis 3.3 m (nominal 2.5 m
  camera height — metric figures are height-dependent and should be read as suggestive;
  the angular numbers are assumption-free).
- **The Annapolis "k ÷ apparent width is constant to 3%" identity does not hold here** —
  São Paulo lands at 0.089 against 0.106 / 0.111 / 0.107. Apparent size still *orders* the
  constant; what it does not do is predict it to better than ~±10%, because the constant
  also absorbs how tightly apparent size tracks range within the city (São Paulo's r
  −0.91 vs Annapolis' −0.62).

Same standing rule as the other cities: if a re-annotation by a different annotator ever
happens, render **without** prefill from this file or from `../boxes.json`.
