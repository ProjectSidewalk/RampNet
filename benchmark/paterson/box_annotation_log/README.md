# Paterson box-annotation log

Progress snapshots from the annotation session (jonf, 2026-08-15) that produced
`../boxes.json` — whole-apron extent gold over Paterson's adjudicated ramps (#116),
drawn under box rule **v2**, the same rule as Richmond's completed gold.

## This gold is a deliberate PARTIAL sample — read this before scoring it

Unlike `benchmark/richmond/boxes.json` (all 310 adjudicated ramps), this set covers a
**subset** of Paterson's 395. That is by design, not an abandoned session.

Richmond established that the within-city answer converges early: the rule ranking was
stable across its 112 → 246 → 299 checkpoints, so boxes beyond ~100 bought CI width and
nothing else. The open questions after Richmond + the first Paterson boxes are all
*between*-city — does the scale constant vary by provider, by pano height, by city? —
and no amount of additional Paterson annotation can answer them. So the effort that
would have finished Paterson is spread at ~100 boxes each across
**paterson (GSV, 16384×8192) · annapolis (Mapillary, 8000×4000) · sao_paulo (GSV,
hills)**, with morgantown (Mapillary, 4096×2048 — the only real sample of the regime
where the v1 resolution normalization is a factor of 2) held as an optional fourth.

**The subset is an unbiased random sample of panos.** `scripts/box_gallery.py` orders
the gallery by `sha1(pano_id)` with a pano's own ramps kept consecutive
(`box_gallery.py`, "Stable pano shuffle"), and that hash is independent of every pano
property — capture date, resolution, ramp count, detection count. So any *prefix* of
the gallery is a random sample of panos, and stopping partway is statistically clean.
Two consequences for anyone using this file:

- Per-ramp rates (containment, context ratio, the scale constant) are estimated fine.
- Anything that needs *complete* coverage of a pano set is not — e.g. counting ramps
  per pano, or a neighbour-contamination metric evaluated against "all other gold
  ramps in this pano", which is only valid on the panos that were fully annotated
  (every pano in this file is complete; the sampling unit is the pano, not the ramp).

`scripts/analysis/crop_window_eval.py --bundle` prints a `completeness_warning` naming
the covered fraction, and every rate it reports is over that subset.

## Snapshots

The canonical gold is `../boxes.json`; these exist for replicability. Interim numbers
regenerate by pointing the scorer at a bundle whose `boxes.json` is the snapshot.

| file | items | box rule | notes |
|---|---|---|---|
| `2026-08-15_061of395.json` | 58 boxed + 3 can't | v2 | First checkpoint. All 13 panos at 16384×8192 (4096 px crops); 50 of the 58 det-prompted. Source of the "the ~3.5× constant does not transfer — GSV wants ~2.5×" reading on #114. |
| `2026-08-17_119of395.json` | 109 boxed + 10 can't | v2 | Session close — the ~100-box GSV target is met, so annotation moves to annapolis. 30 panos: 28 at 16384×8192, one 13312×6656, one 3328×1664; 86 of the 109 det-prompted. Superset of the first checkpoint with no edits to it (verified box-by-box). |

## Annotation note

Paterson's ramps sit closer together than any other benchmark city's, which is the
population where an over-sized crop window pulls a *neighbouring* ramp into the frame.
Nothing measures that yet — the scorer has containment, context, road-margin and
edge-cut, but no neighbour-contamination metric, so today only containment pushes on
the scale constant and bigger always scores safer. This gold is the material for the
opposite bound if that metric gets built.

Same standing rule as Richmond: if a re-annotation by a different annotator ever
happens, render **without** prefill from these files or from `../boxes.json`.
