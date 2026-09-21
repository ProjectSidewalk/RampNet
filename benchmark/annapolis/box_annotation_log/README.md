# Annapolis box-annotation log

Progress snapshot from the annotation session (jonf, 2026-08-17) that produced
`../boxes.json` — whole-apron extent gold over Annapolis' adjudicated ramps (#116),
drawn under box rule **v2**, the same rule as Richmond's and Paterson's gold.

## This gold is a deliberate PARTIAL sample — read this before scoring it

Like `benchmark/paterson/`, and unlike `benchmark/richmond/` (all 310 adjudicated
ramps), this set covers a **subset**: 142 of Annapolis' 294. That is by design. Richmond
established that the within-city answer converges by ~100 boxes, so the effort is spread
at ~100 each across **paterson (GSV, 16384×8192) · annapolis (Mapillary, 8000×4000) ·
sao_paulo (GSV, hills)** — the open questions are all *between*-city.

**The subset is an unbiased random sample of panos.** `scripts/box_gallery.py` orders the
gallery by `sha1(pano_id)` with a pano's ramps kept consecutive, and that hash is
independent of every pano property, so any prefix is a random sample and stopping partway
is statistically clean. Per-ramp rates are estimated fine; anything needing *complete*
coverage of a pano set is not (every pano in this file is complete — the sampling unit is
the pano, not the ramp).

## Why this city mattered

Annapolis was annotated to separate two confounded explanations for Paterson scoring
better than Richmond under every depression-only sizing rule: imagery provider
(Mapillary SfM vs GSV) versus panorama resolution. Annapolis is Mapillary at 8000×4000 —
higher resolution than most of Richmond's panos — so it splits them.

It answered cleanly, and against resolution: Annapolis behaves like Richmond
(depression-only R² 0.431 vs Richmond's 0.478, Paterson's 0.600; scale-free spread
3.24/2.98/2.90 vs Richmond's 3.24/3.02/3.08 and Paterson's 2.06/1.94/2.44).

The rival explanation — that boxes drawn in a coarser view are quantised harder — is also
ruled out here: median box width in the drawing view is 336 px (Richmond 278, Paterson
398), so a 1 px hand-jitter is 0.30% of a box (0.36% / 0.25%). Annapolis sits *between*
the other two on annotation zoom while matching Richmond on spread.

## Snapshots

| file | items | box rule | notes |
|---|---|---|---|
| `2026-08-17_142of294.json` | 131 boxed + 11 can't | v2 | Single session, the whole pass. All 42 panos at 8000×4000 (2000 px crops); 94 of the 131 det-prompted; one edge-flagged. |

## Annotation note

Annapolis' ramps are the widest of the three cities annotated so far — median implied
cross-range extent 3.3 m against Paterson's 2.3 m and Richmond's 2.7 m — and that
difference turns out to drive the whole scale-constant question (see #114). Anyone
re-reading this gold for a different purpose should not treat the three cities'
box-size distributions as interchangeable.

Same standing rule as the other cities: if a re-annotation by a different annotator ever
happens, render **without** prefill from this file or from `../boxes.json`.
