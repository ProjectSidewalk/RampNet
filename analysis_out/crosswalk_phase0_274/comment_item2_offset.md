## Phase 0 item 2: crowd-vs-OSM offset read, 17 dense cities

**Method.** In each of the 17 cities from the census with ≥ 100 Crosswalk labels and a median ≤ 5 m to the nearest OSM crossing node: every crowd label (server lat/lng, no re-raycast) is joined to its nearest `highway=crossing` node within 10 m; a seeded sample of up to 1,500 matched labels per city (all of them in Seattle); the ways through the sampled nodes are pulled from Overpass with geometry. The crossing axis is the `footway=crossing` (or cycleway/path `=crossing`) way through the node when one exists, else the normal to the road way at the node. The offset vector label − node is projected on that axis (**along** the crossing, i.e. toward one kerb or the other) and on the road axis (**across**, i.e. up or down the street). Under a centre convention the along-crossing offset is small and symmetric about zero; under an end convention it has a mode near half the carriageway width (3–6 m) with either sign. Same script family as item 1; cached pulls; CSV per city with `label_uid`, node id, axis source, both components, the node's marking tag and the human verdict.

### Per-city table

| city | labels | matched <=10 m | sampled | axis from footway | median |along| m | p90 |along| | median |across| m | p90 |across| | |along| > 3 m | |across| > 3 m |
|---|---|---|---|---|---|---|---|---|---|---|
| chicago-il | 28091 | 25938 | 1500 | 98% | 0.87 | 3.07 | 0.69 | 2.19 | 10% | 6% |
| seattle-wa | 6283 | 6034 | 6034 | 99% | 1.3 | 4.3 | 1.04 | 3.3 | 21% | 12% |
| amsterdam | 2976 | 2571 | 1126 | 17% | 0.94 | 3.39 | 1.12 | 3.05 | 13% | 10% |
| st-louis-mo | 2890 | 2018 | 1500 | 98% | 1.47 | 4.75 | 1.28 | 4.06 | 23% | 17% |
| gainesville-fl | 2420 | 2227 | 1497 | 92% | 0.95 | 4.14 | 0.83 | 2.21 | 18% | 4% |
| burnaby | 1728 | 1693 | 1496 | 100% | 1.34 | 4.92 | 0.96 | 3.12 | 27% | 11% |
| pittsburgh-pa | 1018 | 819 | 819 | 97% | 1.08 | 3.99 | 0.9 | 3.11 | 17% | 10% |
| columbus-oh | 889 | 739 | 739 | 71% | 1.05 | 3.56 | 1.27 | 3.41 | 12% | 14% |
| tucson-az | 691 | 510 | 510 | 99% | 1.18 | 4.23 | 1.21 | 3.88 | 17% | 16% |
| danville-il | 488 | 419 | 419 | 99% | 1.45 | 5.34 | 1.41 | 3.24 | 26% | 13% |
| detroit-mi | 283 | 212 | 212 | 33% | 0.9 | 2.84 | 0.71 | 2.24 | 8% | 7% |
| waltham-ma | 281 | 251 | 251 | 96% | 0.99 | 3.61 | 0.76 | 2.19 | 15% | 5% |
| auckland | 273 | 254 | 252 | 97% | 1.61 | 4.61 | 1.11 | 3.71 | 24% | 13% |
| rancagua-chile | 148 | 112 | 111 | 12% | 1.37 | 5.3 | 2.58 | 5.69 | 22% | 42% |
| madison-wi | 137 | 107 | 107 | 99% | 1.6 | 5.06 | 0.89 | 2.95 | 25% | 8% |
| santiago-chile | 128 | 109 | 109 | 18% | 1.53 | 5.38 | 2.49 | 5.84 | 32% | 40% |
| sao-paulo-brazil | 105 | 80 | 80 | 56% | 0.94 | 3.3 | 2.5 | 6.72 | 15% | 41% |

### Pooled cuts

| subset | n | median \|along\| | p90 | median \|across\| | p90 | \|along\| > 3 m | signed along median |
|---|---|---|---|---|---|---|---|
| all sampled | 16762 | 1.17 | 4.26 | 1.00 | 3.22 | 19% | -0.06 |
| human-confirmed | 3512 | 1.18 | 4.11 | 1.03 | 3.15 | 19% | -0.04 |
| human-rejected | 183 | 1.21 | 5.51 | 1.09 | 4.52 | 26% | -0.31 |
| axis = footway way | 14961 | 1.20 | 4.30 | 0.99 | 3.15 | 20% | -0.06 |
| axis = road normal | 1801 | 1.00 | 3.77 | 1.14 | 3.64 | 15% | -0.08 |
| node tagged marked | 12604 | 1.22 | 4.29 | 1.02 | 3.18 | 20% | -0.07 |
| node tagged unmarked | 1182 | 1.12 | 4.48 | 1.21 | 4.12 | 19% | -0.04 |
| node untagged | 2976 | 1.03 | 3.98 | 0.90 | 3.06 | 17% | -0.05 |
| on a oneway road | 5076 | 1.25 | 4.63 | 1.05 | 3.47 | 23% | -0.05 |

### Reading 1: the crowd labels the centre, and the OSM node is that centre

Pooled over 16,762 sampled labels the median offset is 1.17 m along the crossing and 1.00 m across it; the signed along-crossing median is -0.06 m, so the labels scatter symmetrically about the node rather than piling up at a kerb, and only 19% are more than 3 m from it along the crossing. There is no half-carriageway mode anywhere in the 17 cities (per-city medians 0.87–1.61 m along). Human-confirmed labels (3,512) read the same (1.18 m along, 1.03 m across), so this is not an artefact of unvalidated labels. These numbers are at or under the server's own placement error for a label raycast from a pano, which means the OSM node is, to the precision we can measure here, the midpoint the labeling convention asks for (the umbrella's Decision on the target convention stands: **centre of the marked extent**).

### Reading 2: the caveats from the review are small or confined

- **Dual carriageways.** Labels whose node sits on a `oneway=yes` road (5,076) read 1.25 m along / 1.05 m across, against 1.17 / 1.00 overall: no separate mode. Each carriageway of a dual road tends to carry its own crossing node, and the crowd labels each half.
- **Junction-node tagging.** The exception is the three Latin American cities, Santiago, Rancagua and São Paulo: only 12–56% of their nodes have a footway crossing way, and the **across**-road component jumps to a 2.5 m median with ~40% beyond 3 m, while the along-crossing component stays at ~1–1.5 m. That is a node tagged on or near the junction node (or a crossing drawn as one node for several marked crossings) rather than at the crossing, so in those cities the node gives the bearing but not the midpoint. CDMX, Cuenca and the rest of the sparse group were not in this read at all.
- **Axis source.** Road-normal and footway axes give the same picture (1.00 vs 1.20 m along), so the result does not hinge on the footway ways being drawn.
- **Marking tag.** Labels on nodes tagged `unmarked` (1,182) sit exactly like labels on `marked` nodes (1.12 vs 1.22 m along). The crowd places them the same way; whether those are stale tags or unmarked crossings the crowd calls `Crosswalk` is the question only the aerial sheet answers (below).

### What this sets for item 3 and for Stage 1

- **Matcher radius for the Vistas probe:** the pooled p90 is 4.3 m along / 3.2 m across the crossing. At the 10–25 m ranges a crossing is seen from in a pano, 4–5 m of ground offset is ~10–25° of bearing, well inside Stage 1's ±18.37° tolerance at the near end and the reason the probe's reading rule uses a radius of 0.05 of the pano width (~18°) rather than the ramp radius.
- **Stage 1 bearing source:** an OSM crossing node is usable as the bearing anchor in the dense group without any midpoint correction; in the junction-tagged cities it needs the footway way or a road-normal offset, which the crop model is there to absorb.
- **Aerial reference (Seattle):** a seeded rating sheet is ready, 120 human-confirmed labels on nodes tagged marked / unmarked / untagged with both positions, the offset and map links, for a person to rate on a basemap: is there a painted crossing at the node, and is the label at its centre. It is the step that decides whether `crossing:markings` can be a positive source at all. It needs a rater and an imagery source, so it is the one item in Phase 0 that waits on a person.

🤖 Generated with [Claude Code](https://claude.com/claude-code) — Fable 5.1, claude-fable-5-1
