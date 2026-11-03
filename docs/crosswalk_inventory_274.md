# Crosswalk outside oracles: Seattle's SDOT inventory, Li et al. 2023, and related work (RampNet#274)

Side work run in parallel with the Phase 1 human GT round, 2026-11-03. Question: is there crosswalk ground truth
we can get without a person rating it? Two outside sources were scored, both against the 120 final human
verdicts of the Phase 0 Seattle aerial spot-check (`analysis_out/crosswalk_phase0_274/aerial_sample_seattle-wa.rated.csv`).
Nothing here touches the Phase 1 rating page or reads anything that could unblind it.

## Findings

**1. SDOT's "Marked Crosswalks" layer agrees with the human verdicts, but misses the fixed bar by 0.002.**
The rule was fixed in `sdot_inventory.py` before the first score: an SDOT point within 10 m of the OSM node
counts; GT-GRADE requires sensitivity ≥ 0.90 with a Wilson lower bound ≥ 0.85, and specificity ≥ 0.75.

| radius | sensitivity (human painted → SDOT point within r) | specificity (human not painted → none) |
|---|---|---|
| 5 m | 83/107 = 0.776 | 12/12 = 1.000 |
| **10 m (rule)** | **98/107 = 0.916 [0.848, 0.96]** | **12/12 = 1.000 [0.76, 1.00]** |
| 15 m | 100/107 = 0.935 | 5/12 = 0.417 |

Verdict under the rule: **THIRD OPINION**, because the lower bound is 0.848 against 0.85. Two readings sit
beside it. Neither changes the verdict:

- **The misses are mostly coverage, not paint.** 5 of the 9 human-painted rows SDOT misses have no SDOT point
  of any status within 100 m. Four are one cluster at about 47.521, −122.358, which looks like the Westwood
  Village shopping-centre lot (private property). One is on Aurora Ave (SR 99, a state route). On the 102 rows
  SDOT covers, sensitivity is 98/102 = 0.961 [0.90, 0.98]. This reading is post hoc: the 100 m "coverage hole"
  cut was chosen after seeing the misses.
- **Specificity depends on the radius.** It holds at 10 m and collapses at 15 m, because a junction's unmarked
  leg sits 10–15 m from its marked leg. 12 rows is also too few to say much either way.

Where SDOT's records reach, the inventory looks good enough to stand in for a rater. Making that official needs
an openly amended rule that adds a coverage mask. That's a decision for Jon, not something done here.

**2. OSM's marked tag finds nearly all of SDOT's crosswalks.** 90.6% [0.90, 0.91] of the 5,887 current SDOT
crosswalks inside Project Sidewalk's Seattle regions have an OSM-marked crossing node within 10 m, and 95.4%
have some crossing node. In the other direction, 71.7% of the 8,055 OSM-marked nodes have an SDOT point within
10 m. 10.8% have none within 100 m, a coverage hole. The remaining ~17% sit 10–100 m from SDOT, and nothing here
says which source is wrong at those nodes. A small rated sample of them would settle it (see open questions).
Seattle's OSM `unmarked` class (25,500 nodes) almost never has an SDOT crosswalk at it (2.2% within 10 m).
Phase 0's 28 of 40 unmarked nodes painted was a sample of nodes the crowd had labelled, not of unmarked nodes
in general.

**3. SDOT separates validated crowd labels sharply.** Of crowd Crosswalk labels judged correct by human votes
(agree − disagree ≥ 2), 83.7% lie within 10 m of an SDOT crosswalk. For labels judged incorrect it's 5.2%, and
for undecided ones 79.8%. So in Seattle, SDOT works as an automatic check on crowd labels, and it suggests
most undecided labels are right.

**4. Li et al. 2023 is a weak opinion and covers almost none of Phase 1.** Its counts come from a model at
each intersection near a US rail station; the latest Seattle year is 2014–2019. Of the 306 Seattle
intersections it covers, it finds a crosswalk at 229 of the 233 where SDOT has one (0.983). But it also reports
one at 50 of the 73 where SDOT has none within 30 m, and at 41 of those OSM has no marked node either. Presence
agreement with SDOT is 0.824, against 0.935 for OSM-marked. Only 9 of the 320 Phase 1 rows lie within 30 m of a
Li intersection, all in Chicago. Not worth carrying further.

**Amendment (2026-11-03, post hoc, adopted with Jon's go-ahead).** The rule gains a coverage mask: a row with no
SDOT point of any status within 100 m is outside the inventory's coverage and leaves the denominator, as
`unclear` does. The thresholds are unchanged, and the original verdict is still printed beside the amended
one. Under it SDOT is **GT-GRADE inside its coverage**: sensitivity 98/102 = 0.961 [0.90, 0.98], specificity
10/10 [0.72, 1.00] (two of the 12 no-rows fall outside coverage too). The mask was chosen after seeing the
misses, so this reading is weaker than a pre-registered pass, and specificity rests on 10 rows.

**5. The map-only bar a detector has to beat** (`map_baseline.py`, `baseline_report.md`). With SDOT as truth
inside its coverage, and no pixels at all:

| predictor | per OSM node (22,020 scored): precision / recall / F1 | per SDOT crosswalk (5,887): found within 10 m |
|---|---|---|
| OSM marked tag | 0.800 / 0.879 / 0.838 | 0.906 |
| OSM marked, or signalised | 0.800 / 0.897 / **0.846** | 0.914 |
| a crowd label within 10 m | 0.801 / 0.416 / 0.548 | 0.408 (0.732 where the crowd has audited) |
| OSM (marked or signalised) or crowd | 0.772 / 0.930 / 0.844 | **0.929** |

So OSM alone answers "is this crossing painted?" at F1 ≈ 0.85 in Seattle. What's left for a detector:

- **553 OSM `unmarked` nodes with SDOT paint within 10 m** (170 of them within 3 m, which look like stale OSM
  tags; the rest may be an adjacent painted leg).
- **122 untagged nodes with SDOT paint** that the map can't call.
- **1,469 map-yes nodes with no SDOT point within 10 m.** Half have one 10–25 m away (likely an offset or the
  next leg), half are 25–100 m away (stale tag, removed paint, or a crosswalk SDOT doesn't record).
- **~7% of SDOT crosswalks with no map positive within 10 m**, 28% for mid-block ones.

The mask also drops 20,731 of the 42,751 nodes, mostly unmarked residential crossings with no crosswalk of any
kind nearby. The scored negatives are therefore the hard ones near painted crossings, which suits a detector
comparison but means these precision numbers don't describe the whole city.

## Open questions for Jon

1. ~~Amend the SDOT rule with a coverage mask?~~ Adopted 2026-11-03 (above).
2. Rate a small sample (~40) of OSM-marked nodes that sit 10–100 m from any SDOT point? That says whether those
   are OSM tag errors, SDOT position offsets or crosswalks SDOT doesn't own. It's the one ambiguity left in
   finding 2.

## Replication

No GPU. Network only in each `fetch` (SDOT's ArcGIS FeatureServer, the Seattle Project Sidewalk server's
`/v3/api/regions`, Zenodo; GET only). OSM tiles and crowd labels come from the Phase 0 census cache
(`crossing_census.py --out OUT`, pulled 2026-10-28; a safety copy is at
`D:/Git/sal-scratch-keep/phase0-census-2026-10-28/census/out`).

```bash
python scripts/analysis/crosswalk_inventory_274/sdot_inventory.py fetch --out analysis_out/crosswalk_inventory_274
python scripts/analysis/crosswalk_inventory_274/sdot_inventory.py score --out analysis_out/crosswalk_inventory_274 \
    --phase0 analysis_out/crosswalk_phase0_274/aerial_sample_seattle-wa.rated.csv --census <census OUT>
python scripts/analysis/crosswalk_inventory_274/map_baseline.py --out analysis_out/crosswalk_inventory_274 \
    --census <census OUT>
python scripts/analysis/crosswalk_inventory_274/li2023_check.py fetch --out analysis_out/crosswalk_inventory_274
python scripts/analysis/crosswalk_inventory_274/li2023_check.py score --out analysis_out/crosswalk_inventory_274 \
    --census <census OUT> [--phase1-sheet <crosswalk-gt-phase1>/analysis_out/crosswalk_gt_phase1_274/sheet.csv]
```

SDOT refreshes daily, so a re-fetch moves the numbers. `inputs.json` records each pull's URL, time, sha256 and
feature count, and `score` refuses an input whose hash has changed since its fetch.

| number | lives in |
|---|---|
| calibration table, verdict, post-hoc coverage line, the 9 disagreements | `analysis_out/crosswalk_inventory_274/report.md` §2; per row in `phase0_calibration.csv` |
| OSM nodes vs SDOT | `report.md` §3; per node in `osm_nodes.csv` |
| SDOT vs OSM | `report.md` §4 |
| crowd labels vs SDOT | `report.md` §5; per label (`seattle-wa:<label_id>`) in `crowd_labels.csv` |
| amended verdict | `report.md` §2 |
| map-only baseline (finding 5) | `baseline_report.md`; per node in `baseline_nodes.csv`, per SDOT crosswalk in `baseline_sdot.csv` |
| Li et al. vs SDOT/OSM, Phase 1 coverage | `li2023_report.md`; per intersection in `li2023_seattle.csv` |

TODO: no figures yet; add a `figures` subcommand if this becomes a PR.

## Related work: crosswalk datasets and methods

Checked 2026-11-03. "Labels" says where each dataset's crosswalk labels come from, which decides whether it can
judge OSM. Several large sets take their labels *from* OSM, so they can't tell us where OSM is wrong.

| work | imagery | labels | released | relevance |
|---|---|---|---|---|
| Ahmetovic, Manduchi, Coughlan & Mascetti, "Zebra Crossing Spotter", ASSETS 2015, doi:10.1145/2700648.2809847 | satellite, then Street View | own | no dataset found | Same cascade as ours (aerial candidates checked at street level), framed for blind travellers. Core related work. |
| Berriel et al., "Deep learning based large-scale automatic satellite crosswalk classification", IEEE GRSL 2017, [arXiv 1706.09302](https://arxiv.org/abs/1706.09302) | Google Static Maps | **OSM** | scripts + regions | 245,768 tiles, 9 countries. Labels inherit OSM's errors. |
| Berriel et al., "Automatic large-scale data acquisition via crowdsourcing for crosswalk classification", [arXiv 1805.11970](https://arxiv.org/abs/1805.11970) | Street View, Brazil | **OSM**, hand-labelled test set | scripts | 657,691 images. Its hand-labelled test set is a published OSM error rate, relevant to São Paulo. |
| Li, Sheng, Irvin, …, Ng & Rodriguez, Environment and Planning B 2023, doi:10.1177/23998083221112157 | Street View, 2007–2020 | model trained on 4,925 hand-labelled images | per-intersection counts, [Zenodo 6803956](https://zenodo.org/record/6803956), CC BY 4.0 | Scored above (finding 4). The hand-labelled images don't appear to be released. |
| Hosseini et al., Tile2Net, CEUS 2023, doi:10.1016/j.compenvurbsys.2023.101950, [repo](https://github.com/VIDA-NYU/tile2net) | aerial | city planimetric data | model + code | Crosswalk polygons from orthos. Tried on labeler#104 (negative); re-read what that tested before reusing it. |
| CrosswalkNet, [arXiv 2506.07885](https://arxiv.org/abs/2506.07885), 2025 (with a MassDOT report) | Massachusetts aerial | 23k hand-annotated | not confirmed | The largest hand-annotated aerial set found; worth asking the authors. |
| IEEE DataPort "Aerial image (roads & sidewalks/crosswalks)", doi:10.21227/fr6t-6t64 | USGS ortho, DC + Cambridge | GIS-derived | DataPort (access terms unchecked) | Lumps sidewalk and crosswalk into one class. |
| OSMDeepOD (Swiss), [repo](https://github.com/codeaudit/OSMDeepOD) | aerial, 50×50 px | own | on request | Small patches. |
| Mapillary Vistas (Neuhold et al., ICCV 2017) | street level, worldwide | hand-annotated, crosswalk classes | research licence | Probed in Phase 0: usable as an extent filter, not a placer. |
| BDD100K (Yu et al., CVPR 2020) / ApolloScape (Huang et al., 2018) | dashcam / vehicle | hand-annotated | yes | Crosswalk boundaries (BDD) or full zebra masks (ApolloScape). Pretraining at most. |
| FPVCrosswalk2025 (Romić et al., Data in Brief 2025, doi:10.1016/j.dib.2025.111755) | first-person video | quadrilateral masks | [Mendeley](https://data.mendeley.com/datasets/mcr2jwk5bp) | 300 real + 3,000 synthetic images. Different viewpoint. |
| Satellite + street-view crosswalk detection with ground-truth checking, [ScienceDirect 2024](https://www.sciencedirect.com/science/article/pii/S2046043024000066) | Bing satellite + street view | own | not checked | Street-view checking improved accuracy on an unseen region. |
| OpenSatMap (NeurIPS 2024 D&B), [arXiv 2410.23278](https://arxiv.org/abs/2410.23278) | satellite | lane-level lines | yes | No crosswalk class found. |

## Other cities' crosswalk inventories (for training labels beyond Seattle)

Searched 2026-11-03. "Live" means the endpoint's metadata and feature count were queried that day. None of these
is derived from OSM. The score is how useful the set would be as Stage 1 labels (5 = full city, current, marked
only, at the crosswalk). None has been scored against human verdicts the way SDOT was, so each one needs that
calibration before it is used.

| city / agency | endpoint | geometry, count | attributes | score |
|---|---|---|---|---|
| Washington DC, OCTO planimetrics 2023 | `maps2.dcgis.dc.gov/.../Planimetrics_2023/MapServer/11`, `DESCRIPTION='Crosswalk'` | polygon, 14,279 | Marked crosswalks visible in aerials; vintages back to 1999. DDOT intersection inventory (`.../MapServer/96`) adds style. | 5 (live) |
| Los Angeles, LADOT | `maps.lacity.org/lahub/rest/services/LADOT/MapServer/4` | point, 22,507 | Continental, standard, ladder, decorative; dates 2005–2026; no removed status | 5 (live) |
| Taipei, 交工處 行穿線 | data.gov.tw dataset 151593 (SHP) | polygon, ~19.6k | Zebra vs ladder bars; 2024-07; Government Open Data Licence | 5 (headers only) |
| Austin, TPW | Socrata 9hak-anfp | line across the crossing, 7,328 | Continental or transverse; active or removed; city streets only | 5 (live) |
| Portland OR, PBOT | `portlandmaps.com/.../MapServer/201`, `LineType=3621` | line, 6,825 | Continental or parallel; last maintenance date | 4 (live) |
| Dallas | `.../TrafficControlDevices/FeatureServer/0` | point, 9,801 crosswalks | Continental flag | 4 (live) |
| Denver | `.../WebEdits_ASSET_SPECIALTYMARKINGS_P/FeatureServer/0` | point, 9,312 continental | Transverse crosswalks probably in a companion lines layer | 4 (live) |
| **Bend OR** | `services5.arcgis.com/JisFYcK2mIVg9ueP/.../Crosswalks/FeatureServer/7` | polyline, 6,652 | **1,117 marked, 5,501 unmarked**: explicit hard negatives; Bend is the experimental city | 4 (live) |
| Cambridge MA, Calgary, Brampton, Zurich (canton roads only), Raleigh | various | 1.5k–17.8k | Pavement-marking layers with a crosswalk class | 3 (live) |
| Toronto, San Francisco, Pittsburgh | various | — | Marked status unclear, mostly undetermined, or stale (2018) | 2 |

Also: Columbus/MORPC (marked vs unmarked lines, "not field verified"); the Ohio DOT ADA inventory; Amsterdam BGT
(CC-0) and the Dutch road-attributes database; Berlin's Fußgängerüberwege (WFS timed out). Nothing was found for
Chicago, Boston, Oakland, San Jose, Philadelphia, Bellevue, Spokane, Gainesville, Montreal or Mexico City.
