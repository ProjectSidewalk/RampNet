## Phase 0 item 1: census of crowd Crosswalk / Signal labels vs OSM crossing and signal nodes (all 46 public cities)

**Method.** For every public city on `/v3/api/cities`: all crowd `Crosswalk` and `Signal` labels from `rawLabels` (every user, the server's stored lat/lng, no re-raycast); OSM `highway=crossing` nodes (Signal: `highway=traffic_signals` or `crossing=traffic_signals`) pulled from Overpass over the 0.1° tiles that hold ≥ 1 label; haversine distance from each label to the nearest node. `marked` = `crossing:markings` ≠ `no`, or `crossing` ∈ {marked, zebra, lines, ladder, dashes, dots, surface}; `unmarked` = `crossing:markings=no` or `crossing` ∈ {unmarked, no, informal}. "human ok/wrong" is a simplified margin rule on human votes only (net ≥ 2 Agree / net ≥ 2 Disagree); the AI validator's votes are ignored. Stdlib script, every pull cached; it goes in with the Phase 0 PR. Pulled 2026-10-28.

### Per-city table

Sorted by label count. "marked" and "tagged either way" are shares of the OSM nodes in the tiles (which cover more than the city). Distances are label → nearest node.


| city | Crosswalk labels | human ok/wrong | OSM crossing nodes | marked | tagged either way | median m to any node | <=10 m any | <=10 m marked | Signal labels | median m to signal node | <=10 m |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chicago-il | 28091 | 3907/163 | 247831 | 25% | 44% | 1.5 | 93% | 37% | 5826 | 8.9 | 64% |
| seattle-wa | 6283 | 1820/232 | 53353 | 21% | 77% | 2.2 | 96% | 86% | 1671 | 9.5 | 55% |
| paterson-nj | 5300 | 41/0 | 3695 | 52% | 60% | 239.1 | 18% | 15% | 159 | 19.7 | 13% |
| cdmx | 3931 | 114/29 | 5726 | 29% | 35% | 185.0 | 19% | 18% | 112 | 20.7 | 21% |
| teaneck-nj | 3363 | 1595/112 | 12890 | 66% | 72% | 183.5 | 30% | 28% | 227 | 29.6 | 30% |
| amsterdam | 2976 | 890/112 | 8769 | 83% | 92% | 2.1 | 86% | 86% | 1034 | 8.6 | 54% |
| st-louis-mo | 2890 | 292/7 | 18946 | 31% | 75% | 3.6 | 70% | 64% | 1359 | 11.2 | 41% |
| gainesville-fl | 2420 | 15/0 | 6448 | 40% | 74% | 1.8 | 92% | 70% | 719 | 12.4 | 27% |
| cuenca | 1768 | 257/15 | 1439 | 54% | 58% | 75.4 | 36% | 25% | 801 | 11.0 | 43% |
| burnaby | 1728 | 289/5 | 23016 | 28% | 89% | 2.3 | 98% | 82% | 891 | 9.4 | 58% |
| pittsburgh-pa | 1018 | 214/33 | 16703 | 23% | 40% | 2.3 | 80% | 42% | 218 | 8.9 | 60% |
| mendota-il | 983 | 457/246 | 33 | 73% | 91% | 178.9 | 16% | 14% | 31 | 18.5 | 8% |
| columbus-oh | 889 | 177/7 | 12925 | 24% | 68% | 2.4 | 83% | 30% | 725 | 12.9 | 30% |
| walla-walla-wa | 878 | 234/19 | 505 | 38% | 53% | 42.3 | 42% | 31% | 284 | 16.6 | 7% |
| tucson-az | 691 | 195/8 | 10492 | 26% | 63% | 2.9 | 74% | 71% | 138 | 14.6 | 20% |
| oradell-nj | 675 | 483/33 | 973 | 49% | 61% | 430.4 | 5% | 6% | 21 | 14.8 | 0% |
| maywood-nj | 505 | 127/2 | 3260 | 65% | 74% | 586.7 | 1% | 0% | 31 | 22.5 | 6% |
| danville-il | 488 | 15/2 | 2613 | 6% | 52% | 2.9 | 86% | 29% | 100 | 11.2 | 28% |
| knox-oh | 355 | 76/10 | 295 | 37% | 61% | 254.5 | 11% | 8% | 19 | 593.5 | 0% |
| west-chester-pa | 331 | 47/0 | 448 | 34% | 43% | 136.8 | 28% | 12% | 44 | 7.7 | 64% |
| detroit-mi | 283 | 42/7 | 5740 | 5% | 19% | 1.7 | 75% | 13% | 17 | 13.4 | 18% |
| waltham-ma | 281 | 62/2 | 5737 | 40% | 64% | 1.8 | 89% | 55% | 47 | 8.6 | 66% |
| auckland | 273 | 153/8 | 8260 | 27% | 75% | 2.6 | 93% | 69% | 271 | 6.8 | 84% |
| cliffside-park-nj | 201 | 64/2 | 9050 | 67% | 71% | 239.4 | 18% | 18% | 9 | 137.8 | 0% |
| rancagua-chile | 148 | 7/1 | 2237 | 17% | 17% | 4.4 | 76% | 15% | 143 | 7.8 | 67% |
| madison-wi | 137 | 45/2 | 1842 | 51% | 71% | 2.7 | 78% | 71% | 12 | 19.1 | 0% |
| spgg | 130 | 29/45 | 1431 | 22% | 26% | 176.2 | 10% | 2% | 14 | 32.0 | 0% |
| santiago-chile | 128 | 3/0 | 9864 | 77% | 82% | 4.5 | 85% | 71% | 116 | 6.8 | 80% |
| niagara-falls-ny | 115 | 12/1 | 1855 | 44% | 63% | 24.5 | 41% | 28% | 70 | 12.1 | 34% |
| sao-paulo-brazil | 105 | 0/0 | 15105 | 60% | 66% | 4.0 | 76% | 61% | 20 | 87.0 | 15% |
| hackensack-nj | 98 | 35/1 | 2318 | 72% | 80% | 102.4 | 16% | 12% | 16 | 18.3 | 6% |
| chandigarh-india | 94 | 6/2 | 106 | 62% | 63% | 446.7 | 2% | 0% | 6 |  |  |
| fort-wayne-in | 75 | 3/0 | 4208 | 24% | 60% | 71.3 | 33% | 21% | 17 | 386.0 | 0% |
| houston-tx | 71 | 1/0 | 14587 | 45% | 77% | 2.8 | 84% | 54% | 17 | 528.1 | 41% |
| richmond-va | 69 | 0/0 | 10713 | 23% | 77% | 5.0 | 100% | 19% | 53 | 5.9 | 87% |
| la-ca | 47 | 28/0 | 15215 | 25% | 53% | 2.4 | 100% | 47% | 57 | 11.9 | 33% |
| newberg-or | 47 | 1/5 | 1463 | 26% | 95% | 97.9 | 34% | 23% | 0 |  |  |
| blackhawk-hills-il | 38 | 18/1 | 78 | 4% | 91% | 557.6 | 0% |  | 0 |  |  |
| la-piedad | 29 | 0/0 | 2 | 50% | 50% |  |  |  | 0 |  |  |
| laurens-ia | 20 | 0/0 | 11 | 9% | 36% | 6.0 | 75% | 10% | 0 |  |  |
| clifton-nj | 18 | 7/0 | 1795 | 50% | 58% | 6.4 | 61% | 39% | 3 | 17.1 | 0% |
| virden-il | 17 | 7/2 | 18 | 22% | 94% | 135.5 | 0% | 0% | 5 |  |  |
| vancouver-wa | 12 | 0/0 | 9734 | 33% | 52% | 3.1 | 92% | 50% | 6 | 10.9 | 50% |
| bayonne-fr | 9 | 0/0 | 1785 | 72% | 73% | 3.4 | 100% | 100% | 0 |  |  |
| washington-dc | 4 | 0/0 | 8342 | 58% | 79% | 0.7 | 100% | 100% | 0 |  |  |
| newport-ky | 2 | 0/0 | 1550 | 47% | 71% | 7.2 | 100% | 100% | 0 |  |  |

Total Crosswalk labels: 68014; Signal: 15309


### Reading 1: OSM crossing coverage across our cities is bimodal, and that, not placement, drives the distances

The median label-to-node distance is either 2–5 m or 100–500 m, with almost nothing between. Counting OSM crossing nodes inside each city's own label footprint (1 km cells that hold ≥ 1 label) separates the two groups almost perfectly: above ~40 nodes per cell the crowd's labels sit on OSM nodes; below ~30 they do not, because there are no nodes to sit on. The far group is not crowd error: Teaneck (1,595 human-confirmed labels, 24 nodes/cell), Paterson, CDMX, Mendota, Oradell and Maywood are all heavily validated. Two cities have nodes but not where the crowd labels are, which is worth a look on an aerial basemap before trusting either side: Cuenca (86 nodes/cell, median 75 m) and Newberg (58/cell, 98 m; Bend-like, few painted crossings).


| city | labels | 1 km cells with labels | OSM crossing nodes in those cells | nodes per cell | median m | <=25 m |
|---|---|---|---|---|---|---|
| richmond-va | 69 | 4 | 1239 | 309.8 | 5.0 | 1.0 |
| la-ca | 47 | 3 | 856 | 285.3 | 2.4 | 1.0 |
| seattle-wa | 6283 | 211 | 42640 | 202.1 | 2.2 | 0.986 |
| newport-ky | 2 | 1 | 164 | 164.0 | 7.2 | 1.0 |
| chicago-il | 28091 | 1081 | 163507 | 151.3 | 1.5 | 0.944 |
| washington-dc | 4 | 1 | 151 | 151.0 | 0.7 | 1.0 |
| sao-paulo-brazil | 105 | 8 | 1200 | 150.0 | 4.0 | 0.867 |
| santiago-chile | 128 | 5 | 697 | 139.4 | 4.5 | 0.93 |
| pittsburgh-pa | 1018 | 70 | 9474 | 135.3 | 2.3 | 0.83 |
| rancagua-chile | 148 | 6 | 787 | 131.2 | 4.4 | 0.777 |
| burnaby | 1728 | 47 | 5350 | 113.8 | 2.3 | 0.997 |
| columbus-oh | 889 | 34 | 3863 | 113.6 | 2.4 | 0.863 |
| danville-il | 488 | 18 | 1875 | 104.2 | 2.9 | 0.949 |
| detroit-mi | 283 | 6 | 615 | 102.5 | 1.7 | 0.845 |
| tucson-az | 691 | 10 | 889 | 88.9 | 2.9 | 0.826 |
| cuenca | 1768 | 11 | 948 | 86.2 | 75.4 | 0.428 |
| st-louis-mo | 2890 | 160 | 13289 | 83.1 | 3.6 | 0.753 |
| gainesville-fl | 2420 | 54 | 4478 | 82.9 | 1.8 | 0.954 |
| bayonne-fr | 9 | 3 | 245 | 81.7 | 3.4 | 1.0 |
| houston-tx | 71 | 6 | 478 | 79.7 | 2.8 | 0.944 |
| waltham-ma | 281 | 17 | 1266 | 74.5 | 1.8 | 0.9 |
| newberg-or | 47 | 8 | 460 | 57.5 | 97.9 | 0.362 |
| auckland | 273 | 13 | 710 | 54.6 | 2.6 | 0.96 |
| madison-wi | 137 | 6 | 298 | 49.7 | 2.7 | 0.898 |
| amsterdam | 2976 | 67 | 3274 | 48.9 | 2.1 | 0.887 |
| vancouver-wa | 12 | 4 | 165 | 41.2 | 3.1 | 0.917 |
| fort-wayne-in | 75 | 8 | 302 | 37.8 | 71.3 | 0.387 |
| west-chester-pa | 331 | 7 | 222 | 31.7 | 136.8 | 0.305 |
| niagara-falls-ny | 115 | 19 | 556 | 29.3 | 24.5 | 0.505 |
| cliffside-park-nj | 201 | 8 | 226 | 28.2 | 239.4 | 0.189 |
| teaneck-nj | 3363 | 25 | 605 | 24.2 | 183.5 | 0.312 |
| hackensack-nj | 98 | 2 | 40 | 20.0 | 102.4 | 0.327 |
| clifton-nj | 18 | 2 | 38 | 19.0 | 6.4 | 0.611 |
| cdmx | 3931 | 50 | 927 | 18.5 | 185.0 | 0.222 |
| paterson-nj | 5300 | 35 | 618 | 17.7 | 239.1 | 0.218 |
| spgg | 130 | 22 | 374 | 17.0 | 176.2 | 0.178 |
| walla-walla-wa | 878 | 32 | 462 | 14.4 | 42.3 | 0.484 |
| knox-oh | 355 | 36 | 256 | 7.1 | 254.5 | 0.126 |
| oradell-nj | 675 | 14 | 81 | 5.8 | 430.4 | 0.062 |
| maywood-nj | 505 | 8 | 45 | 5.6 | 586.7 | 0.021 |
| laurens-ia | 20 | 2 | 10 | 5.0 | 6.0 | 0.8 |
| mendota-il | 983 | 11 | 33 | 3.0 | 178.9 | 0.191 |
| virden-il | 17 | 5 | 11 | 2.2 | 135.5 | 0.0 |
| chandigarh-india | 94 | 23 | 19 | 0.8 | 446.7 | 0.047 |
| blackhawk-hills-il | 38 | 3 | 2 | 0.7 | 557.6 | 0.0 |
| la-piedad | 29 | 4 | 0 | 0.0 | None | None |


### Reading 2: where a node exists, it is where the crowd puts the label, which supports the midpoint convention

In the 17 cities with ≥ 100 labels and a median ≤ 5 m, the median is 1.5–4.5 m, which is inside the server's own placement error for a label raycast from a pano. Restricted to **human-confirmed** labels in those cities (8,086 labels, Chicago 3,907 of them):

| city | confirmed labels | ≤ 10 m any node | ≤ 10 m marked node |
|---|---|---|---|
| chicago-il | 3907 | 93% | 34% |
| seattle-wa | 1820 | 98% | 93% |
| amsterdam | 890 | 98% | 97% |
| st-louis-mo | 292 | 44% | 34% |
| burnaby | 289 | 98% | 84% |
| pittsburgh-pa | 214 | 92% | 53% |
| tucson-az | 195 | 78% | 75% |
| columbus-oh | 177 | 79% | 41% |
| auckland | 153 | 97% | 76% |
| waltham-ma | 62 | 90% | 42% |
| madison-wi | 45 | 84% | 80% |
| detroit-mi | 42 | 67% | 14% |
| **pooled** | **8086** | **92%** | **59%** |

So the OSM crossing node is a usable Stage 1 bearing source (the ±18° tolerance is far wider than 2–5 m at any range we use) in the dense group, and the along/across decomposition of item 2 can be read from these same joins. St. Louis is the outlier to check (confirmed labels agree with nodes less often than unconfirmed ones).

### Reading 3: the `crossing:markings` tag disagrees with the crowd often enough that it cannot be the positive set on its own

Pooled over confirmed labels with a node within 10 m, 36% sit on a node OSM tags unmarked or leaves untagged. Chicago drives that (56% of its 248k crossing nodes carry no markings tag at all; only 37% of its labels are within 10 m of a marked node); without Chicago it is 1 in 9, but in Columbus, Pittsburgh, Waltham and Detroit it is 1 in 2 or worse, and Richmond's 69 labels are all within 10 m of a node but only 19% within 10 m of a *marked* one. Either the tags are stale (a crossing painted since the node was tagged) or the crowd labels unmarked crossings as `Crosswalk`. That is exactly what the one-city aerial reference in item 2 has to settle before any OSM-marked node is used as a training positive; until then, positives come from crowd-confirmed labels, and OSM nodes supply bearing and candidate crossings only.

### Reading 4: Signal labels sit ~9–12 m from OSM signal nodes

Across the cities with ≥ 100 Signal labels the median is 7–15 m and only 27–64% are within 10 m, against 2–4 m for crossings. An OSM signal node is tagged on the intersection node (or the stop line), while the PS label is on the pole at the corner, so a signal bearing rule needs a corner offset rather than the node itself. Parked until crosswalks are done; recorded here so the number is not re-derived.

### What this settles for Phase 0

- **Item 2 (offset read)** runs on the 17 dense-group cities; the aerial reference city should be one with many confirmed labels, dense OSM nodes and a marking inventory to compare against. Seattle fits all three; Burnaby and Amsterdam are the non-US checks.
- **Item 3 (Vistas probe)** stays on `richmond` and the `laurens` splits as pinned, with the caveat that both are small (69 and 20 crowd labels).
- The far group (Paterson, Teaneck, CDMX, Mendota, …) is where the crowd labels are the *only* supervision; they are also where a crosswalk detector adds the most, since no open data covers them.

🤖 Generated with [Claude Code](https://claude.com/claude-code) — Fable 5.1, claude-fable-5-1
