**Stage 0a triggers, 8-split pool (US_SPLITS minus juneau), merged pairs at 0.30**

| metric | pairs | under 8 px | share | arm C (> 1/3) | >= 12 px outside paterson | arm A (>= 30) | >= 12 px, all | 8 to 12 px |
|---|---|---|---|---|---|---|---|---|
| chebyshev | 136 | 25 | 0.184 | not triggered | 38 | RUNS | 62 | 49 |
| euclidean | 136 | 17 | 0.125 | not triggered | 44 | RUNS | 73 | 46 |

**Context: the seven-split pool S7 (cross-checked against `analysis_out/miss_taxonomy.json`)**

| metric | pairs | under 8 px | share | >= 12 px outside paterson |
|---|---|---|---|---|
| chebyshev | 124 | 24 | 0.194 | 31 |
| euclidean | 124 | 16 | 0.129 | 35 |

**Per split**

| split | merged misses | pairs | under 8 px (cheb / euc) | >= 12 px (cheb / euc) |
|---|---|---|---|---|
| richmond | 10 | 10 | 0 / 0 | 10 / 10 |
| annapolis | 5 | 5 | 0 / 0 | 3 / 4 |
| clovis | 1 | 1 | 0 / 0 | 1 / 1 |
| morgantown | 9 | 9 | 2 / 1 | 4 / 5 |
| bend | 14 | 14 | 2 / 1 | 9 / 11 |
| paterson | 80 | 80 | 20 / 14 | 24 / 29 |
| gainesville | 5 | 5 | 0 / 0 | 4 / 4 |
| laurens_mapillary | 12 | 12 | 1 / 1 | 7 / 9 |

**Histogram, 1 px bins (pairs per bin)**

| px | chebyshev | euclidean |
|---|---|---|
| 0-1 | 0 | 0 |
| 1-2 | 0 | 0 |
| 2-3 | 0 | 0 |
| 3-4 | 1 | 0 |
| 4-5 | 2 | 1 |
| 5-6 | 5 | 5 |
| 6-7 | 6 | 3 |
| 7-8 | 11 | 8 |
| 8-9 | 12 | 12 |
| 9-10 | 11 | 7 |
| 10-11 | 17 | 14 |
| 11-12 | 9 | 13 |
| 12-13 | 7 | 9 |
| 13-14 | 6 | 6 |
| 14-15 | 10 | 9 |
| 15-16 | 6 | 7 |
| 16-17 | 6 | 7 |
| 17-18 | 10 | 4 |
| 18-19 | 4 | 7 |
| 19-20 | 3 | 6 |
| 20-21 | 4 | 8 |
| 21-22 | 2 | 4 |
| 22-23 | 2 | 3 |
| 23-24 | 1 | 1 |
| 24-25 | 0 | 1 |
| 25-26 | 1 | 1 |
| 26-27 | 0 | 0 |

