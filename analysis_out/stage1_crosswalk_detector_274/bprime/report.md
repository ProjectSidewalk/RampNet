# (b') Seattle node sample: detector vs the map-only baseline (RampNet#274)

Scored 2026-11-05T14:52:55Z by `score_274.py bprime-score`; protocol in docs/stage1_crosswalk_sdot_274.md (b'). 876 nodes, weighted to the eligible population.

| model | threshold | detector P / R / F1 / MCC | OSM P / R / F1 / MCC | OR F1 | always-yes F1 | F1 det−OSM 95% | F1 OR−OSM 95% | primary: OR beats OSM | secondary: det ≥ OSM |
|---|---|---|---|---|---|---|---|---|---|
| crosswalk_s1 | 0.31 | 0.745 / 0.636 / 0.686 / 0.586 | 0.814 / 0.898 / 0.854 / 0.799 | 0.823 | 0.424 | [-0.224, -0.115] | [-0.059, -0.004] | no | no |
| released | 0.05 | 0.313 / 0.888 / 0.463 / 0.179 | 0.814 / 0.898 / 0.854 / 0.799 | 0.498 | 0.424 | [-0.414, -0.367] | [-0.371, -0.340] | no | no |

## Per cell (unweighted: the share of sampled nodes each predictor calls marked)

| model | OSM class | truth | nodes | detector yes | OSM yes |
|---|---|---|---|---|---|
| crosswalk_s1 | marked | not | 150 | 0.387 | 1.000 |
| crosswalk_s1 | marked | marked | 150 | 0.660 | 1.000 |
| crosswalk_s1 | unknown | not | 150 | 0.040 | 0.000 |
| crosswalk_s1 | unknown | marked | 126 | 0.627 | 0.405 |
| crosswalk_s1 | unmarked | not | 150 | 0.060 | 0.000 |
| crosswalk_s1 | unmarked | marked | 150 | 0.373 | 0.000 |
| released | marked | not | 150 | 0.813 | 1.000 |
| released | marked | marked | 150 | 0.887 | 1.000 |
| released | unknown | not | 150 | 0.353 | 0.000 |
| released | unknown | marked | 126 | 0.865 | 0.405 |
| released | unmarked | not | 150 | 0.820 | 0.000 |
| released | unmarked | marked | 150 | 0.913 | 0.000 |

## Detector recall on marked nodes, by the scored pano's capture year (unweighted)

| model | year | marked nodes | hit |
|---|---|---|---|
| crosswalk_s1 | 2019-2022 | 328 | 0.585 |
| crosswalk_s1 | 2023+ | 56 | 0.411 |
| crosswalk_s1 | <=2018 | 42 | 0.452 |
| released | 2019-2022 | 328 | 0.890 |
| released | 2023+ | 56 | 0.911 |
| released | <=2018 | 42 | 0.857 |
