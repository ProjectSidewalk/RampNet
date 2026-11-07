# (b') Seattle node sample: detector vs the map-only baseline (RampNet#274)

Scored 2026-11-07T14:07:32Z by `score_274.py bprime-score`; protocol in docs/stage1_crosswalk_sdot_274.md (b'). 876 nodes, weighted to the eligible population.

| model | threshold | detector P / R / F1 / MCC | OSM P / R / F1 / MCC | OR F1 | always-yes F1 | F1 det−OSM 95% | F1 OR−OSM 95% | primary: OR beats OSM | secondary: det ≥ OSM |
|---|---|---|---|---|---|---|---|---|---|
| crosswalk_s3 | 0.29 | 0.767 / 0.736 / 0.752 / 0.663 | 0.814 / 0.898 / 0.854 / 0.799 | 0.835 | 0.424 | [-0.152, -0.060] | [-0.047, 0.005] | no | no |
| crosswalk_s2 | 0.31 | 0.789 / 0.674 / 0.727 / 0.641 | 0.814 / 0.898 / 0.854 / 0.799 | 0.841 | 0.424 | [-0.179, -0.079] | [-0.037, 0.009] | no | no |
| crosswalk_s1 | 0.21 | 0.712 / 0.720 / 0.716 / 0.611 | 0.814 / 0.898 / 0.854 / 0.799 | 0.812 | 0.424 | [-0.188, -0.091] | [-0.072, -0.015] | no | no |
| released | 0.10 | 0.333 / 0.869 / 0.481 / 0.221 | 0.814 / 0.898 / 0.854 / 0.799 | 0.523 | 0.424 | [-0.401, -0.346] | [-0.349, -0.312] | no | no |

## Per cell (unweighted: the share of sampled nodes each predictor calls marked)

| model | OSM class | truth | nodes | detector yes | OSM yes |
|---|---|---|---|---|---|
| crosswalk_s3 | marked | not | 150 | 0.533 | 1.000 |
| crosswalk_s3 | marked | marked | 150 | 0.767 | 1.000 |
| crosswalk_s3 | unknown | not | 150 | 0.020 | 0.000 |
| crosswalk_s3 | unknown | marked | 126 | 0.706 | 0.405 |
| crosswalk_s3 | unmarked | not | 150 | 0.053 | 0.000 |
| crosswalk_s3 | unmarked | marked | 150 | 0.420 | 0.000 |
| crosswalk_s2 | marked | not | 150 | 0.427 | 1.000 |
| crosswalk_s2 | marked | marked | 150 | 0.700 | 1.000 |
| crosswalk_s2 | unknown | not | 150 | 0.027 | 0.000 |
| crosswalk_s2 | unknown | marked | 126 | 0.651 | 0.405 |
| crosswalk_s2 | unmarked | not | 150 | 0.040 | 0.000 |
| crosswalk_s2 | unmarked | marked | 150 | 0.400 | 0.000 |
| crosswalk_s1 | marked | not | 150 | 0.540 | 1.000 |
| crosswalk_s1 | marked | marked | 150 | 0.747 | 1.000 |
| crosswalk_s1 | unknown | not | 150 | 0.067 | 0.000 |
| crosswalk_s1 | unknown | marked | 126 | 0.722 | 0.405 |
| crosswalk_s1 | unmarked | not | 150 | 0.073 | 0.000 |
| crosswalk_s1 | unmarked | marked | 150 | 0.427 | 0.000 |
| released | marked | not | 150 | 0.773 | 1.000 |
| released | marked | marked | 150 | 0.867 | 1.000 |
| released | unknown | not | 150 | 0.240 | 0.000 |
| released | unknown | marked | 126 | 0.849 | 0.405 |
| released | unmarked | not | 150 | 0.753 | 0.000 |
| released | unmarked | marked | 150 | 0.900 | 0.000 |

## Detector recall on marked nodes, by the scored pano's capture year (unweighted)

| model | year | marked nodes | hit |
|---|---|---|---|
| crosswalk_s3 | 2019-2022 | 328 | 0.665 |
| crosswalk_s3 | 2023+ | 56 | 0.518 |
| crosswalk_s3 | <=2018 | 42 | 0.476 |
| crosswalk_s2 | 2019-2022 | 328 | 0.613 |
| crosswalk_s2 | 2023+ | 56 | 0.500 |
| crosswalk_s2 | <=2018 | 42 | 0.429 |
| crosswalk_s1 | 2019-2022 | 328 | 0.659 |
| crosswalk_s1 | 2023+ | 56 | 0.536 |
| crosswalk_s1 | <=2018 | 42 | 0.500 |
| released | 2019-2022 | 328 | 0.872 |
| released | 2023+ | 56 | 0.911 |
| released | <=2018 | 42 | 0.833 |
