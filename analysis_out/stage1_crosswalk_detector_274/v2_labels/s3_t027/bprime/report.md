# (b') Seattle node sample: detector vs the map-only baseline (RampNet#274)

Scored 2026-11-08T15:09:17Z by `score_274.py bprime-score`; protocol in docs/stage1_crosswalk_sdot_274.md (b'). 876 nodes, weighted to the eligible population.

| model | threshold | detector P / R / F1 / MCC | OSM P / R / F1 / MCC | OR F1 | always-yes F1 | F1 det−OSM 95% | F1 OR−OSM 95% | primary: OR beats OSM | secondary: det ≥ OSM |
|---|---|---|---|---|---|---|---|---|---|
| crosswalk_s3 | 0.27 | 0.763 / 0.750 / 0.756 / 0.668 | 0.814 / 0.898 / 0.854 / 0.799 | 0.834 | 0.424 | [-0.145, -0.055] | [-0.048, 0.005] | no | no |
| crosswalk_s1 | 0.21 | 0.712 / 0.720 / 0.716 / 0.611 | 0.814 / 0.898 / 0.854 / 0.799 | 0.812 | 0.424 | [-0.188, -0.091] | [-0.072, -0.015] | no | no |

## Per cell (unweighted: the share of sampled nodes each predictor calls marked)

| model | OSM class | truth | nodes | detector yes | OSM yes |
|---|---|---|---|---|---|
| crosswalk_s3 | marked | not | 150 | 0.560 | 1.000 |
| crosswalk_s3 | marked | marked | 150 | 0.780 | 1.000 |
| crosswalk_s3 | unknown | not | 150 | 0.027 | 0.000 |
| crosswalk_s3 | unknown | marked | 126 | 0.714 | 0.405 |
| crosswalk_s3 | unmarked | not | 150 | 0.053 | 0.000 |
| crosswalk_s3 | unmarked | marked | 150 | 0.433 | 0.000 |
| crosswalk_s1 | marked | not | 150 | 0.540 | 1.000 |
| crosswalk_s1 | marked | marked | 150 | 0.747 | 1.000 |
| crosswalk_s1 | unknown | not | 150 | 0.067 | 0.000 |
| crosswalk_s1 | unknown | marked | 126 | 0.722 | 0.405 |
| crosswalk_s1 | unmarked | not | 150 | 0.073 | 0.000 |
| crosswalk_s1 | unmarked | marked | 150 | 0.427 | 0.000 |

## Detector recall on marked nodes, by the scored pano's capture year (unweighted)

| model | year | marked nodes | hit |
|---|---|---|---|
| crosswalk_s3 | 2019-2022 | 328 | 0.677 |
| crosswalk_s3 | 2023+ | 56 | 0.518 |
| crosswalk_s3 | <=2018 | 42 | 0.500 |
| crosswalk_s1 | 2019-2022 | 328 | 0.659 |
| crosswalk_s1 | 2023+ | 56 | 0.536 |
| crosswalk_s1 | <=2018 | 42 | 0.500 |
