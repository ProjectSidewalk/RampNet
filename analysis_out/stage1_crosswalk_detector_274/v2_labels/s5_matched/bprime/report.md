# (b') Seattle node sample: detector vs the map-only baseline (RampNet#274)

Scored 2026-11-10T05:35:30Z by `score_274.py bprime-score`; protocol in docs/stage1_crosswalk_sdot_274.md (b'). 876 nodes, weighted to the eligible population.

| model | threshold | detector P / R / F1 / MCC | OSM P / R / F1 / MCC | OR F1 | always-yes F1 | F1 det−OSM 95% | F1 OR−OSM 95% | primary: OR beats OSM | secondary: det ≥ OSM |
|---|---|---|---|---|---|---|---|---|---|
| crosswalk_s5 | 0.26 | 0.770 / 0.676 / 0.720 / 0.628 | 0.814 / 0.898 / 0.854 / 0.799 | 0.836 | 0.424 | [-0.186, -0.087] | [-0.044, 0.005] | no | no |
| crosswalk_s4 | 0.23 | 0.794 / 0.728 / 0.759 / 0.677 | 0.814 / 0.898 / 0.854 / 0.799 | 0.845 | 0.424 | [-0.141, -0.051] | [-0.033, 0.012] | no | no |
| released | 0.10 | 0.333 / 0.869 / 0.481 / 0.221 | 0.814 / 0.898 / 0.854 / 0.799 | 0.523 | 0.424 | [-0.401, -0.346] | [-0.349, -0.312] | no | no |

## Per cell (unweighted: the share of sampled nodes each predictor calls marked)

| model | OSM class | truth | nodes | detector yes | OSM yes |
|---|---|---|---|---|---|
| crosswalk_s5 | marked | not | 150 | 0.493 | 1.000 |
| crosswalk_s5 | marked | marked | 150 | 0.707 | 1.000 |
| crosswalk_s5 | unknown | not | 150 | 0.020 | 0.000 |
| crosswalk_s5 | unknown | marked | 126 | 0.690 | 0.405 |
| crosswalk_s5 | unmarked | not | 150 | 0.047 | 0.000 |
| crosswalk_s5 | unmarked | marked | 150 | 0.340 | 0.000 |
| crosswalk_s4 | marked | not | 150 | 0.513 | 1.000 |
| crosswalk_s4 | marked | marked | 150 | 0.760 | 1.000 |
| crosswalk_s4 | unknown | not | 150 | 0.033 | 0.000 |
| crosswalk_s4 | unknown | marked | 126 | 0.722 | 0.405 |
| crosswalk_s4 | unmarked | not | 150 | 0.033 | 0.000 |
| crosswalk_s4 | unmarked | marked | 150 | 0.380 | 0.000 |
| released | marked | not | 150 | 0.773 | 1.000 |
| released | marked | marked | 150 | 0.867 | 1.000 |
| released | unknown | not | 150 | 0.240 | 0.000 |
| released | unknown | marked | 126 | 0.849 | 0.405 |
| released | unmarked | not | 150 | 0.753 | 0.000 |
| released | unmarked | marked | 150 | 0.900 | 0.000 |

## Detector recall on marked nodes, by the scored pano's capture year (unweighted)

| model | year | marked nodes | hit |
|---|---|---|---|
| crosswalk_s5 | 2019-2022 | 328 | 0.604 |
| crosswalk_s5 | 2023+ | 56 | 0.464 |
| crosswalk_s5 | <=2018 | 42 | 0.476 |
| crosswalk_s4 | 2019-2022 | 328 | 0.637 |
| crosswalk_s4 | 2023+ | 56 | 0.571 |
| crosswalk_s4 | <=2018 | 42 | 0.500 |
| released | 2019-2022 | 328 | 0.872 |
| released | 2023+ | 56 | 0.911 |
| released | <=2018 | 42 | 0.833 |
