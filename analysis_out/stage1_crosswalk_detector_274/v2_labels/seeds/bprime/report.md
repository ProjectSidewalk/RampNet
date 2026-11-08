# (b') Seattle node sample: detector vs the map-only baseline (RampNet#274)

Scored 2026-11-08T21:59:31Z by `score_274.py bprime-score`; protocol in docs/stage1_crosswalk_sdot_274.md (b'). 876 nodes, weighted to the eligible population.

| model | threshold | detector P / R / F1 / MCC | OSM P / R / F1 / MCC | OR F1 | always-yes F1 | F1 det−OSM 95% | F1 OR−OSM 95% | primary: OR beats OSM | secondary: det ≥ OSM |
|---|---|---|---|---|---|---|---|---|---|
| crosswalk_s1 | 0.21 | 0.712 / 0.720 / 0.716 / 0.611 | 0.814 / 0.898 / 0.854 / 0.799 | 0.812 | 0.424 | [-0.188, -0.091] | [-0.072, -0.015] | no | no |
| crosswalk_s3 | 0.29 | 0.767 / 0.736 / 0.752 / 0.663 | 0.814 / 0.898 / 0.854 / 0.799 | 0.835 | 0.424 | [-0.152, -0.060] | [-0.047, 0.005] | no | no |
| crosswalk_s1_seed2 | 0.24 | 0.829 / 0.653 / 0.731 / 0.655 | 0.814 / 0.898 / 0.854 / 0.799 | 0.859 | 0.424 | [-0.171, -0.077] | [-0.011, 0.018] | no | no |
| crosswalk_s1_seed3 | 0.22 | 0.799 / 0.689 / 0.740 / 0.657 | 0.814 / 0.898 / 0.854 / 0.799 | 0.845 | 0.424 | [-0.163, -0.066] | [-0.032, 0.012] | no | no |
| crosswalk_s3_seed2 | 0.39 | 0.826 / 0.690 / 0.752 / 0.676 | 0.814 / 0.898 / 0.854 / 0.799 | 0.856 | 0.424 | [-0.150, -0.055] | [-0.017, 0.018] | no | no |
| crosswalk_s3_seed3 | 0.31 | 0.793 / 0.639 / 0.708 / 0.622 | 0.814 / 0.898 / 0.854 / 0.799 | 0.844 | 0.424 | [-0.201, -0.094] | [-0.032, 0.010] | no | no |

## Per cell (unweighted: the share of sampled nodes each predictor calls marked)

| model | OSM class | truth | nodes | detector yes | OSM yes |
|---|---|---|---|---|---|
| crosswalk_s1 | marked | not | 150 | 0.540 | 1.000 |
| crosswalk_s1 | marked | marked | 150 | 0.747 | 1.000 |
| crosswalk_s1 | unknown | not | 150 | 0.067 | 0.000 |
| crosswalk_s1 | unknown | marked | 126 | 0.722 | 0.405 |
| crosswalk_s1 | unmarked | not | 150 | 0.073 | 0.000 |
| crosswalk_s1 | unmarked | marked | 150 | 0.427 | 0.000 |
| crosswalk_s3 | marked | not | 150 | 0.533 | 1.000 |
| crosswalk_s3 | marked | marked | 150 | 0.767 | 1.000 |
| crosswalk_s3 | unknown | not | 150 | 0.020 | 0.000 |
| crosswalk_s3 | unknown | marked | 126 | 0.706 | 0.405 |
| crosswalk_s3 | unmarked | not | 150 | 0.053 | 0.000 |
| crosswalk_s3 | unmarked | marked | 150 | 0.420 | 0.000 |
| crosswalk_s1_seed2 | marked | not | 150 | 0.453 | 1.000 |
| crosswalk_s1_seed2 | marked | marked | 150 | 0.680 | 1.000 |
| crosswalk_s1_seed2 | unknown | not | 150 | 0.027 | 0.000 |
| crosswalk_s1_seed2 | unknown | marked | 126 | 0.667 | 0.405 |
| crosswalk_s1_seed2 | unmarked | not | 150 | 0.013 | 0.000 |
| crosswalk_s1_seed2 | unmarked | marked | 150 | 0.353 | 0.000 |
| crosswalk_s1_seed3 | marked | not | 150 | 0.460 | 1.000 |
| crosswalk_s1_seed3 | marked | marked | 150 | 0.720 | 1.000 |
| crosswalk_s1_seed3 | unknown | not | 150 | 0.027 | 0.000 |
| crosswalk_s1_seed3 | unknown | marked | 126 | 0.651 | 0.405 |
| crosswalk_s1_seed3 | unmarked | not | 150 | 0.033 | 0.000 |
| crosswalk_s1_seed3 | unmarked | marked | 150 | 0.373 | 0.000 |
| crosswalk_s3_seed2 | marked | not | 150 | 0.467 | 1.000 |
| crosswalk_s3_seed2 | marked | marked | 150 | 0.720 | 1.000 |
| crosswalk_s3_seed2 | unknown | not | 150 | 0.020 | 0.000 |
| crosswalk_s3_seed2 | unknown | marked | 126 | 0.659 | 0.405 |
| crosswalk_s3_seed2 | unmarked | not | 150 | 0.020 | 0.000 |
| crosswalk_s3_seed2 | unmarked | marked | 150 | 0.380 | 0.000 |
| crosswalk_s3_seed3 | marked | not | 150 | 0.427 | 1.000 |
| crosswalk_s3_seed3 | marked | marked | 150 | 0.667 | 1.000 |
| crosswalk_s3_seed3 | unknown | not | 150 | 0.027 | 0.000 |
| crosswalk_s3_seed3 | unknown | marked | 126 | 0.611 | 0.405 |
| crosswalk_s3_seed3 | unmarked | not | 150 | 0.033 | 0.000 |
| crosswalk_s3_seed3 | unmarked | marked | 150 | 0.353 | 0.000 |

## Detector recall on marked nodes, by the scored pano's capture year (unweighted)

| model | year | marked nodes | hit |
|---|---|---|---|
| crosswalk_s1 | 2019-2022 | 328 | 0.659 |
| crosswalk_s1 | 2023+ | 56 | 0.536 |
| crosswalk_s1 | <=2018 | 42 | 0.500 |
| crosswalk_s3 | 2019-2022 | 328 | 0.665 |
| crosswalk_s3 | 2023+ | 56 | 0.518 |
| crosswalk_s3 | <=2018 | 42 | 0.476 |
| crosswalk_s1_seed2 | 2019-2022 | 328 | 0.598 |
| crosswalk_s1_seed2 | 2023+ | 56 | 0.482 |
| crosswalk_s1_seed2 | <=2018 | 42 | 0.381 |
| crosswalk_s1_seed3 | 2019-2022 | 328 | 0.607 |
| crosswalk_s1_seed3 | 2023+ | 56 | 0.518 |
| crosswalk_s1_seed3 | <=2018 | 42 | 0.429 |
| crosswalk_s3_seed2 | 2019-2022 | 328 | 0.613 |
| crosswalk_s3_seed2 | 2023+ | 56 | 0.500 |
| crosswalk_s3_seed2 | <=2018 | 42 | 0.452 |
| crosswalk_s3_seed3 | 2019-2022 | 328 | 0.564 |
| crosswalk_s3_seed3 | 2023+ | 56 | 0.500 |
| crosswalk_s3_seed3 | <=2018 | 42 | 0.405 |
