# (b') Seattle node sample: detector vs the map-only baseline (RampNet#274)

Scored 2026-11-14T00:43:29Z by `score_274.py bprime-score`; protocol in docs/stage1_crosswalk_sdot_274.md (b'). 876 nodes, weighted to the eligible population.

| model | threshold | detector P / R / F1 / MCC | OSM P / R / F1 / MCC | OR F1 | always-yes F1 | F1 det−OSM 95% | F1 OR−OSM 95% | primary: OR beats OSM | secondary: det ≥ OSM |
|---|---|---|---|---|---|---|---|---|---|
| crosswalk_s6full | 0.17 | 0.724 / 0.736 / 0.730 / 0.630 | 0.814 / 0.898 / 0.854 / 0.799 | 0.814 | 0.424 | [-0.171, -0.077] | [-0.070, -0.012] | no | no |
| crosswalk_s6full_seed2 | 0.27 | 0.777 / 0.610 / 0.683 / 0.592 | 0.814 / 0.898 / 0.854 / 0.799 | 0.843 | 0.424 | [-0.227, -0.117] | [-0.034, 0.010] | no | no |
| crosswalk_s6full_seed3 | 0.24 | 0.761 / 0.674 / 0.715 / 0.620 | 0.814 / 0.898 / 0.854 / 0.799 | 0.830 | 0.424 | [-0.193, -0.087] | [-0.050, 0.001] | no | no |
| crosswalk_s6half | 0.23 | 0.748 / 0.680 / 0.712 / 0.614 | 0.814 / 0.898 / 0.854 / 0.799 | 0.829 | 0.424 | [-0.194, -0.093] | [-0.053, 0.000] | no | no |
| crosswalk_s6half_seed2 | 0.23 | 0.818 / 0.681 / 0.743 / 0.665 | 0.814 / 0.898 / 0.854 / 0.799 | 0.854 | 0.424 | [-0.160, -0.065] | [-0.020, 0.016] | no | no |
| crosswalk_s6half_seed3 | 0.25 | 0.772 / 0.755 / 0.764 / 0.678 | 0.814 / 0.898 / 0.854 / 0.799 | 0.839 | 0.424 | [-0.136, -0.046] | [-0.040, 0.008] | no | no |
| crosswalk_s4_steps1397 | 0.24 | 0.765 / 0.688 / 0.724 / 0.632 | 0.814 / 0.898 / 0.854 / 0.799 | 0.830 | 0.424 | [-0.181, -0.081] | [-0.050, 0.000] | no | no |
| crosswalk_s4 | 0.23 | 0.794 / 0.728 / 0.759 / 0.677 | 0.814 / 0.898 / 0.854 / 0.799 | 0.845 | 0.424 | [-0.141, -0.051] | [-0.033, 0.012] | no | no |
| crosswalk_s4_seed2 | 0.30 | 0.796 / 0.633 / 0.705 / 0.620 | 0.814 / 0.898 / 0.854 / 0.799 | 0.845 | 0.424 | [-0.203, -0.096] | [-0.032, 0.010] | no | no |
| crosswalk_s4_seed3 | 0.13 | 0.707 / 0.700 / 0.703 / 0.595 | 0.814 / 0.898 / 0.854 / 0.799 | 0.812 | 0.424 | [-0.202, -0.102] | [-0.073, -0.013] | no | no |

## Per cell (unweighted: the share of sampled nodes each predictor calls marked)

| model | OSM class | truth | nodes | detector yes | OSM yes |
|---|---|---|---|---|---|
| crosswalk_s6full | marked | not | 150 | 0.527 | 1.000 |
| crosswalk_s6full | marked | marked | 150 | 0.767 | 1.000 |
| crosswalk_s6full | unknown | not | 150 | 0.053 | 0.000 |
| crosswalk_s6full | unknown | marked | 126 | 0.738 | 0.405 |
| crosswalk_s6full | unmarked | not | 150 | 0.073 | 0.000 |
| crosswalk_s6full | unmarked | marked | 150 | 0.400 | 0.000 |
| crosswalk_s6full_seed2 | marked | not | 150 | 0.467 | 1.000 |
| crosswalk_s6full_seed2 | marked | marked | 150 | 0.633 | 1.000 |
| crosswalk_s6full_seed2 | unknown | not | 150 | 0.027 | 0.000 |
| crosswalk_s6full_seed2 | unknown | marked | 126 | 0.635 | 0.405 |
| crosswalk_s6full_seed2 | unmarked | not | 150 | 0.033 | 0.000 |
| crosswalk_s6full_seed2 | unmarked | marked | 150 | 0.340 | 0.000 |
| crosswalk_s6full_seed3 | marked | not | 150 | 0.440 | 1.000 |
| crosswalk_s6full_seed3 | marked | marked | 150 | 0.700 | 1.000 |
| crosswalk_s6full_seed3 | unknown | not | 150 | 0.033 | 0.000 |
| crosswalk_s6full_seed3 | unknown | marked | 126 | 0.690 | 0.405 |
| crosswalk_s6full_seed3 | unmarked | not | 150 | 0.053 | 0.000 |
| crosswalk_s6full_seed3 | unmarked | marked | 150 | 0.380 | 0.000 |
| crosswalk_s6half | marked | not | 150 | 0.507 | 1.000 |
| crosswalk_s6half | marked | marked | 150 | 0.707 | 1.000 |
| crosswalk_s6half | unknown | not | 150 | 0.040 | 0.000 |
| crosswalk_s6half | unknown | marked | 126 | 0.690 | 0.405 |
| crosswalk_s6half | unmarked | not | 150 | 0.053 | 0.000 |
| crosswalk_s6half | unmarked | marked | 150 | 0.387 | 0.000 |
| crosswalk_s6half_seed2 | marked | not | 150 | 0.473 | 1.000 |
| crosswalk_s6half_seed2 | marked | marked | 150 | 0.713 | 1.000 |
| crosswalk_s6half_seed2 | unknown | not | 150 | 0.027 | 0.000 |
| crosswalk_s6half_seed2 | unknown | marked | 126 | 0.619 | 0.405 |
| crosswalk_s6half_seed2 | unmarked | not | 150 | 0.020 | 0.000 |
| crosswalk_s6half_seed2 | unmarked | marked | 150 | 0.353 | 0.000 |
| crosswalk_s6half_seed3 | marked | not | 150 | 0.573 | 1.000 |
| crosswalk_s6half_seed3 | marked | marked | 150 | 0.787 | 1.000 |
| crosswalk_s6half_seed3 | unknown | not | 150 | 0.027 | 0.000 |
| crosswalk_s6half_seed3 | unknown | marked | 126 | 0.754 | 0.405 |
| crosswalk_s6half_seed3 | unmarked | not | 150 | 0.047 | 0.000 |
| crosswalk_s6half_seed3 | unmarked | marked | 150 | 0.413 | 0.000 |
| crosswalk_s4_steps1397 | marked | not | 150 | 0.460 | 1.000 |
| crosswalk_s4_steps1397 | marked | marked | 150 | 0.720 | 1.000 |
| crosswalk_s4_steps1397 | unknown | not | 150 | 0.047 | 0.000 |
| crosswalk_s4_steps1397 | unknown | marked | 126 | 0.667 | 0.405 |
| crosswalk_s4_steps1397 | unmarked | not | 150 | 0.047 | 0.000 |
| crosswalk_s4_steps1397 | unmarked | marked | 150 | 0.347 | 0.000 |
| crosswalk_s4 | marked | not | 150 | 0.513 | 1.000 |
| crosswalk_s4 | marked | marked | 150 | 0.760 | 1.000 |
| crosswalk_s4 | unknown | not | 150 | 0.033 | 0.000 |
| crosswalk_s4 | unknown | marked | 126 | 0.722 | 0.405 |
| crosswalk_s4 | unmarked | not | 150 | 0.033 | 0.000 |
| crosswalk_s4 | unmarked | marked | 150 | 0.380 | 0.000 |
| crosswalk_s4_seed2 | marked | not | 150 | 0.420 | 1.000 |
| crosswalk_s4_seed2 | marked | marked | 150 | 0.660 | 1.000 |
| crosswalk_s4_seed2 | unknown | not | 150 | 0.020 | 0.000 |
| crosswalk_s4_seed2 | unknown | marked | 126 | 0.635 | 0.405 |
| crosswalk_s4_seed2 | unmarked | not | 150 | 0.033 | 0.000 |
| crosswalk_s4_seed2 | unmarked | marked | 150 | 0.333 | 0.000 |
| crosswalk_s4_seed3 | marked | not | 150 | 0.553 | 1.000 |
| crosswalk_s4_seed3 | marked | marked | 150 | 0.727 | 1.000 |
| crosswalk_s4_seed3 | unknown | not | 150 | 0.040 | 0.000 |
| crosswalk_s4_seed3 | unknown | marked | 126 | 0.730 | 0.405 |
| crosswalk_s4_seed3 | unmarked | not | 150 | 0.080 | 0.000 |
| crosswalk_s4_seed3 | unmarked | marked | 150 | 0.393 | 0.000 |

## Detector recall on marked nodes, by the scored pano's capture year (unweighted)

| model | year | marked nodes | hit |
|---|---|---|---|
| crosswalk_s6full | 2019-2022 | 328 | 0.659 |
| crosswalk_s6full | 2023+ | 56 | 0.554 |
| crosswalk_s6full | <=2018 | 42 | 0.500 |
| crosswalk_s6full_seed2 | 2019-2022 | 328 | 0.558 |
| crosswalk_s6full_seed2 | 2023+ | 56 | 0.482 |
| crosswalk_s6full_seed2 | <=2018 | 42 | 0.381 |
| crosswalk_s6full_seed3 | 2019-2022 | 328 | 0.619 |
| crosswalk_s6full_seed3 | 2023+ | 56 | 0.500 |
| crosswalk_s6full_seed3 | <=2018 | 42 | 0.429 |
| crosswalk_s6half | 2019-2022 | 328 | 0.628 |
| crosswalk_s6half | 2023+ | 56 | 0.500 |
| crosswalk_s6half | <=2018 | 42 | 0.405 |
| crosswalk_s6half_seed2 | 2019-2022 | 328 | 0.595 |
| crosswalk_s6half_seed2 | 2023+ | 56 | 0.500 |
| crosswalk_s6half_seed2 | <=2018 | 42 | 0.357 |
| crosswalk_s6half_seed3 | 2019-2022 | 328 | 0.671 |
| crosswalk_s6half_seed3 | 2023+ | 56 | 0.571 |
| crosswalk_s6half_seed3 | <=2018 | 42 | 0.548 |
| crosswalk_s4_steps1397 | 2019-2022 | 328 | 0.604 |
| crosswalk_s4_steps1397 | 2023+ | 56 | 0.482 |
| crosswalk_s4_steps1397 | <=2018 | 42 | 0.452 |
| crosswalk_s4 | 2019-2022 | 328 | 0.637 |
| crosswalk_s4 | 2023+ | 56 | 0.571 |
| crosswalk_s4 | <=2018 | 42 | 0.500 |
| crosswalk_s4_seed2 | 2019-2022 | 328 | 0.576 |
| crosswalk_s4_seed2 | 2023+ | 56 | 0.429 |
| crosswalk_s4_seed2 | <=2018 | 42 | 0.381 |
| crosswalk_s4_seed3 | 2019-2022 | 328 | 0.625 |
| crosswalk_s4_seed3 | 2023+ | 56 | 0.571 |
| crosswalk_s4_seed3 | <=2018 | 42 | 0.548 |
