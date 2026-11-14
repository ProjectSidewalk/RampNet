# First crosswalk detector: pre-registered scores (RampNet#274)

Scored 2026-11-14T00:43:29Z by `score_274.py score`. Protocol: `docs/stage1_crosswalk_sdot_274.md`.

## Models

| label | sha256 | val-F1 threshold |
|---|---|---|
| crosswalk_s6full | `08cbfabfc99d638d…` | 0.17 |
| crosswalk_s6full_seed2 | `14817415cb7a2133…` | 0.27 |
| crosswalk_s6full_seed3 | `93108fbd7fd407ec…` | 0.24 |
| crosswalk_s6half | `821970cf726acdf0…` | 0.23 |
| crosswalk_s6half_seed2 | `e23360b1ee3313d3…` | 0.23 |
| crosswalk_s6half_seed3 | `eca55ffdacb23d00…` | 0.25 |
| crosswalk_s4_steps1397 | `03937938482494a8…` | 0.24 |
| crosswalk_s4 | `ebad379b5f7c1645…` | 0.23 |
| crosswalk_s4_seed2 | `48a9928f5bda772b…` | 0.30 |
| crosswalk_s4_seed3 | `6f143e48f2f9d413…` | 0.13 |
| released | `024a987cabed7cc9…` | 0.10 |
| label_ceiling | `n/a…` | 0.05 |

## (a) Pixels, test split, at each model's own val-F1 threshold

Precision over complete panos (dense_ignore + verified_empty); recall over every test positive. 95% Wilson intervals.

| model | precision | recall | F1 | TP (complete) | FP | ignored | positives |
|---|---|---|---|---|---|---|---|
| crosswalk_s6full | 0.846 [0.816, 0.871] | 0.834 [0.812, 0.855] | 0.840 | 963 (548) | 100 | 123 | 1154 |
| crosswalk_s6full_seed2 | 0.870 [0.840, 0.894] | 0.810 [0.787, 0.832] | 0.839 | 935 (527) | 79 | 101 | 1154 |
| crosswalk_s6full_seed3 | 0.834 [0.803, 0.860] | 0.828 [0.806, 0.849] | 0.831 | 956 (551) | 110 | 120 | 1154 |
| crosswalk_s6half | 0.847 [0.817, 0.872] | 0.831 [0.808, 0.852] | 0.839 | 959 (553) | 100 | 120 | 1154 |
| crosswalk_s6half_seed2 | 0.861 [0.832, 0.886] | 0.809 [0.786, 0.831] | 0.834 | 934 (533) | 86 | 112 | 1154 |
| crosswalk_s6half_seed3 | 0.843 [0.814, 0.869] | 0.855 [0.834, 0.874] | 0.849 | 987 (571) | 106 | 148 | 1154 |
| crosswalk_s4_steps1397 | 0.855 [0.825, 0.880] | 0.832 [0.809, 0.852] | 0.843 | 960 (547) | 93 | 117 | 1154 |
| crosswalk_s4 | 0.815 [0.784, 0.842] | 0.846 [0.824, 0.865] | 0.830 | 976 (564) | 128 | 127 | 1154 |
| crosswalk_s4_seed2 | 0.879 [0.850, 0.903] | 0.805 [0.781, 0.827] | 0.840 | 929 (530) | 73 | 99 | 1154 |
| crosswalk_s4_seed3 | 0.825 [0.794, 0.851] | 0.850 [0.828, 0.870] | 0.837 | 981 (564) | 120 | 128 | 1154 |
| released | 0.220 [0.197, 0.245] | 0.370 [0.343, 0.398] | 0.276 | 427 (250) | 888 | 258 | 1154 |
| label_ceiling | 1.000 [0.994, 1.000] | 1.000 [0.997, 1.000] | 1.000 | 1154 (671) | 0 | 0 | 1154 |

### Test, per city

| model | group | panos | precision | recall | F1 | FP | positives |
|---|---|---|---|---|---|---|---|
| crosswalk_s6full | amsterdam | 64 | n/a | 0.870 | n/a | 0 | 131 |
| crosswalk_s6full | burnaby | 49 | n/a | 0.874 | n/a | 0 | 87 |
| crosswalk_s6full | seattle-wa | 379 | 0.846 | 0.825 | 0.835 | 100 | 899 |
| crosswalk_s6full | teaneck-nj | 23 | n/a | 0.838 | n/a | 0 | 37 |
| crosswalk_s6full_seed2 | amsterdam | 64 | n/a | 0.855 | n/a | 0 | 131 |
| crosswalk_s6full_seed2 | burnaby | 49 | n/a | 0.908 | n/a | 0 | 87 |
| crosswalk_s6full_seed2 | seattle-wa | 379 | 0.870 | 0.794 | 0.830 | 79 | 899 |
| crosswalk_s6full_seed2 | teaneck-nj | 23 | n/a | 0.811 | n/a | 0 | 37 |
| crosswalk_s6full_seed3 | amsterdam | 64 | n/a | 0.840 | n/a | 0 | 131 |
| crosswalk_s6full_seed3 | burnaby | 49 | n/a | 0.851 | n/a | 0 | 87 |
| crosswalk_s6full_seed3 | seattle-wa | 379 | 0.834 | 0.828 | 0.831 | 110 | 899 |
| crosswalk_s6full_seed3 | teaneck-nj | 23 | n/a | 0.757 | n/a | 0 | 37 |
| crosswalk_s6half | amsterdam | 64 | n/a | 0.832 | n/a | 0 | 131 |
| crosswalk_s6half | burnaby | 49 | n/a | 0.885 | n/a | 0 | 87 |
| crosswalk_s6half | seattle-wa | 379 | 0.847 | 0.829 | 0.838 | 100 | 899 |
| crosswalk_s6half | teaneck-nj | 23 | n/a | 0.757 | n/a | 0 | 37 |
| crosswalk_s6half_seed2 | amsterdam | 64 | n/a | 0.863 | n/a | 0 | 131 |
| crosswalk_s6half_seed2 | burnaby | 49 | n/a | 0.828 | n/a | 0 | 87 |
| crosswalk_s6half_seed2 | seattle-wa | 379 | 0.861 | 0.802 | 0.830 | 86 | 899 |
| crosswalk_s6half_seed2 | teaneck-nj | 23 | n/a | 0.757 | n/a | 0 | 37 |
| crosswalk_s6half_seed3 | amsterdam | 64 | n/a | 0.870 | n/a | 0 | 131 |
| crosswalk_s6half_seed3 | burnaby | 49 | n/a | 0.885 | n/a | 0 | 87 |
| crosswalk_s6half_seed3 | seattle-wa | 379 | 0.843 | 0.851 | 0.847 | 106 | 899 |
| crosswalk_s6half_seed3 | teaneck-nj | 23 | n/a | 0.838 | n/a | 0 | 37 |
| crosswalk_s4_steps1397 | amsterdam | 64 | n/a | 0.855 | n/a | 0 | 131 |
| crosswalk_s4_steps1397 | burnaby | 49 | n/a | 0.897 | n/a | 0 | 87 |
| crosswalk_s4_steps1397 | seattle-wa | 379 | 0.855 | 0.823 | 0.839 | 93 | 899 |
| crosswalk_s4_steps1397 | teaneck-nj | 23 | n/a | 0.811 | n/a | 0 | 37 |
| crosswalk_s4 | amsterdam | 64 | n/a | 0.863 | n/a | 0 | 131 |
| crosswalk_s4 | burnaby | 49 | n/a | 0.874 | n/a | 0 | 87 |
| crosswalk_s4 | seattle-wa | 379 | 0.815 | 0.843 | 0.829 | 128 | 899 |
| crosswalk_s4 | teaneck-nj | 23 | n/a | 0.784 | n/a | 0 | 37 |
| crosswalk_s4_seed2 | amsterdam | 64 | n/a | 0.863 | n/a | 0 | 131 |
| crosswalk_s4_seed2 | burnaby | 49 | n/a | 0.885 | n/a | 0 | 87 |
| crosswalk_s4_seed2 | seattle-wa | 379 | 0.879 | 0.789 | 0.831 | 73 | 899 |
| crosswalk_s4_seed2 | teaneck-nj | 23 | n/a | 0.811 | n/a | 0 | 37 |
| crosswalk_s4_seed3 | amsterdam | 64 | n/a | 0.847 | n/a | 0 | 131 |
| crosswalk_s4_seed3 | burnaby | 49 | n/a | 0.862 | n/a | 0 | 87 |
| crosswalk_s4_seed3 | seattle-wa | 379 | 0.825 | 0.849 | 0.836 | 120 | 899 |
| crosswalk_s4_seed3 | teaneck-nj | 23 | n/a | 0.865 | n/a | 0 | 37 |
| released | amsterdam | 64 | n/a | 0.435 | n/a | 0 | 131 |
| released | burnaby | 49 | n/a | 0.195 | n/a | 0 | 87 |
| released | seattle-wa | 379 | 0.220 | 0.389 | 0.281 | 888 | 899 |
| released | teaneck-nj | 23 | n/a | 0.081 | n/a | 0 | 37 |
| label_ceiling | amsterdam | 64 | n/a | 1.000 | n/a | 0 | 131 |
| label_ceiling | burnaby | 49 | n/a | 1.000 | n/a | 0 | 87 |
| label_ceiling | seattle-wa | 379 | 1.000 | 1.000 | 1.000 | 0 | 899 |
| label_ceiling | teaneck-nj | 23 | n/a | 1.000 | n/a | 0 | 37 |

### Test, per supervision mode

| model | group | panos | precision | recall | F1 | FP | positives |
|---|---|---|---|---|---|---|---|
| crosswalk_s6full | dense_ignore | 257 | 0.846 | 0.817 | 0.831 | 100 | 671 |
| crosswalk_s6full | sparse | 227 | n/a | 0.859 | n/a | 0 | 483 |
| crosswalk_s6full | verified_empty | 31 | n/a | n/a | n/a | 0 | 0 |
| crosswalk_s6full_seed2 | dense_ignore | 257 | 0.870 | 0.785 | 0.825 | 79 | 671 |
| crosswalk_s6full_seed2 | sparse | 227 | n/a | 0.845 | n/a | 0 | 483 |
| crosswalk_s6full_seed2 | verified_empty | 31 | n/a | n/a | n/a | 0 | 0 |
| crosswalk_s6full_seed3 | dense_ignore | 257 | 0.834 | 0.821 | 0.827 | 110 | 671 |
| crosswalk_s6full_seed3 | sparse | 227 | n/a | 0.839 | n/a | 0 | 483 |
| crosswalk_s6full_seed3 | verified_empty | 31 | n/a | n/a | n/a | 0 | 0 |
| crosswalk_s6half | dense_ignore | 257 | 0.847 | 0.824 | 0.835 | 100 | 671 |
| crosswalk_s6half | sparse | 227 | n/a | 0.841 | n/a | 0 | 483 |
| crosswalk_s6half | verified_empty | 31 | n/a | n/a | n/a | 0 | 0 |
| crosswalk_s6half_seed2 | dense_ignore | 257 | 0.861 | 0.794 | 0.826 | 86 | 671 |
| crosswalk_s6half_seed2 | sparse | 227 | n/a | 0.830 | n/a | 0 | 483 |
| crosswalk_s6half_seed2 | verified_empty | 31 | n/a | n/a | n/a | 0 | 0 |
| crosswalk_s6half_seed3 | dense_ignore | 257 | 0.843 | 0.851 | 0.847 | 106 | 671 |
| crosswalk_s6half_seed3 | sparse | 227 | n/a | 0.861 | n/a | 0 | 483 |
| crosswalk_s6half_seed3 | verified_empty | 31 | n/a | n/a | n/a | 0 | 0 |
| crosswalk_s4_steps1397 | dense_ignore | 257 | 0.855 | 0.815 | 0.834 | 93 | 671 |
| crosswalk_s4_steps1397 | sparse | 227 | n/a | 0.855 | n/a | 0 | 483 |
| crosswalk_s4_steps1397 | verified_empty | 31 | n/a | n/a | n/a | 0 | 0 |
| crosswalk_s4 | dense_ignore | 257 | 0.815 | 0.841 | 0.828 | 128 | 671 |
| crosswalk_s4 | sparse | 227 | n/a | 0.853 | n/a | 0 | 483 |
| crosswalk_s4 | verified_empty | 31 | n/a | n/a | n/a | 0 | 0 |
| crosswalk_s4_seed2 | dense_ignore | 257 | 0.879 | 0.790 | 0.832 | 73 | 671 |
| crosswalk_s4_seed2 | sparse | 227 | n/a | 0.826 | n/a | 0 | 483 |
| crosswalk_s4_seed2 | verified_empty | 31 | n/a | n/a | n/a | 0 | 0 |
| crosswalk_s4_seed3 | dense_ignore | 257 | 0.825 | 0.841 | 0.832 | 120 | 671 |
| crosswalk_s4_seed3 | sparse | 227 | n/a | 0.863 | n/a | 0 | 483 |
| crosswalk_s4_seed3 | verified_empty | 31 | n/a | n/a | n/a | 0 | 0 |
| released | dense_ignore | 257 | 0.221 | 0.373 | 0.278 | 879 | 671 |
| released | sparse | 227 | n/a | 0.366 | n/a | 0 | 483 |
| released | verified_empty | 31 | 0.000 | n/a | n/a | 9 | 0 |
| label_ceiling | dense_ignore | 257 | 1.000 | 1.000 | 1.000 | 0 | 671 |
| label_ceiling | sparse | 227 | n/a | 1.000 | n/a | 0 | 483 |
| label_ceiling | verified_empty | 31 | n/a | n/a | n/a | 0 | 0 |

### Test, per capture year

| model | group | panos | precision | recall | F1 | FP | positives |
|---|---|---|---|---|---|---|---|
| crosswalk_s6full | 2019-2022 | 318 | 0.821 | 0.837 | 0.829 | 64 | 676 |
| crosswalk_s6full | 2023+ | 167 | 0.876 | 0.832 | 0.853 | 35 | 428 |
| crosswalk_s6full | <=2018 | 30 | 0.875 | 0.820 | 0.847 | 1 | 50 |
| crosswalk_s6full_seed2 | 2019-2022 | 318 | 0.849 | 0.808 | 0.828 | 49 | 676 |
| crosswalk_s6full_seed2 | 2023+ | 167 | 0.891 | 0.811 | 0.849 | 30 | 428 |
| crosswalk_s6full_seed2 | <=2018 | 30 | 1.000 | 0.840 | 0.913 | 0 | 50 |
| crosswalk_s6full_seed3 | 2019-2022 | 318 | 0.831 | 0.837 | 0.834 | 61 | 676 |
| crosswalk_s6full_seed3 | 2023+ | 167 | 0.833 | 0.820 | 0.827 | 49 | 428 |
| crosswalk_s6full_seed3 | <=2018 | 30 | 1.000 | 0.780 | 0.876 | 0 | 50 |
| crosswalk_s6half | 2019-2022 | 318 | 0.850 | 0.828 | 0.839 | 52 | 676 |
| crosswalk_s6half | 2023+ | 167 | 0.840 | 0.841 | 0.841 | 48 | 428 |
| crosswalk_s6half | <=2018 | 30 | 1.000 | 0.780 | 0.876 | 0 | 50 |
| crosswalk_s6half_seed2 | 2019-2022 | 318 | 0.844 | 0.815 | 0.829 | 53 | 676 |
| crosswalk_s6half_seed2 | 2023+ | 167 | 0.882 | 0.806 | 0.842 | 32 | 428 |
| crosswalk_s6half_seed2 | <=2018 | 30 | 0.875 | 0.760 | 0.813 | 1 | 50 |
| crosswalk_s6half_seed3 | 2019-2022 | 318 | 0.843 | 0.859 | 0.851 | 57 | 676 |
| crosswalk_s6half_seed3 | 2023+ | 167 | 0.843 | 0.857 | 0.850 | 48 | 428 |
| crosswalk_s6half_seed3 | <=2018 | 30 | 0.857 | 0.780 | 0.817 | 1 | 50 |
| crosswalk_s4_steps1397 | 2019-2022 | 318 | 0.843 | 0.834 | 0.839 | 54 | 676 |
| crosswalk_s4_steps1397 | 2023+ | 167 | 0.865 | 0.829 | 0.847 | 39 | 428 |
| crosswalk_s4_steps1397 | <=2018 | 30 | 1.000 | 0.820 | 0.901 | 0 | 50 |
| crosswalk_s4 | 2019-2022 | 318 | 0.804 | 0.845 | 0.824 | 73 | 676 |
| crosswalk_s4 | 2023+ | 167 | 0.824 | 0.848 | 0.836 | 55 | 428 |
| crosswalk_s4 | <=2018 | 30 | 1.000 | 0.840 | 0.913 | 0 | 50 |
| crosswalk_s4_seed2 | 2019-2022 | 318 | 0.869 | 0.817 | 0.842 | 43 | 676 |
| crosswalk_s4_seed2 | 2023+ | 167 | 0.888 | 0.790 | 0.836 | 30 | 428 |
| crosswalk_s4_seed2 | <=2018 | 30 | 1.000 | 0.780 | 0.876 | 0 | 50 |
| crosswalk_s4_seed3 | 2019-2022 | 318 | 0.821 | 0.851 | 0.836 | 66 | 676 |
| crosswalk_s4_seed3 | 2023+ | 167 | 0.827 | 0.850 | 0.839 | 53 | 428 |
| crosswalk_s4_seed3 | <=2018 | 30 | 0.875 | 0.840 | 0.857 | 1 | 50 |
| released | 2019-2022 | 318 | 0.185 | 0.359 | 0.245 | 523 | 676 |
| released | 2023+ | 167 | 0.288 | 0.411 | 0.338 | 322 | 428 |
| released | <=2018 | 30 | 0.023 | 0.160 | 0.040 | 43 | 50 |
| label_ceiling | 2019-2022 | 318 | 1.000 | 1.000 | 1.000 | 0 | 676 |
| label_ceiling | 2023+ | 167 | 1.000 | 1.000 | 1.000 | 0 | 428 |
| label_ceiling | <=2018 | 30 | 1.000 | 1.000 | 1.000 | 0 | 50 |

Precision is n/a where a group has no complete pano (every non-Seattle city is `sparse`), or where a complete group drew no detection at all (`verified_empty` with 0 FP).


### Val, overall (where the threshold was chosen)

| model | threshold | precision | recall | F1 |
|---|---|---|---|---|
| crosswalk_s6full | 0.17 | 0.796 | 0.814 | 0.805 |
| crosswalk_s6full_seed2 | 0.27 | 0.837 | 0.782 | 0.808 |
| crosswalk_s6full_seed3 | 0.24 | 0.787 | 0.812 | 0.799 |
| crosswalk_s6half | 0.23 | 0.807 | 0.811 | 0.809 |
| crosswalk_s6half_seed2 | 0.23 | 0.813 | 0.783 | 0.798 |
| crosswalk_s6half_seed3 | 0.25 | 0.803 | 0.814 | 0.809 |
| crosswalk_s4_steps1397 | 0.24 | 0.797 | 0.812 | 0.805 |
| crosswalk_s4 | 0.23 | 0.771 | 0.808 | 0.789 |
| crosswalk_s4_seed2 | 0.30 | 0.835 | 0.779 | 0.806 |
| crosswalk_s4_seed3 | 0.13 | 0.769 | 0.810 | 0.789 |
| released | 0.10 | 0.190 | 0.368 | 0.251 |
| label_ceiling | 0.05 | 1.000 | 1.000 | 1.000 |

## (b) Per OSM node, Seattle test panos, against the map-only baseline

Nodes of `baseline_nodes.csv` within 25 m of a Seattle test pano. A node is predicted marked when a detection at the model's val-F1 threshold, raycast to flat ground at 2.5 m and kept within 25 m range, lands within 10 m. Truth: a current SDOT point within 10 m.

| model | nodes (true) | detector P / R / F1 | `osm_marked_signal` P / R / F1 | detector OR OSM F1 | passes (detector F1 ≥ OSM) |
|---|---|---|---|---|---|
| crosswalk_s6full | 463 (315) | 0.823 / 0.943 / 0.879 | 0.854 / 0.946 / 0.898 | 0.889 | no |
| crosswalk_s6full_seed2 | 463 (315) | 0.817 / 0.933 / 0.871 | 0.854 / 0.946 / 0.898 | 0.887 | no |
| crosswalk_s6full_seed3 | 463 (315) | 0.819 / 0.946 / 0.878 | 0.854 / 0.946 / 0.898 | 0.892 | no |
| crosswalk_s6half | 463 (315) | 0.811 / 0.943 / 0.872 | 0.854 / 0.946 / 0.898 | 0.885 | no |
| crosswalk_s6half_seed2 | 463 (315) | 0.825 / 0.930 / 0.875 | 0.854 / 0.946 / 0.898 | 0.893 | no |
| crosswalk_s6half_seed3 | 463 (315) | 0.821 / 0.946 / 0.879 | 0.854 / 0.946 / 0.898 | 0.888 | no |
| crosswalk_s4_steps1397 | 463 (315) | 0.824 / 0.924 / 0.871 | 0.854 / 0.946 / 0.898 | 0.892 | no |
| crosswalk_s4 | 463 (315) | 0.815 / 0.940 / 0.873 | 0.854 / 0.946 / 0.898 | 0.890 | no |
| crosswalk_s4_seed2 | 463 (315) | 0.830 / 0.927 / 0.876 | 0.854 / 0.946 / 0.898 | 0.892 | no |
| crosswalk_s4_seed3 | 463 (315) | 0.827 / 0.943 / 0.881 | 0.854 / 0.946 / 0.898 | 0.892 | no |
| released | 463 (315) | 0.727 / 0.965 / 0.829 | 0.854 / 0.946 / 0.898 | 0.840 | no |
| label_ceiling | 463 (315) | 0.882 / 0.949 / 0.914 | 0.854 / 0.946 / 0.898 | 0.893 | n/a (ceiling) |

On nodes OSM leaves untagged (`osm_class` unknown):

| model | nodes (true) | detector P / R / F1 | `osm_marked_signal` P / R / F1 |
|---|---|---|---|
| crosswalk_s6full | 38 (9) | 0.583 / 0.778 / 0.667 | 0.625 / 0.556 / 0.588 |
| crosswalk_s6full_seed2 | 38 (9) | 0.538 / 0.778 / 0.636 | 0.625 / 0.556 / 0.588 |
| crosswalk_s6full_seed3 | 38 (9) | 0.667 / 0.889 / 0.762 | 0.625 / 0.556 / 0.588 |
| crosswalk_s6half | 38 (9) | 0.583 / 0.778 / 0.667 | 0.625 / 0.556 / 0.588 |
| crosswalk_s6half_seed2 | 38 (9) | 0.667 / 0.889 / 0.762 | 0.625 / 0.556 / 0.588 |
| crosswalk_s6half_seed3 | 38 (9) | 0.700 / 0.778 / 0.737 | 0.625 / 0.556 / 0.588 |
| crosswalk_s4_steps1397 | 38 (9) | 0.636 / 0.778 / 0.700 | 0.625 / 0.556 / 0.588 |
| crosswalk_s4 | 38 (9) | 0.615 / 0.889 / 0.727 | 0.625 / 0.556 / 0.588 |
| crosswalk_s4_seed2 | 38 (9) | 0.636 / 0.778 / 0.700 | 0.625 / 0.556 / 0.588 |
| crosswalk_s4_seed3 | 38 (9) | 0.667 / 0.889 / 0.762 | 0.625 / 0.556 / 0.588 |
| released | 38 (9) | 0.375 / 1.000 / 0.545 | 0.625 / 0.556 / 0.588 |
| label_ceiling | 38 (9) | 1.000 / 0.889 / 0.941 | 0.625 / 0.556 / 0.588 |

Exploratory (not part of the pass rule): a node-bootstrap 95% interval for detector F1 minus OSM F1 (nodes are spatially correlated, so it is too narrow):

- crosswalk_s6full: [-0.040, 0.002]; 1140 detections on 379 panos, 1014 placed within range
- crosswalk_s6full_seed2: [-0.048, -0.005]; 1060 detections on 379 panos, 905 placed within range
- crosswalk_s6full_seed3: [-0.040, 0.001]; 1156 detections on 379 panos, 963 placed within range
- crosswalk_s6half: [-0.047, -0.005]; 1149 detections on 379 panos, 991 placed within range
- crosswalk_s6half_seed2: [-0.044, -0.002]; 1085 detections on 379 panos, 948 placed within range
- crosswalk_s6half_seed3: [-0.039, 0.002]; 1201 detections on 379 panos, 1041 placed within range
- crosswalk_s4_steps1397: [-0.049, -0.004]; 1132 detections on 379 panos, 965 placed within range
- crosswalk_s4: [-0.047, -0.003]; 1198 detections on 379 panos, 1057 placed within range
- crosswalk_s4_seed2: [-0.045, -0.001]; 1036 detections on 379 panos, 884 placed within range
- crosswalk_s4_seed3: [-0.036, 0.004]; 1209 detections on 379 panos, 1048 placed within range
- released: [-0.094, -0.043]; 1920 detections on 379 panos, 1744 placed within range
- label_ceiling: [-0.005, 0.041]; 899 detections on 379 panos, 774 placed within range

Heading source for the Seattle test panos: {'ps': 152, 'streetlevel': 196, 'ps_negatives': 31}.


## Exploratory, post hoc (not pre-registered: explains (b), never re-decides it)

### crosswalk_s6full: node F1 across thresholds (val pick 0.17)

| threshold | detector P / R / F1 | detector OR OSM F1 | detector AND OSM F1 |
|---|---|---|---|
| 0.05 | 0.810 / 0.971 / 0.883 | 0.885 | 0.896 |
| 0.10 | 0.821 / 0.959 / 0.884 | 0.888 | 0.894 |
| 0.15 | 0.820 / 0.943 / 0.877 | 0.888 | 0.887 |
| 0.20 | 0.824 / 0.933 / 0.875 | 0.890 | 0.882 |
| 0.25 | 0.823 / 0.917 / 0.868 | 0.892 | 0.873 |
| 0.30 | 0.826 / 0.902 / 0.862 | 0.893 | 0.865 |
| 0.35 | 0.826 / 0.889 / 0.856 | 0.893 | 0.859 |
| 0.40 | 0.833 / 0.870 / 0.851 | 0.894 | 0.853 |
| 0.45 | 0.839 / 0.844 / 0.842 | 0.897 | 0.840 |
| 0.50 | 0.838 / 0.835 / 0.836 | 0.897 | 0.834 |
| 0.55 | 0.841 / 0.790 / 0.815 | 0.900 | 0.810 |
| 0.60 | 0.836 / 0.727 / 0.778 | 0.902 | 0.768 |
| 0.65 | 0.846 / 0.683 / 0.756 | 0.904 | 0.743 |
| 0.70 | 0.861 / 0.648 / 0.739 | 0.907 | 0.721 |
| 0.75 | 0.861 / 0.552 / 0.673 | 0.905 | 0.659 |
| 0.80 | 0.854 / 0.410 / 0.554 | 0.902 | 0.541 |
| 0.85 | 0.868 / 0.314 / 0.462 | 0.903 | 0.445 |
| 0.90 | 0.861 / 0.197 / 0.320 | 0.897 | 0.318 |
| 0.95 | 0.909 / 0.127 / 0.223 | 0.896 | 0.223 |

OSM alone: F1 0.898. Saying yes at every node: F1 0.810 (its precision is the share of nodes that are marked).

### crosswalk_s6full: detector vs OSM vs truth at the val threshold (node counts)

| detector | OSM | truth | nodes |
|---|---|---|---|
| yes | yes | marked | 286 |
| yes | yes | not | 44 |
| yes | no | marked | 11 |
| yes | no | not | 20 |
| no | yes | marked | 12 |
| no | yes | not | 7 |
| no | no | marked | 6 |
| no | no | not | 77 |

### crosswalk_s6full_seed2: node F1 across thresholds (val pick 0.27)

| threshold | detector P / R / F1 | detector OR OSM F1 | detector AND OSM F1 |
|---|---|---|---|
| 0.05 | 0.808 / 0.978 / 0.885 | 0.885 | 0.898 |
| 0.10 | 0.812 / 0.971 / 0.884 | 0.888 | 0.894 |
| 0.15 | 0.815 / 0.952 / 0.878 | 0.885 | 0.891 |
| 0.20 | 0.817 / 0.952 / 0.880 | 0.885 | 0.892 |
| 0.25 | 0.819 / 0.946 / 0.878 | 0.887 | 0.889 |
| 0.30 | 0.814 / 0.917 / 0.863 | 0.887 | 0.873 |
| 0.35 | 0.814 / 0.905 / 0.857 | 0.888 | 0.866 |
| 0.40 | 0.820 / 0.867 / 0.843 | 0.890 | 0.848 |
| 0.45 | 0.821 / 0.816 / 0.818 | 0.894 | 0.819 |
| 0.50 | 0.830 / 0.762 / 0.795 | 0.900 | 0.788 |
| 0.55 | 0.830 / 0.695 / 0.756 | 0.901 | 0.747 |
| 0.60 | 0.848 / 0.622 / 0.718 | 0.905 | 0.703 |
| 0.65 | 0.856 / 0.530 / 0.655 | 0.899 | 0.647 |
| 0.70 | 0.870 / 0.381 / 0.530 | 0.899 | 0.521 |
| 0.75 | 0.830 / 0.248 / 0.381 | 0.892 | 0.385 |
| 0.80 | 0.950 / 0.121 / 0.214 | 0.896 | 0.215 |
| 0.85 | 0.952 / 0.063 / 0.119 | 0.898 | 0.119 |
| 0.90 | 1.000 / 0.013 / 0.025 | 0.898 | 0.025 |
| 0.95 | 1.000 / 0.013 / 0.025 | 0.898 | 0.025 |

OSM alone: F1 0.898. Saying yes at every node: F1 0.810 (its precision is the share of nodes that are marked).

### crosswalk_s6full_seed2: detector vs OSM vs truth at the val threshold (node counts)

| detector | OSM | truth | nodes |
|---|---|---|---|
| yes | yes | marked | 283 |
| yes | yes | not | 44 |
| yes | no | marked | 11 |
| yes | no | not | 22 |
| no | yes | marked | 15 |
| no | yes | not | 7 |
| no | no | marked | 6 |
| no | no | not | 75 |

### crosswalk_s6full_seed3: node F1 across thresholds (val pick 0.24)

| threshold | detector P / R / F1 | detector OR OSM F1 | detector AND OSM F1 |
|---|---|---|---|
| 0.05 | 0.801 / 0.981 / 0.882 | 0.883 | 0.897 |
| 0.10 | 0.805 / 0.968 / 0.879 | 0.885 | 0.892 |
| 0.15 | 0.819 / 0.962 / 0.885 | 0.891 | 0.891 |
| 0.20 | 0.817 / 0.949 / 0.878 | 0.891 | 0.884 |
| 0.25 | 0.819 / 0.946 / 0.878 | 0.892 | 0.882 |
| 0.30 | 0.818 / 0.930 / 0.871 | 0.892 | 0.875 |
| 0.35 | 0.827 / 0.911 / 0.867 | 0.895 | 0.869 |
| 0.40 | 0.830 / 0.898 / 0.863 | 0.895 | 0.864 |
| 0.45 | 0.830 / 0.886 / 0.857 | 0.896 | 0.857 |
| 0.50 | 0.827 / 0.863 / 0.845 | 0.896 | 0.844 |
| 0.55 | 0.830 / 0.835 / 0.832 | 0.899 | 0.828 |
| 0.60 | 0.831 / 0.794 / 0.812 | 0.900 | 0.805 |
| 0.65 | 0.823 / 0.724 / 0.770 | 0.901 | 0.761 |
| 0.70 | 0.812 / 0.660 / 0.729 | 0.899 | 0.720 |
| 0.75 | 0.821 / 0.568 / 0.672 | 0.901 | 0.659 |
| 0.80 | 0.835 / 0.451 / 0.586 | 0.905 | 0.564 |
| 0.85 | 0.869 / 0.359 / 0.508 | 0.902 | 0.490 |
| 0.90 | 0.901 / 0.232 / 0.369 | 0.899 | 0.359 |
| 0.95 | 0.939 / 0.146 / 0.253 | 0.895 | 0.254 |

OSM alone: F1 0.898. Saying yes at every node: F1 0.810 (its precision is the share of nodes that are marked).

### crosswalk_s6full_seed3: detector vs OSM vs truth at the val threshold (node counts)

| detector | OSM | truth | nodes |
|---|---|---|---|
| yes | yes | marked | 285 |
| yes | yes | not | 46 |
| yes | no | marked | 13 |
| yes | no | not | 20 |
| no | yes | marked | 13 |
| no | yes | not | 5 |
| no | no | marked | 4 |
| no | no | not | 77 |

### crosswalk_s6half: node F1 across thresholds (val pick 0.23)

| threshold | detector P / R / F1 | detector OR OSM F1 | detector AND OSM F1 |
|---|---|---|---|
| 0.05 | 0.789 / 0.971 / 0.871 | 0.876 | 0.892 |
| 0.10 | 0.798 / 0.965 / 0.874 | 0.881 | 0.890 |
| 0.15 | 0.805 / 0.956 / 0.874 | 0.882 | 0.889 |
| 0.20 | 0.812 / 0.949 / 0.876 | 0.885 | 0.888 |
| 0.25 | 0.813 / 0.937 / 0.870 | 0.885 | 0.882 |
| 0.30 | 0.817 / 0.924 / 0.867 | 0.888 | 0.876 |
| 0.35 | 0.819 / 0.917 / 0.865 | 0.889 | 0.873 |
| 0.40 | 0.821 / 0.902 / 0.859 | 0.889 | 0.867 |
| 0.45 | 0.839 / 0.876 / 0.857 | 0.894 | 0.859 |
| 0.50 | 0.846 / 0.857 / 0.852 | 0.896 | 0.852 |
| 0.55 | 0.844 / 0.806 / 0.825 | 0.896 | 0.824 |
| 0.60 | 0.846 / 0.749 / 0.795 | 0.900 | 0.788 |
| 0.65 | 0.857 / 0.705 / 0.774 | 0.902 | 0.763 |
| 0.70 | 0.849 / 0.590 / 0.697 | 0.898 | 0.690 |
| 0.75 | 0.852 / 0.495 / 0.627 | 0.898 | 0.620 |
| 0.80 | 0.828 / 0.305 / 0.445 | 0.893 | 0.447 |
| 0.85 | 0.829 / 0.184 / 0.301 | 0.895 | 0.299 |
| 0.90 | 0.906 / 0.092 / 0.167 | 0.896 | 0.168 |
| 0.95 | 0.938 / 0.048 / 0.091 | 0.896 | 0.091 |

OSM alone: F1 0.898. Saying yes at every node: F1 0.810 (its precision is the share of nodes that are marked).

### crosswalk_s6half: detector vs OSM vs truth at the val threshold (node counts)

| detector | OSM | truth | nodes |
|---|---|---|---|
| yes | yes | marked | 286 |
| yes | yes | not | 46 |
| yes | no | marked | 11 |
| yes | no | not | 23 |
| no | yes | marked | 12 |
| no | yes | not | 5 |
| no | no | marked | 6 |
| no | no | not | 74 |

### crosswalk_s6half_seed2: node F1 across thresholds (val pick 0.23)

| threshold | detector P / R / F1 | detector OR OSM F1 | detector AND OSM F1 |
|---|---|---|---|
| 0.05 | 0.796 / 0.965 / 0.872 | 0.877 | 0.893 |
| 0.10 | 0.804 / 0.952 / 0.872 | 0.883 | 0.887 |
| 0.15 | 0.819 / 0.949 / 0.879 | 0.890 | 0.886 |
| 0.20 | 0.819 / 0.937 / 0.874 | 0.892 | 0.879 |
| 0.25 | 0.825 / 0.927 / 0.873 | 0.893 | 0.877 |
| 0.30 | 0.827 / 0.908 / 0.865 | 0.893 | 0.869 |
| 0.35 | 0.837 / 0.895 / 0.865 | 0.894 | 0.867 |
| 0.40 | 0.838 / 0.873 / 0.855 | 0.893 | 0.859 |
| 0.45 | 0.843 / 0.851 / 0.847 | 0.894 | 0.849 |
| 0.50 | 0.843 / 0.835 / 0.839 | 0.895 | 0.839 |
| 0.55 | 0.845 / 0.778 / 0.810 | 0.897 | 0.808 |
| 0.60 | 0.841 / 0.724 / 0.778 | 0.898 | 0.773 |
| 0.65 | 0.837 / 0.619 / 0.712 | 0.897 | 0.706 |
| 0.70 | 0.851 / 0.527 / 0.651 | 0.895 | 0.648 |
| 0.75 | 0.880 / 0.397 / 0.547 | 0.897 | 0.540 |
| 0.80 | 0.890 / 0.308 / 0.458 | 0.899 | 0.447 |
| 0.85 | 0.889 / 0.178 / 0.296 | 0.895 | 0.298 |
| 0.90 | 0.902 / 0.117 / 0.208 | 0.898 | 0.208 |
| 0.95 | 1.000 / 0.022 / 0.043 | 0.898 | 0.043 |

OSM alone: F1 0.898. Saying yes at every node: F1 0.810 (its precision is the share of nodes that are marked).

### crosswalk_s6half_seed2: detector vs OSM vs truth at the val threshold (node counts)

| detector | OSM | truth | nodes |
|---|---|---|---|
| yes | yes | marked | 282 |
| yes | yes | not | 45 |
| yes | no | marked | 11 |
| yes | no | not | 17 |
| no | yes | marked | 16 |
| no | yes | not | 6 |
| no | no | marked | 6 |
| no | no | not | 80 |

### crosswalk_s6half_seed3: node F1 across thresholds (val pick 0.25)

| threshold | detector P / R / F1 | detector OR OSM F1 | detector AND OSM F1 |
|---|---|---|---|
| 0.05 | 0.807 / 0.968 / 0.880 | 0.883 | 0.895 |
| 0.10 | 0.815 / 0.962 / 0.882 | 0.884 | 0.895 |
| 0.15 | 0.816 / 0.959 / 0.882 | 0.886 | 0.894 |
| 0.20 | 0.815 / 0.949 / 0.877 | 0.884 | 0.890 |
| 0.25 | 0.821 / 0.946 / 0.879 | 0.888 | 0.889 |
| 0.30 | 0.824 / 0.937 / 0.877 | 0.890 | 0.883 |
| 0.35 | 0.827 / 0.927 / 0.874 | 0.893 | 0.878 |
| 0.40 | 0.830 / 0.914 / 0.870 | 0.894 | 0.872 |
| 0.45 | 0.824 / 0.879 / 0.851 | 0.894 | 0.853 |
| 0.50 | 0.828 / 0.838 / 0.833 | 0.896 | 0.832 |
| 0.55 | 0.828 / 0.778 / 0.802 | 0.895 | 0.801 |
| 0.60 | 0.823 / 0.692 / 0.752 | 0.898 | 0.746 |
| 0.65 | 0.837 / 0.635 / 0.722 | 0.902 | 0.710 |
| 0.70 | 0.841 / 0.537 / 0.655 | 0.899 | 0.645 |
| 0.75 | 0.840 / 0.435 / 0.573 | 0.898 | 0.565 |
| 0.80 | 0.850 / 0.324 / 0.469 | 0.896 | 0.466 |
| 0.85 | 0.897 / 0.222 / 0.356 | 0.897 | 0.354 |
| 0.90 | 0.941 / 0.152 / 0.262 | 0.895 | 0.264 |
| 0.95 | 0.963 / 0.083 / 0.152 | 0.896 | 0.152 |

OSM alone: F1 0.898. Saying yes at every node: F1 0.810 (its precision is the share of nodes that are marked).

### crosswalk_s6half_seed3: detector vs OSM vs truth at the val threshold (node counts)

| detector | OSM | truth | nodes |
|---|---|---|---|
| yes | yes | marked | 287 |
| yes | yes | not | 44 |
| yes | no | marked | 11 |
| yes | no | not | 21 |
| no | yes | marked | 11 |
| no | yes | not | 7 |
| no | no | marked | 6 |
| no | no | not | 76 |

### crosswalk_s4_steps1397: node F1 across thresholds (val pick 0.24)

| threshold | detector P / R / F1 | detector OR OSM F1 | detector AND OSM F1 |
|---|---|---|---|
| 0.05 | 0.804 / 0.975 / 0.881 | 0.881 | 0.898 |
| 0.10 | 0.811 / 0.956 / 0.878 | 0.883 | 0.892 |
| 0.15 | 0.817 / 0.949 / 0.878 | 0.887 | 0.889 |
| 0.20 | 0.825 / 0.940 / 0.878 | 0.892 | 0.883 |
| 0.25 | 0.824 / 0.924 / 0.871 | 0.892 | 0.876 |
| 0.30 | 0.829 / 0.921 / 0.872 | 0.895 | 0.874 |
| 0.35 | 0.835 / 0.914 / 0.873 | 0.897 | 0.872 |
| 0.40 | 0.830 / 0.886 / 0.857 | 0.897 | 0.856 |
| 0.45 | 0.843 / 0.870 / 0.856 | 0.900 | 0.852 |
| 0.50 | 0.844 / 0.844 / 0.844 | 0.900 | 0.840 |
| 0.55 | 0.850 / 0.813 / 0.831 | 0.901 | 0.825 |
| 0.60 | 0.842 / 0.762 / 0.800 | 0.901 | 0.792 |
| 0.65 | 0.848 / 0.708 / 0.772 | 0.906 | 0.757 |
| 0.70 | 0.853 / 0.629 / 0.724 | 0.907 | 0.706 |
| 0.75 | 0.858 / 0.537 / 0.660 | 0.907 | 0.641 |
| 0.80 | 0.892 / 0.444 / 0.593 | 0.906 | 0.576 |
| 0.85 | 0.909 / 0.317 / 0.471 | 0.903 | 0.457 |
| 0.90 | 0.875 / 0.178 / 0.296 | 0.898 | 0.292 |
| 0.95 | 0.857 / 0.076 / 0.140 | 0.898 | 0.140 |

OSM alone: F1 0.898. Saying yes at every node: F1 0.810 (its precision is the share of nodes that are marked).

### crosswalk_s4_steps1397: detector vs OSM vs truth at the val threshold (node counts)

| detector | OSM | truth | nodes |
|---|---|---|---|
| yes | yes | marked | 279 |
| yes | yes | not | 43 |
| yes | no | marked | 12 |
| yes | no | not | 19 |
| no | yes | marked | 19 |
| no | yes | not | 8 |
| no | no | marked | 5 |
| no | no | not | 78 |

### crosswalk_s4: node F1 across thresholds (val pick 0.23)

| threshold | detector P / R / F1 | detector OR OSM F1 | detector AND OSM F1 |
|---|---|---|---|
| 0.05 | 0.803 / 0.971 / 0.879 | 0.882 | 0.895 |
| 0.10 | 0.805 / 0.959 / 0.875 | 0.885 | 0.888 |
| 0.15 | 0.809 / 0.952 / 0.875 | 0.886 | 0.886 |
| 0.20 | 0.809 / 0.943 / 0.871 | 0.886 | 0.882 |
| 0.25 | 0.815 / 0.940 / 0.873 | 0.890 | 0.880 |
| 0.30 | 0.832 / 0.930 / 0.879 | 0.896 | 0.879 |
| 0.35 | 0.834 / 0.927 / 0.878 | 0.896 | 0.879 |
| 0.40 | 0.835 / 0.914 / 0.873 | 0.896 | 0.873 |
| 0.45 | 0.835 / 0.898 / 0.865 | 0.896 | 0.865 |
| 0.50 | 0.839 / 0.879 / 0.859 | 0.894 | 0.861 |
| 0.55 | 0.833 / 0.822 / 0.827 | 0.894 | 0.828 |
| 0.60 | 0.836 / 0.775 / 0.804 | 0.897 | 0.801 |
| 0.65 | 0.838 / 0.692 / 0.758 | 0.899 | 0.752 |
| 0.70 | 0.836 / 0.600 / 0.699 | 0.900 | 0.690 |
| 0.75 | 0.832 / 0.473 / 0.603 | 0.901 | 0.589 |
| 0.80 | 0.865 / 0.365 / 0.513 | 0.904 | 0.494 |
| 0.85 | 0.871 / 0.235 / 0.370 | 0.898 | 0.364 |
| 0.90 | 0.810 / 0.108 / 0.190 | 0.895 | 0.192 |
| 0.95 | 0.900 / 0.057 / 0.107 | 0.898 | 0.107 |

OSM alone: F1 0.898. Saying yes at every node: F1 0.810 (its precision is the share of nodes that are marked).

### crosswalk_s4: detector vs OSM vs truth at the val threshold (node counts)

| detector | OSM | truth | nodes |
|---|---|---|---|
| yes | yes | marked | 283 |
| yes | yes | not | 45 |
| yes | no | marked | 13 |
| yes | no | not | 22 |
| no | yes | marked | 15 |
| no | yes | not | 6 |
| no | no | marked | 4 |
| no | no | not | 75 |

### crosswalk_s4_seed2: node F1 across thresholds (val pick 0.30)

| threshold | detector P / R / F1 | detector OR OSM F1 | detector AND OSM F1 |
|---|---|---|---|
| 0.05 | 0.814 / 0.975 / 0.887 | 0.886 | 0.899 |
| 0.10 | 0.817 / 0.965 / 0.885 | 0.887 | 0.895 |
| 0.15 | 0.820 / 0.952 / 0.881 | 0.887 | 0.892 |
| 0.20 | 0.818 / 0.940 / 0.874 | 0.887 | 0.885 |
| 0.25 | 0.829 / 0.940 / 0.881 | 0.892 | 0.886 |
| 0.30 | 0.830 / 0.927 / 0.876 | 0.892 | 0.881 |
| 0.35 | 0.829 / 0.911 / 0.868 | 0.893 | 0.872 |
| 0.40 | 0.833 / 0.870 / 0.851 | 0.893 | 0.854 |
| 0.45 | 0.835 / 0.851 / 0.843 | 0.893 | 0.845 |
| 0.50 | 0.842 / 0.794 / 0.817 | 0.896 | 0.816 |
| 0.55 | 0.849 / 0.752 / 0.798 | 0.896 | 0.796 |
| 0.60 | 0.852 / 0.692 / 0.764 | 0.896 | 0.761 |
| 0.65 | 0.847 / 0.632 / 0.724 | 0.899 | 0.717 |
| 0.70 | 0.826 / 0.511 / 0.631 | 0.895 | 0.628 |
| 0.75 | 0.837 / 0.406 / 0.547 | 0.893 | 0.544 |
| 0.80 | 0.842 / 0.270 / 0.409 | 0.894 | 0.406 |
| 0.85 | 0.879 / 0.184 / 0.304 | 0.895 | 0.302 |
| 0.90 | 0.886 / 0.098 / 0.177 | 0.896 | 0.178 |
| 0.95 | 1.000 / 0.041 / 0.079 | 0.898 | 0.079 |

OSM alone: F1 0.898. Saying yes at every node: F1 0.810 (its precision is the share of nodes that are marked).

### crosswalk_s4_seed2: detector vs OSM vs truth at the val threshold (node counts)

| detector | OSM | truth | nodes |
|---|---|---|---|
| yes | yes | marked | 281 |
| yes | yes | not | 42 |
| yes | no | marked | 11 |
| yes | no | not | 18 |
| no | yes | marked | 17 |
| no | yes | not | 9 |
| no | no | marked | 6 |
| no | no | not | 79 |

### crosswalk_s4_seed3: node F1 across thresholds (val pick 0.13)

| threshold | detector P / R / F1 | detector OR OSM F1 | detector AND OSM F1 |
|---|---|---|---|
| 0.05 | 0.811 / 0.956 / 0.878 | 0.883 | 0.892 |
| 0.10 | 0.824 / 0.949 / 0.882 | 0.889 | 0.890 |
| 0.15 | 0.825 / 0.930 / 0.875 | 0.890 | 0.882 |
| 0.20 | 0.826 / 0.921 / 0.871 | 0.891 | 0.876 |
| 0.25 | 0.829 / 0.908 / 0.867 | 0.891 | 0.872 |
| 0.30 | 0.830 / 0.898 / 0.863 | 0.893 | 0.867 |
| 0.35 | 0.833 / 0.889 / 0.860 | 0.893 | 0.864 |
| 0.40 | 0.832 / 0.883 / 0.857 | 0.893 | 0.860 |
| 0.45 | 0.836 / 0.876 / 0.856 | 0.894 | 0.858 |
| 0.50 | 0.843 / 0.854 / 0.849 | 0.895 | 0.849 |
| 0.55 | 0.841 / 0.825 / 0.833 | 0.895 | 0.833 |
| 0.60 | 0.842 / 0.778 / 0.809 | 0.899 | 0.803 |
| 0.65 | 0.839 / 0.730 / 0.781 | 0.901 | 0.773 |
| 0.70 | 0.833 / 0.648 / 0.729 | 0.899 | 0.722 |
| 0.75 | 0.828 / 0.521 / 0.639 | 0.897 | 0.633 |
| 0.80 | 0.856 / 0.416 / 0.560 | 0.897 | 0.553 |
| 0.85 | 0.897 / 0.333 / 0.486 | 0.901 | 0.474 |
| 0.90 | 0.900 / 0.229 / 0.365 | 0.901 | 0.350 |
| 0.95 | 0.907 / 0.124 / 0.218 | 0.898 | 0.213 |

OSM alone: F1 0.898. Saying yes at every node: F1 0.810 (its precision is the share of nodes that are marked).

### crosswalk_s4_seed3: detector vs OSM vs truth at the val threshold (node counts)

| detector | OSM | truth | nodes |
|---|---|---|---|
| yes | yes | marked | 286 |
| yes | yes | not | 44 |
| yes | no | marked | 11 |
| yes | no | not | 18 |
| no | yes | marked | 12 |
| no | yes | not | 7 |
| no | no | marked | 6 |
| no | no | not | 79 |

### released: node F1 across thresholds (val pick 0.10)

| threshold | detector P / R / F1 | detector OR OSM F1 | detector AND OSM F1 |
|---|---|---|---|
| 0.05 | 0.721 / 0.978 / 0.830 | 0.836 | 0.893 |
| 0.10 | 0.727 / 0.965 / 0.829 | 0.840 | 0.887 |
| 0.15 | 0.735 / 0.952 / 0.830 | 0.843 | 0.885 |
| 0.20 | 0.738 / 0.940 / 0.827 | 0.844 | 0.880 |
| 0.25 | 0.741 / 0.937 / 0.827 | 0.847 | 0.878 |
| 0.30 | 0.746 / 0.924 / 0.826 | 0.849 | 0.874 |
| 0.35 | 0.749 / 0.921 / 0.826 | 0.851 | 0.872 |
| 0.40 | 0.749 / 0.921 / 0.826 | 0.851 | 0.872 |
| 0.45 | 0.749 / 0.921 / 0.826 | 0.851 | 0.872 |
| 0.50 | 0.757 / 0.911 / 0.827 | 0.854 | 0.870 |
| 0.55 | 0.757 / 0.908 / 0.825 | 0.854 | 0.868 |
| 0.60 | 0.758 / 0.905 / 0.825 | 0.854 | 0.868 |
| 0.65 | 0.759 / 0.902 / 0.824 | 0.855 | 0.866 |
| 0.70 | 0.759 / 0.902 / 0.824 | 0.855 | 0.866 |
| 0.75 | 0.759 / 0.892 / 0.820 | 0.856 | 0.861 |
| 0.80 | 0.758 / 0.876 / 0.813 | 0.855 | 0.855 |
| 0.85 | 0.765 / 0.857 / 0.808 | 0.853 | 0.852 |
| 0.90 | 0.755 / 0.771 / 0.763 | 0.854 | 0.803 |
| 0.95 | 0.786 / 0.537 / 0.638 | 0.873 | 0.656 |

OSM alone: F1 0.898. Saying yes at every node: F1 0.810 (its precision is the share of nodes that are marked).

### released: detector vs OSM vs truth at the val threshold (node counts)

| detector | OSM | truth | nodes |
|---|---|---|---|
| yes | yes | marked | 287 |
| yes | yes | not | 45 |
| yes | no | marked | 17 |
| yes | no | not | 69 |
| no | yes | marked | 11 |
| no | yes | not | 6 |
| no | no | not | 28 |

