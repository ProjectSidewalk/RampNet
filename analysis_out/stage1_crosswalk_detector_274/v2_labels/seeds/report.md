# First crosswalk detector: pre-registered scores (RampNet#274)

Scored 2026-11-08T21:58:56Z by `score_274.py score`. Protocol: `docs/stage1_crosswalk_sdot_274.md`.

## Models

| label | sha256 | val-F1 threshold |
|---|---|---|
| crosswalk_s1 | `6f467ff4afd7d092…` | 0.21 |
| crosswalk_s3 | `32f9356178faa571…` | 0.29 |
| crosswalk_s1_seed2 | `dcdbd4ee008bd330…` | 0.24 |
| crosswalk_s1_seed3 | `c3126aa644f8bfd7…` | 0.22 |
| crosswalk_s3_seed2 | `86a42bfca9dcd0f0…` | 0.39 |
| crosswalk_s3_seed3 | `5de289f7f5f46a27…` | 0.31 |

## (a) Pixels, test split, at each model's own val-F1 threshold

Precision over complete panos (dense_ignore + verified_empty); recall over every test positive. 95% Wilson intervals.

| model | precision | recall | F1 | TP (complete) | FP | ignored | positives |
|---|---|---|---|---|---|---|---|
| crosswalk_s1 | 0.850 [0.820, 0.875] | 0.845 [0.823, 0.865] | 0.847 | 975 (565) | 100 | 118 | 1154 |
| crosswalk_s3 | 0.858 [0.830, 0.883] | 0.840 [0.817, 0.860] | 0.849 | 969 (569) | 94 | 125 | 1154 |
| crosswalk_s1_seed2 | 0.885 [0.857, 0.908] | 0.791 [0.767, 0.814] | 0.836 | 913 (524) | 68 | 90 | 1154 |
| crosswalk_s1_seed3 | 0.817 [0.786, 0.844] | 0.841 [0.818, 0.861] | 0.828 | 970 (557) | 125 | 128 | 1154 |
| crosswalk_s3_seed2 | 0.890 [0.862, 0.912] | 0.795 [0.771, 0.818] | 0.840 | 918 (533) | 66 | 112 | 1154 |
| crosswalk_s3_seed3 | 0.855 [0.825, 0.880] | 0.795 [0.771, 0.818] | 0.824 | 918 (537) | 91 | 113 | 1154 |

### Test, per city

| model | group | panos | precision | recall | F1 | FP | positives |
|---|---|---|---|---|---|---|---|
| crosswalk_s1 | amsterdam | 64 | n/a | 0.840 | n/a | 0 | 131 |
| crosswalk_s1 | burnaby | 49 | n/a | 0.874 | n/a | 0 | 87 |
| crosswalk_s1 | seattle-wa | 379 | 0.850 | 0.846 | 0.848 | 100 | 899 |
| crosswalk_s1 | teaneck-nj | 23 | n/a | 0.757 | n/a | 0 | 37 |
| crosswalk_s3 | amsterdam | 64 | n/a | 0.802 | n/a | 0 | 131 |
| crosswalk_s3 | burnaby | 49 | n/a | 0.862 | n/a | 0 | 87 |
| crosswalk_s3 | seattle-wa | 379 | 0.858 | 0.845 | 0.852 | 94 | 899 |
| crosswalk_s3 | teaneck-nj | 23 | n/a | 0.784 | n/a | 0 | 37 |
| crosswalk_s1_seed2 | amsterdam | 64 | n/a | 0.840 | n/a | 0 | 131 |
| crosswalk_s1_seed2 | burnaby | 49 | n/a | 0.851 | n/a | 0 | 87 |
| crosswalk_s1_seed2 | seattle-wa | 379 | 0.885 | 0.780 | 0.829 | 68 | 899 |
| crosswalk_s1_seed2 | teaneck-nj | 23 | n/a | 0.757 | n/a | 0 | 37 |
| crosswalk_s1_seed3 | amsterdam | 64 | n/a | 0.855 | n/a | 0 | 131 |
| crosswalk_s1_seed3 | burnaby | 49 | n/a | 0.897 | n/a | 0 | 87 |
| crosswalk_s1_seed3 | seattle-wa | 379 | 0.817 | 0.834 | 0.825 | 125 | 899 |
| crosswalk_s1_seed3 | teaneck-nj | 23 | n/a | 0.811 | n/a | 0 | 37 |
| crosswalk_s3_seed2 | amsterdam | 64 | n/a | 0.779 | n/a | 0 | 131 |
| crosswalk_s3_seed2 | burnaby | 49 | n/a | 0.851 | n/a | 0 | 87 |
| crosswalk_s3_seed2 | seattle-wa | 379 | 0.890 | 0.795 | 0.840 | 66 | 899 |
| crosswalk_s3_seed2 | teaneck-nj | 23 | n/a | 0.730 | n/a | 0 | 37 |
| crosswalk_s3_seed3 | amsterdam | 64 | n/a | 0.771 | n/a | 0 | 131 |
| crosswalk_s3_seed3 | burnaby | 49 | n/a | 0.805 | n/a | 0 | 87 |
| crosswalk_s3_seed3 | seattle-wa | 379 | 0.855 | 0.802 | 0.828 | 91 | 899 |
| crosswalk_s3_seed3 | teaneck-nj | 23 | n/a | 0.703 | n/a | 0 | 37 |

### Test, per supervision mode

| model | group | panos | precision | recall | F1 | FP | positives |
|---|---|---|---|---|---|---|---|
| crosswalk_s1 | dense_ignore | 257 | 0.850 | 0.842 | 0.846 | 100 | 671 |
| crosswalk_s1 | sparse | 227 | n/a | 0.849 | n/a | 0 | 483 |
| crosswalk_s1 | verified_empty | 31 | n/a | n/a | n/a | 0 | 0 |
| crosswalk_s3 | dense_ignore | 257 | 0.858 | 0.848 | 0.853 | 94 | 671 |
| crosswalk_s3 | sparse | 227 | n/a | 0.828 | n/a | 0 | 483 |
| crosswalk_s3 | verified_empty | 31 | n/a | n/a | n/a | 0 | 0 |
| crosswalk_s1_seed2 | dense_ignore | 257 | 0.885 | 0.781 | 0.830 | 68 | 671 |
| crosswalk_s1_seed2 | sparse | 227 | n/a | 0.805 | n/a | 0 | 483 |
| crosswalk_s1_seed2 | verified_empty | 31 | n/a | n/a | n/a | 0 | 0 |
| crosswalk_s1_seed3 | dense_ignore | 257 | 0.817 | 0.830 | 0.823 | 125 | 671 |
| crosswalk_s1_seed3 | sparse | 227 | n/a | 0.855 | n/a | 0 | 483 |
| crosswalk_s1_seed3 | verified_empty | 31 | n/a | n/a | n/a | 0 | 0 |
| crosswalk_s3_seed2 | dense_ignore | 257 | 0.890 | 0.794 | 0.839 | 66 | 671 |
| crosswalk_s3_seed2 | sparse | 227 | n/a | 0.797 | n/a | 0 | 483 |
| crosswalk_s3_seed2 | verified_empty | 31 | n/a | n/a | n/a | 0 | 0 |
| crosswalk_s3_seed3 | dense_ignore | 257 | 0.855 | 0.800 | 0.827 | 91 | 671 |
| crosswalk_s3_seed3 | sparse | 227 | n/a | 0.789 | n/a | 0 | 483 |
| crosswalk_s3_seed3 | verified_empty | 31 | n/a | n/a | n/a | 0 | 0 |

### Test, per capture year

| model | group | panos | precision | recall | F1 | FP | positives |
|---|---|---|---|---|---|---|---|
| crosswalk_s1 | 2019-2022 | 318 | 0.856 | 0.848 | 0.852 | 51 | 676 |
| crosswalk_s1 | 2023+ | 167 | 0.842 | 0.846 | 0.844 | 48 | 428 |
| crosswalk_s1 | <=2018 | 30 | 0.875 | 0.800 | 0.836 | 1 | 50 |
| crosswalk_s3 | 2019-2022 | 318 | 0.838 | 0.843 | 0.841 | 59 | 676 |
| crosswalk_s3 | 2023+ | 167 | 0.883 | 0.841 | 0.862 | 34 | 428 |
| crosswalk_s3 | <=2018 | 30 | 0.875 | 0.780 | 0.825 | 1 | 50 |
| crosswalk_s1_seed2 | 2019-2022 | 318 | 0.883 | 0.800 | 0.840 | 37 | 676 |
| crosswalk_s1_seed2 | 2023+ | 167 | 0.884 | 0.780 | 0.829 | 31 | 428 |
| crosswalk_s1_seed2 | <=2018 | 30 | 1.000 | 0.760 | 0.864 | 0 | 50 |
| crosswalk_s1_seed3 | 2019-2022 | 318 | 0.802 | 0.843 | 0.822 | 74 | 676 |
| crosswalk_s1_seed3 | 2023+ | 167 | 0.833 | 0.836 | 0.835 | 50 | 428 |
| crosswalk_s1_seed3 | <=2018 | 30 | 0.875 | 0.840 | 0.857 | 1 | 50 |
| crosswalk_s3_seed2 | 2019-2022 | 318 | 0.889 | 0.793 | 0.838 | 35 | 676 |
| crosswalk_s3_seed2 | 2023+ | 167 | 0.888 | 0.806 | 0.845 | 31 | 428 |
| crosswalk_s3_seed2 | <=2018 | 30 | 1.000 | 0.740 | 0.851 | 0 | 50 |
| crosswalk_s3_seed3 | 2019-2022 | 318 | 0.838 | 0.791 | 0.814 | 55 | 676 |
| crosswalk_s3_seed3 | 2023+ | 167 | 0.878 | 0.806 | 0.841 | 34 | 428 |
| crosswalk_s3_seed3 | <=2018 | 30 | 0.778 | 0.760 | 0.769 | 2 | 50 |

Precision is n/a where a group has no complete pano (every non-Seattle city is `sparse`), or where a complete group drew no detection at all (`verified_empty` with 0 FP).


### Val, overall (where the threshold was chosen)

| model | threshold | precision | recall | F1 |
|---|---|---|---|---|
| crosswalk_s1 | 0.21 | 0.800 | 0.806 | 0.803 |
| crosswalk_s3 | 0.29 | 0.817 | 0.801 | 0.809 |
| crosswalk_s1_seed2 | 0.24 | 0.870 | 0.770 | 0.817 |
| crosswalk_s1_seed3 | 0.22 | 0.799 | 0.806 | 0.803 |
| crosswalk_s3_seed2 | 0.39 | 0.869 | 0.773 | 0.818 |
| crosswalk_s3_seed3 | 0.31 | 0.832 | 0.767 | 0.798 |

## (b) Per OSM node, Seattle test panos, against the map-only baseline

Nodes of `baseline_nodes.csv` within 25 m of a Seattle test pano. A node is predicted marked when a detection at the model's val-F1 threshold, raycast to flat ground at 2.5 m and kept within 25 m range, lands within 10 m. Truth: a current SDOT point within 10 m.

| model | nodes (true) | detector P / R / F1 | `osm_marked_signal` P / R / F1 | detector OR OSM F1 | passes (detector F1 ≥ OSM) |
|---|---|---|---|---|---|
| crosswalk_s1 | 463 (315) | 0.821 / 0.949 / 0.881 | 0.854 / 0.946 / 0.898 | 0.894 | no |
| crosswalk_s3 | 463 (315) | 0.819 / 0.946 / 0.878 | 0.854 / 0.946 / 0.898 | 0.888 | no |
| crosswalk_s1_seed2 | 463 (315) | 0.825 / 0.927 / 0.873 | 0.854 / 0.946 / 0.898 | 0.892 | no |
| crosswalk_s1_seed3 | 463 (315) | 0.814 / 0.943 / 0.874 | 0.854 / 0.946 / 0.898 | 0.890 | no |
| crosswalk_s3_seed2 | 463 (315) | 0.821 / 0.930 / 0.872 | 0.854 / 0.946 / 0.898 | 0.886 | no |
| crosswalk_s3_seed3 | 463 (315) | 0.816 / 0.930 / 0.869 | 0.854 / 0.946 / 0.898 | 0.890 | no |

On nodes OSM leaves untagged (`osm_class` unknown):

| model | nodes (true) | detector P / R / F1 | `osm_marked_signal` P / R / F1 |
|---|---|---|---|
| crosswalk_s1 | 38 (9) | 0.667 / 0.889 / 0.762 | 0.625 / 0.556 / 0.588 |
| crosswalk_s3 | 38 (9) | 0.583 / 0.778 / 0.667 | 0.625 / 0.556 / 0.588 |
| crosswalk_s1_seed2 | 38 (9) | 0.875 / 0.778 / 0.824 | 0.625 / 0.556 / 0.588 |
| crosswalk_s1_seed3 | 38 (9) | 0.615 / 0.889 / 0.727 | 0.625 / 0.556 / 0.588 |
| crosswalk_s3_seed2 | 38 (9) | 0.636 / 0.778 / 0.700 | 0.625 / 0.556 / 0.588 |
| crosswalk_s3_seed3 | 38 (9) | 0.615 / 0.889 / 0.727 | 0.625 / 0.556 / 0.588 |

Exploratory (not part of the pass rule): a node-bootstrap 95% interval for detector F1 minus OSM F1 (nodes are spatially correlated, so it is too narrow):

- crosswalk_s1: [-0.039, 0.005]; 1156 detections on 379 panos, 1028 placed within range
- crosswalk_s3: [-0.040, 0.001]; 1161 detections on 379 panos, 1049 placed within range
- crosswalk_s1_seed2: [-0.047, -0.002]; 996 detections on 379 panos, 873 placed within range
- crosswalk_s1_seed3: [-0.045, -0.003]; 1209 detections on 379 panos, 1059 placed within range
- crosswalk_s3_seed2: [-0.046, -0.005]; 1044 detections on 379 panos, 916 placed within range
- crosswalk_s3_seed3: [-0.050, -0.005]; 1101 detections on 379 panos, 970 placed within range

Heading source for the Seattle test panos: {'ps': 152, 'streetlevel': 196, 'ps_negatives': 31}.


## Exploratory, post hoc (not pre-registered: explains (b), never re-decides it)

### crosswalk_s1: node F1 across thresholds (val pick 0.21)

| threshold | detector P / R / F1 | detector OR OSM F1 | detector AND OSM F1 |
|---|---|---|---|
| 0.05 | 0.801 / 0.971 / 0.878 | 0.883 | 0.893 |
| 0.10 | 0.807 / 0.959 / 0.877 | 0.888 | 0.886 |
| 0.15 | 0.814 / 0.956 / 0.879 | 0.889 | 0.887 |
| 0.20 | 0.821 / 0.949 / 0.881 | 0.894 | 0.884 |
| 0.25 | 0.828 / 0.949 / 0.885 | 0.894 | 0.888 |
| 0.30 | 0.832 / 0.930 / 0.879 | 0.895 | 0.881 |
| 0.35 | 0.830 / 0.902 / 0.865 | 0.893 | 0.868 |
| 0.40 | 0.832 / 0.879 / 0.855 | 0.893 | 0.858 |
| 0.45 | 0.836 / 0.857 / 0.846 | 0.895 | 0.847 |
| 0.50 | 0.834 / 0.829 / 0.831 | 0.894 | 0.832 |
| 0.55 | 0.842 / 0.794 / 0.817 | 0.898 | 0.813 |
| 0.60 | 0.840 / 0.752 / 0.794 | 0.901 | 0.786 |
| 0.65 | 0.839 / 0.695 / 0.760 | 0.899 | 0.753 |
| 0.70 | 0.837 / 0.635 / 0.722 | 0.901 | 0.712 |
| 0.75 | 0.848 / 0.530 / 0.652 | 0.904 | 0.636 |
| 0.80 | 0.850 / 0.432 / 0.573 | 0.903 | 0.555 |
| 0.85 | 0.880 / 0.327 / 0.477 | 0.905 | 0.455 |
| 0.90 | 0.872 / 0.216 / 0.346 | 0.901 | 0.331 |
| 0.95 | 0.837 / 0.114 / 0.201 | 0.898 | 0.197 |

OSM alone: F1 0.898. Saying yes at every node: F1 0.810 (its precision is the share of nodes that are marked).

### crosswalk_s1: detector vs OSM vs truth at the val threshold (node counts)

| detector | OSM | truth | nodes |
|---|---|---|---|
| yes | yes | marked | 285 |
| yes | yes | not | 45 |
| yes | no | marked | 14 |
| yes | no | not | 20 |
| no | yes | marked | 13 |
| no | yes | not | 6 |
| no | no | marked | 3 |
| no | no | not | 77 |

### crosswalk_s3: node F1 across thresholds (val pick 0.29)

| threshold | detector P / R / F1 | detector OR OSM F1 | detector AND OSM F1 |
|---|---|---|---|
| 0.05 | 0.795 / 0.971 / 0.874 | 0.876 | 0.896 |
| 0.10 | 0.801 / 0.959 / 0.873 | 0.881 | 0.890 |
| 0.15 | 0.810 / 0.959 / 0.878 | 0.886 | 0.890 |
| 0.20 | 0.811 / 0.956 / 0.878 | 0.887 | 0.888 |
| 0.25 | 0.818 / 0.956 / 0.881 | 0.888 | 0.891 |
| 0.30 | 0.819 / 0.946 / 0.878 | 0.888 | 0.887 |
| 0.35 | 0.819 / 0.933 / 0.872 | 0.889 | 0.880 |
| 0.40 | 0.821 / 0.930 / 0.872 | 0.889 | 0.880 |
| 0.45 | 0.834 / 0.927 / 0.878 | 0.890 | 0.885 |
| 0.50 | 0.834 / 0.892 / 0.862 | 0.892 | 0.867 |
| 0.55 | 0.838 / 0.851 / 0.844 | 0.894 | 0.845 |
| 0.60 | 0.837 / 0.813 / 0.824 | 0.896 | 0.824 |
| 0.65 | 0.834 / 0.768 / 0.800 | 0.898 | 0.795 |
| 0.70 | 0.847 / 0.667 / 0.746 | 0.902 | 0.735 |
| 0.75 | 0.846 / 0.559 / 0.673 | 0.903 | 0.660 |
| 0.80 | 0.833 / 0.460 / 0.593 | 0.899 | 0.583 |
| 0.85 | 0.823 / 0.324 / 0.465 | 0.896 | 0.457 |
| 0.90 | 0.848 / 0.248 / 0.383 | 0.897 | 0.378 |
| 0.95 | 0.842 / 0.152 / 0.258 | 0.895 | 0.255 |

OSM alone: F1 0.898. Saying yes at every node: F1 0.810 (its precision is the share of nodes that are marked).

### crosswalk_s3: detector vs OSM vs truth at the val threshold (node counts)

| detector | OSM | truth | nodes |
|---|---|---|---|
| yes | yes | marked | 287 |
| yes | yes | not | 45 |
| yes | no | marked | 11 |
| yes | no | not | 21 |
| no | yes | marked | 11 |
| no | yes | not | 6 |
| no | no | marked | 6 |
| no | no | not | 76 |

### crosswalk_s1_seed2: node F1 across thresholds (val pick 0.24)

| threshold | detector P / R / F1 | detector OR OSM F1 | detector AND OSM F1 |
|---|---|---|---|
| 0.05 | 0.802 / 0.962 / 0.874 | 0.884 | 0.888 |
| 0.10 | 0.811 / 0.952 / 0.876 | 0.887 | 0.886 |
| 0.15 | 0.816 / 0.946 / 0.876 | 0.891 | 0.882 |
| 0.20 | 0.817 / 0.933 / 0.871 | 0.891 | 0.877 |
| 0.25 | 0.827 / 0.924 / 0.873 | 0.892 | 0.877 |
| 0.30 | 0.828 / 0.914 / 0.869 | 0.893 | 0.872 |
| 0.35 | 0.837 / 0.898 / 0.867 | 0.893 | 0.870 |
| 0.40 | 0.840 / 0.867 / 0.853 | 0.894 | 0.855 |
| 0.45 | 0.842 / 0.844 / 0.843 | 0.897 | 0.842 |
| 0.50 | 0.862 / 0.813 / 0.837 | 0.902 | 0.829 |
| 0.55 | 0.860 / 0.743 / 0.797 | 0.904 | 0.787 |
| 0.60 | 0.860 / 0.644 / 0.737 | 0.902 | 0.727 |
| 0.65 | 0.856 / 0.565 / 0.681 | 0.898 | 0.673 |
| 0.70 | 0.865 / 0.489 / 0.625 | 0.899 | 0.615 |
| 0.75 | 0.855 / 0.394 / 0.539 | 0.902 | 0.524 |
| 0.80 | 0.896 / 0.273 / 0.418 | 0.903 | 0.401 |
| 0.85 | 0.908 / 0.187 / 0.311 | 0.898 | 0.303 |
| 0.90 | 0.971 / 0.105 / 0.189 | 0.896 | 0.190 |
| 0.95 | 1.000 / 0.048 / 0.091 | 0.898 | 0.091 |

OSM alone: F1 0.898. Saying yes at every node: F1 0.810 (its precision is the share of nodes that are marked).

### crosswalk_s1_seed2: detector vs OSM vs truth at the val threshold (node counts)

| detector | OSM | truth | nodes |
|---|---|---|---|
| yes | yes | marked | 280 |
| yes | yes | not | 43 |
| yes | no | marked | 12 |
| yes | no | not | 19 |
| no | yes | marked | 18 |
| no | yes | not | 8 |
| no | no | marked | 5 |
| no | no | not | 78 |

### crosswalk_s1_seed3: node F1 across thresholds (val pick 0.22)

| threshold | detector P / R / F1 | detector OR OSM F1 | detector AND OSM F1 |
|---|---|---|---|
| 0.05 | 0.795 / 0.975 / 0.876 | 0.880 | 0.893 |
| 0.10 | 0.798 / 0.968 / 0.875 | 0.881 | 0.892 |
| 0.15 | 0.811 / 0.956 / 0.878 | 0.887 | 0.888 |
| 0.20 | 0.814 / 0.943 / 0.874 | 0.890 | 0.881 |
| 0.25 | 0.812 / 0.933 / 0.869 | 0.888 | 0.877 |
| 0.30 | 0.813 / 0.924 / 0.865 | 0.890 | 0.872 |
| 0.35 | 0.817 / 0.895 / 0.855 | 0.891 | 0.860 |
| 0.40 | 0.824 / 0.879 / 0.851 | 0.892 | 0.855 |
| 0.45 | 0.838 / 0.851 / 0.844 | 0.896 | 0.844 |
| 0.50 | 0.843 / 0.838 / 0.841 | 0.897 | 0.839 |
| 0.55 | 0.848 / 0.800 / 0.824 | 0.900 | 0.818 |
| 0.60 | 0.847 / 0.740 / 0.790 | 0.901 | 0.782 |
| 0.65 | 0.847 / 0.670 / 0.748 | 0.902 | 0.738 |
| 0.70 | 0.836 / 0.600 / 0.699 | 0.899 | 0.691 |
| 0.75 | 0.866 / 0.492 / 0.628 | 0.895 | 0.625 |
| 0.80 | 0.865 / 0.365 / 0.513 | 0.896 | 0.507 |
| 0.85 | 0.894 / 0.267 / 0.411 | 0.898 | 0.405 |
| 0.90 | 0.857 / 0.171 / 0.286 | 0.897 | 0.283 |
| 0.95 | 0.900 / 0.114 / 0.203 | 0.898 | 0.198 |

OSM alone: F1 0.898. Saying yes at every node: F1 0.810 (its precision is the share of nodes that are marked).

### crosswalk_s1_seed3: detector vs OSM vs truth at the val threshold (node counts)

| detector | OSM | truth | nodes |
|---|---|---|---|
| yes | yes | marked | 284 |
| yes | yes | not | 46 |
| yes | no | marked | 13 |
| yes | no | not | 22 |
| no | yes | marked | 14 |
| no | yes | not | 5 |
| no | no | marked | 4 |
| no | no | not | 75 |

### crosswalk_s3_seed2: node F1 across thresholds (val pick 0.39)

| threshold | detector P / R / F1 | detector OR OSM F1 | detector AND OSM F1 |
|---|---|---|---|
| 0.05 | 0.794 / 0.965 / 0.871 | 0.874 | 0.895 |
| 0.10 | 0.805 / 0.959 / 0.875 | 0.880 | 0.893 |
| 0.15 | 0.806 / 0.952 / 0.873 | 0.882 | 0.889 |
| 0.20 | 0.806 / 0.952 / 0.873 | 0.882 | 0.889 |
| 0.25 | 0.809 / 0.952 / 0.875 | 0.883 | 0.889 |
| 0.30 | 0.809 / 0.943 / 0.871 | 0.881 | 0.887 |
| 0.35 | 0.814 / 0.943 / 0.874 | 0.884 | 0.887 |
| 0.40 | 0.823 / 0.927 / 0.872 | 0.888 | 0.881 |
| 0.45 | 0.827 / 0.908 / 0.865 | 0.890 | 0.872 |
| 0.50 | 0.825 / 0.870 / 0.847 | 0.891 | 0.852 |
| 0.55 | 0.827 / 0.835 / 0.831 | 0.894 | 0.832 |
| 0.60 | 0.834 / 0.765 / 0.798 | 0.899 | 0.793 |
| 0.65 | 0.845 / 0.692 / 0.761 | 0.899 | 0.755 |
| 0.70 | 0.842 / 0.590 / 0.694 | 0.901 | 0.685 |
| 0.75 | 0.843 / 0.460 / 0.595 | 0.905 | 0.579 |
| 0.80 | 0.817 / 0.298 / 0.437 | 0.903 | 0.418 |
| 0.85 | 0.850 / 0.216 / 0.344 | 0.901 | 0.329 |
| 0.90 | 0.925 / 0.117 / 0.208 | 0.895 | 0.210 |
| 0.95 | 0.905 / 0.060 / 0.113 | 0.896 | 0.113 |

OSM alone: F1 0.898. Saying yes at every node: F1 0.810 (its precision is the share of nodes that are marked).

### crosswalk_s3_seed2: detector vs OSM vs truth at the val threshold (node counts)

| detector | OSM | truth | nodes |
|---|---|---|---|
| yes | yes | marked | 283 |
| yes | yes | not | 43 |
| yes | no | marked | 10 |
| yes | no | not | 21 |
| no | yes | marked | 15 |
| no | yes | not | 8 |
| no | no | marked | 7 |
| no | no | not | 76 |

### crosswalk_s3_seed3: node F1 across thresholds (val pick 0.31)

| threshold | detector P / R / F1 | detector OR OSM F1 | detector AND OSM F1 |
|---|---|---|---|
| 0.05 | 0.794 / 0.968 / 0.873 | 0.880 | 0.890 |
| 0.10 | 0.802 / 0.965 / 0.876 | 0.885 | 0.888 |
| 0.15 | 0.805 / 0.959 / 0.875 | 0.886 | 0.886 |
| 0.20 | 0.812 / 0.949 / 0.876 | 0.887 | 0.885 |
| 0.25 | 0.816 / 0.943 / 0.875 | 0.889 | 0.883 |
| 0.30 | 0.816 / 0.930 / 0.869 | 0.890 | 0.876 |
| 0.35 | 0.820 / 0.924 / 0.869 | 0.890 | 0.876 |
| 0.40 | 0.818 / 0.911 / 0.862 | 0.890 | 0.868 |
| 0.45 | 0.819 / 0.892 / 0.854 | 0.892 | 0.858 |
| 0.50 | 0.830 / 0.883 / 0.855 | 0.896 | 0.855 |
| 0.55 | 0.830 / 0.838 / 0.834 | 0.897 | 0.832 |
| 0.60 | 0.835 / 0.803 / 0.819 | 0.899 | 0.814 |
| 0.65 | 0.825 / 0.746 / 0.783 | 0.897 | 0.779 |
| 0.70 | 0.822 / 0.660 / 0.732 | 0.898 | 0.726 |
| 0.75 | 0.846 / 0.575 / 0.684 | 0.895 | 0.681 |
| 0.80 | 0.858 / 0.460 / 0.599 | 0.898 | 0.590 |
| 0.85 | 0.861 / 0.314 / 0.460 | 0.893 | 0.460 |
| 0.90 | 0.857 / 0.190 / 0.312 | 0.898 | 0.304 |
| 0.95 | 0.853 / 0.092 / 0.166 | 0.899 | 0.161 |

OSM alone: F1 0.898. Saying yes at every node: F1 0.810 (its precision is the share of nodes that are marked).

### crosswalk_s3_seed3: detector vs OSM vs truth at the val threshold (node counts)

| detector | OSM | truth | nodes |
|---|---|---|---|
| yes | yes | marked | 280 |
| yes | yes | not | 44 |
| yes | no | marked | 13 |
| yes | no | not | 22 |
| no | yes | marked | 18 |
| no | yes | not | 7 |
| no | no | marked | 4 |
| no | no | not | 75 |

