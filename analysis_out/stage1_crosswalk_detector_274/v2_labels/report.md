# First crosswalk detector: pre-registered scores (RampNet#274)

Scored 2026-11-06T00:14:47Z by `score_274.py score`. Protocol: `docs/stage1_crosswalk_sdot_274.md`.

## Models

| label | sha256 | val-F1 threshold |
|---|---|---|
| crosswalk_s1 | `6f467ff4afd7d092…` | 0.21 |
| released | `024a987cabed7cc9…` | 0.10 |
| label_ceiling | `n/a…` | 0.05 |

## (a) Pixels, test split, at each model's own val-F1 threshold

Precision over complete panos (dense_ignore + verified_empty); recall over every test positive. 95% Wilson intervals.

| model | precision | recall | F1 | TP (complete) | FP | ignored | positives |
|---|---|---|---|---|---|---|---|
| crosswalk_s1 | 0.850 [0.820, 0.875] | 0.845 [0.823, 0.865] | 0.847 | 975 (565) | 100 | 118 | 1154 |
| released | 0.220 [0.197, 0.245] | 0.370 [0.343, 0.398] | 0.276 | 427 (250) | 888 | 258 | 1154 |
| label_ceiling | 1.000 [0.994, 1.000] | 1.000 [0.997, 1.000] | 1.000 | 1154 (671) | 0 | 0 | 1154 |

### Test, per city

| model | group | panos | precision | recall | F1 | FP | positives |
|---|---|---|---|---|---|---|---|
| crosswalk_s1 | amsterdam | 64 | n/a | 0.840 | n/a | 0 | 131 |
| crosswalk_s1 | burnaby | 49 | n/a | 0.874 | n/a | 0 | 87 |
| crosswalk_s1 | seattle-wa | 379 | 0.850 | 0.846 | 0.848 | 100 | 899 |
| crosswalk_s1 | teaneck-nj | 23 | n/a | 0.757 | n/a | 0 | 37 |
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
| crosswalk_s1 | dense_ignore | 257 | 0.850 | 0.842 | 0.846 | 100 | 671 |
| crosswalk_s1 | sparse | 227 | n/a | 0.849 | n/a | 0 | 483 |
| crosswalk_s1 | verified_empty | 31 | n/a | n/a | n/a | 0 | 0 |
| released | dense_ignore | 257 | 0.221 | 0.373 | 0.278 | 879 | 671 |
| released | sparse | 227 | n/a | 0.366 | n/a | 0 | 483 |
| released | verified_empty | 31 | 0.000 | n/a | n/a | 9 | 0 |
| label_ceiling | dense_ignore | 257 | 1.000 | 1.000 | 1.000 | 0 | 671 |
| label_ceiling | sparse | 227 | n/a | 1.000 | n/a | 0 | 483 |
| label_ceiling | verified_empty | 31 | n/a | n/a | n/a | 0 | 0 |

### Test, per capture year

| model | group | panos | precision | recall | F1 | FP | positives |
|---|---|---|---|---|---|---|---|
| crosswalk_s1 | 2019-2022 | 318 | 0.856 | 0.848 | 0.852 | 51 | 676 |
| crosswalk_s1 | 2023+ | 167 | 0.842 | 0.846 | 0.844 | 48 | 428 |
| crosswalk_s1 | <=2018 | 30 | 0.875 | 0.800 | 0.836 | 1 | 50 |
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
| crosswalk_s1 | 0.21 | 0.800 | 0.806 | 0.803 |
| released | 0.10 | 0.190 | 0.368 | 0.251 |
| label_ceiling | 0.05 | 1.000 | 1.000 | 1.000 |

## (b) Per OSM node, Seattle test panos, against the map-only baseline

Nodes of `baseline_nodes.csv` within 25 m of a Seattle test pano. A node is predicted marked when a detection at the model's val-F1 threshold, raycast to flat ground at 2.5 m and kept within 25 m range, lands within 10 m. Truth: a current SDOT point within 10 m.

| model | nodes (true) | detector P / R / F1 | `osm_marked_signal` P / R / F1 | detector OR OSM F1 | passes (detector F1 ≥ OSM) |
|---|---|---|---|---|---|
| crosswalk_s1 | 463 (315) | 0.821 / 0.949 / 0.881 | 0.854 / 0.946 / 0.898 | 0.894 | no |
| released | 463 (315) | 0.727 / 0.965 / 0.829 | 0.854 / 0.946 / 0.898 | 0.840 | no |
| label_ceiling | 463 (315) | 0.882 / 0.949 / 0.914 | 0.854 / 0.946 / 0.898 | 0.893 | n/a (ceiling) |

On nodes OSM leaves untagged (`osm_class` unknown):

| model | nodes (true) | detector P / R / F1 | `osm_marked_signal` P / R / F1 |
|---|---|---|---|
| crosswalk_s1 | 38 (9) | 0.667 / 0.889 / 0.762 | 0.625 / 0.556 / 0.588 |
| released | 38 (9) | 0.375 / 1.000 / 0.545 | 0.625 / 0.556 / 0.588 |
| label_ceiling | 38 (9) | 1.000 / 0.889 / 0.941 | 0.625 / 0.556 / 0.588 |

Exploratory (not part of the pass rule): a node-bootstrap 95% interval for detector F1 minus OSM F1 (nodes are spatially correlated, so it is too narrow):

- crosswalk_s1: [-0.039, 0.005]; 1156 detections on 379 panos, 1028 placed within range
- released: [-0.094, -0.043]; 1920 detections on 379 panos, 1744 placed within range
- label_ceiling: [-0.005, 0.041]; 899 detections on 379 panos, 774 placed within range

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

