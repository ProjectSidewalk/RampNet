# Crosswalk validator vs Jon's votes, Juneau AI Crosswalk labels

votes file: `votes_snapshot.csv` (350 rows, 348 labels after keeping each label's latest vote); predictions: `predictions_r3.csv`; map: `confidence_precision_map.json`

Crops: {'ok': 3579}; scored 3579 of 3579 AI Crosswalk labels.

## All 3,579 AI Crosswalk labels (no reference): what the production rule would do

| slice | n | AI Agree vote | Unsure (Agree verdict below gate) | Unsure (Disagree verdict, capped) |
|---|---:|---:|---:|---:|
| all | 3579 | 2283 (64%) | 225 (6%) | 1071 (30%) |
| rig = GoPro Max | 292 | 162 (55%) | 9 (3%) | 121 (41%) |
| rig = Trimble mx7 | 3287 | 2121 (65%) | 216 (7%) | 950 (29%) |

## First batch only (the 236 queue/gallery votes, NOT random)

The set the brief describes: votes from the validation queue and the gallery in geographic clusters.

n = 232 decided votes (133 Agree, 99 Disagree) on 158 panos; 4 Unsure votes excluded.

- **AUC(incorrect), round 3 (b8498b8):** 0.985 [0.963, 1.000]
- AUC(incorrect), 2025-08 model (b7936e5): 0.991 [0.973, 1.000]

Production rule (gate 0.92 on the shipped map):

| Jon \ AI | Agree vote | Unsure (Agree verdict) | Unsure (Disagree verdict) | total |
|---|---:|---:|---:|---:|
| Agree | 124 | 6 | 3 | 133 |
| Disagree | 0 | 0 | 99 | 99 |

- Agree recall (Jon-Agree labels that get an AI Agree vote): 124/133 = 0.93 [0.88, 0.96]
- Precision of AI Agree votes (Jon agrees): 124/124 = 1.00 [0.97, 1.00]
- Jon-Disagree labels that get an AI Agree vote (false approvals): 0/99 = 0.00 [0.00, 0.04]
- Raw Disagree verdicts (argmax, ungated): 102; precision 99/102 = 0.97 [0.92, 0.99]; recall of Jon-Disagree 99/99 = 1.00 [0.96, 1.00]
- Raw-verdict accuracy 0.987; Jon-Agree labels with a raw Disagree verdict: 3/133 = 0.02 [0.01, 0.06]

Filter sweep (drop a label when the validator's log-odds of `incorrect` exceeds t; equivalent to a p_incorrect threshold):

| p_incorrect > | Jon-Disagree removed | Jon-Agree removed (lost) | remaining precision |
|---:|---:|---:|---:|
| 0.5 | 99/99 | 3/133 | 1.000 (n=130) |
| 0.1 | 99/99 | 5/133 | 1.000 (n=128) |
| 0.01 | 99/99 | 7/133 | 1.000 (n=126) |
| 0.001 | 99/99 | 23/133 | 1.000 (n=110) |
| 0.0001 | 99/99 | 45/133 | 1.000 (n=88) |

## All Jon-voted labels (first batch + later votes incl. the random sample)

Pooled: the non-random first batch plus every later vote (most of them on the seeded random sample). Dominated by the non-random batch; use for failure modes, not city-wide rates.

n = 344 decided votes (208 Agree, 136 Disagree) on 250 panos; 4 Unsure votes excluded.

- **AUC(incorrect), round 3 (b8498b8):** 0.980 [0.964, 0.993]
- AUC(incorrect), 2025-08 model (b7936e5): 0.988 [0.975, 0.997]

Production rule (gate 0.92 on the shipped map):

| Jon \ AI | Agree vote | Unsure (Agree verdict) | Unsure (Disagree verdict) | total |
|---|---:|---:|---:|---:|
| Agree | 188 | 12 | 8 | 208 |
| Disagree | 5 | 1 | 130 | 136 |

- Agree recall (Jon-Agree labels that get an AI Agree vote): 188/208 = 0.90 [0.86, 0.94]
- Precision of AI Agree votes (Jon agrees): 188/193 = 0.97 [0.94, 0.99]
- Jon-Disagree labels that get an AI Agree vote (false approvals): 5/136 = 0.04 [0.02, 0.08]
- Raw Disagree verdicts (argmax, ungated): 138; precision 130/138 = 0.94 [0.89, 0.97]; recall of Jon-Disagree 130/136 = 0.96 [0.91, 0.98]
- Raw-verdict accuracy 0.959; Jon-Agree labels with a raw Disagree verdict: 8/208 = 0.04 [0.02, 0.07]

Filter sweep (drop a label when the validator's log-odds of `incorrect` exceeds t; equivalent to a p_incorrect threshold):

| p_incorrect > | Jon-Disagree removed | Jon-Agree removed (lost) | remaining precision |
|---:|---:|---:|---:|
| 0.5 | 130/136 | 8/208 | 0.971 (n=206) |
| 0.1 | 131/136 | 12/208 | 0.975 (n=201) |
| 0.01 | 131/136 | 18/208 | 0.974 (n=195) |
| 0.001 | 134/136 | 37/208 | 0.988 (n=173) |
| 0.0001 | 136/136 | 71/208 | 1.000 (n=137) |

### By rig

| rig | n | Disagree | AUC [95% pano CI] | AI Agree votes | of which Jon-Disagree |
|---|---:|---:|---:|---:|---:|
| GoPro Max | 65 | 39 | 1.000 [1.000, 1.000] | 26 | 0 |
| Trimble mx7 | 279 | 97 | 0.978 [0.959, 0.992] | 167 | 5 |

### By false-positive class (Jon's comment on Disagree votes)

| FP class | n | raw Disagree verdict | AI Agree vote (leak) | median p_incorrect | AUC vs Jon-Agree labels |
|---|---:|---:|---:|---:|---:|
| (no comment) | 2 | 2 | 0 | 0.977 | 0.978 |
| brick/paver sidewalk | 40 | 40 | 0 | 1 | 0.994 |
| capture-car roof | 52 | 51 | 1 | 0.989 | 0.976 |
| curb ramp | 3 | 3 | 0 | 1 | 0.994 |
| driveway crossing | 3 | 3 | 0 | 1 | 1.000 |
| fence | 1 | 1 | 0 | 0.998 | 0.990 |
| fence/tree shadows | 11 | 11 | 0 | 1 | 0.991 |
| median | 1 | 1 | 0 | 0.612 | 0.962 |
| other car roof | 1 | 1 | 0 | 0.997 | 0.990 |
| other road markings (chalk/paint, bike lane) | 4 | 4 | 0 | 0.999 | 0.988 |
| parking-lot / access-aisle stripes | 5 | 2 | 2 | 0.134 | 0.921 |
| placement off the crosswalk | 1 | 0 | 1 | 0.00273 | 0.865 |
| plain sidewalk | 1 | 1 | 0 | 1 | 0.995 |
| reflection in a window | 1 | 0 | 1 | 0.000117 | 0.668 |
| stairs | 6 | 6 | 0 | 0.997 | 0.986 |
| turn lane | 4 | 4 | 0 | 1 | 0.992 |
| *(Jon Agree)* | 208 | 8 | 188 | 2.38e-05 | — |

### By label dip below the horizon (capture-car roofs sit at steep dip)

| dip_deg | n | Disagree | AUC | AI Agree votes (Jon-Disagree among them) |
|---|---:|---:|---:|---:|
| (0, 15] | 160 | 46 | 0.978 | 112 (4) |
| (15, 25] | 81 | 24 | 0.992 | 50 (0) |
| (25, 40] | 90 | 56 | 0.982 | 29 (1) |
| (40, 90] | 13 | 10 | 1.000 | 2 (0) |

### Excluding capture-car roofs (a geometric mask could remove them before validation)

n = 292 (84 Disagree): AUC 0.982 [0.965, 0.994]; AI Agree votes 192, of which Jon-Disagree 4

## Seeded random sample (50 kept-stratum + 30 dropped-stratum labels)

The 80 labels of `random_sample_2026-11-17.json` (50 from the 2,717-label OSM-proximity keep set, 30 from its 862-label drop set). Pooled counts below are NOT reweighted to the population; the per-stratum lines are each a simple random sample of their stratum. Five dropped-stratum labels had first-batch votes before the draw.

n = 79 decided votes (49 Agree, 30 Disagree) on 75 panos; 0 Unsure votes excluded.

- **AUC(incorrect), round 3 (b8498b8):** 0.959 [0.910, 0.989]
- AUC(incorrect), 2025-08 model (b7936e5): 0.967 [0.926, 0.993]

Production rule (gate 0.92 on the shipped map):

| Jon \ AI | Agree vote | Unsure (Agree verdict) | Unsure (Disagree verdict) | total |
|---|---:|---:|---:|---:|
| Agree | 40 | 5 | 4 | 49 |
| Disagree | 4 | 0 | 26 | 30 |

- Agree recall (Jon-Agree labels that get an AI Agree vote): 40/49 = 0.82 [0.69, 0.90]
- Precision of AI Agree votes (Jon agrees): 40/44 = 0.91 [0.79, 0.96]
- Jon-Disagree labels that get an AI Agree vote (false approvals): 4/30 = 0.13 [0.05, 0.30]
- Raw Disagree verdicts (argmax, ungated): 30; precision 26/30 = 0.87 [0.70, 0.95]; recall of Jon-Disagree 26/30 = 0.87 [0.70, 0.95]
- Raw-verdict accuracy 0.899; Jon-Agree labels with a raw Disagree verdict: 4/49 = 0.08 [0.03, 0.19]

Filter sweep (drop a label when the validator's log-odds of `incorrect` exceeds t; equivalent to a p_incorrect threshold):

| p_incorrect > | Jon-Disagree removed | Jon-Agree removed (lost) | remaining precision |
|---:|---:|---:|---:|
| 0.5 | 26/30 | 4/49 | 0.918 (n=49) |
| 0.1 | 26/30 | 6/49 | 0.915 (n=47) |
| 0.01 | 26/30 | 9/49 | 0.909 (n=44) |
| 0.001 | 29/30 | 11/49 | 0.974 (n=39) |
| 0.0001 | 30/30 | 18/49 | 1.000 (n=31) |

| stratum (population) | n decided | Jon Disagree | AUC | AI Agree votes | of which Jon-Disagree | Jon-Agree with AI Agree vote |
|---|---:|---:|---:|---:|---:|---:|
| kept (2717) | 49 | 5/49 = 0.10 [0.04, 0.22] | 0.968 | 38 | 1 | 37/44 |
| dropped (862) | 30 | 25/30 = 0.83 [0.66, 0.93] | 0.888 | 6 | 3 | 3/5 |

Stratified estimates for all 3,579 AI Crosswalk labels (population-weighted; 95% stratified bootstrap; small strata, read as rough):

| quantity | estimate [95% CI] |
|---|---:|
| Jon-Disagree share (RampNet false-positive rate) | 0.278 [0.208, 0.356] |
| share of labels that get an AI Agree vote | 0.637 [0.535, 0.730] |
| precision of AI Agree votes | 0.938 [0.868, 0.988] |
| Agree recall (correct labels that get an AI Agree vote) | 0.828 [0.711, 0.928] |
| raw Disagree verdict precision | 0.814 [0.653, 0.970] |
| raw Disagree verdict recall | 0.858 [0.714, 0.973] |
| precision of labels left after dropping raw Disagree verdicts | 0.944 [0.883, 0.989] |

Random-sample labels scored: 80 of 80; votes so far: 79

