# Second-pass verifier on RampNet's candidates (#258)

**Issue:** [#258](https://github.com/ProjectSidewalk/RampNet/issues/258). The plan is the
comment of 2026-10-21 (https://github.com/ProjectSidewalk/RampNet/issues/258#issuecomment-6316627866).
**Status:** steps 1 and 2 are done (2026-10-21, 2026-10-22). The production CurbRamp validator ranks
RampNet's candidates worse than RampNet's own score on 13 of 14 splits; see Step 2.

## The idea

Two passes. Lower RampNet's threshold to admit more candidates, then score each candidate with a
crop-level model and drop the false positives. The first judge is the production CurbRamp validator
from [sidewalk-validator-ai](https://github.com/ProjectSidewalk/sidewalk-validator-ai): it works on
crops, it was trained on curb ramps, it already runs in Project Sidewalk, and it costs nothing to run.
Why now, when the #35 cascade did not transfer: see the plan comment. In short, #35's only large gains
were on `laurens_mapillary`, and #265 puts most open-rig misses below threshold rather than silent.

## Step 1: the candidate set

`scripts/analysis/verifier_candidates_258.py` writes every stored RampNet peak (score ≥ 0.05) from
the committed `analysis_out/op_cache/<split>.json` on all 14 pinned splits to
`analysis_out/verifier_258/candidates/<split>.json`, one row per peak. A row holds the pano, x, y,
RampNet score, label, a `redundant` flag, the review stratum (`group`), `source` and `camera_model`.
CPU only, no network, no imagery; `--check` re-derives every file byte for byte, and
`tests/test_verifier_candidates_258.py` covers it.

**Candidates are all peaks, not only sub-threshold ones.** A two-pass pipeline filters everything above
its lowered threshold, peaks at or above 0.55 included.

**One label per peak holds at every threshold.** Peaks are matched in score order by the shared greedy
matcher with the seam wrap, then the ignore-point fallback, exactly as `score_pano` scores the full floor
set. Applying a threshold keeps a prefix of that order, and greedy matching never revisits an earlier
assignment, so no kept peak's label changes. The script asserts this against `score_pano` at 0.05, 0.30
and 0.55 on every split, and the test checks seven thresholds on juneau.

| split | panos | GT | candidates | TP | FP (redundant) | ignored |
|---|--:|--:|--:|--:|--:|--:|
| richmond | 124 | 310 | 537 | 279 | 197 (0) | 61 |
| bend | 110 | 327 | 416 | 289 | 98 (2) | 29 |
| clovis | 125 | 195 | 392 | 183 | 185 (0) | 24 |
| morgantown | 125 | 267 | 480 | 237 | 225 (1) | 18 |
| annapolis | 125 | 294 | 419 | 265 | 139 (2) | 15 |
| paterson | 125 | 395 | 389 | 299 | 71 (0) | 19 |
| gainesville | 125 | 272 | 430 | 242 | 171 (1) | 17 |
| laurens_mapillary | 94 | 249 | 374 | 195 | 165 (0) | 14 |
| juneau | 125 | 297 | 434 | 257 | 159 (5) | 18 |
| laurens_gsv | 86 | 220 | 261 | 180 | 72 (2) | 9 |
| budapest_district5 | 125 | 300 | 586 | 227 | 330 (4) | 29 |
| sao_paulo | 125 | 281 | 512 | 244 | 218 (1) | 50 |
| bayonne | 125 | 301 | 565 | 218 | 318 (3) | 29 |
| manual_gold | 1000 | 3919 | 4709 | 3607 | 1102 (38) | 0 |
| **total** | | | **10,504** | | | |

TP at the 0.05 floor over GT is the recall ceiling any verifier can reach: 195/249 = 0.78 on
laurens_mapillary, 218/301 = 0.72 on bayonne.

**Caveats that travel with every number from these rows.**

- **TP labels on sub-threshold peaks are a lower bound.** On verdict splits the GT is anchored to
  reviewed RampNet detections exported at a 0.55 floor, so a sub-threshold peak is TP only if the
  reviewer added that ramp as a missed mark. A real ramp the reviewer did not mark is labelled `fp`, so
  verifier precision measured on these labels is a lower bound. `manual_gold`, with independent
  YOLO-format labels, does not have this bias.
- **`redundant` FPs** are second hits on a ramp a higher-scored peak already claimed. A verifier that
  says "ramp" there is right about the image and still costs an FP, so AUC is to be reported with and
  without them.
- The op_caches predate the seam-wrapped NMS (#248, opt-in), as every op_cache does.

## Leakage check: no benchmark pano could have been in the validator's training data

`scripts/analysis/validator_leakage_258.py`; output `analysis_out/verifier_258/leakage.json`;
run 2026-10-21 (CSV fetch times and sha256s are in the JSON).

**What is known about the training set.** The validator's training images were never published:
sidewalk-validator-ai's 2026-10-14 audit, finding 13, shows the HF CurbRamp dataset repo is a copy of
the Crosswalk one ([sidewalk-validator-ai#7](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/7)).
The selection code is known. At sidewalk-validator-ai `eb9bc65`, `data_generation.py` reads
`rawLabels` from 19 deployments (cdmx, la-piedad, spgg, amsterdam, blackhawk-hills, chicago,
cliffside-park, columbus, knox, mendota, newberg, oradell, pittsburgh, sea, st-louis, teaneck,
keelung, new-taipei, taipei) and selects labels by agree/disagree counts. None of the 13 benchmark
cities is on that list.

**The check is conservative by construction.** It treats every current public CurbRamp label in those
19 cities (284,494 labels on 109,371 panos) as possibly trained on, a superset of any selection.

| result | every split |
|---|---|
| exact `pano_id` overlap | **0** on all 14 |
| benchmark panos within 50 m / 500 m of a training-city CurbRamp label | **0 / 0** on all 14 |
| nearest training label | 2.3 km (manual_gold: NYC, near Cliffside Park / Teaneck); 7.9 km (paterson); every other split ≥ 79 km |

**Limits.** The live API omits labels deleted since the 2025-08 data pull. The 2025-08-16 model run's
city list is not recorded; the list above is the code's. Neither can put a benchmark pano in training
unless a deleted label sat on one, or the run used a city the code does not name.

## Step 2: the validator on every candidate (run 2026-10-22)

**Result.** The validator separates RampNet's true ramps from its false positives worse than
RampNet's own score does, on 13 of 14 splits. By the stop rule in the design below, the validator
as the ranker stops here. Below RampNet's 0.30 operating point the order flips: there the validator
ranks better than RampNet on the pooled open and reference splits, but not on manual_gold, and the
untuned two-pass rule raises F1 only on laurens_mapillary.

### How it was run

`scripts/analysis/validator_score_258.py` cuts the crop production would cut for each of the
10,504 candidates and records the validator's P(correct). The crop path is copied from the
serving code, sidewalk-ai-api `main.py` at `745e0c5`; the perspective function is copied verbatim.

- **Model:** `projectsidewalk/sidewalk-validator-ai-curbramp` at revision `969ad5a3…` (the
  `model_map.py` pin, trained 2025-08-16), `AutoImageProcessor(backend="pil")`, float32. Its classes
  are `correct` and `incorrect`.
- **Crop:** pano resized to 8192×4096 (`INTER_LINEAR`), a 90° perspective of 2048×2048 centred on the
  candidate, Depth-Anything-V2 metric (`vitb`, `vkitti`, max depth 80, repo commit `a561b84`) at the
  centre pixel, crop half-size `int(6100 / depth)`, resized to 640 on the long side (`INTER_CUBIC`).
  The depth checkpoint's sha256 is asserted against serving's pinned Hub file (`4dad67a7…`).
- **Score:** P(correct), the softmax of the `correct` class. Serving returns the argmax label and its
  confidence; the argmax is recorded too, as `pred`.
- **Imagery:** every split's panos were checked against its `imagery_manifest.json` on klone before
  scoring (the `.slurm` stops otherwise). juneau and bayonne were copied to klone for this run, from
  the main checkout and from the bayonne bundle, after the same check passed locally.
- **Compute:** klone job `42152480`, `ckpt-g2`, one L40S, 2026-10-22 06:51–07:21 local, 1,822 s =
  0.51 GPU-hours, $0, no preemption. The row is in `analysis_out/compute_log.jsonl`, and the sacct
  dump is `docs/data/compute/sacct_klone_2026-10-22_validator258.txt`. The job log is
  `docs/data/validator_258_42152480.log`.

**Differences from serving that remain:**

1. **manual_gold's panos are 4096×2048** (the HF benchmark copy), so its crops are upsampled 2× where
   serving would read native imagery. This cannot be removed without the native-resolution panos (#21),
   and it matters for reading manual_gold below.
2. **Package versions.** torch 2.14.1+cu126 and torchvision 0.29.1 match serving's
   `requirements.txt`; transformers is 5.15.1 against serving's 5.20.0, and opencv 5.0.0 / pillow
   12.3.0 match. xFormers was not installed, so DINOv2 inside Depth-Anything ran its plain attention
   path; whether the serving image has xFormers was not checked.
3. **Serving re-reads the pano per request.** This reads it once per pano and reuses it for every
   candidate on that pano; the array is the same.

The validator's own training-versus-serving mismatch (training crops used Depth-Anything ViT-L,
serving uses ViT-B) is the validator's, not this setup's. It was left as served.

### Primary: AUC of the validator against RampNet's own score

All panos, `ignored` rows dropped, redundant FPs kept. 95% intervals are a paired pano bootstrap
(B = 2000, seed 258); the pooled rows resample (split, pano) clusters. The open pool is
`laurens_mapillary`, `bayonne`, `budapest_district5`, `juneau`, `laurens_gsv`; the reference pool is
the seven US splits of #46 (both from `open_rig_misses_258.py`).

| split | TP | FP | RampNet score | validator P(correct) | validator - RampNet |
|---|--:|--:|--:|--:|--:|
| richmond | 279 | 197 | 0.956 [0.936, 0.974] | 0.860 [0.820, 0.896] | -0.096 [-0.134, -0.062] |
| bend | 289 | 98 | 0.945 [0.922, 0.967] | 0.870 [0.820, 0.914] | -0.075 [-0.127, -0.031] |
| clovis | 183 | 185 | 0.928 [0.898, 0.955] | 0.824 [0.776, 0.867] | -0.104 [-0.153, -0.057] |
| morgantown | 237 | 225 | 0.951 [0.931, 0.970] | 0.888 [0.855, 0.922] | -0.063 [-0.100, -0.025] |
| annapolis | 265 | 139 | 0.939 [0.917, 0.961] | 0.881 [0.843, 0.915] | -0.059 [-0.102, -0.018] |
| paterson | 299 | 71 | 0.954 [0.927, 0.977] | 0.896 [0.843, 0.941] | -0.059 [-0.110, -0.015] |
| gainesville | 242 | 171 | 0.923 [0.895, 0.946] | 0.830 [0.784, 0.873] | -0.093 [-0.142, -0.044] |
| laurens_mapillary | 195 | 165 | 0.847 [0.804, 0.887] | 0.902 [0.865, 0.935] | 0.055 [0.008, 0.102] |
| juneau | 257 | 159 | 0.935 [0.908, 0.960] | 0.833 [0.793, 0.868] | -0.103 [-0.145, -0.064] |
| laurens_gsv | 180 | 72 | 0.874 [0.809, 0.931] | 0.812 [0.739, 0.875] | -0.062 [-0.120, -0.008] |
| budapest_district5 | 227 | 330 | 0.881 [0.847, 0.913] | 0.774 [0.727, 0.820] | -0.107 [-0.155, -0.061] |
| sao_paulo | 244 | 218 | 0.922 [0.894, 0.946] | 0.875 [0.839, 0.908] | -0.047 [-0.085, -0.011] |
| bayonne | 218 | 318 | 0.858 [0.823, 0.889] | 0.803 [0.762, 0.838] | -0.056 [-0.100, -0.011] |
| manual_gold | 3607 | 1102 | 0.953 [0.945, 0.961] | 0.849 [0.834, 0.865] | -0.104 [-0.119, -0.090] |
| **open pooled** | 1077 | 1044 | 0.878 [0.861, 0.894] | 0.823 [0.802, 0.843] | -0.054 [-0.077, -0.033] |
| **reference pooled** | 1794 | 1086 | 0.946 [0.938, 0.954] | 0.872 [0.856, 0.886] | -0.074 [-0.091, -0.060] |

Dropping the redundant FPs, or restricting to the unbiased subset (`group != "top"`), changes no sign:
the largest shift in the difference is 0.017 (juneau, unbiased subset without redundant FPs), and the
open pool moves from −0.054 to between −0.046 and −0.051. All four variants are in `analysis_out/verifier_258/validator_auc.json`.

**laurens_mapillary is the one split where the validator ranks better**, +0.055 [0.008, 0.102]. It is
also the split where RampNet's own ranking is weakest (0.847) and where the recall gap (#265) is largest.

### Below the operating point

The same AUC over only the candidates under RampNet's 0.30 operating point, the ones a lowered
threshold would add. **This stratum was added after richmond, the first split scored, and before any
other split's scores were read** (commit `b8287cc`); it was not in the design posted on #258.

| split | TP | FP | RampNet score | validator P(correct) | validator - RampNet |
|---|--:|--:|--:|--:|--:|
| richmond | 22 | 169 | 0.667 [0.534, 0.796] | 0.751 [0.637, 0.867] | 0.084 [-0.039, 0.222] |
| bend | 20 | 76 | 0.603 [0.468, 0.734] | 0.844 [0.727, 0.939] | 0.241 [0.091, 0.372] |
| clovis | 23 | 157 | 0.765 [0.652, 0.872] | 0.746 [0.653, 0.831] | -0.020 [-0.182, 0.123] |
| morgantown | 22 | 198 | 0.629 [0.519, 0.739] | 0.785 [0.675, 0.880] | 0.157 [-0.012, 0.315] |
| annapolis | 27 | 113 | 0.673 [0.570, 0.768] | 0.858 [0.783, 0.919] | 0.185 [0.081, 0.294] |
| paterson | 15 | 56 | 0.739 [0.618, 0.844] | 0.779 [0.606, 0.915] | 0.040 [-0.171, 0.224] |
| gainesville | 32 | 136 | 0.724 [0.612, 0.823] | 0.729 [0.631, 0.820] | 0.005 [-0.127, 0.146] |
| laurens_mapillary | 64 | 148 | 0.674 [0.584, 0.755] | 0.892 [0.846, 0.933] | 0.218 [0.135, 0.309] |
| juneau | 31 | 137 | 0.702 [0.591, 0.815] | 0.728 [0.641, 0.806] | 0.026 [-0.122, 0.162] |
| laurens_gsv | 36 | 60 | 0.727 [0.621, 0.834] | 0.740 [0.639, 0.835] | 0.013 [-0.113, 0.136] |
| budapest_district5 | 34 | 250 | 0.619 [0.538, 0.710] | 0.698 [0.595, 0.792] | 0.080 [-0.041, 0.188] |
| sao_paulo | 20 | 163 | 0.724 [0.603, 0.831] | 0.846 [0.740, 0.938] | 0.121 [-0.047, 0.268] |
| bayonne | 68 | 269 | 0.767 [0.702, 0.827] | 0.762 [0.694, 0.824] | -0.005 [-0.094, 0.082] |
| manual_gold | 111 | 764 | 0.709 [0.656, 0.761] | 0.705 [0.656, 0.752] | -0.004 [-0.071, 0.065] |
| **open pooled** | 233 | 864 | 0.698 [0.659, 0.738] | 0.768 [0.735, 0.801] | 0.070 [0.024, 0.117] |
| **reference pooled** | 161 | 905 | 0.689 [0.645, 0.730] | 0.782 [0.741, 0.821] | 0.092 [0.036, 0.148] |

Pooled, the validator ranks the sub-threshold candidates better than RampNet does: +0.070 on the open
splits and +0.092 on the reference splits, both intervals above zero. Per split, the intervals clear
zero only on bend, annapolis and laurens_mapillary. On manual_gold, the only split with independent
labels, the difference is −0.004 [−0.071, 0.065].

Two readings of the manual_gold row, which this run cannot separate:

- **Label bias would work against the validator on the verdict splits, not for it.** There a
  sub-threshold peak is TP only if the reviewer added that ramp as a missed mark, so a real ramp the
  reviewer did not mark is labelled FP. A validator that calls it a ramp is penalized for being right.
  So the verdict-split gain is not explained by that bias.
- **manual_gold's crops are upsampled 2×** (above), which the verdict splits' are not. A blurrier crop
  could cost the validator exactly where the decision is hardest. Native-resolution manual_gold panos
  (#21) would settle it.

### Secondary: the untuned two-pass rule

RampNet alone at 0.30 against RampNet at a lowered threshold followed by the validator's own verdict
(`pred == "correct"`, P(correct) > 0.5, what production returns). Nothing is fitted on these rows.
All panos; F1 (precision / recall).

| split | GT | rampnet@0.30 | rampnet@0.05+validator | rampnet@0.30+validator |
|---|--:|--:|--:|--:|
| richmond | 310 | 0.864 (0.90 / 0.83) | 0.766 (0.70 / 0.85) | 0.847 (0.92 / 0.79) |
| bend | 327 | 0.871 (0.92 / 0.82) | 0.854 (0.86 / 0.85) | 0.862 (0.94 / 0.80) |
| clovis | 195 | 0.836 (0.85 / 0.82) | 0.738 (0.71 / 0.77) | 0.793 (0.92 / 0.70) |
| morgantown | 267 | 0.845 (0.89 / 0.81) | 0.801 (0.78 / 0.82) | 0.840 (0.94 / 0.76) |
| annapolis | 294 | 0.853 (0.90 / 0.81) | 0.828 (0.80 / 0.86) | 0.850 (0.95 / 0.77) |
| paterson | 395 | 0.818 (0.95 / 0.72) | 0.808 (0.90 / 0.74) | 0.815 (0.96 / 0.71) |
| gainesville | 272 | 0.812 (0.86 / 0.77) | 0.777 (0.74 / 0.81) | 0.797 (0.89 / 0.72) |
| laurens_mapillary | 249 | 0.660 (0.89 / 0.53) | 0.739 (0.84 / 0.66) | 0.624 (0.93 / 0.47) |
| juneau | 297 | 0.829 (0.91 / 0.76) | 0.764 (0.79 / 0.74) | 0.780 (0.93 / 0.67) |
| laurens_gsv | 220 | 0.766 (0.92 / 0.65) | 0.770 (0.84 / 0.71) | 0.726 (0.94 / 0.59) |
| budapest_district5 | 300 | 0.674 (0.71 / 0.64) | 0.585 (0.54 / 0.63) | 0.641 (0.76 / 0.56) |
| sao_paulo | 281 | 0.800 (0.80 / 0.80) | 0.754 (0.73 / 0.78) | 0.788 (0.87 / 0.72) |
| bayonne | 301 | 0.600 (0.75 / 0.50) | 0.589 (0.63 / 0.55) | 0.551 (0.88 / 0.40) |
| manual_gold | 3919 | 0.902 (0.91 / 0.89) | 0.864 (0.84 / 0.89) | 0.894 (0.93 / 0.86) |

The rule lowers F1 on 12 of 14 splits. It raises it on laurens_mapillary, 0.660 → 0.739 (recall
0.53 → 0.66), and leaves laurens_gsv flat (0.766 → 0.770). The validator calls most candidates correct
(80% of richmond's candidates, 87% of manual_gold's), so its own 0.5 cut is too permissive as a filter
at a 0.05 floor. Applied on top of 0.30 it removes more TPs than FPs everywhere.

### What this does and does not settle

- **Settled:** as a replacement for RampNet's score, the production validator is worse on 13 of 14
  splits, by 0.05–0.11 AUC. The design's stop condition is met.
- **Not settled:** whether a combined rule helps, i.e. keep everything RampNet scores ≥ 0.30 and admit a
  sub-threshold candidate only when P(correct) clears a high bar. The below-operating-point table says
  the validator carries information RampNet's score lacks in that band. Testing that rule needs a
  threshold fitted without touching the split it is scored on (leave-one-split-out), and it goes beyond
  the stop rule as written, so it is proposed here, not run.
- **Not run:** manual_gold at native resolution (#21); the other validators in the plan.

## Step 2 design, as posted before the run

Score every candidate with the deployed CurbRamp validator, **reproducing the serving path** in
[sidewalk-ai-api](https://github.com/ProjectSidewalk/sidewalk-ai-api) (`main.py`, `sidewalk_ai_api/`,
read at `745e0c5`), because the serving path is what Project Sidewalk deploys.

- **Primary metric:** AUC of P(correct) against RampNet's own score on the same candidates, per split,
  pano-bootstrap intervals; with and without `redundant` rows; on all panos and on the unbiased subset.
  If it does not beat RampNet's score on the open-rig splits, stop and write it up.
- **Where:** klone, one GPU, $0. About 10.5k crops.
- **Open items then, resolved as above:** bayonne had no local panos (copied from the bundle);
  manual_gold's local panos are 4096 wide (run, mismatch recorded); depth model ViT-B as served.
- **Mapillary and Panoramax panos** (5760–12288 px wide) are resized to 8192 exactly as serving resizes
  a cached scrape. The validator never saw these rigs in training; that is the transfer being tested,
  not a defect of the setup.

## Reproduce

```bash
python scripts/analysis/verifier_candidates_258.py           # CPU, seconds
python scripts/analysis/verifier_candidates_258.py --check
python scripts/analysis/validator_leakage_258.py             # 19 GETs to the public API, ~1 min

# step 2, GPU (klone; see the header of the .slurm for the inputs and their paths)
PY=... PANOS_ROOT=... DAV2_DIR=... DAV2_CKPT=... HF_HOME=... \
    sbatch -A ckpt-makelab -p ckpt-g2 --export=ALL scripts/analysis/validator_score_258.slurm
python scripts/analysis/validator_auc_258.py                 # CPU, ~40 s
python scripts/analysis/validator_auc_258.py --check
pytest -q tests/test_verifier_candidates_258.py tests/test_validator_auc_258.py
```

The scores in `analysis_out/verifier_258/validator_scores/` are committed, so the read-out re-derives
from a clean clone with no GPU. Re-running step 2 needs the benchmark panos: nine splits are on HF
(`rampnet-benchmark`); laurens_gsv, laurens_mapillary, bayonne and juneau are not uploaded yet
(`docs/replication.md`), and manual_gold at native resolution is #21.
