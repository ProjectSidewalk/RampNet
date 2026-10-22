# Second-pass verifier on RampNet's candidates (#258)

**Issue:** [#258](https://github.com/ProjectSidewalk/RampNet/issues/258). The plan is the
comment of 2026-10-21 (https://github.com/ProjectSidewalk/RampNet/issues/258#issuecomment-6316627866).
**Status:** steps 1, 2 and 3 are done (2026-10-21, 2026-10-22). On the pre-registered band
(RampNet score in [0.05, 0.55)) the production CurbRamp validator does not beat RampNet's score on the
open pool (+0.017 [−0.024, 0.059] AUC), is worse on manual_gold, and is better on laurens_mapillary.
Fused with RampNet's score (fitted on sao_paulo only), it buys +0.024 [+0.002, +0.054] recall at
matched precision on the open pool and +0.007 [−0.000, +0.012] over all 13 held-out splits, led by
laurens_mapillary. See Steps 2 and 3.

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

**This departs from the posted step 1**, which defined the candidates as RampNet peaks with score in
[0.05, 0.55). All peaks were kept so a pipeline that re-judges every detection could be scored on the
same rows. The cost is that over the full range the candidate set was selected by RampNet's own
score, which favours RampNet in any AUC comparison, so step 2 reads its stop rule on the posted band
and reports the all-peaks AUC only as a secondary table.

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
  YOLO-format labels, does not have this bias. For the AUC comparison the consequence runs one way:
  an unmarked real ramp is scored high by the validator and low by RampNet, so the bias penalizes the
  validator, which makes any sub-threshold gain it shows conservative.
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

**Result, on the pre-registered band.** The design's stop rule is read on its own candidate set,
RampNet peaks in [0.05, 0.55). There the validator does **not** beat RampNet's score on the open
pool: +0.017 [−0.024, 0.059] AUC, an interval across zero. It is worse on manual_gold, the only split
with independent labels: −0.074 [−0.117, −0.031]. It is better on laurens_mapillary, +0.129
[0.066, 0.198], and bayonne is flat at −0.023 [−0.088, 0.045]. So, read literally, the stop
condition is met on the open pool and on manual_gold and not met on laurens_mapillary. Below the
0.30 operating point, a stratum added after the design, the open-pool gain is larger and its interval
clears zero, +0.076 [0.027, 0.129], while manual_gold stays flat (−0.004). The richmond resolution
control (below) shows upsampling does not explain most of manual_gold's row; resolution is a partial
contributor at most. Step 3 was
run as the PR #269 review asked; its result is in the next section.

**All 14 splits were scored, not only the five the design named.** The design's step 2 named the
open-rig splits for the stop rule. The run scored all 14 and fitted nothing on any of them: the AUCs
below are read-outs, not fits, so every split except sao_paulo (the dev split) was still held out
for step 3.

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
   serving would read native imagery. Native-resolution panos are #21; the richmond control below
   measures what the downsample costs on a verdict split.
2. **Package versions.** torch 2.14.1+cu126 and torchvision 0.29.1 match serving's
   `requirements.txt`; transformers is 5.15.1 against serving's 5.20.0, and opencv 5.0.0 / pillow
   12.3.0 match. xFormers was not installed, so DINOv2 inside Depth-Anything ran its plain attention
   path; whether the serving image has xFormers was not checked.
3. **Serving re-reads the pano per request.** This reads it once per pano and reuses it for every
   candidate on that pano; the array is the same.

The validator's own training-versus-serving mismatch (training crops used Depth-Anything ViT-L,
serving uses ViT-B) is the validator's, not this setup's. It was left as served.

### How the AUC tables are read

AUC of the validator's P(correct) against RampNet's own score, on the same candidates, `ignored`
rows dropped. 95% intervals are a paired pano bootstrap (B = 2000, seed 258); the pooled rows
resample (split, pano) clusters. All numbers are in `analysis_out/verifier_258/validator_auc.json`
(`scripts/analysis/validator_auc_258.py`).

- **Pools.** The open pool is the four non-GSV splits: laurens_mapillary, bayonne,
  budapest_district5 and juneau. laurens_gsv is GSV, so it is not pooled; it is read beside
  laurens_mapillary as the same-town control, as #265 does. The first version of this read-out
  pooled laurens_gsv in (the `OPEN_SPLITS` tuple of `open_rig_misses_258.py`); that pool is kept in
  the JSON as `open_with_laurens_gsv` so its numbers stay re-derivable, and it is not used below.
  #265's own tables pool only the GoPro Max pair, so parity with #265 does not need the old pool; the
  GoPro Max pool (laurens_mapillary + bayonne) is in the JSON as `gopro_max`. The reference pool is
  the seven US splits of #46.
- **Pooled rows are the inferential ones; per-split rows are descriptive.** There are 14 splits ×
  4 strata × 2 subsets, so a per-split interval that clears zero is not evidence by itself.
- **Label bias works against the validator here, so a sub-threshold gain is conservative.** A real
  ramp the reviewer did not mark is labelled FP (step 1). The validator tends to score such a ramp
  high and RampNet, by construction of the band, scores it low, so the mislabel costs the validator's
  AUC and not RampNet's.

### Primary: the pre-registered band, RampNet score in [0.05, 0.55)

All panos, redundant FPs kept.

| split | TP | FP | RampNet score | validator P(correct) | validator − RampNet |
|---|--:|--:|--:|--:|--:|
| richmond | 41 | 188 | 0.769 [0.687, 0.846] | 0.740 [0.639, 0.835] | −0.029 [−0.141, 0.079] |
| bend | 42 | 93 | 0.726 [0.621, 0.819] | 0.810 [0.728, 0.882] | 0.084 [−0.030, 0.198] |
| clovis | 44 | 172 | 0.827 [0.743, 0.896] | 0.743 [0.658, 0.833] | −0.084 [−0.206, 0.042] |
| morgantown | 42 | 220 | 0.752 [0.659, 0.830] | 0.796 [0.722, 0.865] | 0.044 [−0.069, 0.160] |
| annapolis | 48 | 133 | 0.733 [0.650, 0.809] | 0.864 [0.800, 0.918] | 0.131 [0.036, 0.218] |
| paterson | 30 | 63 | 0.806 [0.705, 0.887] | 0.793 [0.681, 0.884] | −0.013 [−0.133, 0.105] |
| gainesville | 59 | 161 | 0.764 [0.700, 0.831] | 0.743 [0.663, 0.822] | −0.021 [−0.111, 0.083] |
| laurens_mapillary | 98 | 154 | 0.763 [0.693, 0.825] | 0.893 [0.853, 0.930] | 0.129 [0.066, 0.198] |
| laurens_gsv (same-town control) | 79 | 65 | 0.839 [0.748, 0.915] | 0.761 [0.679, 0.840] | −0.078 [−0.178, 0.016] |
| bayonne | 104 | 293 | 0.792 [0.747, 0.839] | 0.769 [0.718, 0.823] | −0.023 [−0.088, 0.045] |
| budapest_district5 | 74 | 308 | 0.730 [0.660, 0.799] | 0.705 [0.638, 0.772] | −0.025 [−0.103, 0.051] |
| juneau | 53 | 152 | 0.766 [0.681, 0.853] | 0.711 [0.640, 0.778] | −0.054 [−0.161, 0.053] |
| sao_paulo (dev) | 61 | 199 | 0.812 [0.747, 0.870] | 0.813 [0.740, 0.880] | 0.001 [−0.098, 0.097] |
| manual_gold | 278 | 944 | 0.785 [0.752, 0.814] | 0.711 [0.679, 0.743] | −0.074 [−0.117, −0.031] |
| **open pooled** (4 non-GSV) | 329 | 907 | 0.746 [0.712, 0.779] | 0.763 [0.732, 0.795] | **0.017 [−0.024, 0.059]** |
| **GoPro Max pooled** | 202 | 447 | 0.773 [0.734, 0.812] | 0.825 [0.793, 0.860] | 0.052 [0.005, 0.101] |
| **reference pooled** | 306 | 1030 | 0.770 [0.742, 0.799] | 0.784 [0.753, 0.815] | 0.014 [−0.030, 0.057] |

On the unbiased subset (`group != "top"`) the open pool reads +0.022 [−0.017, 0.060], the GoPro Max
pool +0.060 [0.011, 0.110] and the reference pool +0.010 [−0.034, 0.053]; no pooled sign changes.

**The gain is laurens_mapillary, not "open rigs".** Within the GoPro Max pair, laurens_mapillary is
+0.129 and bayonne −0.023 on this band, and +0.218 [0.135, 0.309] against −0.005 [−0.094, 0.082]
below 0.30. The GoPro Max pool clears zero only because laurens_mapillary is in it. bayonne, the
other GoPro Max split (Panoramax), does not show the effect in the AUC, and the #35 cascade also
gained only on laurens_mapillary. So the evidence is one town. Candidate explanations, none tested
here: Laurens is where RampNet's own ranking is weakest (0.847 AUC over all peaks, the lowest of the
14), so a second opinion has the most room there; and #265 found sub-threshold misses dominate
there. laurens_gsv, the same town on GSV, does not show the gain (−0.078 [−0.178, 0.016]), so it
goes with the rig, not the streets.

### Below the operating point, RampNet score in [0.05, 0.30)

The candidates a lowered threshold adds below the 0.30 operating point, which moved from 0.55 after
the plan's wording. **This stratum was added after richmond, the first split scored, and before any
other split's scores were read** (commit `b8287cc`); it was not in the design posted on #258. At that
commit richmond's difference was +0.084 [−0.039, 0.222], not distinguishable from zero, so the
stratum was not added because the first split looked favourable.

| split | TP | FP | RampNet score | validator P(correct) | validator − RampNet |
|---|--:|--:|--:|--:|--:|
| richmond | 22 | 169 | 0.667 [0.534, 0.796] | 0.751 [0.637, 0.867] | 0.084 [−0.039, 0.222] |
| bend | 20 | 76 | 0.603 [0.468, 0.734] | 0.844 [0.727, 0.939] | 0.241 [0.091, 0.372] |
| clovis | 23 | 157 | 0.765 [0.652, 0.872] | 0.746 [0.653, 0.831] | −0.020 [−0.182, 0.123] |
| morgantown | 22 | 198 | 0.629 [0.519, 0.739] | 0.785 [0.675, 0.880] | 0.157 [−0.012, 0.315] |
| annapolis | 27 | 113 | 0.673 [0.570, 0.768] | 0.858 [0.783, 0.919] | 0.185 [0.081, 0.294] |
| paterson | 15 | 56 | 0.739 [0.618, 0.844] | 0.779 [0.606, 0.915] | 0.040 [−0.171, 0.224] |
| gainesville | 32 | 136 | 0.724 [0.612, 0.823] | 0.729 [0.631, 0.820] | 0.005 [−0.127, 0.146] |
| laurens_mapillary | 64 | 148 | 0.674 [0.584, 0.755] | 0.892 [0.846, 0.933] | 0.218 [0.135, 0.309] |
| laurens_gsv (same-town control) | 36 | 60 | 0.727 [0.621, 0.834] | 0.740 [0.639, 0.835] | 0.013 [−0.113, 0.136] |
| bayonne | 68 | 269 | 0.767 [0.702, 0.827] | 0.762 [0.694, 0.824] | −0.005 [−0.094, 0.082] |
| budapest_district5 | 34 | 250 | 0.619 [0.538, 0.710] | 0.698 [0.595, 0.792] | 0.080 [−0.041, 0.188] |
| juneau | 31 | 137 | 0.702 [0.591, 0.815] | 0.728 [0.641, 0.806] | 0.026 [−0.122, 0.162] |
| sao_paulo (dev) | 20 | 163 | 0.724 [0.603, 0.831] | 0.846 [0.740, 0.938] | 0.121 [−0.047, 0.268] |
| manual_gold | 111 | 764 | 0.709 [0.656, 0.761] | 0.705 [0.656, 0.752] | −0.004 [−0.071, 0.065] |
| **open pooled** (4 non-GSV) | 197 | 804 | 0.690 [0.647, 0.732] | 0.766 [0.725, 0.806] | **0.076 [0.027, 0.129]** |
| **GoPro Max pooled** | 132 | 417 | 0.721 [0.667, 0.772] | 0.817 [0.775, 0.857] | 0.096 [0.032, 0.160] |
| **reference pooled** | 161 | 905 | 0.689 [0.645, 0.730] | 0.782 [0.741, 0.821] | 0.092 [0.036, 0.148] |

Below 0.30 the pooled gains clear zero on the open and reference pools; manual_gold is flat at
−0.004 [−0.071, 0.065].

### Secondary: every peak ≥ 0.05, with the selection caveat

**This is the step-1 candidate set, and it departs from the posted design** (see Step 1).
**Selection caveat:** the candidate set was selected by RampNet's own score, so over the full range
RampNet is asked to separate its confident TPs from its 0.05-floor FPs, a separation it wins by
construction. The ranker under test defined the sample. This table answers "could the validator
replace RampNet's score as the ranker", which is not the question the stop rule asks.

| split | TP | FP | RampNet score | validator P(correct) | validator − RampNet |
|---|--:|--:|--:|--:|--:|
| richmond | 279 | 197 | 0.956 [0.936, 0.974] | 0.860 [0.820, 0.896] | −0.096 [−0.134, −0.062] |
| bend | 289 | 98 | 0.945 [0.922, 0.967] | 0.870 [0.820, 0.914] | −0.075 [−0.127, −0.031] |
| clovis | 183 | 185 | 0.928 [0.898, 0.955] | 0.824 [0.776, 0.867] | −0.104 [−0.153, −0.057] |
| morgantown | 237 | 225 | 0.951 [0.931, 0.970] | 0.888 [0.855, 0.922] | −0.063 [−0.100, −0.025] |
| annapolis | 265 | 139 | 0.939 [0.917, 0.961] | 0.881 [0.843, 0.915] | −0.059 [−0.102, −0.018] |
| paterson | 299 | 71 | 0.954 [0.927, 0.977] | 0.896 [0.843, 0.941] | −0.059 [−0.110, −0.015] |
| gainesville | 242 | 171 | 0.923 [0.895, 0.946] | 0.830 [0.784, 0.873] | −0.093 [−0.142, −0.044] |
| laurens_mapillary | 195 | 165 | 0.847 [0.804, 0.887] | 0.902 [0.865, 0.935] | 0.055 [0.008, 0.102] |
| laurens_gsv (same-town control) | 180 | 72 | 0.874 [0.809, 0.931] | 0.812 [0.739, 0.875] | −0.062 [−0.120, −0.008] |
| bayonne | 218 | 318 | 0.858 [0.823, 0.889] | 0.803 [0.762, 0.838] | −0.056 [−0.100, −0.011] |
| budapest_district5 | 227 | 330 | 0.881 [0.847, 0.913] | 0.774 [0.727, 0.820] | −0.107 [−0.155, −0.061] |
| juneau | 257 | 159 | 0.935 [0.908, 0.960] | 0.833 [0.793, 0.868] | −0.103 [−0.145, −0.064] |
| sao_paulo (dev) | 244 | 218 | 0.922 [0.894, 0.946] | 0.875 [0.839, 0.908] | −0.047 [−0.085, −0.011] |
| manual_gold | 3607 | 1102 | 0.953 [0.945, 0.961] | 0.849 [0.834, 0.865] | −0.104 [−0.119, −0.090] |
| **open pooled** (4 non-GSV) | 897 | 972 | 0.877 [0.860, 0.893] | 0.821 [0.799, 0.843] | −0.056 [−0.080, −0.033] |
| **GoPro Max pooled** | 413 | 483 | 0.850 [0.824, 0.875] | 0.848 [0.822, 0.874] | −0.002 [−0.036, 0.030] |
| **reference pooled** | 1794 | 1086 | 0.946 [0.938, 0.954] | 0.872 [0.856, 0.886] | −0.074 [−0.091, −0.060] |

As a replacement ranker the validator is worse on 13 of 14 splits. Dropping the redundant FPs or
restricting to the unbiased subset changes no pooled sign; all variants are in the JSON
(`with_redundant`, `without_redundant`).

### The resolution control: richmond at native resolution and at 4096

manual_gold's crops are cut from 4096×2048 panos upsampled 2× to the serving resize, the verdict
splits' from native imagery. To separate "the validator does not help on independent labels" from
"the validator is blurred on manual_gold", richmond (native 8192 GSV) was downsampled to 4096×2048
with manual_gold's own transform and scored again with the same scorer.

- **Transform:** `scripts/analysis/downsample_panos_258.py`, the last step of
  `rampnet.gsv.fetch_panorama` that produced manual_gold's images (`cv2.resize` to 4096×2048,
  `INTER_LINEAR`) and its plain `cv2.imwrite` (JPEG quality 95). Google's server-side zoom-3 render,
  which manual_gold's fetch started from, cannot be reproduced; the HF `rampnet-benchmark`
  `4096x2048` config is a different transform (PIL BILINEAR, q82) and was not used. The native panos
  were checked against richmond's `imagery_manifest.json` first; the per-file sha256s of source and
  output are in `analysis_out/verifier_258/validator_scores_4096/richmond_downsample_manifest.json`.
- **Run:** `scripts/analysis/validator_score_258_4096.slurm`, klone job `42153716`, `ckpt-g2`, one
  L40, 2026-10-22 08:55–09:02 local, 443 s = 0.12 GPU-hours, $0. Scores:
  `analysis_out/verifier_258/validator_scores_4096/richmond.json`; job log
  `docs/data/validator_258_4096_42153716.log`; ledger row in `analysis_out/compute_log.jsonl`; sacct
  dump `docs/data/compute/sacct_klone_2026-10-22_validator258_4096.txt`. The native run was on an
  L40S, so the difference below also contains any GPU numerics difference; both are bounded by it.

Validator AUC on richmond's candidates, native against 4096, paired on the same candidates and
panos (`resolution_control` in `validator_auc.json`):

| stratum | TP | FP | validator, native | validator, 4096 | 4096 − native |
|---|--:|--:|--:|--:|--:|
| pre-registered band [0.05, 0.55) | 41 | 188 | 0.740 [0.639, 0.835] | 0.723 [0.615, 0.825] | −0.017 [−0.041, 0.004] |
| below 0.30 | 22 | 169 | 0.751 [0.637, 0.867] | 0.754 [0.636, 0.868] | +0.003 [−0.025, 0.028] |
| every peak ≥ 0.05 | 279 | 197 | 0.860 [0.820, 0.896] | 0.855 [0.814, 0.891] | −0.005 [−0.011, 0.000] |

**The downsample costs richmond little**: +0.003 [−0.025, 0.028] below 0.30, against the ~0.08 to
0.10 that separates manual_gold's below-0.30 difference (−0.004) from the pooled verdict splits (+0.076
open, +0.092 reference), and −0.017 [−0.041, 0.004] on the pre-registered band. The band interval
allows resolution to account for up to about half of manual_gold's −0.074 on that band. So the
upsampling does not explain most of manual_gold's row: manual_gold stands as counter-evidence, with
resolution a partial contributor at most: on the one split with
independent labels, the validator adds nothing below the operating point. One split's control, and a
GSV one; it does not cover the open rigs.

### What step 2 does and does not settle

- **Settled:** as a replacement for RampNet's score over all peaks, the validator is worse on 13 of
  14 splits (secondary table, with its selection caveat).
- **On the pre-registered band:** the stop condition is met on the open pool and on manual_gold, and
  not met on laurens_mapillary. bayonne is flat.
- **Upsampling does not explain most of manual_gold's row** (richmond control; on the band it could
  account for up to about half), so manual_gold is counter-evidence, with resolution a partial
  contributor at most, on the one split whose labels do not penalize the validator.
- **Not settled by step 2 alone:** whether the information the validator carries below 0.30 is
  worth anything once combined with RampNet's score. Step 3 measures that.
- **Not run:** manual_gold at native resolution (#21); the other validators in the plan.

## Step 3: fusion fitted on sao_paulo, recall at matched precision (run 2026-10-22)

`scripts/analysis/validator_fusion_258.py`; output `analysis_out/verifier_258/validator_fusion.json`;
`--check` re-derives it byte for byte; `tests/test_validator_fusion_258.py`. CPU only, about 40 s.

**The rule, fixed before any held-out split was read.** A two-feature logistic regression,
P(tp) = sigmoid(b0 + b1·logit(RampNet score) + b2·logit(P(correct))), fitted by unpenalized
maximum likelihood on sao_paulo's 462 TP/FP candidates (244 TP; `ignored` dropped, redundant FPs
kept). It is the smallest rule that weighs the two scores against each other with fitted weights:
three parameters on 462 rows, with no search over models, features or penalties. Fitted:
b0 = −0.031, b1 = 1.183 (RampNet), b2 = 0.200 (validator). The validator gets about a sixth of
RampNet's weight on the logit scale.

**Scoring.** Each rule keeps a subset of every pano's op_cache floor peaks, and the kept set is
re-scored exactly with `score_pano` (seam wrap; each peak keeps its own RampNet score as its match
order, as in `cascade_cost_35.py`). Precision is over every pano and recall over the `fn_confirmed`
panos, `aggregate`'s convention, using `cascade_cost_35`'s `_pano_counts` and `_prf`. Re-scoring
matters because a fused rule can drop a high-scored peak and keep a lower one beside the same ramp,
which the per-peak labels of step 1 (valid only for score-order prefixes) would mis-count. The
script asserts that the re-scored RampNet@0.30 equals the step-1 labels on every split.

**Matched precision, `cascade_cost_35`'s definition.** `cascade_cost_35.py` reads a rule against its
baseline as `best_recall_at_precision_ge_baseline`: among the settings whose precision is at least
the baseline's (RampNet at 0.30), the highest recall, ties by F1. This applies that definition
exactly, to both arms on the same split. The threshold-only arm sweeps RampNet's threshold over
`cascade_cost_35.THR_SWEEP` (0.05 to 0.95 by 0.01); the fused arm sweeps the fused probability from
0.01 to 0.99 by 0.01, the same step as the threshold-only arm, so neither arm gets a finer grid. Both arms' thresholds are read on the scored split, so this compares the two
rankings at one operating precision; the fused weights come from sao_paulo only. The bootstrap is
selection aware: each of the 2,000 pano draws recomputes the precision bar and re-picks both arms'
thresholds.

**Fully held out ("deployed").** The fused threshold is also fixed on sao_paulo by the same rule:
the best recall at P ≥ P(RampNet@0.30) there, which is 0.27. The same rule applied to the
threshold-only arm on sao_paulo picks 0.30 itself, so RampNet@0.30 is the matched threshold-only
control. The 0.27 cut is applied unchanged to every other split.

### Recall at matched precision

| split | precision bar | threshold-only R (t) | fused R (t) | fused − threshold-only |
|---|--:|--:|--:|--:|
| richmond | 0.902 | 0.829 (0.33) | 0.823 (0.49) | −0.006 [−0.032, +0.024] |
| bend | 0.924 | 0.823 (0.30) | 0.829 (0.41) | +0.006 [−0.027, +0.043] |
| clovis | 0.851 | 0.821 (0.30) | 0.821 (0.36) | +0.000 [−0.052, +0.041] |
| morgantown | 0.888 | 0.805 (0.32) | 0.809 (0.34) | +0.004 [−0.015, +0.033] |
| annapolis | 0.901 | 0.809 (0.30) | 0.861 (0.29) | +0.051 [+0.013, +0.075] |
| paterson | 0.950 | 0.719 (0.30) | 0.729 (0.38) | +0.010 [−0.027, +0.027] |
| gainesville | 0.857 | 0.772 (0.30) | 0.783 (0.46) | +0.011 [−0.025, +0.065] |
| laurens_mapillary | 0.885 | 0.526 (0.30) | 0.659 (0.18) | +0.133 [+0.035, +0.192] |
| laurens_gsv (same-town control) | 0.923 | 0.654 (0.30) | 0.677 (0.38) | +0.023 [−0.090, +0.094] |
| bayonne | 0.754 | 0.498 (0.30) | 0.578 (0.22) | +0.080 [+0.003, +0.126] |
| budapest_district5 | 0.707 | 0.643 (0.31) | 0.647 (0.35) | +0.003 [−0.034, +0.029] |
| juneau | 0.911 | 0.761 (0.30) | 0.758 (0.36) | −0.003 [−0.051, +0.024] |
| sao_paulo (dev, fitted here) | 0.803 | 0.797 (0.30) | 0.819 (0.27) | +0.021 [−0.018, +0.061] |
| manual_gold | 0.912 | 0.892 (0.30) | 0.893 (0.40) | +0.001 [−0.005, +0.006] |
| **open pooled** (4 non-GSV) | 0.806 | 0.610 (0.30) | 0.635 (0.32) | **+0.024 [+0.002, +0.054]** |
| **GoPro Max pooled** | 0.810 | 0.511 (0.30) | 0.607 (0.22) | +0.096 [+0.042, +0.143] |
| **reference pooled** | 0.900 | 0.793 (0.30) | 0.803 (0.39) | +0.010 [+0.001, +0.023] |
| **all 13 held-out splits pooled** | 0.895 | 0.813 (0.30) | 0.821 (0.37) | +0.007 [−0.000, +0.012] |

### Deployed: the sao_paulo cut (0.27) applied unchanged, against RampNet@0.30

| split | RampNet@0.30 P / R / F1 | fused P / R / F1 | ΔP | ΔR | ΔF1 |
|---|--:|--:|--:|--:|--:|
| richmond | 0.902 / 0.829 / 0.864 | 0.838 / 0.852 / 0.845 | −0.064 [−0.099, −0.032] | +0.023 [+0.006, +0.042] | −0.019 [−0.040, +0.002] |
| bend | 0.924 / 0.823 / 0.871 | 0.901 / 0.859 / 0.879 | −0.024 [−0.045, −0.003] | +0.037 [+0.013, +0.063] | +0.009 [−0.006, +0.025] |
| clovis | 0.851 / 0.821 / 0.836 | 0.810 / 0.831 / 0.820 | −0.041 [−0.080, −0.002] | +0.010 [−0.025, +0.046] | −0.015 [−0.042, +0.011] |
| morgantown | 0.888 / 0.805 / 0.845 | 0.872 / 0.817 / 0.843 | −0.016 [−0.056, +0.028] | +0.011 [−0.008, +0.030] | −0.002 [−0.023, +0.021] |
| annapolis | 0.901 / 0.809 / 0.853 | 0.897 / 0.864 / 0.880 | −0.004 [−0.033, +0.025] | +0.054 [+0.032, +0.077] | +0.027 [+0.008, +0.047] |
| paterson | 0.950 / 0.719 / 0.818 | 0.942 / 0.734 / 0.825 | −0.008 [−0.025, +0.007] | +0.015 [+0.000, +0.032] | +0.007 [−0.005, +0.019] |
| gainesville | 0.857 / 0.772 / 0.812 | 0.819 / 0.831 / 0.825 | −0.038 [−0.073, −0.002] | +0.059 [+0.029, +0.092] | +0.012 [−0.012, +0.037] |
| laurens_mapillary | 0.885 / 0.526 / 0.660 | 0.900 / 0.615 / 0.730 | +0.015 [−0.010, +0.042] | +0.088 [+0.048, +0.135] | +0.070 [+0.037, +0.106] |
| laurens_gsv (same-town control) | 0.923 / 0.654 / 0.766 | 0.909 / 0.723 / 0.805 | −0.015 [−0.040, +0.015] | +0.068 [+0.031, +0.108] | +0.039 [+0.013, +0.066] |
| bayonne | 0.754 / 0.498 / 0.600 | 0.783 / 0.551 / 0.647 | +0.029 [−0.009, +0.071] | +0.053 [+0.015, +0.089] | +0.047 [+0.014, +0.080] |
| budapest_district5 | 0.707 / 0.643 / 0.674 | 0.661 / 0.663 / 0.662 | −0.046 [−0.084, −0.010] | +0.020 [+0.000, +0.040] | −0.011 [−0.034, +0.010] |
| juneau | 0.911 / 0.761 / 0.829 | 0.888 / 0.771 / 0.825 | −0.024 [−0.046, −0.005] | +0.010 [−0.019, +0.041] | −0.004 [−0.025, +0.016] |
| sao_paulo (dev, fitted here) | 0.803 / 0.797 / 0.800 | 0.804 / 0.819 / 0.811 | +0.001 [−0.025, +0.032] | +0.021 [−0.008, +0.053] | +0.011 [−0.009, +0.035] |
| manual_gold | 0.912 / 0.892 / 0.902 | 0.891 / 0.902 / 0.896 | −0.021 [−0.027, −0.015] | +0.010 [+0.005, +0.014] | −0.005 [−0.009, −0.001] |
| **open pooled** (4 non-GSV) | 0.806 / 0.610 / 0.695 | 0.794 / 0.651 / 0.716 | −0.013 [−0.030, +0.006] | +0.041 [+0.025, +0.057] | +0.021 [+0.007, +0.035] |
| **GoPro Max pooled** | 0.810 / 0.511 / 0.626 | 0.835 / 0.580 / 0.684 | +0.025 [−0.001, +0.054] | +0.069 [+0.042, +0.099] | +0.058 [+0.034, +0.084] |
| **reference pooled** | 0.900 / 0.793 / 0.843 | 0.872 / 0.823 / 0.847 | −0.028 [−0.040, −0.017] | +0.030 [+0.022, +0.040] | +0.004 [−0.004, +0.011] |
| **all 13 held-out splits pooled** | 0.895 / 0.813 / 0.852 | 0.873 / 0.835 / 0.854 | −0.022 [−0.028, −0.017] | +0.022 [+0.018, +0.027] | +0.002 [−0.002, +0.005] |

### Reading step 3

- **At matched precision the fused rule buys recall on the open pool, +0.024 [+0.002, +0.054], and
  little elsewhere:** +0.010 [+0.001, +0.023] on the reference pool, +0.007 [−0.000, +0.012] over all
  13 held-out splits, +0.001 [−0.005, +0.006] on manual_gold. The GoPro Max pool's +0.096 is led by
  laurens_mapillary (+0.133), but bayonne gains too here (+0.080 [+0.003, +0.126]), unlike in the
  AUC tables; budapest_district5 and juneau do not.
- **Deployed with the sao_paulo cut,** the open pool gains +0.041 recall and +0.021 [+0.007, +0.035]
  F1, with precision −0.013 [−0.030, +0.006]. On the reference pool and manual_gold the same cut trades precision for
  recall and F1 is flat. The cut is one number fitted on one split, so where a split's precision
  falls (richmond −0.064) it is that threshold, not the weights, that moved it.
- **laurens_gsv gains when deployed** (+0.039 F1) but not at matched precision (+0.023
  [−0.090, +0.094]), so on laurens_gsv the deployed gain is mostly a lower effective threshold.
- **Cost.** Validator wall-clock on klone (from the job log, one L40S) per ramp the deployed rule
  adds over RampNet@0.30: 5.5 s on the open pool (259 s for +47 TP), 2.3 s on laurens_mapillary, and
  8.9 s over all 13 held-out splits (1,452 s for +163 TP). $0, on klone's ckpt partition; the
  GPU-hours are the 42152480 ledger row.
- **Caveats.** Every caveat of the candidate labels applies. The sub-threshold TP lower bound
  penalizes the fused rule, which admits sub-threshold peaks, and not RampNet@0.30. The fusion was
  run on all panos only, not on the unbiased subset. The op_caches predate the seam fix.

## Appendix: the production cut as a filter, for reference

The validator's own verdict (`pred == "correct"`, P(correct) > 0.5, what production returns) applied
to RampNet's peaks at a 0.05 or a 0.30 floor, against RampNet@0.30. Nothing is fitted, and this is
not the rule the design proposed (step 3 is). F1, with ΔF1 against RampNet@0.30 and its paired pano
bootstrap interval, from `validator_fusion.json` (`production_cut`), exactly re-scored. The first
version of this table (in `validator_auc_258.py`) counted per-peak labels instead of re-scoring;
the two agree to within 0.003 F1.

| split | RampNet@0.30 | RampNet ≥ 0.05, then validator | RampNet ≥ 0.30, then validator |
|---|--:|--:|--:|
| richmond | 0.864 | 0.766 (−0.098 [−0.138, −0.062]) | 0.847 (−0.017 [−0.036, +0.001]) |
| bend | 0.871 | 0.854 (−0.017 [−0.042, +0.007]) | 0.862 (−0.008 [−0.022, +0.004]) |
| clovis | 0.836 | 0.738 (−0.097 [−0.142, −0.053]) | 0.793 (−0.043 [−0.081, −0.006]) |
| morgantown | 0.845 | 0.801 (−0.044 [−0.079, −0.008]) | 0.840 (−0.005 [−0.029, +0.017]) |
| annapolis | 0.853 | 0.828 (−0.025 [−0.059, +0.010]) | 0.850 (−0.003 [−0.019, +0.014]) |
| paterson | 0.818 | 0.808 (−0.010 [−0.033, +0.010]) | 0.815 (−0.004 [−0.015, +0.006]) |
| gainesville | 0.812 | 0.777 (−0.036 [−0.077, +0.004]) | 0.797 (−0.016 [−0.039, +0.005]) |
| laurens_mapillary | 0.660 | 0.739 (+0.079 [+0.027, +0.130]) | 0.624 (−0.036 [−0.066, −0.009]) |
| laurens_gsv | 0.766 | 0.770 (+0.004 [−0.042, +0.049]) | 0.726 (−0.040 [−0.071, −0.015]) |
| bayonne | 0.600 | 0.589 (−0.011 [−0.064, +0.038]) | 0.551 (−0.049 [−0.083, −0.016]) |
| budapest_district5 | 0.674 | 0.588 (−0.086 [−0.131, −0.046]) | 0.641 (−0.033 [−0.063, −0.005]) |
| juneau | 0.829 | 0.764 (−0.066 [−0.106, −0.029]) | 0.780 (−0.049 [−0.077, −0.024]) |
| sao_paulo | 0.800 | 0.754 (−0.046 [−0.082, −0.010]) | 0.788 (−0.012 [−0.034, +0.011]) |
| manual_gold | 0.902 | 0.865 (−0.037 [−0.044, −0.030]) | 0.894 (−0.008 [−0.012, −0.004]) |
| **open pooled** (4 non-GSV) | 0.695 | 0.663 (−0.031 [−0.054, −0.007]) | 0.655 (−0.040 [−0.055, −0.025]) |
| **GoPro Max pooled** | 0.626 | 0.655 (+0.028 [−0.010, +0.066]) | 0.585 (−0.042 [−0.065, −0.021]) |
| **reference pooled** | 0.843 | 0.799 (−0.044 [−0.057, −0.031]) | 0.831 (−0.012 [−0.019, −0.004]) |

The validator calls most candidates correct, so its own 0.5 cut is too permissive as a filter at a
0.05 floor, and on top of 0.30 it removes more TPs than FPs.

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
python scripts/analysis/validator_auc_258.py                 # CPU, ~80 s
python scripts/analysis/validator_auc_258.py --check

# the richmond 4096 resolution control (GPU, klone; STAGE is shared storage, never /tmp)
PY=... PANOS_ROOT=... STAGE=... DAV2_DIR=... DAV2_CKPT=... HF_HOME=...     sbatch -A ckpt-makelab -p ckpt-g2 --export=ALL scripts/analysis/validator_score_258_4096.slurm

# step 3, CPU, ~40 s
python scripts/analysis/validator_fusion_258.py
python scripts/analysis/validator_fusion_258.py --check
pytest -q tests/test_verifier_candidates_258.py tests/test_validator_auc_258.py     tests/test_validator_fusion_258.py
```

The scores in `analysis_out/verifier_258/validator_scores/` are committed, so the read-out re-derives
from a clean clone with no GPU. Re-running step 2 needs the benchmark panos: nine splits are on HF
(`rampnet-benchmark`); laurens_gsv, laurens_mapillary, bayonne and juneau are not uploaded yet
(`docs/replication.md`), and manual_gold at native resolution is #21.
