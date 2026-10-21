# Second-pass verifier on RampNet's candidates (#258)

**Issue:** [#258](https://github.com/ProjectSidewalk/RampNet/issues/258). The plan is the
comment of 2026-10-21 (https://github.com/ProjectSidewalk/RampNet/issues/258#issuecomment-6316627866).
**Status:** step 1 (candidate set) and the leakage check are done. Step 2 (scoring the candidates
with the CurbRamp validator) has **not been run**; its design is below.

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

## Step 2 design (not yet run)

Score every candidate with the deployed CurbRamp validator, **reproducing the serving path** in
[sidewalk-ai-api](https://github.com/ProjectSidewalk/sidewalk-ai-api) (`main.py`, `sidewalk_ai_api/`,
read at `745e0c5`), because the serving path is what Project Sidewalk deploys. The training path differs
from it (audit finding 6 on preprocessing; training crops used Depth-Anything ViT-L where serving uses
ViT-B), and that difference is the validator's, not ours to fix here.

- **Model:** `projectsidewalk/sidewalk-validator-ai-curbramp` at revision `969ad5a3…` (the
  `model_map.py` pin, trained 2025-08-16), `AutoImageProcessor(backend="pil")`, float32.
- **Crop:** pano resized to 8192×4096, a 90° perspective of 2048×2048 centred on the candidate,
  Depth-Anything-V2 metric (`vitb`, `vkitti`, max depth 80) at the centre pixel, crop half-size
  `int(6100 / depth)`, resized to 640 on the long side with cubic interpolation.
- **Score:** P(correct), i.e. the softmax of the `correct` class, not the argmax confidence.
- **Primary metric:** AUC of P(correct) against RampNet's own score on the same candidates, per split,
  pano-bootstrap intervals; with and without `redundant` rows; on all panos and on the unbiased subset.
  If it does not beat RampNet's score on the open-rig splits, stop and write it up.
- **Where:** klone, one GPU, $0. About 10.5k crops.

**Open before step 2:**

1. **Imagery.** Panos are local at native resolution for 13 splits. bayonne has no local panos.
   manual_gold's local panos are 4096×2048 (the HF benchmark copy); serving resizes to 8192, so manual_gold
   crops would be upsampled 2×. That is an instrument difference to record, or to remove with native-res
   panos (#21).
2. **Mapillary and Panoramax panos** (5760–12288 px wide) are resized to 8192 exactly as serving resizes
   a cached scrape. The validator never saw these rigs in training; that is the transfer being tested,
   not a defect of the setup.

## Reproduce

```bash
python scripts/analysis/verifier_candidates_258.py           # CPU, seconds
python scripts/analysis/verifier_candidates_258.py --check
python scripts/analysis/validator_leakage_258.py             # 19 GETs to the public API, ~1 min
pytest -q tests/test_verifier_candidates_258.py
```
