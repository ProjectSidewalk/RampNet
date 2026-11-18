# Production Crosswalk validator on Juneau AI Crosswalk labels (Mapillary), offline, 2026-11-17

Run by an Opus 5.5 subagent; the main session saved this summary and checked the headline independently.
Every table is in `results/tables.md`, and per-label scores for all 3,579 labels are in `results/all_labels_scored.csv`.
`score_votes.py --votes <csv>` re-scores after new votes.

## Models

- No published Crosswalk validator was trained on Mapillary.
- Primary result: round 3, hub `projectsidewalk/sidewalk-validator-ai-crosswalk@b8498b82eb`, all-GSV training manifest (3,120 rows).
- Secondary: the 2025-08 model `b7936e5bd1`.

## Crops

Rendered with the repo's own path: `analysis/ai_labels_eval.py` -> `data_generation.read_store_panorama` -> `create_perspective_crop`, on the klone checkout at 838287d. That commit is identical to 76850a1 for every file used.

- `coord_frame "rig"` with no tilt (AI labels, F9). Each crop is centred on its detection to within 0.7 px.
- Crop size comes from Depth Anything V2, not GSV depth, so no fallback was needed.
- Panos come from the klone store copy, resized to 8192x4096. GoPro panos are therefore upsampled.

Deviations from serving:
- Serving never receives this traffic.
- The renderer is the GPU torch path, not the API's CPU path.
- The store is the klone copy, not the API host's mount.
- A repo bug was worked around: `ai_labels_eval.read_labels` turns all-digit Mapillary pano ids into ints. The fix is `dtype={"pano_id": str}`; it has not been made.

## Results (round 3)

| Votes | Decided | AUC [95% CI] |
|---|---:|---|
| First batch (clustered) | 232 | 0.985 [0.963, 1.000] |
| All votes | 344 | 0.980 [0.964, 0.993] |
| Seeded random sample | 79 | 0.959 [0.910, 0.989] |

On the first batch, 0 of 99 rejected labels got an Agree vote.

Weighted to all 3,579 labels with the main session's stratified estimator (`../xwalk_validator_check.py`):

| Filter | Precision [95% CI] | Real crosswalks kept | Labels |
|---|---|---|---|
| As sent | 0.72 [0.65, 0.79] | 1.00 | 3,579 |
| Drop validator Disagree verdicts | 0.94 [0.89, 0.99] | 0.93 | ~2,540 |
| Validator Agree vote (production rule) | 0.94 [0.87, 0.99] | 0.83 | ~2,290 |
| Drop Disagree verdicts + RampNet conf >= 0.40 | 0.99 [0.95, 1.00] | 0.79 | ~2,070 |

The last row's rule was chosen after seeing the sample, which is small. Treat it as an estimate.

As deployed, the validator removes nothing:
- The served map caps `incorrect` at 0.9199, so no Disagree vote is ever cast.
- An Agree vote needs `p_correct` >= 0.9925.
- SidewalkWebpage never sends it AI-authored or Mapillary labels.

Per class (all votes), labels getting a raw Disagree verdict:

| Class | Disagree verdicts / labels |
|---|---|
| Capture-car roof | 51/52 |
| Brick sidewalk | 40/40 |
| Shadows | 11/11 |
| Stairs | 6/6 |
| Turn lane | 4/4 |
| Paint/chalk/bike markings | 4/4 |
| Curb ramp | 3/3 |
| Driveway | 3/3 |
| Parking/access-aisle stripes | 2/5 |
| Window reflection | 0/1 |
| Label 20 px off | 0/1 |

It rejects 8 of 208 of Jon's Agrees, mostly faded or distant crossings.

Examples are in `examples/`.

## Not done

The VLM leg (Gemma 4 31B) did not run: the scrubbed venv `vlm-scoreboard` was purged. The klone work dir `/gscratch/scrubbed/jfroehli/validator_xwalk_juneau/` holds all 3,579 crops and will be purged after 21 days.
