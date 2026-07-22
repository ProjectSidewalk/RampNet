# Validation benchmark

Human-validated ground truth for RampNet's curb-ramp detector, on **real deployment
imagery** — the fixed comparison target for model changes (see issues #21, #22, #26).
Each city is one split; per pano every model detection is judged correct / incorrect /
unsure and the reviewer marks ramps the model missed, so both precision and recall are
measurable.

## Layout

```
benchmark/<city>/
  records.jsonl   detection records for the validated panos (detections + pano metadata)
  verdicts.json   human verdicts (crop judgments + missed-ramp marks), self-contained
```

`records.jsonl` + `verdicts.json` are all the **scoring** needs — they're image-free, so
precision/recall reproduce with no imagery. Score with:

```
python scripts/score_validation.py benchmark/<city>
```

The **native-resolution panos** (for the labeling UI and the resolution experiment, #25)
are archived separately and published to HF (#21); they are intentionally not in git.

## Current splits

| City | Source | Panos | Precision | Recall |
|------|--------|-------|-----------|--------|
| richmond | Mapillary 360 | 124 | 0.965 | 0.895 |
| bend | GSV (Google Street View) | 110 | 0.958 | 0.831 |

Both splits are **self-contained**: the reviewer's complete-scan attestation is baked into
`no_missed` (set on fully-judged panos with no missed marks), so the numbers reproduce with a
plain `python scripts/score_validation.py benchmark/<city>` — no `--assume-scanned` needed.
This matters because the recall gate otherwise excludes unconfirmed panos and biases recall
low (it over-weights panos where a miss *was* found).
