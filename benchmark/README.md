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

## How a split is produced — the two-repo boundary

RampNet and [`sidewalk-auto-labeler`](https://github.com/ProjectSidewalk/sidewalk-auto-labeler)
split by **question, not by city**. A validation city (Richmond, Bend, Clovis, …) is simply
where both questions land on the same panos — neither repo "owns" a city.

- *"What are the curb ramps in this city?"* → **auto-labeler**. Enumerate → thin → fetch
  imagery → detect → submit. It is the only thing that fetches pixels (RampNet has no
  `sources/`), so it also hands RampNet the imagery bundle.
- *"How good is the model, and is it improving?"* → **RampNet**. Ground truth, scoring, and
  the benchmark itself.

Per-step, for adding a city to this benchmark:

| Step | Repo | Tool |
|------|------|------|
| Enumerate → thin → **detect** the city | auto-labeler | `main.py --source <mapillary\|gsv>` |
| Export the native-res bundle (`panos/` + `records.jsonl`) | auto-labeler | `scripts/export_benchmark.py` |
| GT-verify a sample → `verdicts.json` | **RampNet** | `scripts/gt_gallery.py benchmark/<city>` |
| Score P/R + Wilson CIs + threshold sweep | **RampNet** | `scripts/score_validation.py` / `rampnet.validation` |
| Add the split to the HF benchmark dataset | **RampNet** | `scripts/build_benchmark_dataset.py` |

The GT gallery and scorer are **canonical in RampNet** (`scripts/gt_gallery.py`,
`rampnet/validation.py` — decoupled from any imagery source, no network). The auto-labeler
still carries transitional copies (`scripts/spot_check_gallery.py`, `scripts/score_validation.py`)
marked for deletion; run the RampNet versions, not those.

## Current splits

| City | Source | Panos | Precision | Recall |
|------|--------|-------|-----------|--------|
| richmond | Mapillary 360 | 124 | 0.960 | 0.765 |
| bend | GSV (Google Street View) | 110 | 0.954 | 0.758 |
| clovis | Mapillary 360 (GoPro Fusion) | 125 | 0.914 | 0.713 |

All three splits are **self-contained**: the reviewer's complete-scan attestation is baked into
`no_missed` (set on fully-judged panos with no missed marks), so the numbers reproduce with a
plain `python scripts/score_validation.py benchmark/<city>` — no `--assume-scanned` needed.
This matters because the recall gate otherwise excludes unconfirmed panos and biases recall
low (it over-weights panos where a miss *was* found).

**Clovis is stratified, and harder.** Its 125 panos are sampled 5 top-detection / 95 random /
25 empty (`sample.json`; each pano's `benchmark_group` is in `records.jsonl`). The table shows
the all-125 figure `score_validation.py` prints first, but the number comparable to richmond and
bend is the **unbiased subset** — random + empty, 120 panos, dropping the 5 hand-picked
high-density panos: **P 0.889 / R 0.650**. Both are below the other two cities because clovis is
100% soft, 2018-era GoPro Fusion 360 imagery, where richmond mixes in the sharper NCTECH iSTAR
Pulsar (camera provenance is in the records, added in #50).

## The manual_gold split (in-distribution gold reference)

`benchmark/manual_gold/` (issue #58) is a different kind of split: its ground truth is the
repo's 1,000-pano manually labeled gold set (`manual_labels/*.txt` — 3,919 curb ramps, 207
negative panos), labeled **independently of any model**. Unlike the city splits it is not
derived from reviewing RampNet's detections, so it carries no RampNet anchoring — and at 4×
the size of the largest city split it gives every model tighter confidence intervals. It is
also **in-distribution**: GSV imagery from the training cities (NYC / Portland / Bend), held
out of Stage-2 training as the HF dataset's test split. Treat it as the in-domain reference
that complements the deployment cities above, never as their replacement. It shares zero
panos with bend, richmond, or clovis (verified by `scripts/fetch_manual_gold.py --audit`).

The layout differs accordingly — there is no verdict review:

```
benchmark/manual_gold/
  gt_source.json    points at manual_labels/ (GT = YOLO box centers, no ignore points,
                    every pano recall-confirmed)
  records.jsonl     pano metadata + RampNet detections (built by the two scripts below)
  panos/            imagery from the HF test split (git-ignored, like every split)
```

No verdicts means `scripts/score_validation.py` and `scripts/gt_gallery.py` do **not** apply
here; the split is scored by the model-comparison harness only:

```
python scripts/fetch_manual_gold.py --audit      # id membership/overlap audit, no download
python scripts/fetch_manual_gold.py              # imagery (HF test split, ~44 GB; run on Hyak)
python scripts/export_gold_records.py --checkpoint <stage2.pth>   # RampNet detections + gate
python scripts/model_comparison/compare.py benchmark/manual_gold --models rampnet --op-threshold 0.55
```

The exporter ends with a reproduction gate against the published gold-set numbers
(P 0.949 / R 0.873 @ conf >= 0.55, TTA). Read the manual-gold section of
`docs/model_comparison.md` before quoting numbers from this split.

**All three splits were reviewed at model resolution** with the pan/zoom labeler (`scripts/gt_gallery.py`),
which shows the full pano at the model's input resolution (4096×2048) with pan/zoom, rather than a
downscaled overview. For richmond and bend this was a *re-review*: reviewing at model resolution
surfaced genuinely-missed ramps that the earlier 1600 px overview hid — small/distant curb ramps a
reviewer literally could not resolve — correcting recall down from earlier, optimistic numbers
(richmond 0.895 → 0.765, bend 0.831 → 0.758). Precision was essentially unchanged (the zoom mostly
resolved `unsure` detections, not misclassifications). The correction is consistent across both
imagery sources (GSV and Mapillary), and these are the honest per-pano-comprehensive figures; clovis
was reviewed at model resolution from the start. Richmond and bend each include one `duplicate`
verdict — a redundant second detection on one physical ramp, scored as a false positive by default
(`--lenient-duplicates` scores the other way; see `scripts/score_validation.py`); clovis has none.
