# The benchmark scoring protocol as code (`rampnet.eval`)

Issue [#150](https://github.com/ProjectSidewalk/RampNet/issues/150), the "evaluation protocol
as code" item only. Written 2026-10-05.

**Result.** `rampnet/eval.py` scores a prediction file against a committed benchmark bundle
through the same functions every committed benchmark number came from, and
`scripts/analysis/eval_protocol_150.py --check` proves it: all **165** cells of
`analysis_out/scoreboard.json` (21 model legs, 12 splits) and all **30** (arm, split) cells of
`scripts/model_comparison/yolo_baseline/benchmark_eval/` re-derive **equal**, 1,650 and 480
field comparisons respectively, with **0** differing and **0** unchecked. The split files are
pinned by content hash in `benchmark/split_pins.json`. No number changed, and the four
fingerprinted scorer files are byte-identical to `main` (fingerprint `f4aef67aba03`).

## What the protocol is

This is the rule in the README's erratum (July 2026), as implemented by
`rampnet.detection_eval.score_pano` and `aggregate`. `rampnet.eval` adds no scoring logic.

| part | rule |
|---|---|
| ground truth, city splits | from the human review in `verdicts.json`: detections marked `True` plus non-`unsure` missed marks are GT points; `unsure` detections and `unsure` missed marks are *ignore* points; `False` and `duplicate` detections are neither |
| ground truth, `manual_gold` | the box centres of the 1,000 independent YOLO label files in `manual_labels/`; no ignore points; every pano is recall-confirmed |
| matching | greedy one-to-one, predictions in descending confidence order (input order when there is no confidence); each claims the nearest **unclaimed** GT point **strictly** within the radius; a second hit on a claimed ramp is a false positive |
| radius and space | 0.022 of the panorama width, in a 1024 × 512 scaled space (so the radius is anisotropic in normalised units) |
| seam | x wraps across the 0/1 seam (`wrap_x=True`, #140) |
| ignore zones | an unmatched prediction within the radius of an ignore point is neither TP nor FP, and stays out of the PR curve |
| precision | over every scored pano |
| recall | over the GT of recall-confirmed panos only (`fn_confirmed`: the reviewer attested a complete scan, or marked at least one miss) |
| intervals | Wilson 95% on precision and recall |
| AP | VOC all-point interpolated, over the recall-confirmed panos, from the **full** confidence range of the predictions as given |
| operating point | an operating threshold truncates P, R, F1 and the counts. It never truncates AP |

**AP is truncated at the export floor.** AP integrates whatever confidence range the
predictions carry, so it is a function of the floor they were exported at. The published
convention (the AP-provenance block in [`model_scoreboard.md`](model_scoreboard.md)) is that
the YOLO and open-vocabulary detectors are exported at a 0.05 floor, and RampNet's city-split
AP is read from `analysis_out/op_cache/` (also 0.05), because the city bundles' own detections
stop at the deployed 0.55. Every `score_split` result carries an `ap_note` that states the floor: "AP over
predictions as given; truncated at floor F" when `--floor` is declared, or the lowest
confidence present when it is not.

## Prediction format

Exactly the `benchmark/model_detections/` file shape, and no second format:

```json
{"model": "my-detector",
 "city": "richmond",
 "detections": {
   "1273933840289887": [[0.2851, 0.5859, 0.95], [0.8154, 0.5703, 0.93]],
   "934739365184374": []}}
```

`x` and `y` are normalised to [0, 1] on the equirectangular panorama, and x wraps. The third
element is a confidence, or `null` for a detector that emits none; such a model gets no AP and
no sweep. A pano absent from `detections` is scored as zero predictions, and the result
reports how many there were (`n_panos_without_predictions`), along with prediction panos the
bundle does not have (`n_prediction_panos_not_in_bundle`). The loader rejects a malformed file
with a message that names the pano and the point index. `--predictions rampnet` scores the
bundle's own `records.jsonl` detections through the same path.

## How to score one file

```powershell
python -m rampnet.eval score --bundle benchmark/richmond --predictions benchmark/model_detections/y11l_pano__richmond.json --op-threshold 0.25 --floor 0.05 --out analysis_out/eval_protocol_150/example_richmond_y11l.json
python -m rampnet.eval score --bundle benchmark/manual_gold --predictions rampnet --op-threshold 0.55 --floor 0.05
python -m rampnet.eval protocol
```

Output of the first two on this branch:

```
richmond / y11l_pano @ op 0.25: P 0.9252  R 0.4387  F1 0.5952  AP 0.7238  tp/fp/fn/ign 136/11/174/4  (124 panos, 310 GT in recall pool)
manual_gold / rampnet @ op 0.55: P 0.9474  R 0.8727  F1 0.9085  AP 0.9173  tp/fp/fn/ign 3420/190/499/0  (1000 panos, 3919 GT in recall pool)
```

`--out` writes the full result: the protocol constants, the operating point and floor,
P/R/F1 with Wilson intervals, AP and its note, the counts, `--sweep` and `--pr-curve` if
asked, the live split pins, the scorer fingerprint, `eval_sha256` (this module's own hash),
and `predictions_sha256`. `analysis_out/eval_protocol_150/example_richmond_y11l.json` is the
first command's output, committed as an example.

The operating points the committed tables use are per model class
(`scoreboard.OPERATING_POINT`): RampNet 0.55, the supervised YOLO arms 0.25, the open
detectors and Vistas at their export floor (op 0), and the chat VLMs at op 0 (they carry no
confidence, so a threshold is a no-op for them).

## The reproduction check

```powershell
$env:PYTHONPATH = "<checkout>"
python scripts/analysis/eval_protocol_150.py           # writes analysis_out/eval_protocol_150/reproduction.json
python scripts/analysis/eval_protocol_150.py --check   # exit 1 if any cell differs or the JSON is stale
```

It runs on the CPU from committed files only and took 18 to 30 s per run (four runs) on the desktop it was written on
(Windows, Python 3.12). Result:

| source | cells | equal | differs | unchecked | fields compared |
|---|--:|--:|--:|--:|--:|
| `analysis_out/scoreboard.json` | 165 | 165 | 0 | 0 | 1,650 |
| `benchmark_eval/` | 30 | 30 | 0 | 0 | 480 |

What is compared, and how:

- **Scoreboard cells.** For every (model, split) cell in `per_split`: precision, recall,
  F1, AP, `ap_bundle`, tp, fp, fn, n_panos and n_gt_recall, at the cell's operating point.
  The scoreboard stores floats rounded to 6 decimals (`scoreboard_render.JSON_PRECISION`), so
  the re-derived float is rounded to 6 decimals and compared with `==`; integers are compared
  as they are. For the 11 RampNet cells whose `ap_source` is the op_cache (every split but
  `laurens_gsv`, which has no op_cache, and `manual_gold`, whose bundle is already at 0.05),
  `analysis_out/op_cache/<split>.json` is converted to the prediction format and scored
  against the **bundle's** ground truth through `score_split`, and that AP equals the stored
  one. The scoreboard itself scores that cache against the GT stored inside the cache, so
  this also shows the two GTs agree.
- **benchmark_eval cells.** For the three YOLO pano arms on the ten splits, at op 0.25 and
  floor 0.05: the headline row of `<split>.txt` as printed (P, its CI, R, its CI, F1, AP,
  tp/fp/fn/ignored), every sweep row as printed, and `pr_<split>/pr_<arm>.json` exactly
  (`ap`, `n_gt`, and the full recall and precision lists, unrounded).

No cell needed a fix, and none could not be checked: every scoreboard cell has a committed
detections file (or is RampNet), and every YOLO arm has its file on all ten splits.

## Three scoring paths, and which number comes from which

Three pieces of code score RampNet-style detections. They share the 1:1 matcher
(`rampnet/metrics.py`) but are not interchangeable, and the README's headline does not come
from the benchmark scorer.

| path | what it scores | committed numbers it produced |
|---|---|---|
| `rampnet.detection_eval` (this protocol, via `rampnet.eval`) | any model's points against the model-agnostic GT, recall gated on `fn_confirmed`, ignore zones | `analysis_out/scoreboard.json`, `docs/model_scoreboard.md`, the per-split tables in `docs/model_comparison.md`, `benchmark_eval/` |
| `stage_two/evaluate.py` | RampNet only, from heatmaps it decodes itself from the gold-set images | the erratum's corrected `manual_gold` row (P 0.949 / R 0.873 / AP 0.9205), `stage_two/evaluation_results_new/` |
| `rampnet/validation.py` | RampNet's own detections against the reviewers' per-detection verdicts | the "RampNet verdict-based cross-check" blocks in `benchmark_eval/<split>.txt` and `compare.py` output; Wilson intervals for the other two |

So `manual_gold` RampNet at 0.55 is **P 0.947** on this protocol (the scoreboard row, and the
second command above) and **P 0.949** in the README. The gap is not the matcher. Per
[`decode_e2e_221.md`](decode_e2e_221.md), `stage_two/evaluate.py` run on the raw Hugging Face
image bytes gives P 0.94737 / R 0.87267 (3,610 predictions), the same as this protocol; the
committed `evaluation_results_new/` was produced from quality-95 JPEG re-encodes
(`download_dataset.py`'s path), and re-running on q95 re-encodes reproduces its P 0.94921
exactly. Recall gating does not arise on `manual_gold`, because every pano there is
recall-confirmed. The AP figures also differ (0.917 here against 0.9205 in the erratum); this
document does not attribute that gap beyond noting that the two paths decode their own peaks
from different image bytes and the bundle's detections were exported at a 0.05 floor.

## Leave-one-city-out: a reporting convention, not a training run

The issue asks for "a leave-one-city-out runner over rampnet-benchmark". There is no LOCO
training runner in this repo, and none was built here: nothing is retrained without the
held-out city. `python -m rampnet.eval loco --model <stem> --op-threshold T` is a **reporting**
convention. For each of the eight pooled in-distribution splits (`POOLED_SPLITS`, which a test
holds equal to `low_floor_sweep.US_SPLITS`), it reports the model's score on that split beside
the micro-pooled score (counts summed; precision over all panos, recall over recall-confirmed
GT) and the macro-mean over the other seven. RampNet at 0.55:

```
LOCO report for rampnet @ op 0.55 (reporting only, no retraining)
  held out                 F1  rest F1 micro  rest F1 macro
  richmond              0.855          0.794          0.783
  bend                  0.850          0.794          0.783
  clovis                0.801          0.803          0.790
  morgantown            0.835          0.798          0.785
  annapolis             0.839          0.797          0.785
  paterson              0.805          0.802          0.790
  gainesville           0.803          0.803          0.790
  laurens_mapillary     0.543          0.828          0.827
```

RampNet's training cities are NYC, Portland and Bend, so every pooled split except `bend` is
already outside its training distribution (`bend` is a training city and overlaps the training
set by four panoramas; see `benchmark/README.md`). For RampNet this report is therefore a
per-city spread, not a held-out estimate. Whether this is what the FAIROS deliverable means by LOCO is a decision for Jon.

## Split pins

`benchmark/split_pins.json` holds, for each of the 12 scored splits
(`low_floor_sweep.ALL_SPLITS`): the LF-normalised sha256 of `records.jsonl`; of `verdicts.json`
for a city split, or of `gt_source.json` plus a 16-hex digest over the 1,000 `manual_labels/*.txt`
files (built like `imagery_manifest.digest_of`, `stem|sha256` sorted by stem) for
`manual_gold`; the imagery digest read from the committed `imagery_manifest.json`; and the
record and review counts. `python -m rampnet.eval pins --verify` exits 1 on any drift, and
`tests/test_eval_protocol_150.py` runs the same check in CI. Every `score` result embeds the
live pins for its split, so a result names the exact files it was scored against.

Left out on purpose: `bayonne` (staged, not reviewed), `vancouver` (#224, in flux) and the #48
neighbourhood bundles, which borrow another split's verdicts. They can be added once their
files settle.

## Code layout

- `rampnet/bundles.py`: the bundle loaders, moved (not copied) from
  `scripts/model_comparison/compare.py`, which re-exports them, so `C.load_bundle is
  bundles.load_bundle`. New: `ground_truths` (either bundle kind, in the pano order the
  committed scorers use), `rampnet_predictions`, `split_pins`.
- `rampnet/eval.py`: `rescore`, `operating_report`, `sweep_rows` and `has_confidences`, also
  moved from `compare.py` and re-exported; `score_split`, the prediction-file validator, the
  pins, `loco` and the CLI.
- `scripts/analysis/eval_protocol_150.py`: the reproduction check.
- `tests/test_eval_protocol_150.py`: a synthetic three-pano bundle with known counts, the
  re-exports, all 12 bundles loading, the pins, three representative cells, the fingerprint,
  the validator and the protocol constants. CPU-only and offline.

**The fingerprint guard.** `rescore_benchmark_eval.py` stamps every `benchmark_eval/` file with
a sha256 over `rampnet/{geometry,metrics,detection_eval,validation}.py`, and CI fails on any
byte change to them. Those four files were not edited, and `rampnet/eval.py` is deliberately
not added to the fingerprinted set: it packages the scorer without changing it.
`tests/test_eval_protocol_150.py::test_scorer_fingerprint_unchanged` pins `f4aef67aba03` with
a message that says what to re-run if it ever changes.

## Not done here

The rest of #150 is not touched: no DOI, no dataset or model card changes, no leaderboard,
no Hugging Face upload, and no Croissant work (draft PR #190 covers that). Those are Jon's
call.
