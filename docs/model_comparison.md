# Model comparison: RampNet vs. VLMs

Uses the standardized curb-ramp benchmark (`benchmark/{bend,richmond}/`) to compare
RampNet against general-purpose vision-language models that now do bounding-box detection —
**Gemini (Flash 3.5)** via API and the latest open **Qwen3-VL** (intended to run on Hyak).
The question: does a general VLM match or beat the purpose-trained RampNet on real
deployment imagery (GSV + Mapillary 360)? The harness is model-agnostic, so future models
(issue #20) plug in the same way.

## Why the benchmark verdicts can't be reused directly

The bundle's `verdicts.json` holds a human judgment for **each RampNet detection** (aligned
positionally to `records.jsonl`). Those verdicts describe RampNet's points and can't score a
different model, which produces different boxes. So we derive a **model-agnostic ground
truth** from the same review and score every model against it identically.

## Methodology

Per pano (`rampnet/detection_eval.py`):

- **GT ramp points** = detections the reviewer confirmed real (`True`) **∪** ramps they
  marked as missed (non-`unsure`). This is the reviewer's complete enumeration of the real
  curb ramps in the pano.
- **Ignore points** = `unsure` detections **∪** `unsure` missed marks. A prediction landing
  here is scored as **neither** TP nor FP (mirrors `validation.collect`'s `unsure`
  abstention) — the reviewer couldn't tell from the imagery, so no model is rewarded or
  penalized there.
- `False` / `duplicate` detections join neither set. A duplicate is a second hit on a ramp
  already in GT, so it becomes a false positive naturally under greedy 1:1 matching.

Scoring is uniform across models: every detector's output is reduced to center points
`(x, y[, confidence])` — a VLM box → its center — and greedily matched to GT within the
normalized radius **0.022** (the pano value in `rampnet/metrics.py`), with the same
anisotropic 1024/512 scaling. **Precision** counts detections on every pano; **recall** counts
only panos whose missed-ramp check is confirmed (`no_missed` set, or a missed mark exists), so
un-scanned panos can't bias it — the same gate `validation.collect` uses. Each of P/R/F1
carries a **Wilson 95% CI** (`rampnet.validation.wilson_interval`).

## Harness self-validation

Scoring RampNet's **own** bundle detections against this derived GT reproduces the published
verdict-based numbers within a small tolerance, which validates the harness before any VLM
spend:

| City | Harness (derived GT) | Published (verdict-based) |
|------|----------------------|---------------------------|
| richmond | P 0.964 / R 0.768 | P 0.960 / R 0.765 |
| bend | P 0.961 / R 0.761 | P 0.954 / R 0.758 |

The ~0.005 upward drift is expected: a RampNet `False` detection occasionally falls within
radius of a real GT point, which the per-detection human verdict scored differently. The
`compare.py` CLI prints both side by side.

## Caveats (read before quoting numbers)

- **RampNet-anchored GT.** The GT was assembled during a RampNet review. A reviewer scanning
  fresh for another model might catch a few more ramps; the complete-scan attestation
  (`no_missed`) mitigates this, but it is a known asymmetry.
- **Box → point reduction.** VLM boxes are scored by their centers, at the same radius as
  RampNet's point detections. Localization differences finer than the radius aren't measured.
- **Equirectangular projection disadvantages VLMs.** RampNet was trained on 2048×4096
  equirect panos; Gemini/Qwen were not, and ramps are tiny in a warped 4k+ pano. The current
  **whole-pano** input is therefore a **lower bound** for the VLMs — see tiling below.
- **No AP for VLMs.** VLM box detection carries no calibrated per-box confidence, so we report
  **operating-point** P/R for all models; AP / PR-curve (via `rampnet.metrics`) applies only
  where confidences exist (RampNet).

## Status

- **Shipped:** the model-agnostic scorer (`rampnet/detection_eval.py`), the comparison CLI
  (`scripts/model_comparison/compare.py`), and the RampNet-from-bundle baseline
  (`BundleRampNetDetector`). Tested (`tests/test_detection_eval.py`, `tests/test_model_comparison.py`).
- **Scaffolded (this increment does not run VLMs):** `GeminiDetector` / `QwenDetector`. Their
  image prep (whole-pano downscale) and box→point parsing (`gemini_boxes_to_points`,
  `qwen_boxes_to_points`) are real and unit-tested; the live model call (`_raw_detect`) raises
  `NotImplementedError` with wiring instructions.

## Running it

```bash
# RampNet baseline (no GPU, no keys — reads detections from the bundle):
python scripts/model_comparison/compare.py benchmark/richmond --models rampnet
python scripts/model_comparison/compare.py benchmark/bend --models rampnet

# Once a VLM is wired (below), add it; unwired models are skipped with a clear note:
python scripts/model_comparison/compare.py benchmark/richmond --models rampnet,gemini,qwen
```

## Next increments

1. **Wire the live VLM calls.** Implement `GeminiDetector._raw_detect` (against the current
   `google-genai` SDK; needs `GOOGLE_API_KEY`) and `QwenDetector._raw_detect` (load Qwen3-VL
   once in `_ensure_ready`, run on **Hyak** — the A40 OOMs at native res). Deps:
   `pip install -r requirements-vlm.txt`. The parse functions already exist.
2. **Equirectangular tiling** (`# TODO(tiling)` in `detectors.py`): slice each pano into
   overlapping crops, detect per tile, remap boxes to pano-normalized coords, dedup at seams —
   the fair-comparison input. Report whole-pano and tiled side by side.
3. **Add the `clovis` split** once the auto-labeler hands back its bundle; the harness is
   city-generic (it just needs `records.jsonl` + `verdicts.json` + `panos/`).

## Files

- `rampnet/detection_eval.py` — model-agnostic GT + scorer (pure, torch-free).
- `scripts/model_comparison/detectors.py` — `Detector` protocol, RampNet baseline, VLM scaffolds.
- `scripts/model_comparison/compare.py` — CLI.
- `requirements-vlm.txt` — optional VLM deps.
- `tests/test_detection_eval.py`, `tests/test_model_comparison.py` — guards.
