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
  equirect panos; Gemini/Qwen were not, and ramps are tiny in a warped 4k+ pano. The fair
  input is **perspective reprojection** (default, below): the pano is reprojected into
  overlapping rectilinear views. Whole-pano (`--tiling none`) remains available as a lower
  bound.
- **No AP for VLMs.** VLM box detection carries no calibrated per-box confidence, so we report
  **operating-point** P/R for all models; AP / PR-curve (via `rampnet.metrics`) applies only
  where confidences exist (RampNet).

## VLM input: perspective reprojection (fair) vs whole-pano (lower bound)

`--tiling perspective` (default) reprojects the pano into a ring of overlapping rectilinear
views (`equirect_tiling.default_views`: 90° FOV, −30° pitch toward the ground, 6 yaws 60°
apart → 30° overlap), runs the detector per view, maps each detection's center back to pano
coordinates (`perspective_point_to_equirect`), and merges detections across the overlaps
(`dedup_points`, with 0/1 seam wrap). This is what a VLM expects — undistorted photos —
so it's the fair comparison. `--tiling none` sends one downscaled whole-pano call as a
lower bound.

**Seams.** Neighboring views overlap by 30°, so a ramp near a tile boundary is seen whole
in at least one view; the duplicate detection from the adjacent view is merged by
`dedup_points` (within the match radius). Residual nuance: a ramp truncated at a tile edge
can yield a box whose center is offset enough to escape the dedup radius and double-count as
a false positive — the mitigation is enough overlap that each ramp is near-centered in some
view, which is a rig-tuning question (`fov_h_deg` / `n_yaw` / `pitch_deg`) to calibrate
empirically once the live VLM runs.

## Validating the reprojection

- **Numerical** (`tests/test_equirect_tiling.py`): round-trip identity (view point → pano →
  view recovers the original), the view center looks at its (yaw, pitch), and the
  gnomonic-correctness invariants — the horizon renders as a straight horizontal and
  meridians as straight verticals (a warped projection would bend them). Plus renderer/scalar
  agreement.
- **Visual** (`scripts/model_comparison/dump_views.py`): renders a real pano's views with a
  graticule overlay. Great-circle meridians + the equator must render as **straight lines**;
  buildings/poles should look like normal photos. `python scripts/model_comparison/dump_views.py
  benchmark/richmond --out <dir>`.

## Status

- **Shipped:** the model-agnostic scorer (`rampnet/detection_eval.py`), the comparison CLI
  (`scripts/model_comparison/compare.py`), the RampNet-from-bundle baseline
  (`BundleRampNetDetector`), the **perspective reprojection + dedup** (`equirect_tiling.py`),
  and the **live `GeminiDetector`** (google-genai; API key or Vertex+ADC). Tested
  (`test_detection_eval.py`, `test_model_comparison.py`, `test_equirect_tiling.py`).
- **Scaffolded:** `QwenDetector` (Qwen3-VL on Hyak) — image prep, reprojection wiring, and
  box→point parsing are real and tested; only the live `_raw_detect` raises `NotImplementedError`.

## Gemini credentials

The `GeminiDetector` reads credentials from the environment; `compare.py` auto-loads a
git-ignored repo-root `.env` (so nothing lands in the shell or transcript). Two options:

- **Vertex AI + ADC** (for orgs that disallow API keys). In `.env`:
  ```
  GOOGLE_GENAI_USE_VERTEXAI=true
  GOOGLE_CLOUD_PROJECT=your-project-id
  GOOGLE_CLOUD_LOCATION=global
  ```
  and once, in your own terminal:
  `gcloud auth application-default login && gcloud auth application-default set-quota-project <project>`
  (the SDK finds the ADC file automatically at runtime; gcloud itself isn't needed after login).
- **API key** (if allowed): `GOOGLE_API_KEY=...` in `.env`.

**Location matters for model availability.** The newest Gemini flash ids
(`gemini-3.6-flash`, `gemini-3.5-flash`) are served only on the `global` Vertex location;
regional endpoints (e.g. `us-west1`) lag — there they cap at `gemini-2.5-flash`. Use
`global` unless an org data-residency policy requires a region (the benchmark imagery is
public GSV/Mapillary, so residency is not a concern here). Vertex model ids differ from the
AI-Studio aliases (`gemini-flash-latest` only resolves on `global`); pin them explicitly with
`gemini:<model-id>` in `--models`.

## Running it

```bash
# RampNet baseline (no GPU, no keys — reads detections from the bundle):
python scripts/model_comparison/compare.py benchmark/richmond --models rampnet

# RampNet vs Gemini variants (needs credentials above). Each --models token is a
# provider or provider:model_id; variants of one provider become separate rows:
python scripts/model_comparison/compare.py benchmark/richmond \
    --models rampnet,gemini:gemini-2.5-flash,gemini:gemini-3.6-flash

# Cost control / smoke: cap panos; whole-pano lower bound instead of tiling:
python scripts/model_comparison/compare.py benchmark/richmond --models rampnet,gemini --limit 20
python scripts/model_comparison/compare.py benchmark/richmond --models gemini --tiling none
```

Unwired models (Qwen) are skipped with a clear note rather than crashing the run.

## Next increments

1. **Wire the live Qwen call.** Implement `QwenDetector._raw_detect` (load Qwen3-VL once in
   `_ensure_ready`, run on **Hyak** — the A40 OOMs at native res). Deps:
   `pip install -r requirements-vlm.txt`. Reprojection and the parse functions already exist.
   (Gemini is wired — see above.)
2. **Calibrate the reprojection rig** against the first live VLM run: tune `fov_h_deg`,
   `n_yaw`, `pitch_deg` (and consider trimming the wasted nadir/hood region) so ramps land
   near-centered in some view, minimizing seam-truncation false positives. Report perspective
   vs `--tiling none` side by side.
3. **Add the `clovis` split** once the auto-labeler hands back its bundle; the harness is
   city-generic (it just needs `records.jsonl` + `verdicts.json` + `panos/`).

## Files

- `rampnet/detection_eval.py` — model-agnostic GT + scorer (pure, torch-free).
- `scripts/model_comparison/detectors.py` — `Detector` protocol, RampNet baseline, VLM scaffolds.
- `scripts/model_comparison/equirect_tiling.py` — perspective reprojection + point mapping + dedup.
- `scripts/model_comparison/compare.py` — comparison CLI.
- `scripts/model_comparison/dump_views.py` — visual de-distortion QA (graticule overlay).
- `requirements-vlm.txt` — optional VLM deps.
- `tests/test_detection_eval.py`, `tests/test_model_comparison.py`,
  `tests/test_equirect_tiling.py` — guards.
