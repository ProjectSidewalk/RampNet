# Model comparison: RampNet vs. general-purpose models

Uses the standardized curb-ramp benchmark (`benchmark/{bend,richmond}/`) to compare
RampNet against off-the-shelf models. The question: does a general model match or beat the
purpose-trained RampNet on real deployment imagery (GSV + Mapillary 360)? The harness is
model-agnostic, so new models (issues #20, #39) plug in the same way.

Three classes of challenger, which fail differently and are worth keeping distinct:

| class | models | output | tunable? |
|---|---|---|---|
| **chat VLMs** | `gemini-3.6-flash`, `gemini-3.1-pro-preview`, `Qwen/Qwen3-VL-*` | boxes, no score | no — one operating point |
| **open-vocab detectors** | `google/owlv2-large-patch14-ensemble`, `IDEA-Research/grounding-dino-base` | boxes **with calibrated scores** | yes — AP, PR curve, threshold sweep |
| **pointing models** | `allenai/Molmo2-8B`, `allenai/MolmoPoint-8B` | **points** (RampNet's native format) | no score, but no box→point reduction |

The chat VLMs are all doing localization as a side skill, and they lose the same way: they
are false-positive-heavy (119–293 FP against RampNet's 9). The other two classes exist in
this harness to test whether that is a property of *general models* or of *chat models*.

**It is not.** See the results below — the purpose-built open-vocabulary detectors do far
*worse* than the chat VLMs on this task, not better.

## Results

Perspective tiling, match radius 0.022, all models scored against the same derived GT.
Open detectors are shown at their 0.05 cache floor; their tuned operating points are in the
sweep below. Run on Hyak (L40S); RampNet and Gemini rows are cache-scored.

**richmond** (124 reviewed panos, 310 GT ramps)

| model | P | R | F1 | AP | tp/fp/fn |
|---|---|---|---|---|---|
| **rampnet** | **0.964** | 0.768 | **0.855** | 0.763 | 238/9/72 |
| gemini-3.1-pro-preview | 0.631 | 0.700 | 0.664 | – | 217/127/93 |
| gemini-3.6-flash | 0.626 | 0.642 | 0.634 | – | 199/119/111 |
| Qwen3-VL-32B-Instruct | 0.760 | 0.297 | 0.427 | – | 92/29/218 |
| Qwen3-VL-8B-Instruct | 0.323 | 0.452 | 0.377 | – | 140/293/170 |
| owlv2-large-patch14-ensemble | 0.033 | **0.971** | 0.064 | 0.104 | 301/8799/9 |
| grounding-dino-base | 0.028 | 0.852 | 0.053 | 0.032 | 264/9321/46 |

**bend** (110 reviewed panos, 327 GT ramps)

| model | P | R | F1 | AP | tp/fp/fn |
|---|---|---|---|---|---|
| **rampnet** | **0.961** | 0.761 | **0.850** | 0.754 | 249/10/78 |
| gemini-3.1-pro-preview | 0.706 | 0.581 | 0.638 | – | 190/79/137 |
| gemini-3.6-flash | 0.608 | 0.587 | 0.597 | – | 192/124/135 |
| Qwen3-VL-32B-Instruct | 0.706 | 0.294 | 0.415 | – | 96/40/231 |
| Qwen3-VL-8B-Instruct | 0.379 | 0.336 | 0.357 | – | 110/180/217 |
| owlv2-large-patch14-ensemble | 0.037 | 0.951 | 0.070 | 0.093 | 311/8187/16 |
| grounding-dino-base | 0.038 | 0.850 | 0.073 | 0.049 | 278/6969/49 |

### What the numbers say

1. **RampNet still wins by a wide margin**, and nothing tested comes close on F1.
2. **Purpose-built detectors did worse than chat models, not better.** OWLv2's best F1 over
   the whole threshold sweep is **0.184** (thr 0.25: P 0.130 / R 0.310); Grounding DINO's is
   **0.073**. Both are far below Gemini-3.6-flash's 0.634. The issue-#39 hypothesis — that
   the chat VLMs' weakness was a *chat* problem — is refuted. Open-vocabulary detection with
   a text query is simply not selective enough for an object that looks like a slightly
   different patch of concrete.
3. **Capacity isn't the chat VLMs' problem either.** Qwen-32B moved to the *precise* end
   (P 0.760 / R 0.297) versus 8B's FP flood (P 0.323 / R 0.452) — the operating point moved,
   F1 barely did (0.427 vs 0.377).
4. **But OWLv2 has an extraordinary recall ceiling.** At its floor it finds **97.1%** of
   richmond's ramps, against RampNet's 76.8%.

### The recall-first angle: OWLv2 as a candidate generator

Recall matters more than precision here (a false negative is a permanently missing ramp; a
false positive is a cheap review). So the question is not OWLv2's F1 but whether it *sees*
what RampNet misses. It does — nearly all of it:

| OWLv2 thr | RampNet FN | recovered by OWLv2 | union recall | OWLv2 FP | FP per recovered ramp |
|---|---|---|---|---|---|
| 0.05 | 72 | **69** | **0.990** | 8799 | 128 |
| 0.10 | 72 | 60 | 0.961 | 5167 | 86 |
| 0.15 | 72 | 46 | 0.916 | 2923 | 64 |
| 0.20 | 72 | 34 | 0.877 | 1507 | 44 |
| 0.25 | 72 | 18 | 0.826 | 640 | 36 |

A RampNet ∪ OWLv2 oracle would reach **0.990** recall on richmond. The cost is the story: at
128 false positives per recovered ramp, versus **~6** for Gemini-3.6-flash (which recovered
20 of the 72 at 119 FP total — see issue #35), OWLv2 is a **6–20× less efficient**
complement. It is a recall oracle, not a usable candidate generator, unless a downstream
verifier can reject ~128 proposals per find more cheaply than Gemini can propose 6.

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
- **Box → point reduction.** Box models are scored by their box centers, at the same radius
  as RampNet's point detections. Localization differences finer than the radius aren't
  measured. Molmo is the exception — it emits points natively, so nothing is reduced.
- **Equirectangular projection disadvantages the challengers.** RampNet was trained on
  2048×4096 equirect panos; the others were not, and ramps are tiny in a warped 4k+ pano.
  The fair input is **perspective reprojection** (default, below): the pano is reprojected
  into overlapping rectilinear views. Whole-pano (`--tiling none`) remains available as a
  lower bound.
- **No AP for *chat* VLMs.** Gemini/Qwen box detection carries no calibrated per-box
  confidence, so those rows are a single **operating point** and their AP column reads `-`.
  AP / PR curves (via `rampnet.metrics`) are reported wherever confidences exist: RampNet,
  OWLv2, and Grounding DINO.
- **AP is measured on a slightly different slice than P/R.** AP needs one consistent recall
  denominator, so it is computed over the **recall-confirmed panos** only (the `no_missed`
  gate), while the precision column counts detections on every pano. On the current bundles
  nearly every pano is recall-confirmed, so the two slices are close — but they are not the
  same set. Note also that `--op-threshold` truncates the curve it is computed from: the AP
  printed alongside a thresholded row is the AP *of that row's operating range*, so quote
  full-range AP from a run without `--op-threshold`.
- **AP is not comparable across models at different floors — and RampNet's is truncated.**
  Every model's curve stops where its detections stop. RampNet's bundle detections were
  extracted at a **0.5** peak threshold, so its curve has no low-confidence tail at all
  (visible in `--sweep`: every row below 0.5 is identical) and its AP — 0.763 richmond /
  0.754 bend — is a **lower bound**, close to its recall ceiling of 0.768 / 0.761 times a
  near-1.0 precision envelope. The open detectors are cached down to 0.05, so their curves
  extend into a region RampNet's simply doesn't cover. Compare AP between OWLv2 and
  Grounding DINO freely; against RampNet, compare operating points, or re-extract RampNet's
  detections at a lower peak threshold first.
- **A swept threshold is tuned on the test set.** The `--sweep` table's best-F1 row is
  chosen on the benchmark itself. There is no separate val split, so quote it as an
  optimistic upper bound on what threshold tuning buys, not as a held-out result.

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

## Validating the box mapping

Reprojection is only half the pipeline; the other half is turning a provider's boxes back into
pano points, and that half has a silent failure mode — **box coordinate conventions differ by
provider and even between Qwen generations**. `scripts/model_comparison/dump_detections.py`
overlays a detector's raw boxes (red) on each view together with the pano's ground-truth ramps
(green) and ignore points (amber), so a mapping error shows up as boxes sitting consistently
off the ramps:

```bash
python scripts/model_comparison/dump_detections.py benchmark/richmond \
    --model qwen:Qwen/Qwen3-VL-8B-Instruct --out view_dump/qwen
```

### Qwen box coordinates are normalized 0–1000

`gemini_boxes_to_points` divides by 1000; `qwen_boxes_to_points` takes an explicit
`coord_space` because the family changed convention:

- **Qwen3-VL** (`norm1000`, the default): `bbox_2d = [x1, y1, x2, y2]` normalized to **0–1000**,
  as in the upstream 2D-grounding cookbook (`bbox_2d[0] / 1000 * width`). Being
  resolution-independent, the processor's smart-resize (which rounds to multiples of 28)
  **cannot** shift them — this retires the earlier "normalize by the processed size" caveat.
- **Qwen2/2.5-VL** (`pixels`): absolute pixels of the image the processor actually fed the model.

`infer_qwen_coord_space` picks by model id; `--qwen-coord-space` overrides. The two are *not*
auto-detected, because at a 1024px view they differ by only 2.4% — a wrong choice does not
crash, it introduces a small systematic localization bias. Verified empirically by rendering
one view at 512 / 1024 / 1400 px: the returned coordinates stayed in the same ~0–1000 band
instead of scaling with the image, and the overlay put boxes squarely on tactile ramps.

`dump_detections.py` draws all three prediction shapes: plain boxes (Gemini, Qwen), **scored**
boxes (OWLv2, Grounding DINO — the score is printed next to each box, since that is the
number the threshold sweep tunes), and **points** (Molmo, drawn as a red crosshair-in-circle
with the same visual weight as a box, so a scale error is equally obvious).

## What each model class buys you

### Open-vocabulary detectors: real confidences, so a real curve

`OwlV2Detector` and `GroundingDinoDetector` are text-prompted *detectors*, not chat models:
the "prompt" is a short query (`"a photo of a curb ramp"` for OWLv2, which is CLIP-based;
`"curb ramp."` — lowercase, period-terminated — for Grounding DINO), and every box comes
back with a **calibrated score**. The harness threads that score all the way through
(`pixel_boxes_to_points` → `dedup_points` keeps the highest-scoring copy of a cross-view
duplicate → `score_pano` matches greedily in score order), which unlocks three things no
chat VLM in this harness can offer:

- **AP** in the main table,
- **PR curves** (`--pr-out DIR` → one JSON per model plus a combined PNG),
- a **threshold sweep** (`--sweep`) — P/R/F1 at each cutoff, best-F1 row flagged.

That last one matters directly for the recall-first direction: a detector you can *tune*
toward recall is worth more than a chat model pinned at one operating point.

**`--score-threshold` is a cache floor, not the operating point.** Detections are computed
once down to a low score (default **0.05**) and cached; every higher operating point is then
a free local re-score (`--op-threshold`, `--sweep`) with no second model run. The floor is
part of the detector signature, so *lowering* it invalidates the cache and re-runs the model
— raising the reported threshold never does.

**OWLv2's boxes are relative to a padded square.** Its image processor pads to
`max(h, w)` (bottom/right) before resizing, so boxes live in that square's frame with the
image in the top-left corner; `owlv2_target_size` states that frame explicitly and
`pixel_boxes_to_points` normalizes by the *original* width/height, dropping centers that
land in the pad. Current transformers already scales OWLv2 boxes by `max(h, w)` internally
(`_scale_boxes`: *"for owlv2 image is padded to max size"*), so on this version passing the
square and passing the image's own `(h, w)` agree — verified on a 2:1 crop, where both put
the top box at y 0.815 against a true position of 0.817. Square views (the default rig) are
unaffected either way; whole-pano mode (`--tiling none`) is the only place the distinction
could bite, and passing the square is also correct under the older per-axis scaling.

### Molmo: points, not boxes

Molmo is the one challenger whose native output is a **point**, which is RampNet's own
output format — so it is the only apples-to-apples comparison in the table, with no
box→center reduction. There is no per-point score, so Molmo gets an operating point but no
PR curve.

**Its coordinate convention changed between generations**, and unlike Qwen's two box
conventions the two are distinguishable by *syntax*, so `molmo_points_from_text` infers the
scale per tag (override with `--molmo-coord-scale`):

- **Molmo 1** — `<point x="35.4" y="61.2" alt="...">` / `<points x1=… y1=… x2=… y2=…>`:
  coordinates are **percentages (0–100)**.
- **Molmo 2** — `<points coords="0 354 612; 1 700 480"/>`, triplets of `id x y`:
  coordinates are **scaled by 1000**, per the model card's own regex. (Issue #39 expected
  0–100 for all of Molmo; that holds for Molmo 1 only.)

A wrong scale here fails loudly rather than silently: points outside `[0,1]` after scaling
are dropped (as the model card's reference implementation does), so mis-scaled 0–1000
numbers divided by 100 land out of frame and the model appears to detect nothing.

`MolmoPoint-8B` is different again — it emits points as **special tokens** that only the
model can decode (`extract_image_points`, with metadata from the processor and a
constrained-decoding logits processor). `infer_molmo_mode` picks that path by model id;
`molmo_token_points_to_items` reads only the last two values of each returned row, because
the model card documents the leading ids two different ways.

**The `transformers==4.57.1` pin on the Molmo cards is real, not cautionary.** Under 5.14.1
Molmo's own Hub code fails at import:

```
TypeError: Unexpected keyword argument image_use_col_tokens.
  ... transformers_modules/allenai/Molmo2_hyphen_8B/.../processing_molmo2.py line 93
```

It also needs `einops` **and `requests`**, which the lean cluster env lacked. The fix is a
**dedicated env at the pin**, not a downgrade of the env the other models use — the harness
itself imports fine on 4.57.1, so only Molmo's interpreter differs:

```bash
conda create -p /gscratch/scrubbed/$USER/envs/molmo python=3.11 -y
MOLMOPY=/gscratch/scrubbed/$USER/envs/molmo/bin/python
$MOLMOPY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
$MOLMOPY -m pip install "transformers==4.57.1" accelerate pillow numpy einops requests
PYTHON=$MOLMOPY MODELS=rampnet,molmo:allenai/Molmo2-8B \
    sbatch -A <account> scripts/model_comparison/run_open_models.slurm
```

**Status: verify the overlay before quoting any Molmo number.** The parser has unit tests
over both syntaxes, but the coordinate scale has not been confirmed against real weights.
Run `dump_detections.py` on one pano first and check the red crosshairs sit on ramps — as was
done for Qwen. Nothing detected at all means the scale is wrong (try `--molmo-coord-scale`).

## Status

- **Shipped:** the model-agnostic scorer (`rampnet/detection_eval.py`), the comparison CLI
  (`scripts/model_comparison/compare.py`), the RampNet-from-bundle baseline
  (`BundleRampNetDetector`), the **perspective reprojection + dedup** (`equirect_tiling.py`),
  the **live `GeminiDetector`** (google-genai; API key or Vertex+ADC), the **live
  `QwenDetector`** (transformers; Qwen3-VL on a cluster GPU), the **live `OwlV2Detector` /
  `GroundingDinoDetector`** (with AP, PR curves and a threshold sweep), and the
  **`MolmoDetector`** (points; wired, unverified). Tested (`test_detection_eval.py`,
  `test_model_comparison.py`, `test_equirect_tiling.py`).
- **Smoke-tested locally** on `Qwen/Qwen3-VL-2B-Instruct` (the largest that fits an 8 GB dev
  GPU) to validate wiring, JSON parsing, and box mapping before spending cluster time. 2B is
  far too weak to benchmark — the real runs are 8B and 32B on Hyak.
- **Where runs happen:** benchmark numbers come from **Hyak** (or makelab2), never the dev
  box. The desktop is for de-risking a cluster job — a 1–2 pano wiring probe and a
  `dump_detections.py` overlay — and those results are smoke tests, not results.
- **Desktop and cluster agree exactly.** The 2-pano smoke on an RTX 3070 and on an L40S
  produced *identical* numbers (OWLv2 18/156/1/5, AP 0.356; Grounding DINO 18/160/1/3, AP
  0.247), and the overlay job reproduced the same 94 OWLv2 boxes across the same six views.
  So a desktop probe is a faithful rehearsal of the cluster job — worth knowing before
  spending an allocation on a wiring bug.

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

# Qwen3-VL (open weights, needs a GPU — see the Hyak runbook below):
python scripts/model_comparison/compare.py benchmark/richmond \
    --models rampnet,qwen:Qwen/Qwen3-VL-8B-Instruct

# Open-vocabulary detectors: AP in the table, plus the curve and the sweep.
python scripts/model_comparison/compare.py benchmark/richmond \
    --models rampnet,owlv2,gdino --sweep --pr-out evaluation_results/pr_richmond

# Scoring only (no GPU, no model load) once .model_cache holds the detections.
# Every operating point is a free re-score of that cache:
python scripts/model_comparison/compare.py benchmark/richmond \
    --models rampnet,owlv2,gdino --op-threshold 0.2
```

A model that can't run (missing credentials, missing client lib, remote code that won't load
on this transformers version) is skipped with a clear note rather than crashing the run — so
one broken model can't cost you the models that already ran.

## Running the open-weight models on Hyak

Benchmark runs go on the cluster, not the dev box — Qwen3-VL-8B is ~16 GB in bf16 (32B
~64 GB) and Molmo-8B ~16 GB, and even the small detectors should produce their reported
numbers where every other model's came from. Two launchers:

- `scripts/model_comparison/run_qwen.slurm` — the Qwen leg.
- `scripts/model_comparison/run_open_models.slurm` — OWLv2 + Grounding DINO (default), or
  Molmo via `MODELS=`. OWLv2-large and Grounding DINO-base are ~1–2 GB and finish in
  minutes on one card; Molmo-8B takes hours because it generates text per view.

**The results come back through the detection cache.** `cache_key` hashes only
`(label, detector signature, city, pano id)` — nothing machine-specific — so detections computed
on Hyak drop straight into a local `.model_cache/`. And when every pano of a model is already
cached, `score_model` skips `detector.prepare()` entirely, so the final table can be produced on
a laptop that cannot load Qwen at all.

```bash
# 1. Stage the repo plus the (git-ignored) bundle imagery. Send the NATIVE panos:
#    the harness downscales in-process, and pre-resizing re-encodes the JPEG,
#    which is not free (a past gold-set re-eval moved P +2.2 / R -1.8 on
#    re-encoding alone).
rsync -av --exclude .venv --exclude .model_cache --exclude 'benchmark/*/panos' \
      RampNet/ klone:~/RampNet/
rsync -av benchmark/richmond/panos/ klone:~/RampNet/benchmark/richmond/panos/

# 2. On a login node: build an env. The full environment.yml works (remember
#    CONDA_OVERRIDE_CUDA=12.6, or conda-forge silently installs CPU-only torch),
#    but this leg needs only numpy/PIL/torch/torchvision/transformers -- the
#    RampNet baseline reads detections from the bundle, so no timm, no model load.
#    A lean env off the CUDA wheel index is faster and has no CPU-fallback trap:
module load conda/Miniforge3-25.9.1-0
conda create -p /gscratch/scrubbed/$USER/envs/qwenvl python=3.11 -y
ENVPY=/gscratch/scrubbed/$USER/envs/qwenvl/bin/python
$ENVPY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
$ENVPY -m pip install "transformers>=4.57" accelerate pillow numpy

# 3. Pre-download the weights so the GPU job isn't billed for the transfer.
#    (~17 GB for Qwen-8B; OWLv2-large + Grounding DINO-base are ~2 GB together.)
export HF_HOME=/gscratch/scrubbed/$USER/hf
$ENVPY -c 'from huggingface_hub import snapshot_download as d; [d(m) for m in [
    "Qwen/Qwen3-VL-8B-Instruct",
    "google/owlv2-large-patch14-ensemble",
    "IDEA-Research/grounding-dino-base"]]'

# 4. Submit. -A is required (find yours: sacctmgr -nP show assoc user=$USER
#    format=Account,QOS). 8B fits one L40S; 32B needs two (device_map shards it).
mkdir -p logs
export PYTHON=$ENVPY
sbatch -A <account> scripts/model_comparison/run_qwen.slurm
BUNDLE=benchmark/bend sbatch -A <account> scripts/model_comparison/run_qwen.slurm
QWEN_MODEL=Qwen/Qwen3-VL-32B-Instruct sbatch -A <account> --gpus=2 \
    scripts/model_comparison/run_qwen.slurm

# 4b. The open-vocabulary detectors: minutes, one card, both cities.
sbatch -A <account> scripts/model_comparison/run_open_models.slurm
BUNDLE=benchmark/bend sbatch -A <account> scripts/model_comparison/run_open_models.slurm

# 4c. Molmo (hours — it generates text per view). Verify the box/point mapping on
#     one pano FIRST; the Molmo path has never seen real weights:
$ENVPY scripts/model_comparison/dump_detections.py benchmark/richmond \
    --model molmo:allenai/Molmo2-8B --out view_dump/molmo
MODELS=rampnet,molmo:allenai/Molmo2-8B \
    sbatch -A <account> scripts/model_comparison/run_open_models.slurm

# 5. Bring the detections home and score every model side by side, no GPU needed.
rsync -av klone:~/RampNet/.model_cache/ .model_cache/
python scripts/model_comparison/compare.py benchmark/richmond --sweep \
    --pr-out evaluation_results/pr_richmond \
    --models rampnet,gemini:gemini-3.6-flash,qwen:Qwen/Qwen3-VL-8B-Instruct,owlv2,gdino
```

Runs are resumable: a job that is preempted or times out has already cached everything it
finished, so re-submitting picks up where it stopped.

## Next increments

1. **Calibrate the reprojection rig — now measurable, and demonstrably costly.** With
   `pitch_deg=-30` the bottom ~40% of every view is the capture vehicle's hood and the black
   nadir cap, so roughly a third of every paid call is spent on pixels that cannot contain a
   curb ramp. The open-detector overlays show this is not merely wasteful: Grounding DINO's
   **highest-scoring box in a view (0.40) is the hood itself**, outranking its correct 0.22
   box on a real tactile pad. Because AP ranks by score, hood detections at the top of the
   ranking depress AP directly — which is a plausible part of why Grounding DINO's AP (0.032)
   trails OWLv2's (0.104) despite similar operating points. Masking the nadir/hood region is
   now a change whose benefit can be *measured* (ΔAP), not just argued. Report perspective vs
   `--tiling none` side by side.
2. **Add the `clovis` split** once the auto-labeler hands back its bundle; the harness is
   city-generic (it just needs `records.jsonl` + `verdicts.json` + `panos/`).
3. **Verify Molmo against real weights** (overlay first, then the run), and decide whether
   `MolmoPoint-8B`'s special-token path or `Molmo2-8B`'s XML path is the one to report.
4. **Prompt-sweep the open detectors before writing them off entirely.** `--owlv2-query` /
   `--gdino-query` are free hyperparameters and these models are cheap to run (43
   detections/min on one L40S; a full city is ~15 min). The current queries are a single
   untuned phrase each; "curb cut", "wheelchair ramp at a crosswalk", or a multi-query
   ensemble might move them. Given F1 0.184 vs RampNet's 0.855 this will not change the
   verdict, but it would tell us whether the ceiling is the *query* or the *model class*.
5. **If a recall-first candidate generator is wanted, compare complements on FP-per-find,
   not F1.** That metric ranks Gemini-3.6-flash (~6) far above OWLv2 (36–128) despite
   OWLv2's much higher recall ceiling — see the table above.

## Files

- `rampnet/detection_eval.py` — model-agnostic GT + scorer, AP/PR curve (pure, torch-free).
- `scripts/model_comparison/detectors.py` — `Detector` protocol, RampNet baseline, VLM /
  open-vocabulary / pointing detectors.
- `scripts/model_comparison/equirect_tiling.py` — perspective reprojection + point mapping + dedup.
- `scripts/model_comparison/compare.py` — comparison CLI (table, sweep, PR curves).
- `scripts/model_comparison/dump_views.py` — visual de-distortion QA (graticule overlay).
- `scripts/model_comparison/dump_detections.py` — visual mapping QA (boxes/points vs ground truth).
- `scripts/model_comparison/run_qwen.slurm` — Hyak launcher for the Qwen leg.
- `scripts/model_comparison/run_open_models.slurm` — Hyak launcher for OWLv2 / Grounding DINO / Molmo.
- `requirements-vlm.txt` — optional VLM deps.
- `tests/test_detection_eval.py`, `tests/test_model_comparison.py`,
  `tests/test_equirect_tiling.py` — guards.
