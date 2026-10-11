# Running the model comparison

This is the operational half of [`model_comparison.md`](model_comparison.md): what is shipped,
the credentials the paid legs need, how to run a leg, the Hyak launchers for the open-weight
models, and the file index. The six sections below were moved verbatim out of that document
under #145, headings unchanged and in their original order. Results, mechanisms and caveats
stay in `model_comparison.md`; where the moved text says "above", "below" or "the Molmo
section" about something not in this file, it means that document.

## Status

- **Shipped:** the model-agnostic scorer (`rampnet/detection_eval.py`), the comparison CLI
  (`scripts/model_comparison/compare.py`), the RampNet-from-bundle baseline
  (`BundleRampNetDetector`), the **perspective reprojection + dedup** (`equirect_tiling.py`),
  the **live `GeminiDetector`** (google-genai; API key or Vertex+ADC), the **live
  `QwenDetector`** (transformers; Qwen3-VL on a cluster GPU), the **live `OwlV2Detector` /
  `GroundingDinoDetector`** (with AP, PR curves and a threshold sweep), and the
  **`MolmoDetector`** (points; `Molmo2-8B` verified by overlay and run on all four scored
  cities — see the Molmo section; `MolmoPoint-8B`'s special-token path is wired but has not
  met real weights). Tested (`test_detection_eval.py`, `test_model_comparison.py`,
  `test_equirect_tiling.py`).
- **Every split now carries the full standing roster** (morgantown and budapest_district5 run
  2026-07-28). The remaining gaps, tracked in the coverage matrix at the top, are the
  GT-completeness correction (#55, done on two cities) and the null-recall pass on
  `manual_gold`.
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

## Claude on Vertex (#122): two constraints worth knowing before you wire it up

The Claude leg runs through **Vertex AI on the same credentials as the Gemini legs** —
same ADC, same project, same `global` location, no Anthropic API key and no new secret.
`--models claude:claude-sonnet-5`. Two things are not obvious and cost an afternoon to
discover:

**1. Structured outputs are blocked by org policy; a plain tool is the way in.**
This project's GCP organization sets `constraints/vertexai.allowedPartnerModelFeatures`,
which allow-lists *features* of partner models rather than the models themselves.
`structured_outputs` is not on the list, so `output_config.format` returns **400
FAILED_PRECONDITION** — and so does a tool marked `strict: True`, because that is
implemented as structured outputs underneath. A plain, un-`strict` tool passes and
returns the same schema-shaped `{"boxes": [...]}` in a `tool_use` block, which is what
`ClaudeDetector` uses. Measured 2026-08-15. `output_config.effort` is **allowed**, so the
cost lever survives.

**The tool is offered, not forced** (`--claude-tool-choice auto`, the default). Forcing it
with `tool_choice={"type": "tool", ...}` guarantees the answer arrives as a tool call, but
it **suppresses thinking entirely**, which makes `--claude-effort` inert — measured on one
view, forced gives 60 output tokens and 0 thinking at *both* `low` and `high`, while `auto`
gives 0 / 42 / 237 thinking at `low` / `high` / `max`. `forced` remains available and is the
better choice at `effort=low`, where there is no thinking to lose. Both settings are in the
cache signature, because both change what comes back.

The cost of *not* forcing is that a turn can end in prose instead of a tool call — a refusal,
a preamble, a fenced JSON block — so `boxes_from_claude_response` must treat the text path as
a first-class case rather than an afterthought. It scans for the first balanced JSON value
(the same `_first_json_blob` the Qwen path uses) and yields no boxes when there is none.
Getting this wrong is expensive in a specific way: a parse exception propagates out of
`_raw_detect` and costs **all six views of the panorama**, not one box. That is not
hypothetical — it is how the sonnet/low leg originally lost a pano (see the table below).

Unblocking `strict: True` would need an org admin to add
`publishers/anthropic/models/<model>:structured_outputs` to that constraint. It would buy
hard schema validation on top of the current shape guarantee — worth having, not worth
blocking on. Until then the parser treats the schema as a hint: a malformed *item* costs
one box, a malformed *response* costs no boxes, and neither costs a panorama.

**2. Enablement is per model, and it propagates unevenly.** Each Claude model is enabled
separately in Vertex Model Garden (Sonnet 5 and Opus 5 are different Marketplace
services), and for some hours afterwards Vertex intermittently answers a valid request
with `404 Publisher model ... was not found or your project does not have access to it`.
Measured the same day: 12/12 identical calls succeeded in one burst, then 3 of 5 panos
404'd minutes later. The Anthropic SDK does not retry 404 — it is a 4xx and normally
permanent — so `ClaudeDetector` retries it explicitly with backoff. Without that a leg
silently loses panos to a transient lie. A genuinely un-enabled model still fails, just
after four tries.

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
#
#    Partition: the launchers default to `-p gpu-l40s` (non-preemptible, lab-capped
#    at 2 GPUs, so legs queue behind each other). The ckpt scavenger queue is faster
#    when it's free and preemption only costs ~one pano of cache -- but CHECK
#    `squeue -u $USER` FIRST. If a long run is already parked there (the issue #51
#    YOLO baseline lives on ckpt-g2 for ~1-2 weeks), stay on gpu-l40s: these legs
#    are 12-30 min each and are not worth any risk to a multi-day job.
mkdir -p logs
export PYTHON=$ENVPY
sbatch -A <account> scripts/model_comparison/run_qwen.slurm
BUNDLE=benchmark/bend sbatch -A <account> scripts/model_comparison/run_qwen.slurm
# 32B needs two cards. Use --gpus-per-node, NOT --gpus: the launcher already
# sets --nodes=1 --gpus-per-node=1, and `--gpus=2` against that is rejected
# ("required nodes (2) doesn't fall between min_nodes (1) and max_nodes (1)").
QWEN_MODEL=Qwen/Qwen3-VL-32B-Instruct sbatch -A <account> --gpus-per-node=2 \
    scripts/model_comparison/run_qwen.slurm

# 4b. The open-vocabulary detectors: minutes, one card, both cities.
sbatch -A <account> scripts/model_comparison/run_open_models.slurm
BUNDLE=benchmark/bend sbatch -A <account> scripts/model_comparison/run_open_models.slurm

# 4c. Molmo (hours — it generates text per view). Verify the point mapping on one
#     pano FIRST — this overlay is what caught the image-index parsing bug on the
#     first real run (see the Molmo section), and any new checkpoint can pull the
#     same kind of trick:
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

## Files

- `rampnet/detection_eval.py` — model-agnostic GT + scorer, AP/PR curve (pure, torch-free);
  includes the YOLO manual-label loader for `manual_gold`.
- `scripts/fetch_manual_gold.py` — `manual_gold` imagery + records from the HF test split
  (`--images-only` for the per-machine imagery re-fetch, plus the `--audit` id checks).
- `scripts/export_gold_records.py` — RampNet detections for `manual_gold` + the
  reproduction gate (GPU).
- `scripts/model_comparison/detectors.py` — `Detector` protocol, RampNet baseline, VLM /
  open-vocabulary / pointing detectors.
- `scripts/model_comparison/equirect_tiling.py` — perspective reprojection + point mapping + dedup.
- `scripts/model_comparison/compare.py` — comparison CLI (table, sweep, PR curves).
- `scripts/model_comparison/dump_views.py` — visual de-distortion QA (graticule overlay).
- `scripts/model_comparison/dump_detections.py` — visual mapping QA (boxes/points vs ground truth).
- `scripts/model_comparison/run_qwen.slurm` — Hyak launcher for the Qwen leg.
- `scripts/model_comparison/run_open_models.slurm` — Hyak launcher for OWLv2 / Grounding DINO / Molmo.
- `scripts/analysis/null_recall.py` — real vs chance recall at a model's box density (the
  "how much of a detector's recall is real?" table). Cache-only; no GPU, no keys.
- `scripts/analysis/fp_taxonomy.py` — what the FP flood is made of (duplicate / near_gt / hood /
  isolated), with an exact chance baseline for the near-GT share. Cache-only; no GPU, no keys.
- `scripts/analysis/empty_response_check.py` — whether a Gemini leg's zero-detection panoramas
  are real or lost responses (#120 review). Published detections only; no cache, no GPU, no keys.
- `scripts/model_comparison/pricing.py` — verified-only per-token price table (each entry
  carries the date it was checked); prices what `--usage-log` records.
- `scripts/analysis/vertex_usage.py` — server-side reconciliation: actual billed tokens per
  model from Cloud Monitoring. Needs ADC on the billing project, so only its output is
  replicable from this repo.
- `scripts/analysis/vertex_effort_split.py` — divides one such daily total between two legs
  of the same model (the #122 low/high pairs) using minute alignment plus the deterministic
  input geometry. Refuses with `NOT SEPARABLE` when the legs leave no distinguishable
  trace, which is the Sonnet case. Same ADC requirement.
- `analysis_out/usage_log.jsonl` — committed, append-only record of what each paid run spent.
- `requirements-vlm.txt` — optional VLM deps.
- `tests/test_detection_eval.py`, `tests/test_model_comparison.py`,
  `tests/test_equirect_tiling.py` — guards.
