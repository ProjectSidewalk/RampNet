# Recall error-analysis scripts

The analysis behind [`docs/detection_recall_analysis.md`](../../docs/detection_recall_analysis.md)
and [`docs/operating_point.md`](../../docs/operating_point.md). Scripts read the committed
benchmark bundles in `benchmark/`; the ones that need pixels also need the native-res `panos/`
(git-ignored — they must be present locally).

Outputs go to `$RAMPNET_ANALYSIS_OUT` (default `analysis_out/`), which is git-ignored **except**
for two things committed on purpose so results survive without a GPU:

- `analysis_out/op_cache/*.json` — the low-floor detection caches (image-free, ~780 KB). Every
  number in `docs/operating_point.md` re-derives from these on CPU.
- `analysis_out/op/*.csv` and `*.json` — the derived result tables, so a figure quoted in prose
  can be checked against the table it came from.

The gallery crops under `analysis_out/op/*_incremental_fp/` stay ignored (181 MB of regenerable
PNGs); their irreplaceable part, the human A/B tags, is committed at
`benchmark/<city>/incremental_fp_tags.json`.

**`low_floor_sweep.py` and the two `plot_*.py` scripts need no GPU and no imagery** — they read
the committed caches only, so anyone can reproduce the operating-point numbers from a clean
checkout.

## Run order

| script | GPU | what it answers |
|---|---|---|
| `miss_analysis.py` | no | Are misses localization near-misses or blind? Are they hard (a VLM also missed) or RampNet-specific? |
| `complementarity.py [model] [split]` | no | Oracle-union recall + the RampNet-miss ∩ VLM-hit set (issue #35 gate). Reads cached VLM detections from `.model_cache`; split defaults to richmond. |
| `precision_by_distance.py` | no | Is precision worse at distance — i.e. is culling far detections worth it? (No.) |
| `threshold_sweep.py` | **yes** | Re-runs inference on all benchmark panos and sweeps `threshold_abs` × `min_distance`. |
| `peak_nms_check.py` | no | Would suppressing peaks closer than the match radius help? (No — 6 of the 10 within-R pairs in the reviewed records are real ramp pairs; issue #62.) Reads all seven splits' committed records, no panos needed. |
| `null_recall.py <bundle>` | no | How much of a model's recall is real detection vs. what the match radius hands out for free at that box density? (Open-vocab detectors: mostly the latter.) Re-scores cached detections from `.model_cache`; skips models that aren't cached rather than running them. |
| `depth_extract_da3.py [n]` | **yes** | Metric depth for every GT ramp via Depth Anything 3 on the reprojected views → `gt_depth_da3.json`. |
| `depth_analysis.py` | no | Recall vs true distance / apparent size + the resolution forecast. Needs `gt_depth_da3.json`. |
| `size_analysis.py` | no | Geometry-only size stratification (no depth model) + the hard-miss montage figure. |
| `overlap_test.py` | **yes** | Do the threshold and resolution levers target the same ramps? Needs `gt_depth_da3.json`. |
| `operating_point_curve.py extract` | **yes** | Inference once → all peaks down to a low score floor → per-pano cache (issue #54). Handles both bundle kinds, so `manual_gold` (independent YOLO GT, no verdict review) is covered too. `--tta` extracts the horizontal-flip-TTA arm instead (#78) — two passes per pano, mirrored heatmap un-flipped and maxed exactly as `stage_two/evaluate.py`; each arm must live in its own `--cache` dir (mixing is refused). |
| `operating_point_curve.py curve` | no | Continuous PR curve + honest AP + F1-vs-threshold from the cache (#54). |
| `operating_point_curve.py gallery` | no | Incremental-FP crops for the GT-completeness spot-check → corrected precision with an error band (#54). |
| `low_floor_sweep.py parity` | no | **Gate — run first.** Do the cached peaks at 0.55 reproduce each split's committed `records.jsonl`? Measured in match radii, since bit-exactness is the wrong bar (#54). |
| `low_floor_sweep.py sweep` | no | P/R/F1 **and detections-per-pano** vs threshold, per split, pooled, and per **imagery tier** (tier assigned per pano from camera provenance, not per split). |
| `low_floor_sweep.py hist` | no | GT-true vs GT-false confidence calibration with Wilson intervals — the promotion floor input for auto-labeler#27 stage 4. |
| `low_floor_sweep.py gtbias` | no | Measures the GT-anchoring bias: below 0.55 every TP comes from a reviewer *missed mark*, never a reviewed detection, so sub-0.55 precision is a lower bound by construction (#54/#55). |
| `low_floor_sweep.py corrected` | no | Applies the committed #55 A/B tags → corrected P/R per split and pooled, with an uncertainty band. |
| `low_floor_sweep.py floor` | no | Does the labeler's `DETECTION_STORAGE_FLOOR = 0.1` discard recoverable ramps? (Yes — 2.7% of GT.) Plus the recall **ceiling** on multi-view consensus. |
| `low_floor_sweep.py distance` | no | Where the recall gain from a lower threshold lands on the distance axis (uniform — so it stacks with multi-view rather than overlapping it). |
| `low_floor_sweep.py tagcheck` | no | Do the committed #55 tags still resolve against this cache? Tag ids are keyed to peak *coordinates*, so a re-extraction can silently orphan reviewer work. |
| `low_floor_sweep.py tta` | no | Flip-TTA vs single-pass at the operating points (#78): both arms on identical grid/GT per split + pooled US, AP per arm, and the four-lever decomposition — drop alone, TTA alone, both, and the **marginal TTA-after-the-drop** row the 2×-GPU decision prices against. `manual_gold` needs no TTA cache (its committed detections *are* a TTA export); the city splits read `extract --tta`'s `op_cache_tta/`. |
| `stage1_label_recall.py` | no | **E1 (#59)** — is the far-field cliff inherited from the Stage-1 *labels*, or is it the model? Stage-1 label recall vs model recall on the same 1,000 gold panos. Fetches two columns of the Hub test split over HTTP range requests on first run, then caches. |
| `miss_decomposition.py` | no | Of the recall we're missing, how much can more data even reach? Splits misses into far-field (pixel-starved, 57.8%) and near-field (42.2%) with a multi-view ceiling (#59, #38, #48). Committed caches only. |
| `miss_taxonomy.py` | no | **What actually caused each miss (#46).** Buckets every miss into merged / sub_threshold / localization / silent, so the near-field population above resolves into causes: only **0.023 of the 0.087 recall points is sourcing-addressable**. Includes a greedy-vs-optimal matcher check (a wash) and an azimuth-randomized null per bucket. Committed caches only. |
| `fp_taxonomy.py` | no | **What the FP flood is made of (#46).** Buckets every model's false positives into duplicate / near_gt / hood / isolated, with an *exact* arc-geometry chance baseline for the near-GT share — which shows OWLv2's and Grounding DINO's near-ramp FPs are entirely density (excess −0.2% and −0.7%). Reads `.model_cache`; no GPU and no model load. |
| `silent_witness.py` | no | Did any *other* model detect a ramp where RampNet was silent? Witnessed ⇒ the imagery contains a recognizable ramp, so the failure is RampNet-specific (confirmed vocabulary). Brackets the sourcing-addressable population at **0.009–0.022 recall pts**. Chance-corrected, because OWLv2 witnesses 121/128 by density alone. Reads `.model_cache`; no GPU. |
| `miss_gallery.py` | no | Crops for the misses geometry cannot explain (#46 gallery half). **Checks the instrument before rendering**: `geom()` sizes ramps at the model's 4096-px input, but stored panos run 4096–16384 px wide, so it classifies each crop `parity` vs `advantaged` and renders a third "as the model saw it" panel so a reviewer compares pixel budgets instead of inferring. Needs `benchmark/<city>/panos` (`--panos-root` if run from a worktree). |
| `fp_gallery.py` | no | The FP half of the gallery: worst-N `isolated` false positives per model, through the same instrument and manifest as `miss_gallery.py`. Ranked by the model's own confidence where it has one; the sample size and what was left out are always printed, never silent. Reads `.model_cache` + `panos/`. |
| `make_tagger.py <gallery>` | no | Turns a rendered gallery into a keyboard-driven `tagger.html` beside it — one keystroke per crop, auto-advance, `localStorage` autosave, and an export keyed exactly like `benchmark/<city>/incremental_fp_tags.json`. Picks the verdict scheme from the manifest's own contents (miss vs FP). Local page by design: the crops are git-ignored files on disk. |
| `plot_operating_point.py` | no | The headline figure: PR response per split + F1-vs-threshold → `docs/figures/operating_point_pr.png`. |
| `plot_storage_floor.py` | no | Storage-floor cost + recall ceiling → `docs/figures/storage_floor_ceiling.png`. |

`run_low_floor_extract.slurm` is the Hyak launcher for the one GPU step (one L40S, ~45 min for
1,859 panos across all eight splits); it is resumable, skipping splits that already have a cache.
Submitting from a non-interactive shell needs `PYTHON=<interpreter>` set explicitly — the
`source activate sidewalkcv2` fallback only works from a conda-initialized login shell.

The GPU scripts reproduce the deployment inference path exactly (resize 2048×4096 bilinear,
ImageNet norm, no TTA — see `sidewalk-auto-labeler/detectors/curb_ramp.py`), so
`threshold_sweep.py` at `(0.55, 10)` reproduces the committed `records.jsonl` detections.

`operating_point_curve.py` is the issue #54 operating-point analysis: unlike `threshold_sweep.py`
(which re-extracts peaks per discrete threshold), it extracts once at a low floor and carries each
peak's height as its confidence, so a single inference pass yields the whole continuous curve + AP.
Its `curve`/`gallery` steps are CPU-only and read the cache `extract` writes.

`low_floor_sweep.py` is the **cross-split** layer on that same cache — pooling, per-tier grouping,
calibration, the GT-anchoring measurement, the #55 correction and the storage-floor check. Run
`parity` before trusting anything else: it is the gate that catches a preprocessing divergence,
which every downstream number would otherwise silently inherit. The five Mapillary splits
reproduce their committed records bit-exactly; bend does not, and that is expected rather than a
failure — it is the only GSV split, and the GSV production path fed the model a 4096×2048
intermediate rather than the native-res bundle pano. `manual_gold` is exempt from the gate
entirely (its committed detections used flip-TTA).

Model weights load from the published HF artifact **by state_dict**, matching the deployment
inference path (not `AutoModel`); the pure scoring logic lives in `rampnet/detection_eval.py` and
`rampnet/metrics.py` and is unit-tested in `tests/test_operating_point_curve.py`.

## Depth Anything 3 setup

`depth_extract_da3.py` needs DA3, which ships its own package (not `transformers`):

```bash
git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git
export DA3_SRC=$PWD/Depth-Anything-3/src        # imported from src/, NOT pip-installed
pip install omegaconf einops addict opencv-python-headless plyfile pycolmap trimesh evo
```

Deliberately **skip** their `numpy<2`, `xformers` and `open3d` pins — they are unnecessary for
inference and will churn a working CUDA env. One import (`moviepy`, used only by the Gaussian-splat
video export) must be stubbed; create an empty `stubs/moviepy/__init__.py` + `editor.py` next to
`$DA3_SRC/..` and it is picked up automatically.

**Critical:** pass the *known* intrinsics. We synthesise the rectilinear views, so
`focal = (W/2) / tan(fov_h/2)` exactly (512 px for the default 90° FOV, 1024 px views). With
intrinsics supplied, `prediction.depth` is **already in metres** — do *not* apply the
`× focal / 300` formula from the DA3 README, which is for the no-intrinsics path and over-corrects
by ~1.65×. Intrinsics-naive models (e.g. Depth-Anything-V2 metric) come out ~3× long on these
wide-FOV views.

## Not part of the recall analysis

Two scripts in this directory belong to a different question and read none of the caches above:

| script | GPU | what it answers |
|---|---|---|
| `stage2_epoch_curve.py` | no | **The Stage 2 epoch curve (#84).** Extracts per-epoch auto-label validation loss from the committed TensorBoard events of Run A (`stage_two/run_a_84_events/`) and compares it to the paper run. Standard library only — it parses the TFRecord framing and the two protobuf messages directly rather than importing tensorboard, so it runs from a clean clone with no extra install. See [`docs/stage2_epoch_curve_84.md`](../../docs/stage2_epoch_curve_84.md). |
| `plot_epoch_curve.py` | no | The figure for the above: both runs' curves on one absolute axis, and each epoch's excess over Run A's own minimum → `docs/figures/stage2_epoch_curve_84.png`. |
