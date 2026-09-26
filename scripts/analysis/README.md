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
| `complementarity.py [model] [split]` | no | Oracle-union recall + the RampNet-miss ∩ challenger-hit set, with a chance null on that subset and the FP bill a naive union would actually pay (issue #35 gate). Takes any model spec (`provider` or `provider:model_id`); a bare non-provider token is read as a Gemini model id. Reads cached detections from `.model_cache`; split defaults to richmond. Pass the same `--vistas-input-size` the run used — it is part of the cache key. |
| `precision_by_distance.py` | no | Is precision worse at distance — i.e. is culling far detections worth it? (No.) |
| `threshold_sweep.py` | **yes** | Re-runs inference on all benchmark panos and sweeps `threshold_abs` × `min_distance`. |
| `peak_nms_check.py` | no | Would suppressing peaks closer than the match radius help? (No — 6 of the 10 within-R pairs in the reviewed records are real ramp pairs; issue #62.) Reads all seven splits' committed records, no panos needed. |
| `null_recall.py <bundle>` | no | How much of a model's recall is real detection vs. what the match radius hands out for free at that box density? (Open-vocab detectors: mostly the latter.) Re-scores cached detections from `.model_cache`; skips models that aren't cached rather than running them. |
| `depth_extract_da3.py [n]` | **yes** | Metric depth for every GT ramp via Depth Anything 3 on the reprojected views → `gt_depth_da3.json`. |
| `depth_analysis.py` | no | Recall vs true distance / apparent size + the resolution forecast. Needs `gt_depth_da3.json`. |
| `recall_by_depth_112.py` | no | **The distance axis on GSV's own depth (#112).** Re-derives distance for every GT point and detection of the four archived GSV splits from the labeler's depth payloads (its `depth.py`, `--labeler-root`), excludes stand-in-ground panos, and re-issues the recall-by-distance / size / forecast / precision tables on the flat 2.5 m axis, the depth axis and the labeler's depth-frame-scaled axis side by side, per split and pooled. Commits every row, so `--check` re-derives all tables with no payload; `--check --doc-tables` prints the doc's §0 tables. Looks planes up with image column = raw payload column (not the labeler's mirrored stored-column mapping; see the next row). |
| `depth_image_alignment_112.py` | no | **Which way round is a GSV depth payload against a benchmark JPEG? (#112)** Four independent checks (sky mask vs image, ground plane under GT points, plane boundaries vs image edges, seam continuity of the raw-space ray) of the image↔payload column mapping `recall_by_depth_112.py` uses → `analysis_out/depth_image_alignment_112.json`. Needs the payloads and the native-res `panos/` (`--panos-root`). |
| `farfield_forensics.py` | no | Is the far-field `visible` verdict deck a representative sample, and does apparent size actually separate a far-field hit from a far-field silent miss? (#46 Phase 0.) Committed caches, witness list, gallery verdicts and imagery manifests only. |
| `silent_activation.py` | **yes** | Is a `silent` miss attenuated or absent? Reads the heatmap inside each silent miss's match window plus a per-pano azimuth null (#46 Phase 1). Needs the native-res `panos/` (HF `projectsidewalk/rampnet-benchmark`); `--panos-root` points at the checkout holding them. |
| `compare_silent_activation.py` | no | Does a re-run of Phase 1 reproduce the committed `silent_activation.json`? Three outcomes (bytes / values / what moved), joined per miss, with class and null-p95 flips called out (#131). |
| `cascade_gate.py` | **yes** | Is a gated cascade possible? Partitions every GT ramp into `complementarity.py`'s four cells, then reads RampNet's heatmap at each with #46 Phase 1's instrument, so the question "does RampNet already produce something a prior could promote at the ramps the challenger recovers?" gets a bounded answer (#126). Floor peaks come from `analysis_out/op_cache/<split>.json`, not the bundle records. Needs the native-res `panos/` and the challenger already cached — pass the same `--vistas-input-size` the run used. |
| `cascade_cost_35.py` | no | What does the gated cascade cost? Promotes RampNet floor peaks in [T_lo, T_hi) that sit within r_gate of a challenger box, scores the result with the benchmark scorer against baseline, threshold-only and naive-union controls, and a cyclic-shift null, over a (T_lo, r_gate, c_min) grid (#35). Reads only committed inputs: `analysis_out/op_cache/`, `benchmark/model_detections/`, the bundles. Writes `analysis_out/cascade_cost_35/`. `--all-published` sweeps every (split, leg); `--summary` prints the tables in `docs/cascade_cost_35.md`. |
| `size_analysis.py` | no | Geometry-only size stratification (no depth model) + the hard-miss montage figure. |
| `overlap_test.py` | **yes** | Do the threshold and resolution levers target the same ramps? Writes per-GT-ramp hits at four thresholds to `analysis_out/overlap.json` (committed, #171) and stops there. It does not cross-tab by distance: `detection_recall_analysis.md` §5's per-band table came from a join of `overlap.json` to `gt_depth_da3.json` that is not committed, and the depth file is not committed either. |
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
| `silent_witness.py` | no | Did any *other* model detect a ramp where RampNet was silent? Witnessed ⇒ the imagery contains a recognizable ramp, so the failure is RampNet-specific (confirmed vocabulary). Brackets the sourcing-addressable population — see `docs/curb_ramp_data_sourcing.md` §0b for the current bracket, which is a function of the witness pool and so is not restated here. Chance-corrected, because the dense detectors witness most misses by density alone. `--models` defaults to the **frozen** `roster.WITNESS_POOL_46` (#122), the pool the #46 human pass was rated under; the pool it ran over is recorded in its JSON. Reads `.model_cache`; no GPU. |
| `miss_gallery.py` | no | Crops for the misses geometry cannot explain (#46 gallery half). **Checks the instrument before rendering**: `geom()` sizes ramps at the model's 4096-px input, but stored panos run 4096–16384 px wide, so it classifies each crop `parity` vs `advantaged` and renders a third "as the model saw it" panel so a reviewer compares pixel budgets instead of inferring. Needs `benchmark/<city>/panos` (`--panos-root` if run from a worktree). |
| `fp_gallery.py` | no | The FP half of the gallery: worst-N `isolated` false positives per model, through the same instrument and manifest as `miss_gallery.py`. Ranked by the model's own confidence where it has one; the sample size and what was left out are always printed, never silent. Reads `.model_cache` + `panos/`. |
| `make_tagger.py <gallery>` | no | Turns a rendered gallery into a keyboard-driven `tagger.html` beside it — one keystroke per crop, auto-advance, `localStorage` autosave, and an export keyed exactly like `benchmark/<city>/incremental_fp_tags.json`. Picks the verdict scheme from the manifest's own contents (miss vs FP). Local page by design: the crops are git-ignored files on disk. |
| `ps_supervision_audit.py fetch` / `report` / `hf-index` | no | **What supervision Project Sidewalk actually holds for tags, severity and position (#86, RampNet 2.0).** `fetch` pulls `rawLabels`, `validations`, `labelEdits` and `labelTags` from every deployment (public ones from the cities API, private ones from a committed hostname list) into a gitignored cache; `report` reduces it to `docs/ps_supervision_audit.md` plus the committed tables under `analysis_out/ps_audit/`: tier sizes (all / crowd-validated / trusted rater), by year and deployment, per-tag counts against each deployment's own vocabulary, severity vs tags with the tags→severity κ baseline, the two tag-reviewed corpora, label-edit provenance, and Owner rater drift. `hf-index` reads the CSVs out of the 30 GB HF `sidewalk-tagger-ai-validated` zip by HTTP range so §8 (tag drift since 2024, train/test pano leak) needs no download. `--trusted-users` takes a `role|username|user_id` list (not committed); without it tier 3 is the Owners. |
| `yolo_geometry_51.py` | no | The #51 equirect control: reads the committed geometry-pair reports under `docs/data/yolo_geometry_51/` and decomposes the tiles-vs-pano difference into geometry and training budget. `--check` fails on artifact drift. Its split population is **pinned to the 2026-08-30 run**, not the live registry. |
| `yolo_warmup_dip_72.py` | no | The YOLO baseline's warmup-LR collapse (#72), read from the committed `results.csv` files: per run, the epoch-1 mAP@50, the dip minimum and its epoch, the epoch it regained the epoch-1 level, no-box epochs, and the `lr/pg0` peak. `--markdown` prints the table in `yolo_baseline/README.md`; `--check` fails if any fact the caveat states stops holding, including the schedule and grid cell in each run's `args.yaml`. Nine independent runs — the grid plus the three seed replicates — show the dip (eleven committed curves; two are continuations that repeat their parents' early epochs). |
| `operating_point_parity_51.py` | no | RampNet vs the YOLO legs at **matched** operating points (#51): one uniform threshold per model, selected on a split the headline is never reported over. `--sensitivity` re-runs the selection on every candidate dev split; `--check` fails on artifact drift. |
| `plot_operating_point.py` | no | The headline figure: PR response per split + F1-vs-threshold → `docs/figures/operating_point_pr.png`. |
| `plot_storage_floor.py` | no | Storage-floor cost + recall ceiling → `docs/figures/storage_floor_ceiling.png`. |

`run_low_floor_extract.slurm` is the Hyak launcher for the one GPU step (one L40S, ~45 min for
1,859 panos across the splits it covers); it is resumable, skipping splits that already have a
cache.
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

These scripts belong to other questions and read none of the caches above:

| script | GPU | what it answers |
|---|---|---|
| `stage2_epoch_curve.py` | no | **The Stage 2 epoch curve (#84).** Extracts per-epoch auto-label validation loss from the committed TensorBoard events of Run A (`stage_two/run_a_84_events/`) and compares it against the paper run's own rescued events (`docs/data/rampnet1_stage2_run/`, #104) at full float32 precision. Reads them with `stage2_train_cost.read_scalars` — standard library only, no tensorboard install, and one parser rather than two. Checks every file against the `SHA256SUMS` committed beside it. See [`docs/stage2_epoch_curve_84.md`](../../docs/stage2_epoch_curve_84.md). |
| `plot_epoch_curve.py` | no | The figure for the above: both runs' curves on one absolute axis, and each epoch's excess over Run A's own minimum → `docs/figures/stage2_epoch_curve_84.png`. |
| `tag_benchmark_86.py` (+ `tag_benchmark_86.sh`, `tag_benchmark_86_snap.sh`) | **yes** (`infer`, `tagger-eval`, `train`); no (`labels`, `resplit`, `score`, `test-only`, `collect`) | **The curb-ramp tag benchmark of record (#86, RampNet 2.0 plan item 2).** Reproduces the ASSETS'24 DINOv2 tag baseline on the HF `sidewalk-tagger-ai-validated` CurbRamp test split with the tagger's own code at a pinned commit, then scores the same checkpoint on the pano-disjoint (leak-free) and pano-shared test subsets, and retrains the recipe on pano- and block-grouped re-splits (`tag_benchmark_86.sh train`, then `snapshots` / `finish`). `score` re-derives every number from the committed predictions on CPU. See [`docs/tag_benchmark_86.md`](../../docs/tag_benchmark_86.md). |
| `tag_review_list.py build` / `power` / `size` / `pitch-check` / `recent-edits` | no | **The RampNet 2.0 tag review list (#86 item 3).** Draws the stratified, seeded 500-item list `benchmark/tag_review/review_list.csv` (tag state x distance band x city, rare tags up-weighted, one item per physical ramp) from PR #175's `ps_supervision_audit.py fetch` cache, and writes `review_list.meta.json` with every input's sha256. `power` prints the kappa-precision table and `size` the positives each list size gives over ten seeds; `pitch-check` shows `pano_y` is world-frame (no camera-pitch term in the distance band); `recent-edits` counts the CurbRamp edits in the 30 days before the fetch (the protocol's known-limit figure). Every listed item carries per-rater prior-contact and validation-study proximity flags. The rubric it serves is a **working draft**: [`docs/tag_rubric_draft.md`](../../docs/tag_rubric_draft.md). |
| `tag_review_pull.py prod` / `sheet` / `sheet-template` | no | One rater's pass to `benchmark/tag_review/<rater>.json`, rubric text embedded: from production (`/v3/api/labelEdits` + `/v3/api/validations`, network) or from the blind review sheet. Steps: [`docs/tag_review_protocol.md`](../../docs/tag_review_protocol.md). |
| `tag_review_agreement.py` | no | Two rater exports to per-tag Cohen's kappa, prevalence, positive-specific agreement and severity weighted kappa; refuses exports made under different rubric text or lists. |
| `sam2_extent_83.py run` / `summarize` / `gallery` | **yes** (`run` only) | **SAM2 extent vs the whole-apron gold (#83 path 1, RampNet 2.0 plan item 7).** Point-prompted SAM2.1 Hiera-L on a gnomonic view centered on the point vs the plain equirect crop, from the gold box center and from the recorded (detection) point, at 90/76/60°; each mask's seam-aware equirect bbox is scored by IoU against `benchmark/<city>/boxes.json`. `summarize` and `gallery` are CPU and read the committed `analysis_out/sam2_extent_83/*_rows.csv`. Exact commands: `sam2_extent_83_runbook.sh`; results and decision: [`docs/sam2_extent_83.md`](../../docs/sam2_extent_83.md). |
| `crop_cutter_validation.py` | no | **The crop cutter (#86, RampNet 2.0 plan item 2b).** Does `scripts/crop_cutter.py` reproduce the HF `sidewalk-tagger-ai-validated` crops, and how much of the label set has a pano in the makelab2 store? Four steps (`sample` → `coverage` on makelab2 → `pick` → `compare`); reads individual HF crops by HTTP range, never the 30 GB zip. Committed outputs in `docs/data/crop_cutter/`. See [`docs/crop_cutter.md`](../../docs/crop_cutter.md). |
| `context_fov_86.py` (+ `context_fov_86.sh`, `context_fov_86.slurm`, `context_fov_86_env.slurm`) | no (the script); **yes** for the `train` stage it sequences (`tag_benchmark_86.py train` / `infer`, one Slurm job per arm on klone) | **The context experiment (#86, RampNet 2.0 plan item 4).** Does the tagger do better when it sees more of the street? Re-cuts the benchmark labels from the pano store with `crop_cutter.py` at the labeler's viewport and at label-centred 25° / 50° / 90°, trains the #178 recipe on each, and compares them on the same test rows with a paired pano bootstrap (`contrast`). `context_fov_86.sh` runs it end to end in stages; `contrast` and `report` re-derive every number from the committed predictions on CPU, and so does `control` with `CONTROL=final` (with `CONTROL=interim` it reads a makelab2-only file). See [`docs/context_fov_86.md`](../../docs/context_fov_86.md). |
