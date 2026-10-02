# Index of `docs/`

One row per document in `docs/`, so that every write-up the paper can be written from is findable
from one page. Each row gives the issue(s) the document opens with, what kind of document it is,
whether it is current, its own headline (lifted from its text, with a number only where the
document states one unambiguously, and with the qualifier the document attaches to it), and the
committed script its numbers come from, with whether that script has a check mode.

This is the **document-level** view. For the **per-finding** view of RampNet 1.0 (one line per
result, with the caveat that travels with it), read [`rampnet1_findings.md`](rampnet1_findings.md);
this page quotes headlines but does not repeat the per-finding caveats. Cite the document, not
this page.

**Adding a document:** add one row to the group it belongs to, in the same column order;
`tests/test_docs_index.py` fails if a document in `docs/` has no row, if a row links a file that
does not exist, if a `scripts/` path in the *reproduce* column is missing or unlabelled, if a
check label does not match the script, or if a number in a hook no longer appears in the document
it describes. When a PR listed under [Arriving on open PRs](#arriving-on-open-prs) merges, its
line moves into the table above (the test fails until it does).

Column key. **Kind**: result, negative result, protocol/rubric, plan/proposal, ledger, how-to,
report/index. **Status**: *final*; *final; part superseded by &lt;file&gt;* (the part named is
replaced, the rest stands); *in progress* (the document says it is a draft; an open PR is named
when there is one); *proposed* (a plan not yet approved). **Reproduce**: every script carries its
own label. "(check)" means the script has a `--check` flag or `check` subcommand that re-derives
committed output and fails on drift; where that check covers only part of the output, or is an
instrument check rather than an output check, the label says so. "(verify)" is a script's own
verification mode (a `verify` subcommand or a `--verify` flag). "(checked via …)" means the named script's check covers it. "(no check)" means
the script re-derives the numbers but does not compare them.

Supporting directories (not indexed row by row):

- [`assets/`](assets/): contact sheets and galleries embedded by `crop_cutter.md`,
  `crop_window_eval.md` and `sam2_extent_83.md`.
- [`data/`](data/): committed inputs and outputs too small for `analysis_out/` or pinned beside the
  doc that reads them (the paper run's Stage 1 and Stage 2 logs, `sacct` dumps, seed-variance and
  epoch-curve detections, the Vertex minute series).
- [`figures/`](figures/): the PNG/JPG figures the documents embed (scoreboard, operating point,
  epoch curve, recall misses, `multiview_48/`, `sidewalk_width_217/`).

## Evaluation & operating point

| file | issue(s) | kind | status | hook | reproduce |
|---|---|---|---|---|---|
| [`eval_protocol_verification.html`](eval_protocol_verification.html) | #9 | result | final | The over-counting in the published matcher is real, confirmed three independent ways; swapping only the matching rule moves precision −1.0 pt and recall −4.4 pts at the published operating point. | none in `scripts/` (the published code is executed inline in the report; corrected results in `stage_two/evaluation_results_new/`) |
| [`operating_point.md`](operating_point.md) | #54, #55 | result | final | Lowering the peak threshold from 0.55 to 0.30 buys +7.1 recall points pooled at a shallow GT-completeness-corrected precision cost, while detections per pano rise only from 1.86 to 2.23. | `scripts/analysis/low_floor_sweep.py` (no check), `scripts/analysis/operating_point_curve.py` (no check) |
| [`operating_point_parity_51.md`](operating_point_parity_51.md) | #51 | result | final; its 0.039 is re-read at n=9 in [`seed_variance_51_135.md`](seed_variance_51_135.md) | The published RampNet-vs-YOLO gap is mostly an operating-point artifact: at parity it is 0.039 F1, not 0.160. | `scripts/analysis/operating_point_parity_51.py` (check) |
| [`seed_variance_51_135.md`](seed_variance_51_135.md) | #51, #135 | result | final | RampNet's recipe beats the `y11x_tiles` recipe at matched operating points on the seven US splits by 0.016 F1, Welch 95% CI [0.008, 0.024] (n=9 RampNet replicates, extended under Amendment 2 after the n=3 read); the published 0.039 overstates it by about 2.5×, and RampNet still loses `manual_gold` at matched thresholds. | `scripts/analysis/seed_variance_read_51_135.py` (check) |
| [`seam.md`](seam.md) | #130, #132 | result | final | The 360° seam: three real defects (Stage 1 double-labels 8,361 label pairs, 10 duplicate GT pairs in `manual_gold`, cached detections dropped near the seam) and two claims retracted. | `scripts/analysis/stage1_seam_scan.py` (no check), `scripts/analysis/seam_review.py` (no check), `scripts/analysis/seam_response.py` (no check) |
| [`stage2_epoch_curve_84.md`](stage2_epoch_curve_84.md) | #84, #135 | result | final; part superseded by [`stage2_run_b_power_135.md`](stage2_run_b_power_135.md) (the curve shape) | Run A, the Stage 2 epoch curve: there is no resolvable human-labelled peak; read paired (#135 amendment), the plateau is epochs 2–6 and epochs 7 and 8 sit measurably below it. | `scripts/analysis/stage2_epoch_curve.py` (verify), `scripts/analysis/stage2_manual_gold_curve.py` (no check) |
| [`stage2_run_b_power_135.md`](stage2_run_b_power_135.md) | #135, #84 | result | final; part superseded by [`stage2_cosine_rung_135.md`](stage2_cosine_rung_135.md) (the recommendation) | `manual_gold` can resolve the effect Run B would plausibly produce only if read paired; pooling the benchmark splits does not help. No GPU time spent. | `scripts/analysis/benchmark_power_135.py` (no check), `scripts/analysis/dump_peaks_from_cache.py` (verify) |
| [`stage2_cosine_rung_135.md`](stage2_cosine_rung_135.md) | #135 | negative result | final | Pre-registered 8-epoch cosine rung: the primary is a TIE, no epoch clears the pre-registered bar at both ends of the measured s.e. bracket, and Run B was decided against on 2026-09-03. | `scripts/analysis/stage2_manual_gold_curve.py` (no check), `scripts/analysis/run_b_gate_135.py` (check), `scripts/analysis/benchmark_power_135.py` (no check) |

## Model comparison & cost

| file | issue(s) | kind | status | hook | reproduce |
|---|---|---|---|---|---|
| [`model_comparison.md`](model_comparison.md) | #20, #39, #145 | result | final | The comprehensive log of RampNet against general-purpose models; its headline, RampNet beats every off-the-shelf model tested, is real and wide but carries qualifiers that travel with it (zero-shot challengers, a single untuned prompt, box-centre scoring). | `scripts/model_comparison/compare.py` (no check) runs the legs; `scripts/analysis/scoreboard.py` (check) re-scores the committed detections |
| [`model_scoreboard.md`](model_scoreboard.md) | #171 | report/index | final | Every model, one table: twenty-one model legs and twelve splits, generated from the committed detections and checked against `model_comparison.md`. | `scripts/analysis/scoreboard.py` (check), `scripts/analysis/scoreboard_render.py` (checked via `scripts/analysis/scoreboard.py` --check), `scripts/analysis/scoreboard_figures.py` (no check) |
| [`claude_legs_122.md`](claude_legs_122.md) | #122, #156 | result | final | The Claude legs: first annapolis results for both models at both effort levels, the Claude Fable legs on the first-party API, how to reproduce the four legs, and how one billed day was split by effort. | `scripts/analysis/vertex_effort_split.py` (no check), `scripts/analysis/export_model_cache.py` (verify) |
| [`vistas_transfer_126.md`](vistas_transfer_126.md) | #126, #163 | result | final | Supervised transfer from Mapillary Vistas `Curb Cut`: at 1024 resolution parity F1 moves 0.516 → 0.534 on richmond, so "transfers but does not compete" stands; the resolution handicap was real and almost entirely a recall handicap. | `scripts/model_comparison/compare.py` (no check) runs the legs and `scripts/analysis/scoreboard.py` (check) scores them; secondary analyses `scripts/analysis/complementarity.py` (no check), `scripts/analysis/cascade_gate.py` (no check) |
| [`cascade_cost_35.md`](cascade_cost_35.md) | #35, #126 | result | final | The gated RampNet + Vistas cascade is VIABLE on richmond under the pre-stated rule, and what it buys is recall, not F1; the richmond setting does not transfer: under a rule stated before scoring it passes on gainesville only of the three GSV splits, and that pass turns on one false positive. | `scripts/analysis/cascade_cost_35.py` (check), `scripts/analysis/cascade_transfer_35.py` (check) |
| [`yolo_geometry_51.md`](yolo_geometry_51.md) | #51 | result | final; part superseded by [`operating_point_parity_51.md`](operating_point_parity_51.md) (the headline against RampNet) | Is the RampNet-vs-YOLO gap architecture or equirectangular input? About a third of it is not architecture: the residual is 0.160 F1 at the published operating points, not 0.252, and the doc says to read it as 0.039 at matched operating points; the geometry decomposition stands. | `scripts/analysis/yolo_geometry_51.py` (check) |
| [`stage2_training_cost.md`](stage2_training_cost.md) | #84 | result | final | One Stage 2 epoch is 3.5 hours on 16 GPUs, not 36, measured from the paper run's own TensorBoard events. | `scripts/analysis/stage2_train_cost.py` (verify) |

## Data provenance & sourcing

| file | issue(s) | kind | status | hook | reproduce |
|---|---|---|---|---|---|
| [`data_provenance.md`](data_provenance.md) | — | report/index | final | Where every piece of training data came from, which external services regeneration depends on, and which cities entered training (the contamination registry). | `scripts/analysis/gov_provenance.py` (no check), `scripts/build_street_derivative.py` (verify) |
| [`stage1_generation_cost.md`](stage1_generation_cost.md) | #18, #172 | result | final | Stage 1 generation: 97.91% yield, and the entire 2.09% loss is panoramas Google refused to serve, measured from the paper run's own logs; also the rescued paper Stage 1 accuracy and the corrected agreement figure (#172). | `scripts/analysis/stage1_yield.py` (no check), `scripts/analysis/stage1_agreement_172.py` (no check) |
| [`curb_ramp_data_sourcing.md`](curb_ramp_data_sourcing.md) | #59 | plan/proposal | final | Which cities a larger Stage 1 corpus could be sourced from, how much each would buy, and what a retrain would cost; nothing here has been acted on. | `scripts/analysis/sourcing_tables.py` (check), `scripts/analysis/fetch_inventory.py` (no check) |
| [`data_scaling_59.md`](data_scaling_59.md) | #59 | negative result | final | Would more training data buy recall? E1: the harm hypothesis is NOT supported (verdict FLAT: the model's distance cliff is ~2.5× steeper than Stage 1's label recall), plus the miss buckets. | `scripts/analysis/stage1_label_recall.py` (no check), `scripts/analysis/miss_decomposition.py` (no check), `scripts/analysis/silent_activation.py` (no check) |
| [`location_precision_assessment_96.md`](location_precision_assessment_96.md) | #96, #103 | result | final | The location-precision assessment, city by city, that gates every route to 500k: the temporal gate, the Stage 1 tolerance curve, Denver (Good under our own stated threshold, not the paper's), Seattle's offset, and the street-level review instrument. | `scripts/analysis/stage1_offset_tolerance.py` (no check), `scripts/analysis/temporal_gap.py` (no check), `scripts/analysis/inventory_review_summary.py` (no check), `scripts/analysis/sourcing_tables.py` (check) |

## Recall & geometry analyses

| file | issue(s) | kind | status | hook | reproduce |
|---|---|---|---|---|---|
| [`detection_recall_analysis.md`](detection_recall_analysis.md) | #112 | result | final | Where RampNet's recall goes: recall is distance-limited (reliable to ~18 m, effectively blind past 25 m) and precision is not the problem; metre labels re-measured on GSV depth in §0. | `scripts/analysis/recall_by_depth_112.py` (check), `scripts/analysis/depth_analysis.py` (no check) |
| [`input_res_sweep_25.md`](input_res_sweep_25.md) | #25 | negative result | final | The frozen checkpoint does not benefit from more input pixels on any split; at 2× (r4096) the pre-stated rule reads "hurts" on 10 of 11 splits (on bend, richmond and sao_paulo that rests on the precision drop, a lower bound). | `scripts/analysis/input_res_sweep_25.py` (check: an instrument check, r2048 must reproduce `op_cache`; its `sums` subcommand verifies the committed outputs) |
| [`two_scale_197.md`](two_scale_197.md) | #197, #25 | negative result | final | Two-scale inference (near field from 1×, far field from 2× upsampled), pre-stated primary: HURTS, ΔF1 −0.018 over the eight US splits; a threshold alone reaches the same recall for fewer false positives. | `scripts/analysis/two_scale_197.py` (check) |
| [`aug_transfer_82.md`](aug_transfer_82.md) | #82 | result | in progress (PR #235) | Augmentation as a rig-transfer lever: a frozen-model probe of which pixel-statistics axis the released checkpoint reacts to, the `--aug` flags in `stage_two/train.py`, and a paired fine-tune screen on all 12 bundles. | `scripts/analysis/aug_probe_82.py` (check: an instrument check, the untransformed arm must reproduce the #25 `r2048` caches; `report --check` re-derives the tables), `scripts/analysis/aug_finetune_82.py` (check) |
| [`laurens_paired_151.md`](laurens_paired_151.md) | #151 | result | final | Laurens paired on the corners both rigs saw: RampNet's own rig effect survives pairing (ΔF1 +0.112), but that RampNet is more rig-sensitive than the YOLO pano arms is not established. | `scripts/analysis/laurens_paired_151.py` (check) |
| [`perspective_photos_218.md`](perspective_photos_218.md) | #218 | negative result | final (the Richmond gallery is unrated and Seoul is not scored) | The released checkpoint mostly does not transfer to Richmond's flat Mapillary photos: the canvas embed hits 0.199 of in-view pool ramps at 0.30 against a chance floor of 0.093, and the same ramps from 360 panos are 0.458 above chance against 0.097 for the flat photos; no input mapping fixes it, and the pose shows no systematic bearing error. | `scripts/analysis/perspective_photos_218.py` (no check), `scripts/analysis/perspective_bearing_check_218.py` (no check), `scripts/analysis/perspective_figures_218.py` (no check), `scripts/analysis/seoul_photos_218.py` (no check) |
| [`bearing_audit_218.md`](bearing_audit_218.md) | #218 | negative result | final | Audit of the bearing geometry behind the flat-photo hit test: no systematic instrument error, so the headline in `perspective_photos_218.md` stands; no global geometry correction moves the canvas arm's above-chance rate at 0.30 by more than +0.009, and GoPro HERO11 frames (a split chosen after looking) hit 0.574 of their pairs at 0.55 against 0.726 for the panos on the same 36 ramps. | `scripts/analysis/bearing_audit_218.py` (no check), `scripts/analysis/bearing_audit_figure_218.py` (no check) |
| [`da3_calibration_101.md`](da3_calibration_101.md) | #101, #112 | result | final | Depth Anything 3 calibrated against GSV depth: DA3 reads range ~10.6% longer than Google's (median 1.106); pooled the difference is a scale, on Google's 2025–26 rig it is not. | `scripts/analysis/da3_calibration_101.py` (check) |
| [`subcell_decode_221.md`](subcell_decode_221.md) | #221 | result | final | The released model puts 99.0% of `manual_gold` peak columns on hi-res pixel 3 or 4 mod 8; a Gaussian decode from each peak's 3x3 coarse neighbourhood moves detections toward the human box centres on all five splits measured (CI excludes zero on four), by -0.727 px [-0.783, -0.673] on `manual_gold`, and no detection metric moves. | `scripts/analysis/subcell_decode_221.py` (no check) re-derives `results.json` from the committed detections, and its `compare` subcommand diffs a re-run against it; `scripts/analysis/subcell_decode_221_figures.py` (no check) draws the Examples figures |
| [`decode_e2e_221.md`](decode_e2e_221.md) | #221 | result | final | End-to-end check of the shipped sub-cell decode: `--decode argmax` gives byte-identical CSVs to main's evaluate.py, the gaussian decode cuts position error 5.080 → 4.353 px on a real GPU run, the HF export round-trips byte for byte, and the gap to the committed `evaluation_results_new/` is the JPEG quality-95 re-encode. | `scripts/analysis/decode_e2e_221.py` (no check), `scripts/analysis/decode_e2e_221_gpu.sh` (no check), `scripts/analysis/decode_e2e_221_q95.sh` (no check) |

## Multi-view & auto-labeler

| file | issue(s) | kind | status | hook | reproduce |
|---|---|---|---|---|---|
| [`multiview_48.md`](multiview_48.md) | #48 | result | final | Multi-view evidence per physical ramp, Phase 1: multi-view recall is already in production (0.945–1.000 of ramps per city), extra captures give diminishing returns, and misses are correlated across views (2.0× the count expected under independence). | `scripts/analysis/multiview_evidence_48.py` (no check), `scripts/analysis/multiview_challengers_48.py` (no check), `scripts/analysis/residual_gt_check_48.py` (no check) |
| [`per_ramp_recall_38.md`](per_ramp_recall_38.md) | #38, #48 | result | final | Per-ramp recall and how correlated per-view misses are: misses are correlated beyond a city- and range-stratified independence null (ratio 2.00 [1.74, 2.25]), the ratio grows with views per ramp, and 46 of 1,298 ramps (3.5%) are missed by every view within 25 m. | `scripts/analysis/per_ramp_recall_38.py` (check) |
| [`crossview_align_48.md`](crossview_align_48.md) | #48 | result | final | The cross-view placement harness and all arm families combined: the best arm is the post hoc `mapa_posed_pair`, confirmed but smaller on 686 fresh pairs (3.26° against `proj_height_auto`'s 4.22°). | `scripts/analysis/crossview_align_48.py` (no check), `scripts/analysis/crossview_combined_48.py` (no check), `scripts/analysis/crossview_fresh_48.py` (check: `pairs --check` covers the fresh pair list only; `score`, which gives 3.26°, has no check) |
| [`crossview_align_48/depth.md`](crossview_align_48/depth.md) | #48 | result | final | Monocular metric depth arms on the frozen 300 pairs: on GSV no depth arm beats `proj_height_auto`; on Mapillary (Richmond) raw depth from UniDepth v2 or DA3 beats the projection, but that is one city, chosen from 17 arms after the fact. | `scripts/analysis/crossview_depth_48.py` (no check), `scripts/analysis/crossview_arms/depth_mono.py` (no check) |
| [`crossview_align_48/matching.md`](crossview_align_48/matching.md) | #48 | result | final | Pairwise image matching arms on the frozen 300 pairs: the fallback problem is a matcher problem, and RoMa solves it (7% fallback against `lg`'s 70%); on GSV no matching arm beats the free `proj_height_auto` prior. | `scripts/analysis/crossview_matching_48.py` (no check), `scripts/analysis/crossview_arms/pairwise_matching.py` (no check) |
| [`crossview_align_48/multiview_3d.md`](crossview_align_48/multiview_3d.md) | #48 | result | final | Multi-view 3D arms (SfM, feed-forward 3D, splatting) on the frozen 300 pairs: feed-forward 3D (MapAnything) beats `proj_height_auto` (median 4.06° on these pairs) and does not fall back; its best arm, `mapa_posed_pair`, is post hoc. | `scripts/analysis/crossview_arms/_mv3d.py` (no check), `scripts/analysis/crossview_arms/ff3d.py` (no check) |
| [`crossview_align_48/semantic.md`](crossview_align_48/semantic.md) | #48 | result | final | Semantic and structural arms on the frozen 300 pairs: the best is `sem_chamfer_auto`, median 3.30° from the reference against 4.06° for `proj_height_auto`; it is not among the arms the 83-arm Bonferroni screen keeps. | `scripts/analysis/crossview_arms/semantic.py` (no check) |
| [`flat_mapillary_3d.md`](flat_mapillary_3d.md) | #214, #48 | negative result | final | Flat Mapillary imagery does not improve cross-view placement in Richmond (verdict proposed, not decided: do not pursue it for #48 placement); 12% of images within 30 m of the pool ramps are flat (1,422 of 12,150), and the un-thinned 360 panos did not help placement either. | `scripts/analysis/flat_mapillary_48.py` (no check), `scripts/analysis/flat3d/reconstruct.py` (no check) |
| [`mined_placement_158.md`](mined_placement_158.md) | #158 | result | final | Image-based placement of the labeler's mined targets: on Richmond `roma_local` moves 9 false positives within 5 m of a detected ramp (all-mined 0.431 → 0.627), but the 9 are not shown to be the mined ramp (6 of the 9 land on another fused site), and on GSV no arm helps; the measurement itself lives in the labeler. | `scripts/analysis/mined_placement_158.py` (no check) |

## RampNet 2.0 planning

| file | issue(s) | kind | status | hook | reproduce |
|---|---|---|---|---|---|
| [`rampnet2_plan.md`](rampnet2_plan.md) | #86 | plan/proposal | proposed | RampNet 2.0 plan: find, tag, rate. DRAFT, not yet approved (its status line says the census scripts are not yet committed; that predates `ps_supervision_audit.py`, the next row). | none (text only) |
| [`ps_supervision_audit.md`](ps_supervision_audit.md) | #86 | result | final | Sizes Project Sidewalk's label store as supervision for tags, severity and measurement, by tier, era, deployment and tag; replaces the hand-run census on #86. | `scripts/analysis/ps_supervision_audit.py` (no check) |
| [`tag_benchmark_86.md`](tag_benchmark_86.md) | #86 | result | final (edited on open PR #189) | The ASSETS'24 tagger reproduces exactly (mAP 0.34); a model trained without the train/test pano leak does not score lower. | `scripts/analysis/tag_benchmark_86.py` (no check) |
| [`crop_cutter.md`](crop_cutter.md) | #86 | result | final | Label crops cut from the makelab2 pano store at any field of view, validated against the HF crops (viewport NCC median 0.780 over 200 labels). The store is an unpublished local input. | `scripts/crop_cutter.py` (no check), `scripts/analysis/crop_cutter_validation.py` (no check) |
| [`context_fov_86.md`](context_fov_86.md) | #86 | negative result | final | Field of view vs tag accuracy: wider is worse at a fixed 256 px input (90° loses 0.074 mAP), and the street-dependent tags show no detected gain from any wider crop. | `scripts/analysis/context_fov_86.py` (no check) |
| [`crop_window_eval.md`](crop_window_eval.md) | #114, #83 | result | final | Crop-window rules scored against the `manual_gold` boxes, and what those boxes are: `manual_labels/` w/h is NOT object-extent gold. | `scripts/analysis/crop_window_eval.py` (no check) |
| [`sam2_extent_83.md`](sam2_extent_83.md) | #83, #86 | negative result | final | SAM2 extent from a point prompt is not production-grade (median IoU 0.260 against the whole-apron box on Richmond), and the projection does not matter. | `scripts/analysis/sam2_extent_83.py` (no check) |
| [`sidewalk_width_217.md`](sidewalk_width_217.md) | #217, #86 | result | final | Sidewalk width from one photo, arm 1 on the Seoul set with laser-measured ground truth and no GSV: on the held-out half, clear width to a mean absolute error of 0.77 m, and all 9 photos with GT below 1.2 m flagged at a precision of 0.41; the tails are heavier than the four VLMs', and this is the easy geometry (camera on the sidewalk), so it says nothing yet about width seen from the street. | `scripts/analysis/seoul_fetch_217.py` (verify), `scripts/analysis/sidewalk_width_217.py` (no check) re-derives the numbers and its `verify-seg` subcommand checks the label maps against `seg_meta.json`; `scripts/analysis/sidewalk_width_217_figures.py` (no check) draws the figures |

## Human review protocols

| file | issue(s) | kind | status | hook | reproduce |
|---|---|---|---|---|---|
| [`tag_rubric_draft.md`](tag_rubric_draft.md) | #86 | protocol/rubric | in progress (working draft, no open PR) | The curb-ramp tag rubric and review list: an explicit "tag it when" / "do not tag when" threshold per tag, with Jon's decisions applied; one rater for now, so the pass produces no κ. | none (text only) |
| [`tag_review_protocol.md`](tag_review_protocol.md) | #86 | protocol/rubric | in progress (working draft, no open PR) | How a tag review pass is run, recorded per rater with the rubric embedded, and how a second blind pass would be compared. | none (text only) |

## Ledgers, how-tos & indexes

| file | issue(s) | kind | status | hook | reproduce |
|---|---|---|---|---|---|
| [`rampnet1_findings.md`](rampnet1_findings.md) | #162 | report/index | final | RampNet 1.0's findings, one line each, with the document that holds each number and the caveat that travels with it. | none (text only; each row names its source) |
| [`rampnet1_report.md`](rampnet1_report.md) | — | report/index | in progress (draft, no open PR) | The lab's internal account of RampNet 1.0: what was built, what was measured after publication, what was corrected, and what it says about RampNet 2.0. | none (text only; each section names its source) |
| [`replication.md`](replication.md) | #143, #21 | ledger | final | The replication ledger: for every experiment, what a new student needs to reproduce it from a clean clone, and what is blocking them where they cannot. | none (text only; per-row commands in the doc) |
| [`compute_cost.md`](compute_cost.md) | #143 | ledger | final | What cluster compute has cost: 2,684.4 GPU-hours on klone at $0 in the 2026-08-19 pull (2,650.0 of them RampNet's; later klone pulls are recorded by job in their own sections), plus 674.7 GPU-hours on Tillicum at $607.24, the only billed compute. | `scripts/analysis/slurm_usage.py` (no check), `scripts/analysis/gpu_hours_as_of.py` (no check) |
| [`running_model_comparison.md`](running_model_comparison.md) | #145, #122 | how-to | final | The operational half of `model_comparison.md`: what is shipped, credentials for the paid legs, how to run a leg, the Hyak launchers, and the file index. | `scripts/model_comparison/compare.py` (no check) |
| [`adding_a_benchmark_city.md`](adding_a_benchmark_city.md) | — | how-to | final | End-to-end runbook and checklist for adding a city to the validation benchmark, and the committed numbers, figures and documents a new split invalidates. | none (text only; the runbook names each script) |
| [`tillicum.md`](tillicum.md) | #51, #70 | how-to | final (working notes; the opening banner predates the first runs) | Running RampNet jobs on Tillicum, UW-IT's usage-billed GPU cluster: access, the scheduler and cost model, migrating the Slurm scripts, and what the first runs measured. | none (text only; launchers under `scripts/model_comparison/`) |

## Arriving on open PRs

Not on `main` yet; listed so a reader knows they are coming. When each PR merges, its line moves
into the right group above; `tests/test_docs_index.py` fails until it does.

| file | PR | what it is |
|---|---|---|
| `docs/fair_metadata_150.md` | #190 | Croissant 1.1 + GeoCroissant 1.0 + RAI metadata for the dataset and benchmark (#150). |
| `docs/pu_training_86.md` | #182 | Plan: PU training of the tag head (#86 item 5), proposed. |
