# RampNet slides for the UChicago Distinguished Lecture (October 2026)

**Status: draft outline and figures, 2026-10-05.** Six to seven slides on RampNet inside a talk on
urban computing. Every number here is read from [`rampnet1_findings.md`](../../rampnet1_findings.md)
or from the source named beside it; the figures regenerate with

```
python scripts/talks/uchicago_2026_figures.py --pano-dir benchmark/bend/panos
```

(`--pano-dir` is only needed for the pipeline figure; the benchmark imagery is on the Hub and
`scripts/unpack_benchmark_panos.py` restores it. The other four figures need nothing but the repo.)

Numbers not to put on a slide, because they were superseded: a 0.039 or 0.252 RampNet-vs-YOLO gap,
"blind at the seam", "72% seam dropout", any Tillicum cost other than the ledgered one. The list is
§3 of the findings page.

## Slide 1. Cities do not know where their curb ramps are

- A curb ramp is the gating feature of a street crossing for a wheelchair user. Cities are required
  to inventory them; most have no usable inventory.
- Deitz, Lobben and Alferez (2021) scored 178 US municipalities: 90% publish open street data,
  34% sidewalk data, 10% curb ramps (`curb_ramp_data_sourcing.md` §4, citing the paper's §3.1).
- Street-level imagery covers far more of the world than any inventory does.

Figure: a GSV panorama with every curb ramp circled, or a Project Sidewalk screenshot if the
audience has already seen one. Not built here.

## Slide 2. The idea: the cities that publish ramp coordinates already labelled their imagery

- Government GPS point + panorama pose = a bearing in the panorama; a small crop model trained on
  Project Sidewalk crops places the pixel label. No human in the loop.
- 214,376 panoramas, 849,895 labels, three cities (NYC 78% of records, Portland, Bend), 20% negatives.
- Agreement with a 1,000-panorama human gold set: precision 0.915, recall 0.928 (the corrected
  figures; the paper said 0.940 / 0.925).

Figure: `pipeline_stage1.png`. One real training panorama in Bend with four government points,
their bearings, the crop windows and the labels Stage 1 produced. The reviewer later confirmed all
four. Inputs: `stage1_example_DJ8Zp111zu6KnMZz-0PHgQ.json` (this panorama's row in the published
dataset), `benchmark/bend/records.jsonl`, `stage_one/dataset_generation/location_data/bend.geojson`.

## Slide 3. The model: whole 360° panoramas, points not boxes

- ConvNeXt V2 backbone, a single-channel keypoint heatmap over the 2048×4096 panorama, peaks are
  detections.
- Gold set, one-to-one matching: precision 0.949, recall 0.873, AP 0.92 (`README.md` erratum).

Figure: `heatmap_demo_paterson.png`. One Paterson panorama (not in training), the street band
of the input above and the model's heatmap below in the style of the 2023 talk slide; 9 confirmed
ramps, 9 peaks at the deployed threshold. The heatmap is saved beside the figure
(`heatmap_paterson_*.npz`, with checkpoint and run metadata) so the figure regenerates on CPU;
`uchicago_2026_heatmap.py` is the one GPU step. Two richmond heatmaps are saved too
(`1847752429062443`, 8 of 8; `1273933840289887`, 10 found and 4 far-field misses) if a harder
example is wanted: pass them as `split`/`pano` to `fig_heatmap`.

Video: `showcase/showcase.mp4` (1920×1080, 32 panoramas × 2.5 s). Each frame is the raw street
band above and the heatmap below with a ring on every peak ≥ 0.55, captioned with city, imagery
and the reviewer's verdict. Selection is mechanical from the committed verdicts
(`showcase/manifest.json`): per split, the two panoramas with the most confirmed ramps where every
detection was confirmed and nothing was missed, plus one seeded-random other, plus four true
negatives (no ramps, no detections). Eleven splits, three imagery sources, four countries. The
32 heatmaps are committed as 8-bit PNGs; frames and the mp4 are not (regenerate with
`uchicago_2026_showcase.py render` then `video`, needs ffmpeg).

## Slide 4. How it compares

Three baselines, in increasing strength:

| baseline | number | source |
|---|---|---|
| Prior curb-ramp detectors | Weld et al. 0.38 AP on our gold set vs RampNet 0.92; Tohme and Weld were at 26–34% precision | paper §2, `rampnet1_report.md` §3 |
| Nine zero-shot models (Gemini 3.1 Pro, Claude Opus 5, Qwen3-VL, Molmo2, OWLv2, Grounding DINO, ...) | RampNet 0.792 pooled F1 over eight US cities vs 0.575 for the best; leads on all twelve bundles by 0.11–0.34 | `model_scoreboard.md`, findings §2 row 1 |
| A supervised YOLO trained on the same dataset | 0.599 at its default threshold; at matched operating points and across seeds the gap is 0.016 F1, 95% CI [0.008, 0.024] | `seed_variance_51_135.md`, findings §2 row 3 |

The honest framing, which Jon chose: the dataset is the result; the keypoint formulation is a
small, real bonus. On the in-distribution gold set the two supervised models are level.

Figure: `comparison_f1.png` (eight rows, pooled F1, operating points in the footnote). Variants:
`comparison_f1_v2.png` states the N; `comparison_pr.png` shows precision and recall instead of
F1; `comparison_f1_with_gold.png` adds each model's manual_gold F1 as a diamond;
`comparison_f1_annapolis.png` is one city with every leg ever run there (18 challengers,
including the Claude Fable 5 / 5.1, Opus 5 and Sonnet 5 legs).

Why `manual_gold` is not in the pooled bar: the scoreboard holds it out as the in-distribution
reference. It is GSV from the three training cities, so pooling it with deployment cities would
mix two questions; and two of the challengers (Gemini 3.1 Pro, Claude Opus 5) have no published
manual_gold detections, so the pooled mean would not be over the same models.

## Slide 5. Trained on US Google Street View, it transfers to other cameras and countries

- Twelve benchmark bundles, three imagery sources (GSV, Mapillary 360° on consumer and survey rigs,
  Panoramax), four countries (US, Brazil, Hungary, France). RampNet keeps the top F1 on every split.
- What drops out of distribution is recall, not precision: the generalization gap from the gold set
  to deployed cities is −0.12 for RampNet, −0.19 to −0.28 for the YOLO arms; zero-shot models sit on
  the diagonal (`figures/scoreboard_generalization.png` is slide-ready as is).
- Same rural town, two cameras: GoPro Max 0.54, GSV 0.66. The rig matters more than the town.
- Bayonne (France, Panoramax) is the newest and weakest: F1 0.52, recall 0.38, still the top F1 on
  the split by 0.09. It is on [PR #239](https://github.com/ProjectSidewalk/RampNet/pull/239), not
  yet on main, and its ground truth is single-rater at medium confidence.

Figure: `transfer_imagery_country.png` (RampNet vs the best zero-shot model per split, grouped by
imagery source, labelled by country). Reads bayonne from a constant in the script until #239 merges.

## Slide 6. Where recall goes

- Recall is distance-limited; precision is not. Recall 0.84 within 8 m, 0.56 at 18–25 m, 0.18 past
  25 m; precision 0.94–1.00 in every band. Culling far detections would lose 132 true ramps to
  remove 4 false ones.
- The levers are the decoder and calibration, not vocabulary: of 427 misses, 29% were merged with a
  neighbour's peak, 39% fired below threshold, and only 8% of the silent remainder show no response.
  A bigger corpus of the same kind buys ~0.013 recall points.

Figure: `recall_by_distance.png`. Caveat on the slide: flat-ground distance at 2.5 m camera height,
two cities; the per-band rows are from `detection_recall_analysis.md` §1 and are not re-derivable
from the repo (the depth file is not committed), the 487/637 total is.

## Slide 7. Deployed

- The auto-labeler runs RampNet over a city's imagery, fuses views in world space and submits
  labels to that city's Project Sidewalk server, where volunteers validate them.
- Human validators agree with 97% of the AI's ramp clusters in Vancouver (GSV), 96% in Richmond
  (Mapillary, iSTAR Pulsar) and 90% in Laurens (Mapillary, GoPro Max). 64,814 / 12,962 / 1,575 AI
  labels submitted.
- In Laurens, one auditor labelling streets independently of the AI: the AI has a cluster within
  7.5 m of 68–85% of their ramps one-to-one (chance ~0.25), 91% by any-cluster coverage.
- Caveats that must travel: mostly one validator per city, the validation queue is not a random
  sample, the Laurens audit was 44 of 169 streets complete at the pull.

Figure: `deployment_validation.png`. Source: sidewalk-auto-labeler `docs/server-agree-check.md` at
`66c76d6` (PR [#119](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/119), merged
2026-10-04).

## Slide 8. RampNet 2.0

- Find, then measure, condition and tag. Multi-view fusion across panoramas. Composition and
  geometry over volume (NYC is 78% of training records; assessed inventories cannot reach 500k).
- North star: an AI labeller at least as good as a human, feeding Project Sidewalk.

Figure: `rampnet2_roadmap.png`. Three rows (find, tag, rate) with where each stands and what
2.0 adds, from `rampnet2_plan.md` §1 and §4.

Three more future-work figures, each from real data:

- **A corner over time**, `timelapse/timelapse_E15SCY6RTiDuNUu_trZk4Q.png` (and a second corner,
  `timelapse_0CvMs02xHlg3mCMHJdc6Xg.png`). The same Bend intersection in three Street View
  captures with the model's heatmap on each, and the city's inventory install dates: one ramp
  through 2021, nine ramps built 2023-05 per the city, nine detected in 2024-08. Built by
  `uchicago_2026_timelapse.py` (`candidates` ranks every benchmark panorama whose GSV history
  brackets an install date; `fetch` pulls metadata and tiles through the production Stage 1 path;
  `infer`; `render --dates`). Manifests and heatmaps committed; the historical panoramas are
  cached and gitignored, sha256 in the manifest. A finding on the way: on the first corner the
  model sees the ramps the inventory dates to 2010-02 already present in 2008-10, so that
  InstallDate is a record date, not a construction date. That is the dating question
  [#238](https://github.com/ProjectSidewalk/RampNet/issues/238) is about.
- **From a point to a measurement**, `measure_chain.png`. One Richmond ramp: the deployed
  keypoint, the whole-apron extent box from the gold set, the range from calibrated Depth
  Anything 3, the width that follows (an illustration of the geometry, labelled as such), and
  slope as the measurement that does not exist yet.
- **3D fly-around**, `flyaround/flyaround_richmond_99.mp4` and `flyaround_bend_7.mp4` (8 s
  orbits, 1920×1080). The MapAnything reconstructions behind the cross-view placement viewer
  (#48, PR #210): the corner's point cloud, every camera that saw it, and the ramp's ground-truth
  point placed in 3D. Point clouds committed as `flyaround/*.npz`; frames and mp4 gitignored,
  `uchicago_2026_flyaround.py render` then `video` rebuilds them (CPU, a few minutes). The clouds
  are the viewer's 150k-point subsample, which is why they read as points rather than surfaces.

## Figures in this folder

| file | slide | inputs |
|---|---|---|
| `pipeline_stage1.png` | 2 | `stage1_example_DJ8Zp111zu6KnMZz-0PHgQ.json`, `benchmark/bend/{records.jsonl,panos/}`, Bend inventory and streets geojson |
| `comparison_f1.png` | 4 | `analysis_out/scoreboard.json` |
| `comparison_f1_v2.png` | 4 | same chart with the N stated: cities, panoramas, ramps, challengers and providers (`--only comparison_v2`) |
| `transfer_imagery_country.png` | 5 | `analysis_out/scoreboard.json`; bayonne from PR #239 (constant in the script) |
| `transfer_imagery_us.png` | 5 | same, US splits only, so it reads as camera transfer alone (`--only transfer_us`) |
| `heatmap_demo_paterson.png` | 3 | `heatmap_paterson_*.npz` (from `uchicago_2026_heatmap.py`, GPU), `benchmark/paterson/{records.jsonl,verdicts.json,panos/}` |
| `showcase/showcase.mp4` | 3 | `showcase/manifest.json`, `showcase/heat_*.png` (from `uchicago_2026_showcase.py infer`, GPU), benchmark panos; frames + mp4 regenerate with `render` and `video` |
| `comparison_pr.png` | 4 | `analysis_out/scoreboard.json` (`--only comparison_pr`) |
| `comparison_f1_with_gold.png` | 4 | `analysis_out/scoreboard.json` (`--only comparison_gold`) |
| `comparison_f1_annapolis.png` | 4 | `analysis_out/scoreboard.json` (`--only comparison_annapolis`) |
| `recall_by_distance.png` | 6 | `docs/detection_recall_analysis.md` §1–§2 (constants in the script) |
| `timelapse/timelapse_*.png` | 8 | `timelapse/manifest_*.json`, `timelapse/heat_*.png` (GSV history via `uchicago_2026_timelapse.py fetch` + `infer`, GPU), Bend inventory |
| `measure_chain.png` | 8 | `benchmark/richmond/{boxes.json,records.jsonl,panos/}`, `analysis_out/da3_calibration_101/rows_points.jsonl` (`--only measure`) |
| `flyaround/flyaround_*.mp4` | 8 | `flyaround/*.npz` (MapAnything clouds from the #48 viewer); `uchicago_2026_flyaround.py render` + `video` |
| `rampnet2_roadmap.png` | 8 | `docs/rampnet2_plan.md` §1, §4 (constants in the script) |
| `deployment_validation.png` | 7 | sidewalk-auto-labeler `docs/server-agree-check.md` at 66c76d6 (constants in the script) |

Palette and mark conventions follow `scripts/analysis/scoreboard_figures.py`: one hue for emphasis,
neutral inks, class and group carried by position and labels.
