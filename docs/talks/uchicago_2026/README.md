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

Figure: a panorama with the predicted heatmap and extracted peaks. Not built here; needs a
checkpoint and a GPU (`stage_two/demo.py`).

## Slide 4. How it compares

Three baselines, in increasing strength:

| baseline | number | source |
|---|---|---|
| Prior curb-ramp detectors | Weld et al. 0.38 AP on our gold set vs RampNet 0.92; Tohme and Weld were at 26–34% precision | paper §2, `rampnet1_report.md` §3 |
| Nine zero-shot models (Gemini 3.1 Pro, Claude Opus 5, Qwen3-VL, Molmo2, OWLv2, Grounding DINO, ...) | RampNet 0.792 pooled F1 over eight US cities vs 0.575 for the best; leads on all twelve bundles by 0.11–0.34 | `model_scoreboard.md`, findings §2 row 1 |
| A supervised YOLO trained on the same dataset | 0.599 at its default threshold; at matched operating points and across seeds the gap is 0.016 F1, 95% CI [0.008, 0.024] | `seed_variance_51_135.md`, findings §2 row 3 |

The honest framing, which Jon chose: the dataset is the result; the keypoint formulation is a
small, real bonus. On the in-distribution gold set the two supervised models are level.

Figure: `comparison_f1.png` (eight rows, pooled F1, operating points in the footnote).

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

## Slide 8 (optional). RampNet 2.0

- Find, then measure, condition and tag. Multi-view fusion across panoramas. Composition and
  geometry over volume (NYC is 78% of training records; assessed inventories cannot reach 500k).
- North star: an AI labeller at least as good as a human, feeding Project Sidewalk.

Figure: not built; the multi-view figures under `figures/multiview_48/` are candidates.

## Figures in this folder

| file | slide | inputs |
|---|---|---|
| `pipeline_stage1.png` | 2 | `stage1_example_DJ8Zp111zu6KnMZz-0PHgQ.json`, `benchmark/bend/{records.jsonl,panos/}`, Bend inventory and streets geojson |
| `comparison_f1.png` | 4 | `analysis_out/scoreboard.json` |
| `comparison_f1_v2.png` | 4 | same chart with the N stated: cities, panoramas, ramps, challengers and providers (`--only comparison_v2`) |
| `transfer_imagery_country.png` | 5 | `analysis_out/scoreboard.json`; bayonne from PR #239 (constant in the script) |
| `recall_by_distance.png` | 6 | `docs/detection_recall_analysis.md` §1–§2 (constants in the script) |
| `deployment_validation.png` | 7 | sidewalk-auto-labeler `docs/server-agree-check.md` at 66c76d6 (constants in the script) |

Palette and mark conventions follow `scripts/analysis/scoreboard_figures.py`: one hue for emphasis,
neutral inks, class and group carried by position and labels.
