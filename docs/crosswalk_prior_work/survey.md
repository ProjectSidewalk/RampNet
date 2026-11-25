# We found crosswalk data, no comparable benchmark

**What this is.** A web survey of public crosswalk datasets, models, benchmarks and city inventories,
run on 2026-11-08 for #274 by Claude research agents, and this synthesis of their five notes, which sit
beside it in this folder:
[`aerial_and_municipal_ground_truth.md`](aerial_and_municipal_ground_truth.md),
[`av_hd_map_datasets.md`](av_hd_map_datasets.md),
[`model_hubs_and_pretrained.md`](model_hubs_and_pretrained.md),
[`panoramic_street_view_accessibility.md`](panoramic_street_view_accessibility.md) and
[`model_name_candidates.md`](model_name_candidates.md). Committed on 2026-11-25 from an untracked copy
in the sidewalk-auto-labeler checkout, lightly edited; numbers and links are as the agents read them on
2026-11-08 (see the evidence tags below). The benchmark plan that builds on it is
[`../crosswalk_benchmark_plan.md`](../crosswalk_benchmark_plan.md). "The assignment" in the text means the
research brief the agents were given; it is not committed.

There is not "nothing", but in the searches run nothing tests this task. Crosswalks are labelled in about a dozen public road-scene datasets, two small human-labelled panoramic sets (PASS and WildPASS) have a crosswalk class, and autonomous-driving HD mapping routinely publishes a pedestrian-crossing AP on nuScenes. What we did **not find, in the searches run**, is a benchmark, leaderboard or pretrained model that scores individual marked crosswalks as points on full equirectangular panoramas, or any head-to-head of street-level crosswalk detectors on a shared independent test set. The most useful independent test is one the project has to assemble: held-out cities scored against official inventories with a blind street-level audit, backed by WildPASS (same imagery family) and Argoverse 2 (different cameras). Baselines should be **run on our ground truth**, because no published number is comparable to the Seattle audit (P 0.980, R ~0.85, F1 0.908, an internal figure from the assignment with no public URL). For the name, use **XwalkNet**, or PedXingNet if unmarked crossings become a firm part of the roadmap; keep CrossNet as an internal codename at most.

**Evidence tags used throughout.** **[P]** was read on the primary source: paper text, a taxonomy or schema file, a live REST endpoint, or a hub API on 2026-11-08. **[F]** was read through a fetch tool that returns a small-model summary, so table values were reported rather than seen; paper values whose note does not say how they were read are also tagged [F], to be conservative. **[T]** is a third-party summary. **[S]** is a search-result snippet, with the page never opened. Spot-check anything not tagged [P] before it goes in a paper.

## The assumption holds for image space and fails for BEV maps

The four literature scans agree once their scopes are separated. In HD mapping there **is** a de facto per-class benchmark: `AP_ped` on nuScenes, [reported across later methods](https://arxiv.org/pdf/2403.08639) [F] and matched by Chamfer distance at thresholds of 0.5, 1.0 and 1.5 ([MapTRv2](https://arxiv.org/html/2308.05736) [F]; [threshold usage](https://arxiv.org/pdf/2312.03341) [F]). It scores BEV polygons from calibrated multi-camera rigs, so it does not apply to single-pano detection. In image space, each paper found used its own data, task and metric, with at least five metrics across at least five viewpoints ([CDNet](https://link.springer.com/article/10.1007/s00521-022-07007-9) [S]; [X-CDNet](https://www.bohrium.com/paper-details/x-cdnet-a-real-time-crosswalk-detector-based-on-yolox/1011454158751924229-2631) [S]; [ROD](https://arxiv.org/html/2611.00681) [F]). No challenge track scoring crosswalks alone was found; the closest pool crossings with other elements, as in the [OLUS metric](https://raw.githubusercontent.com/OpenDriveLab/OpenLane-V2/master/docs/metrics.md) [P]. A search for independent reuse of CDSet in 2024-2025 papers found no head-to-head.

**On Hugging Face, no community-standard crosswalk model was found.** Among cards that could be read, none reports a metric on a named public test set. The most-downloaded crosswalk-capable checkpoint is the generic [Mask2Former Vistas model](https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-semantic) [P], which the project has already probed with classes 8 and 23 ([RampNet#274](https://github.com/ProjectSidewalk/RampNet/issues/274) [F]). The only crosswalk model found with a metric block is Project Sidewalk's own [crop validator](https://huggingface.co/projectsidewalk/sidewalk-validator-ai-crosswalk) [P], which is a classifier and out of scope for whole-pano detection per its card.

**Scope of these negatives.** General web search in English; Hugging Face, GitHub and Zenodo through APIs, with Hugging Face matching repo names only; ArcGIS Hub (about 2,600 items) and several national portals. Google Scholar, Semantic Scholar, ACM DL and IEEE Xplore were not searched directly, Chinese-language sources were not searched, and this was not a systematic review. An unindexed workshop or non-English paper could exist.

The three closest prior works differ from ours in specific ways. [Berriel et al. 2018](https://arxiv.org/abs/1805.11970) [P] used OSM `highway=crossing` to auto-label GSV 90-degree crops, but its output is binary per image, so **any claim that map-derived weak supervision for street-level crosswalks is new would be wrong**. [Li et al. 2023](https://journals.sagepub.com/doi/full/10.1177/23998083221112157) [S] is the closest audit application, with crosswalk counts aggregated to intersection-years from 922,144 GSV images and a [public CSV](https://zenodo.org/record/6803956) [F]. [PASS](https://ar5iv.labs.arxiv.org/html/1909.07721) and [WildPASS](https://ar5iv.labs.arxiv.org/html/2103.05687) [F] are the closest input modality, but they produce semantic masks on a 70-degree band with no instance or point evaluation. Our own [RampNet](https://arxiv.org/html/2508.09415v1) [F] supplies the architecture, so a reviewer may call this a new label type on an existing method; the contribution has to be the task, the data and the benchmark.

Li 2023 appears here in one role only: **related work to cite and to read**. The project's earlier decision to drop it as an inventory source, stated in the assignment, is a separate judgement about using its output as ground truth, and nothing in these notes revisits it. Its performance metrics, view geometry and output unit remain unread, and they are what the novelty wording depends on.

Do not claim "first". The defensible wording, **pending a read of Li 2023**, is: "We are not aware of a prior detector that localizes individual marked crosswalks as points in full equirectangular street-level panoramas", citing Berriel 2018 for map-derived supervision.

## Run one held-out-city test, two external checks, and baselines on our ground truth

Each row below answers a different question, so report them separately and never pool them. Seattle stays the development city and is excluded from all held-out scoring.

| Step | Test | Ground truth | Protocol | What it tells you | Main risks |
|---|---|---|---|---|---|
| 1 (primary) | **Held-out cities: Los Angeles, Austin, Paris, Bend, plus Calgary or Portland** | Filtered official inventories (table below) | Native 4096x2048 input. Filter on status and date fields, and gate by capture year. Score raw, then blind-audit samples of **both** FPs and FNs on street-level imagery, and report both figures | Cross-city generalization on the real task, which answers the likeliest reviewer objection | Pano coverage unmeasured (`--scan-only` each first). Completeness unmeasured. One feature is not always one crossing |
| 2 (external, same imagery family) | **WildPASS**: 500 annotated panos, 25 cities ([paper](https://ar5iv.labs.arxiv.org/html/2103.05687) [F]) | Independent human masks; connected-component centroids or mask containment | Upsample 2x (5.7 to 11.4 px/deg) and pad the 360x70-degree band into 2:1 | New geography, independent labels | Model has not seen padded input. Whether the [archive](https://raw.githubusercontent.com/elnino9ykl/WildPASS/master/README.md) [P] holds all 500 masks is unknown, as are crosswalk counts. **Open the archive first**; it decides whether recall is estimable |
| 3 (external, camera shift) | **Argoverse 2 Sensor + Map**: 1,000 scenarios, 6 US cities, 7-camera ring ([paper](https://ar5iv.labs.arxiv.org/html/2301.00493) [F]) | Vector crossings as [two 3D edges](https://argoverse.github.io/user-guide/api/hd_maps.html) [F] | Render each camera to equirect without blending; a detection counts if any view fires. Score "peak falls inside polygon" in world space | Camera-domain robustness with labels independent of any image model | **No marked/unmarked attribute**, so sample-audit precision. Ring resolution unconfirmed. [CC BY-NC-SA 4.0, open download](https://argoverse.github.io/user-guide/getting_started.html) [F] |

Three rules apply to every row. Fix the point definition before scoring and publish a radius-sensitivity curve: RampNet's [88 px true-positive radius](https://arxiv.org/html/2508.09415v1) [F] should not be assumed to carry over, because a crossing is an elongated object several metres long. Confirm that no held-out area contributed training panos; the crowd-label census spans [46 cities](https://github.com/ProjectSidewalk/RampNet/issues/274) [F], and those labels are positives only, never recall ground truth. Measure precision twice, against raw labels and after audit, since unmarked or stale records would otherwise depress it. nuScenes is a reasonable follow-up to step 3, with a measured [8.03 px map-to-image error](https://arxiv.org/abs/2407.21331) [F], which is about 0.35 degrees at its front-camera resolution by our arithmetic.

Baselines go on the audited Seattle set and the held-out cities. The order is by information per unit of effort.

| # | Baseline | Why | Protocol | Access |
|---|---|---|---|---|
| 1 | **OneFormer or InternImage on Vistas**, beside the existing Mask2Former probe | Separates a model effect from a Vistas-label effect | Union of crosswalk classes, connected components to points | Checkpoints via [OneFormer](https://github.com/SHI-Labs/OneFormer) and [InternImage](https://github.com/OpenGVLab/InternImage/blob/master/segmentation/README.md) [P] |
| 2 | **SAM 3** with a text prompt | Current zero-shot option, returns masks | Perspective views, reprojected | [Gated](https://huggingface.co/facebook/sam3) [P] |
| 3 | **Grounding DINO Base, OWLv2** | Open and cheap | Perspective views; box centres are a proxy | Apache-2.0 ([GDINO](https://huggingface.co/IDEA-Research/grounding-dino-base), [OWLv2](https://huggingface.co/google/owlv2-large-patch14-ensemble)) [P] |
| 4 | **Molmo2** | The only model confirmed here with native point output | Perspective views; tolerance in angular or ground distance | [Apache-2.0](https://huggingface.co/allenai/Molmo2-8B) [P] |
| 5 | **YOLO or CDNet trained on CDSet + RLMD** | "What public data gets you" | Perspective views | [CDSet](https://zenodo.org/records/8289874) CC BY 4.0, [RLMD](https://huggingface.co/datasets/veetinator/Road_Line_Marking_Dataset) MIT [P]; [CDNet](https://github.com/zhangzhengde0225/CDNet) weights untested |
| 6 | **Tag-filtered OSM crossing nodes** | The cheap map alternative | Filter `crossing:markings` and `crossing=`; `unmarked` is [not a safe negative](https://github.com/ProjectSidewalk/RampNet/issues/274) [F] | Open; Overture is [OSM-derived](https://docs.overturemaps.org/guides/transportation) [F], so do not count it separately |

Perspective projection for rows 2 to 5 is engineering judgement, not a measured finding. Open-vocabulary models are poorly suited to raw equirectangular input ([OOOPS](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5584_ECCV_2024_paper.php) [F]; [PanoSAMic](https://arxiv.org/pdf/2601.07447v1.pdf) [S]; [SphMind](https://arxiv.org/html/2609.33462v2) [S]), and **no published evaluation of these models on crosswalks or road markings was found**.

Do not use the following as benchmarks. Roboflow and Kaggle listings are unaudited and were never opened. [Metropolis](https://raw.githubusercontent.com/mapillary/metropolis_sdk/master/FORMAT.md) [P] has true equirect input but machine-generated masks, so it is an input-format sanity check at most, and it is possibly circular with a Vistas-trained filter. [FPVCrosswalk2025](https://data.mendeley.com/datasets/mcr2jwk5bp/1) [P] is 91% synthetic. [Cosmos-Drive-Dreams](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams) [F] video is synthetic. Vistas is not fully independent of any pipeline that uses a Vistas-trained model.

## What exists: inventory tables

**None of the scores below are comparable to each other or to our point F1.** They mix BEV Chamfer AP, mask IoU, image-level accuracy, box AP and F1, on different data. Berriel's dataset is about 95% negative, so its accuracy says little without class-balanced numbers.

### Street-level and panoramic crosswalk work

| Work | Task and unit | Imagery | Reported, as stated | Released | Evidence |
|---|---|---|---|---|---|
| [Berriel 2018](https://arxiv.org/abs/1805.11970) | Binary, per image; OSM labels | GSV 90-degree crops; 657,691 images, Brazil | 94.12% accuracy automatic; 96.30% with part hand-labelled | Scripts and models promised; [repo](https://github.com/rodrigoberriel/streetview-crosswalk-classification) not opened | [P] |
| [Li 2023](https://journals.sagepub.com/doi/full/10.1177/23998083221112157) | Counts per image, aggregated to intersection-year | GSV; 922,144 images, 4,925 hand-labelled | **Not obtained** | [CSV](https://zenodo.org/record/6803956), CC BY 4.0, no weights | Paper [S]; Zenodo [F] |
| [Adams 2022](https://www.mdpi.com/1660-4601/19/8/4548) | Image classifier for MAPS-Mini features | GSV, 5 cities | Accuracy: zebra 94.62%, line 80.20%. **Unverified**; the abstract's ">84%" conflicts with 80.20% | Not confirmed | [S] |
| [Mind Your Crossings](https://pmc.ncbi.nlm.nih.gov/articles/PMC5531764) | World-level zebra instances; satellite candidates validated on GSV | 4 regions, 7.5 km² | P 0.902-0.971, R 0.772-0.950. **Unverified**; closest metric unit to ours | Not found | [T] via [summary](https://a11y-paradise.onrender.com/reviews/69a488f012b3bc5f14238fcf) |
| [IJTST 2024](https://www.sciencedirect.com/science/article/pii/S2046043024000364) | Classification under occlusion | Aerial and street view, about 18,000 images | Street 82.58% vs aerial 55.59%. **Unverified** | Not confirmed | [S] |
| [PASS](https://ar5iv.labs.arxiv.org/html/1909.07721) | Pixel masks, crosswalk class | Annular 360-degree, 2048x692 | Crosswalk 52.9-62.1 by method on the PASS test set | [Repo](https://github.com/elnino9ykl/DS-PASS); licence unconfirmed | [F] |
| [WildPASS](https://ar5iv.labs.arxiv.org/html/2103.05687) | Pixel masks, crosswalk class | 360x70-degree, 2048x400, Street View; 500 labelled panos | Table 2: 4.2-28.9 baselines, up to 71.1. Metric unresolved (see conflicts) | Drive archive; code MIT, data terms unstated | [F] |
| [Mapillary detections](https://blog.mapillary.com/update/2017/09/22/editing-map-data-with-ai-detections.html) | Per-image detection, map features via [API](https://www.mapillary.com/developer/api-documentation) | Crowdsourced street level | **No published accuracy found** | Via API | [S] / docs [F] |

Other panoramic sets checked have no crosswalk class: [DensePASS](https://ar5iv.labs.arxiv.org/html/2108.06383), [CVRG-Pano](https://github.com/semihorhan/semseg-outdoor-pano) and [OmniCity](https://ar5iv.labs.arxiv.org/html/2208.00928) [F]. [Waymo PVPS](https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/src/waymo_open_dataset/protos/camera_segmentation.proto) [P] has only a generic road-marker class. KITTI-360's [label file](https://raw.githubusercontent.com/autonomousvision/kitti360Scripts/master/kitti360scripts/helpers/labels.py) [P] has no crosswalk label. The class lists of SHIFT, 360+x and PanorAMS were not resolved. Heatmap-style work on crosswalk centres was not found in two targeted queries, which is weak evidence; [LYTNet](https://ar5iv.labs.arxiv.org/html/1907.09706) [F] regresses the midline endpoints of a single crossing per phone image.

### Datasets with crosswalk labels

| Dataset | Label and geometry | View, scale | Licence and access | Published crosswalk score (metric) | Evidence |
|---|---|---|---|---|---|
| [Vistas v1.2 / v2.0](https://onedl-mmsegmentation.readthedocs.io/en/latest/_modules/mmseg/datasets/mapillary.html) | `Crosswalk - Plain`, `Lane Marking - Crosswalk`; v2.0 adds marking-only; masks | Perspective, 25,000 images | [CC BY-NC-SA 4.0](https://datasetninja.com/mapillary-vistas-dataset) [T] | SegFormer-B0 [63.80 / 21.09 IoU](https://arxiv.org/abs/2108.07007) [P] | Classes [P] |
| [BDD100K](https://raw.githubusercontent.com/bdd100k/bdd100k/master/bdd100k/label/label.py) | `crosswalk` lane category | Dashcam, 70K train | Devkit BSD-3; portal unreachable 2026-11-08 | [ODS-F 9.37](https://ar5iv.labs.arxiv.org/html/1805.04687) at tau=10 [F] | Class [P] |
| [ApolloScape Lane](https://raw.githubusercontent.com/ApolloScapeAuto/dataset-api/master/lane_segmentation/helpers/laneMarkDetection.py) | `zebra`, mask | 2 cameras, about 165K, China | Unconfirmed | None per class found | Class [P] |
| [A2D2](https://ar5iv.labs.arxiv.org/html/2004.06320) | `Zebra crossing`, mask | 41,277 frames, Germany | CC BY-ND 4.0, [S3](https://registry.opendata.aws/aev-a2d2/) | None found | Class from third-party loader |
| [CeyMo](https://ar5iv.labs.arxiv.org/html/2110.11867) | `Pedestrian Crossing`, polygons | Dashcam, 839 instances | Drive, no licence stated | F1 94.95-96.86 (road-marking detection) | [F] |
| [VPGNet](https://ar5iv.labs.arxiv.org/html/1710.06288) | `Crosswalk`, grid cells | Seoul, 13,632 instances | Request form | None found | [F] |
| [CDSet](https://huggingface.co/datasets/zzd0225/crosswalk-detection-dataset) | Boxes | Dashcam, 3,434 images | Apache-2.0 on HF, [CC BY 4.0 on Zenodo](https://zenodo.org/records/8289874) | [F1 94.83%](https://link.springer.com/article/10.1007/s00521-022-07007-9) (CDNet, own data) **unverified** [S] | Dataset [P] |
| [CD9K](https://www.sciencedirect.com/science/article/abs/pii/S1047320324001627) | Boxes, derived from BDD100K | Dashcam, 9,387 images | No confirmed download | [AP50 93.3](https://www.bohrium.com/paper-details/x-cdnet-a-real-time-crosswalk-detector-based-on-yolox/1011454158751924229-2631) (X-CDNet) **unverified** [S] | [S] |
| [RLMD](https://huggingface.co/datasets/veetinator/Road_Line_Marking_Dataset) | Polygons, 25 categories | Driving, 2,137 images | MIT | Paper not opened | [P] |
| [ROD](https://arxiv.org/html/2611.00681) | Boxes | Pedestrian view, 2,493 instances, 189 test | [MIT, constituents vary](https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO) | Recall 0.783-0.815, YOLO nano family | [F] |
| [SANPO](https://arxiv.org/html/2309.12172) | Panoptic "thing" class | Chest and head stereo, 112K masked | CC BY 4.0, wording ambiguous | None per class | [F] |
| [nuScenes map](https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/python-sdk/nuscenes/map_expansion/map_api.py) | `ped_crossing` polygon | 6 cameras, 40K keyframes | [CC BY-NC-SA 4.0](https://ar5iv.labs.arxiv.org/html/1903.11027) | [AP_ped 46.3-68.1](https://arxiv.org/html/2308.05736) (BEV Chamfer) [F] | Class [P] |
| [Argoverse 2](https://argoverse.github.io/user-guide/api/hd_maps.html) | Two 3D edges | 7-camera ring, 6 US cities | CC BY-NC-SA 4.0, open | MapTRv2 [62.9 (2D), 60.7 (3D)](https://arxiv.org/html/2308.05736) (BEV Chamfer) [F] | [F] |
| [Waymo Perception](https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/src/waymo_open_dataset/protos/map.proto) | Map polygon | 5 cameras, front and sides | [Non-commercial](https://waymo.com/open/about/) | None found | [P] |
| [HRDX](https://arxiv.org/html/2606.17080) | Polygon | 6 wide cameras, Bay Area | [Non-commercial, request](https://usa.honda-ri.com/hrdx) | AP 43.82-73.29 (BEV) | [F] |
| [KITScenes](https://arxiv.org/html/2606.02956) | Separate `zebra_crossing`, `pedestrian_crossing` | 9 cameras, 360-degree, Germany | [CC BY-NC 4.0, gated, early release](https://kitscenes.com/) | Leaderboard planned, crosswalk scoring unconfirmed | [F] |

Marked status matters most when reading this table: **nuScenes, Argoverse 2 and Waymo carry no marking attribute**, so an unmarked polygon counts against a marked-crosswalk detector. No crosswalk or road-marking label appears in [Cityscapes](https://raw.githubusercontent.com/mcordts/cityscapesScripts/master/cityscapesscripts/helpers/labels.py), [IDD](https://raw.githubusercontent.com/AutoNUE/public-code/master/helpers/anue_labels.py) or [nuImages](https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/docs/instructions_nuimages.md) [P]. [ACDC](https://ar5iv.labs.arxiv.org/html/2104.13395) [F] and [CULane](https://xingangpan.github.io/projects/CULane.html) [F] also lack one, while [WildDash 2](https://www.wilddash.cc/) [F] has only generic road markings.

**Roboflow and Kaggle are snippet-only [S], with pages never opened.** Listed sizes include [SafeWalkBD, 10.2k, multi-class](https://universe.roboflow.com/search?q=class%3Azebra+crossing), [Pedestrian-Zebra, about 5.2k](https://universe.roboflow.com/distracted-driving-yu8re/pedestrian-zebra), whose overlapping crosswalk and zebra classes illustrate the fragmentation, and [Segmentation 3, 1,949](https://universe.roboflow.com/janissy/segmentation-3-pqrfl). On Kaggle, [DataCluster Zebra Crossing](https://www.kaggle.com/datasets/dataclusterlabs/zebra-crossing) reads as a commercial teaser. Image counts, annotation quality and provenance are unverified.

### Models

| Model | Output | Crosswalk evidence | Evidence |
|---|---|---|---|
| [Mask2Former Vistas](https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-semantic) | Masks; [classes 8 and 23](https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-semantic/raw/main/config.json) | No per-class score on card or [model zoo](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md) | [P] |
| [SegFormer-B0 on Vistas](https://arxiv.org/abs/2108.07007) | Masks | Only verified per-class number found: `Lane Marking - Crosswalk` 63.80, `Crosswalk - Plain` 21.09 IoU | [P] |
| [CDNet](https://github.com/zhangzhengde0225/CDNet) | Boxes, YOLOv5-era, dashcam | F1 94.83% on own data, **unverified**; only downloadable dedicated detector that merits running | Repo [P]; metric [S] |
| [X-CDNet](https://github.com/xgyutu/X-CDNet) | Boxes | AP50 93.3 on own data, **unverified**; weights unchecked | [S] |
| [CrosswalkNet](https://arxiv.org/abs/2506.07885) | Aerial oriented boxes | P 96.5%, R 93.3% (Massachusetts); [no released weights shown](https://github.com/z00bean/CrosswalkNet) | [F] |
| [Project Sidewalk validator](https://huggingface.co/projectsidewalk/sidewalk-validator-ai-crosswalk) | Crop classifier | Accuracy 0.9573 on own split | [P] |

The remaining Hugging Face hits are toys or empty cards, for example [filepile](https://huggingface.co/filepile/crosswalk-traffic-light) ("TRAINED WITH 30 IMAGES") and [Aleton/Zebra_crossing](https://huggingface.co/Aleton/Zebra_crossing) [P]. The largest crosswalk image set on the hub by file count, [SightLinks](https://huggingface.co/datasets/SightLinks/YOLO-OBB-Zebra-Crossings-Dataset) (8,550 images) [P], appears to be aerial and has no licence or card.

### Overhead sources and inventories

Overhead work is the most developed thread but offers no shared benchmark. [SkyScapes](https://arxiv.org/pdf/2007.06102) [F] is the only one found with a held-out test and a crosswalk class, and it is 16 images of Munich. [Tile2Net](https://www.evl.uic.edu/documents/ceus_mappingthewalk_miranda.pdf) [F] reports crosswalk IoU 75.42% with P and R 0.86, and [Berriel 2017](https://arxiv.org/pdf/1706.09302) [F] reports 97.11% image-level accuracy. None should serve as street-level ground truth without a street-level audit, because canopy and shadow cause undercounts.

The official inventories are the richest find. Counts below came from live queries on 2026-11-08 [P].

| City | Layer | Geometry, count | Pattern recorded | Currency | Note |
|---|---|---|---|---|---|
| **Los Angeles** | [LADOT](https://maps.lacity.org/lahub/rest/services/LADOT/MapServer/4) | Point, 22,507 | Continental 12,338, Standard 6,906, Ladder 2,988, others | Dates to 2026-10-27 | Best schema found; apparently one point per leg |
| **Austin** | [Short Line](https://services.arcgis.com/0L95CJ0VTaxqcmED/arcgis/rest/services/Markings_Short_Line_Read_Only/FeatureServer/0) | Line, 7,328 | Continental 6,831, Transverse 465 | Edited 2026-11-08 | 7,289 active |
| **Paris** | [Passages piétons](https://opendata.paris.fr/explore/dataset/plan-de-voirie-passages-pietons/) | Polygon, 22,236 | None | Processed 2026-10-10 | ODbL; "aires mixtes" scope needs reading |
| **Bend** | [Crosswalks](https://services5.arcgis.com/JisFYcK2mIVg9ueP/arcgis/rest/services/Crosswalks/FeatureServer/7) | Line, 6,655 | Marked 1,117, **unmarked 5,504** | Edited 2026-11-08 | Explicit negatives; city-managed only |
| Calgary | [Crosswalks](https://data.calgary.ca/resource/hxgg-rpad.json) | Point, 4,216 (about 2,925 real) | Regular, ladder, others | Updated 2026-11-06 | Must drop stop lines |
| Portland | [PBOT](https://www.portlandmaps.com/od/rest/services/COP_OpenData_Transportation/MapServer/201) | Line, 7,094 | Field exists, unread | Unread | Close to Seattle |
| Seattle (dev only) | [Marked Crosswalks](https://services.arcgis.com/ZOyb2t4B0UYuYNYH/arcgis/rest/services/Marked_Crosswalks_CDL/FeatureServer/0) | Point, 6,692 | `LADER` 5,895, `PARSL` 724, others | Edited 2026-11-06 | Filter 348 planned |
| DC | [DDOT](https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Transportation_Traffic_Calming_WebMercator/MapServer/96) | Point, 16,117 | Standard, "Longitudal", Diagonal | **Unknown** | Secondary only; intersection-only collection |
| Arlington VA | [Pavement Markings](https://arlgis.arlingtonva.us/arcgis/rest/services/Open_Data/od_Pavement_Markings_Points/FeatureServer/0) | Point, 3,690 | High Visibility 2,703, Standard 987 | Installed to 2026-10-28 | Reserve city |

No citywide layer was found in the catalogs searched for NYC, San Francisco, Chicago, Boston, Toronto or Montreal, and no national US marked-crosswalk inventory was found. That is a result about those catalogs, and several other portals could not be read. OSM's coverage is large but noisy: [`crossing:markings` is on 7,378,658 nodes](https://taginfo.openstreetmap.org/api/4/key/stats?key=crossing:markings) [P], and against 1,272 hand-labelled crosswalks in Leon County, [OSM scored recall 77.8%, precision 52.2%](https://arxiv.org/pdf/2406.09731) [F], which may partly reflect unmarked crossings. No published precision or recall for `crossing:markings` values, and no city-by-city OSM-vs-inventory comparison, was found.

### Conflicts between notes, and how they resolve

The MapTRv2 and PASS/WildPASS tables below were re-read for this report through a summarizing fetch [F].

| Conflict | Resolution |
|---|---|
| MapTRv2 `AP_ped` 59.8 vs 68.1 | **Same model, different training length, not different thresholds.** [Table 1](https://arxiv.org/html/2308.05736), nuScenes val, R50 camera-only: MapTR 46.3 (24 ep), 56.2 (110 ep); MapTRv2 59.8 (24 ep), 68.1 (110 ep). Thresholds are 0.5/1.0/1.5 for both. MapTRv2's text gives no unit; [MapTR's](https://ar5iv.labs.arxiv.org/html/2208.14437) is metres. |
| WildPASS: IoU or per-class accuracy | **Unresolved in the paper.** [Table 2](https://ar5iv.labs.arxiv.org/html/2103.05687) is captioned "Accuracy analysis" and Section 4.4 says per-class accuracy, while Section 4.2 states mIoU as the metric. Do not label these values IoU without opening the PDF. |
| WildPASS ranges 4.2-28.9 vs 6.4 | Not a conflict. 6.4 is the ERF-PSPNet row inside the 4.2-28.9 baseline range. Vistas training per row is **inferred from the text**, so the contrast with 71.1 and 28.0 is not established. |
| "PASS 71.1" | That row is a **method** on WildPASS, not a score on the PASS dataset. |
| "Crosswalk IoU up to 62-71%" | One note merged two datasets. [PASS Table II](https://ar5iv.labs.arxiv.org/html/1909.07721) tops out at 62.1 (SwaftNet), with ERF-PSPNet 52.9 and ERF-APSPNet 57.2 being two models. 62.3 is SwaftNet on **Vistas val**. The 71.1 is WildPASS. Quote them separately. |
| Only panoramic set with a crosswalk class | PASS and WildPASS are the only **human-labelled** ones found. Metropolis has the class in machine-generated masks. |
| Li 2023: closest prior work vs dropped | Different roles; see first section. |
| Berriel 2017: 97.11% vs 96.78% | [The paper](https://arxiv.org/pdf/1706.09302) [F] gives 97.11%; the [lab page](https://www.lcad.inf.ufes.br/wiki/index.php/Deep_Learning_Based_Large-Scale_Automatic_Satellite_Crosswalk_Classification) [S] lists 96.78%. Cite the paper. |
| Vistas class counts | 65 in the [checkpoint](https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-semantic/raw/main/config.json) [P] vs 66 in the [blog](https://blog.mapillary.com/update/2021/01/18/vistas-2-dataset.html). v2.0 is 124 there and [152 on the current listing](https://www.mapillary.com/datasets). Unreconciled. |

## XwalkNet is the best name; CrossNet is crowded

This section is judgement plus a name-collision scan run on 2026-11-08. "No collision" means none in the searches listed: arXiv and OpenAlex titles, GitHub repo names, Hugging Face repo IDs and PyPI, plus web search where stated. Google Scholar could not be read.

**CrossNet should stay an internal codename at most.** The string is free on [PyPI](https://pypi.org/project/crossnet/), but it has [5 arXiv titles](https://export.arxiv.org/api/query?search_query=ti:%22CrossNet%22), [46 OpenAlex works](https://api.openalex.org/works?filter=title.search:%22CrossNet%22) and [79 GitHub repos](https://github.com/search?q=CrossNet+in%3Aname&type=repositories). The worst collisions share this model's paradigm: ["CrossNet: Detecting Objects as Crosses"](https://researchr.org/publication/LengLWHG22) predicts a center map, and a [second CenterNet-based CrossNet](https://aminer.org/pub/61c9e9155244ab9dcbb81774/crossnet-computing-friendly-lightweight-anchor-free-detector) exists. There is also [CROSS-Net](https://arxiv.org/abs/2310.18215) in transportation, the [`CrossNet` layer in DeepCTR](https://deepctr-doc.readthedocs.io/en/latest/deepctr.layers.interaction.html), and a [volleyball brand](https://en.wikipedia.org/wiki/Crossnet). "Cross" names no object, and it will read as cross-city in a paper about cross-city generalization. **CrosswalkNet is unavailable in practice**: it is a [2025 aerial crosswalk detector](https://arxiv.org/abs/2506.07885).

| Rank | Name | Collision result | Fit | Weakness |
|---|---|---|---|---|
| 1 | **XwalkNet** | None (API scan, two web searches). Root neighbours only: [Crosswalk Project](https://en.wikipedia.org/wiki/Crosswalk_Project), [Xwalk protein tool](https://docs.rc.ufl.edu/software/apps/xwalk) | True RampNet sibling; all marking patterns | "Ex-walk" misreading; US word |
| 2 | **PedXingNet** | None (API scan, web search). Root neighbour [XING](https://en.wikipedia.org/wiki/XING) | Best if unmarked is added, since [OSM treats unmarked as a category of crossing](https://wiki.openstreetmap.org/wiki/Key:crossing:markings) | Long; "zing" misreading |
| 3 | XMarkNet | None. Root neighbour [XMark](https://projects.cwi.nl/xmark/index.html) | "X marks the spot" suits points | Excludes unmarked; needs explaining |
| 4 | CrosswalkPoint | None; generic phrase in [a patent](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12272088) | Names the paradigm | Breaks the -Net pattern |
| 5 | CrossStripeNet / CrossLineNet | None | Clear | Each misses a [common pattern](https://en.wikipedia.org/wiki/Crosswalks_in_North_America) |

Avoid "Zebra": two parallel lines are the [most frequent US style](https://en.wikipedia.org/wiki/Crosswalks_in_North_America), and [OSM](https://wiki.openstreetmap.org/wiki/Key:crossing:markings) separates zebra, lines and ladder. RampNet-Crosswalk had no collision, but it ranks last because it would make RampNet a de facto umbrella and pre-empt the deferred naming decision. If you adopt XwalkNet, state the pronunciation on first use ("XwalkNet, pronounced crosswalk-net"), keep the single capital, and follow the [existing org pattern](https://huggingface.co/api/models?author=projectsidewalk&limit=50) with `xwalknet-model`. No reader testing was done, so poll the lab, then re-run the exact-name checks and read Google Scholar by hand before committing.

## Blocked sources limit which numbers can be cited

Some publisher sites and hubs blocked automated access. PMC, MDPI, preprints.org, ResearchGate, CVF supplements, air.unimi and the Mapillary help center also refused, Kaggle returned only page titles, and Google Scholar stopped at a verification page. One literature search was also refused and later ran normally when rephrased.

The affected claims are every Roboflow and Kaggle listing; the metrics for Li 2023 (not obtained), Adams 2022, IJTST 2024, Mind Your Crossings, CDNet and X-CDNet; and any Vistas class definition, since the Vistas paper was unreachable. Someone with library access should open Li 2023, Adams 2022, IJTST 2024 and Mind Your Crossings before any of their numbers are cited. Portals that could not be read include catalog.data.gov, data.gv.at, hri.fi, data.overheid.nl, Barcelona and data.gov.tw, so absent inventories there mean nothing. Two internal Seattle baselines mentioned in the notes have no citable URL there, so their figures are omitted; take them from the project's Seattle evaluation when writing the comparison table.

## Conclusion

The gap the project can credibly fill is the benchmark more than the detector. A multi-city, exhaustively audited, point-scored panoramic test would be useful whatever the model's novelty is judged to be, and a city-level OSM-vs-inventory comparison looks like a cheap by-product that no one in these searches had published.

Two small actions could change this report's conclusions, so do them first: read Li 2023's full text, which decides the novelty wording, and open the WildPASS archive, which decides whether the only independent human-labelled panoramic test has enough crossings to estimate recall.
