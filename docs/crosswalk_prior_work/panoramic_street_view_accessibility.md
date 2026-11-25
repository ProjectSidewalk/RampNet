# Crosswalk detection in street-view and panoramic imagery, and accessibility datasets

Notes written 2026-11-08 by Claude (Opus 5.5, claude-opus-5-5) for the Project Sidewalk crosswalk detector literature check.

**How to read the provenance tags.** Numbers rarely compare across these papers, so each entry says what was actually read.

- `[PDF read]`: I searched the paper's own extracted text directly. This applies only to Berriel et al. 2018.
- `[page read]`: the paper or page was opened through a fetch tool that returns a small-model summary. Table values were reported to me, not seen by me, so spot-check any number before it goes in a paper.
- `[secondary]`: a third-party summary of the paper was opened, not the paper.
- `[snippet only]`: the figure came from a search-result excerpt. The paper was **not opened**, usually because PMC, MDPI, Sage, ScienceDirect, ResearchGate or IEEE refused automated access. This falls short of the assignment's "open each paper" requirement and is marked so nobody mistakes it for verified.

## Who has detected or audited crosswalks from GSV, Baidu, Tencent or Mapillary imagery?

### Takeaway

Street-level crosswalk work exists and goes back at least a decade, but what I found is image-level classification, satellite-first pipelines, or forward-facing vehicle or phone imagery. I found no published crosswalk detector that takes a full equirectangular panorama and returns instances or points, other than Project Sidewalk's own in-progress work. The assumption that there is no prevailing benchmark held up within this search: every study below uses its own data, task and metric.

### Cited Findings

Two corrections to the brief, because the named works are split differently than the brief implies:

- Berriel's "Street View" and "satellite" papers are two separate papers: a 2017 satellite paper and a 2018 street-level paper. — [Berriel 2017](https://arxiv.org/abs/1706.09302); [Berriel 2018](https://arxiv.org/abs/1805.11970)
- ZebraRecognizer (ICPR 2014) is a separate user-side recognition module for visually impaired users. The satellite plus Street View pipeline is Zebra Crossing Spotter (ASSETS 2015) and its journal extension Mind Your Crossings (TACCESS 2017). I could not establish how, or whether, the two connect. — [ASSETS 2015 abstract](https://assets15.sigaccess.org/acceptedPapers/24.html); [PMC record](https://pmc.ncbi.nlm.nih.gov/articles/PMC5531764)

**Comparison table.** All metrics are as reported and are not comparable across rows.

| Work | Task and unit | Imagery | Geography and N | Method | Reported metric, conditions | Released | Provenance |
|---|---|---|---|---|---|---|---|
| [Berriel et al. 2018, Computers & Graphics](https://arxiv.org/abs/1805.11970) | Binary "crosswalk in image?"; **image-level, no localization** | GSV Image API perspective crops: FOV 90°, pitch 0°, heading computed along the route | 657,691 images (33,959 positive, 623,732 negative), 20 Brazilian states plus the federal district, 24 regions. Cross-database: IARA vehicle (manually labeled) and GOPRO (11,070 images) | VGG; labels **automatic from OSM `highway=crossing`** | 94.12% overall accuracy fully automatic; 96.30% with part of the database manually annotated | Scripts promised; models via [repo](https://github.com/rodrigoberriel/streetview-crosswalk-classification) (repo not opened); other two datasets' sequences (>23,500 images) | [PDF read] |
| [Berriel et al. 2017, IEEE GRSL](https://arxiv.org/abs/1706.09302) | Binary, image-level | Satellite (Google Static Maps) | >240,000 images, 3 continents, 9 countries, >20 cities | CNN; labels from OSM | 97.11% in the arXiv/DeepAI abstract; [the LCAD lab page](https://www.lcad.inf.ufes.br/wiki/index.php/Deep_Learning_Based_Large-Scale_Automatic_Satellite_Crosswalk_Classification) lists 96.78%. Unresolved | Repo exists, not opened | [snippet only] |
| [Li et al. 2023, Environment and Planning B](https://journals.sagepub.com/doi/full/10.1177/23998083221112157) | Counts parallel-line and high-visibility crosswalks per image; aggregated to intersection-year | GSV | Trained on 4,925 hand-labeled images; run on 922,144 historical images; 38,350 intersections within 250 m of 4,417 US TOD stations, 2007–2020; missing years imputed | CNNs | **Not obtained** (publisher 403; not in abstracts found) | [CSV of intersection-year counts](https://zenodo.org/record/6803956), `inf_crosswalk.csv`, 142.1 MB, CC BY 4.0. No images or weights listed | Zenodo [page read]; paper [snippet only] |
| [Adams et al. 2022, IJERPH](https://www.mdpi.com/1660-4601/19/8/4548) | Image classifiers for 8 MAPS-Mini features including zebra and line crosswalks | GSV; headings and FOV not confirmed | 5 cities; applied to 512 WalkIT Arizona participants' neighborhoods | EfficientNet-B5 | Overall accuracy: zebra 94.62%, line 80.20%. The abstract says all metrics exceed 84%, which the per-feature table contradicts | Not confirmed | [snippet only]; PMC, MDPI and preprints.org all refused |
| [Ahmetovic et al. 2015/2017, Zebra Crossing Spotter / Mind Your Crossings](https://pmc.ncbi.nlm.nih.gov/articles/PMC5531764) | Zebra localization in the world; satellite candidates validated on GSV | Satellite, then registered GSV panoramas | Four regions (2 San Francisco, NYC, Milan), 7.5 km². ASSETS version: 141 zebra and 152 transverse crossings in SF | Line-segment detector and geometric filters; GSV validation | Precision 0.902–0.971, recall 0.772–0.950 across regions. Crowd validation in Milan raised recall 0.772 to as high as 0.97, precision 0.948 to 0.989 | Not confirmed | [secondary, a11y-paradise summary](https://a11y-paradise.onrender.com/reviews/69a488f012b3bc5f14238fcf); [abstract page read](https://assets15.sigaccess.org/acceptedPapers/24.html) |
| [Ahmetovic et al. 2014, ZebraRecognizer](https://air.unimi.it/handle/2434/270734) | Phone-side zebra recognition | Smartphone | Not obtained | Not obtained | "No false positives", 93% of zebras identified | Not confirmed | [snippet only]; proceedings PDF 404 |
| [IJTST 2024, occluded crosswalks](https://www.sciencedirect.com/science/article/pii/S2046043024000364) | Crosswalk classification, three-way comparison | Aerial, street-view, combined | Heavily occluded area; N not obtained | VGG16 (per snippet) | Street-view 82.58% vs aerial 55.59%; combined best | Not confirmed | [snippet only] |
| [USM dissertation (no. 2034)](https://aquila.usm.edu/dissertations/2034/) | Crosswalk detection and measurement | Bing aerial and street view; MDOT satellite | N not in abstract | CNN, segmentation, dual-perspective model | Aerial 99.23% on MDOT imagery. Dual-perspective vs aerial-only on occluded data: accuracy +49%, recall +383%, precision +15% (**relative gains only**) | Full-text PDF linked | Abstract [page read] |
| [Verma & Ukkusuri 2023, TRR](https://trid.trb.org/View/2296683) | Crosswalk detection, then assignment to intersection legs | **Satellite**, not street view | 4 North American cities | YOLOv5 | 71% (DC), 89% (LA) are **leg-assignment** accuracy, not detection | Not confirmed | [snippet only] |
| [Mapillary AI detections](https://blog.mapillary.com/update/2017/09/22/editing-map-data-with-ai-detections.html) | Per-image detection, class shown as "Crosswalk--zebra" | Crowdsourced street-level | Global product | Proprietary | **No published accuracy found** | Via API | [snippet only]; help center refused |
| [Roadside facility detection on Mapillary](https://www.researchgate.net/publication/366416809_Development_of_a_Large-Scale_Roadside_Facility_Detection_Model_Based_on_the_Mapillary_Dataset) | Multi-class boxes including crosswalks | Mapillary | Not obtained | YOLOv7 | P 87.57%, R 72.60% **across all classes**, not crosswalk-specific | Not confirmed | [snippet only] |

**Project Sidewalk's own line of work**

- Tohme (Hara et al., UIST 2014) finds curb ramps in GSV. Its CV subsystem alone finds 67% of ramps. With crowd workers it reaches F-measure 84% vs 86% for manual labeling, at 13% less human time. A later lab paper cites Tohme's CV as 26% precision and 67% recall. Curb ramps only. — [paper](https://makeabilitylab.cs.washington.edu/media/publications/Hara_TohmeDetectingCurbRampsInGoogleStreetViewUsingCrowdsourcingComputerVisionAndMachineLearning_UIST2014.pdf) [snippet only]
- Project Sidewalk (Saha et al., CHI 2019): 18-month DC deployment, over 255,000 labels on curb ramps, missing ramps, obstacles and surface problems. — [paper](https://makeabilitylab.cs.washington.edu/media/publications/Saha_ProjectSidewalkAWebBasedCrowdsourcingToolForCollectingSidewalkAccessibilityDataAtScale_CHI2019.pdf); [report](https://digital.lib.washington.edu/researchworks/bitstreams/6253f97b-d36a-49f0-8130-2ae65aaba8e1/download) [snippet only]
- Weld et al. (ASSETS 2019) classifies curb ramp, missing curb ramp, surface problem and obstruction in GSV panoramas. Crosswalk is not among its classes. — [repo](https://github.com/ProjectSidewalk/sidewalk-cv-assets19) [snippet only]
- RampNet (O'Meara et al., ICCV 2025 CV4A11y workshop): full detail under the point/keypoint question. — [arXiv](https://arxiv.org/html/2508.09415v1) [page read]

**VLM work, 2024–2026**

- OmniAcc reports 97.5% zero-shot crosswalk accuracy with GPT-4o, but on **1 m NAIP aerial imagery**, not street level, despite how search excerpts present it. 200 images per configuration, balanced, Oxford, Ohio. Best config (blurred) P 96.11%, R 99.00%, F1 97.53%; the plain config scored P 68.03%, R 100%, F1 80.97%. — [arXiv](https://arxiv.org/html/2509.07220v1) [page read]
- WalkCLIP has GPT-4o caption street-view composites, including listing visible crosswalks, as pseudo-labels for walkability prediction. It is not a crosswalk audit. — [arXiv](https://arxiv.org/pdf/2511.21947) [snippet only]
- Jang and Kim (2024) report ChatGPT and Gemini street-view audits "strongly agree" with DeepLabv3+. The abstract gives no metric and does not mention crosswalks. — [VTechWorks record](https://vtechworks.lib.vt.edu/items/bb66089d-2443-4fa5-884b-37a562e2452d) [page read]
- A Singapore LLM walkability study reports a prompt variant misreading crosswalks as fixed furniture. — [paper](https://ual.sg/publication/2025-sdsc-llm-walkability/2025-sdsc-llm-walkability.pdf) [snippet only]
- GPT-4V has been applied to crossing safety for blind and low-vision pedestrians on robot-captured images (risk assessment, not detection). — [arXiv](https://arxiv.org/abs/2402.06794) [snippet only]
- A Zurich master's thesis trains YOLO11s on curb ramps, marked crossings, tram tracks and traffic lights. — [UZH](https://lean-gate.geo.uzh.ch/typo3conf/ext/qfq/Classes/Api/download.php/mastersThesis/1209) [snippet only, no numbers obtained]

**Baidu and Tencent**

- Searches returned Baidu Street View work on green view, enclosure, neighborhood quality and road cracks, but nothing on crosswalk audits. — [example](https://pmc.ncbi.nlm.nih.gov/articles/PMC9779280/) [snippet only]

### Inferences

- The closest large-scale prior is Berriel 2018 for method and Li 2023 for application. Neither localizes crosswalks in a panorama, as far as I could confirm for Li.
- Our P 0.980, R ~0.85, F1 0.908 (instance-level, Seattle audit) cannot be set beside 94.12%, 94.62% or 82.58%, which are image-level accuracies, in Berriel's case on a dataset that is about 95% negative. Accuracy there is close to uninformative without class-balanced numbers.
- Precision and recall from Mind Your Crossings are the nearest in unit (world-level instances), but its areas are small and it depends on satellite visibility.

### Gaps

- Li et al. 2023 performance metrics, view geometry and output unit. The paper refused automated access. The full text is the most important unread item here.
- Adams 2022 and IJTST 2024 precision and recall; only accuracies seen, and only in excerpts.
- Mind Your Crossings primary numbers were read only through a third-party summary. Berriel's related work does independently note the Street View step was "evaluated in only 406 portions of Street View images". — [Berriel 2018](https://arxiv.org/abs/1805.11970) [PDF read]
- Chinese-language sources were not searched. "No Baidu or Tencent crosswalk audit found" reflects English-language queries only and is weak evidence of absence.
- I found no VLM study reporting crosswalk precision and recall on street-level imagery, but only a few queries were run.

## Which 360/panoramic road-scene datasets include a crosswalk class, and what crosswalk IoU is reported?

### Takeaway

The PASS/WildPASS family is the only panoramic benchmark I confirmed with a crosswalk class. It is semantic segmentation on a 360°×70–75° band at low resolution, so no instances or points. Everything else checked lacks a crosswalk class or could not be confirmed.

### Cited Findings

| Dataset | Panoramic? | Crosswalk class? | Size and format | Crosswalk result | Provenance |
|---|---|---|---|---|---|
| [PASS / DS-PASS](https://ar5iv.labs.arxiv.org/html/1909.07721) | Annular, 360° | **Yes** (car, road, crosswalk, curb; DS-PASS adds sidewalk and person) | DS-PASS annotates 400 images, 2048×692 | IoU on PASS test: SwaftNet 62.1%, ERF-APSPNet 57.2%, ERFNet 56.4%, SwiftNet 55.9%, BiSeNet 18.1%. SwaftNet on Vistas val: 62.3% | [page read]; table was garbled in conversion and matched by column |
| [WildPASS](https://ar5iv.labs.arxiv.org/html/2103.05687) | 360°×70°, 2048×400 | **Yes** (8 classes: car, road, sidewalk, crosswalk, curb, person, truck, bus) | 2,500 panos, 65 cities, 6 continents; 500 annotated (25 cities), 2,000 unlabeled (40 cities). Images reportedly from GSV ([snippet](https://arxiv.org/pdf/2103.05687)). [Repo](https://github.com/elnino9ykl/WildPASS) not opened | Baselines described as Vistas-trained: 4.2%–28.9% (DANet best). PASS (ERF-PSPNet) 71.1%; OOSS 28.0%; omni-supervised ERF-PSPNet 70.5%; ECANet with attention 67.8%, with fusion 67.9% | [page read]; training condition per row is partly inferred |
| [DensePASS](https://ar5iv.labs.arxiv.org/html/2108.06383) | 360°, 2048×400 | No (19 Cityscapes classes) | 100 labeled, 2,000 unlabeled | None. Best mIoU 39.76% (42.47% with WildDash source) | [page read] |
| [SynPASS / Trans4PASS+](https://arxiv.org/html/2207.11860v5) | Synthetic, 1024×2048 | Not seen. Has `roadline`; no crosswalk in the portion read (first 100k of 162k characters) | 9,080 images, 22 classes | None | [page read, partial] |
| [Waymo PVPS](https://ar5iv.labs.arxiv.org/html/2206.07704) | Five cameras stitched, 220° | No dedicated class. [Proto](https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/src/waymo_open_dataset/protos/camera_segmentation.proto) has `TYPE_LANE_MARKER` and `TYPE_ROAD_MARKER` ("all markings on the road other than lane markers") | 28 classes, 2,860 sequences, ~100k images | None per class found | [page read] |
| [CVRG-Pano](https://github.com/semihorhan/semseg-outdoor-pano) | Outdoor panoramas | No (20 classes; flat = ground, road, sidewalk, parking) | 600 images | None | [page read] |
| [OmniCity](https://ar5iv.labs.arxiv.org/html/2208.00928) | NYC street panoramas | No (7 land-use classes; building instances) | >100K annotated images overall | None | [page read] |

Pinhole context, since these are the usual sources of a crosswalk class:

- Mapillary Vistas has "Lane Marking - Crosswalk", and v2.0 also separates "Lane Marking (only) - Crosswalk". SegFormer-B0 scores **63.80%** IoU on Vistas val at 1024×2048. — [Flying Guide Dog](https://ar5iv.labs.arxiv.org/html/2108.07007) [page read]; class names from [mmseg listing](https://huggingface.co/spaces/KyanChen/TTP/blob/refs%2Fpr%2F1/mmseg/datasets/mapillary.py) [snippet only]
- A panoptic quality of 46.1 for a zebra-crosswalk marking class on Vistas is reported in Learning to Fuse Things and Stuff. — [arXiv](https://arxiv.org/pdf/1812.01192) [snippet only]
- Surround-camera HD-map work is a nearby formulation. On nuScenes (6 cameras, 360° horizontal), MapTR-tiny camera-only reaches AP_ped 46.3 at 24 epochs and 56.2 at 110 epochs; VectorMapNet 36.1; HDMapNet 14.4. AP averages Chamfer thresholds of 0.5, 1.0 and 1.5 m, and crossings are closed polygons in BEV. — [MapTR](https://ar5iv.labs.arxiv.org/html/2208.14437) [page read]

The WildPASS authors call road and crosswalk the most context-critical classes. — [snippet](https://arxiv.org/pdf/2103.05687)

### Inferences

- A crosswalk IoU in the 60–70% range is what is reported for panoramic segmentation, on a narrow band at 400–692 px tall. It measures pixels, not whether each crossing was found, so it is not a substitute for instance recall.
- The 70–75° vertical band of the PASS family presumably excludes the steep below-horizon region where a close crossing appears in a full equirectangular pano. That is my reading of the geometry and is not stated by those papers.
- HD-map work is the most likely source of a reviewer's "this is already solved" objection. The answer is that it needs calibrated surround rigs and evaluates polygons in a local BEV frame, on driving datasets, rather than on archival single panoramas.

### Gaps

- SHIFT, KITTI-360 and 360+x class lists were not obtained. 360+x has a `classes.json` that I did not read. — [360+x](https://x360dataset.github.io)
- Matterport, Structured3D and PanoContext are, to my understanding, indoor, but I did not check them, so I make no sourced claim.
- Whether PanorAMS's 22 categories include a crossing or road-marking category was not established (see OSM question).
- The WildPASS table does not label training conditions per row, so the 71.1% vs 28.0% contrast needs reading in the PDF.

## What accessibility and sidewalk datasets are relevant, and do they label crosswalks?

### Takeaway

Project Sidewalk's crowd labels look like the largest crosswalk point-label source tied to panoramas that I found. Assistive datasets with crosswalks exist (SANPO, PTL) but are egocentric, not panoramic, and report little or nothing for crosswalks as a class.

### Cited Findings

- **Project Sidewalk API:** offers `rawLabels`, `labelClusters`, `validations`, and label types, with a `pano_source` field. Crosswalks and curb ramps carry Good/Okay/Bad ratings. The `/v3` API is in preview. The page did not render the label-type list or field schemas. — [API page](https://sidewalk-sea.cs.washington.edu/api) [page read]
- **Crowd Crosswalk census:** 68,014 labels across 46 cities. Crowd labels are used for positives only, never as recall ground truth, because the labeling is not exhaustive. — [RampNet#274](https://github.com/ProjectSidewalk/RampNet/issues/274) [page read]; RampNet notes PS does not require comprehensive labeling — [RampNet](https://arxiv.org/html/2508.09415v1)
- **SANPO** (WACV 2025): crosswalk is a "thing" class in a 31-class taxonomy. 701 real sessions, 617,408 stereo frames, 112K with panoptic masks, 1,961 synthetic sessions; San Francisco, Mountain View, Boulder, NYC; chest and head stereo cameras, **no 360**. **No per-class crosswalk result**; crosswalk is omitted from the per-label table. CC BY 4.0, though the paper's release wording is ambiguous. — [arXiv](https://arxiv.org/html/2309.12172) [page read]
- **ImVisible / PTL:** 5,059 images (3,456 train, 864 val, 739 test), Shanghai intersections, iPhone. Labels traffic-light state and the zebra midline. Dataset published. — [LYTNet](https://ar5iv.labs.arxiv.org/html/1907.09706) [page read]; [repo](https://github.com/samuelyu2002/ImVisible) [snippet only]
- **"PTL-Crosswalk":** I found no dataset under exactly that name. The nearest is "Dynamic Crosswalk Scene Understanding for the Visually Impaired": 7,447 PTL images, 1,006 key-object images, 3,336 crosswalk images. — [HAL](https://hal-college-de-france.archives-ouvertes.fr/UNIV-NANTES/hal-03330267v1) [snippet only]
- **SideGuide:** ~350K images with boxes, 100K with polygon masks, 180K stereo pairs. Crosswalk class not confirmed. — [IEEE](https://ieeexplore.ieee.org/document/9340734/) [snippet only]
- **SideSeeing:** 46 videos, 12 km, four cities. A sidewalk assessment taxonomy, not a segmentation benchmark. — [arXiv](https://arxiv.org/pdf/2407.06464) [snippet only]
- **CitySurfaces:** sidewalk material segmentation, 90.5% mIoU, trained on NYC and Boston, tested on six others. Crosswalk not mentioned in the abstract. — [arXiv](https://arxiv.org/abs/2201.02260) [page read]
- **OpenSidewalks:** a network schema, not imagery. Crossings exist only on the street surface; `crossing=marked` converts to `crossing:markings=yes`. About 2.25M OSM nodes carried `crossing=marked` as of January 2026, and the wiki notes inconsistent use. — [schema](https://gh.r-mj.com/OpenSidewalks/OpenSidewalks-Schema); [mapping rules](https://tcat.cs.washington.edu/wp-content/uploads/OpenSidewalks_mapping_rulesForMapping.pdf); [OSM wiki](https://wiki.openstreetmap.org/wiki/Tag:crossing=marked) [snippet only]
- **Cross-dataset harmonization:** a 2026 preprint maps SideGuide, SANPO and others to a 9-class taxonomy (SENSATION-DS). — [arXiv](https://arxiv.org/html/2607.21137v1) [snippet only]

### Inferences

- Project Sidewalk's combination of panorama-registered points, validation votes and multi-city coverage appears unusual among the datasets found, but it is non-exhaustive. Any benchmark built from it needs exhaustive re-annotation, as already planned on the project side.
- SANPO is the most usable external source of crosswalk instance masks, but in a different viewpoint and with no reported baseline.

### Gaps

- "TILES" and "Sidewalk-AI" could not be resolved to specific datasets from the queries run. I make no claim about them.
- Tactile-paving and curb-ramp-specific datasets were not surveyed; no query targeted them.
- SideGuide's class list and Taskar Center TDEI data contents were not confirmed.

## Who uses OSM crossing nodes or municipal inventories as labels for street-level imagery?

### Takeaway

Using OSM crossing nodes to label street-level imagery has direct precedent: Berriel 2018 did it for GSV in 2017–18. Municipal or open-map labels on panoramas also have precedent (PanorAMS, RampNet). The piece I did not find elsewhere is map-derived labels producing **per-crossing points in a full panorama**.

### Cited Findings

- Berriel 2018 queries `highway=crossing` from OSM via Overpass, retrieves the nearest GSV pano within 50 m, computes a heading, and fetches 90° crops as automatic positives and negatives. The system has no human annotation in its fully automatic arm. — [arXiv](https://arxiv.org/abs/1805.11970) [PDF read]
- PanorAMS auto-generates over 14 million noisy boxes on 771,299 Amsterdam panoramas "solely from open data sources", across 22 categories, with 147,075 crowdsourced clean boxes on 7,348 images. — [arXiv](https://arxiv.org/abs/2208.14295) [page read, abstract]
- RampNet projects government curb-ramp locations into panoramas: 849,895 labels on 214,376 panos, 94.0% precision and 92.5% recall against manual labels. — [arXiv](https://arxiv.org/html/2508.09415v1) [page read]
- Li 2023 used OSM road intersections to locate sites, and reports only 14.22% of OSM crossings have attributes indicating presence and category, concentrated in Europe. — [paper](https://journals.sagepub.com/doi/full/10.1177/23998083221112157) [snippet only]
- A 100-city US satellite study notes some OSM crossings are only single nodes and matches them to the nearest road edge. — [arXiv](https://arxiv.org/pdf/2507.20497) [snippet only]
- An OSM contributor built orthophoto training data from mapped crossings and recast labeling as "is a crossing at the image center". Aerial, not street-level. — [blog](https://monicz.dev/osm-yolo-crossings) [snippet only]
- Aerial segmentation from online maps has a long history; noisy map labels need much larger training sets to match hand labels. — [Kaiser et al.](https://arxiv.org/pdf/1707.06879) [snippet only]
- Project Sidewalk's internal Phase 0 measured crowd labels a median 1.17 m along and 1.00 m across from OSM nodes in 17 dense cities (16,762 labels). It also found `unmarked` nodes in Seattle were painted 20 of 40 times, so that tag cannot serve as negatives. — [RampNet#274](https://github.com/ProjectSidewalk/RampNet/issues/274) [page read]

### Inferences

- Any claim that map-derived weak supervision for street-level crosswalks is new would be wrong because of Berriel 2018, and it should cite that paper explicitly.
- The defensible difference is label geometry: Berriel needs only "is there a crossing in this view", while a point-output detector needs sub-pano placement, which makes OSM node precision the binding constraint.

### Gaps

- I found no published paper training a street-level crosswalk detector on `crossing:markings` tags specifically.
- Whether PanorAMS includes road markings was not established from the abstract.

## Does anyone predict crosswalk centres or use heatmap localisation?

### Takeaway

I found no paper predicting crosswalk centres via heatmaps. Point or endpoint regressions exist for assistive phone views, and heatmap peak detection on panoramas exists for curb ramps in Project Sidewalk's own RampNet.

### Cited Findings

- RampNet takes 4096×2048 equirectangular input and outputs a 1024×512 heatmap, using ConvNeXt V2. Peaks are kept above 0.55. A true positive is a ground-truth point within 88 px on the 4096×2048 pano. AP 0.924 on 1,000 manually labeled panos (3,919 ramps), versus 0.380 for Weld et al. on the same ground truth. — [arXiv](https://arxiv.org/html/2508.09415v1) [page read]
- LYTNet regresses the two endpoints of the zebra midline as image coordinates, then applies a homography. Angle error 6.27° (abstract says 6.35°), startpoint error 0.0763, endpoint error 0.0510, on a single crossing per image. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1907.09706) [page read; one table value is inconsistent in the source]
- A joint regression formulation detects zebra crossings and estimates direction. — [Wu et al. 2021, IET](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/csy2.12006) [snippet only]
- MapTR represents crossings as closed polygons of 20 ordered points in BEV, a vector rather than point output. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2208.14437) [page read]
- Lane-marker keypoint heatmaps exist (FOLOLane), but for lanes. — [arXiv](https://arxiv.org/abs/2105.13680) [snippet only]
- Street-view object geotagging reports traffic-light recovery with precision and recall both above .92, through triangulated point outputs, not crosswalks. — [arXiv](https://arxiv.org/pdf/1708.08417) [snippet only]

### Inferences

- A centre point for an extended, anisotropic object is a design choice that a reviewer will probe. A crossing is several metres long, so the correct point is less well defined than for a curb ramp. A distance-radius criterion should be justified, and its sensitivity shown.
- The RampNet 88 px radius should not be assumed to carry over to crosswalks without that analysis.

### Gaps

- Two targeted queries found nothing on CenterNet-style crosswalk detection. This is weak negative evidence: an unindexed workshop paper or a non-English paper could exist.

## How would a reviewer assess the novelty claim, and which three prior works are closest?

### Takeaway

The narrow combination (full equirectangular input, marked crosswalks, per-instance point output, map-derived supervision) was not found in prior work, but each ingredient has precedent, and curb-ramp RampNet supplies the architecture. A reviewer would likely accept a claim about the combination and a useful benchmark, and reject any broader "first crosswalk detector from street view" framing.

### Cited Findings

**Three closest prior works**

1. **Berriel et al. 2018.** OSM crossing labels on GSV, with the largest confirmed N. Differences from ours: 90° perspective crops rather than a full equirectangular pano, binary image-level output with no localization, Brazil only. — [arXiv](https://arxiv.org/abs/1805.11970) [PDF read]
2. **Li et al. 2023.** GSV at national US scale with a public longitudinal product, and the closest audit application. Differences as far as confirmed: hand labels (4,925), counts per image aggregated to intersections, no reported point localization, and metrics I could not obtain. — [Sage](https://journals.sagepub.com/doi/full/10.1177/23998083221112157) [snippet only]; [Zenodo](https://zenodo.org/record/6803956) [page read]
3. **PASS/WildPASS (Yang et al.).** The closest input modality with an explicit crosswalk class: 360° street-level images, crosswalk IoU up to about 62–71%. Differences: cylindrical 70–75° band at low resolution, semantic masks with no instance or point evaluation. — [WildPASS](https://ar5iv.labs.arxiv.org/html/2103.05687); [DS-PASS](https://ar5iv.labs.arxiv.org/html/1909.07721) [page read]

Also expect these to be raised:

- **Ahmetovic et al.** evaluates at world-level instance precision and recall, which is our closest metric unit, though it depends on satellite detection first. — [summary](https://a11y-paradise.onrender.com/reviews/69a488f012b3bc5f14238fcf) [secondary]
- **RampNet** is our own precedent; a reviewer may call crosswalks a new label type on an existing method. — [arXiv](https://arxiv.org/html/2508.09415v1) [page read]
- **MapTR and successors** localize crossings from surround cameras, in a different problem setting. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2208.14437) [page read]
- **Mapillary's** production "Crosswalk--zebra" detector has no published accuracy that I found, but it undercuts any "no one does this at scale" statement. — [blog](https://blog.mapillary.com/update/2017/09/22/editing-map-data-with-ai-detections.html) [snippet only]

### Inferences

Likely reviewer objections, in rough order of seriousness:

- **Single-city audit.** Seattle P 0.980 / R ~0.85 / F1 0.908 says little about generalization. The strongest response is a multi-city, exhaustive benchmark, which would also fill the gap that no shared benchmark was found.
- **Method novelty.** The architecture is carried over from RampNet. The contribution has to be framed as the task, the data and the benchmark.
- **Incomparable baselines.** No external number is directly comparable. A credible paper should run at least one baseline on our own ground truth, for example a Vistas-trained crosswalk mask converted to instances, or a VLM on rendered views, rather than quoting published figures.
- **Point definition.** Needs a stated definition and a radius-sensitivity curve.

Suggested wording, pending a check of Li 2023: "We are not aware of a prior detector that localizes individual marked crosswalks as points in full equirectangular street-level panoramas". Avoid "first", and cite Berriel 2018 for map-derived supervision.

### Gaps

- The three items that would most change this assessment: Li 2023's full text, Mind Your Crossings' primary text, and a Chinese-language search.
- Google, Apple, Mapillary and others probably have internal detectors; published evidence for accuracy was not found.

### Search terms and limits

**Queries run** (web search, English only): crosswalk detection Google Street View deep learning audit; Berriel satellite crosswalk classification; Ahmetovic ZebraRecognizer StreetView satellite; panoramic semantic segmentation crosswalk class IoU; OpenStreetMap crossing nodes weak labels street-level; VLM / GPT-4V street view crosswalk audit; large language model virtual street audit crosswalk; GPT-4o Gemini crosswalk precision recall; RampNet; MAPS automated CV; Li marked crosswalks TOD; occluded crosswalks IJTST; Trans4PASS+ SynPASS classes; Mapillary Vistas crosswalk IoU; Baidu street view zebra crossing; keypoint heatmap crosswalk centre; CenterNet zebra crossing; Project Sidewalk crosswalk, Weld, Hara, Tohme; ImVisible PTL; SideGuide CitySurfaces SANPO SideSeeing; Mapillary detections taxonomy; CVRG-Pano; Waymo PVPS classes; MapTR pedestrian crossing; municipal inventory as labels; crosswalk 360 equirectangular YOLO; Verma Ukkusuri; street view pedestrian crossing inventory Mapillary; OpenSidewalks TDEI; OmniCity 360+x SHIFT; crossing:markings Mapillary classifier.

**Limits**

- Google Scholar, Semantic Scholar, ACM DL, IEEE Xplore and TRID were not searched directly. Discovery relied on a general web search tool, whose ranking and coverage I cannot characterize.
- PMC, MDPI, preprints.org, Sage, ScienceDirect, ResearchGate, air.unimi and Mapillary's help center refused automated access. Those papers remain unread. Someone with library access should open Li 2023, Adams 2022, IJTST 2024 and Mind Your Crossings before any of their numbers are cited.
- No claim here that something does not exist is stronger than "not found in these searches".
