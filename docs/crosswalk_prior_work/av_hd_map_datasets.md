# AV perception and HD-map datasets with crosswalk annotations, as candidate independent tests of a street-level crosswalk detector

Notes compiled 2026-11-08. "Confirmed" means read from a taxonomy file, schema, devkit, paper or official page during this session. Where a fact comes from a third party or a search snippet, the row says so. Pixel-per-degree figures and all test-design reasoning are my own and sit under Inferences.

## Is the "no prevailing crosswalk benchmark" assumption true?

### Takeaway
Half true. No crosswalk-only benchmark or leaderboard turned up in the AV literature, and nothing scores image-plane point detections on panoramas. But crosswalk is a widely annotated class, and per-class pedestrian-crossing AP is a routinely published number in vectorized HD-mapping (nuScenes, Argoverse 2, HRDX). The surprise find is that two small panoramic segmentation benchmarks (PASS, WildPASS) have a `Crosswalk` class with published scores; WildPASS is built from Google Street View panoramas.

### Cited Findings
- De facto per-class benchmark: `AP_ped` on nuScenes is reported in the VectorMapNet, MapTR and MapTRv2 tables, for example HDMapNet camera 14.4, VectorMapNet camera 36.1, MapTR-tiny R50 110 ep 56.2, MapTRv2 R50 110 ep 68.1 — [VectorMapNet](https://arxiv.org/html/2206.08920v6); [MapTR](https://arxiv.org/html/2208.14437); [MapTRv2](https://arxiv.org/html/2308.05736).
- These are BEV (bird's-eye-view) vector metrics matched by Chamfer distance at thresholds {0.5, 1.0, 1.5}; the MapTRv2 text does not state the units — [MapTRv2](https://arxiv.org/html/2308.05736); [VectorMapNet](https://arxiv.org/html/2206.08920v6).
- Panoramic segmentation has a Crosswalk class: WildPASS has 500 annotated panoramas from 25 cities, 360°x70° at 2048x400, built on Google Street View, with 8 classes (Car, Road, Sidewalk, Crosswalk, Curb, Person, Truck, Bus) — [ECANet paper](https://ar5iv.labs.arxiv.org/html/2103.05687).
- PASS has 400 images with Road, Car, Crosswalk and Curb originally, plus Sidewalk and Person added later — [DS-PASS paper](https://ar5iv.labs.arxiv.org/html/1909.07721).
- Published crosswalk scores on perspective road-marking sets exist too: CeyMo Pedestrian Crossing F1 94.95 to 96.86 by model — [CeyMo paper](https://ar5iv.labs.arxiv.org/html/2110.11867). BDD100K crosswalk ODS-F 9.37 at tau=10, the lowest lane category, with 10K training images — [BDD100K paper](https://ar5iv.labs.arxiv.org/html/1805.04687).
- Small crosswalk-specific sets exist outside the AV mainstream: CD9K (9,387 images at 1280x720, boxes re-annotated from BDD100K) — [X-CDNet](https://www.sciencedirect.com/science/article/abs/pii/S1047320324001627). FPVCrosswalk2025 (300 real first-person images plus synthetic, binary masks) — [Data in Brief](https://pmc.ncbi.nlm.nih.gov/articles/PMC12206053/). Neither has an associated leaderboard that I found.

### Inferences
- A fair statement for the PI: crosswalks are annotated everywhere and scored in two sub-fields (BEV map construction and panoramic segmentation), but no shared, crosswalk-focused benchmark exists, and none uses point-at-centre scoring.
- The HD-map `AP_ped` numbers are not comparable with our P 0.980 / R ~0.85 / F1 0.908. The task, the geometry, the matching rule and the input all differ.

### Gaps
- I searched the AV and HD-map literature, not the assistive-tech or remote-sensing literature in depth. Those are other researchers' scope.

## Perception datasets: which actually label crosswalks?

### Takeaway
Labelled: Vistas (both versions), BDD100K, ApolloScape, A2D2, CeyMo, VPGNet, PASS, WildPASS, ZOD (as a road-painting flag), UT CODa (LiDAR terrain). Metropolis has the class in machine-generated masks only. Not labelled: Cityscapes, KITTI-360, IDD, ACDC, nuImages, ONCE, TuSimple, CULane. Waymo's camera panoptic set folds crosswalk paint into a generic road-marker class, and WildDash 2 has only a generic "road markings" label.

### Cited Findings

**Datasets with a crosswalk label**

| Dataset | Exact class name(s) | Geometry | Marked / style distinction | Scale | Geography | Camera | Licence and access (2026) | Crosswalk benchmark? |
|---|---|---|---|---|---|---|---|---|
| Mapillary Vistas v1.2 | `Crosswalk - Plain`; `Lane Marking - Crosswalk` — [mmseg list](https://onedl-mmsegmentation.readthedocs.io/en/latest/_modules/mmseg/datasets/mapillary.html). IDs `construction--flat--crosswalk-plain`, `marking--crosswalk-zebra` — [third-party blog](https://jiaruixu.github.io/2018/05/02/Mapillary-Vistas-Dataset) | 2D pixel mask; both instance-labelled per the blog | Plain vs zebra at class level; no parallel-line class | 25,000 images, 66 classes — [Mapillary blog](https://blog.mapillary.com/update/2021/01/18/vistas-2-dataset.html) | Global (not re-confirmed from the paper) | Mixed consumer and pro, perspective (paper unreachable) | CC BY-NC-SA 4.0 — [Dataset Ninja](https://datasetninja.com/mapillary-vistas-dataset), third party. A Mapillary page snippet adds a Research Use License on download — [Mapillary](https://www.mapillary.com/dataset) | None found |
| Mapillary Vistas v2.0 | Adds `Lane Marking (only) - Crosswalk` to the two above — [mmseg list](https://onedl-mmsegmentation.readthedocs.io/en/latest/_modules/mmseg/datasets/mapillary.html). IDs `construction--flat--crosswalk-plain`, `marking--discrete--crosswalk-zebra`, `marking-only--discrete--crosswalk-zebra` appear in the [Metropolis SDK colour map](https://raw.githubusercontent.com/mapillary/metropolis_sdk/master/metropolis/utils/color_map.py), which shares the taxonomy | 2D pixel mask | Plain vs zebra vs "marking only" | 18k train, 2k val, 5k test (test unlabeled) — [Dataset Ninja](https://datasetninja.com/mapillary-vistas-dataset). 124 classes — [Mapillary blog](https://blog.mapillary.com/update/2021/01/18/vistas-2-dataset.html). Mapillary's listing now says 152 categories, unreconciled — [Mapillary](https://www.mapillary.com/datasets) | Global | As v1.2 | Same; "direct download" per the 2021 blog | None found |
| BDD100K | `crosswalk` (lane category id 0, alongside double/single white/yellow/other and road curb) — [label.py](https://raw.githubusercontent.com/bdd100k/bdd100k/master/bdd100k/label/label.py) | Lane-marking category (see Gaps on polyline format) | Parallel vs vertical direction attribute exists; no zebra vs parallel-line class — [paper](https://ar5iv.labs.arxiv.org/html/1805.04687) | 100K videos/images; 70K training images for lane marking; no crosswalk count — [paper](https://ar5iv.labs.arxiv.org/html/1805.04687) | New York, SF Bay Area and others, mostly US | Dashcam, 720p | Portal cert expired and doc host unresolved on 2026-11-08; devkit is BSD-3 and not archived, last push 2024-03 (GitHub API) | Lane task only, crosswalk ODS-F 9.37 |
| ApolloScape Lane Mark | `c_wy_z`, id 214, category `zebra` — [label file](https://raw.githubusercontent.com/ApolloScapeAuto/dataset-api/master/lane_segmentation/helpers/laneMarkDetection.py). Paper: "crosswalk w zebra"; separate `d_wy_za` "diamond w/y zebra attention" — [paper](https://ar5iv.labs.arxiv.org/html/1803.06184) | 2D pixel mask (lane segmentation) | Zebra only | 165,949 images stated; benchmark section gives 132,189 train + 33,790 test, unreconciled — [paper](https://ar5iv.labs.arxiv.org/html/1803.06184) | 3 road sites, China | 2 front cameras, 3384x2710 | Not confirmed; [site lane page](https://apolloscape.auto/lane_segmentation.html) returned 404 | Lane segmentation only |
| A2D2 | `Zebra crossing` (also `Painted driv. instr.`) — [xmuda loader](https://raw.githubusercontent.com/valeoai/xmuda/master/xmuda/data/a2d2/a2d2_dataloader.py), third party; official site unreachable | 2D pixel mask | No | 41,277 frames with segmentation, 38 classes — [paper](https://ar5iv.labs.arxiv.org/html/2004.06320) | Three cities (German source) | 6 cameras | CC BY-ND 4.0, commercial use allowed; anonymous S3 — [paper](https://ar5iv.labs.arxiv.org/html/2004.06320); [AWS registry](https://registry.opendata.aws/aev-a2d2/) | None found |
| CeyMo | `Pedestrian Crossing (PC)` — [README](https://raw.githubusercontent.com/oshadajay/CeyMo/main/README.md) | Polygons (GT), plus boxes and masks | No | 2,887 images, 4,706 instances, 839 PC (611 train, 228 test) — [paper](https://ar5iv.labs.arxiv.org/html/2110.11867) | Not stated (authors are from Sri Lanka) | Dashcam-type, 1920x1080 | Google Drive links; no licence stated — [README](https://raw.githubusercontent.com/oshadajay/CeyMo/main/README.md) | Eval script plus published PC F1 |
| VPGNet | `Crosswalk` (17 classes) — [paper](https://ar5iv.labs.arxiv.org/html/1710.06288) | Corner-point polygons converted to 8x8 grid cells | No | ~20,000 images, 13,632 crosswalk instances | Seoul | 1288x728, vehicle camera | Survey form then emailed link — [README](https://raw.githubusercontent.com/SeokjuLee/VPGNet/master/README.md) | None found |
| PASS | `Crosswalk` (6 eval classes) — [DS-PASS](https://ar5iv.labs.arxiv.org/html/1909.07721) | 2D pixel mask on unfolded annular image | No | 400 images, 2048x692 | Not stated | Panoramic annular lens, 360° horizontal, wearable | [DS-PASS repo](https://github.com/elnino9ykl/DS-PASS); licence not confirmed | Crosswalk IoU: ERF-PSPNet 52.9, SwiftNet 55.9, SwaftNet 62.1 |
| WildPASS | `Crosswalk` (8 classes) — [ECANet paper](https://ar5iv.labs.arxiv.org/html/2103.05687) | 2D pixel mask | No | 500 annotated panoramas (25 cities) plus 2,000 unlabeled (40 cities) | 6 continents | Street View panorama, 360°x70° band, 2048x400 | Google Drive; repo MIT (code), data terms unstated — [README](https://raw.githubusercontent.com/elnino9ykl/WildPASS/master/README.md) | Crosswalk column in a paper table: 67.9 (ECANet), 70.5 (omni-supervised), 6.4 (Vistas-trained ERF-PSPNet baseline); caption says per-class accuracy |
| ZOD | Road-painting annotation flag `ContainsCrossWalk` — [lane.py](https://raw.githubusercontent.com/zenseact/zod/main/zod/anno/lane.py) | 2D point list | No | Not confirmed | Not confirmed | Not confirmed | Not confirmed (HF card returned 401) | None found |
| UT CODa | `Crosswalk` terrain class, id 18 — [coda-devkit](https://github.com/ut-amrl/coda-devkit) `helpers/constants.py` (GitHub code search) | 3D LiDAR point labels; not images | No | 5,000 frames terrain semantic annotations — [paper](https://ar5iv.labs.arxiv.org/html/2309.13549) | UT Austin campus | Robot | Texas Data Repository | None found |
| Mapillary Metropolis | Masks use the Vistas-style names above — [colour map](https://raw.githubusercontent.com/mapillary/metropolis_sdk/master/metropolis/utils/color_map.py) | "Machine-generated panoptic masks" on the panorama — [FORMAT.md](https://raw.githubusercontent.com/mapillary/metropolis_sdk/master/FORMAT.md) | Zebra vs plain by class | Counts not stated | Not stated | **360° equirectangular "source of truth"** plus 4 warped perspective views — [FORMAT.md](https://raw.githubusercontent.com/mapillary/metropolis_sdk/master/FORMAT.md) | Available "at the Mapillary dataset page" per the [2021 tutorial](https://research.mapillary.com/MetropolisTutorial); dataset licence not confirmed; SDK Apache 2.0, not archived (GitHub API) | Not for crosswalks; the tutorial lists 2D/3D detection, tracking, depth |

**Datasets that do NOT label crosswalks**

| Dataset | Evidence |
|---|---|
| Cityscapes | No crosswalk or road-marking label among its 34 — [labels.py](https://raw.githubusercontent.com/mcordts/cityscapesScripts/master/cityscapesscripts/helpers/labels.py) |
| KITTI-360 | No crosswalk or road-marking label among its 46 — [labels.py](https://raw.githubusercontent.com/autonomousvision/kitti360Scripts/master/kitti360scripts/helpers/labels.py) |
| IDD | No crosswalk or road-marking label — [anue_labels.py](https://raw.githubusercontent.com/AutoNUE/public-code/master/helpers/anue_labels.py) |
| ACDC | Annotates the 19 Cityscapes evaluation classes; 8,012 images — [paper](https://ar5iv.labs.arxiv.org/html/2104.13395) |
| nuImages | Surface classes are `flat.driveable_surface` and `vehicle.ego` only — [instructions](https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/docs/instructions_nuimages.md) |
| Waymo camera panoptic | 28 classes with no crosswalk; `TYPE_ROAD_MARKER` is "All markings on the road other than lane markers" — [camera_segmentation.proto](https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/src/waymo_open_dataset/protos/camera_segmentation.proto) |
| WildDash 2 | Adds a generic "road markings" label; no crosswalk class named; 4,256 public frames; 3 leaderboards, none crosswalk-specific — [wilddash.cc](https://www.wilddash.cc/) |
| ONCE | 5 classes (Car, Bus, Truck, Pedestrian, Cyclist) — [ONCE site](https://once-for-auto-driving.github.io), via search result |
| TuSimple | Lane markings only, highway — [TIB catalogue](https://service.tib.eu/ldmservice/dataset/tusimple-lane-detection-benchmark), secondary |
| CULane | "Other lane markings are not annotated" — [CULane page](https://xingangpan.github.io/projects/CULane.html) |

### Inferences
- WildPASS and PASS are the only public human-labelled panoramic datasets found with a crosswalk class. The Vistas-trained baseline's 6.4% on WildPASS suggests a perspective-trained model does poorly on panoramic crosswalks, which is the gap our detector is meant to close.
- Vistas's `plain` vs `zebra` split is the only widely available marked-style distinction, but no fetched source defines the two. Treat it as a naming convention until checked against the paper.
- Metropolis uses machine-generated masks, so it is not ground truth. If those masks came from a Vistas-trained model (likely, not confirmed), they are circular with our Mask2Former extent filter.

### Gaps
- Vistas paper returned 403, so class definitions, per-class crosswalk counts and camera mix are unconfirmed. Dataset Ninja's per-class table was not visible in what I could fetch.
- BDD100K's `poly2d`-style geometry could not be confirmed because `doc.bdd100k.com` did not resolve and `format.rst` returned 404.
- Whether WildPASS's Drive archive contains masks for all 500 panoramas or only a validation subset is not stated. The archive has to be opened to find out.
- A2D2's official `class_list.json` was not read; the class name comes from third-party loader code and copies of the file in other repos.
- TRoM (712 images, 19 classes) is listed in a search result; its crosswalk class is unconfirmed.
- Bosch Small Traffic Lights returned HTTP 500 and was not confirmed from the source. Its name implies traffic-light-only labels; I did not verify that.
- Metropolis's city, image count, current download state and dataset licence are unconfirmed: the Mapillary pages are JavaScript-rendered and returned nothing useful.

## HD-map and vectorized datasets: which publish a ped_crossing metric, and could their imagery be projected into a usable test?

### Takeaway
nuScenes, Argoverse 2 and Waymo all have polygon crosswalks with no marked-vs-unmarked attribute, and nuScenes and Argoverse 2 are the established `AP_ped` targets. Argoverse 2 is the strongest candidate of the three for a projection test because its ring is described as fully panoramic, its downloads are open, and each crosswalk is a simple quadrilateral.

### Cited Findings

| Dataset | Class / geometry | Marked distinction | Scale and geography | Cameras | Licence and access | ped_crossing metric |
|---|---|---|---|---|---|---|
| nuScenes map expansion | `ped_crossing` polygon layer; `stop_line_type` may be `PED_CROSSING` — [map_api.py](https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/python-sdk/nuscenes/map_expansion/map_api.py) | No type field seen. Described as "regions with white markings indicating legal pedestrian crossing points" — [VectorMapNet](https://arxiv.org/html/2206.08920v6) | 1,000 scenes, 40k keyframes, Boston and Singapore — [paper](https://ar5iv.labs.arxiv.org/html/1903.11027) | 6 pinhole, 1600x900; front/side 70° HFOV offset 55°, rear 110° | CC BY-NC-SA 4.0 — [paper](https://ar5iv.labs.arxiv.org/html/1903.11027); registration needed | `AP_ped` (figures above) |
| Argoverse 2 Sensor + Map | "Pedestrian crossing": `edge1`, `edge2`, each a 3D polyline with 2 waypoints — [AV2 docs](https://argoverse.github.io/user-guide/api/hd_maps.html); endpoints define a polygon — [paper](https://ar5iv.labs.arxiv.org/html/2301.00493) | None | 1,000 scenarios of 15 s; Austin, Detroit, Miami, Palo Alto, Pittsburgh, Washington DC — [paper](https://ar5iv.labs.arxiv.org/html/2301.00493) | 7 ring cameras, "fully panoramic field of view", plus 2 stereo | CC BY-NC-SA 4.0; `s5cmd --no-sign-request` from `s3://argoverse/datasets/av2/sensor/` — [guide](https://argoverse.github.io/user-guide/getting_started.html) | MapTRv2 2D 62.9, 3D 60.7 — [MapTRv2](https://arxiv.org/html/2308.05736) |
| Waymo Perception | `Crosswalk`: closed polygon of `MapPoint`, no type — [map.proto](https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/src/waymo_open_dataset/protos/map.proto) | None | 2,030 segments of 20 s, ~390k frames, v1.4.3 / v2.0.1 — [Waymo](https://waymo.com/open/about/) | 5 cameras (front and sides) | Non-commercial — [Waymo](https://waymo.com/open/about/) | None found |
| Waymo Motion | Same map polygon | None | 103,354 segments, 574 h; v1.3.1 Oct 2025; SF, Phoenix, Mountain View, LA, Detroit, Seattle — [Waymo](https://waymo.com/open/about/) | 8 cameras but **embeddings only**, no images | Non-commercial | None found |
| OpenLane-V2 | "Pedestrian crossing", treated as area — [features](https://raw.githubusercontent.com/OpenDriveLab/OpenLane-V2/master/docs/features.md) | None | Builds on nuScenes and AV2 (users must accept both terms) — [README](https://raw.githubusercontent.com/OpenDriveLab/OpenLane-V2/master/README.md) | Multi-view | CC BY-NC-SA 4.0 | No separate number: `DET_a` pools crossings with road boundaries, Chamfer, into OLUS — [metrics](https://raw.githubusercontent.com/OpenDriveLab/OpenLane-V2/master/docs/metrics.md) |
| Lyft / Woven Prediction | `pedestrian_crosswalk`: "ground polygon around an area where pedestrians cross the road and vehicles yield" — [road_network.proto](https://raw.githubusercontent.com/woven-planet/l5kit/master/l5kit/l5kit/data/proto/road_network.proto) | None | ~1,118 h, Palo Alto route — [arXiv](https://arxiv.org/pdf/2006.14480v2.pdf) | Not image-based for this dataset | l5kit repo **archived** (GitHub API); [woven.toyota page](https://woven.toyota/en/prediction-dataset) exists, state unverified | None |
| nuPlan | `SemanticMapLayer.CROSSWALK` — [maps_datatypes.py](https://github.com/motional/nuplan-devkit/blob/master/nuplan/common/maps/maps_datatypes.py) | Not confirmed | 4 locations (Singapore, Boston, Las Vegas, Pittsburgh); sensor blobs per log — [setup](https://raw.githubusercontent.com/motional/nuplan-devkit/master/docs/dataset_setup.md) | 8 cameras | Account plus Terms of Use | Planning-focused; none found |
| HRDX (2026) | `Crosswalk`, polygon, "Pedestrian crossing region", no attributes — [paper](https://arxiv.org/html/2606.17080) | None | 40+ h, 1,400 km, SF Bay Area | 6 wide-angle cameras at 10 Hz, plus aerial | Non-commercial; university email and request form — [Honda](https://usa.honda-ri.com/hrdx) | Crosswalk AP: MapTRv2 43.82, StreamMapNet 54.36, SQD-MapNet 51.49, MapTracker 65.40, up to 73.29 with aerial |
| KITScenes Multimodal (2026) | `zebra_crossing` and `pedestrian_crossing` under Lane Markings — [paper Table 16](https://arxiv.org/html/2606.02956); no definitions given | **Two separate classes** | ~1,000 scenarios, 162 km, 62 km² map; Karlsruhe, Frankfurt, Sindelfingen — [HF card](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal) | 9 global-shutter cameras, 360° | CC BY-NC 4.0 plus extra terms, gated; early release v1.0.x, "not recommended for final benchmark reporting" — [site](https://kitscenes.com/) | Leaderboard planned for the `test-e2e` split; crosswalk scoring not confirmed |

- HDMapNet and VectorMapNet are methods on nuScenes, not datasets; their per-class numbers are above — [VectorMapNet](https://arxiv.org/html/2206.08920v6).
- MapTR's README lists only overall mAP; per-class numbers are in the papers — [README](https://raw.githubusercontent.com/hustvl/MapTR/main/README.md).
- nuScenes map-to-image alignment has a measured error: "around 8.03 pixels reprojection error on average" — [CAMAv2](https://arxiv.org/abs/2407.21331).

### Inferences
- **How far from equirect:** our input is 4096 px / 360° = 11.4 px/deg. nuScenes is 22.9 px/deg (front/side) and 14.5 (rear), so stitched output would be downsampled, not upsampled. WildPASS is 5.7 px/deg, requiring 2x upsampling.
- **Stitching cost:** the optical centres differ, so a synthesized equirect has parallax seams and exposure steps. Rendering each camera into equirect separately, without blending, and letting a detection count if any view fires avoids seams but changes the input distribution.
- **Nadir:** nuScenes front/side cameras (70° HFOV at 16:9) cover about ±21.5° vertically, assuming a centred pinhole. With a 1.5 to 1.8 m camera height (assumed, not verified), a crosswalk centre closer than about 4 m falls out of frame. Most crosswalks in view are farther, so the missing nadir is a modest loss. The black fill is the bigger risk, since the model has never seen it.
- **Registration error is small here:** 8 px at 22.9 px/deg is about 0.35°, much smaller than a crosswalk's angular extent at normal range.
- **Marked vs unmarked is the main label confound.** None of nuScenes, AV2 or Waymo carries a marking attribute, and Lyft's definition does not require paint. An unmarked crossing polygon would count against our marked-crosswalk precision. Argoverse's and Waymo's map content would need a sample audit before trusting precision.
- **Scoring:** for elongated polygons, "peak falls inside polygon, projected in world space" is safer than distance-to-centroid. AV2's four corner points make a centroid trivial if one is still wanted.

### Gaps
- No source gives a total crosswalk count for nuScenes, AV2, Waymo, HRDX or KITScenes.
- AV2 ring camera resolution and vertical FOV were not confirmed from a fetched source.
- nuPlan's sensor-data completeness and whether its crosswalk layer differentiates marking type were not checked.
- OpenLane-V2's challenge servers are "active" per its README, but I did not verify them live.

## Newer (2024-2026) datasets

### Takeaway
Most new large releases carry no crosswalk labels, or only in map form. HRDX and KITScenes (both 2026) are the two new datasets with crosswalk classes and real imagery. NVIDIA's Cosmos-Drive-Dreams has 3D crosswalk polygons, but only over synthetic video so far.

### Cited Findings
- NVIDIA PhysicalAI-Autonomous-Vehicles: 306,151 clips, ~1,700 h, 25 countries, 7 cameras; the card lists "autogenerated (non-GT) machine labels" and never mentions lanes or crosswalks — [HF card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).
- Its licence is restricted to internal AV development using NVIDIA technology, bars derivative works, and expires 12 months after download — [HF card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).
- PhysicalAI-Autonomous-Vehicles-NCore: ~1.1k clips with 3D cuboids only — [HF card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore).
- Cosmos-Drive-Dreams: "Crosswalk annotations, polygon format" for 5,843 RDS-HQ clips; 81,802 videos, all synthetic; real videos promised "soon"; CC BY 4.0, AV use only — [HF card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams).
- WOD-E2E: 4,021 segments of 20 s, 8 cameras at 360°, built for planning; no crosswalk annotation was found — [challenge page](https://waymo.com/open/challenges/2025/e2e-driving/).
- MAN TruckScenes: "does not provide map data"; 4 cameras; CC BY-NC-SA 4.0; Germany — [paper](https://arxiv.org/html/2407.07462v2).
- DriveLM-nuScenes (696 scenes) and OmniDrive are VQA layers on nuScenes; no crosswalk class was found in either — [DriveLM](https://arxiv.org/pdf/2312.14150); [OmniDrive](https://arxiv.org/pdf/2405.01533).
- A strategic survey of AV dataset creation appeared in 2026 — [arXiv 2607.00710](https://arxiv.org/pdf/2607.00710). I did not read it in detail.

### Inferences
- Cosmos-Drive-Dreams would become useful only if the real multi-view videos ship, and its AV-only restriction would need checking against our use.
- Synthetic Cosmos video is not a valid independent test of a detector meant for real imagery.

### Gaps
- Whether Cosmos real-world videos have been released since the fetched card revision is unconfirmed.
- "CoDA" is ambiguous. I confirmed UT CODa; the separate CODA corner-case detection dataset was not checked.
- Coverage of 2025-2026 arXiv releases came from a small number of searches, so a newer dataset could have been missed.

## Is there any challenge or workshop track that scores crosswalks as a task?

### Takeaway
None that scores crosswalks alone. The closest are the CVPR 2023 online HD-map track and the CVPR 2024 Mapless Driving track, where pedestrian crossings are one element of a pooled metric.

### Cited Findings
- CVPR 2024 Autonomous Grand Challenge Mapless Driving predicted centerlines, boundaries and pedestrian crossings, scored by OLUS — [MapVision report](https://arxiv.org/pdf/2406.10125). OLUS averages `DET_l`, `DET_a`, `DET_t` and two topology terms, with crossings inside `DET_a` — [OpenLane-V2 metrics](https://raw.githubusercontent.com/OpenDriveLab/OpenLane-V2/master/docs/metrics.md).
- CVPR 2023 online HD-map construction used reshaped Argoverse 2 with an mAP leaderboard — [MapNeXt](https://arxiv.org/pdf/2401.07323).
- Autonomous Grand Challenge 2025 listed NAVSIM v2, 1X World Model and AgiBot tracks, with no map or crosswalk track — [OpenDriveLab](https://opendrivelab.com/challenge2025/).
- CVPR 2026 items found: a V2I cooperative perception track — [DriveX](https://drivex-workshop.github.io/cvpr2026); a Physical AI AV Reasoning challenge with chain-of-causation and auto-labeling tracks — [NVIDIA](https://nvidia-physicalai-av-ood-reasoning-challenge-2026.static.hf.space/index.html); and a KITScenes LongTail challenge at WAD, per search results. None mentions crosswalks.

### Inferences
- Online HD-map tracks appear to have faded from flagship challenge programs since 2024, which reduces the chance of a live crosswalk leaderboard.

### Gaps
- EvalAI, Codabench, AI City Challenge 2026 and ICCV 2025 RoboSense were not enumerated. A crosswalk track there is unlikely but not excluded.

## Which would be the best independent cross-domain test for our detector?

### Takeaway
Run two complementary tests. **WildPASS** is the only human-labelled crosswalk ground truth on real street-level panoramas across many cities. **Argoverse 2 Sensor + Map** is the largest openly downloadable test with independent vector ground truth and a genuine camera-domain shift. Neither is a drop-in, and both need a label-quality audit.

### Cited Findings
- WildPASS provides 500 Street View-based panoramas from 25 cities, with masks for Crosswalk — [ECANet paper](https://ar5iv.labs.arxiv.org/html/2103.05687); [README](https://raw.githubusercontent.com/elnino9ykl/WildPASS/master/README.md).
- AV2 provides 1,000 scenarios in 6 US cities, 7-camera "fully panoramic" ring, crosswalks as two 3D edges — [paper](https://ar5iv.labs.arxiv.org/html/2301.00493); [docs](https://argoverse.github.io/user-guide/api/hd_maps.html).
- Metropolis provides equirectangular panoramas but "machine-generated panoptic masks" — [FORMAT.md](https://raw.githubusercontent.com/mapillary/metropolis_sdk/master/FORMAT.md).

### Inferences

Ranking, by independence and fit with a point-at-centre panorama detector:

| Rank | Dataset | Why | Main risks |
|---|---|---|---|
| 1 | WildPASS | Same imagery family as our Seattle audit, new geography, independent human masks; connected-component centroids or mask-containment give point GT | Only a 360°x70° band at 2048x400: needs 2x upsampling and padding into 4096x2048, and the model has not seen padded input. Small N. Mask release unverified. The README documents no pano IDs (only a city-name variant), so re-fetching full-resolution originals looks unlikely; check the archive |
| 2 | Argoverse 2 Sensor + Map | Real camera-domain shift, 6 US cities, vector GT independent of any image model, easy download | Ring-to-equirect rendering needed; no marked/unmarked attribute; non-commercial |
| 3 | nuScenes | Largest literature footprint and Singapore geography | Same projection issues; known 8 px map-image misalignment; non-commercial |
| 4 | KITScenes | Only source separating `zebra_crossing` from `pedestrian_crossing`; 360° global-shutter rig, German domain | Early release; class semantics undefined; gated |
| 5 | HRDX | Fresh and crosswalk AP published | University-only request; same Bay Area region as many other US sets; no 360° rig claim |
| 6 | Vistas v2.0 | Human zebra masks, global | Perspective images, and already used to train our extent filter via Mask2Former, so not fully independent |
| 7 | PASS | Panoramic with crosswalks | Annular lens, wearable, small |
| 8 | Metropolis | True equirect input | Machine masks, likely circular; useful only as an input-format sanity check |

- Waymo Perception is excluded from the top group because only 5 cameras cover front and sides, so a 360° render would be incomplete; Motion has no pixels at all.
- A clean protocol reports WildPASS and AV2 separately: one tests geography with the same camera type, and the other tests camera domain.
- For both, measure precision twice: raw against the labels, and after a human audit of apparent false positives, since unmarked or stale labels will otherwise depress precision.

### Gaps
- Whether RampNet behaves sensibly on a padded 360°x70° band has not been tested.
- WildPASS crosswalk instance count and the share of panoramas containing any crosswalk are unknown until the masks are inspected, and determine whether recall can be estimated reliably from it.
- Image capture dates versus map dates are unknown for the HD-map sets, and crosswalks can be repainted or removed between the two.
