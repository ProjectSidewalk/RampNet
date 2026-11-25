# Public crosswalk models, small datasets and baseline candidates: hub inventory (accessed 2026-11-08)

Access limits that affect every section below:

- **Hugging Face (HF), GitHub and Zenodo** were read directly through their APIs. Counts are as of 2026-11-08.
- **Roboflow Universe returned HTTP 403 to automated fetching.** Every Roboflow entry below comes from web-search index snippets only. **Those pages were not opened**, so the assignment's "confirm from the actual page" standard is not met for them.
- **Kaggle:** WebFetch returned only the page title (the page is JS-rendered), so Kaggle entries are also snippet-only. FPVCrosswalk2025 was confirmed from its Mendeley Data primary page instead.
- **Also blocked (403 or browser check):** the CVF Vistas supplement, the MDPI SAM paper, and the PMC copy of the FPVCrosswalk paper. Claims that rest on search snippets of those sources are labelled.
- **HF search matches repo names, not card text.** A dataset named without "crosswalk" or similar can be missed.

## Hugging Face: what exists for crosswalk, zebra crossing, pedestrian crossing, road marking

### Takeaway
HF holds one serious street-level dashcam crosswalk dataset (CDSet, 3,434 images), one serious road-marking dataset (RLMD, 2,137 images), and one multi-class pedestrian-view set with a useful crosswalk class (ROD). There are also two aerial sets. **Among cards that could be read, no HF crosswalk model reports a metric on a named public test set.** Project Sidewalk's own validator reports metrics on its own split and is a crop classifier. The GPSO U-Net's card could not be opened. The most-downloaded crosswalk-related checkpoint anywhere on HF is the generic Mask2Former Vistas model we already ran.

### Cited Findings

Datasets, most useful first. Downloads are HF's rolling count.

| Repo | Size | Annotation | Viewpoint | Licence | Downloads / last change | Verdict |
|---|---|---|---|---|---|---|
| [zzd0225/crosswalk-detection-dataset](https://huggingface.co/datasets/zzd0225/crosswalk-detection-dataset) (CDSet-3434) | 3,434 images (3,080 train, 354 test), plus 1,770 images labelled with or without crosswalk; 1.1 GB | Boxes; classes crosswalk and guide arrows | Vehicle-mounted camera | Apache-2.0 on HF; [CC BY 4.0 on Zenodo](https://zenodo.org/records/8289874) | 237 / 2023-09 | **Serious but narrow.** The most-cited public street-level crosswalk set found; dashcam, not 360 |
| [veetinator/Road_Line_Marking_Dataset](https://huggingface.co/datasets/veetinator/Road_Line_Marking_Dataset) (RLMD) | 2,137 images; 1.36 GB | Polygon masks, 25 categories including crosswalk and box junction | Driving scene | MIT | 363 / 2025-06 | **Serious**, with a peer-reviewed [paper](https://ieeexplore.ieee.org/document/10226935) (paper page not opened) |
| [Abtinzandi/Obstacle-Detection-Dataset-YOLO](https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO) (ROD) | 24,326 images, 25 classes; [crosswalk class 2,493 instances (1,875 / 429 / 189 train/val/test)](https://arxiv.org/html/2611.00681) | Boxes | [Pedestrian eye and phone height; Tehran, Toronto, plus 29 Roboflow collections](https://arxiv.org/html/2611.00681) | MIT (constituent sources vary) | 6,928 / 2026-11-03 | Usable as a pedestrian-view test. It is an aggregate of scraped Roboflow sets, and only 189 crosswalk instances are in test |
| [SightLinks/YOLO-OBB-Zebra-Crossings-Dataset](https://huggingface.co/datasets/SightLinks/YOLO-OBB-Zebra-Crossings-Dataset) | 8,550 images + 8,550 label files; 2.28 GB | Label files; oriented boxes per repo name | Aerial or satellite per the [sibling model's name](https://huggingface.co/SightLinks/YOLO-OBB-Zebra-Crossings-Satellite) (model is gated) | Unknown | 1,160 / 2025-03 | Largest crosswalk image set on HF by file count. **Blank card** |
| [mandresyandri/gpso-crosswalks-20cm-dataset](https://huggingface.co/datasets/mandresyandri/gpso-crosswalks-20cm-dataset) | 563 tiles, about 433 distinct crossings per card; 1.08 GB | Binary masks, single annotator, SAM 3 "pedestrian crossing" pre-annotation | Aerial, IGN 20 cm | Etalab 2.0 | 687 / 2026-09 | Honest, well-documented card, but one territory and aerial. The card's limits section states that tiles without crossings are mostly absent, so FP rate is under-tested. The linked model repo could not be opened |
| [nicO1asFr crosswalks-france family](https://huggingface.co/datasets/nicO1asFr/crosswalks-france-detection) (detection, grounding, sam, segmentation, pedestrian-crosswalks v1/v2) | 75 to 381 JPEGs per repo | Boxes or grounding JSON | Aerial (one sample inspected; filenames are OSM nodes in Marseille) | No licence tag | 33 to 408 / 2025-10 | Toy. No cards |
| [filepile/crosswalk](https://huggingface.co/datasets/filepile/crosswalk) | 214 images, 83 MB | Not documented | Not documented | Apache-2.0 | 69 / 2024-05 | Toy; the [companion model](https://huggingface.co/filepile/crosswalk-traffic-light) says it was "TRAINED WITH 30 IMAGES" |
| [Zhonghua/Crosswalk_Pixelwise_Groundtruth](https://huggingface.co/datasets/Zhonghua/Crosswalk_Pixelwise_Groundtruth) | One 223 MB zip | Pixel masks per name | Unknown (zip not opened) | Apache-2.0 | 20 / 2024-08 | Unknown; blank card |
| [projectsidewalk/sidewalk-validator-ai-dataset-crosswalk](https://huggingface.co/datasets/projectsidewalk/sidewalk-validator-ai-dataset-crosswalk) | 1,585 WebP crops, 541 MB | Train/val/test folders | Perspective crops from street-level panos | Not tagged | 8 / 2025-08 | Ours; it is crop classification, not localization |

Non-imagery or irrelevant hits, listed so nobody re-checks them:

- [EmbedEarth/uncontrolled-crosswalk](https://huggingface.co/datasets/EmbedEarth/uncontrolled-crosswalk) is OSM features (ODbL), not images.
- [oscur/NYC_raised_crosswalk](https://huggingface.co/datasets/oscur/NYC_raised_crosswalk) has 119 tabular rows.
- [french-open-data/passages-pietons](https://huggingface.co/datasets/french-open-data/passages-pietons) and similar are open-data tables.
- [Asilbek14/Crosswalk_Violation](https://huggingface.co/datasets/Asilbek14/Crosswalk_Violation) is a project dump; its README is about licence-plate models.

Most "crosswalk" hits on HF are metadata crosswalks (occupational codes, governance frameworks).

Search terms: "paso de cebra", "passage pieton", "Zebrastreifen", "pedestrian crossing", "road markings" and "crossing detection" returned zero or near-zero repos by name through the HF API. Single-token searches for "cebra", "pieton" and "zebra" surfaced only the unrelated repos above.

Models:

| Repo | What | Reported metrics | Downloads / last change | Verdict |
|---|---|---|---|---|
| [facebook/mask2former-swin-large-mapillary-vistas-semantic](https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-semantic) | 65-class Vistas v1.2 | Card gives none per class | 39,992 / 2023-01 | The de facto crosswalk-capable checkpoint, and already run |
| [projectsidewalk/sidewalk-validator-ai-crosswalk](https://huggingface.co/projectsidewalk/sidewalk-validator-ai-crosswalk) | DINOv2 label-validation crop classifier | Accuracy 0.9573, AUC 0.9799, macro F1 0.9565 on own test split | 101 / 2026-10 | Ours. The only crosswalk model found on HF with a model-index metric block. Out of scope for whole-pano detection per its card |
| [Aleton/Zebra_crossing](https://huggingface.co/Aleton/Zebra_crossing) | YOLOv8n-seg, 1 class | None; architecture stats only | 23 / 2026-05 | Toy; no training data described |
| [SatyaShodhaka/florence-crosswalk-*](https://huggingface.co/SatyaShodhaka/florence-crosswalk2-qa-json) (about 10 repos) | Florence-2 fine-tunes | Default template card, "More Information Needed" | 2 to 13 / 2024-25 | Unusable |
| [calinMoglan/crosswalk-detector-v1.3 to v2.3](https://huggingface.co/calinMoglan/crosswalk-detector-v2.3) | LFM2-VL-3B LoRA adapters | Default template card | 3 to 4 / 2026-01 | Unusable |
| [snooplsm/reported-bike-lane-crosswalk-detector](https://huggingface.co/snooplsm/reported-bike-lane-crosswalk-detector) | YOLOv8s-seg ONNX, 13 classes | No metrics seen in the excerpt read | 0 / 2026-06 | Detects "blocked crosswalk", not crosswalks |
| [8bits-ai/SegFormer-Lane-Marking](https://huggingface.co/8bits-ai/SegFormer-Lane-Marking) | SegFormer | Blank card | 24 / 2024-08 | Unusable |
| [SightLinks/YOLO-OBB-Zebra-Crossings-Satellite](https://huggingface.co/SightLinks/YOLO-OBB-Zebra-Crossings-Satellite) | YOLO OBB | Gated; card not readable | 0 / 2025-03 | Could not assess |

### Inferences
- The PI's impression holds for HF: no community-standard crosswalk model exists there. Download counts are in single or low double digits, excluding the generic Vistas segmenters.
- CDSet and RLMD are the only pure-play street-level sets worth a held-out test. Neither is panoramic, so they measure transfer rather than our task.

### Gaps
- HF search is name-only, so cards that mention crosswalks in text are not covered. No full-text search was available.
- RLMD paper metrics were not read (IEEE page not opened).

## Roboflow Universe and Kaggle

### Takeaway
Listings run to a few thousand images at most for crosswalk-focused projects, and a claimed 10.2k for a multi-class one. All are unaudited community uploads. **None of these pages could be opened, so image counts are from search snippets, class definitions are thin, and provenance is unverified.** Do not treat any of them as a benchmark.

### Cited Findings (search-index snippets only, pages not opened)
- SafeWalkBD: 10.2k images, 6 models, multi-class including crosswalk — [Roboflow search listing](https://universe.roboflow.com/search?q=class%3Azebra+crossing).
- [Pedestrian-Zebra](https://universe.roboflow.com/distracted-driving-yu8re/pedestrian-zebra): about 5.2k images; classes car, crosswalk, cyclist, person and zebra crossing, so the two crossing classes overlap; object detection.
- [Segmentation 3 (Janissy)](https://universe.roboflow.com/janissy/segmentation-3-pqrfl): 1,949 images, Zebra Crossing, instance segmentation, CC BY 4.0.
- CrosSight 1.9k images; crosswalk-TA 1.1k images, instance segmentation — [Roboflow listing](https://universe.roboflow.com/search?q=class:crosswalk).
- [pedestrianCross 1.75k images and "Group 2 Safety and People" 1.83k images](https://universe.roboflow.com/browse/transportation/crosswalk).
- Toy end of the range: [Telkom University zebra-cross, 40 images](https://universe.roboflow.com/telkom-university-ifw99/zebra-cross-dataset); [SeniorDesignProject zebras, 501 images](https://universe.roboflow.com/seniordesignproject/zebras-nlay8).
- [ROD](https://arxiv.org/html/2611.00681) aggregates 29 Roboflow Universe collections, with mostly CC BY 4.0 and one that states no licence. It shows these sets get reused as ingredients.
- Kaggle: [DataCluster Labs Zebra Crossing](https://www.kaggle.com/datasets/dataclusterlabs/zebra-crossing) advertises over 3,000 images from over 200 locations, in COCO, YOLO, VOC and TFRecord formats. The search snippet says the full set appears to require contacting the vendor, and the description has a copy-paste reference to excavators, so treat it as a commercial teaser.
- Kaggle: [FPVCrosswalk2025](https://www.kaggle.com/datasets/jocelyndumlao/fpvcrosswalk2025-first-person-view-crosswalk) is a mirror. The primary record is [Mendeley Data](https://data.mendeley.com/datasets/mcr2jwk5bp/1): 3,000 Stable Diffusion synthetic plus 300 real first-person images, quadrilateral binary masks, CC BY 4.0, published 2025-04-11.
- A [ConvNeXt-DeepLabv3 paper](https://jiki.cs.ui.ac.id/index.php/jiki/article/view/1757) reports 0.946 mean IoU and 0.972 Dice on FPVCrosswalk (search snippet). Masks are quadrilaterals and 91% of the images are synthetic, so that number says little about field performance.

### Inferences
- Class definitions conflict even inside one project (crosswalk and zebra crossing as separate classes), which supports the fragmentation argument.
- Any use of Roboflow data needs a manual label audit first.

### Gaps
- Roboflow image counts, annotation quality, mAP, provenance and last-updated dates are unverified. Confirming them needs a human opening the pages, or a Roboflow API key used within its terms. Kaggle download counts were not obtained.

## GitHub and Zenodo

### Takeaway
The only dedicated street-level crosswalk detector of note on GitHub is [CDNet](https://github.com/zhangzhengde0225/CDNet) (145 stars, a 2022 YOLOv5 derivative whose README documents a model zoo). The rest are course projects and small assistive apps. Zenodo adds a real street-view crosswalk product and a small mixed-class set, but no detector weights were found.

### Cited Findings
GitHub search, 2026-11-08, sorted by stars for "crosswalk detection" and "zebra crossing detection":

| Repo | Stars / GitHub `updatedAt` (can reflect metadata, not commits) | Notes |
|---|---|---|
| [geometalab/OSMDeepOD](https://github.com/geometalab/OSMDeepOD) | 197 / 2026-01 | Aerial, OSM-seeded; MIT |
| [zhangzhengde0225/CDNet](https://github.com/zhangzhengde0225/CDNet) | 145 / 2026-04 | Dashcam; README lists a model zoo, datasets and 33.1 FPS on Jetson Nano. [Paper reports average F1 94.83%](https://link.springer.com/article/10.1007/s00521-022-07007-9) (snippet). Listed free for non-commercial use per [search summary](https://github.com/zhangzhengde0225/CDNet/blob/master/docs/README_zh_cn.md) |
| [xN1ckuz/Crosswalks-Detection-using-YOLO](https://github.com/xN1ckuz/Crosswalks-Detection-using-YOLO) | 46 / 2026-10 | University course project, YOLOv5 then YOLOv8, self-collected photos, GPL-3.0 |
| [Sh-Zh-7/intelligent-transportation-system](https://github.com/Sh-Zh-7/intelligent-transportation-system) | 57 / 2026-09 | Course-style integrated project with a crossing component (description only; README not read) |
| [TomMao23/ZebraCrossing_Detection](https://github.com/TomMao23/ZebraCrossing_Detection) | 13 / 2025-08 | Small |
| [z00bean/CrosswalkNet](https://github.com/z00bean/CrosswalkNet) | 1 / 2025-03 | Aerial; README is figures only; paper at [arXiv 2506.07885](https://arxiv.org/pdf/2506.07885) |
| [xgyutu/X-CDNet](https://github.com/xgyutu/X-CDNet) | 0 / 2025-08 | Code for the X-CDNet paper; [93.3 AP50, 123 FPS on its own CD9K set](https://www.bohrium.com/paper-details/x-cdnet-a-real-time-crosswalk-detector-based-on-yolox/1011454158751924229-2631) (snippet). No confirmed CD9K download |
| [25-p-project-team-6/VIA-Safe-Crosswalk](https://github.com/25-p-project-team-6/VIA-Safe-Crosswalk) | 2 / 2026-06 | Assistive Android app; on-device traffic-light detection |

Everything else returned had seven stars or fewer, or was unrelated.

Zenodo:

- [CDSet](https://zenodo.org/records/8289874): CC BY 4.0, 1.1 GB. This is the only hit for an exact "zebra crossing" dataset search besides an unrelated vehicle-pedestrian interaction record.
- [Marked crosswalks in US transit-oriented station areas, 2007-2020](https://zenodo.org/records/6803956): CC BY 4.0, per-intersection, per-year presence of parallel-line and high-visibility crosswalks **derived by computer vision from street view imagery**. Tabular results only, no weights, 159 downloads. It is the closest prior work to our viewpoint in this scan.
- [Outdoor Navigation Dataset](https://zenodo.org/records/14043015): 1,251 images at 640x640, YOLO segmentation, classes Crosswalk, Pedestrian, Ramp, Road, Sidewalk; CC BY 4.0. Small but includes Ramp, which is unusual.
- [A Dashcam Road Segmentation Dataset for Zero-Shot SAM Evaluation](https://zenodo.org/records/22301900): road surface only, 151 manual labels. Not crosswalk.

### Inferences
- CDNet is the only downloadable dedicated detector that merits running. Expect weak transfer to panoramas: it is a YOLOv5-era model trained on 3,080 vehicle-mounted camera images. The geographic spread of those images was not established here.

### Gaps
- CDNet weight download links and terms were not exercised. Whether the X-CDNet repo has weights was not checked.

## Pretrained semantic and panoptic models with a crosswalk class

### Takeaway
Vistas is the only widely trained label set with explicit crosswalk classes, and the same two classes carry over to every Vistas checkpoint. **Published per-class crosswalk IoU is nearly absent.** The only verified number found is SegFormer-B0: `Lane Marking - Crosswalk` 63.80, `Crosswalk - Plain` 21.09.

### Cited Findings
- The Mask2Former Vistas checkpoint has 65 classes; crosswalk-relevant ones are `Crosswalk - Plain` (id 8), `Lane Marking - Crosswalk` (id 23) and `Lane Marking - General` (id 24) — [config.json](https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-semantic/raw/main/config.json).
- **SegFormer-B0 on Vistas validation at 1024x2048: `Crosswalk - Plain` 21.09, `Lane Marking - Crosswalk` 63.80, Road 85.20, Curb 53.20, overall mIoU 41.83** — [Flying Guide Dog, arXiv 2108.07007, Table I](https://arxiv.org/abs/2108.07007). Read from the PDF.
- Mask2Former semantic on Vistas: R50 57.4 mIoU (59.0 ms+flip), Swin-L 63.2 (64.7 ms+flip); overall only — [MODEL_ZOO](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md).
- OneFormer publishes Vistas checkpoints for Swin-L, ConvNeXt-L and DiNAT-L, with rows such as 46.7 / 62.9 / 64.1 for Swin-L. Weights are hosted on shi-labs.com, not HF. Overall metrics only — [OneFormer README](https://github.com/SHI-Labs/OneFormer). The three columns are probably PQ, mIoU and ms+flip mIoU, but the header row was not read.
- InternImage publishes a Mapillary UperNet InternImage-L 512x1024 checkpoint on HF, plus Mapillary-pretrained Cityscapes models, which lose the crosswalk classes because Cityscapes has none — [segmentation README](https://github.com/OpenGVLab/InternImage/blob/master/segmentation/README.md).
- MMSegmentation includes Vistas v1 (65 class) and v2 dataset configs and a DeepLabV3+ R50 result of 47.35 mIoU at 1280x1280 — [configs](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/_base_/datasets), [DeepLabV3+ README](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/deeplabv3plus/README.md).
- An HF API name search found no SegFormer, OneFormer or InternImage repo with "vistas" or "mapillary" in the name. Other Vistas checkpoints on HF are [mask2former panoptic](https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-panoptic) (5,777 downloads) and [maskformer-resnet50-vistas](https://huggingface.co/facebook/maskformer-resnet50-vistas) (170).
- A snippet of an [MDPI paper on SAM for road objects](https://www.mdpi.com/2313-433X/11/6/189) shows a `lane-marking-crosswalk` row with mean IoU about 0.05 for each SAM backbone. The page could not be opened, and it is unclear from the snippet whether that row is the train or val table. Treat it as unconfirmed.
- Project Sidewalk's prior use: [RampNet#126](https://github.com/ProjectSidewalk/RampNet/issues/126) ran the same checkpoint on curb ramps (at 1024 parity recall 0.884, precision 0.383, F1 0.534; RampNet alone had F1 0.864). [RampNet#274](https://github.com/ProjectSidewalk/RampNet/issues/274) pins the union of classes 8 and 23 with connected components for the crosswalk probe.
- Objects365, a common training set for open-vocabulary detectors, has a "Crosswalk Sign" class. No plain "Crosswalk" class was seen in the [search results for its label file](https://huggingface.co/datasets/huggingface/label-files/blob/main/object365-id2label.json), but the full list was not checked.

### Inferences
- The `Crosswalk - Plain` 21.09 vs `Lane Marking - Crosswalk` 63.80 split matches our finding that Vistas models struggle with non-zebra forms. The observed confusion with `Lane Marking - General` is a plausible consequence of a three-way label boundary, though no paper tested that.
- Running OneFormer or InternImage Vistas is cheap, and would tell us whether the 0.821 recall@0.05 is a Mask2Former property or a Vistas-label property. Same labels imply the ceiling is likely similar.

### Gaps
- No per-class crosswalk IoU was found for Mask2Former, OneFormer or InternImage. The original [Vistas supplement](https://openaccess.thecvf.com/content_ICCV_2017/supplemental/Neuhold_The_Mapillary_Vistas_ICCV_2017_supplemental.pdf) and a [CVPRW 2023 supplement](https://openaccess.thecvf.com/content/CVPR2023W/SAIAD/supplemental/Zhou_Category_Differences_Matter_CVPRW_2023_supplemental.pdf) may have some, but CVF returned 403. Running the public checkpoints on Vistas val would produce the numbers directly.

## Open-vocabulary and foundation models with a "crosswalk" prompt

### Takeaway
**No published evaluation of any of these models specifically on crosswalks or road markings was found.** Molmo and Molmo2 are the only models confirmed here to have native point output; several others were not checked. Direct use on equirectangular input is poorly supported; perspective crops plus reprojection is the only defensible protocol.

### Cited Findings

Availability, from the HF API on 2026-11-08 unless noted:

| Model | Access / licence | Downloads | Output (verified from) |
|---|---|---|---|
| [facebook/sam3](https://huggingface.co/facebook/sam3) | Gated manual approval; licence "other" | 1,982,669 | [Text noun-phrase prompts](https://ai.meta.com/blog/segment-anything-model-3/), masks and IDs for all matches |
| [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base) | Open, Apache-2.0 | 1,563,882 | Boxes |
| Grounding DINO 1.5, 1.6 and DINO-X | [API token through DeepDataSpace](https://github.com/IDEA-Research/Grounding-DINO-1.5-API); [DINO-X also API](https://github.com/idea-research/dino-x-api). A [2026 guide](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/open-vocabulary-detection-grounding-dino-florence-2-rtdetr-rfdetr) says API-only. The repos' Apache-2.0 notices appear to cover wrapper code | n/a | Boxes; not downloadable as weights by this evidence |
| [google/owlv2-large-patch14-ensemble](https://huggingface.co/google/owlv2-large-patch14-ensemble) | Open, Apache-2.0 | 72,208 | Boxes |
| [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) | Open, MIT | 398,121 | Boxes and phrase grounding per card; no point task in the card |
| [google/paligemma2-3b-pt-896](https://huggingface.co/google/paligemma2-3b-pt-896), [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) | Gated, Gemma licence | 1,052; 364,325 | Not verified here |
| [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Open, Apache-2.0 | 6,046,339; 7,107,787 | Not verified here |
| [allenai/Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924), [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B) | Open, Apache-2.0 | 10,388; 43,632 | **Native points.** Molmo2 card parses `<points coords=...>` with coordinates scaled by 1000. [Molmo counts by pointing](https://arxiv.org/html/2409.17146v2) |
| [vikhyatk/moondream2](https://huggingface.co/vikhyatk/moondream2) | Open, Apache-2.0 | 2,306,622 | Not verified here |

Evaluations, none crosswalk-specific:

- YOLO-World was not evaluated on crosswalks. A 2026 roadside-infrastructure benchmark, [arXiv 2601.10551](https://arxiv.org/pdf/2601.10551), reports YOLO-World zero-shot at 22.9 mAP@50 versus 69.1 fine-tuned (snippet; class list not confirmed, and the abstract does not mention crosswalks).
- [WHU-Infra3D, arXiv 2606.09882](https://arxiv.org/pdf/2606.09882): fine-tuning open-vocabulary models bridges the cross-city gap (snippet).
- SAM 3 on aerial data: [arXiv 2607.09583](https://arxiv.org/html/2607.09583v1) reports text-only prompts at 11.67 mAP on DIOR and 11.85 on iSAID, with 0 mAP on some classes. No road-marking class was included.
- Older zero-shot SAM on pedestrian infrastructure: [arXiv 2310.09918](https://arxiv.org/abs/2310.09918). The abstract gives qualitative results only.
- Third-party usage: the [GPSO dataset](https://huggingface.co/datasets/mandresyandri/gpso-crosswalks-20cm-dataset) used SAM 3 with the prompt "pedestrian crossing" for pre-annotation on 20 cm aerial, with a human correcting afterward. That is usage, not an evaluation.
- Crosswalk use of VLMs: a [UMass GPT-4V crossing-risk study](https://www.umass.edu/robotics/sites/default/files/2024-11/ur.pdf) raised accuracy from 0.2545 to 0.4545 by adding segmentation masks. It is a risk task, not detection, and the numbers come from a snippet.

Equirectangular input:

- [Open Panoramic Segmentation (OOOPS, ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5584_ECCV_2024_paper.php) needed a deformable adapter and random equirectangular projection augmentation to handle distortion.
- [PanoSAMic, arXiv 2601.07447](https://arxiv.org/pdf/2601.07447v1.pdf) says SAM cannot fully handle panoramas and that supervised methods still beat open-vocabulary ones on them (snippet).
- A [2026 survey](https://arxiv.org/html/2606.27745v1) attributes VLM ERP weakness to near-total absence of ERP in pretraining; [SphMind](https://arxiv.org/html/2609.33462v2) reports Qwen2.5-VL 3B direction accuracy of only 9.5 to 17.5% on raw ERP (snippets).
- No result evaluated Grounding DINO or OWLv2 on 360 imagery.

### Inferences
- A crosswalk spans a wide horizontal angle at ground level and sits in the most distorted lower band of the ERP image. Direct 4096x2048 input also hits model input-size limits, so a fair baseline must project to perspective views and map results back to the panorama. This is engineering judgment, not a measured finding.
- None of these models output the centre of a marked crossing natively. Box or mask centres are proxies, and Molmo's points are model-chosen, so scoring them against our centre-point GT needs a tolerance in angular or ground distance.
- Recommended baseline shortlist, in order of information per unit of effort: (1) OneFormer or InternImage Vistas to separate model effect from label effect; (2) SAM 3 text prompt on perspective views, because it is the current zero-shot standard and has masks; (3) Grounding DINO Base and OWLv2, open and cheap; (4) Molmo2 for native points; (5) a YOLO or CDNet trained on CDSet plus RLMD, as the "what public data gets you" baseline. Skip DINO-X unless a paid API result is wanted for completeness.

### Gaps
- Point output for Qwen, Gemma, PaliGemma, Moondream and frontier VLMs was not confirmed from primary documentation. Gemini's coordinate convention was not found by the searches run.
- No frontier VLM accuracy on crosswalk localization was found.

## Published head-to-heads and benchmark fragmentation

### Takeaway
**No street-level crosswalk detector head-to-head on a shared independent test set was found.** Each paper reports on its own dataset or split, with its own metric. The closest things to shared benchmarks are the nuScenes pedestrian-crossing AP (bird's-eye-view vectorized maps) and Vistas per-class IoU (rarely reported).

### Cited Findings

| Source | Test data | Viewpoint | Metric and value | Compared against |
|---|---|---|---|---|
| [CDNet](https://link.springer.com/article/10.1007/s00521-022-07007-9) (2022) | Own CDSet | Dashcam | Average F1 94.83% | Own YOLOv5 baseline; the [repo](https://github.com/zhangzhengde0225/CDNet) claims it exceeds YOLOv5 |
| [X-CDNet](https://www.bohrium.com/paper-details/x-cdnet-a-real-time-crosswalk-detector-based-on-yolox/1011454158751924229-2631) (2024) | Own CD9K, partly from BDD100K | Dashcam | AP50 93.3 | YOLOX (no CDNet-vs-X-CDNet F1 found) |
| [FPVCrosswalk paper line](https://jiki.cs.ui.ac.id/index.php/jiki/article/view/1757) | FPVCrosswalk | First-person | mIoU 0.946, Dice 0.972 | Unknown |
| [ROD](https://arxiv.org/html/2611.00681) (2026) | Own 189 test crosswalks | Pedestrian | **Recall only**, conf 0.25 IoU 0.7: YOLOv8n 0.783, v9t 0.809, v10n 0.794, 11n 0.815, 12n 0.815, 26n 0.799 | YOLO nano family only |
| [CrosswalkNet](https://arxiv.org/pdf/2506.07885) (2025) | Own | Aerial | Not extracted | Not extracted |
| [Multi-modal zebra crossing framework](https://link.springer.com/article/10.1007/s12559-026-10547-0) (2026) | Own | Assistive | 99.76% F1 (snippet) | Different setup |
| [SegFormer-B0 on Vistas](https://arxiv.org/abs/2108.07007) (2021) | Vistas val | Street | IoU 63.80 / 21.09 | None for this class |
| [MapTR → MapTRv2](https://arxiv.org/abs/2308.05736) on nuScenes val | Shared | BEV vectorized, multi-camera | **Pedestrian-crossing AP 46.3 (MapTR baseline, R50) → 59.8 (MapTRv2 R50)**, 24 epochs, Table 3 | Dozens of later methods report this [AP_ped](https://arxiv.org/pdf/2403.08639), using [Chamfer thresholds 0.5, 1.0 and 1.5 m](https://arxiv.org/pdf/2312.03341) |

A search for "CDSet" comparisons in 2024-2025 papers found no independent re-use reporting a head-to-head.

### Inferences
- The fragmentation is real and documentable: **at least five different metrics (F1, AP50, IoU, recall at fixed confidence, Chamfer AP) across at least five viewpoints**, each on its creator's data. No two rows above can be compared.
- The PI's assumption should be refined rather than confirmed outright. There **is** a prevailing crosswalk benchmark, but it is the autonomous-driving HD-map one (nuScenes ped crossing), which measures BEV polygons from calibrated multi-camera rigs and does not apply to single-pano detection. In the street-level image space there is none.
- A reported F1 near 0.95 on a creator's dashcam set is not comparable to our audited 0.908 on Seattle, which is measured on full panoramas against an audited GT.

### Gaps
- This scan was not a systematic literature review; independent CDSet re-use may exist in papers not surfaced by search. Scholar citation tracing was not done.
- ScienceDirect and Springer full texts were not read, so CDNet and X-CDNet numbers come from abstracts or snippets.
