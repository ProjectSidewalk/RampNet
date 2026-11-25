# Model name candidates for the Project Sidewalk marked-crosswalk detector

Scan date 2026-11-08. This is a name-collision survey. "No collision found" always means "in the searches listed here"; it never means the name is free.

## How crowded is "CrossNet" / "Cross-Net", and is it usable?

### Takeaway
CrossNet is usable only in the narrowest sense: PyPI's `crossnet` is unregistered and Hugging Face has one zero-download match. It is a poor choice, because the name is heavily reused in ML, including two center-heatmap object detectors (the same paradigm as this model) and a transportation paper, and it is a consumer brand with a retail presence. I recommend treating it as an internal codename only.

### Cited Findings
**Registry and index counts (run 2026-11-08)**
- arXiv title-phrase search: 5 results, all literally CrossNet variants — [arXiv API](https://export.arxiv.org/api/query?search_query=ti:%22CrossNet%22).
- OpenAlex title search: 46 works, all with the literal string in the title — [OpenAlex](https://api.openalex.org/works?filter=title.search:%22CrossNet%22).
- GitHub repository-name search: 79 repos for "CrossNet". "Cross-Net" returns 870, but tokenization makes most of those noise (for example Cross Stage Partial Networks) — [GitHub](https://github.com/search?q=CrossNet+in%3Aname&type=repositories).
- PyPI: `crossnet` and `crossnets` both return 404 (not registered); the same check returned 200 for the controls `requests`, `crosswalk` and `xwalk` — [PyPI](https://pypi.org/project/crossnet/).
- Hugging Face models: one match on "CrossNet", `fabricioslv/biododeca-crossnet-3d`, 0 downloads; no datasets — [HF search](https://huggingface.co/models?search=CrossNet).

**Uses in ML/CV, roughly by prominence**
- Reference-based super-resolution: "CrossNet: An End-to-End Reference-Based Super Resolution Network Using Cross-Scale Warping", ECCV 2018, 253 citations in OpenAlex — [CVF](https://openaccess.thecvf.com/content_ECCV_2018/html/Haitian_Zheng_CrossNet_An_End-to-end_ECCV_2018_paper.html), [DOI](https://doi.org/10.1007/978-3-030-01231-1_6). Official code has 100 GitHub stars — [repo](https://github.com/htzheng/ECCV2018_CrossNet_RefSR). Its extension CrossNet++ (TPAMI 2020) has 41 citations — [DOI](https://doi.org/10.1109/tpami.2020.2997007).
- Recommender systems: the "Cross Network" of Deep & Cross Network is exposed as a class literally named `CrossNet` in DeepCTR — [DeepCTR docs](https://deepctr-doc.readthedocs.io/en/latest/deepctr.layers.interaction.html). TensorFlow Recommenders uses "Cross Network" and a `crossnet` variable in its official tutorial — [TF Recommenders](https://www.tensorflow.org/recommenders/examples/dcn). xDeepFM also calls it CrossNet — [arXiv](https://arxiv.org/pdf/1803.05170). The source paper is [arXiv 1708.05123](https://arxiv.org/abs/1708.05123). Derived name: XCrossNet (2021) — [aitopics](https://aitopics.org/tag/xcrossnet).
- **Same paradigm as this model (the worst collisions):** "CrossNet: Detecting Objects as Crosses" (IEEE TMM 2022) models each object as a center keypoint plus size and predicts a center map — [researchr](https://researchr.org/publication/LengLWHG22), [DOI](https://doi.org/10.1109/TMM.2021.3060278), 8 citations per [Crossref](https://api.crossref.org/works?query.title=CrossNet%20Detecting%20Objects%20as%20Crosses&rows=2). "CrossNet: Computing-Friendly Lightweight Anchor-Free Detector" (2021) is a second, CenterNet-based detector with the name — [AMiner](https://aminer.org/pub/61c9e9155244ab9dcbb81774/crossnet-computing-friendly-lightweight-anchor-free-detector).
- Transportation: "CROSS-Net: Region-Agnostic Taxi-Demand Prediction Using Feature Disentanglement", IEEE T-ITS — [arXiv](https://arxiv.org/abs/2310.18215), [DOI](https://doi.org/10.1109/tits.2026.3683763).
- Speech separation: CrossNet (2024) — [arXiv 2403.03411](https://arxiv.org/abs/2403.03411); follow-ups [X-CrossNet](http://arxiv.org/abs/2411.13811) and [AV-CrossNet](http://arxiv.org/abs/2406.11619).
- Image translation: "CrossNet: Latent Cross-Consistency for Unpaired Image Translation" — [arXiv 1901.04530](http://arxiv.org/abs/1901.04530).
- Cross-view learning code: `viibridges/crossnet`, 59 stars — [GitHub](https://github.com/viibridges/crossnet).
- Architecture: "CrossNets: A New Approach to Complex Learning" (DAG networks) — [DeepAI](https://deepai.org/publication/crossnets-a-new-approach-to-complex-learning).
- Neuromorphic hardware: "CrossNets" (2003, 120 citations; 2011, 114 citations) — [DOI](https://doi.org/10.1196/annals.1292.010), [DOI](https://doi.org/10.1166/sam.2011.1177).
- Cross-Net in medical imaging: "Convolutional descriptors aggregation via cross-net for skin lesion recognition" (2020, 50 citations) — [DOI](https://doi.org/10.1016/j.asoc.2020.106281).

**Non-ML uses**
- The top GitHub hit by stars is `dr0op/CrossNet-Beta` (367 stars), described in Chinese as a red-team tool for generating phishing executables with antivirus evasion — [GitHub](https://github.com/dr0op/CrossNet-Beta).
- Crossnet is a US four-way volleyball game and company, founded 2017, sold at Walmart, Target and Dick's and shown on national TV — [Wikipedia](https://en.wikipedia.org/wiki/Crossnet).
- A third-party aggregator lists 6 US trademark filings for CROSSNET LLC, mostly Class 28 (toys and sporting goods), including CROSSNET itself — [trademarkdir](https://trademarkdir.com/owner/crossnet-llc). That is a secondary source, and I did not read USPTO records directly.
- A search for a transportation company or pedestrian-crossing product called CrossNet found none; results were other crossing products such as Pelco IntelliCross — [Pelco PDF](https://pelcoreport.pelcoinc.com/wp-content/uploads/2016/07/072915_IntelliCross_PR.pdf).

### Inferences
- "Cross" alone names no object. RampNet names the thing it finds, while CrossNet could be read as cross-view, cross-domain or cross-city. That is a real ambiguity in this project's own writing, which is likely to discuss cross-city generalization.
- Two prior detectors with the center-heatmap design make CrossNet the most awkward possible choice for reviewers and for search.
- The volleyball brand is mainly a search-noise problem.

### Gaps
- I did not confirm a "CrossNet" few-shot paper specifically; the brief mentions one, and these searches did not surface it.
- An unverified wiki mirror says Crossnet was acquired in 2025; I did not confirm it and do not rely on it.

## Which candidates collide across arXiv, OpenAlex (Google Scholar proxy), Hugging Face, GitHub, PyPI and the web?

### Takeaway
Most short descriptive names are taken (CrosswalkNet directly so, by a 2025 crosswalk-detection paper). XwalkNet, PedXingNet, XMarkNet, CrossStripeNet, CrossLineNet, CrosswalkPoint and similar coined names returned no collision in the searches run.

### Cited Findings
**Method.** One script ran on 2026-11-08 for each name: arXiv title phrase ([template](https://export.arxiv.org/api/query?search_query=ti:%22XwalkNet%22)), OpenAlex title search ([template](https://api.openalex.org/works?filter=title.search:%22XwalkNet%22)), GitHub repo-name search ([template](https://github.com/search?q=XwalkNet+in%3Aname&type=repositories)), Hugging Face model and dataset search ([template](https://huggingface.co/models?search=XwalkNet)) and PyPI JSON lookup ([template](https://pypi.org/project/xwalknet/)). Web search was added for names where noted. Network failures were reported as errors, and none occurred.

**Counts.** "OA lit." is OpenAlex works whose title literally contains the name (capped at the top 200 by citations). "HF" is models/datasets.

| Name | arXiv title | OA lit. | GitHub names | HF | PyPI | Status |
|---|---|---|---|---|---|---|
| CrossNet | 5 | 46 | 79 | 1/0 | not registered | crowded |
| Cross-Net | 4 | 7 | 870 (noisy) | 3/18 (noisy) | not registered | crowded |
| CrosswalkNet | 1 | 1 | 1 | 0/0 | not registered | **direct collision** |
| ZebraNet | 1 | 9 | 2 | 0/0 | not registered | taken (famous) |
| CrossingNet | 0 | 0 | 5 | 0/0 | not registered | near-collision |
| PedXNet | 0 | 0 | 1 | 0/0 | not registered | taken |
| CrossPoint | 8 | 200+ | 384 | 0/2 | **registered** | crowded |
| WalkNet | 0 | 9 | 10 | 0/0 | not registered | taken |
| StripeNet | 0 | 0 | 7 | 2/0 | not registered | taken nearby |
| LadderNet | 1 | 9 | 13 | 0/0 | not registered | taken |
| PaintNet | 1 | 2 | 51 | 0/0 | not registered | taken |
| MarkNet | 0 | 2 | 35 | 0/0 | not registered | noisy |
| CrossRamp | 0 | 1 | 0 | 0/0 | not registered | off-meaning |
| XingNet | 0 | 0 | 2 | 0/0 | not registered | noisy word |
| ZebraCross | 0 | 2 | 25 | 1/0 | not registered | generic |
| ZebraPoint | 0 | 0 | 1 | 0/0 | not registered | one tiny repo |
| StripePoint | 0 | 1 | 0 | 0/0 | not registered | one paper |
| PedXing | 2 (0 literal) | 0 | 2 | 0/0 | not registered | two tiny repos |
| XWNet | 0 | 0 | 4 (unrelated) | 0/0 | not registered | noisy |
| XwalkNet, PedXingNet, XMarkNet, CrosswalkPoint, XwalkPoint, CrossStripeNet, CrossLineNet, MarkedWalkNet, PedCrossNet, CrosswayNet, RungNet | 0 | 0 | 0 | 0/0 | not registered | no collision found in the searches run (API scan plus web search) |
| RampNet-Crosswalk, RampNet-X, RampXNet, CrosswalkRampNet, ZebraRamp, ZebraWalkNet, XwalkDet, MarkedCrossNet, XingPoint, WalkMarkNet, WalkStripeNet, CrossPaintNet, CrossWalkFinder, CrossBarNet, CrossMarkNet | 0 | 0 | 0 | 0/0 | not registered | no collision found in the API scan only |

**Collision evidence**
- **CrosswalkNet:** a YOLOv8 oriented-box framework for pedestrian crosswalk detection in aerial images, UMass, June 2025 — [arXiv 2506.07885](https://arxiv.org/abs/2506.07885), with a GitHub repo showing no released weights or license — [z00bean/CrosswalkNet](https://github.com/z00bean/CrosswalkNet). Same task, different viewpoint.
- **ZebraNet:** Princeton's wildlife-tracking sensor network, ASPLOS 2002, ACM SIGMOBILE Test of Time award 2017 — [Princeton](https://www.cs.princeton.edu/news/martonosi-group-paper-receives-acm-sigmobile-test-time-paper-award). "Hardware design experiences in ZebraNet" has 533 citations — [DOI](https://doi.org/10.1145/1031495.1031522). Nothing connects it to crossing detection in the searches run.
- **CrossingNet:** "Crossing Nets", CVPR 2017, for hand pose — [CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Wan_Crossing_Nets_Combining_CVPR_2017_paper.html); code is literally `melonwan/crossingNet` — [GitHub](https://github.com/melonwan/crossingNet).
- **PedXNet:** PedXnets, pediatric radiograph transfer learning, Scientific Reports 2024 — [Nature](https://www.nature.com/articles/s41598-024-58163-y), code at [babbu3682/PedXnet](https://github.com/babbu3682/PedXnet). Pedestrian-side neighbor: the PedX dataset — [CMU](https://ri.cmu.edu/publications/pedx-benchmark-dataset-for-metric-3d-pose-estimation-of-pedestrians-in-complex-urban-intersections).
- **CrossPoint:** CVPR 2022 3D contrastive learning, 327 citations — [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Afham_CrossPoint_Self-Supervised_Cross-Modal_Contrastive_Learning_for_3D_Point_Cloud_Understanding_CVPR_2022_paper.html). PyPI `crosspoint` is an API client — [PyPI](https://pypi.org/project/crosspoint/). A popular e-reader firmware uses the name (8,698 stars) — [GitHub](https://github.com/crosspoint-reader/crosspoint-reader). Crosspoint is also standard memory and switch terminology — [OpenAlex](https://api.openalex.org/works?filter=title.search:%22CrossPoint%22).
- **WalkNet:** Walknet, the hexapod walking controller (1998, 271 citations) — [DOI](https://doi.org/10.1016/s0893-6080(98)00067-7); also "WalkNet: A Neural-Network-Based Interactive Walking Controller" (2017) — [DOI](https://doi.org/10.1007/978-3-319-67401-8_2). A small inclusive-mobility app uses the name too — [BrandonCabra/WalkNet](https://github.com/BrandonCabra/WalkNet).
- **StripeNet:** a City of Boston Office of Emerging Technology pair of models, with empty model cards — [OETBoston/smart-StripeNet-mask2former](https://huggingface.co/OETBoston/smart-StripeNet-mask2former). Boston's related curb digitization project uses 360 street-level imagery — [boston.gov](https://www.boston.gov/smartgrant). A StripeNet student network also appears in stripe-like space-target detection — [arXiv 2407.18097](https://arxiv.org/pdf/2407.18097).
- **LadderNet:** U-Net chain for medical segmentation, 307 citations — [arXiv 1810.07810](https://arxiv.org/abs/1810.07810), [GitHub](https://github.com/juntang-zhuang/LadderNet) (145 stars). Other LadderNet titles cover 360 video and Arctic sea ice — [OpenAlex](https://api.openalex.org/works?filter=title.search:%22LadderNet%22).
- **PaintNet:** robotic spray-painting dataset and model, IROS 2023 — [arXiv 2211.06930](https://arxiv.org/abs/2211.06930); a separate 2020 fashion-generation PaintNet — [DOI](https://doi.org/10.1007/s11042-020-09009-y); GitHub noise from Paint.NET — [GitHub](https://github.com/search?q=PaintNet+in%3Aname&type=repositories).
- **MarkNet:** a marketing-information facility in Indian agriculture papers — [OpenAlex](https://api.openalex.org/works?filter=title.search:%22MarkNet%22); 35 small GitHub repos — [GitHub](https://github.com/search?q=MarkNet+in%3Aname&type=repositories). A web search found no neural-network paper with the name.
- **CrossRamp:** appears as an elliptical-trainer term — [DOI](https://doi.org/10.25710/5aa9-ak50).
- **Xing / XingNet:** XING is a large German professional network — [Wikipedia](https://en.wikipedia.org/wiki/XING). XingNet hits are two tiny unrelated repos, one a coffee shop — [GitHub](https://github.com/search?q=XingNet+in%3Aname&type=repositories).
- **ZebraCross:** generic in GitHub detection repos — [GitHub](https://github.com/search?q=ZebraCross+in%3Aname&type=repositories); "zebracross" is an Indonesian-language word in paper titles — [DOI](https://doi.org/10.57235/jerumi.v3i2.7478).
- **ZebraPoint:** one 1-star repo — [smario-7/zebrapoint](https://github.com/smario-7/zebrapoint). Neighbor zebra-named work: Zebra-CoT — [arXiv 2507.16746](https://arxiv.org/abs/2507.16746). ZebraRecognizer is a crossing detector, so "Zebra" prefixes already exist in this niche — [ICPR paper](https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICPR-2014/data/5209c566.pdf).
- **StripePoint:** "StripePoint-YOLO" for weld-seam keypoints, 2026 — [DOI](https://doi.org/10.3390/s26154980).
- **PedCrossNet:** no exact hit, but PedCross is a pedestrian crossing-intention system — [NDHU seminar](https://www.csie.ndhu.edu.tw/news-zh/113-2-專題演講－114-04-11-國立中央大學資訊工程學系-孫敏德/), and Ped-Cross GAN generates crossing scenarios — [Coventry](https://pureportal.coventry.ac.uk/en/publications/generation-of-pedestrian-crossing-scenarios-using-ped-cross-gener/).
- **"Xwalk" (the root, not XwalkNet):** Intel's discontinued Crosswalk web runtime used `xwalk` as its package namespace — [Wikipedia](https://en.wikipedia.org/wiki/Crosswalk_Project), [APKMirror](https://www.apkmirror.com/apk/crosswalk-project/crosswalk-project-runtime/). Xwalk is a protein cross-linking tool — [UF RC docs](https://docs.rc.ufl.edu/software/apps/xwalk). PyPI `xwalk` is a yanked protein-identifier package — [PyPI](https://pypi.org/project/xwalk/). Hugging Face has a user namespace `xwalk` — [HF](https://huggingface.co/xwalk). XWalk is also a crypto travel platform — [white book](https://xwalk.gitbook.io/xwalk-white-book/technical-architecture/5.1-key-technical-features).
- **XMark (the root, not XMarkNet):** the VLDB 2002 XML benchmark — [CWI](https://projects.cwi.nl/xmark/index.html), and a PC benchmark — [Neowin](https://www.neowin.net/news/dacris-software-releases-xmark-7/).
- **CrossLineNet neighbor:** CrossDet, "Crossline Representation for Object Detection", ICCV 2021 — [CVF PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Qiu_CrossDet_Crossline_Representation_for_Object_Detection_ICCV_2021_paper.pdf).
- **CrosswalkPoint neighbor:** "crosswalk points" is a descriptive phrase in a lidar patent — [US 12272088](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12272088).
- **RungNet:** the web search returned only RUNNet, the Russian academic network, treated as the likely intended term — [Wikipedia](https://en.wikipedia.org/wiki/RUNNet).

**Web searches that returned no exact match:** XwalkNet (two queries), XingNet and ZebraPoint and XwalkPoint and CrosswalkPoint, PedXingNet and CrosswayNet and PedCrossNet, CrossStripeNet and CrossLineNet and MarkedWalkNet, and XMarkNet. A search for a pedestrian-crossing company or product named XWalk found none.

### Inferences
- The zero rows are coined compounds, so low counts are expected. The real test is whether a root word has a strong meaning, which is why the Xwalk and XMark neighbors above matter.
- Boston's StripeNet is probably a street-marking model, given the street-imagery project, but its model card is empty, so that remains a guess. Even so, it makes StripeNet a poor pick for a street-marking model.
- Project Sidewalk's own Hugging Face org already has `rampnet-model` plus `sidewalk-validator-ai-crosswalk` and `sidewalk-tagger-ai-crosswalk` — [HF API](https://huggingface.co/api/models?author=projectsidewalk&limit=50). A name like `xwalknet-model` would fit the first pattern and stay distinct from the validator and tagger repos.

### Gaps
- **Google Scholar could not be read:** the request stopped at a session-verification page. OpenAlex is the proxy, and it can miss names used only in abstracts or very recent preprints.
- The arXiv and OpenAlex checks are title-only, GitHub is repo-name-only, and Hugging Face matches repo IDs rather than card text.
- The 15 names marked "API scan only" had no general web search.
- DNS lookups for xwalknet.com, .org and .ai, and pedxingnet.com, returned no address, while crossnet.com and stripenet.com did. No address does not mean a domain is unregistered, so this is weak evidence.
- I recalled Crossref's Crossmark service as a neighbor for CrossMarkNet, but my one search did not confirm it. Check it before using that name.

## How well does each name fit marked, non-zebra and possibly unmarked crossings, in the US and globally?

### Takeaway
"Zebra" is the wrong anchor for a US-trained model, since the most frequent US style is two parallel lines and "zebra" means different things in different places. "Crosswalk" and "crossing" fit every pattern, and "crossing" is the only family that clearly survives a later move to unmarked crossings.

### Cited Findings
- In the US and Canada, two parallel lines are used "most frequently", with continental and ladder styles as higher-visibility options — [Wikipedia](https://en.wikipedia.org/wiki/Crosswalks_in_North_America).
- OpenStreetMap separates `lines`, `zebra`, `ladder`, `ladder:skewed`, `dashes` and others. It notes that zebra is called "continental" in the US and that `ladder:skewed` is sometimes called "zebra" in the US — [OSM](https://wiki.openstreetmap.org/wiki/Key:crossing:markings).
- OSM also has `crossing:markings=no`, implied by `crossing=unmarked`, so "unmarked crossing" is an established category — [OSM](https://wiki.openstreetmap.org/wiki/Key:crossing:markings).
- Germany uses dashes at nearly all signalized crossings — [OSM](https://wiki.openstreetmap.org/wiki/Key:crossing:markings).
- "Zebra crossing" is the British term and "crosswalk" the American one, and "pedestrian crossing" is the general term; these are low-authority glossary sources — [differentwordfor](https://differentwordfor.com/pedestrian-crossing), [ProZ](https://proz.com/kudoz/english-to-arabic/transport-transportation-shipping/3166855-zebra-crossings.html), [Wikipedia](https://en.wikipedia.org/wiki/Zebra_crossing).
- US usage treats a crosswalk as a marked or unmarked area — [Wikipedia](https://en.wikipedia.org/wiki/Crosswalks_in_North_America). PED XING signs are used near marked or unmarked crosswalks — [SlashGear](https://www.slashgear.com/2283888/what-does-ped-xing-mean-road-sign/).
- PED XING is a US warning-sign legend — [Traffic Safety Warehouse](https://www.trafficsafetywarehouse.com/Ped-Xing-Sign/productinfo/W11-2P).

### Inferences
| Family | Marked, US | Marked, global | Unmarked later | Note |
|---|---|---|---|---|
| Xwalk / Crosswalk | good | fair (US word, widely understood) | fair | US "crosswalk" can include unmarked |
| PedXing / Crossing | good | fair | **best** | "crossing" is the broadest concept |
| XMark / Marked / CrossMark | good | good | **poor** | "mark" excludes unmarked |
| Stripe | partial (not parallel lines) | good | poor | suits zebra, continental, ladder |
| Line | good for parallel and ladder | partial | poor | weak for pure bar patterns |
| Ladder / Rung | partial | partial | poor | names one US pattern |
| Zebra | **poor** | good | poor | helps non-US readers, misdescribes the common US style |
| Ped- prefixes | good | good | good | reads as pedestrian behavior prediction |

- Stripe-based names don't travel well, given dashes in Germany and two-line crossings in the US.
- Zebra helps with international recognition and hurts accuracy for a model trained partly on US inventories. If non-US reach becomes the priority, ZebraPoint is the best of that family.

### Gaps
- I did not read the current MUTCD text (Section 3B.18) directly, and US style frequency comes from Wikipedia.
- No authoritative global terminology source was found; the term split rests on general and glossary sources.

## How do the names rate on pronounceability, memorability, search distinctiveness and pairing with RampNet?

### Takeaway
Only single-compound "Noun+Net" names read as true RampNet siblings. XwalkNet is the strongest of those, with one known weakness: some readers will say "ex-walk".

### Cited Findings
- RampNet is a two-stage curb-ramp pipeline presented at the ICCV 2025 CV4A11y workshop — [CVF](https://openaccess.thecvf.com/content/ICCV2025W/CV4A11y/html/OMeara_RampNet_A_Two-Stage_Pipeline_for_Bootstrapping_Curb_Ramp_Detection_in_ICCVW_2025_paper.html), [arXiv](https://arxiv.org/abs/2508.09415).
- "X" for "cross" has a road-sign precedent in PED XING — [SlashGear](https://www.slashgear.com/2283888/what-does-ped-xing-mean-road-sign/).

### Inferences
These are judgment calls, not measurements.
- **Sentence test:** "RampNet and XwalkNet locate curb ramps and crosswalks in street-level panoramas" reads cleanly. "RampNet and PedXingNet" is heavier. "RampNet and CrosswalkPoint" breaks the pattern.
- **Title test:** "XwalkNet: Bootstrapping Marked Crosswalk Detection from City Inventories" works, and follows RampNet's title form.
- **Pronunciation:** state it on first use, for example "XwalkNet (pronounced crosswalk-net)". PedXingNet has the same issue ("zing", or "shing" for Mandarin readers), and XingNet is worse.
- **Casing:** use XwalkNet, not XWalkNet, to match Ramp+Net's single capital; package IDs are lowercase regardless.
- **Search:** the exact compound XwalkNet returned nothing, so it should be easy to own. Plain "Xwalk" will still surface the Intel and protein tools.

### Gaps
- No reader or pronunciation testing was done. A quick poll of lab members would settle the "ex-walk" question.

## What is the ranked shortlist?

### Takeaway
Top pick: **XwalkNet**. Choose **PedXingNet** instead if unmarked crossings are a firm part of the roadmap. Avoid CrosswalkNet and CrossNet.

### Cited Findings
Collision evidence for each name is in the sections above. Every entry below is "no collision found in the searches run" unless a neighbor is stated.
1. **XwalkNet** — no collision found (API scan and two web searches). Neighbors share only the root "Xwalk": [Crosswalk Project](https://en.wikipedia.org/wiki/Crosswalk_Project), [Xwalk protein tool](https://docs.rc.ufl.edu/software/apps/xwalk).
2. **PedXingNet** — no collision found (API scan and web search). Root neighbors: [XING](https://en.wikipedia.org/wiki/XING), two tiny PedXing repos ([yxlin/pedxing](https://github.com/yxlin/pedxing)).
3. **XMarkNet** — no collision found (API scan and web search). Root neighbor: [XMark benchmark](https://projects.cwi.nl/xmark/index.html).
4. **CrosswalkPoint** — no collision found (API scan and web search). Generic phrase in [a lidar patent](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12272088).
5. **CrossStripeNet** — no collision found (API scan and web search).
6. **CrossLineNet** — no collision found (API scan and web search). Neighbor: [CrossDet](https://openaccess.thecvf.com/content/ICCV2021/papers/Qiu_CrossDet_Crossline_Representation_for_Object_Detection_ICCV_2021_paper.pdf).
7. **ZebraPoint** — one 1-star repo, [smario-7/zebrapoint](https://github.com/smario-7/zebrapoint); otherwise no collision found.
8. **MarkedWalkNet** — no collision found (API scan and web search).
9. **XingNet** — two tiny unrelated repos ([GitHub](https://github.com/search?q=XingNet+in%3Aname&type=repositories)); root neighbor [XING](https://en.wikipedia.org/wiki/XING).
10. **PedCrossNet** — no exact hit (API scan and web search); neighbors [PedCross](https://www.csie.ndhu.edu.tw/news-zh/113-2-專題演講－114-04-11-國立中央大學資訊工程學系-孫敏德/) and [Ped-Cross GAN](https://pureportal.coventry.ac.uk/en/publications/generation-of-pedestrian-crossing-scenarios-using-ped-cross-gener/).
11. **RampNet-Crosswalk** — no collision found (API scan only).

**Not recommended, with the deciding evidence:** [CrosswalkNet](https://arxiv.org/abs/2506.07885) (same task), CrossNet (see first section), [Crossing Nets](https://openaccess.thecvf.com/content_cvpr_2017/html/Wan_Crossing_Nets_Combining_CVPR_2017_paper.html), [ZebraNet](https://www.cs.princeton.edu/news/martonosi-group-paper-receives-acm-sigmobile-test-time-paper-award), [CrossPoint](https://openaccess.thecvf.com/content/CVPR2022/html/Afham_CrossPoint_Self-Supervised_Cross-Modal_Contrastive_Learning_for_3D_Point_Cloud_Understanding_CVPR_2022_paper.html), [PedXnet](https://www.nature.com/articles/s41598-024-58163-y), [Walknet](https://doi.org/10.1016/s0893-6080(98)00067-7), [StripeNet](https://huggingface.co/OETBoston/smart-StripeNet-mask2former), [LadderNet](https://arxiv.org/abs/1810.07810), [PaintNet](https://arxiv.org/abs/2211.06930), and [RungNet](https://en.wikipedia.org/wiki/RUNNet) (reads as RUNNet, ladder-only).

### Inferences
| # | Name | Sibling of RampNet | Fit | Main weakness |
|---|---|---|---|---|
| 1 | XwalkNet | strong | all marking patterns; unmarked plausible | "ex-walk" misreading; US word |
| 2 | PedXingNet | moderate | best for unmarked later | long; "zing" misreading; US sign idiom |
| 3 | XMarkNet | strong | marked only; hints "X marks the spot" for point detection | doesn't say crosswalk; needs explaining |
| 4 | CrosswalkPoint | weak | clear, names the point-detection paradigm | breaks the -Net pattern; descriptive phrase is hard to own |
| 5 | CrossStripeNet | moderate | misses parallel-line style | long |
| 6 | CrossLineNet | moderate | misses pure bar patterns | could read as a geometry term |
| 7 | ZebraPoint | weak | global appeal; misdescribes common US style | "zebra" mismatch |
| 8 | MarkedWalkNet | moderate | clear; excludes unmarked | clunky |
| 9 | XingNet | strong in form | broad | unclear pronunciation; "Xing" noise |
| 10 | PedCrossNet | moderate | broad | reads as crossing-intention prediction |
| 11 | RampNet-Crosswalk | literal | clear | puts "Ramp" on a crosswalk model and makes RampNet a de facto umbrella, pre-empting the deferred naming decision |

- RampNet-Crosswalk is ranked last because of that conflict with the project constraint, not because of collisions.

### Gaps
- This ranking weighs my own judgment on fit and sound, and the PI may weight those differently.
- Before committing, re-run the exact-name checks and read Google Scholar by hand.
