# Aerial crosswalk datasets, municipal inventories and held-out benchmark cities

Research date: 2026-11-08. "Live query" means I read the count, fields and value distribution directly from the layer's REST endpoint on that date. Catalog sweeps covered ArcGIS Hub (about 2,600 items over 11 queries), Socrata, opendata.swiss, data.gov.uk, data.gov.au, GovData.de, data.gouv.fr, data.europa.eu, Boston, San Jose, Montreal, Toronto and Madrid CKAN.

Portals I could not read this session (so absence there is not evidence of absence): catalog.data.gov CKAN path returned 404; data.gv.at CKAN 404; hri.fi (Helsinki) 403; data.overheid.nl and Barcelona failed TLS verification; data.gov.tw API 405; eoc-datasets.dlr.de did not resolve for direct fetch; mass.gov, london.gov.uk and tandfonline.com returned 403 to direct fetch; help.mapillary.com 403; Rostock's download host timed out.

## Overhead crosswalk datasets and benchmarks

### Takeaway
The PI's assumption mostly survives: I found no shared crosswalk benchmark with a leaderboard or wide reuse. SkyScapes is the only overhead benchmark found with a held-out test set and a crosswalk class, but it is 16 images of Munich. The other datasets are single-paper, downloader-only, login-gated, merged with sidewalks, or unreleased.

### Cited Findings

**Datasets with a crosswalk label**

| Dataset | Exact label | Imagery / GSD | N | Licence and access | Reported metric |
|---|---|---|---|---|---|
| Berriel et al. 2017 | Image-level binary crosswalk / no-crosswalk, labels from OSM `highway=crossing` | Google Static Maps, zoom 20, 200x225 px covering about 22x25 m | 245,768 images (74,047 positive); 3 continents, 9 countries, 20+ cities | No licence stated; repo ships a downloader that needs your own API key, not images | 97.11% global accuracy; 94.8% mean cross-city/country/continent; 96.1% cross-level. [Paper](https://arxiv.org/pdf/1706.09302), [repo](https://github.com/rodrigoberriel/satellite-crosswalk-classification) |
| SkyScapes (DLR, ICCV 2019) | `crosswalk (CW)` and separate `zebra zone (ZZ)`, within 12 lane-marking classes | Helicopter, about 13 cm; Munich | 16 images of 5616x3744 px, 5.7 km2; 50/12.5/37.5 split | [DLR terms](https://eoc-datasets.dlr.de/license-terms): non-commercial, no redistribution, registration; test labels withheld for an [online benchmark](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-skyscapes) | Best SkyScapes-Lane mean IoU 51.93% over 13 classes; per-class CW value not extracted. [Paper](https://arxiv.org/pdf/2007.06102) |
| Tile2Net (Hosseini et al., CEUS 2023) | `Crosswalk` segmentation class, labels from city planimetrics | Sub-metre aerial tiles; trained on Cambridge MA, DC, NYC | 20,000 tiles per [secondary summary](https://www.evl.uic.edu/pubs/2703) | Tool is open; imagery varies by city; licence not read | Crosswalk IoU 75.42%, precision 0.86, recall 0.86. [Paper](https://www.evl.uic.edu/documents/ceus_mappingthewalk_miranda.pdf) |
| CrosswalkNet (2025) | Oriented boxes; three types, one grouping zebra/continental/ladder | MassGIS 2019 and 2021, 15 cm | 23,000+ annotated crosswalks | No release found | 96.5% precision, 93.3% recall on Massachusetts; applied without tuning to NH, VA, ME. [arXiv](https://arxiv.org/abs/2506.07885) |
| Waterloo Urban Scene (2024) | Authors added 14 classes including crosswalks | 12 cm, 205.8 km2, Waterloo ON | Not extracted | Availability not checked | Per-class crosswalk metric not extracted. [arXiv](https://arxiv.org/pdf/2410.05717) |
| IEEE DataPort "Aerial Image (Roads & Sidewalks/Crosswalks)" | Two classes; sidewalks and crosswalks **merged** | USGS orthos, zoom 20; DC, Cambridge | 3.93 GB | CC BY 4.0 but behind login | None on page. [DataPort](https://ieee-dataport.org/documents/aerial-image-roads-sidewalks-crosswalks) |
| Koester et al. 2016 | Zebra crossings | German aerial imagery | About 10 km of road, about 3,100 zebras | Not checked | Search snippet only; [PDF](https://cvhci.iar.kit.edu/~dkoester/publications/koesterICCHP2016zebrax.pdf) not opened |

**Related work without a reusable crosswalk dataset**
- Ahmetovic et al. describe a satellite plus Street View zebra pipeline; I found no dataset release. [SKI](https://www.ski.org/publication/mind-your-crossings-mining-gis-imagery-for-crosswalk-localization/)
- A Florida change-detection study labelled 1,272 crosswalk points in Leon County; I found no release statement. [arXiv](https://arxiv.org/pdf/2406.09731)
- A Hartford occlusion study compares aerial and street-view models on about 18,000 Bing Maps images; this was the only Bing use I saw, from a search snippet, not opened. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2046043024000364)
- An Amsterdam study filtered Tile2Net polygons with LiDAR and validated against **Project Sidewalk** labels: raw precision 0.343 to 0.758, filtered 1.0, recall 0.652 to 0.897, over 4 districts of 32 to 107 labels. [BNAIC 2024](https://bnaic2024.sites.uu.nl/wp-content/uploads/sites/986/2024/11/Where-Not-to-Cross-the-Street.pdf)
- Moran et al. used about three million satellite images across the 100 largest US cities for crossing **distance**, using OSM crossing edges, so it is not a marking dataset. [arXiv](https://arxiv.org/abs/2507.20497)

**Named in the brief, no crosswalk class confirmed**
- DOTA v1.0 to v2.0: no crosswalk or road-marking class in the 15 to 18 categories. [DOTA](https://captain-whu.github.io/DOTA/dataset.html)
- UAVid: 8 classes, Road is the only road class; CC BY-NC-SA 4.0. [UAVid](https://uavid.nl/)
- xView: 60 classes at 0.3 m, class list not on the abstract page. [arXiv](https://arxiv.org/abs/1802.07856)
- FAIR1M: 5 categories, 37 sub-categories, names not on the abstract page. [arXiv](https://arxiv.org/abs/2103.05569)
- OpenEarthMap: 8 land-cover classes at 0.25 to 0.5 m. [OEM](https://open-earth-map.org/overview.html)
- Lane-only sets: AEL has 7,763 images of lane polylines from Google Maps ([arXiv](https://arxiv.org/pdf/2312.13449)); AerialLanes18 is lane-marking segmentation ([arXiv](https://doi.org/10.48550/arXiv.1803.06904)).
- Ground-level, out of scope but easily confused: FPVCrosswalk2025, 3,000 synthetic plus 300 real first-person images ([Mendeley](https://data.mendeley.com/datasets/mcr2jwk5bp/1)); CD9K, 9,387 BDD100K frames ([X-CDNet](https://www.sciencedirect.com/science/article/abs/pii/S1047320324001627)).

### Inferences
- Berriel's GSD works out to about 0.11 m per pixel (22 m over 200 px).
- Berriel's Table I prints a total of 245,313, but its rows and text sum to 245,768. That is a typo in the table, not a second count.
- Overhead sources will undercount under canopy and shadow, so none should serve as ground truth for a street-level model without a street-level audit.
- SkyScapes' "zebra zone" is listed apart from "crosswalk" and may mean hatched road area. Do not map it to crosswalks without reading the class definitions.
- Why there may be no benchmark: labels come cheaply from OSM or city planimetrics, and each paper builds its own. This is my reading, not a sourced claim.

### Gaps
- "AerialLanes25" could not be confirmed under that name in two searches. SkyLane was not searched separately.
- iSAID, DeepGlobe, SpaceNet, LoveDA, MUDS and AeroScapes class lists were not confirmed this session. RoadTracer-adjacent sets were not checked.
- No per-class crosswalk IoU extracted for SkyScapes or Waterloo.
- Absence of a benchmark is a negative search result across arXiv, general web search and a targeted Hugging Face query. I did not enumerate Kaggle or Hugging Face directly.
- The "Li 2023" paper was not re-examined.

## Municipal and national inventories

### Takeaway
Usable inventories exist and are richer than expected. Los Angeles, Austin, Seattle, Bend, Paris, Calgary, Arlington VA and Portland are the strongest. Only some record pattern, and several US and Canadian portals have no citywide layer at all. NYC, San Francisco, Chicago, Boston, Toronto and Montreal had nothing citywide in the catalogs I could search.

### Cited Findings

**North America, schema confirmed by live query**

| City | Layer | Geometry, count | Pattern recorded? | Currency | Notes |
|---|---|---|---|---|---|
| **Los Angeles** | [LADOT Crosswalks](https://maps.lacity.org/lahub/rest/services/LADOT/MapServer/4) | Point, 22,507 | **Yes**: Continental 12,338, Standard 6,906, Ladder 2,988, Decorative 151, Pavers 72, Other 36, Scramble 16. Colour: White 16,266, Yellow 6,240 | `Date_` 2005-11-01 to 2026-10-27 | `Location` holds an angle, which suggests one point per leg. Hub item says modified 2020, which understates recency. Companion [Decorative layer](https://maps.lacity.org/lahub/rest/services/Street_Information/MapServer/1): 745 lines |
| **Seattle** | [Marked Crosswalks](https://services.arcgis.com/ZOyb2t4B0UYuYNYH/arcgis/rest/services/Marked_Crosswalks_CDL/FeatureServer/0) | Point, 6,692 | **Yes**: LADER 5,895, PARSL 724, BRICK 42, CUSTOM 26, OTHER 5 | Edited 2026-11-06 | **Filter status**: INSVC 6,342, PLANNED 348 |
| **Austin** | [Short Line markings](https://services.arcgis.com/0L95CJ0VTaxqcmED/arcgis/rest/services/Markings_Short_Line_Read_Only/FeatureServer/0), also [Socrata](https://datahub.austintexas.gov/resource/9hak-anfp.json) | Line, 7,328 crosswalks of 16,674 | **Yes**: Continental 6,831, Transverse 465 | Edited 2026-11-08; work dates to 2026-11-03 | 7,289 ACTIVE. Assets "referenced using NearMap imagery and work orders" |
| **Bend** | [Crosswalks](https://services5.arcgis.com/JisFYcK2mIVg9ueP/arcgis/rest/services/Crosswalks/FeatureServer/7) | Line, 6,655 | Marked/unmarked only: **marked 1,117, unmarked 5,504**, bicycle 31, closed 3 | Edited 2026-11-08 | Explicit negatives. Defined as connections between ADA ramps, city-managed only ([portal](https://data.bendoregon.gov/datasets/crosswalks)). LifeCycleStatus includes 199 PROP |
| **DC** | [DDOT Pavement Marking](https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Transportation_Traffic_Calming_WebMercator/MapServer/96) | Point, 16,117 crosswalks of 60,094 | **Yes**: Standard 11,674, "Longitudal" 4,302, Diagonal 141 | **Unknown**: CREATED and EDITED blank, Hub item 2023-03 | Collected only within 10 m of curb faces at intersections, so midblock crossings are likely missing |
| **Arlington VA** | [Pavement Markings](https://arlgis.arlingtonva.us/arcgis/rest/services/Open_Data/od_Pavement_Markings_Points/FeatureServer/0) | Point, 3,690 crosswalks of 12,980 | **Yes**: High Visibility 2,703, Standard 987 | Installed through 2026-10-28; synced 2026-11-08 | Cartegraph asset system |
| **Portland** | [PBOT Pavement Marking Lines](https://www.portlandmaps.com/od/rest/services/COP_OpenData_Transportation/MapServer/201), `LineType=3621` | Line, 7,094 | `LineStyle` exists, values not read | Not read | Code 3621 = crosswalk per [PedPDX layer](https://www.portlandmaps.com/od/rest/services/COP_OpenData_Transportation/MapServer/1400) (4,658 points including non-marked, April 2023) |
| **Cambridge MA** | [Pavement Markings](https://services1.arcgis.com/WnzC35krSYGuYov4/arcgis/rest/services/Pavement_Markings/FeatureServer/0) | Line, PM-CROSSWALK 17,828, raised 58 | No | Layer edited 2025-09-09 | Tile2Net training city, so not independent of that model |
| **Oakland** | [CrossWalks](https://services.arcgis.com/9tC74aDHuml0x5Yz/arcgis/rest/services/CrossWalks/FeatureServer/0) | Line, 5,734 | **Yes**: Continental 3,220, Basic 2,514; White 4,431, Yellow 1,303 | `Updated` to 2023-09-08, **aging** | Some remarks cite Google imagery as the source of updates |
| **Raleigh** | [Crosswalk inventory](https://services.arcgis.com/v400IkDOw1ad7Yad/arcgis/rest/services/Raleigh_Crosswalks/FeatureServer/0) | Point, 1,459 | Domain Standard, High Visibility, Ladder, Other | Edited 2026-11-06, inspected to 2026-02 | Small for a city its size |
| **Calgary** | [Crosswalks](https://data.calgary.ca/resource/hxgg-rpad.json) ([metadata](https://data.calgary.ca/api/views/hxgg-rpad.json)) | Point, 4,216 | **Yes**, but mixed: REGULAR 2,035, LADDER 867, STOP LINE 693, RIGHT TURN 436, others | Updated 2026-11-06 | Licence "See Terms of Use". About 2,925 records are actual crosswalks after dropping stop lines and similar |
| Tucson | [Crosswalks](https://gis.tucsonaz.gov/arcgis/rest/services/PublicMaps/OpenData_Transportation/MapServer/6) | Line, 709 | High visibility yes 121, no 588 | Hub 2025-09 | Self-described as all crosswalks |
| Kitchener ON | [Crosswalks](https://services1.arcgis.com/qAo1OsXi67t7XgmS/arcgis/rest/services/Crosswalks/FeatureServer/1) | Line, 4,720 | Surface only: ASPHALT (PAINTED) 1,813, ASPHALT 2,855 | Updated 2026-09 | "Painted" is a weak proxy |
| Nashville | [Crosswalks view](https://services2.arcgis.com/HdTo6HJqh92wn4D8/arcgis/rest/services/Crosswalks_2_view/FeatureServer/0) | Point, 1,823 | Longitudinal / Transverse | Audit 2023 | |

**Partial or not usable as a citywide inventory**
- NYC: only [Raised Crosswalks](https://data.cityofnewyork.us/api/views/uh2s-ftgh.json) (159) and [Enhanced Crossings](https://data.cityofnewyork.us/api/views/6ax4-q5k4.json) (197). A domain search for "crosswalk" returned no planimetric crosswalk layer.
- Toronto: [Pedestrian crossovers](https://services3.arcgis.com/b9WvedVPoizGfvfD/arcgis/rest/services/COTGEO_PEDESTRIAN_CROSSOVER/FeatureServer/0) (491, last edited 2021-11, **stale**, flashing-amber crossovers only). The Open Data [Pedestrian Network](https://open.toronto.ca) matched "crosswalk" but its fields were not read.
- Vancouver: [432 points](https://services3.arcgis.com/qrcTTRTwUoS8N47o/arcgis/rest/services/TrafficCrosswalk/FeatureServer/0) with only OBJECTID and Location; the count suggests a subset.
- Minneapolis: "Marked Crosswalk 2023" is [public but empty](https://services.arcgis.com/afSMGVsC7QlRK1kZ/arcgis/rest/services/Marked_Crosswalk/FeatureServer/0) (0 records).
- Charlotte's [layer](https://services.arcgis.com/9Nl857LBlQVyzq54/arcgis/rest/services/Pedestrian_Crossings/FeatureServer/0) is a 465-row project list, not an inventory.
- Ottawa [2019 crossovers](https://services.arcgis.com/G6F8XLCl5KtAlZ2G/arcgis/rest/services/2019_Pedestrian_Cross_over_Locations/FeatureServer) are **stale**. London ON [crossovers](https://maps.london.ca/server/rest/services/OpenData/OpenData_Transportation/MapServer/12): 307. Victoria BC [marked crosswalks](https://maps.victoria.ca/server/rest/services/OpenData/OpenData_Transportation/MapServer/18): 171.
- Boston: [Sidewalk Centerline](https://data.boston.gov/dataset/sidewalk-centerline) (PDDL) matched "crosswalk", fields not read. The [pathVu layer](https://services7.arcgis.com/lCps1TIE7mFpTJoN/arcgis/rest/services/boston_crosswalks/FeatureServer) is a vendor copy from 2017, **stale**.
- Socrata searches of San Francisco and Chicago portals for "crosswalk" returned 0. Hub searches for Bellevue, Denver, Philadelphia and San Jose returned no official layer in the top results.
- Montreal CKAN: no crosswalk inventory found ([portal](https://donnees.montreal.ca)).

**Europe**
- **Paris**, [Plan de voirie - Passages pietons](https://opendata.paris.fr/explore/dataset/plan-de-voirie-passages-pietons/): 22,236 **polygons**, median area 27.6 m2 (range 0.015 to 2,859), ODbL, processed 2026-10-10, publisher Direction de la Voirie. No pattern field. Dataset text is headed "Aires mixtes".
- **Hamburg**, [Fussgaengerueberwege WFS](https://geodienste.hamburg.de/wfs_fussgaengerueberwege?SERVICE=WFS&VERSION=1.1.0&REQUEST=GetFeature&typename=de.hh.up:fussgaengerueberwege&maxFeatures=2): points, sign VZ 293 zebras only, `quelle: CycloMedia`, `erfassungsdatum 2023-10-23` in the two sampled rows. dl-de/by-2-0 per the [EU portal listing](https://data.europa.eu). Total count not obtained.
- **Rostock** [listing](https://www.govdata.de/ckan/api/3/action/package_show?id=fussgangeruberwege): CSV, GeoJSON, KML; download timed out, schema unseen.
- **Canton Zurich** [Fussgaengerstreifen](https://ckan.opendata.swiss/api/3/action/package_show?id=fussgangerstreifen1): includes unmarked crossings on state parcels, with field-collected marking and signalisation detail; open terms; served via map viewer, schema unseen.
- Weak Swiss sources: [Lucerne](https://ckan.opendata.swiss/api/3/action/package_show?id=fussgangerstreifen) is itself deep-learning output from 2020 aerials; [Aargau](https://ckan.opendata.swiss/api/3/action/package_show?id=fussgangerstreifen-kantonsstrassen) was digitized from 2002/2004 orthophotos (**stale**); [St. Gallen](https://ckan.opendata.swiss/api/3/action/package_show?id=fussgangerstreifen-in-der-stadt-st-gallen) is frozen from a 2021 hackathon.
- **Brno**: [2,617 polylines](https://services6.arcgis.com/fUWVlHWZNxUvTUh8/arcgis/rest/services/prechody_pro_chodce/FeatureServer/0), updated 2026-11-06, no attributes.
- French metro layers exist for Brest, Rennes, Tours, Nantes, Roubaix and others via [data.gouv.fr](https://www.data.gouv.fr/); not schema-checked.
- MobiData BW's [Baden-Wuerttemberg crossings](https://www.govdata.de/ckan/api/3/action/package_show?id=gebundelte-daten-fussgangeruberwege-baden-wurttemberg) are re-served OSM, so they are not independent.
- London UK: a 2022 Assembly answer, seen only as a search snippet, says TfL does not provide open data on pedestrian crossing locations ([page](https://www.london.gov.uk/who-we-are/what-london-assembly-does/questions-mayor/find-an-answer/open-data-pedestrian-crossings), 403 to direct fetch). TfL's [Traffic Signals](https://gis-tfl.opendata.arcgis.com/datasets/TfL::traffic-signals/explore) layer covers signalised types, not zebras.
- Scattered UK council files: [data.gov.uk search](https://www.data.gov.uk/search?q=pedestrian).

**Asia-Pacific**
- Seoul: [OA-15554](https://data.seoul.go.kr/dataList/OA-15554/S/1/datasetView.do), monthly, updated 2026-11-01, EPSG:5186, KOGL Type 1; [OA-21209](https://data.seoul.go.kr/dataList/OA-21209/S/1/datasetView.do), arterials with crossing length, refreshed daily but described as current as of 2020. Field definitions not readable; no pattern evidence.
- Taipei: I found only [marked walkways (標線型人行道)](https://data.taipei/dataset/detail?id=d647b3af-4bbd-4e8f-b420-8f6c1bc1b597), which is not a crosswalk inventory. The Taipei inventory the team already knows was not re-found.
- Singapore: [Pedestrian Facilities](https://data.gov.sg/datasets/d_45175a1ff466b44414a9a1aeef564e6f/view) is annual counts only; LTA's [DataMall static list](https://datamall.lta.gov.sg/content/datamall/en/static-data.html) was not opened.
- Melbourne: [Pedestrian Network](https://data.gov.au/data/dataset/melbourne-pedestrian-network) is a route network, not crossings.
- Victoria state: [Pavement Marking Inventory](https://services-ap1.arcgis.com/vCUFK3DUHjGhHO6G/arcgis/rest/services/Victoria_Pavement_Marking_Inventory/FeatureServer/0), 104,197 points, edited 2026-10-01, but the sub-type field is uniform, so a crosswalk filter is unconfirmed.
- Canberra ([ACT Pedestrian Crossing Assets](https://data.gov.au/data/en/dataset/actgov-pedestrian-crossing-assets)) and Western Australia ([Main Roads, CC BY](https://data.gov.au/data/en/dataset/mrwa-pedestrian-crossings)) exist; schemas not checked.

### Inferences
- LA is the single best schema found: large, per-leg, patterned and dated.
- Inventories are maintenance records, so they are likely strongest for city-maintained paint and weakest for private, state-owned or recently removed markings. Seattle's and Austin's status fields show why filtering matters.
- Independence varies. Austin was digitized from aerial imagery, Hamburg from CycloMedia, and Oakland partly from Google imagery. All three remain independent of RampNet, but Hamburg and Oakland are street-level sources and may share recency effects.
- One feature does not always equal one crossing. Cambridge and Portland line layers may hold several lines per crossing, so counts are not comparable until a sample is dissolved.
- DC's "Longitudal" probably means continental bars. This is not documented.

### Gaps
- Stated completeness is almost never published. Only Tucson claims "all crosswalks".
- Amsterdam, Vienna, Helsinki and Barcelona are unresolved because portals failed. Sydney/TfNSW was not checked.
- Hamburg's total count, and Seoul's fields and counts, were not obtained.

## US federal and research sources, Overture, Mapillary

### Takeaway
I found no national US marked-crosswalk inventory. Overture is OSM-derived and adds no independent signal. Mapillary exposes crosswalk map features, but I found no published recall for them.

### Cited Findings
- Overture lists 2,664,824 `road / footway / crosswalk` features as of October 2026. Its primary source is OSM, enhanced with TomTom and local authoritative data, and the guide mentions no markings property. [Overture](https://docs.overturemaps.org/guides/transportation)
- Meta announced the end of Daylight on 2024-05-03 and redirected effort to Overture. [Daylight](https://daylightmap.org/2024/05/03/sunsetting-daylight.html)
- MassDOT funded an AI crosswalk inventory from aerials ([final report](https://www.mass.gov/doc/artificial-intelligence-framework-for-crosswalk-detection-across-massachusetts-final-report/download), 403 to direct fetch, known from search snippet). I found no public layer from it.
- State layers seen on Hub but not probed: [FDOT Mid-Block Crosswalks](https://services1.arcgis.com/O1JpcwDW8sjYuddV/arcgis/rest/services/Mid_Block_Crosswalks__View/FeatureServer), [VDOT Unsignalized Crosswalk](https://services.arcgis.com/p5v98VHDX9Atv3l7/arcgis/rest/services/Unsignalized_Crosswalk/FeatureServer), [Alaska DOT yearly layers](https://services.arcgis.com/r4A0V7UzH9fcLVvv/arcgis/rest/services/Crosswalk_2024/FeatureServer), [Ohio ADA Crosswalk Inventory](https://tims.dot.state.oh.us/ags/rest/services/MapViewers/ADA_Crosswalk_Inventory/MapServer/0/metadata).
- UW TCAT's OpenSidewalks schema reportedly includes `crossing:markings` ([mirror of schema repo](https://awesome.ecosyste.ms/projects/github.com%2FOpenSidewalks%2FOpenSidewalks-Schema)). TDEI demonstration regions are six WA, OR and MD counties ([data.gov](https://catalog.data.gov/dataset/its4us-demonstration-region-data-on-the-tdei)).
- Mapillary point classes include `marking--discrete--crosswalk-zebra` and `construction--flat--crosswalk-plain`. [Points list](https://www.mapillary.com/developer/api-documentation/points)
- Mapillary exposes features via z14 tiles (`mly_map_feature_point`) and Graph `/map_features`; bbox queries must be smaller than 0.01 degrees square; the website has a GeoJSON download. No precision or recall appears in the docs. [API docs](https://www.mapillary.com/developer/api-documentation)
- A 2022 Mapillary-based roadside detection study reports 87.57% precision and 72.60% recall, but it is not crosswalk-specific. [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9781587/)

### Inferences
- Mapillary features would be a competing detector, not ground truth. They could serve as a baseline next to OSM, since it is street-level AI and has no published crosswalk accuracy.
- Overture should not be counted as separate from the OSM baseline.

### Gaps
- FHWA MIRE crosswalk elements, NCHRP data products, Replica and Microsoft road-marking releases: my one combined search returned only Daylight material, so these are unresolved rather than negative.
- The Annotations for Pedestrian Environment dataset surfaced in a snippet (sidewalks, crossings, corner bulbs) and was not opened.

## Studies of OSM crossing completeness and `crossing:markings`

### Takeaway
Direct evidence is thin. I found one quantitative OSM-vs-truth crossing comparison, and no study measuring `crossing:markings` accuracy. The tag is widely used, but much of it records absence or generic presence.

### Cited Findings
- Leon County FL, 1,272 manually labelled crosswalks, 30 ft match: OSM (n=2,312) recall 77.8%, precision 52.2%, quality 39.8%. The authors' aerial model scored recall 85.9%, precision 88.7%. [Antwi et al., Table 1](https://arxiv.org/pdf/2406.09731)
- Berriel estimated OSM-derived satellite labels at 95.41% accurate, with 4.04% false positives and 4.83% false negatives, on 44,175 manually checked images. [Berriel](https://arxiv.org/pdf/1706.09302)
- NYU's Laefer: "If we had only used OpenStreetMap, we would have been left with a lot of crosswalks missing." No percentage given; about 49,000 crossings across Paris, San Francisco and Irvine. [NYU](https://engineering.nyu.edu/news/nyu-tandon-study-maps-pedestrian-crosswalks-across-entire-cities-helping-improve-road-safety)
- Moran et al. report that some OSM crossings exist only as single centre-of-road nodes; a search snippet says over half in some cities. [arXiv](https://arxiv.org/pdf/2507.20497)
- Taginfo, 2026-11-08: `highway=crossing` on 13,744,736 nodes; `crossing:markings` on 7,378,658 nodes and 4,307,703 ways. [Stats](https://taginfo.openstreetmap.org/api/4/key/stats?key=crossing:markings)
- `crossing:markings` values: no 39.25%, zebra 28.41%, yes 18.3%, lines 6.97%, ladder 2.83%. [Values](https://taginfo.openstreetmap.org/api/4/key/values?key=crossing:markings&page=1&rp=15&sortname=count_all&sortorder=desc)
- `crossing=` values: uncontrolled 5,334,076, unmarked 5,164,543, traffic_signals 2,587,311, marked 2,232,772, zebra 365,575. [Values](https://taginfo.openstreetmap.org/api/4/key/values?key=crossing&page=1&rp=12&sortname=count_all&sortorder=desc)
- Tag semantics differ from US terms: `zebra` is US "continental", `lines` is US "standard/transverse", `ladder` is separate. [OSM wiki](https://wiki.openstreetmap.org/wiki/Key:crossing:markings)
- OSM sidewalk accuracy findings do not carry over directly: Maryland OSM sidewalks were accurate but incomplete against Street View. [CDC 2026](https://www.cdc.gov/pcd/issues/2026/26_0255.htm)

### Inferences
- Leon County's low OSM precision may partly reflect unmarked crossings counted as false positives. The ground truth was visible crosswalks in aerials, so a tag-filtered OSM baseline could score higher. That is consistent with the team's own 0.846 to 0.898 F1 baseline in Seattle, which filters to marked or signalised nodes.
- Berriel's figure measures label noise in OSM-sampled images, not completeness over real crosswalks, and dates from 2017.
- Node and way counts should not be added, because one physical crossing can carry the tag on both.

### Gaps
- No published precision or recall for `crossing:markings` values was found in two targeted searches.
- A city-by-city OSM-vs-inventory comparison appears not to exist publicly, which makes it a cheap contribution here. The JAPA Moran paper returned 403 and was not read.

## Best 3-5 cities for a held-out cross-city benchmark

### Takeaway
Recommended core: **Los Angeles, Austin, Paris and Bend**, plus **Calgary or Portland** as the fifth. Keep Seattle as the development city and exclude it from the held-out set. Panorama coverage still needs a `--scan-only` check for each city before committing.

### Cited Findings

| City | Why | Paint and geography | Main risks |
|---|---|---|---|
| Los Angeles | Largest, per-leg, dated to 2026-10-27 ([layer](https://maps.lacity.org/lahub/rest/services/LADOT/MapServer/4)) | Continental, standard, ladder, decorative, pavers, scramble; white and yellow | Licence unread; old `Date_` values may be stale records |
| Austin | Line geometry, edited 2026-11-08 ([layer](https://services.arcgis.com/0L95CJ0VTaxqcmED/arcgis/rest/services/Markings_Short_Line_Read_Only/FeatureServer/0)) | 93% continental, some transverse | City-maintained assets only; aerial-digitized |
| Paris | Only strong non-US polygon set, ODbL, 2026-10-10 ([dataset](https://opendata.paris.fr/explore/dataset/plan-de-voirie-passages-pietons/)) | European zebra geometry, dense streets | No pattern field; "aires mixtes" scope needs reading; some polygon outliers |
| Bend | Explicit unmarked negatives: 1,117 marked, 5,504 unmarked ([layer](https://services5.arcgis.com/JisFYcK2mIVg9ueP/arcgis/rest/services/Crosswalks/FeatureServer/7)) | Small city, suburban | Small positive set; the project already uses Bend as a sandbox |
| Calgary (option) | Regular vs ladder, Canadian winter wear ([data](https://data.calgary.ca/resource/hxgg-rpad.json)) | Regular, ladder, multi-use | Must filter non-crosswalk rows; records can span several crosswalks |
| Portland (option) | 7,094 crosswalk lines ([layer](https://www.portlandmaps.com/od/rest/services/COP_OpenData_Transportation/MapServer/201)) | Pacific Northwest, close to Seattle | Feature-to-crossing ratio unknown; less diversity from Seattle |

Excluded, with reasons:
- DC has the right size and types, but no dates and intersection-only collection ([layer](https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Transportation_Traffic_Calming_WebMercator/MapServer/96)). Use it as a secondary city if recency can be established.
- Hamburg covers only legal zebras, not signalised crossings, and German Street View was [refreshed in 2023 for the first time in about a decade](https://gigazine.net/gsc_news/en/20230726-google-street-view-germany/), so capture dates need checking.
- Seoul has rich official data, but Street View in South Korea is [subject to security restrictions](https://thepeninsulaqatar.com/article/27/02/2026/south-korea-finally-agrees-to-share-data-with-google-maps) (search-snippet level, coverage not measured), and the pipeline has no Korean imagery source.
- Oakland's inventory stops in 2023 ([layer](https://services.arcgis.com/9tC74aDHuml0x5Yz/arcgis/rest/services/CrossWalks/FeatureServer/0)). Cambridge is a Tile2Net training city and is geographically close to prior work.

### Inferences
- The set spans inventory styles (points, lines, polygons), paint families (continental, standard, ladder, European zebra, decorative) and negatives (Bend).
- Stamped brick and rainbow are too rare to score as classes anywhere. LA's Pavers (72) and Decorative (151), plus Seattle's BRICK (42), can only support a reported recall with wide confidence intervals.
- No inventory should be treated as perfect ground truth. Score against the filtered inventory, then blind-audit a sample of both false positives and false negatives on street-level imagery. Report raw and audited figures, as in the Seattle work.
- Apply capture-date gating per the project's crosswalk temporal-alignment rule, and use each city's status and date fields where they exist.

### Gaps
- Panorama coverage and vintage were not measured for any city. `main.py --scan-only` on a sample polygon would settle this, and GSV versus Panoramax availability in Paris should be checked first.
- Completeness of LA, Austin and Paris is unmeasured. An OSM-vs-inventory disagreement audit would estimate it.
- Paris polygon semantics (raised or mixed-surface crossings, per-bar vs per-crossing outliers) need a visual sample.
