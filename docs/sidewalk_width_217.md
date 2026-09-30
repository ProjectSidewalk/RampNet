# Sidewalk width from one photo: arm 1 on the Seoul set (#217)

Issue [#217](https://github.com/ProjectSidewalk/RampNet/issues/217), part of the #86
measurement track. This is **arm 1** (single view, nearest point: Vistas segmentation →
sidewalk edges → ground-plane geometry → metric scale from camera height), run on the one
dataset we have with laser-measured width. **No GSV imagery was used**; the street-view arm is
not built. The second half of this doc lists which benchmark cities publish width ground truth
for that arm.

## Result, in one paragraph

On the held-out half of the Seoul set (254 photos, tuned on the other 260), the estimator
gets **clear width to a mean absolute error of 0.77 m (95% CI 0.50–1.06)**, with a mean bias of
+0.29 m (0.08–0.55) and a median bias of +0.13 m (−0.02 to 0.28). It flags **all 9 of the 9
photos with GT below 1.2 m** (exact 95% CI on recall 0.66–1.00) at a precision of 9/22 = 0.41
(0.21–0.64). The median bias is smaller than any of the paper's four VLMs (+0.40 to +1.05 m),
**but the tails are worse**: 90% of errors are within 2.05 m (1.06–2.93), against the paper's
best calibrated 90% interval of ±1.0 m. The large errors are on sidewalks wider than 3 m,
where the segmenter's extent of "sidewalk" and the surveyor's passage differ (shop-frontage
paving counted, near edges out of frame); the median relative bias is under 10% at every
width, so the geometry itself is not scaling wrong. **This is the easy geometry** (the camera stands on the
sidewalk, width is lateral); it says nothing yet about width seen from the street.

## Data

- **Seoul Sidewalk Accessibility Image Dataset** (Lieu et al., arXiv 2609.17882; Zenodo
  [10.5281/zenodo.22699523](https://doi.org/10.5281/zenodo.22699523), CC0). 514 photos from
  four sites (7–11 July 2026), "positioned 1.0 m above ground level and aligned with the
  center of the pedestrian walking path", iPhone 17 (26 mm equivalent) and iPhone 16 Pro
  (24 mm equivalent). Width "was measured as the unobstructed pedestrian passage using a
  Sincon SD-70 laser distance meter, excluding permanent street furniture and other fixed
  obstacles" — i.e. **effective (clear) width**. GT range 0.92–10.15 m, mean 2.97 m; 22 photos
  below 1.2 m, 69 below 1.5 m.
- Committed: [`data/seoul_sidewalk_217/summary_attributes.csv`](../data/seoul_sidewalk_217/summary_attributes.csv)
  (byte-identical to Zenodo's, md5 `4f1b49f9…`, pinned `binary` in `.gitattributes`) and
  [`manifest.json`](../data/seoul_sidewalk_217/manifest.json) (every image's size and sha256,
  and each Zenodo archive's md5). The images are not committed.
- **Facts about the files that the paper does not state**, found on download:
  - The CSV names 464 photos `.HEIC` and 50 `.JPG`; the archives hold only `.jpg`. Matched on
    the case-folded stem.
  - **None of the 514 JPEGs carries EXIF** (checked with PIL on all of them). So the phone,
    the focal length and the orientation cannot be read per photo. 480 are 5712×4284 and 34
    are 4032×3024, all landscape.
  - The paper does not say where along the sidewalk the width was taken relative to the
    photo, or how the phone was aimed. **The phones were not held level**: the vanishing point
    of the sidewalk edges puts the median pitch at **3.4° up** (half A), and single photos
    reach 10° up (see below).
- The shared copy is at `makelab2:/homes/gws/jonf/seoul_sidewalk/` (README there), also used by
  #218.

## Method

[`scripts/analysis/sidewalk_width_217.py`](../scripts/analysis/sidewalk_width_217.py), three
stages.

1. **segment** (GPU). `facebook/mask2former-swin-large-mapillary-vistas-semantic` at revision
   `4772b6bf101d91f2534c106dc524d906aeb3c68a` (the revision `crossview_arms/semantic.py`
   pins), each photo resized to 1440 px wide, 768×1024 model input, full 65-class argmax.
   Label maps are not committed; their sha256 values are in
   [`analysis_out/sidewalk_width_217/seg_meta.json`](../analysis_out/sidewalk_width_217/seg_meta.json)
   together with the environment (torch 2.8.0+cu128, transformers 4.57.6, NVIDIA A40).
2. **measure** (CPU). For each image row, the **span of walkable pixels connected to the
   walking line** (the image centre column, since the camera stood on the path's centre).
   Walkable = Sidewalk, Pedestrian Area, Curb Cut, and the manholes / catch basins / potholes
   that sit on it. Two spans per row:
   - **total**: people, bicycles and fixed obstacles are passable (the span runs through them
     and is trimmed back to walkable pixels at its ends);
   - **clear**: only people and bicycles are passable; a pole, bench, sign, bollard, etc. ends
     the span.
   A row whose span touches the frame edge (3 px) is dropped: the true edge is out of view.
   Both span ends are back-projected onto a flat ground plane 1.0 m below the camera (pinhole,
   principal point at the centre, focal length from 25 mm equivalent, no roll). The path
   direction is the mean slope of the two edge lines fitted in ground coordinates (MAD-trimmed
   least squares), and width is the end-point separation along the path's normal, so a camera
   yawed off the path does not inflate it. The per-image width is a statistic over a depth
   band: rows from the first valid depth ≥ `zmin` to `zmin + band`, median or 10th percentile.
3. **score** (CPU, from the committed widths CSV alone). Tune on half A, report on half B.

**Horizon.** Width at depth Z scales as 1/(v − v_horizon), so the horizon row is the single most
important number. Three variants: `level` (horizon at the image centre, the protocol's
nominal level phone), `vp` (the horizon through the vanishing point of the two fitted edge
lines; no estimate if it is not found or implies more than 15° of pitch), and `vp_prior` (the
VP, else the median VP pitch of half A).

**Tuning grid** (240 configurations per measure): walkable set with/without Bike Lane × which
fixed classes are obstacles (furniture only, or furniture + vegetation + terrain) × lane
markings as boundary or surface × 3 horizons × `zmin` ∈ {1.5, 2.5, 4} m × `band` ∈ {1, 3} m ×
{median, p10}. **Rule:** lowest MAE on half A among configurations that estimate at least 90%
of half A.

**Split.** ~100 m lat/lon cells (0.001°), so repeat photos of one sidewalk stay in one half;
cells stratified by whether they contain a GT < 1.2 m photo, then halved by a seeded shuffle
(seed 217). Half A: 260 photos, 27 cells, 13 below 1.2 m. Half B: 254 photos, 25 cells, 9
below 1.2 m. **CIs** are 95% cluster-bootstrap percentiles over half-B cells (10,000 draws,
seed 217), plus exact Clopper–Pearson intervals for recall and precision of the < 1.2 m flag
(the bootstrap one degenerates to [1, 1] with 9 positives).

### What was chosen on half A

| measure | walkable | obstacles | markings | horizon | band | stat | MAE A |
|---|---|---|---|---|---|---|---|
| clear | base | furniture | surface | vp | 1.5–2.5 m | median | 0.62 m |
| total | base | furniture + vegetation | surface | vp | 4–5 m | p10 | 0.55 m |

The clear pick is the literal "nearest point" of the issue: the first metre of valid ground.
All top-5 half-A configurations for both measures use the VP horizon and markings as surface,
and they differ from the winner by at most 0.02 m of MAE, so the choice is not a knife edge.

## Results (half B, n = 254)

| | clear width | total width |
|---|---|---|
| estimated (coverage) | 244 (0.96) | 246 (0.97) |
| **MAE** | **0.77 m** [0.50, 1.06] | 0.72 m [0.45, 1.01] |
| relative MAE | 25% [18, 33] | 24% [18, 32] |
| mean bias | +0.29 m [0.08, 0.55] | +0.29 m [0.08, 0.52] |
| median bias | +0.13 m [−0.02, 0.28] | +0.11 m [−0.03, 0.22] |
| error, 5th–95th pct | −1.39 to +2.50 m | −1.08 to +2.69 m |
| 90th pct of \|error\| | 2.05 m [1.06, 2.93] | 1.73 m [1.00, 3.42] |
| within 0.3 m / 0.5 m | 45% / 63% | 46% / 66% |
| 3-class accuracy (<1.2, 1.2–1.5, ≥1.5) | 0.87 [0.80, 0.93] | 0.88 [0.81, 0.94] |
| **recall, GT < 1.2 m** | **9/9 = 1.00** [0.66, 1.00] | 8/9 = 0.89 [0.52, 1.00] |
| **precision, flagged < 1.2 m** | **9/22 = 0.41** [0.21, 0.64] | 8/18 = 0.44 [0.22, 0.69] |

Brackets are 95% CIs (bootstrap; exact for recall and precision). A photo with no estimate is
never flagged, so it counts against recall.

**Confusion, clear width** (rows GT, columns estimate):

| GT \ est | < 1.2 | 1.2–1.5 | ≥ 1.5 | none |
|---|---|---|---|---|
| < 1.2 (9) | **9** | 0 | 0 | 0 |
| 1.2–1.5 (29) | 10 | **9** | 10 | 0 |
| ≥ 1.5 (216) | 3 | 8 | **195** | 10 |

The flag's false positives are mostly sidewalks just above the line: 10 of the 13 are 1.2–1.5 m.

### Where the error is

| GT width | n | MAE clear | mean bias clear |
|---|---|---|---|
| < 1.5 m | 38 | 0.37 m | +0.17 m |
| 1.5–3 m | 91 | 0.36 m | +0.16 m |
| 3–5 m | 109 | 1.16 m | +0.58 m |
| ≥ 5 m | 16 | 1.59 m | −0.72 m |

and by filename series (a proxy for capture session; the paper gives no photo-to-site map):
IMG_4xxx MAE 0.44 m (n = 68), IMG_6xxx 1.02 m (n = 157), IMG_89xx–90xx 0.24 m (n = 29). The
median *relative* bias is under 10% in every width bin, so the geometry is not scaling wrong;
the large absolute errors are on wide sidewalks where the segmenter's sidewalk and the
surveyor's passage are different extents. Two half-A examples (half B was not inspected
after scoring): IMG_6754, GT 4.17 m, estimate 9.78 m — the paved shop-frontage apron is
labelled Sidewalk and counted; IMG_6387, GT 5.02 m, estimate 2.15 m — the near rows run off the
left of the frame and the first in-frame rows are cut by a row of parked share-bikes and a
planter.

### What the vanishing point buys, and pitch sensitivity

Same configuration with the horizon at the image centre (`level`): clear MAE 0.85 m, mean bias
**−0.66 m**; total MAE 1.26 m, bias −1.03 m, precision of the flag 0.10. The phones were
tilted up (median VP pitch −3.4° on half A), so a level assumption understates every width;
the VP removes most of that. Found on 96% (A) and 98% (B) of photos.

Measured sensitivity on half B, shifting the pitch used by ±2° from the VP estimate: median
width change −10% (+2°, down) and +12% (−2°, up) for clear width (−16% / +16% for total, whose
band is farther out). So the ±2° the issue asked about is worth ±10–16% of width: about
±0.3 m on a 3 m sidewalk. Pitch, not focal length or camera height, is the dominant geometric
error term here.

### Against the paper's VLMs

The paper reports, on all 514 photos, median width bias +0.40 m (GPT-5.2) to +1.05 m
(InternVL3.5-8B), and a best calibrated 90% interval half-width of ±1.0 m (GPT-5.2, coverage
0.91). Here, on half B: median bias +0.13 m [−0.02, 0.28], i.e. below every VLM; but the 90th
percentile of |error| is 2.05 m [1.06, 2.93], wider than GPT-5.2's ±1.0 m interval. The two are
not the same statistic (theirs is a conformal prediction interval, ours an empirical error
quantile), and they are on different subsets (all 514 vs half B), so this is a rough
comparison: **less bias, heavier tails.** A half-A scale calibration (×0.98) changes nothing
material (clear MAE 0.75 m).

## Caveats (they apply to every number above)

- **Easy geometry.** The camera stood on the sidewalk facing along it, so width is lateral and
  well resolved. From the street, width lies along the viewing ray, where one degree is 0.3 m
  of ground at 6 m range from 2.5 m up. None of this transfers to GSV without its own GT.
- **Definition mismatch.** GT is effective width after excluding permanent furniture; "total"
  here is Vistas's sidewalk extent and "clear" subtracts only what Vistas labels. Neither
  knows the surveyor's rule about frontage zones, planting strips or where the passage ends.
  That total beats clear slightly on MAE (0.72 vs 0.77 m, well inside each other's CIs) says the
  obstacle subtraction is not yet adding information.
- **Focal length is ±4% unknown.** No EXIF; 25 mm equivalent is the midpoint of the two phones.
  Every width carries up to ±4% from this alone.
- **Camera height is taken as exactly 1.0 m** (the stated protocol); a 5 cm error is a 5%
  width error. Roll is assumed zero; the ground is assumed flat. GT running and cross slopes
  have medians of 1.1° and 1.2° but reach 11.8° and 9.7°; a running slope shifts the edges'
  vanishing point and so is partly absorbed into the VP pitch, a cross slope is not modelled.
- **Small positive class.** 9 photos below 1.2 m in half B; recall's exact CI runs from 0.66.
- **Where GT was measured** relative to the photo is not documented. Width varies along a
  sidewalk; some of the error is the two measuring different cross-sections.
- **Development contact with half B.** While debugging, per-image output was printed for 38
  label maps, 19 of them in half B (IMG_4293–IMG_4335), and one half-B photo (IMG_4315) was
  viewed (and one half-A photo, IMG_4284). That output motivated three options: markings as surface, the `vp_prior` horizon,
  and raising the VP pitch cap from 10° to 15°. The first two went into the tuning grid and
  were chosen on half A by the rule above, not fixed by hand; the cap was changed directly.
  After scoring, only half-A images were inspected.
- The group split is by 0.001° cells; a sidewalk run that crosses a cell boundary can still
  have photos in both halves.

## Reproduce

```bash
# 0. images (≈4.6 GB; about 10 min on makelab2 with three parallel curl downloads)
python scripts/analysis/seoul_fetch_217.py fetch --dest $SEOUL        # md5 vs Zenodo, then
                                                                      # sha256 vs the manifest
# 1. label maps (GPU; 490 s on a shared A40)
python scripts/analysis/sidewalk_width_217.py segment --images $SEOUL/images --out $SEG
# 2. widths for every configuration (CPU; 264 s)
python scripts/analysis/sidewalk_width_217.py measure --seg $SEG \
    --out analysis_out/sidewalk_width_217/widths.csv.gz
# 3. tune on A, report on B (CPU; ~20 s)
python scripts/analysis/sidewalk_width_217.py score \
    --widths analysis_out/sidewalk_width_217/widths.csv.gz \
    --out analysis_out/sidewalk_width_217/results.json
```

Step 3 needs only committed files. The committed `widths.csv.gz` has sha256
`877304c5a4611d28f6d6edc43fd7d995b903e38e08e98248eeabac233971c825` (recorded in
`results.json`). A re-run of step 1 on other hardware may not reproduce the label maps
byte-for-byte (GPU nondeterminism); compare against `seg_meta.json` and expect the widths CSV
to differ slightly if they do. `tests/test_sidewalk_width_217.py` checks the geometry on
synthetic label maps of known width (level, pitched, yawed, off-centre, truncated by the
frame, with obstacles) and that the committed half-B headline recomputes from the committed
per-image estimates.

## Cost

Free compute. makelab2 A40 (shared with two other jobs): segmentation 490 s = **0.136 GPU-h**;
measure 264 s CPU; score seconds on the desktop. Two `paid: false` rows in
`analysis_out/usage_log.jsonl` (`sidewalk-width-217:segment-vistas`, `:measure`). Download
about 10 minutes of network, no compute.

## Not done

- **Per-pano camera height and the street-view geometry** (arm 1 on GSV) — needs GT; see below.
- **Arm 2** (multi-view fallback) and **arm 3** (VLM contrast on our own imagery).
- **Sidewalk-plane fit.** Here the plane is the stated 1.0 m below the camera; nothing is
  fitted. On GSV the issue's plan fits the sidewalk plane, not the road.
- **Obstacle subtraction that helps.** Clear width does not beat total; a better rule (e.g.
  subtracting only obstacles whose base lies inside the band) was not tried.

## Ground truth for the street-view arm: which benchmark cities publish sidewalk width

Searched 2026-09-30. "Verified" means the live schema was opened (ArcGIS REST `?f=json`, WFS
`DescribeFeatureType`, or the DBF inside the downloaded zip) and the rows that actually carry a
width were counted with group-by queries; everything else is marked. Every source below is an
open download or API. **No source says whether its width is total pavement width or clear
width**, except Bend's metadata, which calls its field a minimum width. Treat every width here
as total surface width until shown otherwise. The Seoul GT above is clear (effective) width, so
the two are not the same quantity.

| City | Dataset | URL | Geometry | Width field | Verified | Notes |
|---|---|---|---|---|---|---|
| **São Paulo** | GeoSampa `geoportal:calcada` (Calçadas) | WFS `http://wfs.geosampa.prefeitura.sp.gov.br/geoserver/ows?service=wfs&version=1.0.0&request=DescribeFeatureType&typeName=geoportal:calcada` (GetFeature with `outputFormat=application/json` works) | **Polygon**, one per block face, EPSG:31983 | `qt_largura_minima_trecho` / `_maxima_` / `_media_`, metres (sample 1.66 / 2.44 / 2.05); also area and slope min/max/mean. Total vs clear unknown | Yes, schema + 2 features | 491,383 polygons. No update date in the WFS; news items date the release to 2019 |
| **Bend OR** | City of Bend "Sidewalk" | REST `https://services5.arcgis.com/JisFYcK2mIVg9ueP/arcgis/rest/services/Sidewalk/FeatureServer/0` (item `013ed9a6e2054947b6c787e5064cee8d`) | Line | `SWWidth` is a coded bin (MIN3/4/5/6/8/10, Multi-Use, OTHER), "the minimum width of the sidewalk, in feet". `ClearWidth` is a YES/NO/Pending flag, not a number | Yes | 19,983 rows, 16,477 PRESENT; SWWidth filled on 14,390 (MIN5 9,820; MIN6 1,984; MIN4 1,499; MIN8 544; MIN10 300). Last edit 2026-09-30. Deschutes County: nothing |
| **Richmond VA** | (a) City "Transportation Surfaces", SubType 8 = Sidewalk | REST `https://services1.arcgis.com/k3vhq11XkBNeeOfM/arcgis/rest/services/Transportation_Surface/FeatureServer/0` | **Polygon** (planimetric) | none (derivable from geometry) | Yes | 113,870 sidewalk polygons; last edit 2023-12-08 |
| | (b) VDOT Virginia Statewide Sidewalk Inventory | REST `https://services.arcgis.com/p5v98VHDX9Atv3l7/arcgis/rest/services/Virginia_Statewide_Sidewalk_Inventory/FeatureServer/0` | Line (digitised from aerials) | `width`, feet, integer. Total vs clear unknown | Yes | Richmond City 18,511 rows; 6,091 have width 0 (missing); then 3 ft 4,635, 4 ft 4,359, 5 ft 1,661, 6–9 ft. Also Henrico, Chesterfield. Item modified 2026-08-18 |
| | (c) City "Sidewalks_View" | `.../Sidewalks_View/FeatureServer/0` (item `97296c18e4984ada822dc1d03f813b5e`) | Polygon | none (material, condition, side) | Yes | 10,808 rows; last edit 2026-09-21. Henrico `WIDTH_FEET` filled on 30 of 3,706; Chesterfield and PlanRVA: no usable width |
| **Annapolis MD** | Anne Arundel County "Sidewalks" (the city publishes none) | REST `https://gis.aacounty.org/arcgis/rest/services/OpenData/Structure_OpenData/MapServer/9` | **Polygon** | none (derivable) | Yes | 362,233 polygons "captured from 2023 orthophotos"; 3,677 in a small downtown Annapolis box. The county centreline layer has no width |
| **Gainesville FL** | (a) FDOT `Sidewalk_Width_Sep_TDA` | REST `https://services1.arcgis.com/O1JpcwDW8sjYuddV/arcgis/rest/services/Sidewalk_Width_Sep_TDA/FeatureServer/0` | Line, linear-referenced by side | "width is recorded to the nearest foot"; the field is probably `SWSCD` (inferred from its values, not documented) | Yes | **State highways only**: 855 rows in Alachua County, mostly 5 ft. Updated 2026-09-26 |
| | (b) City Socrata "sidewalks" | `https://data.cityofgainesville.org/api/views/swi2-fkvs.json` | Line | `width` filled on 5 of 4,931 rows | Yes | Last updated 2020-09-21. City Public Works layer: no width |
| **Paterson NJ** | NJDOT County Road Sidewalk Inventory, Passaic | `https://www.nj.gov/transportation/refdata/countysidewalks/zip/shapefiles/Passaic.zip` | Line, milepost-referenced | `width`, integer, presumably feet (0, 3–15) | Yes, DBF read | County routes only, collected about 2007. ~570 rows per side, ~150 zeros each. Nothing from the city or Passaic County |
| **Clovis CA** | — | city GIS page `https://www.clovisca.gov/services/technology/gis.php` | — | — | Yes (REST tree walked) | No sidewalk layer (curb and gutter only). Nothing for Fresno County |
| **Morgantown WV** | MMMPO Bike/Ped Plan "Existing Sidewalks" | `https://services7.arcgis.com/lE5mQkgxehcTjzKf/arcgis/rest/services/MMMPO_Bike_Pedestrian_Plan/FeatureServer/2` | Line | `Width` = 0 on all 1,016 rows | Yes | Nothing usable |
| **Vancouver BC** | City: sidewalk condition rating 2021 (lines), right-of-way widths (property line to property line); TransLink Regional Sidewalk Data 2025 | TransLink `https://services7.arcgis.com/WpS8F3vcmrEQUG8m/arcgis/rest/services/Regional_Sidewalk_Data_2025_WFL1/FeatureServer/1` | Line | none | Yes | Nothing usable |
| **Budapest** | Budapest Közút "Üzemeltetett utak" | `https://kapu.budapestkozut.hu/arcgis/rest/services/kozutfigyelo/kozutfigyelo/MapServer/2` | Road centreline | only sidewalk type (both / one side / none) | Yes (REST tree walked) | Nothing usable |

OpenStreetMap is a fallback only where sidewalks are mapped as separate ways with `width=*`;
coverage for these cities was not counted.

**Where to start the street-view arm.** São Paulo is the one benchmark split with measured
metric widths (min / mean / max per block face) on every sidewalk polygon. Richmond is the best
US option: widths can be derived from the city's 114k planimetric polygons and cross-checked
against VDOT's integer-foot inventory. Bend's minimum-width bins can check the width classes
but not metric error. Annapolis polygons (2023) also allow derived widths. Not yet done: none of
these layers has been fetched into the repo or joined to a GSV pano.
