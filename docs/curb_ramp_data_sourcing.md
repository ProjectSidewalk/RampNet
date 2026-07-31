# Curb-ramp data sourcing: candidate cities for a larger Stage 1 corpus

Working notes for [#59](https://github.com/ProjectSidewalk/RampNet/issues/59) — "would more
training data buy recall?" This document answers the narrower, prior question that #59 assumes
away: **which cities could we even source from, how much would each buy, and what would it cost
to retrain?**

Everything with a number attached was verified live on **2026-07-30** by querying the publishing
agency's own API (method and endpoint recorded per row, so any number here is re-derivable).
Counts drift — these datasets are updated weekly in some cities — so treat them as a snapshot,
not a constant.

**Nothing here has been acted on.** No city has been sourced, no lat/lng quality gate has been
run, and no retrain has been attempted. This is the pre-work.

## 1. The current training corpus is mostly one city

The Stage 1 corpus is built from three cities' open-government curb-ramp inventories
(`docs/data_provenance.md` §1). Counting the source inventories rather than the derived panos:

| Training city | Ramp records | Share of training ramps |
| :--- | ---: | ---: |
| New York City, NY | 217,679 | **78.2%** |
| Portland, OR | 46,065 | 16.5% |
| Bend, OR | 14,800 | 5.3% |
| **Total** | **278,544** | |

Which yields the published corpus: **214,376 panoramas / 849,895 labels**, split 70/20/10 with a
**150,063-panorama train split** (paper §4.2).

**This is the most consequential number in this document.** In inventory terms RampNet is largely
an NYC curb-ramp detector with some Portland and a little Bend. It bears directly on the failure
modes the benchmark keeps surfacing — Paterson's paired tactile surface indicators, Gainesville's
large diagonal ramps into wide arterials (`docs/model_comparison.md`) — neither of which is NYC
design vocabulary.

The implication for #59 is that **composition, not volume, is the lever this corpus is missing**.
Adding ~164k ramps across four cities is only +59% in count, but it moves NYC from 78% to ~49% of
the corpus.

### Ramps → panoramas calibration

278,544 ramp records produced 214,376 panoramas, so:

> **≈ 0.77 panoramas per government ramp record** (inclusive of the ~20% negative panoramas)

Use this to size any candidate. Two caveats: the ratio will run **higher in sprawl cities** (Austin,
Charlotte — ramps are sparser, so fewer share a panorama) and **lower in dense grids** (NYC). It is
a planning heuristic derived from three cities, not a law.

## 2. Verified candidate inventories

Ordered by size. "Verified" means the count was returned by the agency's own API on 2026-07-30.

| Jurisdiction | Ramp records | Endpoint / source | Notes |
| :--- | ---: | :--- | :--- |
| **VDOT** (Virginia, statewide) | **83,000** | `services.arcgis.com/p5v98VHDX9Atv3l7/…/ADA_Curb_Ramp_Condition_FS_9_View` | State highway ROW **only** — arterials through many towns, not whole cities |
| **Austin, TX** | **49,796** | `services.arcgis.com/0L95CJ0VTaxqcmED/…/TRANSPORTATION_curb_ramps` | City of Austin CTM |
| **WisDOT** (statewide) | ~49,000 | `data-wisdot.opendata.arcgis.com` | *Documented estimate, not queried.* Desktop inventory built 2014/15 from photo log + satellite; annual updates |
| **Seattle, WA** | **46,386** total / **38,468 active** | `services.arcgis.com/ZOyb2t4B0UYuYNYH/…/Curb_Ramps_(Active)` | Weekly refresh. Carries **condition, width, install date** — see §5 |
| **Charlotte, NC** | **40,601** | `gis.charlottenc.gov/…/CDOT_ADA/ADA_Curb_Ramps/MapServer/0` | From Charlotte's ADA self-evaluation |
| **Washington, DC** | **34,859** | `maps2.dcgis.dc.gov/…/Transportation_ADA_WebMercator/MapServer/3` | ⚠️ **Captured 2016** — a 10-year-old snapshot; see §4 |
| **Raleigh, NC** | ~14,550 | City sidewalk/pedestrian facility assessment (PDF) | *Reported in a study document, not queried* |
| Spokane WA · Tacoma WA · Los Angeles CA · Dallas TX | exists, **count not pulled** | ArcGIS Hub / city portals | Confirmed to publish a curb-ramp layer |
| **Columbus, OH** | **none found** | — | See below |

### Columbus: no public point inventory located

Columbus scored 9/14 in Deitz et al. and is **already contamination-burned** via the crop model
(`docs/data_provenance.md` §1), which would have made it a free city to train on. A title search of
ArcGIS Online returned only a *UIRF Planned Projects* layer (planned work, not an inventory).

This is consistent with Deitz's Table 6: at score 9, only 1 of 2 cities carried curb-ramp data, and
Columbus appears to be the one that does not. **Recorded as a negative result** — someone should
check `opendata.columbus.gov` directly before concluding it, but do not assume Columbus is available.

### The supply is much larger than the literature suggests

A title search for "curb ramp" on ArcGIS Online returns **1,469 items**. Two categories that the
municipal-portal literature misses entirely:

- **State DOT inventories** (VDOT 83k, WisDOT ~49k, MnDOT, DelDOT). These cover many cities at once
  but only along state highway right-of-way — arterials and connecting highways. Partial per city,
  and skewed toward exactly the wide-arterial context Gainesville's failures cluster in.
- **County and small-city layers** (Leon County FL, Tacoma, Westfield IN, Fitchburg WI, …). Mostly
  small — Leon County returns **191** features — so the tail is long but thin.

## 3. What Deitz et al. (2021) does and does not give us

[Deitz, Lobben & Alferez 2021](https://doi.org/10.1177/20539517211047735), *Squeaky wheels: Missing
data, disability, and power in the smart city*, scored 178 US municipalities on 14 accessibility
data features. It is the only systematic survey we know of, and it is **useful as a candidate
generator**:

- Curb-ramp data appears **only in municipalities scoring ≥7** of 14 — sole exception Los Angeles
  (score 4, and the authors flag it as incomplete, 2014 installs only). So their Table 9
  (all cities scoring ≥6) is effectively the candidate pool.
- **Seattle is the highest scorer in the entire sample (13/14).** Portland and Washington DC tie at
  12; NYC is at 10. RampNet's existing choice of Portland and NYC is corroborated independently.
- Only **17%** of municipalities with any accessibility data had curb ramps — 18 cities of 178.

Three limits that matter for our purposes:

1. **Six years stale.** Portal review ran June 2019 – March 2020. Charlotte, Austin, and Seattle
   have all published or substantially grown inventories since.
2. **It undercounts by construction.** The sample is municipalities >150k population, plus the ten
   most populous per census subregion and the most populous per state. **Bend, OR is not in the
   sample at all** — one of RampNet's own three training cities, with a perfectly good 14,800-ramp
   inventory, is invisible to their method. Their 17% is a floor for large cities, not a ceiling for
   all cities.
3. **Portal-based discovery misses ArcGIS Online.** They searched municipal open data portals; the
   1,469 ArcGIS items above include state DOTs and county layers their method would not surface.

## 4. Prerequisite: the lat/lng quality gate

**This gates everything downstream and should run before any city is sourced.** Stage 1 projects a
government GPS point onto a panorama; the label is only as good as that coordinate. A city with a
large but poor inventory is worse than a small good one, because of the mechanism in #59: an
unlabeled ramp inside a positive panorama gets a **zero** in the target heatmap, so bad or missing
records actively train the model to suppress detections.

Per candidate city, before committing:

| Check | Why it matters | Cheap test |
| :--- | :--- | :--- |
| **Positional accuracy** | Stage 1 assumes the point is *at* the ramp. Systematic offset (e.g. recorded at the intersection centroid or the parcel) shifts every label in that city. | Sample ~50 points, compare against the ramp's visible position in GSV |
| **Per-ramp vs per-corner** | If a city records one point per *corner*, paired ramps collapse to one label — **the exact supervision gap behind Paterson's failure**. NYC's 217,679 over ~120k intersections implies ~1.8/intersection, i.e. per-ramp. | Count records per intersection; inspect a paired corner |
| **Staleness** | DC's data was **captured in 2016**. Ramps built since are missing (label-recall loss); ramps removed are phantom labels (label-precision loss). | Check the capture/update date; compare install-date distribution against the GSV capture date |
| **Completeness** | This is the label-recall term that sets the ceiling. Stage 1 agreement is currently **P .9403 / R .9245** against the manual gold set; a patchier city lowers it. | Spot-check N intersections in GSV for ramps absent from the inventory |
| **CRS / datum** | A wrong projection silently shifts an entire city. | Confirm the declared CRS and round-trip a few known points |
| **Active vs retired** | Seattle publishes 46,386 total but **38,468 active** — the difference is retired records that may no longer exist on the ground. | Prefer the active filter where the publisher provides one |

Seattle is the strongest here and DC the weakest: Seattle refreshes weekly and publishes condition,
width, and install date; DC is a single 2016 capture.

## 5. Beyond location: attributes we currently discard

Per [#86](https://github.com/ProjectSidewalk/RampNet/issues/86), RampNet 2.0 expands scope from
*find* to *measure / condition / tag*, and we currently discard government attributes at ingest
(keeping lat/lng/date only). Some candidate inventories carry exactly that supervision:

- **Seattle** — condition, ramp width, install date, direction, assessment date.
- **VDOT** — the layer is explicitly *ADA Curb Ramp **Conditions***.
- **Charlotte** — built from an ADA self-evaluation, so compliance attributes.

If #86 proceeds, city selection should weigh attribute richness, not just record count. Seattle is
the standout on this axis.

## 6. Cost of the five-city scenario

Scenario: add **DC + Seattle + Austin + Charlotte + Columbus**. Columbus has no locatable inventory
(§2), so the arithmetic below is the **four-city** version. Seattle uses the *active* count.

| | Ramps | Panoramas (× 0.77) | Train split (70%) | Steps (÷16) |
| :--- | ---: | ---: | ---: | ---: |
| Current | 278,544 | 214,376 | 150,063 | 9,379 |
| **+ DC, Seattle, Austin, Charlotte** | **442,268** | **~340,400** | **~238,300** | **~14,890** |
| Growth | +59% | **1.59×** | | |

NYC's share falls from **78.2% → 49.2%**.

### Wall-clock

**Stage 2 training** is the only stage with a published anchor: the paper trained **1 epoch on 16
L40s** (4 nodes × 4 GPUs, `stage_two/run_train.slurm`), batch size 1 per GPU (VRAM-bound), and the
README states it *"will take a very long time (> 24 hours)"*. Neither the paper nor the repo records
an exact wall-clock, so **>24 h is a floor, not a measurement**.

Scaling that floor by 1.59×:

- **Stage 2, 1 epoch: ≳ 38 h on 16 L40s** (≳ 610 GPU-hours).
- If #84's epoch curve runs first (2–4 epochs), multiply accordingly — 4 epochs is ≳ 150 h.

**Stage 1 dataset generation is the long pole and it is entirely unmeasured.** Roughly 126k new
panoramas must be fetched from Google's undocumented tile endpoints (32 tiles each at zoom 3 ≈ **4M
tile requests**) and passed through the crop model. `run_download_dataset.slurm` allocates
`--time=100:00:00`, which suggests the authors expected it to be long. The fetch runs 26 panoramas
concurrently (`ThreadPoolExecutor(max_workers=26)`, 50 threads per pano for tiles). Rate limiting
against unofficial endpoints is the dominant risk and cannot be estimated from the repo.

**Order-of-magnitude total: about a week of wall-clock**, dominated by the Stage 1 fetch, assuming
nothing throttles or breaks. The estimate has wide error bars on the Stage 1 half.

**The crop model does not need retraining.** It is a keypoint placer, loaded from a hardcoded path
by `dataset_generation/inference_isolator.py`. Reusing it is also the better experiment: only the
data changes, so any delta is attributable.

## 7. Selection rule

**Train on cities you would never want as a benchmark split.** Every city added to training is
permanently disqualified as clean evaluation ground.

- **Seattle is already burned** — it is in the crop-model contamination registry via Project
  Sidewalk (`docs/data_provenance.md` §1), so training on it costs nothing we still hold. Columbus
  would have been too, but appears to have no inventory.
- **Austin, Charlotte, and DC are registry-clean**, so training on them forecloses them as future
  splits. Acceptable — none is among the current nine — but it should be deliberate.
- **Do not add Paterson or Gainesville to training.** They are two of only three GSV benchmark
  splits, i.e. nearly all of our in-domain-imagery evaluation.

### The tension worth naming

**Seattle is the safest add and the weakest diversity add.** It is a third Pacific Northwest city
alongside Portland and Bend, sharing regional design standards, climate, and street-grid era. The
cities that actually attack the vocabulary gap behind the Paterson and Gainesville failures are the
Sunbelt/Southeast ones — **Austin and Charlotte**.

If only one city is added first, adding Seattle tests the *least* interesting axis. Pair it with
Austin or Charlotte in the same run.

## 8. Caveats

- **Counts are a 2026-07-30 snapshot.** Several of these refresh weekly.
- **Record count ≠ ramp count.** Inventories may hold multiple records per physical ramp, or one per
  corner. §4 covers the test; it has not been run.
- **No inventory here has passed a quality gate.** §4 is the checklist, not a result.
- **Two counts are secondhand** (WisDOT ~49,000, Raleigh ~14,550) — from published documents, not
  queried APIs. They are marked as such above and should be verified before use.
- **The 0.77 panoramas-per-ramp ratio is derived from three cities**, two of which are dense grids.
  It is the weakest link in the §6 arithmetic.
- **Nothing here measures whether more data helps.** That is #59's E1–E3. This document only
  establishes what sourcing would cost if the answer turns out to be yes.
