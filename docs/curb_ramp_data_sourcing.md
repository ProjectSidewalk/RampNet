# Curb-ramp data sourcing: candidate cities for a larger Stage 1 corpus

Working notes for [#59](https://github.com/ProjectSidewalk/RampNet/issues/59) — "would more
training data buy recall?" This document answers the narrower, prior question: **which cities could
we source from, how much would each buy, and what would it cost to retrain?**

Live counts were verified on **2026-07-30** by querying each publishing agency's own API; the
endpoint is recorded per row so every number is re-derivable. Counts drift (several of these refresh
weekly), so treat them as a snapshot.

**Nothing here has been acted on.** No city has been sourced, no location-precision assessment has
been run on any new city, and no retrain has been attempted. This is pre-work.

> **Read §2 first.** The paper already assessed location precision for eight cities, and that
> assessment — not inventory size — is the binding constraint on everything below.

## 0. E1 result: the harm hypothesis is NOT supported

Everything below prices out *sourcing*. #59 raised a prior objection — that scaling the same
pipeline could make the worst failure mode **worse** — and that had to be settled before any of it
mattered. It is now measured. Script: `scripts/analysis/stage1_label_recall.py`
(17 tests in `tests/test_stage1_label_recall.py`); result JSON in
`analysis_out/e1_stage1_label_recall.json`.

**The hypothesis.** Stage-1 agreement against the manual gold set is P .9403 / R .9245, so ~7.5% of
gold-visible ramps are unlabeled inside training panoramas. In heatmap regression that is not
neutral — the target is **zero** there, so the loss actively pushes activations down. If those
misses cluster in the far/small regime (plausible: projection error and the crop model both degrade
with distance), we have been training the model to suppress detections exactly where it is blind,
and scaling bakes that in harder.

**The measurement.** Two recall curves over the same **3,919 gold ramps in 793 panoramas**, on the
#25 bins with the identical `geom()` estimator `size_analysis.py` uses. Model detections at 0.55.

| distance | n | Stage-1 labels | model | gap |
| :--- | ---: | ---: | ---: | ---: |
| 0–8 m | 1,374 | 0.959 | 0.943 | +0.016 |
| 8–12 m | 1,065 | 0.918 | 0.894 | +0.024 |
| 12–18 m | 865 | 0.924 | 0.842 | +0.082 |
| 18–25 m | 498 | 0.900 | 0.779 | +0.120 |
| 25–40 m | 113 | **0.779** | **0.487** | **+0.292** |
| **drop-off** | | **0.180** | **0.457** | |

Apparent size tells the same story: Stage-1 falls 0.951 → 0.797 across 80+ px down to 20–32 px
(−0.154); the model falls 0.938 → 0.541 (−0.397).

**Verdict: FLAT.** Stage-1 label recall does decline with distance, but **the model's cliff is ~2.5×
steeper**. At 25–40 m the labels find 78% of gold ramps while the model detects 49% — the far ramps
*are* being labeled; the model is not reaching the ceiling the labels already set. The gap widens
monotonically with distance and with shrinking apparent size, which is the signature of a
resolution/model limit, not an inherited label limit. Consistent with #25's forecast (+0.103 recall
at 2× linear resolution).

**Instrument check.** Model overall recall comes out at **0.873**, reproducing the published
gold-set figure at 0.55 exactly; Stage-1 overall lands at 0.928 against the documented .9245. Both
curves reproduce known numbers, so the comparison is not an artifact of this script.

**What this does and does not license:**

- ✅ **Naive scaling is not contraindicated by this mechanism.** The #59 objection that motivated
  caution does not hold up. Sourcing work below is worth doing.
- ❌ **It does not show that scaling helps.** That is E2 (#84's epoch curve — are we even
  data-limited at 1 epoch?) and E3. This closes an objection; it does not make a case.
- ⚠️ **The implicit-hard-negative mechanism is real, just not binding.** 22% of ramps at 25–40 m
  are still unlabeled and still train as zeros. It is a second-order effect here, not a
  first-order one.
- ⚠️ **This is an in-distribution result.** The gold set is drawn from the NYC/Portland/Bend test
  split, so it says nothing about the out-of-distribution failures (Paterson's paired TSIs,
  Gainesville's diagonal ramps) that motivate the diversity argument in §1.

One analysis defect worth recording, since it nearly decided the experiment: the drop-off was
initially computed between the nearest and farthest *populated* buckets, and the gold set has **4**
ramps beyond 40 m. At n=4 a single ramp moves recall by 0.25, and that bucket inverted the sign of
Stage-1's drop-off (to −0.041). Buckets now require n ≥ 30, and a regression test pins it.

## 1. The current training corpus is mostly one city

Stage 1 is built from three cities' open-government inventories (`docs/data_provenance.md` §1).
Counting source inventories rather than derived panos, at today's live counts:

| Training city | Ramp records (live) | Paper Table 1 | Share |
| :--- | ---: | ---: | ---: |
| New York City, NY | 217,679 | 217,680 | **78.2%** |
| Portland, OR | 46,065 | 45,324 | 16.5% |
| Bend, OR | 14,800 | 13,611 | 5.3% |
| **Total** | **278,544** | 276,615 | |

Yielding **214,376 panoramas / 849,895 labels**, split 70/20/10 with a **150,063-panorama train
split** (paper §4.2).

**NYC is 78.2% of the training ramps.** In inventory terms RampNet is largely an NYC curb-ramp
detector with some Portland and a little Bend. That bears on the failure modes the benchmark keeps
surfacing — Paterson's paired tactile surface indicators, Gainesville's large diagonal ramps into
wide arterials (`docs/model_comparison.md`) — neither of which is NYC design vocabulary. The lever
this corpus is missing is plausibly **composition, not volume**.

The live counts validate cleanly against the paper's Table 1 (NYC differs by 1; DC below is
*identical*), which is a useful cross-check on the query method used throughout this document.

### Ramps → panoramas calibration

278,544 ramp records produced 214,376 panoramas:

> **≈ 0.77 panoramas per government ramp record** (inclusive of the ~20% negative panoramas)

The ratio runs **higher in sprawl cities** (ramps sparser, fewer share a panorama) and **lower in
dense grids**. A planning heuristic from three cities, not a law.

## 2. Location precision is the binding constraint, and the paper already measured it

**Paper §3.1, Table 1.** Eight government datasets were assessed by **manually overlaying curb-ramp
locations on an aerial imagery base map** and judging whether coordinates land on the physical ramp
(Fig. 2). One city rated Poor, four OK, three Good.

> *"For our purposes, we use all data from the good category: New York City, NY; Portland, OR; and
> Bend, OR — all which offer precise and diverse curb ramp styles."*

**The selection rule was "Good only."** The current corpus is not three cities because three were
available — it is three cities because **only three passed**.

| City | Paper count | **Precision** | Status |
| :--- | ---: | :--- | :--- |
| New York City, NY | 217,680 | **Good** | in training |
| Portland, OR | 45,324 | **Good** | in training |
| Bend, OR | 13,611 | **Good** | in training |
| Los Angeles, CA | 91,759 | OK | assessed, unused |
| Austin, TX | 48,995 | OK | assessed, unused |
| Washington, DC | 34,859 | OK | assessed, unused |
| Nashville, TN | 18,285 | OK | assessed, unused |
| **Seattle, WA** | 45,653 | **Poor** | **assessed, rejected** |

### ⚠️ Seattle is rated Poor — do not treat it as the obvious first add

Seattle looks like the ideal candidate on every other axis: the highest scorer in Deitz's 178-city
sample (13/14), a weekly refresh, ~46k records, rich attributes (condition, width, install date),
and it is **already contamination-burned** via the crop model, so training on it would cost no
evaluation ground. An earlier draft of this document recommended it first on exactly that reasoning.

**Table 1 rates its location precision Poor**, which is the one criterion that disqualifies a
dataset for Stage 1 — the pipeline projects the government coordinate onto the panorama, so a
misplaced coordinate produces a misplaced label. Rich attributes do not compensate.

Seattle may still matter for [#86](https://github.com/ProjectSidewalk/RampNet/issues/86) (condition
and width supervision does not need the point to be pixel-accurate), and the rating predates several
years of weekly updates so a re-assessment is defensible. But **it cannot be the default first add**,
and any argument for it has to start by contesting Table 1.

### What this reframes

The candidate pool splits three ways, and only the third has real upside:

- **Good (276,615)** — already fully used. There is no unexploited good data in Table 1.
- **OK (193,898)** — LA, Austin, DC, Nashville. Available, assessed, deliberately not used. Taking
  them is a **quality step-down the paper chose not to make**, and doing so should be an explicit,
  recorded decision rather than a side effect of chasing a number.
- **Unassessed** — every city in §3 that is not in Table 1. This is where the upside is, and none of
  it is known-good yet.

## 3. Candidate inventories not in Table 1 (unassessed)

Live counts, 2026-07-30. **None has had its location precision assessed.** Ordered by size.

| Jurisdiction | Ramp records | Endpoint | Notes |
| :--- | ---: | :--- | :--- |
| **VDOT** (Virginia, statewide) | **83,000** | `services.arcgis.com/p5v98VHDX9Atv3l7/…/ADA_Curb_Ramp_Condition_FS_9_View` | State highway ROW only. ⚠️ **includes Richmond** — see §6 |
| **Denver, CO** | **72,770** | `services1.arcgis.com/zdB7qR0BtYrg0Xpl/…/ODC_TRANS_CURBRAMPS_P` (layer **228**) | Largest city inventory found. ⚠️ *"delineated from 2022 aerial imagery"* — derived; recent, but verify it is per-ramp |
| **San Francisco, CA** | **50,096** | `services.arcgis.com/Zs2aNLFN00jrS4gG/…/Curb_Ramps_from_DataSF_pulled_weekly_` | Mirrors DataSF, **pulled weekly** |
| **WisDOT** (statewide) | ~49,000 | `data-wisdot.opendata.arcgis.com` | *Not queried.* ⚠️ 2014/15 **desktop** inventory from photo log + satellite |
| **NYSDOT** (statewide) | **42,297** | Socrata `data.ny.gov/resource/hmbc-hni2` | State ROW. NYC already in training — de-dup needed |
| **Charlotte, NC** | **40,601** | `gis.charlottenc.gov/…/CDOT_ADA/ADA_Curb_Ramps/MapServer/0` | From Charlotte's ADA self-evaluation |
| **CDOT** (Colorado, statewide) | **24,549** | Socrata `data.colorado.gov/resource/sb9m-ecvv` | State ROW only |
| **Boston, MA** | **24,022** | `gisportal.boston.gov/…/Infrastructure/OpenData/MapServer/3` | *Pedestrian Ramp Inventory*, BostonGIS |
| **Sioux Falls, SD** | **19,977** | `gis.siouxfalls.gov/…/Data/Transportation/MapServer/15` | Remarkable for a 200k city — matches Deitz's account of a post-FHWA-complaint data programme |
| **Minneapolis, MN** | **18,447** | `services.arcgis.com/afSMGVsC7QlRK1kZ/…/Minneapolis_ADA_Ped_Ramps_-_View_Layer_` | Consistent with the 17,800 in their 2020 ADA Transition Plan |
| **Arlington, VA** | **10,342** | `arlgis.arlingtonva.us/…/Open_Data/od_Sidewalk_ADA_Ramps` | |
| **Raleigh, NC** | ~14,550 | City pedestrian facility assessment (PDF) | *Reported in a study document, not queried* |
| Spokane WA · Tacoma WA · Dallas TX | exists, not pulled | ArcGIS Hub / city portals | Confirmed to publish a layer |
| **Phoenix, AZ** | gated | CurbPHX | Comprehensive, rich attributes (width, curb height, surface condition) — **behind a city login**. Request-access candidate |
| **Chicago, IL** | none published | — | See below |
| **Atlanta, GA** | 4,517 (wrong polarity) | `services2.arcgis.com/zLeajbicrDRLQcny/…/MAF_Missing_ADA_Ramps_Draft` | See below |
| Columbus OH · Houston TX · Des Moines IA · Pittsburgh PA · Oakland CA | none found | — | See below |

**Unassessed city total: ~236,000 ramps** (Denver, SF, Charlotte, Boston, Sioux Falls, Minneapolis,
Arlington). Plus ~199,000 across four state DOTs.

### Negative results

- **Chicago** would have been the highest-value candidate — CDOT states **137,000+ ramps installed
  since 2006**, and Chicago is **already contamination-burned**, so it would have been free. But that
  is a program statistic on a department page; the Socrata portal returns only crash data. **The
  ramps exist, the count is public, the geolocated inventory is not.**
- **Pittsburgh** — also contamination-burned, also unavailable. WPRDC carries steps, signalized
  intersections and centerlines but no ramps; the regional SPC layer's host refused connection; the
  two ArcGIS hits are vendor consulting deliverables (913 features). Worth one email to WPRDC.
- **Columbus** (score 9, contamination-burned) — only a *UIRF Planned Projects* layer.
- **Houston** — sidewalk permits, a sidewalk asset layer, service areas. No ramp inventory.
- **Des Moines** — sidewalk centerlines only. **Oakland** — Socrata dataset returns 4 records.

### Atlanta publishes *missing* ramps — wrong polarity for Stage 1, right one for #86

Atlanta's *Moving Atlanta Forward Draft ADA Ramp Installation Locations* holds **4,517 features**,
but they are locations where a ramp is **planned because none exists**. Useless for Stage 1.

Potentially valuable elsewhere: **absence ≠ negative** is named in #86 as one of RampNet 2.0's hard
parts, and it is the mechanism behind this issue's label-ceiling argument — we cannot distinguish
"no ramp here" from "a ramp we failed to record." A municipally-attested list of corners that
**lack** a ramp is **confirmed-absence supervision**: true negatives with provenance. Opportunistic
find; not part of any route below, and "Draft" should be taken seriously.

### On search method: terminology is the trap

Naming is wildly inconsistent — *Curb Ramps* (Seattle, Portland), *Pedestrian Ramp Locations* (NYC),
*ADA Ped Ramps* (Minneapolis), *Pedestrian Ramp Inventory* (Boston), *Access Ramps* (LA), *Sidewalk
ADA Ramps* (Arlington), *sCurbRamps* (Bend).

An earlier pass of this document searched titles for "curb ramp" only, concluded supply was thin,
and recommended verifying rather than searching further. **That was wrong** — a title search for
"curb ramp" does not match NYC's own dataset. Re-running across *pedestrian ramp*, *ADA ramp*, *ped
ramp*, *access ramp* and *curb cut* surfaced Denver, San Francisco, Boston, Sioux Falls, Nashville
and Arlington in a single pass, adding ~195k ramps. Anyone extending this list should search the
synonym set, not the phrase.

## 4. Deitz et al. (2021) — already cited by the paper as [10]

[Deitz, Lobben & Alferez 2021](https://doi.org/10.1177/20539517211047735) scored 178 US
municipalities on 14 accessibility data features. **The RampNet paper already draws on it** (§3.1:
"of the 178 US cities studied in [10], 90% published open street data but only 34% had sidewalk data
and far fewer (10%) included curb ramps").

Useful additional structure for candidate generation:

- Curb-ramp data appears **only in municipalities scoring ≥7** of 14 — sole exception Los Angeles
  (score 4, flagged incomplete). Their Table 9 (cities ≥6) is effectively the candidate pool, and it
  is how Denver, Boston, SF, Sioux Falls, Madison, Nashville and Arlington were located here.
- **Seattle is the highest scorer in the sample (13/14)** — a reminder that open-data maturity and
  *coordinate precision* are different things, since Table 1 rates it Poor.

Three limits: the portal review ran **June 2019 – March 2020**; the sample is large municipalities
only, so **Bend is not in it at all**; and portal-based discovery misses the ~1,469 ArcGIS items,
including state DOTs.

A **1-hit-in-6 test of cities below the cutoff** (Minneapolis, Des Moines, Houston, Phoenix, Atlanta,
Pittsburgh) found only Minneapolis — which itself scores 7, i.e. above the line. Weak evidence
(n=6), but it points the same way as the paper: hunting below the cutoff has poor yield, while the
score-≥7 pool is productive.

## 5. The location-precision assessment (the critical path)

The paper's method is **qualitative**: overlay coordinates on aerial imagery, judge alignment
visually, bucket Good / OK / Poor. No thresholds are published. Replicating it for a new city is
cheap — no GPU, no pipeline run, no sourcing commitment — and it is the **gate every unassessed city
in §3 must pass**.

Worth quantifying while extending it, so the tiers stop being a judgment call:

| Check | Why it matters | Test |
| :--- | :--- | :--- |
| **Positional offset** | The Stage 1 mechanism. Fig. 2b is the failure. | Sample ~50 points/city, measure metres from the true ramp on aerial imagery; report a distribution, not a bucket |
| **Per-ramp vs per-corner** | If a city records one point per *corner*, paired ramps collapse to one label — **the supervision gap behind Paterson's failure** | Records per intersection; inspect a paired corner. NYC's ~1.8/intersection implies per-ramp |
| **Staleness** | DC's live count is *identical* to the paper's, corroborating a static 2016 capture. Ramps built since are missing (recall loss); removed ramps are phantom labels (precision loss) | Compare capture date to GSV capture date |
| **Completeness** | The label-recall term that sets the ceiling. Stage 1 agreement is **P .9403 / R .9245** | Spot-check N intersections in GSV for ramps absent from the inventory |
| **CRS / datum** | A wrong projection silently shifts a whole city | Round-trip known points |
| **Active vs retired** | Seattle publishes 46,386 total but 38,468 active | Prefer the publisher's active filter |

A quantitative offset distribution would also let the OK tier be re-examined: "OK" may mean 2 m or
8 m, and that difference plausibly decides whether LA's 91,759 records are usable.

**One discrepancy to resolve.** Table 1 rates **LA "OK"** at 91,759, but LA's published Access Ramps
layer is documented as *"the geographic center of corner polygon features, with attributes indicating
the presence or absence of a wheelchair access ramp"*, derived from 2014 aerial imagery — i.e.
corner-centroid, not ramp-located. Either they assessed a different layer or "OK" tolerates
corner-level placement. Worth settling before counting LA's 91,759.

## 5a. Temporal distance is a second, independent gate — and it is not the #11 check

Positional precision asks *is the coordinate on the ramp?* Temporal distance asks *did the ramp and
the pixels exist at the same time?* They are independent, and a city can pass one and fail the other.

**This is not covered by the existing temporal-consistency filter.** [#11](https://github.com/ProjectSidewalk/RampNet/issues/11)
(closed) fixed two real bugs in that filter — a `2000-01-01` sentinel that let undated records bypass
the check, and a month comparison that ignored the year. What it now does correctly is an
**ordering** test: discard a panorama unless the ramp was installed strictly before the month of
capture (`generate_dataset_meta.py`, `(year, month)` tuple compare). Ordering is not distance, and
three failure modes survive it:

1. **Inventory older than the imagery → unlabeled positives.** Ramps built after the inventory
   snapshot are visible in the panorama but absent from the government data, so no ordering check can
   see them. The target heatmap is **zero** at a real ramp — the label-ceiling mechanism this issue
   is built around. **DC is the worst case in hand:** its live count is *identical* to the paper's
   Table 1 figure, corroborating a static 2016 capture, against GSV imagery a decade newer.
2. **Undated records → phantom labels.** `TREAT_UNDATED_AS_PREDATING = True` is a deliberate,
   documented choice, but it means every dateless record is *assumed* to predate the panorama. Where
   a city's undated fraction is high the filter is effectively off for those records, and ramps not
   yet built when the pano was captured become labels at empty pixels. **The undated fraction is
   therefore a per-city measure of how much of the filter actually functions**, and it has never been
   reported for any city.
3. **Rebuilt or removed ramps** change appearance or vanish between the two dates, and carry no
   signal either way.

Note the two errors have **opposite signs**: (1) suppresses detections at real ramps, (2) trains
detections at nothing. Both degrade the corpus, so a city's *net* temporal exposure is not captured
by any single number — report both.

**This is cheap to measure and needs no human and no imagery** — install-date distribution, undated
fraction, inventory snapshot date, and the GSV/Mapillary capture-date distribution are all metadata.
It should therefore run *before* the visual precision assessment (§5), because it can disqualify a
city for a few minutes of compute rather than a few hours of review.

### The imagery half is already collected: Streetscape Tracker

[Streetscape Tracker](https://github.com/jonfroehlich/streetscape-tracker) (`D:\Git\gsv-tracker`,
dashboard at `makeabilitylab.cs.washington.edu/public/streetscape-tracker/`) samples a frozen
geographic grid per city and records **`capture_date` per panorama** for both GSV and Mapillary, in
immutable dated snapshots. Its catalogue holds **1,204 cities / 1,863 snapshots** on makelab1
(`/projects/makeabilitylab/streetscape-tracker/data/`).

Coverage of this document's candidates is good, and critically **includes all three training
cities**, so the assessment can be calibrated against known-Good imagery:

- **Tracked** — Austin, Bend, Boston, Denver, Los Angeles, Minneapolis, Nashville, New York,
  Portland OR, Seattle, Sioux Falls, Washington DC. Also the Gainesville, Paterson and Richmond
  benchmark splits.
- **Not tracked — worth queueing**: **Charlotte NC, San Francisco CA, Arlington VA** (only Arlington
  TN and Arlington WA are in the catalogue).
- Several tracked cities have stale runs (DC 2024-04-11, Seattle 2024-12-19, Portland OR
  2024-12-21) and would benefit from a re-run.

**First measurement (2026-07-30), GSV capture-year distribution vs inventory date:**

| City | Precision (Table 1) | Inventory vintage | Modal imagery years | Gap |
| :--- | :--- | :--- | :--- | :--- |
| Bend, OR | **Good** (training) | rolling / live | **2024** (83% of 189k) | ~0 yr |
| Austin, TX | OK | live API | **2024–25** (73% of 2.0M) | ~0 yr |
| Denver, CO | unassessed | 2022 aerial delineation | **2022–24** (70% of 2.3M) | ~0–2 yr |
| **Washington, DC** | OK | **2016 static capture** | **2022–23** (63% of 159k) | **~6 yr** ⚠️ |

**DC is confirmed as the worst temporal case in the pool** — its inventory predates the modal
imagery by roughly six years, so every ramp DC built between 2016 and 2023 is in the pixels and
absent from the labels. **Denver reads better than its "aerial-derived" flag suggested**: a 2022
delineation against 2022–24 imagery is nearly contemporaneous, which is the best temporal alignment
of any unassessed candidate.

**Caveat on using the tracker this way.** Its GSV series is a *grid sample* (nearest pano per grid
point) spread uniformly over the city, whereas Stage 1 selects panoramas within 10 m of curb-ramp
locations — i.e. concentrated at intersections. The distribution above is therefore a **screening**
signal, good for ranking cities, not a substitute for the per-ramp Δt the pipeline would actually
see.

### The gate is implemented: `scripts/analysis/temporal_gap.py`

Reports the two exposures **separately** — they have opposite signs, so the script deliberately
emits no net "gap" number. It also reproduces the pipeline's ordering check (including the #11
regression) so the filter's data cost is measured rather than assumed, handles the format zoo
(ArcGIS epoch-milliseconds, Socrata ISO strings, bare years) and treats the `2000-01-01` sentinel as
undated while reporting it separately. Pure core, unit-tested in `tests/test_temporal_gap.py`
(21 tests); no GPU, no network, no imagery.

```
python scripts/analysis/temporal_gap.py --city "Washington, DC" \
    --inventory dc.json --inventory-date-field INSTALLDATE \
    --tracker-snapshot washington--district-of-columbia--...2024-04-11.csv.gz \
    --snapshot-date 2016-01
```

### First city assessed: DC is disqualified, and worse than "stale"

Querying DC's layer schema and a `groupBy` over its inspection year settles it:

- **There is no install-date field.** The layer carries `CONDITION`, `YEAR_INSPECTED`,
  `ESTIMATED_YEAR_OF_IMPROVEMENT`, `CREATED_DATE` and `LAST_EDITED_DATE` — record-keeping timestamps
  and a *future* improvement year, but nothing recording when a ramp was built. **DC's undated
  fraction is 100%**, so `TREAT_UNDATED_AS_PREDATING = True` waves all 34,859 records through and
  **the ordering filter is entirely inert for this city.**
- **`YEAR_INSPECTED` is 2016 for all 34,859 records, with no other value.** A single one-shot survey,
  never updated — which confirms the 2016 snapshot empirically rather than inferring it from the
  record count matching the paper's Table 1.

So DC maximises both exposures at once: a decade of unrecorded construction (unlabeled positives)
against a filter that cannot reject anything (phantom labels). **Drop it from every route in §6.**
It was carrying 34,859 ramps in the "+ all OK" tier, so that tier's 470,513 becomes **435,654** —
pushing 500k further out of reach on assessed data.

Worth noting for [#86](https://github.com/ProjectSidewalk/RampNet/issues/86) even so: DC publishes a
`CONDITION` field, which is condition supervision we currently discard at ingest. Being unusable for
Stage 1 localisation does not make a dataset unusable for everything.

**Correction to an earlier commit on this branch.** That commit said DC "maximises both exposures at
once". Only one of the two. DC's `YEAR_INSPECTED = 2016` is an **existence bound** — every record
demonstrably existed in 2016, and the imagery is 2022-23, so no DC record can be un-built at capture
time and **phantoms are structurally impossible**. DC's exposure is entirely one-sided: unlabeled
positives. The generalisation is in §5b, and the tool now models it.

### 5b. What bounds phantoms is the existence date, not the install date

A ramp audited in 2016 demonstrably existed in 2016, whatever its install field says. So for any
panorama captured after a record's **existence bound** — an audit date, an inspection year, or the
vintage of the aerial imagery a layer was delineated from — a phantom label is *structurally
impossible*, however many records are undated.

This matters far more than it first appears, because **install-date coverage turns out to be poor
almost everywhere** (§5c). Three of the seven cities assessed have no install-date field at all.
Were the install date the operative mechanism, most of the candidate pool would be unusable. It is
not: every one of these inventories carries *some* existence evidence, and that is the quantity the
gate should be built on.

It also reframes `TREAT_UNDATED_AS_PREDATING = True`. As a blanket assumption it is unjustified. As
a *derived consequence* of "the whole inventory was surveyed before this imagery was captured", it
is exactly right — and checkable per city.

### 5c. Six cities assessed (2026-07-30)

Schema and histogram queries against each publisher's API; GSV capture years from Streetscape
Tracker. Nothing here required a GPU, a pipeline run, or a human reviewer.

| City | Ramps | Install-date field | **Undated** | Existence bound | Median capture | Verdict |
| :--- | ---: | :--- | ---: | :--- | :--- | :--- |
| **Denver** | 72,770 | *none* | 100% | **2022** aerial delineation | 2022 | ✅ near-contemporaneous |
| **Sioux Falls** | 19,977 | `INSTALLDATE` / `InstallYear` | **37.6%** | `INSPECTEDDATE` | 2024 | ✅ best date coverage found |
| **Minneapolis** | 18,447 | `YearBuilt` | 67.9% | `stamp_date` | 2022 | ✅ usable |
| **Austin** | 49,796 | `YEAR_BUILT` | **84.6%** | `ASSESSMENT_DATE` | 2024 | ⚠️ large but date-poor |
| **Nashville** | 18,388 | *none* | 100% | `DateAudited`, spanning **1998–2025** | 2022 | ⚠️ bound spans 26 yr |
| **Boston** | 24,022 | `CONST_DATE`, **all `18991230`** | **100%** | `INSP_DATE`, **2007–2010** | 2022 | ❌ ~12-year gap |
| *Washington, DC* | 34,859 | *none* | 100% | `YEAR_INSPECTED` 2016 | 2022–23 | ❌ ~6-year gap |

**Four findings, in order of how much they change the plan:**

1. **Install-date coverage is poor across the entire pool — this is systemic, not a DC quirk.** The
   *best* city found is 37.6% undated. Three of seven have no install-date field at all, and two
   more exceed 67%. `TREAT_UNDATED_AS_PREDATING` is therefore load-bearing for every candidate we
   would add, which makes §5b's existence bound the primary mechanism rather than a refinement.
2. **Boston is the worst candidate, not DC.** Its `CONST_DATE` column is uniformly the string
   `18991230` (the spreadsheet zero date) — the field exists and carries nothing — and its
   inspection survey ran **2007–2010** against median 2022 imagery. That is a **~12-year**
   one-directional gap, during which Boston has been actively building ramps.
3. **Denver is the strongest large candidate, and for the reason I had flagged against it.** Being
   *"delineated from 2022 aerial imagery"* is a positional-precision concern, but it is a temporal
   **strength**: it fixes a hard existence bound for the entire layer at 2022, against median 2022
   GSV capture. 72,770 ramps at near-zero temporal gap. The two gates genuinely pull in opposite
   directions here, which is the clearest argument for keeping them separate.
4. **Austin is the awkward one.** 49,796 ramps — the second-largest city inventory — but 84.6% of
   `YEAR_BUILT` is empty, so it rests entirely on `ASSESSMENT_DATE` as its bound. Worth resolving,
   because it is also the richest #86 source found (`RATING` A–F, `CURB_RAMP_TYPE`,
   `DETECTABLE_WARNING`, `DATE_CONSTRUCTION_COMPLETED`).

**Two parser defects that only real data exposed**, both of which would have silently corrupted
results rather than failing loudly:

- **Compact `YYYYMMDD` read as epoch milliseconds.** Boston's `"18991230"` parsed as 1970 — turning
  a null placeholder into a plausible install date, and making the city look temporally fine.
- **Typo years defining the snapshot.** Minneapolis carries a single `2926` among 18k records; the
  default snapshot used `max()`, so one row would have set the city's snapshot to the 30th century
  and zeroed its exposure. Now a 99th-percentile quantile.

Both are fixed and regression-tested. Boston's sentinel also generalised the placeholder handling:
the tool now knows `1899-12` alongside #11's `2000-01`, and flags any *unrecognised* implausibly-old
value that dominates an inventory, since every source invents its own null date.

### For #86: attribute richness is inversely related to nothing useful

Assessed as a side effect, since these schemas had to be read anyway. **Minneapolis is the richest
source found** — per-ramp running slope, cross slope, landing dimensions, ramp width, gutter
condition rating, plus a `Retired` flag. **Sioux Falls** carries `WIDTH`, `SLOPE`, `CROSSSLOPE`,
`COMPLIANCERATING`, `Condition` and a `PHOTO` reference. **Austin** carries an A–F `RATING`.
**Nashville** and **Boston** both carry extensive slope measurements.

This is measurement-and-condition supervision at a scale we do not currently collect, and it is
uncorrelated with Stage 1 usability — Boston is disqualified for #59 and still interesting for #86.

## 6. Routes to a 500,000-ramp corpus

**Be explicit about which 500k is meant:**

- **500k ramp *records*** → need **+221,456** on today's 278,544.
- **500k *panoramas*** → needs ~650k ramps at the 0.77 ratio. A much larger programme.

For the ramp target, tiering by Table 1 gives the decisive result:

| Pool | Ramps | Cumulative |
| :--- | ---: | ---: |
| **Good** (NYC + Portland + Bend) — already used | 276,615 | 276,615 |
| **+ all OK** (LA + Austin + DC + Nashville) | 193,898 | **470,513** ❌ |
| **+ unassessed cities** (Denver, SF, Charlotte, Boston, Sioux Falls, Minneapolis, Arlington) | ~236,000 | ~706,500 |
| **+ state DOTs** (VDOT, WisDOT, NYSDOT, CDOT) | ~198,800 | ~905,300 |

> **500,000 is not reachable on assessed data alone.** Good + *every* OK city reaches **470,513** —
> about 30k short — and that is already after accepting a quality tier the paper deliberately
> rejected. **Every route to 500k depends on cities whose location precision nobody has checked.**

That makes §5 the critical path, not discovery. Concretely: assessing the seven unassessed cities is
a few days of visual work with no compute, and it determines whether 500k is a real target or an
arithmetic one. If roughly 60% of that ~236k passes at Good, 500k clears comfortably on
city-inventory data alone, with no state-DOT tail.

### ⚠️ VDOT would burn the Richmond benchmark split

VDOT is the largest single source (83,000) and the most tempting one-step fix. It is **statewide
Virginia, which includes Richmond** — one of the nine benchmark splits and the Mapillary
out-of-distribution flagship. Training on it unclipped destroys that split's held-out status.

Mitigation: clip to exclude the Richmond deployment footprint (the same dissolved PS-regions polygon
the benchmark uses) before ingest. Feasible, but the failure mode is silent. **NYSDOT** needs the
same treatment for NYC overlap.

## 7. Retrain cost

Scenario: **+ DC, Austin, Charlotte, Minneapolis** (a mix of OK-tier and unassessed, sized to be
illustrative rather than recommended).

| | Ramps | Panos (× 0.77) | Train split | Steps (÷16) |
| :--- | ---: | ---: | ---: | ---: |
| Current | 278,544 | 214,376 | 150,063 | 9,379 |
| + four cities | ~422,000 | ~325,000 | ~227,500 | ~14,200 |
| Growth | +52% | **1.52×** | | |

- **Stage 2: ≳36 h on 16 L40s** for one epoch (≳580 GPU-h) — the paper trained 1 epoch on 16 L40s,
  batch 1 per GPU (VRAM-bound), and the README says it *"will take a very long time (> 24 hours)"*.
  **>24 h is a floor, not a measurement**; no exact wall-clock is recorded anywhere.
- **Stage 1 generation is the long pole and is entirely unmeasured** — ~111k new panoramas at 32
  tiles each ≈ **3.5M tile requests** against Google's undocumented endpoints, fetched 26 panoramas
  at a time. `run_download_dataset.slurm` allocates 100 h. Rate limiting is the dominant risk.
- **The crop model needs no retrain** — reusing it is also the cleaner experiment, since only the
  data changes.

Order of magnitude: **about a week of wall-clock**, wide error bars on the Stage 1 half. Add #84's
epoch curve and multiply.

## 8. Selection rule

**Train on cities you would never want as a benchmark split.** Every city added to training is
permanently disqualified as clean evaluation ground.

- **Already contamination-burned, so free** — Seattle, Columbus, Chicago, Pittsburgh, St. Louis,
  Knoxville and the other crop-model cities (`docs/data_provenance.md` §1). Painfully, **Seattle is
  rated Poor and the other three publish no inventory**, so this category currently yields nothing.
- **Registry-clean, so costly** — Austin, Charlotte, DC, Denver, SF, Boston, Sioux Falls,
  Minneapolis, Arlington. Training on them forecloses them as future splits. Acceptable, since none
  is among the current nine, but it should be deliberate.
- **Never add Paterson or Gainesville** — two of only three GSV benchmark splits, i.e. nearly all
  in-domain-imagery evaluation.

### Diversity, if precision permits

The composition argument in §1 favours cities that are *not* NYC-like and not Pacific-Northwest-like:
Austin, Charlotte and Nashville attack the Sunbelt/Southeast vocabulary gap behind the Paterson and
Gainesville failures. But **precision gates diversity** — a Good-rated bland city beats a Poor-rated
diverse one, because the Poor city's labels are wrong wherever they land.

## 9. Archive the inventories, because they drift

**The source inventories are not in this repo.** `stage_one/dataset_generation/location_data/` and
`street_data/` are neither present nor tracked — the README tells you to download them from live
portal links. `docs/data_provenance.md` §3 records that the exact NYC/Portland/Bend files used for
the paper are archived in the **paper's supplemental material**, so RampNet 1.0 is replicable, but
**not from this repository**, and not by following the README, which serves current data.

The counts collected here quantify how much that matters. Comparing paper Table 1 against the same
endpoints today:

| City | Paper (Tab. 1) | 2026-07-30 | Drift |
| :--- | ---: | ---: | ---: |
| **Bend, OR** | 13,611 | **14,800** | **+8.7%** |
| Portland, OR | 45,324 | 46,065 | +1.6% |
| Seattle, WA | 45,653 | 46,386 | +1.6% |
| Austin, TX | 48,995 | 49,796 | +1.6% |
| Nashville, TN | 18,285 | 18,388 | +0.6% |
| New York City, NY | 217,680 | 217,679 | −0.0005% |

Bend has grown **8.7%** — it is the smallest training city, so it drifts fastest in relative terms,
and it is 5.3% of the corpus. Anyone re-running Stage 1 from the README links today builds a
measurably different dataset from the paper's and has no way to detect the difference. (NYC's
one-record change is its own signal: that inventory is effectively frozen.)

**Recommendation: commit a dated snapshot of every inventory at ingest**, alongside the fetch URL
and query, the way `benchmark/*/records.jsonl` already pins benchmark inputs. These are point files
— tens of thousands of rows, a few MB gzipped — so the cost is trivial next to losing
reproducibility. That also makes the §5c numbers re-derivable later, since every count in this
document is a snapshot of a moving target.

Doing this for the RampNet 2.0 corpus is straightforward. Doing it retroactively for 1.0 means
recovering the three files from the paper's supplemental material and committing them, which is
worth doing while it is still easy: they are the only artifacts that make the published dataset
reproducible from source, and they exist in exactly one place.

## 10. Caveats

- **Counts are a 2026-07-30 snapshot**; several refresh weekly.
- **Only Table 1's eight cities have any precision assessment.** Everything in §3 is unassessed, and
  size says nothing about usability.
- **Record count ≠ ramp count.** Inventories may hold multiple records per ramp, or one per corner.
- **Two counts are secondhand** — WisDOT ~49,000, Raleigh ~14,550 — from documents, not APIs.
- **The 0.77 panoramas-per-ramp ratio comes from three cities**, two of them dense grids. It is the
  weakest link in §7.
- **Nothing here measures whether more data helps.** That is #59's E1–E3. This document only
  establishes what sourcing would cost if the answer is yes.
