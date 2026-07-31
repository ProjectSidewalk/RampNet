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

Both columns are recall against the same human ground truth, but they measure different things, and
the asymmetry is the point:

- **"Stage-1"** = *was a training label present at this ramp?* Stage 1 is **not a detector** — it is
  handed the ramp's location by the government inventory, projects that coordinate onto the
  panorama, and uses the crop model only to refine where the point lands. So this column measures
  **whether the supervision existed**, which is precisely the ceiling the hypothesis is about.
- **"model"** = *did the trained detector fire there?* It gets no positional hint and must locate
  ramps from pixels alone.

Stage-1 should therefore be the higher curve everywhere; the question is only whether it *falls off
with distance the same way the model does*.

*Caveat:* these Stage-1 labels come from the **test** split, so the model never trained on these
particular panoramas. Same pipeline and same cities, so they are a fair proxy for the label quality
the train split received — but a proxy, not a direct measurement of it.

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
steeper**. At 25–40 m a training label was present at 78% of gold ramps while the model detected 49%
— the far ramps *were* labeled; the model is not reaching the ceiling that supervision set. The gap widens
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

## 0a. How much of the missing recall can more data even reach?

E1 closed the objection to scaling but also implied something sharper: the far-field cliff is a
**pixel-count** problem, so more cities cannot fix it — while the *vocabulary* failures the
benchmark keeps surfacing (Paterson's paired tactile surfaces, Gainesville's diagonal arterial
ramps) plausibly are fixable that way. Those are two different populations with two different
programmes attached, and nobody had sized them.

Script: `scripts/analysis/miss_decomposition.py` (15 tests). Reads the committed low-floor caches,
so no GPU, no network, no imagery. Threshold 0.30 (the #79 recommendation); boundary 18 m, the last
distance at which the model still has adequate signal.

**Pooled across the seven US splits — 2,060 GT ramps, 427 misses, recall 0.793:**

| population | misses | share | recall points | fixable by |
| :--- | ---: | ---: | ---: | :--- |
| **Far-field** (≥ 18 m) | 247 | **57.8%** | 0.120 | multi-view (#48/#38), resolution (#25) — **not** more cities |
| **Near-field** (< 18 m) | 180 | **42.2%** | 0.087 | broader/more diverse training corpus |

**Neither dominates.** Roughly three-fifths of the missing recall is pixel-starved and two-fifths is
not. Both programmes have a real target, and the sourcing work below is aimed at a population worth
about **8.7 recall points** pooled.

| split | tier | GT | recall | miss | far | near | far % | MV ceiling |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| richmond | mapillary | 310 | 0.829 | 53 | 32 | 21 | 60.4% | 0.889 |
| bend | gsv | 327 | 0.823 | 58 | 36 | 22 | 62.1% | 0.904 |
| clovis | mapillary | 195 | 0.821 | 35 | 21 | 14 | 60.0% | 0.907 |
| morgantown | mapillary | 267 | 0.805 | 52 | 15 | 37 | **28.8%** | 0.829 |
| annapolis | mapillary | 294 | 0.810 | 56 | 37 | 19 | 66.1% | 0.902 |
| paterson | gsv | 395 | 0.719 | 111 | 64 | 47 | 57.7% | 0.792 |
| gainesville | gsv | 272 | 0.772 | 62 | 42 | 20 | 67.7% | 0.872 |
| *budapest* † | – | 300 | 0.643 | 107 | 40 | 67 | 37.4% | 0.667 |
| *manual_gold* † | – | 3,919 | 0.892 | 423 | 141 | 282 | **33.3%** | 0.915 |

† swept, not pooled.

**The geometry caveat was checked, and it does not drive the result.** Flat-ground distance is
`camera_height / tan(depression)`, which fails on unleveled rigs and hills —
`docs/detection_recall_analysis.md` reports it agreeing with DA3 metric depth at Spearman **0.95 on
GSV but only 0.81 on Mapillary**, and four of the seven pooled splits are Mapillary. A tilted rig
pushes a near ramp toward the horizon, so geometry would call it *far* and **overstate** the
far-field share. Two checks say that is not what happened:

- **Above-horizon GT ramps** — geometrically impossible for a ground ramp, hence a direct tell —
  number **5 across 1,066 Mapillary ramps (0.5%) and 0 across 994 GSV ramps.**
- The **GSV** tier shows a *higher* far share (61.5%) than **Mapillary** (53.6%) — the opposite
  direction to the bias.

(Budapest, held out, has **12 of 300 (4%)** above the horizon — eight times the US rate, consistent
with its consumer rig and its low reviewer confidence. A useful independent corroboration of why it
is held out.)

**Apparent size is the starker cut**, and is what makes the pixel-limit reading concrete:

| apparent size | n GT | recall |
| :--- | ---: | ---: |
| 12–20 px | 49 | **0.163** |
| 20–32 px | 230 | 0.543 |
| 32–50 px | 427 | 0.745 |
| 50–80 px | 677 | 0.885 |
| 80+ px | 643 | 0.885 |

Below ~32 px the model is largely blind, and recall saturates around 0.885 above 50 px — so extra
pixels stop helping well before perfect, which is itself a caution against expecting resolution
alone to close the gap.

**Optimistic multi-view ceiling: 0.868 (+0.075 recall).** That assumes a closer capture exists for
every far ramp and that re-observation succeeds at the measured near-field rate; it ignores fusion
cost and the extra false positives more looks would generate. It is "what is on the table", not a
forecast — but +7.5 points with **no new data collection** is a serious number next to a
multi-month sourcing campaign.

**Caveat on the near-field population.** Calling all 42.2% "vocabulary" is an inference, not a
measurement. A near-field miss can equally be occlusion (a parked car), deep shadow, or surface
debris — Gainesville's reviewer flagged debris explicitly — or a GT disagreement. **The near-field
figure bounds the sourcing-addressable population from above**, and separating those causes needs
the miss taxonomy in #46.

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

## 5d. Per-ramp vs per-corner no longer needs a reviewer (2026-07-31)

Of §5's six checks, one is pure geometry and can be settled from the point set alone. It is also
the check with the most at stake: **if a city records one point per corner rather than per ramp,
paired ramps collapse to a single label** — the supervision gap behind Paterson's failure, where
#46 found 72% of near-field misses were adjacent-pair merges. Script:
`scripts/analysis/inventory_geometry.py` (30 tests in `tests/test_inventory_geometry.py`);
results in `analysis_out/inventory_geometry_*.json`.

A corner in a modern build carries **two** ramps, one per crossing direction, metres apart. So the
two conventions separate on the **nearest-neighbour distance distribution**: a per-ramp inventory
has a strong mode at the within-corner spacing, a per-corner inventory's nearest neighbour is the
next corner across a crosswalk.

**NYC calibrates it, so the number is not a judgment call.** NYC publishes `rampid` *and*
`cornerid`, making it ground truth for this question. Single-link clustering at a 6 m link
reproduces NYC's own corner grouping at **precision 0.976 / recall 0.973** (135,421 geometric vs
134,127 published groups). That is what licenses running the same clustering on cities that
publish no corner key.

### The cross-city result, which is not the one I expected

Run over every frozen snapshot (§9). **Ordered by pairing density, and the ordering is the finding.**

| City | Records | Rec/corner (6 m) | Singleton | Share ≤6 m | vs published corner key |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **NYC** — Good, in training | 217,679 | **1.61** | 0.402 | 0.750 | **P .976 / R .973** |
| **Portland** — Good, in training | 46,101 | **1.38** | 0.635 | 0.539 | — |
| **Bend** — Good, in training | 14,805 | **1.31** | 0.726 | 0.444 | — |
| Sioux Falls | 19,991 | 1.22 | 0.785 | 0.356 | — |
| **Denver** | 72,770 | **1.21** | 0.794 | 0.345 | — |
| Arlington | 10,342 | 1.12 | 0.887 | 0.209 | — |
| Charlotte | 40,600 | 1.09 | 0.917 | 0.155 | P .621 / R .406 ⚠️ |
| Minneapolis | 18,453 | 1.09 | 0.917 | 0.156 | P .924 / R .859 |
| Boston | 24,022 | 1.08 | 0.924 | 0.145 | — |
| San Francisco | 50,096 | **6.64** | 0.038 | 0.994 | — ⚠️ see below |

**All three training cities sit above every candidate.** That is the result, and it is not a Denver
finding — Denver is second-highest of the candidates. An earlier draft of this section compared
Denver against NYC alone and called it weak on pairing; **that framing was wrong**, and it was wrong
in the way single-baseline comparisons usually are. NYC at 1.61 is the outlier, and Minneapolis's
own published corner key independently confirms 1.09 is real, not an artifact of the clustering.

So the honest statement is about the corpus, not the city: **every candidate would dilute pairing
density relative to what RampNet trains on today.** Whether that is a problem is genuinely open, and
it cuts both ways — it is a *vocabulary* difference (which §8 argues we want, and which is exactly
what Paterson and Gainesville punished us for lacking), but it also means fewer paired examples to
learn the σ/`min_distance` separation that #46 found us failing.

### The mechanism, from the two cities that publish ramp type

Sioux Falls (`RAMPTYPE`) and Charlotte (`RP_Type`) let the ratio be decomposed rather than guessed:

| Sioux Falls corner composition | Groups | Records/group |
| :--- | ---: | ---: |
| **Diagonal** only | 6,154 | **1.004** |
| **Directional** only | 9,455 | **1.365** |
| mixed / other | 800 | 1.134 |

A diagonal corner really is one ramp, and it is recorded as one record. Charlotte agrees from the
other direction using its own published corner key: corners containing a diagonal type average
**1.059** records against **1.176** for those without. Sioux Falls is 31% `Diagonal`, and Charlotte
is 30% `*Diag`.

**So a low records-per-corner is substantially ramp-design vocabulary, not under-recording** — which
is the benign reading of Denver's 1.21, and the one the evidence now favours. It does not settle
Denver, because Denver publishes no type field at all; that is what `ramps_visible` on the review
sheet is for.

### The two anomalies, both caught without imagery

**⚠️ San Francisco is disqualified for Stage 1 — its coordinates are intersection centroids.** The
50,096 records carry only **7,553 distinct coordinates**, mapping 1:1 to `cnn` (SF's intersection
node id). Every ramp at an intersection is stamped with *the intersection's* single point: a modal
6–8 rows per coordinate, up to 29. `curbreturnloc` (N/NE/E/…) records which return each row
describes, but the geometry does not follow it. Feeding this to Stage 1 would project ~4.7 identical
labels onto one pixel, none of them on a ramp. Its 6.64 records/corner and 0.994 share-within-6 m
are the signature, and no reviewer time was needed to find it.

Also note **14,414 of its rows carry `crexist = 0`** — confirmed *absence*, the same polarity as
Atlanta's layer (§3) and Charlotte's `NoRamp`. Useless for Stage 1, potentially valuable for #86.

**⚠️ Charlotte's coordinates disagree with Charlotte's own corner key.** Geometric recovery is
P .621 / R .406 against `RP_IntID`+`RP_LocInInt`, far below NYC's .976/.973 and Minneapolis's
.924/.859. The cause is spread: of Charlotte's 3,839 multi-record published corners, **52.5% have
members more than 6 m apart, 14.4% more than 20 m, and the 99th percentile is 308 m.** Large
arterial corners explain the tail up to ~30 m; they do not explain 308 m. Either `RP_IntID` groups
more loosely than it appears or the coordinates are poorly placed — and the second is a
positional-precision red flag that Charlotte's review sheet must specifically test.

### The confound this survives

A single threshold calibrated on Manhattan is not obviously transferable — NYC's corner radii are
tight, and a city built to suburban geometry would space *the same pair* further apart and score as
per-corner purely for being wide. Sweeping the link distance separates the two readings.
`groups_per_intersection` is the guard: it starts near 4 and collapses once the link bridges the
crossing, past which records-per-group means nothing.

| link | NYC rec/group | NYC groups/intersection | Denver rec/group | Denver groups/intersection |
| ---: | ---: | ---: | ---: | ---: |
| 3 m | 1.355 | 4.22 | 1.015 | 4.29 |
| 4 m | 1.528 | 3.75 | 1.046 | 4.16 |
| 5 m | 1.580 | 3.62 | 1.138 | 3.82 |
| **6 m** | **1.607** | **3.56** | **1.213** | **3.59** |
| 8 m | 1.638 | 3.49 | 1.344 | 3.24 |
| 10 m | 1.721 | 3.33 | 1.582 | 2.75 |
| 12 m | 2.462 | 2.32 | 1.819 | 2.39 |

**NYC plateaus at 1.53 → 1.64 across 4–8 m while its groups stay resolved; Denver never plateaus.**
Denver climbs monotonically and only reaches NYC's ratio at a 10 m link, by which point its
groups-per-intersection has already fallen to 2.75 — the rise is the link bridging *different
corners*, not resolving pairs. Compared at matched merge state (≈3.6 groups/intersection, i.e. 6 m
for both), **Denver records 1.21 points per corner where NYC records 1.61**.

So the wider-radii explanation does not hold, and Denver carries materially less pairing than the
corpus's dominant city.

### What this does *not* settle

Two mechanisms produce the same signature, and geometry cannot separate them:

1. **Denver records one point per corner** — a recording convention, and a supervision defect. Its
   delineation is from aerial imagery, whose own metadata concedes *"imagery resolution is not high
   enough to discern"* ADA compliance, so under-separating a close pair is plausible.
2. **Denver's corners physically carry one ramp** — the single diagonal apron at the corner apex,
   standard in pre-1990s residential build-out. That is not a defect at all; one ramp, one label is
   correct, and it is exactly the non-NYC vocabulary §8 argues for.

Distinguishing them needs eyes on imagery, which is why `ramps_visible` is a required field on the
review sheet below. **The count is the evidence**, and it is the single most decision-relevant thing
a reviewer of Denver can produce.

## 5e. Denver: the rest of the automated gate (2026-07-31)

Everything here comes from the frozen snapshot (§9) and needed no reviewer.

**Footprint — passes.** 62,006 of 72,770 records (85.2%) fall strictly inside Denver County, tested
against the city's own `County_Boundary__Area_` layer (main ring 404.5 km² against the county's
400.7 km² reference, so the polygon is right). Of the 10,764 outside, **98.8% are within 1 km of the
boundary** — shared-ROW spillover on arterials — and only **128 records (0.18%)** are more than 2 km
out, at 7–20 km, i.e. Denver Mountain Parks. This is Denver's inventory, not a regional one.

**Schema — thin, and poor for #86.** `OBJECTID`, `CREATEDATE`, `CREATEUSER`, `COMMENTS`,
`UPDATE_STATUS`, `UNIQUE_ID`. No install date, no ramp type, no width, no condition, no
detectable-warning field. Native CRS is **EPSG:2877** (NAD83 / Colorado Central, US survey feet),
server-reprojected to 4326 on request. Against Minneapolis's per-ramp slopes and landing dimensions,
Denver contributes nothing to #86.

**⚠️ The "2022 imagery" claim does not survive contact with the data.** The service describes itself
as *"sidewalk ramps delineated from 2022 aerial imagery"*, and §5c graded Denver ✅ near-contemporaneous
on an existence bound of 2022. Crosstabbing `UPDATE_STATUS` against `CREATEDATE`:

| status | meaning | records | share | CREATEDATE years |
| :--- | :--- | ---: | ---: | :--- |
| `NC` | No Change | 69,986 | **96.2%** | 2015 (54,120), 2017 (7,391), 2019 (5,527), 2021 (2,854), 2022 (**84**), 2016 (10) |
| `A` | Add | 2,784 | 3.8% | 2023 (2,770), 2024 (14) |
| `M` | Modify | **0** | 0% | — |

**74.4% of the layer carries a 2015 creation date, only 84 records carry 2022, and the `Modify` code
is used zero times across 72,770 records.** Two readings are consistent with `NC`: either every
feature was re-verified against 2022 imagery and confirmed unchanged (the bound holds), or `NC`
simply means "not touched in this pass" and a 2015 delineation is being carried forward (the bound
is 2015, a ~7-year gap against median 2022 GSV capture — comparable to the ~6-year gap that
disqualified DC in §5c).

The zero `M` count is evidence for the second reading: a genuine re-examination of 70k features that
produced *no* modifications and only 84 new records in 2022 is hard to credit. **§5c's ✅ for Denver
should be treated as unconfirmed pending an answer from the publisher.** Note also that nothing in
the schema records removals, so a demolished ramp has no mechanism to leave the layer — phantom
labels have no upper bound from this data.

**Coincident duplicates:** 72 records within 0.5 m of another (0.10%), against NYC's 22 (0.01%).
Small, but each is two identical labels in one panorama.

### The review sheet is built — and the obvious basemap was not good enough

`scripts/analysis/inventory_review_sheet.py` renders the §5 positional instrument: an aerial chip
per sampled record, centred on the published coordinate, with range rings at 1/2/5/10 m so the
reviewer reads an **offset in metres** instead of forming an impression.

**The first attempt was unusable, and the failure mode was silent.** Esri World Imagery — the
default anywhere ArcGIS is involved — renders Denver leaf-on, hazy and visibly upsampled at an
effective ~1 m, turning a ramp and its detectable-warning pad into a smudge; and at z=21 it serves
*"Map data not yet available"* as a flat grey tile, which the fetcher pasted into the sheet as
though it were imagery. Measuring a 1–2 m offset against that is not possible, and *appearing* to
is worse than not trying.

Denver's own **`Aerial2018_tilecache`** is leaf-off, sharp and 0.23 m/px at this latitude. The
generalisable lesson: **every city needs its municipal basemap located before its sheet is worth a
reviewer's time**, the global fallback will not do, and blank tiles must be detected rather than
presented. Both are now enforced in the tool, along with a `--tile-source` registry that records
which imagery produced which verdict.

### The instrument is registered and scaled — checked, not asserted

The sheet asks a reviewer to judge a **1-2 m** offset, so two claims have to hold or every
verdict it produces is quietly wrong: the crosshair is on the published coordinate, and the rings
really are 1/2/5/10 m. Neither is visible by looking at the sheet, because the error and the
measurement would come from the same code. Both are now checked against something external —
`scripts/analysis/verify_chip_georeference.py` (15 tests), evidence in
`analysis_out/georef_check/`.

**Tile scheme.** Denver's `Aerial2016` cache is standard Web Mercator: 256 px tiles, EPSG:3857,
origin −20037508.342787, and LOD resolutions matching 156543.03392800014 / 2^z to **3×10⁻¹⁰**
relative. So the projection assumption is not an assumption.

**Scale, against the WGS84 ellipsoid.** Points are constructed an exact ground distance away using
the local radii of curvature — maths that shares nothing with the Web Mercator `cos(lat)` factor
it is validating, so an error there cannot cancel itself — then projected and measured, at eight
bearings. Worst error **0.26% at every radius: 2.6 mm on the 1 m ring, 26 mm on the 10 m ring.**
Constant in *relative* terms across radii, which identifies it as the expected sphere-vs-ellipsoid
residual rather than a bug (an additive error would shrink proportionally as the ring grows). A
regression test confirms the checker would report >20% if the latitude correction were ever
dropped.

**Registration, against the city's own centrelines.** Denver's LRS street geometry is drawn into
the imagery with the same projection that places the crosshair, and the offset to the roadway's
optical centre is measured on cross-sections every 4 m. Centrelines are ground-level, so unlike
building footprints they carry no roof-lean parallax.

| Neighbourhood | east median | north median | resultant |
| :--- | ---: | ---: | ---: |
| Park Hill | +0.11 m | −0.11 m | 0.16 m |
| Berkeley | −0.06 m | +0.06 m | 0.08 m |
| Athmar | −0.06 m | −0.06 m | 0.08 m |
| Hampden | −0.06 m | −0.46 m | 0.46 m |
| Montbello | −0.06 m | +0.11 m | 0.13 m |

937 usable cross-sections. **No systematic shift**: a NAD83/WGS84 datum mismatch applied on one
side and not the other — the plausible failure, since Denver publishes in EPSG:2877 and the server
reprojects — would be **~1 m and consistent in direction** across every site. It is not there.

Two methodological notes, because the first version of this measurement was wrong twice. Offsets
must be resolved into a **geographic** frame: the segment normal's sign flips with the direction a
segment happens to be digitised in, so a real eastward shift cancels in the median. And each
cross-section must be credited **only to the axis it crosses** — a north-south street says nothing
about the north component, and pooling both axes in a grid city fills each median with structural
zeros and reports 0.00 m whatever the truth is.

**What this does not certify.** A centreline is a cartographic construct, not a survey of the
pavement midline; crowned roads, one-sided parking bays and kerb extensions move the optical
centre without moving the true one. Read this as *no gross error* — the instrument is sound at the
scale it is being asked to measure — not as a calibration certificate.

**Sheet as built:** 59 chips (one dropped — outside the basemap footprint), record-weighted sample,
seed 20260731, frame restricted to `UPDATE_STATUS=NC` because the 2,784 records added in 2023–24
postdate the 2016 imagery and are expected to be absent. Output in `analysis_out/review_denver-co/`.

**⚠️ The imagery is near-contemporaneous with the delineation, which makes this a lower bound.**
`Aerial2016` was chosen for resolution — 0.057 m/px against the 2018 cache's 0.23 m/px, which would
render a 40 m chip as 174 px and lose detectable-warning pads entirely — and a positional check does
not normally care about capture year, because ramps do not move. But 74.4% of Denver's records carry
a **2015** `CREATEDATE`, so for the bulk of the frame we are checking a delineation against imagery
of nearly the same date, quite possibly the imagery it was digitised from.

That measures **digitising precision**, which is the right quantity for "does the coordinate land on
the physical ramp". It is *not* the whole error a Stage 1 label carries, because that label is
projected into a GSV panorama captured ~2022, and everything that changed in between — ramps rebuilt,
moved, or demolished — is invisible to this instrument. So both headline numbers from this review are
lower bounds on their Stage 1 equivalents:

- the **offset distribution** excludes any post-2016 drift, and
- the **phantom rate** excludes any ramp that existed in 2016 and was gone by the panorama date,
  which matters more than usual here because Denver's schema has no removal mechanism at all.

The temporal gate (§5a) is the separate instrument for that component; this one should be read as
*positional error at the time of delineation*, and the two composed rather than either quoted alone.

### The rubric is part of the instrument (2026-07-31)

The first ten minutes of the actual review produced four questions the sheet could not answer, and
every one of them would have changed the number: *what is the "correct corner" when the schema has
no corner key? where on the ramp is the reference point? how many ramps do I count on a chip
containing four corners? do I click when it already looks perfect?* A convention that lives only in
the reviewer's head gets applied two ways in one sitting, and **`0.9 m` is uninterpretable without
the rule saying what it is 0.9 m from**. So the rules are now a `RUBRIC` constant that renders
beside the field it governs, opens in full with `?`, and is **copied verbatim into the exported
manifest**. `verdicts.json` cannot be read without them.

The clauses that carry the most risk:

- **Click the centre of the concrete apron — never the detectable-warning pad.** PROWAG R305 puts
  the pad at the back of curb on perpendicular, blended and diagonal ramps and on the street-level
  landing of a parallel ramp, so pad centres sit ~0.6–0.9 m down-slope of ramp centres. The pad is
  the most visible thing in 0.057 m/px imagery, which makes pad-clicking the *easy* mistake, and it
  would add that 0.6–0.9 m to every record as a systematic bias **indistinguishable from real
  positional error**. That is most of the 1 m ring, i.e. enough to move the bucket on its own.
  (Legacy ramps with fully-domed surfaces are the one case where the two coincide.)
- **Count ramps by containment** — what is reachable from the crosshair without crossing a roadway
  — which is per-corner, not per-chip. "One ramp per crossing" was tried and is wrong: a median
  island has two cut-through ends serving a single crossing.
- **Click every chip, including a dead-centre one.** Otherwise near-zero cases are recorded by
  omission and the low tail becomes an artefact of reviewer confidence. Related: offsets below
  ~0.3 m are at the instrument's floor (≈5 px; the registration check's per-site medians are
  0.08–0.46 m), so the left tail is reported as floor-limited rather than as centimetres.

**A readable corner with no ramp is now a verdict rather than a gap.** Such a chip was previously
*uncompletable* — nothing to click, so the offset stayed null, so `done()` was never true and "next
unreviewed" walked straight back to it — leaving only a wrong exit: `unjudgeable`, which asserts "I
cannot see" rather than "I can see, and it is not there". The new `no_ramp` state records a
**phantom**, and the phantom rate is a headline number for Denver specifically, since nothing in its
schema records removals.

### The per-corner comparison, and a threshold artefact it exposed

Each chip now also carries **how many records Denver itself publishes within 6 m and 10 m**
(`count_neighbours`) — the same per-corner quantity from the published side. Differencing it against
the reviewer's `ramps_visible` is what §5d explicitly deferred to imagery. **The published count
stays hidden until the reviewer has entered their own**, because showing it first would anchor the
judgment it exists to be compared against.

The sample is representative on this axis: **23 of 59 chips (39.0%) have a published neighbour
within 6 m, against 34.5% for the full 72,770-record inventory** — inside one standard error at
n=59, computed by code sharing nothing with `inventory_geometry.py`.

**⚠️ The 6 m clustering threshold under-groups large corners.** Chip `66519` is a channelising
"pork-chop" island whose three ramps sit at 0.0, 5.8 and 7.0 m from the sampled record; single-link
at 6 m **splits that island in two and scores one of its three ramps as a singleton**. The reviewer
independently counted three ramps there, and Denver publishes three records — so the inventory is
per-ramp at that corner and the clustering is what loses it. Since 6 m was calibrated against NYC's
tight urban corners (P .976 / R .973), this is exactly the failure NYC could not have revealed:
**part of Denver's 1.21 records/corner may be large suburban corner radii and channelised islands
rather than a vocabulary difference.** Re-running the clustering at 8 m and 10 m and watching Denver
move *relative to* NYC would settle it. Not yet done.

Chip `67585` is the sample's other outlier — **4 published records within 6 m** (at 5.2, 5.4, 6.0 m),
where nothing else in the 59 exceeds 2. Flagged for a careful count: either four ramps genuinely
cluster at that median end, or Denver has duplicates there.

**The Good/OK/Poor question for Denver remains open** — the review is in progress against the
rubric above.

## 6. Routes to a 500,000-ramp corpus

**Be explicit about which 500k is meant:**

- **500k ramp *records*** → need **+221,456** on today's 278,544.
- **500k *panoramas*** → needs ~650k ramps at the 0.77 ratio. A much larger programme.

For the ramp target, tiering by Table 1 gives the decisive result:

| Pool | Ramps | Cumulative |
| :--- | ---: | ---: |
| **Good** (NYC + Portland + Bend) — already used | 276,615 | 276,615 |
| **+ all OK** (LA + Austin + DC + Nashville) | 193,898 | **470,513** ❌ |
| **+ unassessed cities** — corrected, see below | **180,673** | **651,186** |
| **+ state DOTs** (VDOT, WisDOT, NYSDOT, CDOT) | ~198,800 | ~850,000 |

> **500,000 is not reachable on assessed data alone.** Good + *every* OK city reaches **470,513** —
> about 30k short — and that is already after accepting a quality tier the paper deliberately
> rejected. **Every route to 500k depends on cities whose location precision nobody has checked.**

### ⚠️ The unassessed pool is 23% smaller than first counted (2026-07-31)

The original ~236,000 counted published *records*. Reading the frozen snapshots (§5d, §5e) shows
two of those counts are not ramp locations:

| City | Listed in §3 | Corrected | Why |
| :--- | ---: | ---: | :--- |
| Denver | 72,770 | 72,770 | — |
| **San Francisco** | 50,096 | **0** | Coordinates are **intersection centroids** — 7,553 distinct points for 50,096 rows. Unusable for Stage 1 at any precision tier |
| **Charlotte** | 40,601 | **35,095** | 5,505 records are `RP_Type = NoRamp` — confirmed *absence*, not ramps |
| Boston | 24,022 | 24,022 | temporal gate already ❌ |
| Sioux Falls | 19,977 | 19,991 | live drift |
| Minneapolis | 18,447 | 18,453 | live drift; 4 rows carry null geometry |
| Arlington | 10,342 | 10,342 | — |
| **Total** | ~236,000 | **180,673** | **−55,327** |

**The consequence is sharper than the headline number.** Good (276,615) + *every* unassessed city,
even if all of them passed at Good, is **457,288 — still short of 500,000.** So the "no OK tier, no
state DOT" route that looked available is closed: **500k now requires either accepting the OK tier
the paper rejected, or the state-DOT tail with its Richmond/NYC clipping hazards.** That is a
decision for the programme, not a detail.

§5 remains the critical path — assessing what is left is a few days of visual work with no compute —
but the arithmetic it is feeding is tighter than when this document was written. Note also that both
corrections were found by *reading the data*, not by reviewing imagery: the cheap automated checks in
§5d pay for themselves before any reviewer is booked.

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

### Done as of 2026-07-31 — `data/inventories/`

`scripts/analysis/fetch_inventory.py` (24 tests) writes gzipped JSONL plus a sidecar manifest
recording the endpoint, the exact query, the fetch date, the declared-vs-retained count and a
sha256 of the payload. **Nothing in this programme is analysed from a live endpoint any more**;
every number in §5d/§5e is derived from a committed file.

| Snapshot | Records | Note |
| :--- | ---: | :--- |
| `nyc-ny-2026-07-31` | 217,679 | Paper Tab. 1: 217,680 — **−0.0005%, effectively frozen** |
| `portland-or-2026-07-31` | 46,101 | Paper Tab. 1: 45,324 — **drifted +1.7%** |
| `bend-or-2026-07-31` | 14,805 | Paper Tab. 1: 13,611 — **drifted +8.8%** |
| `denver-co-2026-07-31` | 72,770 | First candidate assessed |

Two gzip header fields are pinned (`mtime=0`, `filename=""`) so identical records hash identically;
without that the digest tracks when and where the file was written rather than what is in it, and
is useless as a drift signal. This was a real bug, caught by the test rather than by inspection.

**Two things this does *not* fix, stated plainly:**

1. **These are not the paper's files.** Portland and Bend have drifted +1.7% and +8.8%, so a Stage 1
   re-run from `data/inventories/` reproduces *today's* dataset, not the ICCV one. The paper-exact
   NYC/Portland/Bend files exist in exactly one place — the paper's supplemental material — and
   recovering and committing them is still open. It gets harder, not easier, with time.
2. **The basemap imagery behind any §5 verdict is not redistributable.** The review sheet embeds
   Esri or municipal tiles under terms that do not permit re-hosting, so `verdicts.json` records the
   tile-source URL template, zoom and per-chip tile keys instead. A replicator can re-fetch the
   exact tiles; they cannot get them from this repo. That is a stated blocker, not a solved problem.

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
