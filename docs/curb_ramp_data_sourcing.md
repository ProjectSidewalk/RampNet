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

§0–§0c (E1, how much missing recall more data can reach, the miss buckets, the far-field `visible` anomaly) moved verbatim to [`data_scaling_59.md`](data_scaling_59.md), section numbers unchanged (#145).

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

§5–§5o (the location-precision assessment: the temporal gate, the per-city results, Denver, the Stage 1 tolerance curve, Seattle, Charlotte, supply, the street-level instrument) moved verbatim to [`location_precision_assessment_96.md`](location_precision_assessment_96.md), section numbers unchanged (#145).

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

- **Stage 2: 3.49 h on 16 GPUs for one epoch (~56 GPU-h)** — **measured** 2026-08-03 from the paper
  run's own TensorBoard events at 1.341 s/step, 9,378 steps/epoch. Scaled to this scenario
  (~14,200 steps): **~5.3 h/epoch**. Full method, the 500k projection, and the caveats are in
  [`stage2_training_cost.md`](stage2_training_cost.md).
  > **Correction.** This bullet previously read *"≳36 h on 16 L40s for one epoch (≳580 GPU-h)"*,
  > extrapolated from the README's *"> 24 hours"*. That was **~10× too high**: the ">24 h" covers
  > the paper's *whole ~12-epoch, preemption-riddled run* (44.7 h active / 74.6 h calendar), not one
  > epoch. Stage 2 is an overnight job; **the epoch count matters ~12× more than the corpus size**.
- **Stage 1 generation is the long pole — and is now measured at ≤4,370 panoramas/h** (≥49.1 h for
  the paper's 214,599, a lower bound: `--requeue` hides incarnations and the run was finished
  interactively). ~111k new panoramas at 32 tiles each ≈ **3.5M tile requests** against Google's
  undocumented endpoints ⇒ **≥25 h**; a full 385k-panorama rebuild is **≥88 h**. Rate limiting
  remains the dominant risk, and **storage is a real prerequisite** — the paper run hit a disk-quota
  wall at ~214k panoramas (it recovered, losing nothing). Yield was **97.91%**, the shortfall being
  almost exactly the panoramas Google refused to serve. Full method and caveats:
  [`stage1_generation_cost.md`](stage1_generation_cost.md).
- **The crop model needs no retrain** — reusing it is also the cleaner experiment, since only the
  data changes.

Order of magnitude: **about a week of wall-clock**, wide error bars on the Stage 1 half — and now
that Stage 2 is measured, that week is **essentially all Stage 1**. #84's epoch curve is the one
multiplier that still moves the Stage 2 half (×5 at 1.0's auto-val optimum, ×12 at what the paper
run actually did).

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

**Done for RampNet 1.0 — the inventories are now committed.** This section originally recorded the
opposite, and the recommendation at the end of it is what got acted on:
`stage_one/dataset_generation/location_data/` holds the three NYC/Portland/Bend files the paper's
run consumed (71.8 MB, hash-pinned in `docs/data_provenance.md` §3), and `street_data/` holds a
42.9× derivative of the centrelines with an equivalence proof, because the raw 801.6 MB download
cannot go in git. The README's portal links still serve *current* data, so they remain a way to
build a different dataset, not a way to rebuild this one.

Two caveats travel with that:

- **The committed files hold 276,071 records against Table 1's 276,615.** The 544-record gap is
  between Table 1 and the files rather than anywhere in the pipeline; `data_provenance.md` §3.3
  reconciles it and says plainly what is still unexplained.
- **Publishing the inputs did not make the paper's Stage 1 replayable**, only re-runnable. The
  negative-panorama sampler is unseeded, so the paper's negatives come from
  `negativepanosSHORTENED.jsonl` and cannot be regenerated from any street file.

The counts collected here are what quantified the need. Comparing paper Table 1 against the same
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

**Recommendation, now standing policy: commit a dated snapshot of every inventory at ingest**,
alongside the fetch URL and query, the way `benchmark/*/records.jsonl` already pins benchmark
inputs. These are point files — tens of thousands of rows, a few MB gzipped — so the cost is
trivial next to losing reproducibility. That also makes the §5c numbers re-derivable later, since
every count in this document is a snapshot of a moving target.

**This is done for 1.0 and applies to every city added to the RampNet 2.0 corpus.** The 1.0 files
were recovered from the paper run's scratch directory rather than the supplemental material, and
they existed in exactly one place when they were found — which is the argument for pinning a
city's inventory at the moment it is ingested rather than hoping to reconstruct it later.

### Done as of 2026-07-31 — `data/inventories/`

`scripts/analysis/fetch_inventory.py` (30 tests) writes gzipped JSONL plus a sidecar manifest
recording the endpoint, the exact query, the fetch date, the declared-vs-retained count and a
sha256 of the payload. **Nothing in this programme is analysed from a live endpoint any more**;
every number in §5d/§5e/§5i is derived from a committed file.

| Snapshot | Records | Note |
| :--- | ---: | :--- |
| `nyc-ny-2026-07-31` | 217,679 | Paper Tab. 1: 217,680 — **−0.0005%, effectively frozen** |
| `portland-or-2026-07-31` | 46,101 | Paper Tab. 1: 45,324 — **drifted +1.7%** |
| `bend-or-2026-07-31` | 14,805 | Paper Tab. 1: 13,611 — **drifted +8.8%** |
| `denver-co-2026-07-31` | 72,770 | First candidate assessed |
| `seattle-wa-centerlines-2026-07-31` | 34,484 | **Reference geometry**, not an inventory — SDOT SND |
| `denver-co-centerlines-2026-07-31` | 7,866 | **Reference geometry** — Denver's control for §5i |

**Street centrelines are frozen on the same terms**, via `--geometry polyline`. They are not curb
ramps and are never counted as supply; they are the independent reference §5i measures the ramp
coordinates against, and an analysis that exonerates a city's coordinates must not depend on a live
endpoint for the thing it exonerated them against. Each city's centrelines must come from the **same
publisher and CRS as its ramp layer**, which is what makes a shared datum fault cancel and a
ramp-layer defect show.

Two gzip header fields are pinned (`mtime=0`, `filename=""`) so identical records hash identically;
without that the digest tracks when and where the file was written rather than what is in it, and
is useless as a drift signal. This was a real bug, caught by the test rather than by inspection.

**Two things this does *not* fix, stated plainly:**

1. **These are not the paper's files.** Portland and Bend have drifted +1.7% and +8.8%, so a Stage 1
   re-run from `data/inventories/` reproduces *today's* dataset, not the ICCV one. The paper-exact
   NYC/Portland/Bend files are committed separately, at
   `stage_one/dataset_generation/location_data/` — recovered from the paper run's scratch directory
   rather than from the supplemental material, which is where this section used to say they would
   have to come from. The two directories answer different questions and both are needed: a
   paper-exact re-run reads `location_data/`, and `data/inventories/` is the frozen 2026-07-31 state
   the RampNet 2.0 supply analysis is computed from.
2. **The basemap imagery behind any §5 verdict is not in this repo.** The review sheet embeds Esri
   or municipal tiles, which we do not re-host, so `verdicts.json` records the tile-source URL
   template, zoom and per-chip tile keys instead. A replicator re-fetches the exact tiles from the
   same service; they cannot get them from here. That is a stated blocker, not a solved problem.

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
