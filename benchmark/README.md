# Validation benchmark

Human-validated ground truth for RampNet's curb-ramp detector, on **real deployment
imagery** — the fixed comparison target for model changes (see issues #21, #22, #26).
Each city is one split; per pano every model detection is judged correct / incorrect /
unsure and the reviewer marks ramps the model missed, so both precision and recall are
measurable.

## Layout

```
benchmark/<city>/
  records.jsonl   detection records for the validated panos (detections + pano metadata)
  verdicts.json   human verdicts (crop judgments + missed-ramp marks), self-contained
```

`verdicts.json` also carries the reviewer's **notes**, which nothing scores: a top-level
`review_notes` block about the review itself (what fought the rubric, how confident the
reviewer is in their own pass) and an optional per-pano `note`. Write them in the gallery's
*Review notes* panel and the per-pano note box; `score_validation.py` prints `review_notes`
**above** the numbers and the per-pano notes below them, so a caveat reaches whoever quotes
a precision figure instead of sitting in a README they didn't open. They round-trip through
re-reviews (`gt_gallery.py` prefills and re-exports them). Schema:
`rampnet/validation.py`. **Budapest is the split this exists for — see below.**

`records.jsonl` + `verdicts.json` are all the **scoring** needs — they're image-free, so
precision/recall reproduce with no imagery. Score with:

```
python scripts/score_validation.py benchmark/<city>
```

The **native-resolution panos** (for the labeling UI and the resolution experiment, #25)
are archived separately and published to HF (#21); they are intentionally not in git.

## How a split is produced — the two-repo boundary

RampNet and [`sidewalk-auto-labeler`](https://github.com/ProjectSidewalk/sidewalk-auto-labeler)
split by **question, not by city**. A validation city (Richmond, Bend, Clovis, …) is simply
where both questions land on the same panos — neither repo "owns" a city.

- *"What are the curb ramps in this city?"* → **auto-labeler**. Enumerate → thin → fetch
  imagery → detect → submit. It is the only thing that fetches pixels (RampNet has no
  `sources/`), so it also hands RampNet the imagery bundle.
- *"How good is the model, and is it improving?"* → **RampNet**. Ground truth, scoring, and
  the benchmark itself.

**To add a city, follow [`docs/adding_a_benchmark_city.md`](../docs/adding_a_benchmark_city.md)** —
the full protocol with a checklist. The table below is the two-repo summary; the protocol covers
what the table does not, which is everything a new split invalidates downstream (pooled numbers,
per-tier curves, the #55 correction, both figures, four documents).

Per-step, for adding a city to this benchmark:

| Step | Repo | Tool |
|------|------|------|
| Enumerate → thin → **detect** the city | auto-labeler | `main.py --source <mapillary\|gsv>` |
| Export the native-res bundle (`panos/` + `records.jsonl`) | auto-labeler | `scripts/export_benchmark.py` |
| GT-verify a sample → `verdicts.json` | **RampNet** | `scripts/gt_gallery.py benchmark/<city>` |
| Score P/R + Wilson CIs + threshold sweep | **RampNet** | `scripts/score_validation.py` / `rampnet.validation` |
| Add the split to the HF benchmark dataset | **RampNet** | `scripts/build_benchmark_dataset.py` |

⚠️ The last step lags: `build_benchmark_dataset.py` is still hardcoded to `bend` + `richmond`, so
clovis, morgantown, annapolis, paterson, gainesville, budapest, and sao_paulo are **not** in the published dataset — and it does not yet carry
`review_notes` or per-pano `note` into the parquet rows or the dataset card. Whoever finishes #21
should make the caveats travel with the data, since that is the audience most likely to read a
number with no idea how it was labeled.

The GT gallery and scorer are **canonical in RampNet** (`scripts/gt_gallery.py`,
`rampnet/validation.py` — decoupled from any imagery source, no network). The auto-labeler
still carries transitional copies (`scripts/spot_check_gallery.py`, `scripts/score_validation.py`)
marked for deletion; run the RampNet versions, not those.

## Current splits

| City | Source | Panos | Precision | Recall |
|------|--------|-------|-----------|--------|
| richmond | Mapillary 360 (iSTAR Pulsar + GoPro Max) | 124 | 0.960 | 0.765 |
| bend | GSV (Google Street View) | 110 | 0.954 | 0.758 |
| clovis | Mapillary 360 (GoPro Fusion) | 125 | 0.914 | 0.713 |
| morgantown | Mapillary 360 (GoPro Max) | 125 | 0.975 | 0.730 |
| annapolis | Mapillary 360 (Trimble MX7) | 125 | 0.964 | 0.728 |
| paterson | GSV (Google Street View) | 125 | 0.975 | 0.684 |
| gainesville | GSV (Google Street View) | 125 | 0.945 | 0.695 |
| budapest_district5 † | Mapillary 360 (GoPro Max) | 125 | 0.873 | 0.503 |
| sao_paulo ‡ | GSV (Google Street View) | 125 | 0.888 | 0.676 |

† **Budapest is not comparable to the seven US splits without its caveats** — the reviewer rated
their own pass low confidence and the rubric does not transfer cleanly. Read the section below
(or just run the scorer, which prints the warning first). It is a real signal, not a clean number.

‡ **São Paulo is the second non-US split**, reviewed at **high** confidence — unlike Budapest it
is comparable with context. It is held out of the pooled recommendation for geography, not GT
quality. Read its section below.

All nine city splits are **self-contained**: the reviewer's complete-scan attestation is baked into
`no_missed` (set on fully-judged panos with no missed marks), so the numbers reproduce with a
plain `python scripts/score_validation.py benchmark/<city>` — no `--assume-scanned` needed.
This matters because the recall gate otherwise excludes unconfirmed panos and biases recall
low (it over-weights panos where a miss *was* found).

**Every city split is stratified, so read the unbiased column too.** Each is sampled 5 top-detection /
N random / M empty (`sample.json` where the sampler wrote one; each pano's `benchmark_group` is in
`records.jsonl` and its `group` in `verdicts.json`). The table above is the all-panos figure
`score_validation.py` prints first; the **unbiased subset** — random + empty, dropping the 5
hand-picked high-density panos — is the honest between-city comparison, and
`score_validation.py` prints it second:

| City | Unbiased panos | Precision | Recall |
|------|----------------|-----------|--------|
| richmond | 119 | 0.961 | 0.740 |
| bend | 105 | 0.972 | 0.738 |
| clovis | 120 | 0.889 | 0.650 |
| morgantown | 120 | 0.969 | 0.684 |
| annapolis | 120 | 0.961 | 0.692 |
| paterson | 120 | 0.987 | 0.650 |
| gainesville | 120 | 0.943 | 0.647 |
| budapest_district5 † | 120 | 0.885 | 0.459 |
| sao_paulo ‡ | 120 | 0.869 | 0.626 |

Clovis is below the other cities on both metrics because it is 100% soft, 2018-era GoPro Fusion
360 imagery, where richmond mixes in the sharper NCTECH iSTAR Pulsar (camera provenance is in the
records, added in #50 and backfilled for morgantown/budapest in 2026-07-25). Note bend samples only
10 empty panos where the others take 25.

**Precision tracks the camera across the US Mapillary splits**, now that every split carries
`camera_make`/`camera_model`: clovis (100% GoPro Fusion, 2018) 0.914 → richmond (62% iSTAR Pulsar,
27% GoPro Max) 0.960 → annapolis (100% Trimble MX7, 2020+2023) 0.964 → morgantown (100% GoPro Max,
2024) 0.975. **Recall does not** — richmond 0.765 > morgantown 0.730 ≈ annapolis 0.728 > clovis
0.713 > gainesville 0.695 > paterson 0.684 — so sharper imagery buys fewer false positives more
reliably than it buys fewer misses. The two far-domain GSV splits are the sharpest counterexample:
the freshest, sharpest imagery in the benchmark, the highest precision (paterson), and the two
lowest US recalls.
Recall looks more sensitive to how far away and how dense the ramps are than to sensor sharpness,
and **annapolis is where that stops being a guess** — see its section below, which measures the
distance dependence directly.

**Budapest breaks that camera story, which is the point of including it.** It is 99% GoPro Max —
the same camera as morgantown, on fresher imagery (118 of 125 panos captured 2025-09 or later,
median quality 0.869 vs morgantown's 0.882) — yet it scores 0.873/0.503 against morgantown's
0.975/0.730. Sensor sharpness cannot explain a gap that large in the same direction on both
metrics. What is left is the city itself (out-of-distribution infrastructure for a US-trained
model) and the labeling rubric (a US-derived definition of "curb ramp" applied to a place it
doesn't fit). Those two are **confounded in this split** and this review cannot separate them.

**Morgantown is the precision high-water mark** — 0.975 over 200 judged detections, only 5 false
positives (on the unbiased subset it is a tie with bend, 0.969 vs 0.972). Every one of those 5 FPs
sits below confidence 0.75, so precision is a clean **1.000** at that threshold (at the cost of
recall 0.730 → 0.524). Its imagery is the newest and most uniform in the benchmark: 2024-era
Mapillary 360 shot entirely on a **GoPro Max** by a single contributor across 80 sequences,
uniformly 4096×2048 (3 panos at 5760×2880), median Mapillary quality score 0.882 —
legible enough that the reviewer abstained on only 4.3% of detections and 21.7% of missed marks —
the lowest missed-mark abstention of any split (on detections, bend's 1.9% and annapolis's 2.2%
are lower). Recall is the middling part of the story: 0.730 all-125 /
0.684 unbiased, between clovis and bend. Worth noting for the negative-sample check: all 25
`empty`-group panos held **no detections and no missed ramps**, i.e. the model's "nothing here"
was correct on every one.

## Annapolis — the split that shows misses are a distance problem

Annapolis (125 panos, P 0.964 / R 0.728; unbiased 0.961 / 0.692) is the first split shot on a
**survey-grade rig rather than a consumer action camera**: a vehicle-mounted **Trimble MX7**,
uniformly 8000×4000, one contributor across 101 sequences, median Mapillary quality 0.857, split
between two capture vintages (83 panos 2023-10, 42 panos 2020-05). It is also the densest city in
the benchmark by detection rate — **27.96% of the city's 53,232 panos carry at least one
detection**, against morgantown's 11.2%.

**That density is real, not a false-positive artifact.** This split existed to settle exactly that
question, and precision 0.964 settles it: only 8 of 222 judged detections were wrong. Annapolis is
a compact colonial grid plus the Naval Academy; it genuinely has more curb ramps per pano.

The imagery is the most legible of the Mapillary splits. The reviewer abstained on just **2.2% of
detections (5 of 227)** — beating morgantown's 4.3%; only bend, the GSV split, is lower at 1.9%.

**The finding worth carrying elsewhere: the model's misses are overwhelmingly far away.** Treating
the equirectangular elevation of each point as a ground distance (assuming a ~2.5 m camera mount):

| | n | median est. distance |
|---|---|---|
| True detections | 214 | **11.5 m** |
| Missed ramps (confident) | 80 | **20.4 m** |
| Missed ramps (unsure) | 34 | 24.3 m |

**80% of confidently-missed ramps lie beyond the median distance of a successful detection**, and
97% of the unsure ones do. This ratio does not depend on the assumed camera height — changing it
rescales every distance by the same factor, so only the metre labels move. Restricting to ramps
within ~12.5 m, recall rises from **0.733 to 0.870 at essentially unchanged precision (0.958)**.

This reframes the ~0.70 recall that every city split reports. It is substantially a *viewpoint*
limitation rather than a detection-quality one: a ramp 20–40 m down the street is missed, and the
same ramp is found once the vehicle drives closer. The practical consequence belongs to the
deployment repo — **multi-view aggregation across the un-thinned pano stream should recover most of
this**, since the city run has 53,232 panos and this benchmark deliberately thins to 30 m spacing
so the same ramp never appears twice. The gain lives on the *distance* axis, not the confidence
axis: in the near field, raising the threshold to 0.90 gives precision 1.000 but recall 0.282, so
"seen close in some view" has headroom that "seen confidently in some view" does not. Measuring the
real gain needs a split this benchmark does not yet have — a dense, un-thinned corridor with ground
truth at the *ramp* level in world coordinates rather than per pano.

Smaller notes:

- **Negative check: 21 of 25 `empty`-group panos were clean.** The other 4 held 10 real ramps the
  reviewer marked — but **zero false detections**. The model was never wrong when it fired on these
  panos, only silent, which is the same far-field story.
- **Vintage is a non-finding.** 2020-05 scores P 0.987 / R 0.704 and 2023-10 scores P 0.952 /
  R 0.742, but the counts are small and the intervals overlap. Detection density is near-identical
  across the two (1.86 vs 1.80 per pano), so vintage is not confounded with sampling.
- **3 duplicate marks** (2 in the unbiased subset). Scored as false positives by default; with
  `--lenient-duplicates` precision is 0.977 all-panos / 0.972 unbiased.
- **How the distances above were computed — no depth estimation is involved.** An equirectangular
  pano maps the vertical axis linearly to elevation, so a point at `y` sits at depression
  `(y - 0.5)·180°` below the horizon, and on flat ground a camera at height `H` sees it at
  `H / tan(θ)`. The whole model is that one line, with `H` assumed to be 2.5 m.
  - `H` **cancels out of every comparison here** — it scales all distances linearly, so the
    hit/miss ratio is 1.79 whether the mount is 1.8 m or 3.5 m. Only the metre labels move.
  - **The metre labels are soft near the horizon, which is exactly where the misses sit.**
    `H/tan(θ)` is steep there: at `y = 0.51` a shift of 0.01 (≈20 px in the 2048-tall render)
    takes the estimate from 79.6 m to 39.7 m. Read "20.4 m" as *far*, not as a measurement.
  - **The conclusion does not depend on any of that**, because `H/tan((y-0.5)π)` is strictly
    monotonic in `y` — so the claims are rank statements that survive any monotonic distance
    model, including a correct one. Testing raw `y` with no distance model at all: Mann-Whitney
    **z = -5.69** (p < 1e-7), and **P(a random missed ramp sits closer to the horizon than a
    random detected one) = 0.716**. The 80% / 97% figures above are likewise rank statements.
    Only the cutoff table depends on the metres, and only for where the cutoffs sit.
  - **Unverified assumptions**: `camera_pitch` and `camera_roll` are `null` on all 125 panos, so
    the horizon is assumed to be exactly at `y = 0.5` (plausible — Mapillary equirects are
    gravity-aligned at stitch time and the MX7 is a fixed mount — but unchecked), and the ground
    is assumed flat. Annapolis is coastal and fairly flat, which is why this is defensible here;
    **do not reuse this method on morgantown's hillsides** without accounting for grade.
- **This split carries no `review_notes`.** Unlike budapest, nothing about the rubric fought back
  here, but the reviewer's own confidence rating is not on the record — worth adding on a
  re-review.

## Paterson — the second GSV city, and the split whose misses aren't under-confident

Paterson, NJ (125 panos, P 0.975 / R 0.684; unbiased **0.987 / 0.650**, reviewer-rated
confidence **HIGH**) was added 2026-07-29 for one job: bend, until now the only GSV split, is
also a Stage-2 **training** city, so "GSV" and "in-domain" were confounded in every GSV
number. Paterson is registry-clean (`docs/data_provenance.md`), out-of-training but
NYC-metro-adjacent — recorded at Phase 0 as "second GSV city; out-of-training but
metro-adjacent", **not** a geographic-transfer test. It settled the confound immediately: the
GSV tier's F1-optimum moved from bend's outlier 0.50 to 0.26 with paterson pooled in
(`docs/operating_point.md`). It is also the first split that is a live Project Sidewalk
deployment (`sidewalk-paterson.cs.washington.edu`), whose regions API supplied the boundary —
68 neighborhood polygons dissolving to ~22.5 km², essentially the whole municipality — so a
future crowd-agree-rate comparison shares the benchmark's exact footprint by construction.

The imagery is GSV without camera provenance (structural for the GSV path — the split relies
on its `source` field for tier assignment, like bend). Capture dates skew fresh (85/125 from
2024–2025, mirroring ~72% city-wide) with two 2007/2008 gen-1 panos (3328×1664) that made the
sample honestly. Whole-city context: 34,427 panos, of which **31.7% carry at least one
detection — the densest city in the benchmark** (annapolis: 27.96%), and precision 0.987 says
that density is real ramps, not noise. The reviewer abstained on 2.5% of detections (7 of
284) — the normal band — and the negative check is the strongest of any split: 24 of 25
`empty`-group panos attested clean, with a single missed mark across the stratum.

**Precision is the benchmark's high-water mark; recall is its floor, and for a structural
reason.** Two reviewer-documented populations drive the misses (`review_notes`, printed by
the scorer before the numbers):

- **Paired tactile indicators.** Many Paterson corners carry *two* TSIs almost side by side,
  offset ~60–90°, one per crossing direction; each was marked as its own ramp. The model
  typically fires once per such corner, so the partner lands as a miss — a recall effect,
  not a precision one (this split produced exactly 1 duplicate verdict, against budapest's
  7). No other split has this corner style.
- **Far-field ramps.** Recall at the deployed threshold falls from 0.794 (<12.5 m) through
  0.707 (12.5–25 m) to **0.432 (>25 m)** — the annapolis distance finding, reproduced on GSV.

What makes paterson analytically different: those misses mostly produce **no candidate at any
confidence**. Its recall ceiling at the 0.05 extraction floor is **0.757**, against 0.88–0.94
for every other US split, and only 1 GT ramp sits in the storage-floor band (others: 6–9). So
threshold-lowering, storage-floor changes and flip-TTA are all the wrong levers here; the open
question — whether `peak_local_max`'s `min_distance=10` is *suppressing* the paired-TSI
partner peak rather than the model never firing — is logged in `docs/operating_point.md`
("What would change this") and is untested. The #55 spot-check found only 10 incremental FPs
in `[0.25, 0.55)` (other US cities: 23–30), with an A-rate of 20% — the same shallow
threshold response, measured a second way.

## Gainesville — same far-domain recall as paterson, opposite mechanism

Gainesville, FL (125 panos, P 0.945 / R 0.695; unbiased **0.943 / 0.647**, reviewer-rated
confidence **HIGH**) was added 2026-07-30 to answer the question paterson raised: is the
~0.65 far-domain GSV recall *fabric-specific* (paterson's paired tactile indicators) or
*generic to out-of-domain GSV*? Gainesville is the first far-domain clean GSV city — no
Stage-2 training city is anywhere near Florida — with conventional street fabric, sampled
from the dissolved Project Sidewalk regions polygon (~47.6 km² deployment footprint, not
the ~160 km² municipality). The imagery is the freshest in the benchmark: 105 of 125 panos
captured 2024 or later (88 from 2026), 115 at gen-4 16384×8192, with a long tail back to
2011 that made the sample honestly.

**The recall number replicates — 0.647 vs paterson's 0.650 — with none of paterson's
fabric.** Two far-domain GSV cities now independently land at ~0.65, so the deficit is the
domain gap, not New Jersey's corners. But the *mechanism* is the opposite
(`docs/operating_point.md`): gainesville's recall ceiling at the 0.05 extraction floor is
**0.890**, squarely in the normal US band (0.86–0.91), against paterson's structural 0.757.
Paterson's misses produce no candidate at any confidence; gainesville's misses **fire below
threshold** — so threshold-lowering, the wrong lever in paterson, is the right one here:
0.55 → 0.30 buys gainesville **+9.9 recall points** (0.673 → 0.772 on the extraction
cache), nearly 3× paterson's +3.8. The reviewer's two miss impressions fit weak-activation
failures: **significant debris sitting on ramps** (a surface-legibility failure, new to the
benchmark — a candidate bucket for the #46 failure taxonomy) and **distance** — measured at
R 0.778 near / 0.740 mid / **0.300 far (>25 m)**, the worst far band of any US split
(paterson: 0.432).

Precision is the more ordinary story, and the reviewer's impression that it slipped is
correct but structured: 9 FPs + 2 duplicates (P 0.943 unbiased, lowest of any clean-imagery
US split), yet the FPs hug the review floor — four of nine within 0.02 of the 0.55 cutoff,
FP median confidence 0.623 vs 0.813 for TPs, and a clean 1.000 from 0.80 up. The flip side
of that floor-hugging: the sub-0.55 band is dense. The #55 spot-check gallery held **34**
incremental FPs in `[0.25, 0.55)` — above the 23–30 US precedent and 3× paterson's 10 — and
its A-rate came back the **highest of any split: 35.3%** (12 A / 21 B / 1 unsure, jonf
2026-07-30). Over a third of the "false positives" a lower threshold adds here are real
ramps the GT missed; corrected precision at the recommended 0.30 is 0.886 (from 0.857 raw),
which lands just above clovis's 0.883 — so gainesville did not displace clovis as the
benchmark's binding split (`docs/operating_point.md`).

Smaller notes: ramp styles were **new to the benchmark** — many wide intersections, and
large diagonal curb ramps spanning both crossing directions at new or renovated
intersections (unlike budapest's diagonal aprons, these did not fight the rubric; the split
produced only 2 duplicate verdicts). Abstention was 2.4% of detections (5 of 205), the
normal band. Negative check: 23 of 25 `empty`-group panos attested clean; the other 2 held
4 confidently-missed real ramps and zero false detections — the annapolis pattern. Recall
pool is all 125 panos. Whole-city detection-rate context is not quoted here because the
auto-labeler run summary was not exported with the bundle; it lives in
`sidewalk-auto-labeler`'s run log.

## Budapest District V — the split whose ground truth is itself uncertain

**Read this before quoting 0.873 / 0.503 anywhere.** Budapest is the first non-US split and the
first one where *the rubric*, not the imagery or the model, is the dominant source of doubt. The
reviewer (jonf, 2026-07-27, single pass, no second rater) rated their own confidence **low** and
asked for that to be on the record. The full first-person account lives in the split's
`review_notes` — `python scripts/score_validation.py benchmark/budapest_district5` prints it above
the numbers. The short version:

- **Sweeping diagonal corner ramps.** Budapest corners frequently carry one broad ramped apron
  spanning much of the corner, far larger than a US curb ramp and serving two crossing directions
  at once. The reviewer generally marked these as **two** ramps, one per direction of travel — a
  *rubric decision, not an observation*. RampNet often did the same thing unprompted, which is
  interesting on its own. Where a second detection looked purely redundant it was marked
  `duplicate` instead, so the same physical geometry did not always get scored the same way.
- **That call is worth ~4 points of precision**: 0.873 (duplicates as false positives, the default)
  vs 0.910 (`--lenient-duplicates`). Budapest carries **7 duplicate marks** where richmond and bend
  have 1 each and clovis and morgantown have none — the quantitative fingerprint of the ambiguity.
  6 of the 7 land in the 5 hand-picked `top` panos, so the unbiased subset barely moves
  (0.885 → 0.891). Say which scoring you used.
- **"Curb ramp" vs. "intended pedestrian path."** Many District V surfaces are genuinely ambiguous
  between a curb ramp and a continuously graded pedestrian route that never presents a curb at all.
  The US-derived rubric does not decide these, and the reviewer's line between them likely drifted
  over the session.
- **Highest abstention of any split** — the honest shadow of the two points above: 16 of 189
  detections (8.5%) and 48 of 197 missed marks (24.4%) were marked unsure and abstain from both
  metrics. Compare morgantown's 4.3% / 21.7%.
- **Recall 0.503 rests on 149 confident missed marks**, roughly double any other split (bend 79,
  richmond 73, morgantown 72, clovis 56). Some of that is a real domain gap; some is the reviewer
  counting more things as ramps. This pass cannot separate them.

What Budapest **is** good for: a genuine out-of-distribution stress test, and the clearest evidence
in the benchmark that RampNet's US-trained performance does not transfer wholesale. What it is
**not** good for: a precise number, or a row placed beside the US cities without this context. The
two things that would fix it are a Budapest-specific rubric written from Hungarian street design
rather than adapted from the US one, and a **second independent rater** on the same 125 panos to
measure how much of the gap is the model and how much is the labeler. Of every split here, this is
the one most in need of a second opinion.

## São Paulo — the second non-US split, and the one where the rubric fits

`benchmark/sao_paulo/` (issue #98) exists to give the non-US axis a second point — and to
de-confound the first. Budapest is non-US *and* Mapillary/GoPro Max; São Paulo is non-US on
GSV, the same imagery path as bend/paterson/gainesville. The footprint is four central
districts — Brás plus its Centro-side neighbours Sé, Cambuci and Bom Retiro (13.95 km², sampled
from a 22,741-pano run) — not the whole municipality; Brazilian curb ramps and tactile paving
follow the NBR 9050 standard, a design vocabulary the model never saw in training.

**The headline is what did *not* happen: the Budapest recall collapse did not replicate.** São
Paulo scores 0.888 / 0.676 all-panos (0.869 / 0.626 unbiased) at **high** reviewer confidence —
"relatively easy to assess" (jonf, 2026-08-01, single pass). Its unbiased recall sits inside the
US GSV band (paterson 0.650, gainesville 0.647), not near Budapest's 0.503. With this pair the
non-US question splits cleanly: unfamiliar *infrastructure vocabulary* alone did not break the
model or the rubric here; Budapest's uncertainty localizes to its named ambiguities — the
full-corner diagonal aprons and the level path-street seams (reviewer retrospective, issue #74).
Only 1 duplicate mark against Budapest's 7 is the quantitative echo: São Paulo's ramp geometry
did not fight the one-point-per-ramp rubric.

Two mechanisms did fight the review, both recorded in `review_notes`:

- **Mid-block captures.** In more than 10 panos the camera sat mid-segment, so judging curb
  ramps meant looking back at the nearest intersection — often too far away to see clearly.
  This drives the benchmark's highest abstention: 14.7% of detections and 32.1% of missed
  marks are `unsure` (Budapest: 8.5% / 24.4%). Given the high reviewer confidence these read
  as honest distance-driven abstentions, not rubric failure — the far-field problem measured
  in the annapolis section, arriving through sampling geometry rather than optics.
- **White-painted curbs.** A set of intersections carry white paint along the curb but rarely
  have actual curb ramps, even where crosswalks are present. The model fired on these
  repeatedly (reviewer impression, not yet a measurement), and the reviewer was initially
  confused too before reconciling on the pattern. A candidate split-specific false-positive
  mechanism — likely a chunk of why precision (0.869 unbiased) is the lowest of any
  confident-GT split — worth a pass in the FP taxonomy (#46). The #55 A/B pass points the
  same way: sao_paulo's incremental FPs in `[0.25, 0.55)` are overwhelmingly *genuine* (38 of
  48 tagged B; A-rate 12.5%, the lowest of any split), so the low precision is the model
  mis-firing on real distractors, not the ground truth missing ramps.

The negative check behaves: 20 of the 25 `empty`-stratum panos were attested clean, and the
other 5 held 11 missed ramps — the annapolis-shaped result that the model's "nothing here" is
mostly, not entirely, right. A live Project Sidewalk deployment
(sidewalk-sao-paulo.cs.washington.edu) sits inside the footprint, so an agree-rate comparison
is possible later; it covers Brás only and is 8.2% audited, so that comparison would be thin.

## The manual_gold split (in-distribution gold reference)

`benchmark/manual_gold/` (issue #58) is a different kind of split: its ground truth is the
repo's 1,000-pano manually labeled gold set (`manual_labels/*.txt` — 3,919 curb ramps, 207
negative panos), labeled **independently of any model**. Unlike the city splits it is not
derived from reviewing RampNet's detections, so it carries no RampNet anchoring — and at 4×
the size of the largest city split it gives every model tighter confidence intervals. It is
also **in-distribution**: GSV imagery from the training cities (NYC / Portland / Bend), held
out of Stage-2 training as the HF dataset's test split. Treat it as the in-domain reference
that complements the deployment cities above, never as their replacement. It shares zero
panos with bend, richmond, or clovis (verified by `scripts/fetch_manual_gold.py --audit`).

The layout differs accordingly — there is no verdict review:

```
benchmark/manual_gold/
  gt_source.json    points at manual_labels/ (GT = YOLO box centers, no ignore points,
                    every pano recall-confirmed)
  records.jsonl     pano metadata + RampNet detections (built by the two scripts below)
  bundle_meta.json  which source built the imagery, and when (committed)
  panos/            imagery from the HF test split (git-ignored, like every split)
```

No verdicts means `scripts/score_validation.py` and `scripts/gt_gallery.py` do **not** apply
here; the split is scored by the model-comparison harness only:

```
python scripts/fetch_manual_gold.py --audit       # id membership/overlap audit, no download
python scripts/fetch_manual_gold.py --images-only # imagery for THIS machine (run on Hyak)
python scripts/export_gold_records.py --checkpoint <stage2.pth>   # RampNet detections + gate
python scripts/model_comparison/compare.py benchmark/manual_gold --models rampnet --op-threshold 0.55
```

`--images-only` is the fetch to run on a fresh clone, and `scripts/run_gold_bundle.slurm`
runs it for you. The bundle mixes two lifecycles: `records.jsonl` and `bundle_meta.json` are
**committed** (the records carry the exported detections), while `panos/` is git-ignored, so
the imagery is normally the only missing piece. `--images-only` fetches it and writes nothing
committed; it skips panos already on disk that match `records.jsonl`, so a preempted run
resumes rather than starting over. A bare `python scripts/fetch_manual_gold.py` is the
*first* build only — with `records.jsonl` present it refuses, and `--force` would rebuild the
records and **discard the detections**.

Two caveats travel with that fetch:

- **Cost, measured 2026-08-14 on makelab2:** the `hf` path goes through `load_dataset`, which
  downloaded and arrow-materialized **all three splits, ~2.5 h end to end**, despite
  `split="test"` — not the "~44 GB test split only" this section previously stated. A
  shard-scoped fetch via `HfFileSystem` + pyarrow (what `--audit` already does) would cut
  that; it has not been done, deliberately, to keep the change away from the
  byte-fidelity-sensitive read path.
- **No content hash yet.** Every city split carries `benchmark/<city>/imagery_manifest.json`
  (sha256 per pano, from `scripts/analysis/imagery_manifest.py`); `manual_gold` does **not**,
  because nobody has run the writer on a machine holding all 1,000 panos. The fetch checks
  the manifest when it exists and otherwise prints the command that writes it, so until then
  this split's imagery is verified by `bundle_meta.json`'s recorded source and each pano's
  pixel size in `records.jsonl` — weaker than the other nine. Writing that manifest is the
  open item.

The exporter ends with a reproduction gate against the published gold-set numbers
(P 0.949 / R 0.873 @ conf >= 0.55, TTA). Read the manual-gold section of
`docs/model_comparison.md` before quoting numbers from this split.

**All eight city splits were reviewed at model resolution** with the pan/zoom labeler (`scripts/gt_gallery.py`),
which shows the full pano at the model's input resolution (4096×2048) with pan/zoom, rather than a
downscaled overview. For richmond and bend this was a *re-review*: reviewing at model resolution
surfaced genuinely-missed ramps that the earlier 1600 px overview hid — small/distant curb ramps a
reviewer literally could not resolve — correcting recall down from earlier, optimistic numbers
(richmond 0.895 → 0.765, bend 0.831 → 0.758). Precision was essentially unchanged (the zoom mostly
resolved `unsure` detections, not misclassifications). The correction is consistent across both
imagery sources (GSV and Mapillary), and these are the honest per-pano-comprehensive figures; clovis,
morgantown, annapolis, and budapest were reviewed at model resolution from the start. (That
correction — the ramps the overview hid were the *small and distant* ones — is the same effect the
annapolis section later measures directly: misses are predominantly far-field.) Richmond and bend each
include one `duplicate` verdict — a redundant second detection on one physical ramp, scored as a
false positive by default (`--lenient-duplicates` scores the other way; see
`scripts/score_validation.py`); clovis and morgantown have none, annapolis has 3, paterson has 1,
gainesville has 2, and budapest has 7 (see its section above — there the duplicate call is a live
rubric question, not a stray click).
