# `crosswalk_s8`: Juneau in training, a held-out Juneau slice (RampNet#298)

**DRAFT pre-registration for Jon to approve. Not approved.** Nothing has been trained, inferred, rated, posted or
pushed. The prep artefacts below (input snapshot, slice, candidate pools, card and power estimates) are CPU-only
and change no model. Drafted 2026-11-17 by Claude Opus 5.5 (`claude-opus-5-5`) from Jon's decisions of the same
day; reviewed the same day by Claude Fable 5.2 (`claude-fable-5-2`, `review_s8_fable.md`, outside the repo), and
the parent session's consensus of that review is applied here (it replaced the first draft's validator audit, made
the buffer 60 m, and added the roof-free condition, the GoPro balance and the automatic slice checks). Once Jon
approves, this text is fixed; later changes go in dated amendments below it.

Branch `crosswalk-s8`, based on `crosswalk-supervision-s7` (the stack is main ← `stage1-label-type-274` ←
`crosswalk-followups-285` ← `crosswalk-supervision-s7`). The name `crosswalk_s7` is taken by the stopped
supervision plan (`docs/crosswalk_supervision_s7.md`).

## Summary

- **Goal.** Cut the look-alikes `crosswalk_s4` sends on Mapillary in Juneau without losing crosswalks, Seattle,
  brick crossings or the Laurens car-top fix.
- **Data.** `root_v8` = `root_v4` + Juneau train panos (at most 331: 255 new, 76 from `root_v5`) + up to 100
  brick-sidewalk negatives on `root_v4`'s brick panos. Every Juneau negative that trains is Jon's verdict: 148
  already exist, and up to 125 more come from one blind page of validator-mined candidates. Juneau positives never
  outnumber Juneau negatives: 210 each if half the page's cards come back "no crosswalk", 273 each if all do.
- **Held out.** Three seeded Juneau areas holding 437 of the 3,579 AI labels (12.2%) on 989 panos, with a 60 m
  buffer (546 panos, 311 labels) that neither trains nor scores.
- **Primary.** On a seeded sample of 200 slice panos, look-alikes per 100 panos, three-seed mean of s8 minus
  three-seed mean of s4, each seed at its recall-matched threshold, from Jon's blind ratings. Met if the 95%
  pano-bootstrap interval is below 0, seed 1's own difference is below 0, **and** the same three-seed difference
  without capture-vehicle cards has a point estimate below 0.
- **Guards.** Five automatic (test F1, (b′) recall, Columbia brick recall, Seattle retention, Laurens car-top
  false alarms), one automatic slice check (s8 keeps the slice crosswalks Jon already agreed with), and one rated
  (crosswalks found on the slice sample). A rated Seattle look-alike guard is added only if an automatic trigger
  fires.
- **Jon's rating.** About 17 to 29 minutes before training (the mined-negative page, Claude's dropped negatives,
  a brick page that may be skipped) and about 28 to 33 minutes after, only if every automatic check holds; 6 to 10
  more minutes if the Seattle trigger fires. Total about 45 to 62 minutes.
- **Why the validator only mines.** Five of the six real crosswalks the validator confidently calls "incorrect" on
  Jon's earlier blind pages are Juneau's red decorative-concrete crossings (section 3). Validator negatives would
  teach exactly the crossing class guard 3 protects, so none trains without Jon's verdict.

## Why

`crosswalk_s4` (seed 1, HF `rampnet-crosswalk@ebad379b5f7c`, threshold 0.23, range gate 5.625°) sent 3,579 AI
Crosswalk labels to juneau-test on 2026-11-17 (Mapillary; Trimble MX7 about 92% of panos, GoPro Max about 8%).
On the seeded random sample Jon voted on, precision as sent is **0.72 [0.65, 0.79]**
(`random_sample_2026-11-17.json`, scored in the validator report). Jon's comments name the false-positive
classes: brick or paver sidewalk (the largest; likely a side effect of s4's 548 brick-crosswalk panos), the
capture car's own roof (GoPro Max and Trimble), shadows, stairs, curb ramps, turn-lane arrows, parking
access-aisle stripes, chalk or paint, bike-lane marks and driveways.

This was predicted. s4's adoption read the Juneau confirmation draw as 5.8 → 20.8 look-alikes per 100 panos and
called it "the first thing to fix before s4 labels go live anywhere" (`docs/stage1_crosswalk_sdot_274.md`). What
the later rounds showed, and what this plan takes from each:

- **s3** (166 Jon-rated hard negatives) cut Seattle look-alikes from 13.3 to 5.0 per 100 panos. Negatives work.
- **s4** (124 Laurens car-top negatives) cut Laurens Mapillary look-alikes from 41.2 to 11.2. Rig-specific
  negatives work for that rig, which is also why a roof fix alone cannot be the whole result here (section 6).
- **s5** (164 Juneau positives, 53 negatives, 3:1) found 49% more real Juneau crosswalks but left look-alikes
  flat (−2.5 [−10.8, +5.0]). Positives buy recall, not precision.
- **s6** (3,040 auto-placed sparse positives) raised Juneau recall but multiplied false alarms on
  confirmed-empty panos (guard 6: 6.7 → 112). Many sparse positives can shift the model toward firing.
- **s7** stopped at its leak check: whole-frame background taught where s4, OSM and Vistas are silent leaked
  12.4% of Jon's missed crosswalks. So s8 keeps `sparse` supervision and teaches background only at points a
  person verified.

Tracked as RampNet#298.

## Jon's decisions (2026-11-17) that this plan implements

1. **Juneau goes into training**, reversing the 2026-11-12 hold-out, except a small geographic held-out slice
   (a few whole areas, seeded, with a buffer) that carries the primary. Two Mapillary cities launching on PS in
   the coming weeks will be later held-out tests; they are not available to this decision.
2. **Negatives:** (a) Jon's Disagree labels outside the slice; (b) the production Crosswalk validator's confident
   "incorrect" verdicts, trusted only after Jon checks them; (c) brick **sidewalk** negatives paired with the
   brick-crosswalk positives, without touching the Columbia brick test. Jon's (b) asked for a ~60-card audit at a
   0.95 bar; under the consensus the validator only mines and Jon rates every validator negative that trains
   (section 3 says why the audit was dropped).
3. **Positives:** Jon's Agree labels plus the validator's confident "correct" verdicts, outside the slice. Under
   the consensus, validator positives are used only on GoPro Max panos, to balance the GoPro negatives.
4. **Evaluation is human-rated, never validator-scored** (the validator chose training data). Three seeds,
   recall-matched thresholds against s4's seeds, s4/s6 guards reused, and a primary on the held-out slice from
   Jon's blind ratings, kept cheap to rate.

## 1. The held-out slice

**Rule** (`s8_prep.py slice`, seed 298, fixed before any training or rating; outputs in
`analysis_out/crosswalk_s8/slice/`):

- Strata: 150 m cells over the 5,333 Juneau panos (both runs), ranked by AI-label count. `dense` = the fewest
  cells that hold half the labels (downtown); `other` = every other cell.
- Three areas. Area 0's centre is a pano drawn uniformly from `dense`, areas 1 and 2 from `other`, each centre at
  least 400 m from any pano already in the slice. An area takes the nearest unsliced panos in order of distance
  until it holds one third of 12% of all AI labels, then every pano within that final radius.
- **Buffer: every pano within 60 m of a slice pano** (50 m in the first draft; widened because Juneau is hilly and
  the Trimble's camera height is unverified: at 3 m the range gate reaches 30 m). Buffer panos neither train nor
  score. The 10 extra metres cost 117 panos and 6 planned points; the slice and the primary sample are unchanged.
- **Label-position rule:** a training label whose PS-stored position lies within 35 m of a slice pano is
  excluded (it could be a crosswalk a slice pano sees). It removes none: the nearest training label is 44.9 m away.

Why this shape: Juneau's labels are very clustered (one 300 m cell holds 27% of them), so whole grid cells
either overshoot the target or leave the densest core unselectable. Areas grown to a label count give a fixed
size, stay contiguous (a buffer around one disc costs less than around scattered blocks), and keep downtown,
where the brick sidewalks are, in play. The areas were drawn without looking at votes or verdicts. The random
sample's 10 labels in the slice are there by chance, not by steering.

| Area | Stratum | Centre pano | Radius | Panos | AI labels |
|---|---|---|---|---|---|
| 0 | dense | `3656210694644148` (58.30333, −134.40857) | 79 m | 133 | 146 |
| 1 | other | `428884450036407` (58.30339, −134.41736) | 186 m | 743 | 145 |
| 2 | other | `726578559457461` (58.30620, −134.43070) | 76 m | 113 (56 GoPro Max) | 146 |

| | Slice | Buffer (60 m) | Training area |
|---|---|---|---|
| Panos | 989 | 546 | 3,798 |
| Panos with an AI label | 224 | 134 | 1,022 |
| AI labels (GoPro Max) | 437 (87) | 311 (9) | 2,831 (196) |
| Validator verdict Agree / Disagree | 294 / 143 | 251 / 60 | 1,963 / 868 |
| Jon's juneau-test votes Agree / Disagree / Unsure | 50 / 13 / 0 | 10 / 4 / 0 | 148 / 119 / 4 |
| Random-sample labels (voted) | 10 (9) | 9 (9) | 61 (61) |
| recall285 panos / sure marks within 25 m | 11 / 19 | 10 / 18 | 59 / 73 |
| Confirmation / fresh-eval / s5-mining draw panos | 33 / 14 / 52 | 7 / 15 / 33 | 80 / 91 / 215 |

**Expected contents of the slice** (rough sizes from city-wide rates, not measurements of the slice): 437 s4
labels, about 315 real crosswalk labels and 120 look-alikes at the city's 0.72 precision (the validator's Agree
share in the slice is 0.67, a little below the city's 0.70); about 550 visible crosswalks within 25 m if s4's
Juneau recall of 0.57 holds there. 36 of the slice's 141 look-alike stand-ins (Jon's vote, else the validator's
verdict) are rear-seam, the capture-car position.

**The primary's pano sample.** 200 panos drawn uniformly from the 989 (seed 299), plus a seeded reserve of 50
for panos that turn out unusable (`primary_sample.txt`, `primary_reserve.txt`). Drawn before any s8 checkpoint.
It holds 88 s4 labels on 45 panos and 25 look-alike stand-ins on 19 panos (8 rear-seam). **Its mix is lopsided by
chance:** 152 of the 200 panos are from area 1, 27 from area 0 and 21 from area 2, and only 9 are GoPro Max (18 of
the 88 labels). Camera-split results on the sample will be nearly empty; the GoPro check is the automatic count on
all 56 GoPro slice panos (section 6).

### Leakage controls

- **Spatial:** the slice plus the 60 m buffer, plus the 35 m label-position rule. The range gate keeps every
  sent label at least 5.625° below the horizon (about 25 m on flat ground from a 2.5 m camera, about 30 m from a
  3 m camera; the shallowest sent label is at 5.985°). A crosswalk within 30 m of a slice pano is therefore more
  than 30 m from every training pano, so it cannot be a training point there, and `sparse` supervision puts no
  loss on pixels away from the training points. Steep slopes can still loosen this.
- **Sequence:** excluding whole capture sequences is not possible. Juneau has 41 sequences, 22 of them cross the
  slice, and 2,264 of the 3,798 training-area panos share a sequence with it. Shared sequences mean the same rig,
  day and light; the same physical crosswalks are kept apart by the buffer. The one object the buffer cannot
  separate is the capture car itself, which travels with the camera (`eLOsiMypD8rIQc0qlHVAPg` puts rear-seam
  look-alikes in the slice and rear-seam negatives in training: the same roof). That is why "met" needs a
  roof-free part (section 6).
- **recall285 marks** (Jon's missed-crosswalk clicks) stay measurement-only, as approved on 2026-11-15: the 80
  Juneau recall285 panos carry no training point (116 AI labels on them are left out), and neither do the panos
  with live PS Crosswalk labels.
- **Seattle:** no new Seattle pano; val, test and (b′) files byte for byte as `root_v4`; the 35 m Phase 1 rule
  stays in `add-panos`.
- **Columbia SC:** no `columbia-sc` pano enters training in any role, and the brick-sidewalk negatives come only
  from the non-Columbia brick panos already in `root_v4` (section 4). `add-panos --exclude` gets the Columbia
  pano list as a check.
- **Laurens:** no new Laurens pano. The 92 confirmed-empty Laurens panos keep the known limit A4 stated (37
  within 30 m of a `root_v4` Laurens negative pano), the same for both models.

### What happens to the earlier Juneau evaluation sets

Training on Juneau contaminates every earlier Juneau check that lies in the training area, so they move to
"reported, deciding nothing":

- **Juneau recall on the 110 marks (A4's primary):** 73 of the marks are on training-area panos, whose
  neighbours train. Reported on all 110 and, separately, on the 19 inside the slice (too few to gate).
- **Guard 6's 27 Juneau empty panos:** 22 are in the training area. Reported only; the Laurens 92 carry the
  car-top guard.
- **Fresh Juneau draw (s5 guard 5) and the confirmation draw:** most of their panos are in the training area, and
  Jon's blind verdicts on their s4 cards become training labels (section 2). Not used for evaluation.

## 2. Training data, `root_v8`

```
root_v8 = root_v4                                   (every val, test, (b′) file byte-identical)
        + brick-sidewalk negative_points on root_v4 brick train panos   (≤ 100; section 4; may be skipped)
        + root_v5's Juneau mining panos in the training area           (76 panos: 128 positives, 42 negatives,
                                                                         their ignore points; JSONs reused)
        + new Juneau train panos from the PS AI labels                  (≤ 255 panos; table below)
```

`root_v5` itself is not the base: its Juneau panos in the slice and buffer must not train.

**Vote accounts (decision 1).** The vote snapshot has two accounts. A read-only lookup of
`sidewalk_login.sidewalk_user` on the test instance (2026-11-17, parent session) gives
`2d8335ef-daff-4382-afe2-7e1eb6567fd9` = `jonfroehlich` (114 votes, 74 of the 79 random-sample votes) and
`4ff63c52-88d6-40bb-ad08-ceff4e14f8bc` = `29vs30oz1c7daxjc` (236 votes, 16:20 to 16:44, before the other
session; an auto-generated name, very likely Jon voting while not logged in). Prep uses both. `candidates.csv`
carries `vote_user_id`, `load_inputs(accounts=…)` keeps the latest vote per label among the accounts it is
given, and votes are frozen at the 2026-11-17 snapshot (no later juneau-test vote is carried). **No root is built
until Jon confirms the second account is his;** if it is not, its 184 training-area votes drop out mechanically.

**Sources of Juneau training points,** by priority on each AI label:

1. A `root_v5` pano keeps exactly its own points (Jon-rated under the disc rule, reviewed by Claude, 2026-11-09).
   PS-derived points on those panos are dropped (158 labels).
2. **Jon's verdict:** his juneau-test vote (latest per label), else his blind verdict on an earlier page's s4 card
   within 0.01 of the label (`confirm_s3/juneau/s4`, `s5_inputs/juneau_eval/s4`). Agree / `crosswalk` → positive;
   Disagree / `not_crosswalk` → negative; Unsure → ignore point.
3. **Validator-mined negative candidates** (unrated labels with `p_incorrect > 0.9999` and at least 8.4375° below
   the horizon, about 17 m and nearer) go to **Jon's blind page** (below). They never train on the validator's word.
4. **Validator positives** (`p_correct > 0.9999`) only on GoPro Max panos, only as many as the GoPro balance
   needs (below), trusted with Claude's drop-only review. The validator agrees with Jon on 165 of 166 rated labels
   at this cut and on all 31 random-sample votes.
5. Everything else on a training pano → ignore point.

**The mined-negative page.** 125 candidates (seeded order, after the caps below), one uniform blind page: the
cross and the negative's disc are drawn on **every** card, so the page reveals nothing about the source, with
three buttons: *crosswalk under the cross* → positive; *no crosswalk in the disc* → negative; *crosswalk in the
disc but not under the cross, or unsure* → ignore point. Definition `paint_brick`; one-click class tags (capture
vehicle, brick or paver sidewalk, decorative crossing, curb ramp, lane marking, shadow, other). No AI pre-screen.

**How a label becomes a training target.** Positives are `sparse` points at the AI label's pixel (normalized
from the 8000×4000 store pixel; the same point on the 4096×2048 prep). Negatives are
`supervision.negative_points`, background discs of the trainer's 30 heatmap-pixel radius that train in any mode.
A pano with only negatives trains as `sparse` with no positive, like `root_v4`'s Laurens car-top panos (153 of the
255 new panos are negative-only before the page). s5's rules carry over: two positives closer than 0.05 merge (the
higher confidence stays); an ignore point within 0.05 of a kept positive is not written (A4's amendment: it would
zero the positive's loss); a negative within 0.05 of any label that is or may be a crosswalk (a positive, an
Unsure, or a validator `p_correct > 0.5`) is dropped by an automatic disc screen. **Claude then renders every
negative that would train** (Jon's existing ones and the page's) with its disc and may only **drop** a point, as in
s5; Jon sees only the dropped ones and can overrule.

**Caps** (applied in this order, Jon's negatives ranked first, then a seeded order 298):

- Rear-seam negatives (x < 0.04 or > 0.96, where the car roof sits on Trimble panos): at most 10 per sequence.
- All negatives: at most 30 per capture sequence (the first draft had one sequence supplying 23% of them), at
  most 3 per pano.
- Mined candidates: at most 125 on the page.
- **GoPro balance:** GoPro validator positives are added (seeded, at most 2 per pano) until GoPro positives are at
  least GoPro negatives; if there are not enough, GoPro negatives are cut to the GoPro positive count. Laurens and
  perhaps the new cities are GoPro Max, so the negatives must not teach "no crosswalks on GoPro panos". Today:
  33 GoPro negatives (24 Jon's, 9 `root_v5`), 33 GoPro positives (5 Jon's, 7 `root_v5`, 21 validator).
- **Juneau positives ≤ Juneau negatives in every outcome of the page.** When the rule binds, positives are dropped
  in a seeded order, tier by tier: Jon's PS-vote positives on Trimble panos (not blind), then his blind
  page-verdict positives on Trimble panos, and only if both run out, `root_v5` positives. Positives the page itself
  yields ("crosswalk under the cross") count too. `pos_drop_rank` in `candidates.csv` fixes the order now.
- Brick-sidewalk negatives: at most 1 per pano and 100 in all.

**Planned counts** (`s8_prep.py pools`; before the page, Claude's disc review and the brick mining):

| | Negatives | Positives | Panos |
|---|---|---|---|
| Jon's verdicts (votes and blind page verdicts), after screen and caps | 106 (22 rear seam, 24 GoPro) | 172 (130 from PS votes) | |
| `root_v5` Juneau points in the training area | 42 | 128 | 76 |
| GoPro validator positives | | 21 | |
| Mined candidates for Jon's page (0 GoPro, 4 rear seam) | up to 125 | | |
| New Juneau panos | | | 255 |
| **Juneau, if 0 / half / all 125 page cards are "no crosswalk"** | **148 / 210 / 273** | **148 / 210 / 273** | |
| Positives dropped to keep the rule (Jon's / `root_v5`) | | 167 + 6 / 111 / 48 | |
| Brick-sidewalk negatives (section 4) | ≤ 100 | | on existing panos |
| Ignore points on the new panos | 439 (475 other AI labels, 36 merged into a positive) | | |

Only if fewer than 6 of the 125 page cards come back "no crosswalk" does the rule reach `root_v5`'s positives.
At most 30 negatives come from any one sequence. For scale, `root_v4` has 8,034 positives and 290 negatives on
4,294 train panos; s8 adds about 3% more positives and takes the negatives from 290 to about 440 to 660, depending on the page and the brick mining.

**Train set and steps.** At most 4,294 + 76 + 255 = 4,625 train panos (exact at build), so
`STEPS = round(1000 × train / 4294)` ≈ 1,077 (A4's rule: each pano is seen as often as in s4).

## 3. The validator, and why it only mines

The validator is `projectsidewalk/sidewalk-validator-ai-crosswalk@b8498b82eb` (round 3, GSV-only training), run
offline on Mapillary rig-frame crops of all 3,579 labels (`inputs/validator_REPORT.md`). Against Jon's verdicts:

| Validator says | Cut | Labels | Jon's juneau-test votes agree | Random-sample votes agree | Jon's blind page verdicts agree (labels without a vote) |
|---|---|---|---|---|---|
| incorrect | p > 0.999 | 657 | 66 of 68 | 16 of 16 | 12 of 18 |
| incorrect, ≥ 8.4375° dip | p > 0.999 | 637 | 66 of 67 | 16 of 16 | 11 of 17 |
| **incorrect, ≥ 8.4375° dip** | **p > 0.9999** | 358 | 27 of 27 | 5 of 5 | **6 of 12** |
| correct | p > 0.9999 | 1,535 | 137 of 138 (1 Unsure) | 31 of 31 | 28 of 28 |
| correct | p > 0.99999 | 648 | 58 of 58 | 18 of 18 | 13 of 13 |

(`pools/summary.json`, `validator_cut_sweep`.)

- **The six real crosswalks it calls incorrect are on 3 panos, in two sequences** (`476976021506190` three,
  `1095037398234330` two, `420094107518324` one): `juneau-ak:5038`, `5039`, `6892`, `7245`, `7246`, `7248`. The
  parent session rendered the six crops (contact sheet `s8_six/sheet.jpg`, outside the repo; viewed again while
  writing this): **five are Juneau's red-tinted decorative-concrete crossings across the road**, real crosswalks
  with no white paint; `6892` is herringbone pavers beside the capture car's hood, unclear from the crop. So
  the validator rejects Juneau's decorative crossings. That is three events, not six, and the random-sample votes
  (5 of 5 at this cut) do not show it; both samples are tiny.
- **Why it matters.** Decorative and brick crossings are exactly the class guard 3 (Columbia brick recall)
  protects and decision 3 of s4 added. Validator negatives taken on trust would teach that class as background.
  Hence: the validator only orders and mines candidates, and Jon's verdict decides every negative that trains.
- **Why the audit was dropped.** A one-sided exact 95% lower bound reaches 0.95 only at 60 of 60 correct; one
  error needs 92 of 93. The audit's disc rule is stricter than what the validator judges (in s5, 4 of 58 of Jon's
  own negatives had a crossing in the disc), so even a perfect validator would often fail it. Its pass chance was
  about 0.27 at a true precision of 0.97, and its failure branch changed the recipe (positives would outnumber
  negatives). Rating the ~125 candidates directly costs about the same time and makes every negative Jon's.
- The decorative-concrete crossings get their own tag on the primary page, and their count is reported
  (section 6).

## 4. Brick-sidewalk negatives

**Source:** `root_v4`'s 548 brick train panos (PS crowd `brick/cobblestone` Crosswalk labels: Chicago 254,
Cuenca 133, Gainesville 67, Columbus 27, Pittsburgh 25 and smaller cities). None is a Columbia pano, and none is
a Seattle val, test or (b′) pano. These panos already teach "brick crossing" as positives, so a brick sidewalk on
the same pano is the matched negative.

**Order of work.**
1. **Mine first** (free; one klone job, `score_274.py mine` as s3's Source A): every `crosswalk_s4` seed-1
   detection at ≥ 0.23 on those panos that is not within 0.05 of a positive or an ignore point.
2. **Claude looks at a contact sheet** of the candidates (no verdicts recorded, nothing trained from it).
3. **If fewer than about 30 candidates look like brick sidewalk, the page is skipped** and the doc says so.
   Juneau's own brick sidewalks are already in: 35 of Jon's 106 planned negatives have "brick" in his comment.
4. Otherwise one blind page of **at most 100 cards** (seeded, at most 1 per pano), the same three-button,
   disc-on-every-card design as the mined-negative page, definition `paint_brick`. Jon's "no crosswalk in the
   disc" verdicts become negatives; crossings the crowd did not label become ignore points, never positives.

No AI pre-screen: its "not a crosswalk" calls were right on only 22 of 77 street crops in s3's pre-screen. PS
`SurfaceProblem` labels tagged `brick/cobblestone` are not read this round. Guard 3 (Columbia brick recall) checks
that "brick sidewalk is not a crosswalk" did not cost brick crossings.

## 5. Recipe and compute

**Recipe.** s4's, as A4 ran it: released weights; seeds 1, 2 and 3; batch 16; LR 1e-5;
`--loss-mask --mask-norm pixels`; `TRIM_EPOCH=1`; final checkpoint only; a 20-step smoke run first; about 1,077
steps (above). Each seed gets its own `RUNDIR`, `COPY_DEST` and `DATA_ROOT`, and a job refuses an existing run
directory, as `s6_train.sh` does. A run preempted on `ckpt-all` resumes from its latest checkpoint and is then
not bit-identical to an uninterrupted run (s5 seed 1); any resume is reported.

**Step control** (reported, deciding nothing): `root_v4` at s8's step count, seed 1, scored on the automatic
checks only. Without it a gain cannot be told from the longer schedule (A4's step control found 7 more Juneau
marks from steps alone).

**Threshold.** Recall-matched on Seattle val, each s8 seed against the s4 seed with the same number at s4's
adopted thresholds 0.23, 0.30 and 0.13 (`match-threshold --base-threshold`). s8's own val pick is reported. s4's
seed 3 runs unusually low (0.13; later models' seed 3 matched at 0.24 to 0.31) and has the most Juneau marks and
false alarms, so part of any three-seed change can be seed 3 alone; the per-seed differences and the mean without
seed 3 are reported (section 6).

**Calibration before any s8 job** (automatic): s4 seeds 1, 2 and 3 on all 989 slice panos, on Columbia (seeds 2
and 3 are new there) and on the Seattle confirmation draw. On the 200-pano sample, seed 1 must reproduce the 88
sent labels from the 4096 prep within the 0.01 card radius; a shortfall is reported before anything else runs.
The seed spreads for guards 3 and 5, the slice check and the Seattle trigger come from here.

**Compute: klone `ckpt-all`, free.** Five training jobs (smoke, three seeds, step control) at about 1.2 to 1.8 h
each on four GPUs, about 30 GPU-hours; inference about ten short jobs (val/test, (b′), Columbia, Seattle
confirmation draw, the 989 slice panos, the Laurens and Juneau recall285 panos, the brick mining set). Elapsed
time and GPU-hours go into `analysis_out/compute_log.jsonl` with `scripts/analysis/slurm_usage.py`, from the main
checkout. Not Tillicum: nothing here needs non-preemptible time.

**Storage.**

- The Juneau natives are on klone scrubbed (`juneau_store_xwalk/juneau-ak`, 8000×4000, copied 2026-11-16/17). The
  21-day purge would reach them about 2026-12-07. That is a cost, not a loss: the natives are still in the PS pano
  store, but a re-copy runs on the host and needs Jon and Duo. So the 4096 preps for the slice (989 panos) and the
  training panos (about 331) are made by `bprime-prep` and **tarred the same day**.
- The validator's 3,579 crops are in `/gscratch/scrubbed/jfroehli/validator_xwalk_juneau/` (about 0.2 GB; purge
  about 2026-12-08). They are the validator result's evidence: archive them first, as one tar.
- `/gscratch/makelab` was at 939 of 1,024 GB and 930,517 of 1,000,000 files on 2026-11-03, so everything staged
  there is a tar, never loose files; check headroom before staging the root (about 1 GB new) and four
  checkpoints (about 1 GB). Write a real file to scrubbed before submitting, since scrubbed refused writes on
  2026-11-13 (A4's rerun).

## 6. Evaluation

**What is scored.** Detections at each model's threshold, after the labeler's send-time Crosswalk filters: the
49° rig mask (y ≤ 0.772) and the 5.625° range gate (y ≥ 0.53125), border `keep`. That is what PS would
receive. Peaks come from `score_274.py infer` on the 4096×2048 preps, as for every earlier round.

### Primary (one; 95% interval, pano bootstrap, 2,000 draws, seed 298)

On the 200 slice-sample panos: **look-alikes per 100 panos, the three-seed mean of s8 minus the three-seed mean
of s4**, each seed at its own threshold. A look-alike is a "not a crosswalk" verdict; "unsure" is neither, as in
A4. **Met** if all three hold:

1. the interval's upper end is below 0;
2. seed 1's own difference is below 0;
3. **roof-free part:** the same three-seed difference over cards that are neither rear-seam (x < 0.04 or > 0.96)
   nor tagged "capture vehicle" has a point estimate below 0.

Part 3 exists because the capture car travels with the camera: the same roof is in the slice and in training,
so a roof-only gain would be s4's Laurens result again and would say nothing about other rigs or cities. Both the
full and the roof-free differences are reported with intervals. Part 3 costs no rating: it reads the same cards
and tags.

- **Cards.** Every detection of the six models on the 200 panos, merged at 0.01 (`merge-cards`) into one blind
  page, shuffled, with no model, seed or confidence shown. Inheritance: a card within 0.01 of a label Jon judged
  **blind** under the same question (the confirmation and fresh-eval pages) inherits that verdict (1 label in the
  sample). The 13 sample labels with a juneau-test vote are **re-rated blind** on the page: those votes were cast
  in PS's interface, knowing the label was the AI's, and only s4 has such cards, so carrying them would treat the
  two models differently (about 1.5 minutes). Verdicts from the s5 mining page are not carried, because that page
  asked the disc question.
- **Order on the page.** Cards fired by unequal numbers of s8 and s4 seeds come first. Cards fired by equal
  numbers change neither the primary nor guard 6 (86 of 486 on the stand-in below); they are rated last and may be
  skipped if Jon runs out of time, losing only the reported precision.
- **Expected size.** On the 80 Juneau recall285 panos, s4's three seeds and the three s6 half-dose seeds made 3.42
  merged cards per s4 seed-1 detection (`cards/summary.json`), so about 300 cards here, about 250 of them
  unequal-count.
- **Rating aid.** Claude pre-rates every card (blind, the same image Jon sees) with a verdict and a tag; the page
  shows the prefill and Jon confirms or changes each. AI "not a crosswalk" calls were right on only 22 of 77
  street crops in s3's pre-screen, so they are shown first and need Jon's look; a "crosswalk" prefill only needs a
  confirming click. Only Jon's final verdict counts.
- **Power** (300 simulated outcomes per cell; s4 look-alikes stood in by Jon's votes where he voted and the
  validator's verdict elsewhere; seed noise not modelled). **On the drawn sample** (`power/power_drawn_sample.csv`;
  25 stand-ins on 19 panos, 12.5 per 100):

| s8 removes / adds (share of s4's look-alikes) | Net cut | Pano bootstrap | Sequence-block bootstrap (sensitivity) |
|---|---|---|---|
| 70% / 15% | 55% | 0.91 | 0.93 |
| 50% / 0% | 50% | 1.00 | 1.00 |
| 70% / 30% | 40% | 0.39 | 0.52 |
| 50% / 15% | 35% | 0.56 | 0.64 |
| 30% / 0% | 30% | 0.96 | 0.96 |
| 50% / 30% | 20% | 0.11 | 0.20 |

  In words: the drawn sample detects a halving of Juneau look-alikes with high probability; a net cut of a third
  clears about half the time, less when s8 also brings new look-alikes, which add pano-to-pano variance; a cut of
  a fifth will not clear. Taking Juneau from 0.72 to the 0.9 precision a prod send would want is a cut of about 70%
  at the same number of real crosswalks. The sequence-block interval resamples 21 blocks of unequal size and comes
  out slightly narrower here; with so few blocks it is a sensitivity, not a correction. `power/power.csv` keeps the
  redrawn-sample sizes for comparison (150 panos: 0.51; 300: 0.81, at 50%/15%).

### Guards (all must hold). 1 to 5 and S need no rating

1. **Test F1:** three-seed mean on the v2 labels ≥ 0.826 (s4's mean 0.836 − 0.01).
2. **(b′) weighted marked-node recall:** three-seed mean drops by ≤ 0.03 against s4's three-seed mean (0.687).
3. **Columbia brick recall:** three-seed mean of s8 minus three-seed mean of s4 ≥ −0.03 (s4 seed 1: 183 of 210).
   A4 used seed 1 alone; three seeds because s8 teaches brick-sidewalk negatives, and because the validator's
   blind spot is decorative crossings, so this is the guard most at risk.
4. **Seattle retention:** s8 seed 1 has a detection within 0.05 of at least 194 of the 234 crosswalks Jon
   confirmed for s4 on the Seattle confirmation draw (A4's calibrated rule).
5. **Laurens car-top false alarms:** rig-masked detections on the 92 Laurens panos Jon confirmed empty;
   three-seed mean of s8 minus three-seed mean of s4 ≤ +3, or the s4 seed spread on those panos if larger. s4's
   three seeds have 2 such detections between them. The unmasked count is reported.
- **S. Slice retention (automatic, before any rating):** s8 seed 1 has a detection within 0.05 of at least
  **45 of the 50** slice labels Jon voted Agree, **or 50 minus the s4 seed spread on those 50 if that is lower**
  (A4's guard 4 form). Chosen over a fixed 45 because A4 showed fixed-share retention rules are stricter than the
  difference between two trainings of the same model (s4's own seeds kept 83% and 91% of seed 1's Seattle
  crosswalks), so a fixed bar can fail a no-change retrain by seed luck; and over the spread alone because a small
  spread would then demand nearly all 50. The spread is measured in calibration, before any s8 job.
6. **Crosswalks found on the slice sample (rated):** crosswalk verdicts per 100 panos, three-seed mean of s8
   minus three-seed mean of s4, point estimate ≥ −2.5. This is the slice's retention guard on cards nobody has
   rated yet: a model that buys precision by firing less in Juneau fails it. **It is a point rule on small counts
   and can fail by chance**, as A4 said of its guards 7 and 8: −2.5 is about 5 cards, and a rough estimate from
   s4's seed spread (63, 59 and 77 of the 110 marks) puts a no-change model's failure chance near one in eight.
   That is accepted and stated now.

**Seattle look-alikes: rated only on an automatic trigger.** A4's guard 7 (Seattle confirmation draw, seed 1,
paired look-alikes ≤ +2.5 per 100 panos; 50 to 90 cards) is rated only if s8 seed 1's matched threshold is below
0.20, **or** s8 seed 1's detection count on the Seattle confirmation draw exceeds s4 seed 1's by more than the s4
seed spread there. Recall-matching restores Seattle recall by lowering the threshold, which is how look-alikes
return; the trigger catches that without a page when it has not happened.

**Order, to spare rating:** guards 1 to 5 and S first. If any fails, s4 stays and nothing is rated. Then the
Seattle trigger is read; then the one primary page (plus the Seattle page if triggered).

### Outcomes

- **Adopt (for Juneau, with non-regression elsewhere):** the primary is met (all three parts) and every guard
  holds, including guard 7 if triggered. s8 seed 1 becomes the crosswalk detector of record. This shows a gain on
  held-out Juneau areas and no loss on Seattle, Columbia and Laurens; it does not show a gain in other cities,
  which the two new Mapillary cities will test. Any Juneau resend is a separate plan with Jon's clearance (live
  labels are insert-only).
- **Promising, not established:** the point estimate falls but the primary is not met, every guard holding. s4
  stays.
- **s4 stays:** any guard fails.

### Reported, deciding nothing

On the slice sample: precision and look-alikes per 100 panos per model with intervals, the primary per seed and
as the mean of seeds 1 and 2 (without seed 3), the roof-free difference with its interval, and look-alikes by tag
class; **decorative crossings** (cards tagged "decorative crossing" that Jon calls crosswalks, s8 against s4), the
class the validator rejects; camera split (thin: 9 GoPro panos). On all 989 slice panos: three-seed detections per
pano, s8 against s4 (a large ratio predicts the page size and a guard-6 failure), and the three-seed detection
count on the 56 GoPro slice panos. Also: recall on the 110 Juneau marks (contaminated outside the slice) and on the
19 slice marks; the 27 Juneau empty panos; the step control; s8's own val pick; (b′) F1; test per city; the
Bayonne and PS-label probes; the class mix of the mined page's tags; the `fa-sheet` contact sheet of guard 5's
detections, looked at before any page is built.

### Known limits

- The positives are s4's own peaks (self-training), placed on the detector's coarse grid. They teach where s4
  already fires; they do not teach the crosswalks s4 misses. Recall gains, if any, come from more Juneau
  context, as in s5.
- The slice is three areas of one city and one capture season, and the sample is mostly area 1. The two new
  Mapillary cities are the real out-of-city test and come later.
- Jon's juneau-test votes were cast in PS's validation interface, not under the disc rule; hence the disc screen
  and Claude's disc review of every negative.
- The pano table, the votes and the validator scores are snapshots of 2026-11-17; votes cast later are not used.

## 7. What Jon rates, and how long

Rates assumed: about 7 s per blind card with a disc and tags, about 4 s per confirm-only card. Neither has been
timed on these pages.

| When | Page | Cards | Time |
|---|---|---|---|
| Before training | Mined-negative page (three buttons, disc on every card) | 125 | about 15 min |
| Before training | Claude's dropped negatives (disc review of every negative), to overrule | about 10 to 20 | about 2 min |
| Before training, only if mining finds ≥ about 30 brick sidewalks | Brick-sidewalk page | ≤ 100 | 0 to 12 min |
| After training, only if guards 1 to 5 and S hold | Primary page (three seeds per model, Claude prefill; the 13 PS-vote labels re-rated blind) | about 300 (about 250 required, the rest skippable) | 28 to 33 min |
| After training, only if the Seattle trigger fires | A4 guard 7 page | 50 to 90 | 6 to 10 min |
| **Total** | | **about 385 to 635** | **about 45 to 62 min, plus 6 to 10 if triggered; three or four sittings** |

## 8. Archive

Following the earlier rounds and Jon's rule that everything is saved, to
`/projects/makeabilitylab/RampNet/archives/crosswalk_s8/` (README, `manifest.json`, `SHA256SUMS`). Claude stages;
Jon runs the makelab2 write. Transfers from klone run on a lab host or klone, never through the desktop (the home
connection's data cap). Image sets travel as tars.

| Item | Where it is now | Deadline |
|---|---|---|
| Validator crops, all 3,579 (one tar) | klone scrubbed | before about 2026-12-08 |
| Juneau 4096 preps (slice and training panos), tarred the day they are made | klone scrubbed | before about 2026-12-07 |
| `root_v8` tar and labels digest; `new_panos.csv`; the brick mining candidates | built on klone | at build |
| Checkpoints (three seeds, step control), sha256, TensorBoard runs | `/gscratch/makelab/jonf/crosswalk274/` | after training |
| Slurm logs | klone repo `logs/` | after each job |
| Rating pages (crops embedded) and Jon's raw exports | local scratch | the day each is rated |
| Predictions, scores, ratings CSVs, the input snapshot | this repo | each commit |

## 9. Decisions for Jon

The review's other questions are settled in the text above (the mined page instead of the audit; GoPro-only
validator positives; the caps and the positives ≤ negatives rule; mine-then-maybe brick page of ≤ 100; three
seeds on the primary page; the slice as drawn with a 60 m buffer; the older Juneau checks report-only; reusing
`root_v5`'s 76 panos; the Seattle trigger). What remains:

1. **Confirm the second vote account is yours.** `29vs30oz1c7daxjc` (`4ff63c52…`, 236 votes on 2026-11-17,
   16:20 to 16:44) looks like you voting while not logged in. Nothing is built until you confirm; if it is not
   you, its votes drop out mechanically and the counts are re-run.
2. **Approve the plan as a whole** (or name what to change). Once approved, the text is fixed.
3. **The card budget:** about 125 cards before training (plus up to 100 brick), about 300 after (250 required),
   about 45 to 62 minutes in all. A cheaper primary is possible (seed 1 only, about 150 cards) but is most exposed
   to seed luck, since s4 seed 1 is its high-false-alarm seed (29 unmasked false alarms against 8 and 11).
4. **Push and PR** when you are ready (a draft PR, without a session link).

Context for the `root_v5` reuse, now settled: s5's (b′) "failure" was 0.676 against s4 seed 1's 0.728; against
s4's three-seed mean of 0.687, the rule s8 uses, s5 seed 1 is −0.011. s5's seeds 2 and 3 on (b′) were not scored.

## Where each number lives

All under `analysis_out/crosswalk_s8/`.

| Number | File |
|---|---|
| Input hashes (sources and copies) | `inputs/inputs_sha256.json` |
| Juneau pano table (position, sequence, camera, detection counts) | `inputs/juneau_panos.csv` |
| Validator scores, all 3,579 labels; its report and tables | `inputs/validator_all_labels_scored.csv`, `inputs/validator_REPORT.md`, `inputs/validator_tables.md` |
| Jon's juneau-test votes; the random sample | `inputs/juneau_test_votes_2026-11-17.csv`, `inputs/random_sample_2026-11-17.json` |
| Vote accounts and usernames | `s8_prep.VOTE_ACCOUNTS`, repeated in `pools/summary.json` |
| Areas, per-role counts, expected contents, slice checks, the primary sample's mix | `slice/summary.json`, `slice/areas.json` |
| Every pano's role; every label's role and distance to the slice | `slice/panos.csv`, `slice/labels.csv` |
| The primary sample and reserve | `slice/primary_sample.txt`, `slice/primary_reserve.txt` |
| Candidate and planned training points, per label (`planned`, `pos_drop_rank`, `vote_user_id`) | `pools/candidates.csv` |
| Pool counts, cut sweep, caps, GoPro balance, the page-outcome branches, ignore points | `pools/summary.json` |
| Cards per s4 detection; equal-count cards | `cards/summary.json` |
| Power, redrawn sizes and the drawn sample | `power/power.csv`, `power/power_drawn_sample.csv`, `power/summary.json` |
| Audit bounds quoted in section 3 | computed with `scipy.stats.beta` / `binom`; not a script output |
| The six decorative-crossing crops | parent session's `s8_six/sheet.jpg`, outside the repo; to be archived with the validator crops |

## Replication (prep only)

Code at the commit that adds this text. CPU, under a minute except `power` (about 6 minutes).

```bash
S=scripts/analysis/stage1_crosswalk_sdot_274
L=<labeler worktree on store-records-crosswalk>   # runs/juneau_crosswalk, runs/juneau_expansion_crosswalk
V=<validator_xwalk scratch dir>                   # results/all_labels_scored.csv, REPORT.md, results/tables.md
python $S/s8_prep.py snapshot --records $L/runs/juneau_crosswalk/results.jsonl \
    $L/runs/juneau_expansion_crosswalk/results.jsonl --validator $V/results/all_labels_scored.csv \
    --validator-docs $V/REPORT.md $V/results/tables.md \
    --votes $L/runs/juneau_crosswalk/validation/juneau_test_votes_2026-11-17.csv \
    --random-sample $L/runs/juneau_crosswalk/validation/random_sample_2026-11-17.json
python $S/s8_prep.py slice       # seed 298; 3 areas, 12%, 60 m buffer; sample 200 + reserve 50 (seed 299)
python $S/s8_prep.py pools       # caps and branches; seed 298
python $S/s8_prep.py cards
python $S/s8_prep.py power       # seed 298, 300 sims, 1,000 bootstrap draws per sim
pytest tests/test_s8_prep.py     # the slice and sample reproduce; pool invariants hold
```

Everything after `snapshot` reads only committed files, so a clean clone reproduces the slice, pools, cards and
power from `inputs/`. The `results.jsonl` files themselves are gitignored in the labeler (12 MB); their sha256 are
in `inputs/inputs_sha256.json`. The validator's per-label scores came from an offline run on klone
(`inputs/validator_REPORT.md`); its crops are on klone scrubbed until archived (section 8).
