# `crosswalk_s8`: Juneau in training, a held-out Juneau slice (RampNet#298)

**DRAFT pre-registration for Jon to approve. Not approved.** Nothing has been trained, inferred, rated, posted or
pushed. The prep artefacts below (input snapshot, slice, candidate pools, card and power estimates) are CPU-only
and change no model. Drafted 2026-11-17 by Claude Opus 5.5 (`claude-opus-5-5`) from Jon's decisions of the same
day. Once approved, this text is fixed; later changes go in dated amendments below it.

Branch `crosswalk-s8`, based on `crosswalk-supervision-s7` (the stack is main ← `stage1-label-type-274` ←
`crosswalk-followups-285` ← `crosswalk-supervision-s7`). The name `crosswalk_s7` is taken by the stopped
supervision plan (`docs/crosswalk_supervision_s7.md`).

## Summary

- **Goal.** Cut the look-alikes `crosswalk_s4` sends on Mapillary in Juneau without losing crosswalks, Seattle,
  brick crossings or the Laurens car-top fix.
- **Data.** `root_v8` = `root_v4` + Juneau train panos (about 450) carrying Jon-verified and validator-sourced
  positives and negatives + brick-sidewalk negatives on `root_v4`'s brick panos. Planned before audits: **Juneau
  376 negatives and 376 positives** (Jon-verified: 150 negatives, 303 positives), plus at most 200 brick-sidewalk
  negatives.
- **Held out.** Three seeded Juneau areas holding 437 of the 3,579 AI labels (12.2%) on 989 panos, with a 50 m
  buffer (429 panos, 275 labels) that neither trains nor scores.
- **Primary.** On a seeded sample of 200 slice panos, look-alikes per 100 panos, three-seed mean of s8 minus
  three-seed mean of s4, each seed at its recall-matched threshold, from Jon's blind ratings. Met if the 95%
  pano-bootstrap interval is below 0 and seed 1's own difference is below 0. Simulated power: 0.9 or better if s8
  cuts s4's look-alikes by half net; about 0.6 at a 35% net cut; 0.16 at 20%.
- **Guards.** Five automatic (test F1, (b′) recall, Columbia brick recall, Seattle retention, Laurens car-top
  false alarms, all from A4) and one rated (crosswalks found on the slice sample must not fall by more than 2.5
  per 100 panos).
- **Jon's rating.** About 25 to 40 minutes before training (validator audit, validator positives, brick-sidewalk
  page) and 30 to 35 minutes after, only if every automatic guard holds. Cheaper options are in the decision list.
- **A finding that bears on Jon's decision 2.** Against Jon's verdicts from the random confirmation draws, the
  validator's confident "incorrect" calls are wrong far more often than against his juneau-test votes (section
  3). A 60-card audit at the 0.95 bar probably fails. Decision 1 at the end asks how to proceed.

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
  negatives work for that rig.
- **s5** (164 Juneau positives, 53 negatives, 3:1) found 49% more real Juneau crosswalks but left look-alikes
  flat (−2.5 [−10.8, +5.0]) and failed the (b′) guard on seed 1 (−0.052). Positives buy recall, not precision,
  and Juneau data can cost Seattle.
- **s6** (3,040 auto-placed sparse positives) raised Juneau recall but multiplied false alarms on
  confirmed-empty panos (guard 6: 6.7 → 112). Many sparse positives can shift the model toward firing.
- **s7** stopped at its leak check: whole-frame background taught where s4, OSM and Vistas are silent leaked
  12.4% of Jon's missed crosswalks. So s8 keeps `sparse` supervision and teaches background only at
  verified points.

Tracked as RampNet#298.

## Jon's decisions (2026-11-17) that this plan implements

1. **Juneau goes into training**, reversing the 2026-11-12 hold-out, except a small geographic held-out slice
   (a few whole areas, seeded, with a 50 m buffer) that carries the primary. Two Mapillary cities launching on PS
   in the coming weeks will be later held-out tests; they are not available to this decision.
2. **Negatives:** (a) Jon's Disagree labels outside the slice; (b) the production Crosswalk validator's confident
   "incorrect" verdicts on other Juneau AI labels, trusted only after Jon audits a seeded sample of about 60 cards,
   with the cut chosen so the audit can establish ≥ 0.95; (c) brick **sidewalk** negatives paired with the
   brick-crosswalk positives, without touching the Columbia brick test.
3. **Positives:** Jon's Agree labels plus the validator's confident "correct" verdicts, outside the slice.
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
- **Buffer:** every pano within 50 m of a slice pano. Buffer panos neither train nor score.
- **Label-position rule:** a training label whose PS-stored position lies within 35 m of a slice pano is
  excluded (it could be a crosswalk a slice pano sees). It removes none: the nearest training label is 37.8 m away.

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

| | Slice | Buffer | Training area |
|---|---|---|---|
| Panos | 989 | 429 | 3,915 |
| Panos with an AI label | 224 | 113 | 1,043 |
| AI labels (GoPro Max) | 437 (87) | 275 (6) | 2,867 (199) |
| Validator verdict Agree / Disagree | 294 / 143 | 225 / 50 | 1,989 / 878 |
| Jon's juneau-test votes Agree / Disagree / Unsure | 50 / 13 / 0 | 10 / 2 / 0 | 148 / 121 / 4 |
| Random-sample labels (voted) | 10 (9) | 9 (9) | 61 (61) |
| recall285 panos / sure marks within 25 m | 11 / 19 | 9 / 16 | 60 / 75 |
| Confirmation / fresh-eval / s5-mining draw panos | 33 / 14 / 52 | 6 / 13 / 22 | 81 / 93 / 226 |

**Expected contents of the slice** (rough sizes from city-wide rates, not measurements of the slice): 437 s4
labels, about 315 real crosswalk labels and 120 look-alikes at the city's 0.72 precision (the validator's Agree
share in the slice is 0.67, a little below the city's 0.70); about 550 visible crosswalks within 25 m if s4's
Juneau recall of 0.57 holds there.

**The primary's pano sample.** 200 panos drawn uniformly from the 989 (seed 299), plus a seeded reserve of 50
for panos that turn out unusable (`primary_sample.txt`, `primary_reserve.txt`). Drawn now, before any s8
checkpoint. The sample holds 88 s4 labels on 45 panos (18 GoPro Max); 13 of the 88 already carry a Jon vote.

### Leakage controls

- **Spatial:** the slice plus the 50 m buffer, plus the 35 m label-position rule. The range gate keeps every
  sent label at least 5.625° below the horizon (about 25 m on flat ground from a 2.5 m camera; the shallowest sent
  label is at 5.985°). A crosswalk within 25 m of a slice pano is therefore more than 25 m from every training
  pano, so it cannot be a training point there, and `sparse` supervision puts no loss on pixels away from the
  training points. Slopes and camera height loosen this a little.
- **Sequence:** excluding whole capture sequences is not possible. Juneau has 41 sequences, 22 of them cross the
  slice, and 2,344 of the 3,915 training-area panos share a sequence with it. Shared sequences mean the same rig,
  day and light, which is the domain this round is meant to learn; the same physical crosswalks are kept apart
  by the buffer, not by sequence. Stated, not controlled.
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

- **Juneau recall on the 110 marks (A4's primary):** 75 of the marks are on training-area panos, whose
  neighbours train. Reported on all 110 and, separately, on the 19 inside the slice (too few to gate).
- **Guard 6's 27 Juneau empty panos:** 22 are in the training area. Reported only; the Laurens 92 carry the
  car-top guard.
- **Fresh Juneau draw (s5 guard 5) and the confirmation draw:** most of their panos are in the training area, and
  Jon's verdicts on their s4 cards become training labels (section 2). Not used for evaluation.

## 2. Training data, `root_v8`

```
root_v8 = root_v4                                   (every val, test, (b′) file byte-identical)
        + brick-sidewalk negative_points on root_v4 brick train panos   (≤ 200; section 4)
        + root_v5's Juneau mining panos in the training area           (77 panos: 130 positives, 42 negatives,
                                                                         their ignore points; JSONs reused)
        + new Juneau train panos from the PS AI labels                  (372 panos; table below)
```

`root_v5` itself is not the base: its Juneau panos in the slice (and buffer) must not train.

**Sources of Juneau training points,** by priority on each AI label:

1. A `root_v5` pano keeps exactly its own points (Jon-rated under the disc rule, reviewed by Claude, 2026-11-09).
   PS-derived points on those panos are dropped (160 labels).
2. **Jon's verdict:** his juneau-test vote (latest per label), else his verdict on an earlier page's s4 card
   within 0.01 of the label (`confirm_s3/juneau/s4`, `s5_inputs/juneau_eval/s4`; 87 labels carry one). Agree /
   `crosswalk` → positive; Disagree / `not_crosswalk` → negative candidate; Unsure → ignore point.
3. **Validator, unrated labels only:** `p_correct > 0.9999` → positive candidate; `p_incorrect > 0.9999` and at
   least 8.4375° below the horizon (about 17 m and nearer; the edge of the detector's third row cell, where the
   look-alike classes sit and the validator's crops are not tiny) → negative candidate (section 3 for the cut).
4. Everything else on a training pano → ignore point.

**How a label becomes a training target.** Positives are `sparse` points at the AI label's pixel (normalized
from the 8000×4000 store pixel; the same point on the 4096×2048 prep). Negatives are
`supervision.negative_points`, background discs of the trainer's 30 heatmap-pixel radius that train in any mode.
A pano with only negatives trains as `sparse` with no positive, like `root_v4`'s Laurens car-top panos. s5's
rules carry over: two positives closer than 0.05 merge (the higher confidence stays); an ignore point within
0.05 of a kept positive is not written (A4's amendment: it would zero the positive's loss); a negative within
0.05 of any label that is or may be a crosswalk (a positive, an Unsure, or a validator `p_correct > 0.5`) is
dropped by an automatic disc screen. Every Jon-sourced negative is then rendered with its disc and reviewed by
Claude, who may only **drop** a point (as in s5); Jon sees only the dropped ones.

**Caps,** so no class or sequence dominates:

- Negatives: at most 3 per pano; at most 10 rear-seam negatives (x < 0.04 or > 0.96, where the car roof sits on
  Trimble panos) per capture sequence. One GoPro Max sequence (`eLOsiMypD8rIQc0qlHVAPg`) alone has 27.
- Positives: Jon-verified positives are always kept. Validator positives fill only up to the number of Juneau
  negatives kept, at most 2 per pano, in a seeded order (298). Juneau positives therefore never outnumber Juneau
  negatives, the opposite of s5's 3:1, which raised recall and left look-alikes flat.
- Brick-sidewalk negatives: at most 1 per pano and 200 in all.
- Brick and car-roof classes cannot be capped directly on validator negatives, whose class is unknown. The
  geometric proxy (rear seam) and the per-pano cap are what is enforceable; the class mix is read from the audit's
  tags and reported.

**Planned counts** (`s8_prep.py pools`; before the audit, the disc review and the brick page, all of which can
only remove points):

| Source | Negatives | Positives | Panos |
|---|---|---|---|
| Jon's votes and page verdicts (after disc screen and caps) | 108 (29 rear seam) | 173 | |
| Validator candidates (after caps; audited) | 226 (of 253) | 73 (of 964) | |
| New Juneau panos from these | | | 372 |
| `root_v5` Juneau points in the training area | 42 | 130 | 77 |
| **Juneau total** | **376** | **376** | **449** |
| Jon-verified only (if no validator point is used) | 150 | 303 | about 400 |
| Brick-sidewalk negatives (section 4) | ≤ 200 | | on existing panos |
| Ignore points on the new panos | 650 (718 other AI labels, 68 merged into a positive) | | |

For scale, `root_v4` has 8,034 positives and 290 negatives on 4,294 train panos. s8 adds about 5% more
positives and takes the negatives from 290 to about 670, or about 870 with a full brick page.

**Train set and steps.** About 4,294 + 77 + 372 = 4,743 train panos (exact at build), so
`STEPS = round(1000 × train / 4294)` ≈ 1,105 (A4's rule: each pano is seen as often as in s4).

## 3. The validator: what the audit can and cannot establish

The validator is `projectsidewalk/sidewalk-validator-ai-crosswalk@b8498b82eb` (round 3, GSV-only training), run
offline on Mapillary rig-frame crops of all 3,579 labels (`inputs/validator_REPORT.md`). Against Jon's verdicts:

| Validator says | Cut | Labels | Jon's juneau-test votes agree | Jon's page verdicts agree (labels without a vote) | Unrated, in training area |
|---|---|---|---|---|---|
| incorrect | p > 0.999 | 657 | 66 of 68 | 12 of 18 | 477 |
| incorrect, ≥ 8.4375° dip | p > 0.999 | 637 | 66 of 67 | 11 of 17 | 463 |
| **incorrect, ≥ 8.4375° dip** | **p > 0.9999** | 358 | 27 of 27 | **6 of 12** | 261 |
| correct | p > 0.9999 | 1,535 | 137 of 138 (1 Unsure) | 28 of 28 | 1,028 |
| correct | p > 0.99999 | 648 | 58 of 58 | 13 of 13 | 456 |

(`pools/summary.json`, `validator_cut_sweep`, which pools the two kinds of verdict; a page verdict counts only
where the label has no vote. The last column excludes the recall285 panos but still includes the `root_v5`
panos, which section 2 takes out: 253 negatives remain at the chosen cut.)

- **The two kinds of Jon verdict disagree about the validator.** On his juneau-test votes, the confident
  "incorrect" calls are nearly always right. On his blind page verdicts from the seeded confirmation draw, 6 of
  the 12 at the chosen cut were real crosswalks (`juneau-ak:5038`, `5039`, `6892`, `7245`, `7246`, `7248`; all
  Trimble, 9 to 35° dip, s4 confidence 0.28 to 0.67, no notes). Across all page verdicts, the validator's
  Disagree verdicts are right on 23 of 31 (0.74), close to the random-sample estimate of 0.81 [0.65, 0.97] and
  far from the 0.94 on all votes. The juneau-test votes came mostly from clustered queue and gallery browsing,
  so the page verdicts are the more representative check. Why the validator misses these crossings is not known;
  brick or faded paint is a guess, and the cards carry no notes.
- **What 60 cards can show.** A one-sided exact (Clopper-Pearson) 95% lower bound reaches 0.95 only at 60 of 60
  correct (0.951). One error needs 93 cards (92 of 93); two need 124. The Wilson bound at 60 of 60 is 0.940.
- **Proposed audit rule** (Jon's decision 2, made exact): a seeded draw of 60 cards from the 226 planned
  validator negatives, rated blind with the negative disc drawn, the s3 disc rule ("not a crosswalk" only if no
  crosswalk paint lies anywhere inside the disc) and one-click class tags. 60 of 60 → pass. Exactly one error →
  rate 33 more from the seeded reserve; pass at 92 of 93. Two or more errors → fail. Under this rule the chance
  of passing is 0.27 if the pool's true precision is 0.97, 0.49 at 0.98, 0.79 at 0.99 and 0.93 at 0.995. The page
  verdicts above point lower than 0.97. (Checking at 60 and again at 93 spends a little more than the nominal 5%
  error; stated, not corrected.)
- **If the audit passes,** the planned validator negatives train, and the audited cards train with Jon's own
  verdicts. **If it fails,** no unaudited validator negative trains; the audited cards still train as Jon's
  verdicts (their "not a crosswalk" verdicts become negatives). Decision 1 offers the alternative of having Jon
  confirm every validator negative that would train (226 cards on a confirm-only page) instead of the audit.
- **Validator positives.** At most 73 train under the cap. At this cut the validator agrees with Jon on 165 of
  166 rated labels (the other is an Unsure). Proposed: Jon confirms each of the 73 on the same page as the audit
  (one click each), so no unconfirmed validator positive trains.

## 4. Brick-sidewalk negatives

**Source:** `root_v4`'s 548 brick train panos (PS crowd `brick/cobblestone` Crosswalk labels: Chicago 254,
Cuenca 133, Gainesville 67, Columbus 27, Pittsburgh 25 and smaller cities). None is a Columbia pano, and none is
a Seattle val, test or (b′) pano. These panos already teach "brick crossing" as positives, so a brick sidewalk on
the same pano is the matched negative.

**Mining** (`score_274.py mine`, as s3's Source A): every `crosswalk_s4` seed-1 detection at ≥ 0.23 on those
panos that is not within 0.05 of a positive or an ignore point becomes a card. Inference is one klone job. The
yield is unknown before it runs. s3's mining found 659 cards on about 1,540 sparse train panos (about 0.4 per
pano), but s4 trained on these very panos and should fire on them less; expect perhaps 100 to 250 cards. A seeded
cap of 200 cards keeps the page bounded.

**Rating:** one blind page, `--neg-disc`, definition `paint_brick` (a brick crossing is a crosswalk), with tags
(brick or paver sidewalk, curb ramp, other). No AI pre-screen: its "not a crosswalk" calls were right on only 22
of 77 street crops in s3, and every negative needs Jon's verdict anyway. Jon's "not a crosswalk" verdicts become
negatives, at most one per pano, at most 200. Crossings Jon finds that the crowd did not label become ignore
points, never positives.

**Why not elsewhere.** PS `SurfaceProblem` labels tagged `brick/cobblestone` would point at brick sidewalks
directly, but need a new prod read, new store panos and their own page; listed as an option, not planned. Juneau's
own brick sidewalks enter through Jon's brick-sidewalk Disagrees (40 in all, most in the training area) and the
validator pool. Guard 3 (Columbia brick recall) is the check that teaching "brick sidewalk is not a crosswalk"
did not cost brick crossings.

## 5. Recipe and compute

**Recipe.** s4's, as A4 ran it: released weights; seeds 1, 2 and 3; batch 16; LR 1e-5;
`--loss-mask --mask-norm pixels`; `TRIM_EPOCH=1`; final checkpoint only; a 20-step smoke run first; about 1,105
steps (above). Each seed gets its own `RUNDIR`, `COPY_DEST` and `DATA_ROOT`, and a job refuses an existing run
directory, as `s6_train.sh` does.

**Step control** (reported, deciding nothing): `root_v4` at s8's step count, seed 1, scored on the automatic
checks only. Without it a gain cannot be told from the longer schedule (A4's step control found 7 more Juneau
marks from steps alone).

**Threshold.** Recall-matched on Seattle val, each s8 seed against the s4 seed with the same number at s4's
adopted thresholds 0.23, 0.30 and 0.13 (`match-threshold --base-threshold`). s8's own val pick is reported.

**Calibration before any s8 job** (automatic): s4 seeds 1, 2 and 3 on all 989 slice panos and on Columbia
(seeds 2 and 3 are new there). On the 200-pano sample, seed 1 must reproduce the 88 sent labels from the 4096
prep within the 0.01 card radius; a shortfall is reported before anything else runs. The seed spreads for
guards 3 and 5 come from here.

**Compute: klone `ckpt-all`, free.** Five training jobs (smoke, three seeds, step control) at about 1.2 to 1.8 h
each on four GPUs, about 30 GPU-hours; inference about ten short jobs (val/test, (b′), Columbia, Seattle
confirmation draw, the 989 slice panos, the Laurens and Juneau recall285 panos, the brick mining set). Elapsed
time and GPU-hours go into `analysis_out/compute_log.jsonl` with `scripts/analysis/slurm_usage.py`, from the main
checkout. Not Tillicum: nothing here needs non-preemptible time.

**Storage.**

- The Juneau natives are on klone scrubbed (`juneau_store_xwalk/juneau-ak`, 8000×4000, copied 2026-11-16/17). The
  21-day purge would reach them about 2026-12-07. The 4096 preps for the about 449 training panos and the 989 slice
  panos (about 2 GB) are made there by `bprime-prep` and copied to durable storage the same day.
- The validator's 3,579 crops are in `/gscratch/scrubbed/jfroehli/validator_xwalk_juneau/` (about 0.2 GB; purge
  about 2026-12-08). They are the audit page's raw material and the validator result's evidence: archive them first.
- `/gscratch/makelab` was at 939 of 1,024 GB on 2026-11-03; check headroom before staging the root (about 1 GB
  new) and four checkpoints (about 1 GB). Write a real file to scrubbed before submitting, since scrubbed refused
  writes on 2026-11-13 (A4's rerun).

## 6. Evaluation

**What is scored.** Detections at each model's threshold, after the labeler's send-time Crosswalk filters: the
49° rig mask (y ≤ 0.772) and the 5.625° range gate (y ≥ 0.53125), border `keep`. That is what PS would
receive. Peaks come from `score_274.py infer` on the 4096×2048 preps, as for every earlier round.

### Primary (one; 95% interval, pano bootstrap, 2,000 draws, seed 298)

On the 200 slice-sample panos: **look-alikes per 100 panos, the three-seed mean of s8 minus the three-seed mean
of s4**, each seed at its own threshold. A look-alike is a "not a crosswalk" verdict; "unsure" is neither, as in
A4. **Met** if the interval's upper end is below 0 **and** seed 1's own difference is below 0.

- **Cards.** Every detection of the six models on the 200 panos, merged at 0.01 (`merge-cards`) into one blind
  page, shuffled, with no model, seed or confidence shown. A card within 0.01 of a label Jon already judged under
  the same question ("is the outlined cross on a crosswalk?": his juneau-test votes, the confirmation and
  fresh-eval pages) inherits that verdict (13 s4 labels in the sample have a vote). Verdicts from the s5 mining
  page are not carried, because that page asked the disc question. The paired difference depends only on cards fired by unequal
  numbers of s8 and s4 seeds, but every card is rated so that rates and precision can be reported.
- **Expected size.** On the 80 Juneau recall285 panos, s4's three seeds and the three s6 half-dose seeds made 3.42
  merged cards per s4 seed-1 detection (1.83 for seed 1 against seed 1; `cards/summary.json`). That puts this page
  at about 300 cards (about 160 with seed 1 only).
- **Rating aid.** Claude pre-rates every card (blind, the same image Jon sees) with a verdict and a tag; the page
  shows the prefill and Jon confirms or changes each. AI "not a crosswalk" calls were right on only 22 of 77
  street crops in s3's pre-screen, so they are shown first and need Jon's look; a "crosswalk" prefill only needs a
  confirming click. Only Jon's final verdict counts.
- **Power** (`power/power.csv`; 300 simulated draws per cell; s4 look-alikes on the slice stood in by Jon's votes
  where he voted and the validator's verdict elsewhere, 14.3 per 100 panos; seed noise not modelled):

| Panos | s8 removes / adds (share of s4's look-alikes) | Net cut | Chance the primary is met |
|---|---|---|---|
| 200 | 70% / 15% | 55% | 0.90 |
| 200 | 50% / 0% | 50% | 1.00 |
| 200 | 70% / 30% | 40% | 0.55 |
| 200 | 50% / 15% | 35% | 0.62 |
| 200 | 30% / 0% | 30% | 0.95 |
| 200 | 50% / 30% | 20% | 0.16 |
| 150 | 50% / 15% | 35% | 0.51 |
| 300 | 50% / 15% | 35% | 0.81 |

  New look-alikes cost more power than the net cut suggests, because they add pano-to-pano variance: a clean 30%
  cut (0.95) is easier to see than a 40% net cut made of 70% removed and 30% added (0.55).

  In words: 200 panos detect a halving of Juneau look-alikes with high probability. A net cut of a third clears
  between about half and nine times in ten, depending on how many new look-alikes s8 brings. A cut of a fifth
  will not clear. Taking Juneau from 0.72 to the 0.9 precision a prod send would want is a cut of about 70% at
  the same number of real crosswalks, well inside what this sample sees. Averaging three seeds should narrow the
  interval by an amount the calibration will show. 150 panos would be noticeably weaker (0.51 against 0.62 in
  the middle row) for about a quarter fewer cards.

### Guards (all must hold). 1 to 5 need no rating

1. **Test F1:** three-seed mean on the v2 labels ≥ 0.826 (s4's mean 0.836 − 0.01).
2. **(b′) weighted marked-node recall:** three-seed mean drops by ≤ 0.03 against s4's three-seed mean (0.687).
   This is the guard s5 failed, and s5's Juneau points are reused here.
3. **Columbia brick recall:** three-seed mean of s8 minus three-seed mean of s4 ≥ −0.03 (s4 seed 1: 183 of 210).
   A4 used seed 1 alone; three seeds because s8 teaches brick-sidewalk negatives, so this is the guard most at
   risk. s4 seeds 2 and 3 are run on Columbia in calibration.
4. **Seattle retention:** s8 seed 1 has a detection within 0.05 of at least 194 of the 234 crosswalks Jon
   confirmed for s4 on the Seattle confirmation draw (A4's calibrated rule).
5. **Laurens car-top false alarms:** rig-masked detections on the 92 Laurens panos Jon confirmed empty;
   three-seed mean of s8 minus three-seed mean of s4 ≤ +3, or the s4 seed spread on those panos if larger. s4's
   three seeds have 2 such detections between them. The unmasked count is reported.
6. **Crosswalks found on the slice sample (rated):** crosswalk verdicts per 100 panos, three-seed mean of s8
   minus three-seed mean of s4, point estimate ≥ −2.5. This is the slice's retention guard: a model that buys
   precision by firing less in Juneau fails it. Recall-matching on Seattle val does not prevent that on its own.

**Order, to spare rating:** guards 1 to 5 first. If any fails, s4 stays and nothing is rated. Otherwise the one
primary page is built; the primary and guard 6 are scored from it.

### Outcomes

- **Adopt:** the primary is met and every guard holds. s8 seed 1 becomes the crosswalk detector of record. Any
  Juneau resend is a separate plan with Jon's clearance (live labels are insert-only).
- **Promising, not established:** the point estimate falls but the primary is not met, every guard holding. s4
  stays.
- **s4 stays:** any guard fails.

### Reported, deciding nothing

Precision and look-alikes per 100 panos on the slice sample per model (with intervals), by Jon's tag class and by
camera (GoPro Max vs Trimble); recall on the 110 Juneau marks (contaminated outside the slice) and on the 19 slice
marks; the 27 Juneau empty panos; the step control; s8's own val pick; (b′) F1; test per city; the Bayonne and
PS-label probes; the class mix of the validator negatives from the audit tags; the `fa-sheet` contact sheet of
guard 5's detections, looked at before any page is built.

### Known limits

- The positives are s4's own peaks (self-training), placed on the detector's coarse grid. They teach where s4
  already fires; they do not teach the crosswalks s4 misses. Recall gains, if any, come from more Juneau
  context, as in s5.
- The slice is three areas of one city and one capture season. The two new Mapillary cities are the real
  out-of-city test and come later.
- Jon's juneau-test votes were cast in PS's validation interface, not under the disc rule; hence the disc screen
  and Claude's disc review of every Jon negative.
- The prep pano table, the votes and the validator scores are snapshots of 2026-11-17; votes cast later are not
  in this plan.

## 7. What Jon rates, and how long

Rates assumed: about 7 s per blind card with a disc and tags, about 4 s per confirm-only card. Neither has been
timed on these pages.

| When | Page | Cards | Time |
|---|---|---|---|
| Before training | Validator negative audit (60, up to 93) and the 73 validator positives, one page | 133 to 166 | 12 to 16 min |
| Before training | Claude's dropped Jon negatives (disc review), to overrule | about 10 to 20 | 2 min |
| Before training | Brick-sidewalk page | 100 to 200 | 12 to 23 min |
| After training, only if guards 1 to 5 hold | Primary page (three seeds per model, Claude prefill) | about 290 | 30 to 35 min |
| **Total** | | | **about 60 to 75 min, in three or four sittings** |

Ways to cut it are in the decisions below: seed 1 only on the primary page (about 150 cards, saves about 15
min), skipping the brick page (saves 12 to 23 min, loses the paired brick negatives), or replacing the audit with
a confirm-all page (costs about 7 more minutes but removes the likely-failing audit).

## 8. Archive

Following the earlier rounds and Jon's rule that everything is saved, to
`/projects/makeabilitylab/RampNet/archives/crosswalk_s8/` (README, `manifest.json`, `SHA256SUMS`). Claude stages;
Jon runs the makelab2 write. Transfers from klone run on a lab host or klone, never through the desktop (the home
connection's data cap).

| Item | Where it is now | Deadline |
|---|---|---|
| Validator crops, all 3,579 | klone scrubbed | before about 2026-12-08 |
| Juneau 4096 preps (training and slice panos) and the native store copies they came from | klone scrubbed | before about 2026-12-07 |
| `root_v8` tar and labels digest; `new_panos.csv`; the brick mining candidates | built on klone | at build |
| Checkpoints (three seeds, step control), sha256, TensorBoard runs | `/gscratch/makelab/jonf/crosswalk274/` | after training |
| Slurm logs | klone repo `logs/` | after each job |
| Rating pages (crops embedded) and Jon's raw exports | local scratch | the day each is rated |
| Predictions, scores, ratings CSVs, the input snapshot | this repo | each commit |

## 9. Decisions for Jon

1. **Validator negatives: audit or confirm-all?** Given section 3 (page verdicts 6 of 12 at the cut), a 60-card
   audit at the 0.95 bar is more likely to fail than pass. (a) Run the audit as decided, falling back to
   Jon-verified negatives on failure. (b) Skip the audit: Jon confirms each of the 226 planned validator negatives
   on a confirm-only page (about 15 min), so every negative is Jon-verified. (c) Drop validator negatives. The
   draft text uses (a).
2. **The audit bar.** One-sided exact 95% lower bound ≥ 0.95 (60 of 60, or 92 of 93) as drafted, or a weaker bar.
3. **The cut.** `p_incorrect > 0.9999` and ≥ 8.4375° dip (pool 253) as drafted, or `> 0.999` (pool 447 after the same
   exclusions; 11 of 17 on page verdicts, no better).
4. **Validator positives:** Jon confirms the at most 73 that train (drafted), or trust them at `p_correct > 0.9999`.
5. **Positive cap:** Juneau positives ≤ Juneau negatives (drafted, about 376 each), or more positives for recall.
6. **Brick-sidewalk source:** mine s4 on `root_v4`'s 548 brick panos with a page of up to 200 cards (drafted),
   skip it, or add PS `SurfaceProblem` `brick/cobblestone` locations (new prod read and store panos).
7. **Primary page:** three seeds per model, about 290 cards (drafted), or seed 1 only, about 150 cards. Seed noise
   on Juneau counts has been large (s4's seeds found 63, 59 and 77 of the 110 marks).
8. **Slice:** accept the three areas as drawn (seed 298, 12%, 50 m buffer, 200-pano sample), or change a
   parameter before anything else is built. A map of the areas can be made on request.
9. **Earlier Juneau checks** (recall on the 110 marks, the 27 Juneau empty panos, the fresh and confirmation
   draws) become report-only. Agree?
10. **Reuse `root_v5`'s 77 training-area Juneau panos**, knowing s5 failed the (b′) guard on seed 1 alone.
11. **Seattle look-alikes (A4's rated guard 7)** are left out to save rating; Seattle is covered by test F1,
    (b′) recall and retention. Add it back (about 50 to 90 more cards)?
12. **Two validator accounts in the vote snapshot.** `juneau_test_votes_2026-11-17.csv` carries two
    `validator_user_id`s (`4ff63c52…`, 236 votes; `2d8335ef…`, 114 votes). The README says Jon validated. Confirm
    both are Jon before his votes train.
13. **Push and PR.** The branch is local only.

## Where each number lives

All under `analysis_out/crosswalk_s8/`.

| Number | File |
|---|---|
| Input hashes (sources and copies) | `inputs/inputs_sha256.json` |
| Juneau pano table (position, sequence, camera, detection counts) | `inputs/juneau_panos.csv` |
| Validator scores, all 3,579 labels; its report and tables | `inputs/validator_all_labels_scored.csv`, `inputs/validator_REPORT.md`, `inputs/validator_tables.md` |
| Jon's juneau-test votes; the random sample | `inputs/juneau_test_votes_2026-11-17.csv`, `inputs/random_sample_2026-11-17.json` |
| Areas, per-role counts, expected contents, the primary sample's counts | `slice/summary.json`, `slice/areas.json` |
| Every pano's role; every label's role and distance to the slice | `slice/panos.csv`, `slice/labels.csv` |
| The primary sample and reserve | `slice/primary_sample.txt`, `slice/primary_reserve.txt` |
| Candidate and planned training points, per label | `pools/candidates.csv` |
| Pool counts, cut sweep, planned counts, ignore points | `pools/summary.json` |
| Cards per s4 detection | `cards/summary.json` |
| Power table | `power/power.csv`, `power/summary.json` |
| Audit bounds and pass chances | computed in section 3 with `scipy.stats.beta` / `binom`; not a script output |

## Replication (prep only)

Code at the commit that adds this doc. CPU, under a minute except `power` (about 4 minutes).

```bash
S=scripts/analysis/stage1_crosswalk_sdot_274
L=<labeler worktree on store-records-crosswalk>   # runs/juneau_crosswalk, runs/juneau_expansion_crosswalk
V=<validator_xwalk scratch dir>                   # results/all_labels_scored.csv, REPORT.md, results/tables.md
python $S/s8_prep.py snapshot --records $L/runs/juneau_crosswalk/results.jsonl \
    $L/runs/juneau_expansion_crosswalk/results.jsonl --validator $V/results/all_labels_scored.csv \
    --validator-docs $V/REPORT.md $V/results/tables.md \
    --votes $L/runs/juneau_crosswalk/validation/juneau_test_votes_2026-11-17.csv \
    --random-sample $L/runs/juneau_crosswalk/validation/random_sample_2026-11-17.json
python $S/s8_prep.py slice       # seed 298; 3 areas, 12%, 50 m buffer; sample 200 + reserve 50 (seed 299)
python $S/s8_prep.py pools
python $S/s8_prep.py cards
python $S/s8_prep.py power       # seed 298, 300 sims, 1,000 bootstrap draws per sim
pytest tests/test_s8_prep.py
```

Everything after `snapshot` reads only committed files, so a clean clone reproduces the slice, pools, cards and
power from `inputs/`. The `results.jsonl` files themselves are gitignored in the labeler (12 MB); their sha256 are
in `inputs/inputs_sha256.json`. The validator's per-label scores came from an offline run on klone
(`inputs/validator_REPORT.md`); its crops are on klone scrubbed until archived (section 8).
