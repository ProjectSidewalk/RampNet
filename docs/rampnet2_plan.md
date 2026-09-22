# RampNet 2.0 plan: find, tag, rate

**Status:** DRAFT, 2026-09-22, not yet approved. Umbrella issue: [#86](https://github.com/ProjectSidewalk/RampNet/issues/86).
Every number here is recorded on that issue's thread, or in a committed doc named at the point of
use. The scripts behind the census numbers are scratch probes and are **not yet committed**; item 1
in §4 turns them into one committed script, and until it lands none of the census numbers is
reproducible from a clean clone. Reviewed by an Opus 5 subagent on PR #174; its findings are
applied here.

## 1. Goal and where each part stands

The north star (vision note, 2026-07-23) is an AI accessibility labeler that (1) finds problems,
(2) tags fine-grained nuance, and (3) rates severity, as well as or better than humans, starting
with curb ramps because they are a designed object with an objectively measurable severity.

| goal | state | bottleneck |
|---|---|---|
| **Find** | F1 0.908 on the un-anchored manual gold set; leads the best zero-shot challenger on all twelve bundles by 0.114–0.340 F1 (`rampnet1_findings.md`). Against a YOLO baseline trained on the same data, at matched operating points, the residual is 0.039 and the tiles arm is ahead on morgantown and manual_gold (`operating_point_parity_51.md`) | close to done on GSV and car-mounted rigs; #158 (open) argues rig-native retraining is still required for open imagery (laurens_mapillary F1 0.543, recall 0.390). Per-ramp deployment recall across a city has never been measured (`rampnet1_report.md` §6); the labeler's fusion eval reports union recall 0.941 against Project Sidewalk labels, a metric its own doc caveats (`sidewalk-auto-labeler/docs/ps-clustering-eval.md`). |
| **Tag** | Usable supervision and a published baseline exist (§2) | tags are positive-unlabeled and rater-dependent; no rubric yet |
| **Rate** | Severity as recorded is a 3-level *quality* scale, 88 / 9 / 3 %. It is mostly the tag set: P(≥2) is 2.7 % untagged vs 40 % tagged, monotone in tag count; tags alone predict it at weighted κ 0.44. Two trained raters agree at weighted κ 0.21 | the recorded scale is a reporting-threshold artefact and cannot anchor a "better than humans" claim; severity stays in scope as a **derived** quantity, explained by tags now and by measurements later (§2.4, §4) |

The largest gap between the vision and the data is the third row's measurement half: no
field-measured curb-ramp ground truth exists anywhere. That needs a decision from Jon (fund a small
field campaign, or accept "screen, don't certify"), not an experiment.

## 2. What the supervision census established (all on #86)

### 2.1 Sizes

- 58 Project Sidewalk deployments returned curb-ramp labels (39 public + 19 private hosts, incl.
  DC, Taipei, Kaohsiung, the validation study; 44 public deployments were probed, 5 returned none).
  Human curb-ramp labels: **505,193**; 118,384 tagged (23 %); 296,070 validated correct; 209,632
  on private hosts, 147,937 of them DC.
- Median gap between label date and imagery capture date is 1.2–2.4 years in most cities (cuenca
  8.2 y). Temporal mismatch is the normal case for condition, not an edge case.
- Tags entered the schema on **2018-04-29** (SidewalkWebpage evolution 14). DC (2015–2018) is
  pre-tag, not tag-negative. The tag rate is flat at 26–45 % from 2019 on, so the cutoff is a
  label date, not a deployment exclusion or a calendar year.
- Each city can exclude tags through its config (`config.excluded_tags` in SidewalkWebpage, so
  `/v3/api/labelTags` is a per-city view). A tag absent in a city may be excluded rather than
  unobserved; the audit reads the per-city tag list. How many tags each city hides is not yet
  measured; item 1 records it.

### 2.2 Tags are positive-unlabeled and rater-dependent

- Validation study, 395 same-ramp pairs (dsnyde8 vs mikey): tag rate 61 % vs 14 %; when both tag
  they agree on *which* tag (missing tactile warning 15/15), so the disagreement is the threshold
  of reporting, not perception. Per-tag κ 0.00–0.21.
- Rater × era drift: "not level with street" is ~0 % for both Owners before 2021, 6–10 % on Jon's
  labels since; Jon's any-tag rate rose from 48 % (2019) to 71–82 % (2025–26); mikey's stayed
  19–48 %.
- Consequence: an untagged label is never a negative. A tag negative exists only when someone
  reviewed the full tag set and left the tag off.

### 2.3 What counts as tag-reviewed today

| set | labels | provenance |
|---|---:|---|
| HF `projectsidewalk/sidewalk-tagger-ai-validated`, CurbRamp | 10,857 (train 8,674 / test 2,183), 10 cities | the ASSETS'24 validation pass, frozen 2024-10-29; 94.8 % tag-identical to live today; validators edited 1,049 of 2,612 Seattle labels incl. 567 from empty ⇒ **absence is affirmed** |
| Expert validate (`/expertValidate`, admin only, records Agree + old/new tags) | 2,432 labels, 11 accounts, 21 deployments | the only validate UI that shows tag controls; ordinary Validate/Mobile Agree says nothing about tags |
| Trusted-rater recent labels (Owners, placed 2024+) | ~1.5k (of 3,261 tagged Owner labels all-time) | Laurens 105/105, recent Newberg/Columbus blocks. Not tag-reviewed by a second person; an anchor set, not gold |

Published baseline on the HF set (ASSETS'24, DINOv2-B): **mAP 0.34, macro-F1 0.31, micro-F1 0.67**.
Only "missing tactile warning" is strong; every other ramp tag is under ~0.4 precision at any
recall. Caveat: that split is by label, not by pano.

### 2.4 Severity, measurement and extent

- Severity as recorded behaves like a weighted count of the defects the rater bothered to note:
  P(severity ≥ 2) is 0.027 with no tags, 0.267 / 0.633 / 0.832 / 0.918 with 1 / 2 / 3 / 4 tags;
  `steep` and `not enough landing space` carry the most. A logistic model from tag indicators
  reaches quadratic-weighted κ 0.44 on tagged labels (5-fold CV). Between the two validation-study
  raters, severity agreement is κ 0.21 weighted, one-sided (dsnyde8 rated 32 ramps ≥ 2 that mikey
  rated 1), the same reporting-threshold pattern as the tags. Proposed framing (an answer to Jon's
  question on 2026-09-22, not yet a recorded decision): keep inferring severity, but as a quantity
  to be *explained* (by tags, then by measurements), not a scale to be reproduced. The
  ~6,400 untagged labels rated ≥ 2 are worth a gallery: an unnamed defect, or a rate-but-don't-tag
  rater.
- Where a tag encodes a threshold ("narrow" = 36 in; PROWAG says 48 in), a measurement (width,
  running slope, landing) gives both the tag and the continuous quantity.
- Feasibility note (2026-07-29 on #86): lateral measurements are well conditioned; slope is
  possible but limited by everything the idealisation assumes away; cross slope and lip height are
  out of reach from imagery.
- Extent: `manual_labels` boxes are tactile-pad marks, not aprons (#114). Whole-apron gold exists
  on Richmond (complete) and partially on three more (#116). SAM2 on gnomonic views is the designed
  first experiment (#83).
- GSV native per-pano depth is harvested on the labeler side (`sidewalk-auto-labeler/scripts/
  harvest_depth.py`, `runs/<city>/depth`) for bend, paterson, gainesville and sao_paulo (78.6k /
  34.4k / 35.2k / 22.7k panos on disk, 2026-09-22); not for laurens_gsv or richmond. #111, the
  RampNet-side archive of the benchmark panos, is still open.

### 2.5 The unit of ground truth is the real-world ramp

Project Sidewalk labelers inventory features, they do not annotate every pano exhaustively. So:
labels cluster to ramps (the labeler already does this); positives union across trusted raters;
absence is inferred from nothing; a reviewed item carries the pano id it was judged against.

## 3. Ground truth is built on prod, not in a side file

- Per-label edit URLs exist today: `/gallery?labelType=CurbRamp&labelId=<id>` and
  `/labelMap?labelId=<id>` open the tag and severity editor (owner or admin). A list-driven queue
  (`?labelIds=`) is filed as
  [SidewalkWebpage#5444](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5444) and being
  worked on in another session (Jon, 2026-09-22); the issue itself is open and unassigned.
- The gallery editor writes the label and a `label_history` row but **no validation row, and
  nothing when unchanged**. A review that confirms a label must also vote Agree, or it leaves no
  trace. Expert validate records both in one submit.
- Every edit is retrievable through `/v3/api/labelEdits` (old/new tags, severity, type; user;
  source; time; validation id). That pull is the reproduction path, but prod is a live, mutable
  database with private hosts, so each review pass is also **exported per rater** to
  `benchmark/tag_review/<rater>.json` with the rubric text embedded in the file (the precedent is
  `benchmark/*/boxes.json`, which carries the box rule inside it) and committed.
- Every GPU item below records a row in `analysis_out/compute_log.jsonl` per `docs/compute_cost.md`.
  Crops re-cut from the makelab2 pano store depend on an unpublished local input; the plan says so
  next to each number that uses them.
- **No new tags during a review pass.** Vocabulary ideas are parked as
  [SidewalkWebpage#5448](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5448) (ramp type
  tags) and [#5449](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5449) (blocked ramp).
- **The rubric is not decided.** It is drafted with Jon, from the labeling guide plus PROWAG
  citations, with a "when required" rule per tag, because the two-rater result says reporting
  threshold is what differs. The retired `tactile warning` tag drops out (`parallel lines` is still
  in use, 784 rated labels, and stays).

## 4. Experiments, in order

Each one names the decision it makes. GPU work runs on makelab2 or Hyak (never the desktop).

### Phase 0, instruments (this week; CPU except item 2)

1. **Audit script + doc.** One committed script with the cutoff date, tiers, host list and
   per-city excluded tags as CLI args, plus the labelEdits pull; writes
   `docs/ps_supervision_audit.md`. Turns the #86 thread into something reproducible.
2. **Tag benchmark of record.** Reproduce the ASSETS'24 numbers on the committed `test.csv` split
   first (the only split comparable to the paper) using the released checkpoint
   `validated-dino-cls-b-curbramp-tags-best.pth` from HF `projectsidewalk/sidewalk-tagger-ai-models`
   (`sidewalk-tagger-ai/REPRODUCE_RESULTS.md`); report a pano-grouped re-split beside it as the
   leak-free number; hold expert-validate as a second test set. ~1 GPU-hour if the checkpoint loads;
   retraining their recipe is the fallback and costs more.
2b. **Crop cutter.** A committed script that cuts a crop for any (city, label_id, field of view)
   from the makelab2 pano store, since only labels placed since 2023-10-12 (~196k) have production
   crops. CPU/IO; needed by items 4 and 5; its input is unpublished.

### Phase 1, tagging

3. **Rubric draft + review list** (drafted with Jon; the rubric is not decided). List stratified by
   city, distance band and tag state. Adjudicating the 94 disputed "missing tactile warning" ramps
   is deferred (Jon, #86, 2026-09-22) and they are not in this list. Jon rates first; a second
   rater (Mikey, if available; unconfirmed) repeats the same list blind. Per-tag agreement is the
   human ceiling every model is measured against.
4. **Context experiment.** Re-cut the 8,674 train and 2,183 test crops (item 2b) at three fields
   of view; train the same DINOv2 head at each; compare per-tag AP on the test split. Decides
   whether the street-dependent tags (points into traffic, not level, landing space) fail for lack
   of context or lack of vision. ~1 GPU-day.
5. **Positive-unlabeled training.** Same head on the 118k tagged plus the ~236k untagged tag-era
   labels with a PU loss, vs the clean 8.7k. ASSETS found noisy data hurt under naive training; PU
   is untested. Needs item 2b for every label without a production crop. ~2 GPU-days.
6. **One-pass detect + tag.** Per-tag channels on the RampNet keypoint head, scored at the detected
   point. A Stage 2 retrain: ~56 GPU-hours per epoch on 16 L40S (`stage2_training_cost.md`), and
   the recipe is one epoch. Gated on 4 and 5.

### Phase 1b, severity (alongside tagging; proposed, see §2.4)

- **S1, tags → severity baseline.** Done on the census (§2.4): weighted κ 0.44 from tag
  indicators against the *recorded* severity of tagged labels. Before S2 it is re-fit on the
  review-pass labels against *consensus* severity, so the two are scored on the same items and the
  same target.
- **S2, severity head on the tag benchmark.** Add an ordinal severity output to the head from
  item 4; one additional head-training run. Report weighted κ against (a) recorded severity and
  (b) consensus severity from item 3. It counts only if it beats the re-fit S1 on (b). Depends on
  items 3 **and** 4.
- **S3, severity vs measurements.** Once width / slope / landing estimates exist (items 7–8),
  regress severity on measurements + tags and report the variance each explains. If measurements
  add nothing over tags, that is the result; if they do, severity becomes a function of quantities
  a city can act on.
- **S4, anchor the scale.** The review-pass rubric gives 1 / 2 / 3 stated criteria so the consensus
  severity has a definition; the recorded scale never had one.

### Phase 2, extent and measurement

7. **SAM2 on gnomonic views** vs the Richmond whole-apron gold, as designed on #83. ~1 GPU-day,
   unblocked, prerequisite for width.
8. **Cross-view repeatability** of width and slope using the labeler's fusion clusters and its GSV
   depth archives. No ground truth needed; decides whether measurement is precise enough to justify
   field truth.
9. **Field truth.** ~50 Seattle ramps with tape and inclinometer. The only route to a defensible
   "better than humans" claim on slope or width. Jon's call. Check first whether the one public
   field-measured set (Seoul, 514 images, laser-measured width and slope, CC0 on Zenodo
   10.5281/zenodo.22699523, per the 2026-09-21 related-work note on #86) covers enough curb ramps
   to serve; its authors dropped ramps for lack of variance, so probably not, but it is free.

### Phase 3, the data engine (alongside)

10. Review pass on prod (§3) and the labelEdits pull into the audit.
11. Close the loop: AI-proposed tags surfaced in expert validate with add/remove thresholds (the
    tagger repo already specifies the rule); human review of proposals becomes the clean training
    set.

## 5. Start order

Items 1, 2, 2b, 3 and 7 are independent and can begin now; 4 follows 2 and 2b within days; S2
follows 3 and 4. No Stage 2 retrain (item 6) until 4 and 5 have reported; S3 waits on 7 and 8.

Deliberately out of scope: absence detection from the 130k NoCurbRamp labels (well supervised, 58 %
tagged, but a fourth goal).
