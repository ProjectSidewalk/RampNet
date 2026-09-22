# RampNet 2.0 plan: find, tag, rate

**Status:** DRAFT, 2026-09-22, not yet approved. Umbrella issue: [#86](https://github.com/ProjectSidewalk/RampNet/issues/86).
Every number here is on that issue's thread with the script that produced it; nothing below is re-derived.

## 1. Goal and where each part stands

The north star (vision note, 2026-07-23) is an AI accessibility labeler that (1) finds problems,
(2) tags fine-grained nuance, and (3) rates severity, as well as or better than humans, starting
with curb ramps because they are a designed object with an objectively measurable severity.

| goal | state | bottleneck |
|---|---|---|
| **Find** | F1 0.908 on the un-anchored manual gold set; wins every benchmark split by 0.12–0.34 F1 | rig shift on open imagery (#151, #158); per-ramp deployment recall already 0.94 via the labeler's fusion. Not the 2.0 bottleneck. |
| **Tag** | Usable supervision and a published baseline exist (§2) | tags are positive-unlabeled and rater-dependent; no rubric yet |
| **Rate** | Severity as recorded is a 3-level *quality* scale, 88 / 9 / 3 %. It is mostly the tag set: P(≥2) is 2.7 % untagged vs 40 % tagged, monotone in tag count; tags alone predict it at weighted κ 0.44. Two trained raters agree at weighted κ 0.21 | the recorded scale is a reporting-threshold artefact and cannot anchor a "better than humans" claim; severity stays in scope as a **derived** quantity, explained by tags now and by measurements later (§2.4, §4) |

The largest gap between the vision and the data is the third row's measurement half: no
field-measured curb-ramp ground truth exists anywhere. That needs a decision from Jon (fund a small
field campaign, or accept "screen, don't certify"), not an experiment.

## 2. What the supervision census established (all on #86)

### 2.1 Sizes

- 58 Project Sidewalk deployments answer the rawLabels API (44 public + 20 private hosts, incl. DC,
  Taipei, Kaohsiung, the validation study). Human curb-ramp labels: **505,193**; 118,384 tagged
  (23 %); 296,070 validated correct; 209,632 on private hosts, 147,937 of them DC.
- Tags entered the schema on **2018-04-29** (SidewalkWebpage evolution 14). DC (2015–2018) is
  pre-tag, not tag-negative. The tag rate is flat at 26–45 % from 2019 on, so the cutoff is a
  label date, not a deployment exclusion or a calendar year.
- Each city excludes tags through its config (Seattle hides 28 of the master list). A tag absent in
  a city may be excluded rather than unobserved; the audit reads the per-city tag list.

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
| Trusted-rater recent labels (Owners, 2024+) | ~3k | Laurens 105/105, recent Newberg/Columbus blocks |

Published baseline on the HF set (ASSETS'24, DINOv2-B): **mAP 0.34, macro-F1 0.31, micro-F1 0.67**.
Only "missing tactile warning" is strong; every other ramp tag is under ~0.4 precision at any
recall. Caveat: that split is by label, not by pano.

### 2.4 Severity, measurement and extent

- Severity as recorded behaves like a weighted count of the defects the rater bothered to note:
  P(severity ≥ 2) is 0.027 with no tags, 0.267 / 0.633 / 0.832 / 0.918 with 1 / 2 / 3 / 4 tags;
  `steep` and `not enough landing space` carry the most. A logistic model from tag indicators
  reaches quadratic-weighted κ 0.44 on tagged labels (5-fold CV). Between the two validation-study
  raters, severity agreement is κ 0.21 weighted, one-sided (dsnyde8 rated 32 ramps ≥ 2 that mikey
  rated 1), the same reporting-threshold pattern as the tags. So: keep inferring severity, but as a
  quantity to be *explained* (by tags, then by measurements), not a scale to be reproduced. The
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
- GSV native depth is already archived on the labeler side for four cities (#111 done there).

### 2.5 The unit of ground truth is the real-world ramp

Project Sidewalk labelers inventory features, they do not annotate every pano exhaustively. So:
labels cluster to ramps (the labeler already does this); positives union across trusted raters;
absence is inferred from nothing; a reviewed item carries the pano id it was judged against.

## 3. Ground truth is built on prod, not in a side file

- Per-label edit URLs exist today: `/gallery?labelType=CurbRamp&labelId=<id>` and
  `/labelMap?labelId=<id>` open the tag and severity editor (owner or admin). A list-driven queue
  (`?labelIds=`) is in progress on
  [SidewalkWebpage#5444](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5444).
- The gallery editor writes the label and a `label_history` row but **no validation row, and
  nothing when unchanged**. A review that confirms a label must also vote Agree, or it leaves no
  trace. Expert validate records both in one submit.
- Every edit is retrievable through `/v3/api/labelEdits` (old/new tags, severity, type; user;
  source; time; validation id), so the review pass is reproducible from a clean clone.
- **No new tags during a review pass.** Vocabulary ideas are parked as
  [SidewalkWebpage#5448](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5448) (ramp type
  tags) and [#5449](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5449) (blocked ramp).
- **The rubric is not decided.** It is drafted with Jon, from the labeling guide plus PROWAG
  citations, with a "when required" rule per tag, because the two-rater result says reporting
  threshold is what differs. Two dead tags (`tactile warning`, `parallel lines`) drop out.

## 4. Experiments, in order

Each one names the decision it makes. GPU work runs on makelab2 or Hyak (never the desktop).

### Phase 0, instruments (CPU, this week)

1. **Audit script + doc.** One committed script with the cutoff date, tiers, host list and
   per-city excluded tags as CLI args, plus the labelEdits pull; writes
   `docs/ps_supervision_audit.md`. Turns the #86 thread into something reproducible.
2. **Tag benchmark of record.** Re-split the HF test set by pano; reproduce the ASSETS'24 numbers
   with their released checkpoint under our scorer; hold expert-validate as a second test set.
   ~1 GPU-hour.

### Phase 1, tagging

3. **Rubric draft + review list** (with Jon). List stratified by city, distance band and tag state,
   including the 94 disputed "missing tactile warning" ramps. Jon and Mikey review the same list
   blind; per-tag agreement is the human ceiling every model is measured against.
4. **Context experiment.** Re-cut the 2,183 test labels from the makelab2 pano store at three
   fields of view; train the same DINOv2 head at each; compare per-tag AP. Decides whether the
   street-dependent tags (points into traffic, not level, landing space) fail for lack of context
   or lack of vision. ~1 GPU-day.
5. **Positive-unlabeled training.** Same head on the full 118k tagged + untagged with a PU loss vs
   the clean 8.7k. ASSETS found noisy data hurt under naive training; PU is untested. ~2 GPU-days.
6. **One-pass detect + tag.** Per-tag channels on the RampNet keypoint head, scored at the detected
   point. A Stage 2 retrain, gated on 4 and 5.

### Phase 1b, severity (alongside tagging; no GPU beyond items 4–6)

- **S1, tags → severity baseline.** Done (§2.4): weighted κ 0.44 from tag indicators. This is the
  number any severity model has to beat.
- **S2, severity head on the tag benchmark.** Add an ordinal severity output to the head from
  item 4, trained on the same labels; report weighted κ against (a) the recorded severity and
  (b) the consensus severity from the review pass (item 3). It counts only if it beats S1 on (b).
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
   "better than humans" claim on slope or width. Jon's call.

### Phase 3, the data engine (alongside)

10. Review pass on prod (§3) and the labelEdits pull into the audit.
11. Close the loop: AI-proposed tags surfaced in expert validate with add/remove thresholds (the
    tagger repo already specifies the rule); human review of proposals becomes the clean training
    set.

## 5. Start order

Items 1, 2, 3 and 7 are independent and can begin now; 4 follows 2 within days, with S2 riding on
it. No Stage 2 retrain (item 6) until 4 and 5 have reported; S3 waits on 7 and 8.

Deliberately out of scope: absence detection from the 130k NoCurbRamp labels (well supervised, 58 %
tagged, but a fourth goal).
