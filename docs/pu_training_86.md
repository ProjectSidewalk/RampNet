# Positive-unlabeled training of the tag head (#86, RampNet 2.0 plan item 5)

**Status: PROPOSED, 2026-09-24. Nothing here has run.** This is the plan for item 5 of
[`rampnet2_plan.md`](rampnet2_plan.md) §4, written for review before any crop is cut or any job
is submitted. Every number is taken from a committed document or a committed table, named at the
point of use; the few that are arithmetic on those numbers say so. Decisions Jon has not made are
marked PROPOSED in the text and collected in §8.

Umbrella issue: [#86](https://github.com/ProjectSidewalk/RampNet/issues/86). Inputs it builds on:
the supervision audit (item 1, `ps_supervision_audit.md`, PR #175), the benchmark of record
(item 2, `tag_benchmark_86.md`, PR #178), the crop cutter (item 2b, `crop_cutter.md`, PR #177)
and the context experiment (item 4, `context_fov_86.md`, PR #180).

## 1. Question, and the decision it makes

The plan's wording (item 5): *same head on the 118k tagged plus the ~236k untagged tag-era labels
with a PU loss, vs the clean 8.7k. ASSETS found noisy data hurt under naive training; PU is
untested. Needs item 2b for every label without a production crop. ~2 GPU-days.*

The question in one line: **does the tag head get better when it is trained on every human
curb-ramp label of the tag era, with the untagged ones treated as unlabeled rather than
negative, than when it is trained on the reviewed ASSETS'24 crops alone?** The reviewed set here
is the part of the 8,674 HF train crops that survives the test exclusions (§3.1: at most 4,279),
because the rest sit on or beside the test panoramas; the published model trained on all 8,674
is reported as a reference row.

Why it is not obvious. The ASSETS'24 paper ran exactly the naive version of this experiment on
its whole 33-class dataset (a ~24k cleaned set against a ~87k "larger but noisier crowdsourced"
set) and reports that the larger set was *detrimental* (`sidewalk-tagger-ai` README, abstract).
The audit explains why that should happen for curb ramps: an untagged label is not a negative.
Tags are positive-unlabeled (plan §2.2; audit §4, §7): the two validation-study raters tagged the
same ramps at 61 % vs 14 %, and when both tagged they agreed on *which* tag, so the difference is
the threshold of reporting, not perception. Training with "no tag = negative" teaches the model
that threshold. A PU loss is the standard tool for exactly this label structure; whether it
recovers what the naive run loses is the thing nobody has measured.

**What "beats" means.** Per-tag average precision and mAP on the benchmark's eight scored tags
(`tag_benchmark_86.md` §1: the tags with at least 10 test positives), on three test sets, all
pano-clustered bootstrap CIs as the benchmark reports them:

1. the committed HF `test.csv` rows (2,183 labels), the only split comparable to the published
   0.34 mAP;
2. the pano-grouped re-split's held-out panos (`resplit_pano_grouped_seed86.csv`, 2,197 labels),
   the leak-free read;
3. the expert-validate corpus (2,452 labels, 22 deployments, audit §6) as a second, independent
   test set, once its crops exist (§2.4; this is also the unfinished last clause of item 2,
   `tag_benchmark_86.md` §8).

The comparison is paired on test labels: every new arm, and the matched `clean` control, is
scored on the same rows, and none of them has trained on or within 10 m of any of the three test
sets. A gain counts if the paired difference against `clean` has a 95 % interval that excludes
zero on (1) *and* (2); (3) is reported beside them. The #178 arms are reference rows only, each
on the one test set it did not train on (§3.1, §8 decision 13).
The benchmark's seed variance is unmeasured (`context_fov_86.md` §6), so a difference smaller than
a seed's worth is not a result; §7 says what that costs to fix.

**Decision it makes.** If PU beats the clean set: the 2.0 tag head trains on the full label
store, and the review pass (item 3) is a test set, not a training set. If naive beats clean:
the reporting-threshold reading is wrong somewhere and §7 says what to look at. If neither beats
clean: the tag head is data-limited by *reviewed* labels, and the effort goes into the review
pass and the close-the-loop item (plan item 11), not into more crops.

## 2. The label universe

All counts are human CurbRamp labels (SidewalkAI's account excluded by user id), from
`analysis_out/ps_audit/tiers.csv` on the audit branch at `fdd73c6` (PR #175's post-merge
timestamp fix, which keeps 354 labels whose timestamps lack fractional seconds).
**`main` still carries the pre-fix table** (tag era 361,863, tagged 118,293) until that commit
lands in a follow-up PR; the differences are 0.1 % and change nothing below, but the plan cites
the fixed numbers so the scripts written from it match.

### 2.1 Tiers

| tier | tag-era labels (placed ≥ 2018-04-29) | tagged | untagged | with a production crop |
|---|---:|---:|---:|---:|
| tier 1: every human label | 362,217 | 118,402 | 243,815 | 196,742 |
| tier 2: crowd-validated correct (`correct == true`) | 295,957 | 89,514 | 206,443 | 168,952 |
| tier 3: placed by an Owner | 8,063 | 3,277 | 4,786 | 1,564 |

`untagged` is arithmetic on the two columns beside it. The production-crop column counts labels
placed since 2023-10-12, which is inside the tag era, so it is the same number in every era.

**PROPOSED: the universe is tier 2, not tier 1.** Tier 2 is a *position* tier: an Agree from
`/validate` or `/mobile` says the ramp is there, not that its tags are right (audit §2). That is
exactly what a PU loss wants: every crop really shows a curb ramp, and only the tag set is
uncertain. Tier 1 adds 66,260 labels that nobody confirmed, and a crop of a non-ramp with
"unlabeled" tags is noise a PU loss cannot model (its unlabeled term assumes the class prior of
the positive-labeled population). The cost is 28,888 tagged labels (118,402 − 89,514). Tier 1 is
recorded as a not-run arm in §7.

### 2.2 Which tags

The eight tags the benchmark scores, and no others: *missing tactile warning, points into
traffic, narrow, surface problem, not level with street, not enough landing space, debris /
pooled water, steep* (`tag_benchmark_86.md` §2). The other four in the union vocabulary are out:
`tactile warning` is retired (0 of its 471 HF train positives still carry it, `tag_benchmark_86.md`
§2), `parallel lines` and `not aligned with crosswalk` are city-specific (1 and 3 cities with
labels, audit §4), `not visible` has 5 labels.

Observed positive rate in the tag era, tier 2, the proposed universe (`tags.csv`
`labels_correct` over 295,957; `plan_numbers.json`, `observed_rate_tier2`): missing tactile
warning 15.4 %, points into traffic 9.3 %, narrow 4.9 %, surface problem 3.7 %, not enough landing
space 3.5 %, not level with street 2.8 %, debris / pooled water 1.3 %, steep 0.93 %. Tier 1's
(audit §4) are 16.9, 9.2, 5.3, 4.2, 3.6, 4.0, 1.3 and 1.3 %. These are the *labeled* frequencies
the PU loss sees, not the class priors (§3.2).

### 2.3 Excluded tags are masked cells, not negatives

Each deployment lists its own tag subset (`config.excluded_tags`; audit §4 and
`analysis_out/ps_audit/excluded_tags.csv`). Where a tag is not offered, a label cannot carry it,
so its absence there says nothing. Of the eight target tags, two deployments hide some: amsterdam
hides *missing tactile warning* and *points into traffic* (yet 771 and 809 of its tag-era labels
carry them anyway, placed before the exclusion, audit §4); chandigarh-india lists only 3 tags and hides six of the
eight (119 labels). Everywhere else all eight are listed.

Rule: the loss is computed per (label, tag) cell, and a cell whose tag is not in the label's
deployment tag list at fetch time is **masked** out of both the positive and the unlabeled term
(a tag carried anyway, as in amsterdam, stays a positive). The mask comes from the audit's
`labelTags` pull, keyed on deployment, so it is reproducible from the committed table. This is a
few lines in the loss and must not be skipped: with amsterdam's 12,246 tag-era labels treated as
missing-tactile negatives, the second-strongest tag in the benchmark acquires 12k wrong
negatives.

### 2.4 Nothing in a test set, or near it, is in training

Keys are the composite `(city, label_id)` (`label_uid`), never `label_id` alone. Removed from
every training arm, by panorama, not by label:

- every panorama carrying an HF `test.csv` label (1,869 panos, `tag_benchmark_86.md` §1) and every
  panorama in the pano-grouped re-split's test side (1,283 panos, §5 there), plus every other
  label of the same city within 10 m of any of those test labels, the benchmark's stricter rule
  (`tag_benchmark_86.md` §3: the same corner seen from a neighbouring panorama);
- every panorama carrying one of the 2,452 expert-validate labels, and the same 10 m rule;
- every panorama carrying one of the 500 item-3 review items
  (`benchmark/tag_review/review_list.csv`, PR #176), and the same 10 m rule, because a PU win
  makes the review a test set (§1);
- the validation slice of §5.3, which is itself drawn only from HF train rows that survive the
  rules above.

The exclusion list is written once, committed as `analysis_out/pu_training_86/excluded_panos.csv`
with the reason per row, and asserted by the test suite against every arm's training table
(training ∩ each test set ∩ slice = ∅, by panorama and by the 10 m rule) and against the slice
itself (slice ∩ each test set = ∅, same two rules).
The four HF train labels with no live row (`tag_benchmark_86.md` §3) are simply absent.

## 3. The PU formulation

### 3.1 Arms

Every arm is the benchmark's recipe (`tag_benchmark_86.py train`: DINOv2-B/14 with registers,
full fine-tune, Adam 1e-6, batch 4, BCE-with-logits, no augmentation, seed 86) with only the
training table and the loss changed.

**The existing #178 control cannot be the paired control.** Crossing the committed
`hf_curbramp_labels.csv` with `resplit_pano_grouped_seed86.csv` (both on `bench/tag-benchmark-86`;
`analysis_out/pu_training_86/plan_numbers.json`, from `scripts/analysis/pu_training_86_plan.py`):

- **1,772 of the re-split's 2,197 test labels (81 %) are HF train rows**, so #178 `train_control`
  trained on them and cannot be scored on test (2). Symmetrically, 1,758 of the 2,183 HF test
  labels are re-split train rows, so #178's `pano` arm cannot be scored on test (1).
- Applying §2.4's exclusions to the 8,674 HF train rows removes 3,565 that sit on a panorama of
  either test set and 830 more within 10 m of a test label of the same city, leaving **4,279**
  (4 of them the labels with no live row), before the expert-validate and review-list exclusions,
  which need those sets' coordinates. A new arm cannot carry the other half of the HF train rows
  without training on or beside the test sets, so it cannot be a superset of #178's control.
- On test (1), #178's control keeps the label-level leak (56 % of test labels share a panorama
  with train, `tag_benchmark_86.md` §1) and the new arms do not. That bias runs against the new
  arms; it was bounded at about 0.02 mAP at epoch index 4, and the 100-epoch bound is not yet
  measured (`tag_benchmark_86.md` §5).

**PROPOSED: a new, matched `clean` control, and the #178 arms as reference rows.**

| arm | training labels | loss on an untagged label | what it isolates |
|---|---|---|---|
| **clean** (control, new run) | the HF train rows that survive §2.4, minus the §5.3 selection slice: at most 4,279 − ~430 ≈ 3,850 reviewed crops, cut at the §4.1 framing | negative (correct: the ASSETS'24 pass affirmed empty tag sets, audit §8) | the reviewed set alone, matched to the new arms in exclusions, framing and selection rule |
| **naive** | tier 2 tag era minus §2.4, united with every `clean` row (on `label_uid`; an HF row that is not tier 2 or not tag-era is added, not dropped) | negative for every tag | the ASSETS'24 "Experiment 2" arm on curb ramps; the thing PU is supposed to fix |
| **nnPU** | same table as naive | unlabeled: non-negative PU risk per tag with a class prior per tag (§3.2) | the experiment |
| soft-negative (PROPOSED, only if the budget allows a fourth arm) | same table | negative with a soft target (§3.2) instead of 0 | the cheapest fix; tells whether nnPU's unlabeled term is doing anything a soft target does not |
| reference: #178 `train_control` | all 8,674 HF train rows, HF screenshot framing | negative | the published benchmark; scored on test (1) only, leak included |
| reference: #178 `pano` | the re-split's 8,660 train rows, HF framing | negative | the leak-free recipe; scored on test (2) only |

The two reference rows are **pending**: the 100-epoch #178 arms died with the makelab2 reboot of
2026-09-23 and were relaunched from scratch at 2026-09-24 04:09 UTC, ending about 20:30 UTC
(PR #178 comment of 2026-09-24 04:12 UTC). They answer "does the store beat the published
tagger", unpaired in data and framing; the `clean` control answers "does the store add to the
reviewed rows available under the same exclusions", which is the paired question. The cost of
the matched control: it carries about half the HF train rows, so it will score below the
published model, and it is one more job, 100 epochs over ~3,850 crops, ~50 s per epoch at the
measured ~77 crops/s (§5.1), about 1.4 h plus preprocessing on one L40S (arithmetic).

Both `naive` and `nnPU` contain every `clean` row at the same framing, so every new arm is a true
superset of the control. On the HF rows the absence *is* affirmed, so their cells stay hard
negatives in every arm (they are the only labels in the universe for which that is true, plus the
expert-validate rows, which are test); the loss paragraph below says how they enter.

**The loss per tag, PROPOSED.** For tag t, every unmasked (label, tag) cell falls in one of three
disjoint sets: P_t, labels carrying t; N_t, HF rows (the `clean` rows) not carrying t, whose
absence the ASSETS'24 pass affirmed; and U_t, every other tier-2 label not carrying t. ℓ is the
**logistic loss**, ℓ(z, y) = log(1 + e^(−yz)), which is the control's BCE-with-logits, so `nnPU`
and `naive` differ only in how U_t is treated, not in the loss family. (Kiryo et al., 2017 used
the sigmoid loss in their experiments, and their estimation-error analysis assumes a bounded
loss, which the logistic loss is not; a sigmoid-loss `nnPU` is a not-run ablation, §7.)

- `naive`: ℓ(f_t, −1) on every U_t and N_t cell, ℓ(f_t, +1) on every P_t cell; the batch mean,
  as in the recipe.
- `nnPU`: the non-negative PU risk of Kiryo et al. (2017) on the non-HF rows, in its
  case-control form, where the unlabeled sample U*_t is *every* unmasked non-HF label, tagged t
  or not, and π_t is the marginal class prior of §3.2:

      R_PU,t = π_t · E_P[ℓ(f_t, +1)] + max(0, E_U*[ℓ(f_t, −1)] − π_t · E_P[ℓ(f_t, −1)])

  plus the ordinary BCE on the HF rows (their positives and their affirmed negatives), each crop
  weighted by its share of the table as in `naive`, so the HF rows carry the same weight in both
  arms (about 1 % of the table). A PNU-style mixing weight on the HF term (Sakai et al., 2017) is
  not tuned; an up-weighted HF term is a not-run ablation.

**Why U*_t includes the tagged labels.** The first draft paired an untagged-only U_t with π_t,
which mixes two settings. With U restricted to the labels *not* tagged t, the right prior is the
positive fraction among untagged labels, π_U,t = (π_t − o_t) / (1 − o_t), where o_t is the tag's
observed rate in the table; using π_t there over-subtracts the negative risk, so the clamp fires
for the wrong reason. The two correct forms are algebraically identical: expanding
E_U*[ℓ] = o_t · E_P[ℓ] + (1 − o_t) · E_U[ℓ], the bracket becomes (1 − o_t) · E_U[ℓ(f_t, −1)] −
(π_t − o_t) · E_P[ℓ(f_t, −1)], which is (1 − o_t) times the untagged-only bracket with π_U,t.
The trainer implements the case-control form (one prior per tag, no o_t needed at train time),
and the loss PR's unit test checks it against the censoring form on a hand-computed batch. Both
rest on the selected-completely-at-random assumption: a true positive's chance of being tagged
does not depend on what the crop looks like. The rater and era effects (audit §7) say that is
only approximately true; §7 records it.

**The estimator at batch 4, PROPOSED.** In tier 2, *steep* is on 2,738 of 295,957 labels (0.93 %,
audit `tags.csv`), so a batch of 4 holds a *steep* positive about 3.7 % of the time
(1 − 0.9907⁴, arithmetic), and a per-batch E_P for *steep* is empty in ~96 % of batches. The
first draft left that undefined. The proposal is the per-sample form with **global**
normalisation: for a batch B of size b drawn uniformly from the N non-HF rows,
Ê_P[g] = (N / (b · n_P,t)) · Σ_{i ∈ B ∩ P_t} g_i and Ê_U*[g] = (1 / b) · Σ_{i ∈ B} g_i. Both are
unbiased over batches, both are defined for any batch (an empty P term is 0, never NaN), and the
clamp is applied to the batch's estimate as in the paper, with its gradient-ascent step when the
bracket falls below −β (β and the step size γ are CLI arguments, recorded in the run meta). The
cost is variance: one *steep* positive in a batch carries a weight of N / (4 · n_P) ≈ 27. The
trainer logs, per tag, how often the clamp fires; a rare tag whose clamp fires on most steps is
the signal for the batch-16 follow-up (§8 decision 12). Not chosen: positive-stratified sampling
(changes the data distribution and needs re-weighting) and a running-average clamp (a
refinement of the same estimator, left for the batch-16 run).

**PROPOSED: a tagged label that lacks tag t is unlabeled for t, not negative.** A rater who
tagged *narrow* looked at the ramp and did not tag *steep*; that is stronger evidence than an
untagged label, but the validation study says the reporting threshold varies by rater and the
audit says it varies by year, so it is not an affirmed absence either. Treating it as unlabeled
is the conservative reading; the alternative is a cheap ablation on `nnPU` and is listed in §7.

### 3.2 The class prior per tag

nnPU needs π_t, the fraction of curb ramps that *truly* carry tag t, which the observed rate o_t
under-reports by the labeling frequency c_t = o_t / π_t. Four candidate sources; the first three
are committed, the fourth is pending.

| source | n labels | missing tactile | points into traffic | surface problem | narrow | landing | not level | steep | pooled water |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **HF train rows that survive §2.4** (proposed; `plan_numbers.json`, `priors.survivors`) | 4,279 | 42.5 % | 13.2 % | 7.9 % | 9.3 % | 6.4 % | 4.2 % | 1.5 % | 1.2 % |
| ASSETS'24 validated set, all 10,857 rows (reference only: includes the 2,183 HF test rows and 1,772 re-split test rows; positives from the #86 comment of 2026-09-22T12:50, rates in `plan_numbers.json`, `priors.hf_all_10857`) | 10,857 | 38.9 % | 12.7 % | 9.9 % | 8.5 % | 5.8 % | 4.1 % | 1.4 % | 1.4 % |
| jonfroehlich, own labels 2024–2026 (`rater_drift.csv` from PR #183, rate weighted by labels per year) | 1,134 | 38.4 % | 20.4 % | 9.5 % | 16.1 % | 11.6 % | 9.0 % | 4.4 % | — |
| mikey, own labels 2024–2026 (same) | 364 | 30.2 % | 15.1 % | 8.5 % | 1.6 % | 2.7 % | 1.7 % | 0.0 % | — |
| item-3 review, `untagged` stratum (`tag_rubric_draft.md`, PR #176; not yet reviewed) | 150 | measures π_U,t directly, see below | | | | | | | |

`rater_drift.csv` does not carry pooled water. The two trusted raters disagree by 2× or more on
four of the seven tags they share (narrow 16.1 vs 1.6, landing 11.6 vs 2.7, not level 9.0 vs 1.7,
steep 4.4 vs 0.0; missing tactile, points into traffic and surface problem are within 1.4×),
which is the reporting-threshold effect again, now inside the trusted tier.

**The item-3 review measures the quantity PU needs.** PR #176's review list samples 150
`untagged` items (no tags, never tag-reviewed) from crop-era labels in 35 deployments,
explicitly as "where the review measures how often an untagged label actually carries a
condition, which is the number the positive-unlabeled experiment (plan item 5) needs"
(`tag_rubric_draft.md`, Strata). That is a direct estimate of π_U,t, from the population the PU
arms train on rather than from the HF set's 10 cities. At 150 items its intervals are wide (a
20 % rate is about ±6.4 points at 95 %, arithmetic), so it checks the prior rather than
replacing it, and it only exists once the review is done. Its 500 items are excluded from
training (§2.4), since a PU win makes the review a test set (§1).

**Labeling frequencies against the universe's own rates.** The universe is tier 2, so o_t is
the tier-2 rate (`tags.csv` `labels_correct` over 295,957; §2.2). With the proposed prior
(`plan_numbers.json`, `pu_tier2.survivors`):

| | missing tactile | points into traffic | surface problem | narrow | landing | not level | steep | pooled water |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| o_t, tier 2 | 15.4 % | 9.3 % | 3.7 % | 4.9 % | 3.5 % | 2.8 % | 0.93 % | 1.3 % |
| c_t = o_t / π_t | 0.36 | 0.71 | 0.47 | 0.52 | 0.55 | 0.67 | 0.61 | 1.08 |
| π_U,t (untagged labels truly carrying t) | 32.1 % | 4.2 % | 4.3 % | 4.7 % | 3.0 % | 1.4 % | 0.6 % | 0 (clamped) |

Three readings. First, c_t is not ordered by rarity: *surface problem* is tagged about as
rarely as *missing tactile warning*, and *steep* and *not level* sit mid-range, so the first
draft's "the rarer the defect the more likely it was tagged" does not hold. (That draft also
compared the prior against tier-1 rates; tier 2's are lower, e.g. *not level* 2.8 % vs 4.0 %.)
Second, where c_t ≥ 1, π_U,t is clamped to 0 and nnPU reduces to `naive` for that tag: with this
prior that is *pooled water* (with the all-10,857 prior its c_t is 0.92), so any PU effect can
only show on the other seven. Third, c_t compares a prior from 10 cities with an observed rate
from 56 deployments, so it mixes the labeling frequency with the city mix; it is a plausibility
check on the prior, not a measurement of c_t.

**PROPOSED: π_t from the HF train rows that survive §2.4**, because the HF set is the only
source whose absences were affirmed over a full tag set (one ASSETS'24 validation per label,
10,356 on 10,356 labels, five editor accounts pooled; audit §6 and §8), and because only the
survivors are free of test rows: a prior from all 10,857 sets a training hyperparameter from
test labels. The survivors are 2.9× the trusted-rater 2024+ set. They are a spatially selected
subset (rows away from any test label), and their rates differ from the full set's by up to 3.6
points (*missing tactile*, 42.5 % vs 38.9 %); both rows are committed so the difference is
visible. The trusted-rater rates are the sensitivity check (one extra `nnPU` run with Jon's
priors, §7, if the main result is close), and the item-3 `untagged` stratum checks π_U,t once
reviewed. The known bias travels with the prior: 10 cities and the ASSETS'24 reviewers' own
threshold. Everything about the prior is a CLI argument (`--prior-csv`), never a constant edited
in a session.

**The soft-negative target.** The first draft set the target for an untagged cell to 1 − c_t,
which is P(untagged | positive). The target for an untagged cell is P(positive | untagged), which
is π_U,t. For *missing tactile warning* that is 0.32, not 0.64; for *steep* it is 0.006, not
0.39, about 65× smaller (`plan_numbers.json`). Proposed: target π_U,t on U_t cells, 0 on HF
negatives, 1 on positives.

## 4. Cut plan

### 4.1 What gets cut, and at which field of view

**Every training label is re-cut from the makelab2 pano store, the production crops are not
used.** Production stores a browser crop only for labels placed since 2023-10-12 (168,952 of
tier 2), framed by the labeler's viewport at their zoom, and served through a signed,
referer-checked route (`tag_benchmark_86.md` §8); mixing those with cutter crops for the older
127,005 labels would put two framings in one arm. One cutter, one field of view, one JPEG encode
for every crop, including the HF train and test rows (whose re-cut at the control's framing is
the `viewport` arm of item 4).

**The field of view comes from item 4's results: PROPOSED `fov25`.** `context_fov_86.md` §4
(PR #180, on `exp/context-fov-86`) scored four framings on 2,182 common test rows, paired
against an interim control (the #178 control's epoch-index-49 snapshot) and against `viewport`:

| arm minus reference | mAP, paired 95 % interval |
|---|---|
| `viewport` − control | −0.010 [−0.035, +0.017] |
| `fov25` − control | −0.006 [−0.035, +0.020] |
| `fov25` − `viewport` | +0.004 [−0.024, +0.025] |
| `fov50` − control | **−0.036 [−0.064, −0.009]** |
| `fov50` − `viewport` | **−0.026 [−0.053, −0.002]** |
| `fov90` − control | **−0.075 [−0.105, −0.048]** |

So `fov50` and `fov90` are measurably worse (*surface problem*, *steep* and *missing tactile
warning* all drop at 50°), and `fov25` and `viewport` tie the control and each other. The
choice between the two tied framings is a trade-off:

- **`fov25` (proposed):** label-centred, one field of view for every label, one JPEG encode, and
  independent of the labeler's zoom. Zoom varies by rater, and rater is the axis the PU loss is
  trying not to learn, so a framing that does not depend on it is the safer default. Its one hint
  (*points into traffic* +0.045 [−0.003, +0.089] against the control) needs seeds before it is
  anything.
- **`viewport`:** reproduces the control's framing (the labeler's zoom, 12.5°–48°, ramp
  off-centre), so a PU arm would differ from the HF-framed reference rows (§3.1) only in data. It
  needs each label's zoom, re-saves the 640 px box as a second JPEG encode, and had one tag off 0
  against the control (*surface problem* −0.054 [−0.102, −0.009]).

Item 4's control column is interim until the #178 re-run finishes (`context_fov_86.md` §6); if
the final control changes this reading, the default is revisited before the cut. The cut command
is the cutter's, unchanged (`scripts/crop_cutter.py --fov 25 --size 640 --aspect 1.0 --tilt mm`,
PR #177).

Crops to cut: 295,957 (tier 2 tag era) plus the expert-validate 2,452, minus overlap. Store
coverage on a seeded 5,000-label tag-era sample was 99.16 % (`crop_cutter.md` §4); the misses
are infra3d cities (0 / 26, no directory in the store) and 16 GSV panos. Expect about 2,500
labels to drop with status `missing_pano`, recorded per row in the manifest as item 4 did
(`dropped_labels.csv`).

### 4.2 Throughput, and the worker cap

Measured on item 4's cut (`analysis_out/context_fov_86/cut_summary_*.json`, makelab2, 48 cores):

| pass | crops | panos opened | workers | wall | crops / s | labels / s |
|---|---:|---:|---:|---:|---:|---:|
| three fields of view in one pass | 32,559 | 6,295 | 12 | 1,767 s | 18.4 | 6.1 |
| viewport (1440×960) | 10,853 | 6,295 | 12 | 1,122 s | 9.7 | 9.7 |

The pano decode dominates (one decode serves all three crops of a label in the first row), so a
single-field-of-view pass over 296k labels runs at roughly 6–10 labels/s at 12 workers, **8–14 h**,
and about twice that at 6 workers. That is an estimate from two passes over 6,295 panos; the first
thing the cut does is time the same 5,000-label coverage sample and write the measured rate into
this section before the full run.

**12 workers is not allowed again.** The item 4 cut at 12 workers beside the three #178 training
arms drove makelab2's load to 60, its sshd refused connections from about 12:30 UTC, and the box
was rebooted at 15:15 UTC on 2026-09-23, killing all three arms at epoch 62–66 of 100 with no
resume (PR #178 comments of 2026-09-23 13:23 UTC and 2026-09-24). PROPOSED rule: **at most 6
workers, `nice 10`, and never while a GPU job of ours is training on makelab2**; the #178 rerun
launched 2026-09-24 04:09 UTC ends about 20:30 UTC the same day, and the cut waits for it. The
alternative, cutting from makelab1 over NFS, keeps makelab2's CPUs free but reads 15 TB of store
over the network and its throughput is unmeasured; it is the fallback if makelab2 is busy, after a
timed sample run.

### 4.3 Size and transfer

Bytes per crop for item 4's crops, **observed on klone and not committed** (2026-09-24,
`find -printf %s` over the 10,848 crops per arm): fov25 155.0 KB, fov50 161.2 KB, fov90 133.6 KB,
viewport after the 640 px box 123.1 KB. At fov25's 155.0 KB, 295,957 crops are about **46 GB**
(arithmetic). The cutter's manifest records every crop's `bytes` and `sha256`, so the real total
is summed from it and committed with the cut summary rather than projected.

Neither makelab nor klone holds a key for the other, so item 4's tar went makelab → this
desktop → klone in two hops; the 6.3 GB push needed one resume (observed during item 4's
transfer, not recorded in a committed file). Eight times that is a day of
relay. PROPOSED: split the tar per city (`tar` one archive per deployment, sha256 each), push with
`rsync --partial` from the desktop, and verify the per-city sums on klone before unpacking; the
manifest already carries a per-crop sha256, so an incomplete city is detected, not guessed. A
direct makelab → klone path (a key on makelab1 for klone, or Globus) is Jon's call and would
remove the relay entirely; it is not assumed.

### 4.4 The crops are an unpublished input

The makelab2 pano store is the Project Sidewalk scraper's archive, not a published dataset
(`crop_cutter.md`, top). So the training crops for `naive` and `nnPU` cannot be re-derived by
someone without the store, and every number from those arms says so beside it. What a reader
without the store can still check: the committed training tables (label ids, tags, priors, mask,
exclusion list), the per-crop sha256 manifests, and the CPU scoring of the committed per-label
predictions. What would publish the input: a Hugging Face dataset of the 640 px crops (~45 GB, the
same shape as `sidewalk-tagger-ai-validated`, which ships 30.8 GB of crops of the same labels'
subset today). Whether to publish is a decision for Jon (§8); the doc records the gap until then.

## 5. Training on klone

### 5.1 Measured cost of the recipe

| where | train crops | s / epoch | source |
|---|---:|---:|---|
| klone L40S, one arm per GPU (item 4, `gpu-l40s-makelab`) | 8,666 | 111.5–113.3 mean over 100 epochs (arms `fov25`, `fov50`, `viewport`) | `analysis_out/context_fov_86/train_<arm>_train_log.csv` on klone, to be committed with #180 |
| makelab2 A40, three arms sharing | 8,674 | ~605 | `tag_benchmark_86.md` §5 |

So the recipe processes **~77 crops/s on a dedicated L40S** (8,666 / 112 s), and one epoch over
tier 2's 295,957 crops is **1.06 h**; over tier 1's 362,217 it would be 1.30 h. Item 4's jobs
ran 3:58–4:57 h wall for 100 epochs including preprocessing and inference (`sacct`, jobs
40486691–40486693).

### 5.2 Epoch budget

The recipe is 100 epochs over 8,674 crops, 867k sample presentations. On tier 2 that is
**2.9 epochs**. Two budgets, both PROPOSED, run in this order:

1. **Presentation-matched, 3 epochs** (~3.2 h per arm on one L40S, plus prep). The same number
   of optimizer steps as the published recipe (867k presentations), so a difference from the
   reference rows is data, not compute. The matched `clean` control runs the recipe's 100 epochs
   over its ~3,850 crops, about 385k presentations, so it sees fewer steps than any new arm; that
   is stated beside its row, not corrected by running it longer. Cheap enough to run `naive`,
   `nnPU` and the soft-negative arm all three: ~10 GPU-hours, plus ~1.4 h for `clean`.
2. **30 epochs** (~32 h per arm) for `naive` and `nnPU`, ~64 GPU-hours, with a checkpoint scored
   on the validation slice every 5 epochs so the curve says whether 30 was enough. This is the
   arm the headline comes from.

The plan's "~2 GPU-days" is budget 1 plus one 30-epoch arm; both 30-epoch arms are ~3 GPU-days.
Stated so the ledger is not a surprise.

### 5.3 Selection without reading the test set

The recipe keeps the checkpoint with the best *training* exact-match accuracy
(`tag_benchmark_86.md` §5). Under a PU loss that number is meaningless (the targets are not
labels), and on 296k crops the last epoch is not necessarily the best. PROPOSED: hold out a
**10 % pano-grouped slice of the HF train rows that survive §2.4** (10 % of at most 4,279, so
about 430 reviewed labels, absence affirmed, seed 86) from every arm, `clean` included, score
every 5-epoch checkpoint on it (every 10 epochs for `clean`'s 100), and select by its mAP over the
tags with at least 10 slice positives (the benchmark's rule, `tag_benchmark_86.md` §1). The
slice is drawn from the survivors, never from all 8,674 HF train rows: 1,772 of those are test (2)
rows and 3,565 sit on a test panorama (§3.1). It is the only place in the universe with affirmed
negatives that is not a test set.

The slice is small. At the survivors' rates (`plan_numbers.json`) 430 labels carry about 6–7 *steep*
and 5 *pooled water* positives, so those two tags will usually fall below the 10-positive rule
and selection rests on the other six. A 20 % slice (~860) would fix that and take another ~430
rows out of `clean`, which already has half the HF train rows; 10 % is proposed, 20 % is the
alternative (§8 decision 6).

Because `clean` is a new run, it is selected by the same rule, so the selection asymmetry the
first draft carried is gone for the paired comparison. The #178 reference rows keep their
published selection (best training exact-match accuracy), stated beside them.

### 5.4 What the trainer needs before a 30-hour job is safe

`tag_benchmark_86.py train` was written for 8.7k crops and one sitting. Three changes, each a
small PR against the benchmark script with its own test, before any budget-2 job is submitted:

- **Preprocess once, outside the job.** The trainer decodes every crop into a uint8
  3×266×266 tensor in memory at start: 212,268 bytes per crop, 62.8 GB for tier 2, and the
  decode ran at 3.3–24.8 crops/s from `/gscratch` in item 4's jobs (`prep_s` 349–2,634 s for
  8,666 crops, `train_<arm>_train_meta.json`), i.e. **3–25 h of serial decode per job**. A
  `prep` stage writes the uint8 array once as a `.npy` memmap on `/gscratch/makelab` (one file
  per arm framing, sha256 in the manifest), in parallel on a CPU allocation; the trainer maps it.
  The 48 GB the item 4 jobs requested becomes `--mem=96G` for the memmap's page cache, well
  inside an L40S node's 1.5 TB.
- **Resume.** Save model, optimizer, epoch and the shuffle generator's state every epoch; on
  start, load if present. The #178 arms were lost to a reboot for want of this, and it is the
  precondition for `ckpt-all`, where `PreemptMode=REQUEUE` (`scontrol show partition`). The
  stateless-schedule rule from `stage2_cosine_rung_135.md` applies: anything that depends on the
  step index is recomputed from the index, never carried in optimizer state.
- **The PU loss and the mask** as a `--loss {bce,nnpu,soft}` switch with `--prior-csv` and
  `--mask-csv`, unit-tested against a hand-computed batch (positive term, unlabeled term, the
  non-negative clamp, a masked cell contributing zero to both).

Batch size stays at 4 (PROPOSED). Raising it would cut wall-clock and improve the unlabeled-term
estimate, and would also change the recipe the control was trained under; if the 3-epoch read
looks promising, a batch-16 `nnPU` run is the first follow-up, not part of this comparison.

### 5.5 Where, and the rows it writes

`gpu-l40s-makelab` (`MaxTime=UNLIMITED`, no preemption, one L40S per job, 8 CPUs) for every arm,
one Slurm job per arm as `context_fov_86.slurm` does it; `ckpt-makelab` only once resume exists
and only for budget 2, since a requeue there costs the epoch in flight. Each job writes an
`in_progress` row at start and an `ok` row at the end with the same `run_id` to
`analysis_out/usage_log.jsonl` (`tag_benchmark_86.py log-usage --host --gpu`, as #180 does), and
`scripts/analysis/slurm_usage.py` adds the `sacct` record to `analysis_out/compute_log.jsonl`
after each job (`compute_cost.md`). Both ledgers live in the main checkout and are committed with
the results, never left in the run directory. The cut and the prep stage record their wall-clock
in the cut summaries and the prep manifest respectively.

## 6. What we learn under each outcome

| outcome on tests (1) and (2), paired, 30 epochs | reading | next |
|---|---|---|
| `nnPU` > `clean`, `naive` ≤ `clean` | the store is usable supervision once absence is treated as unknown; ASSETS'24's negative result was the loss, not the data | train the 2.0 tag head on the full store; the review pass is a test set; item 6 (one-pass detect + tag) inherits the PU loss |
| `naive` > `clean` too | the untagged labels are closer to negatives than the two-rater study suggests, or the volume outweighs the label noise at this size | check which tags gained: if only *missing tactile warning*, the gain is the easy tag and the read is "more data helps the visible tag"; report per tag |
| neither > `clean` (and the 3-epoch read agrees) | the head is limited by reviewed labels, not by crops | stop cutting; the effort is the review pass and the closed loop (plan items 10–11); keep the arms' checkpoints for item 6's ablation |
| `nnPU` < `naive` | the priors are wrong or the unlabeled term at batch 4 is too noisy | the sensitivity run with Jon's priors and a batch-16 run, before any conclusion about PU |

Per-tag results are the content either way. The street-dependent tags (*points into traffic, not
level, landing space*) gained nothing from wider context at any field of view in item 4
(`context_fov_86.md` §4), which left two readings: they are limited by the labels
(positive-unlabeled, rater-dependent), or they need context and resolution together. This
experiment tests the first. A gain concentrated on those tags supports it; no gain on them, with
gains elsewhere, points to the second, which a wider single crop cannot test and a larger input or
a two-crop model would.

## 7. Not run, and caveats that travel with the numbers

- **One seed per arm.** Same caveat as item 4; the benchmark's seed variance is unmeasured. A
  second seed of `nnPU` at budget 1 (~3 h) is the cheapest way to put a number on it and is
  PROPOSED as part of the run if the paired difference at budget 2 is under ~0.02 mAP, the
  size of the leak bound in `tag_benchmark_86.md` §1.
- **Prior mis-estimation is confounded with label noise.** If `nnPU` fails, the priors are the
  first suspect (§3.2); the sensitivity run with Jon's 2024+ rates separates the two only
  partly, because those rates are one rater's threshold. A prior estimated from the data (e.g.
  the KM or TIcE estimators) is a further arm, not run.
- **Selected completely at random is assumed, not shown.** Both forms of the nnPU risk (§3.1)
  assume a true positive's chance of being tagged does not depend on the crop. Rater and era
  effects (audit §7) make it at best approximate; a PU result is conditional on it.
- **Not run: a sigmoid-loss `nnPU`** (the loss of Kiryo et al.'s experiments; the plan uses the
  logistic loss so that `nnPU` and `naive` share a loss family, §3.1), and **an up-weighted HF
  term** (a PNU-style mixing weight on the affirmed-negative rows).
- **Rater × era drift** (audit §7). The universe spans 2018–2026 and the reporting threshold moved
  over it; a PU loss with one prior per tag assumes a single labeling frequency. A per-era prior
  is the natural refinement and is not run.
- **Tier 1 not run.** The 66,260 unvalidated tag-era labels are excluded (§2.1).
- **"Tagged but not this tag" as negative, not run.** The ablation of §3.1's PROPOSED rule.
- **The expert-validate test set depends on its crops existing**, which is the same cutter run;
  until then tests (1) and (2) carry the result.
- **Training crops are an unpublished input** (§4.4). The control's crops are public.
- **The framing decision inherits item 4's result**, including its one-seed caveat and its
  interim control column (the #178 epoch-index-49 snapshot, until the re-run finishes).
- **The #178 reference rows are selected by a different rule** than the new arms and `clean`
  (best training accuracy vs the §5.3 slice), and each is scored on one test set only (§3.1).
- **The 8 scored tags are the benchmark's**; a tag with no test positives cannot be scored no
  matter how many training labels it gains (`parallel lines`, `tactile warning`).

## 8. Decisions for Jon

1. **Universe:** tier 2 (validated correct, 295,957 tag-era labels) as proposed, or tier 1
   (362,217)?
2. **Framing:** cut at `fov25` (proposed: ties the control and `viewport` in item 4, centred,
   independent of the labeler's zoom) or `viewport` (ties too, and reproduces the control's
   framing, at the cost of a second JPEG encode and a per-label zoom); `fov50` is out, measured
   worse (−0.036 [−0.064, −0.009] mAP against the control). Re-cut the HF rows the same way, with
   no production crops in any arm?
3. **Priors:** the HF train rows that survive the test exclusions (4,279) as the main prior
   (proposed; the all-10,857 rates include test rows and are shown for reference only),
   trusted-rater 2024+ as the sensitivity check, and the item-3 review's 150 `untagged` items as
   a direct check on π_U,t once reviewed?
4. **Arms:** `naive` + `nnPU` at both budgets, soft-negative (target π_U,t on untagged cells,
   §3.2) at budget 1 only?
5. **Budgets:** 3 epochs (presentation-matched) then 30 epochs; ~3 GPU-days on the lab L40S
   allocation for the full set, against the plan's ~2?
6. **Selection:** a 10 % pano-grouped slice (~430 labels) of the HF train rows that survive the
   test exclusions, held out from every arm including `clean`, or 20 % (~860, keeps *steep* and
   *pooled water* scoreable, costs `clean` another ~430 rows)?
7. **Cut host and cap:** makelab2 at ≤ 6 workers, `nice`, after the #178 rerun finishes; makelab1
   over NFS only after a timed sample?
8. **Transfer:** per-city tars relayed through the desktop, or set up a direct makelab → klone
   path?
9. **Trainer changes first:** the prep memmap, resume and the loss switch as small PRs against
   `tag_benchmark_86.py` before any 30-epoch job?
10. **Tagged-but-not-this-tag** as unlabeled (proposed) or negative?
11. **Publish the cut crops** as a Hugging Face dataset so the arms are reproducible without the
    store, or record them as an unpublished input?
12. **Batch size** stays 4 for the comparison, batch 16 as a follow-up?
13. **Control per test set:** a new `clean` control trained on the surviving HF train rows at the
    §4.1 framing, selected on the §5.3 slice, as the paired control on all three test sets
    (proposed; ~1.4 h on one L40S), with #178 `train_control` as a reference row on test (1) and
    #178 `pano` as a reference row on test (2)? The alternative, pairing against #178's control,
    scores it on its own training labels on test (2) and is not proposed.
