# Should seam-wrapped NMS be the default? (#248)

**Issue:** [#248](https://github.com/ProjectSidewalk/RampNet/issues/248). **Status:** in progress.
This commit holds only the **pre-registration**: the context, every count already seen, the
rating protocol and the decision rule. It was committed and pushed on 2026-11-15, before any
score was computed under wrapped NMS, before any card was rendered or looked at, and before any
model peak pair was joined to Jon's seam verdicts. Results are added in later commits; the git
history is the evidence of the order.

Plan: written by Fable 5.2 (`claude-fable-5-2`) and revised after an adversarial review by Opus 5.5
(`claude-opus-5-5`), both on 2026-11-15. Implementation and AI pre-rating: Opus 5.5
(`claude-opus-5-5`).

## 1. Context

[PR #249](https://github.com/ProjectSidewalk/RampNet/pull/249) added
`rampnet.subcell.detect_peaks(wrap_nms=True)` (default off; [`seam_nms_248.md`](seam_nms_248.md)).
It counted straddling pairs on Run A peaks but measured no detection metric and nothing on the
published model. The open question is whether the default should flip.

The published model's peaks are already committed for both pools that matter, so tonight's study is
**CPU only, from committed files**:

| Pool | File(s) | How extracted |
|---|---|---|
| `manual_gold_tta` | `benchmark/manual_gold/records.jsonl` (`detections_meta.json`: `rampnet_stage2_hub606a1195.pth`, `tta: true`, floor 0.05, `peak_min_distance` 10) | flip TTA, `exclude_border=False`; the README-headline configuration |
| `cities13` | `analysis_out/input_res_sweep_25/cache/r2048/<split>.json` (11 splits) plus `analysis_out/op_cache/{bayonne,juneau}.json` | single pass, `threshold_sweep.peaks_to_dets(h, 0.05, 10)` |

`cities11` (the 11 r2048 splits) is reported beside `cities13`. manual_gold is never pooled with
the cities.

**What stored peaks can and cannot say.** Write `d = 10`. Without exact ties the default output is
every pixel that is the maximum of its clipped `(2d+1)` window; wrap keeps the subset with no higher
pixel in the across-seam part of the window. So, per threshold `t`:

- `C`, **certain removals**: a strip peak (heatmap column < 10 or >= 1014) with a stronger stored
  peak across the seam at wrapped Chebyshev distance <= 10 px. Exact wrap removes it for any
  heatmap, TTA included (preconditions on the committed data: no tied pairs, every pair <= 8 px).
- `E`, **envelope-only candidates**: strip peaks not in `C` with a stronger stored peak across the
  seam within 20 px. A heuristic for removals caused by the slope of a stronger peak whose maximum is
  more than 10 px away (the heatmap is an x8 bilinear upsample, so a slope is one or two coarse cells
  wide). Whether wrap removes an `E` peak is **unknown** without the heatmap. The 30-px count is
  reported as a sensitivity.
- A peak counts at `t` when it is >= `t` (its stronger partner then is too).

Scenario `lo` removes `C`; scenario `hi` removes `C` and `E`. Nothing else changes.

## 2. Everything already seen before this rule was written (section 4.0 of the plan)

Counts only. Nobody has computed TP/FP/FN, P, R, F1 or AP under wrap, joined any model pair to
`seam_verdicts__jon.json`, or looked at any crop.

Seen by the planner and, independently, by the reviewer:

| | peaks >= 0.30 | strip | pairs | `C` | 20-px env. | 30-px env. | peaks >= 0.55 | strip | pairs | `C` | 20-px | 30-px |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cities, r2048, 11 splits (1,289 panos) | 2,880 | 36 | 7 | 7 | 12 | 13 | 2,257 | 16 | 4 | 4 | 4 | 4 |
| manual_gold TTA records (1,000 panos) | 3,981 | 38 | 8 | 8 | 14 | 15 | 3,610 | 23 | 6 | 6 | 9 | 10 |

- Pair distances (wrapped Chebyshev px): cities {4, 4, 4, 4, 4, 4, 7} at 0.30 (bend 4, 4, 7;
  budapest_district5, morgantown, paterson, richmond 4 each); manual_gold {1, 4, 4, 4, 4, 4, 7, 7}.
- Per split at 0.30, strip / pairs: annapolis 2/0, bend 7/3, budapest_district5 11/1, clovis 3/0,
  gainesville 1/0, laurens_gsv 0/0, laurens_mapillary 0/0, morgantown 5/1, paterson 3/1,
  richmond 4/1, sao_paulo 0/0. At the 0.05 floor per split, pairs: 1, 3, 10, 3, 2, 0, 0, 4, 1, 1,
  0 in the same order.
- op_caches: bayonne 39 strip / 8 pairs at the floor, 4 strip / 0 pairs at 0.30; juneau 13 / 1 at
  the floor, 2 / 0 at 0.30; laurens_mapillary 4 / 0; the other 11 files and all of `op_cache_tta`,
  0. The `C` and `E` counts for bayonne and juneau had **not** been computed when the rule was
  committed. **Appended by the `bounds` commit, before any card existed:** bayonne `C` 0 / `E` 0
  and juneau `C` 0 / `E` 0 at both 0.30 and 0.55 (at the 0.05 floor the two add 9 `C` and 5 `E`).
  So `cities13` and `cities11` hold the same 12 candidates at 0.30. `bounds` also reproduced every
  number in the table above exactly, found 3,919 manual_gold GT points, and found one pair of tied
  across-seam peaks (manual_gold, both clipped to 1.0, 23 px apart: outside the 20-px envelope, so
  it touches neither `C` nor `E`; the reviewer's "0 tied pairs" was within 10 px and holds).
- manual_gold records: 5,072 peaks, 132 strip peaks at the floor, 270 scores equal to 1.0 (one in
  the strip, unpaired). r2048: 33 raw scores >= 1.0, none in the strip. 0 tied pairs.
- City `benchmark/*/records.jsonl` (labeler output): 0 strip peaks in all 13 splits.
- Also seen: PR #249's Run A counts, the labeler's Laurens counts, and Jon's verdict totals from
  [`seam.md`](seam.md) (10 `one` pairs at 2.95 to 12.81 px, 4 `two` pairs at 17.28 to 25.06 px).
- Seen by the implementer before this commit, beyond the above: the schema (keys only) of the input
  files, and `imagery_manifest.py --verify --panos-root D:/Git/RampNet`, which reported all 14 local
  splits byte-identical to their manifests (2026-11-15, 21 s). No verdict value of Jon's was read.

What this means for blindness: the size of the removal sets was known when the rule was written, so
the rule does not gate on any count of duplicates. It gates only on the ratings, which do not exist
yet.

## 3. Rating protocol (blinded, calibrated)

**Cards.** One card per member of `C` or `E` at 0.30 in `cities13` and `manual_gold_tta`, except a
manual_gold card that inherits Jon's existing verdict (below), plus **14 calibration cards**: Jon's
adjudicated GT seam pairs (`benchmark/manual_gold/seam_verdicts__jon.json`, marks `a` and `b`). All
in one deck, shuffled with seed 248, each with an opaque id.

**What a card shows.** Crops of the panorama rolled so the seam is at the centre, wide enough for
both marks plus context, and a zoom; two marks labelled A and B, the letters assigned at random
(seed 248). **No scores, no "removed" or "kept" label, no separation in pixels or degrees, no pool
or pano id on the card face, no GT marks.**

**Questions per card.**
1. Pair: `same` (A and B are the same physical object) / `different` / `unsure`.
2. Per mark, A and B: `ramp` / `not_ramp` / `unsure` (rubric: [`seam_review.py`](../scripts/analysis/seam_review.py)
   `RUBRIC`, version 1; a driveway is not a curb ramp; mark only what is visible).

**Files.** `analysis_out/seam_nms_default_248/review/items.json` (opaque id, deck order, seed,
sha256 of each card image) is committed with the sheet. The key (opaque id to pano, pool, which mark
is the weaker peak, set `C` / `E` / calibration) is written to `review/key.json` and **committed only
after the verdict file**, so it is not in the tree while rating. Card images and the HTML are
git-ignored and regenerate from `sheet --panos-root <checkout>`; their hashes are in `items.json`.

**The AI pre-rater (Opus 5.5).** Views each card image in deck order. Does not open `key.json`,
`seam_verdicts__jon.json`, the table in [`seam.md`](seam.md) section 2, or any pano-id lookup until
the verdict file is committed and pushed. `unsure` is a legitimate answer and costs nothing: it sends
the card to Jon. **This blinding is on the honour system** (Jon's verdict file is in the repo).
Verdicts are committed as `review/verdicts__claude-opus-5-5.json` (rubric text, version and the
item-list digest inside) **before** `report` is written or run.

**Calibration.** After the verdict commit, and before any study card is classified or any score is
computed, the AI is scored on the 14 calibration cards against Jon (agreement on the 10 `one` and on
the 4 `two`, by separation band), and that result is committed. **The AI is "calibrated" only if it
answered `different` on all 4 of Jon's `two` pairs.** If it is not, the rating route stops: the AI's
verdicts are not used, every un-inherited card counts as `U`, and the outcome follows from the rule.

**Verdict inheritance on manual_gold (the join).** A `C` or `E` card inherits Jon's existing verdict
only if its two peaks are each nearest to a **different** mark of one adjudicated pair, each within
6.5 px wrapped Chebyshev on the 1024 x 512 grid. Otherwise the card goes on the sheet, including
every card on a `two` panorama (Jon's `two` pairs are 17 to 25 px apart, every model pair is 1 to
8 px apart, so a model pair there is most likely the split of one ramp).

**Classification of a card** ("weaker" is the peak wrap would remove):

| Pair answer | Weaker mark | Class |
|---|---|---|
| `same` | any | `D` (duplicate) if the stronger mark is `ramp`, else `N` |
| `different` | `ramp` | `L` (a real ramp would be lost) |
| `different` | `not_ramp` | `N` (a non-ramp would be removed) |
| `different` | `unsure` | `U` |
| `unsure` | any | `U` |
| not rated | | `U` |

An inherited Jon verdict maps `one` to `D`, `two` to `L` (both marks are GT ramps) and `unsure` to
`U`. **Effective verdict of a card:** Jon's if he has one (inherited, or from a later
`review/verdicts__jon.json` audit); otherwise the AI's if the AI is calibrated; otherwise `U`.

## 4. Decision rule (pre-registered 2026-11-15)

### 4.1 Quantities

At threshold 0.30, over `cities13` plus `manual_gold_tta`:

- `L_jon_C`: cards in `C` classed `L` by **Jon** (inherited or audited).
- `L_opt_C`: cards in `C` classed `L` by their effective verdict.
- `L_pess`: cards in `C` or `E` classed `L` or `U`.

### 4.2 Outcomes, evaluated in this order

1. **KEEP OPT-IN AND CLOSE** if `L_jon_C >= 1`, or `L_opt_C >= 2`. Wrap certainly removes those
   peaks, and they are real, different ramps.
2. **FLIP, CONDITIONAL ON JON'S AUDIT** if the AI is calibrated and `L_pess <= 1`. The
   recommendation is "make wrap the default once Jon has rated the removal cards and the result still
   satisfies this clause". If Jon has already rated every `C` and `E` card, the outcome is simply
   **FLIP**. If `L_pess = 1`, the doc names that card first.
3. **NEXT STEP NEEDED** otherwise. The doc names the step exactly: the `k` cards Jon must rate
   (every `U`, every AI `L`, and all cards if the AI is not calibrated), and, for any `E` card
   classed `L` or `U`, the panoramas whose heatmaps an exact (GPU) tier must produce to learn whether
   wrap removes that peak at all.

An `E` card cannot cause KEEP: its removal is not established. If it is a different real ramp, the
right next step is a forward pass on that one panorama, not a verdict.

**Reported, never gating:** all score deltas (P, R, F1, paired 95% CIs; `lo` and `hi`; GT-A and
GT-B for manual_gold, where GT-B merges Jon's 10 `one` pairs in memory to their wrapped midpoint),
the `D` / `L` / `N` / `U` counts, the 0.55 versions of every quantity, and `cities11` beside
`cities13`. At these counts (at most 3 changed panoramas in any split) every per-split CI includes 0
by construction, which is not evidence of no effect. **Tripwire:** if the pooled paired dF1 under
`lo` is below -0.001 in `cities13` or in manual_gold GT-B (at 0.30 or 0.55), the doc must explain
the conflict with the ratings before stating any recommendation.

The rule deliberately has no minimum duplicate count: whether a rare, correct fix is worth moving a
published number is Jon's value judgement. Anything decided outside this rule is post hoc and will be
labelled so; if new information makes the rule unworkable, it is amended openly here and the affected
read is demoted to exploratory.

Structural point: inside the strip, wrapped NMS does what the default already does at every interior
column, which is to suppress the weaker of two peaks within 10 px. Two real ramps closer than 10 px
are already merged everywhere else in the image.
