# Rubrics for every human-review pass in this benchmark

**Why this file exists.** A verdict whose definition lives only in a docstring is a verdict nobody
else can reproduce — a second rater would have to read source to learn what they were being asked.
These rubrics were extracted from the tools that produced them so they sit **beside the data they
describe**, and so a rating pass can be repeated without reverse-engineering it.

Three human passes exist. Each is defined below, with the tool that renders it and the file its
judgments land in.

---

## 1. Ground-truth verification — `benchmark/<city>/verdicts.json`

**Tool:** `python scripts/gt_gallery.py benchmark/<city>` → open `index.html` → Export.
**Scored by:** `scripts/score_validation.py`, and `rampnet/detection_eval.build_ground_truth`.

The reviewer does two things per panorama: judge **every model detection**, and **scan the whole
panorama** for ramps the model missed.

### Per-detection verdicts (`dets`)

| verdict | meaning | effect on ground truth |
| :--- | :--- | :--- |
| `true` | a real curb ramp | becomes a **GT point** |
| `false` | not a curb ramp | contributes to neither set (counts as an FP) |
| `"unsure"` | cannot tell from this imagery | becomes an **ignore point** — no model is rewarded or penalised there |
| `"duplicate"` | a second detection of a ramp already counted | neither; a redundant hit is charged as an FP by the matcher |

### Missed-ramp marks (`missed`)

Points the reviewer adds where the model detected nothing. A mark with `unsure: true` becomes an
**ignore point** rather than a GT point.

### The complete-scan attestation (`no_missed`)

Set when the reviewer has scanned the whole panorama and is confident nothing was missed. **This
gates recall**: a panorama is only counted in recall when its missed-ramp check is confirmed —
either `no_missed` is set, or at least one missed mark exists. Without that gate, unscanned panos
would deflate recall and over-weight the ones where a miss happened to be found.

### ⚠️ Resolution fairness (issue #26)

`gt_gallery.py` renders crops **and** the full-panorama view at the **model's input resolution
(4096×2048), never the native image**. Mapillary panos are commonly 11000×5500 and GSV up to
16384×8192; showing a reviewer more than the model saw would bias recall. The full-pano view has
pan/zoom so a reviewer scanning for misses sees exactly the model's pixels.

### Reviewer notes

`review_notes` (split-level: what fought the rubric, how confident the reviewer is) and per-pano
`note` both round-trip through the tool, so re-reviewing a city revises its caveats instead of
silently deleting them. `score_validation.py` prints `review_notes` above the numbers.

---

## 2. Incremental-FP A/B spot-check — `benchmark/<city>/incremental_fp_tags.json`

**Tool:** `python scripts/analysis/operating_point_curve.py gallery --city <city>
--op-threshold 0.25 --upper 0.55 --panos benchmark/<city>/panos --out <dir>`
**Scored by:** `scripts/analysis/low_floor_sweep.py corrected`.
**Issue:** #55, riding on #54.

**The problem it solves.** The benchmark's GT was assembled during a review of RampNet's detections
at a 0.55 floor. So an unmatched prediction *below* 0.55 may be a real ramp the GT never had a
chance to record — the left half of the PR curve understates precision by construction. This pass
converts that bound into a number.

**The population:** unmatched predictions scoring in `[0.25, 0.55)` — the "incremental" FPs.

| tag | key | meaning | effect |
| :--- | :--- | :--- | :--- |
| **A** | `a` | **a real ramp the GT missed** | re-counted as a **true positive** |
| **B** | `b` | **a genuine false positive** | stays an FP |
| **U** | `u` | unsure | excluded, and widens the **error band** |

The gallery pre-flags **likely duplicates** — a prediction within `2R` of an already-detected ramp
is a double-count of one ramp rather than a find, which is precisely the distinction the A/B
question turns on. Between `2R` and `3R` is ambiguous territory where genuine corner pairs live.

**Tag ids are `{pano}_{x:.5f}_{y:.5f}`** — coordinates are part of the identity, so a re-extraction
that moves a peak orphans the tag. `low_floor_sweep.py tagcheck` exists to catch that, and should
be run after any re-extraction. *(Checked 2026-07-31: 8/8 cities, 100% of tags still resolve.)*

---

## 3. Miss taxonomy — `benchmark/miss_taxonomy_46/silent__<rater>.json`

**Tool:** `python scripts/analysis/make_tagger.py benchmark/miss_taxonomy_46/silent_gallery`
**Scored by:** `scripts/analysis/tag_results.py`.
**Issue:** #46.

**This rubric travels inside each verdict file** (`scheme` field), which is the pattern the two
passes above should adopt — the file is self-describing, so it can never be compared against a
different rubric by accident. The gallery is committed, so this pass is repeatable today.

The question for each crop: **are the ramp's own pixels present, in the model-resolution panel, and
carrying its appearance?**

| verdict | meaning | programme |
| :--- | :--- | :--- |
| `visible` | the ramp itself is resolvable | **vocabulary — the sourcing target** |
| `context-only` | ramp not resolvable; crosswalk / apron / curb-cut cues imply one | learnable, from scene layout |
| `occluded` | something is physically in the way (**even if still identifiable**) | capture |
| `lighting` | exposure **destroyed** it — clipped white or crushed black | capture |
| `surface` | debris, snow, leaves, construction covering it | environment |
| `not-a-ramp` | nothing ramp-like here | GT error |
| `definition` | imagery clear; whether this **class** counts is the question (e.g. at-grade median cut-through) | rubric question |
| `unclear` | cannot tell even with context | excluded from every rate |

Full guidance, including worked cases, is rendered into the tagging page itself by
`make_tagger.py`.


---

## 4. Curb-ramp labelling — `manual_labels/*.txt` (the 1,000-pano `manual_gold` pass)

**Stated by:** Jon Froehlich, 2026-08-18.
**Issues:** #130, #132.

**This rubric is retroactive, and that is worth saying plainly.** The original 1,000-pano pass
(3,919 ramps, 207 negative panos) ran without a written rubric — `benchmark/manual_gold/gt_source.json`
records the *format* and the independence of the pass, but never what counts as a ramp. Sections 1–3
above cover the verdict GT, the FP spot-check and the miss taxonomy; the labelling pass that produced
the largest ground-truth set in this benchmark was not among them.

The gap has a measured consequence: #130 found **11 ramps double-marked across the 360° seam**, a
systematic error the rule below would have prevented, and one that survived because the labelling
viewer clamped its crop window instead of wrapping it — so a reviewer never saw the two halves as one
object.

Written down now so that a second pass measures **labelling disagreement** rather than
labelling disagreement *plus* undocumented rubric drift. Anyone comparing two passes must check
they were made under the same version.

| question | rule |
| :--- | :--- |
| Does a driveway count? | **No.** Driveway aprons are not curb ramps, however ramp-like they look. |
| Where does the point go? | On the **centre of the ramp, or just below the TSI** (the truncated-dome panel). |
| How far out do you mark? | **Every ramp you can see.** No distance cutoff. |
| May you mark a ramp you infer? | **No.** Only ramps you can actually see. A crossing implying a ramp is not a ramp. |
| Partial occlusion? | **Mark it**, if enough of the ramp is showing that you can confidently call it a ramp. |
| A ramp split by the panorama edge? | **One ramp.** We are marking *physical* ramps; the seam is an artifact of the projection. |
| Unsure? | **Yes** — mark the ramp and flag the point `unsure`. A coerced verdict is noise; an abstention is data. |
| A pano with no ramps? | The labeller must **explicitly attest** that this pano contains none. Silence is not a negative. |
| Resolution? | Label at **full stored resolution** — give the human every pixel. See the note below. |

### Tooling this rubric requires

- **A seam check.** The viewer must be able to shift the seam (90° / 180°) with the labels attached,
  so a labeller can confirm a judgment is not an artifact of where the edge happens to fall.
  `scripts/analysis/seam_review.py` implements this as its A/B views.
- **A wrapping crop window.** Not a clamping one. `rampnet/geometry.py::crop_left` is the shared
  definition; `scripts/gt_gallery.py` clamped, which is the direct cause of the #130 defect.

### The resolution question, and why it does not conflict with #26

#26 requires model-resolution review for **verdict GT**, so that recall is not inflated by ramps the
model's downsampled input never contained. That is right *for scoring*.

It is not right for labelling. Ground truth about **what ramps physically exist** must not depend on
our model's input size, or the truth changes whenever the model does. So: label at full resolution,
and **record each mark's apparent size** so a scorer can filter to "what the model could have seen"
afterwards. One pass then serves both purposes and survives a change of input resolution.
`scripts/analysis/miss_gallery.py` already computes that parity metric.

For `manual_gold` specifically the question is moot: all 1,000 panoramas are stored at 4096×2048,
which is exactly the model's input size, so reviewer pixels and model pixels are the same thing.
It binds for splits stored larger — richmond at 12288 and bend at 16384.

### Known open question

Whether two marks close together are one ramp or two is **not** settled by this rubric, and cannot be
settled by a rule. `manual_gold` holds 234 within-radius pairs away from the seam, 87 of them at
near-identical elevation on the horizon, and those are overwhelmingly genuine adjacent far-field
ramps. Adjudication is per-pair and human: `scripts/analysis/seam_review.py` for the seam cases,
nothing yet for the rest.
