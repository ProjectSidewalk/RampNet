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
