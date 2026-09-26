# Supervised transfer: Mapillary Vistas (#126, #137, #163)

The Mapillary Vistas arms: the supervised-transfer baseline, its instrument checks, the
richmond result, resolution parity, complementarity with RampNet, and the cascade gate.
Moved verbatim from [`model_comparison.md`](model_comparison.md) under #145, where it sat under
§ "What each model class buys you"; heading levels are unchanged, so the document starts at
`###`. "Above" and "below" in the moved text refer to `model_comparison.md` when the thing
they point at is not in this file.

### Supervised transfer: Mapillary Vistas (#126)

Every model above is **zero-shot** — a general VLM told what a curb ramp is, or an
open-vocabulary detector given the phrase. None of them has ever been trained on a curb ramp.
So the roster answers *"can a general model be prompted to do this?"*, and the YOLO baseline
(#51) answers *"architecture versus data, within our dataset"*. Neither answers the third
question: **do somebody else's supervised curb-cut labels transfer to deployment panoramas?**

That question is sharpened by the existing results rather than academic. OWLv2 and Grounding
DINO reach recall 0.85–0.97 at precision 0.03 on richmond: the concept is findable, the
discrimination is not. A supervised-transfer arm tests directly whether real labels — just not
*our* labels — fix the precision side.

`facebook/mask2former-swin-large-mapillary-vistas-semantic` is a public checkpoint requiring
**no training and no dataset download**. Verified against its published `config.json` on
2026-08-18: it has a **65-class head** — the Vistas v1.2 label set — carrying **`Curb Cut`
(id 9)** and **`Curb` (id 2)**. The 124-class v2.0 set is not needed. (For contrast,
Cityscapes has no curb-related class at all, which is why Vistas specifically is the dataset
worth the effort here, not scene-parsing datasets in general.)

**This is a baseline. It is not, and must not become, a supervision source.** The RampNet
paper (arXiv 2508.09415) already reviewed this exact class and rejected it as a data source:
*"their categorization was overly broad and included driveways labeled as curb cuts."* That
assessment stands and is cited rather than rediscovered. It also makes a testable prediction —
driveway aprons should appear as a characteristic false-positive mode — which
`scripts/analysis/fp_taxonomy.py` can name directly.

**Two arms**, because Vistas draws the ramp/curb boundary somewhere we do not, and whether
recall hides on the other side of it is measurable:

```bash
python scripts/model_comparison/compare.py benchmark/richmond \
    --models rampnet,vistas:curb-cut,vistas:curb-cut+curb
```

The spec's `model_id` slot carries the **class set**, not a model id — the checkpoint comes
from `--vistas-model` — so these are the one provider whose label cannot be derived from its
spec, and it is declared in `rampnet/roster.py`'s `LABEL_OVERRIDES` instead.

**Masks become points by connected component**, one point at each component's centroid, scored
by the mean class confidence over it. One point *per class per view* would cap recall at 1 and
place that point in the empty space between two real ramps; a semantic segmenter has no notion
of instances, so the components supply them. The score is carried through, which puts this arm
in the OWLv2/Grounding DINO/YOLO tier that gets AP and a PR curve rather than the chat-VLM tier
pinned at one operating point — and the precision side is the entire question. `min_area_px` is
a **cache floor**, not the operating point, exactly as `--score-threshold` is for the
open-vocabulary detectors. In practice, at the input resolution described below, it is
**inert**: masks arrive at 96×96 and are upsampled 10.67×, so the smallest component that
upsample can produce is on the order of 114 px and the 16 px floor drops nothing. It is left
at 16 rather than recalibrated because it sits in the cache signature — changing it would
orphan both arms' published detections to no effect.

**That inertness is a property of 384×384, not of the floor, and it does not carry to the
parity run below.** At 1024×1024 input the mask logits arrive at 256×256 and are upsampled
only 4×, so the smallest component the upsample can produce is about **16 px — exactly the
floor**. `min_area_px=16` is therefore marginally *binding* at parity where it was provably
inert at 384, and it is the one setting shared by both that does not mean the same thing in
each. It was still left untouched, because changing it would confound the single variable the
parity run exists to isolate.

The **rig** is the identical six-view one every other tiled leg uses
(`equirect_tiling.default_views()`: 6 yaws, 90°×90°, pitch −30°, 1024×1024, source capped at
4096) — but **the rig is not what the model sees, and that is a real caveat on the headline
below.** The checkpoint's own `preprocessor_config.json` is
`{"height": 384, "width": 384}` with `do_resize`, so each 1024×1024 view is downsized to
**about 1/7 the pixel area** before inference; the mask logits come back at 96×96 and are
bilinearly upsampled 10.67× to the view. **So "supervised transfer does not compete" is
measured under a resolution handicap no other leg carries, and it is not a like-for-like
comparison on input.** It was also entirely unrecorded until the review of this PR: nothing
pinned it, nothing overrode it, and it was not in the detection signature, so a `transformers`
upgrade could have changed every mask under an unchanged cache key.

What changed here: `--vistas-input-size H W` now overrides the processor, and
`--vistas-revision` pins the checkpoint. Both are recorded in the signature **only when set**,
so the published richmond detections keep their key and nothing already paid for is orphaned —
and a future run at parity is a distinct, self-describing cache entry. **The parity run has now
been done** — see *Resolution parity* below. Every number in the table that follows is still at
384×384, and the parity numbers are reported separately rather than replacing them: the 384 run
is the one every other Vistas figure in this document was read from, and the parity arm is its
own published leg (`mask2former-vistas-curb-cut-1024x1024`, #163).

Note for anyone reading #126: that issue says `scripts/box_gallery.py` already cuts
perspective views. **It does not** — its `--fov` sizes an axis-aligned crop of the
equirectangular image, so the distortion is preserved, which is the wrong input for a
perspective-trained checkpoint.

#### Instrument checks, before any number

Run on richmond, RTX 3070, fp16, **2.3 s/view**:

- **The semantic map is assembled here, not taken from `post_process_semantic_segmentation`**,
  because that helper returns the argmax only and this arm needs a per-pixel confidence. The
  two agree on **99.73%** of pixels; the residual is at class boundaries.
- **The scary load warning on this checkpoint is noise.** transformers reports
  `pixel_level_module.encoder.swin.layernorm.{weight,bias}` as MISSING and newly initialized.
  Measured: scrambling them to (7.5, −3.25) changes **0 of 1,048,576 pixels**, because the
  Mask2Former path consumes per-stage hidden states, not the final normed output. Recorded so
  nobody else has to re-derive it.
- **Overlay first, always.** `dump_detections.py --masks` also dumps the class mask per view,
  which the points cannot substitute for: a wrong class id yields confident points on the wrong
  object and reads as an ordinary bad model. On richmond pano `1273933840289887`, all four
  predictions in the yaw-240 view land on the yellow detectable-warning pads, scores 0.49–0.79,
  no coordinate offset — and **nothing on the vehicle hood**, so the #47 concern did not
  materialise there.

Class ids are pinned constants so `signature()` works without weights (a fresh clone must be
able to reconstruct signatures with no GPU), and `_ensure_ready` cross-checks them against the
loaded `id2label` — segmenting class 9 of a *different* label set would not raise, it would
quietly score the wrong object.

#### Result: richmond (124 panos, 310 GT ramps), 2026-08-18

RampNet's row reproduces its committed numbers exactly (0.964 / 0.768 / 0.855, 238/9/72), which
is the check that this run is comparable to the ones above. The leg was also run twice, either
side of a change to the detector signature that altered the cache key without touching the
detections; both runs agree to every digit printed here.

| model | P | R | F1 | AP | tp/fp/fn |
|---|---|---|---|---|---|
| **rampnet** | **0.964** | 0.768 | **0.855** | **0.763** | 238/9/72 |
| gemini-3.1-pro-preview | 0.634 | 0.703 | 0.667 | – | 218/126/92 |
| gemini-3.6-flash | 0.626 | 0.642 | 0.634 | – | 199/119/111 |
| **mask2former-vistas-curb-cut** | **0.411** | **0.697** | **0.517** | **0.513** | 216/309/94 |
| molmo2-8B (points) | 0.410 | 0.516 | 0.457 | – | 160/230/150 |
| Qwen3-VL-32B-Instruct | 0.760 | 0.297 | 0.427 | – | 92/29/218 |
| Qwen3-VL-8B-Instruct | 0.323 | 0.452 | 0.377 | – | 140/293/170 |
| *mask2former-vistas-curb-cut+curb* | *0.126* | *0.648* | *0.210* | *0.089* | 201/1399/109 |
| owlv2-large-patch14-ensemble | 0.033 | **0.971** | 0.064 | 0.104 | 301/8799/9 |
| grounding-dino-base | 0.028 | 0.852 | 0.053 | 0.033 | 264/9321/46 |

**Every row above is at one operating point — no confidence floor — which is what the rest of
this document's roster tables use.** An earlier version of this table scored the two Vistas rows
at `--op-threshold 0.30` and carried the OWLv2 and Grounding DINO rows over at 0, then compared
them; the comparison below is at a common point. For the three rows that carry confidences, here
is what the deployment threshold does:

| model @ conf ≥ 0.30 | P | R | F1 |
|---|---:|---:|---:|
| mask2former-vistas-curb-cut | 0.419 | 0.697 | 0.524 |
| owlv2-large-patch14-ensemble | 0.146 | 0.110 | 0.125 |
| grounding-dino-base | 0.030 | 0.090 | 0.045 |

**The question this arm was built to answer gets a clear answer: supervised transfer fixes most
of the precision problem, and does not close the gap.** Against the open-vocabulary detectors,
which is the comparison #126 set up — *the concept is findable, the discrimination is not* —
somebody else's real labels buy **12.4× the precision of OWLv2 at a common floor of none**
(0.411 vs 0.033), and at conf 0.30 **2.9× the precision with six times the recall** (0.419/0.697
against 0.146/0.110). Either way the discrimination failure of the open detectors is about
*supervision*, not about the concept being intrinsically hard to localize — and at the
deployment threshold the supervised arm dominates them on both axes rather than trading one for
the other.

But it is still third of nine challengers on F1, and RampNet leads it by **0.338** — the wide
end of this benchmark's 0.12–0.34 range. Vistas' curb-cut labels transfer; they do not compete.

**On AP it is the best *zero-training* model on this split — 0.513, five times OWLv2's 0.104 —
but it is not the best non-RampNet model, and the thing that beats it is in this same
document.** The supervised YOLO baseline (#51), trained on the RampNet dataset and scored on
this same split, reports richmond AP **0.748** (`y11x_pano_h200`), **0.724** (`y11l_pano`) and
**0.537** (`y26_pano`; 0.536 before the #140 seam wrap, re-scored under #148) — all above
0.513 — and `y11l_pano` also beats it on F1 (0.595 at conf 0.25 against 0.517). That comparison
belongs here rather than being left out, because it is the sharpest version of what this arm
was built to test: **somebody else's labels for a neighbouring class transfer usefully, and our
own labels for the actual class do substantially better.**
Against the *untrained* field the AP point still stands — the chat VLMs above it on F1 have no
AP at all, emitting boxes without scores, so they are pinned at one operating point and cannot
be tuned, and a tunable model at AP 0.513 is a more useful starting point than an untunable one
at F1 0.664.

#### Resolution parity: the handicap was real, and it was not the problem (2026-08-18)

Everything above is measured at 384×384. This section removes that handicap and changes one
variable. Read fixed in advance and posted to #126 **before** the scored output was read: the
gap closes by **< 0.05 F1** ⇒ *"transfers but does not compete"* stands and this stays a
one-split arm; more than that ⇒ the write-up is revised and 3–4 further splits get costed.

Run on **makelab2 (A40, fp16, transformers 5.15.0 / torch 2.13.0+cu130)**, in a scratch worktree
with a private `--cache-dir`, so nothing here shares a cache directory with the published
detections. RampNet is re-run alongside as the comparability check and **reproduces its committed
richmond row to every digit** (0.964 / 0.768 / 0.855, AP 0.763, 238/9/72).

**Run twice.** The first run, 2026-08-18, is the one the numbers below were written from; its
cache was private and was later lost (see *How it is published* below). The second run,
2026-09-20 (#163), repeated both A40 arms on the same host in the same environment from a fresh,
empty cache — worktree at `fe97940`, panos from `benchmark/richmond/`, log and environment
record in `docs/data/vistas_rerun_163/` — and **every printed figure in this section reproduced
to the digit**: the parity row, the control row, both `conf ≥ 0.30` re-scores, the four
complementarity cells at both operating points, the null read, and both cascade partitions. The
detections published under `benchmark/model_detections/` are the 2026-09-20 run's; the table
below is unchanged by it.

| arm | model input | P | R | F1 | AP | tp/fp/fn |
|---|---|---:|---:|---:|---:|---|
| rampnet (committed, reproduced) | — | 0.964 | 0.768 | **0.855** | 0.763 | 238/9/72 |
| vistas curb-cut — **published** (RTX 3070) | 384×384 | 0.411 | 0.697 | 0.517 | 0.513 | 216/309/94 |
| vistas curb-cut — **same-env control** (A40) | 384×384 | 0.411 | 0.694 | 0.516 | 0.510 | 215/308/95 |
| vistas curb-cut — **parity** (A40) | **1024×1024** | 0.383 | **0.884** | **0.534** | **0.649** | 274/442/36 |

At conf ≥ 0.30: control **0.419 / 0.694 / 0.522** (published: 0.419 / 0.697 / 0.524), parity
**0.384 / 0.884 / 0.536**. Note the threshold barely bites at parity — it removes 3 false
positives against 10 at 384 — because the higher-resolution masks are more confident. The
bottom two rows are committed files since #163: the parity row is
`benchmark/model_detections/mask2former-vistas-curb-cut-1024x1024__richmond.json` and the
control row is
`benchmark/model_detections/replicates/makelab2-a40-2026-09-20/mask2former-vistas-curb-cut__richmond.json`;
`tests/test_scoreboard.py` re-scores the first against the committed bundle in CI.

**The env control was not in the original plan, and it is what makes the parity delta
attributable.** The published run was on Jon's RTX 3070 on an older `transformers`; makelab2 is a
major version on. Since the `transformers` version is *not* in the detection signature — a hazard
this document already flags — parity-vs-published would have differed in **two** things.
Re-running 384 in the parity env separates them: it lands within **one detection out of 523** of
the published run (215/308/95 vs 216/309/94, F1 0.516 vs 0.517). So the 4.x→5.15 jump is benign
for this checkpoint, the residual is fp16 kernel nondeterminism rather than a version break, and
**the whole parity delta is attributable to input size.** With the control published (#163) that
residual can be stated at the detection level rather than the metric level: the control emits 553
points to the published run's 555, and **550 of the 555 pair with a control point within 0.005 of
the pano width** (median offset 0.00002, i.e. sub-pixel; median score difference 0.0006, 95th
percentile 0.006, largest 0.077). Five published points have no counterpart at that tolerance and
three control points are new; only 10 of the 124 panos are byte-identical. That is what "within
one detection" is made of — the masks are the same masks with fp16 jitter on their edges and
scores, not a re-drawn segmentation — and it is the size of environment effect a reader should
expect from any re-run of this arm. That also retires the "an upgrade could
have changed every mask under an unchanged cache key" worry for this arm, as a measurement rather
than an assurance.

**The decision rule returns "stands", and it is not close.** Against the same-env control, parity
moves F1 **0.516 → 0.534, +0.018** — about a third of the 0.05 bar. RampNet's lead goes 0.339 →
**0.321**. One split, and no case for costing more.

**But the mechanism underneath that flat F1 is the actual finding, and it is not the one the
caveat predicted.** The handicap was real and it was large — it was just almost entirely a
*recall* handicap:

- **Recall 0.694 → 0.884 (+0.190).** Misses fall from 95 to **36**, a 62% reduction. At parity
  this arm **out-recalls RampNet** (0.884 vs 0.768) while remaining a model that has never seen a
  curb ramp label of ours.
- **AP 0.510 → 0.649 (+0.139)**, a 27% relative gain — the ranking, not just the operating point,
  is substantially better.
- **Precision 0.411 → 0.383 (−0.028)**: slightly *worse*. False positives rise 308 → 442, faster
  than true positives rise 215 → 274.

So resolution was buying recall the whole time, and F1 stayed flat only because precision is the
binding constraint and resolution does nothing for it. **That sharpens rather than softens the
conclusion #126 was built to test.** The original framing — *the concept is findable, the
discrimination is not* — was stated against OWLv2 and Grounding DINO; it now holds against the
supervised arm at equal input too, and it is no longer confounded with how many pixels the model
was given. Vistas' curb-cut labels **find** curb ramps on deployment panoramas better than our own
model does; what they cannot do is tell a curb ramp from the things that look like one, which is
exactly the failure the RampNet paper predicted when it rejected this class as a supervision
source for being *"overly broad"*.

**What this changes above.** Two claims in this section were stated at 384 and do not survive
parity unqualified:

- The AP comparison against the YOLO baseline said richmond AP **0.748** (`y11x_pano_h200`),
  **0.724** (`y11l_pano`) and **0.537** (`y26_pano`) are *"all above 0.513"*. At parity the arm is
  at **0.649**, so **`y26_pano` no longer clears it** — somebody else's labels for a neighbouring
  class, at equal input, beat one of our own three YOLO arms on AP. The two stronger YOLO arms
  still lead, so the sentence's conclusion holds; its arithmetic does not.
- "Best *zero-training* model on AP" is unchanged and strengthened: 0.649 against OWLv2's 0.104.

**Caveats that travel with these numbers.** `min_area_px=16` is inert at 384 but sits exactly at
the smallest achievable blob at 1024 (see above), so the two rows do not share that setting's
meaning even though they share its value. And this is still **richmond only**.

**How it is published (#163).** The 2026-08-18 detections were written to a private
`--cache-dir` in a scratch worktree on makelab2 and nowhere else; when the time came to publish
them (2026-09-17) the checkout at `/homes/gws/jonf/RampNet` had no worktree registered, its
`.model_cache` held no shard written 2026-08-17..20, and the parity run's shard names
(`compare.cache_key` over the 1024 signature) were absent from the home directory, `/tmp`,
`/var`, the root filesystem and the lab mounts. So the two rows, the 1024 complementarity column,
the operating-point-correction table, both cascade tables and both `analysis_out/cascade_gate*.json`
rested for three days on the committed artifacts alone, and this section said so. The re-run of
2026-09-20 closed that; publishing it needed three decisions about the registry, all of which
the roster's own rules left open. Each is settled in `rampnet/roster.py` and held by
`tests/test_roster.py`:

- **The parity arm is a pinned leg, and the published 384 file keeps its bare name.** The leg is
  `mask2former-vistas-curb-cut` pinned on `vistas_input_size = (1024, 1024)` and published as
  `mask2former-vistas-curb-cut-1024x1024` (`roster.pin_token` spells a size as `1024x1024`;
  the pin-naming rule used to assume a scalar). The registry's rule that *every* leg of a pinned
  model is qualified — written against a bare `claude-sonnet-5` file sitting beside an
  `-effort-high` one, where the bare file hid which effort it was — has one exception now, and it
  is a principled one: `vistas_input_size` is an **opt-in** knob, default `None`, absent from the
  detection signature unless set. The bare 384 file therefore carries no `input_size` key and
  the parity file carries `[1024, 1024]`, so the two describe themselves without a rename, and a
  bare name can only ever mean "every opt-in knob unset". Claude's `effort` is not opt-in
  (`low` is in the signature either way), so the Claude legs stay fully qualified.
  `roster.needs_qualified_name` is the rule; renaming the published file to
  `mask2former-vistas-curb-cut-384` would have touched every reference to it in this document
  and in `tests/` for no information.
- **The same-env control is a replicate, not a leg.** It has the published arm's signature and
  cache key by construction, so no pin could name it. `roster.REPLICATES` registers it
  (`of = mask2former-vistas-curb-cut`, `tag = makelab2-a40-2026-09-20`) and it publishes under
  `benchmark/model_detections/replicates/<tag>/`, a directory the exporter derives from the
  registry (`--replicate <tag>`) rather than taking from the keyboard; inside it the file is
  exactly a published file, and `tests/test_roster.py` asserts that every replicate directory is
  registered, holds exactly the registered (leg, split) files, and that a replicate shares the
  header (`model`, `published_as`, `signature`) of the file it replicates while matching no
  pinned sibling's pins. What it may differ in is the detections, and the size of that
  difference is the result it exists to record. The flag exists because of what the review of
  this write-up found (PR #167): a replicate has the published file's signature by
  construction, so the exporter's overwrite guard — which compared signatures only — let the
  control's cache, exported at the default `--out`, replace the published 384 file's detections
  in place with no error and a passing `--verify`. The guard now also refuses a same-signature
  file whose detections differ unless `--replace` is passed, and `tests/test_export_model_cache.py`
  re-exports both committed #163 files from a cache rebuilt out of them and requires the bytes
  back, then aims the replicate's cache at the published directory and requires the refusal.
- **`export_model_cache.py` takes `--vistas-input-size H W`**, which is what lets it address the
  1024 cache entry and lets the filename come from the roster rather than from a `--publish-as`
  typed from memory.

The re-run's cache shards are not committed (the exports are their published form, and
`--verify` reported both identical to the cache that produced them); the run logs, the
environment record and the four free reads are, under `docs/data/vistas_rerun_163/`.

#### Complementarity: 61% of RampNet's misses are recoverable, and a union still loses

Parity raised the obvious recall-first question — this arm misses 36 ramps where RampNet misses
72, so how much of *RampNet's* miss set does a free, zero-training model already cover? Run
through the #35 gate (`scripts/analysis/complementarity.py`, generalized past its Gemini-only
form for this), scoring-side only:

| | vistas @384 (published) | **vistas @1024 (parity)** |
|---|---:|---:|
| found by BOTH | 194 | 220 |
| rampnet ONLY | 44 | 18 |
| **challenger ONLY** (rampnet-miss ∩ hit) | 22 | **54** |
| found by NEITHER | 50 | **18** |
| of rampnet's 72 misses, recovered | 22 (31%) | **54 (75%)** |
| null on that subset (same boxes, wrong pano) | 0.090 | 0.143 |
| **attributable after the null** | **~15** | **~44** |
| oracle-union recall | 0.839 | **0.942** |
| boxes/pano · above chance (`null_recall.py`) | 4.5 · 0.661 | 6.2 · **0.864** |

**Which 384 run each column is.** The 384 column is the **published** arm, so both sides of it
are committed: `benchmark/model_detections/mask2former-vistas-curb-cut__richmond.json` for the
challenger and `benchmark/richmond/records.jsonl` for RampNet. One wrinkle in reading it back —
`complementarity.py` and `null_recall.py` take their challenger detections from `.model_cache`
rather than from the published export the way `fp_taxonomy.py` and `silent_witness.py` do, so on a
clean clone the export has to be written into a cache directory first (*Reproducing it* below).
`tests/test_complementarity.py` does that in a temporary directory and asserts this column cell
for cell, so `pytest tests/test_complementarity.py` checks it with no setup at all.

An earlier version of this column was the same-env A40 control rather than the published run. The
two differ by one detection in 523, which moved two cells by one ramp each: challenger-only 22 →
21 and found-by-nobody 50 → 51. Nothing turned on it, but at the time the control was not
published either, so the column as printed was not re-derivable and did not say which run it
was. The control is published now (the replicate under *Resolution parity*), and the 2026-09-20
re-run reads 194 / 44 / **21** / **51** off it — the same one-ramp shift, from a fresh inference.

**The 1024 column is committed and checked in CI since #163.** The parity detections are
published, and `tests/test_complementarity.py` rebuilds this column from that file and the
committed bundle, cell for cell, at both operating points — between 2026-09-17 and 2026-09-20
it rested on this table alone, because the private makelab2 cache that held the first run's
detections could not be found. The re-run reproduced every cell (220 / 18 / 54 / 18, and the
miss-subset null of 0.143 that `complementarity.py` reports beside them; `null_recall.py`'s
split-wide read is 6.2 boxes/pano, null 0.145, above-chance 0.864 — two nulls, two scripts, both
reproduced).

**Discounted for chance, a free zero-training model finds ~44 of the 72 ramps RampNet misses —
61%.** The null here is measured on the miss subset rather than extrapolated from the split-wide
one, because RampNet's misses are a biased sample (far-field, adjacent pairs) and that is exactly
where density differs; it lands at 0.143 against the split-wide 0.145, so in this case the
extrapolation would have been fair. And the recall is real detection, not density: at 6.2
boxes/pano the arm's **above-chance is 0.864, higher than RampNet's own 0.754** — nothing like
OWLv2's 0.733 null at 74 boxes/pano.

**The resolution fix mattered far more here than the headline suggested.** Parity moved F1 by
+0.018 and was correctly judged not to change the ranking — but it nearly **tripled** the
attributable complementary gain (~15 → ~44 ramps) and shrank the found-by-nobody core from 50 to
**18**, 5.8% of GT. That core is much smaller than paterson's 88 (22%) or gainesville's 55 (20%),
though those are different splits against a different challenger, so read it as suggestive rather
than a like-for-like. The general lesson is worth keeping: **a flat headline metric hid a large
change in the structure underneath it**, and only the complementarity read surfaced it.

**A naive union of the two loses to RampNet alone, and it is not close.** The oracle-union recall
of 0.942 is a *ceiling* — it assumes a combiner that keeps every right call and discards every
wrong one, which does not exist. What a real union pays is both FP bills:

| | P | R | F1 |
|---|---:|---:|---:|
| rampnet alone | 0.964 | 0.768 | **0.855** |
| naive union with vistas @1024 | 0.393 | 0.942 | 0.555 |
| naive union with vistas @384 | 0.450 | 0.839 | 0.586 |

The false-positive bill decides it: those 54 ramps arrive with 442 false positives, about
**8.2 FPs per recovered ramp**, against the 9 FPs RampNet currently pays for 238 true
positives. So ensembling by union is not a close call at any operating point on this arm's
PR curve.

**What that leaves is a gated cascade, and it is a real open question rather than a plan.** The
useful form is not "take both models' boxes" but "use this arm's candidates as a *spatial prior*
to locally relax RampNet's threshold", which would keep RampNet's precision and buy back some of
the 54. Whether it can work is empirical and decidable: #131 measured RampNet's silent misses as
8% absent / 62% adjacent-tail / 30% faint, so most misses *do* have sub-threshold heatmap signal —
but nobody has checked whether that holds at these 54 locations specifically. If the signal is
absent there, the miss is genuine and the cascade has nothing to work with. `silent_activation.py`
is the instrument. **Not run, not costed here.**

**Caveats.** richmond only, one imagery tier. The 442 FPs are not free even in a recall-first
framing — at 3.6 FP/pano against RampNet's 0.07 they are a ~50× review burden, so "FPs are cheap"
is a claim about the labeling workflow that would need its own justification at this ratio.

#### The operating-point correction: a third of that gain is RampNet's own

**Everything above scores RampNet from the committed bundle detections, which are the *shipped*
operating point — on richmond every one of them is ≥ 0.5519. This document has recommended
**0.30** since #54/#55 (PR #79).** For a complementarity read those are different models, and the
difference decides who gets credit for a recovery.

Checked before relying on it: `analysis_out/op_cache/richmond.json` filtered at ≥ 0.5519
reproduces the published row **exactly** (P 0.9636 / R 0.7677 / F1 0.8546, 238/9/72), so it is the
same source. At 0.30 those same peaks give **P 0.9018 / R 0.8290 / F1 0.8639, 257/28/53** —
matching the committed `analysis_out/op/corrected_at_0.3.csv`. `complementarity.py
--rampnet-op-threshold 0.30` re-bases the gate on it:

| | rampnet @0.55 (published) | **rampnet @0.30 (recommended)** |
|---|---:|---:|
| rampnet recall | 0.768 (238) | **0.829 (257)** |
| rampnet F1 | 0.855 | **0.864** |
| rampnet misses | 72 | **53** |
| challenger recovers | 54 (75%) | **38 (72%)** |
| **attributable after the null** | ~44 | **~30** |
| found by NEITHER | 18 | **15** |
| oracle-union recall | 0.942 | 0.952 |
| naive union F1 | 0.555 | 0.549 |

**Both columns are the parity arm, and both re-derive from a clean clone since #163** — the
challenger side is the published 1024 file (see *Resolution parity*), and RampNet's side is
committed on both columns: `benchmark/richmond/records.jsonl` at 0.55 and
`analysis_out/op_cache/richmond.json` at 0.30. `tests/test_complementarity.py` pins the 0.30
column (236 / 21 / 38 / 15) the same way as the shipped one.

**So 16 of the 54 ramps the challenger got credit for recovering are ramps RampNet already has at
the operating point we recommend — the shipped threshold was discarding them.** That is 16 raw,
**~14 after the chance null** (~44 → ~30). RampNet gains 19 hits going 0.55 → 0.30 (72 misses →
53); 16 of the 19 come out of the challenger-recovered cell and the other 3 out of the
found-by-nobody cell, which is why the headline falls by 16 and not by 19. The deployable
complementary gain is **~30, not ~44**. The recovery *rate* barely moves (75% → 72%): the
challenger is not preferentially finding the easy sub-threshold ones, there are simply fewer
misses to find. And a naive union still loses against the stronger baseline (0.549 vs 0.864).

#### The cascade gate: live, but the ceiling is ~19 ramps, not 54

`scripts/analysis/cascade_gate.py` (new) asks the one question that decides whether a gated
cascade is possible at all: **at the ramps the challenger recovers, does RampNet already produce
something a prior could promote?** It partitions all 310 GT ramps into the four cells and reads
RampNet's heatmap at each, reusing #46 Phase 1's instrument verbatim (`site_profile`,
`null_percentile`, `nearest_peak`, `class_of`) so the numbers are comparable to that phase.
Read pre-registered on #126 before running. Artifacts: `analysis_out/cascade_gate.json` (shipped
point) and `analysis_out/cascade_gate_op030.json` (recommended point).

**Both artifacts are committed, and both were regenerated on 2026-09-20 from the published parity
detections (#163).** The challenger side of the partition is now a committed file (see *Resolution
parity*); the RampNet half — one forward per pano — still needs the native-res panoramas from
`projectsidewalk/rampnet-benchmark` and a GPU, so a clean clone cannot regenerate them without
those, but it can check each file against itself: `tests/test_cascade_gate.py` re-derives every
`cells[]` figure from the same file's `sites` list, so a hand-copied number in the tables below
fails the build, and `tests/test_complementarity.py` re-derives the four cell counts from the
published detections alone.

**What the regeneration changed, and what it did not.** The originals were written at `b7342dc`
and `4c192ca`, before the round-1 review fixes (`d1860e5`, `cc2299e`) and before this branch took
the #132 seam wrap, and the prediction recorded here at the time was that a re-run would add three
payload keys (`rampnet_op_threshold` on the shipped file, `null_rng: "per-site"` and
`panos_without_floor_peaks` on both), one per-site key (`nearest_peak_claimed`), and move every
miss-cell null by up to ~0.075 with everything else unchanged. Measured against the originals
(`makelab2`, A40, 128 s and 119 s wall): the **partition is identical** — 220 / 18 / 54 / 18 and
236 / 21 / 38 / 15, zero of 310 sites changed cell in either file — and so is **every
heatmap-derived per-site column**: `act`, `center`, `argmax_off_px`, `nearest_peak_px`,
`nearest_peak_score`, `peak_in_radius`, `class` and `seam` agree on all 310 sites to the recorded
precision, which also says the RampNet forward on this host is deterministic across a month. The
only columns that moved are the null ones: 59 of 72 miss-cell nulls in `cascade_gate.json` and 46
of 53 in `cascade_gate_op030.json`, by up to 0.09 and 0.08, and the cell medians with them
(`challenger_only` 0.895 → 0.88 at the shipped point and 0.88 → 0.865 at 0.30; `neither`
0.925 → 0.905 at 0.30; `above_null_p95` down by one in two of the four miss cells — `neither` at
the shipped point 7 → 6 and `challenger_only` at 0.30 9 → 8, the other two holding at 12 and 5).
That is the per-site seeding, not the heatmap, and it has one visible benefit: the 53 sites that carry a null
in both files now carry the *same* null in both, where before 43 of them differed. The two tests
that pinned the pre-fix state were rewritten in the same commit
(`test_both_artifacts_record_the_threshold_key_and_the_current_envelope`,
`test_the_committed_nulls_are_seeded_per_site_and_agree_across_the_two_files`). The `newline=""`
pinning is what made that comparison a field-by-field diff rather than a guess.

At **rampnet@0.30**, of the 38 genuinely-complementary ramps:

| what RampNet has there | n | what it means |
|---|---:|---|
| floor peak in radius, **0.05–0.30** | **19** | **promotable** — a peak exists, below threshold. This is the cascade's real target. |
| floor peak in radius, ≥0.30 but unmatched | 4 | the greedy matcher gave that peak to an **adjacent GT**. A matcher/σ problem (#130), not a threshold one. |
| no floor peak in radius | 15 | nothing *of this ramp's* to promote — but not, for most of them, nothing nearby. `act` across these 15 is 0.272 median (0.369 mean); the nearest floor peak is a median **35.0 px** away (R = 22.5 px) and the in-window maximum sits on the window edge (median `argmax_off_px` **22.4**, 11 of 15 within 0.5 px of R). That is a neighbouring mode's shoulder reaching into the window, not mass the extractor overlooked. See the re-cut below. |

The activation figure in the last row is the median over those **15 rows**, not over the 38-ramp
cell. The cell's own median — `cells[].act_median` in `analysis_out/cascade_gate_op030.json` — is
0.2152, and the 19 promotable rows sit lower still at 0.153. Three subsets, three medians, which
is why the row says which one it is.

**What the 15 "no peak in radius" sites are, from the artifact's own columns.** An earlier version
of this table called them "unpeaked heatmap mass `peak_local_max` never called a maximum", and
that reading is withdrawn: the committed `sites[]` rows, joined to `analysis_out/op_cache/richmond.json`,
say the opposite for most of them. Re-cutting the 38 exhaustively, by where each site's nearest
floor peak is and whether the greedy match at 0.30 already gave that peak to another GT on the
pano (`cascade_gate.claimed_by_adjacent`, pinned in `tests/test_cascade_gate.py`):

| the 38 recoverable ramps at rampnet@0.30 | n | mechanism |
|---|---:|---|
| floor peak in radius, 0.05–0.30 | **19** | promotable — the cascade's target, unchanged |
| nearest floor peak ≥0.30 and **claimed by an adjacent GT** — 4 inside R, 7 at 1–2 R (26.8–44.1 px) | **11** | the #130 matcher/σ mechanism; the old "4" row and 7 of the old "15" row are one cause |
| nearest floor peak at 1–2 R, unclaimed (23.5 px @0.643, 24.3 px @0.242, 32.2 px @0.358, 42.5 px @0.143) | 4 | a peak just outside the window; would need a wider radius (and, for the two below 0.30, a lower threshold as well) |
| no floor peak within 2 R (51–117 px) | 4 | genuinely nothing near — and one of the 4 is the seam site below, where the heatmap *has* a peak the op_cache dropped |

So the 15-row that read as "two-fifths of the recoverable set has no peak to raise" is 7 parts
matching problem, 4 parts near-miss geometry and 4 parts absence; with the old 4-row folded in, the
38 are 19 / 11 / 4 / 4. (The 4 "absence" sites also have a claimed nearest peak, just beyond 2 R —
51–117 px away, scores 0.85 / 0.86 / 0.34 / 0.94 — which is why the 11 in the table are 4 + 7 and
not 4 + 11.) `class_of`, the #46 Phase 1
decomposition the pre-registration promised for comparability, puts the 15 at **12 `tail` / 3
`faint_local` / 0 `absent`** (80 / 20 / 0%, against Phase 1's 62 / 30 / 8% over silent misses);
over the whole 38-ramp cell it is 35 / 3 / 0, and the 15-ramp `neither` cell is 15 / 0 / 0. Both
artifacts carry these in `cells[].classes`. The seam site is the one case of a different kind:
`723487737079243` at x = 0.0069 has `act` 0.946 **7.4 px** from the ramp, centre 0.78, and no
op_cache peak within 117 px — the `f4c71c8` seam dropout bounded abstractly further down, made
concrete. A regenerated op_cache would almost certainly list that peak, which makes the site a
RampNet **hit** at 0.30 and moves it out of `challenger_only` (38 → 37) rather than into any
row of this table.

**So the cascade is live and its ceiling is ~19 ramps on richmond — +6.1 recall points (0.829 →
0.890) before any false-positive cost, which is unmeasured.** That is a real number and it is a
long way below the 54 the raw complementarity suggested. Of the other 19, eleven are a matching
problem (#130) that no threshold prior can reach — a σ or matcher change is what would act on
them — four sit just outside the window, and four have nothing near them.

**A negative worth recording: RampNet's own activation does not tell you which misses are
recoverable.** `challenger_only` sits at null percentile **0.865** and the hard-core `neither` at
**0.905** (0.88 and 0.925 on the 2026-08-18 artifacts; the 2026-09-20 regeneration re-drew the
nulls per site and moved both medians by 0.015–0.02, with the ordering and the conclusion intact)
— the ramps *nobody* finds look, if anything, *stronger* on raw heatmap mass than the
ones the challenger recovers (they contain 6 of 15 matcher-claimed peaks ≥0.30, which inflates
it). Median argmax offset is 19.0 px inside a 22.5 px radius, i.e. near the window edge rather
than on the ramp. So there is no cheap self-gating shortcut: you cannot skip the second model and
find these by looking harder at RampNet's confidence. Against the pre-registered rule this is the
**PARTIAL** branch — signal present, but not at the site — and the peak-level column, not the
activation, is what supplies the bounded answer.

One note on those two null percentiles, and on any per-site null read across the two
artifacts. The 2026-08-18 files were written with **one** random stream consumed in pano order
over the miss-cell sites only, so a site's draw depended on which sites came before it; the miss
set differs between the files (19 cell transitions), and of the 53 sites that carried a null in
both, **43 differed, by up to 0.075**, while `act` and `nearest_peak_px` agreed on every one.
`cascade_gate.py` seeds per site now (`site_rng`), and the 2026-09-20 regeneration is what that
produces: the same 53 sites carry the **same** null in both files
(`test_the_committed_nulls_are_seeded_per_site_and_agree_across_the_two_files`), individual
values moved by up to 0.09 against the originals, the medians by 0.015–0.02, and nothing that
depends on the heatmap moved at all. Both files were regenerated together, which is the only
way this comparison is meaningful.

**What would have to be true for the cascade to pay.** Promoting sub-0.30 peaks gated on
challenger candidates also promotes them wherever the challenger fires on a driveway and RampNet
has a faint bump — and 442 of the challenger's 716 boxes are false positives. That cost is **not
measured here**, so "+6.1 recall points" is a ceiling on the benefit with the cost still blank.
The next step, if this is ever picked up, is to build the gate and score it, not to reason further
about it.

**Seam exposure: the two committed artifacts predate the #132 seam fixes, and the effect has now
been measured rather than bounded.** This work branched at `5e20d11`, before `eccadda` (wrap the
360° seam in the matcher) and `f4c71c8` (`peaks_to_dets` dropped peaks beside the seam) landed.
The branch has since merged `main`, so both fixes are in it, and `complementarity.py`'s own
`matched_gt` — which produces the four cells — now calls the shared wrapping matcher instead of
re-deriving the distance inline. That matters here because `score_pano` supplies the false-positive
counts and union P/R/F1 printed in the same tables and wraps by default, so the two halves of one
output were on different matchers.

The earlier version of this paragraph quoted `score_pano`'s docstring as saying wrapping *"moves
no metric on any committed split"*. **That is half the sentence.** In full: *"Wrapping moves no
RampNet or YOLO metric on any committed split — but it does move the challengers."* The challenger
is the side being partitioned here, so the half that was dropped is the one that applies, and the
right way to settle it is to measure rather than to cite.

**Measured, on the cells themselves** (both / rampnet-only / challenger-only / neither), wrapping
against not wrapping:

| arm | RampNet's side | cells | wrapped |
|---|---|---|---|
| vistas 384 (published) | bundle, ≥ 0.5519 | 194 / 44 / 22 / 50 | identical |
| vistas 384 (published) | op_cache ≥ 0.30 | 202 / 55 / 14 / 39 | identical |
| vistas 384 (published) | op_cache ≥ 0.05 | 213 / 66 / 3 / 28 | identical |
| gemini-3.1-pro-preview, paterson (#35 gate) | bundle | 188 / 83 / 36 / 88 | identical |
| gemini-3.1-pro-preview, paterson (#35 gate) | op_cache ≥ 0.30 | 194 / 90 / 30 / 81 | identical |
| gemini-3.1-pro-preview, paterson (#35 gate) | op_cache ≥ 0.05 | 201 / 98 / 23 / 73 | identical |

**Zero cell flips, at every threshold, on both arms.** So the published 384 column and the
committed #35 gate numbers are unchanged by the wrap. **The parity arm has now been re-checked
too** (#163): the 2026-08-18 cascade artifacts were partitioned by the pre-wrap matcher, and the
2026-09-20 regeneration — a re-run whose every printed figure and every one of 310 site assignments
matched, published, wrapping matcher — reproduces both
partitions site for site (220 / 18 / 54 / 18 and 236 / 21 / 38 / 15, zero of 310 sites moved), so
the wrap moves nothing at 1024 either, as a measurement rather than a bound.

One residual, unchanged by the merge: `analysis_out/op_cache/richmond.json` was last written at
`c7098be` (2026-07-28), i.e. **before** `f4c71c8`, so it can still be missing peaks that sit
beside the seam. That would make a site read "no floor peak in radius" when one exists — it can
only *understate* the promotable count, never inflate it.

**Measured exposure on the artifacts: 6 of richmond's 310 GT ramps straddle the seam, and only 1
of them is in `challenger_only`** (the other 5 are in `both`, where neither fix can move the
partition in a direction that matters). So the worst case for the ~19-ramp ceiling is one ramp in
38, and no conclusion here turns on it. That one ramp is now identified rather than bounded:
`723487737079243` at x = 0.0069 (see the re-cut of the 38 above) has a 0.946 heatmap peak 7.4 px
from the ramp and no op_cache peak within 117 px, so on a regenerated op_cache it is a RampNet hit
at 0.30 and leaves `challenger_only` (38 → 37) — the ceiling of 19 does not move. The 2026-09-20
regeneration of both cascade artifacts (#163) did **not** regenerate the op_cache, so that site
still reads the same way in the committed files (`test_the_seam_site_in_the_recovered_cell_has_a_peak_the_op_cache_lacks`);
retiring it is an op_cache regeneration for richmond, a separate run.

##### Reproducing it

The scored runs. These need a GPU and the native-resolution panoramas
(`projectsidewalk/rampnet-benchmark`), and they are what produced the parity table:

```bash
# parity (the measurement)
python scripts/model_comparison/compare.py benchmark/richmond \
    --models rampnet,vistas:curb-cut --vistas-input-size 1024 1024

# same-env control (the attribution) -- default 384, no override. It has the SAME
# signature and cache key as the published 384 run, so on a clone that has written
# the published detections into .model_cache (below) compare.py would find all 124
# panos cached and never run the model: --no-cache (or a fresh --cache-dir) is what
# makes this an actual re-inference rather than a re-score of the published arm.
python scripts/model_comparison/compare.py benchmark/richmond --models vistas:curb-cut --no-cache

# either, re-scored at the deployment threshold (free, reads the cache)
python scripts/model_comparison/compare.py benchmark/richmond \
    --models vistas:curb-cut --vistas-input-size 1024 1024 --op-threshold 0.30
```

The complementarity, null and cascade reads. Every one of these re-scores detections that are
already cached, so they cost nothing; only `cascade_gate.py` needs a GPU, because it runs
RampNet. `--vistas-input-size` is part of the cache key, so it is also what selects which run
is being read: drop it to read the **published 384** arm instead of the parity arm.

```bash
# the four cells at parity, with the chance null measured on the miss subset
python scripts/analysis/complementarity.py vistas:curb-cut richmond \
    --vistas-input-size 1024 1024

# the same, re-based on the operating point this document recommends
python scripts/analysis/complementarity.py vistas:curb-cut richmond \
    --vistas-input-size 1024 1024 --rampnet-op-threshold 0.30

# density vs detection: boxes/pano, the shifted-pano null, above-chance
python scripts/analysis/null_recall.py benchmark/richmond \
    --models rampnet,vistas:curb-cut --vistas-input-size 1024 1024

# the cascade gate, at the shipped point and at the recommended one. Needs a GPU
# and the native-res panos; --panos-root is the checkout that holds them.
python scripts/analysis/cascade_gate.py --panos-root /path/to/RampNet \
    --model vistas:curb-cut --vistas-input-size 1024 1024 \
    --json-out analysis_out/cascade_gate.json
python scripts/analysis/cascade_gate.py --panos-root /path/to/RampNet \
    --model vistas:curb-cut --vistas-input-size 1024 1024 \
    --rampnet-op-threshold 0.30 --json-out analysis_out/cascade_gate_op030.json
```

`complementarity.py` and `null_recall.py` read `--cache-dir` (default `.model_cache`), which is
git-ignored, so on a clean clone the published detections have to be written into one first.
`benchmark/model_detections/<model>__<split>.json` records the signature they were cached under
and `compare.cache_key(model, signature, city, pano_id)` is the shard name;
`tests/test_complementarity.py` does the whole thing in eight lines and is the shortest working
example, for the 384 arm and, since #163, for the parity arm
(`mask2former-vistas-curb-cut-1024x1024__richmond.json`, whose recorded signature carries
`input_size: [1024, 1024]`). So none of the reads above needs a GPU on a clean clone; only
`cascade_gate.py` does, for RampNet's forward.

**Publishing the two A40 arms (#163), exactly as run on 2026-09-20.** Both `compare.py` runs
above were made from a worktree of this branch at `fe97940` on makelab2 with a fresh, empty
`--cache-dir`, then the cache was copied back and exported:

```bash
# the parity leg: --vistas-input-size selects the 1024 cache entry, and the registry
# supplies the filename (mask2former-vistas-curb-cut-1024x1024__richmond.json)
python scripts/analysis/export_model_cache.py --cache-dir <that run's cache> \
    --models vistas:curb-cut --splits richmond --vistas-input-size 1024 1024
python scripts/analysis/export_model_cache.py --verify --cache-dir <that run's cache> \
    --models vistas:curb-cut --splits richmond --vistas-input-size 1024 1024

# the same-env 384 control: a REPLICATE of the published arm (same signature, same cache
# key), so it publishes under its registered tag's directory, which --replicate derives
# from the registry
python scripts/analysis/export_model_cache.py --cache-dir <that run's cache> \
    --models vistas:curb-cut --splits richmond --replicate makelab2-a40-2026-09-20
python scripts/analysis/export_model_cache.py --verify --cache-dir <that run's cache> \
    --models vistas:curb-cut --splits richmond --replicate makelab2-a40-2026-09-20
```

On 2026-09-20 the replicate was exported with the directory spelled out as
`--out benchmark/model_detections/replicates/makelab2-a40-2026-09-20`; `--replicate` arrived with
PR #167, resolves to the same directory, and the committed file is byte-identical under either
(the round-trip test named above). Both `--verify` runs reported the published file scoring identically to the cache. Under
`docs/data/vistas_rerun_163/` are the environment record (`env.txt`) and ten logs: the four
`compare.py` runs (`parity_1024`, `parity_1024_op030`, `control_384`, `control_384_op030`), the
three `complementarity.py` runs (`complementarity_1024`, `complementarity_1024_op030`,
`complementarity_384control`), `null_recall_1024`, and both `cascade_gate` runs. The two export
and two `--verify` invocations were not captured to a file; what they printed is recorded in the
sentence above, and the round-trip test is the stronger check. The two `compare.py` model-load
logs had their `tqdm` weight-loading progress collapsed to its final line (PR #167 n2); nothing
else in any log was edited.

**Cost, in both units.** Money: **$0** — makelab2 is lab-owned hardware with no metered
billing, and every read above is free. Time: one leg of three was timed, and **the full 124-pano
parity run takes 3m38s** on one A40. That was measured rather than guessed because the estimate
going in was 3–4× and it was wrong in the cheap direction: the GPU forward goes 0.078 s →
0.092 s per view, only **1.17×**. Swin's windowed attention scales far better than
pixel count, and the documented "2.3 s/view" is dominated by reprojection and CPU work, not the
encoder. Peak GPU memory is 1.64 GB. The same-env 384 control was **not timed** (the same 124
panos at 1/7 the pixel area, so it is the cheaper of the two), and RampNet's row in that table
costs no GPU at all — `--models rampnet` scores the bundle's committed detections without loading
a model. **So for the 2026-08-18 session there is no total to quote; what is recorded is 3m38s for the
one leg that was measured.** The 2026-09-20 re-run (#163) timed everything, and the rows are in
`analysis_out/usage_log.jsonl` with `paid: false` (makelab2 is not a Slurm host, so it has no
`sacct` record and no row in `compute_log.jsonl`; the usage ledger is where a free GPU leg's
runtime goes): parity **237.2 s** wall (29.5 s model load, 207.3 s inference, 1.67 s/pano over
124 panos) and the 384 control **191.3 s** (12.8 s load, 178.2 s inference, 1.44 s/pano), so the
parity arm costs 1.16× the control per pano, in line with the 1.17× per-view forward measured in
August. The two cascade-gate regenerations were 128 s and 119 s. The A40 was shared with an idle
process holding 8.8 GB of its 46 GB throughout; total GPU time for the session, about 11 minutes,
at **$0**. Verified before the run rather than assumed: the override
reaches the model — `pixel_values` (1, 3, 384, 384) → (1, 3, 1024, 1024) and mask logits
(1, 100, 96, 96) → (1, 100, 256, 256) — which a silently no-opping `processor.size` assignment on
a new major version would not have done, and which would have made "parity" a second 384 run
under a different cache key.

Two mechanisms, both measured rather than assumed:

- **The `curb-cut+curb` union is a clean negative result.** It was run to test whether recall
  hides on the other side of Vistas' ramp/curb boundary. It does not — the union *loses* recall
  (0.697 → 0.648) while precision collapses (0.411 → 0.126, 1,399 FPs). `Curb` fires along every
  kerb line in the scene, and because points come from connected components, it also **fuses
  adjacent ramps into one component**, which is where the recall goes. Same adjacent-pair merge
  mechanism the σ analysis in #46 identified from the other direction.
- **The hood worry did not materialise; the opposite did.** `fp_taxonomy.py` puts only **1.9%**
  of this arm's false positives in the ego-vehicle band, against **15.0%** for OWLv2 — eight
  times *better*, not worse, despite masks being the output most likely to bleed into the hood.
  Whatever #47 costs, it does not cost this arm. Densities: 4.48 boxes/pano, the same class as
  RampNet's 4.2 and an order of magnitude below the open detectors' 55–88; the union arm is 13.31.

**What this does not show.** 79.6% of the arm's false positives are `isolated`, which is an upper
bound on hallucination, not a measurement of it — a driveway, a crosswalk and a flight of stairs
all land there and only imagery separates them. So the paper's specific prediction, that these
labels confuse driveway aprons with curb ramps, is **consistent with** this FP profile but is not
confirmed by it; confirming it needs #46's gallery half pointed at this arm.

**Coverage: richmond only.** One split, one imagery tier (Mapillary 360, OOD). Nothing here says
how it behaves on GSV or in-domain, and the roster's own history is that rankings are robust but
not invariant across splits. Both arms are therefore registered as published-but-off-roster.

#### On a cluster

No new launcher; the arm needs nothing beyond the `transformers` + `torchvision` the other open
models already use, and Mask2Former is in-library (no `trust_remote_code`).

```bash
PYTHON=$ENVPY MODELS=rampnet,vistas:curb-cut BUNDLE=benchmark/richmond \
    sbatch -A <account> scripts/model_comparison/run_open_models.slurm
```

