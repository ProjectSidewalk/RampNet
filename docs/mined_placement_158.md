# Image-based placement of mined targets ([#158](https://github.com/ProjectSidewalk/RampNet/issues/158) phase 2)

This doc covers the RampNet side of #158 phase 2: running the #48 placement arms on the
labeler's mined candidates. The measurement itself, its tables and its caveats are in the
labeler's `docs/mined-precision.md`
([sidewalk-auto-labeler#110](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/110)).
That is where a placed pixel is adjudicated against the benchmark verdicts. Nothing in
this repo reads a verdict for this.

## What runs here

`scripts/analysis/mined_placement_158.py` turns the labeler's `sources.csv` into a
harness-shaped pair list. That is one row per mined candidate:

- the **source view** is the member the pre-registered source rule picked (nearest member
  camera to the target);
- the **other view** is the target pano, centred on the step-2 flat projection.

The script then drives registered #48 arms through the harness's own `Context` and
`run_arm`. The pair list has no answer columns at all, and the harness withholds
detections as it does for #48. The multi-view arms get a corner manifest in `_mv3d`'s
format, with one two-view corner per pair and `_mv3d.build_manifest`'s pose priors.

Outputs are in `analysis_out/mined_placement_158/`:

- `pairs.csv`, `keys.csv` (pair_id → city, site_id, pano_id) and `corners.json`;
- `predictions/<arm>.jsonl` + `.meta.json` for `mapa_posed_pair`, `roma` and
  `roma_local`.

## Result (details and caveats in the labeler doc)

The comparison is paired, candidate by candidate, against the step-2 flat projection on 127
candidates:

- **Richmond (Mapillary).**
  - Under the pre-registered 5 m world match, `roma_local` moves 9 false positives onto a
    detected ramp within 5 m and breaks none (sign test p = 0.004; the test itself was
    chosen after the predictions existed, and it survives Bonferroni over 3 arms). All 9
    become `already_detected`, so there are no new hard positives.
  - all-mined 0.431 → 0.627 [0.49, 0.75]; hard-only 0.310 → 0.406.
  - `roma` gives 8 fixed / 0 broken, and `mapa_posed_pair` 4 / 2.
- **The 9 are not shown to be the mined ramp** (post hoc, added after review). The detection
  each one lands on belongs to *another* fused site in 6 of the 9: a multi-pano site 5.4–11 m
  away. At least 2 are demonstrably a different ramp. Site 1287's own source pano detects
  both ramps separately, about 45° apart, and RoMa carried the click from one onto the
  target's detection of the other. Site 350 lands on the same detection. The other 3 land
  on singleton sites whose identity is undetermined. An earlier version of this doc (and the
  #158 comment) said `roma_local` "fixes" richmond's placement; **that is retracted**. What
  holds: the all-mined numbers, under the rubric, where a label on any real ramp within 5 m
  is correct. Read as "placed on the mined ramp", the paired count lies between 0 / 0
  (no discordant pair, so the sign test is undefined) and 7 / 0 (p = 0.016). It is 3 / 0
  (p = 0.25) only if the 3 singleton landings, 8.1 m from the site and of undetermined
  identity, are *taken* to be the mined ramp; that is a reading, not a lower bound. The
  labeler's `scripts/mined_placement_attribution.py` produces the per-candidate table.
- **Pooled.** all-mined 0.605 → 0.702 [0.61, 0.78] and hard-only 0.483 → 0.534 with
  `roma_local`. But pooled `tp` falls 42 → 39. On GSV the same pull turns 7 true hard
  positives into `already_detected`. In 5 of the 7 the landed detection belongs to a site
  that the source pano also detects separately, 0.9–3.8 m away on the same corner.
- **GSV.** No arm helps.
- **Where the arms stand in #48.** [#220](https://github.com/ProjectSidewalk/RampNet/pull/220)
  (open; [results](https://github.com/ProjectSidewalk/RampNet/issues/48#issuecomment-5895553631))
  re-tested the MapAnything arms against `proj_height_auto` on fresh pairs. It confirmed
  `mapa_posed_pair`, the arm run here, which stays robust under #220's sensitivity reads
  (GSV α/3 lower bound ≥ +0.105°; Mapillary +1.32° [0.73, 2.34]), and `mapa_posed_corner`.
  `mapa_k_pair` is confirmed as pre-specified but borderline on GSV: under #220's review and
  [sensitivity reads](https://github.com/ProjectSidewalk/RampNet/issues/48#issuecomment-5896570175)
  its GSV lower bound is 0.000 / −0.009 / −0.048, so that cell is not confirmed under any of
  them. `mapa_k_pair` was best on Mapillary (+1.89°); it was not run in phase 2 (step 3 runs it; see below). `roma` and
  `roma_local` were **not** in that re-test. In #48 no matching arm beats auto on GSV, and
  their Mapillary gains come from a stratum #48 did not screen for multiplicity. `roma_local` is post hoc there. `roma`
  inherits `lg`'s post hoc 5° ground band, so it is not fully pre-specified either.
- **For step 3.** On a dense corner, the 5 m nearest-point world match cannot tell the mined
  ramp from a neighbour. Any later `already_detected` gain should carry the own / other
  attribution beside it.

## Step 3 ([#222](https://github.com/ProjectSidewalk/RampNet/pull/222))

Step 3 of #158 (peak-anchored targets) is in the labeler:
[sidewalk-auto-labeler#112](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/112),
`docs/mined-precision.md` § Step 3. On the RampNet side it adds two things:

- **`mapa_k_pair` on the same 127 mined pairs.** It was named as an arm before scoring. It
  fell back on 1 pair and took 50 s on the makelab2 A40.
  - Under the rubric, richmond reads 11 fixed / 0 broken. All 11 become `already_detected`,
    so, as with `roma_local`, they are not shown to land on the mined ramp.
  - Its #48 status is confirmed as pre-specified but borderline on GSV (above).
- **`paid: false` ledger rows** for that run and for the labeler's richmond and bend floor
  passes. The bend pass failed its instrument gate and was not used. The likely cause, found
  after the review of #222 and #112, is the pixel source: bend's production run fed the model
  Google's zoom-3 rendition, while the floor pass read the archive's max-zoom JPEGs (labeler
  doc, step 3 "Post hoc").
- **`mapa_k_pair.meta.json` provenance.** As written by the run, it says
  `"pre_specified_for_158": false`, because `PLANNED_ARMS` then listed only the phase-2 arms.
  The step-3 plan named the arm before the run and before scoring, so the file now also
  carries `"pre_specified_in": "step 3"` and a dated `provenance_correction`. `PLANNED_ARMS`
  records the plan per arm, and new metas write `pre_specified_in`. No prediction changed.

## Step 4 (#158): the mined-label gallery

`scripts/analysis/mined_label_check_158.py` builds the gallery Jon will rate. It reads the
labeler's city-wide miner output (`sidewalk-auto-labeler` `docs/figures/mined-precision/data/step4/`),
uses the rule fixed there before any crop was cut, and reuses this repo's #48 GT-check
machinery (rater ids, manifest digest, integrity checks, `agreement`, the page) and its
cutter.

- **Sample.** 100 of the 557 labels emitted off the benchmark panos, allocated by band
  (0–8 m 18, 8–12 m 37, 12–15 m 45) with seed 158.
- **Instrument items.** 10 cards (8 with a known Yes, 2 with a known No) are flagged only in
  `items.json`.
- **Cards.** 110 cards with 220 crops.
- **Files:**
  - page: `benchmark/mined_label_check_158/gallery.html` (open it locally);
  - rater file: `analysis_out/mined_label_check_158/mined_label_check__jonf.json`,
    verdicts empty;
  - `manifest.json` holds each crop's sha256 (digest `bf3c00686e50e0da`).
- **Scoring:** `python scripts/analysis/mined_label_check_158.py rates <file> [<file2>]`.
  It reports precision (Yes / (Yes + No), Wilson) pooled and per band, read against #158's
  rule, plus agreement on the instrument items and between two raters.

### Pass 1 (Jon, 2026-09-29): 110 of 110 answered, 57 notes

`rates mined_label_check__jonf.json`, sample cards only (the 10 instrument items are
scored apart):

| band | Yes | No | Can't tell | precision [95% Wilson] |
|---|--:|--:|--:|---:|
| all 100 | 59 | 11 | 30 | 0.843 [0.740, 0.910] |
| 0–8 m | 11 | 2 | 5 | 0.846 [0.578, 0.957] |
| 8–12 m | 23 | 2 | 12 | 0.920 [0.750, 0.978] |
| 12–15 m | 25 | 7 | 13 | 0.781 [0.612, 0.890] |

- **Against the rule:** the point estimate is *build* (≥ 0.80) but the interval reaches
  into *visibility*, so the reading is **not decisive**. Can't tell is 30 of 100; counting
  every one as No gives 0.59, as Yes 0.89, so the answer lies in that range.
- **Instrument items:** 8 of 10 decided, 7 agree with the earlier verdict. `c96405a60` was
  No earlier and Yes now; `cacfb48e9` (earlier No) and `ca7ee8f77` (earlier Yes) are Can't
  tell. The "earlier" answers are step-3 benchmark verdicts joined through the 5 m world
  match, not Jon's direct earlier reads of these crops, so a disagreement can be the match.
- **What the notes say** (57 notes, read 2026-09-30). 13 of the 30 Can't-tell notes name
  washed-out lighting, and four of those say brightness / contrast / saturation controls
  would decide it. 6 of the 11 Nos and 6 of the 59 Yeses say the ramp is there but the ring
  is 1–5 ring-diameters off it: the miner's peak sits beside the ramp. Two Can't-tell cards
  have the ring between two ramps; two Nos sit on a utility panel or cover. Under the
  rubric ("at the ring or touching it") an off-ramp point can still read Yes, so as a
  *training-label* precision 0.84 is an upper reading; with the six offset Yeses counted
  against, it is 53/70 = 0.76.

### Pass 2 (planned 2026-09-30, rebuilt 2026-10-01): the Can't tell cards, re-cut native and unringed, with image controls

Pass 1 was rated without image controls, at 360 px, with the ring drawn into the JPEG.
Pass 2 shows the 32 pass-1 Can't tell cards (30 sample + 2 instrument) again. It is the same
rater's second look, not a second rater; rubric and question are unchanged, Can't tell stays
valid, and the context view is pass 1's crop.

**What changed, and why (2026-10-01).** The first pass-2 page (2026-09-30) kept the pass-1
crops and added brightness / contrast / saturation sliders. Jon's first look at it: the
sliders did not rescue the washed-out cards, and the ring itself hides the spot it marks.
Measured on the 32 pass-1 Can't tell crops (`scripts/analysis/mined_label_check_158.py`'s
inputs; the numbers below are from a one-off read of the crop pixels, not a committed script):

- **Clipping is in the source.** In the 9 cards whose note says washed out, 30–76% of the
  pixels in the ring area are ≥ 245 in all three channels. The camera wrote them as white;
  no filter brings them back.
- **Resolution was thrown away by us.** The 36° window is 1,100 px wide on an 11,000 px pano
  and was resized to 360 px (JPEG q82) before anyone looked. 23 of the 31 panos behind these
  cards are 11,000 px wide (one is 12,288, one 5,760); 6 are 4,096 px, where native is 410
  px and gains nothing. About half the Can't tell notes cite angle, distance or graininess
  rather than lighting.
- **The ring cannot be hidden** when it is baked into the pixels, and the context crop is
  another pano, so no unringed copy of the rated view existed.

So pass 2 was rebuilt:

- **New cuts.** `plan-pass2` writes `analysis_out/mined_label_check_158/items_pass2.json`
  (the 32 target views, `ring: false`, `cut: {native, quality 90}`, bound to the pass-1
  file by sha256 and to the pass-1 gallery by digest). `cut-crops --native` (no resize)
  cut them on makelab2 from the native-res archive in 11 s of CPU; a second cut reproduced
  all 32 files byte for byte. They are committed in
  `benchmark/mined_label_check_158/crops_pass2/` (3.6 MB; 24 at 1100×733, 6 at 410×273,
  one 576×384, one 1229×819) with per-file sha256 in `manifest_pass2.json`, digest
  **`9a8045c952207339`**, over pass-1 digest `bf3c00686e50e0da`. The pass-1 crops and
  manifest are unchanged (same `cut_one`, same defaults).
- **The ring is an overlay.** The page draws it as SVG in crop units (same centre, radius
  and dark halo as the baked ring); **R** or the button hides it. The view opens at 1:1 pixels (never below the 540 px fit width, so the six 410 px cuts are not shrunk); **Z** fits it to the card. The context view stays beside it: the page drops its 1,500 px width cap so both fit side by side on a wide screen. Added 2026-10-02 at Jon's request, before any pass-2 answer: on a 27" monitor the fit view threw away the native resolution pass 2 exists for.
- **Image controls, per card, saved with the answer** (`image`, slider units): levels (black
  and white point), gamma and local contrast as an SVG filter, then brightness, contrast and
  saturation as CSS filters. All are browser filters on the committed JPEG, so the page works
  from `file://` and no pixel is rewritten; the setting is in the export so the re-rating is
  reproducible. A clipped pixel stays white under every setting; what the controls recover is
  the unclipped low-contrast range around it.
- **Washout fix** (added 2026-10-02, before any pass-2 answer): one slider that at t% sets black point 1.7t, local contrast 1.5t and saturation 100+0.4t on the sliders above (`WASHOUT` in the script). It is a shortcut only: the export records the resulting slider values, not t, so an answer reads the same however it was reached.
- **Binding.** The pass-2 rater file carries `manifest_digest` (pass 2),
  `pass1_manifest_digest`, `from_file` / `from_sha256` (the pass-1 file) and exactly its
  Can't tell cards as items; `rates --pass2` refuses anything else. The earlier, unanswered
  pass-2 file (0 of 32) was replaced; nothing from it was rated.

Commands (desktop unless marked):

```bash
python scripts/analysis/mined_label_check_158.py plan-pass2 \
    --pass2-from analysis_out/mined_label_check_158/mined_label_check__jonf.json
python scripts/analysis/multiview_evidence_48.py cut-crops \
    analysis_out/mined_label_check_158/items_pass2.json \
    --archive-root /projects/makeabilitylab/sidewalk-auto-labeler/runs --out crops_p2   # makelab2
python scripts/analysis/mined_label_check_158.py gallery \
    --pass2-from analysis_out/mined_label_check_158/mined_label_check__jonf.json \
    --crops crops_p2 --init-rater jonf-p2
python scripts/analysis/mined_label_check_158.py rates \
    analysis_out/mined_label_check_158/mined_label_check__jonf.json \
    --pass2 analysis_out/mined_label_check_158/mined_label_check__jonf-p2.json
```

`rates --pass2` reports pass 1, the pass-2 subset, and **combined**: pass 1 with the pass-2
answers written over its Can't tells (an unanswered pass-2 card keeps Can't tell). The
combined read is the step-4 number; pass 1 stays reported beside it. The pass-1 page was
rebuilt from the same code path (it gains the new controls; its ring stays baked, so no
toggle); its digest and Jon's pass-1 file are unchanged.

**Status (2026-10-01):** pass 2 is open and unrated (0 of 32).

## Cost

| step | where | wall-clock | GPU-h |
|---|---|---|---|
| build | desktop CPU | 8 s | 0 |
| cut-views (254 views, 211 panos) | makelab2 CPU | 40 s | 0 |
| step 4, pass 2: `cut-crops --native` (32 views, 31 panos) | makelab2 CPU | 11 s | 0 |
| `mapa_posed_pair` | makelab2 A40 (shared, free memory checked first) | 134 s | 0.037 |
| `roma` + `roma_local` | desktop RTX 3070 | 204 s | 0.057 |
| step 3: `mapa_k_pair` (ended 18:13:07Z) | makelab2 A40 (shared, free memory checked first) | 50 s | 0.014 |
| step 3: labeler floor passes, richmond + bend | makelab2 A40 | 214 s + 245 s | 0.059 + 0.068 |

Every GPU run has a `paid: false` row in `analysis_out/usage_log.jsonl`; the RoMa row
(`mined-placement-158:roma+roma_local`) covers both RoMa arms. The two floor-pass rows'
`ts` are the end of each pass rounded to the minute, so they are approximate. Nothing ran
on Tillicum and no paid API was called.

## Reproduce

The commands are in the labeler doc, "Reproduce (phase 2)". Environments are as in #48:

- MapAnything used makelab2's `crossview48_sfm/venv`, with `HF_HOME` / `TORCH_HOME` from
  `crossview48_sfm`. That holds for step 3's `mapa_k_pair` too: its meta records torch
  2.6.0+cu124, which is that venv's, and its log reads the DINOv2 weights from
  `crossview48_sfm/torch_home`. The command is in the labeler doc, "Reproduce (step 3)".
- RoMa ran on the desktop RTX 3070 against RampNet's `.venv` (torch 2.6.0+cu126, numpy
  2.5.1, opencv 5.0.0, python 3.12.10), plus these packages, installed `--no-deps` into a
  scratch directory put first on `PYTHONPATH`:
  - `romatch==0.1.2`
  - `kornia==0.8.3`, with `kornia_rs` 0.2.0
  - `loguru==0.7.3`
  - `win32_setctime` 1.2.0 (Windows only)

  As one command (Windows; drop `win32_setctime` elsewhere):

  ```bash
  pip install --no-deps --target <dir> romatch==0.1.2 kornia==0.8.3 kornia_rs==0.2.0 loguru==0.7.3 win32_setctime==1.2.0
  ```

  `docs/crossview_align_48/matching.md` §7 lists the same packages but not `kornia_rs`. The scratch
  directory was a Claude Code session scratchpad (`%TEMP%\claude\...\scratchpad\pkgs`). It is
  not durable, so reinstall from those lines.
- **Match cache.** `roma` ran first with `--extra match_cache=CACHE`. It filled `CACHE/roma/`
  (7.7 MB, 17:07–17:10Z, including first-use weight downloads) in 192.8 s.
  `roma_local` then read the same cache and took 4.2 s. So:
  - `roma_local` alone on an empty cache takes about as long as `roma`;
  - GPU nondeterminism can change a pair or two of the match set (#48 `matching.md` §2).

  The cache was in the same session scratchpad and is not kept.
- **Recorded versions.** The committed `roma*.meta.json` record cv2, kornia, numpy, python
  and torch. They name romatch 0.1.2 only inside `config`. Since the review fix,
  `predict` also records the installed `romatch` / `loguru` / `lightglue` versions and the
  `--extra` arguments (the match-cache path) in each new meta. The committed files are
  unchanged, because the labeler holds byte-identical copies.

The views are not published. They are on makelab2 at `/homes/gws/jonf/mined158/views`
(`views.tar` sha256 `8a8bd69029736ffa1207de69484826b453109046485de028fadf2cabd7cccad6`) and
regenerate from the archive with `cut-views`.
