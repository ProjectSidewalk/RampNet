# Bayonne (#159): the first Panoramax split, staged ahead of its ground-truth review

**Status: staged, NOT reviewed.** `benchmark/bayonne/` has no `verdicts.json`, and nothing in
this document is scored against ground truth. Every number below is ground-truth-free. The
review is the only step left; this document says how to start it, what is already cached for
the moment it ends, and what remains after it.

Issue [#159](https://github.com/ProjectSidewalk/RampNet/issues/159); plan in
[this comment](https://github.com/ProjectSidewalk/RampNet/issues/159#issuecomment-5956603646).
The bundle comes from sidewalk-auto-labeler
[PR 125](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/125)
(`docs/panoramax-bayonne.md` there covers the run itself).

## 1. What the split is for (Phase 0)

- **First Panoramax split**, and the first place the model meets a single-account municipal
  GoPro Max capture: 123 of 125 panos are from `sig_bayonne` on `panoramax.ign.fr`, 2 from
  another producer on `panoramax.openstreetmap.fr`. It is the Laurens rig class without
  Laurens' mixed contributors and seasons.
- **First non-US deployment target**: Bayonne is the first Panoramax Project Sidewalk city, so
  this is its ground-truth gate.
- **Imagery tier:** `action-modern` (GoPro Max). The plan proposed a Panoramax branch in
  `tier_of()`. It is not needed and would be wrong: tiers are by rig, and the records carry
  `GoPro` / `Max`, which the existing branch already classifies
  (`tests/test_bayonne_159.py::test_bayonne_lands_in_the_modern_action_cam_tier_without_a_new_branch`).
- **Pooling (proposed, Jon's call):** held out of the pooled recommendation as a non-US split,
  like `sao_paulo`, whose `HELD_OUT` reason ("the pooled recommendation is a US-deployment
  basis") applies unchanged.
- **Naming:** bare `bayonne`, one imagery source (the `_<source>` suffix rule).

**Imagery credit.** Municipal imagery © sig_bayonne via Panoramax (panoramax.ign.fr), Licence
Ouverte / Etalab 2.0. The other producer is credited per pano by the `copyright` and `license`
fields of `records.jsonl` (`Arretche`, CC-BY-SA-4.0); the gallery shows both next to each pano.

## 2. The staged bundle

| file | committed | what |
|---|---|---|
| `records.jsonl` | yes | 125 records: 5 `top` / 95 `random` / 25 `empty`, 147 detections ≥ 0.55 |
| `sample.json` | yes | sampler settings (seed 0, 30 m spacing, sample 100, empty 25) |
| `imagery_manifest.json` | yes | sha256 + bytes + size per pano, written at staging time, digest `eb844c2e67625ea0` |
| `bundle_provenance.json` | yes | labeler commits, run command, `results.jsonl` sha256 `4f38ff52…b4b3` (28,524 records) |
| `nadir_band.json` | yes | the logo band's top edge, measured per pano |
| `index.csv`, `panos/` | no (git-ignored) | the exporter's manifest and the 125 native panos (67 at 5760×2880, 58 at 5376×2688) |

The panos are in `D:/Git/labeler-wt/bayonne-bundle/panos/` and in the makelab2 archive
(`/projects/makeabilitylab/sidewalk-auto-labeler/runs/bayonne/panos/`, all 28,524). Both copies
were checked byte for byte against `imagery_manifest.json` on 2026-10-02.

**The nadir logo band.** 123 municipal panos carry a white band across the bottom of the frame.
Its top edge is y 0.791 on 114 of them and 0.771-0.789 on 9; an overlay on those 9 shows the band
itself starting higher on two-wheeler captures, not a measurement error. The two non-municipal
panos have no white band; one (`f8759625…`) has a green-and-white one this measurement does not
detect, and it is plain to see in the gallery.

## 3. Start the review (the only step left)

```bash
# 1. imagery into the checkout (git-ignored), then check it
cp -r D:/Git/labeler-wt/bayonne-bundle/panos benchmark/bayonne/
python scripts/analysis/bayonne_159.py verify            # expect: verify: OK

# 2. build and open the gallery
python scripts/gt_gallery.py benchmark/bayonne
# open benchmark/bayonne/gallery/index.html, review, "Export verdicts",
# save the download over benchmark/bayonne/verdicts.json
```

The export downloads as `bayonne_verdicts.json`; it becomes `benchmark/bayonne/verdicts.json`.

**What was checked on the instrument before handing it over** (2026-10-02, built from this
branch and opened in headless Edge):

- **Resolution.** Every full pano is rendered at 4096×2048 and every crop at 512×512 cut from
  that image: model resolution, below native (5760 or 5376 wide), as for every split.
- **Band.** The gallery reads `nadir_band.json`, hatches the band from its measured top edge,
  labels the line, explains it in a banner ("123 of 125 panos"), and asks before accepting a
  missed-ramp mark inside it. Screenshot check: the dashed line sits on the band's top edge.
- **Heading and seam.** The gallery does not rotate panos and draws detections at the records'
  normalized x, y. Records and pixels share one frame: on the gallery page checked and in the
  147 pre-read crops (section 6, cut at the same coordinates) the marker sits on a kerb or
  crossing in all but a handful of cases, and those handful are detector errors (a roof, a
  manhole), not an offset. The equirect's left and right edges are the 360 seam, as in every
  split. Production drops detections within about 3.5° of the seam (section 4), so a ramp at
  the left or right edge of a pano shows up as a missed ramp, which is how it should be scored.
- **Link and credit.** The pano id links to the picture on its own Panoramax instance (it used
  to link every non-Mapillary pano to Google Maps); producer and licence are shown beside it.

**Rubric notes for this city** (from the labeler hand-off and the AI pre-read in section 6):
French lowered kerbs at crossings are often a plain lowering with a white-painted kerb face and
no flare, sometimes with white tactile paving; they count. Driveway and garage lowerings, common
here, do not. Write anything that fought the rubric in the gallery's *Review notes*.

## 4. Ground-truth-free reads

Everything here re-derives from committed files with `scripts/analysis/bayonne_159.py`
(`checks`, `firing`, `frame`), and `tests/test_bayonne_159.py` pins each output byte for byte.

### 4.1 The pipeline reproduces before it reads Bayonne

`laurens_mapillary` (the other GoPro Max split, 94 panos) was re-extracted with this branch on
makelab2's A40. It lands on the committed `analysis_out/op_cache/laurens_mapillary.json` cell
for cell: **374 of 374 peaks, none extra, none missing**, scores within 5.4e-5
(`analysis_out/bayonne_159/checks.json`, `replication_control`).

### 4.2 Bayonne's cache reproduces its records, except at the seam

`analysis_out/bayonne_159/op_cache/bayonne.json`: 125 panos, peaks down to 0.05, extracted with
`operating_point_curve.py extract --unreviewed` (placeholder GT, `meta.gt = "unreviewed"`).
Against the committed records, all **147 of 147** detections ≥ 0.55 land in the identical heatmap
cell. The cache has **3 more** peaks ≥ 0.55, all at the seam (x = 0 or 0.996):
`135b5214` (0.551), `2873e2ea` (0.562) and `5eb6919e` (0.590, an `empty`-stratum pano).

The cause is in the labeler, not here: `detectors/curb_ramp.py` calls `peak_local_max` without
`exclude_border=False`, so the production path drops every peak within 10 heatmap cells of the
array edge, the defect RampNet's extractor had until f4c71c8 (#132). That affects every labeler
city's output, not only Bayonne's (see "Decisions for Jon" in the PR). For this split it means
the review judges exactly what production reported, and a ramp the model saw at the seam is
scored as a miss.

### 4.3 Is Bayonne quiet at every threshold?

Peaks per pano by sampler stratum, against every split's committed op_cache
(`analysis_out/bayonne_159/firing.json`). The committed op_caches predate f4c71c8 and so lack
border peaks; the `interior` columns drop the 10-cell border ring from **every** split so all are
measured alike, and they are the ones quoted. The `random` stratum is conditioned on ≥ 1
detection at 0.55 and `empty` on none, so neither is a city rate.

| split | random: peaks/pano @0.10 | @0.30 | @0.55 | 0.10/0.55 | empty: peaks/pano @0.30 (panos firing) |
|---|---:|---:|---:|---:|---:|
| **bayonne** | 3.389 | 1.790 | 1.232 | **2.75** | 0.120 (3 of 25) |
| laurens_mapillary | 3.250 | 1.875 | 1.297 | 2.51 | 0.160 (3 of 25) |
| budapest_district5 | 3.874 | 2.484 | 1.632 | 2.37 | 0.600 (10 of 25) |
| sao_paulo | 3.947 | 2.895 | 1.979 | 1.99 | 0.200 (4 of 25) |
| gainesville | 3.137 | 2.179 | 1.642 | 1.91 | 0.080 (1 of 25) |
| morgantown | 3.337 | 2.232 | 1.768 | 1.89 | 0.040 (1 of 25) |
| clovis | 2.495 | 1.747 | 1.358 | 1.84 | 0.080 (2 of 25) |
| richmond | 3.979 | 2.936 | 2.383 | 1.67 | 0.000 |
| annapolis | 2.947 | 2.368 | 1.937 | 1.52 | 0.080 (2 of 25) |
| bend | 3.337 | 2.737 | 2.253 | 1.48 | 0.000 (of 10) |
| paterson | 3.221 | 2.789 | 2.516 | 1.28 | 0.000 |

Reading: **no.** At 0.55 Bayonne's detection panos carry the fewest detections of any split
(1.23, with laurens_mapillary 1.30 and clovis 1.36). At 0.10 it is mid-range (3.39). Its
0.10/0.55 ratio, 2.75, is the steepest of the eleven: per 0.55 detection it carries more
sub-threshold peaks than any other split, the shape laurens_mapillary (2.51) and budapest (2.37)
also have. Whether that mass is real ramps or noise is exactly what the review, and then the #55
incremental-FP pass at op-threshold 0.25, will say; nothing here can.

### 4.4 Where the peaks sit in the frame

GoPro Max panos only, interior peaks (`analysis_out/bayonne_159/frame.json`):

| split | panos | peaks ≥ 0.30 | y p50 | y p90 | share y ≥ 0.70 | share in rig mask (y ≥ 0.772) | share in band (y ≥ 0.791) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **bayonne** | 125 | 208 | 0.570 | 0.648 | 0.024 | 0 | 0 |
| laurens_mapillary | 94 | 153 | 0.570 | 0.615 | 0.013 | 0 | 0 |
| morgantown | 125 | 254 | 0.570 | 0.633 | 0.028 | 0.008 | 0.008 |
| budapest_district5 | 124 | 291 | 0.570 | 0.633 | 0.024 | 0.003 | 0 |
| richmond (GoPro Max panos) | 33 | 65 | 0.539 | 0.602 | 0 | 0 | 0 |

Bayonne's peaks sit a little lower in the frame than the other GoPro Max splits (p90 0.648 vs
0.602-0.633), i.e. a little nearer the camera, and none in the band or the rig mask at 0.30
(none at 0.55 either). On the other splits the band column is a counterfactual (they have no
band): the band would have covered under 1% of their peaks. So the band costs little detection
mass directly; what it hides is ramps within about 2 m of the camera, which the review cannot
see either and so cannot count.

## 5. Free challengers, run ahead of the verdicts

All on makelab2's A40 (shared with two other jobs), 2026-10-02, with
`compare.py benchmark/bayonne --unreviewed`: detections are cached and exported, **nothing is
scored**. Each leg wrote a `paid: false` row to `analysis_out/usage_log.jsonl`. The exported
files are in `analysis_out/bayonne_159/model_detections/`, in the
`benchmark/model_detections/` format, and stay out of that directory until the split is
registered. `export_model_cache.py --verify` was not run: it re-scores against verdicts, and
there are none yet.

| leg | env | settings | panos | wall time | s/pano | points/pano at the list's threshold |
|---|---|---|---:|---:|---:|---:|
| y11l_pano | RampNet `.venv-eval` | #71 protocol: `--tiling none --yolo-imgsz 1280` | 125 | 49 s | 0.33 | 0.424 (conf ≥ 0.25) |
| y26_pano | same | same | 125 | 40 s | 0.31 | 0.536 |
| y11x_pano_h200 | same | same | 125 | 40 s | 0.30 | 0.456 |
| OWLv2 (large, ensemble) | same | perspective tiling, score floor 0.05 | 125 | 1,495 s | 11.31 | 5.536 (≥ 0.25) |
| Grounding DINO (base) | same | same | 125 | 656 s | 5.10 | 27.4 (≥ 0.15) |
| Qwen3-VL-8B-Instruct | same | perspective tiling | 125 | 2,943 s | 22.10 | 2.84 (no scores) |
| Molmo2-8B | `~/envs/molmo` (transformers 4.57.1) | perspective tiling | **0** | 284 s | — | **failed** |

0 pano failures on every leg except Molmo. **Molmo2-8B did not run.** It ran in its own env (so
not the documented trap of "completing" with Molmo skipped), but `device_map="auto"` offloaded
part of the model to the CPU on the shared GPU ("Some parameters are on the meta device"), and
every pano failed with `Cannot copy out of meta tensor`; the leg stopped after 10 consecutive
failures. Its ledger row records the 284 s it spent. Not retried: making it fit needs a code or
env change (an explicit dtype or a GPU to itself), outside "use the environments as they are".
**Qwen3-VL-32B** was not run: its weights are not cached on makelab2.

### Candidate misses, for the reviewer's attention only

`analysis_out/bayonne_159/candidates.json` (`bayonne_159.py candidates`): locations where
RampNet has **no peak ≥ 0.30** within the 0.022 match radius and **≥ 2 challenger legs** put a
point within it, each leg at the threshold in the table above. OWLv2 and Grounding DINO are
listed as support and never count as a vote: they are the roster's dense legs, and counted as
votes, 417 of 425 draft candidates were those two agreeing with each other.

**11 candidates, all in the `random` stratum, none in the 25 `empty` panos, none in the band.**
Six have a RampNet peak in the radius below 0.30 (0.08-0.28). Seven are YOLO arms agreeing
with each other (three arms of one supervised recipe, so correlated), and five include Qwen.

| pano | x | y | legs (votes) | support | RampNet best peak in radius |
|---|---:|---:|---|---|---:|
| `2157f981` | 0.7344 | 0.5558 | y11x_pano_h200, y26_pano | — | 0.283 |
| `59e47e8b` (top) | 0.3532 | 0.5626 | y11l_pano, y11x_pano_h200 | gdino | 0.280 |
| `733095d5` | 0.6226 | 0.6299 | qwen3-vl-8b, y11l, y11x_h200, y26 | gdino | 0.227 |
| `88490116` | 0.7205 | 0.6605 | y11l_pano, y26_pano | gdino | — |
| `8ce50263` | 0.1570 | 0.5844 | y11x_pano_h200, y26_pano | — | 0.076 |
| `ade1ccfb` | 0.8991 | 0.5182 | qwen3-vl-8b, y26_pano | gdino, owlv2 | — |
| `b0f0b5d9` | 0.8954 | 0.6169 | qwen3-vl-8b, y26_pano | gdino, owlv2 | — |
| `b1b7ddd2` | 0.3272 | 0.6074 | y11l_pano, y11x_pano_h200 | — | — |
| `cc057703` | 0.1185 | 0.6624 | qwen3-vl-8b, y26_pano | gdino | 0.215 |
| `f5af69b9` | 0.9373 | 0.7234 | y11l_pano, y11x_pano_h200 | — | 0.139 |
| `fbbe99ca` | 0.7941 | 0.6647 | qwen3-vl-8b, y26_pano | gdino | — |

Use it **after** judging a pano, as a second sweep: shown first, it would steer the miss scan
toward where other models looked. It is not a recall estimate: the challengers' own precision on
this split is unknown until the review.

## 6. AI pre-read of the 147 detections (not ground truth)

`analysis_out/bayonne_159/ai_preread/preread__claude-opus-5-5.json` holds a **model's** labels,
with the rubric it applied in the same file. Rater: Claude Opus 5.5 (`claude-opus-5-5`),
2026-10-02, looking at native-resolution square crops (side 0.09 × native width, 484-518 px,
shown 1:1, ring at the peak) in 25 contact sheets of six. It is a heads-up on what the detector
fires on in Bayonne. It is not a verdict file, it is not in `benchmark/bayonne/`, and **no number
from it is a precision.** The crops show more pixels than the model or the reviewer sees, which
favours `ramp` and `cant_tell`.

| detection confidence | ramp | not_ramp | cant_tell |
|---|---:|---:|---:|
| 0.55-0.70 | 49 | 7 | 20 |
| 0.70-0.85 | 35 | 2 | 7 |
| 0.85-1.00 | 25 | 1 | 1 |
| all 147 | 109 | 10 | 28 |

What it says for the review: most detections are lowered kerbs at the ends of zebra crossings,
often with a white-painted kerb face or white tactile paving. The doubtful ones are driveway and
garage lowerings with no crossing (one at 0.97), manhole covers on the footway, and one roof in a
pano whose horizon is tilted; they cluster below 0.70. Re-derive the table with
`python scripts/analysis/bayonne_159.py preread-summary` (`ai_preread/summary.json`); the crops
regenerate with `preread-crops`.

## 7. Paid legs: not run

No money was spent. Expected cost for 125 panos, from the ledger's measured dollars per pano
(rows before 2026-10-02; `analysis_out/bayonne_159/paid_legs_estimate.json`,
`bayonne_159.py paid-estimate`):

| leg | $/pano (measured panos) | expected |
|---|---:|---:|
| gemini-3.6-flash | 0.01234 (182) | $1.54 |
| gemini-3.1-pro-preview | 0.01621 (180) | $2.03 |
| gemini-3.7-flash | 0.01140 (180) | $1.42 |
| claude-opus-5, effort low | 0.04727 (277) | $5.91 |
| total | | $10.90 |

Dollars are estimates and the billing console is authoritative; the rates rest on 180-277
measured panos per leg. To run them after the review (they need verdicts only to be scored, not
to run): `python scripts/model_comparison/compare.py benchmark/bayonne --unreviewed --models
gemini:gemini-3.6-flash,gemini:gemini-3.1-pro-preview,claude:claude-opus-5` with the default
ledger.

## 8. Commands, in order

```bash
# staging (done)
python scripts/analysis/imagery_manifest.py --write --cities bayonne
python scripts/analysis/bayonne_159.py verify
python scripts/analysis/bayonne_159.py band --write              # needs panos/

# GPU, makelab2 A40 (done 2026-10-02; log committed as analysis_out/bayonne_159/gpu_run.log)
bash scripts/analysis/bayonne_159_gpu.sh    # repro, extract, parity, yolo, open, qwen, molmo, export

# CPU reads (done; each has --write, and checks against the committed file without it)
python scripts/analysis/bayonne_159.py checks
python scripts/analysis/bayonne_159.py firing --print
python scripts/analysis/bayonne_159.py frame --print
python scripts/analysis/bayonne_159.py candidates --print
python scripts/analysis/bayonne_159.py paid-estimate --print
python scripts/analysis/bayonne_159.py preread-crops && python scripts/analysis/bayonne_159.py preread-summary
```

**Cost: $0, about 1.7 GPU-hours of a shared A40.** The seven challenger legs total 5,508 s
of wall time (one `paid: false` row each in `analysis_out/usage_log.jsonl`, Molmo's failed leg
included); the two extractions took 2 min 50 s (laurens_mapillary, 94 panos) and 3 min 16 s
(Bayonne, 125 panos), timed in `analysis_out/bayonne_159/gpu_run.log` (the run's log, progress
bars stripped). The A40 was shared with other jobs throughout, so these times are upper bounds
for the hardware.

The `GPU` step ran from a git worktree of this branch on makelab2 (`~/wt-bayonne159`), with the
Bayonne panos copied from the archive and checked against `imagery_manifest.json`, and
`benchmark/laurens_mapillary/panos` linked from the main checkout there.

## 9. What is left after the review

1. Commit `benchmark/bayonne/verdicts.json` with `review_notes` (reviewer, confidence, what
   fought the rubric: driveway lowerings, the band, the two-wheeler captures).
2. `python scripts/score_validation.py benchmark/bayonne`; quote the unbiased column.
3. `python scripts/analysis/operating_point_curve.py attach-gt --cities bayonne --cache
   analysis_out/bayonne_159/op_cache` → `analysis_out/op_cache/bayonne.json` (CPU, no GPU re-run).
4. Register the split (the diff below), then `low_floor_sweep.py parity --cities bayonne`
   (expect the 3 seam peaks; consider a `PARITY_EXCEPTIONS` entry if the count arm trips), and
   `sweep` / `hist` / `gtbias` / `floor` / `distance`.
5. #55 incremental-FP gallery at **op-threshold 0.25**, tags into
   `benchmark/bayonne/incremental_fp_tags.json`, then `corrected --op-threshold 0.30` and
   `tagcheck`. Section 4.3 predicts a larger item count than paterson's 10.
6. Move the challenger detections from `analysis_out/bayonne_159/model_detections/` into
   `benchmark/model_detections/` and score the row: `compare.py benchmark/bayonne --models ...`
   (cache hits, no GPU).
7. Decide the paid legs (section 7).
8. `python scripts/analysis/train_overlap_check.py` (network, ~10 min) and commit
   `benchmark/train_overlap.json`; only then does `export_benchmark.py` accept the split.
9. Docs: `benchmark/README.md` (both tables + a Bayonne section), `docs/model_comparison.md`
   coverage matrix, `docs/operating_point.md` (held-out rows).
10. HF publish of the panos (Jon's call; not done here).

**Registration diff, prepared and not landed** (it needs verdicts: `train_overlap_check.py`
exits on any split in `BENCHMARK_SPLITS` without `verdicts.json`, and the registries feed tables
and tests that would then claim a reviewed split):

- `scripts/analysis/low_floor_sweep.py`: add `"bayonne"` to `CITY_SPLITS`, and
  `HELD_OUT["bayonne"] = "non-US city -- the pooled recommendation is a US-deployment basis
  (first Panoramax split; held out for geography, not GT quality)"`, adjusting the GT-quality
  clause to what the review records.
- `scripts/analysis/miss_decomposition.py`: the same `HELD_OUT` entry (the two registries must
  agree, `test_registries_agree_with_low_floor_sweep`).
- `scripts/export_benchmark.py`: add `"bayonne"` to `BENCHMARK_SPLITS`.
- `scripts/analysis/plot_operating_point.py`: `SERIES["bayonne"] = "#52514e"` (neutral ink,
  held out) and a `HELD_DASH` entry.
- `tier_of()`: no change (section 1).

## 10. Not done here, and why

- **The review**, and anything scored against it: it is Jon's.
- **Registration**: needs verdicts (section 9).
- **Paid legs**: no spend today; expected cost in section 7.
- **HF publish**: Jon's call.
- **The labeler's seam defect**: found here, not fixed; the labeler repo was out of scope today.
- **Qwen3-VL-32B**: its weights are not cached on makelab2, and downloading them for one
  leg was not worth it ahead of the review; it can run after.
- **Molmo2-8B**: ran in its own env and failed on every pano (section 5); not retried.
