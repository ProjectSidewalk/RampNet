# Lyon label audit (#159, #263): are Lyon's self-labels good enough to train on?

**Status (2026-10-20): staged, not reviewed.** The sample is drawn and committed, the plan and
the pass bar are written down (`benchmark/lyon/audit_plan.json`), and the known-artifact counts
are in. The panos are not local yet, so the overlay measurement, the imagery pin and the gallery
check are still to do (section 7). Nobody has judged a detection. The plan, the pass bar and the
group list are a **working draft** posted on #159 before the review; Jon can amend them until the
review opens. Lyon is not approved for training and nothing about it is frozen.

Issue [#159](https://github.com/ProjectSidewalk/RampNet/issues/159) (decision of
[2026-10-06](https://github.com/ProjectSidewalk/RampNet/issues/159#issuecomment-6020299907)),
campaign [#263](https://github.com/ProjectSidewalk/RampNet/issues/263), track 2. The sampler is
sidewalk-auto-labeler [PR 167](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/167);
the run is sidewalk-auto-labeler [PR 163](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/163)
(`docs/panoramax-lyon.md` section 8 there).

## 1. What Lyon is for (Phase 0)

Lyon exists to answer whether this rig class's self-labels are good enough to train on, per rig
and producer. If Lyon trains, the same bundle later gives the in-city held-out read. **These
panos never train**: every `panorama_id` in `benchmark/lyon/records.jsonl` is held out of every
training pool, whatever the audit reads.

- **Not a benchmark split.** Lyon stays training-only under #158's rule that a city trains or
  evaluates, never both. `benchmark/lyon/` is staged the way a split is so that the same tools
  review and score it, but it is not registered (`BENCHMARK_SPLITS`, `US_SPLITS`, `HELD_OUT`),
  enters no pooled number, and is not published to the Hub benchmark.
- **Why per rig and producer.** The 5 m run mixes a professional multi-camera rig (Grand Lyon's
  Point Grey Ladybug, 8192x4096, solid black below y ≈ 0.81) with consumer GoPro Max captures
  from many accounts. One account, ecartip, holds most of the 2026 GoPro Max panos, with a large
  roof rack, detections that sit lower in the frame, and pose reported as exactly 0/0. A pooled
  precision would let the largest producers decide what "GoPro Max" or "Ladybug" means.
- **Imagery tier, if it mattered:** the GoPro Max panos fall in `action-modern` like Bayonne's;
  the Ladybug panos have no tier because nothing on the board uses that rig. No `tier_of()`
  branch is added, because Lyon is not scored as a split.

## 2. The sample

`scripts/export_benchmark.py` with the new stratified mode, on the 5 m run of record
(`results.jsonl`, 174,405 records, sha256 `60e688b2…61ef`). Seed 159, 30 m spacing across all
strata, `top` 10 / `random` 140 / `empty` 25 = **175 panos**. The `random` and `empty` targets are
allocated across four groups in proportion to each group's pool, with at least 20 `random` and 3
`empty` panos per group. The strata are defined at 0.55, as for every split: `random` panos have at
least one detection at 0.55, `empty` panos none. The records keep every detection down to **0.30**
so the reviewer also judges the 0.30 tier (section 4).

The four groups are the four strata of Jon's decision. `benchmark/lyon/group_map.json` maps
`camera_model|copyright` to them (first match wins):

- **`grand_lyon_ladybug`**: Grand Lyon 8192x4096, both account spellings (`grand lyon`,
  `grand-lyon`), the Point Grey Ladybug panos and the make-less panos of the same Métropole
  campaign.
- **`ecartip_gopro_max`**: ecartip's GoPro Max panos.
- **`other_gopro_max`**: every other account's GoPro Max panos.
- **`other_cameras`**: everything else (Kandao, Insta360, RICOH, Garmin, GoPro MAX2, the make-less
  10560x5280 bike-route captures, and the few Ladybug panos not from Grand Lyon).

**Group table** (`python scripts/analysis/lyon_review_159.py table`, from `audit_plan.json`):

| group | run panos with a 0.55 detection / without | top | random | empty | detections ≥ 0.55 (random) | detections ≥ 0.30 (random + empty) |
|---|---:|---:|---:|---:|---:|---:|
| grand_lyon_ladybug | 17,089 / 69,151 | 6 | 52 | 12 | 95 | 165 |
| ecartip_gopro_max | 14,576 / 30,829 | 3 | 44 | 5 | 81 | 148 |
| other_gopro_max | 7,710 / 25,654 | 1 | 24 | 5 | 43 | 61 |
| other_cameras | 1,513 / 7,883 | 0 | 20 | 3 | 25 | 54 |
| **all** | 40,888 / 133,517 | **10** | **140** | **25** | **244** | **428** |

With the 10 `top` panos the bundle holds **330 detections at 0.55 and 524 at 0.30** to judge.

**Inside each group** (all strata; `rig_breakdown` in `audit_plan.json`; descriptive, no bar):

- `grand_lyon_ladybug`, 70 panos: 44 tagged Point Grey Ladybug (2020 14, 2021 5, 2022 25), 26
  make-less (2020-2023). 60 from `grand lyon`, 10 from `grand-lyon`.
- `ecartip_gopro_max`, 52 panos: 44 from 2026, 8 from 2025.
- `other_gopro_max`, 30 panos from 9 accounts; the largest are `mike en gyroroue` 9 and
  `ign_ddc_dtce_np` 8.
- `other_cameras`, 23 panos: 8 make-less 10560x5280 (one bike-route account), 5 Insta360 X3,
  4 Kandao QooCam 3, 2 Garmin VIRB 360, 2 RICOH THETA S, 1 Insta360 X5, 1 Kandao QooCam 3 Ultra.

So the long-tail group mixes seven cameras with 1-8 panos each. Its precision speaks for "the
tail as a whole", not for any one camera in it.

**Producer credit and one masked name.** 152 panos are on `panoramax.ign.fr` (Etalab 2.0) and 23
on `panoramax.openstreetmap.fr` (CC BY-SA 4.0); the gallery shows each pano's producer and licence
and links it on its own instance. One producer name is an email address; `stage` masks it to
`j***@***` in the committed records, as the labeler census does (sidewalk-auto-labeler PR 150).
That is the only difference between the committed `records.jsonl` and the exporter's.

## 3. The pass bar (posted before the review)

- **Metric:** precision of the detections at confidence ≥ 0.55, the shipped threshold.
- **Read on:** each group's unbiased subset (`random` + `empty`; the `top` panos are excluded, as
  for every split). `empty` panos carry no 0.55 detection, so at 0.55 this is the `random` stratum.
- **Bar:** a group **passes if its point estimate is ≥ 0.80**, the #158 build line (≥ 0.80 build,
  0.50-0.80 visibility, < 0.50 drop). The 95% Wilson interval is reported beside it, and an
  interval that straddles 0.80 is reported as **not decisive**.
- **Consequence:** a group that fails is excluded from any Lyon training pool.
- **Reported alongside, with no bar:** recall at 0.55; precision and recall at 0.30; unsure and
  duplicate counts; per-rig rows inside each group.
- **Duplicates** count as false positives (the `rampnet.validation` default);
  `--lenient-duplicates` gives the other reading from the same verdicts.

How decisive this can be: with 25 detections (`other_cameras`) a point estimate of 0.80 has a
Wilson interval of about [0.61, 0.91]; with 95 (`grand_lyon_ladybug`) about [0.71, 0.87]. A group
near the bar will read "not decisive". More panos in a group is the only fix, and it is cheaper to
decide that now than after the review.

`python scripts/analysis/lyon_review_159.py score` computes all of this from `verdicts.json` and
refuses to run before it exists.

## 4. What the reviewer sees

The standard ground-truth gallery (`scripts/gt_gallery.py`, `benchmark/RUBRICS.md` §1), with two
differences from a benchmark session:

- **Detections down to 0.30 are shown.** Each crop's caption carries its confidence. Judge every
  one the same way. Mark as missed only a ramp with no detection at all. At scoring time a
  correct 0.40 detection counts as a ramp the model missed at 0.55, so recall at 0.55 stays
  correct, and the 0.30 tier gets its own precision without a second pass.
- **Nadir overlays are named.** The Grand Lyon panos are solid black below about y 0.81 (no
  downward camera). `nadir_band.json` will carry each pano's measured edge and a label, and the
  gallery hatches the region, labels it ("black nadir (no camera)") and asks before a missed-ramp
  mark lands inside it. ecartip's roof rack is real imagery, not an overlay: it is not hatched, and
  ramps hidden by it cannot be marked.

As for every split, the gallery renders each pano and crop at the model's input resolution
(4096x2048, crops 512x512 cut from it), never at native resolution (8192 to 11136 wide here).

**Rubric.** `benchmark/RUBRICS.md` as for every split, including the 2026-10-04 ruling that a
bike-lane cut is not a curb ramp. From the Bayonne review: French lowered kerbs at crossings are
often a plain lowering with a white-painted kerb face and no flare, sometimes with tactile paving,
and they count; driveway and garage lowerings do not.

**Before you start, do not open** `analysis_out/lyon_review_159/artifacts.json`'s `items` list.
It names detections you are about to judge. The counts below are the part meant to be read first.

## 5. Known artifacts to look at

Both from #159's decision comment, counted on this bundle's own detections
(`analysis_out/lyon_review_159/artifacts.json`, from the run's `results.jsonl`, no ground truth):

| group | detections ≥ 0.30 / ≥ 0.55 | in a fixed-pixel cell, ≥ 0.30 / ≥ 0.55 | above the horizon (y < 0.5), ≥ 0.30 / ≥ 0.55 |
|---|---:|---:|---:|
| grand_lyon_ladybug | 224 / 147 | 2 / 2 | 2 / 1 |
| ecartip_gopro_max | 177 / 107 | 0 / 0 | 12 / 4 |
| other_gopro_max | 69 / 51 | 0 / 0 | 3 / 2 |
| other_cameras | 54 / 25 | 1 / 0 | 3 / 2 |

(All strata, so these totals include the `top` panos.)

- **The Grand Lyon fixed-pixel cluster.** A "fixed-pixel cell" is a 1024x512 heatmap cell that
  holds a ≥ 0.30 peak in at least 5 panos of the pano's own sequence. On the first 10 m slice, 2.1%
  of Grand Lyon's 0.30 detections sat in such cells, mostly at (x 0.503, y 0.756), straight ahead
  just above the black nadir, and none reached 0.55. On this bundle 2 Grand Lyon detections sit in
  one, and both are ≥ 0.55: at 5 m spacing a sequence has twice the panos, so more cells repeat.
  A random sample shows the reviewer only a couple of these, so the review alone cannot measure
  how often the cluster is a vehicle part rather than a ramp. If that matters for mining from the
  0.30 tier, it needs a targeted look (section 9).
- **ecartip above the horizon.** 4 of ecartip's 107 detections at 0.55 in the bundle sit above the
  horizon (3.7%), against 10.3% in the first 10 m slice. ecartip reports pose as exactly 0/0, so
  its panos may not be levelled. These detections are judged like any other; a ray cast from them
  hits no ground, which matters for geometric mining whatever the verdict.

## 6. Caveats that travel with the numbers

- **Single rater, one pass.** The audit has one reviewer. The verdicts are per-rater and the rubric
  is `benchmark/RUBRICS.md`, so a second rater can repeat the pass on the same list.
- **Recall is not comparable 1:1 with the benchmark splits.** Those were reviewed with only the
  0.55 detections shown. Here the 0.30 detections are shown too, which can help the reviewer find
  ramps, so the GT may be more complete and recall at 0.55 lower than it would read under the
  benchmark protocol. Precision at 0.55 is unaffected.
- **Strata weights.** The pooled `all` row mixes `random` and `empty` panos at the sample's
  proportions (140 to 25), not the run's (about 1 to 3). That is the convention for every split;
  read the per-group rows, not the pool.
- **The seam.** The production extractor drops peaks within 10 heatmap cells of the 360 seam
  (`detection_border: exclude`, sidewalk-auto-labeler#130). The bundle's detections are what Lyon
  would contribute to training, so a ramp at the seam with no detection is a miss, as in Bayonne.
- **Mount and district are confounded with producer.** A group that fails says its labels are not
  good enough to train on; it does not say why.

## 7. Start the review (blocked on the imagery)

The 175 native panos are in the makelab2 archive (`/projects/makeabilitylab/sidewalk-auto-labeler/
runs/lyon/panos/`, sha256 per file in its `index.csv`). They have not been copied yet: makelab2 was
put on hold on 2026-10-20, before the fetch. Steps, in order, once it is clear:

```bash
# 1. on makelab2: tar only the sampled panos (ids from records.jsonl) and their index rows;
#    copy the tar here and unpack into benchmark/lyon/panos/ (git-ignored), with the
#    sampled rows of the archive's index.csv as benchmark/lyon/index.csv (git-ignored)
# 2. pin the imagery and measure the overlays
python scripts/analysis/imagery_manifest.py --write --cities lyon --panos-root .
python scripts/analysis/lyon_review_159.py band --write
python scripts/analysis/lyon_review_159.py verify             # expect: verify: OK
# 3. build and open the gallery
python scripts/gt_gallery.py benchmark/lyon
# open benchmark/lyon/gallery/index.html, review, "Export verdicts",
# save the download (lyon_verdicts.json) over benchmark/lyon/verdicts.json
# 4. after the review
python scripts/analysis/lyon_review_159.py score --write
```

## 8. Reproduce

```bash
# in sidewalk-auto-labeler, branch lyon-review-bundle-159 (PR 167), the 5 m run's results.jsonl:
python scripts/export_benchmark.py runs/lyon/results.jsonl --bundle <dir> \
    --sample 150 --empty-sample 25 --seed 159 --min-spacing 30 --top 10 \
    --group-key camera_model --group-key copyright --group-map <dir>/group_map.json \
    --group-min 20 --empty-group-min 3 --record-confidence 0.30 --records-only
# here (<dir>/group_map.json is benchmark/lyon/group_map.json):
python scripts/analysis/lyon_review_159.py stage --from <dir>
python scripts/analysis/lyon_review_159.py plan          # recomputes, compares to the committed file
python scripts/analysis/lyon_review_159.py artifacts --results runs/lyon/results.jsonl
pytest -q tests/test_lyon_review_159.py
```

The draw is deterministic: re-running the exporter from the committed code reproduced
`records.jsonl` and `sample.json` byte for byte (`bundle_provenance.json`). The run's
`results.jsonl` (798 MB) is not in either repo: it is on Hyak, on makelab2 and in the labeler
worktree `D:/Git/labeler-wt/lyon5m`, and `bundle_provenance.json` pins its sha256.

## 9. Not done, and decisions for Jon

- **The imagery, the overlay file, the manifest and the gallery check** (section 7): waiting on
  makelab2.
- **The review itself.** Nothing is judged; there is no `verdicts.json`.
- **Review load.** 175 panos and 524 detections, against Bayonne's 125 and 147. Showing only the
  0.55 detections would cut it to 330 (same panos, records re-staged without the 0.30-0.55
  peaks), at the cost of the 0.30 tier, which would then need a #55-style tag pass. Your call
  before the review opens.
- **Group sizes against the bar.** `other_cameras` (25 detections at 0.55) and `other_gopro_max`
  (43) will be hard to read as decisive near 0.80. Raising `--group-min` before the review is
  cheap; after it, it means a second draw.
- **The fixed-pixel cluster** needs a targeted look if the 0.30 tier is to feed mining (section 5).
  Not built.
- **Per-producer caps on training** (#159 decision, point 2) are a training-side step, not part of
  this audit.
