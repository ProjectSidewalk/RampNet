# Where RampNet's misses live on the open-imagery rigs (#263 Track 1, #258)

Issue: [#263](https://github.com/ProjectSidewalk/RampNet/issues/263) (Track 1), background in
[#258](https://github.com/ProjectSidewalk/RampNet/issues/258) and
[#46](https://github.com/ProjectSidewalk/RampNet/issues/46). Run 2026-10-20. CPU only, no paid
model, no GPU (the one GPU step, laurens_gsv, was not run; see [Gaps](#gaps)).

RampNet loses about a third of its recall on the open rigs Project Sidewalk deployments are
moving to (#151, #159). This doc asks what kind of miss that is, using the #46 miss taxonomy
(`scripts/analysis/miss_taxonomy.py`) on the open-imagery splits.

## The reading rule (fixed in #263 before any number was looked at)

Each missed ramp goes into one of four buckets, from the committed 0.05-floor peak caches:

- **merged**: a peak above threshold sat inside the match radius, but a neighbouring ramp
  claimed it. One peak for two ramps.
- **sub_threshold**: a peak sat inside the radius, scoring between 0.05 and the threshold. The
  model found the ramp and was not confident enough.
- **localization**: no peak in the radius, but an unclaimed one within twice the radius.
- **silent**: nothing at all, even at the 0.05 floor.

The rule: **if silent dominates, the gap is labels** (the model does not respond, so it has to be
taught). **If sub_threshold dominates, the gap is calibration**, and the cheap verifier rungs in
#258 come before any retrain.

"Dominates" is made concrete here as: the pano-bootstrap 95% interval of (sub_threshold share
minus silent share) excludes zero. Otherwise neither leads. The bootstrap resamples panos within
each split, because misses cluster by pano (a merged pair is two misses on one pano), so it is
wider than the Wilson interval, which treats misses as independent. Both are reported.

## Result

At the 0.30 operating point (#79), **neither bucket dominates on the open rigs taken together**.
On the three GoPro Max splits pooled, sub_threshold is 46.9% of misses and silent is 40.1%, and
the difference is −2 to +16 points. Split by split, only laurens_mapillary passes the rule
(calibration). Bayonne is a tie, budapest leans silent and juneau leans sub_threshold, and none
of those three is separated.

Counted in recall points, the gap is about half calibration and half silence. GoPro Max is 0.236
recall points behind the #46 reference pool at 0.30. Sub_threshold misses account for +0.128 of
that and silent misses for +0.116. Merged misses go the other way (−0.026).

At the 0.55 deployed threshold sub_threshold leads almost everywhere. That is mostly the 0.30 to
0.55 band that #79 already proposes to give up, so it is not new information.

The same camera does fine inside the reference pool. Morgantown is GoPro Max (95 of 95 scored
panos) and reaches recall 0.805 at 0.30, with silent misses costing 0.075 recall points against
the reference's 0.062. So the silent excess is not the camera model. It is concentrated in
bayonne (+0.171), budapest (+0.095) and laurens_mapillary (+0.074).

So #263's decision point does not get a single answer from this. The two levers are about the
same size on GoPro Max: cheap rescoring of sub-threshold peaks can reach at most the
sub_threshold excess, and labels or data are the only thing that reaches the silent excess.
Which to do first is Jon's call.

### Bucket shares at 0.30 (Wilson 95% in brackets)

| population | panos | GT | misses | recall | merged | sub_threshold | localization | silent | sub − silent, boot 95% (pts) | reading |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| `laurens_mapillary` | 59 | 249 | 118 | 0.526 | 12 · 10.2% [6–17] | 70 · 59.3% [50–68] | 2 · 1.7% [0–6] | 34 · 28.8% [21–38] | +14 to +48 | sub_threshold leads: calibration |
| `bayonne` | 87 | 301 | 152 | 0.495 | 3 · 2.0% [1–6] | 72 · 47.4% [40–55] | 7 · 4.6% [2–9] | 70 · 46.1% [38–54] | −14 to +16 | neither leads |
| `budapest_district5` | 89 | 300 | 107 | 0.643 | 14 · 13.1% [8–21] | 35 · 32.7% [25–42] | 11 · 10.3% [6–17] | 47 · 43.9% [35–53] | −29 to +6 | neither leads |
| `juneau` | 94 | 297 | 72 | 0.758 | 10 · 13.9% [8–24] | 36 · 50.0% [39–61] | 2 · 2.8% [1–10] | 24 · 33.3% [24–45] | −4 to +37 | neither leads |
| GoPro Max pooled (laurens_mapillary, bayonne, budapest) | 235 | 850 | 377 | 0.556 | 29 · 7.7% [5–11] | 177 · 46.9% [42–52] | 20 · 5.3% [3–8] | 151 · 40.1% [35–45] | −2 to +16 | neither leads |
| `morgantown` (GoPro Max, in the reference) | 95 | 267 | 52 | 0.805 | 9 · 17.3% [9–30] | 22 · 42.3% [30–56] | 1 · 1.9% [0–10] | 20 · 38.5% [26–52] | −16 to +26 | neither leads |
| **#46 reference, 7 US splits** | 657 | 2060 | 427 | 0.793 | 124 · 29.0% [25–34] | 166 · 38.9% [34–44] | 9 · 2.1% [1–4] | 128 · 30.0% [26–34] | +0.5 to +17 | sub_threshold leads: calibration |
| `laurens_gsv` | — | — | — | — | not run: no op_cache ([Gaps](#gaps)) | | | | | |

The reference row is the #46 table (29.0 / 38.9 / 2.1 / 30.0), reproduced exactly by
`miss_taxonomy.py --cities` with the seven original US splits (pinned in
`tests/test_open_rig_misses_258.py`). It is often called the GSV pool, but it is not: three of
the seven are GSV (bend, paterson, gainesville) and four are Mapillary (richmond, clovis,
morgantown, annapolis). Paterson alone contributes 80 of its 124 merged misses.

"panos" counts panos with at least one GT ramp, the population the taxonomy scores
(laurens_mapillary has 94 panos, 59 with a ramp).

### Recall points per bucket at 0.30, and the excess over the reference

| population | merged | sub_threshold | localization | silent | recall gap to reference |
|---|--:|--:|--:|--:|--:|
| `laurens_mapillary` | 0.048 (−0.012) | 0.281 (+0.201) | 0.008 (+0.004) | 0.137 (+0.074) | +0.267 |
| `bayonne` | 0.010 (−0.050) | 0.239 (+0.159) | 0.023 (+0.019) | 0.233 (+0.171) | +0.298 |
| `budapest_district5` | 0.047 (−0.013) | 0.117 (+0.036) | 0.037 (+0.032) | 0.157 (+0.095) | +0.149 |
| `juneau` | 0.034 (−0.026) | 0.121 (+0.041) | 0.007 (+0.002) | 0.081 (+0.019) | +0.035 |
| GoPro Max pooled | 0.034 (−0.026) | 0.208 (+0.128) | 0.024 (+0.019) | 0.178 (+0.116) | +0.236 |
| `morgantown` | 0.034 (−0.026) | 0.082 (+0.002) | 0.004 (−0.001) | 0.075 (+0.013) | −0.013 |
| #46 reference | 0.060 | 0.081 | 0.004 | 0.062 | — |

Shares hide that a split can have the same mix as the reference and three times the misses. This
table is the one to read for "how much recall could each lever buy".

### Near field and far field at 0.30 (18 m boundary, flat-ground distance)

| population | near GT | near misses | near recall | near merged / sub / loc / silent | far GT | far misses | far recall | far merged / sub / loc / silent |
|---|--:|--:|--:|---|--:|--:|--:|---|
| `laurens_mapillary` | 216 | 98 | 0.546 | 7 / 61 / 2 / 28 | 33 | 20 | 0.394 | 5 / 9 / 0 / 6 |
| `bayonne` | 191 | 93 | 0.513 | 0 / 44 / 6 / 43 | 110 | 59 | 0.464 | 3 / 28 / 1 / 27 |
| `budapest_district5` | 201 | 67 | 0.667 | 6 / 19 / 9 / 33 | 99 | 40 | 0.596 | 8 / 16 / 2 / 14 |
| `juneau` | 190 | 27 | 0.858 | 0 / 17 / 2 / 8 | 107 | 45 | 0.579 | 10 / 19 / 0 / 16 |
| GoPro Max pooled | 608 | 258 | 0.576 | 13 / 124 / 17 / 104 | 242 | 119 | 0.508 | 16 / 53 / 3 / 47 |
| #46 reference | 1360 | 180 | 0.868 | 48 / 84 / 3 / 45 | 700 | 247 | 0.647 | 76 / 82 / 6 / 83 |

The open-rig deficit is a near-field deficit. On the reference, most misses are far (247 of
427). On GoPro Max, 258 of 377 are near, where the model has the pixels. Near-field silent misses
are 104 of 608 near ramps on GoPro Max (17%) against 45 of 1,360 on the reference (3.3%). Those
104 are the population that labels could plausibly reach. Juneau, the Trimble MX7 city, looks
like its sibling annapolis: near recall 0.858 and most misses far (45 of 72).

### Merged pairs, how far below threshold the sub-threshold peaks sit, and the null (0.30)

| population | merged pairs | Chebyshev median px | > min_distance 10 | Euclid ≤ 2σ | < 8 px | sub-threshold peak median [q25, q75] | share ≥ 0.15 | null rate sub / merged |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `laurens_mapillary` | 12 | 12.2 | 10 | 10 | 1 | 0.143 [0.087, 0.206] | 49% | 8.3% / 7.1% |
| `bayonne` | 3 | 13.7 | 2 | 3 | 1 | 0.186 [0.107, 0.253] | 54% | 12.2% / 12.8% |
| `budapest_district5` | 14 | 13.1 | 10 | 13 | 2 | 0.159 [0.118, 0.206] | 60% | 9.7% / 10.1% |
| `juneau` | 10 | 15.0 | 8 | 8 | 0 | 0.180 [0.120, 0.239] | 58% | 10.1% / 11.7% |
| GoPro Max pooled | 29 | 12.4 | 22 | 26 | 4 | 0.157 [0.099, 0.219] | 53% | 10.2% / 9.2% |
| #46 reference | 124 | 11.4 | 78 | 108 | 24 | 0.157 [0.094, 0.228] | 53% | 7.6% / 10.1% |

- **Merged is small on every open split** (2 to 14% of misses), and the pairs that do merge
  look like the reference's: most are farther apart than the extractor's `min_distance` (22 of
  29), so the heatmap had one mode. Nothing here argues for the sigma or head work (#221).
- **The sub-threshold peaks sit exactly as far below threshold as the reference's**: median
  0.157 on both, 53% at or above 0.15 on both. It is the same calibration problem as on the
  reference, with more of it. A verifier tuned on the reference band would see the same score
  range here.
- **The null** is the chance a peak lands in the radius by accident, with the ramp's elevation
  held and its azimuth randomized. It is 8 to 12% for sub_threshold on the open splits (7.6% on
  the reference). Bayonne's is the highest, so up to about one in eight of its sub-threshold
  misses could be a peak that happened to be nearby.

### At 0.55 (the published operating point)

| population | GT | misses | recall | merged | sub_threshold | localization | silent | sub − silent, boot 95% (pts) | reading | recall gap to reference (sub / silent excess) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|
| `laurens_mapillary` | 249 | 152 | 0.390 | 9 · 5.9% [3–11] | 107 · 70.4% [63–77] | 0 · 0.0% [0–2] | 36 · 23.7% [18–31] | +33 to +60 | sub_threshold leads: calibration | +0.333 (+0.272 / +0.080) |
| `bayonne` | 301 | 188 | 0.375 | 3 · 1.6% [1–5] | 108 · 57.5% [50–64] | 2 · 1.1% [0–4] | 75 · 39.9% [33–47] | +3 to +32 | sub_threshold leads: calibration | +0.347 (+0.201 / +0.184) |
| `budapest_district5` | 300 | 147 | 0.510 | 8 · 5.4% [3–10] | 81 · 55.1% [47–63] | 2 · 1.4% [0–5] | 56 · 38.1% [31–46] | −1 to +34 | neither leads | +0.212 (+0.112 / +0.122) |
| `juneau` | 297 | 93 | 0.687 | 9 · 9.7% [5–17] | 58 · 62.4% [52–72] | 0 · 0.0% [0–4] | 26 · 28.0% [20–38] | +17 to +52 | sub_threshold leads: calibration | +0.035 (+0.037 / +0.022) |
| GoPro Max pooled | 850 | 487 | 0.427 | 20 · 4.1% [3–6] | 296 · 60.8% [56–65] | 4 · 0.8% [0–2] | 167 · 34.3% [30–39] | +18 to +35 | sub_threshold leads: calibration | +0.295 (+0.190 / +0.132) |
| #46 reference | 2060 | 572 | 0.722 | 110 · 19.2% [16–23] | 325 · 56.8% [53–61] | 3 · 0.5% [0–2] | 134 · 23.4% [20–27] | +26 to +40 | sub_threshold leads: calibration | — |

Raising the threshold to 0.55 moves misses into sub_threshold and leaves silent nearly where it
was (151 to 167 on GoPro Max), so at 0.55 the rule reads "calibration" almost everywhere. The
silent excess in recall points barely moves (+0.116 to +0.132). The 0.30 table is the one that
separates the two levers. Recall at 0.55 matches the scoreboard to within a point (laurens 0.390,
juneau 0.687; budapest 0.510 here vs 0.503 and bayonne 0.375 vs 0.372, because the taxonomy
scores only panos whose missed-ramp check was confirmed and that carry at least one GT ramp).

## Per rig

- **GoPro Max on Mapillary, Laurens (laurens_mapillary).** Calibration. 59% of misses peak below
  0.30, and the shortfall against the reference is mostly sub_threshold (+0.201 of +0.267
  recall points). The model finds these ramps and scores them low. This fits #151, where the
  rig effect survived pairing on corners both rigs saw (ΔF1 +0.112), and #82's frozen-model
  probe, where pushing GSV panos toward the measured GoPro pixel statistics on all axes at once
  cost 0.209 recall at 0.30 (`all@gopro`, `docs/aug_transfer_82.md`).
- **GoPro Max on Panoramax, Bayonne (bayonne).** A tie. Sub-threshold and silent are 47% and
  46% of misses and cost about the same recall (+0.159 and +0.171). Half of what Bayonne misses,
  the model never responds to. Bayonne's GT is single-rater at medium confidence (bollards,
  raised crossings and roundabouts fought the rubric), so part of that silent bucket may be
  marks a second rater would not make.
- **GoPro Max on Mapillary, Budapest (budapest_district5).** Leans silent (44% vs 33%, not
  separated) and has the highest localization share on the board (10%). Budapest's GT is
  single-rater at low confidence, and the rubric does not transfer cleanly, so this is the split
  to trust least. The brief for this work called Budapest a consumer 360 rig distinct from the
  GoPro Max splits; `records.jsonl` says it is GoPro Max (89 of 89 scored panos).
- **Trimble MX7 on Mapillary, Juneau (juneau).** Close to the reference: 0.035 recall points
  behind, misses mostly far field. Juneau is not part of the open-rig deficit. Its sibling,
  annapolis (same rig, same contributor), is in the reference pool.
- **GSV in the same town (laurens_gsv).** Not run. This is the cleanest control (same town,
  largely the same ramps, the rig as the only change) and it needs one GPU extract. See
  [Gaps](#gaps).
- **GoPro Max inside the reference (morgantown).** At reference level (recall 0.805, silent
  excess +0.013). Whatever makes laurens, bayonne and budapest hard, the camera model alone
  is not it.

## Caveats

- **GT is anchored to RampNet.** On every split here, GT is the reviewed RampNet detections
  exported at a 0.55 floor plus the ramps the reviewer marked as missed. So every miss at 0.55
  is a ramp the reviewer added, and a peak between 0.30 and 0.55 counts as a hit only where the
  reviewer added a mark there. A ramp that both RampNet and the reviewer missed is in no bucket.
  The bucket mix describes the misses a reviewer noticed.
- **Single rater on every open split** (jonf). Bayonne and juneau are rated medium confidence and
  budapest low. Laurens's confidence rating is not on the record. No second rater has repeated
  any of them, so a reviewer who marks more distant or more ambiguous ramps moves misses into
  silent.
- **n is small per split.** 72 to 152 misses at 0.30, so a share carries a Wilson interval of
  about ±8 to ±11 points and a wider pano-bootstrap interval. The per-split readings that say
  "neither leads" are mostly underpowered, not ties.
- **"silent" is an upper bound on "the model saw nothing."** On the #46 splits,
  `silent_activation.py` found only 8% of silent misses had a flat heatmap. The rest were the
  tail of an adjacent mode (62%) or a faint response on site (30%)
  (`docs/rampnet1_findings.md`). That breakdown
  needs the raw heatmaps (GPU) and has not been run on the open splits. Until it is, the silent
  excess is not proof of a label gap.
- **The distance is approximate.** The near/far split uses the flat-ground estimate with a
  2.5 m camera height (`stage1_label_recall.geom`). GoPro Max rigs sit at about 1.8 to 2.1 m
  (`docs/da3_calibration_101.md`), so true distances on these splits are 15 to 27% shorter than
  estimated, and some "far" ramps are near. Flat-ground geometry also agrees with depth less
  well on Mapillary (Spearman 0.81) than on GSV (0.95). The near-field counts above are, if
  anything, low.
- **The reference pool is mixed-rig**, not GSV (see under the first table).

## Gaps

- **laurens_gsv is not bucketed.** It has no op_cache. Producing one is a GPU extract of 86
  panos with the repo's own extractor: `scripts/analysis/open_rig_misses_258_gpu.sh` (committed
  on this branch) re-extracts laurens_mapillary into a scratch cache as a repro check, then
  writes `analysis_out/op_cache/laurens_gsv.json`. A first attempt on makelab2 on 2026-10-20 was
  launched but not confirmed: the call that launched it timed out on the client side, and by
  the standing rule there were no retries and no further contact with the host. Whether the
  cache was written there is unknown. To fill it:

  ```bash
  # on a GPU host with benchmark/laurens_gsv/panos and benchmark/laurens_mapillary/panos
  bash scripts/analysis/open_rig_misses_258_gpu.sh 2>&1 | tee open_rig_258_gpu.log
  # verify: tp/fp/fn at 0.55 must equal the scoreboard's laurens_gsv RampNet cell
  #         (P 0.925 / R 0.505, benchmark/README.md)
  # then commit analysis_out/op_cache/laurens_gsv.json, record the paid:false usage row,
  # and re-run (laurens_gsv then appears in the JSON instead of under "gaps"):
  python scripts/analysis/open_rig_misses_258.py
  ```

  Adding the cache also changes `miss_taxonomy.py`'s default output (laurens_gsv stops
  printing "no cache") and the scoreboard's AP-provenance row for laurens_gsv, so those
  committed outputs need regenerating in the same change.
- **The silent activation breakdown** (see Caveats) on the open splits, also GPU.
- **No second rater** on any open split.

## Reproduce

From a clean clone, CPU only (about 10 seconds):

```bash
pip install -e .
# The machine-readable table (analysis_out/open_rig_misses_258.json), and its check:
python scripts/analysis/open_rig_misses_258.py
python scripts/analysis/open_rig_misses_258.py --check
# The taxonomy itself on the open splits, at either threshold:
python scripts/analysis/miss_taxonomy.py --cities laurens_mapillary,bayonne,budapest_district5,juneau
python scripts/analysis/miss_taxonomy.py --cities laurens_mapillary,bayonne,budapest_district5,juneau --threshold 0.55
# The #46 reference table, unchanged:
python scripts/analysis/miss_taxonomy.py --cities richmond,bend,clovis,morgantown,annapolis,paterson,gainesville
```

Inputs, all committed: `analysis_out/op_cache/<split>.json` (0.05-floor peaks from the
published `projectsidewalk/rampnet-model`, with verdict-derived GT) and
`benchmark/<split>/records.jsonl` (camera models). The bootstrap uses 2,000 replicates with seed
20261020; the null uses 200 trials per miss, seeded per split.

What `miss_taxonomy.py --cities` changes: it restricts the run to the named splits and pools the
named ones that are in `US_SPLITS`. With no flag the output is byte-identical to before. Its
title still says "seven US splits" although laurens_mapillary and juneau have since joined
`US_SPLITS` (the default pool is nine splits, 2,606 GT). That stale label is kept so the default
output does not move; `--cities` runs print the real count.

Cost: no GPU and no paid model were used for any number in this doc, so there is no usage row.
