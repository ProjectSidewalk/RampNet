# Detector Phase 1, Stage 0: merged pairs and the horizon band (#263)

Issue [#263](https://github.com/ProjectSidewalk/RampNet/issues/263). This is Stage 0 of the plan
"Detector Phase 1 arms and an upscaled horizon-band pass" (2026-11-15, revised after its review;
section 4). **The plan is committed, with corrections, as
[`detector_phase1_263.md`](detector_phase1_263.md) (PR #306); its original file,
`detector-phase1-horizon-2x-2026-11-15.md`, is not in the repository.** Its three sections that hold the Stage 0 rules (3, 4 and 7) are copied verbatim in
[Appendix A](#appendix-a-the-plans-stage-0-rules-verbatim), and the rules as written in this doc and
that appendix are the record. The filename in both scripts' docstrings and in `protocol.plan` of
both JSON outputs names that plan. The plan cites a Phase 1 design doc, `docs/detector_phase1_263.md`;
the design it meant was on the unmerged branch `docs/detector-phase1-263`
([PR #270](https://github.com/ProjectSidewalk/RampNet/pull/270), closed without merging), and that
path now holds the 2026-11-15 plan itself (PR #306).

**Run date:** 2026-11-15; the narrow-minus-wide interval, the figure fix and the 0c script fixes
after review 1, 2026-11-25. **Scripts:** `scripts/analysis/merged_separation_263.py` (0a) and
`scripts/analysis/horizon_band_263.py` (0b; 0c written, not run). **Tests:**
`tests/test_merged_separation_263.py`, `tests/test_horizon_band_263.py`. **Outputs:**
`analysis_out/horizon_band_263/{merged_pairs,simulate}.{json,md}` and two figures in
`docs/figures/horizon_band_263/`. **Compute for the results:** CPU only, about 1 minute; committed
caches only, no network, no spend, so no ledger row.

| stage | status | binds? | verdict |
|---|---|---|---|
| 0a merged-pair histogram | run | yes: arm A and arm C triggers | **arm C not triggered** (25 of 136 pairs under 8 px, 18%); **arm A runs** (38 pairs at 12 px or more outside paterson, at least 30 needed) |
| 0b band simulation | run | no (exploratory) | **on S7 pooled**, every band config beats the matched-recall threshold, and F1 against the 1x pass is not above zero under the standard scorer. On Mapillary4 alone, three of the four configs are wholly below the 1x pass on F1 and 2x-narrow does not beat the matched threshold; on S7 without paterson, both 2x configs are wholly below the 1x pass on F1. Leave-one-split-out picks the **narrow** core on 7 of 7 splits at both scales, but narrow minus wide on F1 is not separated from zero. The 1.5x rows use real pixels; production and 0c upsample |
| 0c real strips on the A40 | script written, **not run** | yes: the binding gate | — |

## 0a. Merged pairs

A merged miss (`miss_taxonomy.py`, #46) is a missed ramp whose in-radius peak at 0.30 went to a
neighbouring ramp. The plan's rules: more than one third of the pairs under 8 px triggers arm C (the
stride-16 head); arm A (sigma 10 to 6) runs only if at least 30 pairs outside paterson are 12 px or
more apart. Pixels are heatmap px on the 512×1024 grid.

| pool | metric | pairs | under 8 px | share | arm C | at 12 px or more, outside paterson | arm A |
|---|---|---|---|---|---|---|---|
| 8 splits (US_SPLITS minus juneau) | Chebyshev | 136 | 25 | 0.184 | not triggered | 38 | **runs** |
| 8 splits | Euclidean | 136 | 17 | 0.125 | not triggered | 44 | **runs** |
| S7 (context) | Chebyshev | 124 | 24 | 0.194 | — | 31 | — |
| the plan's 9 splits (juneau not re-read; see below) | Chebyshev | 146 | 25 | 0.171 | not triggered | ≥ 38 | runs |

Both metrics give the same verdicts. Paterson holds 80 of the 136 pairs, and 20 of the 25 under 8 px.
The 9-split row is the 8 splits plus juneau's 10 merged pairs, none under 8 px, which
`open_rig_misses_258.md` already reports (its merged-pair table); juneau's caches were not read here.
Adding juneau back can only add pairs, so the 8-split arm A count of 38 is a lower bound on the
9-split count. The arm A margin is thin on S7 alone: 31 pairs against the bar of 30. On the 8 splits
the verdict holds up to a 14 px bar (38, 34, 31 pairs at 12, 13, 14 px; 26 at 15). The S7 recount
matches the committed `analysis_out/miss_taxonomy.json` (124 pairs, 78 above `min_distance`); the
script asserts this.

![Merged-pair separations](figures/horizon_band_263/merged_separation.png)

## 0b. Band simulation (exploratory, binds nothing)

The band is simulated from committed full-pano caches (#196): the in-core peaks of `u4096` (2×,
upsampled) or `r3072` (1.5×, real pixels where the native image has them) are merged into the 1×
`r2048` peaks at 0.30 by #197's seam-wrapped `dedupe`. Splits: S7 (richmond, annapolis, clovis,
morgantown, bend, paterson, gainesville). Intervals: paired pano bootstrap, stratified by split,
2,000 draws, pooled (micro) counts as in the plan's table. The matched-recall baseline is the highest
1× threshold whose recall reaches the band's (#197's definition). These splits chose the band
(review finding 2), so none of this is a test.

**S7 at 0.30, far unmatched detections as FPs (the standard scorer).** 1× alone: P 0.8928, R 0.7961,
F1 0.8417.

| config | F1 | ramps gained | FPs added | ΔR | ΔF1 | matched 1× threshold | its FPs added | band − matched ΔF1 |
|---|---|---|---|---|---|---|---|---|
| 2x-wide (0.45–0.60) | 0.8343 | +100 | +174 | +0.049 [+0.039, +0.059] | −0.007 [−0.015, +0.001] | 0.12 | +357 | +0.035 [+0.025, +0.044] |
| 2x-narrow (0.47–0.56) | 0.8369 | +89 | +146 | +0.043 [+0.034, +0.053] | −0.005 [−0.013, +0.003] | 0.14 | +296 | +0.029 [+0.020, +0.039] |
| 1.5x-wide | 0.8416 | +106 | +146 | +0.051 [+0.042, +0.061] | −0.000 [−0.008, +0.008] | 0.11 | +402 | +0.049 [+0.039, +0.058] |
| 1.5x-narrow | 0.8439 | +95 | +120 | +0.046 [+0.037, +0.056] | +0.002 [−0.005, +0.010] | 0.13 | +327 | +0.040 [+0.031, +0.050] |

**The 1.5x rows use real pixels; production and 0c upsample.** `r3072` is the native image resized
to 3072×6144 (`input_res_sweep_25.md`, arm table), so it has real detail wherever the native image
has more than 2048 rows. Production and the 0c strip resize native to 2048×4096 and then upsample
bicubic. So the 1.5x rows are an optimistic stand-in, and 1.5x looking better than 2x here (F1
0.8416 against 0.8343, wide) compares real pixels with upsampled ones. #196 found r4096 and u4096
indistinguishable at 2x; nothing has measured it at 1.5x. The plan says the same: "The 1.5x cache is
real pixels; production must upsample, which is untested at 1.5x."

**The same, with unmatched detections beyond 25 m ignored (upper bound).** 1× alone: F1 0.8460.

| config | F1 | FPs added | ΔF1 | band − matched ΔF1 |
|---|---|---|---|---|
| 2x-wide | 0.8525 | +105 | +0.006 [−0.001, +0.014] | +0.041 [+0.031, +0.051] |
| 2x-narrow | 0.8553 | +77 | +0.009 [+0.002, +0.017] | +0.037 [+0.027, +0.046] |
| 1.5x-wide | 0.8561 | +96 | +0.010 [+0.003, +0.018] | +0.051 [+0.041, +0.060] |
| 1.5x-narrow | 0.8585 | +70 | +0.013 [+0.006, +0.019] | +0.044 [+0.034, +0.053] |

The ramps gained and FPs added reproduce the plan's scratch table exactly (100/174/357, 89/146/296,
106/146/402, 95/120/327); `tests/test_horizon_band_263.py` pins them. On S7 pooled, every config
beats the matched-recall threshold by an interval above zero under both bounds. Against the 1× pass
at 0.30, no config gains F1 under the FP bound (every interval includes zero); under the ignore
bound three of the four do, and 2x-wide's interval (+0.006 [−0.001, +0.014]) includes zero. The two
bounds give different readings of F1 against base. That is the case where the plan schedules a
100-detection far-field audit, but only at 0c and only if the bounds disagree there.

**Leave-one-split-out choice between the cores.** For each held-out split, the core with the higher
pooled F1 at 0.30 (FP bound) on the other six was chosen; ties would go to wide. **Narrow won on all
seven, at both scales**, by 0.002 to 0.004 F1 each time. The LOSO-assembled result is therefore
identical to the narrow rows above. The seven folds share five sixths of their data, though, so 7 of
7 is close to a single comparison. The paired interval of narrow minus wide on S7 pooled (FP bound,
same bootstrap) is ΔF1 +0.002 [−0.001, +0.005] at 1.5× and +0.003 [−0.000, +0.006] at 2×, with ΔR
−0.005 [−0.009, −0.002] at both scales and ΔP +0.010 [+0.005, +0.014] at 1.5× (+0.010 [+0.006,
+0.015] at 2×). Narrow is not separated from wide on F1; it trades recall for precision. That
supports keeping wide as the binding config: Stage 0c tests the wide 1.5× strip first, as the plan
fixes, and the narrow strip runs beside it. (Both bounds in `simulate.md`; `narrow_minus_wide_S7` in
`simulate.json`.)

**By imagery (context, FP bound).** Mapillary4 (richmond, annapolis, clovis, morgantown) gains 37
ramps for 107 FPs at 2x-wide (ΔF1 −0.022 [−0.033, −0.011]) and 45 for 94 at 1.5x-wide (−0.013
[−0.024, −0.002]); 2x-narrow is also wholly below zero (−0.018 [−0.029, −0.008]) and does not beat
the matched threshold (+0.000 [−0.012, +0.013]). GSV3 (bend, paterson, gainesville) gains 63 for 67
at 2x-wide (+0.009 [−0.003, +0.021]) and 61 for 52 at 1.5x-wide (+0.014 [+0.002, +0.026]). Without
paterson, S7 gains 78 for 136 at 1.5x-wide (ΔF1 −0.007 [−0.017, +0.002]), and both 2× configs are
wholly below zero (2x-wide −0.016 [−0.025, −0.007], 2x-narrow −0.013 [−0.021, −0.004]). So the S7
pooled reading leans on the GSV splits. "F1 against the 1× pass not wholly below zero" is one of the
three 0c gate conditions, so a non-GSV source could fail it where S7 pooled does not.
laurens_mapillary, reported only, gains 5 ramps for 1 FP in every config (it has 9 of 249 GT ramps
beyond 25 m).

**S7 misses at 0.30 by flat-ground range, 1.5× (scorer definition: a GT ramp the 1:1 greedy match
leaves unmatched).**

| range | GT | misses | recovered by the full 1.5× pass | by the wide band | by the narrow band |
|---|---|---|---|---|---|
| 0–8 m | 452 | 46 | 1 | 0 | 0 |
| 8–18 m | 908 | 132 | 30 | 22 | 10 |
| 18–25 m | 395 | 85 | 30 | 25 | 24 |
| ≥ 25 m | 300 | 153 | 69 | 64 | 64 |
| at/above horizon | 5 | 4 | 2 | 2 | 2 |

All 132 misses at 8–18 m have their GT row inside the wide core and 55 inside the narrow one, which
is where the wide core's extra ramps come from. The 2× table, the per-split rows and the plan's
approximate miss counts (43/87/51/112, recovered 1/18/13/59, all reproduced) are in `simulate.md`.

![Band configs against lowering the 1x threshold](figures/horizon_band_263/band_vs_threshold.png)

## 0c. Real strips on the A40 (written, not run)

`horizon_band_263.py extract | calibrate | report`, or all three in order with `run0c`. Jon, or a
later session that Jon clears, runs it on makelab2:

```bash
python scripts/analysis/horizon_band_263.py run0c --panos-root /path/to/RampNet
```

`--panos-root` is a directory shaped like a RampNet checkout, holding
`benchmark/<split>/panos/<pano_id>.jpg` for S7, sao_paulo, manual_gold, juneau and bayonne; `extract`
checks that every pano is present before it touches the GPU. Expected cost, from the #196 timings:
about 3 to 4 GPU-h. The 1× pass and the full-pano 1.5× and 2× passes on S7's 859 panos dominate it.
`run0c` resumes in both stages: a selection split whose cache exists is skipped, and so is a
held-out split (step 4).

**Where the panos come from.** `extract` needs the **native** files: it resizes native to 2048×4096
itself (`threshold_sweep.PRE`, bilinear), so a copy that was resized beforehand, such as the Hub's
`4096x2048` config, gives different pixels.

| splits | source of the native panos |
|---|---|
| S7, sao_paulo | Hugging Face [`projectsidewalk/rampnet-benchmark`](https://huggingface.co/datasets/projectsidewalk/rampnet-benchmark), `native` config (dataset commit `63d5ffd0`): `python scripts/unpack_benchmark_panos.py --out <root> --cities richmond,annapolis,clovis,morgantown,bend,paterson,gainesville,sao_paulo` |
| manual_gold | the test split of `projectsidewalk/rampnet-dataset`, which stores these 1,000 panos at 2048×4096 already: `python scripts/fetch_manual_gold.py --images-only` (into the checkout it runs in) |
| juneau | **not published.** The makelab2 archive, `/projects/makeabilitylab/sidewalk-auto-labeler/runs/juneau/benchmark_bundle_254/panos/` (`juneau_split_254.md`) |
| bayonne | **not published.** The makelab2 archive, `/projects/makeabilitylab/sidewalk-auto-labeler/runs/bayonne/panos/`; also public on Panoramax under the ids in `records.jsonl`, but whether a re-fetch is byte-identical is unchecked (`benchmark/README.md`, Bayonne section) |

Every split's `imagery_manifest.json` pins each file's sha256, and
`python scripts/analysis/imagery_manifest.py --verify --panos-root <root>` checks a root against them.
juneau and bayonne are the whole open-rig claim, and they are not on the Hub (`docs/replication.md`,
the `benchmark/*/panos/` row), so a clean clone cannot re-run that part of 0c. Uploading both to
`rampnet-benchmark` would unblock it; that is Jon's call (`bayonne_split_159.md` §10,
`juneau_split_254.md` §8).

**What `run0c` does, in order:**

1. `extract --splits selection` (S7 + sao_paulo). For every pano it runs the 1× pass; each strip
   (1.5x-wide, 1.5x-narrow, 2x-wide); and, on S7 only, the full-pano pass at 1.5× and 2× with the
   same input/4 head and scaled `min_distance` (15, 20). It records GPU-side seconds (copy, forward,
   peak extraction) and peak memory per pass, and appends `paid: false` rows to
   `analysis_out/usage_log.jsonl`. If a run dies, its rows still count the panos finished in the
   split it died in. Each cache is written to a temporary name and renamed when complete, so a crash
   never leaves a truncated cache that would count as done.
2. `calibrate` freezes the context rule (S7) and the band thresholds (sao_paulo) into
   `analysis_out/horizon_band_263/calibration.json`. It refuses when sao_paulo has no 1× in-core
   detection at 0.30 for a config, since there is then no 1× precision for the band threshold to
   equal. This refusal is not skipped for `calibrate --smoke`, so a smoke calibration on a few
   panos (one per split, as in the local test below) can stop on it when its sao_paulo pano has no
   1× in-core detection at 0.30. That is harmless: a smoke calibration can never gate a held-out
   read. Extract more sao_paulo panos (`--limit`) for a smoke run that gets through.
   On a re-run, an existing non-smoke `calibration.json` is **not rewritten**: `run0c` runs
   `calibrate --check` instead and stops if it does not reproduce from the selection caches.
3. If the context rule fails for 1.5x-wide, `run0c` **stops** and the held-out splits stay unread.
4. Otherwise it runs `extract --heldout-read-once` on each held-out split that has no cache yet. Each
   held-out cache records the sha256 of the calibration it was read under. A held-out split whose
   cache exists was read once already and is not named again; if its recorded sha256 differs from
   the current `calibration.json`, `run0c` stops. A split that died part way wrote no cache, so it
   is read again from its first pano; nothing from the partial read was written or seen.
   `extract`'s guard is unchanged: it still refuses any named held-out split whose cache exists.
5. `report` writes `gate_0c.json` / `gate_0c.md` (`--out` moves them). The decision line follows
   1.5x-wide's verdict. When that config passes the gate but its measured cost is over the limit, it
   reads "passes the gate; over the 1.75x cost limit (…x), Jon's call"; when the held-out read ran
   under `--despite-failed-context`, it says so. The cost limit is reported in the decision, not
   enforced: no gate bar depends on it.

Then commit the caches, `calibration.json` and the gate files, and confirm with
`calibrate --check` and `report --check`. `calibration.json` records no commit, so committing it
changes neither its bytes nor the sha256 the held-out caches carry.

**The strip.** The production image is the 2048×4096 pano (`threshold_sweep.PRE`), upsampled
bicubic as #196's u4096 was. From it, `extract` cuts the rows around the core, adds 128 px of
circular wrap each side (192 px at 1.5×), and runs the forward pass with the head's Upsample set to
input/4. Only peaks in the core rows and the unwrapped columns are kept, mapped to pano coordinates
on the full-pass grid, and merged by #197's `dedupe`. The window is aligned to the backbone's
stride-32 grid, so the strip is sampled on the same grid as the full pass. That does not make its
interior features equal the full pass's: the backbone's GlobalResponseNorm layers normalise over the
whole feature map, so the context rule measures context plus that statistic (amendment 4):

| config | 1× rows (normalized) | strip input | heatmap | `min_distance` | MP |
|---|---|---|---|---|---|
| 1.5x-wide (tested first) | 0.3750–0.6562 | 864×6528 | 216×1632 | 15 | 5.64 |
| 1.5x-narrow | 0.4062–0.6250 | 672×6528 | 168×1632 | 15 | 4.39 |
| 2x-wide | 0.3984–0.6562 | 1056×8704 | 264×2176 | 20 | 9.19 |

**Rules, as the plan states them and as the script reads them:**

- **Context, S7.** The reference is every full-pass peak at 0.30 or more in the core. A peak counts as
  reproduced when a strip core peak lies within the match radius (1:1, seam-wrapped). The rule needs
  ≥ 95% reproduced and |median score shift (strip − full)| < 0.05. A stricter reading also requires
  the strip peak to reach 0.30; it is reported, but it is not the rule.
- **Calibration, sao_paulo.** 1× and strip peaks in the core are paired 1:1, closest first, within
  the radius; a pair counts when either score reaches 0.30. A band threshold is set if |median shift|
  > 0.03, or if strip and 1× in-core precision at 0.30 differ by more than 0.05. When set, it is the
  lowest 0.01-grid threshold at which the strip's in-core precision reaches the 1× in-core precision
  at 0.30. If no grid threshold reaches it, the config cannot be calibrated and its gate is FAIL.
- **Gate, manual_gold (binding for 1.5x-wide).** Three conditions, each on a paired bootstrap
  interval: the recall gain is above zero; band minus matched-recall threshold F1 is above zero (an
  unreachable matched threshold counts as not beaten, as in #197); F1 against the 1× pass is not
  wholly below zero. All three must hold under both bounds: unmatched detections beyond 25 m (or at or
  above the horizon) scored as FPs, and dropped. The same bound applies to every arm being compared.
  Fail: the band is dropped for the shipped weights and kept only as a read for arm B.
- **Open rig, juneau + bayonne pooled.** The recall gain must be at least +0.02 with an interval
  above zero; otherwise the band is not offered for open sources.
- **Cost.** The GPU-side multiple is (1× + strip) / 1×, pooled over every 0c cache. The limit is 1.75×
  (Jon's to change); `report` names it in the decision line and does not enforce it. The 2× strip
  also needs the limit lifted and must gain at least 1.25 times the 1.5× strip's ramps on
  manual_gold.

**Held-out protocol in code.** `simulate` reads no held-out split; no upscaled cache exists for them.
`extract` refuses a held-out split in each of these cases:

- `--heldout-read-once` is missing;
- it is mixed with selection splits;
- `--limit` or `--force` is given;
- `calibration.json` is absent or from a `--smoke` calibrate;
- the context rule failed for 1.5x-wide (`--despite-failed-context` overrides; Jon's call);
- its cache already exists.

`report` refuses a held-out cache whose recorded calibration sha256 differs from the current
`calibration.json`. `tests/test_horizon_band_263.py` covers each refusal. It also runs calibrate and
report end to end on synthetic caches with a known verdict, and `run0c` on a stand-in for `extract`
(the stop on a failed context rule, and a resume after a held-out split died).

**What was verified locally, and what was not.** The extract path ran end to end on one pano each of
richmond and sao_paulo, into a scratch cache root that is not committed. On that one richmond pano,
every full-pass core peak at 0.30 was reproduced by the strip at the same heatmap pixel (median
offset 0 px). That shows the geometry is right; it is not a result. The run used the local RTX 3070,
not the CPU as intended (see Deviations). Not verified: A40 time and memory, and every 0c number.

## Amendments (choices the plan left open, made before any 0c data exists)

1. **0a pool.** The plan names the 9-split pool. juneau is held out until 0c, so it was not read; the
   pool is the other eight. The plan's own 9-split arm C figure (25 of 146) is quoted. Both pools give
   the same verdicts, and arm A's count can only grow with juneau.
2. **0a metric.** "Under 8 px" is Chebyshev in `miss_taxonomy.py`. The plan does not name a metric for
   "12 px or more apart", so both rules were evaluated under both metrics. They agree.
3. **0b misses.** The plan's range and split counts used an approximate test (any 1× peak within the
   radius). The doc's tables use the scorer's 1:1 match. The approximate figures are reproduced
   exactly in `simulate.md`.
4. **Strip size.** The plan's "rows 0.40–0.65 upsampled to 768×6144" is not on the stride-32 grid. A
   window off the grid changes the backbone's sampling phase, which would confound the context rule
   with a shift effect. The script uses the smallest grid-aligned window that keeps at least 0.05 of
   context: 864×6528 (5.64 MP, against the plan's 5.0 MP) for 1.5x-wide. That raises the plan's
   pixel-linear cost estimate from 1.6× to about 1.67×, still under 1.75×; 0c measures the real time.
   Grid alignment removes the sampling-phase effect only. (Corrected after review 1: this item first
   said that an aligned strip's interior features equal the full pass's, so the context rule
   measures context alone. That is not right for this backbone.) Each of the 36 blocks of
   `KeypointModel`'s ConvNeXtV2 backbone has a GlobalResponseNorm layer (`timm/layers/grn.py`), which
   takes each channel's L2 norm over the whole spatial map (`x.norm(p=2, dim=spatial)`) and scales the
   channel by that norm relative to the mean over channels. A strip has different norms from the
   whole image however much context it carries. So the context rule measures context plus this
   statistic, and adding context rows would not necessarily cure a shortfall. The rule is still the
   right test of strip against full pass; only the reading of a failure changes. The size of the
   effect on the shipped weights is unmeasured: it needs the checkpoint and a pano, and the one-pano
   smoke test above reports peak positions, not score shifts.
5. **Context-rule reading.** Reference peaks at 0.30, reproduced at the score floor, |median shift|.
   The stricter reading (strip peak also at 0.30) is reported beside it. If the two readings
   disagree, 0c reports both, and the first binds (Jon, 2026-11-25; see "Decided before any 0c data
   exists" below).
6. **Calibration details.** Symmetric closest-first pairing (pairing on the 1× peaks above 0.30 alone
   would bias the shift downward); precision on in-core peaks of both passes; the threshold rule
   above.
7. **Far bound.** "Beyond 25 m" includes detections at or above the horizon (no ground range). The
   ignore bound drops only unmatched detections, so every TP and the recall are unchanged; the
   matched-recall threshold therefore does not depend on the bound.
8. **Stop before the held-out read.** The plan does not say what happens when the context rule
   fails. `run0c` stops before reading manual_gold, juneau and bayonne, so that their one read is not
   spent on a strip known to be unfaithful.
9. **0b pooling and LOSO criterion.** Micro-pooled counts, as the plan's 0b table; LOSO on pooled F1
   at 0.30 under the standard (FP) scorer, ties to wide.
10. **Seam.** The unwrapped full-pass heatmap can report a blob's tail at column 0 as an extra border
    maximum (`exclude_border=False`); the wrapped strip does not. It touches only peaks within a few
    px of the seam (about 0.2% of peaks, `seam_nms_248.md`), and it shows up, if at all, as a context
    shortfall.
11. **No 1× in-core detection on sao_paulo** (added after review 1). The calibration rule sets the
    band threshold to equal the 1× in-core precision at 0.30, which does not exist when sao_paulo has
    no 1× in-core detection at 0.30. `calibrate` then refuses, so the held-out splits stay unread
    until someone looks. It used to return the grid floor, 0.05.

## Deviations

- **The local GPU was used for about 3.5 minutes.** The extract smoke test was meant to run on the
  CPU via `CUDA_VISIBLE_DEVICES=""`. On Windows that empty value did not hide the GPU, so the test ran
  on the shared RTX 3070. Its peak memory (15.7 GiB for the full 2× pass) spilled into shared system
  memory. `nvidia-smi` had not been checked first. The GPU showed 1% use and no other compute process
  afterwards. The run used `--usage-log none`, so it has no ledger row, and nothing from it is
  committed. `extract --device cpu` now exists for smoke tests on a shared machine.

## Where each number lives

| number | file | key |
|---|---|---|
| 0a pairs, under 8 px, at 12 px or more, triggers | `analysis_out/horizon_band_263/merged_pairs.json` | `pool_8.triggers.{chebyshev,euclidean}` |
| 0a S7 context and cross-check | same | `context_S7`, `cross_check` |
| 0a per-pair separations and histogram | same | `pairs`, `pool_8.histogram` |
| 0b S7 and context pools, both bounds | `analysis_out/horizon_band_263/simulate.json` | `pools.<pool>.configs.<cfg>.{fp,ignore}` |
| 0b 1× base | same | `pools.<pool>.base.{fp,ignore}` |
| 0b LOSO choices | same | `loso.<scale>` |
| 0b narrow minus wide, S7 pooled | same | `narrow_minus_wide_S7.<scale>.{fp,ignore}` |
| 0b misses by range | same | `misses_by_range.<scale>` |
| 0b per split, laurens_mapillary | same | `per_split`, `report_only` |
| plan's approximate counts | same | `scratch_reproduction` |
| 0c strip geometry | same | `protocol.strip_geometry_for_0c` |
| figure data (threshold frontier) | same | `threshold_frontier_S7` |

Every results table in this doc is pasted from `merged_pairs.md` or `simulate.md`. `simulate.md`
prints its intervals to three decimals from values `simulate.json` has already rounded to four, so
they are rounded twice: 10 of the 144 intervals in its pool tables have one number 0.001 off the
once-rounded value. Four of the ten are quoted in this doc: the S7 ignore-bound ΔF1 of 2x-wide,
+0.006 (+0.007 once rounded; quoted twice), and of 1.5x-narrow, +0.013 (+0.012); 1.5x-wide's
ignore-bound band minus matched, +0.051 (+0.050); and the upper bound of GSV3 2x-wide's ΔF1, +0.021
(+0.020). No bound changes sign and no verdict or sentence reads differently. The narrow-minus-wide
table is stored at six decimals and rounded once.

## What the plan says happens next

**0c has not been run.** Jon decided that this PR merges as Stage 0 with 0c not run. The plan's
section 9 scheduled 0c for 11-16 to 11-18 and the gate for 11-19; those dates have passed, and 0c now
runs on makelab2 only when Jon approves and starts it. Its manual_gold gate is the binding read for
the band; 0b binds nothing. For the
arms, rung 1 runs F (LR 3e-6, the control) first, then B (zoom), then A, which 0a now permits. C stays
unscheduled because 0a did not trigger it. Nothing past Stage 0 has been run.

**Decided before any 0c data exists (Jon, 2026-11-25): the context rule binds on the plan's
reading, not the strict one.** A full-pass peak counts as reproduced when a strip core peak lies
within the radius at any score at or above the floor, with |median shift| < 0.05; the strict
reading (strip peak also at 0.30) is still reported beside it, as the script already does. Reasons:
the median-shift clause already bounds a systematic score drop; calibration on sao_paulo exists to
move the band's threshold when strip scores shift, so 0.30 is not the right bar for strip scores;
and a strip that loses ramps fails the binding manual_gold gate as a fail, never as a false win.
The cost of this reading is that a strip which then fails could spend the held-out read; the cost
of the strict one would be stopping a usable band before calibration can correct its scores. No
code change follows from this decision.

**Open before 0c runs (to be fixed in a follow-up PR before 0c can run; none is done in this PR).**

- The calibration refusal also fires under `calibrate --smoke` (step 2 above). The fix: in smoke
  mode, record a null band threshold instead of refusing.
- A hand-run `extract --splits juneau,bayonne --heldout-read-once` does not compare the calibration
  sha256 recorded by the other held-out splits' caches with the current `calibration.json`; only
  `run0c` does. `report` refuses such a mix, so no mixed result can come out, but the one read of
  juneau and bayonne would be spent. The fix: move `run0c`'s sha256 comparison into `guard_extract`.

## Reproduction

```bash
pip install -e . && pip install -r requirements-dev.txt

# 0a: histogram + triggers from analysis_out/op_cache (~1 s); --check writes nothing
python scripts/analysis/merged_separation_263.py
python scripts/analysis/merged_separation_263.py --check

# 0b: simulation from the committed #196 caches (~60 s, CPU); --check writes nothing
python scripts/analysis/horizon_band_263.py simulate
python scripts/analysis/horizon_band_263.py simulate --check
python scripts/analysis/horizon_band_263.py figures

# tests (~80 s, most of it the two drift guards)
python -m pytest -q tests/test_merged_separation_263.py tests/test_horizon_band_263.py

# both checks through the gate
python scripts/check_all.py --only merged_separation_263 --only horizon_band_263
```

Inputs: `analysis_out/op_cache/{richmond,annapolis,clovis,morgantown,bend,paterson,gainesville,laurens_mapillary}.json`
for 0a, and `analysis_out/input_res_sweep_25/cache/{r2048,r3072,u4096}/<S7 split and laurens_mapillary>.json`
for 0b. Their sha256 values are recorded in each output's `protocol.inputs`, and the #196 caches are
also covered by that directory's `SHA256SUMS` (`python scripts/analysis/input_res_sweep_25.py sums`).
The code is branch `analysis/horizon-band-263` as merged by
[PR #295](https://github.com/ProjectSidewalk/RampNet/pull/295);
`git log -- docs/horizon_band_263.md` lists its commits, including the review-1 fixes.

## Appendix A. The plan's Stage 0 rules, verbatim

From "Plan: detector Phase 1 arms and an upscaled horizon-band pass (2026-11-15, revised after
review)", `detector-phase1-horizon-2x-2026-11-15.md`, which is not in the repository as that file (it is
committed, with corrections, as `detector_phase1_263.md`). Sections 3, 4
and 7 are copied unchanged below; they hold every Stage 0 rule this doc applies (the 0a triggers,
the 0b brief, the 0c rules and gate, and the cost limit). Where the code departs from them, the
Amendments section above says so. The plan's "Scratch" numbers came from an uncommitted script; 0b
reproduces them.

> ## 3. Split roles
>
> | | selection | held out, read once |
> |---|---|---|
> | band | S7 = richmond, annapolis, clovis, morgantown, bend, paterson, gainesville (already read at 2x); sao_paulo for thresholds | manual_gold, juneau, bayonne (never run upscaled) |
> | arms | sao_paulo (threshold, checkpoint); manual_gold (rung 1 gate) | macro US7; laurens_mapillary, juneau, bayonne, budapest_district5; laurens_gsv |
>
> laurens_mapillary was read at 2x in #197 and again in my 11-15 scratch run. It enters no band rule and is reported only.
>
> ## 4. Stage 0 (no training)
>
> **0a. Merged pairs (CPU).** `miss_taxonomy.py` gives 146 pairs on the 9-split pool, 25 (17%) under 8 px. Commit the per-pair histogram. More than one third under 8 px triggers arm C (not met now). Arm A runs only if at least 30 pairs outside paterson are 12 px or more apart.
>
> **0b. Band simulation (CPU, exploratory, binds nothing).** Scratch, S7, 0.30, base F1 0.8417:
>
> | pass, band (y) | ramps gained | FPs added | FPs for a threshold with the same recall |
> |---|---:|---:|---:|
> | 2x, 0.45-0.60 | 100 | 174 | 357 |
> | 2x, 0.47-0.56 | 89 | 146 | 296 |
> | 1.5x, 0.45-0.60 | 106 | 146 | 402 |
> | 1.5x, 0.47-0.56 | 95 | 120 | 327 |
>
> S7 misses by flat-ground range (n; inside 0.45-0.60; recovered by 2x): 0-8 m 43; 1; 1. 8-18 m 87; 87; 18. 18-25 m 51; 51; 13. Beyond 25 m 112; 112; 59. The 1.5x cache is real pixels; production must upsample, which is untested at 1.5x. Commit this with intervals and a leave-one-split-out choice between the two cores.
>
> **What it shows about imagery.** Recovery by split: richmond 18 of 42 misses and annapolis 13 of 51 (Mapillary), bend 14 of 42 and gainesville 17 of 57 (GSV). The four Mapillary splits gain 37 ramps for 107 FPs at 2x; the three GSV splits 63 for 67. laurens_mapillary recovers 5 of 106, but only 9 of its 249 GT ramps are beyond 25 m, which may be GT anchoring. So: the band works where far-field GT exists, at a worse FP cost on Mapillary; GoPro Max and Panoramax are unmeasured.
>
> **0c. Real strips, shipped weights (A40, about 4 GPU-h). The binding gate.** Tested first: **1.5x, core 0.45-0.60 plus 0.05 context**: rows 0.40-0.65 of the production image upsampled to 768x6144, plus 192 px circular wrap each side, head at input/4; core peaks only; merged by #197's seam-wrapped `dedupe`. Also run: the narrow core (0.47-0.56) and the 2x strip. Rules:
> - Context: on S7, the strip reproduces at least 95% of in-band peaks from a full-pano pass at the same scale, same input/4 head and scaled `min_distance`; median score shift under 0.05.
> - Calibration: on sao_paulo, a band threshold is set (to equal 1x precision) if the median paired score shift exceeds 0.03 or precision at 0.30 differs by more than 0.05.
> - Gate, manual_gold: recall interval above zero, paired F1 interval above the matched-recall threshold baseline, F1 against base not wholly below zero. Unmatched detections beyond 25 m are scored twice, as FPs and as ignore points; the gate must hold under both.
> - Open-rig claim: juneau plus bayonne recall gain at least +0.02 with interval above zero; otherwise the band is not offered for open sources.
>
> Fail the gate: the band is dropped for the shipped weights and kept only as a read for arm B.
>
> ## 7. Production cost of the band
>
> Base 0.61 s/pano. Pixel-linear estimates: 1.5x wide core 5.0 MP, +0.35 s, **1.6x**; 1.5x narrow core 3.8 MP, 1.4x; 2x wide core 8.9 MP, 2.1x. 0c measures time and peak memory. Limit: 1.75x GPU-side, because the forward pass is the serialized step on GPU-bound GSV runs; rate-limited open sources are download-bound. The limit is Jon's to change; the 2x strip needs it lifted and must gain at least 1.25 times the 1.5x strip's ramps on manual_gold.
