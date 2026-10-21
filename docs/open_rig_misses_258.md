# Where RampNet's misses live on the open-imagery rigs (#263 Track 1, #258)

Issue: [#263](https://github.com/ProjectSidewalk/RampNet/issues/263) (Track 1), background in
[#258](https://github.com/ProjectSidewalk/RampNet/issues/258) and
[#46](https://github.com/ProjectSidewalk/RampNet/issues/46). Run 2026-10-20, revised the same day
after review. Every number in it is CPU only, with no paid model. The one GPU step, the
laurens_gsv extract, ran but failed its parity gate, so laurens_gsv is not in the tables (see
[Gaps](#gaps)).

RampNet loses about a third of its recall on the open rigs Project Sidewalk deployments are
moving to (#151, #159). This doc asks what kind of miss that is, using the #46 miss taxonomy
(`scripts/analysis/miss_taxonomy.py`) on the open-imagery splits.

## The buckets

Each missed ramp goes into one of four buckets, from the committed 0.05-floor peak caches. R is
the match radius.

- **merged**: a peak at or above threshold sat inside R, but a neighbouring ramp claimed it. One
  peak for two ramps.
- **sub_threshold**: a peak sat inside R, scoring between 0.05 and the threshold. The model found
  the ramp and was not confident enough.
- **localization**: no peak inside R, and the nearest at-or-above-threshold peak between R and 2R
  is unclaimed.
- **silent**: everything else. That means no floor peak inside R, and no unclaimed
  at-or-above-threshold peak within 2R. It does **not** mean the model gave nothing: a silent
  ramp can still have a sub-threshold peak between R and 2R, or a neighbour's peak there.

Because "silent" is looser than it sounds, this doc adds a stricter count beside it.
**strict silent** means no floor peak at all (score 0.05 or more) within 2R. It is a subset of
silent, computed in `open_rig_misses_258.py`; `classify_miss` is unchanged.

## The reading rule, and when its test was chosen

#263 fixed the rule before this work started: **if silent dominates, the gap is labels** (the
model does not respond, so it has to be taught). **If sub_threshold dominates, the gap is
calibration**, and the cheap verifier rungs in #258 come before any retrain.

#263 did not say what "dominates" means. This PR makes it concrete: the pano-bootstrap 95%
interval of (sub_threshold share minus silent share) lies strictly above or strictly below zero,
otherwise neither leads. A bound of exactly zero counts as including zero, and the tables print it
unsigned, as 0.0. The
bootstrap resamples panos within each split, because misses cluster by pano (a merged pair is two
misses on one pano), so it is wider than the Wilson interval, which treats misses as independent.
Both are reported.

When that test was chosen: I chose it while writing `open_rig_misses_258.py`. By then I had
already seen the per-split 0.30 bucket shares for these four splits, because the default
`miss_taxonomy.py` run (used to check that `--cities` left the default output unchanged) prints
every split. I had not seen any interval, any pooled open-rig number or any recall-point
decomposition. A plurality reading of "dominates" would give the same answer on the primary pool
at 0.30 (52.6% against 38.5%).

## What each reading uses

- The **headline** reads the GoPro Max pool at 0.30 on all panos: bucket shares for the rule,
  recall points for the size of each lever.
- `benchmark/README.md` names the **unbiased subset** (random and empty strata, dropping the 5
  hand-picked high-density panos per split) as the honest between-city comparison. Every
  between-city table below is therefore repeated on the unbiased subset. No reading changes
  between the two.
- **budapest_district5 is never pooled.** It is GoPro Max, but `miss_decomposition.HELD_OUT`
  holds it out for single-rater GT at low reviewer confidence, so it gets its own row. An
  earlier draft of this doc pooled it with the other two GoPro Max splits, and the pool then
  read "neither leads". That is why it is held out here, and why that reading is withdrawn.

## Result

At the 0.30 operating point (#79), **sub_threshold leads on the GoPro Max pool** (laurens_mapillary
and bayonne). It is 52.6% of misses against 38.5% silent, and the bootstrap difference is +2.1
to +25.2 points. The unbiased subset gives +1.9 to +25.4.

Split by split, only laurens_mapillary passes the rule on its own (calibration). Bayonne is close
(47.4% against 46.1%) and underpowered. Budapest (held out) leans silent and juneau leans
sub_threshold, and neither of those is separated.

**In recall points, the two levers are both large, and this data cannot say which is larger.**
The GoPro Max pool is 0.284 recall points behind the #46 reference at 0.30 (bootstrap 0.229 to
0.333):

- Sub-threshold misses cost +0.178 more than on the reference.
- Silent misses cost +0.127 more.
- The difference between those two excesses is +0.051, with an interval of −0.008 to +0.109.
  On the unbiased subset it is +0.051 (−0.016 to +0.114).

**What rescoring could reach at most.** A verifier that rescores RampNet's own sub-threshold
peaks can recover at most all of them. On the GoPro Max pool that is 0.258 recall points at 0.30,
of which 0.178 is the excess over the reference (the reference has its own 0.081).

**What labels might reach.** The strict silent misses have no peak of any kind within 2R. They
cost 0.126 recall points on the GoPro Max pool against 0.026 on the reference, an excess of
+0.100 (0.069 to 0.131). That is the population with the best claim to being a label or data
problem, and it is still an upper bound on single-rater GT (see Caveats).

**Merged is not part of the open-rig story.** Against the full reference, merged misses look
like they go the other way (−0.033). That is Paterson: it holds 80 of the reference's 124 merged
misses. Against the reference without Paterson, the GoPro Max merged excess is +0.001.

At the 0.55 deployed threshold, sub_threshold leads on every population except budapest. Here
the excess difference separates: +0.096 (+0.028 to +0.166) on the GoPro Max pool. Most of that is
the 0.30 to 0.55 band that #79 already proposes to give up, so it adds little.

So #263's decision point gets a lean, not a verdict. On shares, calibration leads on the GoPro
Max pool. In recall points both levers are large: about 0.18 excess for sub_threshold and 0.13
for silent, 0.10 of it strictly silent. The interval does not separate them. Which to do first
is Jon's call.

### Bucket shares at 0.30, all panos (Wilson 95% in brackets)

<!-- BEGIN GENERATED: shares-0.30-all -->
| population | panos | GT | misses | recall | merged | sub_threshold | localization | silent | of which strict | sub − silent, boot 95% (pts) | reading |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| `laurens_mapillary` | 59 | 249 | 118 | 0.526 | 12 · 10.2% [6–17] | 70 · 59.3% [50–68] | 2 · 1.7% [0–6] | 34 · 28.8% [21–38] | 23 · 19.5% [13–28] | +12.6 to +48.0 | sub_threshold leads: calibration |
| `bayonne` | 87 | 301 | 152 | 0.495 | 3 · 2.0% [1–6] | 72 · 47.4% [40–55] | 7 · 4.6% [2–9] | 70 · 46.1% [38–54] | 46 · 30.3% [24–38] | −13.6 to +14.9 | neither leads |
| `budapest_district5` (held out, not pooled) | 89 | 300 | 107 | 0.643 | 14 · 13.1% [8–21] | 35 · 32.7% [25–42] | 11 · 10.3% [6–17] | 47 · 43.9% [35–53] | 25 · 23.4% [16–32] | −28.2 to +5.8 | neither leads |
| `juneau` | 94 | 297 | 72 | 0.758 | 10 · 13.9% [8–24] | 36 · 50.0% [39–61] | 2 · 2.8% [1–10] | 24 · 33.3% [24–45] | 12 · 16.7% [10–27] | −4.2 to +35.6 | neither leads |
| **GoPro Max pool** (laurens_mapillary + bayonne) | 146 | 550 | 270 | 0.509 | 15 · 5.6% [3–9] | 142 · 52.6% [47–58] | 9 · 3.3% [2–6] | 104 · 38.5% [33–44] | 69 · 25.6% [21–31] | +2.1 to +25.2 | sub_threshold leads: calibration |
| `morgantown` (GoPro Max, in the reference) | 95 | 267 | 52 | 0.805 | 9 · 17.3% [9–30] | 22 · 42.3% [30–56] | 1 · 1.9% [0–10] | 20 · 38.5% [26–52] | 13 · 25.0% [15–38] | −17.6 to +26.5 | neither leads |
| #46 reference without paterson (6 splits) | 559 | 1665 | 316 | 0.810 | 44 · 13.9% [11–18] | 150 · 47.5% [42–53] | 8 · 2.5% [1–5] | 114 · 36.1% [31–42] | 50 · 15.8% [12–20] | +0.9 to +21.6 | sub_threshold leads: calibration |
| **#46 reference** (7 US splits) | 657 | 2060 | 427 | 0.793 | 124 · 29.0% [25–34] | 166 · 38.9% [34–44] | 9 · 2.1% [1–4] | 128 · 30.0% [26–34] | 53 · 12.4% [10–16] | +0.9 to +17.4 | sub_threshold leads: calibration |
| `laurens_gsv` | — | — | — | — | not run: no op_cache (see Gaps) | | | | | | |
<!-- END GENERATED: shares-0.30-all -->

The reference row is the #46 table (29.0 / 38.9 / 2.1 / 30.0). `miss_taxonomy.py --cities`
reproduces it exactly from the seven original US splits, and `tests/test_open_rig_misses_258.py`
pins that.

It is often called the GSV pool, but it is not one. Three of the seven are GSV (bend, paterson,
gainesville) and four are Mapillary (richmond, clovis, morgantown, annapolis). "panos" counts
panos with at least one GT ramp, the population the taxonomy scores; laurens_mapillary has 94
panos, 59 of them with a ramp.

### Recall points per bucket at 0.30, all panos, with the excess over the #46 reference

Each cell is the bucket's recall points, then the excess over the reference and its pano-bootstrap
95% interval, all in recall points (fractions of GT).

Morgantown's excess interval is conservative. Morgantown is one of the seven reference splits,
and the bootstrap resamples it independently of the reference that contains it, so it ignores
their positive covariance.

<!-- BEGIN GENERATED: excess-0.30-all -->
| population | merged | sub_threshold | localization | silent | strict silent | recall gap | sub excess − silent excess |
|---|--:|--:|--:|--:|--:|--:|--:|
| `laurens_mapillary` | 0.048 (−0.012; −0.044 to +0.025) | 0.281 (+0.201; +0.137 to +0.260) | 0.008 (+0.004; −0.005 to +0.017) | 0.137 (+0.074; +0.028 to +0.121) | 0.092 (+0.067; +0.028 to +0.105) | +0.267 (+0.195 to +0.334) | +0.126 (+0.038 to +0.214) |
| `bayonne` | 0.010 (−0.050; −0.066 to −0.034) | 0.239 (+0.159; +0.104 to +0.213) | 0.023 (+0.019; +0.002 to +0.040) | 0.233 (+0.171; +0.119 to +0.225) | 0.153 (+0.127; +0.081 to +0.176) | +0.298 (+0.224 to +0.365) | −0.012 (−0.092 to +0.062) |
| `budapest_district5` (held out, not pooled) | 0.047 (−0.013; −0.044 to +0.021) | 0.117 (+0.036; −0.005 to +0.079) | 0.037 (+0.032; +0.009 to +0.060) | 0.157 (+0.095; +0.048 to +0.144) | 0.083 (+0.058; +0.019 to +0.102) | +0.149 (+0.078 to +0.216) | −0.059 (−0.123 to +0.004) |
| `juneau` | 0.034 (−0.026; −0.049 to −0.003) | 0.121 (+0.041; −0.005 to +0.083) | 0.007 (+0.002; −0.006 to +0.013) | 0.081 (+0.019; −0.016 to +0.057) | 0.040 (+0.015; −0.010 to +0.044) | +0.035 (−0.029 to +0.099) | +0.022 (−0.031 to +0.070) |
| **GoPro Max pool** (laurens_mapillary + bayonne) | 0.027 (−0.033; −0.052 to −0.013) | 0.258 (+0.178; +0.136 to +0.220) | 0.016 (+0.012; +0.002 to +0.025) | 0.189 (+0.127; +0.090 to +0.165) | 0.126 (+0.100; +0.069 to +0.131) | +0.284 (+0.229 to +0.333) | +0.051 (−0.008 to +0.109) |
| `morgantown` (GoPro Max, in the reference) | 0.034 (−0.026; −0.050 to +0.000) | 0.082 (+0.002; −0.030 to +0.038) | 0.004 (−0.001; −0.007 to +0.008) | 0.075 (+0.013; −0.021 to +0.048) | 0.049 (+0.023; −0.002 to +0.050) | −0.013 (−0.070 to +0.046) | −0.011 (−0.054 to +0.034) |
| #46 reference without paterson (6 splits) | 0.026 | 0.090 | 0.005 | 0.069 | 0.030 | — | — |
| **#46 reference** (7 US splits) | 0.060 | 0.081 | 0.004 | 0.062 | 0.026 | — | — |
<!-- END GENERATED: excess-0.30-all -->

The same, against the reference without Paterson (the merged column is the one that moves):

<!-- BEGIN GENERATED: excess-0.30-all-no-paterson -->
| population | merged | sub_threshold | localization | silent | strict silent | recall gap | sub excess − silent excess |
|---|--:|--:|--:|--:|--:|--:|--:|
| `laurens_mapillary` | 0.048 (+0.022; −0.008 to +0.057) | 0.281 (+0.191; +0.129 to +0.250) | 0.008 (+0.003; −0.006 to +0.016) | 0.137 (+0.068; +0.022 to +0.115) | 0.092 (+0.062; +0.024 to +0.101) | +0.284 (+0.212 to +0.351) | +0.123 (+0.038 to +0.207) |
| `bayonne` | 0.010 (−0.016; −0.029 to −0.002) | 0.239 (+0.149; +0.092 to +0.204) | 0.023 (+0.018; +0.002 to +0.038) | 0.233 (+0.164; +0.111 to +0.220) | 0.153 (+0.123; +0.076 to +0.172) | +0.315 (+0.233 to +0.389) | −0.015 (−0.095 to +0.063) |
| `budapest_district5` (held out, not pooled) | 0.047 (+0.020; −0.009 to +0.053) | 0.117 (+0.027; −0.017 to +0.071) | 0.037 (+0.032; +0.009 to +0.058) | 0.157 (+0.088; +0.041 to +0.137) | 0.083 (+0.053; +0.016 to +0.097) | +0.167 (+0.094 to +0.236) | −0.062 (−0.130 to +0.006) |
| `juneau` | 0.034 (+0.007; −0.013 to +0.031) | 0.121 (+0.031; −0.013 to +0.076) | 0.007 (+0.002; −0.006 to +0.013) | 0.081 (+0.012; −0.023 to +0.051) | 0.040 (+0.010; −0.014 to +0.040) | +0.053 (−0.016 to +0.117) | +0.019 (−0.033 to +0.068) |
| **GoPro Max pool** (laurens_mapillary + bayonne) | 0.027 (+0.001; −0.016 to +0.020) | 0.258 (+0.168; +0.127 to +0.209) | 0.016 (+0.012; +0.001 to +0.024) | 0.189 (+0.121; +0.082 to +0.157) | 0.126 (+0.096; +0.065 to +0.128) | +0.301 (+0.246 to +0.355) | +0.048 (−0.009 to +0.104) |
| `morgantown` (GoPro Max, in the reference) | 0.034 (+0.007; −0.014 to +0.034) | 0.082 (−0.008; −0.043 to +0.027) | 0.004 (−0.001; −0.008 to +0.008) | 0.075 (+0.006; −0.026 to +0.041) | 0.049 (+0.019; −0.007 to +0.045) | +0.005 (−0.054 to +0.062) | −0.014 (−0.061 to +0.029) |
| #46 reference without paterson (6 splits) | 0.026 | 0.090 | 0.005 | 0.069 | 0.030 | — | — |
| **#46 reference** (7 US splits) | 0.060 | 0.081 | 0.004 | 0.062 | 0.026 | — | — |
<!-- END GENERATED: excess-0.30-all-no-paterson -->

### The unbiased subset at 0.30 (the between-city comparison)

<!-- BEGIN GENERATED: shares-0.30-unbiased -->
| population | panos | GT | misses | recall | merged | sub_threshold | localization | silent | of which strict | sub − silent, boot 95% (pts) | reading |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| `laurens_mapillary` | 54 | 212 | 110 | 0.481 | 12 · 10.9% [6–18] | 65 · 59.1% [50–68] | 2 · 1.8% [0–6] | 31 · 28.2% [21–37] | 20 · 18.2% [12–26] | +12.6 to +49.5 | sub_threshold leads: calibration |
| `bayonne` | 82 | 261 | 144 | 0.448 | 2 · 1.4% [0–5] | 68 · 47.2% [39–55] | 6 · 4.2% [2–9] | 68 · 47.2% [39–55] | 45 · 31.2% [24–39] | −14.8 to +14.7 | neither leads |
| `budapest_district5` (held out, not pooled) | 84 | 268 | 104 | 0.612 | 14 · 13.5% [8–21] | 33 · 31.7% [24–41] | 11 · 10.6% [6–18] | 46 · 44.2% [35–54] | 24 · 23.1% [16–32] | −30.3 to +4.8 | neither leads |
| `juneau` | 89 | 262 | 67 | 0.744 | 8 · 11.9% [6–22] | 33 · 49.2% [38–61] | 2 · 3.0% [1–10] | 24 · 35.8% [25–48] | 12 · 17.9% [11–29] | −7.6 to +33.3 | neither leads |
| **GoPro Max pool** (laurens_mapillary + bayonne) | 136 | 473 | 254 | 0.463 | 14 · 5.5% [3–9] | 133 · 52.4% [46–58] | 8 · 3.1% [2–6] | 99 · 39.0% [33–45] | 65 · 25.6% [21–31] | +1.9 to +25.4 | sub_threshold leads: calibration |
| `morgantown` (GoPro Max, in the reference) | 90 | 225 | 51 | 0.773 | 9 · 17.6% [10–30] | 21 · 41.2% [29–55] | 1 · 2.0% [0–10] | 20 · 39.2% [27–53] | 13 · 25.5% [16–39] | −18.2 to +23.5 | neither leads |
| #46 reference without paterson (6 splits) | 529 | 1414 | 306 | 0.784 | 40 · 13.1% [10–17] | 146 · 47.7% [42–53] | 8 · 2.6% [1–5] | 112 · 36.6% [31–42] | 50 · 16.3% [13–21] | 0.0 to +22.0 | neither leads |
| **#46 reference** (7 US splits) | 622 | 1768 | 415 | 0.765 | 119 · 28.7% [25–33] | 162 · 39.0% [34–44] | 9 · 2.2% [1–4] | 125 · 30.1% [26–35] | 53 · 12.8% [10–16] | +0.7 to +17.3 | sub_threshold leads: calibration |
| `laurens_gsv` | — | — | — | — | not run: no op_cache (see Gaps) | | | | | | |
<!-- END GENERATED: shares-0.30-unbiased -->

<!-- BEGIN GENERATED: excess-0.30-unbiased -->
| population | merged | sub_threshold | localization | silent | strict silent | recall gap | sub excess − silent excess |
|---|--:|--:|--:|--:|--:|--:|--:|
| `laurens_mapillary` | 0.057 (−0.011; −0.048 to +0.032) | 0.307 (+0.215; +0.143 to +0.279) | 0.009 (+0.004; −0.007 to +0.019) | 0.146 (+0.075; +0.022 to +0.131) | 0.094 (+0.064; +0.023 to +0.111) | +0.284 (+0.217 to +0.348) | +0.140 (+0.033 to +0.233) |
| `bayonne` | 0.008 (−0.060; −0.077 to −0.042) | 0.261 (+0.169; +0.114 to +0.223) | 0.023 (+0.018; 0.000 to +0.042) | 0.261 (+0.190; +0.135 to +0.240) | 0.172 (+0.142; +0.094 to +0.191) | +0.317 (+0.249 to +0.381) | −0.021 (−0.104 to +0.061) |
| `budapest_district5` (held out, not pooled) | 0.052 (−0.015; −0.051 to +0.020) | 0.123 (+0.032; −0.012 to +0.079) | 0.041 (+0.036; +0.010 to +0.067) | 0.172 (+0.101; +0.051 to +0.155) | 0.090 (+0.060; +0.018 to +0.107) | +0.153 (+0.081 to +0.223) | −0.069 (−0.140 to +0.002) |
| `juneau` | 0.030 (−0.037; −0.060 to −0.013) | 0.126 (+0.034; −0.013 to +0.082) | 0.008 (+0.003; −0.006 to +0.015) | 0.092 (+0.021; −0.019 to +0.062) | 0.046 (+0.016; −0.012 to +0.049) | +0.021 (−0.049 to +0.090) | +0.013 (−0.042 to +0.068) |
| **GoPro Max pool** (laurens_mapillary + bayonne) | 0.030 (−0.038; −0.060 to −0.014) | 0.281 (+0.190; +0.148 to +0.233) | 0.017 (+0.012; −0.001 to +0.026) | 0.209 (+0.139; +0.099 to +0.179) | 0.137 (+0.107; +0.074 to +0.142) | +0.302 (+0.253 to +0.355) | +0.051 (−0.016 to +0.114) |
| `morgantown` (GoPro Max, in the reference) | 0.040 (−0.027; −0.056 to +0.004) | 0.093 (+0.002; −0.036 to +0.042) | 0.004 (−0.001; −0.008 to +0.010) | 0.089 (+0.018; −0.019 to +0.057) | 0.058 (+0.028; −0.000 to +0.058) | −0.008 (−0.065 to +0.052) | −0.017 (−0.067 to +0.037) |
| #46 reference without paterson (6 splits) | 0.028 | 0.103 | 0.006 | 0.079 | 0.035 | — | — |
| **#46 reference** (7 US splits) | 0.067 | 0.092 | 0.005 | 0.071 | 0.030 | — | — |
<!-- END GENERATED: excess-0.30-unbiased -->

On the unbiased subset, the reference's own "calibration" verdict is marginal: the lower bound is
+0.7 points on the full reference, and the reference without Paterson reads "neither leads". It
is marginal on all panos too (+0.9).

### Near field and far field at 0.30, all panos (18 m boundary, flat-ground distance)

Strict silent counts are in parentheses.

<!-- BEGIN GENERATED: nearfar-0.30-all -->
| population | near GT | near misses | near recall | near merged / sub / loc / silent (strict) | far GT | far misses | far recall | far merged / sub / loc / silent (strict) |
|---|--:|--:|--:|---|--:|--:|--:|---|
| `laurens_mapillary` | 216 | 98 | 0.546 | 7 / 61 / 2 / 28 (21) | 33 | 20 | 0.394 | 5 / 9 / 0 / 6 (2) |
| `bayonne` | 191 | 93 | 0.513 | 0 / 44 / 6 / 43 (30) | 110 | 59 | 0.464 | 3 / 28 / 1 / 27 (16) |
| `budapest_district5` (held out, not pooled) | 201 | 67 | 0.667 | 6 / 19 / 9 / 33 (19) | 99 | 40 | 0.596 | 8 / 16 / 2 / 14 (6) |
| `juneau` | 190 | 27 | 0.858 | 0 / 17 / 2 / 8 (6) | 107 | 45 | 0.579 | 10 / 19 / 0 / 16 (6) |
| **GoPro Max pool** (laurens_mapillary + bayonne) | 407 | 191 | 0.531 | 7 / 105 / 8 / 71 (51) | 143 | 79 | 0.448 | 8 / 37 / 1 / 33 (18) |
| `morgantown` (GoPro Max, in the reference) | 216 | 37 | 0.829 | 5 / 17 / 1 / 14 (8) | 51 | 15 | 0.706 | 4 / 5 / 0 / 6 (5) |
| #46 reference without paterson (6 splits) | 1134 | 133 | 0.883 | 13 / 79 / 3 / 38 (15) | 531 | 183 | 0.655 | 31 / 71 / 5 / 76 (35) |
| **#46 reference** (7 US splits) | 1360 | 180 | 0.868 | 48 / 84 / 3 / 45 (16) | 700 | 247 | 0.647 | 76 / 82 / 6 / 83 (37) |
<!-- END GENERATED: nearfar-0.30-all -->

**Read the distances as approximate, on both sides.** The boundary uses the flat-ground estimate
with a 2.5 m camera height (`stage1_label_recall.geom`). Real rigs sit lower on *both* sides of
the comparison: GoPro Max at about 1.8 to 2.1 m, and in the reference paterson and gainesville's
GSV at about 1.8 m, clovis at 2.02 m and annapolis at 2.12 m (`docs/da3_calibration_101.md`). So
true distances are shorter than estimated everywhere. The bias does not establish a direction
for the near/far contrast between the open rigs and the reference.

The open-rig deficit is mostly a near-field deficit. On the reference, most misses are far (247
of 427). On the GoPro Max pool, 191 of 270 are near, where the model has the pixels.

- Strict silent near-field misses are 51 of 407 near ramps on the GoPro Max pool (12.5%), against
  16 of 1,360 on the reference (1.2%). These are ramps the model shows no peak for within 2R.
- The looser silent count is 71 of 407 against 45 of 1,360.

Either is an upper bound on what labels could reach, because the GT is one rater's. Juneau, the
Trimble MX7 city, looks like its sibling annapolis: near recall 0.858, and most misses far (45 of
72).

### Merged pairs, how far below threshold the sub-threshold peaks sit, and the null (0.30, all panos)

<!-- BEGIN GENERATED: detail-0.30-all -->
| population | merged pairs | Chebyshev median px | > min_distance 10 | Euclid ≤ 2σ | < 8 px | sub-threshold peak median [q25, q75] | share ≥ 0.15 | null rate sub / merged |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `laurens_mapillary` | 12 | 12.2 | 10 | 10 | 1 | 0.143 [0.087, 0.206] | 49% | 8.3% / 7.1% |
| `bayonne` | 3 | 13.7 | 2 | 3 | 1 | 0.186 [0.107, 0.253] | 54% | 12.2% / 12.8% |
| `budapest_district5` (held out, not pooled) | 14 | 13.1 | 10 | 13 | 2 | 0.159 [0.118, 0.206] | 60% | 9.7% / 10.1% |
| `juneau` | 10 | 15.0 | 8 | 8 | 0 | 0.180 [0.120, 0.239] | 58% | 10.1% / 11.7% |
| **GoPro Max pool** (laurens_mapillary + bayonne) | 15 | 12.2 | 12 | 13 | 2 | 0.156 [0.097, 0.227] | 51% | 10.3% / 8.2% |
| `morgantown` (GoPro Max, in the reference) | 9 | 10.3 | 5 | 8 | 2 | 0.124 [0.091, 0.163] | 36% | 6.7% / 7.8% |
| #46 reference without paterson (6 splits) | 44 | 15.8 | 35 | 32 | 4 | 0.153 [0.091, 0.228] | 52% | 7.8% / 9.8% |
| **#46 reference** (7 US splits) | 124 | 11.4 | 78 | 108 | 24 | 0.157 [0.094, 0.228] | 53% | 7.6% / 10.1% |
<!-- END GENERATED: detail-0.30-all -->

- **Merged is small on every open split** (2 to 14% of misses). The pairs that do merge look like
  the reference's: most are farther apart than the extractor's `min_distance`, so the heatmap had
  one mode. Nothing here argues for the sigma or head work (#221).
- **The sub-threshold peaks sit as far below threshold as the reference's.** The median is 0.156
  on the GoPro Max pool against 0.157 on the reference, with about half at or above 0.15 on both.
  A verifier tuned on the reference band would see the same score range here.
- **The null** is the chance a peak lands inside R by accident, with the ramp's elevation held and
  its azimuth randomized. For sub_threshold it is 8 to 12% on the open splits and 7.6% on the
  reference. Bayonne's is the highest, so up to about one in eight of its sub-threshold misses
  could be a peak that happened to be nearby.

### At 0.55 (the published operating point)

<!-- BEGIN GENERATED: shares-0.55-all -->
| population | panos | GT | misses | recall | merged | sub_threshold | localization | silent | of which strict | sub − silent, boot 95% (pts) | reading |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| `laurens_mapillary` | 59 | 249 | 152 | 0.390 | 9 · 5.9% [3–11] | 107 · 70.4% [63–77] | 0 · 0.0% [0–2] | 36 · 23.7% [18–31] | 23 · 15.1% [10–22] | +32.5 to +60.5 | sub_threshold leads: calibration |
| `bayonne` | 87 | 301 | 188 | 0.375 | 3 · 1.6% [1–5] | 108 · 57.5% [50–64] | 2 · 1.1% [0–4] | 75 · 39.9% [33–47] | 46 · 24.5% [19–31] | +3.4 to +30.6 | sub_threshold leads: calibration |
| `budapest_district5` (held out, not pooled) | 89 | 300 | 147 | 0.510 | 8 · 5.4% [3–10] | 81 · 55.1% [47–63] | 2 · 1.4% [0–5] | 56 · 38.1% [31–46] | 25 · 17.0% [12–24] | −1.4 to +33.8 | neither leads |
| `juneau` | 94 | 297 | 93 | 0.687 | 9 · 9.7% [5–17] | 58 · 62.4% [52–72] | 0 · 0.0% [0–4] | 26 · 28.0% [20–38] | 12 · 12.9% [8–21] | +16.7 to +51.5 | sub_threshold leads: calibration |
| **GoPro Max pool** (laurens_mapillary + bayonne) | 146 | 550 | 340 | 0.382 | 12 · 3.5% [2–6] | 215 · 63.2% [58–68] | 2 · 0.6% [0–2] | 111 · 32.6% [28–38] | 69 · 20.3% [16–25] | +20.3 to +40.8 | sub_threshold leads: calibration |
| `morgantown` (GoPro Max, in the reference) | 95 | 267 | 72 | 0.730 | 8 · 11.1% [6–20] | 43 · 59.7% [48–70] | 0 · 0.0% [0–5] | 21 · 29.2% [20–41] | 13 · 18.1% [11–28] | +11.9 to +50.8 | sub_threshold leads: calibration |
| #46 reference without paterson (6 splits) | 559 | 1665 | 446 | 0.732 | 34 · 7.6% [6–10] | 290 · 65.0% [60–69] | 2 · 0.4% [0–2] | 120 · 26.9% [23–31] | 50 · 11.2% [9–14] | +29.3 to +46.7 | sub_threshold leads: calibration |
| **#46 reference** (7 US splits) | 657 | 2060 | 572 | 0.722 | 110 · 19.2% [16–23] | 325 · 56.8% [53–61] | 3 · 0.5% [0–2] | 134 · 23.4% [20–27] | 53 · 9.3% [7–12] | +26.0 to +40.9 | sub_threshold leads: calibration |
| `laurens_gsv` | — | — | — | — | not run: no op_cache (see Gaps) | | | | | | |
<!-- END GENERATED: shares-0.55-all -->

<!-- BEGIN GENERATED: excess-0.55-all -->
| population | merged | sub_threshold | localization | silent | strict silent | recall gap | sub excess − silent excess |
|---|--:|--:|--:|--:|--:|--:|--:|
| `laurens_mapillary` | 0.036 (−0.017; −0.044 to +0.014) | 0.430 (+0.272; +0.191 to +0.347) | 0.000 (−0.002; −0.003 to 0.000) | 0.145 (+0.080; +0.033 to +0.124) | 0.092 (+0.067; +0.028 to +0.105) | +0.333 (+0.252 to +0.405) | +0.192 (+0.090 to +0.294) |
| `bayonne` | 0.010 (−0.043; −0.058 to −0.028) | 0.359 (+0.201; +0.139 to +0.258) | 0.007 (+0.005; −0.003 to +0.020) | 0.249 (+0.184; +0.134 to +0.239) | 0.153 (+0.127; +0.081 to +0.176) | +0.347 (+0.277 to +0.414) | +0.017 (−0.078 to +0.104) |
| `budapest_district5` (held out, not pooled) | 0.027 (−0.027; −0.047 to −0.004) | 0.270 (+0.112; +0.047 to +0.175) | 0.007 (+0.005; −0.002 to +0.016) | 0.187 (+0.122; +0.067 to +0.174) | 0.083 (+0.058; +0.019 to +0.102) | +0.212 (+0.129 to +0.286) | −0.009 (−0.101 to +0.078) |
| `juneau` | 0.030 (−0.023; −0.045 to −0.000) | 0.195 (+0.037; −0.012 to +0.090) | 0.000 (−0.002; −0.003 to 0.000) | 0.087 (+0.022; −0.012 to +0.061) | 0.040 (+0.015; −0.010 to +0.044) | +0.035 (−0.034 to +0.104) | +0.015 (−0.045 to +0.075) |
| **GoPro Max pool** (laurens_mapillary + bayonne) | 0.022 (−0.032; −0.048 to −0.014) | 0.391 (+0.233; +0.186 to +0.284) | 0.004 (+0.002; −0.003 to +0.011) | 0.202 (+0.137; +0.100 to +0.173) | 0.126 (+0.100; +0.069 to +0.131) | +0.341 (+0.287 to +0.390) | +0.096 (+0.028 to +0.166) |
| `morgantown` (GoPro Max, in the reference) | 0.030 (−0.023; −0.046 to +0.002) | 0.161 (+0.003; −0.043 to +0.053) | 0.000 (−0.002; −0.003 to 0.000) | 0.079 (+0.014; −0.020 to +0.048) | 0.049 (+0.023; −0.002 to +0.050) | −0.008 (−0.073 to +0.056) | −0.011 (−0.064 to +0.048) |
| #46 reference without paterson (6 splits) | 0.020 | 0.174 | 0.001 | 0.072 | 0.030 | — | — |
| **#46 reference** (7 US splits) | 0.053 | 0.158 | 0.002 | 0.065 | 0.026 | — | — |
<!-- END GENERATED: excess-0.55-all -->

Unbiased subset at 0.55:

<!-- BEGIN GENERATED: shares-0.55-unbiased -->
| population | panos | GT | misses | recall | merged | sub_threshold | localization | silent | of which strict | sub − silent, boot 95% (pts) | reading |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| `laurens_mapillary` | 54 | 212 | 143 | 0.326 | 9 · 6.3% [3–12] | 101 · 70.6% [63–77] | 0 · 0.0% [0–3] | 33 · 23.1% [17–31] | 20 · 14.0% [9–21] | +32.3 to +62.2 | sub_threshold leads: calibration |
| `bayonne` | 82 | 261 | 176 | 0.326 | 2 · 1.1% [0–4] | 100 · 56.8% [49–64] | 2 · 1.1% [0–4] | 72 · 40.9% [34–48] | 45 · 25.6% [20–32] | +1.6 to +30.4 | sub_threshold leads: calibration |
| `budapest_district5` (held out, not pooled) | 84 | 268 | 143 | 0.466 | 8 · 5.6% [3–11] | 78 · 54.5% [46–62] | 2 · 1.4% [0–5] | 55 · 38.5% [31–47] | 24 · 16.8% [12–24] | −2.9 to +32.6 | neither leads |
| `juneau` | 89 | 262 | 88 | 0.664 | 7 · 8.0% [4–16] | 55 · 62.5% [52–72] | 0 · 0.0% [0–4] | 26 · 29.5% [21–40] | 12 · 13.6% [8–22] | +15.2 to +51.4 | sub_threshold leads: calibration |
| **GoPro Max pool** (laurens_mapillary + bayonne) | 136 | 473 | 319 | 0.326 | 11 · 3.5% [2–6] | 201 · 63.0% [58–68] | 2 · 0.6% [0–2] | 105 · 32.9% [28–38] | 65 · 20.4% [16–25] | +20.1 to +40.3 | sub_threshold leads: calibration |
| `morgantown` (GoPro Max, in the reference) | 90 | 225 | 71 | 0.684 | 8 · 11.3% [6–21] | 42 · 59.2% [48–70] | 0 · 0.0% [0–5] | 21 · 29.6% [20–41] | 13 · 18.3% [11–29] | +11.5 to +49.3 | sub_threshold leads: calibration |
| #46 reference without paterson (6 splits) | 529 | 1414 | 433 | 0.694 | 31 · 7.2% [5–10] | 282 · 65.1% [61–69] | 2 · 0.5% [0–2] | 118 · 27.3% [23–32] | 50 · 11.6% [9–15] | +28.3 to +46.7 | sub_threshold leads: calibration |
| **#46 reference** (7 US splits) | 622 | 1768 | 557 | 0.685 | 106 · 19.0% [16–22] | 317 · 56.9% [53–61] | 3 · 0.5% [0–2] | 131 · 23.5% [20–27] | 53 · 9.5% [7–12] | +26.0 to +40.5 | sub_threshold leads: calibration |
| `laurens_gsv` | — | — | — | — | not run: no op_cache (see Gaps) | | | | | | |
<!-- END GENERATED: shares-0.55-unbiased -->

Raising the threshold to 0.55 moves misses into sub_threshold and leaves silent nearly where it
was (104 to 111 on the GoPro Max pool). So at 0.55 the rule reads "calibration" almost
everywhere, and the 0.30 tables are the ones that separate the two levers.

**These recall figures match the scoreboard.** On all four open splits, the taxonomy's hits at
0.55 equal `analysis_out/scoreboard.json`'s RampNet tp exactly (laurens_mapillary 97, bayonne 113,
budapest 153, juneau 204).

`benchmark/README.md` prints budapest 0.503 and bayonne 0.372 instead of 0.510 and 0.375. That
table comes from `score_validation.py`, which counts the detections a reviewer judged correct.
Matching peaks to GT points also credits a peak the reviewer judged wrong or unsure when it lands
within R of a confident missed mark. There is one such detection on bayonne (judged false) and
two on budapest (judged unsure), which are exactly the 1 and 2 ramps of difference.

## Per rig

- **GoPro Max on Mapillary, Laurens (laurens_mapillary).** Calibration. 59% of misses peak below
  0.30, and most of the shortfall against the reference is sub_threshold (+0.201 of +0.267 recall
  points). The model finds these ramps and scores them low. This fits two earlier results:
  - #151, where the rig effect survived pairing on corners both rigs saw (ΔF1 +0.112);
  - #82's frozen-model probe, where pushing GSV panos toward the measured GoPro pixel statistics
    on all axes at once cost 0.209 recall at 0.30 (`all@gopro`, `docs/aug_transfer_82.md`).
- **GoPro Max on Panoramax, Bayonne (bayonne).** Close and underpowered. Sub-threshold and silent
  are 47% and 46% of misses and cost about the same recall (+0.159 and +0.171). 46 of its 70
  silent misses are strict.
  - Bayonne's GT is single-rater at medium confidence (bollards, raised crossings and roundabouts
    fought the rubric), so part of that silent bucket may be marks a second rater would not make.
- **GoPro Max on Mapillary, Budapest (budapest_district5, held out).** Leans silent (44% against
  33%, not separated) and has the highest localization share on the board (10%). Budapest's GT is
  single-rater at low confidence and the rubric does not transfer cleanly, so this is the split to
  trust least, and it is not pooled.
- **Trimble MX7 on Mapillary, Juneau (juneau).** Close to the reference: 0.035 recall points
  behind (−0.029 to +0.099), with most misses far field. Its sibling annapolis (same rig, same
  contributor) is in the reference pool.
- **GSV in the same town (laurens_gsv).** Not run. It is the cleanest control (same town, largely
  the same ramps, rig the only change) and it needs one GPU extract. See [Gaps](#gaps).
- **GoPro Max inside the reference (morgantown).** At reference level: recall 0.805, gap −0.013
  (−0.070 to +0.046). That interval is conservative, because morgantown is part of the
  reference. It suggests the camera model alone does not explain the open-rig
  deficit. It is one split with 52 misses, though, and it differs from bayonne in country and in
  how well the rubric fit, so it is not a controlled comparison. Richmond also has 18 GoPro Max
  panos beside its 64 iSTAR Pulsar.

## Caveats

- **GT is anchored to RampNet.** On every split here, GT is the reviewed RampNet detections
  exported at a 0.55 floor plus the ramps the reviewer marked as missed.
  - So every miss at 0.55 is a ramp the reviewer added.
  - A peak between 0.30 and 0.55 counts as a hit only where the reviewer added a mark there.
  - A ramp that both RampNet and the reviewer missed is in no bucket.

  The bucket mix describes the misses a reviewer noticed.
- **Single rater on every open split** (jonf). Bayonne and juneau are rated medium confidence and
  budapest low; laurens's confidence is unrecorded. No second rater has repeated any of them, and
  a reviewer who marks more distant or more ambiguous ramps moves misses into silent.
- **n is small per split.** There are 72 to 152 misses at 0.30, so a share carries a Wilson
  interval of about ±8 to ±11 points and a wider pano-bootstrap interval. The per-split "neither
  leads" readings are underpowered, not ties.
- **Even strict silent is not proof that the model saw nothing.** On the #46 splits,
  `silent_activation.py` found only 8% of silent misses had a flat heatmap. The rest were the
  tail of an adjacent mode (62%) or a faint response on site (30%) (`docs/rampnet1_findings.md`).
  That breakdown needs the raw heatmaps (GPU) and has not been run on the open splits.
- **Seams are not handled the same way across the populations.**
  - The taxonomy matches without x-wrapping, whereas `score_pano` has wrapped since #140. With
    wrapping, 1 bayonne and 1 juneau miss at 0.30 would be hits.
  - The extractor treats the seam differently too. The seven reference caches and budapest's
    predate the `f4c71c8` border fix and carry no floor peaks within 10 columns of the seam,
    against 4 / 39 / 13 on laurens_mapillary / bayonne / juneau.
  - Only 5 reference GT ramps sit in that strip, all missed (richmond 1 silent, bend 1 silent,
    paterson 1 silent and 2 merged).
  - Bound: about 0.002 recall points on either side. No reading changes.
- **The reference caches are not all at parity with the scoreboard.** On bend, paterson and
  gainesville, the cache's hits at 0.55 fall 2, 2 and 7 short of the records-based scoreboard tp.
  That is the GSV-path drift that `low_floor_sweep.py parity` tolerates. The open splits match
  exactly.

## Gaps

- **laurens_gsv is not bucketed: its extract ran and failed the parity gate.** The extract ran
  on makelab2 (A40) on 2026-10-20 at 9f4b7ce, using `scripts/analysis/open_rig_misses_258_gpu.sh`.
  The script header lists each step and its gate.

  **Panos.** The laurens_gsv panos are not on Hugging Face yet. The copies of record are in
  makelab2's `~/RampNet/benchmark/{laurens_gsv,laurens_mapillary}/panos/`, and each split's
  committed `imagery_manifest.json` pins their bytes. The run's `manifest` step verified them (86
  and 94 panos, `OK`).

  **What the run did, step by step** (log: `analysis_out/open_rig_258/gpu_run.log`):
  - **repro passed within tolerance.** laurens_mapillary re-extracted with all 374 peaks at their
    committed positions. Scores differed by at most 5.4e-5, and no peak crossed 0.30 or 0.55.
  - **extract ran.** 86 panos in 164.6 s.
  - **The usage rows were written before the gate**, so they are in `analysis_out/usage_log.jsonl`.
  - **parity FAILED.** The script stopped there, so the score report did not run.
    - At 0.55 the records hold 121 detections and the cache 111.
    - 74.8% land in the identical heatmap cell and 92.8% within 0.5 R. Every displacement is under
      0.44 R, so the position arm passes and the count arm fails.

  **Why the count arm fails.** The cache sees the same peaks at the same places as the deployed
  run, but scores many of them lower. Of the 18 records detections at or above 0.55 with no
  cache peak at or above 0.55 within 0.5 R, all 18 have a cache peak within 0.44 R (14 in the
  same cell) scoring 0.38 to 0.54. Scored against verdict GT at 0.55 with `wrap_x=True`, the cache gives tp 101 /
  fp 7, against the scoreboard's records-based 112 / 8.

  This is the same gap #151 observed and left uninvestigated: the deployed GSV run beats a
  re-extraction from the native-resolution bundle (`docs/laurens_paired_151.md`, "The deployed
  GSV run and the r2048 re-extraction differ on `laurens_gsv`"). The other GSV caches show it
  only faintly (bend 2, paterson 2 and gainesville 7 TPs short). sao_paulo failed the same count
  arm and was admitted as a ratified exception (`low_floor_sweep.PARITY_EXCEPTIONS`).

  **The candidate cache is kept, not used.** It is committed as
  `analysis_out/open_rig_258/laurens_gsv_candidate_op_cache.json` (sha256 `af95b501…`), and no
  analysis reads it. Whether to admit it as a parity exception is an open decision. The case
  for and against:
  - Its lower scores would push some laurens_gsv misses into sub_threshold that the deployed
    run hit.
  - On the other side, the bundle downsample is the same preprocessing the other open-split
    caches use.

  If it is admitted: copy it to `analysis_out/op_cache/laurens_gsv.json`, add a
  `PARITY_EXCEPTIONS` entry with this diagnosis, and re-run `open_rig_misses_258.py`. That also
  changes `miss_taxonomy.py`'s default output (laurens_gsv stops printing "no cache") and the
  scoreboard's AP-provenance row for laurens_gsv. Those committed outputs then need
  regenerating in the same change.

  **The first launch.** A first launch at d74d680, earlier the same day, also ran to completion,
  but nobody confirmed it at the time. Its outputs were discarded and its log is kept as
  `analysis_out/open_rig_258/gpu_run_d74d680_superseded.log`. Its GPU time (91 s repro and 167 s
  extract, from the log stamps) is recorded as two `SUPERSEDED` rows.
- **The silent-activation breakdown** on the open splits (see Caveats), also GPU.
- **No second rater** on any open split.

## Reproduce

From a clean clone, CPU only (about 30 seconds):

```bash
pip install -e .
# The machine-readable table (analysis_out/open_rig_misses_258.json) and every generated
# table in this doc, then the check that both re-derive byte for byte:
python scripts/analysis/open_rig_misses_258.py
python scripts/analysis/open_rig_misses_258.py --check
# The taxonomy itself on the open splits, at either threshold:
python scripts/analysis/miss_taxonomy.py --cities laurens_mapillary,bayonne,budapest_district5,juneau
python scripts/analysis/miss_taxonomy.py --cities laurens_mapillary,bayonne,budapest_district5,juneau --threshold 0.55
# The #46 reference table, unchanged:
python scripts/analysis/miss_taxonomy.py --cities richmond,bend,clovis,morgantown,annapolis,paterson,gainesville
```

Inputs, all committed:

- `analysis_out/op_cache/<split>.json`: 0.05-floor peaks from the published
  `projectsidewalk/rampnet-model`, with verdict-derived GT.
- `benchmark/<split>/verdicts.json`: the `top` stratum for the unbiased subset.
- `benchmark/<split>/records.jsonl`: camera models.

The bootstraps use 2,000 replicates with seed 20261020, and the null uses 200 trials per miss,
seeded per split. Every table between `BEGIN GENERATED` and `END GENERATED` markers is written by
the script from the JSON, and `--check` fails if one has been edited by hand.

**What `miss_taxonomy.py --cities` changes.** It restricts the run to the named splits and pools
the named ones that are in `US_SPLITS`. With no flag the output is byte-identical to before.
That includes the title "seven US splits", which is stale: laurens_mapillary and juneau have since
joined `US_SPLITS`, so the default pool is nine splits and 2,606 GT. The label is kept so the
default output does not move. `--cities` runs print the real count.

**Cost.** No GPU and no paid model were used for any number in this doc. The laurens_gsv extract
(see Gaps) is not used by any number either, but it spent A40 time. That time is recorded as four
`paid: false` rows in `analysis_out/usage_log.jsonl`:
- this run: `open-rig-258:repro`, 89.5 s, and `open-rig-258:extract`, 164.6 s;
- the superseded d74d680 launch: 91 s and 167 s.
