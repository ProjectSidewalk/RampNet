# VLM prompt experiments, round 1: Gemini and Qwen on the curb-ramp benchmark (#276)

**Status: Step 5 done (2026-11-11), write-up in progress.** Steps 0–5 of the plan are done
(Step 3 on three model families), the Step 2 zero-shot columns are published and on the
scoreboard, and the Step 6 thinking arm has run. Step 4 is a negative result on both models,
and Step 5, the frozen held-out run, finds the **pre-registered primary endpoint null**: the
optimized Flash definition does not beat zero-shot on held-out cities. The one wording that
moves Flash on held-out data is Jon's blind expert definition (an amendment arm, so
exploratory). Nothing moves Argon. Step 7 replicates and the in-context-images control are
pending.
Every dev-split number below regenerates from the detection caches with
`python scripts/analysis/prompt_276_results.py`, every held-out number with
`python scripts/analysis/prompt_276_heldout.py` (`--check` compares with the committed JSON;
`scripts/check_all.py --only prompt_276_results,prompt_276_heldout --allow local-cache` runs both).

## What and why

Every VLM number the org held before this round was zero-shot: one untuned text prompt, one
operating point, no confidence ([#276](https://github.com/ProjectSidewalk/RampNet/issues/276)).
This round asks three questions, in the order prompt-engineering practice suggests:

1. **Current models, same prompt.** How do `gemini-4-argon` and `gemini-3.8-flash` do on the
   as-run prompt, on every split? (Step 2; the scoreboard rows.)
2. **How much is wording?** Repeat the seed prompt (noise floor), try three meaning-preserving
   paraphrases (the "best of K prompts" null), and an expert definition written blind to model
   errors. (Step 3, on the dev split only, on three model families.)
3. **Does an exemplar-driven rewrite of the class definition beat that null?** DetPO-style:
   the model's own worst false positive and worst miss on `dev_train`, each confirmed by a
   human, drive a rewrite; selection on `dev_select`; a frozen held-out run only if the
   dev_select gate passes. (Steps 4–5, *pending*.)

The literature behind the design: naive visual few-shot often hurts chat VLMs, while rewriting
the class description from the worst FP and FN helps a little and reliably (DetPO, 2026);
elicited per-box confidence gives a chat VLM a PR curve; tune on the dev split only, freeze,
then score held-out. Our own prior from #123: thinking/effort is an operating-point dial, not a
quality lever.

Funding: the Google.org Cloud credit (Vertex, project `rampnet`). Spend is reconciled against
Cloud Monitoring after each paid day (`scripts/analysis/vertex_usage.py --days 3 --reconcile`);
billed and logged tokens matched exactly on 2026-11-09.

## The dev split and the rules

`sao_paulo` is the one split tuning may look at (#51 convention). It is cut once, seeded, into
`dev_train` (60 panos) and `dev_select` (65), committed as
`benchmark/prompt_opt_276/sao_paulo_split.json` (`prompt_opt_276_splits.py --check` rebuilds
it). `dev_train` feeds the optimizer its exemplars; `dev_select` picks among candidates, once.
Exemplars come only from the model's own `dev_train` errors: the miss-taxonomy silent gallery
and every other split's FP tags are held-out material and are never used. Each picked exemplar
is shown to a human on a local review page before the rewrite sees it.

The prompt a model receives, once per 1024×1024 perspective view (6 views per pano), is the
instruction with the definition in the middle; `--definition NAME` swaps only the definition
(`benchmark/prompt_opt_276/README.md` has the full text and every definition's provenance).
Temperature stays at the as-run 0.0 for every leg: Step 1 found no observable difference at 1.0
on either model (tp/fp/fn identical on Argon, recall unchanged on Flash), and changing it would
have broken continuity with the published 3.x legs.

## Step 1: diagnostics on 3.8 Flash's low recall (~$1.40)

3.8 Flash's first smoke leg scored R 0.136 where 3.7 Flash had 0.30 on the same panos. The
cheap checks, in order: every call for both models ends `STOP` (finish-reason tally added in
Step 0); the raw dump shows Flash's empty views are real answers, each with 100–1,000 thinking
tokens, not parse failures; temperature 1.0 and thinking `low` leave the first-10 dev_select
tallies unchanged (Argon 5/3/4 tp/fp/fn at every setting; Flash 2/3/7, 2/2/7, 2/5/7, recall
0.222 throughout). The recall gap is the model's. Per-view concurrency 6 (Step 0) brought wall
time from 45/56 s per pano (Flash/Argon, sequential) to 11–12 / 22–24 s.

## Step 2: zero-shot on all fourteen splits, both models

Both legs: the as-run prompt, 6 views, temperature 0.0, no confidence, concurrency 6. Exported
with `export_model_cache.py --verify` (14 of 14 splits identical to the cache) and registered
`standing=True` in `rampnet/roster.py` in the same commits (`e00b4325` Flash, `0833e205` Argon).

| leg | panos | pooled F1 (9 US cities) | P | R | manual_gold F1 | cost | s/pano |
|---|---|---|---|---|---|---|---|
| Gemini 3.8 Flash | 2,539 | 0.527 | 0.778 | 0.409 | 0.504 | ≈ $44 | 10–14 |
| Gemini 4 Argon | 2,539 | **0.734** | 0.790 | 0.690 | 0.740 | ≈ $58 | 27–47 |
| RampNet (reference) | | 0.793 | 0.953 | 0.686 | 0.908 | | |

What it changed on the board ([`model_scoreboard.md`](model_scoreboard.md)): Argon is the best
zero-shot challenger, cutting RampNet's lead from 0.081 (Opus 5.5) to 0.059. It is ahead of
Opus 5.5 on seven of the nine pooled cities (Opus keeps morgantown and paterson), so the
scoreboard's every-city displacement test is not met and the doc says so. As point estimates
Argon beats RampNet on `laurens_mapillary` (0.712 vs 0.543), `laurens_gsv` (0.706 vs 0.659) and
`bayonne` (0.609 vs 0.517), and its pooled recall (0.690) edges RampNet's (0.686): a third model
on finding 3. It is also the most stable strong model across the nine cities (F1 range 0.129).
The supervised-YOLO comparison in [`model_comparison.md`](model_comparison.md) now splits 1–9
against the best zero-shot model, from 5–5. Flash is the least stable strong challenger (range
0.403; recall 0.10 on laurens_gsv to 0.55 on richmond).

Two notes for the record. Argon hit `MAX_TOKENS` on 3 of 15,234 views (clovis 2, laurens_gsv
1), each after 150k+ thinking tokens; those views read as "no ramps" and are counted as such.
And the Laurens paired-interval record (`laurens_paired_151.py`) gained an Argon row: its
Mapillary-minus-GSV difference is −0.010 F1 [−0.086, +0.065], the smallest gap of any model.

## Step 3: the wording battery on the dev split, three model families

Per model: the seed repeated on `dev_train` in a fresh cache dir (noise floor), three
paraphrases of the seed on `dev_select`, the expert definition and its 515-word structured form
on `dev_select`, and (Gemini) the seed with elicited per-box confidence on all 125 panos. The
open-weight mirror, Qwen3-VL-32B-Instruct, ran on klone (job 43076939, 2× L40S, 58 min for six
legs) under the library pins of its published column; see the launcher's header for why the
pins matter. Full table with Wilson intervals:
[`analysis_out/prompt_opt_276/step3_results.md`](../analysis_out/prompt_opt_276/step3_results.md).

| leg (F1) | Flash 3.8 | Argon | Qwen3-VL-32B |
|---|---|---|---|
| seed, dev_train / repeat | 0.400 / 0.387 (spread 0.013) | 0.726 / 0.705 (0.021) | 0.265 / 0.265 (identical) |
| seed, dev_select | 0.262 | 0.661 | 0.156 |
| para1 / para2 / para3 | 0.302 / 0.322 / 0.337 | 0.645 / 0.684 / **0.705** | 0.154 / 0.140 / 0.218 |
| expert | **0.407** (P .661 R .294) | 0.653 (P .690 R .619) | 0.079 (R .048) |
| expert-long | 0.308 | 0.681 | 0.159 |
| zs + confidence, 125 panos | 0.323 (AP 0.176; 92 boxes, 17 distinct scores) | 0.708 (AP 0.651; 322 boxes, 57 distinct scores) | — |

Readings:

- **The noise floor is model-specific and small.** Flash 0.013, Argon 0.021 on 60 panos; Qwen
  with greedy decoding reproduces bit-for-bit, so its floor is 0 and any paraphrase difference
  is real wording sensitivity. The optimizer's acceptance floor is set above each (Flash 0.02,
  Argon 0.03).
- **Wording matters, and one paraphrase wins everywhere.** `para3` is the best paraphrase on all
  three models and clears each floor (Flash +0.075, Argon +0.044, Qwen +0.062 over the seed).
  Whether it joins the held-out arms is a plan amendment for Jon to call; it would be recorded as
  such.
- **The expert definition is not portable.** Written blind to model errors, it lifts Flash by
  0.145 (mostly precision), is neutral on Argon (within its floor, recall up and precision down),
  and halves Qwen's recall. The long structured form is worse than the compact one on Flash and
  Qwen, and within the floor on Argon.
- **Asking for a score costs nothing measurable** (Flash 0.323 vs 0.339 zero-shot on the same
  125 panos; Argon 0.708 vs 0.696). Argon's scores spread over 57 distinct values, a usable PR
  curve; Flash's cluster in 0.6–1.0 with precision 0.66 → 0.93 across that band, real but
  compressed.

## Step 6 (thinking arm): Argon at thinking low vs high, dev_select

Seed prompt, no confidence, `--gemini-thinking-level`; the seed leg's level is the model's
default (unset). Compared against seed dev_select F1 0.661 (P 0.743 R 0.595), floor 0.021.

| thinking | F1 | P | R | tp/fp/n_gt | cost | s/pano | thinking tokens |
|---|---|---|---|---|---|---|---|
| low | 0.617 | 0.683 | 0.564 | 71/33/126 | $1.04 | 48.6 | 0 reported |
| default (seed) | 0.661 | 0.743 | 0.595 | 75/26/126 | | 22–24 | |
| high | 0.667 | 0.852 | 0.548 | 69/12/126 | $3.27 | 34.4 | 205,769 |

High ties the seed within the floor and moves the operating point toward precision (false
positives 26 → 12, true positives 75 → 69); low is twice the floor below the seed. The #123
result, that thinking is an operating-point dial rather than a quality lever, holds on a second
model family. The in-context-images control (4 `dev_train` exemplars, both models) is
*pending* until the exemplars have human verdicts, so that every exemplar in the study is
reviewer-confirmed.

## Step 4: exemplar-driven definition rewrite (DetPO-style), both models: negative

`scripts/model_comparison/prompt_optimize.py`: `pick` scores the current definition on
`dev_train` (confidence on) and takes the single worst false positive (highest-score box with
no GT, ignore or FP-tag point within 2× radius) and the single worst miss (a GT ramp with no
prediction within 2× radius); a local review page asks a human to confirm each; `rewrite`
shows the confirmed crops, boxed, to the same model as critic and asks for a new definition;
`evaluate` keeps it only if `dev_train` F1 rises by more than the model's seed-repeat floor
(Flash 0.02, Argon 0.03); two consecutive rejections stop the loop; `alternative` then asks
for an image-free rewording of the best definition. Jon judged the four pages on
2026-11-09/10: both first picks confirmed on both models; in round 2 the Argon miss and the
Flash false positive were "unsure" and dropped, so those rewrites used one exemplar.

| model | seed, dev_train | r1 | r2 | outcome |
|---|---|---|---|---|
| Flash 3.8 | 0.367 | 0.375 (+0.008) | 0.358 (−0.009) | both rejected, early stop |
| Argon | 0.727 | 0.710 (−0.017) | 0.698 (−0.030) | both rejected, early stop |

Neither model's rewrite cleared its floor. Argon's candidates traded recall for precision each
time (r2: P .778 R .632 vs the seed's .761/.697); Flash's barely moved. The critic's rewrites
were faithful to the crops (Argon r1 added a storm-drain exclusion and a flush-curb clause for
exactly the two it saw), which is the DetPO failure mode on a strong model: one pair of
exemplars does not generalise to the other 58 panos.

**Selection on dev_select** (65 panos; the pre-registered entrants are the seed, the best
accepted candidate, the final, the alternative and the paraphrase null; gate +0.02 over the
seed):

| model | seed | alternative | para3 (null) | gate |
|---|---|---|---|---|
| Flash 3.8 | 0.262 | **0.337** (P .577 R .238), +0.075 | 0.337 | passes |
| Argon | 0.661 | 0.633 (P .737 R .556), −0.028 | 0.705 | fails |

Flash's alternative passes the gate and ties para3 exactly. On Flash every rewording of the
seed scores 0.30–0.34 and the seed 0.26, so the gate detected the paraphrase effect, not
anything learned from the exemplars. The held-out arm for Flash therefore uses
`def276-flash38-alt` as pre-registered, and the write-up reads it as a wording result with
para3 beside it. Argon has no optimized arm. The whole step cost $7.79 and four human
judgements; the record (trajectories, picks, verdicts, rewrites, crops) is under
`benchmark/prompt_opt_276/optimize/`.

## Step 5: frozen held-out run (2026-11-09 16:49 → 2026-11-11 01:00 PST)

Every arm ran with elicited confidence at concurrency 6 on the 12 held-out splits (1,414
panos); the zero-shot baseline also ran the full 1,000-pano `manual_gold`, the other arms its
committed 250-pano subset. Three processes (two Argon chains over disjoint split lists, one
Flash chain), 97 legs, all exit 0 after the two incidents below. Five small `dev_select` legs
(confidence on, one per frozen definition) supplied the thresholds. $269 on the ledger.

| arm | Flash 3.8 | Argon | status in the plan |
|---|---|---|---|
| zs + conf (the baseline of every comparison) | yes | yes | pre-registered |
| opt + conf (`def276-flash38-alt`, passed the gate) | yes | no optimized definition passed | pre-registered primary endpoint (Flash only) |
| para3 + conf | yes | yes | **amendment, 2026-11-09**: the best paraphrase on all three model families in Step 3; added as a held-out arm before any held-out number was seen |
| expert + conf | yes | yes | **amendment, 2026-11-09**: Jon's blind definition; +0.145 on Flash dev_select, within Argon's floor (so Argon's is exploratory) |

**Protocol, as pre-registered.** One threshold per (model, definition), selected on `dev_select`
with that definition and confidence on: the 0.01-grid threshold that maximises dev_select F1
(lowest on a tie). Every held-out split is scored at that threshold, never at its own best.
Primary endpoint: per model, opt+conf vs zs+conf, macro ΔF1 on the 7-city GSV US reference
pool (richmond, bend, clovis, morgantown, annapolis, paterson, gainesville: the board before
`laurens_mapillary` and `juneau` joined), paired pano bootstrap (2,000 draws, seed 276), Holm
across the two models. Argon has no optimized arm, so the primary test is Flash's alone and
Holm reduces to it. The open pool (`laurens_mapillary`, `bayonne`), the nine-city scoreboard
pool, the amendment arms and the remaining splits are exploratory, Holm-corrected within each
(model, pool) family.

**Selected thresholds.** Flash: 0.00 for every definition, i.e. no confidence cut helps a
model whose errors are misses (dev_select F1 0.267 zs, 0.310 alt, 0.314 para3, 0.296 expert).
Argon: 0.35 (zs, dev_select F1 0.705), 0.37 (para3, 0.698), expert *pending*.

**Primary endpoint: null.** Flash alt+conf minus zs+conf on the reference pool is
**−0.011 F1 [−0.027, +0.006], p = 0.22**. The +0.043 (confidence on) / +0.075 (off) that
passed the dev_select gate did not transfer: the optimized wording loses on four of the seven
reference cities and gains on the open pool by +0.028 [−0.011, +0.068]. The gate (65 panos,
≥ 0.02) is a cost gate, and this is what it looks like when selection noise passes it.

**Macro F1 at the selected thresholds** (`step5_results.md` has every split, P/R, AP and the
raw unthresholded number):

| pool | Flash zs | Flash alt | Flash para3 | Flash expert | Argon zs | Argon para3 | Argon expert |
|---|---|---|---|---|---|---|---|
| reference (7 GSV US) | 0.570 | 0.560 | 0.556 | **0.595** | **0.736** | 0.726 | pending |
| open (Mapillary + Panoramax) | 0.256 | 0.284 | 0.267 | **0.324** | 0.658 | 0.671 | pending |
| nine-city scoreboard pool | 0.516 | 0.511 | 0.511 | **0.550** | **0.733** | 0.726 | pending |
| all 12 held-out | 0.448 | 0.445 | 0.446 | **0.481** | **0.705** | 0.698 | pending |

**Arm minus zs, paired pano bootstrap** (ΔF1 [95% CI], p; Holm within family):

| pool | Flash alt (primary) | Flash para3 | Flash expert | Argon para3 | Argon expert |
|---|---|---|---|---|---|
| reference | −0.011 [−0.027, +0.006] p .22 | −0.015 [−0.031, +0.000] p .05 | **+0.025 [+0.009, +0.041] p .002** (Holm .004) | −0.010 [−0.022, +0.002] p .10 | pending |
| open | +0.028 [−0.011, +0.068] p .16 | +0.011 [−0.027, +0.048] p .53 | **+0.068 [+0.027, +0.106] p < .001** | +0.013 [−0.013, +0.041] p .35 | pending |
| all 12 | −0.004 [−0.018, +0.011] p .61 | −0.002 [−0.015, +0.011] p .75 | **+0.033 [+0.018, +0.048] p < .001** | −0.008 [−0.019, +0.002] p .13 | pending |

**What held and what did not.**

- *Exemplar-derived rewriting (Step 4) is a null on held-out data for both models*: Argon's
  candidates never passed dev_train, and Flash's alternative, the only candidate to pass the
  gate, is −0.011 on the reference pool.
- *The paraphrase arm is a null too*: para3 was the best of three rewordings on dev_select for
  all three model families, and on held-out cities it is −0.015 (Flash) and −0.010 (Argon).
  "Best of K prompts" on 65 panos is selection, not signal, at this effect size.
- *The expert definition moves Flash*, +0.025 on the reference pool and +0.068 on the open
  pool, through recall (reference-pool mean recall 0.45 → 0.48, precision 0.80 → 0.79). It
  leads zs on six of the seven reference cities (paterson 0.647 vs 0.609, bend 0.668 vs 0.640)
  and trails on gainesville. It is an amendment arm, so this is exploratory, and it is the
  natural candidate for the Step 7 replicates. On Argon it is *pending*.
- *Confidence elicitation on Argon moves the operating point, not the model.* Pooled over the
  nine scoreboard cities, zs+conf at its selected threshold scores 0.736 against Step 2's
  0.734 without confidence (same panos, same prompt); the cut is worth +0.013 macro over the
  raw zs+conf output (0.733 vs 0.720). Argon's scores spread over 50–60 distinct values per
  split, so its AP (0.27 on budapest to 0.73 on richmond; 0.635 on full `manual_gold`) is a usable curve. On Flash the
  cut is worth nothing (threshold 0.00) and zs+conf scores 0.544 pooled against Step 2's 0.527.
- *Argon on the full `manual_gold`*: 0.741 at the threshold (0.737 raw) against Step 2's
  0.740. Flash: 0.490 (0.504 in Step 2).

## Step 7: replicates — *pending*

## Cost

| | legs | cost (ledger, list price) |
|---|---|---|
| Gemini 3.8 Flash, Steps 1–4 (incl. $2.32 lost to the parser incident below) | 35 | $59.39 |
| Gemini 4 Argon, Steps 1–4 and the Step 6 thinking arm | 34 | $78.24 |
| Qwen3-VL-32B (klone, `gpu-l40s-makelab`) | 6 | $0 |
| Gemini 3.8 Flash, Step 5 (52 chain legs, 3 threshold legs, 5 one-pano re-runs) | 60 | $153.29 |
| Gemini 4 Argon, Step 5 (39 chain legs, 2 threshold legs, 1 re-run; expert threshold leg *pending*) | 41 | $115.77 |
| Argon `manual_gold` run killed by the leg timeout (no ledger row; ≈ 992/1,000 of a $23.26 leg) | | ≈ $23 |
| **Total through 2026-11-11** | | **≈ $430** |

Reconciled against Cloud Monitoring on 2026-11-09 (billed == logged for both models); the
Step 5 reconcile is *pending* the final legs. Argon list price doubles on 2026-12-01; the
credit expires 2026-12-07.

## Incidents and traps, recorded

- **Flash zs+confidence parser ($2.32 lost).** Under a dict response schema google-genai fills
  `resp.parsed` with plain dicts; the parser read `.box_2d` attributes and scored zero boxes.
  Fixed in `b9e29172`; the leg re-ran clean. The failed run's cache entries were discarded.
- **Argon `MAX_TOKENS`** on 3 of 15,234 Step 2 views (above). A view that ends that way is an
  empty answer; `stop_reasons` in the ledger row records the count.
- **Klone, twice.** The scrubbed purge had removed the Hugging Face blob store but left the
  snapshot symlinks, so `snapshot_download` saw a complete cache and `du -sh models--X` read 13
  MB; use `du -shL snapshots/*/`. Then an unpinned rebuild (torch 2.14.1+cu126, transformers
  5.22.2) loaded the 32B shards and failed every pano with `CUDA driver error: invalid
  argument`; the published column's pins (torch 2.13.0+cu126, torchvision 0.28.0, transformers
  5.14.1, accelerate 1.14.0) ran clean. `run_qwen_276.slurm` documents them.
- **Argon manual_gold leg killed by the chain's 12 h per-leg timeout (2026-11-10 13:00 PST).**
  The Step 5 runner wrapped each leg in `timeout 43200`. Argon on the full 1,000-pano manual_gold
  ran at ~50 s/pano with three processes sharing the Vertex quota (25–31 s/pano on the city
  legs; 26.8 s/pano alone in Step 2) and was killed with 992 of 1,000 panos in the cache. The
  cache kept the work; the relaunch (`LEG_TIMEOUT=86400`) called only the 8 missing panos. The
  ledger therefore has **no row for the 992 killed calls (≈ $23)**; the relaunch's row records
  `panos_called: 8`. `vertex_usage.py --reconcile` will show the gap, and the cost table counts it
  from the billed side.
- **Isolated pano failures, `WinError 10038` (6 of 97 Step 5 legs).** One pano per affected leg
  failed every retry with `ReadError: [WinError 10038] An operation was attempted on something
  that is not a socket` (Flash alt on annapolis, gainesville, laurens_gsv; Flash expert on
  annapolis; Flash para3 on dev_select; Argon expert on manual_gold-250). `compare.py` isolates
  such a pano, scores the leg without it and leaves it uncached, so the regenerator reported
  those legs pending. Each leg was re-run, which calls only the missing pano; the re-run's
  ledger row records `panos_called: 1`.
- **Qwen is deterministic** under its greedy decode: the seed repeat reproduced every box, so
  a fresh cache dir measures no floor for it. The Gemini floors came from the same procedure.

## Where each number lives

| number | file | regenerator |
|---|---|---|
| every dev-split P/R/F1, tp/fp/n_gt, distinct-score counts | `analysis_out/prompt_opt_276/step3_results.{json,md}` | `prompt_276_results.py` (`--check`) |
| Step 5: selected thresholds and sweeps, every held-out split at them, pool macro/pooled F1, paired-bootstrap Δ rows | `analysis_out/prompt_opt_276/step5_results.{json,md}` | `prompt_276_heldout.py` (`--check`) |
| the manual_gold 250-pano subset | `benchmark/prompt_opt_276/manual_gold_250.json` | `prompt_opt_276_splits.py --check` |
| Step 2 pooled / per-split rows, scoreboard effects | `analysis_out/scoreboard.json`, `docs/model_scoreboard.md` | `scoreboard.py --check` |
| Laurens paired Argon row | `analysis_out/laurens_paired_151.{json,md}` | `laurens_paired_151.py` |
| costs, s/pano, tokens, finish reasons, `pano_list` sha1, `run_options` per leg | `analysis_out/usage_log.jsonl` | the legs themselves; `vertex_usage.py --reconcile` |
| the dev split | `benchmark/prompt_opt_276/sao_paulo_split.json` | `prompt_opt_276_splits.py --check` |
| definitions and their provenance | `benchmark/prompt_opt_276/definitions/`, `README.md` | committed text |
| optimizer picks, verdicts, rewrites | `benchmark/prompt_opt_276/optimize/<model>/` | `prompt_optimize.py` |

## Replication

The published Step 2 detections re-score from `benchmark/model_detections/` on any clone
(`scoreboard.py --check`). The dev-split legs live in the local `.model_cache` and
`.model_cache_rep1` until the PR exports them, and so do the Step 5 held-out legs;
`check_all.py --only prompt_276_results,prompt_276_heldout --allow local-cache` verifies both
committed tables against those caches. Re-running any paid leg needs
Vertex access to `rampnet`; the Qwen legs need klone's `gpu-l40s` and the pinned env.
