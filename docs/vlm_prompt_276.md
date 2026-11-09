# VLM prompt experiments, round 1: Gemini and Qwen on the curb-ramp benchmark (#276)

**Status: in progress (2026-11-09).** Steps 0–3 of the plan are done on three model families,
the Step 2 zero-shot columns are published and on the scoreboard, and the Step 6 thinking arm
has run. Step 4 (exemplar-driven definition rewrite) is waiting on the reviewer's verdicts for
the first picked exemplars; Steps 5–7 follow it. Sections marked *pending* are placeholders.
Every dev-split number below regenerates from the detection caches with
`python scripts/analysis/prompt_276_results.py` (`--check` compares with the committed JSON;
`scripts/check_all.py --only prompt_276_results --allow local-cache` runs it).

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

## Step 4: exemplar-driven definition rewrite — *pending*

`scripts/model_comparison/prompt_optimize.py` (`pick` → local `review.html` + hand-saved
`verdicts.json` → `rewrite` → `evaluate --floor`; `alternative`; `status`). Both models' first
picks are made from the cached zs+confidence legs ($0): Flash dev_train F1 0.367 (P .731 R .245),
Argon 0.727 (P .761 R .697), each with one worst FP and one pure miss on the review page under
`benchmark/prompt_opt_276/optimize/<model>/iter_01/`. Gate: ≥ 0.02 F1 over the seed on
`dev_select`, else the held-out run is skipped and the negative result recorded here.

## Step 5: frozen held-out run — *pending* (gated on Step 4)

Pre-registered primary endpoint: per model, opt+conf vs zs+conf, macro ΔF1 on the 7-city US
reference pool, paired pano bootstrap, Holm across the two models. The open pool and everything
else is exploratory.

## Step 7: replicates — *pending*

## Cost so far

| | legs | cost (ledger, list price) |
|---|---|---|
| Gemini 3.8 Flash (Steps 1–4, incl. $2.32 lost to the parser incident below) | 29 | $55.76 |
| Gemini 4 Argon (Steps 1–3, Step 6 thinking) | 28 | $74.08 |
| Qwen3-VL-32B (klone, `gpu-l40s-makelab`) | 6 | $0 |
| **Total through 2026-11-09 14:31** | | **$129.84** |

Reconciled against Cloud Monitoring on 2026-11-09 (billed == logged for both models). Argon
list price doubles on 2026-12-01; the credit expires 2026-12-07.

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
- **Qwen is deterministic** under its greedy decode: the seed repeat reproduced every box, so
  a fresh cache dir measures no floor for it. The Gemini floors came from the same procedure.

## Where each number lives

| number | file | regenerator |
|---|---|---|
| every dev-split P/R/F1, tp/fp/n_gt, distinct-score counts | `analysis_out/prompt_opt_276/step3_results.{json,md}` | `prompt_276_results.py` (`--check`) |
| Step 2 pooled / per-split rows, scoreboard effects | `analysis_out/scoreboard.json`, `docs/model_scoreboard.md` | `scoreboard.py --check` |
| Laurens paired Argon row | `analysis_out/laurens_paired_151.{json,md}` | `laurens_paired_151.py` |
| costs, s/pano, tokens, finish reasons, `pano_list` sha1, `run_options` per leg | `analysis_out/usage_log.jsonl` | the legs themselves; `vertex_usage.py --reconcile` |
| the dev split | `benchmark/prompt_opt_276/sao_paulo_split.json` | `prompt_opt_276_splits.py --check` |
| definitions and their provenance | `benchmark/prompt_opt_276/definitions/`, `README.md` | committed text |
| optimizer picks, verdicts, rewrites | `benchmark/prompt_opt_276/optimize/<model>/` | `prompt_optimize.py` |

## Replication

The published Step 2 detections re-score from `benchmark/model_detections/` on any clone
(`scoreboard.py --check`). The dev-split legs live in the local `.model_cache` and
`.model_cache_rep1` until the PR exports them; `check_all.py --only prompt_276_results --allow
local-cache` verifies the committed table against those caches. Re-running any paid leg needs
Vertex access to `rampnet`; the Qwen legs need klone's `gpu-l40s` and the pinned env.
