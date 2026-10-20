# Newest Claude ids on every split: Opus 5.5 and Fable 5.1 (2026-10-19/20)

**Status: done.** Both models have run all 13 splits: 26 legs (2 models x 13 splits). This
campaign ran 25 of them; their ledger rows are stamped 2026-10-19 20:13 to 2026-10-20 08:16 UTC.
The 26th, Fable 5.1 on annapolis, is the #156 leg from 2026-09-05, which the campaign did not
re-run. `juneau`, which is on `main` and which open PR #260 makes a ninth pooled split, was
not run; "Not done yet" says what that changes. The published detections are in `benchmark/model_detections/` and were verified
identical to the cache that produced them (`export_model_cache.py --verify`: 24 of 24 pairs,
then the two bayonne pairs). What the tests re-derive, from those files and
`analysis_out/usage_log.jsonl`: `tests/test_claude_published_legs.py` checks the results table
at r=0.022 (P / R / F1, tp/fp/fn, and the gap column from unrounded F1s), the cost table and the
per-org totals; `tests/test_claude_newest_ids_261.py` checks the paired intervals against
RampNet, the radius table, the emitted-point counts, the per-pano cost ranges, the bayonne leg
costs and the per-call token and thinking figures. Both are CPU-only and read committed files.

**Ledger dollars are estimates.** `est_cost_usd` is list price applied to the token counts
(`scripts/model_comparison/pricing.py`). None of the campaign's rows has been reconciled
against the Anthropic billing console, which is authoritative for what was charged.

**Not on Hugging Face yet.** The `laurens_gsv`, `laurens_mapillary` and `bayonne` panoramas
are not in the published benchmark dataset ([`replication.md`](replication.md), the benchmark
panos row). A clean clone can re-score every leg here from the committed detections, but
re-running those three splits' paid legs needs those panoramas first.

**`bayonne` was missed at first.** The split list for the campaign was taken from the main
checkout, whose branch predates bayonne joining `main`, so the first pass covered 12 splits and
this write-up said "every split" while it was 12 of 13. Review caught it. Both bayonne legs were
run last, on 2026-10-20, with the panos copied from the labeler bundle into this worktree and
verified against `imagery_manifest.json` (125 of 125).

## What and why

Each of Jon's three Claude subscriptions comes with monthly first-party API credit (Max 5x $100,
Max 20x $200, Team up to $500). The credits expire each cycle, so this campaign spends them on the
zero-shot benchmark. Three choices were Jon's, made 2026-10-19:

- **Only the newest ids:** `claude-opus-5-5` and `claude-fable-5-1`. Not Fable 5 and not Opus 5.
- **First-party API only** (`--claude-serving-path anthropic`). The credits don't cover Vertex.
- **Every split, `manual_gold` included.** This reverses #144's decision not to run Claude on
  `manual_gold`, which was made because `gemini-3.1-pro-preview` has no `manual_gold` row to
  compare against. Jon wants the coverage regardless.

Both legs run at `effort=low`, `tool_choice=auto`, 6 perspective views per pano: the same rig
as every earlier Claude leg (#122, #156).

## Who paid, and what it cost

Each org's credit is tagged on every ledger row by `--api-account` (new in this branch). Costs are
list price from `analysis_out/usage_log.jsonl`. The credits pay the bill, and the ledger records
what the run cost, the same as always.

| Org (`--api-account`) | Credit, expiry (UTC) | Leg | Splits finished | Ledger cost |
|---|---|---|---|---|
| `claude5` | $100, 2026-10-23 | Opus 5.5, 12 city splits | 12 of 12 (bayonne's $8.53 on 2026-10-20) | $95.35 (incl. $0.33 calibration) |
| `claude-team` | $500, 2026-10-27 | Fable 5.1, 11 city splits (annapolis is #156's) | 11 of 11 (bayonne's $21.24 on 2026-10-20) | $215.09 |
| `claude-team` | (same) | Fable 5.1, `manual_gold` | done (1,000 panos, 5.0 h) | $167.92 |
| `claude-team` | (same) | Opus 5.5, `manual_gold` | done (moved here from claude20: Team's credit expires first) | $67.98 |
| `claude20` | $200, 2026-11-18 | unused | — | — |

**Per org:** `claude5` spent **$95.35 of $100**; `claude-team` spent **$450.99 of $500**
($215.09 + $167.92 + $67.98); `claude20` spent nothing.

The Team org also holds $63.93 of purchased credit, which is spent after the promotional $500. To
stay inside the free allocation, its runner streams ran under a $440 ledger cap, and they finished
at $429.75. The bayonne leg came after them as a direct `compare.py` call, outside the runner, so
no cap applied to it; it took the org to $450.99, still inside the $500.

**Per pano:** Opus 5.5 **$0.0665–0.0684**. Fable 5.1 **$0.165–0.168** on its first 11 legs of this
campaign, and **$0.170** on bayonne. The 5-pano Opus 5.5 calibration ($0.0665/pano) predicted the
full annapolis leg ($0.0676/pano) to within 2%.

## Results (F1 at the standard radius 0.022, effort low)

The gap column is the difference of the unrounded F1s, so it can differ by 0.001 from the
difference of the printed F1s.

| Split | Opus 5.5 P / R / **F1** | Fable 5.1 P / R / **F1** | Gap |
|---|---|---|---|
| annapolis | 0.694 / 0.748 / **0.720** | 0.637 / 0.585 / **0.610** (2026-09-05 leg) | +0.110 |
| bend | 0.701 / 0.740 / **0.720** | 0.647 / 0.606 / **0.626** | +0.095 |
| budapest_district5 | 0.517 / 0.400 / **0.451** | 0.447 / 0.350 / **0.393** | +0.059 |
| clovis | 0.563 / 0.667 / **0.610** | 0.591 / 0.518 / **0.552** | +0.058 |
| gainesville | 0.622 / 0.761 / **0.684** | 0.521 / 0.588 / **0.553** | +0.132 |
| laurens_gsv | 0.727 / 0.677 / **0.701** | 0.567 / 0.327 / **0.415** | +0.286 |
| laurens_mapillary | 0.706 / 0.655 / **0.679** | 0.637 / 0.402 / **0.493** | +0.187 |
| morgantown | 0.759 / 0.813 / **0.785** | 0.711 / 0.618 / **0.661** | +0.123 |
| paterson | 0.826 / 0.673 / **0.742** | 0.773 / 0.585 / **0.666** | +0.076 |
| richmond | 0.658 / 0.839 / **0.738** | 0.571 / 0.697 / **0.628** | +0.110 |
| sao_paulo | 0.663 / 0.630 / **0.646** | 0.487 / 0.520 / **0.503** | +0.143 |
| bayonne | 0.620 / 0.618 / **0.619** (tp/fp/fn 186/114/115) | 0.552 / 0.528 / **0.540** (tp/fp/fn 159/129/142) | +0.079 |
| manual_gold | 0.719 / 0.725 / **0.722** (tp/fp/fn 2840/1111/1079) | 0.681 / 0.600 / **0.638** (tp/fp/fn 2351/1101/1568) | +0.084 |

**Opus 5.5 leads Fable 5.1 on all 13 splits**: the 12 city splits by 0.06 to 0.29 F1 (median +0.110), and `manual_gold` by +0.084. The gain is
mostly recall. On Laurens, Fable's recall is 0.33/0.40 against Opus 5.5's 0.68/0.66.

These are single runs. There is no seed-variance estimate for either model, so a gap near 0.06 on one
split should not be read as more than suggestive. The direction holding on every split is the
stronger evidence.

## Why Opus 5.5 does better: placement plus detection, not the effort setting

Both models ran at the same effort setting (`low`), but that setting does not produce the same
amount of thinking in the two models (below).

**Nearly the same work per call.** Every ledger row in this campaign shows 2,033 input tokens per
call for both models. Output, which includes thinking (Anthropic bills it as output), is 143–164
tokens per call on every leg: 157 per call for Opus 5.5 and 151 for Fable 5.1, pooled over each
model's campaign rows. Both think little at `effort=low`, but not equally. The ledger's
`thoughts_tokens` is 18 per call for Opus 5.5 (9–33 by leg, about 11% of its output) and 2.6 per
call for Fable 5.1 (0.5–7 by leg, median 1.1, about 2% of its output), about 7x less. Every call
ended in a tool call except two `end_turn`s, both Opus 5.5's: one in clovis and one in
`manual_gold`.

**Part of the gap is where the point lands.** The published detections were re-scored at 1.5x
and 2x the match radius (0.033 and 0.044) on all 13 splits. Fable gains more than Opus 5.5 as the
radius loosens on 12 of the 13, so some of its misses are near misses: the point lands near the
ramp but outside 0.022, and is counted as both a false positive and a miss. At 0.044 the gap
closes by **−6% to 53% of its 0.022 value, median 39%**. On clovis it goes the other way: the gap
grows from +0.058 to +0.062. On sao_paulo it barely moves (6%). Gaps are differences of unrounded
F1s, and the share closed is computed from unrounded values.

| Split | Gap at r=0.022 | r=0.033 | r=0.044 | share closed at 0.044 | Opus 5.5 F1 at 0.044 | Fable 5.1 F1 at 0.044 |
|---|---|---|---|---:|---|---|
| annapolis | +0.110 | +0.077 | +0.067 | 39% | 0.773 | 0.705 |
| bend | +0.095 | +0.077 | +0.068 | 29% | 0.752 | 0.685 |
| budapest_district5 | +0.059 | +0.057 | +0.031 | 47% | 0.547 | 0.516 |
| clovis | +0.058 | +0.051 | +0.062 | −6% | 0.703 | 0.641 |
| gainesville | +0.132 | +0.079 | +0.064 | 51% | 0.730 | 0.666 |
| laurens_gsv | +0.286 | +0.189 | +0.166 | 42% | 0.714 | 0.548 |
| laurens_mapillary | +0.187 | +0.108 | +0.088 | 53% | 0.700 | 0.612 |
| morgantown | +0.123 | +0.098 | +0.091 | 26% | 0.809 | 0.718 |
| paterson | +0.076 | +0.055 | +0.048 | 37% | 0.757 | 0.709 |
| richmond | +0.110 | +0.072 | +0.070 | 36% | 0.776 | 0.706 |
| sao_paulo | +0.143 | +0.140 | +0.134 | 6% | 0.724 | 0.590 |
| bayonne | +0.079 | +0.054 | +0.044 | 45% | 0.688 | 0.644 |
| manual_gold | +0.084 | +0.062 | +0.051 | 39% | 0.784 | 0.732 |

**The rest is detection.** At 2x radius Opus 5.5 still leads on every split, by 0.031
(budapest_district5) to 0.166 (laurens_gsv), median 0.067. On annapolis that lead is almost all
recall at matching precision (R 0.803 vs 0.680, P 0.744 vs 0.733). On annapolis the published
files hold 328 Opus 5.5 points against 285 Fable 5.1 points (+15%); 317 and 270 of them are
scored (the rest fall within the radius of an ignore point, which counts as neither a true nor a
false positive). More of Opus 5.5's points are right.

Re-run, from committed files only (no `.model_cache`, no API key, no panoramas):

```powershell
python scripts/analysis/claude_newest_ids_261.py --check      # compare against analysis_out/claude_newest_ids_261/results.json
python scripts/analysis/claude_newest_ids_261.py --markdown   # print the tables on this page
```

**Fable's September annapolis leg reproduces exactly** (F1 0.610) through this branch's code from
its cached detections, so the two models are being scored the same way.

## How it was run

API keys are Windows user environment variables (`ANTHROPIC_API_KEY_CLAUDE5`, `_CLAUDE20`, `_TEAM`).
Panos and `.model_cache` live only in the main checkout, so `-DataRoot` points there. The one
exception is bayonne, whose panos were only in this worktree (see the top of this page).

```powershell
# calibration: 5 panos, claude5 org
python scripts/model_comparison/compare.py <main>/benchmark/annapolis --models claude:claude-opus-5-5 `
  --claude-serving-path anthropic --claude-effort low --limit 5 --api-account claude5 `
  --cache-dir <main>/.model_cache

# Opus 5.5, 11 city splits, claude5 org
scripts/model_comparison/run_claude_legs_by_account.ps1 -Account claude5 -Model claude-opus-5-5 `
  -PerPano 0.0665 -Cap 95 -DataRoot <main> `
  -Splits annapolis,bend,budapest_district5,clovis,gainesville,laurens_gsv,laurens_mapillary,morgantown,paterson,richmond,sao_paulo

# Fable 5.1, Team org: three streams in parallel, sharing a $440 cap
scripts/model_comparison/run_claude_legs_by_account.ps1 -Account claude-team -Model claude-fable-5-1 `
  -PerPano 0.1654 -Cap 440 -DataRoot <main> -Splits manual_gold
scripts/model_comparison/run_claude_legs_by_account.ps1 -Account claude-team -Model claude-fable-5-1 `
  -PerPano 0.1654 -Cap 440 -DataRoot <main> -Splits gainesville,morgantown,paterson,richmond,sao_paulo
scripts/model_comparison/run_claude_legs_by_account.ps1 -Account claude-team -Model claude-fable-5-1 `
  -PerPano 0.1654 -Cap 440 -DataRoot <main> -Splits bend,budapest_district5,clovis,laurens_gsv,laurens_mapillary

# Opus 5.5, manual_gold, Team org (under the same $440 cap)
scripts/model_comparison/run_claude_legs_by_account.ps1 -Account claude-team -Model claude-opus-5-5 `
  -PerPano 0.0665 -Cap 440 -DataRoot <main> -Splits manual_gold

# bayonne, both models, run last (2026-10-20) as direct compare.py calls from this worktree,
# whose benchmark/bayonne holds the panos. The key is set by hand, as the runner does.
$env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY_CLAUDE5', 'User')
python scripts/model_comparison/compare.py benchmark/bayonne --models claude:claude-opus-5-5 `
  --claude-serving-path anthropic --claude-effort low --api-account claude5 `
  --cache-dir <main>/.model_cache --usage-log <main>/analysis_out/usage_log.jsonl
$env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY_TEAM', 'User')
python scripts/model_comparison/compare.py benchmark/bayonne --models claude:claude-fable-5-1 `
  --claude-serving-path anthropic --claude-effort low --api-account claude-team `
  --cache-dir <main>/.model_cache --usage-log <main>/analysis_out/usage_log.jsonl

# publish + verify (per model; list the finished splits)
python scripts/analysis/export_model_cache.py --cache-dir <main>/.model_cache --models claude:claude-opus-5-5 `
  --claude-effort low --claude-serving-path anthropic --publish-as claude-opus-5-5-effort-low-anthropic `
  --splits <finished splits> --write
python scripts/analysis/export_model_cache.py ... --verify
```

The runner calls ran via a scratch copy of `run_claude_legs_by_account.ps1` that hardcoded the
main checkout path. The committed version takes `-DataRoot` instead. It also has two fixes that
landed in b202470, after every run here: it passes `--usage-log` explicitly, so the cap reads the
same ledger `compare.py` writes, and it stops when `compare.py` exits non-zero. Neither fix changes
a result here. The scratch copy ran both the code and the data from the main checkout, so
`compare.py`'s default ledger and the ledger the cap read were the same file.

## What it changes on the scoreboard

Both legs have full pooled coverage, so `scripts/analysis/scoreboard.py` puts them in the
pooled tables of [`model_scoreboard.md`](model_scoreboard.md). Both stay `standing=False` in
`rampnet/roster.py`, so the standing roster tables in `model_comparison.md` do not change;
each generated city table's footnote lists them.

Every Claude number in these bullets is a single run, with no seed variance, at effort low only,
through the first-party serving path (`anthropic`). Read each comparison with RampNet with that
in mind.

- **Claude Opus 5.5 is the best challenger with full pooled coverage**: pooled F1 0.710
  (P 0.691 / R 0.737) over the eight US city splits. RampNet's lead over it is 0.082 F1;
  before it, the best such challenger was YOLO11l, 0.193 behind RampNet. Bayonne is held out
  of the pool, so it moves none of these pooled numbers.
- **Against RampNet, split by split, with paired intervals.** RampNet minus the Claude leg in
  F1, with a pano-level paired bootstrap (resample the split's panos, rescore both models on the
  same draw; 10,000 draws, seed 0; `scripts/analysis/claude_newest_ids_261.py`, the same draw
  construction as `bayonne_paired_159.py`). RampNet is at its deployed 0.55, as on the
  scoreboard. The four cells where a Claude leg has the higher F1:

  | split | leg | RampNet F1 | leg F1 | RampNet − leg [95% CI] | reading |
  |---|---|---:|---:|---|---|
  | `laurens_mapillary` | Opus 5.5 | 0.543 | 0.679 | −0.136 [−0.216, −0.053] | Opus 5.5 wins |
  | `bayonne` | Opus 5.5 | 0.517 | 0.619 | −0.102 [−0.165, −0.040] | Opus 5.5 wins |
  | `laurens_gsv` | Opus 5.5 | 0.659 | 0.701 | −0.042 [−0.101, +0.020] | point estimate only; the interval includes zero |
  | `bayonne` | Fable 5.1 | 0.517 | 0.540 | −0.023 [−0.096, +0.050] | point estimate only; the interval includes zero |

  So Opus 5.5 **beats RampNet on two splits**, `laurens_mapillary` and `bayonne`. On
  `laurens_gsv` Opus 5.5, and on `bayonne` Fable 5.1, score higher than RampNet, but these
  data do not establish a win. RampNet has the top F1 on the other ten splits, `manual_gold`
  included (0.908 against 0.722), and its lead over Opus 5.5 has an interval above zero on all
  ten; the narrowest is morgantown, +0.050 [+0.005, +0.097]. Over Fable 5.1 RampNet's lead has
  an interval above zero on eleven splits; on `laurens_mapillary` it is +0.051 [−0.036, +0.142].
  These intervals resample panos only. They do not cover run-to-run variation of the Claude
  legs, which has not been measured.
- **Bayonne's GT carries a caveat.** It is single-rater at medium reviewer confidence, and
  French street design fought the US-written rubric ([`bayonne_split_159.md`](bayonne_split_159.md)).
  RampNet's bayonne F1 is low because of its recall (0.375 at 0.55, its lowest anywhere), not
  its precision (0.831). Both Claude legs have lower precision there (0.620, 0.552) and higher
  recall (0.618, 0.528).
- **Its pooled recall, 0.737, is above RampNet's 0.686**, the only model with full pooled
  coverage for which that is true apart from the two open-vocabulary detectors at their 0.05
  floor. Its precision is 0.691 against RampNet's 0.951.
- **Fable 5.1 pools at 0.598**, level with YOLO11l (0.599): the unrounded gap is 0.0002.
- **These pooled numbers depend on the pool staying at eight splits.** See juneau under "Not
  done yet".

## Not done yet

- Seed variance: neither model has been run twice on any split.
- Effort: both legs ran at `low` only. No other effort level was run.
- Serving path: first-party API only, so nothing here says how the same ids behave on Vertex.
- `claude-fable-5` was not part of this campaign, so it is still annapolis-only.
- The `claude20` org's $200 credit was not used.
- **juneau: neither model was run on it.** `benchmark/juneau/` (125 panos) is on `main`, and
  open PR #260 registers it as the ninth pooled US split. When #260 merges, each of these two
  legs has 8 of 9 pooled splits, so neither has full pooled coverage. The pooled F1s above
  (Opus 5.5 0.710, Fable 5.1 0.598), RampNet's 0.082 lead over Opus 5.5, and "all 13 splits"
  then stop being current. Two ways to restore them: run both legs on juneau, or regenerate the
  scoreboard, which moves both legs out of the pooled table into the partial-coverage table.
  Estimated cost of the juneau legs, at this campaign's measured per-pano cost (Opus 5.5
  $0.0665–0.0684, Fable 5.1 $0.165–0.170): about **$8.31–8.56** for Opus 5.5 and
  **$20.60–21.24** for Fable 5.1, **$28.91–29.79** together. Not run: it is paid, and it waits
  on #260.
