# Newest Claude ids on every split: Opus 5.5 and Fable 5.1 (2026-10-19)

**Status: IN PROGRESS.** This is a checkpoint, committed while the runs were still going so the
partial results don't live only on one disk. Every number below is from a split that finished;
splits still running are marked as such. The published detections for every finished split are in
`benchmark/model_detections/` and were verified identical to the cache that produced them
(`export_model_cache.py --verify`, 23 of 23 pairs).

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
| `claude5` | $100, 2026-10-23 | Opus 5.5, 11 city splits | 11 of 11 | $86.82 (incl. $0.33 calibration) |
| `claude-team` | $500, 2026-10-27 | Fable 5.1, 10 city splits | 10 of 10 | $193.85 |
| `claude-team` | (same) | Fable 5.1, `manual_gold` | done (1,000 panos, 5.0 h) | $167.92 |
| `claude-team` | (same) | Opus 5.5, `manual_gold` | running (moved here from claude20: Team's credit expires first) | — |
| `claude20` | $200, 2026-11-18 | unused so far | — | — |

The Team org also holds $63.93 of purchased credit, which is spent after the promotional $500. To
stay inside the free allocation, its streams run under a $440 ledger cap.

**Per pano:** Opus 5.5 **$0.0665–0.068**, Fable 5.1 **$0.166–0.168**. The 5-pano Opus 5.5
calibration ($0.0665/pano) predicted the full annapolis leg ($0.0676/pano) to within 2%.

## Results so far (F1 at the standard radius 0.022, effort low)

| Split | Opus 5.5 P / R / **F1** | Fable 5.1 P / R / **F1** | Gap |
|---|---|---|---|
| annapolis | 0.694 / 0.748 / **0.720** | 0.637 / 0.585 / **0.610** (2026-09-05 leg) | +0.110 |
| bend | 0.701 / 0.740 / **0.720** | 0.647 / 0.606 / **0.626** | +0.094 |
| budapest_district5 | 0.517 / 0.400 / **0.451** | 0.447 / 0.350 / **0.393** | +0.058 |
| clovis | 0.563 / 0.667 / **0.610** | 0.591 / 0.518 / **0.552** | +0.058 |
| gainesville | 0.622 / 0.761 / **0.684** | 0.521 / 0.588 / **0.553** | +0.131 |
| laurens_gsv | 0.727 / 0.677 / **0.701** | 0.567 / 0.327 / **0.415** | +0.286 |
| laurens_mapillary | 0.706 / 0.655 / **0.679** | 0.637 / 0.402 / **0.493** | +0.186 |
| morgantown | 0.759 / 0.813 / **0.785** | 0.711 / 0.618 / **0.661** | +0.124 |
| paterson | 0.826 / 0.673 / **0.742** | 0.773 / 0.585 / **0.666** | +0.076 |
| richmond | 0.658 / 0.839 / **0.738** | 0.571 / 0.697 / **0.628** | +0.110 |
| sao_paulo | 0.663 / 0.630 / **0.646** | 0.487 / 0.520 / **0.503** | +0.143 |
| manual_gold | running | 0.681 / 0.600 / **0.638** (tp/fp/fn 2351/1101/1568) | — |

**Opus 5.5 leads Fable 5.1 on all 11 city splits**, by 0.06 to 0.29 F1 (median +0.110). The gain is
mostly recall. On Laurens, Fable's recall is 0.33/0.40 against Opus 5.5's 0.68/0.66.

These are single runs. There is no seed-variance estimate for either model, so a gap near 0.06 on one
split should not be read as more than suggestive. The direction holding on every split is the
stronger evidence.

## Why Opus 5.5 does better: placement plus detection, not effort

**Same work per call.** Every ledger row in this campaign shows 2,033 input tokens per call for
both models, and 142–164 output tokens per call. Every call ended in a tool call except one
`end_turn` in Opus 5.5's clovis leg. At `effort=low`, neither model is thinking more than the other.

**About half the gap is where the point lands.** The cached detections were re-scored at 1.5x
and 2x the match radius. This cost nothing: no API key was set, so no call was possible. Fable gains
more as the radius loosens, so many of its misses are near misses: the point lands near the ramp
but outside 0.022, and is counted as both a false positive and a miss.

| Split | Gap at r=0.022 | r=0.033 | r=0.044 | Opus 5.5 F1 at 0.044 | Fable 5.1 F1 at 0.044 |
|---|---|---|---|---|---|
| annapolis | +0.110 | +0.077 | +0.068 | 0.773 | 0.705 |
| bend | +0.094 | +0.076 | +0.067 | 0.752 | 0.685 |
| gainesville | +0.131 | +0.079 | +0.064 | 0.730 | 0.666 |

**The rest is detection.** At 2x radius Opus 5.5 still leads by ~0.065, almost all of it recall at
matching precision (annapolis: R 0.803 vs 0.680, P 0.744 vs 0.733). It emits ~17% more points
(annapolis: 317 vs 270) and more of them are right.

The radius check covers 3 splits, chosen because they had finished when it was run. It is
re-runnable for any finished split:

```powershell
# no ANTHROPIC_API_KEY set, so a cache miss cannot become a paid call
python scripts/model_comparison/compare.py <main checkout>/benchmark/<split> `
  --models 'claude:claude-opus-5-5,claude:claude-fable-5-1' `
  --claude-serving-path anthropic --claude-effort low `
  --cache-dir <main checkout>/.model_cache --radius 0.044
```

**Fable's September annapolis leg reproduces exactly** (F1 0.610) through this branch's code from
its cached detections, so the two models are being scored the same way.

## How it was run

API keys are Windows user environment variables (`ANTHROPIC_API_KEY_CLAUDE5`, `_CLAUDE20`, `_TEAM`).
Panos and `.model_cache` live only in the main checkout, so `-DataRoot` points there.

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

# publish + verify (per model; list the finished splits)
python scripts/analysis/export_model_cache.py --cache-dir <main>/.model_cache --models claude:claude-opus-5-5 `
  --claude-effort low --claude-serving-path anthropic --publish-as claude-opus-5-5-effort-low-anthropic `
  --splits <finished splits> --write
python scripts/analysis/export_model_cache.py ... --verify
```

These ran via a scratch copy of `run_claude_legs_by_account.ps1` that hardcoded the main checkout
path. The committed version takes `-DataRoot` instead and is otherwise the same logic.

## Not done yet

- Opus 5.5 on `manual_gold` (running on the Team org).
- Scoreboard regeneration, coverage text in `docs/model_comparison.md`, and the tests that pin
  coverage (`tests/test_scoreboard.py`, `tests/test_claude_published_legs.py`). These wait for the
  full set.
- Seed variance: neither model has been run twice on any split.
