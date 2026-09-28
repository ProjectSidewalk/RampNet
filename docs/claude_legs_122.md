# The Claude legs (#122, #156)

The Claude legs of the model comparison: the first annapolis results at both effort levels,
the Claude Fable legs served off the first-party API, how to reproduce the four legs, and how
one billed day was split between two legs by effort. Moved verbatim from
[`model_comparison.md`](model_comparison.md) under #145, where it sat after § "What each model
class buys you"; heading levels are unchanged. "Above" and "below" in the moved text refer
to `model_comparison.md` when the thing they point at is not in this file.

## Claude on Vertex and first-party (#122, #156)

The wiring notes that sat here, "Claude on Vertex (#122): two constraints worth knowing before you wire it up", moved verbatim to [`running_model_comparison.md`](running_model_comparison.md) (#145).

### First results: annapolis, both models × both effort levels (2026-08-15)

Full 125-pano annapolis split (294 GT ramps, **all four legs on the same denominator** —
see the correction note below), the **shared** `DETECTION_PROMPT` (deliberately not
Claude-tuned, so these numbers stay comparable to the other legs), `auto` tool choice,
identical rig. Box mapping verified by `dump_detections.py` for both models before
reading anything into the numbers — predictions sit tight on the ramps, no offset.

| model | effort | P | R | F1 | tp/fp/fn | thinking tok | cost |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: |
| claude-sonnet-5 | low | 0.589 | 0.381 | 0.463 | 112/78/182 | 57 | $3.60 |
| claude-sonnet-5 | high | 0.506 | 0.415 | 0.456 | 122/119/172 | 17,820 | $3.82 |
| **claude-opus-5** | **low** | 0.572 | 0.605 | **0.588** | 178/133/116 | 523 | $8.94 |
| claude-opus-5 | high | 0.430 | 0.656 | 0.520 | 193/256/101 | 127,227 | $12.46 |

The two Opus costs are the only ones here with independent corroboration: they were recorded
from console output at run time, and Cloud Monitoring's minute series was later solved for
the same split and returned $8.95 / $12.47 (§"Splitting a two-leg day by effort"). The Sonnet
pair has no such check — that day's telemetry does not separate.

**Every number in this table is re-derivable from committed files**, with no
`.model_cache`, no API key and no GPU: the per-panorama detections are published under
`benchmark/model_detections/claude-*-effort-*__annapolis.json`, and
`tests/test_claude_published_legs.py` recomputes the whole table from them on every CI run.
A number edited here without re-running anything fails the suite.

**Effort is an operating-point dial, never a quality lever.** Both models move the same
direction — thinking makes them fire more, so recall rises and precision falls — and in
neither case does F1 improve. Sonnet nets slightly *negative* (0.463 → 0.456); Opus nets
clearly *negative* (0.588 → 0.520), in both cases because precision falls harder than
recall rises. Two models, same direction, so this is a property of the task rather than of
one model: **spend effort to move along the P/R curve, not to get a better detector.**
Same shape as this benchmark's Qwen 8B→32B finding, where scaling flipped the failure mode
instead of fixing it. Note that the expensive setting is the worse one — 127k thinking
tokens to lose 0.068 F1.

**`claude-opus-5` at `low` was the strongest general model measured on annapolis**, at
0.588 — the first to displace `gemini-3.1-pro-preview` (0.567) from that slot — **until the
two Fable legs (0.611 and 0.610, next section) displaced it in turn.** Against
`claude-sonnet-5` at the same effort it gains **+0.224 recall at essentially unchanged
precision** (0.589 → 0.572), which is a capability difference rather than a threshold
shift. RampNet leads it by **0.251** (0.839 vs 0.588).

> **Superseded on the pooled board, 2026-08-19 (#139); re-pooled over eight splits
> 2026-09-17 after #151 added `laurens_mapillary`.** That displacement is an annapolis
> result and it does not generalise. Run on the other splits, `claude-opus-5` at `low` pools
> to **F1 0.568 against `gemini-3.1-pro`'s 0.575** over the eight US city splits — an
> annapolis lead of +0.021 becoming a pooled gap of −0.007, inside any reading of a tie.
> **`gemini-3.1-pro` still tops the table, by less than 0.01.** Read per split rather than
> pooled, the two trade wins — four each on the eight pooled splits, six of eleven overall
> for Opus, with laurens_mapillary (+0.086) the largest pooled gap in Opus's favour (the
> held-out laurens_gsv is wider still, +0.158) and gainesville (−0.069) and richmond
> (−0.066) the largest against — so the honest reading is
> not "Opus is worse" but **"per-split gaps whose range is 0.156 swamped a +0.021 lead"**.
> What survives is the shape rather than the ranking: Opus trades **−0.077 precision for
> +0.052 recall**, the highest recall of any chat VLM with full pooled coverage (0.586). The
> pooled table is in [`model_scoreboard.md`](model_scoreboard.md); the paragraph above is
> left as written because the annapolis numbers in it are still correct and are what the rest
> of this section analyses. (On the seven-split board this note was first written against,
> the same detections pooled to 0.588 against 0.608, 3 wins of 7; the eighth split moved it.)

**Correction, 2026-08-18 — the sonnet/low row originally used a different denominator.**
As first published it read 0.587 / 0.372 / 0.456 on `108/76/182`, which is **290** GT
ramps, not 294: one panorama (`annapolis:1528518111324684`) was lost when a malformed tool
result raised out of the parser and took all six of its views with it — the 1-in-745 case
described above. So that row was scored on 124 panos while the other three used 125, and
the original "Sonnet nets *exactly* flat, 0.456 → 0.456" compared two different pano sets.
The parser was hardened, the panorama was re-run under the fix (6 calls, $0.03, recorded in
`analysis_out/usage_log.jsonl`), and the row above is the whole split. The finding survives
the correction and is slightly strengthened: Sonnet now moves in the same direction as Opus
rather than being a flat tie, so *both* models lose F1 to effort.

Four caveats that travel with these numbers:

- **`claude-opus-5` at `high` has the highest recall of any challenger on this split
  (0.656 vs RampNet's 0.738)** — closer to RampNet than any general model has come here,
  and worth remembering given this project's recall-first framing, where a false negative
  is a permanent loss and a false positive is cheap. F1 ranks it below `low`; a
  recall-weighted objective would not.
- The prompt is **fixed and Gemini-derived**. Claude is documented to underperform on
  prompts carried over from another model, so these numbers bound Claude-on-our-prompt,
  not Claude. Holding it fixed is the right call for comparability; a prompt-variant run
  would be a separate, separately-labelled experiment.
- **The two paid legs are not fed identical pixels.** `ClaudeDetector` JPEG-encodes each
  reprojected view at quality 90; the Gemini leg is handed a PIL image and `google-genai`'s
  `pil_to_blob` encodes it as **lossless PNG** (its JPEG branch requires
  `image.format == "JPEG"` *and* a filename, and a reprojected view is an in-memory
  `Image.fromarray` with neither). So a Claude/Gemini comparison on this split carries one
  extra JPEG round-trip on the Claude side. It costs no tokens to remove —
  `--claude-image-format png` — but it *is* a cache-key change, so switching means re-paying
  for the detections; the published numbers are the JPEG ones and were left as run. The
  direction of any bias is unmeasured, which is the honest statement: on a split whose miss
  story is faint far-field ramps, q90 quantization is not obviously harmless.
- **Decoding is not pinned.** `GeminiDetector` sets `temperature=0.0`; the Claude legs sent
  no temperature and took the provider default, so one paid leg is greedy and the other is
  sampled. `--claude-temperature 0.0` matches them, at the same re-run cost as above. Both
  settings are recorded in `ClaudeDetector.signature()` only when they deviate from what
  these legs ran, precisely so that documenting the gap did not orphan the paid cache.

**Measured cost, `claude-sonnet-5` at `--claude-effort low`** (5 annapolis panos, 2026-08-15):
**2,229 input and 39 output tokens per call, 0 thinking.** The tool definition is
re-sent on every call and accounts for ~700 of that input — about a third of the bill,
and not worth caching, since tools render below Sonnet 5's 1,024-token minimum cacheable
prefix. That puts a 125-pano leg at **≈$3.60** and all ten splits at **≈$61** (halved by
batch). Effort is the dominant lever: thinking bills as output at $10/MTok, and `low`
spends none of it.

### Claude Fable on annapolis (#156): the first legs served off Vertex

**Serving path caveat, and it travels with every number below.** Vertex gates the whole
Fable family behind a project-level publisher data-sharing setting
(`PublisherModelConfig.data_sharing_enabled_provider`), so these two legs did **not** run
on the Vertex path the four legs above used. They ran on Anthropic's first-party API under
`--claude-serving-path anthropic`, a different account and a different rate card for the
same weights. Same rig, same prompt, same tool definition, same `effort=low`, same JPEG
q90 encoding. The path is deliberately **not** part of the detection cache key — it
changes who bills, not what was asked, and putting it in the key would have orphaned the
$28.82 of paid detections above — so it is recorded in `analysis_out/usage_log.jsonl` and
in each published file's `pins` instead. Whether the two paths return bit-identical
detections for one model id is **untested**; nothing here depends on it, because no model
was run on both.

| model | P | R | F1 | tp/fp/fn | boxes/pano | thinking tok | cost, 720 calls (120 of 125 panos) |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| `claude-fable-5` | 0.579 | 0.646 | **0.611** | 190/138/104 | 2.72 | 23,699 | $18.47 |
| `claude-fable-5-1` | 0.637 | 0.585 | **0.610** | 172/98/122 | 2.28 | 254 | $19.86 |
| *`claude-opus-5` (low), for reference* | *0.572* | *0.605* | *0.588* | *178/133/116* | *2.56* | *523* | *$8.94* |

The cost and thinking-token columns are the two 720-call full-leg rows in
`analysis_out/usage_log.jsonl` (2026-09-05 15:52 and 16:47), which cover 120 of the 125
panos: the other 5 were served from the cache the calibration pass had already written.
Counting that pass's first 30 calls on those 5 panos ($0.76 for `claude-fable-5`, $0.82 for
`claude-fable-5-1`), each id's whole split cost **$19.23** and **$20.67**. The calibration's
second pass (42 calls, $1.11, re-issued after a cache-write gap) is in the ledger too but
belongs to neither leg's number. The Opus row is the whole 125-pano leg, from console
output rather than the ledger (see the gap stated below).

**Both displace `claude-opus-5` at the top of this split**, which had itself displaced
`gemini-3.1-pro-preview` (0.567). This is the first time a general-purpose model has beaten
Opus here. RampNet still leads by **0.228** (0.839 vs 0.611).

**Within the family, the version is an operating-point dial — not a quality lever.** The
two are separated by **0.001 F1**, which is nothing, while sitting at visibly different
operating points: 5.1 trades 0.061 recall for 0.058 precision against 5, and emits 0.44
fewer boxes per pano. That is the same shape as the effort finding above and the same shape
as the Qwen 8B→32B inversion: the knob moves *where* on the P/R curve the model sits, and
the ceiling does not move. Which one to prefer is therefore a decision about the
objective, not about the models — and under this project's recall-first framing, where a
false negative is permanent and a false positive is cheap, that argues for `claude-fable-5`
despite it being the older id.

**The always-on-thinking cost premise was wrong, and this is where it was measured.**
Because the Fable family cannot disable thinking (`{"type": "disabled"}` is a 400 and
`budget_tokens` was removed), #156 predicted a cost band "wider than a flat 2x" — no
near-zero-thinking floor to make an `effort=low` leg cheap. It is a flat 2x. Fable is
$10/$50 per MTok against Opus's $5/$25, and per call the legs cost 2.15x and 2.31x the
Opus leg ($0.0257 and $0.0276 against $0.0119) — the whole-split totals above, $19.23 and
$20.67 against $8.94, give the same ratios.
`claude-fable-5` spent ~33 thinking tokens/call and `claude-fable-5-1` ~0.35, against
Opus-low's ~0.7 — always-on thinking is *adaptive*, and on a localization task at low
effort it costs essentially nothing. A 5-pano calibration predicted the full-leg cost to
within 1.5% on both ids.

**These legs are `standing=False` and cover annapolis only (1 of 8 pooled splits).** Both
clear the pre-registered 0.567 gate for expanding to the full split set, so that expansion
is now a live decision rather than a hypothetical — `manual_gold` is 1,000 panos × 6
views = 6,000 calls, so at the measured full-leg rates ($0.0257/call for `claude-fable-5`,
$0.0276/call for `claude-fable-5-1`) it is **~$154 and ~$165 per id** for that split alone.
It has **not** been taken, and no other split has been run.

**Reproducing them** (the detections are committed; nothing below needs an API key):

```bash
pytest -q tests/test_claude_published_legs.py   # recompute both rows from committed files
```

Re-exporting from a `.model_cache` that produced them needs the serving path as well as the
effort — not because it changes the cache lookup (it does not), but because the registry
uses it to resolve the leg's published filename:

```bash
for m in claude-fable-5-1 claude-fable-5; do
  python scripts/analysis/export_model_cache.py --splits annapolis --models claude:$m --claude-effort low --claude-serving-path anthropic
  python scripts/analysis/export_model_cache.py --verify --splits annapolis --models claude:$m --claude-effort low --claude-serving-path anthropic
done
```

Re-running them from scratch needs `ANTHROPIC_API_KEY` (the repo-root `.env` is the
gitignored home for it) and costs ~$40 (the whole-split totals above):

```bash
for m in claude-fable-5-1 claude-fable-5; do
  python scripts/model_comparison/compare.py benchmark/annapolis --models claude:$m --claude-serving-path anthropic --claude-effort low
done
```

Check reachability first — `python scripts/model_comparison/probe_claude_models.py
--serving-path anthropic` — because a key with no credit balance authenticates and then
fails every call with a 400 that says so.

### Reproducing these four legs, and one gap in the record

The detections are committed, so the table above can be re-derived by anyone with a clone
and nothing else:

```bash
pytest -q tests/test_claude_published_legs.py     # recompute the table from committed files
```

Re-exporting them from a `.model_cache` that produced them needs the leg's settings, because
**one model id is several legs here** — effort is part of the cache signature, and both
effort levels of `claude-sonnet-5` would otherwise write the same filename. Hence
`--publish-as`, which names the published file without touching the cache label:

```bash
for m in claude-sonnet-5 claude-opus-5; do for e in low high; do
  python scripts/analysis/export_model_cache.py --splits annapolis \
      --models claude:$m --claude-effort $e --publish-as $m-effort-$e
  python scripts/analysis/export_model_cache.py --verify --splits annapolis \
      --models claude:$m --claude-effort $e --publish-as $m-effort-$e
done; done

# and the eleven-split opus/low leg (#139, #151), which takes no --splits:
python scripts/analysis/export_model_cache.py --verify \
    --models claude:claude-opus-5 --claude-effort low \
    --publish-as claude-opus-5-effort-low
# -> on a cache holding every run, "11 pair(s): published detections score IDENTICALLY
#    to the cache". Nobody has run it on such a cache: observed 2026-09-17 on the desktop
#    cache, which never held the two Laurens runs, it printed "compared 9" and flagged
#    both Laurens files as "NOTHING was compared" -- unverified, not verified.
```

**The gap, stated plainly: the four original legs' token counts were never written to
`analysis_out/usage_log.jsonl`.** The $28.82 total and the per-leg costs in the table above
come from the runs' console output, not from a committed record. A re-run cannot back-fill
them — the detections are cached, so a repeat run makes zero API calls and has no usage to
report. Only the 2026-08-18 single-panorama re-run ($0.03) is in the log.

**Recovered from Cloud Monitoring, 2026-08-19 — this paragraph previously said the counts
"cannot be recovered", and that was wrong.** A re-run cannot back-fill them, but the
server-side metrics can: `vertex_usage.py --days 7` returns billed tokens per model per day,
and the #122 legs are four days inside the ~6-week retention window. They ran 2026-08-15 and
land in the row labelled 2026-08-16, because each row is a 24 h window ending at the query's
time of day rather than a calendar day.

| model (both efforts, one row each) | input | output | billed | console figures |
|---|---:|---:|---:|---:|
| `claude-opus-5` (low + high) | 3,058,702 | 247,222 | **$21.47** | $8.94 + $12.46 = $21.40 |
| `claude-sonnet-5` (low + high) | 3,300,368 | 118,471 | **$7.79** | $3.60 + $3.82 = $7.42 |
| **total** | | | **$29.26** | **$28.82** |

So the console numbers were right to within **1.5%**, and the table above stands as
published. Two things this changes, and one it does not:

- **The Opus per-leg split is recoverable too, at minute resolution.** The daily row is
  per model, so `low` and `high` land in one number — but the metric can be aligned to 60 s
  instead of 86,400 s, and the two legs leave different traces. `vertex_effort_split.py`
  does this and confirms the console figures **to 0.1%**; the working is below.
- **The Sonnet split is not recoverable, and the tool says so rather than guessing.**
  Whether a per-effort split survives depends on whether effort actually changed the
  model's behaviour, which makes this a property of the *result*, not of the telemetry.
- **The method validated itself against the one leg that did log.** The 2026-08-18 Sonnet
  re-run appears in monitoring as 12,594 input / 480 output — token-for-token identical to
  its committed `usage_log.jsonl` record. Layer 3 reproducing layer 1 exactly, on the one
  case where both exist, is what makes the recovered figures above trustworthy.
- **It does not make the loss cheap.** Recovery worked because someone looked within six
  weeks. Past that window this paragraph's original claim becomes true retroactively.

#### Splitting a two-leg day by effort

Cloud Monitoring has **no `effort` label** — effort is a request parameter and never
reaches the metric, whose labels are `type`, `request_type`, `shared_request_type`,
`source`, `explicit_caching` plus the resource's `model_user_id` / `model_version_id` /
`publisher` / `location`. The daily alignment is a *query* choice, though, not a property
of the data, so the lever is time plus two facts this repo already holds:

1. **Input is deterministic** — 12,186 tokens per Opus panorama (6 views × 2,031). Total
   input therefore pins the pano count exactly: the 08-15 Opus day is **251.00 panos** —
   250 leg panos plus one panorama's worth of input (12,186 tokens, about $0.06) whose
   origin is not in the record. The only single-panorama re-run in
   `analysis_out/usage_log.jsonl` is Sonnet's, on 08-18, so it is not that; the likeliest
   source is a smoke call or a 404 retry from the #122 enablement window (the "12/12
   identical calls … 3 of 5 panos 404'd" measurement above, which did not name a model).
   The geometric split drops it from both legs, which is why the two anchors sum to
   $21.41 against the day's $21.47. The input half of the split needs no inference at all.
2. **Effort bills as output, not input.** A high-effort leg has a higher output/input ratio
   *and* a lower throughput, so when the fast leg finishes, both change at once.

That leaves one unknown — how the output divides — and the minute series shows exactly the
predicted shape. The legs ran **concurrently**, not back to back: throughput holds at ~5
panos/min until **18:32 UTC**, then drops **2.54×** to ~1.7 while the output ratio doubles
(0.0675 → 0.1203). That is the `low` leg finishing and leaving `high` running alone.

| anchor | low effort | high effort | sum |
|---|---:|---:|---:|
| tail ratio = pure high (0.1203) | $9.21 | $12.20 | $21.41 |
| low ratio = 0.0349, measured on the 984-pano #139 leg | **$8.95** | **$12.47** | $21.41 |
| **console output, recorded at run time** | **$8.94** | **$12.46** | $21.40 |

**The rate-anchored solve reproduces the console figures to 0.1%, from a completely
independent source.** The anchor was taken from the #139 leg's measured output rate before
either number was compared, so the agreement is a check, not a fit. Quote **$8.94 / $12.46**
— the run-time record — and treat this as the corroboration that they are right.

```bash
python scripts/analysis/vertex_effort_split.py --model claude-opus-5 \
    --start 2026-08-15T17:00:00Z --end 2026-08-15T21:30:00Z \
    --per-pano-input 12186 --anchor-low-ratio 0.034908 \
    --save-series docs/data/vertex_minute_series/claude-opus-5_2026-08-15.json
```

**The same command on `claude-sonnet-5` refuses to answer, and that is the more
transferable result.** Sonnet's ratio is flat across its whole run — throughput drops only
1.78× and the ratio moves the *wrong way* (0.0363 → 0.0273) — so there is no second
component to find and the script prints `NOT SEPARABLE`. (First published as 1.63× and
0.0365 → 0.0281: the changepoint search stopped one position short of the last full window,
which is exactly where this series' largest drop sits, so the cut landed a minute early at
17:35 instead of 17:36. Fixed 2026-09-17; the verdict does not move.) The reason is in the result table
above: Sonnet's high leg spent **17,820** thinking tokens against Opus's **127,227**, so the
dial that this method reads barely moved. A mixture solver run on that series returns "high
effort cost less than low", which is false; the guard exists because the wrong answer is the
plausible-looking one. **A per-effort split is recoverable exactly when effort changed the
model enough to be worth splitting** — the telemetry is not the limiting factor.

**The minute series is committed, so this section no longer has an expiry date.**
Everything above was read out of telemetry with ~6 weeks of retention: the
2026-08-15 series would have aged out around **2026-09-26** and the 2026-08-18 day
around **2026-09-29**, after which nobody, with or without access to the project,
could re-derive a number in it. `--save-series` writes the fetched rows to JSON and
`--from-series` replays one, which needs no credentials, no project and no network:

```bash
python scripts/analysis/vertex_effort_split.py --model claude-opus-5 \
    --from-series docs/data/vertex_minute_series/claude-opus-5_2026-08-15.json \
    --per-pano-input 12186 --anchor-low-ratio 0.034908
python scripts/analysis/vertex_effort_split.py --model claude-sonnet-5 \
    --from-series docs/data/vertex_minute_series/claude-sonnet-5_2026-08-15.json
```

Four snapshots are committed under `docs/data/vertex_minute_series/`, all fetched
2026-09-03, and replaying them reproduces every figure published above exactly:

| file | window (UTC) | active minutes | input | output |
|---|---|---:|---:|---:|
| `claude-opus-5_2026-08-15.json` | 08-15 17:00-21:30, both effort legs | 76 | 3,058,702 | 247,222 |
| `claude-sonnet-5_2026-08-15.json` | 08-15 12:00-21:30, both effort legs | 55 | 3,300,368 | 118,470 |
| `claude-opus-5_2026-08-18.json` | 08-18 00:00 - 08-19 12:00, the #139 leg | 83 | 11,988,993 | 418,503 |
| `vertex_usage_daily_2026-09-03.json` | the daily rows, 25-day lookback | - | - | - |

Four details worth knowing before re-running any of it. The Sonnet window is wider
than the Opus one because that leg started before 17:00 — the narrower window clips
it to 3,157,769 input and moves the head ratio to 0.0374, which is why the figures
quoted above need the wide one. Sonnet's output is one token under the daily row's
118,471, and the cause is not identified: the 60 s deltas over a window that holds the
whole run should sum to the daily delta (the Opus series matches its row to the token),
and the fetch that wrote these files kept only minutes with input tokens, so a minute
holding a single output token and no input — a response finishing after its request
was counted, the shape of the 2-token smoke minutes at 16:13–16:14 — would have been
dropped before the file was saved. That filter is gone (`minute_rows` keeps every
minute with any tokens), but the committed series were fetched under it, so a re-fetch
inside the retention window could carry one more row than these do. The daily-row
snapshot, written by `vertex_usage.py --save-rows`, carries every token type including
the three cache buckets that are zero here: a snapshot that quietly dropped a billed
bucket would be worse than no snapshot. Only `fetched_utc` moves between regenerations;
the rows are byte-stable, and `tests/test_vertex_effort_split.py` asserts that for all
four files — `test_the_daily_snapshot_backs_the_published_cost_table` opens the daily
file, pins the four Claude rows, prices them to the $21.47 / $7.79 / $70.41 / $0.03 in
the table above and round-trips it through `write_json`.

Two guards now stand where that went wrong, and the order matters. `compare.py` **refuses to
start** a paid leg under `--usage-log none` (override: `--allow-unrecorded-spend`), which is
the check that fires while the money is still unspent; `report_usage` still warns loudly at
the end of a leg that logged nothing, for the case where the log path existed but could not
be written. A warning after the fact would still have been worth having: everything above was
reconstructed five days late, and the only reason it worked is that nobody waited six weeks.
What the reconstruction cannot give you is a *guarantee* — Opus separated because its effort
dial moved 127k thinking tokens, Sonnet's did not separate at all, and which case you are in
is not knowable until after the money is spent. Layer 1 is the only layer that always works.

