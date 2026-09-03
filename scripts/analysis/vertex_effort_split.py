"""Split one day's billed Vertex tokens between two legs of the same model (#139, #143).

``vertex_usage.py`` recovers spend **per model per day**, which is where the recovery
stops when two legs of one model ran the same day. That is exactly the #122 case: both
effort levels of ``claude-opus-5`` and of ``claude-sonnet-5`` ran on 2026-08-15, so the
daily row is a sum of two runs we would like to price separately.

Cloud Monitoring carries no ``effort`` label -- the labels on
``publisher/online_serving/token_count`` are ``type``, ``request_type``,
``shared_request_type``, ``source``, ``explicit_caching`` and the resource's
``model_user_id`` / ``model_version_id`` / ``publisher`` / ``location``. Effort is a
request parameter and never reaches the metric. So the only lever is **time**, at
minute resolution, plus two facts this repo already holds:

1. **Input is deterministic.** A pano is a fixed number of views at a fixed size, so
   input tokens per pano are constant (Opus: 12,186 = 6 x 2,031). Total input therefore
   pins the pano count exactly, and a two-leg day splits its input 50/50 by geometry
   with no inference at all.
2. **Effort shows up in output, not input.** Thinking bills as output, so a
   high-effort leg has a higher output/input ratio and a lower throughput than a
   low-effort leg of the same model.

That makes the day a two-component mixture with a known input split, and the only
unknown is how the output divides. This script solves it two ways and reports both,
because agreement between them is the whole basis for trusting the answer:

* **tail anchor** -- find the changepoint where throughput drops (the faster leg
  finishing, leaving the slower one running alone) and take the tail's ratio as the
  slow leg's pure ratio.
* **rate anchor** (``--anchor-low-ratio``) -- use an output/input ratio measured for
  the *same model at the same effort* on some other, cleanly-attributed run.

**It does not always work, and it says so.** Separability needs the effort dial to have
actually changed the model's behaviour. It did for Opus (127,227 thinking tokens across
the high leg, a 2.5x throughput drop, ratios 0.035 vs 0.127) and it did not for Sonnet
(17,820 thinking tokens, a flat ratio across the whole run). When the tail ratio is not
meaningfully above the head ratio there is no separation to find, and this prints
NOT SEPARABLE rather than a confident wrong number -- a mixture solver handed a flat
series will happily return "high effort cost less than low", which is the failure this
guard exists to catch.

Read-only. Needs the same ADC and project as ``vertex_usage.py``.

    python scripts/analysis/vertex_effort_split.py --model claude-opus-5 \
        --start 2026-08-15T17:00:00Z --end 2026-08-15T21:30:00Z \
        --per-pano-input 12186 --anchor-low-ratio 0.034908

**Replication note:** like ``vertex_usage.py`` this reads one cloud project's billing
telemetry, so only someone with access to that project can re-derive it, and Google's
metric retention is ~6 weeks. ``--save-series`` writes the fetched minute rows to a JSON
file and ``--from-series`` replays one, which is what takes the result off that clock:
the series committed under ``docs/data/vertex_minute_series/`` re-runs the whole analysis
with no cloud access at all, long after the metric has aged out.

    python scripts/analysis/vertex_effort_split.py --model claude-opus-5 \\
        --from-series docs/data/vertex_minute_series/claude-opus-5_2026-08-15.json \\
        --per-pano-input 12186 --anchor-low-ratio 0.034908

The numbers it produced are transcribed into ``docs/model_comparison.md`` section
"Reproducing these four legs".
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_comparison"))
from pricing import estimate_cost, price_for      # noqa: E402
from vertex_usage import write_json               # noqa: E402  (same snapshot format)

TOKEN_METRIC = "aiplatform.googleapis.com/publisher/online_serving/token_count"

#: A tail ratio must exceed the head ratio by this factor before the two legs are
#: called separable. Below it the series is flat and any split is fitting noise.
MIN_RATIO_LIFT = 1.25
#: Minutes dropped either side of the changepoint, which is a blend of both legs.
GUARD_MINUTES = 3


def _load_dotenv():
    """Same repo-root .env the rest of the harness reads.

    NOTE: REPO is derived from __file__, so running from a git worktree looks in the
    worktree, not the checkout that holds .env -- pass --project explicitly there.
    That is the read-side face of the #143 bug.
    """
    path = REPO / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def fetch_minute_series(project, model, start, end):
    """Minute-aligned (input, output) deltas for one model, oldest first."""
    try:
        import google.auth
        import google.auth.transport.requests
        import requests
    except ImportError as e:                              # pragma: no cover
        raise SystemExit(f"needs google-auth + requests (pip install -r "
                         f"requirements-vlm.txt): {e}")
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())
    headers = {"Authorization": f"Bearer {creds.token}",
               "x-goog-user-project": project}
    params = {
        "filter": f'metric.type = "{TOKEN_METRIC}" AND '
                  f'resource.labels.model_user_id = "{model}"',
        "interval.startTime": start,
        "interval.endTime": end,
        "aggregation.alignmentPeriod": "60s",
        "aggregation.perSeriesAligner": "ALIGN_DELTA",
        "aggregation.crossSeriesReducer": "REDUCE_SUM",
        "aggregation.groupByFields": ["metric.labels.type"],
        "pageSize": 2000,
    }
    url = f"https://monitoring.googleapis.com/v3/projects/{project}/timeSeries"
    series, token, pages = [], None, 0
    while True:
        page = dict(params, **({"pageToken": token} if token else {}))
        r = requests.get(url, params=page, headers=headers, timeout=90)
        if r.status_code != 200:
            raise SystemExit(f"Cloud Monitoring query failed ({r.status_code}): "
                             f"{r.text[:500]}")
        body = r.json()
        series.extend(body.get("timeSeries", []))
        pages += 1
        token = body.get("nextPageToken")
        if not token:
            break
        if pages >= 50:            # same runaway guard as vertex_usage.py
            raise SystemExit("stopped after 50 pages with more remaining — "
                             "narrow the window, or the totals would be partial")
    buckets = defaultdict(lambda: defaultdict(int))
    for s in series:
        ttype = s.get("metric", {}).get("labels", {}).get("type", "?")
        for pt in s.get("points", []):
            n = int(pt["value"].get("int64Value", 0) or 0)
            if n:
                buckets[pt["interval"]["endTime"]][ttype] += n
    return sorted((ts, d.get("input", 0), d.get("output", 0))
                  for ts, d in buckets.items() if d.get("input", 0))


def save_series(path, model, start, end, rows):
    """Write the fetched minute rows so the analysis outlives metric retention."""
    doc = {
        "model": model,
        "metric": TOKEN_METRIC,
        "alignment_period": "60s",
        "interval_start": start,
        "interval_end": end,
        "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "columns": ["end_time", "input_tokens", "output_tokens"],
        "rows": [[ts, int(i), int(o)] for ts, i, o in rows],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, doc, "rows")
    return doc


def load_series(path):
    """Replay a saved series. Returns (rows, model, start, end)."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [(ts, int(i), int(o)) for ts, i, o in doc["rows"]]
    return (rows, doc.get("model"), doc.get("interval_start"),
            doc.get("interval_end"))


def find_changepoint(rows, window=5):
    """Index of the largest sustained drop in throughput, and the drop factor."""
    best, best_drop = None, 0.0
    for i in range(window, len(rows) - window):
        before = sum(r[1] for r in rows[i - window:i]) / window
        after = sum(r[1] for r in rows[i:i + window]) / window
        if before and after and before / after > best_drop:
            best, best_drop = i, before / after
    return best, best_drop


def main():
    _load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                    help="GCP project (default: $GOOGLE_CLOUD_PROJECT / repo-root .env). "
                         "Pass explicitly when running from a git worktree.")
    ap.add_argument("--model", required=True, help="model_user_id, e.g. claude-opus-5")
    ap.add_argument("--start", help="RFC3339, e.g. 2026-08-15T17:00:00Z. Required "
                                    "unless --from-series is given.")
    ap.add_argument("--end", help="RFC3339. Required unless --from-series is given.")
    ap.add_argument("--save-series", metavar="PATH",
                    help="Write the fetched minute rows to PATH as JSON, so the "
                         "analysis survives the ~6-week metric retention.")
    ap.add_argument("--from-series", metavar="PATH",
                    help="Replay a saved series instead of querying Cloud "
                         "Monitoring. Needs no credentials and no project.")
    ap.add_argument("--per-pano-input", type=int, default=0,
                    help="Deterministic input tokens per panorama; pins the pano count.")
    ap.add_argument("--legs", type=int, default=2,
                    help="Legs of this model that ran in the window (default 2).")
    ap.add_argument("--anchor-low-ratio", type=float, default=0.0,
                    help="output/input ratio for the FAST leg, measured on a cleanly "
                         "attributed run of the same model at the same effort.")
    args = ap.parse_args()

    if args.from_series:
        rows, saved_model, start, end = load_series(args.from_series)
        # A series is a per-model file; replaying one under a different --model would
        # price the wrong rate card against it and say nothing.
        if saved_model and saved_model != args.model:
            raise SystemExit(f"{args.from_series} holds {saved_model}, not "
                             f"{args.model} — pass --model {saved_model}.")
        print(f"(replaying {args.from_series}: {len(rows)} minutes, "
              f"{start} -> {end}, no cloud query)")
    else:
        if not args.project:
            raise SystemExit("no project: pass --project, or set GOOGLE_CLOUD_PROJECT "
                             "in the environment or a repo-root .env.")
        if not (args.start and args.end):
            raise SystemExit("a cloud query needs --start and --end (or replay a "
                             "saved window with --from-series).")
        rows = fetch_minute_series(args.project, args.model, args.start, args.end)
        if args.save_series:
            save_series(args.save_series, args.model, args.start, args.end, rows)
            print(f"(wrote {len(rows)} minute rows to {args.save_series})")

    if len(rows) < 4 * GUARD_MINUTES:
        raise SystemExit(f"only {len(rows)} active minute(s) in the window — widen it")
    tin = sum(r[1] for r in rows)
    tout = sum(r[2] for r in rows)

    print(f"== {args.model}  {rows[0][0]} -> {rows[-1][0]}")
    print(f"   {len(rows)} active minutes, input {tin:,}, output {tout:,}, "
          f"blended ratio {tout / tin:.4f}")
    if args.per_pano_input:
        n = tin / args.per_pano_input
        # A non-integer pano count means the window is clipping a run or the
        # per-pano rate is wrong, and either way the 50/50 input split below is
        # unsound. Say so rather than dividing anyway.
        verdict = (f"{round(n)} exactly" if abs(n - round(n)) < 0.02 else
                   "NOT an integer -- the window or the rate is off")
        print(f"   panos {n:.2f} at {args.per_pano_input:,} input/pano ({verdict})")

    cut, drop = find_changepoint(rows)
    head, tail = rows[:cut - GUARD_MINUTES], rows[cut + GUARD_MINUTES:]
    hi, ho = sum(r[1] for r in head), sum(r[2] for r in head)
    ti, to = sum(r[1] for r in tail), sum(r[2] for r in tail)
    r_head, r_tail = ho / hi, to / ti
    print(f"   changepoint {rows[cut][0]}: throughput /{drop:.2f}, "
          f"output ratio {r_head:.4f} -> {r_tail:.4f}")

    if r_tail < r_head * MIN_RATIO_LIFT:
        print(f"\n   NOT SEPARABLE: the tail ratio is not {MIN_RATIO_LIFT}x the head's, "
              f"so there is no\n   second component to find. Either the legs did not "
              f"overlap the way this\n   assumes, or the effort dial did not move this "
              f"model's output enough to\n   leave a trace. Report the daily total and "
              f"say the split is unrecovered.")
        return 0

    # Input divides by geometry, not inference: every leg made one pass over the split.
    if args.per_pano_input:
        per_leg_in = (round(tin / args.per_pano_input) // args.legs) * args.per_pano_input
    else:
        per_leg_in = tin / args.legs
    print(f"\n   per-leg input (geometry, {args.legs} legs): {per_leg_in:,.0f}")

    # estimate_cost returns None for an id that is not in the verified rate card, and
    # None cannot be formatted as a dollar figure. Decide once, up front, so an
    # unpriced model reports its token split instead of raising inside report().
    priced = price_for(args.model) is not None
    if not priced:
        print(f"   (no verified price for {args.model}; token splits only — add a "
              f"verified entry to scripts/model_comparison/pricing.py)")

    def report(tag, out_slow):
        out_fast = tout - out_slow
        print(f"   [{tag}]")
        for label, out in (("low ", out_fast), ("high", out_slow)):
            cost = (f"  ${estimate_cost(args.model, per_leg_in, out):7.2f}"
                    if priced else "")
            print(f"      {label} effort: output {out:>10,.0f}  "
                  f"ratio {out / per_leg_in:.4f}{cost}")
        if priced:
            total = (estimate_cost(args.model, per_leg_in, out_fast)
                     + estimate_cost(args.model, per_leg_in, out_slow))
            print(f"      sum ${total:7.2f} vs billed day "
                  f"${estimate_cost(args.model, tin, tout):.2f}")

    report("tail anchor", r_tail * per_leg_in)
    if args.anchor_low_ratio:
        report(f"rate anchor (low ratio {args.anchor_low_ratio:.4f})",
               tout - args.anchor_low_ratio * per_leg_in)
    print("\n   Two anchors that disagree are two estimates, not one answer -- quote "
          "the spread.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
