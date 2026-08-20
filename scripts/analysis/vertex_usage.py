"""Actual Vertex AI token usage for a Google Cloud project, from Cloud Monitoring.

The harness records what each run spent as it happens (compare.py --usage-log),
but that only covers runs made after the instrumentation existed, and a local
log can be lost. This script recovers the ground truth server-side: Cloud
Monitoring's publisher-model metrics count every billed token per model id, at
daily granularity, going back ~6 weeks (Google metric retention). It priced the
entire Gemini history of this project retroactively on 2026-08-15 — see the
"Cost accounting" section of docs/model_comparison.md for those numbers.

Read-only. Needs ADC with access to the project (`gcloud auth
application-default login`) — the same credentials the Gemini legs run under, and
the same `GOOGLE_CLOUD_PROJECT` (environment or repo-root `.env`) they read.

    python scripts/analysis/vertex_usage.py                  # 30 days, $GOOGLE_CLOUD_PROJECT
    python scripts/analysis/vertex_usage.py --days 42 --project my-project

**Replication note:** this reads one specific cloud project's billing telemetry, so
only someone with access to that project can re-derive its output. The numbers it
produced are transcribed into docs/model_comparison.md; the per-run token counts in
analysis_out/usage_log.jsonl are the committed, checkable half.

Caveat: each daily row is a 24 h window ending at the query time-of-day (UTC),
NOT a calendar day — a leg run on the evening of the 14th lands in the row
labeled the 15th. Attribute rows to legs by the run record, not by eye.
"""
import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_comparison"))
from rampnet import ledger  # noqa: E402
from pricing import estimate_cost, price_for  # noqa: E402

TOKEN_METRIC = "aiplatform.googleapis.com/publisher/online_serving/token_count"
# The only `type` label values this metric is known to carry. Anything else is
# tokens we would neither print nor price, so it gets reported rather than dropped.
KNOWN_TOKEN_TYPES = ("input", "output")


def _load_dotenv():
    """Reuse compare.py's .env loader so both halves of the harness read the same
    credentials file. Imported inside the function (the export_model_cache idiom)
    so the module still imports without the detector stack on the path."""
    try:
        from compare import load_dotenv
    except ImportError:
        return
    load_dotenv(str(REPO))


def fetch_token_series(project, days):
    try:
        import google.auth
        import google.auth.transport.requests
        import requests
    except ImportError as e:
        raise SystemExit(f"needs google-auth + requests (pip install -r "
                         f"requirements-vlm.txt): {e}")
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())
    headers = {"Authorization": f"Bearer {creds.token}",
               "x-goog-user-project": project}
    end = datetime.now(timezone.utc)
    params = {
        "filter": f'metric.type = "{TOKEN_METRIC}"',
        "interval.startTime": (end - timedelta(days=days)).isoformat(),
        "interval.endTime": end.isoformat(),
        "aggregation.alignmentPeriod": "86400s",
        "aggregation.perSeriesAligner": "ALIGN_DELTA",
        "aggregation.crossSeriesReducer": "REDUCE_SUM",
        "aggregation.groupByFields": ["resource.labels.model_user_id",
                                      "metric.labels.type"],
        "pageSize": 1000,
    }
    # Follow nextPageToken. A dropped page is a SILENT UNDERCOUNT in the one tool
    # whose job is server-side ground truth, and "the total came back low" has no
    # symptom a reader could notice.
    url = f"https://monitoring.googleapis.com/v3/projects/{project}/timeSeries"
    series, token, pages = [], None, 0
    while True:
        page_params = dict(params, **({"pageToken": token} if token else {}))
        r = requests.get(url, params=page_params, headers=headers, timeout=60)
        if r.status_code != 200:
            raise SystemExit(f"Cloud Monitoring query failed ({r.status_code}): "
                             f"{r.text[:500]}")
        body = r.json()
        series.extend(body.get("timeSeries", []))
        pages += 1
        token = body.get("nextPageToken")
        if not token:
            break
        if pages >= 50:   # runaway guard; say so rather than truncating quietly
            raise SystemExit(f"stopped after {pages} pages with more remaining — "
                             f"narrow --days and re-run, or the totals would be partial")
    if pages > 1:
        print(f"(fetched {len(series)} time series across {pages} pages)")
    return series


def ledger_totals_by_model(rows, since=None):
    """Per-model token totals from usage_log.jsonl rows, for reconciliation.

    ``since`` is an ISO date (YYYY-MM-DD); rows stamped earlier are skipped so the
    comparison covers the same window the metric query did. Free legs carry no
    token keys and are skipped: they have nothing to reconcile against a bill.

    **Recovered rows are skipped too, and that exclusion is the point.** A recovered
    row was read off this very bill (``rampnet.ledger.RECOVERED``), so counting it as
    "logged" would compare the bill against itself and report ``ok`` for the exact
    gap this function exists to find — turning the one check that catches a silent
    no-write into a check that cannot."""
    out = defaultdict(lambda: defaultdict(float))
    for rec in rows or []:
        if ledger.row_kind(rec) == ledger.RECOVERED:
            continue
        if not rec.get("input_tokens") and not rec.get("output_tokens"):
            continue
        if since and (rec.get("ts") or "")[:10] < since:
            continue
        model = rec.get("model_id") or "?"
        out[model]["input"] += rec.get("input_tokens") or 0
        out[model]["output"] += rec.get("output_tokens") or 0
        out[model]["rows"] += 1
    return {m: dict(v) for m, v in out.items()}


def reconcile(billed, logged, tolerance=0.02):
    """Compare per-model billed tokens against what the ledger says we recorded.

    This is the only check that catches a **silent no-write** — a paid leg that ran,
    billed, and left no row. #119's guard cannot: it proves a log path was accepted,
    not that the file survived (#139 lost $70.41 that way). A missing row is only
    recoverable while the metric is still retained, ~6 weeks, so the check has to be
    run close to the run.

    Returns one dict per model with the billed and logged totals and a verdict.
    Deliberately one-sided in what it treats as alarming: ledger > billed is odd but
    harmless (a re-run, a mis-stamped window), while billed > ledger is spend with no
    record, which is the failure this exists for."""
    rows = []
    for model in sorted(set(billed) | set(logged)):
        b = billed.get(model, {})
        lg = logged.get(model, {})
        b_in, l_in = b.get("input", 0), lg.get("input", 0)
        if not l_in and b_in:
            verdict = "MISSING - billed, nothing logged"
        elif b_in and abs(b_in - l_in) / b_in > tolerance:
            verdict = ("UNDER — ledger short" if l_in < b_in
                       else "over — ledger exceeds billed")
        else:
            verdict = "ok"
        rows.append({
            "model_id": model, "billed_input": b_in, "billed_output": b.get("output", 0),
            "logged_input": l_in, "logged_output": lg.get("output", 0),
            "logged_rows": int(lg.get("rows", 0)), "verdict": verdict,
        })
    return rows


def print_reconciliation(rows):
    print(f"\n== reconciliation: Cloud Monitoring vs analysis_out/usage_log.jsonl ==")
    print(f"{'model':26s} {'billed in':>14s} {'logged in':>14s} {'rows':>5s}  verdict")
    worst = []
    for r in rows:
        print(f"  {r['model_id']:24s} {r['billed_input']:14,.0f} "
              f"{r['logged_input']:14,.0f} {r['logged_rows']:5d}  {r['verdict']}")
        if r["verdict"].startswith(("MISSING", "UNDER")):
            worst.append(r)
    if worst:
        gap = sum(r["billed_input"] - r["logged_input"] for r in worst)
        print(f"\n  {len(worst)} model(s) billed more than the ledger records "
              f"({gap:,.0f} input tokens unaccounted for).")
        print("  A missing layer-1 row is an emergency with a deadline: this metric "
              "retains ~6 weeks,\n  after which the number is unrecoverable at any "
              "price. Recovery is per-model per-DAY,\n  so per-split attribution is "
              "already gone - record the total against the run it came from.")
    else:
        print("\n  every billed model has a ledger row within tolerance.")
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    # Same env var the Gemini legs run under (detectors.py GeminiDetector, and the
    # .env setup in docs/model_comparison.md), so following those instructions is
    # enough to run this too. Defaulting to a hardcoded project id would send
    # someone else's ADC at a project that isn't theirs and 403 with no hint why.
    ap.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                    help="GCP project to query (default: $GOOGLE_CLOUD_PROJECT, "
                         "which compare.py also reads from a repo-root .env).")
    ap.add_argument("--days", type=float, default=30,
                    help="Lookback window (metric retention is ~6 weeks).")
    ap.add_argument("--reconcile", action="store_true",
                    help="Compare these billed totals against what "
                         "analysis_out/usage_log.jsonl recorded, per model. The only "
                         "check that catches a paid leg which ran, billed, and left "
                         "no row (#139, #143).")
    ap.add_argument("--usage-log", help="Ledger to reconcile against (default: the "
                                        "committed analysis_out/usage_log.jsonl in "
                                        "the main checkout).")
    ap.add_argument("--min-tokens", type=int, default=1000,
                    help="Hide rows below this many input tokens (smoke-test noise). "
                         "Suppressed rows are counted and their cost reported.")
    args = ap.parse_args()

    _load_dotenv()
    if not args.project:
        args.project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not args.project:
        raise SystemExit(
            "no project: pass --project, or set GOOGLE_CLOUD_PROJECT in the "
            "environment or a repo-root .env (same variable the Gemini legs use).")

    daily = defaultdict(lambda: defaultdict(float))   # (day, model) -> type -> tokens
    totals = defaultdict(lambda: defaultdict(float))  # model -> type -> tokens
    for ts in fetch_token_series(args.project, args.days):
        model = ts["resource"]["labels"].get("model_user_id", "?")
        typ = ts["metric"]["labels"].get("type", "?")
        for p in ts.get("points", []):
            val = (int(p["value"].get("int64Value", 0))
                   + float(p["value"].get("doubleValue", 0)))
            daily[(p["interval"]["endTime"][:10], model)][typ] += val
            totals[model][typ] += val

    # Only `input` and `output` are printed and priced below. If Vertex ever adds a
    # third bucket (cached input, a renamed label), those tokens would be billed and
    # invisible here -- so surface it instead of letting the total read complete.
    seen_types = {t for m in totals.values() for t in m}
    unknown = sorted(seen_types - set(KNOWN_TOKEN_TYPES))
    if unknown:
        extra = sum(m[t] for m in totals.values() for t in unknown)
        print(f"WARNING: unpriced token type(s) {', '.join(unknown)} carrying "
              f"{extra:,.0f} tokens are excluded from every figure below. "
              f"Price them in pricing.py or the total understates real spend.\n")

    print(f"{'window end':12s} {'model':26s} {'input':>14s} {'output':>12s}")
    for (day, model) in sorted(daily):
        t = daily[(day, model)]
        if t.get("input", 0) < args.min_tokens:
            continue
        print(f"{day:12s} {model:26s} {t.get('input', 0):14,.0f} "
              f"{t.get('output', 0):12,.0f}")

    print(f"\n== totals, last {args.days:g} days ==")
    grand = 0.0
    unpriced, suppressed, suppressed_cost = [], [], 0.0
    for model in sorted(totals):
        t = totals[model]
        cost = estimate_cost(model, t.get("input", 0), t.get("output", 0))
        if t.get("input", 0) < args.min_tokens:
            # Below the noise floor, but a cost ledger must not quietly drop rows:
            # tally them and report the total they carry.
            suppressed.append(model)
            suppressed_cost += cost or 0.0
            continue
        if cost is None:
            unpriced.append(model)
            cost_txt = "   (no price in pricing.py)"
        else:
            grand += cost
            cost_txt = f"   ~=${cost:7.2f} (as of {price_for(model)['as_of']})"
        print(f"  {model:26s} {t.get('input', 0):14,.0f} in "
              f"{t.get('output', 0):12,.0f} out{cost_txt}")
    print(f"  {'TOTAL (priced models)':26s} {'':14s}    {'':12s}    ~=${grand:7.2f}")
    if unpriced:
        print(f"  unpriced models excluded from total: {', '.join(unpriced)} — "
              "add a verified entry to scripts/model_comparison/pricing.py")
    if suppressed:
        print(f"  {len(suppressed)} model(s) below --min-tokens {args.min_tokens:,} "
              f"excluded (~${suppressed_cost:.2f}): {', '.join(suppressed)}")
    print("\nEstimates only; the billing console is authoritative for spend.")

    if args.reconcile:
        log_path = args.usage_log or str(
            ledger.canonical_repo_root(REPO) / "analysis_out" / "usage_log.jsonl")
        rows = ledger.read_rows(log_path)
        if rows is None:
            print(f"\ncannot reconcile: no ledger at {os.path.abspath(log_path)}")
            return
        since = (datetime.now(timezone.utc)
                 - timedelta(days=args.days)).date().isoformat()
        print(f"(ledger: {os.path.abspath(log_path)}, rows stamped >= {since})")
        print_reconciliation(reconcile({m: dict(t) for m, t in totals.items()},
                                       ledger_totals_by_model(rows, since)))


if __name__ == "__main__":
    main()
