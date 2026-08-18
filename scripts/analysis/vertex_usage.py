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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_comparison"))
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


if __name__ == "__main__":
    main()
