"""Actual Vertex AI token usage for a Google Cloud project, from Cloud Monitoring.

The harness records what each run spent as it happens (compare.py --usage-log),
but that only covers runs made after the instrumentation existed, and a local
log can be lost. This script recovers the ground truth server-side: Cloud
Monitoring's publisher-model metrics count every billed token per model id, at
daily granularity, going back ~6 weeks (Google metric retention). It priced the
entire Gemini history of this project retroactively on 2026-08-15 — see the
"Cost accounting" section of docs/model_comparison.md for those numbers.

Read-only. Needs ADC with access to the project (`gcloud auth
application-default login`) — the same credentials the Gemini legs run under.

    python scripts/analysis/vertex_usage.py                  # 30 days, project rampnet
    python scripts/analysis/vertex_usage.py --days 42 --project rampnet

Caveat: each daily row is a 24 h window ending at the query time-of-day (UTC),
NOT a calendar day — a leg run on the evening of the 14th lands in the row
labeled the 15th. Attribute rows to legs by the run record, not by eye.
"""
import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_comparison"))
from pricing import estimate_cost, price_for  # noqa: E402

TOKEN_METRIC = "aiplatform.googleapis.com/publisher/online_serving/token_count"


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
    r = requests.get(
        f"https://monitoring.googleapis.com/v3/projects/{project}/timeSeries",
        params=params, headers=headers, timeout=60)
    if r.status_code != 200:
        raise SystemExit(f"Cloud Monitoring query failed ({r.status_code}): "
                         f"{r.text[:500]}")
    return r.json().get("timeSeries", [])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--project", default="rampnet")
    ap.add_argument("--days", type=float, default=30,
                    help="Lookback window (metric retention is ~6 weeks).")
    ap.add_argument("--min-tokens", type=int, default=1000,
                    help="Hide rows below this many input tokens (smoke-test noise).")
    args = ap.parse_args()

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

    print(f"{'window end':12s} {'model':26s} {'input':>14s} {'output':>12s}")
    for (day, model) in sorted(daily):
        t = daily[(day, model)]
        if t.get("input", 0) < args.min_tokens:
            continue
        print(f"{day:12s} {model:26s} {t.get('input', 0):14,.0f} "
              f"{t.get('output', 0):12,.0f}")

    print(f"\n== totals, last {args.days:g} days ==")
    grand = 0.0
    unpriced = []
    for model in sorted(totals):
        t = totals[model]
        if t.get("input", 0) < args.min_tokens:
            continue
        cost = estimate_cost(model, t.get("input", 0), t.get("output", 0))
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
    print("\nEstimates only; the billing console is authoritative for spend.")


if __name__ == "__main__":
    main()
