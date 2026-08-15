"""Probe GSV pano availability AT THE RECORDS the street sheet will render
(issue #103).

The street-level analogue of ``probe_basemap_at_sites.py``, inheriting its
lesson and fixing its flaw. The lesson: probe the sample, never one point —
every basemap failure on #96 was *somewhere else in the city*. The flaw: that
probe samples the whole inventory, so with a filtered frame (Denver's
``UPDATE_STATUS=NC``) it does not actually hit the sheet's sites at the same
seed. This one takes ``--sites-from-verdicts`` and probes **exactly** the
records a built aerial sheet reviewed — which for the Denver pilot is also the
list the street sheet will render.

Per site it runs the SAME search and the SAME pick rule the sheet will use
(``street_review_sheet.choose_pano`` — one definition, imported), so its
headline number — the **pick rate** — is not a proxy for whether the sheet
will build: it is the sheet's own dry run, minus the pixels. Reported:

* **coverage** — sites with >=1 panorama at all;
* **pick rate** — sites where the rule finds an eligible pano (in the 4-30 m
  band, captured on/after the record's date);
* **date coverage** — sites with any pano postdating the record (#103's
  argument 4: temporal matching becomes per-record);
* chosen-pano **range and capture-year distributions**, so the review's
  geometry is known before a reviewer sees it;
* per-site failures WITH REASONS — a drop count is a claim about the fetcher
  until it is checked against the sample (§5h's Charlotte lesson).

Searches land in the sheet's own on-disk cache (``gsv_cache/search``), so the
requests this spends are requests the build no longer needs.

    python scripts/analysis/probe_panos_at_sites.py \
        --sites-from-verdicts analysis_out/review_denver-co/verdicts.json \
        --inventory data/inventories/denver-co-2026-07-31.jsonl.gz \
        --date-field CREATEDATE --city denver-co \
        --json analysis_out/probe_panos_denver-co.json

Pure logic (aggregation, the pick rule) is unit-tested without network; the
search itself is the only networked step and every failure of it is recorded,
never folded into a count.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from street_review_sheet import (  # noqa: E402
    RANGE_BAND_M, cached_search, choose_pano, load_sites_from_verdicts)
from inventory_review_sheet import load_inventory  # noqa: E402
from temporal_gap import SENTINEL_YMS, parse_ym  # noqa: E402


def record_ym_of(row, date_field):
    """The record's (year, month) through the ONE date parser, or None."""
    if not date_field:
        return None
    ym = parse_ym(row.get(date_field))
    return None if (ym is None or ym in SENTINEL_YMS) else ym


def probe_site(site, row, date_field, band_m, search_dir, sleep_s):
    """One site's probe result. Networked only through ``cached_search``."""
    rec_ym = record_ym_of(row, date_field)
    out = {"id": site["id"], "stratum": site.get("stratum"),
           "record_ym": None if rec_ym is None else list(rec_ym)}
    try:
        cands = cached_search(site["lat"], site["lon"], search_dir, sleep_s=sleep_s)
    except Exception as exc:                                      # noqa: BLE001
        out.update(status="search_failed",
                   detail="{}: {}".format(type(exc).__name__, exc))
        return out

    chosen, status, stats = choose_pano(cands, site["lat"], site["lon"],
                                        rec_ym, band_m=band_m)
    dated_after = sum(
        1 for c in cands
        if rec_ym is not None and (parse_ym(c.get("date")) or (0, 0)) >= rec_ym)
    out.update(status=status, **stats, n_dated_after_record=dated_after)
    if chosen is not None:
        out.update(chosen_pano=chosen["pano_id"], chosen_date=chosen.get("date"),
                   chosen_range_m=chosen["range_m"])
    return out


def summarise(results):
    """Aggregate the per-site rows into the numbers the build decision needs.

    Pure. Every rate's denominator is all probed sites; failures stay listed
    individually beside the aggregate, so the aggregate can be audited against
    the sample.
    """
    n = len(results)
    picked = [r for r in results if r["status"] == "ok"]
    ranges = sorted(r["chosen_range_m"] for r in picked)
    years = {}
    for r in picked:
        ym = parse_ym(r.get("chosen_date"))
        y = ym[0] if ym else None
        years[str(y)] = years.get(str(y), 0) + 1
    statuses = {}
    for r in results:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1

    def pct(k):
        return round(k / n, 4) if n else None

    mid = len(ranges) // 2
    return {
        "n_sites": n,
        "status_counts": statuses,
        "coverage": pct(sum(1 for r in results if r.get("n_panos", 0) > 0)),
        "pick_rate": pct(len(picked)),
        "date_coverage": pct(sum(
            1 for r in results
            if r.get("record_ym") is None or r.get("n_dated_after_record", 0) > 0)),
        "chosen_range_m": None if not ranges else {
            "min": ranges[0],
            "median": (ranges[mid] if len(ranges) % 2
                       else round((ranges[mid - 1] + ranges[mid]) / 2, 2)),
            "max": ranges[-1]},
        "chosen_year_hist": dict(sorted(years.items())),
        "failures": [r for r in results if r["status"] != "ok"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sites-from-verdicts", required=True,
                    help="a built aerial sheet's verdicts.json — probe ITS sites")
    ap.add_argument("--inventory", required=True,
                    help="frozen snapshot, for the record dates")
    ap.add_argument("--city", required=True,
                    help="names the shared gsv_cache dir (review_<city>-gsv)")
    ap.add_argument("--id-field", default="OBJECTID")
    ap.add_argument("--date-field", default=None)
    ap.add_argument("--band-min", type=float, default=RANGE_BAND_M[0])
    ap.add_argument("--band-max", type=float, default=RANGE_BAND_M[1])
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", default=None,
                    help="write the full payload here (commit it — the probe "
                         "IS the record that the sample was checked)")
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args(argv)

    sites, source = load_sites_from_verdicts(args.sites_from_verdicts)
    if args.limit:
        sites = sites[:args.limit]
    rows = load_inventory(args.inventory)
    by_id = {str(r.get(args.id_field)): r for r in rows}
    missing = [s["id"] for s in sites if s["id"] not in by_id]
    if missing:
        ap.error("{} site ids not in the inventory (first: {})".format(
            len(missing), missing[:3]))

    # The sheet's own cache dir, so every search spent here is one the build
    # no longer pays for.
    search_dir = os.path.join(args.out_dir, "review_{}-gsv".format(args.city),
                              "gsv_cache", "search")
    band = (args.band_min, args.band_max)

    results = []
    for k, site in enumerate(sites):
        r = probe_site(site, by_id[site["id"]], args.date_field, band,
                       search_dir, args.sleep)
        results.append(r)
        print("  [{:>3}/{}] {} {} {}".format(
            k + 1, len(sites), r["id"], r["status"],
            "pano {} {} at {} m".format(r.get("chosen_pano"), r.get("chosen_date"),
                                        r.get("chosen_range_m"))
            if r["status"] == "ok" else r.get("detail", "")))

    s = summarise(results)
    print("\nsites {}  coverage {}  pick rate {}  date coverage {}".format(
        s["n_sites"], s["coverage"], s["pick_rate"], s["date_coverage"]))
    print("chosen range {}  years {}".format(s["chosen_range_m"], s["chosen_year_hist"]))
    if s["failures"]:
        print("failures ({}):".format(len(s["failures"])))
        for f in s["failures"]:
            print("  {} {} {}".format(f["id"], f["status"], f.get("detail", "")))

    payload = {"sites_source": source, "inventory": os.path.basename(args.inventory),
               "date_field": args.date_field, "band_m": list(band),
               "summary": {k: v for k, v in s.items() if k != "failures"},
               "sites": results}
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        print("wrote {}".format(args.json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
