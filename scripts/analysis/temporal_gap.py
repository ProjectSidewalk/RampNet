"""How far apart are a city's curb-ramp inventory and its street-level imagery?

Sourcing gate for a larger Stage 1 corpus (issues #59, #86). See
``docs/curb_ramp_data_sourcing.md`` §5a for why this is a gate at all.

Positional precision (paper Tab. 1) asks *is the coordinate on the ramp?*
This asks the independent question *did the ramp and the pixels exist at the same
time?* — and it is **not** the check fixed in #11. That one is an **ordering**
test: discard a panorama unless the ramp was installed strictly before the month
of capture (``generate_dataset_meta.py``). Ordering is not distance, and two
failure modes survive it, **with opposite signs**:

* **Unlabeled positives** — ramps built after the inventory snapshot are in the
  pixels but absent from the data, so no ordering check can see them. The target
  heatmap is *zero* at a real ramp.
* **Phantom labels** — ``TREAT_UNDATED_AS_PREDATING = True`` means every dateless
  record is *assumed* to predate the panorama, so where a city's undated fraction
  is high the filter is effectively off and un-built ramps become labels at empty
  pixels.

Both are estimated below. Neither is captured by a single "gap" number, so this
never reports one.

Needs no GPU, no network and no imagery — inventory dates and panorama capture
dates are both metadata, which is why this should run *before* the visual
precision assessment: it can disqualify a city in minutes rather than hours.

    python scripts/analysis/temporal_gap.py \
        --city "Washington, DC" \
        --inventory dc_ada_curb_ramp.json --inventory-date-field INSTALLDATE \
        --tracker-snapshot washington--district-of-columbia--...2024-04-11.csv.gz

**Imagery input.** ``--tracker-snapshot`` takes a Streetscape Tracker snapshot
(``D:\\Git\\gsv-tracker``; catalogue on makelab1 at
``/projects/makeabilitylab/streetscape-tracker/data/``), which records
``capture_date`` per panorama for GSV and Mapillary across ~1,200 cities. Its GSV
series is a *grid sample* spread uniformly over the city, whereas Stage 1 selects
panoramas within 10 m of ramp locations — i.e. at intersections. So this is a
**screening** signal, good for ranking cities, not the per-ramp Δt the pipeline
would actually see.

The core below is pure and unit-tested in ``tests/test_temporal_gap.py``; only
``load_*`` and ``main`` touch disk.
"""
import argparse
import csv
import gzip
import json
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))

# Null-date placeholders seen in the wild. Treated as undated, and reported
# separately so a city carrying genuine installs at these dates is not silently
# rewritten.
#   (2000, 1)  — combine_location_data.py substituted 2000-01-01 for any missing
#               install date, silently defeating the ordering filter (#11).
#   (1899, 12) — the spreadsheet/OLE zero date. Boston's entire CONST_DATE column
#               is "18991230", i.e. the field is present but carries no data.
SENTINEL_YMS = frozenset({(2000, 1), (1899, 12)})
SENTINEL_YM = (2000, 1)          # kept for the report header; see SENTINEL_YMS

# A single (year, month) holding this share of an inventory, at an implausible
# date, is a placeholder rather than a fact. Reported so unknown sentinels surface
# instead of being scored as real installs.
SENTINEL_SUSPECT_SHARE = 0.5
SENTINEL_SUSPECT_BEFORE_YEAR = 1950

# Mirrors TREAT_UNDATED_AS_PREDATING in generate_dataset_meta.py. Flipping it
# here does not change the pipeline — it shows what the pipeline's choice costs.
DEFAULT_UNDATED_PREDATES = True

# Years of install history used to estimate a city's build rate. Short enough to
# track current policy (post-settlement programmes ramp up sharply), long enough
# not to ride on one anomalous year.
DEFAULT_RATE_LOOKBACK_YEARS = 3


# --------------------------------------------------------------------------- #
# Pure core (no I/O) — unit-tested in tests/test_temporal_gap.py
# --------------------------------------------------------------------------- #
def parse_ym(value):
    """Any of the formats these sources use -> ``(year, month)``, else ``None``.

    Government inventories are not consistent: ArcGIS FeatureServer emits epoch
    **milliseconds**, Socrata emits ISO-8601 text, and some layers carry a bare
    year. Returning ``None`` for anything unparseable is what makes the undated
    fraction meaningful, so this never guesses.
    """
    if value is None or value == "":
        return None
    # ArcGIS epoch milliseconds, as an int or a numeric string. Negative = pre-1970,
    # which is real for old installs, so the sign is not a validity test.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _numeric_ym(value)
    text = str(value).strip()
    if not text:
        return None
    if text.lstrip("-").isdigit():
        return _numeric_ym(int(text))
    # ISO-8601 and the YYYY/MM/DD variants, with or without a time part.
    head = text.replace("/", "-").split("T")[0].split(" ")[0]
    parts = head.split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 and parts[1] else 1
    except (ValueError, IndexError):
        return None
    if not (1000 <= year <= 2999) or not (1 <= month <= 12):
        return None
    return (year, month)


def _numeric_ym(n):
    """Bare year, compact date, or epoch milliseconds — disambiguated by width.

    Every source picks a different encoding and they collide numerically, so the
    order here matters. Boston's ``CONST_DATE`` is the string ``"18991230"``: read
    as milliseconds that is 1970, which would silently turn a null placeholder
    into a plausible-looking install date.

    * 4 digits  ``2019``     -> year        (1000-2999 spans any install year)
    * 6 digits  ``201905``   -> YYYYMM
    * 8 digits  ``20190514`` -> YYYYMMDD
    * otherwise              -> epoch milliseconds

    Applied to ints and numeric strings alike, so ``2019`` and ``"2019"`` cannot
    disagree.
    """
    n = int(n)
    if 1000 <= n <= 2999:                                   # YYYY
        return (n, 1)
    if 100001 <= n <= 299912 and 1 <= n % 100 <= 12:        # YYYYMM
        return (n // 100, n % 100)
    if 10000101 <= n <= 29991231:                           # YYYYMMDD
        month = (n // 100) % 100
        if 1 <= month <= 12:
            return (n // 10000, month)
    return _from_epoch_ms(n)


def _from_epoch_ms(n):
    try:
        import datetime
        dt = datetime.datetime.fromtimestamp(int(n) / 1000.0, datetime.timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None
    return (dt.year, dt.month)


def build_hist(values, sentinels=SENTINEL_YMS):
    """``(Counter of (year, month), n_undated, n_sentinel)``.

    Sentinel values count as undated *and* are reported separately, because a
    high sentinel share means the source went through a lossy conversion rather
    than genuinely lacking dates — a different problem with a different fix.
    Pass ``sentinels=None`` for capture dates, where an old date is a real
    (ancient) panorama rather than a placeholder.
    """
    hist, n_undated, n_sentinel = Counter(), 0, 0
    known = frozenset(sentinels or ())
    for v in values:
        ym = parse_ym(v)
        if ym is None:
            n_undated += 1
        elif ym in known:
            n_sentinel += 1
            n_undated += 1
        else:
            hist[ym] += 1
    return hist, n_undated, n_sentinel


def suspected_sentinel(hist, share=SENTINEL_SUSPECT_SHARE,
                       before=SENTINEL_SUSPECT_BEFORE_YEAR):
    """An unrecognised placeholder masquerading as an install date, or ``None``.

    Every source invents its own null date, so an allow-list will always lag.
    A single implausibly-old value carrying most of an inventory is a placeholder,
    and scoring it as a real install would make a city look temporally pristine
    when it has no dates at all.
    """
    total = sum(hist.values())
    if not total:
        return None
    for ym, c in hist.items():
        if ym[0] < before and c / total >= share:
            return ym
    return None


def ordering_passes(install_ym, capture_ym, undated_predates=DEFAULT_UNDATED_PREDATES):
    """The pipeline's own check, reproduced.

    ``generate_dataset_meta.py`` keeps a (ramp, pano) pair only when the ramp was
    installed strictly before the **month** of capture, comparing ``(year, month)``
    tuples — the coupling bug fixed in #11.
    """
    if install_ym is None:
        return bool(undated_predates)
    return install_ym < capture_ym


def _weights(hist):
    """Histogram -> ``[(key, share)]`` summing to 1.0 (empty -> [])."""
    total = sum(hist.values())
    if not total:
        return []
    return [(k, c / total) for k, c in sorted(hist.items())]


def _ym_to_years(ym):
    return ym[0] + (ym[1] - 1) / 12.0


def discard_rate(install_hist, capture_hist, n_undated=0,
                 undated_predates=DEFAULT_UNDATED_PREDATES):
    """Share of (ramp, pano) pairs the ordering filter rejects.

    This is the filter's *data cost* — pairs thrown away because the ramp
    postdates the imagery. High is not automatically bad (it means the filter is
    working), but it bounds how much of a city's inventory is usable at all.
    """
    n_dated = sum(install_hist.values())
    total = n_dated + n_undated
    if not total or not sum(capture_hist.values()):
        return 0.0
    rejected = 0.0
    for c_ym, c_w in _weights(capture_hist):
        for i_ym, i_n in install_hist.items():
            if not ordering_passes(i_ym, c_ym):
                rejected += c_w * i_n
        if not undated_predates:
            rejected += c_w * n_undated
    return rejected / total


def phantom_rate(install_hist, capture_hist, n_undated,
                 undated_predates=DEFAULT_UNDATED_PREDATES,
                 existence_bound_ym=None):
    """Expected share of records that become labels at pixels with no ramp.

    A phantom is an undated record whose true install date is *after* the
    panorama's capture — admitted because ``TREAT_UNDATED_AS_PREDATING`` waves it
    through. Per-record truth is unknowable, so this assumes **undated records
    share the dated records' install distribution** and integrates the tail above
    each capture date. That assumption is optimistic if dateless records skew
    recent (plausible: recent construction is likelier to be mid-record-entry),
    so read this as a lower bound.

    ``existence_bound_ym`` is the crucial refinement. **What controls phantoms is
    not the install date but the existence bound** — the date by which every
    record is *known* to have existed. An audit date, an inspection year, or the
    vintage of the aerial imagery a layer was delineated from all supply one: a
    ramp audited in 2016 demonstrably existed in 2016, whatever its install field
    says. For any panorama captured after that bound a phantom is **structurally
    impossible**, so those captures contribute zero regardless of how many records
    are undated.

    This is why a 100%-undated inventory is not automatically disqualified, and
    why the exposure is genuinely one-sided for old surveys: DC has no install
    date at all, but every record was inspected in 2016 against 2022-23 imagery,
    so it cannot produce phantoms — only unlabeled positives.
    """
    if not undated_predates:
        return 0.0                      # they are all discarded instead
    n_dated = sum(install_hist.values())
    total = n_dated + n_undated
    if not total or not n_dated or not sum(capture_hist.values()):
        return 0.0
    undated_share = n_undated / total
    tail = 0.0
    for c_ym, c_w in _weights(capture_hist):
        if existence_bound_ym is not None and c_ym > existence_bound_ym:
            continue                    # every record provably predates this pano
        after = sum(n for ym, n in install_hist.items() if ym >= c_ym)
        tail += c_w * (after / n_dated)
    return undated_share * tail


def install_rate_per_year(install_hist, snapshot_ym, lookback=DEFAULT_RATE_LOOKBACK_YEARS):
    """Mean recorded installs/year over the ``lookback`` years before the snapshot.

    Used to price the unlabeled-positive exposure. Returns 0.0 when the inventory
    carries no dated installs in that window, in which case the exposure below is
    reported as unknown rather than zero.
    """
    if not install_hist or snapshot_ym is None or lookback <= 0:
        return 0.0
    hi = snapshot_ym[0]
    lo = hi - lookback
    n = sum(c for (y, _m), c in install_hist.items() if lo <= y < hi)
    return n / float(lookback)


def missing_exposure(install_hist, capture_hist, snapshot_ym,
                     inventory_size=None, lookback=DEFAULT_RATE_LOOKBACK_YEARS):
    """Ramps built after the inventory snapshot but present in the imagery.

    These are the **unlabeled positives**: invisible to the ordering filter,
    because the record simply does not exist. Estimated as
    ``build_rate x mean_positive_gap``, where the gap is per-panorama
    ``max(0, capture - snapshot)`` in years.

    ``est_missing_ramps`` is ``None`` when the build rate is unknown — an unknown
    is reported as unknown, never as zero.
    """
    gaps = [(c_w, max(0.0, _ym_to_years(c_ym) - _ym_to_years(snapshot_ym)))
            for c_ym, c_w in _weights(capture_hist)] if snapshot_ym else []
    mean_gap = sum(w * g for w, g in gaps)
    share_after = sum(w for w, g in gaps if g > 0)
    rate = install_rate_per_year(install_hist, snapshot_ym, lookback)
    est = rate * mean_gap if rate else None
    pct = (100.0 * est / inventory_size) if (est is not None and inventory_size) else None
    return {
        "inventory_snapshot": snapshot_ym,
        "mean_gap_years": mean_gap,
        "share_imagery_after_snapshot": share_after,
        "install_rate_per_year": rate,
        "est_missing_ramps": est,
        "est_missing_pct_of_inventory": pct,
    }


def quantile_ym(hist, q):
    """Count-weighted quantile ``(year, month)`` of a histogram (``None`` if empty).

    Used instead of ``max()`` for the default snapshot date, because real
    inventories carry typo years — Minneapolis has a single ``2926`` among 18k
    records — and one bad row must not define a city's snapshot.
    """
    total = sum(hist.values())
    if not total:
        return None
    seen, target = 0, total * q
    for ym, c in sorted(hist.items()):
        seen += c
        if seen >= target:
            return ym
    return max(hist)


def median_ym(hist):
    """Count-weighted median ``(year, month)`` of a histogram (``None`` if empty)."""
    total = sum(hist.values())
    if not total:
        return None
    seen, half = 0, total / 2.0
    for ym, c in sorted(hist.items()):
        seen += c
        if seen >= half:
            return ym
    return max(hist)


def summarize(install_values, capture_values, snapshot_ym=None,
              undated_predates=DEFAULT_UNDATED_PREDATES,
              lookback=DEFAULT_RATE_LOOKBACK_YEARS,
              existence_bound_ym=None):
    """Full report for one city. ``snapshot_ym`` defaults to the newest install.

    ``existence_bound_ym`` — the date by which every record is known to have
    existed (audit/inspection date, or the vintage of the imagery a layer was
    delineated from). Set it whenever the source provides one; it is what
    actually bounds phantom labels. Defaults to ``snapshot_ym`` when omitted,
    since a snapshot is itself weak evidence of existence at that date.
    """
    ihist, n_undated, n_sentinel = build_hist(install_values)
    chist, _, _ = build_hist(capture_values, sentinels=None)
    n_dated = sum(ihist.values())
    size = n_dated + n_undated
    if snapshot_ym is None and ihist:
        snapshot_ym = quantile_ym(ihist, 0.99)
    if existence_bound_ym is None:
        existence_bound_ym = snapshot_ym
    return {
        "existence_bound": existence_bound_ym,
        "suspected_sentinel": suspected_sentinel(ihist),
        "inventory_size": size,
        "n_dated": n_dated,
        "n_undated": n_undated,
        "n_sentinel": n_sentinel,
        "undated_fraction": (n_undated / size) if size else 0.0,
        "median_install": median_ym(ihist),
        "median_capture": median_ym(chist),
        "n_panos": sum(chist.values()),
        "ordering_discard_rate": discard_rate(ihist, chist, n_undated, undated_predates),
        "phantom_rate": phantom_rate(ihist, chist, n_undated, undated_predates,
                                     existence_bound_ym),
        "missing": missing_exposure(ihist, chist, snapshot_ym, size, lookback),
        "install_years": Counter({y: c for (y, _m), c in ihist.items()}),
        "capture_years": Counter({y: c for (y, _m), c in chist.items()}),
    }


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_inventory_dates(path, field=None):
    """Install dates from GeoJSON, ArcGIS JSON, Socrata JSON, or CSV.

    With no ``field``, guesses from the usual names. Raises if it cannot find one
    rather than returning an all-undated result, which would look like a finding.
    """
    GUESSES = ("installdate", "install_date", "date_installed", "construction_date",
               "constructiondate", "yearbuilt", "year_built", "install_dt",
               "date_built", "builtdate", "completion_date", "installed")

    def pick(keys):
        if field:
            for k in keys:
                if k.lower() == field.lower():
                    return k
            raise SystemExit(f"field {field!r} not in record; available: {sorted(keys)}")
        for g in GUESSES:
            for k in keys:
                if k.lower() == g:
                    return k
        raise SystemExit(
            "no install-date field found; pass --inventory-date-field. "
            f"Available: {sorted(keys)}")

    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return []
        key = pick(rows[0].keys())
        return [r.get(key) for r in rows]

    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    if isinstance(blob, dict) and "features" in blob:          # GeoJSON / ArcGIS
        feats = blob["features"]
        if not feats:
            return []
        props = [f.get("properties", f.get("attributes", {}) or {}) for f in feats]
    elif isinstance(blob, list):                                # Socrata
        props = blob
    else:
        raise SystemExit(f"unrecognized inventory JSON structure in {path}")
    if not props:
        return []
    key = pick(props[0].keys())
    return [p.get(key) for p in props]


def load_tracker_capture_dates(path, status_ok="OK"):
    """``capture_date`` column of a Streetscape Tracker snapshot (.csv.gz or .csv).

    Rows without a panorama (``ZERO_RESULTS``) are grid points with no imagery;
    they carry no date and are skipped rather than counted as undated, since they
    say something about *coverage*, not about temporal alignment.
    """
    opener = gzip.open if path.endswith(".gz") else open
    dates = []
    with opener(path, "rt", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if status_ok and (row.get("status") or "").strip().upper() != status_ok:
                continue
            d = (row.get("capture_date") or "").strip()
            if d:
                dates.append(d)
    return dates


def format_report(city, rep):
    def ym(t):
        return f"{t[0]}-{t[1]:02d}" if t else "n/a"

    m = rep["missing"]
    L = [f"--- {city} ---",
         f"Inventory records:    {rep['inventory_size']:,}  "
         f"(dated {rep['n_dated']:,}, undated {rep['n_undated']:,})",
         f"Undated fraction:     {rep['undated_fraction']:.1%}"
         + (f"   [{rep['n_sentinel']:,} are {SENTINEL_YM[0]}-01 sentinels]"
            if rep["n_sentinel"] else ""),
         f"Median install:       {ym(rep['median_install'])}",
         f"Median capture:       {ym(rep['median_capture'])}   "
         f"({rep['n_panos']:,} panos)",
         "",
         "Exposure (opposite signs — do not net them):",
         f"  Unlabeled positives (inventory older than imagery)",
         f"    snapshot {ym(m['inventory_snapshot'])}, mean gap "
         f"{m['mean_gap_years']:.2f} yr, "
         f"{m['share_imagery_after_snapshot']:.1%} of imagery postdates it",
         f"    build rate {m['install_rate_per_year']:,.0f}/yr -> est. missing "
         + ("unknown (no dated installs in the lookback window)"
            if m["est_missing_ramps"] is None else
            f"{m['est_missing_ramps']:,.0f} ramps"
            + (f" ({m['est_missing_pct_of_inventory']:.1f}% of inventory)"
               if m["est_missing_pct_of_inventory"] is not None else "")),
         f"  Phantom labels (undated records bypassing the ordering filter)",
         f"    {rep['phantom_rate']:.2%} of records, lower bound"
         + (f"   [existence bound {ym(rep['existence_bound'])}"
            + ("; every pano postdates it, so phantoms are impossible]"
               if rep["phantom_rate"] == 0.0 and rep["n_undated"] else "]")
            if rep.get("existence_bound") else ""),
         "",
         f"Ordering filter discards {rep['ordering_discard_rate']:.1%} of "
         f"(ramp, pano) pairs."]
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--city", required=True, help="Label for the report.")
    p.add_argument("--inventory", required=True,
                   help="Government curb-ramp inventory (GeoJSON/ArcGIS/Socrata JSON, or CSV).")
    p.add_argument("--inventory-date-field", default=None,
                   help="Install-date field name (guessed if omitted).")
    p.add_argument("--tracker-snapshot", required=True,
                   help="Streetscape Tracker snapshot .csv.gz for the same city.")
    p.add_argument("--snapshot-date", default=None,
                   help="Inventory snapshot date YYYY-MM (default: newest install "
                        "date). Set this for a static capture such as DC's 2016.")
    p.add_argument("--existence-bound", default=None,
                   help="YYYY-MM by which every record is KNOWN to have existed — an "
                        "audit/inspection date, or the vintage of the aerial imagery a "
                        "layer was delineated from. This, not the install date, is what "
                        "bounds phantom labels: panos captured after it cannot produce "
                        "one. Defaults to --snapshot-date.")
    p.add_argument("--rate-lookback", type=int, default=DEFAULT_RATE_LOOKBACK_YEARS)
    p.add_argument("--no-undated-predates", action="store_true",
                   help="Score as if TREAT_UNDATED_AS_PREDATING were False.")
    p.add_argument("--json-out", default=None, help="Also write the report as JSON.")
    args = p.parse_args(argv)

    rep = summarize(
        load_inventory_dates(args.inventory, args.inventory_date_field),
        load_tracker_capture_dates(args.tracker_snapshot),
        snapshot_ym=parse_ym(args.snapshot_date) if args.snapshot_date else None,
        undated_predates=not args.no_undated_predates,
        lookback=args.rate_lookback,
        existence_bound_ym=parse_ym(args.existence_bound) if args.existence_bound else None)
    print(format_report(args.city, rep))

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"city": args.city, **rep}, fh, indent=2, default=lambda o: dict(o))
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
