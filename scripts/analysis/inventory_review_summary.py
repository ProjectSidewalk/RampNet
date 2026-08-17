"""Turn a reviewed ``verdicts.json`` into the numbers §5 asks for (issues #96, #59).

The paper bucketed cities Good / OK / Poor by eye and published no thresholds.
§5 asks for the same judgment made quantitative: *"sample ~50 points/city, measure
metres from the true ramp on aerial imagery; report a **distribution**, not a
bucket."* This is the reduction step — verdict file in, distribution out.

Four numbers come out, and each has a denominator that has to be stated or it
means something different from what a reader assumes:

* **Offset distribution** over chips with a usable measurement. A chip marked
  unjudgeable is EXCLUDED even if it carries a click, because "I cannot make a
  call" and "the call is 4.54 m" are contradictory claims and the disowned click
  lands in the tail, where a single stray value moves p90 and the max.
* **Phantom rate** over *judgeable* chips, not all chips — an unjudgeable chip is
  not evidence of a ramp being present or absent.
* **Unjudgeable rate** over all chips.
* **Per-corner agreement** between what the reviewer counted and what the city
  publishes, read against the [6 m, 10 m] bracket rather than either radius,
  because a radius is not a corner (see §5e).

**Wilson intervals, not normal approximations.** At n≈55 with a rate near 5% the
normal interval runs below zero, which would be nonsense on the page.

    python scripts/analysis/inventory_review_summary.py \
        analysis_out/review_denver-co/verdicts.json

Pure apart from file reading; the arithmetic is unit-tested in
``tests/test_inventory_review_summary.py``.
"""
import argparse
import json
import math
import random
import sys


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion. Pure.

    Preferred over the normal approximation because these denominators are ~55
    and these rates are near 5%, where the normal interval extends below zero.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def percentile(sorted_values, q):
    """Linear-interpolated percentile of an already-sorted list. Pure."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * q
    lo = int(math.floor(k))
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def systematic_shift(records, metres_per_pixel, span_px):
    """Is the offset a random error, or is the whole city displaced?

    **The check that separates a bad inventory from a bad basemap**, and it costs
    nothing extra because the reviewer's click already records a *direction*, not
    just a distance.

    Random positional error cancels: the mean offset VECTOR tends to zero while
    the mean offset MAGNITUDE does not. A datum or projection error does not
    cancel — every ramp is displaced the same way, so the two converge. The ratio
    ``|mean vector| / mean magnitude`` is therefore ~0 for noise and ~1 for a
    shift.

    Denver measures 0.10 m resultant against 0.44 m mean magnitude (24%) — no
    gross shift, consistent with §5e's independent centreline check. Seattle's
    first 11 chips measure 2.06 m against 2.37 m (**87%**).

    **⚠️ The ratio is NOT ~0 under the null, and reading it as though it were is
    how this statistic misleads.** With ``n`` offsets of fixed magnitude and
    uniformly random direction the mean vector does not vanish; it shrinks only
    as ``1/sqrt(n)``, so the expected share is roughly ``0.9/sqrt(n)`` — **39% at
    Seattle's n=11**, not 0%. A share has to be read against
    ``systematic_shift_null``, which resamples the observed magnitudes with random
    directions and returns the exceedance probability. Seattle's 87% is genuinely
    improbable under that null (p≈0.001), so the direction is real *in this
    sample*.

    **But a significant share still does not establish a registration error.**
    Seattle is the worked counter-example: its 87% survived the null, and both
    candidate frames were then cleared at high n — the coordinates sit unbiased
    against the city's own street network over 31,430 samples
    (``inventory_centerline_offset.py``, 0.00 m) and the basemap agrees with that
    network to ≤0.32 m *at the eleven reviewed chips themselves*
    (``verify_chip_georeference.py --sites-from-verdicts``). A city-wide
    displacement cannot hide from either. What produces a directional lean at
    n=11 is a handful of large per-record errors that happen to share a heading,
    which this ratio cannot distinguish from a true shift.

    So the rule the tool enforces: a high share **raises** the registration
    question, and only a high-n instrument can answer it. Never conclude "shift"
    from the review sheet alone. Pure.
    """
    C = span_px / 2.0
    vecs = []
    for r in records:
        if r.get("unreadable") or r.get("click_px") is None or r.get("offset_m") is None:
            continue
        px, py = r["click_px"]
        vecs.append(((px - C) * metres_per_pixel, -(py - C) * metres_per_pixel))
    if not vecs:
        return None
    n = len(vecs)
    mean_e = sum(v[0] for v in vecs) / n
    mean_n = sum(v[1] for v in vecs) / n
    resultant = math.hypot(mean_e, mean_n)
    mean_mag = sum(math.hypot(*v) for v in vecs) / n
    out = {
        "n": n,
        "mean_east_m": mean_e, "mean_north_m": mean_n,
        "resultant_m": resultant, "mean_magnitude_m": mean_mag,
        "systematic_share": (resultant / mean_mag) if mean_mag else None,
        "east_positive": sum(1 for v in vecs if v[0] > 0),
        "north_positive": sum(1 for v in vecs if v[1] > 0),
    }
    if out["systematic_share"] is not None:
        out["null"] = systematic_shift_null([math.hypot(*v) for v in vecs],
                                            out["systematic_share"])
    return out


def systematic_shift_null(magnitudes, observed_share, draws=20000, seed=20260731):
    """How large a ``systematic_share`` do these offsets give by chance alone?

    Holds the observed **magnitudes** fixed and randomises only the directions.
    Keeping the magnitudes matters: the ratio is dominated by the largest few
    offsets, so a null built on equal-sized vectors would understate how easily a
    heavy-tailed sample fakes a shift. Seattle's magnitudes run 0.21 m to 8.79 m
    over eleven chips, and the single 8.79 m click moves the mean vector further
    than the six smallest combined.

    Returns the null median and the exceedance probability. Pure apart from a
    seeded RNG, so the p-value is reproducible.
    """
    n = len(magnitudes)
    if n == 0:
        return None
    mean_mag = sum(magnitudes) / n
    if mean_mag <= 0:
        return None
    rng = random.Random(seed)
    shares = []
    for _ in range(draws):
        e = nn = 0.0
        for m in magnitudes:
            th = rng.uniform(0.0, 2.0 * math.pi)
            e += m * math.cos(th)
            nn += m * math.sin(th)
        shares.append(math.hypot(e, nn) / n / mean_mag)
    shares.sort()
    ge = sum(1 for s in shares if s >= observed_share)
    return {
        "draws": draws,
        "median_share": shares[draws // 2],
        "p95_share": shares[int(0.95 * draws)],
        "p_value": ge / float(draws),
        "note": "Expected share under random directions is ~0.9/sqrt(n), NOT 0. "
                "A share below the null median is evidence of nothing; a share "
                "above it says the direction is real in this sample, which is "
                "still not the same as a city-wide registration error.",
    }


def classify(record):
    """One of 'measured', 'phantom', 'unjudgeable', 'todo'. Pure.

    Order matters. ``unjudgeable`` is tested BEFORE the offset, so a chip that
    carries a click the reviewer then disowned is excluded rather than counted.
    """
    if record.get("unreadable"):
        return "unjudgeable"
    if record.get("no_ramp"):
        return "phantom"
    if record.get("offset_m") is not None:
        return "measured"
    return "todo"


def summarise(manifest):
    """Reduce a verdicts manifest to the reportable numbers. Pure."""
    records = manifest["records"]
    buckets = {k: [] for k in ("measured", "phantom", "unjudgeable", "todo")}
    for rec in records:
        buckets[classify(rec)].append(rec)

    offsets = sorted(r["offset_m"] for r in buckets["measured"])
    n_all = len(records)
    n_judgeable = len(buckets["measured"]) + len(buckets["phantom"])

    within = {}
    for t in (0.5, 1.0, 2.0, 3.0, 5.0):
        k = sum(1 for o in offsets if o <= t)
        within[t] = {"n": k, "of": len(offsets),
                     "rate": (k / len(offsets)) if offsets else None,
                     "ci": wilson(k, len(offsets))}

    # Reviewer's per-corner count against the published bracket. A count inside
    # [within_6m, within_10m] is consistent; only outside it is evidence.
    agree = more = fewer = 0
    disagreements = []
    for rec in records:
        seen, p6, p10 = (rec.get("ramps_visible"), rec.get("published_within_6m"),
                         rec.get("published_within_10m"))
        if seen is None or p6 is None:
            continue
        if seen > p10:
            more += 1
            disagreements.append({"id": rec["id"], "seen": seen, "p6": p6, "p10": p10,
                                  "kind": "more_than_published"})
        elif seen < p6:
            fewer += 1
            disagreements.append({"id": rec["id"], "seen": seen, "p6": p6, "p10": p10,
                                  "kind": "fewer_than_published",
                                  "phantom": bool(rec.get("no_ramp"))})
        else:
            agree += 1

    counted = [r["ramps_visible"] for r in records if r.get("ramps_visible") is not None]

    return {
        "city": manifest.get("city"),
        "seed": manifest.get("seed"),
        "sheet_build": manifest.get("sheet_build"),
        "imagery": manifest.get("imagery"),
        "metres_per_pixel": manifest.get("metres_per_pixel"),
        "chips": n_all,
        "reviewed": n_all - len(buckets["todo"]),
        "todo": [r["id"] for r in buckets["todo"]],
        "uncounted": [r["id"] for r in records if r.get("ramps_visible") is None],
        "offset": {
            "n": len(offsets),
            "min": offsets[0] if offsets else None,
            "p25": percentile(offsets, .25), "median": percentile(offsets, .50),
            "p75": percentile(offsets, .75), "p90": percentile(offsets, .90),
            "max": offsets[-1] if offsets else None,
            "mean": (sum(offsets) / len(offsets)) if offsets else None,
            "within_m": within,
        },
        "phantom": {"n": len(buckets["phantom"]), "of_judgeable": n_judgeable,
                    "rate": (len(buckets["phantom"]) / n_judgeable) if n_judgeable else None,
                    "ci": wilson(len(buckets["phantom"]), n_judgeable),
                    "ids": [r["id"] for r in buckets["phantom"]]},
        # Two denominators, because they answer different questions and the wrong
        # one is badly misleading mid-review. Over ALL chips is the number to
        # report once the pass is complete; over ATTEMPTED chips is the only
        # honest reading while chips remain untouched, since the untouched ones
        # are not evidence that the imagery was readable.
        "unjudgeable": {"n": len(buckets["unjudgeable"]), "of": n_all,
                        "rate": len(buckets["unjudgeable"]) / n_all if n_all else None,
                        "ci": wilson(len(buckets["unjudgeable"]), n_all),
                        "of_attempted": n_all - len(buckets["todo"]),
                        "rate_of_attempted": (
                            len(buckets["unjudgeable"]) / (n_all - len(buckets["todo"]))
                            if n_all - len(buckets["todo"]) else None),
                        "ids": [r["id"] for r in buckets["unjudgeable"]]},
        "per_corner": {
            "consistent": agree, "more_than_published": more,
            "fewer_than_published": fewer, "disagreements": disagreements,
            "mean_ramps_seen": (sum(counted) / len(counted)) if counted else None,
            "counted": len(counted),
            "histogram": {str(v): counted.count(v) for v in sorted(set(counted))},
        },
        "systematic_shift": systematic_shift(
            records, manifest.get("metres_per_pixel") or 0.0,
            manifest.get("span_px") or 0),
        "excluded_clicks": [
            {"id": r["id"], "offset_m": r["offset_m"], "note": r.get("note", "")}
            for r in buckets["unjudgeable"] if r.get("offset_m") is not None
        ],
    }


def render(s):
    out = []
    w = out.append
    w("{} -- location precision, seed {} (sheet build {})".format(
        s["city"], s["seed"], s["sheet_build"]))
    w("imagery: {} at {:.4f} m/px".format(s["imagery"], s["metres_per_pixel"]))
    w("")
    w("{} of {} chips reviewed{}".format(
        s["reviewed"], s["chips"],
        "" if not s["todo"] else "   !! NOT DONE: " + ", ".join(s["todo"])))
    if s["uncounted"]:
        w("   !! no ramp count on: " + ", ".join(s["uncounted"]))
    o = s["offset"]
    w("")
    w("OFFSET  (n={}, unjudgeable chips excluded)".format(o["n"]))
    if o["n"]:
        w("  min {:.2f}  p25 {:.2f}  median {:.2f}  p75 {:.2f}  p90 {:.2f}  max {:.2f} m"
          .format(o["min"], o["p25"], o["median"], o["p75"], o["p90"], o["max"]))
        for t, v in sorted(o["within_m"].items()):
            w("  <= {:>4.1f} m : {:>2}/{:<2}  {:5.1f}%   [{:.1f}-{:.1f}]".format(
                t, v["n"], v["of"], 100 * v["rate"], 100 * v["ci"][0], 100 * v["ci"][1]))
    p, u = s["phantom"], s["unjudgeable"]
    w("")
    w("PHANTOM      {}/{} judgeable = {:.1f}%  [{:.1f}-{:.1f}]  {}".format(
        p["n"], p["of_judgeable"], 100 * p["rate"], 100 * p["ci"][0], 100 * p["ci"][1],
        ", ".join(p["ids"])))
    w("UNJUDGEABLE  {}/{} chips     = {:.1f}%  [{:.1f}-{:.1f}]".format(
        u["n"], u["of"], 100 * u["rate"], 100 * u["ci"][0], 100 * u["ci"][1]))
    if s["todo"]:
        w("             {}/{} ATTEMPTED = {:.1f}%  <- the honest reading mid-review".format(
            u["n"], u["of_attempted"], 100 * u["rate_of_attempted"]))
    w("             {}".format(", ".join(u["ids"])))
    c = s["per_corner"]
    w("")
    w("PER-CORNER   consistent {} | saw more {} | saw fewer {}   (n={})".format(
        c["consistent"], c["more_than_published"], c["fewer_than_published"], c["counted"]))
    w("  ramps seen per corner: {}   mean {:.2f}".format(
        c["histogram"], c["mean_ramps_seen"]))
    for d in c["disagreements"]:
        w("   {:<8} saw {} | published {}/{} | {}{}".format(
            d["id"], d["seen"], d["p6"], d["p10"], d["kind"],
            " (phantom)" if d.get("phantom") else ""))
    sh = s.get("systematic_shift")
    if sh:
        w("")
        w("SYSTEMATIC SHIFT  (does the whole city move, or is it random error?)")
        w("  mean magnitude {:.2f} m | mean vector east {:+.2f} north {:+.2f} -> resultant {:.2f} m"
          .format(sh["mean_magnitude_m"], sh["mean_east_m"], sh["mean_north_m"],
                  sh["resultant_m"]))
        w("  systematic share {:.0f}%  (east-positive {}/{}, north-positive {}/{})"
          .format(100 * sh["systematic_share"], sh["east_positive"], sh["n"],
                  sh["north_positive"], sh["n"]))
        nul = sh.get("null")
        if nul:
            w("  null (magnitudes kept, directions randomised): median {:.0f}%, "
              "p95 {:.0f}%".format(100 * nul["median_share"], 100 * nul["p95_share"]))
            w("  P(share this high by chance) = {:.4f}".format(nul["p_value"]))
        if nul and nul["p_value"] >= 0.05:
            w("  -> NOT distinguishable from random direction at n={}. The share is "
              "~0.9/sqrt(n)".format(sh["n"]))
            w("     under the null, not 0, so a large-looking share at small n means "
              "nothing.")
        elif sh["systematic_share"] > 0.5:
            w("  !! DIRECTIONAL beyond chance -- but that is a QUESTION, not a verdict.")
            w("     A registration error displaces every record, so it cannot hide from")
            w("     a high-n instrument. Before calling this a shift, confirm it with")
            w("     BOTH of:")
            w("       inventory_centerline_offset.py   ramps vs the city's own streets")
            w("       verify_chip_georeference.py --sites-from-verdicts   basemap vs those")
            w("     Seattle is the counter-example: 87% at p=0.001, and both came back")
            w("     clean -- the lean was per-record error sharing a heading, not a shift.")

    if s["excluded_clicks"]:
        w("")
        w("CLICKS EXCLUDED as unjudgeable (recorded, not counted):")
        for e in s["excluded_clicks"]:
            w("   {:<8} {:.2f} m   {}".format(e["id"], e["offset_m"], e["note"]))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("verdicts")
    ap.add_argument("--json", help="also write the summary as JSON here")
    args = ap.parse_args(argv)
    with open(args.verdicts, encoding="utf-8") as fh:
        s = summarise(json.load(fh))
    print(render(s))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(s, fh, indent=1)
            fh.write("\n")
        print("\nwrote {}".format(args.json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
