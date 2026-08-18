"""Phase 0 of the far-field `visible` anomaly study: is the rated sample representative? (#46)

The reviewer pass over the silent misses (``benchmark/miss_taxonomy_46/silent__jonf.json``)
rated **34 of 36** rateable far-field crops ``visible`` — the ramp's own pixels present
and carrying its appearance *in the model-resolution panel*. At face value that
contradicts the pixel-starvation framing E1 attached to the far field
(``docs/curb_ramp_data_sourcing.md`` §0a: "a 1.2 m ramp at 30 m is ~25 px — more
examples do not add pixels"), which is the assumption the 18 m far/near split — and
through it the sourcing bracket and the multi-view sizing — stands on.

Before that contradiction is allowed to mean anything, the sample has to be checked
(#46, hypothesis H1): the 37 rated far-field crops are not a random draw from the 83
far-field silent misses. They passed two filters — **unwitnessed** (no other model
detected anything there either) and the **30-source-pixel judgeability floor** — and
the floor bites at a different apparent size on every split, because the stored
panoramas range from 4096 to 16384 px wide while ``geom()`` sizes ramps at the model's
4096-px input. On a 16384-px split the floor admits ramps down to **7.5 model px**; on
morgantown it stops at **30**. If the rated crops are the biggest and closest of the
far field, a 94% visible rate there says little about the population.

This measures that, from committed data alone:

* **the two filters' effect on the sample**, split by split — which populations could
  even reach the deck, and where the rated 37 sit in the far-silent size distribution;
* **whether apparent size discriminates far-field hits from far-field silent misses**
  at all — if the model detects other ramps of the *same* apparent size at a healthy
  rate, a hard pixel floor cannot be what makes these particular ramps silent;
* **the mis-binning guards**: above-horizon clamps (``geom()`` sends y <= 0.5 straight
  to 150 m) and the GSV/Mapillary tier split, since the flat-ground distance estimate
  is weaker on Mapillary rigs (Spearman 0.81 vs 0.95, §0a).

Everything is read from committed files (``analysis_out/op_cache``,
``analysis_out/silent_witness.json``, ``benchmark/miss_taxonomy_46/``,
``benchmark/<city>/imagery_manifest.json``): no GPU, no network, no imagery.

    python scripts/analysis/farfield_forensics.py
    python scripts/analysis/farfield_forensics.py --json-out analysis_out/farfield_forensics.json
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import miss_taxonomy as mt  # noqa: E402
from miss_decomposition import (  # noqa: E402
    DEFAULT_THRESHOLD, FAR_BOUNDARY_M, TIER, US_SPLITS)
from miss_gallery import (  # noqa: E402
    JUDGEABLE_SOURCE_PX, MODEL_WIDTH, load_queue, source_px, tag_key)

# Distance bands inside the far field, in metres. The first two match E1's gold-set
# bins (recall 0.90 at 18-25, 0.49 at 25-40), so the pooled-benchmark rates here are
# directly comparable to the numbers the pixel-starvation framing was built on.
# ``geom()`` clamps above-horizon points to 150 m, so the clamp is its own band —
# those distances are not measurements, they are the estimator giving up.
FAR_BANDS = ((18.0, 25.0), (25.0, 40.0), (40.0, 150.0))

# Tolerance for "same apparent size" when asking how often the model detects OTHER
# far-field ramps of a rated miss's size. +/-20% of the target px is ~ +/-20% of
# distance (px is 1/d), comfortably inside one of E1's bins.
MATCH_TOL = 0.20

# The verdict pass the write-up quotes. A file per rater (``silent__<rater>.json``)
# is the committed shape, so a second pass is a flag, not a patch.
DEFAULT_RATER = "jonf"


# --------------------------------------------------------------------------- #
# Pure core (no I/O) — unit-tested in tests/test_farfield_forensics.py
# --------------------------------------------------------------------------- #
def auc(a, b):
    """P(a random draw from ``a`` exceeds one from ``b``), ties counting half.

    The Mann-Whitney statistic scaled to [0, 1]: 0.5 means the two samples are
    indistinguishable on this variable, 1.0 means every ``a`` exceeds every ``b``.
    O(n*m); the populations here are dozens-to-hundreds, so clarity wins.
    """
    if not a or not b:
        return float("nan")
    wins = ties = 0
    for x in a:
        for y in b:
            if x > y:
                wins += 1
            elif x == y:
                ties += 1
    return (wins + 0.5 * ties) / (len(a) * len(b))


def quartiles(values):
    """``(q1, median, q3)`` by simple index — enough for reporting, no interpolation."""
    if not values:
        return (float("nan"),) * 3
    v = sorted(values)
    n = len(v)
    return v[n // 4], v[n // 2], v[(3 * n) // 4]


def effective_floor_model_px(source_width, floor=JUDGEABLE_SOURCE_PX,
                             model_width=MODEL_WIDTH):
    """The judgeability floor translated into MODEL pixels for one stored width.

    ``source_px = model_px * source_width / model_width``, so a floor fixed in
    source pixels admits smaller model-pixel ramps the wider the stored pano is.
    This asymmetry is what shapes the rated deck's composition across splits.
    """
    return floor * model_width / source_width


def band_of(row, bands=FAR_BANDS):
    """The far band a row falls in, ``'clamp'`` for above-horizon rows, else None.

    The clamp test is on ``y``, mirroring ``miss_decomposition.above_horizon`` —
    a ground ramp cannot sit at or above the horizon, so its 150 m is an artifact
    of an unleveled rig or a hill, not a distance.

    **The top band is closed at its upper edge**, unlike the others, because
    ``geom()`` reaches 150 m by two different routes and only one of them is a
    ``y`` tell: the above-horizon branch, and ``min(d, 150.0)`` for a row that is
    below the horizon but saturates anyway (y = 0.502 puts a ramp 406 m out). A
    half-open [40, 150) drops that second kind from every band while ``y > 0.5``
    keeps it out of ``clamp`` — two pooled far-field GT rows, one of them a silent
    miss, which is exactly the arithmetic the write-up quotes. Pinned by
    ``tests/test_farfield_forensics.py::test_the_bands_partition_the_far_field``.
    """
    if row["y"] <= 0.5:
        return "clamp"
    top = bands[-1][1] if bands else None
    for lo, hi in bands:
        if lo <= row["dist"] < hi or (hi == top and row["dist"] == hi):
            return (lo, hi)
    return None


def matched_rate(rows, px, tol=MATCH_TOL):
    """``(hits, n)`` among ``rows`` whose apparent size is within ``tol`` of ``px``.

    The counterfactual the floor question needs: of every far-field GT ramp the
    benchmark scores at (about) this apparent size, how many did the model find?
    """
    sel = [r for r in rows if abs(r["px"] - px) <= tol * px]
    return sum(1 for r in sel if r["hit"]), len(sel)


def percentile_rank(population, x):
    """Fraction of ``population`` strictly below ``x``, ties counting half."""
    if not population:
        return float("nan")
    below = sum(1 for v in population if v < x)
    ties = sum(1 for v in population if v == x)
    return (below + 0.5 * ties) / len(population)


def partition_check(band_rows, n_far, n_far_silent):
    """Raise unless the band table decomposes the whole far field.

    A band table is a *decomposition*, so its rows must sum to the population the
    section around it quotes. Enforced rather than eyeballed because the failure is
    invisible in the output: a dropped row just makes a column smaller, and the
    only tell is that a total two paragraphs away disagrees. That is exactly how
    the [40, 150) half-open top band shipped, taking 2 GT and 1 silent miss with it.
    """
    gt = sum(b["n_gt"] for b in band_rows.values())
    sil = sum(b["silent"] for b in band_rows.values())
    if (gt, sil) != (n_far, n_far_silent):
        raise ValueError(f"bands do not partition the far field: {gt}/{n_far} GT, "
                         f"{sil}/{n_far_silent} silent — see band_of()")
    return gt, sil


def row_key(row):
    """The identity a row shares with the witness list and the gallery manifest."""
    return (row["city"], row["pano"], round(float(row["x"]), 6),
            round(float(row["y"]), 6))


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def stored_widths(city):
    """``{pano_id: width}`` from the committed imagery manifest, or ``{}``.

    The manifest was committed to pin the imagery by content hash (#94); its
    ``width`` field is what lets this script translate the source-pixel floor
    into model pixels without touching a single image.
    """
    path = os.path.join(REPO, "benchmark", city, "imagery_manifest.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return {pid: rec["width"] for pid, rec in payload.get("panos", {}).items()
            if rec.get("width")}


def verdicts_path(gallery_dir, rater=DEFAULT_RATER):
    """Where one rater's silent-miss verdicts live.

    Per-rater by construction, because a second rater is the top open follow-up
    on #46 and the two passes have to be comparable **without editing source**
    (CLAUDE.md: human judgments are committed per rater, and the rubric travels
    with them). ``--rater`` is the knob; the default is the pass that exists.
    """
    return os.path.join(gallery_dir, f"silent__{rater}.json")


def load_rated(gallery_dir, field="far", rater=DEFAULT_RATER):
    """The reviewer's rated items: manifest entry + verdict, keyed like the tags.

    ``field`` restricts to one distance population; ``None`` returns all 50
    (Phase 1's activation forensics wants the near-field verdicts too).

    **A manifest item with no verdict is not returned.** All 50 are rated today,
    so this changes nothing now — but mid-pass with a second rater an unrated
    item would otherwise carry ``verdict: None`` into ``rated_rows``, the
    per-split table and the survivorship AUC, inflating every one of them
    silently. Queued-but-unrated is a different population from rated, and the
    write-up quotes the latter.
    """
    with open(os.path.join(gallery_dir, "silent_gallery", "manifest.json"),
              encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(verdicts_path(gallery_dir, rater), encoding="utf-8") as fh:
        raw = json.load(fh)["verdicts"]
    out = {}
    for key, item in manifest["items"].items():
        if field is not None and item.get("field") != field:
            continue
        v = raw.get(key)
        v = v if isinstance(v, str) else (v or {}).get("verdict")
        if v is None:
            continue
        out[key] = {**item, "verdict": v}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--witness", default=os.path.join(OUT, "silent_witness.json"))
    p.add_argument("--gallery", default=os.path.join(REPO, "benchmark",
                                                     "miss_taxonomy_46"))
    p.add_argument("--rater", default=DEFAULT_RATER,
                   help="Which reviewer pass to read (benchmark/miss_taxonomy_46/"
                        "silent__<rater>.json). A second rater is a flag, not an edit.")
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    # Populations, all at the deployed threshold. --------------------------------
    pooled = []
    for city in US_SPLITS:
        loaded = mt.load_rows(city, args.threshold, rng=None)
        if loaded is not None:
            pooled.extend(loaded[0])
    far = [r for r in pooled if r["field"] == "far"]
    far_hits = [r for r in far if r["hit"]]
    far_silent = [r for r in far if not r["hit"] and r["bucket"] == "silent"]

    queue = load_queue(args.witness)
    unw_far = [r for r in far_silent if row_key(r) in queue]

    rated = load_rated(args.gallery, rater=args.rater)
    rated_keys = {(v["city"], v["pano"], round(float(v["x"]), 6),
                   round(float(v["y"]), 6)) for v in rated.values()}
    rated_rows = [r for r in unw_far if row_key(r) in rated_keys]
    excluded_rows = [r for r in unw_far if row_key(r) not in rated_keys]

    widths = {city: stored_widths(city) for city in US_SPLITS}
    visible = [v for v in rated.values() if v["verdict"] == "visible"]

    print(f"=== Far-field 'visible' anomaly, Phase 0: sample forensics "
          f"(threshold {args.threshold}, boundary {FAR_BOUNDARY_M:.0f} m, #46) ===\n")
    print(f"far-field GT {len(far)}, hits {len(far_hits)} "
          f"(recall {len(far_hits)/len(far):.3f}), silent misses {len(far_silent)}; "
          f"unwitnessed {len(unw_far)}, rated {len(rated_rows)}, "
          f"below the floor {len(excluded_rows)}")
    tally = {}
    for v in rated.values():
        tally[v["verdict"]] = tally.get(v["verdict"], 0) + 1
    print(f"reviewer verdicts over the rated set: " +
          ", ".join(f"{k} {n}" for k, n in sorted(tally.items(),
                                                  key=lambda kv: -kv[1])))

    # 1. The floor, split by split. ----------------------------------------------
    print(f"\n{'-'*78}\n1. THE FLOOR IS NOT ONE FLOOR — {JUDGEABLE_SOURCE_PX:.0f} "
          f"source px in model pixels, per split\n{'-'*78}")
    print(f"{'split':>12} {'tier':>10} {'stored px':>12} {'floor(model px)':>16} "
          f"{'far-silent':>11} {'unwitn.':>8} {'rated':>6}")
    per_split = {}
    for city in US_SPLITS:
        fs = [r for r in far_silent if r["city"] == city]
        if not fs:
            continue
        ws = sorted({widths[city].get(r["pano"]) for r in fs
                     if widths[city].get(r["pano"])})
        w_lo, w_hi = (ws[0], ws[-1]) if ws else (None, None)
        floor_lo = effective_floor_model_px(w_hi) if w_hi else float("nan")
        floor_hi = effective_floor_model_px(w_lo) if w_lo else float("nan")
        floor_s = (f"{floor_lo:.1f}" if w_lo == w_hi else
                   f"{floor_lo:.1f}-{floor_hi:.1f}")
        n_unw = sum(1 for r in unw_far if r["city"] == city)
        n_rated = sum(1 for r in rated_rows if r["city"] == city)
        stored_s = (f"{w_lo}" if w_lo == w_hi else f"{w_lo}-{w_hi}") if ws else "?"
        print(f"{city:>12} {TIER.get(city, '-'):>10} {stored_s:>12} {floor_s:>16} "
              f"{len(fs):>11} {n_unw:>8} {n_rated:>6}")
        per_split[city] = {"far_silent": len(fs), "unwitnessed": n_unw,
                           "rated": n_rated, "stored_width_min": w_lo,
                           "stored_width_max": w_hi}
    print("\n  The deck's composition follows the floor: the 16384-px splits admit")
    print("  far misses down to 7.5 model px, morgantown stops at 30. Which split a")
    print("  miss happened in decides whether a reviewer ever saw it.")

    # 2. Where the rated set sits in the far-silent population. -------------------
    print(f"\n{'-'*78}\n2. SURVIVORSHIP — where the rated {len(rated_rows)} sit "
          f"among all {len(far_silent)} far-field silent misses\n{'-'*78}")
    print(f"{'population':>34} {'n':>4} {'dist q1/med/q3 (m)':>20} "
          f"{'px q1/med/q3':>15}")
    pops = {
        "rated (reached the deck)": rated_rows,
        "below the floor (excluded)": excluded_rows,
        "witnessed (never queued)": [r for r in far_silent
                                     if row_key(r) not in queue],
        "ALL far-field silent misses": far_silent,
        "far-field hits (for contrast)": far_hits,
    }
    stats = {}
    for name, rows in pops.items():
        dq = quartiles([r["dist"] for r in rows])
        pq = quartiles([r["px"] for r in rows])
        stats[name] = {"n": len(rows), "dist_q": dq, "px_q": pq}
        print(f"{name:>34} {len(rows):>4} "
              f"{dq[0]:>6.1f}/{dq[1]:>5.1f}/{dq[2]:>5.1f} "
              f"{pq[0]:>5.1f}/{pq[1]:>4.1f}/{pq[2]:>4.1f}")
    auc_rated = auc([r["px"] for r in rated_rows],
                    [r["px"] for r in far_silent
                     if row_key(r) not in rated_keys])
    med_rank = percentile_rank([r["px"] for r in far_silent],
                               quartiles([r["px"] for r in rated_rows])[1])
    print(f"\n  AUC(rated px vs unrated far-silent px) = {auc_rated:.3f} "
          f"(0.5 = no size bias)")
    print(f"  the rated median px sits at the {med_rank:.0%} percentile of the "
          f"far-silent population")

    # 3. Does apparent size even separate hit from silent out here? ---------------
    print(f"\n{'-'*78}\n3. DISCRIMINATION — is a far-field silent miss the size "
          f"the model cannot see?\n{'-'*78}")
    auc_hit = auc([r["px"] for r in far_hits], [r["px"] for r in far_silent])
    print(f"  AUC(far-hit px vs far-silent px) = {auc_hit:.3f} "
          f"(1.0 would mean size alone decides)\n")
    print(f"{'band':>12} {'GT':>6} {'recall':>8} {'silent':>8} {'rated':>6} "
          f"{'visible':>8}")
    band_rows = {}
    key_of = {tag_key(v["pano"], v["x"], v["y"]): v for v in rated.values()}
    for band in list(FAR_BANDS) + ["clamp"]:
        sel = [r for r in far if band_of(r) == band]
        if not sel:
            continue
        n_hit = sum(1 for r in sel if r["hit"])
        n_sil = sum(1 for r in sel if not r["hit"] and r["bucket"] == "silent")
        n_rated = sum(1 for r in rated_rows if band_of(r) == band)
        n_vis = sum(1 for r in rated_rows if band_of(r) == band
                    and key_of.get(tag_key(r["pano"], r["x"], r["y"]),
                                   {}).get("verdict") == "visible")
        label = "clamp>=150" if band == "clamp" else f"{band[0]:.0f}-{band[1]:.0f} m"
        print(f"{label:>12} {len(sel):>6} {n_hit/len(sel):>8.3f} {n_sil:>8} "
              f"{n_rated:>6} {n_vis:>8}")
        band_rows[label] = {"n_gt": len(sel), "recall": n_hit / len(sel),
                            "silent": n_sil, "rated": n_rated, "visible": n_vis}

    # The bands must partition the far field, or the table above quietly reports a
    # smaller population than the section it sits in. They did not, until the top
    # band was closed at 150 m — see band_of().
    try:
        banded, banded_sil = partition_check(band_rows, len(far), len(far_silent))
    except ValueError as exc:
        raise SystemExit(str(exc))
    print(f"{'':>12} {banded:>6} {'':>8} {banded_sil:>8}   <- sums to the "
          f"{len(far)} far GT / {len(far_silent)} silent above")

    rates = []
    for v in visible:
        h, n = matched_rate(far, v["model_px"])
        if n:
            rates.append(h / n)
    rq = quartiles(rates)
    print(f"\n  MATCHED-SIZE DETECTION RATE: for each of the {len(visible)} "
          f"'visible' misses, the model's")
    print(f"  recall over ALL far-field GT within ±{MATCH_TOL:.0%} of that miss's "
          f"apparent size:")
    print(f"    q1/median/q3 = {rq[0]:.2f} / {rq[1]:.2f} / {rq[2]:.2f}")
    print(f"  A hard pixel floor would put these near zero. The model routinely")
    print(f"  detects other ramps of the same apparent size; silence at that size is")
    print(f"  therefore not size-fated, and 'pixel-starved' is not a sufficient")
    print(f"  explanation for these misses.")

    # 4. Mis-binning guards. ------------------------------------------------------
    print(f"\n{'-'*78}\n4. GUARDS — how much of this could the distance estimator "
          f"be inventing?\n{'-'*78}")
    n_clamp = {name: sum(1 for r in rows if r["y"] <= 0.5)
               for name, rows in pops.items()}
    print(f"  above-horizon clamps: rated {n_clamp['rated (reached the deck)']}, "
          f"excluded {n_clamp['below the floor (excluded)']}, "
          f"all far-silent {n_clamp['ALL far-field silent misses']}")
    tiers = {}
    for r in rated_rows:
        tiers[TIER.get(r["city"], "-")] = tiers.get(TIER.get(r["city"], "-"), 0) + 1
    print(f"  rated set by tier: " +
          ", ".join(f"{k} {n}" for k, n in sorted(tiers.items())) +
          "  (flat-ground distance is Spearman 0.95 on gsv, 0.81 on mapillary)")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        payload = {
            "threshold": args.threshold, "boundary_m": FAR_BOUNDARY_M,
            "rater": args.rater,
            "judgeable_source_px": JUDGEABLE_SOURCE_PX,
            "match_tolerance": MATCH_TOL,
            "counts": {"far_gt": len(far), "far_hits": len(far_hits),
                       "far_silent": len(far_silent),
                       "unwitnessed_far": len(unw_far),
                       "rated": len(rated_rows),
                       "below_floor": len(excluded_rows)},
            "verdicts": tally,
            "per_split": per_split,
            "populations": {name: {"n": s["n"], "dist_q1_med_q3": s["dist_q"],
                                   "px_q1_med_q3": s["px_q"]}
                            for name, s in stats.items()},
            "auc_rated_vs_unrated_px": auc_rated,
            "rated_median_px_percentile": med_rank,
            "auc_hit_vs_silent_px": auc_hit,
            "bands": band_rows,
            "matched_size_detection_rate_q1_med_q3": rq,
            "above_horizon": n_clamp,
            "rated_by_tier": tiers,
        }
        # newline="" so the artifact is LF on every platform. Python's text
        # mode would write CRLF here on Windows, which makes a regenerated
        # copy fail a byte comparison against the committed one even when
        # every number is identical — and a content hash that only holds on
        # one OS is not a content hash.
        with open(args.json_out, "w", encoding="utf-8", newline="") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
