"""Turn a filled review sheet into the offset distribution §5 asks for.

Second half of the location-precision gate (issues #96, #59). Reads the
``verdicts.json`` written by ``inventory_review_sheet.py`` after a reviewer has
filled it in, and reports what the paper's Good/OK/Poor buckets leave implicit.

    python scripts/analysis/inventory_precision_score.py \
        --verdicts analysis_out/review_denver-co/verdicts.json \
        --control  analysis_out/review_bend-or/verdicts.json

**This deliberately does not assign a tier.** Paper §3.1 / Table 1 published
buckets but no thresholds, so "OK" may mean 2 m or 8 m — and inventing a cutoff
here would replace one unstated judgment with another, dressed up as a
measurement. What makes a number interpretable is a **control**: run the same
sheet over a city Table 1 already rated **Good** (Bend, Portland or NYC, all
already in training so nothing is lost by looking), have the reviewer score it in
the same sitting, and read the candidate against it. ``--control`` does the
comparison and reports the gap; without one, the output is a distribution and an
explicit refusal to grade.

Three things get reported, and the second is the one that decides Denver:

* **Offset distribution** — quantiles and the share within 1/2/5 m. The Stage 1
  mechanism: a coordinate that misses the ramp puts the label on the wrong pixels.
* **``ramps_visible``** — how many ramps the reviewer counted on each corner. This
  is the human half of the per-ramp/per-corner question, and it separates the two
  readings geometry cannot (`inventory_geometry.py` §5d): a city recording one
  point per corner shows **2 ramps visible where it published 1 record**, whereas
  a city of genuine single diagonal aprons shows 1 and 1. The comparison is
  against ``--records-per-corner`` from the geometry run.
* **Unreadable rate** — a sheet that is 40% unreadable has not assessed the city,
  whatever the surviving 60% say, so this is reported before anything else.

The core is pure and unit-tested in ``tests/test_inventory_precision_score.py``.
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))

OFFSET_BINS = [0, 0.5, 1, 2, 3, 5, 8, 12, 20, float("inf")]

# Below this share of judgeable chips, the sample is too thin to describe a city
# and the run should be repeated on better imagery rather than reported.
MIN_READABLE_SHARE = 0.6


def quantiles(values, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {str(q): None for q in qs}
    out = {}
    for q in qs:
        pos = q * (len(vals) - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        out[str(q)] = vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)
    return out


def histogram(values, edges):
    counts = [0] * (len(edges) - 1)
    for v in values:
        if v is None:
            continue
        for k in range(len(edges) - 1):
            if edges[k] <= v < edges[k + 1]:
                counts[k] += 1
                break
    return counts


def partition(records):
    """Split into (judged, unreadable, unscored).

    ``unscored`` — reviewed neither way — is reported rather than folded into
    either side: a half-finished sheet and a hard-to-read one are different
    problems, and silently treating blanks as unreadable would hide the first.
    """
    judged, unreadable, unscored = [], [], []
    for r in records:
        if r.get("unreadable"):
            unreadable.append(r)
        elif r.get("offset_m") is None:
            unscored.append(r)
        else:
            judged.append(r)
    return judged, unreadable, unscored


def score(verdicts):
    """Summarise one filled sheet. Pure."""
    records = verdicts.get("records", [])
    judged, unreadable, unscored = partition(records)
    reviewed = len(judged) + len(unreadable)
    offsets = [r["offset_m"] for r in judged]

    visible = Counter(r["ramps_visible"] for r in judged
                      if r.get("ramps_visible") is not None)
    n_visible = sum(visible.values())

    on_corner = [r["on_corner"] for r in judged if r.get("on_corner") is not None]

    return {
        "city": verdicts.get("city"),
        "inventory": verdicts.get("inventory"),
        "imagery": verdicts.get("imagery"),
        "reviewer": verdicts.get("reviewer"),
        "confidence": verdicts.get("confidence"),
        "chips": len(records),
        "reviewed": reviewed,
        "complete": len(unscored) == 0 and reviewed == len(records),
        "unscored": len(unscored),
        "readable": {
            "judged": len(judged),
            "unreadable": len(unreadable),
            "share": (len(judged) / float(reviewed)) if reviewed else None,
            "sufficient": (len(judged) / float(reviewed) >= MIN_READABLE_SHARE)
                          if reviewed else False,
        },
        "offset_m": {
            "n": len(offsets),
            "quantiles": quantiles(offsets),
            "mean": (sum(offsets) / len(offsets)) if offsets else None,
            "histogram_edges": OFFSET_BINS[:-1] + ["inf"],
            "histogram": histogram(offsets, OFFSET_BINS),
            "share_within_1m": _share(offsets, 1.0),
            "share_within_2m": _share(offsets, 2.0),
            "share_within_5m": _share(offsets, 5.0),
        },
        "on_corner": {
            "n": len(on_corner),
            "share": (sum(1 for v in on_corner if v) / float(len(on_corner)))
                     if on_corner else None,
        },
        "ramps_visible": {
            "n": n_visible,
            "histogram": {str(k): visible[k] for k in sorted(visible)},
            "mean": (sum(k * v for k, v in visible.items()) / float(n_visible))
                    if n_visible else None,
        },
        "tier": None,
        "tier_note": "Not assigned. Table 1 published buckets but no thresholds; "
                     "grade by comparing against a known-Good control, not against "
                     "a cutoff invented here.",
    }


def _share(values, radius):
    if not values:
        return None
    return sum(1 for v in values if v <= radius) / float(len(values))


def compare_to_control(candidate, control):
    """Read a candidate against a Table-1 Good city scored the same way."""
    def gap(a, b):
        return None if (a is None or b is None) else a - b
    c_off, k_off = candidate["offset_m"], control["offset_m"]
    return {
        "control_city": control["city"],
        "median_offset_m": {"candidate": c_off["quantiles"]["0.5"],
                            "control": k_off["quantiles"]["0.5"],
                            "gap": gap(c_off["quantiles"]["0.5"], k_off["quantiles"]["0.5"])},
        "share_within_2m": {"candidate": c_off["share_within_2m"],
                            "control": k_off["share_within_2m"],
                            "gap": gap(c_off["share_within_2m"], k_off["share_within_2m"])},
        "unreadable_share": {
            "candidate": 1 - (candidate["readable"]["share"] or 0),
            "control": 1 - (control["readable"]["share"] or 0)},
        "note": "A candidate at or better than the control on median offset and "
                "share-within-2m has met the bar the paper's Good tier actually "
                "represents. Imagery differs between the two sheets, so the "
                "unreadable shares are a confound, not a finding.",
    }


def compare_to_geometry(scored, records_per_corner):
    """Does the reviewer's ramp count corroborate the geometric corner ratio?

    The decisive test for a city like Denver, whose 1.21 records/corner is
    ambiguous between "records one point per corner" and "has single diagonal
    aprons". If the reviewer counts ~2 ramps visible where the inventory holds
    ~1.2 records, it is the first and the labels merge pairs; if they count ~1.2,
    it is the second and nothing is being lost.
    """
    mean_visible = scored["ramps_visible"]["mean"]
    if mean_visible is None or not records_per_corner:
        return None
    ratio = mean_visible / records_per_corner
    return {
        "mean_ramps_visible": mean_visible,
        "records_per_corner": records_per_corner,
        "visible_per_record": ratio,
        "reading": ("under-recorded: the reviewer sees more ramps than the "
                    "inventory holds, so paired ramps are collapsing to one label"
                    if ratio > 1.25 else
                    "consistent: the inventory records about what is on the ground"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--control", default=None,
                    help="verdicts.json for a Table-1 Good city, scored in the same sitting")
    ap.add_argument("--records-per-corner", type=float, default=None,
                    help="from inventory_geometry.py, to corroborate ramps_visible")
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args(argv)

    with open(args.verdicts) as fh:
        scored = score(json.load(fh))
    if args.control:
        with open(args.control) as fh:
            scored["control"] = compare_to_control(scored, score(json.load(fh)))
    if args.records_per_corner:
        scored["geometry_check"] = compare_to_geometry(scored, args.records_per_corner)

    r = scored["readable"]
    print("{}: {} chips, {} reviewed, {} unscored".format(
        scored["city"], scored["chips"], scored["reviewed"], scored["unscored"]))
    if not scored["complete"]:
        print("  ! sheet is not fully reviewed — figures below are partial")
    print("  readable {}/{} ({})".format(
        r["judged"], r["judged"] + r["unreadable"],
        "sufficient" if r["sufficient"] else "TOO THIN — repeat on better imagery"))
    q = scored["offset_m"]["quantiles"]
    if q["0.5"] is not None:
        print("  offset median {:.2f} m | p95 {:.2f} m | within 2 m {:.3f}".format(
            q["0.5"], q["0.95"], scored["offset_m"]["share_within_2m"]))
    if scored["ramps_visible"]["mean"] is not None:
        print("  ramps visible mean {:.2f}  {}".format(
            scored["ramps_visible"]["mean"], scored["ramps_visible"]["histogram"]))
    if scored.get("geometry_check"):
        print("  vs geometry: {:.2f} visible per record — {}".format(
            scored["geometry_check"]["visible_per_record"],
            scored["geometry_check"]["reading"]))
    if scored.get("control"):
        c = scored["control"]
        print("  vs control {}: median gap {}".format(
            c["control_city"], c["median_offset_m"]["gap"]))
    print("  tier: NOT ASSIGNED — see tier_note")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "inventory_precision_{}.json".format(scored["city"]))
    with open(path, "w") as fh:
        json.dump(scored, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("wrote {}".format(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
