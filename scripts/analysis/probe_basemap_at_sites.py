"""Grade a basemap **at the records it will be used to review** (issue #96).

`probe_basemap.py` checks ONE dense point — "probe where imagery should exist".
Every basemap failure on this issue has been *somewhere else in the city*:
Charlotte passed that probe and then built a 1-of-60 sheet, because its server
404s ~20-35% of tiles at random (§5h). And the vegetation figures quoted in
`inventory_review_sheet.py`'s source notes — Denver 7.2%, King County 28.3%,
Charlotte 16.2% — were computed ad hoc and never committed, so they could not be
re-derived or compared against a new candidate year.

This closes both gaps. It samples a city's **actual inventory records** with the
same `uniform_sample(seed)` the sheet uses, so passing `--seed` from a built
sheet probes exactly the sites that sheet will show, and reports per source:

* **coverage** — fraction of sites whose centre tile returns imagery. This is the
  number that predicts whether a sheet will build.
* **vegetation** — fraction of pixels with excess-green ``ExG = 2G - R - B``
  above a threshold, plus mean ExG. This is the number that predicts whether a
  reviewer can see a ramp under canopy.
* **blank rate** — tiles that return 200 with no content (Esri's grey "not yet
  available" tiles, §5e).

Two or more ``--source`` arguments are compared **at identical sites**, so the
difference is the imagery rather than the sample — the paired design that makes
"is 2025 leafier than 2019?" answerable at n=40 instead of n=400.

    python scripts/analysis/probe_basemap_at_sites.py \
        --inventory data/inventories/seattle-wa-2026-07-31.jsonl.gz \
        --source seattle-2019 --source seattle-2025 --sample 40 --seed 20260731

**Calibration:** the ExG threshold is set so this reproduces the committed
figures for the two anchor sources (Denver leaf-off ~7%, King County 2019 ~28%).
`--calibrate` prints those two and nothing else, so drift is visible.
"""
import argparse
import io
import json
import math
import os
import sys
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inventory_review_sheet import (  # noqa: E402
    TILE_SOURCES, USER_AGENT, load_inventory, uniform_sample)

#: Excess-green cut. ExG = 2G - R - B on 0-255 channels; a pixel above this is
#: called vegetation. Calibrated against the two anchor basemaps — see --calibrate.
EXG_THRESHOLD = 20.0


def tile_xy(lon, lat, zoom):
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    r = math.radians(lat)
    y = int((1.0 - math.log(math.tan(r) + 1.0 / math.cos(r)) / math.pi) / 2.0 * n)
    return x, y


def fetch(url, timeout=30, retries=4):
    """Fetch a tile. Retries a 404 before believing it — Charlotte's server
    404s tiles that exist, and a single-shot probe would report a coverage hole
    that is really a transient (§5h)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    import time
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                return 200, fh.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (404, 400):
                return exc.code, b""
            if attempt < retries - 1:
                time.sleep(0.4 * (attempt + 1))
        except Exception:
            return "ERR", b""
    return 404, b""


def tile_stats(blob, exg_threshold=EXG_THRESHOLD):
    """(vegetation_fraction, mean_exg, stddev_luma) for one tile."""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"), dtype=np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    exg = 2.0 * g - r - b
    luma = a.mean(axis=2)
    return float((exg > exg_threshold).mean()), float(exg.mean()), float(luma.std())


def probe_source(key, sites, exg_threshold=EXG_THRESHOLD, zoom=None):
    src = TILE_SOURCES[key]
    z = zoom or src["max_zoom"]
    got, blank, veg, exg, codes = 0, 0, [], [], {}
    for lon, lat in sites:
        x, y = tile_xy(lon, lat, z)
        code, blob = fetch(src["url"].format(z=z, x=x, y=y))
        codes[code] = codes.get(code, 0) + 1
        if code != 200 or not blob:
            continue
        v, e, sd = tile_stats(blob, exg_threshold)
        if sd < 3.0:
            blank += 1
            continue
        got += 1
        veg.append(v)
        exg.append(e)
    n = len(sites)
    return {
        "source": key, "zoom": z, "attribution": src["attribution"],
        "n_sites": n, "n_imagery": got, "n_blank": blank,
        "coverage": round(got / n, 4) if n else None,
        "blank_rate": round(blank / n, 4) if n else None,
        "vegetation_frac": round(sum(veg) / len(veg), 4) if veg else None,
        "mean_exg": round(sum(exg) / len(exg), 2) if exg else None,
        "http": {str(k): v for k, v in codes.items()},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inventory")
    ap.add_argument("--source", action="append", default=[],
                    help="tile-source key; repeat to compare at identical sites")
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260731,
                    help="pass the SHEET's seed to probe the sheet's own sites")
    ap.add_argument("--zoom", type=int, default=None)
    ap.add_argument("--exg-threshold", type=float, default=EXG_THRESHOLD)
    ap.add_argument("--calibrate", action="store_true",
                    help="probe the two anchor sources and stop")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.calibrate:
        print("ExG threshold %.1f — committed notes say Denver leaf-off ~7.2%%, "
              "King County 2019 ~28.3%%" % args.exg_threshold)
        for inv, key in (("denver-co", "denver-2016"), ("seattle-wa", "seattle-2019")):
            import glob
            c = [f for f in glob.glob(os.path.join(REPO, "data", "inventories", inv + "-*.jsonl.gz"))
                 if "centerline" not in f]
            if not c:
                print(f"  {key}: no frozen inventory for {inv}")
                continue
            rows = load_inventory(c[0])
            idx = uniform_sample(len(rows), args.sample, args.seed)
            sites = [(rows[i]["lon"], rows[i]["lat"]) for i in idx]
            r = probe_source(key, sites, args.exg_threshold, args.zoom)
            print(f"  {key:14s} coverage {r['coverage']:.2f}  "
                  f"vegetation {r['vegetation_frac']:.3f}  mean ExG {r['mean_exg']}")
        return 0

    if not args.inventory or not args.source:
        ap.error("--inventory and at least one --source are required")

    rows = load_inventory(args.inventory)
    idx = uniform_sample(len(rows), args.sample, args.seed)
    sites = [(rows[i]["lon"], rows[i]["lat"]) for i in idx]
    print(f"{len(rows)} records; probing {len(sites)} sites "
          f"(seed {args.seed}) — IDENTICAL sites across sources", file=sys.stderr)

    results = []
    for key in args.source:
        r = probe_source(key, sites, args.exg_threshold, args.zoom)
        results.append(r)
        print(f"  {key} done", file=sys.stderr)

    print(f"\n{'source':16s} {'zoom':>4s} {'coverage':>9s} {'blank':>6s} "
          f"{'vegetation':>10s} {'meanExG':>8s}  http")
    for r in results:
        cov = "-" if r["coverage"] is None else f"{r['coverage']:.3f}"
        veg = "-" if r["vegetation_frac"] is None else f"{r['vegetation_frac']:.3f}"
        print(f"{r['source']:16s} {r['zoom']:>4d} {cov:>9s} {r['blank_rate']:>6.3f} "
              f"{veg:>10s} {str(r['mean_exg']):>8s}  {r['http']}")

    if len(results) == 2:
        a, b = results
        if a["vegetation_frac"] is not None and b["vegetation_frac"] is not None:
            d = b["vegetation_frac"] - a["vegetation_frac"]
            print(f"\npaired at identical sites: {b['source']} vegetation is "
                  f"{d:+.3f} vs {a['source']} "
                  f"({'leafier' if d > 0 else 'clearer'}); coverage "
                  f"{b['coverage'] - a['coverage']:+.3f}")

    payload = {"inventory": os.path.basename(args.inventory), "seed": args.seed,
               "sample": args.sample, "exg_threshold": args.exg_threshold,
               "sources": results}
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
