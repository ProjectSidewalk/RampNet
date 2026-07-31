"""Is the review sheet's crosshair actually on the coordinate, and are the rings true?

Instrument check for the location-precision gate (issues #96, #59).

``inventory_review_sheet.py`` asks a reviewer to judge a **1-2 m** offset. Every
verdict it produces is therefore only as good as two claims that are easy to
assert and easy to get wrong:

1. **Registration** — the crosshair sits on the published coordinate, not a metre
   off. A systematic shift here would bias every offset in the same direction,
   and nothing in the sheet would look wrong.
2. **Scale** — the 1/2/5/10 m rings really are those radii on the ground.

Neither is verifiable by staring at the sheet, because the error and the
measurement come from the same code. So both are checked against something
external:

* **Scale** is checked against the **WGS84 ellipsoid**, not against the projection
  under test. Two points are constructed an exact ground distance apart using the
  local radii of curvature, then projected: if the ring maths is right they land
  exactly one ring-radius apart in pixels. Web Mercator's ``cos(lat)`` scale
  factor plays no part in constructing the pair, so a mistake in it cannot cancel.
* **Registration** is checked against **independent municipal geometry** — street
  centrelines from the city's own LRS, drawn into the chip with the same
  projection the crosshair uses. Centrelines are ground-level, so unlike building
  footprints they carry no roof-lean parallax: if they run down the middle of the
  visible roadway, the imagery and the vector data agree in the chip's pixel
  space to well under a metre.

    python scripts/analysis/verify_chip_georeference.py --city denver-co
    python scripts/analysis/verify_chip_georeference.py --city seattle-wa

Each city in ``CITIES`` registers its own centreline layer, basemap and sample
neighbourhoods; the basemap must be the one its review sheet was built on, or
this is checking a different instrument than the one being attributed.

The scale half is pure and unit-tested (``tests/test_verify_chip_georeference.py``);
the registration half needs network and writes PNGs for a human to look at, which
is the point — it produces evidence, not a boolean.

**This is one leg of a triangle.** With ``inventory_centerline_offset.py`` (ramps
vs centrelines, no imagery) and a filled review sheet (ramps vs imagery), the
three measurements must satisfy

    (ramps vs imagery) = (ramps vs centrelines) + (centrelines vs imagery)

which turns a systematic offset from an unattributed fact into a located one:
whichever pair disagrees is where the error lives. Because each leg is measured
independently, the identity is a check rather than an assumption.

**The residual this exists to expose.** Denver publishes in EPSG:2877 and Seattle
in EPSG:2926, both NAD83, and both servers reproject to 4326; the imagery is
tiled from state-plane sources. If either side applied a real NAD83->WGS84 datum
shift while the other used the null transform, the two would disagree by roughly
a metre in CONUS — and by construction the ramp coordinates would inherit it,
since they travel the same reprojection path as these centrelines. That is
exactly the shape of an error the ramps-vs-centrelines leg is blind to, which is
why both legs are needed.
"""
import argparse
import json
import math
import os
import sys
import urllib.parse
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import inventory_review_sheet as irs  # noqa: E402

OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))

# WGS84
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)

# Per-city registry. ``centerlines`` must be the city's OWN street geometry from
# the SAME publisher as its ramp inventory -- that is what makes the check
# independent of the inventory while sharing its reprojection path.
#
# ``sites`` are neighbourhoods spread across the city, deliberately covering more
# than one street orientation so a shift cannot hide by being parallel to one
# grid. Coordinates are neighbourhood centroids, not features -- an earlier
# version used a made-up "arterial" point that landed on a house, which proved
# nothing. ``visual`` are the two chips a human looks at.
CITIES = {
    "denver-co": {
        "centerlines": ("https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/ArcGIS/rest"
                        "/services/ODC_TRANS_STREETROUTE_U/FeatureServer/146"),
        "tile_source": "denver-2016",
        "sites": [
            ("park-hill", -104.9280, 39.7500),
            ("berkeley", -105.0400, 39.7770),
            ("athmar", -105.0000, 39.6960),
            ("hampden", -104.9200, 39.6700),
            ("montbello", -104.8500, 39.7830),
        ],
        "visual": [
            ("residential", -104.947384, 39.732141),
            ("downtown-diagonal", -104.9911615, 39.7461177),
        ],
    },
    "seattle-wa": {
        # SDOT's Street Network Database -- same ArcGIS org (ZOyb2t4B0UYuYNYH)
        # and same native CRS (EPSG:2926) as Curb_Ramps_(Active), so both travel
        # the identical reprojection to 4326.
        "centerlines": ("https://services.arcgis.com/ZOyb2t4B0UYuYNYH/arcgis/rest"
                        "/services/Street_Network_Database_SND/FeatureServer/0"),
        # Must match the sheet the verdicts came from, or the check is measuring
        # a different instrument than the one being attributed.
        "tile_source": "seattle-2019",
        "sites": [
            ("wallingford", -122.3340, 47.6600),
            ("greenwood", -122.3550, 47.6900),
            ("beacon-hill", -122.3110, 47.5750),
            ("columbia-city", -122.2870, 47.5600),
            ("west-seattle", -122.3870, 47.5610),
        ],
        "visual": [
            ("residential", -122.334000, 47.660000),
            ("arterial", -122.311000, 47.575000),
        ],
    },
}


def local_radii(lat_deg):
    """Meridional and prime-vertical radii of curvature on WGS84, in metres. Pure."""
    s = math.sin(math.radians(lat_deg))
    w = math.sqrt(1 - WGS84_E2 * s * s)
    return (WGS84_A * (1 - WGS84_E2) / w ** 3), (WGS84_A / w)


def offset_lonlat(lon, lat, east_m, north_m):
    """Move a point an exact ground distance on the ellipsoid. Pure.

    Uses the local radii of curvature, which is accurate to well under a
    millimetre at the tens-of-metres scale this checks — and, crucially, is
    derived from the ellipsoid rather than from the Web Mercator formula under
    test, so an error in that formula cannot cancel itself out here.
    """
    m_rad, n_rad = local_radii(lat)
    dlat = north_m / m_rad
    dlon = east_m / (n_rad * math.cos(math.radians(lat)))
    return lon + math.degrees(dlon), lat + math.degrees(dlat)


def ring_scale_error(lat, zoom, radii_m, bearings=(0, 45, 90, 135, 180, 225, 270, 315)):
    """For each ring radius, the worst relative error over several bearings.

    Constructs a point exactly ``r`` metres away on the ellipsoid, projects both
    it and the centre, and compares the pixel separation against the radius the
    sheet would draw (``r / metres_per_pixel``). Several bearings because Web
    Mercator is conformal — an error that only showed up north-south would be
    invisible in a single east-west test.
    """
    mpp = irs.metres_per_pixel(lat, zoom)
    cx, cy = irs.lonlat_to_pixel(0.0, lat, zoom)
    out = []
    for r in radii_m:
        worst, at = 0.0, None
        for b in bearings:
            th = math.radians(b)
            lon2, lat2 = offset_lonlat(0.0, lat, r * math.sin(th), r * math.cos(th))
            px, py = irs.lonlat_to_pixel(lon2, lat2, zoom)
            drawn = r / mpp
            got = math.hypot(px - cx, py - cy)
            err = abs(got - drawn) / drawn
            if err > worst:
                worst, at = err, b
        out.append({"radius_m": r, "drawn_px": r / mpp,
                    "max_rel_error": worst, "worst_bearing_deg": at,
                    "max_abs_error_m": worst * r})
    return out


def road_centre_offset(lum, cx, cy, ux, uy, half_px, min_step):
    """Signed offset from a centreline point to the roadway's optical centre.

    Walks outward from the centreline along the perpendicular ``(ux, uy)`` and
    takes the first strong brightening in each direction as the pavement edge —
    kerb, gutter or the grass beyond it. The midpoint of those two edges is where
    the road actually is; the difference is the registration error in pixels, and
    its sign says which way.

    Returns ``None`` when either edge is not found inside ``half_px``, which is
    the honest answer for a cross-section blocked by a parked car, a tree crown or
    a driveway apron. Those are common enough that the aggregate has to be a
    **median over many cross-sections**, never a single reading.

    Pure: takes a luminance sampler, not an image.
    """
    edges = []
    for sign in (1, -1):
        base, found = None, None
        for t in range(2, int(half_px)):
            v = lum(cx + sign * ux * t, cy + sign * uy * t)
            if v is None:
                break
            if base is None:
                base = v
            base = min(base, v)
            if v - base >= min_step:
                found = t
                break
        if found is None:
            return None
        edges.append(sign * found)
    return (edges[0] + edges[1]) / 2.0


def _segment_normal(p, q):
    """Unit normal to a segment in pixel space, or None for a degenerate one."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return None
    return (-dy / n, dx / n, n)


def _get_json(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": irs.USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.load(fh)


def fetch_centerlines(lon, lat, span_m, layer):
    """Street centrelines intersecting the chip, as lists of (lon, lat)."""
    m_rad, n_rad = local_radii(lat)
    dlat = math.degrees((span_m * 0.75) / m_rad)
    dlon = math.degrees((span_m * 0.75) / (n_rad * math.cos(math.radians(lat))))
    env = {"xmin": lon - dlon, "ymin": lat - dlat,
           "xmax": lon + dlon, "ymax": lat + dlat,
           "spatialReference": {"wkid": 4326}}
    q = urllib.parse.urlencode({
        "f": "json", "where": "1=1", "geometry": json.dumps(env),
        "geometryType": "esriGeometryEnvelope", "inSR": "4326", "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects", "returnGeometry": "true",
        "outFields": "*"})
    d = _get_json(layer + "/query?" + q)
    if "error" in d:
        raise RuntimeError(d["error"])
    return [path for f in d.get("features", [])
            for path in (f.get("geometry") or {}).get("paths", [])]


def draw_registration_chip(lon, lat, zoom, span_m, cache_dir, tile_url, paths):
    """Chip with the municipal centrelines drawn in the chip's own pixel space."""
    from PIL import ImageDraw
    chip, mpp, _keys, _blank = irs.render_chip(lon, lat, zoom, span_m, cache_dir, tile_url)
    span_px = chip.size[0]
    px0, py0 = irs.lonlat_to_pixel(lon, lat, zoom)
    ox, oy = px0 - span_px / 2.0, py0 - span_px / 2.0

    def to_chip(p):
        x, y = irs.lonlat_to_pixel(p[0], p[1], zoom)
        return (x - ox, y - oy)

    d = ImageDraw.Draw(chip, "RGBA")
    for path in paths:
        pts = [to_chip(p) for p in path]
        if len(pts) > 1:
            d.line(pts, fill=(0, 229, 255, 190), width=max(2, span_px // 320))
    c = span_px / 2.0
    for r_m in (2.0, 10.0):
        r = r_m / mpp
        d.ellipse([c - r, c - r, c + r, c + r], outline=(255, 235, 59, 230), width=2)
    for a, b in ((-span_px / 18, -span_px / 46), (span_px / 46, span_px / 18)):
        d.line([c + a, c, c + b, c], fill=(255, 64, 64, 255), width=3)
        d.line([c, c + a, c, c + b], fill=(255, 64, 64, 255), width=3)
    return chip, mpp


def measure_registration(lon, lat, zoom, box_m, cache_dir, tile_url, layer,
                         step_m=4.0, half_m=12.0, min_step=18):
    """Measure centreline-to-roadway offset over a whole neighbourhood.

    Renders one mosaic, drops the city's centrelines into it, and takes a
    cross-section every ``step_m`` along every segment. Reports the distribution
    of offsets in metres.

    **What a non-zero median would mean.** A systematic shift between the vector
    data and the imagery — most plausibly a NAD83/WGS84 datum transform applied on
    one side and not the other, which is about a metre in CONUS. Because the ramp
    coordinates travel the same reprojection path as these centrelines, that shift
    would land in every offset a reviewer records, in the same direction, and
    nothing in the review sheet would look wrong.

    **What it cannot rule out.** A centreline is a cartographic construct, not a
    survey of the pavement's midline: crowned roads, one-sided parking bays and
    kerb extensions all move the optical centre without moving the true one. So a
    median within a few tens of centimetres is evidence of no gross error, not a
    calibration certificate.
    """
    from PIL import Image
    mpp = irs.metres_per_pixel(lat, zoom)
    span_px = int(round(box_m / mpp))
    x0, y0, x1, y1, ox, oy = irs.tile_range(lon, lat, zoom, span_px)
    mosaic = Image.new("RGB", ((x1 - x0 + 1) * irs.TILE_PX, (y1 - y0 + 1) * irs.TILE_PX))
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            mosaic.paste(irs._fetch_tile(tile_url.format(z=zoom, x=tx, y=ty), cache_dir),
                         ((tx - x0) * irs.TILE_PX, (ty - y0) * irs.TILE_PX))
    grey = mosaic.convert("L")
    W, H = grey.size
    px = grey.load()

    def lum(x, y):
        xi, yi = int(x), int(y)
        if xi < 0 or yi < 0 or xi >= W or yi >= H:
            return None
        return px[xi, yi]

    def to_mosaic(p):
        mx, my = irs.lonlat_to_pixel(p[0], p[1], zoom)
        return (mx - x0 * irs.TILE_PX, my - y0 * irs.TILE_PX)

    paths = fetch_centerlines(lon, lat, box_m, layer)
    east, north, attempted = [], [], 0
    for path in paths:
        pts = [to_mosaic(p) for p in path]
        for a, b in zip(pts, pts[1:]):
            nrm = _segment_normal(a, b)
            if nrm is None:
                continue
            ux, uy, seg_px = nrm
            n = int(seg_px * mpp / step_m)
            for k in range(1, max(n, 1)):
                f = k / float(n)
                cx, cy = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
                attempted += 1
                d = road_centre_offset(lum, cx, cy, ux, uy, half_m / mpp, min_step)
                if d is None:
                    continue
                # Resolve into a GEOGRAPHIC frame. The segment normal's sign flips
                # with the direction the segment happens to be digitised in, so a
                # real shift -- say a metre east -- would appear as +1 on one
                # segment and -1 on its neighbour and cancel in the median. That
                # is the failure this whole check exists to detect, so it must not
                # be averaged away. In Web Mercator pixel space +x is east and +y
                # is south.
                #
                # A cross-section only constrains the axis it crosses: on a
                # north-south street the normal is east-west, so the sample says
                # nothing about north and contributes an exact zero to it. Pooling
                # both axes in a grid city therefore fills each median with zeros
                # from the streets that could not measure it, and reports 0.00
                # whatever the truth is. Each sample is assigned to the axis it
                # actually measures.
                if abs(ux) >= abs(uy):
                    east.append(d * ux * mpp / abs(ux))
                else:
                    north.append(-d * uy * mpp / abs(uy))
    if not east and not north:
        return {"cross_sections_attempted": attempted, "usable": 0}

    def stats(vals):
        s = sorted(vals)

        def q(p):
            return s[min(len(s) - 1, int(p * len(s)))]
        return {"median": q(0.5), "p25": q(0.25), "p75": q(0.75)}

    e = stats(east) if east else None
    n_ = stats(north) if north else None
    both = sorted(abs(v) for v in east + north)
    return {
        "cross_sections_attempted": attempted,
        "usable": len(east) + len(north),
        "usable_share": (len(east) + len(north)) / float(attempted or 1),
        "east_m": e, "east_n": len(east),
        "north_m": n_, "north_n": len(north),
        "resultant_shift_m": math.hypot(e["median"] if e else 0.0,
                                        n_["median"] if n_ else 0.0),
        "abs_median_m": both[len(both) // 2],
        "metres_per_pixel": mpp,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--city", choices=sorted(CITIES), default="denver-co")
    ap.add_argument("--tile-source", choices=sorted(irs.TILE_SOURCES), default=None,
                    help="defaults to the city's registered basemap")
    ap.add_argument("--zoom", type=int, default=None)
    ap.add_argument("--span-m", type=float, default=60.0)
    ap.add_argument("--box-m", type=float, default=220.0,
                    help="neighbourhood box for the measured registration check")
    ap.add_argument("--out", default=None,
                    help="defaults to analysis_out/georef_check_<city>")
    ap.add_argument("--skip-imagery", action="store_true",
                    help="run only the scale check, which needs no network")
    ap.add_argument("--sites-from-verdicts", default=None,
                    help="measure at the REVIEWED CHIPS instead of the registered "
                         "neighbourhoods. A city-wide average cannot rule out a "
                         "misregistration confined to where the verdicts were "
                         "actually produced, and orthorectification error is "
                         "local — so when attributing a specific review's offsets, "
                         "measure the imagery under that review.")
    args = ap.parse_args(argv)

    city = CITIES[args.city]
    centerlines = city["centerlines"]
    default_sites, visual_sites = city["sites"], city["visual"]
    if args.sites_from_verdicts:
        with open(args.sites_from_verdicts) as fh:
            vd = json.load(fh)
        default_sites = [(str(r["id"]), r["lon"], r["lat"]) for r in vd["records"]
                         if not r.get("unreadable") and r.get("click_px") is not None
                         and r.get("offset_m") is not None]
        print("measuring at {} reviewed chips from {}".format(
            len(default_sites), os.path.basename(args.sites_from_verdicts)))
    out_dir = args.out or os.path.join(OUT, "georef_check_" + args.city)
    src = irs.TILE_SOURCES[args.tile_source or city["tile_source"]]
    zoom = args.zoom if args.zoom is not None else src["max_zoom"]
    lat = default_sites[0][2]

    print("SCALE — rings against the WGS84 ellipsoid (z{}, lat {:.4f})".format(zoom, lat))
    rows = ring_scale_error(lat, zoom, [r for r in irs.RING_RADII_M])
    for r in rows:
        print("  {:>5.1f} m ring = {:7.2f} px | max error {:.4f}% ({:.1f} mm) "
              "at bearing {}".format(r["radius_m"], r["drawn_px"],
                                     100 * r["max_rel_error"],
                                     1000 * r["max_abs_error_m"], r["worst_bearing_deg"]))
    worst = max(r["max_rel_error"] for r in rows)
    print("  verdict: worst ring error {:.4f}% — {}".format(
        100 * worst, "negligible" if worst < 0.01 else "INVESTIGATE"))

    result = {"city": args.city, "tile_source": args.tile_source or city["tile_source"],
              "zoom": zoom, "latitude": lat, "scale_check": rows,
              "tile_scheme": "verified standard Web Mercator: 256 px, EPSG:3857, "
                             "origin -20037508.342787, LOD resolutions match "
                             "156543.03392800014 / 2^z to 3e-10"}

    if not args.skip_imagery:
        os.makedirs(out_dir, exist_ok=True)
        cache_dir = os.path.join(out_dir, "tiles")
        os.makedirs(cache_dir, exist_ok=True)
        print("\nREGISTRATION (measured) — centreline vs the roadway's optical centre")
        print("  {:>12} {:>5} {:>14} {:>5} {:>14} {:>10}".format(
            "site", "nE", "east median", "nN", "north median", "resultant"))
        measured, worst_shift = [], 0.0
        for name, lon, slat in default_sites:
            m = measure_registration(lon, slat, min(zoom, 20), args.box_m,
                                     cache_dir, src["url"], centerlines)
            m["site"] = name
            measured.append(m)
            if not m.get("usable"):
                print("  {:>12}  no usable cross-sections".format(name))
                continue
            e, n_ = m["east_m"], m["north_m"]
            worst_shift = max(worst_shift, m["resultant_shift_m"])
            print("  {:>12} {:>5} {:>14} {:>5} {:>14} {:>8.2f} m".format(
                name, m["east_n"], "{:+.2f} m".format(e["median"]) if e else "n/a",
                m["north_n"], "{:+.2f} m".format(n_["median"]) if n_ else "n/a",
                m["resultant_shift_m"]))
        print("  verdict: worst resultant shift {:.2f} m — {}".format(
            worst_shift,
            "no datum-scale error; a NAD83/WGS84 mismatch would be ~1 m and "
            "consistent in direction" if worst_shift < 0.6 else "INVESTIGATE"))
        result["registration_measured"] = {
            "sites": measured, "box_m": args.box_m, "zoom": min(zoom, 20),
            "interpretation": "Median offset between the city's own street "
                              "centrelines and the optical centre of the roadway in "
                              "the imagery, resolved per axis. A NAD83/WGS84 datum "
                              "mismatch applied on one side only would show as ~1 m, "
                              "consistent in direction across sites.",
            "limits": "A centreline is a cartographic construct, not a survey of the "
                      "pavement midline: crowned roads, one-sided parking bays and "
                      "kerb extensions move the optical centre without moving the "
                      "true one. Read a small median as no gross error, not as a "
                      "calibration certificate.",
        }

        print("\nREGISTRATION (visual) — centrelines drawn into a chip")
        sites = []
        for name, lon, slat in visual_sites:
            paths = fetch_centerlines(lon, slat, args.span_m, centerlines)
            chip, mpp = draw_registration_chip(lon, slat, zoom, args.span_m,
                                               cache_dir, src["url"], paths)
            path = os.path.join(out_dir, "registration_{}.png".format(name))
            chip.save(path)
            print("  {:>18}: {} centreline paths, {:.4f} m/px -> {}".format(
                name, len(paths), mpp, path))
            sites.append({"site": name, "lon": lon, "lat": slat,
                          "paths": len(paths), "png": os.path.basename(path)})
        result["registration_check"] = {
            "layer": centerlines, "span_m": args.span_m, "sites": sites,
            "how_to_read": "Cyan is the city's own street-centreline geometry, "
                           "projected with the same code that places the crosshair. "
                           "If it tracks the middle of the visible roadway, imagery "
                           "and vector data agree in chip pixel space. Centrelines "
                           "are ground-level, so no roof-lean parallax is involved.",
        }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "georef_check.json"), "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("\nwrote {}".format(os.path.join(out_dir, "georef_check.json")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
