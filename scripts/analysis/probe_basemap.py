"""Is a city's aerial basemap good enough to measure a 1-2 m offset on? (issue #96)

§5e's generalisable lesson is that **every city needs its municipal basemap
located and checked before its review sheet is worth a reviewer's time** — the
global Esri fallback renders leaf-on and visibly upsampled, and past its deepest
level serves a flat grey "Map data not yet available" tile that a naive fetcher
pastes in as evidence. This is that check, made repeatable.

Three failure modes, each of which has actually bitten:

* **Not a cache at all.** A `MapServer` may be dynamic-only, with no
  `/tile/{z}/{y}/{x}` endpoint. It can still be usable through `/export`, which
  renders an arbitrary bbox — that is how New York State's ortho service works —
  but it needs a different fetcher, so the sheet has to know which it is.
* **A non-Web-Mercator cache.** Municipal caches are often built in a state-plane
  CRS. The sheet's tile math assumes EPSG:3857, so a mismatch would place every
  crosshair wrongly with no visible symptom.
* **A declared depth the cache does not have.** King County's Seattle aerials
  advertise `maxLOD 23` (0.013 m/px) and serve **404 above z20** (0.101 m/px).
  Trusting the metadata would have built a sheet of missing tiles. So the deepest
  level is found by *probing*, never by reading.

    python scripts/analysis/probe_basemap.py \
        --url https://gismaps.kingcounty.gov/arcgis/rest/services/BaseMaps/KingCo_Aerial_2021/MapServer \
        --at 47.6089 -122.3356

Network is all this needs. Pure helpers are unit-tested in
``tests/test_probe_basemap.py``.
"""
import argparse
import io
import json
import math
import sys
import urllib.request

USER_AGENT = "RampNet-sourcing/1.0 (+https://github.com/ProjectSidewalk/RampNet)"
WEBMERC_R0 = 156543.03392800014      # m/px at z0, 256 px tiles, at the equator
WEBMERC_WKIDS = (3857, 102100)
# A ramp is 1.2-1.8 m deep and its detectable-warning pad ~0.6 m. Below roughly
# 0.15 m/px the pad stops being individually visible, which is the feature the
# reviewer uses to find the ramp's near edge. Denver's usable cache is 0.057.
GOOD_MPP = 0.08
USABLE_MPP = 0.15


def metres_per_pixel(zoom, lat, tile_px=256):
    """Ground resolution of a Web Mercator pixel at ``lat``. Pure."""
    return WEBMERC_R0 * (256.0 / tile_px) / (2 ** zoom) * math.cos(math.radians(lat))


def tile_xy(lon, lat, zoom):
    """Web Mercator tile containing lon/lat. Pure."""
    n = 2 ** zoom
    s = max(-0.9999, min(0.9999, math.sin(math.radians(lat))))
    return (int((lon + 180.0) / 360.0 * n),
            int((0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n))


def is_web_mercator(tile_info):
    """Does this cache use the standard EPSG:3857 ladder? Pure.

    Both halves matter: the right CRS with a bespoke resolution ladder still
    breaks the sheet's tile math.
    """
    if not tile_info:
        return False
    sr = tile_info.get("spatialReference", {})
    wkid = sr.get("latestWkid") or sr.get("wkid")
    if wkid not in WEBMERC_WKIDS:
        return False
    for lod in tile_info.get("lods", []) or []:
        expected = WEBMERC_R0 * (256.0 / (tile_info.get("rows") or 256)) / (2 ** lod["level"])
        if abs(lod["resolution"] - expected) / expected > 1e-6:
            return False
    return True


def looks_blank(mean, stddev):
    """Served placeholder rather than imagery? Same test as the review sheet."""
    return stddev <= 6.0 and 150 <= mean <= 235


def grade(mpp):
    """Verdict on a resolution, in the terms §5e uses. Pure."""
    if mpp <= GOOD_MPP:
        return "GOOD -- warning pads individually visible"
    if mpp <= USABLE_MPP:
        return "USABLE -- coarser than Denver, offsets floor higher"
    return "TOO COARSE -- cannot measure a 1-2 m offset"


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout).read()


def _stats(blob):
    from PIL import Image, ImageStat
    im = Image.open(io.BytesIO(blob)).convert("L")
    st = ImageStat.Stat(im)
    return im.size, st.mean[0], st.stddev[0]


def probe(service_url, lat, lon, lo=14, hi=23, timeout=30):
    """Describe a service and find the deepest level it ACTUALLY serves."""
    out = {"url": service_url, "lat": lat, "lon": lon}
    try:
        meta = json.loads(_get(service_url + "?f=json", timeout))
    except Exception as exc:
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
        return out
    ti = meta.get("tileInfo")
    out["cached"] = bool(ti)
    out["supports_export"] = "Map" in (meta.get("capabilities") or "") or not ti
    if ti:
        sr = ti.get("spatialReference", {})
        out["wkid"] = sr.get("latestWkid") or sr.get("wkid")
        out["tile_px"] = ti.get("rows")
        out["web_mercator"] = is_web_mercator(ti)
        lods = ti.get("lods") or []
        out["declared_max_lod"] = lods[-1]["level"] if lods else None
    else:
        out["web_mercator"] = None
        out["declared_max_lod"] = None
        return out

    deepest, levels = None, {}
    for z in range(hi, lo - 1, -1):
        x, y = tile_xy(lon, lat, z)
        url = "%s/tile/%d/%d/%d" % (service_url, z, y, x)
        try:
            blob = _get(url, timeout)
        except Exception as exc:
            levels[z] = "unavailable (%s)" % getattr(exc, "code", type(exc).__name__)
            continue
        try:
            _, mean, sd = _stats(blob)
        except Exception:
            levels[z] = "undecodable"
            continue
        if looks_blank(mean, sd):
            levels[z] = "blank placeholder"
            continue
        levels[z] = "imagery (sd %.1f)" % sd
        if deepest is None:
            deepest = z
    out["levels"] = levels
    out["deepest_served"] = deepest
    if deepest is not None:
        out["metres_per_pixel"] = metres_per_pixel(deepest, lat, out.get("tile_px") or 256)
        out["grade"] = grade(out["metres_per_pixel"])
        out["declared_but_absent"] = (out["declared_max_lod"] or 0) - deepest
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", required=True, help="ArcGIS MapServer base URL (no /tile)")
    ap.add_argument("--at", nargs=2, type=float, required=True, metavar=("LAT", "LON"),
                    help="a dense point in the city -- probe where imagery should exist")
    ap.add_argument("--lo", type=int, default=14)
    ap.add_argument("--hi", type=int, default=23)
    args = ap.parse_args(argv)

    r = probe(args.url, args.at[0], args.at[1], args.lo, args.hi)
    if "error" in r:
        print("UNREACHABLE: %s" % r["error"])
        return 1
    print(r["url"])
    if not r["cached"]:
        print("  NOT a cached tile service -- dynamic only.")
        print("  Usable via /export (arbitrary bbox), but that needs a different fetcher")
        print("  than the review sheet's tile path.")
        return 0
    print("  cache CRS      : wkid %s  (%s)" % (
        r["wkid"], "standard Web Mercator" if r["web_mercator"]
        else "NOT standard Web Mercator -- the sheet's tile math does not apply"))
    print("  tile size      : %s px" % r["tile_px"])
    print("  declared maxLOD: %s" % r["declared_max_lod"])
    print("  deepest SERVED : %s" % r["deepest_served"])
    if r.get("declared_but_absent", 0) > 0:
        print("  !! %d declared level(s) are not actually built -- metadata overstates depth"
              % r["declared_but_absent"])
    for z in sorted(r["levels"], reverse=True):
        print("     z%-2d %8.4f m/px  %s" % (z, metres_per_pixel(z, r["lat"],
                                                                r.get("tile_px") or 256),
                                             r["levels"][z]))
    if r.get("metres_per_pixel"):
        print("  => %.4f m/px : %s" % (r["metres_per_pixel"], r["grade"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
