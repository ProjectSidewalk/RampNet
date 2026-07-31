"""Build the aerial-overlay review sheet for an inventory's positional precision.

The half of the location-precision gate that needs a human (issues #96, #59). See
``docs/curb_ramp_data_sourcing.md`` §5.

The paper's method (§3.1, Fig. 2) is to **overlay curb-ramp coordinates on aerial
imagery and judge whether they land on the physical ramp**, bucketing a city
Good / OK / Poor. No thresholds were published, and "OK" may mean 2 m or 8 m —
a difference that plausibly decides whether a 90k-record city is usable at all.
§5 asks for the same judgment made quantitative: *"sample ~50 points/city, measure
metres from the true ramp on aerial imagery; report a **distribution**, not a
bucket."*

This builds the instrument for that: an aerial chip per sampled record, centred on
the published coordinate, with range rings at known radii.

**The reviewer points at the ramp; the page does the measuring.** Clicking the
image in the enlarged view computes the offset from the crosshair exactly, so no
one estimates a distance by eye — the rings are there for orientation, not
arithmetic. Verdicts are entered in the page, kept in ``localStorage`` so a
refresh costs nothing, and exported as a ``verdicts.json`` matching the template
this script also writes.

**Annotations are an SVG overlay, never burned into the image.** They sit exactly
on top of the pixels being judged, so the reviewer has to be able to take them
away to see what is underneath; baked-in marks cannot be removed without
re-rendering the whole sheet, which is not a workflow. The overlay toggles with a
checkbox or ``o``.

**The rubric is part of the instrument** (``RUBRIC``). It renders next to the
field it governs, opens in full with ``?``, and is copied verbatim into the
exported manifest — because ``0.9 m`` is uninterpretable without the rule saying
what it is 0.9 m *from*, and a convention that lives only in someone's head gets
applied two ways in one sitting. The load-bearing clauses, each written after a
real chip raised the question: click the **centre of the concrete apron**, never
the detectable-warning pad (PROWAG R305 puts the pad at the back of curb, ~0.6-0.9 m
down-slope, so pad-clicking would bias every record in one direction); count ramps
by **containment** — what you could reach without crossing a roadway — which is
per-corner rather than per-chip and, unlike "one ramp per crossing", survives a
median island; and click **every** chip including a dead-centre one, or the low
tail of the distribution becomes an artefact of reviewer confidence.

**A readable corner with no ramp is a verdict, not a gap.** ``no_ramp`` records a
**phantom** and completes the chip. Without it such a chip was uncompletable —
nothing to click, so the offset stayed null, so it was never "done" — and the
only exits were to leave it stuck or to mislabel it unjudgeable, which asserts
something different ("I cannot see" rather than "I can see, and it is not
there"). The phantom rate matters on its own: an inventory whose schema has no
removal mechanism gives a demolished ramp no way to leave the layer.

**Each chip also carries how many records the city itself publishes nearby**
(``count_neighbours``), which is the same per-corner quantity from the other
side; differencing the two is what settles whether a low records-per-corner ratio
is under-recording or ramp-design vocabulary (see §5d). **It stays hidden until
the reviewer has entered their own count**, because a published figure shown
first would anchor the judgment it is meant to be compared against.

**The basemap is the instrument, and the obvious basemap is not good enough.**
Esri World Imagery — the default anywhere ArcGIS is involved — renders Denver
leaf-on, hazy, and visibly upsampled to an effective ~1 m, so a ramp and its
detectable-warning pad are a smudge; and past its deepest level it serves "Map
data not yet available" as a blank grey tile that a naive fetcher will happily
paste in as evidence. Denver's own ``Aerial2016`` cache is leaf-off 3-inch
imagery at **0.057 m/px** — 4x the linear detail of its 2018 cache, and the
warning pads are individually visible. **Every city needs its municipal basemap
located before its sheet is worth a reviewer's time**, and the deepest available
level matters more than the capture year: a positional check does not care that
imagery is two years older, because ramps do not move.

    python scripts/analysis/inventory_review_sheet.py \
        --city denver-co --inventory data/inventories/denver-co-2026-07-31.jsonl.gz \
        --tile-source denver-2016 --sample 60 --seed 20260731 \
        --where-field UPDATE_STATUS --where-value NC

**Sampling is record-weighted by default**, because every record becomes a Stage 1
label and the question is how accurate the *labels* will be. ``--sampling
stratified`` spreads the sample over an equal-area grid instead, which buys
peripheral coverage at the cost of no longer estimating anything about the
population — use it to diagnose, not to quote.

Imagery fetching is the only part of this programme that needs network. Tile math
and sampling are pure and unit-tested in ``tests/test_inventory_review_sheet.py``.
"""
import argparse
import base64
import gzip
import io
import json
import math
import os
import random
import sys
import urllib.request
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))

# Named basemaps. ``max_zoom`` is the deepest level the service actually serves —
# past it these return a placeholder rather than an HTTP error, which is why
# ``looks_blank`` exists. ``note`` travels into the manifest so a verdict can
# never be read without knowing what it was made against.
TILE_SOURCES = {
    "esri-world": {
        "url": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery"
                "/MapServer/tile/{z}/{y}/{x}"),
        "max_zoom": 20,
        "attribution": ("Esri World Imagery (Esri, Maxar, Earthstar Geographics, "
                        "and the GIS User Community)"),
        "note": "Global fallback. Leaf-on and visibly upsampled over Denver — "
                "adequate to check gross placement, NOT to measure a 1-2 m offset.",
    },
    "denver-2018": {
        "url": ("https://tiles.arcgis.com/tiles/zdB7qR0BtYrg0Xpl/arcgis/rest/services"
                "/Aerial2018_tilecache/MapServer/tile/{z}/{y}/{x}"),
        "max_zoom": 19,
        "attribution": "City and County of Denver (geospatialDENVER), Aerial 2018",
        "note": "Leaf-off and sharp, but only 0.23 m/px — a 40 m chip is 174 px, "
                "which is thin for reading a 1 m offset. Prefer denver-2016.",
    },
    "denver-2016": {
        "url": ("https://tiles.arcgis.com/tiles/zdB7qR0BtYrg0Xpl/arcgis/rest/services"
                "/Aerial2016/MapServer/tile/{z}/{y}/{x}"),
        "max_zoom": 21,
        "attribution": "City and County of Denver (geospatialDENVER), Aerial 2016",
        "note": "Leaf-off 3-inch imagery — 0.057 m/px at Denver's latitude, 4x the "
                "linear detail of the 2018 cache, and detectable-warning pads are "
                "individually visible. Two years older, which does not matter for a "
                "positional check (ramps do not move) and is in fact closer to the "
                "2015 vintage 74% of Denver's records carry.",
    },
}

USER_AGENT = "RampNet-sourcing/1.0 (+https://github.com/ProjectSidewalk/RampNet)"

TILE_PX = 256
DEFAULT_SPAN_M = 40.0

# Ring radii in metres. 1 m is roughly "on the ramp", 2 m "on the right corner
# quadrant", 5 m "right corner, wrong ramp of the pair", 10 m "wrong corner".
# These are the read-off marks, so the reviewer never estimates a bare distance.
RING_RADII_M = (1.0, 2.0, 5.0, 10.0)

# The rubric. **One source of truth**: it is rendered into the sheet next to the
# field it governs *and* copied verbatim into the exported manifest, because a
# verdict is uninterpretable without the rule that produced it — "0.9 m" means
# nothing unless you know what it is 0.9 m from. Every clause here was written
# after a case that would otherwise have been called two different ways on two
# different days; the examples are the actual chips that raised the question.
RUBRIC = {
    "click_target": (
        "Click the CENTRE of the ramp's concrete apron. NOT the detectable-warning "
        "pad: PROWAG R305 puts the pad at the back of curb on perpendicular, blended "
        "and diagonal ramps, and on the street-level landing of a parallel ramp, so "
        "pad centres sit roughly 0.6-0.9 m down-slope of ramp centres. The pad is the "
        "most visible thing in the frame, so clicking pads is the easy mistake, and it "
        "would add that 0.6-0.9 m to EVERY record as a systematic bias that looks "
        "exactly like real positional error. Parallel ramp (a level landing flanked by "
        "two sloped runs, where 'the centre' has three defensible answers metres "
        "apart): click the centre of the LANDING and note 'parallel'. Legacy ramps "
        "whose entire surface is domed are the one case where pad centre and ramp "
        "centre coincide."
    ),
    "always_click": (
        "Click on EVERY chip, including when the crosshair already looks dead centre "
        "— click the crosshair itself for ~0. Two reasons. Mechanically, a chip with "
        "no click has a null offset and never counts as reviewed. Methodologically, if "
        "you only click when you think you see an error then near-zero cases are "
        "recorded by omission, and the low tail of the distribution becomes an "
        "artefact of reviewer confidence rather than a property of the data."
    ),
    "ramps_visible": (
        "Count only ramps you could reach from the crosshair WITHOUT CROSSING A "
        "ROADWAY. This is per-corner, not per-chip: a 40 m chip on an arterial holds "
        "three or four corners and counting all of them conflates 'ramps in frame' "
        "with 'ramps on this corner'. Perpendicular pair = 2. One diagonal apron "
        "serving two crossings = 1. Median island with a cut-through = 2, one end per "
        "side. Triangular channelising island ('pork chop') = 3, one per leg it "
        "serves. Note that 'one ramp per crossing' is NOT the rule — a median has two "
        "ends serving a single crossing; containment is the rule. THE RINGS DO NOT "
        "BOUND THE COUNT either: they exist to measure the offset, and a ramp sitting "
        "inside the 10 m ring but across a roadway belongs to a different corner. On "
        "chip 66519 four ramps fall inside the 10 m ring and the answer is three."
    ),
    "on_corner": (
        "The same containment test: YES if the crosshair and the ramp you clicked are "
        "on the same corner or island with no roadway between them. It is NOT 'is this "
        "the ramp the digitiser meant' — these inventories carry no corner key, so "
        "that is unknowable. Below ~2-3 m it is yes by construction, so the field only "
        "carries information in the tail, where it separates an imprecise point "
        "(benign for Stage 1: it still projects into roughly the right part of the "
        "panorama) from a misassigned one (wrong side of the street, plausibly a "
        "different panorama altogether). Mid-block ramps and refuge islands resolve "
        "under the same test; note the case."
    ),
    "no_ramp": (
        "The corner is readable and there is definitively no ramp at it. This is a "
        "PHANTOM record, and it is a result rather than a failure — an inventory whose "
        "schema has no removal mechanism gives a demolished ramp no way to leave the "
        "layer, so the phantom rate has no upper bound from the data alone. Kept "
        "distinct from unjudgeable on purpose: 'I can see, and it is not there' is a "
        "different claim from 'I cannot see'."
    ),
    "unjudgeable": (
        "Shadow, occlusion or resolution prevents a call. Mark it rather than "
        "guessing — the unreadable rate is itself a reported number."
    ),
    "resolution_floor": (
        "Offsets below roughly 0.3 m are at the floor of this instrument, not "
        "measurements of real error: see the metres-per-pixel in this manifest for the "
        "pixel size, and the registration check in analysis_out/georef_check/ for how "
        "well the imagery agrees with the city's own vector data. Report the left tail "
        "as floor-limited rather than claiming centimetres."
    ),
    "published_nearby": (
        "The count of published records near each chip is HIDDEN until you have "
        "entered ramps_visible for that chip, and this is deliberate. ramps_visible is "
        "meant to be independent evidence from the imagery; showing the published "
        "count first would anchor it, and the whole value of the comparison is that "
        "the two were arrived at separately. Once revealed, each nearby record is "
        "drawn on the image as a magenta diamond, and THE DIAMONDS ARE THE EVIDENCE "
        "— the counts are only a summary. **A radius is not a corner**, and it fails "
        "in both directions on exactly the complex geometry where the comparison "
        "would matter: 6 m misses the far ramp of a large corner (chip 66519's "
        "channelising island spans 7.0 m) and reaches straight across a 4-5 m slip "
        "lane (chip 67585, where the record 5.2 m ESE is on the far side of a "
        "crossing). Both produced confident false alarms before the panel stopped "
        "issuing verdicts. So: look at where the diamonds fall, decide which are on "
        "your corner, and note a genuine disagreement rather than trusting a number. "
        "A count above the published figure suggests the city under-records (the "
        "pair-merge failure mode); below it suggests phantoms or duplicates."
    ),
}


def find_neighbours(all_points, targets, radius_m, zoom=None):
    """Published records within ``radius_m`` of each target.

    Returns, per target, a list of ``{"d_m", "dx_px", "dy_px"}`` sorted by
    distance — the pixel offsets only when ``zoom`` is given, computed in the
    chip's own Web Mercator projection so a marker drawn at that offset lands
    exactly where the record is. **The target's own record is included** when it
    appears in ``all_points``, so a count taken from this is directly comparable
    to a reviewer's per-corner ramp count rather than off by one against it.

    Neighbours come from the WHOLE inventory, never the sample frame: a
    neighbouring ramp excluded from the frame (Denver's 2023-24 `A` records, say)
    is still a published ramp, and pretending otherwise would understate the city.

    Points are bucketed into a lon/lat grid sized to the radius, so this is O(n)
    rather than targets x records. Distances use an equirectangular
    approximation, exact enough at the tens-of-metres scale asked for here. Pure.
    """
    if not targets:
        return []
    lat_mid = sum(p[1] for p in targets) / len(targets)
    m_per_deg_lat = 111132.0
    cell_lat = radius_m / m_per_deg_lat
    cell_lon = radius_m / (111320.0 * math.cos(math.radians(lat_mid)) or 1e-9)

    grid = defaultdict(list)
    for lon, lat in all_points:
        grid[(int(lon / cell_lon), int(lat / cell_lat))].append((lon, lat))

    out = []
    for lon0, lat0 in targets:
        mlon = 111320.0 * math.cos(math.radians(lat0)) or 1e-9
        cx, cy = int(lon0 / cell_lon), int(lat0 / cell_lat)
        px0, py0 = lonlat_to_pixel(lon0, lat0, zoom) if zoom is not None else (0, 0)
        found = []
        for gx in (-1, 0, 1):
            for gy in (-1, 0, 1):
                for lon, lat in grid.get((cx + gx, cy + gy), ()):
                    d = math.hypot((lon - lon0) * mlon, (lat - lat0) * m_per_deg_lat)
                    if d > radius_m:
                        continue
                    rec = {"d_m": round(d, 2), "dx_px": None, "dy_px": None}
                    if zoom is not None:
                        px, py = lonlat_to_pixel(lon, lat, zoom)
                        rec["dx_px"] = round(px - px0, 1)
                        rec["dy_px"] = round(py - py0, 1)
                    found.append(rec)
        found.sort(key=lambda r: r["d_m"])
        out.append(found)
    return out


def count_neighbours(all_points, targets, radii_m):
    """How many published records fall within each radius of each target.

    A thin projection of :func:`find_neighbours`, so the counts and the markers
    drawn on the chip can never disagree about what is nearby.
    """
    if not radii_m:
        return [[] for _ in targets]
    found = find_neighbours(all_points, targets, max(radii_m))
    return [[sum(1 for n in fs if n["d_m"] <= r) for r in radii_m] for fs in found]


# Radii at which neighbouring published records are counted. 6 m is the threshold
# `inventory_geometry.py` calibrated against NYC's published corner key (P .976 /
# R .973). 10 m is carried alongside because 6 m demonstrably under-groups large
# corners: Denver chip 66519 is a channelising island whose three ramps sit at
# 0.0, 5.8 and 7.0 m, so single-link at 6 m splits it and scores one of the three
# as a singleton. Reporting both makes that visible instead of silent.
NEIGHBOUR_RADII_M = (6.0, 10.0)

# A served-but-empty tile ("Map data not yet available") is near-uniform. Real
# aerial imagery over a street scene never is. Both thresholds have to hold, so a
# genuinely flat subject — fresh snow, a blank roof — is not discarded on
# variance alone.
BLANK_STDDEV_MAX = 6.0
BLANK_MEAN_RANGE = (150, 235)


def lonlat_to_pixel(lon, lat, zoom, tile_px=TILE_PX):
    """Web Mercator (EPSG:3857) global pixel coordinates. Pure."""
    n = tile_px * (2 ** zoom)
    x = (lon + 180.0) / 360.0 * n
    s = math.sin(math.radians(lat))
    s = max(-0.9999, min(0.9999, s))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    return x, y


def metres_per_pixel(lat, zoom, tile_px=TILE_PX):
    """Ground resolution of a Web Mercator pixel at ``lat``. Pure."""
    return (2 * math.pi * 6378137.0 * math.cos(math.radians(lat))) / (tile_px * 2 ** zoom)


def tile_range(lon, lat, zoom, span_px, tile_px=TILE_PX):
    """Tiles covering a ``span_px`` box centred on lon/lat, plus the crop origin.

    Returns ``(x0, y0, x1, y1, origin_px_x, origin_px_y)``; the tile range is
    inclusive and the origin is the box's top-left in global pixel space.
    """
    px, py = lonlat_to_pixel(lon, lat, zoom, tile_px)
    half = span_px / 2.0
    left, top = px - half, py - half
    return (int(math.floor(left / tile_px)), int(math.floor(top / tile_px)),
            int(math.floor((px + half) / tile_px)), int(math.floor((py + half) / tile_px)),
            left, top)


def looks_blank(stats):
    """Is this a served placeholder rather than imagery?

    Takes ``(mean, stddev)`` so the test stays pure and the caller owns PIL.
    """
    mean, stddev = stats
    return stddev <= BLANK_STDDEV_MAX and BLANK_MEAN_RANGE[0] <= mean <= BLANK_MEAN_RANGE[1]


def uniform_sample(n_records, n, seed):
    """Record-weighted sample. Pure and deterministic given ``seed``.

    The default, because every record becomes a Stage 1 label: the quantity being
    estimated is the accuracy of the labels the pipeline would actually produce,
    which is a per-record average, not a per-square-kilometre one.
    """
    rng = random.Random(seed)
    idx = list(range(n_records))
    rng.shuffle(idx)
    return sorted(idx[:n])


def stratified_sample(points, n, seed, grid=8):
    """Pick ``n`` indices spread over an equal-area grid of the point set.

    Cells are filled round-robin from a shuffled per-cell queue, so every occupied
    cell contributes before any cell contributes twice. **Not** an estimator of
    the population — it deliberately over-weights sparse periphery — so it is a
    diagnostic option, never the default.
    """
    if not points or n <= 0:
        return []
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    lo_x, hi_x = min(lons), max(lons)
    lo_y, hi_y = min(lats), max(lats)
    span_x = (hi_x - lo_x) or 1e-9
    span_y = (hi_y - lo_y) or 1e-9
    cells = defaultdict(list)
    for i, (lon, lat) in enumerate(points):
        cx = min(grid - 1, int((lon - lo_x) / span_x * grid))
        cy = min(grid - 1, int((lat - lo_y) / span_y * grid))
        cells[(cx, cy)].append(i)
    rng = random.Random(seed)
    keys = sorted(cells)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(cells[k])
    picked, round_no = [], 0
    while len(picked) < n:
        added = False
        for k in keys:
            if len(cells[k]) > round_no:
                picked.append(cells[k][round_no])
                added = True
                if len(picked) == n:
                    break
        if not added:
            break
        round_no += 1
    return sorted(picked)


class TileMissing(Exception):
    """The basemap has no tile here.

    A city basemap is clipped to that city, so a record outside the municipal
    footprint — Denver publishes ~15% of its ramps within 1 km beyond the county
    line, plus a handful in the mountain parks — has no imagery. That is a
    property of the record, not a failure of the run, so it drops the chip and
    is counted.
    """


def _fetch_tile(url, cache_dir, timeout=60):
    from PIL import Image
    key = url.split("/tile/")[-1].replace("/", "_") + ".jpg"
    path = os.path.join(cache_dir, key)
    if os.path.exists(path):
        if os.path.getsize(path) == 0:
            raise TileMissing(url)
        return Image.open(path).convert("RGB")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            blob = fh.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 400):
            # Cache the absence too, so a re-run does not re-request it.
            open(path, "wb").close()
            raise TileMissing(url)
        raise
    with open(path, "wb") as out:
        out.write(blob)
    return Image.open(io.BytesIO(blob)).convert("RGB")


def render_chip(lon, lat, zoom, span_m, cache_dir, tile_url):
    """Aerial chip centred on lon/lat. **No annotation is drawn.**

    Returns ``(image, metres_per_pixel, tile_keys, blank)``. ``blank`` is True when
    the fetched imagery is a placeholder — the caller drops the chip rather than
    presenting grey pixels as evidence.

    Rings and crosshair used to be burned into the JPEG here. They are now an SVG
    overlay in the sheet instead, for a reason that only shows up in use: the
    annotation sits exactly on top of the pixels being judged, so a reviewer needs
    to take it away to see whether a ramp is under it. Baked-in marks cannot be
    removed, and re-rendering the whole sheet to look underneath is not a
    workflow. Keeping the image clean also means the overlay can be redrawn at any
    display size without resampling the imagery.
    """
    from PIL import Image, ImageStat
    mpp = metres_per_pixel(lat, zoom)
    span_px = int(round(span_m / mpp))
    x0, y0, x1, y1, ox, oy = tile_range(lon, lat, zoom, span_px)
    canvas = Image.new("RGB", ((x1 - x0 + 1) * TILE_PX, (y1 - y0 + 1) * TILE_PX))
    keys = []
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            url = tile_url.format(z=zoom, x=tx, y=ty)
            keys.append("{}/{}/{}".format(zoom, ty, tx))
            canvas.paste(_fetch_tile(url, cache_dir), ((tx - x0) * TILE_PX, (ty - y0) * TILE_PX))
    crop_x = int(round(ox - x0 * TILE_PX))
    crop_y = int(round(oy - y0 * TILE_PX))
    chip = canvas.crop((crop_x, crop_y, crop_x + span_px, crop_y + span_px))

    st = ImageStat.Stat(chip.convert("L"))
    blank = looks_blank((st.mean[0], st.stddev[0]))
    return chip, mpp, keys, blank


def to_data_uri(img, quality=85):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


SHEET_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>__CITY__ — curb-ramp location precision review</title>
<style>
 :root {{ --bg:#111; --panel:#1c1c1c; --line:#2e2e2e; --dim:#9a9a9a; }}
 * {{ box-sizing: border-box; }}
 body {{ font:14px/1.55 system-ui,sans-serif; margin:0; background:var(--bg); color:#eee; }}
 header {{ position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line);
           padding:14px 24px; display:flex; gap:20px; align-items:center; flex-wrap:wrap; }}
 h1 {{ font-size:17px; margin:0; font-weight:600; }}
 .sub {{ color:var(--dim); font-size:12px; width:100%; margin-top:2px; }}
 .ctl {{ display:flex; gap:6px; align-items:center; font-size:13px; color:#ddd; }}
 button {{ font:inherit; background:#2a2a2a; color:#eee; border:1px solid var(--line);
           border-radius:5px; padding:5px 11px; cursor:pointer; }}
 button:hover {{ background:#343434; }}
 button.primary {{ background:#2d5a3d; border-color:#3c7a52; }}
 .prog {{ font-variant-numeric:tabular-nums; color:var(--dim); }}
 main {{ padding:20px 24px 60px; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:14px; }}
 figure {{ margin:0; background:var(--panel); border-radius:7px; padding:7px;
           border:1px solid transparent; }}
 figure.done {{ border-color:#3c7a52; }}
 figure.skip {{ border-color:#7a5a3c; opacity:.65; }}
 figure.phantom {{ border-color:#8a3c3c; }}
 .wrap {{ position:relative; cursor:zoom-in; line-height:0; }}
 .wrap img {{ width:100%; border-radius:4px; display:block; }}
 .wrap svg {{ position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }}
 body.nooverlay .wrap svg {{ display:none; }}
 figcaption {{ font-size:11px; color:#999; margin-top:5px; font-family:ui-monospace,monospace;
               display:flex; justify-content:space-between; }}
 dialog {{ border:none; background:var(--panel); color:#eee; border-radius:10px; padding:0;
           max-width:96vw; max-height:96vh; }}
 dialog::backdrop {{ background:rgba(0,0,0,.82); }}
 .modal {{ display:flex; gap:18px; padding:18px; align-items:flex-start; }}
 .stage {{ position:relative; line-height:0; cursor:crosshair; }}
 .stage img {{ display:block; border-radius:5px;
               width:min(74vh,calc(96vw - 340px)); height:auto; image-rendering:auto; }}
 .stage svg {{ position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }}
 .side {{ width:280px; font-size:13px; }}
 .side h2 {{ font-size:14px; margin:0 0 10px; }}
 .row {{ margin-bottom:13px; }}
 .row label {{ display:block; color:var(--dim); font-size:12px; margin-bottom:4px; }}
 .seg {{ display:flex; gap:4px; flex-wrap:wrap; }}
 .seg button {{ flex:1; min-width:38px; padding:5px 4px; }}
 .seg button[aria-pressed="true"] {{ background:#2d5a3d; border-color:#3c7a52; }}
 .measure {{ font-family:ui-monospace,monospace; font-size:19px; }}
 .measure em {{ color:var(--dim); font-size:12px; font-style:normal; }}
 input[type=text] {{ width:100%; background:#111; color:#eee; border:1px solid var(--line);
                     border-radius:5px; padding:6px; font:inherit; }}
 .nav {{ display:flex; gap:8px; margin-top:16px; }}
 .nav button {{ flex:1; }}
 kbd {{ background:#000; border:1px solid var(--line); border-radius:3px; padding:0 4px;
        font-size:11px; font-family:ui-monospace,monospace; }}
 .help {{ color:var(--dim); font-size:11.5px; margin-top:14px; line-height:1.5; }}
 /* The rule sits under the control it governs. A rubric that lives anywhere else
    is a rubric nobody reads at the moment of judgment. */
 .hint {{ color:#b9b9b9; font-size:11px; line-height:1.45; margin:5px 0 0;
          border-left:2px solid #3a3a3a; padding-left:7px; }}
 .hint b {{ color:#ffd479; font-weight:600; }}
 .pub {{ font-family:ui-monospace,monospace; font-size:12.5px; }}
 .pub.agree {{ color:#7bc98a; }}
 .pub.under {{ color:#ff9d6b; }}
 .pub.over {{ color:#e07a7a; }}
 #rubric-dlg {{ max-width:760px; }}
 #rubric-dlg .modal {{ display:block; padding:24px 26px; }}
 #rubric-dlg h3 {{ font-size:13px; margin:16px 0 4px; color:#ffd479; }}
 #rubric-dlg h3:first-of-type {{ margin-top:6px; }}
 #rubric-dlg p {{ margin:0; color:#ddd; font-size:12.5px; line-height:1.6; }}
</style>

<header>
 <h1>__CITY__ — location precision</h1>
 <label class="ctl"><input type="checkbox" id="ovl" checked> overlay <kbd>o</kbd></label>
 <span class="ctl prog" id="prog"></span>
 <button id="next-todo">next unreviewed <kbd>n</kbd></button>
 <button id="show-rubric">rubric <kbd>?</kbd></button>
 <button class="primary" id="export">export verdicts.json</button>
 <div class="sub">
  __N__ chips · <code>__INV__</code> · __SAMPLING__ sample, seed __SEED__ ·
  imagery <b>__SOURCE__</b> z__ZOOM__ (__MPP__ m/px) · __ATTRIB__.
  <span style="color:#ffb74d">__NOTE__</span>
  Progress is saved in this browser; <b>export before you finish</b> to write it to disk.
 </div>
</header>

<main><div class="grid" id="grid"></div></main>

<dialog id="dlg"><div class="modal">
 <div class="stage" id="stage"><img id="big" alt=""><svg id="bigsvg"></svg></div>
 <div class="side">
  <h2 id="title"></h2>
  <div class="row">
   <label>offset — click the nearest ramp on the image</label>
   <div class="measure" id="offset">—</div>
   <em id="offhint">click to measure · click the crosshair for 0</em>
   <p class="hint"><b>Centre of the concrete apron, not the warning pad</b> — pads sit
    0.6–0.9 m down-slope, and clicking them biases every record. Parallel ramp: centre of
    the level landing, note <code>parallel</code>. <b>Click every chip</b>, including a
    dead-centre one.</p>
  </div>
  <div class="row">
   <label>ramps visible on this corner <b>(the per-corner evidence)</b></label>
   <div class="seg" id="vis"></div>
   <p class="hint"><b>Only what you could reach without crossing a roadway</b> — this is
    per-corner, not per-chip. Perpendicular pair 2 · diagonal apron 1 · median island 2 ·
    pork-chop island 3.<br><b>The rings do not bound the count.</b> They measure distance;
    a ramp inside the 10 m ring but across a roadway is a different corner.</p>
  </div>
  <div class="row" id="pubrow" hidden>
   <label>published records nearby <b>(revealed after you count)</b></label>
   <div class="pub" id="pub">—</div>
  </div>
  <div class="row">
   <label>crosshair on the correct corner?</label>
   <div class="seg" id="corner"></div>
   <p class="hint">Same containment test: same corner or island, <b>no roadway between</b>
    the crosshair and the ramp you clicked. Yes by construction below ~2–3 m.</p>
  </div>
  <div class="row">
   <label>corner is readable and there is no ramp</label>
   <div class="seg" id="noramp"></div>
   <p class="hint"><b>A phantom record — this is a result, not a failure.</b> Distinct from
    unjudgeable: “I can see, and it is not there”.</p>
  </div>
  <div class="row">
   <label>unjudgeable — shadow, occlusion, resolution</label>
   <div class="seg" id="unread"></div>
  </div>
  <div class="row">
   <label>note</label>
   <input type="text" id="note" placeholder="optional">
  </div>
  <div class="nav">
   <button id="prev">← prev</button>
   <button id="nxt">next →</button>
   <button id="close">close <kbd>esc</kbd></button>
  </div>
  <div class="help">
   <b>Mark unjudgeable rather than guessing</b> — the unreadable rate is itself a reported number.
   <kbd>0</kbd>–<kbd>3</kbd> sets ramps visible · <kbd>u</kbd> unjudgeable ·
   <kbd>p</kbd> no ramp · <kbd>←</kbd> <kbd>→</kbd> move · <kbd>o</kbd> overlay ·
   <kbd>?</kbd> full rubric.
  </div>
 </div>
</div></dialog>

<dialog id="rubric-dlg"><div class="modal">
 <h2 style="margin:0 0 4px">Review rubric — __CITY__</h2>
 <p style="color:#9a9a9a;font-size:12px;margin:0 0 6px">
  Exported verbatim into <code>verdicts.json</code>, because a verdict cannot be read later
  without the rule that produced it.</p>
 <div id="rubric-body"></div>
 <div class="nav"><button id="rubric-close">close <kbd>esc</kbd></button></div>
</div></dialog>

<script>
const META = __META__;
const CHIPS = __CHIPS__;
const KEY = "rampnet-verdicts-" + META.city + "-" + META.seed;
const V = JSON.parse(localStorage.getItem(KEY) || "{{}}");
const S = META.span_px, C = S / 2;

// Returns the overlay's *inner* markup only. The <svg> wrapper is created once
// and kept: writing outerHTML would detach the element, so the cached reference
// would go stale and the modal overlay would render exactly once.
function overlayInner(withScale) {{
  let p = "";
  META.rings.forEach((m, i) => {{
    const r = m / META.mpp;
    if (r >= C) return;
    p += `<circle cx="${{C}}" cy="${{C}}" r="${{r}}" fill="none" stroke="#ffeb3b"
           stroke-opacity=".85" stroke-width="${{S / 500}}"/>`;
    const s = i % 2 ? 1 : -1, d = r * 0.7071;
    p += `<text x="${{C + s * d}}" y="${{C - s * d}}" fill="#ffeb3b" font-size="${{S / 42}}"
           font-family="system-ui" text-anchor="${{s > 0 ? 'start' : 'end'}}"
           dy="${{s > 0 ? -2 : 10}}">${{m}}m</text>`;
  }});
  const g = S / 46, a = S / 18, w = S / 220;
  p += `<g stroke="#ff4040" stroke-width="${{w}}">
     <line x1="${{C - a}}" y1="${{C}}" x2="${{C - g}}" y2="${{C}}"/>
     <line x1="${{C + g}}" y1="${{C}}" x2="${{C + a}}" y2="${{C}}"/>
     <line x1="${{C}}" y1="${{C - a}}" x2="${{C}}" y2="${{C - g}}"/>
     <line x1="${{C}}" y1="${{C + g}}" x2="${{C}}" y2="${{C + a}}"/></g>`;
  if (withScale) {{
    const bar = 10 / META.mpp;
    p += `<g><rect x="${{S * .03}}" y="${{S - S * .05}}" width="${{bar}}" height="${{S / 130}}"
      fill="#fff" fill-opacity=".9"/><text x="${{S * .03}}" y="${{S - S * .065}}" fill="#fff"
      font-size="${{S / 40}}" font-family="system-ui">10 m</text></g>`;
  }}
  return p;
}}

function marker(v) {{
  if (!v || v.px == null) return "";
  return `<g><circle cx="${{v.px}}" cy="${{v.py}}" r="${{S / 60}}" fill="none" stroke="#4fc3f7"
    stroke-width="${{S / 200}}"/><line x1="${{C}}" y1="${{C}}" x2="${{v.px}}" y2="${{v.py}}"
    stroke="#4fc3f7" stroke-width="${{S / 300}}" stroke-dasharray="${{S / 90}}"/></g>`;
}}

// Every OTHER published record in the frame, drawn where it actually is. This is
// what turns "3 or 4?" from a judgment into a look: three diamonds on the island
// and one across the crossing is visible in a second and arguable in none.
// Gated on the reviewer having counted first, for the same anti-anchoring reason
// the numbers are.
function pubMarkers(c, v) {{
  if (!c.pub || v.ramps_visible == null) return "";
  const r = S / 90;
  return c.pub.map(p => {{
    const x = C + p[0], y = C + p[1];
    return `<path d="M ${{x}} ${{y - r}} L ${{x + r}} ${{y}} L ${{x}} ${{y + r}} L ${{x - r}} ${{y}} Z"
      fill="none" stroke="#ff6fd8" stroke-width="${{S / 330}}"/>`;
  }}).join("");
}}

function state(id) {{ return V[id] || (V[id] = {{}}); }}
function save() {{ localStorage.setItem(KEY, JSON.stringify(V)); paint(); }}
// A readable corner with no ramp is a finished verdict, not an unfinished one.
// Before `no_ramp` existed such a chip could never be completed: there was
// nothing to click, so the offset stayed null and `next unreviewed` walked
// straight back to it.
function done(v) {{ return v && (v.unreadable || v.no_ramp || v.offset_m != null); }}

function paint() {{
  let n = 0;
  CHIPS.forEach(c => {{
    const v = V[c.id], f = document.getElementById("f" + c.id);
    if (!f) return;
    f.className = v && v.unreadable ? "skip" : (v && v.no_ramp ? "phantom"
                  : (done(v) ? "done" : ""));
    if (done(v)) n++;
    const tag = f.querySelector(".tag");
    tag.textContent = !v ? "" : v.unreadable ? "unjudgeable" : v.no_ramp ? "no ramp"
      : (v.offset_m != null ? v.offset_m.toFixed(1) + " m"
         + (v.ramps_visible != null ? " · " + v.ramps_visible + "\\u00d7" : "") : "");
  }});
  document.getElementById("prog").textContent = n + " / " + CHIPS.length + " reviewed";
}}

const grid = document.getElementById("grid");
grid.innerHTML = CHIPS.map(c => `<figure id="f${{c.id}}"><div class="wrap" data-id="${{c.id}}">
  <img src="${{c.uri}}" alt="${{c.id}}" loading="lazy">
  <svg viewBox="0 0 ${{S}} ${{S}}" xmlns="http://www.w3.org/2000/svg">${{overlayInner(false)}}</svg></div>
  <figcaption><span>${{c.id}}</span><span class="tag"></span></figcaption></figure>`).join("");

const dlg = document.getElementById("dlg"), big = document.getElementById("big"),
      bigsvg = document.getElementById("bigsvg"), stage = document.getElementById("stage");
let cur = 0;

function seg(el, opts, get, set) {{
  el.innerHTML = opts.map(o =>
    `<button data-v="${{o.v}}" aria-pressed="${{String(get() === o.v)}}">${{o.t}}</button>`).join("");
  el.querySelectorAll("button").forEach(b => b.onclick = () => {{
    const raw = b.dataset.v;
    const val = raw === "null" ? null : (raw === "true" ? true : raw === "false" ? false : +raw);
    set(get() === val ? null : val);
    save(); render();
  }});
}}

function render() {{
  const c = CHIPS[cur], v = state(c.id);
  big.src = c.uri; big.alt = c.id;
  bigsvg.setAttribute("viewBox", `0 0 ${{S}} ${{S}}`);
  bigsvg.innerHTML = overlayInner(true) + pubMarkers(c, v) + marker(v);
  document.getElementById("title").textContent =
    `${{c.id}}  (${{cur + 1}}/${{CHIPS.length}})`;
  document.getElementById("offset").textContent =
    v.offset_m == null ? "—" : v.offset_m.toFixed(2) + " m";
  seg(document.getElementById("vis"),
      [0, 1, 2, 3, 4].map(n => ({{v: n, t: n === 4 ? "4+" : String(n)}})),
      () => v.ramps_visible, x => v.ramps_visible = x);
  seg(document.getElementById("corner"),
      [{{v: true, t: "yes"}}, {{v: false, t: "no"}}],
      () => v.on_corner, x => v.on_corner = x);
  // The three terminal states are mutually exclusive: a chip cannot be both
  // "no ramp here" and "cannot tell", and neither can carry an offset.
  seg(document.getElementById("noramp"), [{{v: true, t: "no ramp here"}}],
      () => v.no_ramp || null, x => {{
        v.no_ramp = !!x;
        if (x) {{ v.unreadable = false; v.offset_m = null; v.px = v.py = null;
                 v.ramps_visible = 0; }}
      }});
  seg(document.getElementById("unread"), [{{v: true, t: "unjudgeable"}}],
      () => v.unreadable || null, x => {{
        v.unreadable = !!x;
        if (x) v.no_ramp = false;
      }});
  document.getElementById("note").value = v.note || "";

  // Held back until the count is entered, so the published figure cannot anchor
  // it. The comparison is only worth anything if the two were reached
  // independently.
  const row = document.getElementById("pubrow"), pub = document.getElementById("pub");
  if (v.ramps_visible == null || c.published == null) {{
    row.hidden = true;
  }} else {{
    row.hidden = false;
    // NO automatic verdict. A radius is not a corner, and it fails in BOTH
    // directions on exactly the complex geometry where the comparison would
    // matter: 6 m misses the far ramp of a large corner (chip 66519's island
    // spans 7.0 m) and reaches straight across a 4-5 m slip lane (chip 67585,
    // where the record 5.2 m ESE is on the far side of a crossing). Two false
    // alarms in two chips, in opposite directions. So the panel now shows the
    // records rather than judging them — the diamonds on the image are the
    // evidence, and the reviewer can see which ones are across a roadway.
    const p6 = c.published[0], p10 = c.published[1];
    pub.className = "pub";
    pub.innerHTML = `${{p6}} within 6 m · ${{p10}} within 10 m<br>
      <span style="font-size:11px">◆ marks each one. <b>A radius is not a corner</b> —
      6 m can cross a slip lane and can miss the far ramp of a big corner, so check
      each diamond against what you counted.</span>`;
  }}
}}

function open_(i) {{ cur = (i + CHIPS.length) % CHIPS.length; render();
  if (!dlg.open) dlg.showModal(); }}

grid.querySelectorAll(".wrap").forEach(w => w.onclick = () =>
  open_(CHIPS.findIndex(c => c.id === w.dataset.id)));

// Click-to-measure. The reviewer's job is to point at the ramp, not to estimate a
// distance: the geometry is exact and the rings are only there for orientation.
document.getElementById("stage").onclick = e => {{
  const r = big.getBoundingClientRect();
  const px = (e.clientX - r.left) / r.width * S, py = (e.clientY - r.top) / r.height * S;
  const v = state(CHIPS[cur].id);
  v.px = px; v.py = py;
  v.offset_m = Math.hypot(px - C, py - C) * META.mpp;
  v.unreadable = false; v.no_ramp = false;
  save(); render();
}};

document.getElementById("note").oninput = e => {{
  state(CHIPS[cur].id).note = e.target.value; save();
}};
document.getElementById("prev").onclick = () => open_(cur - 1);
document.getElementById("nxt").onclick = () => open_(cur + 1);
document.getElementById("close").onclick = () => dlg.close();
document.getElementById("ovl").onchange = e =>
  document.body.classList.toggle("nooverlay", !e.target.checked);

function nextTodo() {{
  const i = CHIPS.findIndex(c => !done(V[c.id]));
  if (i < 0) return alert("Every chip has been reviewed. Export when ready.");
  open_(i);
}}
document.getElementById("next-todo").onclick = nextTodo;

// The rubric is built from the same constant the manifest carries, so the rule
// shown to the reviewer and the rule exported beside their verdicts cannot drift.
const rubricDlg = document.getElementById("rubric-dlg");
document.getElementById("rubric-body").innerHTML =
  Object.entries(META.manifest.rubric || {{}}).map(([k, text]) =>
    `<h3>${{k.replace(/_/g, " ")}}</h3><p>${{text}}</p>`).join("");
document.getElementById("show-rubric").onclick = () => rubricDlg.showModal();
document.getElementById("rubric-close").onclick = () => rubricDlg.close();

addEventListener("keydown", e => {{
  if (e.target.tagName === "INPUT") return;
  if (e.key === "?") {{ rubricDlg.open ? rubricDlg.close() : rubricDlg.showModal(); return; }}
  if (rubricDlg.open) return;
  if (e.key === "o") {{ const b = document.getElementById("ovl");
    b.checked = !b.checked; b.onchange({{target: b}}); return; }}
  if (e.key === "n" && !dlg.open) return nextTodo();
  if (!dlg.open) return;
  const v = state(CHIPS[cur].id);
  if (e.key === "ArrowLeft") open_(cur - 1);
  else if (e.key === "ArrowRight") open_(cur + 1);
  else if ("01234".includes(e.key)) {{ v.ramps_visible = +e.key; save(); render(); }}
  else if (e.key === "u") {{ v.unreadable = !v.unreadable;
    if (v.unreadable) v.no_ramp = false; save(); render(); }}
  else if (e.key === "p") {{ v.no_ramp = !v.no_ramp;
    if (v.no_ramp) {{ v.unreadable = false; v.offset_m = null; v.px = v.py = null;
                     v.ramps_visible = 0; }}
    save(); render(); }}
}});

document.getElementById("export").onclick = () => {{
  const out = Object.assign({{}}, META.manifest, {{
    reviewer: META.manifest.reviewer, records: CHIPS.map(c => {{
      const v = V[c.id] || {{}};
      return {{
        id: c.id, lon: c.lon, lat: c.lat, tiles: c.tiles,
        offset_m: v.offset_m == null ? null : +v.offset_m.toFixed(2),
        on_corner: v.on_corner == null ? null : v.on_corner,
        ramps_visible: v.ramps_visible == null ? null : v.ramps_visible,
        unreadable: !!v.unreadable, no_ramp: !!v.no_ramp, note: v.note || "",
        published_within_6m: c.published ? c.published[0] : null,
        published_within_10m: c.published ? c.published[1] : null,
        click_px: v.px == null ? null : [+v.px.toFixed(1), +v.py.toFixed(1)]
      }};
    }})
  }});
  const blob = new Blob([JSON.stringify(out, null, 2) + "\\n"], {{type: "application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "verdicts.json"; a.click();
}};

paint();
</script>
"""


def build_sheet(meta, chips, manifest):
    """Assemble the interactive sheet. Pure — takes rendered chips, returns HTML."""
    subs = {
        "__CITY__": meta["city"],
        "__N__": str(len(chips)),
        "__INV__": meta["inventory"],
        "__SAMPLING__": meta["sampling"],
        "__SEED__": str(meta["seed"]),
        "__SOURCE__": meta["tile_source"],
        "__ZOOM__": str(meta["zoom"]),
        "__MPP__": "{:.3f}".format(meta["mpp"]),
        "__ATTRIB__": meta["attribution"],
        "__NOTE__": meta["note"],
        "__META__": json.dumps({
            "city": meta["city"], "seed": meta["seed"], "mpp": meta["mpp"],
            "span_px": meta["span_px"], "rings": list(RING_RADII_M),
            "manifest": manifest,
        }),
        "__CHIPS__": json.dumps(chips),
    }
    out = SHEET_TEMPLATE.replace("{{", "\x00").replace("}}", "\x01")
    out = out.replace("{", "{").replace("}", "}")
    out = out.replace("\x00", "{").replace("\x01", "}")
    for k, v in subs.items():
        out = out.replace(k, v)
    return out


def load_inventory(path):
    opener = gzip.open if path.endswith(".gz") else open
    rows = []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--city", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--seed", type=int, required=True,
                    help="explicit, so the sample is reproducible and citable")
    ap.add_argument("--sampling", choices=("uniform", "stratified"), default="uniform")
    ap.add_argument("--tile-source", choices=sorted(TILE_SOURCES), default="esri-world")
    ap.add_argument("--zoom", type=int, default=None, help="defaults to the source's max")
    ap.add_argument("--span-m", type=float, default=DEFAULT_SPAN_M)
    ap.add_argument("--grid", type=int, default=8, help="stratification grid per axis")
    ap.add_argument("--id-field", default="OBJECTID")
    ap.add_argument("--where-field", default=None,
                    help="restrict the sample frame, e.g. UPDATE_STATUS")
    ap.add_argument("--where-value", default=None)
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args(argv)

    src = TILE_SOURCES[args.tile_source]
    zoom = args.zoom if args.zoom is not None else src["max_zoom"]
    if zoom > src["max_zoom"]:
        ap.error("{} serves at most z{}; deeper levels return blank placeholders".format(
            args.tile_source, src["max_zoom"]))

    rows = load_inventory(args.inventory)
    frame = list(range(len(rows)))
    if args.where_field:
        frame = [i for i in frame if str(rows[i].get(args.where_field)) == args.where_value]
        print("sample frame: {} of {} records with {}={}".format(
            len(frame), len(rows), args.where_field, args.where_value))
    pts = [(rows[i]["lon"], rows[i]["lat"]) for i in frame]
    if args.sampling == "uniform":
        local = uniform_sample(len(frame), args.sample, args.seed)
    else:
        local = stratified_sample(pts, args.sample, args.seed, grid=args.grid)
    picked = [frame[i] for i in local]
    print("sampled {} records ({}, seed {})".format(len(picked), args.sampling, args.seed))

    review_dir = os.path.join(args.out_dir, "review_{}".format(args.city))
    cache_dir = os.path.join(review_dir, "tiles_{}".format(args.tile_source))
    os.makedirs(cache_dir, exist_ok=True)

    chips, verdicts, blanks, missing, mpp, span_px = [], [], 0, 0, None, 0
    for k, i in enumerate(picked):
        lon, lat = rows[i]["lon"], rows[i]["lat"]
        rid = str(rows[i].get(args.id_field, i))
        try:
            chip, mpp, keys, blank = render_chip(lon, lat, zoom, args.span_m, cache_dir, src["url"])
        except TileMissing:
            missing += 1
            print("  [{:>3}/{}] {} NO IMAGERY — outside the basemap footprint".format(
                k + 1, len(picked), rid))
            continue
        if blank:
            blanks += 1
            print("  [{:>3}/{}] {} BLANK — dropped".format(k + 1, len(picked), rid))
            continue
        span_px = chip.size[0]
        chips.append({"uri": to_data_uri(chip), "id": rid, "lon": lon, "lat": lat,
                      "tiles": keys})
        verdicts.append({
            "id": rid, "lon": lon, "lat": lat, "tiles": keys,
            "offset_m": None, "on_corner": None, "ramps_visible": None,
            "unreadable": False, "no_ramp": False, "note": "",
        })
        print("  [{:>3}/{}] {} {:.6f},{:.6f}".format(k + 1, len(picked), rid, lat, lon))

    # How many records the city itself publishes near each chip — the same
    # per-corner quantity the reviewer reads off the imagery, taken from the
    # other side. Differencing the two is what settles whether a low
    # records-per-corner ratio is under-recording or ramp-design vocabulary
    # (docs/curb_ramp_data_sourcing.md §5d). Counted against the WHOLE inventory,
    # not the sample frame.
    all_pts = [(r["lon"], r["lat"]) for r in rows if r.get("lon") is not None]
    targets = [(c["lon"], c["lat"]) for c in chips]
    # Half the chip span, so every neighbour that could be drawn is found. Pixel
    # offsets come back in the chip's own projection, so a marker lands exactly
    # on the record.
    found = find_neighbours(all_pts, targets, args.span_m / 2.0, zoom=zoom)
    for chip, verdict, near in zip(chips, verdicts, found):
        counts = [sum(1 for n in near if n["d_m"] <= r) for r in NEIGHBOUR_RADII_M]
        chip["published"] = counts
        # Only the OTHER records get a marker; the sampled one is the crosshair.
        chip["pub"] = [[n["dx_px"], n["dy_px"]] for n in near if n["d_m"] > 0.05]
        verdict["published_within_6m"] = counts[0]
        verdict["published_within_10m"] = counts[1]
        verdict["published_neighbours_m"] = [n["d_m"] for n in near if n["d_m"] > 0.05]

    manifest = {
        "city": args.city, "inventory": os.path.basename(args.inventory),
        "seed": args.seed, "sampling": args.sampling, "sample_requested": args.sample,
        "sample_frame": {"field": args.where_field, "value": args.where_value,
                         "size": len(frame), "of": len(rows)},
        "grid": args.grid if args.sampling == "stratified" else None,
        "tile_source": args.tile_source, "tile_url": src["url"],
        "imagery": src["attribution"], "imagery_note": src["note"],
        "zoom": zoom, "metres_per_pixel": mpp, "span_m": args.span_m,
        "span_px": span_px, "ring_radii_m": list(RING_RADII_M),
        "neighbour_radii_m": list(NEIGHBOUR_RADII_M),
        "blank_chips_dropped": blanks,
        "no_imagery_dropped": missing,
        # Travels with the verdicts on purpose: an offset is uninterpretable
        # without the rule that says what it is an offset *from*.
        "rubric": RUBRIC,
        "reviewer": None, "reviewed_on": None, "confidence": None,
    }
    verdict_path = os.path.join(review_dir, "verdicts.json")
    with open(verdict_path, "w") as fh:
        json.dump(dict(manifest, records=verdicts), fh, indent=2)
        fh.write("\n")

    sheet_meta = {
        "city": args.city, "inventory": os.path.basename(args.inventory),
        "sampling": args.sampling, "seed": args.seed, "tile_source": args.tile_source,
        "zoom": zoom, "mpp": mpp or 0.0, "span_px": span_px,
        "attribution": src["attribution"], "note": src["note"],
    }
    sheet_path = os.path.join(review_dir, "review_sheet.html")
    with open(sheet_path, "w", encoding="utf-8") as fh:
        fh.write(build_sheet(sheet_meta, chips, manifest))

    print("\n{} chips, {} blank dropped, {} no imagery".format(
        len(chips), blanks, missing))
    print("wrote {}".format(sheet_path))
    print("wrote {}".format(verdict_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
