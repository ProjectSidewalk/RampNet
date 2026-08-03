"""Build the STREET-LEVEL review sheet: judge records against the imagery
Stage 1 actually consumes (issue #103; #96 §5n).

The aerial sheet (``inventory_review_sheet.py``) measures a metric offset
against a municipal basemap — a *proxy*, which then needs §5g's Monte Carlo to
become a decision, and whose basemap hunt has failed four distinct ways (§5e,
§5h). This sheet asks the real question directly: **does the ramp fall inside
the ±18.37° strip Stage 1 would cut for this record?** It renders, per sampled
record, the exact production view — the GSV panorama fetched and projected by
``rampnet.gsv`` (the code ``download_dataset.py`` itself runs), aimed at the
government point's bearing — and the reviewer clicks the ramp.

**The verdict is an ANGULAR offset, deliberately.** A click at pixel column
``c`` is ``atan((c - 512)/512)`` degrees from the projected government bearing,
signed with §5j's residual convention (**positive = clockwise = right of the
crosshair**), so human review of *candidate* cities lands in the same units as
``stage1_bearing_residual.py``'s automatic null over *corpus* cities
(|mean| <= 0.25° at n=90k) and the two cross-validate. The price is stated in
#103: this yields **no metric number** — nothing comparable to Denver's 0.29 m
— so the aerial sheet is not retired.

**The full 90° view is shown, not the bare 341-px strip.** The reviewer must be
able to distinguish *"ramp just outside the crop"* from *"no ramp here"* — that
distinction is the measurement. The strip edges are drawn where they truly are:
the crop ``persp[:, 341:682]`` is **asymmetric** (−18.458° / +18.368° about the
bearing), and both edges come from the same single definition as
``crop_half_angle_deg()`` rather than from a re-derived constant.

**One panorama per record, picked by a recorded rule** (nearest within
4–30 m whose capture date is on or after the record's date, tie-break newest),
and every unjudgeable verdict must carry a **reason tag** — parked van, pole,
sun — because street level replaces the aerial sheet's canopy selection bias
with a different one that #103 says must be *measured*, not assumed. A
targeted second-vantage pass over just the unjudgeable subset is the cheap
follow-on this enables.

**Neighbour records are always drawn** (dashed magenta bearings), unlike the
aerial sheet's anti-anchoring gate. There is no counting task here for them to
anchor, and they exist to resolve the question that cost Seattle five chips of
notes: *which ramp is this record's?* A diamond on the other visible ramp means
another record claims it.

Sampling reuses the aerial scaffold: ``--sites-from-verdicts`` renders exactly
the records of a built aerial sheet (the Denver pilot pairs per-record with
§5f's trusted answer), and fresh sampling imports ``uniform_sample`` /
``sample_year_strata`` unchanged, so §5l's date-strata discipline carries over.

Caching learns §5h's lessons structurally: assembled panoramas are cached by
pano id; **absences are JSON marker files carrying a reason and an attempt
count, never zero-byte sentinels**, and ``--refetch-absent`` re-tries them — so
the "retry fix inert against an existing cache" trap cannot recur. Every input
record leaves with a terminal status in the manifest (rendered / no pano in
band / no dated pano / fetch failed / …): a drop count is a claim about the
fetcher until it is checked against the sample, and Charlotte's
``no_imagery_dropped: 59`` was believed once already.

    python scripts/analysis/street_review_sheet.py \
        --city denver-co --inventory data/inventories/denver-co-2026-07-31.jsonl.gz \
        --sites-from-verdicts analysis_out/review_denver-co/verdicts.json \
        --date-field CREATEDATE

Network is needed only for the GSV endpoints (undocumented, unauthenticated,
can break without notice — the risk #103 accepts). The geometry, the pano-pick
rule, sampling, and the sheet assembly are pure and unit-tested.
"""
import argparse
import json
import hashlib
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# One definition each, imported never re-derived (#103's standing instruction):
# the crop columns and half-angle from the tolerance analysis, the sign/wrap
# conventions from the §5j residual, the samplers and inventory loader from the
# aerial sheet, and the sentinel-aware date parser from the temporal gate.
from inventory_review_sheet import (  # noqa: E402
    load_inventory, sample_year_strata, stratified_sample, to_data_uri,
    uniform_sample, YEAR_STRATA)
from stage1_bearing_residual import fwd_azimuth_deg, wrap_deg  # noqa: E402
from stage1_offset_tolerance import CROP_HI, CROP_LO, crop_half_angle_deg  # noqa: E402
from temporal_gap import SENTINEL_YMS, parse_ym  # noqa: E402

from rampnet.gsv import (  # noqa: E402
    equirectangular_to_perspective, heading_to_azimuth,
    perspective_col_to_azimuth_deg)

# The production render: download_dataset.py:231-232 does
#   equirectangular_to_perspective(equi, 90, azimuth, -30, 1024, 1024)
#   [0:1024, 341:341+341]
PERSP_PX = 1024
FOV_DEG = 90.0
PITCH_DEG = -30.0

# Context strip (the downsampled full pano). The JS draws in this coordinate
# space too, so the numbers travel through META rather than being repeated in
# the template — the same one-definition rule as everything else here.
CTX_W = 1024
CTX_H = 512

# The strip's true edges. Asymmetric — 341 px left of centre, 340 right — so
# they are computed per edge; crop_half_angle_deg() is the conservative
# symmetric bound §5g/§5j quote, carried alongside for comparability.
STRIP_LEFT_DEG = perspective_col_to_azimuth_deg(CROP_LO)    # -18.4577
STRIP_RIGHT_DEG = perspective_col_to_azimuth_deg(CROP_HI)   # +18.3678

# Records within this range of a panorama each get a Stage 1 crop —
# INCLUSION_DISTANCE_THRESHOLD in generate_dataset_meta.py:12, which cannot be
# imported because that module reads all_locations.csv at import time.
INCLUSION_DISTANCE_M = 35.0

# The pano-pick band. Below ~4 m the geometry degenerates (the ramp is under
# the camera and a fixed coordinate error subtends a huge angle — not the
# regime the corpus median of 11.1 m lives in); above 30 m the ramp is a few
# pixels. Both bounds are manifest entries, not folklore.
RANGE_BAND_M = (4.0, 30.0)

# Degree ruler ticks on the perspective view. The strip edges are drawn
# separately and exactly; these are orientation marks only.
DEGREE_TICKS = (-45, -30, -20, -10, -5, 5, 10, 20, 30, 45)

#: Mandatory reason tags for an unjudgeable verdict — the street-level
#: selection bias (#103: "parked vans, poles and low sun replace tree canopy.
#: Probably smaller, but it must be measured, not assumed"). ``ramp_outside_view``
#: is the one non-occlusion entry: the ramp is visible but beyond the ±45°
#: render, i.e. a coordinate error too large for this instrument to measure —
#: at the 11 m median range that is >11 m tangential, far past anything §5f/§5l
#: measured, so it is recorded as its own category rather than given a fake 45°.
UNREADABLE_REASONS = (
    ("van_or_vehicle", "van/vehicle"),
    ("pole_or_signage", "pole/signage"),
    ("sun_or_shadow", "sun/shadow"),
    ("too_far", "too far"),
    ("image_quality", "image quality"),
    ("ramp_outside_view", "outside view"),
    ("other", "other"),
)

# The provenance fields every record carries IDENTICALLY in three places: the
# chip dict the page renders from, the server-side verdict template, and the
# browser export. The aerial sheet lost `stratum` (§5l) and still silently
# drops `published_neighbours_m` because those three paths were maintained by
# hand; here the chip and template are built from ONE base dict, and the JS
# export copies these fields by iterating this very list (it is substituted
# into the page), so a field added here appears in all three or in none.
SHARED_FIELDS = (
    "id", "lon", "lat", "stratum",
    "pano_id", "pano_capture", "pano_heading_deg", "pano_lat", "pano_lon",
    "range_m", "n_candidates", "az_gov_deg", "theta_deg",
)

# The reviewer-owned fields. The template carries their defaults; the export
# serialises them from page state with per-field null handling.
VERDICT_FIELDS = ("offset_deg", "click_px", "unreadable", "unreadable_reason",
                  "no_ramp", "note")

# The rubric. One source of truth, rendered beside the field it governs and
# copied verbatim into the exported manifest — an angular verdict is
# uninterpretable without the rule saying what it is an angle *to*.
RUBRIC = {
    "click_target": (
        "Click the CENTRE of the ramp's concrete apron, at any height — ONLY THE "
        "HORIZONTAL POSITION IS MEASURED. Stage 1 consumes the government coordinate "
        "for its bearing alone (§5g), so the verdict is the horizontal angle between "
        "the ramp and the red crosshair line, and where you click vertically changes "
        "nothing. Do not click the detectable-warning pad when it is visibly offset "
        "sideways from the apron centre (oblique views): the same systematic-bias "
        "argument as the aerial rubric applies, just in degrees."
    ),
    "which_ramp": (
        "Click the ramp THIS RECORD most plausibly denotes, not merely the nearest "
        "in bearing. Use the magenta dashed bearings: each marks where ANOTHER "
        "published record projects, labelled with its ground distance from this "
        "record — a diamond sitting on the other visible ramp means that ramp is "
        "already claimed. When two ramps flank the crosshair and the assignment is "
        "genuinely undecidable, click your best call and note 'ambiguous' — the note "
        "is part of the record, and §5l found this exact case five times in Seattle."
    ),
    "always_click": (
        "Click on EVERY judgeable chip, including when the ramp sits dead on the "
        "crosshair — click the crosshair line itself for ~0°. Recording near-zero "
        "cases only by omission makes the low tail an artefact of reviewer "
        "confidence, exactly as on the aerial sheet."
    ),
    "strip_edges": (
        "The amber lines are the exact edges of the strip Stage 1 would cut "
        "(asymmetric: -18.46° left, +18.37° right). They are drawn so you can tell "
        "'just outside the crop' from 'no ramp here' — that distinction is the "
        "measurement. THE EDGES DO NOT BOUND WHERE YOU CLICK: click the ramp where "
        "it is, inside or outside."
    ),
    "no_ramp": (
        "The corner at the crosshair bearing is visible and readable, and there is "
        "definitively no curb ramp there. This is a PHANTOM record — a result, not a "
        "failure — and it is deliberately distinct from unjudgeable: 'I can see, and "
        "it is not there' versus 'I cannot see'."
    ),
    "unjudgeable": (
        "Something prevents a call — and the REASON IS MANDATORY, because it is "
        "itself a reported number: street level trades the aerial sheet's canopy "
        "bias for vans, poles and sun, and #103 requires that trade to be measured. "
        "'outside view' is the special case where the ramp IS visible but beyond the "
        "±45° render: that is a coordinate error larger than this instrument can "
        "measure, not an occlusion. A second-vantage pass over the unjudgeable "
        "subset is planned, so an accurate reason directly buys that pass its "
        "target list."
    ),
    "context_strip": (
        "The wide strip under the main view is the full panorama. The bracket marks "
        "the 90° view you are judging; the inner pair of lines is the crop strip. "
        "Use it to orient — e.g. to check whether a ramp you expected is behind the "
        "camera — never to measure."
    ),
    "resolution_floor": (
        "Angular offsets below roughly 1-2° are at this instrument's floor (a click "
        "lands within a few pixels ≈ 0.5°, and 'the centre of the apron' is itself "
        "several degrees wide at typical range). Read the left tail as floor-limited "
        "rather than as fractions of a degree. For scale: the corpus-city automatic "
        "null (§5j) has |median| 2.2-3.4° with the crop model in the loop."
    ),
    "sign_convention": (
        "Positive offset = the ramp is CLOCKWISE of the government bearing = to the "
        "RIGHT of the crosshair in the view. This matches stage1_bearing_residual.py "
        "(§5j), so candidate cities and corpus cities read in the same units with "
        "the same sign. The page computes the sign from your click; nothing to do — "
        "stated so the exported numbers can be read."
    ),
}


# --------------------------------------------------------------------------- #
# pure geometry
# --------------------------------------------------------------------------- #
def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres. Pure, dependency-free.

    Production computes (and discards) range via pyproj's ellipsoidal inverse;
    at the <=35 m scales here the sphere-vs-ellipsoid difference is <0.3%,
    far below the 4-30 m pick band's sensitivity, and staying dependency-free
    keeps this module importable in CI (no pyproj in requirements-dev).
    """
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def neighbour_offsets(rows, rec_lon, rec_lat, pano_lon, pano_lat, az_gov,
                      self_id, id_field="OBJECTID", radius_m=INCLUSION_DISTANCE_M,
                      max_draw_deg=45.0):
    """Other published records near this panorama, as bearing offsets.

    Returns ``(drawable, n_out_of_view)`` where ``drawable`` is
    ``[[offset_deg, dist_from_record_m], ...]`` sorted by |offset|. The
    membership rule is the production one — within ``radius_m`` of the
    *panorama*, because that is which records get their own Stage 1 crops —
    while the label is the ground distance from the *sampled record*, because
    the reviewer's question is association ("is that the adjacent corner's
    record?") and a 5 m label answers it the way the aerial sheet's diamond
    labels did. Records beyond the ±45° render are counted, not drawn. Pure.
    """
    lat_pad = radius_m / 111132.0
    lon_pad = radius_m / (111320.0 * math.cos(math.radians(pano_lat)) or 1e-9)
    drawable, out_of_view = [], 0
    for row in rows:
        if str(row.get(id_field)) == self_id:
            continue
        lon, lat = row.get("lon"), row.get("lat")
        if lon is None or lat is None:
            continue
        if abs(lat - pano_lat) > lat_pad or abs(lon - pano_lon) > lon_pad:
            continue
        if haversine_m(pano_lat, pano_lon, lat, lon) > radius_m:
            continue
        off = wrap_deg(fwd_azimuth_deg(pano_lat, pano_lon, lat, lon) - az_gov)
        if abs(off) <= max_draw_deg:
            d_rec = haversine_m(rec_lat, rec_lon, lat, lon)
            drawable.append([round(off, 2), round(d_rec, 1)])
        else:
            out_of_view += 1
    drawable.sort(key=lambda p: abs(p[0]))
    return drawable, out_of_view


# --------------------------------------------------------------------------- #
# the pano-pick rule — ONE definition, imported by the probe
# --------------------------------------------------------------------------- #
def choose_pano(cands, rec_lat, rec_lon, record_ym, band_m=RANGE_BAND_M):
    """Pick the panorama this record will be judged against.

    ``cands`` are dicts with ``pano_id``, ``lat``, ``lon``, ``date`` (GSV's
    non-padded ``"YYYY-M"``, parsed with the same ``parse_ym`` as everything
    else). The rule, in order:

    1. range band — within ``band_m`` of the record;
    2. temporal — capture ym >= the record's ym, so the ramp exists in the
       imagery (**this is the per-record temporal matching that eliminates the
       §5i/§5l confound**). An undated record (``record_ym is None``) accepts
       any pano — the flag travels in the manifest, not silently;
    3. nearest range, tie-break newest capture, then pano id (deterministic).

    Returns ``(chosen_or_None, status, stats)``; ``status`` is terminal and
    feeds the manifest's per-record accounting.
    """
    stats = {"n_panos": len(cands), "n_in_band": 0, "n_eligible": 0}
    if not cands:
        return None, "no_panos", stats

    enriched = []
    for c in cands:
        r = haversine_m(rec_lat, rec_lon, c["lat"], c["lon"])
        ym = parse_ym(c.get("date"))
        enriched.append((c, r, ym))

    in_band = [(c, r, ym) for c, r, ym in enriched
               if band_m[0] <= r <= band_m[1]]
    stats["n_in_band"] = len(in_band)
    if not in_band:
        return None, "no_pano_in_band", stats

    if record_ym is None:
        eligible = in_band
    else:
        eligible = [(c, r, ym) for c, r, ym in in_band
                    if ym is not None and ym >= record_ym]
    stats["n_eligible"] = len(eligible)
    if not eligible:
        return None, "no_dated_pano_in_band", stats

    def _key(item):
        c, r, ym = item
        months = ym[0] * 12 + ym[1] if ym else -1
        return (round(r, 3), -months, c["pano_id"])

    c, r, ym = min(eligible, key=_key)
    chosen = dict(c, range_m=round(r, 2))
    return chosen, "ok", stats


# --------------------------------------------------------------------------- #
# sites
# --------------------------------------------------------------------------- #
def load_sites_from_verdicts(path):
    """The records of a built aerial sheet, plus its provenance.

    ALL records are taken — including the aerial-unjudgeable ones, which are
    among the most interesting here (street level looking under the canopy is
    argument 3 of #103). Returns ``(sites, source)`` where each site carries
    ``id/lon/lat/stratum`` and ``source`` records where the sample came from.
    """
    with open(path, encoding="utf-8") as fh:
        vd = json.load(fh)
    sites = [{"id": str(r["id"]), "lon": r["lon"], "lat": r["lat"],
              "stratum": r.get("stratum")} for r in vd["records"]]
    source = {"mode": "verdicts", "path": os.path.basename(path),
              "seed": vd.get("seed"), "sheet_build": vd.get("sheet_build"),
              "city": vd.get("city"), "n_records": len(sites)}
    return sites, source


# --------------------------------------------------------------------------- #
# caches — §5h's lessons, structural
# --------------------------------------------------------------------------- #
def _search_cache_path(cache_dir, lat, lon):
    return os.path.join(cache_dir, "search_{:.7f}_{:.7f}.json".format(lat, lon))


def cached_search(lat, lon, cache_dir, sleep_s=0.0):
    """``search_panoramas`` with an on-disk cache.

    A successful search — including one returning zero panoramas — is cached
    as its result; **a failed search is never cached**, so a transient cannot
    masquerade as "no coverage here" (the §5h zero-byte trap, avoided by
    construction: absence-of-panos and failure-to-ask are different records).
    The probe warms this cache and the sheet build consumes it, halving the
    load on the undocumented endpoint. Network import is lazy: search_panos
    pulls pydantic/requests, which CI does not have.
    """
    path = _search_cache_path(cache_dir, lat, lon)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)["panos"]

    sys.path.insert(0, os.path.join(REPO, "stage_one", "dataset_generation"))
    from search_panos import search_panoramas
    panos = [{"pano_id": p.pano_id, "lat": p.lat, "lon": p.lon,
              "heading": p.heading, "date": p.date}
             for p in search_panoramas(lat, lon)]
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"fetched_at": int(time.time()), "panos": panos}, fh)
    if sleep_s:
        time.sleep(sleep_s)
    return panos


def cached_pano_heading(pano_id, cache_dir):
    """Production's ``get_pano_heading`` (the GetMetadata value the pipeline
    itself uses — NOT the search response's heading field, which is a second
    source that has never been verified against it), cached on disk."""
    path = os.path.join(cache_dir, "heading_{}.json".format(pano_id))
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)["heading"]
    sys.path.insert(0, os.path.join(REPO, "stage_one", "dataset_generation"))
    from search_panos import get_pano_heading
    heading = get_pano_heading(pano_id)
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"heading": heading, "fetched_at": int(time.time())}, fh)
    return heading


def fetch_panorama_cached(pano_id, cache_dir, refetch_absent=False, attempts=2,
                          retry_sleep_s=2.0):
    """The production ``fetch_panorama`` behind a cache with HONEST absences.

    Returns ``(equi_bgr_or_None, reason_or_None)``. Success caches the
    assembled 4096x2048 pano as a q95 JPEG keyed by pano id. Failure — after
    ``attempts`` tries, because ``fetch_panorama`` returning ``None`` cannot
    distinguish "no such pano at this zoom" from a transient during its
    dimension probe — writes ``<pano_id>.absent.json`` carrying the reason,
    attempt count and timestamp. **Never a zero-byte sentinel**: §5h's retry
    fix was inert precisely because absences were unreadable, and
    ``--refetch-absent`` (which deletes the marker and retries) needs a marker
    it can reason about. A network *exception* propagates uncached — the
    caller records it as its own status.

    Note: every chosen pano came from a search result, so its id exists as
    metadata and an absence here is *suspicious by construction* — the caller
    prints it loudly rather than folding it into a count.
    """
    import numpy as np
    from PIL import Image

    os.makedirs(cache_dir, exist_ok=True)
    jpg = os.path.join(cache_dir, pano_id + ".jpg")
    marker = os.path.join(cache_dir, pano_id + ".absent.json")

    if os.path.exists(jpg):
        rgb = np.asarray(Image.open(jpg).convert("RGB"))
        return rgb[..., ::-1].copy(), None      # back to production BGR
    if os.path.exists(marker):
        if not refetch_absent:
            with open(marker, encoding="utf-8") as fh:
                reason = json.load(fh).get("reason", "unknown")
            return None, "absent_cached:{}".format(reason)
        os.remove(marker)

    from rampnet.gsv import fetch_panorama
    equi = None
    for attempt in range(attempts):
        equi = fetch_panorama(pano_id)
        if equi is not None:
            break
        if attempt < attempts - 1:
            time.sleep(retry_sleep_s)
    if equi is None:
        with open(marker, "w", encoding="utf-8") as fh:
            json.dump({"reason": "fetch_returned_none", "attempts": attempts,
                       "fetched_at": int(time.time())}, fh)
        return None, "fetch_returned_none"

    Image.fromarray(equi[..., ::-1]).save(jpg, format="JPEG", quality=95)
    return equi, None


# --------------------------------------------------------------------------- #
# rendering — the production path, exactly
# --------------------------------------------------------------------------- #
def render_views(equi_bgr, theta_deg, ctx_size=(CTX_W, CTX_H)):
    """The 90° perspective at the government bearing, plus a context strip.

    Returns ``(persp_rgb_pil, ctx_rgb_pil)``. The perspective is the exact
    production call — ``equirectangular_to_perspective(equi, 90, theta, -30,
    1024, 1024)`` — of which Stage 1 would keep columns 341:682. theta is
    passed wrapped; the renderer is periodic in theta, so this is identical to
    production's unwrapped value. BGR->RGB happens here and only here.
    """
    from PIL import Image
    persp = equirectangular_to_perspective(
        equi_bgr, FOV_DEG, theta_deg, PITCH_DEG, PERSP_PX, PERSP_PX)
    persp_pil = Image.fromarray(persp[..., ::-1])
    ctx_pil = Image.fromarray(equi_bgr[..., ::-1]).resize(
        ctx_size, Image.LANCZOS)
    return persp_pil, ctx_pil


# --------------------------------------------------------------------------- #
# the sheet
# --------------------------------------------------------------------------- #
# NOTE: unlike the aerial template this one contains NO brace-escaping layer —
# build_sheet does pure __TOKEN__ replacement, so every { } below is literal.
# The aerial sheet's {{ }} doubling is a fossil of str.format and produced the
# blank-page hazard class its Node test exists to catch; not inheriting the
# hazard beats testing for it (the Node test is inherited anyway).
SHEET_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>__CITY__ — street-level location review (#103)</title>
<style>
 :root { --bg:#111; --panel:#1c1c1c; --line:#2e2e2e; --dim:#9a9a9a; }
 * { box-sizing: border-box; }
 body { font:14px/1.55 system-ui,sans-serif; margin:0; background:var(--bg); color:#eee; }
 header { position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line);
          padding:14px 24px; display:flex; gap:20px; align-items:center; flex-wrap:wrap; }
 h1 { font-size:17px; margin:0; font-weight:600; }
 .sub { color:var(--dim); font-size:12px; width:100%; margin-top:2px; }
 .ctl { display:flex; gap:6px; align-items:center; font-size:13px; color:#ddd; }
 button { font:inherit; background:#2a2a2a; color:#eee; border:1px solid var(--line);
          border-radius:5px; padding:5px 11px; cursor:pointer; }
 button:hover { background:#343434; }
 button.primary { background:#2d5a3d; border-color:#3c7a52; }
 .prog { font-variant-numeric:tabular-nums; color:var(--dim); }
 .build { font-family:ui-monospace,monospace; font-size:11px; color:#8fb8d8;
          background:#16222c; border:1px solid #24404f; border-radius:4px;
          padding:3px 7px; cursor:help; }
 main { padding:20px 24px 60px; }
 .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(250px,1fr)); gap:14px; }
 figure { margin:0; background:var(--panel); border-radius:7px; padding:7px;
          border:1px solid transparent; }
 figure.done { border-color:#3c7a52; }
 figure.skip { border-color:#7a5a3c; opacity:.65; }
 figure.phantom { border-color:#8a3c3c; }
 figure.partial { border-color:#b08028; }
 figure.partial .tag { color:#d69f3a; }
 .wrap { position:relative; cursor:zoom-in; line-height:0; }
 .wrap img { width:100%; border-radius:4px; display:block; }
 .wrap svg { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
 body.nooverlay svg { display:none; }
 figcaption { font-size:11px; color:#999; margin-top:5px; font-family:ui-monospace,monospace;
              display:flex; justify-content:space-between; }
 dialog { border:none; background:var(--panel); color:#eee; border-radius:10px; padding:0;
          max-width:96vw; max-height:96vh; }
 dialog::backdrop { background:rgba(0,0,0,.82); }
 .modal { display:flex; gap:18px; padding:18px; align-items:flex-start; }
 .left { display:flex; flex-direction:column; gap:8px; }
 .stage { position:relative; line-height:0; cursor:crosshair; }
 .stage img { display:block; border-radius:5px;
              width:min(66vh,calc(96vw - 360px)); height:auto; }
 .stage svg { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
 .ctx { position:relative; line-height:0; }
 .ctx img { display:block; border-radius:4px; width:min(66vh,calc(96vw - 360px)); height:auto; }
 .ctx svg { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
 .side { width:300px; font-size:13px; }
 .side h2 { font-size:14px; margin:0 0 4px; }
 .meta { font-family:ui-monospace,monospace; font-size:11px; color:#9a9a9a; margin:0 0 10px; }
 .row { margin-bottom:13px; }
 .row label { display:block; color:var(--dim); font-size:12px; margin-bottom:4px; }
 .seg { display:flex; gap:4px; flex-wrap:wrap; }
 .seg button { flex:1; min-width:38px; padding:5px 4px; }
 .seg button[aria-pressed="true"] { background:#2d5a3d; border-color:#3c7a52; }
 .measure { font-family:ui-monospace,monospace; font-size:19px; }
 .measure em { color:var(--dim); font-size:12px; font-style:normal; }
 .inout { font-size:12px; font-family:ui-monospace,monospace; }
 .inout.in { color:#7bc98a; }
 .inout.out { color:#e07a7a; }
 input[type=text] { width:100%; background:#111; color:#eee; border:1px solid var(--line);
                    border-radius:5px; padding:6px; font:inherit; }
 .nav { display:flex; gap:8px; margin-top:16px; }
 .nav button { flex:1; }
 kbd { background:#000; border:1px solid var(--line); border-radius:3px; padding:0 4px;
       font-size:11px; font-family:ui-monospace,monospace; }
 .help { color:var(--dim); font-size:11.5px; margin-top:14px; line-height:1.5; }
 .hint { color:#b9b9b9; font-size:11px; line-height:1.45; margin:5px 0 0;
         border-left:2px solid #3a3a3a; padding-left:7px; }
 .hint b { color:#ffd479; font-weight:600; }
 #rubric-dlg { max-width:760px; }
 #rubric-dlg .modal { display:block; padding:24px 26px; }
 #rubric-dlg h3 { font-size:13px; margin:16px 0 4px; color:#ffd479; }
 #rubric-dlg h3:first-of-type { margin-top:6px; }
 #rubric-dlg p { margin:0; color:#ddd; font-size:12.5px; line-height:1.6; }
</style>

<header>
 <h1>__CITY__ — street-level review</h1>
 <label class="ctl"><input type="checkbox" id="ovl" checked> overlay <kbd>o</kbd></label>
 <span class="ctl prog" id="prog"></span>
 <span class="build" title="Content hash of this sheet's logic and rubric. If it does not
match what the tool last printed, this page is stale -- hard-reload.">build __BUILD__</span>
 <button id="next-todo">next unreviewed <kbd>n</kbd></button>
 <button id="show-rubric">rubric <kbd>?</kbd></button>
 <button class="primary" id="export">export verdicts.json</button>
 <div class="sub">
  __N__ chips · <code>__INV__</code> · sites: __SITES__ ·
  GSV panoramas via the Stage 1 production path (rampnet.gsv), 90° view at the
  record's bearing, pitch −30° · amber lines = the exact crop strip
  (−18.46°/+18.37°) · <b>offsets are DEGREES, positive = right of the crosshair</b>.
  Progress is saved in this browser; <b>export before you finish</b> to write it to disk.
 </div>
</header>

<main><div class="grid" id="grid"></div></main>

<dialog id="dlg"><div class="modal">
 <div class="left">
  <div class="stage" id="stage"><img id="big" alt=""><svg id="bigsvg"></svg></div>
  <div class="ctx" id="ctxwrap"><img id="ctximg" alt=""><svg id="ctxsvg"></svg></div>
 </div>
 <div class="side">
  <h2 id="title"></h2>
  <p class="meta" id="meta"></p>
  <div class="row">
   <label>angular offset — click the ramp this record denotes</label>
   <div class="measure" id="offset">—</div>
   <div class="inout" id="inout"></div>
   <em id="offhint">click to measure · click the crosshair line for ~0°</em>
   <p class="hint"><b>Horizontal position only</b> — the verdict is the angle to the red
    line; height does not matter. <b>Click every judgeable chip</b>, even dead-centre.
    The amber strip edges do NOT bound where you click.</p>
  </div>
  <div class="row">
   <label>corner readable and there is no ramp</label>
   <div class="seg" id="noramp"></div>
   <p class="hint"><b>A phantom record — a result, not a failure.</b> Distinct from
    unjudgeable: “I can see, and it is not there”.</p>
  </div>
  <div class="row">
   <label>unjudgeable — and the reason is mandatory</label>
   <div class="seg" id="unread"></div>
   <div class="seg" id="reason" style="margin-top:4px"></div>
   <p class="hint"><b>The reason is a reported number</b>: street level trades canopy for
    vans, poles and sun, and that trade must be measured. <b>outside view</b> = the ramp
    is visible but beyond ±45° — a coordinate error too large for this instrument.</p>
  </div>
  <div class="row">
   <label>note</label>
   <input type="text" id="note" placeholder="optional — e.g. 'ambiguous', 'parallel'">
  </div>
  <div class="nav">
   <button id="prev">← prev</button>
   <button id="nxt">next →</button>
   <button id="close">close <kbd>esc</kbd></button>
  </div>
  <div class="help">
   Magenta dashed lines = bearings of OTHER published records (labelled with ground
   distance from this record) — use them to decide which ramp is this record's.
   <kbd>u</kbd> unjudgeable · <kbd>1</kbd>–<kbd>7</kbd> reason ·
   <kbd>p</kbd> no ramp · <kbd>←</kbd> <kbd>→</kbd> move · <kbd>o</kbd> overlay ·
   <kbd>?</kbd> full rubric.
  </div>
 </div>
</div></dialog>

<dialog id="rubric-dlg"><div class="modal">
 <h2 style="margin:0 0 4px">Review rubric — __CITY__ (street-level)</h2>
 <p style="color:#9a9a9a;font-size:12px;margin:0 0 6px">
  Exported verbatim into <code>verdicts.json</code>, because a verdict cannot be read later
  without the rule that produced it.</p>
 <div id="rubric-body"></div>
 <div class="nav"><button id="rubric-close">close <kbd>esc</kbd></button></div>
</div></dialog>

<script>
const META = __META__;
const CHIPS = __CHIPS__;
// Namespaced: the aerial sheet's key is "rampnet-verdicts-<city>-<seed>", and
// the Denver pilot shares both city and seed with it — an unnamespaced key
// would silently merge two instruments' state.
const KEY = "rampnet-gsv-verdicts-" + META.city + "-" + META.seed;
const V = JSON.parse(localStorage.getItem(KEY) || "{}");
const S = META.persp_px, C = S / 2;
// The click-to-angle map. Same formula as rampnet.gsv's helpers; the Python
// and JS copies are pinned together by the page-logic test, which asserts
// this function against values computed by the Python side.
const F = C / Math.tan(META.fov_deg / 2 * Math.PI / 180);
const colOf = deg => C + F * Math.tan(deg * Math.PI / 180);
const degOf = px => Math.atan((px - C) / F) * 180 / Math.PI;
const insideStrip = deg =>
  deg != null && deg >= META.strip_left_deg && deg <= META.strip_right_deg;

function persp_overlay(c, v) {
  let p = "";
  // Degree ruler (orientation only).
  META.ticks.forEach(t => {
    const x = colOf(t);
    p += `<line x1="${x}" y1="${S - S * .035}" x2="${x}" y2="${S}"
           stroke="#bbb" stroke-opacity=".8" stroke-width="${S / 700}"/>
          <text x="${x}" y="${S - S * .042}" fill="#bbb" font-size="${S / 55}"
           font-family="system-ui" text-anchor="middle">${t}°</text>`;
  });
  // The exact strip edges, drawn separately because the crop is asymmetric.
  [META.strip_left_deg, META.strip_right_deg].forEach(d => {
    const x = colOf(d);
    p += `<line x1="${x}" y1="0" x2="${x}" y2="${S}" stroke="#ffb300"
           stroke-opacity=".9" stroke-width="${S / 400}" stroke-dasharray="${S / 90}"/>`;
  });
  p += `<text x="${colOf(META.strip_right_deg) + S * .008}" y="${S * .035}" fill="#ffb300"
         font-size="${S / 48}" font-family="system-ui">crop edge</text>`;
  // The government bearing — the reference line of the whole measurement.
  p += `<line x1="${C}" y1="0" x2="${C}" y2="${S}" stroke="#ff4040"
         stroke-width="${S / 500}"/>`;
  // Other published records' bearings, always visible: there is no counting
  // task for them to anchor, and they answer "which ramp is this record's".
  (c.neighbors || []).forEach(n => {
    const x = colOf(n[0]), r = S / 70, y = S * .12;
    p += `<line x1="${x}" y1="0" x2="${x}" y2="${S}" stroke="#ff6fd8"
           stroke-opacity=".55" stroke-width="${S / 700}" stroke-dasharray="${S / 140}"/>
          <path d="M ${x} ${y - r} L ${x + r} ${y} L ${x} ${y + r} L ${x - r} ${y} Z"
           fill="none" stroke="#000" stroke-opacity=".7" stroke-width="${S / 150}"/>
          <path d="M ${x} ${y - r} L ${x + r} ${y} L ${x} ${y + r} L ${x - r} ${y} Z"
           fill="none" stroke="#ff6fd8" stroke-width="${S / 400}"/>
          <text x="${x + r * 1.3}" y="${y - r * .4}" fill="#ff6fd8" stroke="#000"
           stroke-width="${S / 400}" paint-order="stroke" font-size="${S / 50}"
           font-family="system-ui">${n[1]}m</text>`;
  });
  if (v && v.click_x != null) {
    p += `<g stroke="#4fc3f7"><line x1="${v.click_x}" y1="0" x2="${v.click_x}" y2="${S}"
           stroke-width="${S / 500}"/><circle cx="${v.click_x}" cy="${v.click_y}" r="${S / 70}"
           fill="none" stroke-width="${S / 300}"/></g>`;
  }
  return p;
}

// The context strip: full pano, with the 90° window and the crop strip marked.
// x is a fraction of the pano width; azimuth a (relative to the pano heading)
// sits at (wrap(a)+180)/360. Dimensions come from META (one definition, the
// Python side that rendered the image).
const CW = META.ctx_w, CH = META.ctx_h;
const wrapDeg = a => { a = (a + 180) % 360; if (a < 0) a += 360; return a - 180; };
const ctxX = a => (wrapDeg(a) + 180) / 360 * CW;

function ctx_overlay(c) {
  const t = c.theta_deg;
  let p = "";
  const vline = (a, color, w) => {
    const x = ctxX(a);
    return `<line x1="${x}" y1="0" x2="${x}" y2="${CH}" stroke="${color}"
             stroke-width="${w}"/>`;
  };
  p += vline(t - META.fov_deg / 2, "#9ecbff", 2) + vline(t + META.fov_deg / 2, "#9ecbff", 2);
  p += vline(t + META.strip_left_deg, "#ffb300", 2) + vline(t + META.strip_right_deg, "#ffb300", 2);
  p += vline(t, "#ff4040", 2);
  p += `<text x="${ctxX(t)}" y="${CH * .1}" fill="#ff4040" font-size="22"
         font-family="system-ui" text-anchor="middle">▼ record bearing</text>`;
  return p;
}

function state(id) { return V[id] || (V[id] = {}); }
function save() { localStorage.setItem(KEY, JSON.stringify(V)); paint(); }
function done(v) { return v && (v.unreadable || v.no_ramp || v.offset_deg != null); }
// Unjudgeable WITHOUT its reason is judged-but-not-recorded: the reason is a
// reported number (the instrument's selection bias), so it gates completeness
// exactly as the aerial sheet's per-corner count did.
function complete(v) {
  if (!v) return false;
  if (v.unreadable) return !!v.unreadable_reason;
  return v.no_ramp || v.offset_deg != null;
}
function partial(v) { return done(v) && !complete(v); }

function fmtOff(v) {
  return (v.offset_deg >= 0 ? "+" : "") + v.offset_deg.toFixed(2) + "°";
}

function paint() {
  let n = 0, part = 0;
  CHIPS.forEach(c => {
    const v = V[c.id], f = document.getElementById("f" + c.id);
    if (!f) return;
    f.className = partial(v) ? "partial"
      : (v && v.unreadable ? "skip" : (v && v.no_ramp ? "phantom"
         : (complete(v) ? "done" : "")));
    if (complete(v)) n++;
    else if (partial(v)) part++;
    const tag = f.querySelector(".tag");
    tag.textContent = !v ? ""
      : partial(v) ? "no reason"
      : v.unreadable ? "unjudgeable · " + (v.unreadable_reason || "")
      : v.no_ramp ? "no ramp"
      : (v.offset_deg != null ? fmtOff(v) : "");
  });
  document.getElementById("prog").textContent =
    n + " / " + CHIPS.length + " complete" + (part ? "  ·  " + part + " partial" : "");
}

const grid = document.getElementById("grid");
grid.innerHTML = CHIPS.map(c => `<figure id="f${c.id}"><div class="wrap" data-id="${c.id}">
  <img src="${c.uri}" alt="${c.id}" loading="lazy">
  <svg viewBox="0 0 ${S} ${S}" xmlns="http://www.w3.org/2000/svg"></svg></div>
  <figcaption><span>${c.id}</span><span class="tag"></span></figcaption></figure>`).join("");

const dlg = document.getElementById("dlg"), big = document.getElementById("big"),
      bigsvg = document.getElementById("bigsvg"), stage = document.getElementById("stage"),
      ctximg = document.getElementById("ctximg"), ctxsvg = document.getElementById("ctxsvg");
let cur = 0;

// Grid thumbnails carry the same overlay as the modal, painted once at load.
CHIPS.forEach(c => {
  const svg = document.getElementById("f" + c.id).querySelector("svg");
  if (svg) svg.innerHTML = persp_overlay(c, null);
});

function seg(el, opts, get, set) {
  el.innerHTML = opts.map(o =>
    `<button data-v="${o.v}" aria-pressed="${String(get() === o.v)}">${o.t}</button>`).join("");
  el.querySelectorAll("button").forEach(b => b.onclick = () => {
    const raw = b.dataset.v;
    const val = raw === "true" ? true : raw === "false" ? false : raw;
    set(get() === val ? null : val);
    save(); render();
  });
}

function render() {
  const c = CHIPS[cur], v = state(c.id);
  big.src = c.uri; big.alt = c.id;
  bigsvg.setAttribute("viewBox", `0 0 ${S} ${S}`);
  bigsvg.innerHTML = persp_overlay(c, v);
  ctximg.src = c.ctx_uri;
  ctxsvg.setAttribute("viewBox", `0 0 ${CW} ${CH}`);
  ctxsvg.innerHTML = ctx_overlay(c);
  document.getElementById("title").textContent =
    `${c.id}  (${cur + 1}/${CHIPS.length})`;
  document.getElementById("meta").textContent =
    `pano ${c.pano_id} · captured ${c.pano_capture} · range ${c.range_m} m` +
    ` · ${c.n_candidates} candidate panos` +
    (c.n_neighbors_out_of_view ? ` · ${c.n_neighbors_out_of_view} published nearby beyond ±45°` : "");
  document.getElementById("offset").textContent =
    v.offset_deg == null ? "—" : fmtOff(v);
  const io = document.getElementById("inout");
  if (v.offset_deg == null) { io.textContent = ""; io.className = "inout"; }
  else if (insideStrip(v.offset_deg)) { io.textContent = "inside the crop strip"; io.className = "inout in"; }
  else { io.textContent = "OUTSIDE the crop strip"; io.className = "inout out"; }
  seg(document.getElementById("noramp"), [{v: true, t: "no ramp here"}],
      () => v.no_ramp || null, x => {
        v.no_ramp = !!x;
        if (x) { v.unreadable = false; v.unreadable_reason = null;
                 v.offset_deg = null; v.click_x = v.click_y = null; }
      });
  // The terminal states are mutually exclusive, and neither carries a click:
  // "I cannot make a call" and "the call is +4.5°" are contradictory claims
  // (the aerial sheet's disowned-click lesson, inherited verbatim).
  seg(document.getElementById("unread"), [{v: true, t: "unjudgeable"}],
      () => v.unreadable || null, x => {
        v.unreadable = !!x;
        if (x) { v.no_ramp = false; v.offset_deg = null; v.click_x = v.click_y = null; }
        else { v.unreadable_reason = null; }
      });
  const rs = document.getElementById("reason");
  if (v.unreadable) {
    seg(rs, META.reasons.map(r => ({v: r[0], t: r[1]})),
        () => v.unreadable_reason || null, x => { v.unreadable_reason = x; });
  } else {
    rs.innerHTML = "";
  }
  document.getElementById("note").value = v.note || "";
}

function open_(i) { cur = (i + CHIPS.length) % CHIPS.length; render();
  if (!dlg.open) dlg.showModal(); }

grid.querySelectorAll(".wrap").forEach(w => w.onclick = () =>
  open_(CHIPS.findIndex(c => c.id === w.dataset.id)));

// Click-to-measure. The reviewer points at the ramp; the page does the trig.
// Only the horizontal position carries information — §5g proved the gate is
// angular — but the row is kept for the marker.
document.getElementById("stage").onclick = e => {
  const r = big.getBoundingClientRect();
  const px = (e.clientX - r.left) / r.width * S, py = (e.clientY - r.top) / r.height * S;
  const v = state(CHIPS[cur].id);
  v.click_x = px; v.click_y = py;
  v.offset_deg = degOf(px);
  v.unreadable = false; v.unreadable_reason = null; v.no_ramp = false;
  save(); render();
};

document.getElementById("note").oninput = e => {
  state(CHIPS[cur].id).note = e.target.value; save();
};
document.getElementById("prev").onclick = () => open_(cur - 1);
document.getElementById("nxt").onclick = () => open_(cur + 1);
document.getElementById("close").onclick = () => dlg.close();
document.getElementById("ovl").onchange = e =>
  document.body.classList.toggle("nooverlay", !e.target.checked);

function nextTodo() {
  let i = CHIPS.findIndex(c => !done(V[c.id]));
  if (i < 0) i = CHIPS.findIndex(c => partial(V[c.id]));
  if (i < 0) return alert("Every chip is fully recorded. Export when ready.");
  open_(i);
}
document.getElementById("next-todo").onclick = nextTodo;

const rubricDlg = document.getElementById("rubric-dlg");
document.getElementById("rubric-body").innerHTML =
  Object.entries(META.manifest.rubric || {}).map(([k, text]) =>
    `<h3>${k.replace(/_/g, " ")}</h3><p>${text}</p>`).join("");
document.getElementById("show-rubric").onclick = () => rubricDlg.showModal();
document.getElementById("rubric-close").onclick = () => rubricDlg.close();

addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "?") { rubricDlg.open ? rubricDlg.close() : rubricDlg.showModal(); return; }
  if (rubricDlg.open) return;
  if (e.key === "o") { const b = document.getElementById("ovl");
    b.checked = !b.checked; b.onchange({target: b}); return; }
  if (e.key === "n" && !dlg.open) return nextTodo();
  if (!dlg.open) return;
  const v = state(CHIPS[cur].id);
  if (e.key === "ArrowLeft") open_(cur - 1);
  else if (e.key === "ArrowRight") open_(cur + 1);
  else if (e.key === "u") { v.unreadable = !v.unreadable;
    if (v.unreadable) { v.no_ramp = false; v.offset_deg = null; v.click_x = v.click_y = null; }
    else { v.unreadable_reason = null; }
    save(); render(); }
  else if (e.key === "p") { v.no_ramp = !v.no_ramp;
    if (v.no_ramp) { v.unreadable = false; v.unreadable_reason = null;
                     v.offset_deg = null; v.click_x = v.click_y = null; }
    save(); render(); }
  else if (v.unreadable && "1234567".includes(e.key)) {
    const r = META.reasons[+e.key - 1];
    if (r) { v.unreadable_reason = r[0]; save(); render(); }
  }
});

document.getElementById("export").onclick = () => {
  const out = Object.assign({}, META.manifest, {
    reviewer: META.manifest.reviewer, records: CHIPS.map(c => {
      const rec = {};
      // The shared provenance fields are copied FROM THE SAME LIST the Python
      // side builds chips and templates from — the third copy of the field
      // list collapses into the first, which is how the aerial sheet's
      // dropped-stratum class of bug (§5l) is prevented rather than tested for.
      META.shared_fields.forEach(f => { rec[f] = c[f] === undefined ? null : c[f]; });
      const v = V[c.id] || {};
      rec.offset_deg = v.offset_deg == null ? null : +v.offset_deg.toFixed(2);
      rec.click_px = v.click_x == null ? null : [+v.click_x.toFixed(1), +v.click_y.toFixed(1)];
      rec.unreadable = !!v.unreadable;
      rec.unreadable_reason = v.unreadable ? (v.unreadable_reason || null) : null;
      rec.no_ramp = !!v.no_ramp;
      rec.note = v.note || "";
      return rec;
    })
  });
  const blob = new Blob([JSON.stringify(out, null, 2) + "\\n"], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "verdicts.json"; a.click();
};

paint();
</script>
"""


def sheet_build_id():
    """Short content hash of the page logic and the rubric — the "is my page
    stale?" answer, shown in the header and written to the manifest."""
    blob = (SHEET_TEMPLATE + json.dumps(RUBRIC, sort_keys=True)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def build_sheet(meta, chips, manifest):
    """Assemble the sheet. Pure — takes rendered chips, returns HTML.

    Plain ``__TOKEN__`` replacement; the template's braces are literal (no
    ``{{``/``}}`` escaping layer to slip on — see the note above the template).
    """
    subs = {
        "__BUILD__": sheet_build_id(),
        "__CITY__": meta["city"],
        "__N__": str(len(chips)),
        "__INV__": meta["inventory"],
        "__SITES__": meta["sites_desc"],
        "__META__": json.dumps({
            "city": meta["city"], "seed": meta["seed"],
            "persp_px": PERSP_PX, "fov_deg": FOV_DEG,
            "ctx_w": CTX_W, "ctx_h": CTX_H,
            "strip_left_deg": STRIP_LEFT_DEG, "strip_right_deg": STRIP_RIGHT_DEG,
            "ticks": list(DEGREE_TICKS),
            "reasons": [list(r) for r in UNREADABLE_REASONS],
            "shared_fields": list(SHARED_FIELDS),
            "manifest": manifest,
        }),
        "__CHIPS__": json.dumps(chips),
    }
    out = SHEET_TEMPLATE
    for k, v in subs.items():
        out = out.replace(k, v)
    return out


def make_base_record(site, chosen, heading, az_gov, theta, n_candidates):
    """The ONE construction site for the shared provenance fields.

    Both the chip dict and the server-side verdict template extend this dict,
    so the two Python paths cannot drift; the JS export path iterates
    ``META.shared_fields``. A test asserts the keys equal ``SHARED_FIELDS``.
    """
    return {
        "id": site["id"], "lon": site["lon"], "lat": site["lat"],
        "stratum": site.get("stratum"),
        "pano_id": chosen["pano_id"], "pano_capture": chosen.get("date"),
        "pano_heading_deg": round(heading, 2),
        "pano_lat": chosen["lat"], "pano_lon": chosen["lon"],
        "range_m": chosen["range_m"], "n_candidates": n_candidates,
        "az_gov_deg": round(az_gov, 2), "theta_deg": round(theta, 2),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--city", required=True)
    ap.add_argument("--inventory", required=True,
                    help="frozen .jsonl.gz snapshot (data/inventories/)")
    ap.add_argument("--sites-from-verdicts", default=None,
                    help="render EXACTLY the records of a built aerial sheet "
                         "(all of them, including its unjudgeables — looking "
                         "under the canopy is the point). The Denver pilot "
                         "uses this for per-record paired calibration against "
                         "the trusted aerial answer.")
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--seed", type=int, default=None,
                    help="required for fresh sampling; with --sites-from-verdicts "
                         "the source sheet's seed is inherited")
    ap.add_argument("--sampling", choices=("uniform", "stratified"), default="uniform")
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--id-field", default="OBJECTID")
    ap.add_argument("--date-field", default=None,
                    help="record date field for the per-record temporal match "
                         "(e.g. CREATEDATE). Parsed with temporal_gap.parse_ym; "
                         "sentinels count as undated. Without it every record "
                         "is undated and any pano is temporally eligible — "
                         "recorded in the manifest, not silent.")
    ap.add_argument("--where-field", default=None)
    ap.add_argument("--where-value", default=None)
    ap.add_argument("--where-not", default=None)
    ap.add_argument("--strata-year-field", default=None)
    ap.add_argument("--strata-year-cutoff", type=int, default=None)
    ap.add_argument("--band-min", type=float, default=RANGE_BAND_M[0])
    ap.add_argument("--band-max", type=float, default=RANGE_BAND_M[1])
    ap.add_argument("--jpeg-quality", type=int, default=82,
                    help="perspective view JPEG quality (the aerial sheet "
                         "hard-wired 85; a 1024px view at 82 is ~150 KB)")
    ap.add_argument("--ctx-quality", type=int, default=70)
    ap.add_argument("--sleep", type=float, default=0.3,
                    help="pause between pano SEARCHES (each is 1 GET + one "
                         "metadata POST per returned pano, against an "
                         "undocumented endpoint)")
    ap.add_argument("--refetch-absent", action="store_true",
                    help="re-try panoramas cached as absent (deletes their "
                         "marker files first) — the §5h lesson: a cached "
                         "absence must be re-testable without hand-deleting "
                         "cache entries")
    ap.add_argument("--limit", type=int, default=None,
                    help="build only the first N sites (smoke runs)")
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args(argv)

    if args.strata_year_field and args.strata_year_cutoff is None:
        ap.error("--strata-year-field needs --strata-year-cutoff")
    if args.where_value is not None and args.where_not is not None:
        ap.error("--where-value and --where-not are mutually exclusive")
    if args.where_field and args.where_value is None and args.where_not is None:
        ap.error("--where-field needs either --where-value or --where-not")

    rows = load_inventory(args.inventory)
    by_id = {str(r.get(args.id_field)): r for r in rows}
    band = (args.band_min, args.band_max)

    # ---- resolve sites ---------------------------------------------------- #
    strata_sizes = None
    if args.sites_from_verdicts:
        sites, sites_source = load_sites_from_verdicts(args.sites_from_verdicts)
        missing = [s["id"] for s in sites if s["id"] not in by_id]
        if missing:
            ap.error("{} verdict record ids not in the frozen inventory "
                     "(first: {}) — wrong snapshot?".format(len(missing), missing[:3]))
        seed = args.seed if args.seed is not None else sites_source["seed"]
        if seed is None:
            ap.error("the source verdicts carry no seed; pass --seed explicitly "
                     "(it namespaces the page's localStorage)")
        sites_desc = "{} records of {} (aerial sheet build {})".format(
            len(sites), sites_source["path"], sites_source["sheet_build"])
        sampling_desc = "sites-from-verdicts"
    else:
        if args.seed is None:
            ap.error("fresh sampling needs an explicit --seed")
        seed = args.seed
        frame = list(range(len(rows)))
        if args.where_field:
            if args.where_not is not None:
                frame = [i for i in frame
                         if str(rows[i].get(args.where_field)) != args.where_not]
            else:
                frame = [i for i in frame
                         if str(rows[i].get(args.where_field)) == args.where_value]
            if not frame:
                ap.error("the sample frame is empty")
            print("sample frame: {} of {} records".format(len(frame), len(rows)))
        stratum_of = {}
        if args.strata_year_field:
            picked, stratum_of, strata_sizes = sample_year_strata(
                rows, frame, args.strata_year_field, args.strata_year_cutoff,
                args.sample, seed)
            got = {k: sum(1 for i in picked if stratum_of[i] == k) for k in YEAR_STRATA}
            print("date strata {} (cutoff {}): frame {} sampled {}".format(
                args.strata_year_field, args.strata_year_cutoff, strata_sizes, got))
        else:
            pts = [(rows[i]["lon"], rows[i]["lat"]) for i in frame]
            local = (uniform_sample(len(frame), args.sample, seed)
                     if args.sampling == "uniform"
                     else stratified_sample(pts, args.sample, seed, grid=args.grid))
            picked = [frame[i] for i in local]
        sites = [{"id": str(rows[i].get(args.id_field, i)),
                  "lon": rows[i]["lon"], "lat": rows[i]["lat"],
                  "stratum": stratum_of.get(i)} for i in picked]
        sites_source = {"mode": "sample", "seed": seed, "sampling": args.sampling,
                        "n_records": len(sites)}
        sites_desc = "{} sampled ({}, seed {})".format(len(sites), args.sampling, seed)
        sampling_desc = args.sampling
    if args.limit:
        sites = sites[:args.limit]

    review_dir = os.path.join(args.out_dir, "review_{}-gsv".format(args.city))
    search_dir = os.path.join(review_dir, "gsv_cache", "search")
    meta_dir = os.path.join(review_dir, "gsv_cache", "meta")
    pano_dir = os.path.join(review_dir, "gsv_cache", "panos")
    for d in (search_dir, meta_dir, pano_dir):
        os.makedirs(d, exist_ok=True)

    # ---- per-site: search, pick, fetch, render ---------------------------- #
    chips, verdicts, site_status = [], [], []
    for k, site in enumerate(sites):
        rid = site["id"]
        row = by_id[rid]
        record_ym = None
        if args.date_field:
            ym = parse_ym(row.get(args.date_field))
            record_ym = None if (ym is None or ym in SENTINEL_YMS) else ym

        def _status(status, detail=None):
            site_status.append({"id": rid, "status": status, "detail": detail})
            print("  [{:>3}/{}] {} {}{}".format(
                k + 1, len(sites), rid, status,
                " ({})".format(detail) if detail else ""))

        try:
            cands = cached_search(site["lat"], site["lon"], search_dir,
                                  sleep_s=args.sleep)
        except Exception as exc:                                  # noqa: BLE001
            _status("search_failed", "{}: {}".format(type(exc).__name__, exc))
            continue

        chosen, pick_status, pick_stats = choose_pano(
            cands, site["lat"], site["lon"], record_ym, band_m=band)
        if chosen is None:
            _status(pick_status, json.dumps(pick_stats))
            continue

        try:
            equi, absent_reason = fetch_panorama_cached(
                chosen["pano_id"], pano_dir, refetch_absent=args.refetch_absent)
        except Exception as exc:                                  # noqa: BLE001
            _status("fetch_error", "{}: {}".format(type(exc).__name__, exc))
            continue
        if equi is None:
            # The pano id came from a search result, so this is suspicious —
            # likely transient, and --refetch-absent will re-try it.
            _status("pano_fetch_absent", "{} {}".format(chosen["pano_id"], absent_reason))
            continue

        try:
            heading = cached_pano_heading(chosen["pano_id"], meta_dir)
        except Exception as exc:                                  # noqa: BLE001
            _status("metadata_failed", "{}: {}".format(type(exc).__name__, exc))
            continue

        # The production geometry, verbatim (download_dataset.py:226-231):
        # bearing pano->record minus the pano azimuth, rendered at pitch -30.
        pano_angle = heading_to_azimuth(heading)
        az_gov = fwd_azimuth_deg(chosen["lat"], chosen["lon"],
                                 site["lat"], site["lon"])
        theta = wrap_deg(az_gov - pano_angle)   # renderer is periodic; wrapped
                                                # value is identical to
                                                # production's unwrapped one
        try:
            persp_pil, ctx_pil = render_views(equi, theta)
        except Exception as exc:                                  # noqa: BLE001
            _status("render_failed", "{}: {}".format(type(exc).__name__, exc))
            continue

        neighbors, n_out = neighbour_offsets(
            rows, site["lon"], site["lat"], chosen["lon"], chosen["lat"],
            az_gov, rid, id_field=args.id_field)

        base = make_base_record(site, chosen, heading, az_gov, theta,
                                n_candidates=pick_stats["n_panos"])
        chips.append(dict(
            base,
            uri=to_data_uri(persp_pil, quality=args.jpeg_quality),
            ctx_uri=to_data_uri(ctx_pil, quality=args.ctx_quality),
            neighbors=neighbors, n_neighbors_out_of_view=n_out,
        ))
        verdicts.append(dict(
            base,
            offset_deg=None, click_px=None, unreadable=False,
            unreadable_reason=None, no_ramp=False, note="",
        ))
        _status("rendered", "pano {} {} at {} m".format(
            chosen["pano_id"], chosen.get("date"), chosen["range_m"]))

    # ---- manifest, with the drop accounting ------------------------------- #
    counts = {}
    for s in site_status:
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    manifest = {
        "city": args.city, "instrument": "street-level (#103)",
        "inventory": os.path.basename(args.inventory),
        "seed": seed, "sampling": sampling_desc,
        "sites_source": sites_source,
        "strata": None if strata_sizes is None else {
            "field": args.strata_year_field, "cutoff": args.strata_year_cutoff,
            "frame_sizes": strata_sizes},
        "date_field": args.date_field,
        "pano_pick": {"band_m": list(band),
                      "rule": "min range, tie-break newest capture",
                      "temporal": "capture ym >= record ym; undated record "
                                  "accepts any pano"},
        "projection": {"persp_px": PERSP_PX, "fov_deg": FOV_DEG,
                       "pitch_deg": PITCH_DEG,
                       "strip_cols": [CROP_LO, CROP_HI],
                       "strip_left_deg": STRIP_LEFT_DEG,
                       "strip_right_deg": STRIP_RIGHT_DEG,
                       "crop_half_angle_deg": crop_half_angle_deg()},
        "sign_convention": "positive = ramp clockwise of the government "
                           "bearing = right of the crosshair (matches "
                           "stage1_bearing_residual.py, §5j)",
        "neighbour_radius_m": INCLUSION_DISTANCE_M,
        "jpeg_quality": args.jpeg_quality, "ctx_quality": args.ctx_quality,
        # Per-record terminal statuses — the drop accounting. A count is a
        # claim about the fetcher until it can be checked record by record.
        "site_status": site_status,
        "status_counts": counts,
        "rubric": RUBRIC,
        "sheet_build": sheet_build_id(),
        "reviewer": None, "reviewed_on": None, "confidence": None,
    }

    verdict_path = os.path.join(review_dir, "verdicts.json")
    with open(verdict_path, "w", encoding="utf-8") as fh:
        json.dump(dict(manifest, records=verdicts), fh, indent=2)
        fh.write("\n")

    sheet_meta = {"city": args.city, "seed": seed,
                  "inventory": os.path.basename(args.inventory),
                  "sites_desc": sites_desc}
    sheet_path = os.path.join(review_dir, "review_sheet.html")
    with open(sheet_path, "w", encoding="utf-8") as fh:
        fh.write(build_sheet(sheet_meta, chips, manifest))

    print("\n{} of {} sites rendered; statuses: {}".format(
        len(chips), len(sites), counts))
    print("sheet build {}".format(sheet_build_id()))
    print("wrote {}".format(sheet_path))
    print("wrote {}".format(verdict_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
