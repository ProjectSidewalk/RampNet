"""Build the stratified, seeded curb-ramp tag review list (RampNet 2.0 plan item 3, #86).

The list is what Jon, and then a second rater, review on production under the draft rubric
in ``docs/tag_rubric_draft.md``; the protocol is ``docs/tag_review_protocol.md``. Per-tag
agreement between the two passes is the human ceiling every tag model is measured against.

Input is the Project Sidewalk API cache written by PR #175's
``scripts/analysis/ps_supervision_audit.py fetch`` (``analysis_out/ps_audit/raw/``,
gitignored): per deployment ``<city>__rawLabels__CurbRamp.csv``,
``<city>__validations__CurbRamp.csv``, ``<city>__labelTags.json`` and the
``fetch_manifest.json`` that records each deployment's host and each file's sha256.
Production is a live database, so a fresh fetch gives a different pool: **the committed
``benchmark/tag_review/review_list.csv`` is the artifact of record**, and this script
reproduces it byte for byte only from a cache whose hashes match
``review_list.meta.json``.

    # the list (defaults = the committed one)
    python scripts/analysis/tag_review_list.py build --cache analysis_out/ps_audit/raw

    # the kappa-precision table, and the positives each list size gives over 10 seeds
    python scripts/analysis/tag_review_list.py power
    python scripts/analysis/tag_review_list.py size --cache analysis_out/ps_audit/raw

    # the check that pano_y is world-frame (why the depression has no camera_pitch term)
    python scripts/analysis/tag_review_list.py pitch-check --cache analysis_out/ps_audit/raw

Strata (full definitions in the rubric doc, section "The review list"):

- **tag state**, precedence top down: ``affirmed_empty`` (no tags now, and a tag-review
  pass affirmed it: an *Agree* vote, or an edit, from ``ExpertValidate`` or the ASSETS'24
  ``ExternalTagValidationASSETS2024`` pass; an Unsure or Disagree vote alone does not
  count), ``tagged_trusted`` (tagged, placed by a trusted account), ``tagged`` (tagged by
  anyone else), ``untagged`` (no tags, never tag-reviewed; *not* a negative);
- **distance band** from the label's depression angle below the horizon,
  ``(pano_y / pano_height - 0.5) * 180`` degrees, turned into a flat-ground distance with a
  fixed 2.5 m camera height: near < 8 m, mid 8-15 m, far >= 15 m. There is **no**
  ``camera_pitch`` term: Project Sidewalk writes ``pano_y`` from the label's world-frame
  pitch (``util.pano.povToPanoCoord`` in ``panoUtilities.js``, called from ``Label.js`` with a
  POV that ``GsvViewer`` / ``PannellumViewer.getPov`` already report in world pitch), so the
  row is already measured from the horizon. This is the same convention, and the same
  2.5 m constant, as ``crop_window_eval.py`` and ``size_analysis.py``. (The first draft of
  this script subtracted ``camera_pitch`` a second time; 67 of 500 items were in the wrong
  band. Measured on crop-era labels placed at the canvas centre, the residual
  ``(0.5 - pano_y / pano_height) * 180 - pov_pitch`` has slope about 0 against
  ``camera_pitch``, not the -1 a double correction needs.);
- **city**, with equal allocation across eligible cities (capped by what each city has), so
  Chicago's 67k crop-era labels do not outvote a deployment with a few hundred.

Within a tagged stratum a label is drawn with weight ``(1 / f) ** rare_power`` (default
power 1.5), where ``f`` is the frequency of the label's rarest tag *within that tag state's
candidates*, so the rare tags reach enough positives for a per-tag kappa; within a (state,
city) cell the draw rotates through the distance bands. At most one label per (city, pano)
and none within ``--min-sep-m`` of an already-drawn label in the same city, so two labels of
one physical ramp (plan §2.5) are not two items. Item order is a seeded shuffle, so a rater
never sees a block of one stratum.

Flags, never filters: per-rater prior contact (``prior_contact_<rater>``: the rater placed,
validated or edited the label before the list was built) and proximity to a
validation-study label (``vstudy_same_pano``, ``vstudy_within_10m``). The validation-study
deployment is excluded *by deployment*; a physical ramp it also covers can still be listed
through another deployment's label, and these columns show where.
"""
import argparse
import csv
import datetime as dt
import glob
import io
import json
import math
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from rampnet import tag_review as tr  # noqa: E402

ANALYSIS_OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
DEFAULT_CACHE = os.path.join(ANALYSIS_OUT, "ps_audit", "raw")
DEFAULT_OUT = os.path.join(REPO, "benchmark", "tag_review", "review_list.csv")

#: The SidewalkAI account (model output, not supervision). Same id as PR #175's audit.
SIDEWALK_AI_USER = "51b0b927-3c8a-45b2-93de-bd878d1e5cf4"
#: Owner accounts = the default trusted placers. Ids are shared across deployments.
OWNERS = {
    "549187e0-82c9-4014-a48d-31f18083d575": "jonfroehlich",
    "18b26a38-24ab-402d-a64e-158fc0bb8a8a": "mikey",
}
#: Validation sources that looked at the tag set (every other source is a position vote).
TAG_REVIEW_SOURCES = ("ExpertValidate", "ExternalTagValidationASSETS2024")
#: The validation-study deployment holds the 94 disputed "missing tactile warning" ramps
#: whose adjudication is deferred (Jon, #86, 2026-09-22); none of its labels enter the list.
DEFAULT_EXCLUDE_CITIES = ("validation-study",)
#: Production stores a browser crop for every label placed on or after this date.
CROP_DATE = "2023-10-12"
#: The deployment whose labels are flagged by proximity (S4 of the PR #176 review).
VSTUDY_CITY = "validation-study"
VSTUDY_RADIUS_M = 10.0
#: Deployments outside the US (for the composition count only; nothing is filtered on it).
NON_US = frozenset({
    "amsterdam", "auckland", "bayonne-fr", "burnaby", "cdmx", "chandigarh-india", "cuenca",
    "kaohsiung", "keelung", "la-piedad", "la-piedad-old", "new-taipei", "rancagua-chile",
    "santiago-chile", "sao-paulo-brazil", "spgg", "taichung", "tainan", "taipei",
    "winterthur-infra3d", "zurich", "zurich-infra3d",
})

STATES = ("affirmed_empty", "tagged_trusted", "tagged", "untagged")
DEFAULT_SHARES = "affirmed_empty=0.20,tagged_trusted=0.15,tagged=0.35,untagged=0.30"
BANDS = ("near", "mid", "far")
CAMERA_HEIGHT_M = 2.5
BAND_EDGES_M = (8.0, 15.0)

LIST_COLUMNS = (
    "item_id", "city", "deployment_visibility", "label_id", "label_uid", "pano_id", "tag_state",
    "distance_band", "depression_deg", "est_distance_m", "camera_pitch", "label_heading_deg",
    "label_pitch_deg", "placed_at", "image_capture_date", "tags_at_list", "severity_at_list",
    "applicable_tags", "tag_reviewed_by", "placed_by_rater", "prior_contact_jonfroehlich",
    "prior_contact_mikey", "vstudy_same_pano", "vstudy_within_10m", "editor_url", "labelmap_url",
    "gsv_url",
)
#: Prior-contact kinds, in the order they are written (``;``-joined, blank = none).
CONTACT_KINDS = ("placed", "validated", "edited")

RAW_COLS = ["label_id", "user_id", "pano_id", "pano_source", "severity", "tags", "time_created",
            "correct", "pano_y", "pano_height", "camera_pitch", "latitude", "longitude",
            "image_capture_date", "pano_x", "pano_width", "camera_heading"]


# ----------------------------------------------------------------------------- geometry

def depression_deg(pano_y, pano_height):
    """Degrees below the horizon of a label's ``pano_y``.

    ``pano_y`` is already world-frame (see the module docstring), so no camera-pitch term."""
    return (np.asarray(pano_y, dtype=float) / np.asarray(pano_height, dtype=float) - 0.5) * 180.0


def flat_ground_distance_m(dep_deg, camera_height_m=CAMERA_HEIGHT_M):
    """Flat-ground distance for a depression angle; ``inf`` at or above the horizon."""
    dep = np.asarray(dep_deg, dtype=float)
    with np.errstate(divide="ignore"):
        d = camera_height_m / np.tan(np.radians(dep))
    return np.where(dep > 0, d, np.inf)


def distance_band(dist_m, edges=BAND_EDGES_M):
    d = np.asarray(dist_m, dtype=float)
    return np.where(d < edges[0], "near", np.where(d < edges[1], "mid", "far"))


def label_view(pano_x, pano_width, camera_heading, dep_deg):
    """(heading, pitch) in degrees that centre a viewer on the label.

    Column 0 of a GSV equirectangular pano faces ``camera_heading - 180``
    (``povToPanoCoord``); the pitch is minus the depression, a world-frame pitch, which is
    what the Google Maps ``pitch`` URL parameter takes."""
    heading = (np.asarray(pano_x, dtype=float) / np.asarray(pano_width, dtype=float) * 360.0
               + np.nan_to_num(np.asarray(camera_heading, dtype=float)) - 180.0) % 360.0
    return heading, -np.asarray(dep_deg, dtype=float)


def gsv_url(pano_id, heading, pitch, fov=60):
    """A Google Maps viewer link pinned to this pano id, centred on the label. Shows the
    imagery only: no Project Sidewalk tags, so it is the blind rater's view (protocol)."""
    return (f"https://www.google.com/maps/@?api=1&map_action=pano&pano={pano_id}"
            f"&heading={heading:.1f}&pitch={pitch:.1f}&fov={fov}")


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ----------------------------------------------------------------------------- load

def load_manifest(cache):
    path = os.path.join(cache, "fetch_manifest.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run PR #175's `ps_supervision_audit.py fetch` first")
    return tr.read_json(path)


def city_vocab(cache, city):
    """CurbRamp tags this deployment offers today (its ``/v3/api/labelTags`` view)."""
    with open(os.path.join(cache, f"{city}__labelTags.json"), encoding="utf-8") as fh:
        tags = json.load(fh)["label_tags"]
    return sorted({t["tag"] for t in tags if t["label_type"] == "CurbRamp"})


def load_pool(cache, cities):
    frames = []
    for city in cities:
        d = pd.read_csv(os.path.join(cache, f"{city}__rawLabels__CurbRamp.csv"), usecols=RAW_COLS,
                        dtype={"correct": "object", "pano_id": "object", "image_capture_date": "object"})
        d["city"] = city
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def _read_validations(cache, city):
    path = os.path.join(cache, f"{city}__validations__CurbRamp.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["label_id", "source", "validation_result", "user_id"])
    return pd.read_csv(path, usecols=["label_id", "source", "validation_result", "user_id"])


def _read_edits(cache, city):
    path = os.path.join(cache, f"{city}__labelEdits.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["label_id", "source", "user_id"])
    return pd.read_csv(path, usecols=["label_id", "source", "user_id"])


def load_tag_reviews(cache, cities):
    """(city, label_id) -> sorted tag-review sources that *affirmed* the label's tag set.

    A tag-review source counts only through an ``Agree`` vote or an edit (the rater changed
    the tags or severity there). An Unsure or Disagree vote from the same source is not an
    affirmation: on the first draft three ``affirmed_empty`` items rested on exactly that."""
    out = {}
    for city in cities:
        v = _read_validations(cache, city)
        v = v[v.source.isin(TAG_REVIEW_SOURCES) & (v.validation_result.astype(str) == "Agree")]
        e = _read_edits(cache, city)
        e = e[e.source.isin(TAG_REVIEW_SOURCES)]
        both = pd.concat([v[["label_id", "source"]], e[["label_id", "source"]]], ignore_index=True)
        for lid, grp in both.groupby("label_id"):
            out[(city, int(lid))] = sorted(set(grp.source))
    return out


def load_prior_contact(cache, cities, raters):
    """(city, label_id) -> {rater name: {"validated", "edited"}} from the cached votes and edits.

    ``placed`` is added from the label's own ``user_id`` by the caller."""
    out = {}
    for city in cities:
        for kind, df in (("validated", _read_validations(cache, city)), ("edited", _read_edits(cache, city))):
            df = df[df.user_id.isin(raters)]
            for lid, uid in zip(df.label_id, df.user_id):
                out.setdefault((city, int(lid)), {}).setdefault(raters[uid], set()).add(kind)
    return out


def load_vstudy(cache):
    """Validation-study label positions and pano ids, or an empty frame if not cached."""
    path = os.path.join(cache, f"{VSTUDY_CITY}__rawLabels__CurbRamp.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["label_id", "pano_id", "latitude", "longitude"])
    v = pd.read_csv(path, usecols=["label_id", "pano_id", "latitude", "longitude"], dtype={"pano_id": "object"})
    return v.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


def vstudy_flags(lat, lon, pano_id, vstudy, radius_m=VSTUDY_RADIUS_M):
    """(same_pano, [validation-study label uids within ``radius_m``]) for one label."""
    same = bool(len(vstudy)) and bool((vstudy.pano_id == pano_id).any())
    if not len(vstudy):
        return same, []
    # cheap prefilter on a lat/lon box, then haversine
    dlat = radius_m / 111_000.0 * 1.5
    dlon = dlat / max(math.cos(math.radians(lat)), 1e-6)
    box = vstudy[(vstudy.latitude.sub(lat).abs() < dlat) & (vstudy.longitude.sub(lon).abs() < dlon)]
    near = sorted((haversine_m(lat, lon, a, b), int(l)) for l, a, b in zip(box.label_id, box.latitude, box.longitude))
    return same, [f"{VSTUDY_CITY}:{l}" for d, l in near if d < radius_m]


def read_trusted(path):
    """User ids from a ``role|username|user_id`` file (PR #175's ``--trusted-users`` format)."""
    ids = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("|")
            if len(parts) == 3 and parts[0] in ("Owner", "Administrator", "Researcher"):
                ids[parts[2]] = parts[1]
    return ids


def parse_shares(s):
    shares = {}
    for part in s.split(","):
        k, v = part.split("=")
        if k not in STATES:
            raise SystemExit(f"unknown tag state {k!r}; states are {STATES}")
        shares[k] = float(v)
    total = sum(shares.values())
    return {k: shares.get(k, 0.0) / total for k in STATES}


# ----------------------------------------------------------------------------- candidates

def build_candidates(cache, *, exclude_cities, require_tags, crop_date, sources, trusted,
                     min_city_pool, raters):
    manifest = load_manifest(cache)
    hosts = {c: h["url"] for c, h in manifest.get("hosts", {}).items()}
    visibility = {c: h.get("visibility", "") for c, h in manifest.get("hosts", {}).items()}
    cities_all = sorted(os.path.basename(p).split("__")[0]
                        for p in glob.glob(os.path.join(cache, "*__rawLabels__CurbRamp.csv")))
    dropped = {}
    cities = []
    vocab = {}
    for c in cities_all:
        if c in exclude_cities:
            dropped[c] = "excluded by --exclude-cities"
            continue
        v = city_vocab(cache, c)
        missing = sorted(set(require_tags) - set(v))
        if missing:
            dropped[c] = "vocabulary lacks " + ", ".join(missing)
            continue
        if c not in hosts:
            dropped[c] = "no host in fetch_manifest.json"
            continue
        vocab[c] = [t for t in v if t not in tr.RETIRED_TAGS]
        cities.append(c)

    d = load_pool(cache, cities)
    n0 = len(d)
    funnel = [("CurbRamp labels in the vocabulary-eligible cities, SidewalkAI included", n0)]
    d = d[d.user_id != SIDEWALK_AI_USER]
    funnel.append(("minus SidewalkAI", len(d)))
    d["t"] = pd.to_datetime(d.time_created, utc=True, format="ISO8601")
    d = d[d.t >= pd.Timestamp(crop_date, tz="UTC")]
    funnel.append((f"placed on/after {crop_date} (has a production crop)", len(d)))
    d = d[d.correct.fillna("").str.lower() != "false"]
    funnel.append(("minus crowd-validated incorrect (correct == false)", len(d)))
    d = d[d.pano_source.isin(sources)]
    funnel.append((f"pano source in {sorted(sources)}", len(d)))
    d = d[(d.pano_height > 0) & d.pano_y.notna() & d.latitude.notna()]
    funnel.append(("with pano geometry and a location", len(d)))

    d = d.copy()
    d["tag_list"] = d.tags.map(tr.parse_tag_list)
    d["n_tags"] = d.tag_list.map(len)
    reviews = load_tag_reviews(cache, cities)
    d["tag_reviewed_by"] = [";".join(reviews.get((c, int(l)), [])) for c, l in zip(d.city, d.label_id)]
    trusted_ids = set(trusted)
    d["state"] = np.select(
        [(d.n_tags == 0) & (d.tag_reviewed_by != ""),
         (d.n_tags > 0) & d.user_id.isin(trusted_ids),
         d.n_tags > 0],
        ["affirmed_empty", "tagged_trusted", "tagged"], default="untagged")
    d["dep"] = depression_deg(d.pano_y, d.pano_height)
    d["dist"] = flat_ground_distance_m(d.dep)
    d["band"] = distance_band(d.dist)
    d["lab_heading"], d["lab_pitch"] = label_view(d.pano_x, d.pano_width, d.camera_heading, d.dep)
    d["placed_by_rater"] = d.user_id.map(lambda u: raters.get(u, ""))
    contact = load_prior_contact(cache, cities, raters)
    for name in sorted(set(raters.values())):
        d[f"prior_contact_{name}"] = [
            ";".join(k for k in CONTACT_KINDS
                     if (k == "placed" and raters.get(u) == name)
                     or k in contact.get((c, int(l)), {}).get(name, ()))
            for c, l, u in zip(d.city, d.label_id, d.user_id)]

    pool = d.groupby("city").size().reindex(cities, fill_value=0)
    small = sorted(pool[pool < min_city_pool].index)
    for c in small:
        dropped[c] = f"fewer than {min_city_pool} eligible labels ({int(pool[c])})"
    d = d[~d.city.isin(small)]
    funnel.append((f"in cities with >= {min_city_pool} eligible labels", len(d)))
    d = d.sort_values(["city", "label_id"]).reset_index(drop=True)
    vs = load_vstudy(cache)
    vs_panos = set(vs.pano_id.dropna())
    pool_stats = {
        "distance_tertiles_m": [round(float(q), 1) for q in np.quantile(d.dist[np.isfinite(d.dist)], [1 / 3, 2 / 3])]
        if len(d) else [],
        "validation_study_panos_shared_with_pool": int(len(vs_panos & set(d.pano_id.dropna()))),
        "validation_study_shared_panos_by_city": {
            c: int(n) for c, n in d[d.pano_id.isin(vs_panos)].drop_duplicates("pano_id").groupby("city").size().items()},
    }
    return d, dict(hosts=hosts, visibility=visibility, vocab=vocab, dropped=dropped, funnel=funnel,
                   vstudy=vs, pool_stats=pool_stats)


# ----------------------------------------------------------------------------- allocation

def largest_remainder(total, shares):
    raw = {k: total * s for k, s in shares.items()}
    out = {k: int(math.floor(v)) for k, v in raw.items()}
    left = total - sum(out.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[:left]:
        out[k] += 1
    return out


def water_fill(total, avail, rng):
    """Split ``total`` as equally as possible over keys, never exceeding ``avail[k]``."""
    alloc = {k: 0 for k in avail}
    active = sorted(k for k in avail if avail[k] > 0)
    remaining = total
    while remaining > 0 and active:
        share = remaining // len(active)
        if share == 0:
            for k in list(rng.permutation(active))[:remaining]:
                alloc[k] += 1
            remaining = 0
            break
        for k in active:
            g = min(share, avail[k] - alloc[k])
            alloc[k] += g
            remaining -= g
        active = [k for k in active if alloc[k] < avail[k]]
    return alloc


def rare_tag_weights(frame):
    """1 / frequency (within ``frame``, i.e. one tag state) of each label's rarest tag; 1 for
    untagged labels. The draw raises this to ``rare_power``."""
    freq = {}
    for tags in frame.tag_list:
        for t in tags:
            freq[t] = freq.get(t, 0) + 1
    return np.array([1.0 / min(freq[t] for t in tags) if tags else 1.0 for tags in frame.tag_list])


def draw(cands, n_total, shares, seed, min_sep_m, rare_power=1.5):
    """The stratified draw. Returns (selected index list, per-cell shortfall report)."""
    rng = np.random.default_rng(seed)
    targets = largest_remainder(n_total, shares)
    taken, taken_panos, taken_pts = [], set(), {}
    shortfall = []

    def ok(i):
        r = cands.loc[i]
        if (r.city, r.pano_id) in taken_panos:
            return False
        for lat, lon in taken_pts.get(r.city, []):
            if haversine_m(r.latitude, r.longitude, lat, lon) < min_sep_m:
                return False
        return True

    def take(i):
        r = cands.loc[i]
        taken.append(i)
        taken_panos.add((r.city, r.pano_id))
        taken_pts.setdefault(r.city, []).append((r.latitude, r.longitude))

    # Rarest state first, so its few candidates are not blocked by a commoner state's picks.
    city_order = {c: k for k, c in enumerate(sorted(cands.city.unique()))}
    for state in STATES:
        sub = cands[cands.state == state]
        if sub.empty:
            if targets[state]:
                shortfall.append((state, "*", targets[state]))
            continue
        w = rare_tag_weights(sub) ** rare_power if state.startswith("tagged") else np.ones(len(sub))
        # Efraimidis-Spirakis keys: ascending -log(u)/w is a weighted sample w/o replacement.
        keys = -np.log(rng.random(len(sub))) / w
        sub = sub.assign(_key=keys).sort_values(["_key", "city", "label_id"])
        queues = {c: {b: list(g.index[g.band == b]) for b in BANDS} for c, g in sub.groupby("city")}
        # Each city's band rotation starts at a different band, so the remainder of a small
        # per-city quota does not always land on "near".
        turn = {c: city_order[c] % len(BANDS) for c in queues}
        remaining = targets[state]
        # Rounds: a city whose candidates are blocked by the de-duplication hands its unmet
        # quota back, and the next round re-splits it over the cities that still have some.
        while remaining > 0:
            avail = {c: sum(len(q) for q in qs.values()) for c, qs in queues.items()}
            if not any(avail.values()):
                break
            alloc = water_fill(remaining, avail, rng)
            got_round = 0
            for city in sorted(alloc):
                got = 0
                while got < alloc[city] and any(queues[city].values()):
                    b = BANDS[turn[city]]
                    turn[city] = (turn[city] + 1) % len(BANDS)
                    while queues[city][b]:
                        i = queues[city][b].pop(0)
                        if ok(i):
                            take(i)
                            got += 1
                            break
                got_round += got
            remaining -= got_round
        if remaining > 0:
            shortfall.append((state, "*", remaining))
    return taken, targets, shortfall


# ----------------------------------------------------------------------------- output

def _fmt(x, nd=3):
    return "" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def list_rows(sel, hosts, vocab, seed, visibility=None, vstudy=None):
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(sel))
    sel = sel.iloc[order].reset_index(drop=True)
    visibility = visibility or {}
    if vstudy is None:
        vstudy = pd.DataFrame(columns=["label_id", "pano_id", "latitude", "longitude"])
    rows = []
    for k, r in sel.iterrows():
        host = hosts[r.city]
        lid = int(r.label_id)
        sev = "" if pd.isna(r.severity) else str(int(r.severity))
        same_pano, near = vstudy_flags(float(r.latitude), float(r.longitude), r.pano_id, vstudy)
        rows.append({
            "item_id": f"tr{k + 1:04d}",
            "city": r.city,
            "deployment_visibility": visibility.get(r.city, ""),
            "label_id": lid,
            "label_uid": f"{r.city}:{lid}",
            "pano_id": r.pano_id,
            "tag_state": r.state,
            "distance_band": r.band,
            "depression_deg": _fmt(float(r.dep), 2),
            "est_distance_m": _fmt(float(r.dist), 1),
            "camera_pitch": _fmt(float(r.camera_pitch), 3) if pd.notna(r.camera_pitch) else "",
            "label_heading_deg": _fmt(float(r.lab_heading), 1),
            "label_pitch_deg": _fmt(float(r.lab_pitch), 1),
            "placed_at": r.t.strftime("%Y-%m-%d"),
            "image_capture_date": "" if pd.isna(r.image_capture_date) else str(r.image_capture_date),
            "tags_at_list": tr.join_tags(r.tag_list),
            "severity_at_list": sev,
            "applicable_tags": tr.join_tags(vocab[r.city]),
            "tag_reviewed_by": r.tag_reviewed_by,
            "placed_by_rater": r.placed_by_rater,
            "prior_contact_jonfroehlich": r.get("prior_contact_jonfroehlich", ""),
            "prior_contact_mikey": r.get("prior_contact_mikey", ""),
            "vstudy_same_pano": "true" if same_pano else "",
            "vstudy_within_10m": ";".join(near),
            "editor_url": f"{host}/gallery?labelType=CurbRamp&labelId={lid}",
            "labelmap_url": f"{host}/labelMap?labelId={lid}",
            "gsv_url": gsv_url(r.pano_id, float(r.lab_heading), float(r.lab_pitch)),
        })
    return rows


def write_csv(path, rows):
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(LIST_COLUMNS), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return tr.sha256_bytes(data)


def read_list(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def composition(rows):
    """Tables the doc and the PR quote: state x band, per city, per-tag list-time positives."""
    df = pd.DataFrame(rows)
    sxb = pd.crosstab(df.tag_state, df.distance_band).reindex(index=list(STATES), columns=list(BANDS),
                                                            fill_value=0)
    per_city = df.groupby("city").size().sort_values(ascending=False)
    tag_pos = {}
    for tags in df.tags_at_list:
        for t in tr.parse_tag_list(tags):
            tag_pos[t] = tag_pos.get(t, 0) + 1
    def col(name):
        return df[name] if name in df else pd.Series([""] * len(df))

    contact = {}
    for name in ("jonfroehlich", "mikey"):
        c = col(f"prior_contact_{name}").fillna("")
        contact[name] = {"any": int((c != "").sum()),
                         **{k: int(c.map(lambda s, k=k: k in s.split(";")).sum()) for k in CONTACT_KINDS}}
    any_contact = (col("prior_contact_jonfroehlich").fillna("") != "") | (col("prior_contact_mikey").fillna("") != "")
    aff = df[df.tag_state == "affirmed_empty"]
    # capture gap: label date minus imagery capture month (first of the month), in years
    cap = pd.to_datetime(col("image_capture_date"), errors="coerce", format="mixed")
    placed = pd.to_datetime(col("placed_at"), errors="coerce")
    gap = ((placed - cap).dt.days / 365.25).dropna()
    vis = col("deployment_visibility").fillna("")
    return {
        "n": len(df),
        "state_by_band": {s: {b: int(sxb.loc[s, b]) for b in BANDS} for s in STATES},
        "per_city": {c: int(n) for c, n in per_city.items()},
        "per_city_range": [int(per_city.min()), int(per_city.max())] if len(per_city) else [],
        "non_us_items": int(df.city.isin(NON_US).sum()),
        "non_us_cities": sorted(set(df.city) & NON_US),
        "tag_positives_at_list": dict(sorted(tag_pos.items())),
        "severity_at_list": {(k or "unrated"): int(v) for k, v in
                             col("severity_at_list").fillna("").value_counts().sort_index().items()},
        "placed_by_rater": {k or "(neither)": int(v) for k, v in df.placed_by_rater.value_counts().sort_index().items()},
        "prior_contact": {**contact, "either_rater": int(any_contact.sum()),
                          "affirmed_empty_validated_by_jonfroehlich": int(
                              col("prior_contact_jonfroehlich")[aff.index].fillna("").str.contains("validated").sum())},
        "tag_reviewed_before": int((df.tag_reviewed_by != "").sum()),
        "affirmed_empty_by_source": {k: int(v) for k, v in aff.tag_reviewed_by.value_counts().sort_index().items()},
        "tagged_items_tag_reviewed": int(((df.tag_state != "affirmed_empty") & (df.tag_reviewed_by != "")).sum()),
        "capture_gap_years": {"n": int(len(gap)), "median": round(float(gap.median()), 2) if len(gap) else None,
                              "at_least_5y": int((gap >= 5).sum())},
        "validation_study_same_pano": int((col("vstudy_same_pano") == "true").sum()),
        "validation_study_within_10m": int((col("vstudy_within_10m").fillna("") != "").sum()),
        "by_deployment_visibility": {k or "?": int(v) for k, v in vis.value_counts().sort_index().items()},
        "private_deployments_in_list": sorted(set(df.city[vis == "private"])),
    }


def cmd_build(args):
    raters = dict(OWNERS)
    trusted = dict(OWNERS)
    if args.trusted_users:
        trusted.update(read_trusted(args.trusted_users))
    cands, info = build_candidates(
        args.cache, exclude_cities=set(args.exclude_cities), require_tags=args.require_tags,
        crop_date=args.crop_date, sources=set(args.sources), trusted=trusted,
        min_city_pool=args.min_city_pool, raters=raters)
    shares = parse_shares(args.shares)
    idx, targets, shortfall = draw(cands, args.n, shares, args.seed, args.min_sep_m, args.rare_power)
    rows = list_rows(cands.loc[idx], info["hosts"], info["vocab"], args.seed,
                     visibility=info["visibility"], vstudy=info["vstudy"])
    digest = write_csv(args.out, rows)
    comp = composition(rows)
    manifest = load_manifest(args.cache)
    used = sorted({r["city"] for r in rows})
    inputs = {}
    keys = [f"{c}__{ep}" for c in sorted(info["vocab"])
            for ep in ("rawLabels__CurbRamp", "validations__CurbRamp", "labelEdits", "labelTags")]
    keys.append(f"{VSTUDY_CITY}__rawLabels__CurbRamp")
    for key in keys:
        m = manifest.get("files", {}).get(key, {})
        inputs[key] = {"sha256": m.get("sha256"), "fetched_at": m.get("fetched_at")}
    meta = {
        "generated_by": "scripts/analysis/tag_review_list.py build",
        "generated_on": dt.date.today().isoformat(),
        "list": {"path": "benchmark/tag_review/review_list.csv", "sha256": digest, "rows": len(rows)},
        "params": {"n": args.n, "seed": args.seed, "shares": shares, "crop_date": args.crop_date,
                   "sources": sorted(args.sources), "exclude_cities": sorted(args.exclude_cities),
                   "require_tags": sorted(args.require_tags), "min_city_pool": args.min_city_pool,
                   "min_sep_m": args.min_sep_m, "rare_power": args.rare_power, "camera_height_m": CAMERA_HEIGHT_M,
                   "band_edges_m": list(BAND_EDGES_M),
                   "depression": "(pano_y / pano_height - 0.5) * 180, pano_y world-frame, no camera_pitch term",
                   "vstudy_radius_m": VSTUDY_RADIUS_M,
                   "trusted_users": "owners only" if not args.trusted_users else "owners + --trusted-users file"},
        "funnel": [{"step": s, "labels": int(n)} for s, n in info["funnel"]],
        "pool_by_state": {s: int((cands.state == s).sum()) for s in STATES},
        "pool_by_city": {c: int(n) for c, n in cands.groupby("city").size().items()},
        "pool_by_state_and_band": {s: {b: int(((cands.state == s) & (cands.band == b)).sum()) for b in BANDS}
                                   for s in STATES},
        "pool_stats": info["pool_stats"],
        "cities_dropped": dict(sorted(info["dropped"].items())),
        "targets": targets,
        "shortfall": [{"state": s, "city": c, "missing": int(m)} for s, c, m in shortfall],
        "composition": comp,
        "cities_in_list": used,
        "inputs": inputs,
    }
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    tr.write_json(meta_path, meta)
    print(f"wrote {args.out} ({len(rows)} items, sha256 {digest})")
    print(f"wrote {meta_path}")
    print(json.dumps({"targets": targets, "shortfall": meta["shortfall"], **comp}, indent=1))


def cmd_size(args):
    """List-time positives per core tag for several list sizes and seeds (the size rationale)."""
    cands, _ = build_candidates(
        args.cache, exclude_cities=set(args.exclude_cities), require_tags=list(tr.CORE_TAGS),
        crop_date=CROP_DATE, sources={"gsv"}, trusted=dict(OWNERS), min_city_pool=args.min_city_pool,
        raters=dict(OWNERS))
    shares = parse_shares(DEFAULT_SHARES)
    print(f"minimum list-time positives over the {len(tr.CORE_TAGS)} core tags; "
          f"a cell is 'min (tag)'; seeds {args.seeds}")
    print("| n | " + " | ".join(f"seed {s}" for s in args.seeds) + " | seeds with every core tag >= 30 |")
    print("|---:|" + "---|" * len(args.seeds) + "---:|")
    for n in args.ns:
        cells, ok = [], 0
        for seed in args.seeds:
            idx, _, _ = draw(cands, n, shares, seed, args.min_sep_m, args.rare_power)
            pos = {t: 0 for t in tr.CORE_TAGS}
            for tags in cands.loc[idx].tag_list:
                for t in tags:
                    if t in pos:
                        pos[t] += 1
            low = min(pos, key=lambda t: (pos[t], t))
            cells.append(f"{pos[low]} ({low})")
            ok += all(v >= 30 for v in pos.values())
        print(f"| {n} | " + " | ".join(cells) + f" | {ok} of {len(args.seeds)} |")


def cmd_pitch_check(args):
    """Is ``pano_y`` world-frame? Slope of the residual against ``camera_pitch``.

    For crop-era GSV labels placed within ``--tol-px`` of the canvas centre, the labeller's
    recorded POV pitch is the label's world pitch. The residual
    ``(0.5 - pano_y / pano_height) * 180 - pov_pitch`` has slope 0 against ``camera_pitch``
    if ``pano_y`` is world-frame, and -1 if it is image-frame (so that ``camera_pitch`` would
    have to be subtracted)."""
    cols = ["time_created", "pano_source", "canvas_x", "canvas_y", "canvas_width", "canvas_height",
            "pitch", "pano_y", "pano_height", "camera_pitch"]
    print("| city | n | slope | camera_pitch sd |\n|---|---:|---:|---:|")
    for city in args.cities:
        d = pd.read_csv(os.path.join(args.cache, f"{city}__rawLabels__CurbRamp.csv"), usecols=cols)
        d = d[(d.pano_source == "gsv") & (d.time_created >= CROP_DATE)]
        d = d[((d.canvas_y - d.canvas_height / 2).abs() <= args.tol_px)].dropna()
        resid = (0.5 - d.pano_y / d.pano_height) * 180.0 - d.pitch
        slope = float(np.polyfit(d.camera_pitch, resid, 1)[0]) if len(d) > 2 else float("nan")
        print(f"| {city} | {len(d):,} | {slope:.3f} | {d.camera_pitch.std():.2f} |")


def cmd_power(args):
    print(f"95% CI half-width of Cohen's kappa (simulated, {args.sims} draws), true kappa {args.kappa:.2f}")
    print("| positives (n x prevalence) | " + " | ".join(f"n={n}" for n in args.ns) + " |")
    print("|---|" + "---|" * len(args.ns))
    for pos in args.positives:
        cells = []
        for n in args.ns:
            p = pos / n
            cells.append(_fmt(tr.kappa_ci_halfwidth(n, p, args.kappa, sims=args.sims, seed=args.seed), 2)
                         if 0 < p < 0.5 else "")
        print(f"| {pos} | " + " | ".join(cells) + " |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="draw the list")
    b.add_argument("--cache", default=DEFAULT_CACHE, help="PR #175 audit cache (analysis_out/ps_audit/raw)")
    b.add_argument("--out", default=DEFAULT_OUT)
    b.add_argument("--n", type=int, default=500, help="list size (see `power` for the rationale)")
    b.add_argument("--seed", type=int, default=86)
    b.add_argument("--shares", default=DEFAULT_SHARES, help="tag-state shares, state=frac,...")
    b.add_argument("--crop-date", default=CROP_DATE)
    b.add_argument("--sources", nargs="+", default=["gsv"], help="pano sources to keep")
    b.add_argument("--exclude-cities", nargs="*", default=list(DEFAULT_EXCLUDE_CITIES))
    b.add_argument("--require-tags", nargs="*", default=list(tr.CORE_TAGS),
                   help="a city must offer all of these tags to be eligible")
    b.add_argument("--rare-power", type=float, default=1.5,
                   help="tagged-state draw weight = (1 / freq of the label's rarest tag) ** this")
    b.add_argument("--min-city-pool", type=int, default=100)
    b.add_argument("--min-sep-m", type=float, default=10.0,
                   help="minimum distance between two listed labels in one city")
    b.add_argument("--trusted-users", default=None,
                   help="role|username|user_id file to widen the trusted placers (not committed)")
    b.set_defaults(func=cmd_build)
    z = sub.add_parser("size", help="min list-time positives per core tag vs list size and seed")
    z.add_argument("--cache", default=DEFAULT_CACHE)
    z.add_argument("--ns", type=int, nargs="+", default=[400, 450, 500])
    z.add_argument("--seeds", type=int, nargs="+", default=[86, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    z.add_argument("--exclude-cities", nargs="*", default=list(DEFAULT_EXCLUDE_CITIES))
    z.add_argument("--min-city-pool", type=int, default=100)
    z.add_argument("--min-sep-m", type=float, default=10.0)
    z.add_argument("--rare-power", type=float, default=1.5)
    z.set_defaults(func=cmd_size)
    g = sub.add_parser("pitch-check", help="is pano_y world-frame? residual slope vs camera_pitch")
    g.add_argument("--cache", default=DEFAULT_CACHE)
    g.add_argument("--cities", nargs="+", default=["seattle-wa", "chicago-il", "taipei"])
    g.add_argument("--tol-px", type=float, default=10.0)
    g.set_defaults(func=cmd_pitch_check)
    p = sub.add_parser("power", help="kappa CI half-width vs list size and positives")
    p.add_argument("--ns", type=int, nargs="+", default=[300, 500, 800])
    p.add_argument("--positives", type=int, nargs="+", default=[10, 20, 30, 50, 80])
    p.add_argument("--kappa", type=float, default=0.6)
    p.add_argument("--sims", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_power)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
