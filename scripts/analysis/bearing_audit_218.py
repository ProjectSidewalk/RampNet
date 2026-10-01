"""Audit of the bearing geometry behind #218's flat-photo hit test (PR #227).

Question: is the low flat-photo hit rate (canvas_level @ 0.30: 0.199 against a swap null
of 0.093) partly an INSTRUMENT error -- a wrong heading field, a sign / handedness flip, a
focal or FOV error, a wrong camera position -- rather than the model not firing? The
figures in #227 showed misses where a visible ramp sits 27-30 deg from the projected pool
ramp bearing and the model fires on it. If the geometry were off, real detections of
real ramps would sit at a systematic offset from the projected bearing, and some global
correction of the geometry would raise the hit rate above its chance floor.

``perspective_bearing_check_218.py`` already read the matched hits' offsets and the
excess within +-60 deg at 0.30. This script adds what that check does not cover:

1. **Correction scan.** The hit rate (scorer's own greedy claims) and its swap-null floor
   under a family of global geometry corrections applied to every detection: a heading
   shift (-45..+45 deg), a focal scale (0.6..1.6), a left-right mirror of the image x
   axis, the device compass instead of the SfM heading, the device GPS instead of the SfM
   position, and the direction of travel (from the sequence's device GPS track) instead of
   the SfM heading. If the committed geometry is right, the identity is at or near the
   maximum of "above chance" and no correction gains more than noise.
2. **Signed-offset excess over the whole frame**, real minus swap null, for every
   height-gated detection >= 0.30 and for every stored peak (>= 0.10, sub-threshold
   included), by camera, by pose source, and by where the ramp sits across the frame
   (left / centre / right third: a focal error moves the two side thirds opposite ways).
3. **Independent heading check.** The direction of travel from consecutive images of the
   same sequence, using the DEVICE GPS (``raw_lat``/``raw_lng``), so it shares nothing with
   Mapillary's SfM; compared with ``computed_compass_angle`` and the device
   ``compass_angle``, per camera.
4. **Pano control.** The 360 panos (bearing test 0.627 at 0.55) go through a different code
   path: the ramp's column is the labeler's ``x_proj``. (a) The flat path's ENU bearing
   code, applied to each pano's ``computed_geometry`` and ``computed_compass_angle``,
   must reproduce ``x_proj``; (b) the panos' signed-offset excess (detections >= 0.55
   minus the 90/180/270 deg rotation null) and their heading-shift scan.

Committed files only (dets_*.jsonl, census, images.csv, captures_R25.csv,
richmond_neighbourhood/records.jsonl): no images, no GPU, no network. About 10 minutes on a
desktop CPU.

    python scripts/analysis/bearing_audit_218.py              # writes bearing_audit/*.json
    python scripts/analysis/bearing_audit_218.py --quick      # 5 null draws, no bootstrap

Outputs: ``analysis_out/perspective_photos_218/bearing_audit/summary.json`` (every number in
docs/bearing_audit_218.md) and ``offsets.csv`` (the excess histograms).
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from rampnet import perspective as P  # noqa: E402
import perspective_photos_218 as PP  # noqa: E402

OUT = os.path.join(PP.OUT, "bearing_audit")
BIN = 5.0
SPAN = 90.0                  # offsets read within +-SPAN deg
N_BOOT = 2000
SHIFTS = list(range(-45, 46, 3))
FSCALES = (0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.4, 1.6)
TRAVEL_MIN_M = 2.0           # displacement needed for a travel bearing
TRAVEL_MAX_S = 20.0          # neighbour frames at most this far apart in time
TRAVEL_MAX_M = 60.0


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def travel_bearings(census):
    """{image_id: (bearing_raw, bearing_computed, span_m)}: direction of travel at each
    image from the previous and next frames of its sequence (any frame of the census,
    pano or flat: a sequence is one camera). Device GPS (raw) and SfM (computed) tracks
    are read separately; the raw one is independent of Mapillary's SfM."""
    by_seq = defaultdict(list)
    for r in census.values():
        if r["sequence"] and r["captured_at"]:
            by_seq[r["sequence"]].append(r)
    out = {}
    for seq, rs in by_seq.items():
        rs.sort(key=lambda r: int(r["captured_at"]))
        t = np.array([int(r["captured_at"]) / 1000.0 for r in rs])
        for k, r in enumerate(rs):
            res = []
            for la, ln in (("raw_lat", "raw_lng"), ("lat", "lng")):
                lat0, lng0 = float(r[la]), float(r[ln])

                def pos(j):
                    return P.enu_offset(lat0, lng0, float(rs[j][la]), float(rs[j][ln]))
                # nearest earlier and later frame that moved >= TRAVEL_MIN_M
                prev = nxt = None
                for j in range(k - 1, -1, -1):
                    if t[k] - t[j] > TRAVEL_MAX_S:
                        break
                    if math.hypot(*pos(j)) >= TRAVEL_MIN_M:
                        prev = j
                        break
                for j in range(k + 1, len(rs)):
                    if t[j] - t[k] > TRAVEL_MAX_S:
                        break
                    if math.hypot(*pos(j)) >= TRAVEL_MIN_M:
                        nxt = j
                        break
                if prev is None and nxt is None:
                    res.append(None)
                    continue
                a = pos(prev) if prev is not None else (0.0, 0.0)
                b = pos(nxt) if nxt is not None else (0.0, 0.0)
                de, dn = b[0] - a[0], b[1] - a[1]
                span = math.hypot(de, dn)
                if span < TRAVEL_MIN_M or span > TRAVEL_MAX_M:
                    res.append(None)
                    continue
                res.append((math.degrees(math.atan2(de, dn)) % 360, span))
            out[r["image_id"]] = res
    return out


def load(arm, n_null):
    rows = PP.read_csv(PP.IMAGES_CSV)
    by_id = {r["image_id"]: r for r in rows}
    census = {r["image_id"]: r for r in PP.read_csv(os.path.join(PP.CENSUS, "images.csv"))}
    ramps = PP.ramp_table()
    ramp_ll = {u: (a, b) for u, a, b in ramps}
    recs = PP.load_dets(arm)
    ids = sorted(recs)
    geo = {i: PP.image_geometry(by_id[i], recs[i]["width"], recs[i]["height"], ramps)
           for i in ids}
    pos = [i for i in ids if geo[i]["positive"]]
    # the scorer's own donors (same seed), so the identity row reproduces results.md
    donors = PP.swap_donors(pos, {i: by_id[i]["nearest_ramp"] for i in pos}, n_null)
    travel = travel_bearings(census)
    images = []
    for iid in pos:
        g, rec, row, c = geo[iid], recs[iid], by_id[iid], census[iid]
        cam = g["cam"]
        # where each in-view ramp lands across the frame (x / width), from 1.5 m
        for r in g["near"]:
            pc = g["R_wc"] @ np.array([r["e"], r["n"], -PP.VIEW_H])
            u, _ = P.project_cam(cam, pc) if pc[2] > 0 else (np.nan, np.nan)
            r["xfrac"] = float((u + 0.5) / cam.width)
        head = float(row["computed_compass_angle"])
        tr = travel.get(iid, [None, None])
        images.append({
            "id": iid, "g": g, "rec": rec, "row": row,
            "camera": f'{row["make"]} {row["model"]}'.strip(),
            "seq": row["sequence"],
            "level_pose": all(abs(x) < 1e-3
                              for x in P.heading_pitch_roll(g["R_wc"])[1:]),
            "d_compass": float(P.wrap_deg(float(c["compass_angle"]) - head)),
            "d_travel_raw": None if tr[0] is None else float(P.wrap_deg(tr[0][0] - head)),
            "d_travel_sfm": None if tr[1] is None else float(P.wrap_deg(tr[1][0] - head)),
            "raw_ll": (float(c["raw_lat"]), float(c["raw_lng"])),
            "donors": [d[iid] for d in donors]})
    return images, recs, ramp_ll


# --------------------------------------------------------------------------- #
# geometry under a correction
# --------------------------------------------------------------------------- #
def outvoted(im, tol):
    """Heading correction (deg) where both device-side headings agree with each other
    within ``tol`` and both differ from the SfM heading by at least ``tol``: two
    independent sources against one. 0 elsewhere, and 0 where the disagreement exceeds
    60 deg: a GoPro Max frame 180 deg from the direction of travel may be its rear lens,
    which the device compass would not see."""
    a, b = im["d_travel_raw"], im["d_compass"]
    if a is None or abs(a - b) > tol or abs(a) < tol or abs(b) < tol or abs(a) > 60:
        return 0.0
    return (a + b) / 2.0


def det_world_tf(dets, cam, R_wc, fscale=1.0, mirror=False):
    """``PP.det_world`` with the focal scaled by ``fscale`` and, with ``mirror``, the
    image x axis flipped (u -> w - 1 - u)."""
    if not dets:
        return np.zeros(0), np.zeros(0)
    c2 = P.Camera(cam.width, cam.height, cam.focal * fscale, cam.k1, cam.k2)
    u = np.array([d["u"] for d in dets], dtype=float)
    if mirror:
        u = cam.width - 1 - u
    ray = P.unproject_cam(c2, u, [d["v"] for d in dets])
    b, dep = P.ray_bearing_depression(ray @ R_wc)
    return b, dep


def near_from(im, ramp_ll, raw_pos):
    """Candidate ramps with bearing / range re-read from the device GPS position when
    ``raw_pos``; the in-view flags stay those of the scored geometry."""
    if not raw_pos:
        return im["g"]["near"]
    lat, lng = im["raw_ll"]
    out = []
    for r in im["g"]["near"]:
        a, b = ramp_ll[r["uid"]]
        e, n = P.enu_offset(lat, lng, a, b)
        out.append(dict(r, range=float(math.hypot(e, n)),
                        bearing=float(math.degrees(math.atan2(e, n)) % 360)))
    return out


class Scorer:
    """Real and swap-null pair hits under a correction, with a cache of the unprojected
    detections per (image, fscale, mirror)."""

    def __init__(self, images, recs, ramp_ll, thr):
        self.images, self.recs, self.ramp_ll, self.thr = images, recs, ramp_ll, thr
        self.cache = {}

    def world(self, im, donor, fscale, mirror):
        key = (im["id"], donor, fscale, mirror)
        if key not in self.cache:
            rec = im["rec"]
            if donor is None:
                dets = [d for d in rec["dets"] if d["score"] >= self.thr]
            else:
                dd = self.recs[donor]
                dets = [d for d in PP.transplant(dd["dets"], dd["width"], dd["height"],
                                                 rec["width"], rec["height"])
                        if d["score"] >= self.thr]
            b, dep = det_world_tf(dets, im["g"]["cam"], im["g"]["R_wc"], fscale, mirror)
            self.cache[key] = (dets, b, dep)
        return self.cache[key]

    def run(self, shift=0.0, fscale=1.0, mirror=False, per_image_shift=None,
            raw_pos=False, n_null=None):
        """Returns (pairs, real[], null_mean[]) over in-view pairs, scorer's claims."""
        pairs, real, nul = [], [], []
        for im in self.images:
            s = shift + (per_image_shift(im) if per_image_shift else 0.0)
            near = near_from(im, self.ramp_ll, raw_pos)
            dets, b, dep = self.world(im, None, fscale, mirror)
            cl = PP.claim_bearing(dets, b + s, dep, near, 0.0)
            donors = im["donors"][:n_null] if n_null else im["donors"]
            ncl = []
            for dn in donors:
                sd, sb, sdep = self.world(im, dn, fscale, mirror)
                ncl.append(PP.claim_bearing(sd, sb + s, sdep, near, 0.0))
            for r in im["g"]["near"]:
                if r["in_view"]:
                    pairs.append((im, r))
                    real.append(r["uid"] in cl)
                    nul.append(np.mean([r["uid"] in c for c in ncl]) if ncl else np.nan)
        return pairs, np.array(real, float), np.array(nul, float)


def cluster_weights(uids, n_boot=N_BOOT, seed=PP.SEED):
    """(n_boot, n_items) weights of a ramp-cluster bootstrap: each item's weight is the
    number of times its ramp was drawn (ramps drawn with replacement, as many as there
    are ramps)."""
    keys = sorted(set(uids))
    pos = {k: i for i, k in enumerate(keys)}
    item = np.array([pos[u] for u in uids])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), (n_boot, len(keys)))
    counts = np.stack([np.bincount(d, minlength=len(keys)) for d in draws])
    return counts[:, item].astype(float)


def ramp_boot(pairs, vals, n_boot=N_BOOT, seed=PP.SEED):
    """95% ramp-cluster bootstrap of the mean of ``vals`` (one per pair)."""
    w = cluster_weights([r["uid"] for _, r in pairs], n_boot, seed)
    m = (w @ np.asarray(vals, float)) / w.sum(axis=1)
    return [float(x) for x in np.percentile(m, [2.5, 97.5])]


def row_of(name, pairs, real, nul, boot=True):
    d = {"name": name, "n_pairs": len(real), "hits": int(real.sum()),
         "rate": float(real.mean()), "null": float(np.nanmean(nul)),
         "above": float(real.mean() - np.nanmean(nul))}
    if boot:
        d["above_ci"] = ramp_boot(pairs, real - nul)
    return d


# --------------------------------------------------------------------------- #
# full re-score with a corrected heading (the in-view set and the donors are redrawn)
# --------------------------------------------------------------------------- #
def rotate_heading(rotvec, delta_deg):
    """``computed_rotation`` (world-to-camera angle-axis) with the camera turned
    clockwise (seen from above) by ``delta_deg`` about the world vertical."""
    from scipy.spatial.transform import Rotation
    d = math.radians(delta_deg)
    c, s = math.cos(d), math.sin(d)
    Q = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])   # world ray -> turned ray
    R = P.world_to_cam_from_rotvec(rotvec) @ Q.T
    return [float(x) for x in Rotation.from_matrix(R).as_rotvec()]


def rescore(arm, thr, delta_fn, images_meta, n_null=PP.N_NULL, boot=True,
            exclude=None):
    """The scorer's bearing test re-run from scratch with every image's heading turned by
    ``delta_fn(meta)`` deg: in-view set, positives, donors (same seed) and claims all
    recomputed. ``exclude(meta)`` drops images. Returns the headline row."""
    rows = PP.read_csv(PP.IMAGES_CSV)
    ramps = PP.ramp_table()
    recs = PP.load_dets(arm)
    geo, by_id = {}, {}
    for row in rows:
        iid = row["image_id"]
        meta = images_meta.get(iid)
        if exclude is not None and meta is not None and exclude(meta):
            continue
        dl = delta_fn(meta) if meta is not None else 0.0
        if dl:
            row = dict(row)
            row["computed_rotation"] = json.dumps(
                rotate_heading(json.loads(row["computed_rotation"]), dl))
            row["computed_compass_angle"] = str(
                (float(row["computed_compass_angle"]) + dl) % 360)
        by_id[iid] = row
        geo[iid] = PP.image_geometry(row, recs[iid]["width"], recs[iid]["height"], ramps)
    pos = [i for i in sorted(geo) if geo[i]["positive"]]
    donors = PP.swap_donors(pos, {i: by_id[i]["nearest_ramp"] for i in pos}, n_null)
    pairs, real, nul = [], [], []
    for iid in pos:
        g, rec = geo[iid], recs[iid]
        b, dep, _ = PP.det_world(rec["dets"], g["cam"], g["R_wc"])
        cl = PP.claim_bearing(rec["dets"], b, dep, g["near"], thr)
        ncl = []
        for dr in donors:
            dd = recs[dr[iid]]
            sd = PP.transplant(dd["dets"], dd["width"], dd["height"], rec["width"],
                               rec["height"])
            sb, sdep, _ = PP.det_world(sd, g["cam"], g["R_wc"])
            ncl.append(PP.claim_bearing(sd, sb, sdep, g["near"], thr))
        for r in g["near"]:
            if r["in_view"]:
                pairs.append(({"id": iid}, r))
                real.append(r["uid"] in cl)
                nul.append(np.mean([r["uid"] in c for c in ncl]))
    real, nul = np.array(real, float), np.array(nul, float)
    d = row_of("", pairs, real, nul, boot)
    d.update({"n_positive": len(pos), "n_ramps": len({r["uid"] for _, r in pairs}),
              "n_changed": sum(1 for m in images_meta.values() if delta_fn(m))})
    if boot:
        d["rate_ci"] = ramp_boot(pairs, real)
    return d


# --------------------------------------------------------------------------- #
# offsets
# --------------------------------------------------------------------------- #
def gated_offsets(b, dep, r):
    """Signed offsets (deg, + = detection clockwise of the ramp) of the detections that
    pass the bearing test's height gate at ramp ``r``'s range."""
    if len(b) == 0:
        return np.zeros(0)
    h = r["range"] * np.tan(np.radians(np.clip(dep, 1e-6, 89.0)))
    ok = (dep > 0) & (h >= PP.H_MIN) & (h <= PP.H_MAX)
    o = P.wrap_deg(b - r["bearing"])
    return o[ok & (np.abs(o) <= SPAN)]


def offset_sets(sc):
    """Per in-view pair: real offsets and, per null draw, the donor offsets."""
    out = []
    for im in sc.images:
        _, b, dep = sc.world(im, None, 1.0, False)
        nulls = [sc.world(im, dn, 1.0, False) for dn in im["donors"]]
        for r in im["g"]["near"]:
            if r["in_view"]:
                out.append({"im": im, "r": r, "o": gated_offsets(b, dep, r),
                            "on": [gated_offsets(nb, nd, r) for _, nb, nd in nulls]})
    return out


def excess(sets, lim=SPAN):
    """Real and per-draw-mean null offset histograms, and the excess's centre (mean of the
    real-minus-null offset mass within +-lim)."""
    edges = np.arange(-SPAN, SPAN + BIN, BIN)
    real = np.zeros(len(edges) - 1)
    nul = np.zeros(len(edges) - 1)
    sr = nr = sn = nn = 0.0
    for p in sets:
        real += np.histogram(p["o"], edges)[0]
        o = p["o"][np.abs(p["o"]) <= lim]
        sr, nr = sr + o.sum(), nr + len(o)
        k = max(1, len(p["on"]))
        for on in p["on"]:
            nul += np.histogram(on, edges)[0] / k
            on = on[np.abs(on) <= lim]
            sn, nn = sn + on.sum() / k, nn + len(on) / k
    centre = (sr - sn) / (nr - nn) if nr - nn > 1.0 else float("nan")
    return edges, real, nul, centre, nr - nn


def centre_boot(sets, lim, n_boot=N_BOOT, seed=PP.SEED):
    """Ramp-cluster bootstrap 95% interval of ``excess(sets, lim)``'s centre. Each pair is
    reduced once to (real count, real sum, null count, null sum) within +-lim; a replicate
    is then a weighted sum (weights = how often its ramp was drawn)."""
    feats = np.zeros((len(sets), 4))
    for k, p in enumerate(sets):
        o = p["o"][np.abs(p["o"]) <= lim]
        feats[k, 0], feats[k, 1] = len(o), o.sum()
        m = max(1, len(p["on"]))
        for on in p["on"]:
            on = on[np.abs(on) <= lim]
            feats[k, 2] += len(on) / m
            feats[k, 3] += on.sum() / m
    w = cluster_weights([p["r"]["uid"] for p in sets], n_boot, seed)
    t = w @ feats                                   # n_boot x 4
    den = t[:, 0] - t[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(den > 1.0, (t[:, 1] - t[:, 3]) / den, np.nan)
    if np.all(np.isnan(c)):
        return [None, None]
    return [float(x) for x in np.nanpercentile(c, [2.5, 97.5])]


def excess_summary(sets, boot=True):
    _, real, nul, c15, x15 = excess(sets, 15.0)
    _, _, _, c30, x30 = excess(sets, 30.0)
    e = real - nul
    # where the excess peaks (5 deg bins), and its mass within +-10 vs 10-40 deg off
    edges = np.arange(-SPAN, SPAN + BIN, BIN)
    mid = (edges[:-1] + edges[1:]) / 2
    d = {"n_pairs": len(sets), "real": float(real.sum()), "null": float(nul.sum()),
         "excess": float(e.sum()),
         "excess_within_10": float(e[np.abs(mid) < 10].sum()),
         "excess_10_40": float(e[(np.abs(mid) > 10) & (np.abs(mid) < 40)].sum()),
         "excess_40_90": float(e[np.abs(mid) > 40].sum()),
         "peak_bin": float(mid[int(np.argmax(e))]),
         "centre_15": c15, "centre_30": c30}
    if boot:
        d["centre_15_ci"] = centre_boot(sets, 15.0)
        d["centre_30_ci"] = centre_boot(sets, 30.0)
    return d


# --------------------------------------------------------------------------- #
# pano control
# --------------------------------------------------------------------------- #
def pano_control(ramp_ll, thr=0.55, boot=True):
    recs = {}
    with open(PP.NEIGHBOURHOOD, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r["pano"]["panorama_id"]] = r
    caps = [r for r in PP.read_csv(PP.CAPTURES)
            if r["city"] == "richmond" and r["is_source"] == "0"
            and PP.RANGE_MIN <= float(r["dist_m"]) <= PP.RANGE_MAX and r["pano_id"] in recs]
    # (a) the flat path's ENU bearing vs the labeler's x_proj
    dx = []
    for c in caps:
        pn = recs[c["pano_id"]]["pano"]
        a, b = ramp_ll[c["ramp_uid"]]
        e, n = P.enu_offset(float(pn["lat"]), float(pn["lng"]), a, b)
        bear = math.degrees(math.atan2(e, n)) % 360
        xp = (0.5 + float(P.wrap_deg(bear - float(pn["camera_heading"]))) / 360.0) % 1.0
        dx.append(float(P.wrap_deg((xp - float(c["x_proj"])) * 360.0)))
    dx = np.abs(np.array(dx))
    # (b) offsets of gated detections >= thr, real vs rotation null; (c) heading scan
    sets = []
    for c in caps:
        dets = [d for d in recs[c["pano_id"]]["detections"] if d["confidence"] >= thr]
        r = {"uid": c["ramp_uid"], "range": float(c["dist_m"]), "bearing": 0.0}

        def offs(shift):
            if not dets:
                return np.zeros(0)
            b = P.wrap_deg((np.array([d["x_normalized"] for d in dets]) + shift
                            - float(c["x_proj"])) * 360.0)
            dep = (np.array([d["y_normalized"] for d in dets]) - 0.5) * 180.0
            return gated_offsets(b, dep, r)
        sets.append({"r": r, "o": offs(0.0), "on": [offs(s) for s in PP.PANO_NULL_SHIFTS],
                     "pano": c["pano_id"]})
    scan = []
    rows0 = PP.pano_bearing_check(thr, 0.0)
    nulls0 = [PP.pano_bearing_check(thr, s) for s in PP.PANO_NULL_SHIFTS]
    base = np.mean([r["bearing_hit"] for r in rows0])
    nmean = np.mean([np.mean([r["bearing_hit"] for r in rr]) for rr in nulls0])
    for s in SHIFTS:
        rr = PP.pano_bearing_check(thr, s / 360.0)
        scan.append({"shift": s, "rate": float(np.mean([r["bearing_hit"] for r in rr]))})
    return {"n_captures": len(caps),
            "xproj_vs_flat_path_abs_deg": {"median": float(np.median(dx)),
                                            "p99": float(np.percentile(dx, 99)),
                                            "max": float(dx.max())},
            "rate": float(base), "rotation_null": float(nmean),
            "offsets": excess_summary(sets, boot), "_sets": sets, "shift_scan": scan}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
FIGURES_JSON = os.path.join(REPO, "docs", "figures", "perspective_photos_218", "figures.json")


def viewed_misses(images, recs, ramp_ll, arm):
    """The 15 miss candidates #227's figure agent looked at by eye: per (image, ramp), the
    pose fields, the independent headings, and the signed offset of every stored peak
    (>= 0.10) from the projected bearing, before and after the travel-heading correction;
    and whether the ramp becomes a hit at 0.30 / 0.10 under that correction."""
    with open(FIGURES_JSON, encoding="utf-8") as f:
        viewed = json.load(f)["misses_viewed"]
    by_id = {im["id"]: im for im in images}
    out = []
    for v in viewed:
        im = by_id.get(v["image_id"])
        if im is None:
            out.append({**v, "note": "not a positive image in this arm"})
            continue
        r = next(x for x in im["g"]["near"] if x["uid"] == v["ramp"])
        dets = im["rec"]["dets"]
        b, dep = det_world_tf(dets, im["g"]["cam"], im["g"]["R_wc"])
        d = {"image_id": v["image_id"], "ramp": v["ramp"], "seen": v["seen"],
             "camera": im["camera"], "level_pose": im["level_pose"],
             "range": r["range"], "dbear": r["dbear"], "d_compass": im["d_compass"],
             "d_travel_raw": im["d_travel_raw"],
             "peaks": [{"score": dd["score"], "offset": float(P.wrap_deg(bb - r["bearing"])),
                        "depression": float(dp)} for dd, bb, dp in zip(dets, b, dep)]}
        for thr in (0.30, 0.55, 0.10):
            for name, s in (("as_scored", 0.0), ("travel", im["d_travel_raw"] or 0.0),
                            ("compass", im["d_compass"])):
                cl = PP.claim_bearing(dets, b + s, dep, im["g"]["near"], thr)
                d[f"hit_{name}_{thr:.2f}"] = v["ramp"] in cl
        out.append(d)
    return out


def all_meta():
    """``d_compass`` / ``d_travel_raw`` for every flat image (``outvoted`` input)."""
    census = {r["image_id"]: r for r in PP.read_csv(os.path.join(PP.CENSUS, "images.csv"))}
    travel = travel_bearings(census)
    out = {}
    for row in PP.read_csv(PP.IMAGES_CSV):
        iid = row["image_id"]
        head = float(row["computed_compass_angle"])
        tr = travel.get(iid, [None, None])[0]
        out[iid] = {"d_compass": float(P.wrap_deg(float(census[iid]["compass_angle"]) - head)),
                    "d_travel_raw": None if tr is None else float(P.wrap_deg(tr[0] - head))}
    return out


def reversed_frame(m):
    """SfM heading more than 60 deg from both the direction of travel and the device
    compass (which agree within 15 deg): the frame may face backward, or the SfM heading
    is wrong; either way its in-view set is suspect."""
    a, b = m["d_travel_raw"], m["d_compass"]
    return a is not None and abs(a) > 60 and abs(b) > 60 and abs(P.wrap_deg(a - b)) <= 15


def heading_stats(images):
    out = {}
    for key in ("d_compass", "d_travel_raw", "d_travel_sfm"):
        groups = defaultdict(list)
        for im in images:
            if im[key] is not None:
                groups["all"].append(im[key])
                groups[im["camera"]].append(im[key])
        g = {}
        for name, v in groups.items():
            v = np.array(v)
            if len(v) < 5:
                continue
            a = np.abs(v)
            fwd = v[a < 45]
            g[name] = {"n": int(len(v)), "median": float(np.median(v)),
                       "abs_median": float(np.median(a)),
                       "within_10": float(np.mean(a <= 10)),
                       "within_20": float(np.mean(a <= 20)),
                       "near_90": float(np.mean((a > 45) & (a < 135))),
                       "near_180": float(np.mean(a >= 135)),
                       "forward_median": float(np.median(fwd)) if len(fwd) else None,
                       "forward_n": int(len(fwd))}
        out[key] = g
    return out


def rnd(o):
    if isinstance(o, float):
        return None if not math.isfinite(o) else round(o, 4)
    if isinstance(o, dict):
        return {k: rnd(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [rnd(v) for v in o]
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="canvas_level,stretch")
    ap.add_argument("--quick", action="store_true", help="5 null draws, no bootstrap")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    n_null = 5 if a.quick else PP.N_NULL
    boot = not a.quick
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    summary = {"issue": 218, "pr": 227, "n_null": n_null, "seed": PP.SEED,
               "lateral_m": PP.LATERAL_M, "h_accept": [PP.H_MIN, PP.H_MAX], "arms": {}}
    hist_rows = []
    images0 = None
    for arm in a.arms.split(","):
        images, recs, ramp_ll = load(arm, n_null)
        images0 = images0 or images
        res = {}
        for thr in (0.30, 0.55, 0.10):
            sc = Scorer(images, recs, ramp_ll, thr)
            key = f"{thr:.2f}"
            pairs, r0, n0 = sc.run()
            ident = row_of("as scored", pairs, r0, n0, boot)

            def corrected(name, **kw):
                p, r, n = sc.run(**kw)
                d = row_of(name, p, r, n, boot)
                d["gain_above"] = float((r - n).mean() - (r0 - n0).mean())
                if boot:
                    d["gain_above_ci"] = ramp_boot(pairs, (r - n) - (r0 - n0))
                return d

            unnamed = lambda im: im["camera"] in ("none none", "")  # noqa: E731
            corr = [ident]
            for name, kw in (
                    ("mirror x (u -> w-1-u)", {"mirror": True}),
                    ("device compass_angle heading", {"per_image_shift":
                                                     lambda im: im["d_compass"]}),
                    ("device GPS position", {"raw_pos": True}),
                    ("travel heading (device GPS track), where available",
                     {"per_image_shift": lambda im: im["d_travel_raw"] or 0.0}),
                    ("travel heading, forward-facing only (|d| < 45)",
                     {"per_image_shift": lambda im: im["d_travel_raw"]
                      if im["d_travel_raw"] is not None and abs(im["d_travel_raw"]) < 45
                      else 0.0}),
                    ("SfM heading outvoted (travel and device compass agree within 10 deg, "
                     "both >= 10 deg from SfM): their mean",
                     {"per_image_shift": lambda im: outvoted(im, 10.0)}),
                    ("SfM heading outvoted, 15 deg rule",
                     {"per_image_shift": lambda im: outvoted(im, 15.0)}),
                    ("unnamed cameras only: travel heading",
                     {"per_image_shift": lambda im: (im["d_travel_raw"] or 0.0)
                      if unnamed(im) else 0.0}),
                    ("unnamed cameras only: heading -20 deg",
                     {"per_image_shift": lambda im: -20.0 if unnamed(im) else 0.0})):
                corr.append(corrected(name, **kw))
            shift_scan, mirror_scan, f_scan = [], [], []
            for s_ in SHIFTS:
                p, r, n = sc.run(shift=float(s_))
                shift_scan.append({"shift": s_, "rate": float(r.mean()),
                                   "null": float(n.mean()),
                                   "above": float(r.mean() - n.mean())})
                p, r, n = sc.run(shift=float(s_), mirror=True)
                mirror_scan.append({"shift": s_, "rate": float(r.mean()),
                                    "null": float(n.mean()),
                                    "above": float(r.mean() - n.mean())})
            for f in FSCALES:
                p, r, n = sc.run(fscale=f)
                f_scan.append({"fscale": f, "rate": float(r.mean()), "null": float(n.mean()),
                               "above": float(r.mean() - n.mean())})
            best = max(shift_scan, key=lambda x: x["above"])
            bestf = max(f_scan, key=lambda x: x["above"])
            # the best of each scan, re-scored with a paired bootstrap of its gain. The
            # scan maximum is chosen on these same pairs, so its gain is biased upward.
            corr.append(corrected(f"best heading shift ({best['shift']:+d} deg; chosen "
                                  f"on these pairs)", shift=float(best["shift"])))
            corr.append(corrected(f"best focal scale (x{bestf['fscale']:g}; chosen on "
                                  f"these pairs)", fscale=float(bestf["fscale"])))
            # per camera: hit rate, floor, and the shift / focal scans on that camera alone
            per_cam = {}
            cams = defaultdict(list)
            for k, (im, r) in enumerate(pairs):
                cams[im["camera"]].append(k)
            for cam, ks in sorted(cams.items(), key=lambda kv: -len(kv[1])):
                if len(ks) < 25:
                    continue
                sub = Scorer([im for im in images if im["camera"] == cam], recs, ramp_ll,
                             thr)
                sub.cache = sc.cache
                pp_, rr, nn_ = sub.run()
                d = {"n_pairs": len(ks), "hits": int(rr.sum()), "rate": float(rr.mean()),
                     "null": float(nn_.mean()), "above": float(rr.mean() - nn_.mean()),
                     "n_ramps": len({r["uid"] for _, r in pp_}),
                     "n_sequences": len({im["seq"] for im, _ in pp_})}
                if boot:
                    d["rate_ci"] = ramp_boot(pp_, rr)
                    d["above_ci"] = ramp_boot(pp_, rr - nn_)
                sscan = []
                for s_ in SHIFTS:
                    _, r, n = sub.run(shift=float(s_))
                    sscan.append((s_, float(r.mean() - n.mean())))
                fscan = []
                for f in FSCALES:
                    _, r, n = sub.run(fscale=f)
                    fscan.append((f, float(r.mean() - n.mean())))
                d["best_shift"] = max(sscan, key=lambda x: x[1])
                d["best_fscale"] = max(fscan, key=lambda x: x[1])
                d["shift_scan_above"] = sscan
                d["focal_scan_above"] = fscan
                per_cam[cam] = d
            # ceiling under ANY re-mapping of detection bearings: each detection >= thr can
            # claim at most one ramp, so an image contributes at most min(#dets, #in view)
            ceil, ceil_by = 0, defaultdict(lambda: [0, 0])
            for im in images:
                nd = sum(d["score"] >= thr for d in im["rec"]["dets"])
                nv = sum(r["in_view"] for r in im["g"]["near"])
                ceil += min(nd, nv)
                ceil_by[im["camera"]][0] += min(nd, nv)
                ceil_by[im["camera"]][1] += nv
            ceiling = {"pairs_claimable": ceil, "n_pairs": len(pairs),
                       "rate": ceil / len(pairs),
                       "by_camera": {k: {"claimable": v[0], "n_pairs": v[1]}
                                     for k, v in ceil_by.items()}}
            # offsets
            sets = offset_sets(sc)
            groups = {"all": lambda p: True}
            for cam in sorted({im["camera"] for im in images}):
                groups[f"camera: {cam}"] = lambda p, cam=cam: p["im"]["camera"] == cam
            groups["pose: SfM orientation"] = lambda p: not p["im"]["level_pose"]
            groups["pose: no SfM orientation (exactly level)"] = \
                lambda p: p["im"]["level_pose"]
            groups["ramp in left third of frame"] = lambda p: p["r"]["xfrac"] < 1 / 3
            groups["ramp in centre third"] = lambda p: 1 / 3 <= p["r"]["xfrac"] <= 2 / 3
            groups["ramp in right third"] = lambda p: p["r"]["xfrac"] > 2 / 3
            groups["device compass within 5 deg of SfM"] = \
                lambda p: abs(p["im"]["d_compass"]) < 5
            groups["device compass >= 5 deg from SfM"] = \
                lambda p: abs(p["im"]["d_compass"]) >= 5
            offs = {}
            for gname, f in groups.items():
                s = [p for p in sets if f(p)]
                if len(s) < 15:
                    continue
                offs[gname] = excess_summary(s, boot and gname in (
                    "all", "ramp in left third of frame", "ramp in centre third",
                    "ramp in right third") or (boot and len(s) >= 40))
            edges, real, nul, _, _ = excess(sets)
            for j in range(len(real)):
                hist_rows.append({"source": f"flat {arm} @ {key}", "bin_lo": edges[j],
                                  "bin_hi": edges[j + 1], "real": round(float(real[j]), 4),
                                  "null": round(float(nul[j]), 4),
                                  "excess": round(float(real[j] - nul[j]), 4)})
            # per-sequence centres for the sequences with the most pairs
            by_seq = defaultdict(list)
            for p in sets:
                by_seq[p["im"]["seq"]].append(p)
            seqs = []
            for sq, s in sorted(by_seq.items(), key=lambda kv: -len(kv[1]))[:12]:
                e = excess(s, 30.0)
                seqs.append({"sequence": sq, "camera": s[0]["im"]["camera"],
                             "n_pairs": len(s), "excess_30": float(e[4]),
                             "centre_30": float(e[3])})
            if arm == "canvas_level" and thr == 0.30:
                res["viewed_misses"] = viewed_misses(images, recs, ramp_ll, arm)
            res[key] = {"corrections": corr, "shift_scan": shift_scan,
                        "mirror_shift_scan": mirror_scan, "focal_scan": f_scan,
                        "per_camera": per_cam, "ceiling_any_geometry": ceiling,
                        "offsets": offs, "sequences": seqs}
            print(f"{arm} @ {key}: as scored {ident['rate']:.3f} null {ident['null']:.3f} "
                  f"above {ident['above']:+.3f}; best shift {best['shift']:+d} above "
                  f"{best['above']:+.3f}; {time.time() - t0:.0f}s", flush=True)
        summary["arms"][arm] = res
    summary["heading_vs_independent"] = heading_stats(images0)
    # full re-scores with corrected headings (every flat image, not only the positives)
    meta_all = all_meta()
    resc = {}
    for arm in a.arms.split(","):
        for thr in (0.30, 0.55):
            for name, fn, ex in (
                    ("as scored", lambda m: 0.0, None),
                    ("SfM heading outvoted, 10 deg rule", lambda m: outvoted(m, 10.0), None),
                    ("SfM heading outvoted, 15 deg rule", lambda m: outvoted(m, 15.0), None),
                    ("images whose SfM heading is > 60 deg from travel and compass dropped",
                     lambda m: 0.0, reversed_frame)):
                d = rescore(arm, thr, fn, meta_all, n_null, boot, ex)
                d["name"] = name
                resc.setdefault(arm, {}).setdefault(f"{thr:.2f}", []).append(d)
                print(f"rescore {arm} @ {thr}: {name}: {d['rate']:.3f} null {d['null']:.3f} "
                      f"above {d['above']:+.3f} ({d['n_pairs']} pairs)", flush=True)
    summary["rescore"] = resc
    pc = pano_control({u: (x, y) for u, x, y in PP.ramp_table()}, boot=boot)
    edges, real, nul, _, _ = excess(pc["_sets"])
    for j in range(len(real)):
        hist_rows.append({"source": "pano @ 0.55", "bin_lo": edges[j], "bin_hi": edges[j + 1],
                          "real": round(float(real[j]), 4), "null": round(float(nul[j]), 4),
                          "excess": round(float(real[j] - nul[j]), 4)})
    summary["pano_control"] = pc
    summary["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8", newline="") as f:
        json.dump(rnd(summary), f, indent=1, sort_keys=True)
        f.write("\n")
    with open(os.path.join(a.out, "offsets.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["source", "bin_lo", "bin_hi", "real", "null", "excess"],
                           lineterminator=chr(10))
        w.writeheader()
        for r in hist_rows:
            w.writerow({**r, "bin_lo": f"{r['bin_lo']:g}", "bin_hi": f"{r['bin_hi']:g}"})
    print(f"wrote {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
