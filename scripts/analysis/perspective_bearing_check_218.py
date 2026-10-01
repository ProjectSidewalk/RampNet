"""Is part of what #218 scores as a flat-photo miss a pose (bearing) error, not the model?

Prompted by the example figures (``perspective_figures_218.py``): in both drawn misses the
pool ramp's projected bearing points mid-road while the model fires on a visible ramp
27-30 deg off. If the SfM heading, the x -> bearing sign, or the focal / FOV were off,
real detections of real ramps would sit at a SYSTEMATIC signed offset from the GT
bearing. This reads that offset from committed files only (dets, census, images.csv): no
images, no GPU, no network. About 4 minutes on a desktop CPU.

Signed offset ``o = wrap(det_bearing - ramp_bearing)``, degrees; positive = the detection
is clockwise of (to the right of) the pool ramp. Read per in-view pair for every detection
>= THR that passes the bearing test's height gate (0.5-4 m at the ramp's range) and lies
within +-WIDE deg, beside the same quantity for the swap null's donor detections (the
scorer's own ``swap_donors`` and ``transplant``, same seed). The EXCESS of real over null
offsets is where real detections of ramps sit; its centre is the bias.

    python scripts/analysis/perspective_bearing_check_218.py            # canvas_level @ 0.30
    python scripts/analysis/perspective_bearing_check_218.py --arm stretch

Prints only; the numbers are quoted in docs/perspective_photos_218.md, "Is the pose
biased?". Step 8 (canvas_level only) also reads the 113 images with no SfM orientation:
the pose spread with and without them, and the canvas_sfm - canvas_level contrast without
them, quoted in section 4 of the same doc.
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from rampnet import perspective as P  # noqa: E402
import perspective_photos_218 as PP  # noqa: E402

WIDE = 60.0          # offsets read within +-WIDE deg of the pool ramp's bearing
CORE = 15.0          # the excess's centre is also read within +-CORE deg (tails excluded)
BIN = 5.0
N_BOOT = 2000


def offsets(b, dep, r, lim=WIDE):
    """Signed offsets of detections that pass the height gate at ramp ``r``."""
    if len(b) == 0:
        return np.zeros(0)
    h = r["range"] * np.tan(np.radians(np.clip(dep, 1e-6, 89.0)))
    ok = (dep > 0) & (h >= PP.H_MIN) & (h <= PP.H_MAX)
    o = P.wrap_deg(b - r["bearing"])
    return o[ok & (np.abs(o) <= lim)]


def collect(arm, thr, n_null):
    """Per positive image: geometry, detections >= thr in world terms, and the swap-null
    donors' detections projected through this image's camera and pose."""
    rows = PP.read_csv(PP.IMAGES_CSV)
    by_id = {r["image_id"]: r for r in rows}
    census = {r["image_id"]: r for r in PP.read_csv(os.path.join(PP.CENSUS, "images.csv"))}
    ramps = PP.ramp_table()
    recs = PP.load_dets(arm)
    ids = sorted(recs)
    geo = {i: PP.image_geometry(by_id[i], recs[i]["width"], recs[i]["height"], ramps)
           for i in ids}
    pos = [i for i in ids if geo[i]["positive"]]
    donors = PP.swap_donors(pos, {i: by_id[i]["nearest_ramp"] for i in pos}, n_null)
    images = []
    for iid in pos:
        g, rec = geo[iid], recs[iid]
        dets = [d for d in rec["dets"] if d["score"] >= thr]
        b, dep, _ = PP.det_world(dets, g["cam"], g["R_wc"])
        null = []
        for dr in donors:
            dd = recs[dr[iid]]
            sd = [d for d in PP.transplant(dd["dets"], dd["width"], dd["height"],
                                           rec["width"], rec["height"]) if d["score"] >= thr]
            sb, sdep, _ = PP.det_world(sd, g["cam"], g["R_wc"])
            null.append((sd, sb, sdep))
        c, row = census[iid], by_id[iid]
        e, n = P.enu_offset(float(c["lat"]), float(c["lng"]), float(c["raw_lat"]),
                            float(c["raw_lng"]))
        images.append({
            "image_id": iid, "g": g, "dets": dets, "b": b, "dep": dep, "null": null,
            "camera": f'{row["make"]} {row["model"]}'.strip(),
            "sfm_cluster": c["has_sfm_cluster"] == "1",
            # computed_rotation exactly level (pitch = roll = 0 to 1e-3 deg): Mapillary
            # returned no SfM orientation for this image, only a heading
            "level_pose": all(abs(x) < 1e-3 for x in P.heading_pitch_roll(g["R_wc"])[1:]),
            "dhead_raw": float(P.wrap_deg(float(c["compass_angle"])
                                          - float(c["computed_compass_angle"]))),
            "dpos_raw": float(math.hypot(e, n))})
    return images


def score(images, shift=lambda im: 0.0, lateral=PP.LATERAL_M):
    """Pair hits (real) and the swap-null mean per pair, with every detection's bearing
    moved by ``shift(im)`` deg, through the scorer's own greedy one-to-one claims."""
    real, nul = [], []
    for im in images:
        s, near = shift(im), im["g"]["near"]
        cl = PP.claim_bearing(im["dets"], im["b"] + s, im["dep"], near, 0.0, lateral)
        ncl = [PP.claim_bearing(sd, sb + s, sdep, near, 0.0, lateral)
               for sd, sb, sdep in im["null"]]
        for r in near:
            if r["in_view"]:
                real.append(r["uid"] in cl)
                nul.append(np.mean([r["uid"] in c for c in ncl]))
    return np.array(real), np.array(nul)


def pairs_of(images):
    """Per in-view pair: offsets (real, per null draw), hit, and the claiming offset."""
    out = []
    for im in images:
        g = im["g"]
        cl = PP.claim_bearing(im["dets"], im["b"], im["dep"], g["near"], 0.0)
        for r in g["near"]:
            if not r["in_view"]:
                continue
            i = cl.get(r["uid"])
            out.append({
                "im": im, "ramp": r["uid"], "range": r["range"], "dbear": r["dbear"],
                "lateral": r["range"] * math.sin(math.radians(r["dbear"])),
                "o": offsets(im["b"], im["dep"], r),
                "on": [offsets(sb, sdep, r) for _, sb, sdep in im["null"]],
                "hit": i is not None,
                "o_hit": None if i is None else float(P.wrap_deg(im["b"][i] - r["bearing"]))})
    return out


def excess(pairs, n_null, lim=WIDE):
    """Real minus null offset histogram, and the excess's mean (its centre)."""
    edges = np.arange(-lim, lim + BIN, BIN)
    real, nul = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    s_r = s_n = n_r = n_n = 0.0
    for p in pairs:
        o = p["o"][np.abs(p["o"]) <= lim]
        real += np.histogram(o, edges)[0]
        s_r, n_r = s_r + o.sum(), n_r + len(o)
        for on in p["on"]:
            on = on[np.abs(on) <= lim]
            nul += np.histogram(on, edges)[0] / n_null
            s_n, n_n = s_n + on.sum() / n_null, n_n + len(on) / n_null
    centre = (s_r - s_n) / (n_r - n_n) if n_r - n_n > 0.5 else float("nan")
    return edges, real, nul, centre, n_r, n_n


def boot(pairs, stat, seed=PP.SEED):
    """Ramp-cluster bootstrap 95% interval of ``stat(pairs)``."""
    by = {}
    for p in pairs:
        by.setdefault(p["ramp"], []).append(p)
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    out = [stat([p for k in rng.integers(0, len(keys), len(keys)) for p in by[keys[k]]])
           for _ in range(N_BOOT)]
    return np.nanpercentile(out, [2.5, 97.5])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="canvas_level")
    ap.add_argument("--thr", type=float, default=PP.PRIMARY_THR)
    ap.add_argument("--n-null", type=int, default=PP.N_NULL)
    a = ap.parse_args(argv)
    images = collect(a.arm, a.thr, a.n_null)
    pairs = pairs_of(images)
    nh = sum(p["hit"] for p in pairs)
    print(f"{a.arm} @ {a.thr}: {len(images)} positive images, {len(pairs)} in-view pairs, "
          f"{nh} hit ({nh / len(pairs):.3f})")

    print("\n1. Matched hits: signed offset of the claiming detection")
    oh = np.array([p["o_hit"] for p in pairs if p["hit"]])
    win = np.array([math.degrees(math.atan(PP.LATERAL_M / p["range"]))
                    for p in pairs if p["hit"]])
    hits = [p for p in pairs if p["hit"]]
    ci = boot(hits, lambda ps: np.median([p["o_hit"] for p in ps]))
    print(f"   n {len(oh)}; mean {oh.mean():+.1f}, median {np.median(oh):+.1f} [{ci[0]:+.1f}, "
          f"{ci[1]:+.1f}], p25/p75 {np.percentile(oh, 25):+.1f}/{np.percentile(oh, 75):+.1f}; "
          f"share right of GT {np.mean(oh > 0):.2f}; offset / window, mean "
          f"{np.mean(oh / win):+.2f} (0 = centred, +-1 = at the window's edge)")
    dbh = np.array([p["dbear"] for p in hits])
    k = np.polyfit(dbh, oh, 1)
    ks = boot(hits, lambda ps: np.polyfit([p["dbear"] for p in ps],
                                          [p["o_hit"] for p in ps], 1)[0])
    print(f"   offset vs the ramp's angle from the heading (|dbear| p10-p90 "
          f"{np.percentile(np.abs(dbh), 10):.0f}-{np.percentile(np.abs(dbh), 90):.0f}): slope "
          f"{k[0]:+.3f} [{ks[0]:+.3f}, {ks[1]:+.3f}], intercept {k[1]:+.1f}. A focal / FOV "
          f"error of x% gives a slope of -0.6x/100 to -x/100 here; a flipped x -> bearing "
          f"sign, -2.")
    print("   by camera (hits, median offset):", ", ".join(
        f"{c} {len(v)} {np.median(v):+.1f}" for c, v in sorted(
            {c: [p["o_hit"] for p in hits if p["im"]["camera"] == c]
             for c in {p["im"]["camera"] for p in hits}}.items(), key=lambda kv: -len(kv[1]))))

    print(f"\n2. Every height-gated detection >= {a.thr} within +-{WIDE:.0f} deg of an in-view "
          f"pool ramp, real vs swap null")
    edges, real, nul, centre, nr, nn = excess(pairs, a.n_null)
    _, _, _, core, _, _ = excess(pairs, a.n_null, CORE)
    cw = boot(pairs, lambda ps: excess(ps, a.n_null)[3])
    cc = boot(pairs, lambda ps: excess(ps, a.n_null, CORE)[3])
    print(f"   real {nr:.0f}, null {nn:.1f} per draw, excess {nr - nn:.1f}; centre of the "
          f"excess {centre:+.1f} [{cw[0]:+.1f}, {cw[1]:+.1f}] over +-{WIDE:.0f}, {core:+.1f} "
          f"[{cc[0]:+.1f}, {cc[1]:+.1f}] over +-{CORE:.0f}")
    print("   bin           real   null  excess")
    for j in range(len(real)):
        print(f"   {edges[j]:+4.0f}..{edges[j + 1]:+4.0f}  {real[j]:5.0f} {nul[j]:6.1f} "
              f"{real[j] - nul[j]:+7.1f}")

    print("\n3. Misses: is there a detection off the bearing that a pose fix could move in?")
    miss = [p for p in pairs if not p["hit"]]
    for lim in (25.0, 35.0, WIDE):
        def has(o, lo=0.0, hi=lim):
            return np.any((np.abs(o) > lo) & (np.abs(o) <= hi))
        rr = np.mean([has(p["o"]) for p in miss])
        rn = np.mean([np.mean([has(on) for on in p["on"]]) for p in miss])
        print(f"   share of {len(miss)} missed pairs with a gated detection within +-{lim:.0f}: "
              f"real {rr:.3f}, swap null {rn:.3f}")
    rr = np.mean([np.any((np.abs(p["o"]) > 20) & (np.abs(p["o"]) <= 40)) for p in miss])
    rn = np.mean([np.mean([np.any((np.abs(on) > 20) & (np.abs(on) <= 40)) for on in p["on"]])
                  for p in miss])
    print(f"   ... with one 20-40 deg off (the figures' case): real {rr:.3f} "
          f"({rr * len(miss):.0f} pairs), swap null {rn:.3f} ({rn * len(miss):.1f})")
    silent = np.mean([len(p["im"]["dets"]) == 0 for p in miss])
    print(f"   share of missed pairs whose image has NO detection >= {a.thr} anywhere: "
          f"{silent:.3f}")

    print("\n4. Pose fields")
    dh = np.array([im["dhead_raw"] for im in images])
    dp = np.array([im["dpos_raw"] for im in images])
    print(f"   device compass_angle - computed_compass_angle (positive images): median "
          f"{np.median(dh):+.1f}, p10/p90 {np.percentile(dh, 10):+.1f}/"
          f"{np.percentile(dh, 90):+.1f}; within 5 deg {np.mean(np.abs(dh) < 5):.2f}")
    print(f"   device GPS vs computed position: median {np.median(dp):.1f} m, p90 "
          f"{np.percentile(dp, 90):.1f} m")

    print("\n5. Sensitivity: hit rate with every detection's bearing shifted (real, swap null, "
          "above chance)")
    variants = [("as scored", lambda im: 0.0),
                ("device compass instead of SfM", lambda im: im["dhead_raw"])]
    variants += [(f"all bearings {s:+d} deg", lambda im, s=s: float(s))
                 for s in (-10, -5, -3, 3, 5, 10)]
    if np.isfinite(core):
        variants.append((f"hit-median bias removed ({-np.median(oh):+.1f})",
                         lambda im: -float(np.median(oh))))
    for name, f in variants:
        r_, n_ = score(images, f)
        print(f"   {name:34s} {r_.mean():.3f} {n_.mean():.3f} {r_.mean() - n_.mean():+.3f}")
    r_, n_ = score(images, lateral=PP.LATERAL_LOOSE_M)
    print(f"   {'10 m lateral (scorer sensitivity)':34s} {r_.mean():.3f} {n_.mean():.3f} "
          f"{r_.mean() - n_.mean():+.3f}")

    print("\n6. Where the projected bearing points: lateral offset of the pool ramp from the "
          "camera's forward axis")
    r_, n_ = score(images)          # same pair order as pairs_of
    lat = np.array([abs(p["lateral"]) for p in pairs])
    rg = np.array([p["range"] for p in pairs])
    for lo, hi in ((0, 2), (2, 4), (4, 8), (8, 99)):
        m = (lat >= lo) & (lat < hi)
        if m.any():
            print(f"   |lateral| {lo}-{hi} m: {m.sum():3d} pairs (median range "
                  f"{np.median(rg[m]):.1f} m), hit {r_[m].mean():.3f}, swap null "
                  f"{n_[m].mean():.3f}, above chance {r_[m].mean() - n_[m].mean():+.3f}")

    print("\n7. By group: pairs, hits, hit rate, above chance; then the excess within +-15 deg "
          "and its centre")
    groups = {"has sfm_cluster": lambda p: p["im"]["sfm_cluster"],
              "no sfm_cluster": lambda p: not p["im"]["sfm_cluster"],
              "pose exactly level": lambda p: p["im"]["level_pose"],
              "pose not level": lambda p: not p["im"]["level_pose"]}
    for cam in sorted({p["im"]["camera"] for p in pairs}):
        groups[cam] = lambda p, cam=cam: p["im"]["camera"] == cam
    for name, f in groups.items():
        m = np.array([bool(f(p)) for p in pairs])
        if m.sum() < 10:
            continue
        e = excess([p for p, k in zip(pairs, m) if k], a.n_null, CORE)
        print(f"   {name:24s} {m.sum():4d} {int(r_[m].sum()):3d} {r_[m].mean():.3f} "
              f"{r_[m].mean() - n_[m].mean():+.3f} | {e[4] - e[5]:6.1f} {e[3]:+6.1f}")

    if a.arm == "canvas_level":
        sfm_dilution(a, pairs)


def sfm_dilution(a, pairs):
    """Images with no SfM orientation (``computed_rotation`` exactly level) get the same
    canvas in canvas_sfm as in canvas_level, so they add zeros to the paired
    canvas_sfm - canvas_level contrast. Reads the pitch / roll spread with and without
    them, and the contrast on the images that do have an orientation."""
    print("\n8. Images with no SfM orientation: pose spread, and the canvas_sfm - "
          "canvas_level contrast without them")
    pr = {}
    for row in PP.read_csv(PP.IMAGES_CSV):
        _, p, r = P.heading_pitch_roll(PP.pose_of(row))
        pr[row["image_id"]] = (p, r)
    ids = sorted(pr)
    lev = {i for i in ids if abs(pr[i][0]) < 1e-3 and abs(pr[i][1]) < 1e-3}
    for name, sel in (("all", ids), ("with an SfM orientation", [i for i in ids
                                                                   if i not in lev])):
        p = np.array([pr[i][0] for i in sel])
        r = np.array([pr[i][1] for i in sel])
        print(f"   {name:24s} {len(sel):5d} images: pitch p5/p95 {np.percentile(p, 5):+.1f}/"
              f"{np.percentile(p, 95):+.1f}, roll p5/p95 {np.percentile(r, 5):+.1f}/"
              f"{np.percentile(r, 95):+.1f}")
    print(f"   exactly level (no SfM orientation): {len(lev)} of {len(ids)}")
    lv, sf = PP.load_dets("canvas_level"), PP.load_dets("canvas_sfm")
    same = sum(1 for i in lev if [(d["u"], d["v"], d["score"]) for d in lv[i]["dets"]]
               == [(d["u"], d["v"], d["score"]) for d in sf[i]["dets"]])
    print(f"   of those, canvas_sfm detections identical to canvas_level: {same}")
    sp = pairs_of(collect("canvas_sfm", a.thr, 0))
    assert [(p["im"]["image_id"], p["ramp"]) for p in sp] == \
        [(p["im"]["image_id"], p["ramp"]) for p in pairs]
    for p, q in zip(pairs, sp):
        p["d_sfm"] = float(q["hit"]) - float(p["hit"])
    for name, f in (("all pairs", lambda p: True),
                    ("pairs on images with an SfM orientation",
                     lambda p: p["im"]["image_id"] not in lev),
                    ("pairs on images without one", lambda p: p["im"]["image_id"] in lev)):
        ps = [p for p in pairs if f(p)]
        d = np.mean([p["d_sfm"] for p in ps])
        ci = boot(ps, lambda qs: np.mean([q["d_sfm"] for q in qs]))
        lvl = np.mean([p["hit"] for p in ps])
        print(f"   {name:40s} {len(ps):4d} pairs: canvas_level {lvl:.3f}, canvas_sfm - "
              f"canvas_level {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] (ramp-cluster bootstrap)")


if __name__ == "__main__":
    main()
