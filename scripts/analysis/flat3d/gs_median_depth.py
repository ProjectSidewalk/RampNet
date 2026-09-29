"""Post-hoc lift from the Gaussian splat (#214): the MEDIAN depth along the click ray
instead of gsplat's expected (alpha-weighted mean) depth.

Added after the pilot, which showed the expected-depth lift (``gs``) to be erratic:
a mean over every Gaussian the ray crosses is pulled toward floaters in front and
background behind. The median is the depth at which the ray's transmittance falls to 0.5,
the usual fix. It is post hoc, so it is reported as a separate arm (``*_gsmed``), not in
place of ``gs``.

For each Gaussian (the saved ``splat.ply``: log-scales, opacity logit, wxyz quaternion)
the ray o + t d meets its density at its closest Mahalanobis approach,
t* = d'A(mu - o) / d'Ad with A the inverse covariance; its alpha is
sigmoid(opacity) * exp(-q(t*) / 2). Sorted by t*, the first t* where the running
transmittance drops below 0.5 is the median depth. This is the 3D (ray-traced) evaluation,
not the rasteriser's 2D splat, so it is close to but not the same as what gsplat renders.

Writes ``lifts.gs_median`` and ``pairs.<pair>.gs_median`` into each corner's result.json.

    python scripts/analysis/flat3d/gs_median_depth.py --corners-root CORNERS
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import crossview_align_48 as H  # noqa: E402

T_MIN = 0.3        # ignore Gaussians closer than this to the camera (m)
ALPHA_MIN = 1e-4


def load_splat(path):
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    mu = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    s = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float64))
    q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float64)))
    return mu, s, q, op


def quat_to_R(q):
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


def median_depth(o, d, mu, s, q, op):
    """Distance along unit ray d at which transmittance falls to 0.5, or None."""
    R = quat_to_R(q)                               # (N, 3, 3), columns are the axes
    # A = R diag(1/s^2) R'; work in each Gaussian's frame
    dl = np.einsum("nji,j->ni", R, d) / s          # R' d / s
    ml = np.einsum("nji,nj->ni", R, mu - o) / s    # R' (mu - o) / s
    dd = np.sum(dl * dl, 1)
    t = np.sum(dl * ml, 1) / dd
    qmin = np.sum((ml - t[:, None] * dl) ** 2, 1)
    a = op * np.exp(-0.5 * qmin)
    keep = (t > T_MIN) & (a > ALPHA_MIN)
    t, a = t[keep], np.clip(a[keep], 0, 0.99)
    order = np.argsort(t)
    t, a = t[order], a[order]
    T = np.cumprod(1.0 - a)
    hit = np.nonzero(T < 0.5)[0]
    return (float(t[hit[0]]), int(keep.sum())) if len(hit) else (None, int(keep.sum()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corners-root", required=True)
    args = ap.parse_args(argv)
    from crossview_arms._registry import ANSWER_KEYS
    pairs = {p["pair_id"]: {k: v for k, v in p.items() if k not in ANSWER_KEYS}
             for p in H.read_frozen_pairs()}
    for rp in sorted(glob.glob(os.path.join(args.corners_root, "*", "result.json"))):
        d = os.path.dirname(rp)
        res = json.load(open(rp, encoding="utf-8"))
        ply = os.path.join(d, "splat.ply")
        if res.get("status") != "ok" or not os.path.exists(ply):
            continue
        cams = {c["name"]: c for c in res["cameras"]}
        src = next(c for c in res["cameras"] if c["kind"] == "pano_src")
        f, cx, cy = src["params"][:3]
        Rs, Cs = np.asarray(src["R_cw"]), np.asarray(src["C"])
        ray = Rs @ np.array([(H.VIEW_W / 2.0 - cx) / f, (H.VIEW_H / 2.0 - cy) / f, 1.0])
        ray /= np.linalg.norm(ray)
        tm, n = median_depth(Cs, ray, *load_splat(ply))
        lift = {"X": None if tm is None else [round(float(v), 4) for v in Cs + tm * ray],
                "click_depth": tm, "n_on_ray": n, "post_hoc": True}
        if tm is None:
            lift["reason"] = "transmittance_never_below_0.5"
        res.setdefault("lifts", {})["gs_median"] = lift
        # the pair's other-view centre comes from the manifest-free result: the view name
        for pid, pr in res["pairs"].items():
            oc = cams.get(f"{pid}_oth.jpg")
            if oc is None or tm is None:
                pr["gs_median"] = {"reason": "oth_not_registered" if oc is None else
                                   "no_lift"}
                continue
            X = Cs + tm * ray
            Ro, Co = np.asarray(oc["R_cw"]), np.asarray(oc["C"])
            p = Ro.T @ (X - Co)
            if p[2] <= 1e-6:
                pr["gs_median"] = {"reason": "behind_other"}
                continue
            fo, cxo, cyo = oc["params"][:3]
            u, v = fo * p[0] / p[2] + cxo, fo * p[1] / p[2] + cyo
            # the other view is centred on today's projection (proj_x, proj_y)
            pp = pairs[pid]
            x, y = H.view_to_pano(u, v, pp["proj_x"], pp["proj_y"])
            pr["gs_median"] = {"x": float(x), "y": float(y), "u": float(u), "v": float(v),
                               "range_oth_m": float(np.linalg.norm(X - Co))}
        with open(rp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(H.rnd(res, 6), indent=1, sort_keys=True) + "\n")
        print(os.path.basename(d), "median depth", tm, "n", n,
              "expected-depth lift", res["lifts"].get("gs", {}).get("click_depth"))


if __name__ == "__main__":
    main()
