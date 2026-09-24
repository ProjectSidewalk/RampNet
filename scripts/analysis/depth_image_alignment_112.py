"""Which way round is a GSV depth payload against a RampNet benchmark JPEG? (#112, PR #184 B1)

``recall_by_depth_112.py`` looks up the payload plane under a benchmark-image point. The payload
stores one plane index per pixel of a 512x256 grid in *raw* column order, and the labeler's
``depth.py`` maps a stored column ``c`` to raw column ``width - 1 - c`` -- right for
streetlevel's rastered depth map, which is mirrored. Whether the benchmark JPEGs share that
raster's orientation is a question about the imagery, so this script asks the imagery. Two
hypotheses, for image column ``c`` on the 512-column grid:

  * ``raw``  -- the payload column under image column c is raw column c;
  * ``flip`` -- it is raw column ``width - 1 - c`` (``depth.py``'s stored->raw mapping).

Four checks, each independent of the others and of ``recall_by_depth_112.py``'s lookup:

  A. **Sky.** Per panorama, cross-correlate the payload's sky mask (index 0) with an image sky
     score (brightness + blueness) over the upper half, at every one of the 512 column shifts,
     for both hypotheses. Count the panoramas whose best (hypothesis, shift) is at |shift| <= 2
     and <= 8, and those where ``raw`` beats ``flip`` at zero shift.
  B. **Ramps are on the ground.** For every reviewer-confirmed GT point and every true-positive
     detection on a measured-ground panorama, is the payload plane under it ground-like (not
     sky, tilt <= 18 degrees) under each hypothesis?
  C. **Edges.** Per panorama, correlate the column profile of plane-index boundaries with the
     column profile of horizontal image gradient in the band around the horizon, both
     hypotheses, all shifts.
  D. **The raw-space ray formula.** Needs no image: where a ground-like plane meets a wall
     (tilt >= 60 degrees) across a horizontal raw-column boundary, the two planes should give
     the same range on the boundary ray. Compares the labeler's ``depth._direction`` azimuth
     (``phi = (width - col)/width * 2pi + pi/2`` at the boundary) against its mirror
     (``phi = col/width * 2pi + pi/2``) by the median |log(range_a / range_b)|.

A, B and C decide the image<->raw column mapping; D decides the ray formula in raw space. Their
composition is ``recall_by_depth_112.raw_column`` / ``image_ray``.

Inputs: the depth payloads (unpublished; see ``docs/detection_recall_analysis.md`` §0.5), the
benchmark JPEGs (HF ``projectsidewalk/rampnet-benchmark``, ``benchmark/<split>/panos/``), and the
committed ``analysis_out/recall_by_depth_112.json`` for the GT / TP coordinates. About 2.5 min
on CPU.

    python scripts/analysis/depth_image_alignment_112.py \\
        --labeler-root D:/Git/sidewalk-auto-labeler --panos-root D:/Git/RampNet
"""
import argparse
import gzip
import json
import math
import os
import sys
import warnings

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
import recall_by_depth_112 as rbd  # noqa: E402

OUT = os.path.join(REPO, "analysis_out", "depth_image_alignment_112.json")
SPLITS = rbd.DEPTH_SPLITS
WALL_MIN_TILT_DEG = 60.0


def load_index(depthlib, labeler_root, city, pid):
    payload, _ = rbd.load_payload(depthlib, os.path.join(labeler_root, "runs", city, "depth"), pid)
    if payload is None:
        return None, None
    return payload, np.frombuffer(payload.indices, np.uint8).reshape(payload.height, payload.width)


def ground_like(payload, i):
    if i == rbd.SKY or i >= len(payload.planes):
        return False
    return math.degrees(math.acos(min(1.0, abs(payload.planes[i].nz)))) <= rbd.GROUND_MAX_TILT_DEG


def xcorr(a, b):
    """Circular cross-correlation over columns, summed over rows: out[s] = <a, roll(b, s)>."""
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    fa, fb = np.fft.fft(a, axis=-1), np.fft.fft(b, axis=-1)
    cc = np.real(np.fft.ifft(fa * np.conj(fb), axis=-1))
    return (cc.sum(0) if cc.ndim == 2 else cc) / a.size


def best_of(cr, cf, w):
    """(hypothesis, |shift|) of the best correlation over both hypotheses and every shift."""
    k_r, k_f = int(np.argmax(cr)), int(np.argmax(cf))
    hyp, k = ("raw", k_r) if cr[k_r] >= cf[k_f] else ("flip", k_f)
    return hyp, min(k, w - k)


def empty_counts():
    return {"panos": 0, "raw_beats_flip_at_zero_shift": 0,
            "best_raw_within_2": 0, "best_raw_within_8": 0,
            "best_flip_within_2": 0, "best_flip_within_8": 0}


def tally(c, cr, cf, w):
    c["panos"] += 1
    c["raw_beats_flip_at_zero_shift"] += int(cr[0] > cf[0])
    hyp, k = best_of(cr, cf, w)
    c[f"best_{hyp}_within_2"] += int(k <= 2)
    c[f"best_{hyp}_within_8"] += int(k <= 8)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labeler-root", default=os.environ.get("LABELER_ROOT", r"D:\Git\sidewalk-auto-labeler"))
    ap.add_argument("--panos-root", default=REPO, help="checkout holding benchmark/<split>/panos/*.jpg")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore", Image.DecompressionBombWarning)
    Image.MAX_IMAGE_PIXELS = None

    depthlib = rbd.load_depthlib(a.labeler_root)
    with open(rbd.OUT_JSON, encoding="utf-8") as fh:
        rows = json.load(fh)
    measured = {(p["city"], p["pano"]) for p in rows["panos"] if p["camera_height_status"] == "measured"}

    sky = {c: empty_counts() for c in SPLITS}
    edge = {c: empty_counts() for c in SPLITS}
    ground = {k: {"n": 0, "raw": 0, "flip": 0} for k in ("gt_points", "tp_detections")}
    seam = {"raw_formula": [], "mirrored_formula": []}
    pts = {}
    for p in rows["points"]:
        pts.setdefault((p["city"], p["pano"]), []).append(("gt_points", p["x"], p["y"]))
    for d in rows["detections"]:
        if d["kind"] == "TP":
            pts.setdefault((d["city"], d["pano"]), []).append(("tp_detections", d["x"], d["y"]))

    for city in SPLITS:
        for pano in sorted(p["pano"] for p in rows["panos"] if p["city"] == city):
            payload, idx = load_index(depthlib, a.labeler_root, city, pano)
            if payload is None:
                continue
            h, w = idx.shape

            # B: the plane under each GT point / TP detection, measured-ground panos only
            if (city, pano) in measured:
                for kind, x, y in pts.get((city, pano), []):
                    col = min(w - 1, int((x % 1.0) * w))
                    row = min(h - 1, int(y * h))
                    ground[kind]["n"] += 1
                    ground[kind]["raw"] += int(ground_like(payload, idx[row, col]))
                    ground[kind]["flip"] += int(ground_like(payload, idx[row, w - 1 - col]))

            # D: seam continuity in raw space, ground-like vs wall across a column boundary
            P = np.array([[q.nx, q.ny, q.nz, q.d] for q in payload.planes], dtype=float)
            if len(P) >= 2:
                tilt = np.degrees(np.arccos(np.minimum(1.0, np.abs(P[:, 2]))))
                left, right = idx[:, :-1].astype(int), idx[:, 1:].astype(int)
                ok = (left != right) & (left > 0) & (right > 0) & (left < len(P)) & (right < len(P))
                for r, c in zip(*np.nonzero(ok)):
                    i, j = left[r, c], right[r, c]
                    lo, hi = sorted((tilt[i], tilt[j]))
                    if lo > rbd.GROUND_MAX_TILT_DEG or hi < WALL_MIN_TILT_DEG:
                        continue
                    theta = (h - r - 0.5) / h * math.pi
                    for key, phi in (("raw_formula", (w - (c + 1)) / w * 2 * math.pi + math.pi / 2),
                                     ("mirrored_formula", (c + 1) / w * 2 * math.pi + math.pi / 2)):
                        v = np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi),
                                      math.cos(theta)])
                        di, dj = abs(P[i, 3] / (v @ P[i, :3])), abs(P[j, 3] / (v @ P[j, :3]))
                        seam[key].append(abs(math.log(di / dj)))

            # A and C need the image
            jpg = os.path.join(a.panos_root, "benchmark", city, "panos", pano + ".jpg")
            im = Image.open(jpg)
            im.draft("RGB", (im.width // 16, im.height // 16))
            rgb = np.asarray(im.convert("RGB").resize((w, h), Image.BILINEAR), dtype=float)
            R_, G_, B_ = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            up = slice(0, h // 2)
            mask = (idx[up] == rbd.SKY).astype(float)
            if 0.03 < mask.mean() < 0.97:
                score = (R_ + G_ + B_)[up] / 3 + 2 * (B_ - R_)[up]
                tally(sky[city], xcorr(score, mask), xcorr(score, mask[:, ::-1]), w)
            band = slice(int(0.30 * h), int(0.60 * h))
            lum = (R_ + G_ + B_)[band] / 3
            grad = np.abs(np.roll(lum, -1, axis=1) - lum).sum(0)
            ib = idx[band]
            bound = (ib != np.roll(ib, -1, axis=1)).sum(0).astype(float)
            if bound.std() > 0:
                # a boundary between raw c and c+1 mirrors to one between w-2-c and w-1-c
                tally(edge[city], xcorr(grad, bound), xcorr(grad, np.roll(bound[::-1], -1)), w)

    def pooled(d):
        out = {k: sum(v[k] for v in d.values()) for k in empty_counts()}
        return {**d, "pooled": out}

    result = {
        "labeler_commit": rbd.labeler_commit(a.labeler_root),
        "hypotheses": {"raw": "image column c = raw payload column c",
                       "flip": "image column c = raw payload column width-1-c (depth.py _raw_column)"},
        "A_sky": pooled(sky),
        "B_ground_under_point": {k: {**v, "raw_share": round(v["raw"] / v["n"], 4),
                                     "flip_share": round(v["flip"] / v["n"], 4)} for k, v in ground.items()},
        "C_edges": pooled(edge),
        "D_seam_continuity": {k: {"n_boundaries": len(v),
                                  "median_abs_log_ratio": round(float(np.median(v)), 4),
                                  "share_below_0p1": round(float(np.mean(np.array(v) < 0.1)), 4)}
                              for k, v in seam.items()},
    }
    rbd.write_json(a.out, result)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
