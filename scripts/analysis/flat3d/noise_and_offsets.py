"""Two checks on the flat-Mapillary 3D models (#214) from committed files only, added after
the review of #210 (C3, C4). Prints; writes nothing.

1. **Treatment-only paired comparison and run-to-run noise.** Seven corners have no flat
   images, so there ``flat`` and ``noflat`` had identical inputs, but they are separate
   reconstructions. Their 13 pairs give a free measurement of run-to-run noise per lift, and
   the other 47 pairs give the flat-vs-noflat comparison without the untreated pairs.
2. **Camera-to-prior offsets by camera kind.** Median horizontal distance of each registered
   camera centre (``cameras[].C``) from its Mapillary prior (``prior_enu``), per variant and
   kind, plus the mean offset per kind in a few corners.

Usage::

    python scripts/analysis/flat3d/noise_and_offsets.py
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import crossview_align_48 as H  # noqa: E402
import flat_mapillary_48 as F  # noqa: E402

LIFTS = ("sfm", "gs", "gsmed", "mvs")
KINDS = ("pano_src", "pano_oth", "pano_extra", "flat")
SHOW = ("richmond:0", "richmond:145", "richmond:25", "richmond:29", "richmond:30", "richmond:96")


def _variant(path):
    b = os.path.basename(path)[:-len(".json")]
    return "noflat" if b.endswith("_noflat") else "mlypano" if b.endswith("_mlypano") else "flat"


def corners():
    for path in sorted(glob.glob(os.path.join(F.OUT, "corners", "*.json"))):
        with open(path, encoding="utf-8") as f:
            yield _variant(path), json.load(f)


def noise_and_treatment():
    with open(os.path.join(F.OUT, "results_richmond.json"), encoding="utf-8") as f:
        pp = {p["pair_id"]: p for p in json.load(f)["per_pair"]}
    uid, treated, untreated = {}, [], []
    for v, c in corners():
        if v != "flat":
            continue
        for pid in c["pairs"]:
            uid[pid] = c["ramp_uid"]
            (treated if c["n_images_by_kind"].get("flat", 0) > 0 else untreated).append(pid)
    print(f"pairs whose corner has flat images: {len(treated)}; without: {len(untreated)}")
    print("run-to-run noise, |error(flat) - error(noflat)| on the identical-input pairs:")
    for lift in LIFTS:
        d = [abs(pp[p][f"flat_{lift}"] - pp[p][f"noflat_{lift}"]) for p in untreated]
        worst = max(zip(d, untreated))
        print(f"  {lift:6s} median {np.median(d):.3f} deg, max {worst[0]:.2f} deg ({worst[1]})")
    groups = {}
    for p in treated:
        groups.setdefault(uid[p], []).append(p)
    groups = [v for _, v in sorted(groups.items())]

    def stat(ii):
        return float(np.median([pp[p]["noflat_sfm"] - pp[p]["flat_sfm"] for p in ii]))
    ci = H.cluster_bootstrap(groups, stat)
    print(f"flat_sfm vs noflat_sfm on the {len(treated)} treated pairs ({len(groups)} ramps): "
          f"{stat(treated):+.3f} [{ci[0]:.3f}, {ci[1]:.3f}] (> 0: flat closer)")


def offsets():
    res, means = {}, {}
    for v, c in corners():
        per = {}
        for cam in c["cameras"]:
            if cam.get("C") is None or cam.get("prior_enu") is None:
                continue
            k = cam["kind"] if cam["kind"] in KINDS else "mapillary_pano"
            dx, dy = cam["C"][0] - cam["prior_enu"][0], cam["C"][1] - cam["prior_enu"][1]
            res.setdefault((v, k), []).append(float(np.hypot(dx, dy)))
            per.setdefault(k, []).append((dx, dy))
        means[(c["ramp_uid"], v)] = {k: np.round(np.mean(x, axis=0), 1).tolist()
                                     for k, x in per.items()}
    print("camera centre vs Mapillary prior, horizontal, median (p90), metres:")
    for v in ("noflat", "flat", "mlypano"):
        cells = [f"{k} {np.median(res[(v, k)]):.2f} ({np.percentile(res[(v, k)], 90):.1f})"
                 for k in KINDS + ("mapillary_pano",) if (v, k) in res]
        print(f"  {v:8s} " + "; ".join(cells))
    print("mean (east, north) offset by kind, flat variant:")
    for u in SHOW:
        print(f"  {u}: {means.get((u, 'flat'))}")


if __name__ == "__main__":
    noise_and_treatment()
    offsets()
