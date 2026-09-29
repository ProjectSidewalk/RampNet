"""Pairwise image-matching family (#48): more matchers, more estimators, and priors.

The pilot's ``lg`` arm (ALIKED + LightGlue, ground homography) aligns only 30% of pairs.
This module asks what lowers that fallback rate without giving up accuracy. Every arm here
uses the same views as ``lg`` (``cut-views``: 1024x768, 75 deg, source centred on the GT
point, other view centred on today's 2.6 m projection) and the same ground band (5 deg below
the horizon, above the rig), so the only thing that changes between arms is named in the
arm's name:

* **matcher** -- ``lg_*`` the pilot's ALIKED + LightGlue matches, ``sp_lg`` SuperPoint, ``disk_lg`` DISK, ``siftlg`` SIFT (each with
  LightGlue, cvg/LightGlue), ``loftr`` (kornia LoFTR outdoor), ``roma`` (RoMa outdoor,
  dense). The pilot's ``lg`` itself (ALIKED, RANSAC) is not re-registered.
* **estimator** -- default: the pilot's RANSAC homography on ground matches (4 px, >= 15
  inliers, mapped point inside the view). ``_magsac``: MAGSAC++ homography, same gates.
  ``_epi``: an essential matrix from *all* matches (not just the ground; the views are
  pinholes with a known focal length) and the auto-height prior snapped onto the GT
  point's epipolar line. ``_hyb``: ground homography if it passes its gates, else ``_epi``,
  else the auto-height prior -- so it never falls back to the 2.6 m projection.

**The auto-height prior** is the committed ``proj_height_auto`` prediction for the pair
(GSV at the labeler's per-year rig height; Mapillary unchanged at 2.6 m). It is another
arm's output and is derived from camera geometry only, never from the reference. The
views were not re-cut around it: the two points are a median ~2 deg apart, well inside a
75 deg view, so it is used as a prior point, not as a view centre.

**Fallback rows carry diagnostics.** Every arm here returns ``{"x": None, "why": ...,
"n_matches": .., "n_ground": .., "inliers": ..}`` when it falls back, so the committed
predictions say *why* each pair fell back (``why`` in: ``no_ground_matches``,
``few_inliers``, ``mapped_outside_view``, ``epi_*``).

**Pre-specified vs post hoc.** Every constant below was fixed before any of these arms was
scored against the reference. Thresholds inherited from the pilot (5 deg band, 4 px, 15
inliers) are inherited, not re-tuned. Matcher-native settings are the matcher's defaults:
RoMa samples 5,000 matches with its own certainty-balanced sampler; LoFTR keeps matches
with confidence >= 0.5 (the value the pilot's informal probe used). If a setting is ever
tuned after scoring, it gets a new arm name and this one stays as the pre-specified row.

Extra inputs: ``--extra match_cache=DIR`` caches each matcher's raw matches per pair as
``.npz`` so the estimator variants of one matcher do not re-run the network. The packages
(``lightglue`` at cvg/LightGlue eb42fee, ``romatch`` 0.1.2) are not in requirements; see
``docs/crossview_align_48/matching.md``.
"""
import json
import os

import numpy as np

import crossview_align_48 as H
from crossview_arms._registry import register
from crossview_arms.matching import CENTRE, MIN_INLIERS, RANSAC_PX, filter_ground

GROUND_MARGIN_DEG = 5.0      # inherited from lg (post hoc there; not re-tuned here)
MAX_KEYPOINTS = 4096          # SuperPoint / DISK / SIFT: top-k by detector score, not raster
LOFTR_MIN_CONF = 0.5
ROMA_SAMPLES = 5000
EPI_MIN_INLIERS = MIN_INLIERS  # essential-matrix inliers needed to trust the epipolar line
EPI_PX = 1.0                   # essential-matrix RANSAC threshold, px (OpenCV's default)
MAGSAC_PX = RANSAC_PX

AUTO_ARM = "proj_height_auto"


# --------------------------------------------------------------------------- #
# matchers: (gray1, gray2, bgr1, bgr2) -> (k1 Nx2, k2 Nx2) in view pixels
# --------------------------------------------------------------------------- #


def _device(ctx):
    import torch
    return "cuda" if torch.cuda.is_available() and not getattr(ctx.args, "cpu", False) else "cpu"


class CvgLightGlue:
    """cvg/LightGlue with one of its extractors (SuperPoint, DISK, SIFT)."""

    def __init__(self, kind, device):
        import torch
        import lightglue as LG
        torch.manual_seed(H.SEED)
        ext = {"superpoint": LG.SuperPoint, "disk": LG.DISK, "sift": LG.SIFT}[kind]
        self.torch, self.device = torch, device
        self.extractor = ext(max_num_keypoints=MAX_KEYPOINTS).eval().to(device)
        self.matcher = LG.LightGlue(features=kind).eval().to(device)
        self.rbd = __import__("lightglue.utils", fromlist=["rbd"]).rbd

    def __call__(self, g1, g2, c1, c2):
        torch = self.torch

        def t(bgr):
            return torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().to(self.device) / 255.0

        with torch.inference_mode():
            f1 = self.extractor.extract(t(c1))
            f2 = self.extractor.extract(t(c2))
            m = self.matcher({"image0": f1, "image1": f2})
        f1, f2, m = [self.rbd(x) for x in (f1, f2, m)]
        idx = m["matches"].cpu().numpy()
        return (f1["keypoints"].cpu().numpy()[idx[:, 0]], f2["keypoints"].cpu().numpy()[idx[:, 1]])


class KorniaLoFTR:
    def __init__(self, device):
        import torch
        import kornia.feature as KF
        self.torch, self.device = torch, device
        self.model = KF.LoFTR(pretrained="outdoor").to(device).eval()

    def __call__(self, g1, g2, c1, c2):
        torch = self.torch

        def t(g):
            return torch.from_numpy(g).float()[None, None].to(self.device) / 255.0

        with torch.inference_mode():
            r = self.model({"image0": t(g1), "image1": t(g2)})
        k1, k2 = r["keypoints0"].cpu().numpy(), r["keypoints1"].cpu().numpy()
        keep = r["confidence"].cpu().numpy() >= LOFTR_MIN_CONF
        return k1[keep], k2[keep]


class RoMa:
    """RoMa outdoor (romatch 0.1.2), its default resolution; 5,000 certainty-balanced
    samples (romatch's own sampler), seeded."""

    def __init__(self, device):
        import torch
        from romatch import roma_outdoor
        self.torch, self.device = torch, device
        torch.manual_seed(H.SEED)
        self.model = roma_outdoor(device=device)

    def __call__(self, g1, g2, c1, c2):
        from PIL import Image
        torch = self.torch
        a = Image.fromarray(c1[:, :, ::-1].copy())
        b = Image.fromarray(c2[:, :, ::-1].copy())
        torch.manual_seed(H.SEED)
        with torch.inference_mode():
            warp, cert = self.model.match(a, b, device=self.device)
            m, _ = self.model.sample(warp, cert, num=ROMA_SAMPLES)
            k1, k2 = self.model.to_pixel_coordinates(m, H.VIEW_H, H.VIEW_W, H.VIEW_H, H.VIEW_W)
        return k1.cpu().numpy(), k2.cpu().numpy()

    def centre(self, c1, c2):
        """RoMa's dense A->B warp read at the source view's centre (the GT point), by
        bilinear interpolation of the warp grid, plus the certainty there. Returns
        (u, v, certainty) in other-view pixels. No planar assumption."""
        from PIL import Image
        import torch.nn.functional as F
        torch = self.torch
        a = Image.fromarray(c1[:, :, ::-1].copy())
        b = Image.fromarray(c2[:, :, ::-1].copy())
        with torch.inference_mode():
            warp, cert = self.model.match(a, b, device=self.device)
            if warp.dim() == 4:                       # batched (1, H, W, 4)
                warp, cert = warp[0], cert[0]
            ws = warp.shape[1] // 2 if self.model.symmetric else warp.shape[1]
            ab = warp[:, :ws, 2:].permute(2, 0, 1)[None].float()          # 1x2xHxW
            cc = cert[:, :ws][None, None].float()
            g = torch.zeros(1, 1, 1, 2, device=ab.device)                # A's centre
            xy = F.grid_sample(ab, g, align_corners=False)[0, :, 0, 0].cpu().numpy()
            c = float(F.grid_sample(cc, g, align_corners=False)[0, 0, 0, 0])
        return float((xy[0] + 1.0) / 2.0 * H.VIEW_W), float((xy[1] + 1.0) / 2.0 * H.VIEW_H), c


class KorniaAliked:
    """The pilot's own ALIKED + LightGlue (crossview_arms.matching.LightGlue), so ``lg_*``
    estimator variants reuse exactly the matches ``lg`` fitted."""

    def __init__(self, device):
        from crossview_arms.matching import LightGlue
        self.m = LightGlue(device)

    def __call__(self, g1, g2, c1, c2):
        return self.m.match(g1, g2)


MATCHERS = {
    "lg": KorniaAliked,
    "sp_lg": lambda dev: CvgLightGlue("superpoint", dev),
    "disk_lg": lambda dev: CvgLightGlue("disk", dev),
    "siftlg": lambda dev: CvgLightGlue("sift", dev),
    "loftr": KorniaLoFTR,
    "roma": RoMa,
}


def _extra(ctx, key, default=None):
    for kv in getattr(ctx.args, "extra", None) or []:
        k, _, v = kv.partition("=")
        if k == key:
            return v
    return default


def matches(pair, ctx, matcher):
    """All matches (no ground filter) for one pair, cached in memory and optionally on disk."""
    cache_dir = _extra(ctx, "match_cache")
    path = os.path.join(cache_dir, matcher, f"{pair['pair_id']}.npz") if cache_dir else None
    if path and os.path.exists(path):
        z = np.load(path)
        return z["k1"], z["k2"]
    key = ("pm_matcher", matcher)
    if key not in ctx.cache:
        ctx.cache[key] = MATCHERS[matcher](_device(ctx))
    import cv2
    c1, c2 = ctx.view(pair, "src"), ctx.view(pair, "oth")
    g1, g2 = cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(c2, cv2.COLOR_BGR2GRAY)
    k1, k2 = ctx.cache[key](g1, g2, c1, c2)
    k1 = np.asarray(k1, dtype=np.float32).reshape(-1, 2)
    k2 = np.asarray(k2, dtype=np.float32).reshape(-1, 2)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, k1=k1, k2=k2)
    return k1, k2


def roma_centre(pair, ctx):
    """(u, v, certainty) of RoMa's dense warp at the GT point; cached like ``matches`` in
    its own subdirectory, so the ``roma`` sample cache is untouched."""
    cache_dir = _extra(ctx, "match_cache")
    path = os.path.join(cache_dir, "roma_centre", f"{pair['pair_id']}.npz") if cache_dir else None
    if path and os.path.exists(path):
        return tuple(float(x) for x in np.load(path)["uvc"])
    key = ("pm_matcher", "roma")
    if key not in ctx.cache:
        ctx.cache[key] = RoMa(_device(ctx))
    uvc = ctx.cache[key].centre(ctx.view(pair, "src"), ctx.view(pair, "oth"))
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, uvc=np.array(uvc))
    return uvc


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #


def auto_prior(pair, ctx):
    """The committed proj_height_auto point for this pair, (x, y) equirect."""
    if "pm_auto" not in ctx.cache:
        path, _ = H.prediction_paths(AUTO_ARM)
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        ctx.cache["pm_auto"] = {r["pair_id"]: r for r in rows}
    r = ctx.cache["pm_auto"].get(pair["pair_id"])
    if r is None or r.get("x") is None:
        return pair["proj_x"], pair["proj_y"]
    return float(r["x"]), float(r["y"])


def _magsac_flag(cv2):
    """MAGSAC++ in OpenCV: USAC_MAGSAC (OpenCV 5) / USAC_MAGIC (OpenCV 4)."""
    return getattr(cv2, "USAC_MAGSAC", None) or getattr(cv2, "USAC_MAGIC")


def _homography(a, b, method):
    import cv2
    if len(a) < 4:
        return None, 0
    cv2.setRNGSeed(H.SEED)
    flag = cv2.RANSAC if method == "ransac" else _magsac_flag(cv2)
    thr = RANSAC_PX if method == "ransac" else MAGSAC_PX
    Hm, m = cv2.findHomography(np.float32(a), np.float32(b), flag, thr, maxIters=5000,
                               confidence=0.999)
    if Hm is None or m is None:
        return None, 0
    return Hm, int(m.sum())


def ground_homography(pair, ctx, k1, k2, method="ransac"):
    """The pilot's estimator (method 'ransac') or MAGSAC++. Returns (out dict, diag)."""
    a, b = filter_ground(k1, k2, ctx.view_centre(pair, "src"), ctx.view_centre(pair, "oth"),
                         GROUND_MARGIN_DEG)
    diag = {"n_matches": int(len(k1)), "n_ground": int(len(a))}
    if len(a) < 4:
        return None, {**diag, "inliers": 0, "why": "no_ground_matches"}
    Hm, n_in = _homography(a, b, method)
    diag["inliers"] = n_in
    uv = H.map_point(Hm, *CENTRE)
    if Hm is None or n_in < MIN_INLIERS:
        return None, {**diag, "why": "few_inliers"}
    if uv is None or not (0 <= uv[0] < H.VIEW_W and 0 <= uv[1] < H.VIEW_H):
        return None, {**diag, "why": "mapped_outside_view"}
    x, y = H.view_to_pano(uv[0], uv[1], *ctx.view_centre(pair, "oth"))
    return {"x": float(x), "y": float(y), "u": uv[0], "v": uv[1], "via": "homography"}, diag


def epipolar_snap(pair, ctx, k1, k2):
    """Essential matrix from all matches (known focal, principal point at the view centre),
    MAGSAC; the GT point's epipolar line in the other view; the auto-height prior moved to
    the nearest point on that line. Returns (out dict | None, diag)."""
    import cv2
    diag = {"n_matches": int(len(k1))}
    if len(k1) < 8:
        return None, {**diag, "epi_inliers": 0, "why": "epi_few_matches"}
    f = H.focal_px()
    K = np.array([[f, 0, H.VIEW_W / 2.0], [0, f, H.VIEW_H / 2.0], [0, 0, 1.0]])
    cv2.setRNGSeed(H.SEED)
    E, m = cv2.findEssentialMat(np.float64(k1), np.float64(k2), K, method=_magsac_flag(cv2),
                                prob=0.999, threshold=EPI_PX)
    if E is None or m is None or E.shape != (3, 3):
        return None, {**diag, "epi_inliers": 0, "why": "epi_no_model"}
    n_in = int(m.sum())
    diag["epi_inliers"] = n_in
    if n_in < EPI_MIN_INLIERS:
        return None, {**diag, "why": "epi_few_inliers"}
    Ki = np.linalg.inv(K)
    F = Ki.T @ E @ Ki
    line = F @ np.array([CENTRE[0], CENTRE[1], 1.0])
    nrm = np.hypot(line[0], line[1])
    if not np.isfinite(nrm) or nrm < 1e-12:
        return None, {**diag, "why": "epi_degenerate"}
    line = line / nrm
    px, py = auto_prior(pair, ctx)
    u0, v0, front = H.pano_to_view(px, py, *ctx.view_centre(pair, "oth"))
    if not front:
        return None, {**diag, "why": "epi_prior_behind"}
    d = line[0] * u0 + line[1] * v0 + line[2]
    u, v = float(u0 - d * line[0]), float(v0 - d * line[1])
    diag["epi_shift_px"] = float(abs(d))
    if not (0 <= u < H.VIEW_W and 0 <= v < H.VIEW_H):
        return None, {**diag, "why": "epi_outside_view"}
    x, y = H.view_to_pano(u, v, *ctx.view_centre(pair, "oth"))
    return {"x": float(x), "y": float(y), "u": u, "v": v, "via": "epipolar"}, diag


def run(pair, ctx, matcher, estimator):
    k1, k2 = matches(pair, ctx, matcher)
    if estimator in ("ransac", "magsac"):
        out, diag = ground_homography(pair, ctx, k1, k2, estimator)
        return {**diag, **out} if out else {"x": None, "y": None, **diag}
    if estimator == "epi":
        out, diag = epipolar_snap(pair, ctx, k1, k2)
        return {**diag, **out} if out else {"x": None, "y": None, **diag}
    if estimator == "hyb":
        out, d1 = ground_homography(pair, ctx, k1, k2, "ransac")
        if out:
            return {**d1, **out}
        out, d2 = epipolar_snap(pair, ctx, k1, k2)
        diag = {**d1, **d2, "why_h": d1.get("why"), "why": d2.get("why")}
        if out:
            return {**diag, **out}
        x, y = auto_prior(pair, ctx)
        return {**diag, "x": x, "y": y, "via": "auto_prior"}
    raise ValueError(estimator)


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #

MATCHER_DESC = {
    "lg": "ALIKED + LightGlue (kornia 0.8.3; the pilot's lg matches)",
    "sp_lg": "SuperPoint + LightGlue (cvg eb42fee, 4096 kp)",
    "disk_lg": "DISK + LightGlue (cvg eb42fee, 4096 kp)",
    "siftlg": "SIFT + LightGlue (cvg eb42fee, 4096 kp)",
    "loftr": "LoFTR outdoor (kornia 0.8.3), conf >= 0.5",
    "roma": "RoMa outdoor (romatch 0.1.2), 5000 samples",
}
EST_DESC = {
    "ransac": "ground homography, RANSAC (as lg)",
    "magsac": "ground homography, MAGSAC++",
    "epi": "essential matrix from all matches; auto-height prior snapped to the epipolar line",
    "hyb": "ground homography, else epipolar snap, else the auto-height prior",
}
BASE_CONFIG = {"ground_margin_deg": GROUND_MARGIN_DEG, "ransac_px": RANSAC_PX,
               "min_inliers": MIN_INLIERS, "rig_limit_deg": H.RIG_LIMIT_DEG,
               "view": [H.VIEW_W, H.VIEW_H, H.HFOV_DEG], "pre_specified": True}


def _register(matcher, estimator):
    name = matcher if estimator == "ransac" else f"{matcher}_{estimator}"
    cfg = {**BASE_CONFIG, "matcher": MATCHER_DESC[matcher], "estimator": EST_DESC[estimator]}
    if matcher == "loftr":
        cfg["loftr_min_conf"] = LOFTR_MIN_CONF
    if matcher == "roma":
        cfg["roma_samples"] = ROMA_SAMPLES
    if matcher in ("sp_lg", "disk_lg", "siftlg"):
        cfg["max_keypoints"] = MAX_KEYPOINTS
    if estimator in ("epi", "hyb"):
        cfg.update(epi_px=EPI_PX, epi_min_inliers=EPI_MIN_INLIERS, prior=AUTO_ARM)
    if estimator == "magsac":
        cfg["magsac_px"] = MAGSAC_PX
    needs = ("views",)

    @register(name, needs=needs, config=cfg,
              description=f"{MATCHER_DESC[matcher]}; {EST_DESC[estimator]}")
    def arm(pair, ctx):
        return run(pair, ctx, matcher, estimator)
    arm.__name__ = name.replace(".", "_")
    return arm


for _m in MATCHERS:
    for _e in EST_DESC:
        if (_m, _e) != ("lg", "ransac"):      # that one is the pilot's committed `lg`
            _register(_m, _e)


# --------------------------------------------------------------------------- #
# RoMa-only arms: a local homography, and the dense warp read at the GT point
# --------------------------------------------------------------------------- #

ROMA_CERT_MIN = 0.05     # romatch's own sample_thresh: its definition of a usable match
LOCAL_RADIUS_PX = 160.0  # inherited from lg_local
LOCAL_MIN = 12


@register("roma_local", needs=("views",),
          config={**BASE_CONFIG, "matcher": MATCHER_DESC["roma"], "roma_samples": ROMA_SAMPLES,
                  "local_radius_px": LOCAL_RADIUS_PX, "local_min": LOCAL_MIN,
                  "estimator": "ground homography from matches within 160 px of the GT point"},
          description="RoMa; ground homography fitted only to matches within 160 px of the GT "
                      "point (as lg_local)")
def roma_local(pair, ctx):
    k1, k2 = matches(pair, ctx, "roma")
    a, b = filter_ground(k1, k2, ctx.view_centre(pair, "src"), ctx.view_centre(pair, "oth"),
                         GROUND_MARGIN_DEG)
    if len(a):
        near = np.hypot(a[:, 0] - CENTRE[0], a[:, 1] - CENTRE[1]) < LOCAL_RADIUS_PX
    else:
        near = np.zeros(0, bool)
    diag = {"n_matches": int(len(k1)), "n_ground": int(len(a)), "n_near": int(near.sum())}
    if near.sum() < LOCAL_MIN:
        return {"x": None, "y": None, **diag, "why": "few_near_matches"}
    out, d = ground_homography(pair, ctx, a[near], b[near])
    diag["inliers"] = d.get("inliers")
    return {**diag, **out} if out else {"x": None, "y": None, **diag, "why": d.get("why")}


def _warp(pair, ctx, hybrid):
    u, v, c = roma_centre(pair, ctx)
    diag = {"certainty": c, "u": u, "v": v}
    if c >= ROMA_CERT_MIN and 0 <= u < H.VIEW_W and 0 <= v < H.VIEW_H:
        x, y = H.view_to_pano(u, v, *ctx.view_centre(pair, "oth"))
        return {**diag, "x": float(x), "y": float(y), "via": "warp"}
    why = "low_certainty" if c < ROMA_CERT_MIN else "warp_outside_view"
    if hybrid:
        x, y = auto_prior(pair, ctx)
        return {**diag, "x": x, "y": y, "via": "auto_prior", "why": why}
    return {**diag, "x": None, "y": None, "why": why}


WARP_CONFIG = {"matcher": "RoMa outdoor (romatch 0.1.2), dense warp at the GT point",
               "cert_min": ROMA_CERT_MIN, "view": [H.VIEW_W, H.VIEW_H, H.HFOV_DEG],
               "pre_specified": True}


@register("roma_warp", needs=("views",), config=WARP_CONFIG,
          description="RoMa dense warp read at the GT point (no plane); falls back below "
                      "certainty 0.05")
def roma_warp(pair, ctx):
    return _warp(pair, ctx, hybrid=False)


@register("roma_warp_hyb", needs=("views",), config={**WARP_CONFIG, "prior": AUTO_ARM},
          description="roma_warp, else the auto-height prior")
def roma_warp_hyb(pair, ctx):
    return _warp(pair, ctx, hybrid=True)
