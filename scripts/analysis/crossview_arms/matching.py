"""Image-matching arms: ALIKED + LightGlue (learned), SIFT and NCC (classical baselines).

Each matches the source view (centred on the GT point) against the other view (centred on
today's projection), keeps matches on the ground, fits a RANSAC homography and maps the
source view's centre -- the GT point -- through it. Too few inliers, or a mapped point
outside the view, returns None (fall back to the projection).

The ground band: keypoints must be at least ``margin`` degrees below the pano-frame horizon
in both views and above the rig. The pre-specified band was 0.5 deg (``lg_band0.5``); it
admitted far-field points whose homography is near a pure rotation and does not transfer to
a ramp 5-18 m away. 5 deg (flat ground within ~30 m of a 2.6 m camera) was chosen after
looking at those failures, so it is post hoc. Both are registered.
"""
import numpy as np

import crossview_align_48 as H
from crossview_arms._registry import register

RANSAC_PX = 4.0
MIN_INLIERS = 15
LOCAL_RADIUS_PX = 160.0
LOCAL_MIN = 12
NCC_TEMPLATE_PX = 64
NCC_MIN = 0.5
NCC_WINDOW_PX = 200        # search +-200 px (~+-16 deg) around the projected point
SIFT_FEATURES = 4000
SIFT_RATIO = 0.8
CENTRE = (H.VIEW_W / 2.0, H.VIEW_H / 2.0)


def fit_homography(p_src, p_oth, seed=H.SEED):
    """RANSAC homography (seeded). Returns (H, n_inliers, inlier_mask) or (None, 0, None)."""
    import cv2
    if len(p_src) < 4:
        return None, 0, None
    cv2.setRNGSeed(seed)
    Hm, m = cv2.findHomography(np.float32(p_src), np.float32(p_oth), cv2.RANSAC, RANSAC_PX,
                               maxIters=5000, confidence=0.999)
    if Hm is None or m is None:
        return None, 0, None
    return Hm, int(m.sum()), m.ravel().astype(bool)


def filter_ground(a, b, src_view, oth_view, margin):
    if len(a) == 0:
        return a, b
    keep = H.ground_mask(a[:, 0], a[:, 1], *src_view, margin=margin) & \
        H.ground_mask(b[:, 0], b[:, 1], *oth_view, margin=margin)
    return a[keep], b[keep]


def map_centre(a, b, oth_view, min_inliers, local=False):
    """Fit on all ground matches (or, if ``local``, only those within LOCAL_RADIUS_PX of the
    GT point in the source view) and map the GT point. Returns the arm's output dict."""
    diag = {"n_ground": int(len(a))}
    if local:
        d = np.hypot(a[:, 0] - CENTRE[0], a[:, 1] - CENTRE[1]) if len(a) else np.zeros(0)
        sel = d < LOCAL_RADIUS_PX
        diag["n_near"] = int(sel.sum())
        if sel.sum() < LOCAL_MIN:
            return None
        a, b = a[sel], b[sel]
    Hm, n_in, _ = fit_homography(a, b)
    diag["inliers"] = n_in
    uv = H.map_point(Hm, *CENTRE)
    if uv is None or n_in < min_inliers or not (0 <= uv[0] < H.VIEW_W and 0 <= uv[1] < H.VIEW_H):
        return None
    x, y = H.view_to_pano(uv[0], uv[1], *oth_view)
    return {"x": float(x), "y": float(y), "u": uv[0], "v": uv[1], **diag}


class LightGlue:
    """ALIKED (aliked-n16) + LightGlue from kornia. Keypoints are NOT truncated: ALIKED
    returns them in raster order, so a [:N] slice drops the bottom of the view -- the
    ground -- which is exactly what an early version of this pilot did."""

    def __init__(self, device):
        import kornia.feature as KF
        import torch
        self.torch, self.KF, self.device = torch, KF, device
        torch.manual_seed(H.SEED)
        self.extractor = KF.ALIKED.from_pretrained("aliked-n16", device=device).eval()
        self.matcher = KF.LightGlueMatcher("aliked").to(device).eval()

    def features(self, gray):
        t = self.torch.from_numpy(gray).float()[None, None].to(self.device) / 255.0
        t = t.repeat(1, 3, 1, 1)
        with self.torch.inference_mode():
            f = self.extractor(t)[0]
        return f.keypoints, f.descriptors

    def match(self, g1, g2):
        KF, torch = self.KF, self.torch
        k1, d1 = self.features(g1)
        k2, d2 = self.features(g2)
        lafs1 = KF.laf_from_center_scale_ori(k1[None], torch.ones(1, len(k1), 1, 1, device=self.device))
        lafs2 = KF.laf_from_center_scale_ori(k2[None], torch.ones(1, len(k2), 1, 1, device=self.device))
        with torch.inference_mode():
            _, idx = self.matcher(d1, d2, lafs1, lafs2, hw1=g1.shape[:2], hw2=g2.shape[:2])
        idx = idx.cpu().numpy()
        return k1.cpu().numpy()[idx[:, 0]], k2.cpu().numpy()[idx[:, 1]]


def _gray_pair(pair, ctx):
    import cv2
    return (ctx.view(pair, "src", cv2.IMREAD_GRAYSCALE), ctx.view(pair, "oth", cv2.IMREAD_GRAYSCALE))


def _lightglue_matches(pair, ctx):
    if "lightglue" not in ctx.cache:
        import torch
        dev = "cuda" if torch.cuda.is_available() and not getattr(ctx.args, "cpu", False) else "cpu"
        ctx.cache["lightglue"] = LightGlue(dev)
    return ctx.cache["lightglue"].match(*_gray_pair(pair, ctx))


def _lg(pair, ctx, margin, local):
    a, b = _lightglue_matches(pair, ctx)
    a, b = filter_ground(a, b, ctx.view_centre(pair, "src"), ctx.view_centre(pair, "oth"), margin)
    return map_centre(a, b, ctx.view_centre(pair, "oth"), MIN_INLIERS, local=local)


LG_CONFIG = {"extractor": "ALIKED aliked-n16 (kornia 0.8.3), untruncated",
             "matcher": "LightGlue (kornia)", "ransac_px": RANSAC_PX,
             "min_inliers": MIN_INLIERS, "rig_limit_deg": H.RIG_LIMIT_DEG,
             "view": [H.VIEW_W, H.VIEW_H, H.HFOV_DEG]}


@register("lg", needs=("views",), config={**LG_CONFIG, "ground_margin_deg": 5.0},
          description="ALIKED+LightGlue, ground homography (band 5 deg, post hoc)")
def lg(pair, ctx):
    return _lg(pair, ctx, 5.0, local=False)


@register("lg_local", needs=("views",),
          config={**LG_CONFIG, "ground_margin_deg": 5.0, "local_radius_px": LOCAL_RADIUS_PX,
                  "local_min": LOCAL_MIN},
          description="ALIKED+LightGlue, homography from matches within 160 px of the GT point")
def lg_local(pair, ctx):
    return _lg(pair, ctx, 5.0, local=True)


@register("lg_band0.5", needs=("views",), config={**LG_CONFIG, "ground_margin_deg": 0.5},
          description="ALIKED+LightGlue, ground band 0.5 deg (the pre-specified setting)")
def lg_band05(pair, ctx):
    return _lg(pair, ctx, 0.5, local=False)


def sift_match(g1, g2):
    import cv2
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2))
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < SIFT_RATIO * n.distance]
    a = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    b = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return a, b


@register("sift", needs=("views",),
          config={"features": SIFT_FEATURES, "ratio": SIFT_RATIO, "ransac_px": RANSAC_PX,
                  "min_inliers": MIN_INLIERS, "ground_margin_deg": 5.0},
          description="OpenCV SIFT + ratio test, same ground band / RANSAC / fallback as lg")
def sift(pair, ctx):
    a, b = sift_match(*_gray_pair(pair, ctx))
    a, b = filter_ground(a, b, ctx.view_centre(pair, "src"), ctx.view_centre(pair, "oth"), 5.0)
    return map_centre(a, b, ctx.view_centre(pair, "oth"), MIN_INLIERS)


def ncc_search(g1, g2, scale):
    """Template of the source view's centre, rescaled by the range ratio, searched in a
    window of +-NCC_WINDOW_PX around the other view's centre (the projected point), i.e.
    inside the projection's uncertainty. Returns ((u, v) in the full view, peak score)."""
    import cv2
    t = NCC_TEMPLATE_PX
    c1 = (g1.shape[1] // 2, g1.shape[0] // 2)
    tpl = g1[c1[1] - t // 2:c1[1] + t // 2, c1[0] - t // 2:c1[0] + t // 2]
    s = float(np.clip(scale, 0.25, 4.0))
    size = max(8, int(round(t * s)))
    tpl = cv2.resize(tpl, (size, size), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    c2 = (g2.shape[1] // 2, g2.shape[0] // 2)
    x0 = max(0, c2[0] - NCC_WINDOW_PX - size // 2)
    y0 = max(0, c2[1] - NCC_WINDOW_PX - size // 2)
    win = g2[y0:c2[1] + NCC_WINDOW_PX + size // 2, x0:c2[0] + NCC_WINDOW_PX + size // 2]
    if tpl.shape[0] >= win.shape[0] or tpl.shape[1] >= win.shape[1]:
        return None, 0.0
    res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    return (x0 + loc[0] + size / 2.0, y0 + loc[1] + size / 2.0), float(mx)


@register("ncc", needs=("views",),
          config={"template_px": NCC_TEMPLATE_PX, "window_px": NCC_WINDOW_PX, "min_score": NCC_MIN},
          description="range-scaled NCC template search within +-200 px of the projection")
def ncc(pair, ctx):
    g1, g2 = _gray_pair(pair, ctx)
    uv, s = ncc_search(g1, g2, pair["src_range_m"] / max(pair["oth_range_m"], 0.5))
    if uv is None or s < NCC_MIN:
        return None
    x, y = H.view_to_pano(uv[0], uv[1], *ctx.view_centre(pair, "oth"))
    return {"x": float(x), "y": float(y), "score": s}
