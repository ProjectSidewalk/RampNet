"""Image-based estimate of an equirectangular panorama's camera tilt (pitch and roll).

Why: open-imagery panoramas (Panoramax, Mapillary) often carry no pose, or a pose of exactly
0.0 / 0.0 that the camera writes whether or not the image is level (GoPro Max writes
``Xmp.GPano.PosePitchDegrees`` = 0 regardless). The tilt has to come from the pixels. See
``docs/pano_tilt_estimate.md`` for the validation against Google's measured pose and the caveats.

Method (``METHOD_VERSION``): the zenith vanishing point of near-vertical line segments.

1. The equirect is reduced to a grey image ``work_width`` wide (default 4096).
2. Eight gnomonic views are rendered around the horizon, one every 45 degrees of bearing, each
   60 degrees wide and covering elevation -45 to +60 degrees. A pitch-0 pinhole view shows a
   gravity-vertical line as a vertical line when the rig is level, and leaning toward the zenith
   vanishing point when it is not.
3. OpenCV's LSD line segment detector runs on each view. A segment is kept if it is at least
   ``min_len_deg`` long, within ``max_lean_deg`` of the view's vertical, and its midpoint falls in
   the view's own 45-degree sector (so overlapping views do not count a segment twice).
4. Each kept segment's two endpoint rays (rig frame) span a plane through the camera centre; its
   unit normal ``n`` is orthogonal to the gravity "up" direction ``u`` if the segment is a
   gravity-vertical edge (a building corner, a pole, a door frame). ``u`` is found by RANSAC over
   segment pairs (``u = n_i x n_j``), then refined as the smallest-eigenvalue eigenvector of
   ``sum w n n^T`` over the inliers. Residual of a segment: ``asin(|n . u|)`` in degrees.
5. ``u`` is converted to (pitch, roll) by inverting :func:`rampnet.stage1_geometry.rig_from_gravity`,
   so the output uses the #113 convention unchanged: pitch > 0 is nose down, roll > 0 is left side
   up, and the gravity horizon at bearing ``b`` sits ``T(b) = pitch cos b + roll sin b`` degrees
   above the image's middle row, with ``b = x / w * 360 - 180``.

The estimate reports its own reliability: the number of inlier segments, how many of the eight
45-degree sectors they cover, and a formal standard error from the weighted least-squares fit.
:func:`estimate_tilt` returns ``status = "ok"`` or ``"declined"`` with the reason; a declined pano
still carries the raw numbers so the gate can be re-tuned without re-running.

Usage::

    from rampnet.pano_tilt import load_equirect_gray, estimate_tilt
    gray = load_equirect_gray("pano.jpg")            # (2048, 4096) float32 in [0, 1]
    est = estimate_tilt(gray)
    if est.status == "ok":
        print(est.pitch_deg, est.roll_deg, est.se_pitch_deg, est.se_roll_deg)
    else:
        print("declined:", est.reason)

Pure CPU (numpy, OpenCV, Pillow). Deterministic for a given image, parameter set and OpenCV build:
RANSAC uses a fixed seed. LSD output can differ slightly between OpenCV builds, so a re-run on
another machine reproduces the estimates closely but not bit for bit; how closely was not
measured (only one machine was used, docs/pano_tilt_estimate.md section 6).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from rampnet import stage1_geometry as sg

#: Bumped whenever the algorithm or a default parameter changes, so a stored estimate says what
#: produced it.
METHOD_VERSION = "pano_tilt v1 (lsd zenith-vp, 8 views, sector-sqrt ransac)"


@dataclass(frozen=True)
class TiltParams:
    """Every tunable of the estimator, with the defaults ``METHOD_VERSION`` names."""
    work_width: int = 4096          # equirect is reduced to this width (height = width / 2)
    n_views: int = 8                # views around the horizon, evenly spaced in bearing
    view_hfov_deg: float = 60.0     # horizontal field of view of each view
    el_min_deg: float = -45.0       # lowest elevation rendered (keeps the vehicle and road out)
    el_max_deg: float = 60.0        # highest elevation rendered
    min_len_deg: float = 1.5        # shortest segment kept, as the angle between its endpoint rays
    max_lean_deg: float = 35.0      # segments leaning more than this from the view's vertical are dropped
    ransac_iters: int = 1500
    ransac_tol_deg: float = 0.75    # inlier threshold on asin(|n . u|) when scoring hypotheses
    refine_tol_deg: float = 0.35    # inlier threshold during refinement
    max_tilt_deg: float = 40.0      # u must be within this of the rig's up axis
    weight_cap_deg: float = 15.0    # segment weight = min(length_deg, cap)
    seed: int = 0
    # decline gate (docs/pano_tilt_estimate.md section 2; chosen on the GSV development quarter,
    # section 3.1, by a sweep that was not committed). `gate_reasons` applies it; the script's
    # `gate_sensitivity` re-applies it to the committed rows at other thresholds.
    min_inliers: int = 12
    sector_min_inliers: int = 3     # a sector "counts" when it holds at least this many inliers
    min_sectors: int = 3            # how many sectors must count
    # A backstop, not a working filter: no committed row comes near it (largest formal SE 0.165
    # deg on the 571 GSV panos, 0.195 deg on the 200 Lyon panos), so it declines nothing there.
    # Kept so a degenerate fit (e.g. a singular one, SE = inf) is always declined.
    max_se_deg: float = 0.25        # largest formal SE of pitch or roll that is accepted
    ambiguity_sep_deg: float = 3.0  # rival hypotheses are those at least this far from the fit
    max_ambiguity: float = 0.8      # best rival's score / the fit's score; above this, decline


@dataclass
class TiltEstimate:
    """One pano's result. Angles in degrees, in the #113 convention (module docstring)."""
    status: str                     # "ok" or "declined"
    reason: str                     # "" when ok
    pitch_deg: Optional[float]
    roll_deg: Optional[float]
    se_pitch_deg: Optional[float]
    se_roll_deg: Optional[float]
    n_segments: int                 # candidate segments after the length / lean filter
    n_inliers: int
    n_sectors: int                  # sectors (of n_views) holding >= sector_min_inliers inliers
    residual_rms_deg: Optional[float]
    ambiguity: Optional[float]      # best rival hypothesis's score / the fit's score (lower is better)
    sector_inliers: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------------------------
# Image loading and view rendering
# --------------------------------------------------------------------------------------------

def load_equirect_gray(path, work_width: int = 4096) -> np.ndarray:
    """Read a JPEG/PNG equirect as float32 grey in [0, 1], resized to ``work_width`` x ``work_width/2``.

    Uses Pillow's JPEG draft mode to decode a large JPEG at a reduced scale (fast), then an
    area (box) resize to the exact working size.
    """
    from PIL import Image
    w_out, h_out = int(work_width), int(work_width) // 2
    old_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None    # local, hash-pinned panos; a 16384 x 8192 GSV pano trips the guard
    try:
        with Image.open(path) as im:
            if im.format == "JPEG":
                im.draft("L", (w_out, h_out))
            g = im.convert("L")
            if g.size != (w_out, h_out):
                g = g.resize((w_out, h_out), Image.Resampling.BOX)
            return np.asarray(g, dtype=np.float32) / 255.0
    finally:
        Image.MAX_IMAGE_PIXELS = old_limit


def gray_from_array(img: np.ndarray, work_width: int = 4096) -> np.ndarray:
    """Same as :func:`load_equirect_gray` for an in-memory (H, W) or (H, W, 3) uint8/float array."""
    from PIL import Image
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = np.clip(a * (255.0 if a.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    im = Image.fromarray(a)
    g = im.convert("L")
    w_out, h_out = int(work_width), int(work_width) // 2
    if g.size != (w_out, h_out):
        g = g.resize((w_out, h_out), Image.Resampling.BOX)
    return np.asarray(g, dtype=np.float32) / 255.0


@dataclass(frozen=True)
class ViewGeom:
    """A pitch-0 gnomonic view at bearing ``yaw_deg`` covering elevations [el_min, el_max]."""
    yaw_deg: float
    focal_px: float
    width: int
    height: int
    cx: float
    cy: float

    def basis(self):
        """(forward, right, up) unit vectors in the rig's RFU frame (stage1_geometry axes)."""
        y = math.radians(self.yaw_deg)
        fwd = np.array([math.sin(y), math.cos(y), 0.0])
        right = np.array([math.cos(y), -math.sin(y), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        return fwd, right, up

    def rays(self, u, v):
        """Unit RFU rays through view pixel coordinates (u right, v down); arrays broadcast."""
        fwd, right, up = self.basis()
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        d = (self.focal_px * fwd + (u - self.cx)[..., None] * right
             + (self.cy - v)[..., None] * up)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)


def make_views(params: TiltParams, pano_width: int):
    """The ``n_views`` views, with a focal length matching the equirect's angular resolution."""
    rad_per_px = 2.0 * math.pi / pano_width
    f = 1.0 / rad_per_px
    half_w = math.tan(math.radians(params.view_hfov_deg / 2.0))
    width = int(round(2 * f * half_w))
    top = math.tan(math.radians(params.el_max_deg))
    bottom = math.tan(math.radians(-params.el_min_deg))
    height = int(round(f * (top + bottom)))
    step = 360.0 / params.n_views
    return [ViewGeom(yaw_deg=k * step if k * step <= 180 else k * step - 360, focal_px=f,
                     width=width, height=height, cx=width / 2.0, cy=f * top)
            for k in range(params.n_views)]


def render_view(gray: np.ndarray, view: ViewGeom) -> np.ndarray:
    """Sample the equirect into the view (bilinear, x wraps). Returns float32 (height, width)."""
    import cv2
    h, w = gray.shape
    uu, vv = np.meshgrid(np.arange(view.width) + 0.5, np.arange(view.height) + 0.5)
    rays = view.rays(uu, vv)
    b, el = sg.bearing_elevation(rays)
    x, y = sg.pixel_from_bearing_elevation(b, el, w, h)
    # pixel_from_bearing_elevation is continuous with pixel i spanning [i, i+1); remap samples
    # integer coordinates at pixel centres, hence the -0.5.
    mx = (x - 0.5).astype(np.float32)
    my = (y - 0.5).astype(np.float32)
    return cv2.remap(gray, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


# --------------------------------------------------------------------------------------------
# Segments
# --------------------------------------------------------------------------------------------

def detect_segments(view_img: np.ndarray) -> np.ndarray:
    """LSD segments of a float [0, 1] view, as an (N, 4) array of (x1, y1, x2, y2) view pixels."""
    import cv2
    img8 = np.clip(view_img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(img8)[0]
    if lines is None:
        return np.zeros((0, 4), dtype=float)
    return lines.reshape(-1, 4).astype(float)


def segment_constraints(segs: np.ndarray, view: ViewGeom, params: TiltParams):
    """Turn one view's segments into great-circle normals, keeping those in the view's own sector.

    Returns a dict of arrays: ``normal`` (N, 3), ``length_deg``, ``bearing_deg`` (of the midpoint)
    and ``lean_deg`` (angle from the view's vertical). The length and lean filters are applied
    later, in :func:`estimate_from_constraints`, so a cache of constraints can be re-gated. LSD's
    coordinates use the same pixel-edge convention :func:`render_view` samples in (pixel i centred
    at i + 0.5), so the rays are taken at the coordinates as given.
    """
    if len(segs) == 0:
        z = np.zeros((0,))
        return {"normal": np.zeros((0, 3)), "length_deg": z, "bearing_deg": z, "lean_deg": z}
    x1, y1, x2, y2 = segs.T
    lean = np.degrees(np.arctan2(np.abs(x2 - x1), np.abs(y2 - y1)))      # 0 = image-vertical
    e1 = view.rays(x1, y1)
    e2 = view.rays(x2, y2)
    length = np.degrees(np.arccos(np.clip(np.sum(e1 * e2, axis=-1), -1.0, 1.0)))
    mb, _ = sg.bearing_elevation(view.rays((x1 + x2) / 2.0, (y1 + y2) / 2.0))
    mb = np.atleast_1d(mb)
    half = 180.0 / params.n_views
    keep = (np.abs(sg.wrap_deg(mb - view.yaw_deg)) <= half) & (length > 0)
    n = np.cross(e1[keep], e2[keep])
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return {"normal": n, "length_deg": length[keep], "bearing_deg": mb[keep], "lean_deg": lean[keep]}


# --------------------------------------------------------------------------------------------
# Up vector <-> pitch/roll (inverse of stage1_geometry.rig_from_gravity)
# --------------------------------------------------------------------------------------------

def up_from_pitch_roll(pitch_deg, roll_deg) -> np.ndarray:
    """Gravity "up" expressed in the rig frame: ``rig_from_gravity(pitch, roll) @ [0, 0, 1]``."""
    return sg.rig_from_gravity(pitch_deg, roll_deg) @ np.array([0.0, 0.0, 1.0])


def pitch_roll_from_up(u) -> tuple:
    """Inverse of :func:`up_from_pitch_roll` for a unit (or any positive-length) vector.

    ``rig_from_gravity(p, r) @ e_z = (-sin r cos p, -sin p, cos r cos p)``, so
    ``p = asin(-u_y)`` and ``r = atan2(-u_x, u_z)``. ``tests/test_pano_tilt.py`` round-trips this
    against :func:`rampnet.stage1_geometry.rig_from_gravity` so the two cannot drift apart.
    """
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -u[1]))))
    roll = math.degrees(math.atan2(-u[0], u[2]))
    return pitch, roll


# --------------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------------

def _residual_deg(normals, u):
    return np.degrees(np.arcsin(np.clip(np.abs(normals @ u), 0.0, 1.0)))


def _orient(u):
    u = u / np.linalg.norm(u)
    return -u if u[2] < 0 else u


def _weighted_null(normals, w):
    m = (normals * w[:, None]).T @ normals
    vals, vecs = np.linalg.eigh(m)
    return _orient(vecs[:, 0])


def sector_of(bearing_deg, n_views):
    """Index of the n_views-wide bearing sector (sector 0 centred on bearing 0) of each bearing."""
    step = 360.0 / n_views
    return np.floor(((np.asarray(bearing_deg) + step / 2.0) % 360.0) / step).astype(int) % n_views


def _sector_score(cand, normals, weights, sectors, tol):
    """Sum over sectors of sqrt(weight of the segments within ``tol`` (as sin) of each candidate)."""
    inl = np.abs(cand @ normals.T) < tol                        # (n_cand, n)
    score = np.zeros(len(cand))
    for k in np.unique(sectors):
        m = sectors == k
        score += np.sqrt(inl[:, m] @ weights[m])
    return score


def fit_up(normals: np.ndarray, weights: np.ndarray, sectors: np.ndarray, params: TiltParams):
    """RANSAC + eigen refinement of the gravity up vector.

    Returns ``(u, inlier mask, ambiguity)``, or ``(None, None, None)`` when nothing fits.

    A hypothesis is scored by ``sum over sectors of sqrt(inlier weight in that sector)``, not by
    the plain inlier weight: a cluster of parallel segments in one direction (a fence, a facade
    seen obliquely, a tree) converges somewhere too, and the square root makes support from
    many directions worth more than the same total from one. The rule was picked by comparing it
    with the plain inlier weight (docs/pano_tilt_estimate.md section 3.1); that comparison was
    not committed, so its numbers cannot be re-derived. What the committed rows do show, for this
    rule on the 144-pano development quarter before the decline gate: 1 fit more than 3 degrees
    off Google's pose and 17 more than 1 degree off (``validation_summary.json``,
    ``by_role.dev.all_estimates_ungated``).
    """
    n = len(normals)
    if n < 2:
        return None, None, None
    rng = np.random.default_rng(params.seed)
    p = weights / weights.sum()
    cos_max = math.cos(math.radians(params.max_tilt_deg))
    pairs = rng.choice(n, size=(params.ransac_iters, 2), p=p)
    c = np.cross(normals[pairs[:, 0]], normals[pairs[:, 1]])
    nc = np.linalg.norm(c, axis=1)
    ok = nc > 1e-6
    cand = c[ok] / nc[ok, None]
    cand = np.where(cand[:, 2:3] < 0, -cand, cand)
    cand = cand[cand[:, 2] >= cos_max]
    if len(cand) == 0:
        return None, None, None
    tol = math.sin(math.radians(params.ransac_tol_deg))
    score = _sector_score(cand, normals, weights, sectors, tol)
    best_u = cand[int(np.argmax(score))]               # first maximum: deterministic
    u = best_u
    inl = _residual_deg(normals, u) < params.ransac_tol_deg
    for _ in range(10):
        if inl.sum() < 2:
            return None, None, None
        u_new = _weighted_null(normals[inl], weights[inl])
        inl_new = _residual_deg(normals, u_new) < params.refine_tol_deg
        done = np.allclose(u_new, u, atol=1e-10) and np.array_equal(inl_new, inl)
        u, inl = u_new, inl_new
        if done:
            break
    if inl.sum() < 2:
        return None, None, None
    u = _weighted_null(normals[inl], weights[inl])
    # Ambiguity: the best RANSAC candidate at least ambiguity_sep_deg away from the fit, scored
    # the same way, over the fit's own score. A fit that a different direction explains about as
    # well is not trusted, whatever its formal SE says.
    s_fit = _sector_score(u[None], normals, weights, sectors, tol)[0]
    far = cand @ u < math.cos(math.radians(params.ambiguity_sep_deg))
    s_alt = float(score[far].max()) if far.any() else 0.0
    return u, inl, (s_alt / s_fit if s_fit > 0 else float("inf"))


def pitch_roll_se(normals, weights, u, eps_deg=1e-3):
    """Formal SE of (pitch, roll) from the weighted least-squares fit on the given segments.

    Residual ``r_i = n_i . u(pitch, roll)`` (radians); Jacobian by central differences; weights
    normalised to mean 1; ``cov = s^2 (J^T W J)^-1`` with ``s^2 = sum W r^2 / (N - 2)``. Segments
    from one physical edge are not independent, so this understates the true error; it is used
    as a ranking for the decline gate, which was calibrated against Google's pose, not as a CI.
    """
    p0, r0 = pitch_roll_from_up(u)
    w = weights / weights.mean()
    res = normals @ up_from_pitch_roll(p0, r0)
    jp = (normals @ up_from_pitch_roll(p0 + eps_deg, r0)
          - normals @ up_from_pitch_roll(p0 - eps_deg, r0)) / (2 * eps_deg)
    jr = (normals @ up_from_pitch_roll(p0, r0 + eps_deg)
          - normals @ up_from_pitch_roll(p0, r0 - eps_deg)) / (2 * eps_deg)
    J = np.stack([jp, jr], axis=1)                     # per degree
    N = len(normals)
    if N < 3:
        return float("nan"), float("nan"), float("nan")
    s2 = float(np.sum(w * res ** 2) / (N - 2))
    A = J.T @ (J * w[:, None])
    try:
        cov = s2 * np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return float("inf"), float("inf"), math.degrees(math.asin(min(1.0, math.sqrt(s2))))
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return float(se[0]), float(se[1]), math.degrees(math.asin(min(1.0, math.sqrt(s2))))


def estimate_from_constraints(cons: dict, params: TiltParams = TiltParams()) -> TiltEstimate:
    """Filter, fit and gate, given the constraints of :func:`collect_constraints`.

    Split out from :func:`estimate_tilt` so the tests can feed synthetic lines and a cached set of
    constraints can be re-fitted under other parameters without re-running LSD.
    """
    keep = (cons["length_deg"] >= params.min_len_deg) & (cons["lean_deg"] <= params.max_lean_deg)
    normals = cons["normal"][keep]
    length_deg = cons["length_deg"][keep]
    bearing_deg = cons["bearing_deg"][keep]
    n_seg = int(len(normals))
    empty = [0] * params.n_views
    if n_seg < 2:
        return TiltEstimate("declined", "too few segments", None, None, None, None, n_seg, 0, 0, None,
                            None, empty)
    weights = np.minimum(length_deg, params.weight_cap_deg)
    sectors = sector_of(bearing_deg, params.n_views)
    u, inl, amb = fit_up(normals, weights, sectors, params)
    if u is None:
        return TiltEstimate("declined", "no consensus", None, None, None, None, n_seg, 0, 0, None,
                            None, empty)
    pitch, roll = pitch_roll_from_up(u)
    se_p, se_r, rms = pitch_roll_se(normals[inl], weights[inl], u)
    counts = np.bincount(sectors[inl], minlength=params.n_views).tolist()
    n_inl = int(inl.sum())
    reasons = gate_reasons(n_inl, counts, se_p, se_r, amb, params)
    n_sec = int(sum(c >= params.sector_min_inliers for c in counts))
    return TiltEstimate("declined" if reasons else "ok", "; ".join(reasons), pitch, roll, se_p, se_r,
                        n_seg, n_inl, n_sec, rms, amb, counts)


def gate_reasons(n_inliers, sector_inliers, se_pitch, se_roll, ambiguity,
                 params: TiltParams = TiltParams()) -> list:
    """The decline gate: the list of failed tests (empty means accept).

    Takes only fields a stored row keeps (inlier count, per-sector inlier counts, formal SEs,
    ambiguity), so the gate can be re-applied to committed rows under other thresholds without
    the images; ``scripts/analysis/pano_tilt_estimate.py`` does that and the tests check it
    reproduces every committed row's status. A missing or non-finite SE or ambiguity fails its
    test.

    Example::

        gate_reasons(40, [5, 6, 0, 4, 3, 0, 0, 2], 0.05, 0.06, 0.92)
        # -> ['ambiguity 0.92 > 0.8']
    """
    def bad(v, limit):
        return v is None or not (v <= limit)            # also catches nan

    n_sec = int(sum(c >= params.sector_min_inliers for c in sector_inliers))
    reasons = []
    if n_inliers < params.min_inliers:
        reasons.append(f"inliers {n_inliers} < {params.min_inliers}")
    if n_sec < params.min_sectors:
        reasons.append(f"sectors {n_sec} < {params.min_sectors}")
    se = None if se_pitch is None or se_roll is None else max(se_pitch, se_roll)
    if bad(se, params.max_se_deg):
        reasons.append(f"se {se:.3f} > {params.max_se_deg}" if se is not None else "se missing")
    if bad(ambiguity, params.max_ambiguity):
        reasons.append(f"ambiguity {ambiguity:.2f} > {params.max_ambiguity}" if ambiguity is not None
                       else "ambiguity missing")
    return reasons


def collect_constraints(gray: np.ndarray, params: TiltParams = TiltParams()) -> dict:
    """Render the views, run LSD, and return every segment's constraint (see segment_constraints)."""
    views = make_views(params, gray.shape[1])
    parts = [segment_constraints(detect_segments(render_view(gray, v)), v, params) for v in views]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def estimate_tilt(gray: np.ndarray, params: TiltParams = TiltParams()) -> TiltEstimate:
    """Estimate (pitch, roll) of a grey equirect (any width; 2:1). See the module docstring."""
    h, w = gray.shape[:2]
    if w != 2 * h:
        raise ValueError(f"equirect must be 2:1, got {w}x{h}")
    return estimate_from_constraints(collect_constraints(gray, params), params)


def _resample_rotated(gray: np.ndarray, R_out_to_in: np.ndarray) -> np.ndarray:
    """Equirect whose output direction v samples the input at ``R_out_to_in @ v`` (bilinear)."""
    import cv2
    h, w = gray.shape[:2]
    xx, yy = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    b, el = sg.bearing_elevation_from_pixel(xx, yy, w, h)
    v_in = sg.direction_rfu(b, el) @ R_out_to_in.T          # row-vector form of R @ v
    gb, gel = sg.bearing_elevation(v_in)
    x, y = sg.pixel_from_bearing_elevation(gb, gel, w, h)
    return cv2.remap(np.ascontiguousarray(gray, dtype=np.float32), (x - 0.5).astype(np.float32),
                     (y - 0.5).astype(np.float32), interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def tilt_equirect(gray: np.ndarray, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Re-render a gravity-levelled equirect as a rig with (pitch, roll) would record it.

    A rig-frame pixel looking along ``v_rig`` sees the gravity-frame direction
    ``R^T v_rig`` with ``R = rig_from_gravity(pitch, roll)``. The tests use this to make tilted
    fixtures with a known answer.
    """
    return _resample_rotated(gray, sg.rig_from_gravity(pitch_deg, roll_deg).T)


def level_equirect(gray: np.ndarray, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """The inverse of :func:`tilt_equirect`: a rig-frame equirect re-rendered gravity-level.

    ``level_equirect(tilt_equirect(g, p, r), p, r)`` is ``g`` up to resampling. Works on grey or
    (H, W, 3) arrays. This is what "levelling a pano before detection" would mean; nothing in
    the pipeline calls it yet.
    """
    R = sg.rig_from_gravity(pitch_deg, roll_deg)
    if gray.ndim == 3:
        return np.stack([_resample_rotated(gray[..., i], R) for i in range(gray.shape[2])], axis=-1)
    return _resample_rotated(gray, R)
