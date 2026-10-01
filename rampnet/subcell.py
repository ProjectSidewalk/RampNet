"""Sub-cell decode of RampNet heatmap peaks (issue #221).

Why this exists
---------------
The pano model's head is ``Conv(3x3) -> ReLU -> Upsample(bilinear, x8) -> Conv(1x1)``
on a stride-32 ConvNeXt feature map. The 1x1 conv is linear (and its bias commutes with
bilinear resampling, whose weights sum to one), so the 512x1024 heatmap is *exactly* a
bilinear upsample of a 64x128 map -- the "coarse" map. A bilinear surface peaks only at
its sample points, so an integer argmax can only land on the two hi-res pixels that
flank a coarse centre (``8i+3`` / ``8i+4`` under ``align_corners=False``). Detections are
therefore quantized to an 8-pixel grid (2.8 deg of heading and of dip on a 4096-wide
pano). The crop model has the same head and the same x8 factor.

The sub-cell position is still present in the coarse neighbourhood of the peak, so it
can be recovered at inference without retraining. This module does that:

- :func:`coarse_from_heatmap` recovers the coarse map from a 512x1024 heatmap (exact up
  to float error, by least squares against the bilinear operator), so a caller that only
  has ``model(x)`` -- such as the auto-labeler -- does not need to hook the head.
- :func:`refine_offset` turns the 3x3 coarse neighbourhood of a peak into a sub-cell
  offset, by one of several standard rules (``METHODS``).
- :func:`refine_peaks` is the end-to-end entry point: peaks as returned by
  ``peak_local_max`` on the heatmap in, refined normalized ``(x, y)`` out.

Coordinates follow the pipeline's existing convention (``stage_two/train.py`` places a
target at pixel ``round(x_norm * W)``; every extractor reports ``x_norm = col / W``), so
a hi-res pixel index ``p`` is the normalized position ``p / W`` and coarse centre ``i``
sits at hi-res index ``f*i + (f-1)/2`` (``8i + 3.5``). An argmax at ``8i+3`` or ``8i+4``
is therefore half a hi-res pixel either side of the coarse centre it belongs to.

Pure numpy, no torch. Usage::

    from rampnet.subcell import refine_peaks
    from skimage.feature import peak_local_max
    pk = peak_local_max(np.clip(h, 0, 1), min_distance=10, threshold_abs=0.3,
                        exclude_border=False)
    xy = refine_peaks(h, pk, method="gaussian")   # (N, 2) normalized x, y
"""
from functools import lru_cache

import numpy as np

#: The model's upsample factor (stride-32 features, 4x-down heatmap) for both heads.
FACTOR = 8

#: Decode rules, in coarse-cell units of offset from the coarse centre.
#:
#: - ``argmax``: no refinement; the hi-res argmax pixel as ``peak_local_max`` returns it.
#: - ``centre``: the coarse centre itself (``8i+3.5``); differs from ``argmax`` by 0.5 px.
#: - ``quarter``: 0.25 cell toward the higher neighbour, per axis (the SimpleBaseline /
#:   HRNet rule).
#: - ``parabola``: vertex of the parabola through the three values on each axis.
#: - ``gaussian``: the same on ``log`` values -- exact for a sampled Gaussian, which is
#:   the training target's shape (sigma 10 hi-res px = 1.25 coarse cells).
#: - ``dark``: 2-D second-order Taylor expansion of the log map (DARK, Zhang et al.
#:   CVPR 2020) with the cross term, without DARK's pre-smoothing (the coarse map is
#:   already smooth at sigma 1.25 cells).
#: - ``centroid``: value-weighted centroid of the 3x3 neighbourhood, after subtracting
#:   the neighbourhood minimum (so a flat background does not pull it to 0).
METHODS = ("argmax", "centre", "quarter", "parabola", "gaussian", "dark", "centroid")

#: Offsets are clamped to half a cell: a peak whose true position is further than that
#: from the centre would have made the neighbour the coarse maximum instead.
MAX_OFFSET = 0.5
_EPS = 1e-6


def bilinear_upsample_matrix(n_in, n_out):
    """``(n_out, n_in)`` matrix ``U`` with ``U @ v`` equal to 1-D
    ``torch.nn.functional.interpolate(..., mode='bilinear', align_corners=False)``.

    PyTorch's rule: source coordinate ``s = (d + 0.5) * n_in / n_out - 0.5``, clamped
    below at 0, then linear between ``floor(s)`` and ``floor(s) + 1`` (the upper index
    clamped at ``n_in - 1``).
    """
    U = np.zeros((n_out, n_in))
    scale = n_in / n_out
    for d in range(n_out):
        s = max((d + 0.5) * scale - 0.5, 0.0)
        i0 = min(int(np.floor(s)), n_in - 1)
        i1 = min(i0 + 1, n_in - 1)
        w1 = s - i0
        U[d, i0] += 1.0 - w1
        U[d, i1] += w1
    return U


@lru_cache(maxsize=8)
def _pinv(n_in, n_out):
    return np.linalg.pinv(bilinear_upsample_matrix(n_in, n_out))


def coarse_from_heatmap(heatmap, factor=FACTOR):
    """Recover the ``(H/f, W/f)`` coarse map whose bilinear upsample is ``heatmap``.

    Exact (to float rounding) when ``heatmap`` really is such an upsample, which the
    RampNet head guarantees; for any other input it is the least-squares coarse map.
    """
    h = np.asarray(heatmap, dtype=np.float64)
    H, W = h.shape
    if H % factor or W % factor:
        raise ValueError(f"heatmap {h.shape} is not a multiple of factor {factor}")
    return _pinv(H // factor, H) @ h @ _pinv(W // factor, W).T


def upsample(coarse, factor=FACTOR):
    """Bilinear ``align_corners=False`` upsample by ``factor`` (numpy; for tests)."""
    c = np.asarray(coarse, dtype=np.float64)
    h, w = c.shape
    return (bilinear_upsample_matrix(h, h * factor) @ c
            @ bilinear_upsample_matrix(w, w * factor).T)


def coarse_cell(row, col, factor=FACTOR):
    """The coarse cell a hi-res pixel belongs to (the nearest coarse centre)."""
    return int(row) // factor, int(col) // factor


def neighbourhood(coarse, i, j, wrap_x=False):
    """3x3 coarse values around ``(i, j)``; NaN where it runs off the map.

    ``wrap_x`` treats the columns as cyclic (the 360 deg seam of a panorama). It is off by
    default: the network pads at the seam rather than wrapping, so the two edge columns
    were not computed as neighbours of each other.
    """
    c = np.asarray(coarse, dtype=np.float64)
    H, W = c.shape
    out = np.full((3, 3), np.nan)
    for a in (-1, 0, 1):
        r = i + a
        if not 0 <= r < H:
            continue
        for b in (-1, 0, 1):
            k = j + b
            if wrap_x:
                k %= W
            elif not 0 <= k < W:
                continue
            out[a + 1, b + 1] = c[r, k]
    return out


def climb(coarse, i, j, wrap_x=False):
    """Move ``(i, j)`` to the maximum of its 3x3 coarse neighbourhood, repeatedly.

    What it is for, as measured on the #221 extraction (``detections.json``): each hi-res
    pixel of a bilinear surface mixes four coarse values, so a strong *diagonal*
    neighbour can pull the integer argmax into the cell next to the true coarse maximum.
    ``peak_local_max`` then returns an on-grid pixel (``8i+3``/``8i+4``) in the wrong cell,
    and ``row // 8, col // 8`` is not the coarse maximum. Climbing re-anchors it. That
    happened to 48 of 5,100 peaks >= 0.30 (39 of 3,868 on manual_gold), every one of
    them on-grid with score <= 1.

    It is *not* what fixes clipped plateaus in that data. ``peak_local_max`` runs on
    ``clip(h, 0, 1)``, so a peak above 1 returns an arbitrary pixel of a flat top, but in
    all 187 such peaks that pixel was already in the coarse-max cell (0 climbed). The
    left-edge plateau (hi-res columns 0-3 all equal coarse column 0) is handled by the
    NaN edge neighbour in :func:`neighbourhood`, not by climbing. Climbing is still the
    safe general rule for both: it only ever moves to a strictly higher coarse value.

    ``wrap_x`` must match the one used for :func:`neighbourhood`: with the seam wrapped,
    the coarse maximum can be across it, and the offset would otherwise saturate at the
    clamp. Returns the new ``(i, j)`` and the number of steps taken.
    """
    c = np.asarray(coarse)
    H, W = c.shape
    steps = 0
    while steps < max(H, W):
        n = neighbourhood(c, i, j, wrap_x)
        a, b = np.unravel_index(np.nanargmax(n), n.shape)
        if (a, b) == (1, 1) or n[a, b] <= c[i, j]:
            return i, j, steps
        i, j, steps = i + a - 1, j + b - 1, steps + 1
        if wrap_x:
            j %= W
    return i, j, steps


def _axis(lo, mid, hi, method):
    """1-D offset in cells from three values along one axis (NaN-safe).

    ``gaussian`` falls back to ``parabola`` when any of the three values is <= 0: the head
    output is not clamped, and the log of a clamped non-positive value would drive the
    offset to the +/-0.5 clamp. (Never triggered on the #221 data, whose smallest axis
    neighbour of a peak >= 0.30 is 0.009, but the labeler runs on many more cities.)
    """
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return 0.0
    if method == "quarter":
        return 0.25 * float(np.sign(hi - lo))
    if method == "gaussian":
        if min(lo, mid, hi) <= 0:
            return _axis(lo, mid, hi, "parabola")
        lo, mid, hi = (np.log(v) for v in (lo, mid, hi))
    elif method != "parabola":
        raise ValueError(method)
    den = lo - 2.0 * mid + hi
    if den >= 0:          # not a maximum along this axis
        return 0.0
    return float(0.5 * (lo - hi) / den)


def refine_offset(n, method):
    """Sub-cell ``(dy, dx)`` in coarse cells from a 3x3 neighbourhood ``n``.

    ``n[1, 1]`` is the coarse maximum. Offsets are clamped to +/-``MAX_OFFSET``.
    ``argmax`` and ``centre`` return ``(0, 0)`` (the caller places them).
    """
    n = np.asarray(n, dtype=np.float64)
    if method in ("argmax", "centre"):
        return 0.0, 0.0
    if method in ("quarter", "parabola", "gaussian"):
        dy = _axis(n[0, 1], n[1, 1], n[2, 1], method)
        dx = _axis(n[1, 0], n[1, 1], n[1, 2], method)
    elif method == "dark":
        if np.any(n[np.isfinite(n)] <= 0):     # log undefined: per-axis rule instead
            return refine_offset(n, "gaussian")
        L = np.log(np.maximum(np.where(np.isfinite(n), n, _EPS), _EPS))
        if not (np.isfinite(n[0, 1]) and np.isfinite(n[2, 1])
                and np.isfinite(n[1, 0]) and np.isfinite(n[1, 2])):
            return refine_offset(n, "gaussian")
        gy = 0.5 * (L[2, 1] - L[0, 1])
        gx = 0.5 * (L[1, 2] - L[1, 0])
        hyy = L[2, 1] - 2 * L[1, 1] + L[0, 1]
        hxx = L[1, 2] - 2 * L[1, 1] + L[1, 0]
        corners = n[[0, 0, 2, 2], [0, 2, 0, 2]]
        hxy = (0.25 * (L[2, 2] - L[2, 0] - L[0, 2] + L[0, 0])
               if np.all(np.isfinite(corners)) else 0.0)
        Hm = np.array([[hyy, hxy], [hxy, hxx]])
        det = np.linalg.det(Hm)
        if not (hyy < 0 and det > 0):          # not negative definite: fall back
            return refine_offset(n, "gaussian")
        dy, dx = -np.linalg.solve(Hm, [gy, gx])
    elif method == "centroid":
        w = np.where(np.isfinite(n), n, np.nan)
        w = np.nan_to_num(w - np.nanmin(w), nan=0.0)
        s = w.sum()
        if s <= 0:
            return 0.0, 0.0
        g = np.array([-1.0, 0.0, 1.0])
        dy = float((w.sum(axis=1) * g).sum() / s)
        dx = float((w.sum(axis=0) * g).sum() / s)
    else:
        raise ValueError(f"unknown method {method!r}; known: {', '.join(METHODS)}")
    clamp = lambda v: float(np.clip(v, -MAX_OFFSET, MAX_OFFSET))  # noqa: E731
    return clamp(dy), clamp(dx)


def refine_peaks(heatmap, peaks, method="gaussian", factor=FACTOR, coarse=None,
                 wrap_x=False):
    """Refined normalized ``(x, y)`` for each ``(row, col)`` peak of ``heatmap``.

    ``peaks`` is what ``peak_local_max`` returns on the hi-res heatmap. ``coarse`` may be
    passed when the caller already has the pre-upsample map; otherwise it is recovered
    with :func:`coarse_from_heatmap`. Returns an ``(N, 2)`` float array in the pipeline's
    ``(col / W, row / H)`` convention. With ``method="argmax"`` it returns the input
    peaks unchanged, so the refinement can be switched off without a code path change.
    """
    H, W = np.asarray(heatmap).shape
    peaks = np.asarray(peaks, dtype=int).reshape(-1, 2)
    if method == "argmax":
        return np.column_stack([peaks[:, 1] / W, peaks[:, 0] / H]).astype(float)
    if coarse is None:
        coarse = coarse_from_heatmap(heatmap, factor)
    out = np.empty((len(peaks), 2))
    half = (factor - 1) / 2.0
    for k, (r, c) in enumerate(peaks):
        i, j = coarse_cell(r, c, factor)
        i, j, _ = climb(coarse, i, j, wrap_x)
        dy, dx = refine_offset(neighbourhood(coarse, i, j, wrap_x), method)
        x = (factor * (j + dx) + half) % W
        y = factor * (i + dy) + half
        out[k] = (x / W, y / H)
    return out
