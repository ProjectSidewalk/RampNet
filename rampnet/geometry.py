"""Cyclic geometry for equirectangular panoramas — defined once (issue #132).

A panorama wraps. Normalized x = 0 and x = 1 address the same column of pixels, so the
horizontal separation between two points is the *shorter way round*, not the arithmetic
difference. Every place in this codebase that measured it arithmetically got the seam
wrong, and each of those places had reimplemented the distance inline:
``rampnet/metrics.py`` (the shared matcher), ``rampnet/detection_eval.py`` (the
ignore-point fallback), and a handful of analysis scripts and viewers.

That is the actual defect class — not "the seam" but *cyclic distance re-derived per
site*. This module exists so there is one definition to be right, and #132's audit is a
list of callers rather than a list of reimplementations.

**y never wraps.** The poles are not identified with each other; a panorama is a cylinder
in this coordinate system, not a torus. Only x is cyclic.

**Wrapping is opt-in, and that is deliberate.** Not every coordinate space here is
panoramic. ``stage_one/crop_model/ps_and_manual_model/evaluate.py`` matches in *crop*
space (1024x352 crops, ``scale_x=341/4``) where x genuinely has two distinct ends, and
the unit-scale synthetic spaces in the tests are not panoramas either. Wrapping those
would introduce a fresh bug while fixing this one, so callers say which space they are
in rather than inheriting a guess.
"""


def fold(dx, period):
    """The shorter way round: fold an already-scaled separation onto a cyclic axis.

    The one primitive the rest of this module and the matcher are built from, so the
    modular arithmetic has a single home.

    Modular by construction, so inputs outside one period cannot produce a negative
    distance. The naive ``min(dx, period - dx)`` silently does exactly that once
    ``dx > period`` — reachable from the unit-scale synthetic coordinates in the tests
    and from any caller that has not normalized — and a negative distance squares back
    to a plausible small number rather than raising.
    """
    dx = abs(dx) % period
    return min(dx, period - dx)


def wrapped_delta_x(ax, bx, scale_x):
    """Shortest horizontal separation between two normalized x, in scaled units."""
    return fold((ax - bx) * scale_x, scale_x)


def dist_sq(ax, ay, bx, by, scale_x, scale_y, wrap_x=False):
    """Squared distance between two normalized points, in scaled units.

    With ``wrap_x`` the x axis is cyclic with period ``scale_x`` — the panorama case.
    Without it the axis is a plain interval — crop space, and the synthetic spaces in
    the tests. y is never cyclic.
    """
    dx = wrapped_delta_x(ax, bx, scale_x) if wrap_x else (ax - bx) * scale_x
    dy = (ay - by) * scale_y
    return dx * dx + dy * dy


def dist_to_seam(x, scale_x):
    """Distance from a normalized x to the nearer seam edge, in scaled units.

    The quantity #132 bins recall by. ``x`` is taken modulo 1 first so a coordinate that
    has drifted outside [0, 1] reports the distance its wrapped position actually has.
    """
    x %= 1.0
    return min(x, 1.0 - x) * scale_x


def crop_left(center_x_px, width_px, side_px):
    """Left column of a ``side_px``-wide crop centred on ``center_x_px``, wrapping.

    May address columns past the right edge; the caller cuts in two pieces and joins
    them (see ``scripts/box_gallery.py::cut_crop``, which already does this). Returned
    modulo ``width_px`` so it is always a valid column index.

    This is the viewer-side half of the same defect: ``scripts/gt_gallery.py`` *clamped*
    here instead, so a ramp on the seam appeared as two objects most of a panorama
    apart, which is why the duplicate ground truth in #130 survived human review.
    """
    return int(round(center_x_px - side_px / 2)) % width_px


def merge_seam_duplicates(points, radius_sq, scale_x, scale_y):
    """Drop the later member of each pair that duplicates *across the seam*.

    Seam-crossing pairs only. Points closer than the radius but *not* spanning the seam
    are left alone: they are common in this data — ``manual_gold`` holds 234 of them away
    from the seam, 87 with near-identical y at the horizon — and they are overwhelmingly
    genuine adjacent far-field ramps rather than duplicates (#130). Merging those would
    delete real ramps in the direction that flatters recall.

    Order-independent in practice *provided the matcher wraps*: with a wrapping matcher
    either member of a merged pair scores identically, which is the argument in #130 for
    why the two fixes have to land together. Without it, which member survives changes
    the score.
    """
    kept = []
    for p in points:
        duplicate = False
        for q in kept:
            if abs(p[0] - q[0]) * scale_x <= scale_x / 2:
                continue                       # not spanning the seam — leave it alone
            if dist_sq(p[0], p[1], q[0], q[1], scale_x, scale_y, wrap_x=True) < radius_sq:
                duplicate = True
                break
        if not duplicate:
            kept.append(p)
    return kept
