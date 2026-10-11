"""The arm registry for scripts/analysis/crossview_align_48.py (#48).

An arm places a source-view GT ramp point in another pano. It is one function plus one
``@register`` line, in any module of this package (a new file per technique keeps
parallel work free of merge conflicts)::

    from crossview_arms._registry import register

    @register("my_arm", needs=("views",), description="one line for the results table",
              config={"threshold": 0.5})
    def my_arm(pair, ctx):
        ...                                   # see crossview_align_48.Context for ctx
        return {"x": x_norm, "y": y_norm, "inliers": 42}   # or None to fall back

``pair`` is one row of the frozen ``pairs.csv`` (floats already parsed): the source pano
and GT pixel (src_pano, src_x, src_y), the other pano (oth_pano), today's projection
(proj_x, proj_y), ranges, dates and ids. The reference columns (ref_x, ref_y, ref_conf,
ref_world_gap_m) are the answer and are **removed** before an arm sees the row, both from
``pair`` and from ``ctx.pairs``; ``ctx.slim`` / ``geometry.at_height`` withhold each pano's
detections for the same reason (the reference is one of them). ``tests/test_crossview_align_48.py``
plants an arm that tries to read them.
``needs`` is documentation plus a guard: "views", "labeler", "panos".
"""
from dataclasses import dataclass, field

ARMS = {}

#: pair-row keys that hold the answer; run_arm and Context hand arms copies without them
ANSWER_KEYS = ("ref_x", "ref_y", "ref_conf", "ref_world_gap_m")


@dataclass
class Arm:
    name: str
    fn: object
    needs: tuple = ()
    description: str = ""
    config: dict = field(default_factory=dict)


def register(name, needs=(), description="", config=None):
    def deco(fn):
        if name in ARMS and ARMS[name].fn is not fn:
            raise ValueError(f"arm {name!r} registered twice")
        ARMS[name] = Arm(name, fn, tuple(needs), description, dict(config or {}))
        return fn
    return deco
