"""Consensus of two independent feed-forward 3D models (#48) -- a POST HOC composite.

Where MapAnything (pair, given intrinsics; ``mapa_k_pair``) and MASt3R (``mast3r_pair``)
place the click within ``CONSENSUS_DEG`` of each other, take their spherical midpoint;
elsewhere fall back (to the projection, or to ``proj_height_auto``'s point). Defined after
both arms were scored on all 300 pairs, from the agreement diagnostic
(``_mv3d.py agreement``): its threshold is about the reference's own noise
(``docs/crossview_align_48.md`` section 2) and was not tuned, but the composite as a whole
is post hoc. It reads the two arms' committed predictions -- never the answer columns --
so it runs on CPU in seconds.
"""
import crossview_align_48 as H
from crossview_arms._registry import register

CONSENSUS_ARMS = ("mapa_k_pair", "mast3r_pair")
CONSENSUS_DEG = 1.5


def _consensus(pair, ctx, else_auto):
    key = ("mv3d_consensus_preds",)
    if key not in ctx.cache:
        ctx.cache[key] = {a: H.read_predictions(a)
                          for a in CONSENSUS_ARMS + ("proj_height_auto",)}
    preds = ctx.cache[key]
    a, b = (preds[n][pair["pair_id"]] for n in CONSENSUS_ARMS)
    out = {"x": None, "y": None}
    if a["x"] is not None and b["x"] is not None:
        d = float(H.angular_error_deg(a["x"], a["y"], b["x"], b["y"]))
        out["inter_model_deg"] = d
        if d <= CONSENSUS_DEG:
            x, y = H.dir_to_pano(H.pano_dir(a["x"], a["y"]) + H.pano_dir(b["x"], b["y"]))
            out.update({"x": float(x), "y": float(y), "used": "consensus"})
            return out
    if else_auto:
        c = preds["proj_height_auto"][pair["pair_id"]]
        out.update({"x": c["x"], "y": c["y"], "used": "proj_height_auto"})
    return out


CONFIG = {"arms": list(CONSENSUS_ARMS), "agree_deg": CONSENSUS_DEG,
          "point": "spherical midpoint of the two",
          "note": "POST HOC composite, defined after both arms were scored; reads their "
                  "committed predictions (no answer columns)"}


@register("mv3d_consensus", needs=(), config={**CONFIG, "otherwise": "fall back (projection)"},
          description="MapAnything (pair, K) and MASt3R midpoint where they agree within "
                      "1.5 deg; else fall back (post hoc)")
def mv3d_consensus(pair, ctx):
    return _consensus(pair, ctx, else_auto=False)


@register("mv3d_consensus_else_auto", needs=(),
          config={**CONFIG, "otherwise": "proj_height_auto's point"},
          description="mv3d_consensus where the two agree, proj_height_auto elsewhere (post hoc)")
def mv3d_consensus_else_auto(pair, ctx):
    return _consensus(pair, ctx, else_auto=True)
