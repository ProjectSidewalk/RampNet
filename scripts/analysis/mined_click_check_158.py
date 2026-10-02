"""#158 step 4, click pass: where is the ramp each Yes label points at?

Pass 1 and pass 2 of `mined_label_check_158.py` asked "Is there a curb ramp at the ring?".
On 9 of the 71 sample Yeses Jon's notes say the ramp is there but the ring is a ring or
two beside it. Under the rubric ("at the ring or touching it") that is a Yes, but a
training label is a point and a heatmap target, so the offset matters. The click pass
measures it: for every sample card answered Yes (pass 1 with the pass-2 answers written
over its Can't tells), the rater clicks the point where they would place the label on the
ramp the ring refers to. The click is converted exactly to panorama coordinates (the crop
is a plain equirect window, `multiview_evidence_48.cut_one`) and scored in RampNet's own
units: heatmap pixels on the 512 x 1024 training grid, against the training sigma (10) and
the evaluation match radius (0.022 x 1024 = 22.5).

    plan     --pass1 FILE --pass2 FILE   -> items_click.json (the cut plan)
    (makelab2) multiview_evidence_48.py cut-crops items_click.json --archive-root ... --out DIR
    gallery  --crops DIR [--init-rater ID] -> gallery_click.html, manifest_click.json
    score    FILE [FILE2]                 -> offsets, rates, two-rater agreement

The rater file is `mined_click_check__<rater>.json`, a different prefix from the yes/no
files, so the pass-1 rater id can be reused.
"""
import argparse
import html
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import multiview_evidence_48 as mv  # noqa: E402
import residual_gt_check_48 as R  # noqa: E402
import mined_label_check_158 as M  # noqa: E402

ITEMS_PATH = os.path.join(M.OUT, "items_click.json")
GALLERY_DIR = M.GALLERY_DIR
CROPS_DIR = "crops_click"
CROPS_REL = "benchmark/mined_label_check_158/" + CROPS_DIR
MANIFEST_PATH = os.path.join(GALLERY_DIR, "manifest_click.json")
GALLERY_REL = "benchmark/mined_label_check_158/gallery_click.html"
EXPORT_PREFIX, EXPORT_SUFFIX = "mined_click_check__", ".json"
QUALITY = M.PASS2_QUALITY
FIT_PX = 540

#: RampNet's units (stage_two/train.py, stage_two/evaluate.py): a 512 x 1024 heatmap
#: over the full equirect pano, Gaussian targets with sigma 10, a detection matches a
#: label within 0.022 x heatmap width.
HEATMAP_W, HEATMAP_H = 1024, 512
TRAIN_SIGMA_PX = 10.0
EVAL_RADIUS_PX = 0.022 * HEATMAP_W

TASK = "RampNet #158 step 4, click pass: where would you place the label?"
QUESTION = "Click where you would place the label on the curb ramp the ring refers to."
STATUSES = [
    ("placed", "Placed",
     "A click on the ramp the ring refers to, where you would put a curb-ramp label "
     "(for you, the centre of the tactile warning strip, or of the ramp if it has none)."),
    ("multi", "Two ramps, can't pick",
     "The ring sits between two or more ramps and you cannot say which one it means."),
    ("cant_place", "Can't place",
     "You answered Yes before, but in this view you cannot put a point on the ramp "
     "(occluded, washed out, too far). Excluded from offsets."),
]
STATUS_KEYS = tuple(k for k, _, _ in STATUSES)
RULES = [
    "The ring is the miner's point. Click the ramp it refers to, not the nearest ramp in the "
    "crop if that is a different one.",
    "R hides the ring, so it does not cover the ramp; the click stays.",
    "Clicking again moves the point. Clear removes it.",
    "Use the same placement convention on every card.",
]


def verdicts_path(rater):
    if not M.RATER_RE.match(rater or ""):
        raise ValueError(f"rater id {rater!r} must match {M.RATER_RE.pattern}")
    return os.path.join(M.OUT, EXPORT_PREFIX + rater + EXPORT_SUFFIX)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def window_top_frac(y):
    """The crop window's top edge as a fraction of pano height (`cut_one` clamps it)."""
    fv = mv.CROP_FOV_V
    return max(0.0, min(180.0 - fv, y * 180.0 - fv / 2)) / 180.0


def click_to_pano(x, y, fx, fy):
    """A click at (fx, fy), fractions of the crop's width and height, to normalized pano
    (x, y). The window is centred on x, so fx = 0.5 is x; x wraps at the seam."""
    px = (x + (fx - 0.5) * mv.CROP_FOV_H / 360.0) % 1.0
    py = window_top_frac(y) + fy * mv.CROP_FOV_V / 180.0
    return px, py


def offset_px(x0, y0, x1, y1):
    """(dx, dy, distance) on the 512 x 1024 training heatmap, dx wrapped at the seam."""
    dx = (x1 - x0 + 0.5) % 1.0 - 0.5
    dx, dy = dx * HEATMAP_W, (y1 - y0) * HEATMAP_H
    return dx, dy, math.hypot(dx, dy)


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def yes_cards(pass1_path, pass2_path):
    """The sample cards answered Yes in the combined read, in committed item order."""
    ref = M.committed_reference()
    p1 = M.load_verdicts(pass1_path, ref)
    p2 = M.load_verdicts(pass2_path, ref, pass1=(pass1_path, p1))
    v = dict(p1["verdicts"])
    for u in p2["items"]:
        if (p2["verdicts"].get(u) or {}).get("answer"):
            v[u] = p2["verdicts"][u]
    ids = [u for u in ref["items"] if not ref["cards"][u]["instrument"]
           and (v.get(u) or {}).get("answer") == "yes"]
    return ids, ref, p1, p2


def _rel(path):
    return os.path.relpath(path, mv.REPO).replace(os.sep, "/")


def cmd_plan(args):
    ids, ref, p1, p2 = yes_cards(args.pass1, args.pass2)
    plan = json.load(open(M.ITEMS_PATH, encoding="utf-8"))
    keep = set(ids)
    items = [dict(it, ring=False) for it in plan["items"]
             if it["ramp_uid"] in keep and M._role(it) == "target"]
    if len(items) != len(ids):
        raise SystemExit(f"{len(ids)} Yes cards but {len(items)} target views")
    mv.write_json(ITEMS_PATH, {
        "task": TASK, "cards": ids, "n": len(items), "items": items,
        "cut": {"native": True, "quality": QUALITY},
        "pass1_manifest_digest": ref["manifest_digest"],
        "from": {"pass1": _rel(args.pass1), "pass1_sha256": R.sha256_file(args.pass1),
                 "pass2": _rel(args.pass2), "pass2_sha256": R.sha256_file(args.pass2)}})
    print(f"{len(ids)} sample Yes cards -> {ITEMS_PATH}")


# --------------------------------------------------------------------------- #
# gallery
# --------------------------------------------------------------------------- #
def committed_reference():
    plan = json.load(open(ITEMS_PATH, encoding="utf-8"))
    man = json.load(open(MANIFEST_PATH, encoding="utf-8"))
    ids = plan["cards"]
    digest = R.manifest_digest(ids, man["crops_sha256"])
    if digest != man["manifest_digest"]:
        raise ValueError("manifest_click.json's digest does not re-derive from its crop sha256s")
    ref1 = M.committed_reference()
    return {"manifest_digest": digest, "items": ids, "plan": plan,
            "items_by_uid": {it["ramp_uid"]: it for it in plan["items"]},
            "cards": {u: ref1["cards"][u] for u in ids}, "crop_px": man["crop_px"]}


def empty_verdicts(ids, digest, rater):
    return {"task": TASK, "question": QUESTION,
            "statuses": [{"key": k, "label": lab, "definition": d} for k, lab, d in STATUSES],
            "rules": RULES, "rater": rater, "items": ids, "manifest_digest": digest,
            "n_items": len(ids), "n_answered": 0, "gallery": GALLERY_REL, "verdicts": {}}


def _target_figure(it, px):
    name = mv.crop_name(it)
    date = html.escape(it.get("capture_date") or "date unknown")
    w, h = px
    fit_h = round(FIT_PX * h / w)
    return (f'<figure class="tgt"><span class="ringwrap clickable" tabindex="-1">'
            f'<img src="{CROPS_DIR}/{html.escape(name)}" width="{FIT_PX}" height="{fit_h}" '
            f'data-natural="{w} {h}" alt="Card {it["ramp_uid"]}: click where the label '
            f'goes" draggable="false">{M._ring_overlay(it)}'
            f'<svg class="mark" viewBox="0 0 100 100" preserveAspectRatio="none" '
            f'aria-hidden="true"><g class="pt" visibility="hidden">'
            f'<line class="k" x1="0" y1="0" x2="0" y2="0"/><line class="k" x1="0" y1="0" '
            f'x2="0" y2="0"/><line class="w" x1="0" y1="0" x2="0" y2="0"/>'
            f'<line class="w" x1="0" y1="0" x2="0" y2="0"/></g></svg></span>'
            f'<figcaption><strong>Click where the label goes</strong> &middot; {date} '
            f'&middot; native {w}&times;{h} px</figcaption></figure>')


def render_gallery(ref_items, ctx_by_uid, ids, crop_px, digest, cards_meta):
    cards = []
    for n, uid in enumerate(ids, 1):
        it = ref_items[uid]
        band = html.escape(cards_meta[uid]["band"])
        st = "".join(f'<label><input type="radio" name="s_{uid}" value="{k}"'
                     f'{" disabled" if k == "placed" else ""}> {html.escape(lab)}</label>'
                     for k, lab, _ in STATUSES)
        ctx = "".join(M._figure(c, uid, mv.CROP_PX[0]) for c in ctx_by_uid.get(uid, [])) \
            or '<p class="muted">No context view.</p>'
        cards.append(
            f'<section class="card" data-uid="{uid}" aria-labelledby="h_{uid}">'
            f'<h2 id="h_{uid}">{n}. {uid} <span class="muted">({band})</span></h2>'
            f'<div class="row"><div class="rate">{_target_figure(it, crop_px[mv.crop_name(it)])}'
            f'{M.image_controls_html(uid, pass2=True)}'
            f'<fieldset><legend>{html.escape(QUESTION)}</legend><div class="opts">{st}'
            f'<button type="button" class="clear">Clear</button></div>'
            f'<p class="where muted" aria-live="polite">No point yet.</p>'
            f'<label class="note">Note (optional) <textarea rows="2" name="n_{uid}">'
            f'</textarea></label></fieldset></div>'
            f'<div class="context" role="group" aria-label="Context for {uid}, not rated">'
            f'<h3>Context: another capture, unmarked</h3><div class="strip">{ctx}</div></div>'
            f'</div></section>')
    meta = {"task": TASK, "question": QUESTION,
            "statuses": [{"key": k, "label": lab, "definition": d} for k, lab, d in STATUSES],
            "rules": RULES, "items": ids, "manifest_digest": digest, "gallery": GALLERY_REL}
    statuses = "".join(f"<dt>{html.escape(lab)}</dt><dd>{html.escape(d)}</dd>"
                       for _, lab, d in STATUSES)
    rules = "".join(f"<li>{html.escape(x)}</li>" for x in RULES)
    return (PAGE_HEAD.replace("__IMGCSS__", M.IMAGE_CONTROLS_CSS)
            .replace("__STATUSES__", statuses).replace("__RULES__", rules)
            .replace("__N__", str(len(ids)))
            + "".join(cards)
            + '<script id="meta" type="application/json">'
            + json.dumps(meta).replace("</", "<\\/") + "</script>"
            + "<script>" + PAGE_JS + M.IMAGE_CONTROLS_JS + "</script></body></html>")


def cmd_gallery(args):
    if args.init_rater is not None and not M.RATER_RE.match(args.init_rater):
        raise SystemExit(f"--init-rater {args.init_rater!r} is not a valid rater id")
    from PIL import Image
    plan = json.load(open(ITEMS_PATH, encoding="utf-8"))
    ids = plan["cards"]
    out = os.path.join(GALLERY_DIR, CROPS_DIR)
    os.makedirs(out, exist_ok=True)
    sha, px, missing = {}, {}, []
    for it in plan["items"]:
        name = mv.crop_name(it)
        src, dst = os.path.join(args.crops, name), os.path.join(out, name)
        if not os.path.exists(src):
            missing.append(name)
            continue
        if os.path.abspath(src) != os.path.abspath(dst):
            with open(src, "rb") as fi, open(dst, "wb") as fo:
                fo.write(fi.read())
        sha[name] = R.sha256_file(dst)
        with Image.open(dst) as im:
            px[name] = list(im.size)
    if missing:
        raise SystemExit(f"{len(missing)} crops missing from {args.crops}, e.g. {missing[:3]}")
    digest = R.manifest_digest(ids, sha)
    # The pass-2 cuts of the same target views were made the same way; any overlap
    # must be byte-identical, which proves the cut is deterministic.
    man2 = json.load(open(M.MANIFEST2_PATH, encoding="utf-8"))
    same = {n: man2["crops_sha256"][n] == s for n, s in sha.items() if n in man2["crops_sha256"]}
    if not all(same.values()):
        raise SystemExit(f"click crops differ from the pass-2 cuts: "
                         f"{[n for n, ok in same.items() if not ok]}")
    mv.write_json(MANIFEST_PATH, {
        "manifest_digest": digest, "n_crops": len(sha), "n_cards": len(ids),
        "crops_sha256": sha, "crop_px": px, "cut": plan["cut"],
        "identical_to_pass2_cuts": len(same),
        "items": _rel(ITEMS_PATH), "items_sha256": R.sha256_file(ITEMS_PATH)})
    p1 = json.load(open(M.ITEMS_PATH, encoding="utf-8"))
    keep = set(ids)
    ctx = {}
    for it in p1["items"]:
        if it["ramp_uid"] in keep and M._role(it) != "target":
            ctx.setdefault(it["ramp_uid"], []).append(it)
    ref1 = M.committed_reference()
    with open(os.path.join(GALLERY_DIR, "gallery_click.html"), "w", encoding="utf-8",
              newline="") as f:
        f.write(render_gallery({it["ramp_uid"]: it for it in plan["items"]}, ctx, ids, px,
                               digest, ref1["cards"]))
    if args.init_rater:
        path = verdicts_path(args.init_rater)
        if not os.path.exists(path):
            mv.write_json(path, empty_verdicts(ids, digest, args.init_rater))
    print(f"wrote {GALLERY_REL} ({len(ids)} cards, digest {digest}; {len(same)} crops "
          f"byte-identical to their pass-2 cuts)")


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def load_verdicts(path, ref):
    d = json.load(open(path, encoding="utf-8"))
    rater = d.get("rater")
    if not isinstance(rater, str) or not M.RATER_RE.match(rater):
        raise ValueError(f"{path}: rater id {rater!r} missing or invalid")
    if os.path.basename(path) != EXPORT_PREFIX + rater + EXPORT_SUFFIX:
        raise ValueError(f"{path}: file name does not match rater {rater!r}")
    if d.get("manifest_digest") != ref["manifest_digest"]:
        raise ValueError(f"{path}: made on gallery {d.get('manifest_digest')}, not "
                         f"{ref['manifest_digest']}")
    if d.get("items") != ref["items"]:
        raise ValueError(f"{path}: item list differs from items_click.json")
    statuses = [{"key": k, "label": lab, "definition": x} for k, lab, x in STATUSES]
    if (d.get("question"), d.get("statuses"), d.get("rules")) != (QUESTION, statuses, RULES):
        raise ValueError(f"{path}: question, statuses or rules differ from this module's")
    for uid, v in d.get("verdicts", {}).items():
        if uid not in ref["cards"]:
            raise ValueError(f"{path}: {uid} is not an item")
        s = v.get("status")
        if s not in STATUS_KEYS + (None,):
            raise ValueError(f"{path}: {uid} status {s!r} not in {STATUS_KEYS}")
        c = v.get("click")
        if (s == "placed") != (c is not None):
            raise ValueError(f"{path}: {uid} has status {s!r} and click {c!r}")
        if c is not None and not (0 <= c.get("fx", -1) <= 1 and 0 <= c.get("fy", -1) <= 1):
            raise ValueError(f"{path}: {uid} click {c!r} is outside the crop")
    return d


def _pct(xs, q):
    """Linear-interpolated percentile (q in 0..100); None for an empty list."""
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(dists):
    n = len(dists)
    within = {"sigma_10": sum(d <= TRAIN_SIGMA_PX for d in dists),
              "eval_radius_22_5": sum(d <= EVAL_RADIUS_PX for d in dists)}
    out = {"n": n, "median_px": _pct(dists, 50), "p75_px": _pct(dists, 75),
           "p90_px": _pct(dists, 90), "max_px": max(dists) if dists else None,
           "mean_px": statistics.fmean(dists) if dists else None}
    for k, c in within.items():
        lo, hi = mv.wilson(c, n)
        out["within_" + k] = {"count": c, "rate": c / n if n else None, "wilson_95": [lo, hi]}
    return out


def offsets(d, ref):
    rows = []
    for uid in ref["items"]:
        v = d["verdicts"].get(uid) or {}
        if v.get("status") != "placed":
            continue
        it = ref["items_by_uid"][uid]
        cx, cy = click_to_pano(it["x"], it["y"], v["click"]["fx"], v["click"]["fy"])
        dx, dy, dist = offset_px(it["x"], it["y"], cx, cy)
        rows.append({"id": uid, "band": ref["cards"][uid]["band"], "x_click": cx,
                     "y_click": cy, "dx_px": dx, "dy_px": dy, "dist_px": dist,
                     "dist_deg": dist * 360.0 / HEATMAP_W})
    return rows


def score(d, ref, n_decided=None):
    rows = offsets(d, ref)
    st = {k: 0 for k in STATUS_KEYS + ("unanswered",)}
    for uid in ref["items"]:
        st[(d["verdicts"].get(uid) or {}).get("status") or "unanswered"] += 1
    out = {"rater": d["rater"], "manifest_digest": d["manifest_digest"], "status": st,
           "pooled": summarize([r["dist_px"] for r in rows]),
           "by_band": {b: summarize([r["dist_px"] for r in rows if r["band"] == b])
                       for b in M.BANDS},
           "units": {"heatmap": f"{HEATMAP_W}x{HEATMAP_H}", "train_sigma_px": TRAIN_SIGMA_PX,
                     "eval_radius_px": EVAL_RADIUS_PX, "deg_per_px": 360.0 / HEATMAP_W},
           "offsets": rows}
    if n_decided:
        # A label counts at a radius only if it was a Yes and the click is within it;
        # "multi" and "can't place" count against, unanswered cards are refused upstream.
        for k, r in (("sigma_10", TRAIN_SIGMA_PX), ("eval_radius_22_5", EVAL_RADIUS_PX)):
            c = sum(x["dist_px"] <= r for x in rows)
            lo, hi = mv.wilson(c, n_decided)
            out["precision_at_" + k] = {"yes_within": c, "n_decided": n_decided,
                                        "precision": c / n_decided, "wilson_95": [lo, hi],
                                        "rule": M.rule_reading(c / n_decided, lo, hi)}
    return out


def agreement(a, b, ref):
    """Two raters' clicks on the same cards: distance between them on the heatmap grid,
    and status agreement."""
    ra = {r["id"]: r for r in offsets(a, ref)}
    rb = {r["id"]: r for r in offsets(b, ref)}
    both = [u for u in ref["items"] if u in ra and u in rb]
    dists = [offset_px(ra[u]["x_click"], ra[u]["y_click"], rb[u]["x_click"],
                       rb[u]["y_click"])[2] for u in both]
    sa = {u: (a["verdicts"].get(u) or {}).get("status") for u in ref["items"]}
    sb = {u: (b["verdicts"].get(u) or {}).get("status") for u in ref["items"]}
    answered = [u for u in ref["items"] if sa[u] and sb[u]]
    return {"raters": [a["rater"], b["rater"]], "both_placed": len(both),
            "between_raters": summarize(dists),
            "status_agree": sum(sa[u] == sb[u] for u in answered), "both_answered": len(answered)}


def cmd_score(args):
    if len(args.files) > 2:
        raise SystemExit("score takes one or two click files")
    ref = committed_reference()
    files = [load_verdicts(p, ref) for p in args.files]
    for p, d in zip(args.files, files):
        un = [u for u in ref["items"] if not (d["verdicts"].get(u) or {}).get("status")]
        if un and not args.allow_unanswered:
            raise SystemExit(f"{p}: {len(un)} of {len(ref['items'])} cards unanswered "
                             f"(--allow-unanswered to score anyway)")
    out = {"score": [score(d, ref, args.n_decided) for d in files]}
    if len(files) == 2:
        out["agreement"] = agreement(files[0], files[1], ref)
    print(json.dumps(mv.rnd(out), indent=1, sort_keys=True))


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
PAGE_HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mined label clicks</title>
<style>
:root { --bg:#fff; --fg:#1b1b1b; --muted:#5f6368; --line:#d9d9d9; --focus:#1a73e8; --card:#fafafa; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#141414; --fg:#e8e8e8; --muted:#a0a0a0; --line:#333; --focus:#8ab4f8; --card:#1c1c1c; } }
:root[data-theme="dark"] { --bg:#141414; --fg:#e8e8e8; --muted:#a0a0a0; --line:#333; --focus:#8ab4f8; --card:#1c1c1c; }
body { background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 16px 64px; }
.intro { max-width:80ch; }
.muted { color:var(--muted); }
.bar { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:8px 0; z-index:5; display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; }
.card { border-top:1px solid var(--line); padding:14px 0; }
.card.done h2::after { content:" \\2713"; color:var(--muted); }
.row { display:flex; flex-wrap:wrap; gap:16px; align-items:flex-start; }
.rate { flex:0 1 540px; }
.context { flex:1 1 360px; min-width:0; }
.strip { display:flex; gap:8px; overflow-x:auto; }
figure { margin:0; } figcaption { color:var(--muted); font-size:12px; }
img { max-width:100%; height:auto; display:block; }
.ringwrap { position:relative; display:inline-block; max-width:100%; }
.halo { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
.clickable { cursor:crosshair; }
.mark { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; overflow:visible; }
.mark line { vector-effect:non-scaling-stroke; }
.mark line.k { stroke:#000; stroke-width:4; }
.mark line.w { stroke:#ff2bd6; stroke-width:2; }
fieldset { border:1px solid var(--line); border-radius:6px; margin:8px 0 0; }
.opts { display:flex; flex-wrap:wrap; gap:4px 16px; align-items:center; }
.note { display:block; margin-top:6px; } .note textarea { width:100%; box-sizing:border-box; }
button { font:inherit; }
:focus-visible { outline:2px solid var(--focus); outline-offset:2px; }
__IMGCSS__</style></head><body>
<h1>Mined labels (#158 step 4): where would you place the label?</h1>
<div class="intro"><p>These are the __N__ sample cards you answered <strong>Yes</strong> (pass 1, with
pass 2 over its Can't tells). The ring is where the miner put the label. <strong>Click the point where you
would place the label</strong> on the curb ramp the ring refers to. The script measures how far that is
from the miner's point, in RampNet's training units. Each view opens at 1:1 native pixels (<strong>Z</strong>
fits it to the card), <strong>R</strong> hides the ring, and the image controls are the same as in pass 2.
The context view on the right is another capture, for orientation only.</p>
<dl>__STATUSES__</dl><ul>__RULES__</ul>
<p class="muted">Answers are saved in this browser under your rater id. Export writes
<code>mined_click_check__&lt;rater&gt;.json</code>; your pass-1 id is fine here.</p></div>
<div class="bar"><label>Rater id <input id="rater" type="text" size="10" autocomplete="off"></label>
<span id="progress" class="muted"></span><button type="button" id="export">Export</button></div>
"""

PAGE_JS = r"""
const META = JSON.parse(document.getElementById('meta').textContent);
const LAST_RATER = "mcc158_last_rater";
function getItem(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
function setItem(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
const CARDS = [...document.querySelectorAll('.card')];
const raterInput = document.getElementById('rater');
const RATER_RE = /^[a-z0-9][a-z0-9_-]{0,31}$/;
let rater = (new URLSearchParams(location.search).get('rater') || getItem(LAST_RATER) || "").trim().toLowerCase();
let saved = {};
function storageKey() { return "mcc158_" + META.manifest_digest + "__" + rater; }
function load() {
  saved = {};
  if (!rater) return;
  try { saved = JSON.parse(getItem(storageKey()) || "{}") || {}; } catch (e) { saved = {}; }
}
function persist() { if (rater) setItem(storageKey(), JSON.stringify(saved)); }
function topCard() {
  let best = null, bd = Infinity;
  CARDS.forEach(c => { const r = c.getBoundingClientRect(); const d = Math.abs(r.top - 60); if (r.bottom > 60 && d < bd) { bd = d; best = c; } });
  return best;
}
function drawMark(card) {
  const v = saved[card.dataset.uid] || {};
  const g = card.querySelector('.mark .pt');
  const where = card.querySelector('.where');
  const placed = card.querySelector('input[value=placed]');
  if (v.click) {
    const x = v.click.fx * 100, y = v.click.fy * 100, a = 2.2, b = 3.3;
    const ls = g.querySelectorAll('line');
    [[x - b, y, x + b, y], [x, y - b * 1.5, x, y + b * 1.5]].forEach((p, i) => {
      [ls[i], ls[i + 2]].forEach(l => { l.setAttribute('x1', p[0]); l.setAttribute('y1', p[1]); l.setAttribute('x2', p[2]); l.setAttribute('y2', p[3]); });
    });
    g.setAttribute('visibility', 'visible');
    where.textContent = 'Point placed (' + (v.click.fx * 100).toFixed(1) + '%, ' + (v.click.fy * 100).toFixed(1) + '%). Click again to move it.';
  } else {
    g.setAttribute('visibility', 'hidden');
    where.textContent = v.status && v.status !== 'placed' ? 'No point: ' + v.status.replace('_', ' ') + '.' : 'No point yet.';
  }
  placed.disabled = !v.click;
  card.querySelectorAll('input[type=radio]').forEach(r => { r.checked = (r.value === v.status); });
  card.querySelector('textarea').value = v.note || "";
  card.classList.toggle('done', !!v.status);
}
function progress() {
  const n = CARDS.filter(c => (saved[c.dataset.uid] || {}).status).length;
  document.getElementById('progress').textContent = rater ? (n + ' of ' + CARDS.length + ' answered as ' + rater) : 'Enter a rater id to start.';
}
function renderAll() { CARDS.forEach(drawMark); progress(); }
function entry(card) {
  if (!rater) { alert('Enter a rater id first.'); raterInput.focus(); return null; }
  return saved[card.dataset.uid] = Object.assign(saved[card.dataset.uid] || {}, {});
}
CARDS.forEach(card => {
  const wrap = card.querySelector('.ringwrap');
  wrap.addEventListener('click', ev => {
    const e = entry(card); if (!e) return;
    const r = card.querySelector('.rate figure img').getBoundingClientRect();
    const fx = (ev.clientX - r.left) / r.width, fy = (ev.clientY - r.top) / r.height;
    if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return;
    e.click = {fx: +fx.toFixed(5), fy: +fy.toFixed(5)};
    e.status = 'placed';
    persist(); drawMark(card); progress();
  });
  card.querySelectorAll('input[type=radio]').forEach(inp => inp.addEventListener('change', () => {
    const e = entry(card); if (!e) { inp.checked = false; return; }
    e.status = inp.value;
    if (inp.value !== 'placed') delete e.click;
    persist(); drawMark(card); progress();
  }));
  card.querySelector('.clear').addEventListener('click', () => {
    const e = entry(card); if (!e) return;
    delete e.click; delete e.status;
    persist(); drawMark(card); progress();
  });
  card.querySelector('textarea').addEventListener('input', ev => {
    const e = entry(card); if (!e) return;
    e.note = ev.target.value; persist();
  });
});
raterInput.value = rater;
raterInput.addEventListener('change', () => {
  const r = raterInput.value.trim().toLowerCase();
  if (r && !RATER_RE.test(r)) { alert('Rater id: lowercase letters, digits, - and _ only.'); return; }
  rater = r; setItem(LAST_RATER, rater); load(); renderAll();
});
document.getElementById('export').addEventListener('click', () => {
  if (!rater) { alert('Enter a rater id first.'); return; }
  const verdicts = {};
  META.items.forEach(u => {
    const v = saved[u]; if (!v) return;
    const o = {status: v.status || null, click: v.click || null};
    if (v.note) o.note = v.note;
    if (v.image) o.image = v.image;
    if (o.status || o.note || o.image) verdicts[u] = o;
  });
  const n = META.items.filter(u => (verdicts[u] || {}).status).length;
  const out = Object.assign({}, {task: META.task, question: META.question, statuses: META.statuses,
    rules: META.rules, rater: rater, items: META.items, manifest_digest: META.manifest_digest,
    n_items: META.items.length, n_answered: n, gallery: META.gallery,
    exported_at: new Date().toISOString(), verdicts: verdicts});
  if (n === 0 && !confirm('No card is answered under "' + rater + '". Export anyway?')) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'}));
  a.download = "mined_click_check__" + rater + ".json";
  document.body.appendChild(a); a.click(); a.remove();
});
load(); renderAll();
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--pass1", required=True)
    p.add_argument("--pass2", required=True)
    g = sub.add_parser("gallery")
    g.add_argument("--crops", required=True)
    g.add_argument("--init-rater", default=None)
    s = sub.add_parser("score")
    s.add_argument("files", nargs="+")
    s.add_argument("--n-decided", type=int, default=None,
                   help="decided cards in the yes/no read (83), for precision at a radius")
    s.add_argument("--allow-unanswered", action="store_true")
    args = ap.parse_args(argv)
    {"plan": cmd_plan, "gallery": cmd_gallery, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
