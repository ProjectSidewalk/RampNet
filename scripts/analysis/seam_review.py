"""Seam adjudication: is this ONE physical ramp split by the panorama edge, or TWO ramps?

Issue #130 / #132. Fourteen ground-truth pairs in ``manual_gold`` straddle the 360 seam
closely enough to be a double-mark of one ramp. Eleven sit inside the 22.53 px match
radius; three more sit just outside it, at 23.8-25.1 px. The radius is a *scoring*
boundary, not a duplication boundary, so all fourteen are in the deck.

**This cannot be decided by a rule.** ``manual_gold`` holds 234 within-radius pairs
*away* from the seam, 87 of them with near-identical elevation at the horizon, and those
are overwhelmingly genuine adjacent far-field ramps. A uniform-azimuth null predicts
~3 of the eleven arise by chance. An automatic merge would therefore delete real ramps,
in the direction that flatters our own recall. Hence a human.

## The seam-bias control

A reviewer who can see where the panorama's edge falls may be nudged toward "one ramp"
simply because the edge is there. So every item renders **two views of the same ramps**:

  A. **as stored** - the seam runs through the pair, drawn as a line
  B. **rolled 180 deg** - the panorama is rotated so the seam is half a world away and
     the pair sits in unbroken imagery, with no edge anywhere near it

If B still reads as one ramp, no seam artifact produced that judgment. Toggle freely; the
verdict is recorded once per item, not per view.

Markers are HTML overlays rather than drawn pixels, so they can be hidden entirely and
the raw imagery judged on its own.

## Rubric

Written by the rater (Jon Froehlich) and embedded in every export, because a verdict
whose scheme is unknown cannot be reused or compared. See ``RUBRIC`` below.

Usage:
    python scripts/analysis/seam_review.py --panos-root benchmark --rater jon
    # open the printed index.html, judge, Export, save next to the bundle
"""
import argparse
import base64
import glob
import hashlib
import io
import json
import os
import sys

from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rampnet.detection_eval import PANO_SCALE_X, PANO_SCALE_Y, radius_sq_for  # noqa: E402
from rampnet.geometry import crop_left, dist_to_seam, fold  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

MODEL_W, MODEL_H = 4096, 2048
R = radius_sq_for() ** 0.5              # 22.53 px on the 1024-wide matcher axis
CANDIDATE_MAX_PX = 30.0                 # wrapped separation that puts a pair in the deck
FOV_DEG = 50                            # crop width in degrees of panorama longitude
RENDER_PX = 900

# The labelling rubric this pass is judged under. Version bumps on ANY wording change
# that could move a verdict; exports carry both the text and the version, so two passes
# under different rubrics can never be silently averaged.
RUBRIC_VERSION = 1
RUBRIC = [
    ("A ramp split by the panorama edge is ONE ramp",
     "We are marking physical ramps. The image seam is an artifact of the projection, "
     "not a feature of the world. Use view B (rolled 180 deg) to check yourself: if the "
     "ramps sit in unbroken imagery and still read as one, they are one."),
    ("A driveway is NOT a curb ramp",
     "Absolutely not. Driveway aprons are excluded however ramp-like they look."),
    ("Mark every ramp you can SEE; never mark one you only infer",
     "No marking a ramp because a crossing implies one should be there."),
    ("A partially occluded ramp counts if enough of it is showing",
     "If enough ramp is visible that you can confidently call it a ramp, mark it."),
    ("Unsure is a real answer",
     "If you cannot tell whether it is one ramp or two, say unsure rather than guessing. "
     "An unsure verdict is data; a coerced one is noise."),
    ("Judge at full stored resolution",
     "manual_gold is stored at 4096x2048, which is also the model's input size, so for "
     "this split there is no difference between reviewer pixels and model pixels. Zoom "
     "freely."),
]


def load_marks(pid, labels_dir):
    """The YOLO label file's points for one pano. Malformed lines raise — these are
    committed benchmark artifacts, not fuzzy input."""
    out = []
    with open(os.path.join(labels_dir, f"{pid}.txt"), encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{pid}.txt:{lineno}: expected 'class cx cy w h'")
            out.append((float(parts[1]), float(parts[2])))
    return out


def seam_pairs(labels_dir):
    """Every GT pair straddling the seam within CANDIDATE_MAX_PX, across manual_gold.

    Deliberately wider than the match radius: 11 pairs sit inside it and 3 more at
    23.8-25.1 px, and nothing about duplication respects a scoring threshold.
    """
    items = []
    for path in sorted(glob.glob(os.path.join(labels_dir, "*.txt"))):
        pid = os.path.splitext(os.path.basename(path))[0]
        pts = load_marks(pid, labels_dir)
        for i, a in enumerate(pts):
            for b in pts[i + 1:]:
                if abs(a[0] - b[0]) * PANO_SCALE_X <= PANO_SCALE_X / 2:
                    continue                                   # not spanning the seam
                dx = fold(a[0] - b[0], 1.0) * PANO_SCALE_X
                dy = abs(a[1] - b[1]) * PANO_SCALE_Y
                sep = (dx * dx + dy * dy) ** 0.5
                if sep < CANDIDATE_MAX_PX:
                    items.append({"pano": pid, "a": a, "b": b,
                                  "sep_px": round(sep, 2),
                                  "dx_px": round(dx, 2), "dy_px": round(dy, 2),
                                  "inside_match_radius": bool(sep < R)})
    items.sort(key=lambda it: it["sep_px"])
    return items


def find_pano(panos_root, pid):
    for ext in ("jpg", "jpeg", "png", "webp"):
        hit = glob.glob(os.path.join(panos_root, "manual_gold", "panos", f"{pid}.{ext}"))
        if hit:
            return hit[0]
    return None


def roll(img, frac=0.5):
    w, h = img.size
    shift = int(w * frac)
    out = Image.new(img.mode, img.size)
    out.paste(img.crop((shift, 0, w, h)), (0, 0))
    out.paste(img.crop((0, 0, shift, h)), (w - shift, 0))
    return out


def cut(img, center_x_px, side):
    """A wrapping crop, square, vertically centred on the horizon band."""
    w, h = img.size
    left = crop_left(center_x_px, w, side)
    top = max(0, min(int(round(h / 2 - side / 2)), h - side))
    if left + side <= w:
        return img.crop((left, top, left + side, top + side)), left, top
    out = Image.new(img.mode, (side, side))
    first = w - left
    out.paste(img.crop((left, top, w, top + side)), (0, 0))
    out.paste(img.crop((0, top, side - first, top + side)), (first, 0))
    return out, left, top


def local_pct(x_norm, y_norm, left, top, side, pano_w, pano_h):
    """Marker position inside the crop, as percentages, for a CSS overlay.

    Overlay rather than drawn pixels so the rater can hide the marks entirely and judge
    the raw imagery — and so a marker cannot be mis-drawn, which already happened once
    in this codebase (#132: every marker rendered a quarter of a frame too low).
    """
    dx = ((x_norm * pano_w) - left) % pano_w
    return 100.0 * dx / side, 100.0 * (y_norm * pano_h - top) / side


def b64(img):
    if img.size[0] != RENDER_PX:
        img = img.resize((RENDER_PX, RENDER_PX), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build(items, panos_root):
    side = int(FOV_DEG / 360.0 * MODEL_W)
    cards, missing = [], []
    for it in items:
        path = find_pano(panos_root, it["pano"])
        if path is None:
            missing.append(it["pano"])
            continue
        img = Image.open(path).convert("RGB")
        if img.size != (MODEL_W, MODEL_H):
            img = img.resize((MODEL_W, MODEL_H), Image.BILINEAR)

        # midpoint of the pair, on the wrapped axis
        a, b = it["a"], it["b"]
        mid = ((a[0] + (fold(b[0] - a[0], 1.0) *
                        (1 if ((b[0] - a[0]) % 1.0) < 0.5 else -1)) / 2) % 1.0)
        cx = mid * MODEL_W

        cropA, leftA, topA = cut(img, cx, side)
        rolled = roll(img, 0.5)
        # the same world point, after the roll
        cropB, leftB, topB = cut(rolled, ((mid + 0.5) % 1.0) * MODEL_W, side)

        def marks_for(left, top, shift):
            out = []
            for name, p in (("a", a), ("b", b)):
                x = (p[0] + shift) % 1.0
                px, py = local_pct(x, p[1], left, top, side, MODEL_W, MODEL_H)
                out.append({"id": name, "x": round(px, 3), "y": round(py, 3)})
            return out

        seam_pct, _ = local_pct(0.0, 0.5, leftA, topA, side, MODEL_W, MODEL_H)
        cards.append({
            **{k: v for k, v in it.items() if k not in ("a", "b")},
            "a": [round(a[0], 6), round(a[1], 6)], "b": [round(b[0], 6), round(b[1], 6)],
            "radius_pct": round(100.0 * (R / PANO_SCALE_X * MODEL_W) / side, 3),
            "seam_pct": round(seam_pct, 3) if 0 <= seam_pct <= 100 else None,
            "viewA": b64(cropA), "marksA": marks_for(leftA, topA, 0.0),
            "viewB": b64(cropB), "marksB": marks_for(leftB, topB, 0.5),
        })
    return cards, missing


def digest(cards):
    """Identity of this exact item list, so two raters cannot be compared across
    different decks without it being obvious."""
    spec = ";".join(f"{c['pano']}|{c['a']}|{c['b']}" for c in cards)
    return hashlib.sha256(spec.encode("utf-8")).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panos-root", default=os.path.join(REPO, "benchmark"))
    ap.add_argument("--labels-dir", default=os.path.join(REPO, "manual_labels"))
    ap.add_argument("--rater", required=True,
                    help="Rater id; the export is named for it so passes never collide.")
    ap.add_argument("--out", default=os.path.join(REPO, "analysis_out", "seam_review"))
    ap.add_argument("--blind", action="store_true",
                    help="Hide each pair's separation from the rater. The first pass "
                         "(rater jon) showed it, and the verdicts came out perfectly "
                         "rank-ordered by separation -- which is either the real signal "
                         "or anchoring, and that pass cannot tell you which. Any "
                         "SECOND rater should use this.")
    args = ap.parse_args()

    items = seam_pairs(args.labels_dir)
    inside = sum(1 for it in items if it["inside_match_radius"])
    print(f"seam pairs within {CANDIDATE_MAX_PX:.0f} px: {len(items)} "
          f"({inside} inside the {R:.2f} px match radius, {len(items) - inside} outside)")

    cards, missing = build(items, args.panos_root)
    if missing:
        print(f"skipped, no local imagery: {missing}")
    dg = digest(cards)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"index_{args.rater}.html")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(render(cards, args.rater, dg, args.blind))
    print(f"items rendered: {len(cards)}   manifest digest: {dg}")
    print(f"\nwrote {path}")


def render(cards, rater, dg, blind=False):
    rubric_html = "".join(
        f"<li><b>{t}</b><br><span class='rq'>{d}</span></li>" for t, d in RUBRIC)
    if blind:
        # strip the numbers the card header prints; the geometry the overlay needs stays
        cards = [{**c, "sep_px": None, "dx_px": None, "dy_px": None} for c in cards]
    payload = json.dumps({"cards": cards, "rater": rater, "digest": dg, "blind": blind,
                          "rubric_version": RUBRIC_VERSION,
                          "rubric": [{"rule": t, "detail": d} for t, d in RUBRIC]})
    return TEMPLATE.replace("__PAYLOAD__", payload).replace("__RUBRIC__", rubric_html) \
                   .replace("__RATER__", rater).replace("__DIGEST__", dg) \
                   .replace("__N__", str(len(cards)))


TEMPLATE = r"""<!-- generated by scripts/analysis/seam_review.py (#130, #132) -->
<meta charset="utf-8"><title>Seam adjudication</title>
<style>
 body{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#14161a;color:#e8e8ea}
 .wrap{max-width:1000px;margin:0 auto;padding:24px 20px 120px}
 h1{font-size:24px;margin:0 0 4px} .sub{color:#9aa0a8;margin:0 0 18px;font-size:14px}
 details{background:#1c1f25;border:1px solid #2c313a;border-radius:10px;padding:12px 16px;margin:0 0 18px}
 summary{cursor:pointer;font-weight:600} ol{margin:10px 0 0;padding-left:20px}
 li{margin:0 0 8px} .rq{color:#9aa0a8;font-size:13.5px}
 .bar{position:sticky;top:0;z-index:10;background:#14161ae6;backdrop-filter:blur(6px);
   padding:10px 0;margin:0 0 14px;border-bottom:1px solid #2c313a;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 button{background:#252a33;color:#e8e8ea;border:1px solid #39404c;border-radius:8px;
   padding:7px 13px;font-size:14px;cursor:pointer}
 button:hover{background:#2f3540}
 button.on{background:#1a9c3e;border-color:#1a9c3e;color:#fff}
 .card{background:#1c1f25;border:1px solid #2c313a;border-radius:12px;padding:16px;margin:0 0 18px}
 .card.done{border-color:#1a9c3e}
 .head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
 .meta{color:#9aa0a8;font-size:13px;margin:2px 0 12px}
 .stage{position:relative;width:100%;aspect-ratio:1;border-radius:10px;overflow:hidden;background:#000}
 .stage img{position:absolute;inset:0;width:100%;height:100%;display:none}
 .stage img.show{display:block}
 .ring{position:absolute;border:3px solid #ffd400;border-radius:50%;transform:translate(-50%,-50%);
   box-shadow:0 0 0 1px #0008, inset 0 0 0 1px #0008;pointer-events:none}
 .dot{position:absolute;width:9px;height:9px;background:#ffd400;border:2px solid #000;
   border-radius:50%;transform:translate(-50%,-50%);pointer-events:none}
 .seam{position:absolute;top:0;bottom:0;width:0;border-left:3px dashed #00e5ff;pointer-events:none}
 .hidem .ring,.hidem .dot{display:none}
 .hides .seam{display:none}
 .vote{display:flex;gap:8px;margin:12px 0 0;flex-wrap:wrap;align-items:center}
 .vote .one.on{background:#1a9c3e;border-color:#1a9c3e}
 .vote .two.on{background:#0b6bcb;border-color:#0b6bcb}
 .vote .unsure.on{background:#8a6d1f;border-color:#8a6d1f}
 input[type=text]{flex:1;min-width:220px;background:#0f1115;border:1px solid #39404c;
   color:#e8e8ea;border-radius:8px;padding:7px 10px;font-size:14px}
 .tag{font-size:12px;padding:2px 8px;border-radius:99px;background:#2a2f38;color:#9aa0a8}
 #done{font-weight:700}
</style>
<div class="wrap">
<h1>Seam adjudication &mdash; one ramp, or two?</h1>
<p class="sub">rater <b>__RATER__</b> &middot; __N__ items &middot; manifest <code>__DIGEST__</code></p>

<details open><summary>Rubric (version 1) &mdash; embedded in the export</summary>
<ol>__RUBRIC__</ol></details>

<div class="bar">
  <button id="tv">View: <b>A &mdash; as stored</b></button>
  <button id="tm" class="on">marks on</button>
  <button id="ts" class="on">seam on</button>
  <span class="tag">A = seam runs through the pair &middot; B = rolled 180&deg;, no edge nearby</span>
  <span style="flex:1"></span>
  <span id="done">0 / __N__</span>
  <button id="ex">Export verdicts</button>
</div>
<div id="cards"></div>
</div>
<script>
const D = __PAYLOAD__;
const V = {};                     // pano|a|b -> {verdict, note}
let view = 'A', marks = true, seam = true;
const key = c => c.pano + '|' + c.a.join(',') + '|' + c.b.join(',');

function draw() {
  const root = document.getElementById('cards');
  root.innerHTML = '';
  D.cards.forEach((c, i) => {
    const k = key(c), v = V[k] || {};
    const el = document.createElement('div');
    el.className = 'card' + (v.verdict ? ' done' : '');
    const ms = (view === 'A' ? c.marksA : c.marksB).map(m =>
      `<div class="ring" style="left:${m.x}%;top:${m.y}%;width:${c.radius_pct*2}%;height:${c.radius_pct*2}%"></div>
       <div class="dot" style="left:${m.x}%;top:${m.y}%"></div>`).join('');
    const sm = (view === 'A' && c.seam_pct !== null)
      ? `<div class="seam" style="left:${c.seam_pct}%"></div>` : '';
    el.innerHTML = `
      <div class="head"><b>${i+1}. ${c.pano}</b>
        ${c.sep_px === null ? '' : `<span class="tag">${c.sep_px} px apart &middot; ${(c.sep_px/1024*360).toFixed(2)}&deg;
        &middot; ${c.inside_match_radius ? 'inside' : 'outside'} the match radius</span>`}</div>
      <p class="meta">${c.sep_px === null ? 'separation hidden (blind pass)' :
        `dx ${c.dx_px} px &middot; dy ${c.dy_px} px`} &middot;
        rings show the scorer's match radius, not the ramp's extent</p>
      <div class="stage ${marks?'':'hidem'} ${seam?'':'hides'}">
        <img class="${view==='A'?'show':''}" src="${c.viewA}">
        <img class="${view==='B'?'show':''}" src="${c.viewB}">
        ${ms}${sm}
      </div>
      <div class="vote">
        <button class="one ${v.verdict==='one'?'on':''}" data-k="${k}" data-v="one">ONE ramp</button>
        <button class="two ${v.verdict==='two'?'on':''}" data-k="${k}" data-v="two">TWO ramps</button>
        <button class="unsure ${v.verdict==='unsure'?'on':''}" data-k="${k}" data-v="unsure">unsure</button>
        <input type="text" placeholder="note (optional)" data-note="${k}" value="${(v.note||'').replace(/"/g,'&quot;')}">
      </div>`;
    root.appendChild(el);
  });
  document.querySelectorAll('[data-v]').forEach(b => b.onclick = () => {
    const k = b.dataset.k;
    V[k] = Object.assign({}, V[k], {verdict: b.dataset.v});
    save(); draw();
  });
  document.querySelectorAll('[data-note]').forEach(inp => inp.onchange = () => {
    const k = inp.dataset.note;
    V[k] = Object.assign({}, V[k], {note: inp.value});
    save();
  });
  document.getElementById('done').textContent =
    Object.values(V).filter(v => v.verdict).length + ' / ' + D.cards.length;
}
function save(){ localStorage.setItem('seam_' + D.digest + '_' + D.rater, JSON.stringify(V)); }
(function load(){
  const raw = localStorage.getItem('seam_' + D.digest + '_' + D.rater);
  if (raw) Object.assign(V, JSON.parse(raw));
})();
document.getElementById('tv').onclick = e => {
  view = view === 'A' ? 'B' : 'A';
  e.currentTarget.innerHTML = 'View: <b>' + (view === 'A' ? 'A &mdash; as stored'
    : 'B &mdash; rolled 180&deg;') + '</b>';
  draw();
};
document.getElementById('tm').onclick = e => {
  marks = !marks; e.currentTarget.classList.toggle('on', marks);
  e.currentTarget.textContent = 'marks ' + (marks ? 'on' : 'off'); draw();
};
document.getElementById('ts').onclick = e => {
  seam = !seam; e.currentTarget.classList.toggle('on', seam);
  e.currentTarget.textContent = 'seam ' + (seam ? 'on' : 'off'); draw();
};
document.getElementById('ex').onclick = () => {
  const out = {
    rater: D.rater, manifest_digest: D.digest,
    rubric_version: D.rubric_version, rubric: D.rubric,
    verdicts: D.cards.map(c => {
      const v = V[key(c)] || {};
      return {pano: c.pano, a: c.a, b: c.b, sep_px: c.sep_px,
              inside_match_radius: c.inside_match_radius,
              verdict: v.verdict || null, note: v.note || null};
    })
  };
  const blob = new Blob([JSON.stringify(out, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'seam_verdicts__' + D.rater + '.json';
  a.click();
};
draw();
</script>
"""


if __name__ == "__main__":
    main()
