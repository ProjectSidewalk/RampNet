"""Rated gallery of mined labels (#158 step 4): is there a curb ramp at the ring?

The labeler's city-wide miner (``sidewalk-auto-labeler/scripts/mine_city.py``, the step-3
``peak_flat`` rule) emits training labels on panos no benchmark verdict has seen. Their
precision can only be measured by looking. This script samples them by the rule fixed in the
labeler's ``docs/mined-precision.md`` (Step 4 plan) before any crop was cut, builds a
one-question gallery in the shape of the #48 GT check (``residual_gt_check_48.py``, whose
rater-id, manifest-digest, integrity and agreement machinery it reuses), and scores the
per-rater verdict files against #158's rule.

    # 1. plan (desktop; the labeler's committed step-4 labels and step-3 files)
    python scripts/analysis/mined_label_check_158.py plan --labeler-root ../sidewalk-auto-labeler
    # 2. crops (makelab2, the native-res archive) -- the #48 cutter, unchanged
    python scripts/analysis/multiview_evidence_48.py cut-crops \\
        analysis_out/mined_label_check_158/items.json \\
        --archive-root /projects/makeabilitylab/sidewalk-auto-labeler/runs --out crops
    # 3. gallery + empty per-rater file
    python scripts/analysis/mined_label_check_158.py gallery --crops crops --init-rater jonf
    # 4. precision (one file), or precision + agreement (two files)
    python scripts/analysis/mined_label_check_158.py rates \\
        analysis_out/mined_label_check_158/mined_label_check__jonf.json [..__<rater2>.json]
    # 5. pass 2 (after pass 1 is read): the 'Can't tell' cards, re-cut native and unringed
    python scripts/analysis/mined_label_check_158.py plan-pass2 \\
        --pass2-from analysis_out/mined_label_check_158/mined_label_check__jonf.json
    python scripts/analysis/multiview_evidence_48.py cut-crops \\
        analysis_out/mined_label_check_158/items_pass2.json \\
        --archive-root /projects/makeabilitylab/sidewalk-auto-labeler/runs --out crops_p2  # makelab2
    python scripts/analysis/mined_label_check_158.py gallery \\
        --pass2-from analysis_out/mined_label_check_158/mined_label_check__jonf.json \\
        --crops crops_p2 --init-rater jonf-p2
    python scripts/analysis/mined_label_check_158.py rates \\
        analysis_out/mined_label_check_158/mined_label_check__jonf.json \\
        --pass2 analysis_out/mined_label_check_158/mined_label_check__jonf-p2.json

**The sample (fixed before crops).** Population: every emitted label whose target pano is
not one of the 124 benchmark panos. 100 labels, allocated to the range bands 0-8 / 8-12 /
12-15 m in proportion to the population (largest remainder), drawn uniformly without
replacement within each band with ``random.Random(SEED)`` over the labels sorted by
(site_id, pano_id). **Instrument items:** 10 drawn with the same seed from step 3's richmond
``peak_flat`` targets that the benchmark adjudicated tp or fp; known answer Yes for tp, No for
fp. They are flagged in ``items.json`` only -- never on a card or in a per-rater file -- and
scored apart from the precision, as agreement with Jon's earlier verdicts.

**Cards.** One ringed crop of the target pano at the emitted pixel (36 x 24 deg, the #48
``cut_one``) and one unringed context crop of the source-rule member pano at its detection.
No confidence, range, band or instrument flag is shown; cards are shuffled with SEED and
titled by an opaque id.

**Precision** = Yes / (Yes + No) over the 100 sample items, 95% Wilson, pooled and per band,
read against #158's rule (>= 0.80 build, 0.50-0.80 add the visibility test, < 0.50 drop) on
the point estimate and the interval. Can't tell is excluded and counted.
"""
import argparse
import csv
import hashlib
import html
import json
import os
import random
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import multiview_evidence_48 as mv  # noqa: E402
import residual_gt_check_48 as R  # noqa: E402

OUT = os.path.join(mv.OUT_ROOT, "mined_label_check_158")
ITEMS_PATH = os.path.join(OUT, "items.json")
GALLERY_DIR = os.path.join(mv.BENCHMARK, "mined_label_check_158")
MANIFEST_PATH = os.path.join(GALLERY_DIR, "manifest.json")
GALLERY_REL = "benchmark/mined_label_check_158/gallery.html"
#: Pass 2: the pass-1 "Can't tell" cards only, with image controls (added after pass 1 was
#: read; Jon's notes said brightness / contrast / saturation would decide most of them).
GALLERY2_REL = "benchmark/mined_label_check_158/gallery_pass2.html"
PASS2_SUBSET = "cant_tell"
#: Pass 2 rates new cuts of the same 32 target views: native resolution (no resize to
#: 360 x 240) and no baked ring, so the ring is a page overlay the rater can hide. They
#: have their own plan, crop dir, manifest and digest; a pass-2 file carries both digests.
ITEMS2_PATH = os.path.join(OUT, "items_pass2.json")
CROPS2_DIR = "crops_pass2"
CROPS2_REL = "benchmark/mined_label_check_158/" + CROPS2_DIR
MANIFEST2_PATH = os.path.join(GALLERY_DIR, "manifest_pass2.json")
PASS2_QUALITY = 90
EXPORT_PREFIX, EXPORT_SUFFIX = "mined_label_check__", ".json"
RATER_RE = R.RATER_RE

SEED = 158
N_SAMPLE = 100
N_INSTRUMENT = 10
BANDS = ("0-8 m", "8-12 m", "12-15 m")
CITY = "richmond"
#: Relative to --labeler-root: the committed inputs the plan reads.
LABELS_REL = "docs/figures/mined-precision/data/step4/richmond/labels.jsonl"
STEP3_PLACEMENT_REL = "docs/figures/mined-precision/data/placement/peak_flat.jsonl"
STEP3_CANDS_REL = "docs/figures/mined-precision/data/frozen/peak_flat/richmond/candidates.csv"
RULE_BUILD, RULE_VISIBILITY = 0.80, 0.50

QUESTION = "Is there a curb ramp at the ring?"
RUBRIC = [
    ("yes", "Yes",
     "A curb ramp is at the ring or touching it. It may be partly hidden or faint, as long "
     "as you can see it is a ramp."),
    ("no", "No",
     "There is no curb ramp at that spot: the ring is on plain curb, a driveway, sidewalk, "
     "street or something else, and the nearest ramp (if any) is more than roughly one ramp "
     "width away."),
    ("cant_tell", "Can't tell",
     "The view does not let you decide (too dark, blocked, too far, too blurry). Excluded "
     "from every rate."),
]
ANSWERS = tuple(k for k, _, _ in RUBRIC)
RULES = [
    "Answer from the ringed view only. The second view is unmarked context and is not rated.",
    "Judge the spot under the ring, not whether a ramp exists somewhere in the crop.",
    "Add a note for anything worth recording, e.g. 'ramp 1 m left of ring' or 'ring on a "
    "driveway'.",
]


def verdicts_path(rater):
    if not RATER_RE.match(rater or ""):
        raise ValueError(f"rater id {rater!r} must match {RATER_RE.pattern}")
    return os.path.join(OUT, EXPORT_PREFIX + rater + EXPORT_SUFFIX)


def opaque_id(site_id, pano_id):
    return "c" + hashlib.sha256(f"{CITY}:{site_id}:{pano_id}".encode()).hexdigest()[:8]


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def allocate(counts, n):
    """Largest-remainder proportional allocation of n over {band: population}."""
    total = sum(counts.values())
    if total <= n:
        return dict(counts)
    quota = {b: n * c / total for b, c in counts.items()}
    alloc = {b: int(q) for b, q in quota.items()}
    rest = sorted(counts, key=lambda b: (-(quota[b] - alloc[b]), BANDS.index(b)))
    for b in rest[:n - sum(alloc.values())]:
        alloc[b] += 1
    return alloc


def draw_sample(labels, n=N_SAMPLE, seed=SEED):
    """The fixed sampling rule over emitted, non-benchmark labels."""
    pop = sorted((r for r in labels if r["emit"] and not r["benchmark_pano"]),
                 key=lambda r: (r["site_id"], r["pano_id"]))
    by_band = {b: [r for r in pop if r["band"] == b] for b in BANDS}
    stray = [r for r in pop if r["band"] not in BANDS]
    if stray:
        raise SystemExit(f"{len(stray)} emitted labels outside the bands, e.g. {stray[0]}")
    alloc = allocate({b: len(v) for b, v in by_band.items()}, n)
    rng = random.Random(seed)
    out = []
    for b in BANDS:
        out += rng.sample(by_band[b], alloc[b])
    return out, alloc, {b: len(v) for b, v in by_band.items()}


def draw_instrument(labels, step3_rows, cands, n=N_INSTRUMENT, seed=SEED):
    """Step 3's richmond peak_flat targets adjudicated tp or fp, with their known answer.
    The city-wide label for the same (site, pano) must carry the same pixel."""
    bucket = {(int(c["site_id"]), c["pano_id"]): c["bucket"] for c in cands}
    by_key = {(r["site_id"], r["pano_id"]): r for r in labels}
    pool = []
    for s in step3_rows:
        if s["city"] != CITY or not s.get("emit"):
            continue
        k = (int(s["site_id"]), str(s["pano_id"]))
        b = bucket.get(k)
        if b not in ("tp", "fp", "false_det_nearby"):
            continue
        lab = by_key.get(k)
        if lab is None or not lab["emit"] or abs(lab["x"] - s["x"]) > 1e-9 \
                or abs(lab["y"] - s["y"]) > 1e-9:
            raise SystemExit(f"city-wide miner does not reproduce step 3's target {k}")
        pool.append((k, "yes" if b == "tp" else "no", lab))
    pool.sort(key=lambda t: t[0])
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool))), len(pool)


def crop_items(uid, lab):
    """Two crops per card: the target (ringed) and the source-rule member (unringed)."""
    items = [{"ramp_uid": uid, "city": CITY, "pano_id": lab["pano_id"], "x": lab["x"],
              "y": lab["y"], "ring": True, "is_source": True, "role": "target",
              "capture_date": lab.get("capture_date") or ""}]
    if lab.get("src_pano"):
        items.append({"ramp_uid": uid, "city": CITY, "pano_id": lab["src_pano"],
                      "x": lab["src_x"], "y": lab["src_y"], "ring": False,
                      "is_source": False, "role": "context",
                      "capture_date": lab.get("src_capture_date") or ""})
    return items


def cmd_plan(args):
    root = args.labeler_root
    labels = [json.loads(line) for line in open(os.path.join(root, LABELS_REL), encoding="utf-8")
              if line.strip()]
    step3 = [json.loads(line) for line in
             open(os.path.join(root, STEP3_PLACEMENT_REL), encoding="utf-8") if line.strip()]
    with open(os.path.join(root, STEP3_CANDS_REL), encoding="utf-8", newline="") as f:
        cands = list(csv.DictReader(f))
    sample, alloc, pop = draw_sample(labels)
    instr, n_pool = draw_instrument(labels, step3, cands)
    cards = [{"id": opaque_id(r["site_id"], r["pano_id"]), "site_id": r["site_id"],
              "pano_id": r["pano_id"], "band": r["band"], "range_m": r["range_m"],
              "peak_conf": r["peak_conf"], "instrument": False, "known_answer": None}
             for r in sample]
    lab_of = {(r["site_id"], r["pano_id"]): r for r in labels}
    for (k, ans, lab) in instr:
        cards.append({"id": opaque_id(*k), "site_id": k[0], "pano_id": k[1],
                      "band": lab["band"], "range_m": lab["range_m"],
                      "peak_conf": lab["peak_conf"], "instrument": True,
                      "known_answer": ans})
    if len({c["id"] for c in cards}) != len(cards):
        raise SystemExit("opaque id collision")
    random.Random(SEED).shuffle(cards)
    crops = [it for c in cards for it in crop_items(c["id"], lab_of[(c["site_id"], c["pano_id"])])]
    sha = {rel: R.sha256_file(os.path.join(root, rel))
           for rel in (LABELS_REL, STEP3_PLACEMENT_REL, STEP3_CANDS_REL)}
    mv.write_json(ITEMS_PATH, {
        "seed": SEED, "n_sample": len(sample), "n_instrument": len(instr),
        "instrument_pool": n_pool, "population_by_band": pop, "allocation": alloc,
        "inputs_sha256": sha, "cards": cards, "items": crops, "n": len(crops)})
    print(f"{len(sample)} sampled ({alloc} of {pop}) + {len(instr)} instrument "
          f"(pool {n_pool}) -> {len(cards)} cards, {len(crops)} crops -> {ITEMS_PATH}")


# --------------------------------------------------------------------------- #
# gallery
# --------------------------------------------------------------------------- #
def empty_verdicts(ids, digest, rater, pass2=None):
    return {"task": "RampNet #158 step 4, mined-label check: " + QUESTION,
            "question": QUESTION,
            "rubric": [{"key": k, "label": lab, "definition": d} for k, lab, d in RUBRIC],
            "rules": RULES, "rater": rater, "items": ids, "manifest_digest": digest,
            "n_items": len(ids), "n_answered": 0, "gallery": GALLERY_REL, "verdicts": {},
            **(pass2 or {})}


def pass2_meta(pass1_path, pass1_digest):
    """The fields a pass-2 plan and file carry, binding them to the pass-1 file they
    re-rate and to the pass-1 gallery (whose context crops pass 2 still shows)."""
    try:
        rel = os.path.relpath(pass1_path, mv.REPO)
    except ValueError:                  # Windows: another drive has no relative path
        rel = os.path.abspath(pass1_path)
    return {"pass": 2, "subset": PASS2_SUBSET, "image_controls": True, "ring_overlay": True,
            "native_crops": True, "crops": CROPS2_REL, "gallery": GALLERY2_REL,
            "pass1_manifest_digest": pass1_digest,
            "from_file": rel.replace(os.sep, "/"), "from_sha256": R.sha256_file(pass1_path)}


def pass2_items(pass1, ref):
    """The pass-1 'Can't tell' cards, in the committed item order (sample and check items alike)."""
    return [u for u in ref["items"]
            if (pass1["verdicts"].get(u) or {}).get("answer") == PASS2_SUBSET]


#: Per-card image controls, applied to that card's two views and saved with the card's
#: answer (`image`, slider units; the defaults below mean "unchanged") so a re-rating can be
#: reproduced. Levels (black / white point), gamma and local contrast are an SVG filter per
#: card; brightness, contrast and saturation are CSS filters on top of it. Everything is a
#: browser filter on the committed JPEG: no pixel is read or written, so the page works from
#: a file:// URL, and a value the camera wrote as white stays white whatever the setting.
IMAGE_CONTROLS_CSS = (
    ".card img { filter: var(--svgf) brightness(var(--br,1)) contrast(var(--ct,1)) "
    "saturate(var(--sa,1)); }\n"
    ".card .fdef { position:absolute; width:0; height:0; overflow:hidden; }\n"
    ".imgctl { display:flex; flex-wrap:wrap; gap:2px 12px; align-items:center; margin:6px 0 0; }\n"
    ".imgctl label { font-size:13px; color:var(--muted); }\n"
    ".imgctl input[type=range] { vertical-align:middle; width:90px; }\n"
    ".imgctl button { padding:2px 8px; font-size:13px; }\n"
    ".imgctl button[aria-pressed=true] { box-shadow: inset 0 0 0 2px var(--focus); }\n"
    ".ringwrap { display:inline-block; max-width:100%; vertical-align:top; }\n"
    ".halo.ring circle { fill:none; vector-effect:non-scaling-stroke; }\n"
    ".halo.ring circle.g { stroke:#00ff5a; stroke-width:2; }\n"
    ".halo.ring circle.h { stroke:#000; stroke-opacity:.9; stroke-width:1.5; }\n"
    ".card.noring .halo.ring { display:none; }\n"
    ".card.zoomed .rate { flex:1 1 100%; }\n"
    ".imgctl .washout { font-weight:600; color:var(--fg); }\n"
    ".imgctl .washout input[type=range] { width:160px; }\n")
#: The washout slider is a shortcut, not a control of its own: at t (0-100) it sets black
#: point 170*t/100, local contrast 150*t/100 and saturation 100+40*t/100 on the sliders
#: after it (the pass-1 washed-out cards have 30-76% of the ring area at >= 245, the
#: surviving detail sits above ~170). Only those sliders' values are saved, so an export
#: reads the same whether the rater used the shortcut or the sliders.
WASHOUT = (("bp", 0, 170), ("lc", 0, 150), ("sa", 100, 140))
#: (key, label, min, max, default); 100 = x1.0 for the multiplicative ones.
IMAGE_RANGES = (("bp", "Black point", 0, 250, 0), ("wp", "White point", 5, 255, 255),
                ("gm", "Gamma", 30, 300, 100), ("lc", "Local contrast", 0, 300, 0),
                ("br", "Brightness", 40, 250, 100), ("ct", "Contrast", 40, 250, 100),
                ("sa", "Saturation", 0, 300, 100))


def _fe_funcs(kind, attrs):
    return "".join(f'<feFunc{c} type="{kind}" {attrs} data-fn="{kind}"/>' for c in "RGB")


def image_controls_html(uid, pass2=False):
    sliders = "".join(
        f'<label>{lab} <input type="range" data-img="{k}" min="{lo}" max="{hi}" value="{dflt}" '
        f'aria-label="{lab} for card {uid}"></label>' for k, lab, lo, hi, dflt in IMAGE_RANGES)
    fdef = (f'<svg class="fdef" aria-hidden="true" focusable="false">'
            f'<filter id="f_{uid}" color-interpolation-filters="sRGB">'
            f'<feComponentTransfer in="SourceGraphic" result="lv">'
            f'{_fe_funcs("linear", "slope=\"1\" intercept=\"0\"")}</feComponentTransfer>'
            f'<feComponentTransfer in="lv" result="gm">'
            f'{_fe_funcs("gamma", "amplitude=\"1\" exponent=\"1\" offset=\"0\"")}'
            f'</feComponentTransfer>'
            f'<feGaussianBlur in="gm" stdDeviation="0" result="bl"/>'
            f'<feComposite in="gm" in2="bl" operator="arithmetic" k1="0" k2="1" k3="0" k4="0"/>'
            f'</filter></svg>')
    extra = (('<button type="button" class="ring_toggle" aria-pressed="true">Hide ring (R)</button>'
              '<button type="button" class="zoom_toggle" aria-pressed="true">Zoom fit (Z)</button>')
             if pass2 else "")
    washout = (f'<label class="washout">Washout fix <input type="range" data-washout min="0" '
               f'max="100" value="0" aria-label="Washout fix for card {uid}: moves black point, '
               f'local contrast and saturation together"></label>')
    return (f'{fdef}<div class="imgctl" role="group" aria-label="Image controls for card {uid}">'
            f'{washout}{sliders}<button type="button" class="img_reset">Reset image</button>{extra}</div>')


IMAGE_CONTROLS_JS = """
const WASHOUT = """ + json.dumps(WASHOUT) + """;
function imgState(card) {
  const s = {};
  card.querySelectorAll('.imgctl input[data-img]').forEach(inp => { s[inp.dataset.img] = +inp.value; });
  return s;
}
function imgDefault(inp) { return +inp.defaultValue; }
function rateImg(card) { return card.querySelector('.rate figure img'); }
function applyImg(card) {
  const s = imgState(card);
  const v = (k, d) => (s[k] == null ? d : s[k]);
  card.style.setProperty('--br', v('br', 100) / 100);
  card.style.setProperty('--ct', v('ct', 100) / 100);
  card.style.setProperty('--sa', v('sa', 100) / 100);
  const f = card.querySelector('.fdef filter');
  if (!f) return;
  const lo = v('bp', 0) / 255, hi = v('wp', 255) / 255;
  const slope = hi > lo ? 1 / (hi - lo) : 1;
  f.querySelectorAll('[data-fn=linear]').forEach(e => { e.setAttribute('slope', slope); e.setAttribute('intercept', -lo * slope); });
  f.querySelectorAll('[data-fn=gamma]').forEach(e => { e.setAttribute('exponent', v('gm', 100) / 100); });
  const a = v('lc', 0) / 100;
  const img = rateImg(card);
  const sd = a > 0 ? Math.max(2, 0.025 * ((img && img.clientWidth) || 540)) : 0;
  f.querySelector('feGaussianBlur').setAttribute('stdDeviation', sd);
  const comp = f.querySelector('feComposite');
  comp.setAttribute('k2', 1 + a); comp.setAttribute('k3', -a);
}
function renderImg() {
  CARDS.forEach(card => {
    const s = (saved[card.dataset.uid] || {}).image || {};
    card.querySelectorAll('.imgctl input[data-img]').forEach(inp => { inp.value = s[inp.dataset.img] == null ? imgDefault(inp) : s[inp.dataset.img]; });
    applyImg(card);
  });
}
function saveImg(card) {
  if (!rater) return;
  saved[card.dataset.uid] = Object.assign(saved[card.dataset.uid] || {}, {image: imgState(card)});
  persist();
}
function setZoom(card, on) {
  card.classList.toggle('zoomed', on);
  const img = rateImg(card);
  if (img) {
    const nat = img.naturalWidth || +((img.dataset.natural || "").split(" ")[0]) || 0;
    const fit = +img.getAttribute('width') || 0;
    img.style.width = (on && nat) ? Math.max(nat, fit) + 'px' : '';
  }
  const b = card.querySelector('.zoom_toggle');
  if (b) { b.setAttribute('aria-pressed', on); b.textContent = on ? 'Zoom fit (Z)' : 'Zoom 1:1 (Z)'; }
  applyImg(card);
}
function setRing(card, shown) {
  card.classList.toggle('noring', !shown);
  const b = card.querySelector('.ring_toggle');
  if (b) { b.setAttribute('aria-pressed', shown); b.textContent = shown ? 'Hide ring (R)' : 'Show ring (R)'; }
}
CARDS.forEach(card => {
  card.style.setProperty('--svgf', 'url(#f_' + card.dataset.uid + ')');
  card.querySelectorAll('.imgctl input[data-img]').forEach(inp => {
    inp.addEventListener('input', () => { applyImg(card); saveImg(card); });
  });
  const wo = card.querySelector('.imgctl input[data-washout]');
  if (wo) wo.addEventListener('input', () => {
    const t = +wo.value / 100;
    WASHOUT.forEach(([k, a, b]) => {
      const inp = card.querySelector('.imgctl input[data-img="' + k + '"]');
      if (inp) inp.value = Math.round(a + (b - a) * t);
    });
    applyImg(card); saveImg(card);
  });
  card.querySelector('.img_reset').addEventListener('click', () => {
    if (wo) wo.value = 0;
    card.querySelectorAll('.imgctl input[data-img]').forEach(inp => { inp.value = imgDefault(inp); });
    applyImg(card); saveImg(card);
  });
  const rt = card.querySelector('.ring_toggle');
  if (rt) rt.addEventListener('click', () => setRing(card, card.classList.contains('noring')));
  const zt = card.querySelector('.zoom_toggle');
  if (zt) {
    zt.addEventListener('click', () => setZoom(card, !card.classList.contains('zoomed')));
    setZoom(card, true);
  }
});
document.addEventListener('keydown', ev => {
  const t = ev.target;
  if (t.tagName === 'TEXTAREA' || (t.tagName === 'INPUT' && t.type === 'text')) return;
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const k = (ev.key || "").toLowerCase();
  if (k !== 'r' && k !== 'z') return;
  const card = (t.closest && t.closest('.card')) || topCard();
  if (!card || !card.querySelector('.ring_toggle')) return;
  ev.preventDefault();
  if (k === 'r') setRing(card, card.classList.contains('noring'));
  else setZoom(card, !card.classList.contains('zoomed'));
});
renderImg();
raterInput.addEventListener('change', renderImg);
"""


def _role(it):
    return it.get("role") or ("target" if it["ring"] else "context")


def _ring_overlay(it):
    """Pass 2: the ring as an SVG overlay in pass-1 crop units (the crop keeps the 3:2
    window, so the fractions hold at any resolution), same radius as the baked ring plus
    its dark halo. It is drawn by the page, so the rater can hide it (R)."""
    fx, fy = R.ring_centre_frac(it["x"], it["y"])
    w, h = mv.CROP_PX
    cx, cy = round(fx * w, 2), round(fy * h, 2)
    return (f'<svg class="halo ring" viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
            f'aria-hidden="true" focusable="false">'
            f'<circle class="h" cx="{cx}" cy="{cy}" r="15.75"/>'
            f'<circle class="g" cx="{cx}" cy="{cy}" r="14"/>'
            f'<circle class="h" cx="{cx}" cy="{cy}" r="10.75"/></svg>')


def _figure(it, uid, width, pass2=False):
    name = mv.crop_name(it)
    date = html.escape(it["capture_date"] or "date unknown")
    target = _role(it) == "target"
    if target:
        alt, cap, cls = (f"Card {uid}: the view to rate, ring on the spot",
                         f"<strong>Rate this view</strong> &middot; {date}", "src")
        if pass2 and it.get("px"):
            cap += (f" &middot; native {it['px'][0]}&times;{it['px'][1]} px, shown at {width} "
                    f"(Z for 1:1)")
    else:
        alt, cap, cls = (f"Card {uid}: another capture of the corner, unmarked",
                         f"Context: another capture of the corner &middot; {date} &middot; "
                         f"unmarked, not rated", "ctx")
    sub = CROPS2_DIR if (pass2 and target) else "crops"
    natural = (f' data-natural="{it["px"][0]} {it["px"][1]}"'
               if (pass2 and target and it.get("px")) else "")
    img = (f'<img src="{sub}/{html.escape(name)}" width="{width}" '
           f'height="{round(width * mv.CROP_PX[1] / mv.CROP_PX[0])}" loading="lazy"{natural} '
           f'alt="{html.escape(alt)}">')
    if target:
        ring = _ring_overlay(it) if pass2 else R._halo(it)
        img = f'<span class="ringwrap">{img}{ring}</span>'
    return f'<figure class="{cls}">{img}<figcaption>{cap}</figcaption></figure>'


def render_gallery(cards, crops, digest, pass2=None):
    """The #48 GT-check page, re-labelled for this question. The JS is the same shape:
    rater id, per-rater localStorage, Y / N / C keys, export to the per-rater file.
    Both passes get the image controls; `pass2` (from `pass2_meta`) marks the page and
    its exports as the re-rating of the pass-1 "Can't tell" cards, reads the target view
    from the native, unringed pass-2 crops and draws the ring as a toggleable overlay."""
    by_card = {}
    for it in crops:
        by_card.setdefault(it["ramp_uid"], []).append(it)
    out_cards = []
    for n, c in enumerate(cards, 1):
        uid = c["id"]
        tgt = [it for it in by_card[uid] if _role(it) == "target"]
        ctx = [it for it in by_card[uid] if _role(it) != "target"]
        radios = "".join(
            f'<label><input type="radio" name="v_{uid}" value="{k}"> {html.escape(lab)}</label>'
            for k, lab, _ in RUBRIC)
        ctx_html = "".join(_figure(it, uid, mv.CROP_PX[0]) for it in ctx) or \
            '<p class="muted">No context view.</p>'
        out_cards.append(
            f'<section class="card" data-uid="{uid}" aria-labelledby="h_{uid}">'
            f'<h2 id="h_{uid}">{n}. {uid}</h2>'
            f'<div class="row"><div class="rate">'
            f'{"".join(_figure(it, uid, 540, pass2=bool(pass2)) for it in tgt)}'
            f'{image_controls_html(uid, pass2=bool(pass2))}'
            f'<fieldset><legend>{html.escape(QUESTION)}</legend><div class="opts">{radios}</div>'
            f'<label class="note">Note (optional) <textarea rows="2" name="n_{uid}">'
            f'</textarea></label></fieldset></div>'
            f'<div class="context" role="group" aria-label="Context for {uid}, not rated">'
            f'<h3>Context: unmarked, not rated</h3><div class="strip">{ctx_html}</div></div>'
            f'</div></section>')
    ids = [c["id"] for c in cards]
    meta = {"question": QUESTION,
            "rubric": [{"key": k, "label": lab, "definition": d} for k, lab, d in RUBRIC],
            "rules": RULES, "items": ids, "manifest_digest": digest, "gallery": GALLERY_REL,
            "task": "RampNet #158 step 4, mined-label check: " + QUESTION,
            "pass2": pass2}
    return _PAGE(out_cards, meta)


def _PAGE(cards_html, meta):
    """Fill the #48 GT-check template: its CSS and script, with this task's text."""
    pass2 = meta.get("pass2")
    tmpl = R.render_gallery(
        [{"uid": "__X__", "class": "x"}],
        [{"ramp_uid": "__X__", "city": "x", "pano_id": "x", "x": 0.5, "y": 0.5,
          "is_source": True, "ring": False, "capture_date": ""}], meta["manifest_digest"])
    head, rest = tmpl.split('<section class="card"', 1)
    tail = rest[rest.index('<script id="meta"'):]
    title = "Mined label check" + (", pass 2" if pass2 else "")
    head = head.replace("<title>Residual GT check</title>", f"<title>{title}</title>")
    h1 = ("Mined labels (#158 step 4, pass 2): the &ldquo;Can't tell&rdquo; cards again, with "
          "image controls" if pass2 else
          "Mined labels (#158 step 4): is there a curb ramp at the ring?")
    head = re.sub(r"<h1>.*?</h1>", f"<h1>{h1}</h1>", head, count=1, flags=re.S)
    intro = (
        '<div class="intro"><p>Each card is one label that the miner would add to RampNet\'s '
        'training data: a spot in a panorama where other captures agree there is a curb '
        'ramp, but this panorama\'s detector did not fire. Look at the large view on the '
        f'left and answer one question: <strong>{html.escape(QUESTION)}</strong> The smaller '
        'view on the right is another capture of the same corner, shown only as context, '
        'without a ring.</p>'
        + ('<p><strong>Pass 2.</strong> These are the cards answered &ldquo;Can\'t tell&rdquo; '
           'in pass 1. The view to rate is cut again from the same panorama at its native '
           'resolution (up to three times the pixels of pass 1) and without a drawn ring: the '
           'ring is an overlay, <strong>R</strong> (or the button) hides and shows it, and '
           'the view opens at 1:1 pixels (<strong>Z</strong> fits it to the card). On a washed-out '
           'card start with <strong>Washout fix</strong>, one slider that raises the black '
           'point and adds local contrast and saturation together. The sliders after it are '
           'levels (black and white point), gamma, '
           'local contrast, brightness, contrast and saturation; they apply to that card only '
           'and the setting is saved with the answer. A pixel the camera recorded as white '
           'holds no detail at any setting. Rate each card afresh under the same rubric; '
           '&ldquo;Can\'t tell&rdquo; is still a valid answer. Use a new rater id, e.g. '
           '<code>jonf-p2</code>.</p>' if pass2 else
           "<p>The sliders under each ringed view adjust that card's levels, gamma, local "
           'contrast, brightness, contrast and saturation; the setting is saved with the '
           'answer.</p>')
        + '</div>')
    head = re.sub(r'<div class="intro">.*?</div>', lambda _m: intro, head, count=1, flags=re.S)
    head = head.replace("</style>", IMAGE_CONTROLS_CSS + "</style>", 1)
    rubric_html = "".join(f"<dt>{html.escape(lab)}</dt><dd>{html.escape(d)}</dd>"
                          for _, lab, d in RUBRIC)
    rules_html = "".join(f"<li>{html.escape(x)}</li>" for x in RULES)
    head = re.sub(r"<dl>.*?</dl>", lambda _m: f"<dl>{rubric_html}</dl>", head, count=1, flags=re.S)
    head = re.sub(r"<ul>.*?</ul>", lambda _m: f"<ul>{rules_html}</ul>", head, count=1, flags=re.S)
    head = head.replace(R.EXPORT_PREFIX, EXPORT_PREFIX)
    meta_js = json.dumps(meta).replace("</", "<" + chr(92) + "/")
    tail = re.sub(r'<script id="meta" type="application/json">.*?</script>',
                  lambda _m: f'<script id="meta" type="application/json">{meta_js}</script>',
                  tail, count=1, flags=re.S)
    tail = tail.replace("mv48_gtcheck_", "mlc158_")
    tail = re.sub(r'const out = \{task: .*?item_class: META\.item_class, ',
                  'const out = {task: META.task, question: META.question, rubric: META.rubric, '
                  'rules: META.rules, rater: rater, items: META.items, ...(META.pass2 || {}), ',
                  tail, count=1, flags=re.S)
    tail = re.sub(r'\s*supersedes: "[^"]*",', '', tail, count=1)
    tail = tail.replace(R.EXPORT_PREFIX, EXPORT_PREFIX)
    # Carry each card's image setting into the export.
    n = tail.count('note: (v.note || "").trim()}')
    if n != 1:
        raise RuntimeError(f"gallery template drifted: {n} export sites, expected 1")
    tail = tail.replace('note: (v.note || "").trim()}',
                        'note: (v.note || "").trim(), image: v.image || null}')
    tail = tail.replace("</script></body></html>", IMAGE_CONTROLS_JS + "</script></body></html>", 1)
    if pass2:
        tail = tail.replace("mlc158_", "mlc158_p2_")
    return head + "".join(cards_html) + "\n" + tail


def cmd_gallery(args):
    if args.init_rater is not None and not RATER_RE.match(args.init_rater):
        raise SystemExit(f"--init-rater {args.init_rater!r} is not a valid rater id")
    if args.pass2_from:
        return cmd_gallery_pass2(args)
    if not args.crops:
        raise SystemExit("gallery (pass 1) needs --crops")
    plan = json.load(open(ITEMS_PATH, encoding="utf-8"))
    cards, crops = plan["cards"], plan["items"]
    out = os.path.join(GALLERY_DIR, "crops")
    os.makedirs(out, exist_ok=True)
    sha, missing = {}, []
    for it in crops:
        name = mv.crop_name(it)
        src, dst = os.path.join(args.crops, name), os.path.join(out, name)
        if not os.path.exists(src):
            missing.append(name)
            continue
        if os.path.abspath(src) != os.path.abspath(dst):
            with open(src, "rb") as fi, open(dst, "wb") as fo:
                fo.write(fi.read())
        sha[name] = R.sha256_file(dst)
    if missing:
        raise SystemExit(f"{len(missing)} crops missing from {args.crops}, e.g. {missing[:3]}")
    ids = [c["id"] for c in cards]
    digest = R.manifest_digest(ids, sha)
    mv.write_json(MANIFEST_PATH, {"manifest_digest": digest, "n_crops": len(sha),
                                  "n_cards": len(cards), "crops_sha256": sha,
                                  "items": "analysis_out/mined_label_check_158/items.json",
                                  "items_sha256": R.sha256_file(ITEMS_PATH)})
    with open(os.path.join(GALLERY_DIR, "gallery.html"), "w", encoding="utf-8",
              newline="") as f:
        f.write(render_gallery(cards, crops, digest))
    if args.init_rater:
        path = verdicts_path(args.init_rater)
        if not os.path.exists(path):
            mv.write_json(path, empty_verdicts(ids, digest, args.init_rater))
    print(f"wrote {GALLERY_DIR}/gallery.html ({len(cards)} cards, {len(sha)} crops, "
          f"digest {digest})")


def cmd_plan_pass2(args):
    """The pass-2 cut plan: the pass-1 'Can't tell' cards' target views, unringed, to be
    cut at native resolution (`cut-crops` reads the plan's `cut` block). Bound to the
    pass-1 file by sha256 and to the pass-1 gallery by digest."""
    ref = committed_reference()
    pass1 = load_verdicts(args.pass2_from, ref)
    ids = pass2_items(pass1, ref)
    if not ids:
        raise SystemExit(f"{args.pass2_from}: no {PASS2_SUBSET!r} answers, nothing to re-rate")
    plan = json.load(open(ITEMS_PATH, encoding="utf-8"))
    keep = set(ids)
    items = [dict(it, ring=False) for it in plan["items"]
             if it["ramp_uid"] in keep and _role(it) == "target"]
    if len(items) != len(ids):
        raise SystemExit("items.json does not hold one target view per pass-2 card")
    meta = pass2_meta(args.pass2_from, ref["manifest_digest"])
    mv.write_json(ITEMS2_PATH, {**meta, "cut": {"native": True, "quality": PASS2_QUALITY},
                                "cards": ids, "items": items, "n": len(items)})
    print(f"{len(items)} pass-2 target views (native, unringed) -> {ITEMS2_PATH}")


def cmd_gallery_pass2(args):
    """The pass-1 'Can't tell' cards only: the target view from the native, unringed
    pass-2 cuts (--crops, cut from items_pass2.json), the context view from the pass-1
    crops; its own manifest and digest; the ring as a page overlay; image controls."""
    ref = committed_reference()
    pass1 = load_verdicts(args.pass2_from, ref)
    ids = pass2_items(pass1, ref)
    if not ids:
        raise SystemExit(f"{args.pass2_from}: no {PASS2_SUBSET!r} answers, nothing to re-rate")
    if not args.crops:
        raise SystemExit("gallery --pass2-from needs --crops DIR: the native, unringed cuts of "
                         "items_pass2.json (cut-crops on makelab2)")
    plan2 = json.load(open(ITEMS2_PATH, encoding="utf-8"))
    if (plan2.get("from_sha256") != R.sha256_file(args.pass2_from) or plan2.get("cards") != ids
            or plan2.get("pass1_manifest_digest") != ref["manifest_digest"]):
        raise SystemExit(f"{ITEMS2_PATH} was not planned from {args.pass2_from} on this "
                         f"gallery; run plan-pass2 again")
    from PIL import Image
    out = os.path.join(GALLERY_DIR, CROPS2_DIR)
    os.makedirs(out, exist_ok=True)
    sha, px, missing = {}, {}, []
    for it in plan2["items"]:
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
        it["px"] = px[name]
    if missing:
        raise SystemExit(f"{len(missing)} crops missing from {args.crops}, e.g. {missing[:3]}")
    digest2 = R.manifest_digest(ids, sha)
    mv.write_json(MANIFEST2_PATH, {
        "manifest_digest": digest2, "pass1_manifest_digest": ref["manifest_digest"],
        "n_crops": len(sha), "n_cards": len(ids), "crops_sha256": sha, "crop_px": px,
        "cut": plan2["cut"], "items": "analysis_out/mined_label_check_158/items_pass2.json",
        "items_sha256": R.sha256_file(ITEMS2_PATH)})
    plan = json.load(open(ITEMS_PATH, encoding="utf-8"))
    keep = set(ids)
    cards = [c for c in plan["cards"] if c["id"] in keep]
    ctx = [it for it in plan["items"] if it["ramp_uid"] in keep and _role(it) != "target"]
    missing = [mv.crop_name(it) for it in ctx
               if not os.path.exists(os.path.join(GALLERY_DIR, "crops", mv.crop_name(it)))]
    if missing:
        raise SystemExit(f"{len(missing)} context crops missing from {GALLERY_DIR}/crops, e.g. "
                         f"{missing[:3]}; build the pass-1 gallery first")
    meta = pass2_meta(args.pass2_from, ref["manifest_digest"])
    with open(os.path.join(GALLERY_DIR, "gallery_pass2.html"), "w", encoding="utf-8",
              newline="") as f:
        f.write(render_gallery(cards, plan2["items"] + ctx, digest2, pass2=meta))
    if args.init_rater:
        if args.init_rater == pass1["rater"]:
            raise SystemExit("pass 2 needs a new rater id (its file would overwrite pass 1)")
        path = verdicts_path(args.init_rater)
        if not os.path.exists(path):
            mv.write_json(path, empty_verdicts(ids, digest2, args.init_rater, pass2=meta))
    print(f"wrote {GALLERY_DIR}/gallery_pass2.html ({len(cards)} cards re-rated from "
          f"{pass1['rater']}'s {len(pass1['verdicts'])} answers; {len(sha)} native crops, "
          f"digest {digest2}, over pass-1 gallery {ref['manifest_digest']})")


# --------------------------------------------------------------------------- #
# rates
# --------------------------------------------------------------------------- #
def committed_reference():
    plan = json.load(open(ITEMS_PATH, encoding="utf-8"))
    man = json.load(open(MANIFEST_PATH, encoding="utf-8"))
    ids = [c["id"] for c in plan["cards"]]
    digest = R.manifest_digest(ids, man["crops_sha256"])
    if digest != man["manifest_digest"]:
        raise ValueError("manifest.json's digest does not re-derive from its crop sha256s")
    return {"manifest_digest": digest, "items": ids, "cards": {c["id"]: c for c in plan["cards"]}}


def committed_reference2():
    """The committed pass-2 gallery: its digest re-derived from items_pass2.json's card
    list and manifest_pass2.json's crop sha256s, and the pass-1 digest it sits over."""
    plan = json.load(open(ITEMS2_PATH, encoding="utf-8"))
    man = json.load(open(MANIFEST2_PATH, encoding="utf-8"))
    ids = plan["cards"]
    digest = R.manifest_digest(ids, man["crops_sha256"])
    if digest != man["manifest_digest"]:
        raise ValueError("manifest_pass2.json's digest does not re-derive from its crop sha256s")
    if man["pass1_manifest_digest"] != plan["pass1_manifest_digest"]:
        raise ValueError("manifest_pass2.json and items_pass2.json name different pass-1 galleries")
    return {"manifest_digest": digest, "pass1_manifest_digest": man["pass1_manifest_digest"],
            "items": ids, "from_sha256": plan["from_sha256"]}


def load_verdicts(path, ref, pass1=None, ref2=None):
    """A verdict file, checked against the committed gallery. A pass-2 file (`pass: 2`)
    is checked against the committed pass-2 gallery (`ref2`, default
    `committed_reference2()`: its own digest, over this pass-1 digest) and against the
    pass-1 file it re-rates, given as `(path, dict)`: same sha256 as recorded, and exactly
    that file's 'Can't tell' cards as its items."""
    d = json.load(open(path, encoding="utf-8"))
    rater = d.get("rater")
    if not isinstance(rater, str) or not RATER_RE.match(rater):
        raise ValueError(f"{path}: rater id {rater!r} missing or invalid")
    if os.path.basename(path) != EXPORT_PREFIX + rater + EXPORT_SUFFIX:
        raise ValueError(f"{path}: file name does not match rater {rater!r}")
    if d.get("pass") == 2:
        if pass1 is None:
            raise ValueError(f"{path}: a pass-2 file is read with the pass-1 file it re-rates "
                             f"(rates <pass1> --pass2 <pass2>)")
        ref2 = ref2 or committed_reference2()
        if ref2["pass1_manifest_digest"] != ref["manifest_digest"]:
            raise ValueError(f"the committed pass-2 gallery sits over pass-1 gallery "
                             f"{ref2['pass1_manifest_digest']}, not {ref['manifest_digest']}")
        if d.get("manifest_digest") != ref2["manifest_digest"]:
            raise ValueError(f"{path}: made on pass-2 gallery {d.get('manifest_digest')}, not "
                             f"{ref2['manifest_digest']}")
        if d.get("pass1_manifest_digest") != ref["manifest_digest"]:
            raise ValueError(f"{path}: pass1_manifest_digest is not {ref['manifest_digest']}")
        p1_path, p1 = pass1
        if d.get("from_sha256") != R.sha256_file(p1_path):
            raise ValueError(f"{path}: from_sha256 is not {p1_path}'s sha256")
        if ref2.get("from_sha256") not in (None, d.get("from_sha256")):
            raise ValueError(f"{path}: the committed pass-2 gallery was planned from another "
                             f"pass-1 file")
        if d.get("subset") != PASS2_SUBSET or d.get("items") != pass2_items(p1, ref) \
                or d.get("items") != ref2["items"]:
            raise ValueError(f"{path}: items are not {p1_path}'s {PASS2_SUBSET!r} cards")
        if rater == p1.get("rater"):
            raise ValueError(f"{path}: pass 2 uses the same rater id as pass 1")
    else:
        if d.get("manifest_digest") != ref["manifest_digest"]:
            raise ValueError(f"{path}: made on gallery {d.get('manifest_digest')}, not "
                             f"{ref['manifest_digest']}")
        if d.get("items") != ref["items"]:
            raise ValueError(f"{path}: item list differs from the committed items.json")
    rubric = [{"key": k, "label": lab, "definition": x} for k, lab, x in RUBRIC]
    if (d.get("question"), d.get("rubric"), d.get("rules")) != (QUESTION, rubric, RULES):
        raise ValueError(f"{path}: question, rubric or rules differ from this module's")
    for uid, v in d.get("verdicts", {}).items():
        if uid not in ref["cards"]:
            raise ValueError(f"{path}: {uid} is not an item")
        if v.get("answer") not in ANSWERS + (None,):
            raise ValueError(f"{path}: {uid} answer {v.get('answer')!r} not in {ANSWERS}")
    return d


def rule_reading(p, lo, hi):
    def band(x):
        return "build" if x >= RULE_BUILD else "visibility" if x >= RULE_VISIBILITY else "drop"
    if p is None:
        return "no decided items"
    b, blo, bhi = band(p), band(lo), band(hi)
    return b + ("" if blo == bhi else f"; the 95% CI spans {blo}..{bhi}, not decisive")


def precision(verdicts, ids):
    c = Counter((verdicts.get(u) or {}).get("answer") or "unanswered" for u in ids)
    n = c["yes"] + c["no"]
    lo, hi = mv.wilson(c["yes"], n)
    p = c["yes"] / n if n else None
    return {"n_items": len(ids), "yes": c["yes"], "no": c["no"], "cant_tell": c["cant_tell"],
            "unanswered": c["unanswered"], "n_decided": n, "precision": p,
            "wilson_95": [lo, hi], "rule": rule_reading(p, lo, hi)}


def rates(d, ref):
    v, cards = d["verdicts"], ref["cards"]
    sample = [u for u in ref["items"] if not cards[u]["instrument"]]
    instr = [u for u in ref["items"] if cards[u]["instrument"]]
    agree = [u for u in instr if (v.get(u) or {}).get("answer") == cards[u]["known_answer"]]
    decided = [u for u in instr if (v.get(u) or {}).get("answer") in ("yes", "no")]
    return {"rater": d["rater"], "manifest_digest": d["manifest_digest"],
            "pooled": precision(v, sample),
            "by_band": {b: precision(v, [u for u in sample if cards[u]["band"] == b])
                        for b in BANDS},
            "instrument": {"n": len(instr), "decided": len(decided),
                           "agree_with_earlier_verdicts": len(agree),
                           "disagreements": [{"id": u, "now": (v.get(u) or {}).get("answer"),
                                              "earlier": cards[u]["known_answer"]}
                                             for u in instr if u not in agree]}}


def combined(pass1, pass2, ref):
    """Pass 1 with the pass-2 answers written over its 'Can't tell' cards. A pass-2 card
    left unanswered keeps pass 1's 'Can't tell'. The read is one rater's two looks at the
    same spots, the second on native-resolution unringed cuts with image controls, not two
    raters."""
    v = dict(pass1["verdicts"])
    resolved = Counter()
    for u in pass2["items"]:
        a = (pass2["verdicts"].get(u) or {}).get("answer")
        resolved[a or "unanswered"] += 1
        if a:
            v[u] = pass2["verdicts"][u]
    d = {"rater": f"{pass1['rater']} then {pass2['rater']}",
         "manifest_digest": pass1["manifest_digest"], "verdicts": v}
    out = rates(d, ref)
    out["pass2"] = {"rater": pass2["rater"], "n_items": len(pass2["items"]),
                    "manifest_digest": pass2["manifest_digest"],
                    "resolved": dict(resolved),
                    "subset_precision": precision(pass2["verdicts"], pass2["items"])}
    return out


def cmd_rates(args):
    if len(args.files) > 2 or (args.pass2 and len(args.files) != 1):
        raise SystemExit("rates takes one or two verdict files, or one file with --pass2")
    ref = committed_reference()
    files = [load_verdicts(p, ref) for p in args.files]
    out = {"rates": [rates(d, ref) for d in files]}
    if args.pass2:
        p2 = load_verdicts(args.pass2, ref, pass1=(args.files[0], files[0]))
        out["combined"] = combined(files[0], p2, ref)
    if len(files) == 2:
        for d in files:        # R.agreement reads item_class only through the items list
            d.setdefault("item_class", {})
        out["agreement"] = R.agreement(*files)
    print(json.dumps(mv.rnd(out), indent=1, sort_keys=True))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--labeler-root", required=True)
    p2 = sub.add_parser("plan-pass2", help="the pass-2 cut plan (items_pass2.json)")
    p2.add_argument("--pass2-from", required=True, metavar="VERDICTS",
                    help="the pass-1 file whose 'Can't tell' cards pass 2 re-rates")
    g = sub.add_parser("gallery")
    g.add_argument("--crops", default=None,
                   help="cut crops to copy in (pass 1: items.json's; pass 2: items_pass2.json's)")
    g.add_argument("--init-rater", default=None)
    g.add_argument("--pass2-from", default=None, metavar="VERDICTS",
                   help="write gallery_pass2.html: that file's 'Can't tell' cards only")
    r = sub.add_parser("rates")
    r.add_argument("files", nargs="+")
    r.add_argument("--pass2", default=None, metavar="VERDICTS",
                   help="a pass-2 file re-rating the first file's 'Can't tell' cards")
    args = ap.parse_args(argv)
    {"plan": cmd_plan, "plan-pass2": cmd_plan_pass2, "gallery": cmd_gallery,
     "rates": cmd_rates}[args.cmd](args)


if __name__ == "__main__":
    main()
