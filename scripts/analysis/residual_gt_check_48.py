"""Residual-miss GT check for #48: is there a curb ramp at the GT point?

Replaces the first residual gallery (``multiview_evidence_48.py gallery``, rubric in
``residual_taxonomy__jonf.json``), which was superseded before any verdicts for two reasons:

1. In the other views its ring was the GT world point projected with a flat-ground 2.6 m
   camera. GT placement error is p50 1.9 m / p90 4.4 m, Mapillary rigs sit lower than
   2.6 m and pose error is unmeasured, so at 12-18 m that ring often lands on plain curb
   beside the ramp (richmond:3). A rater judges what is under the ring, so projection
   error would have been read as GT error.
2. Its one question ("why was this ramp missed?", seven options) mixed a fact (is there a
   ramp at the GT point?) with a diagnosis, and for the 58 merging cases the model did
   detect the ramp, so options like occluded or far do not describe those failures.

This pass asks one question per card, answered from the source view only: is there a curb
ramp at the ring? The ring in the source view is the reviewer's own click in that pano (the
GT point before any raycast). Other views are shown without a ring, as context.

    python scripts/analysis/residual_gt_check_48.py plan
    python scripts/analysis/multiview_evidence_48.py cut-crops \\
        analysis_out/multiview_48/residual_gt_check_plan.json \\
        --archive-root /projects/makeabilitylab/sidewalk-auto-labeler/runs --out crops  # makelab2
    python scripts/analysis/residual_gt_check_48.py gallery --crops crops
    python scripts/analysis/residual_gt_check_48.py rates analysis_out/multiview_48/residual_gt_check__jonf.json [SECOND_RATER.json]

``plan`` and ``gallery`` read only committed files (plus the crops); ``rates`` reads only
the verdict files and ``residual_misses.json``.
"""
import argparse
import csv
import hashlib
import html
import json
import math
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import multiview_evidence_48 as mv  # noqa: E402
from rampnet.detection_eval import build_ground_truth  # noqa: E402

OUT = mv.OUT
RESIDUAL_PATH = os.path.join(OUT, "residual_misses.json")
CAPTURES_PATH = os.path.join(OUT, "captures_R25.csv")
PLAN_PATH = os.path.join(OUT, "residual_gt_check_plan.json")
VERDICTS_PATH = os.path.join(OUT, "residual_gt_check__jonf.json")
GALLERY_DIR = os.path.join(mv.BENCHMARK, "multiview_residual_gt_check_48")
GALLERY_REL = "benchmark/multiview_residual_gt_check_48/gallery.html"
EXPORT_NAME = "residual_gt_check__jonf.json"

QUESTION = "In the source view, is there a curb ramp at the ring?"

#: The rubric. It travels in every verdict file, so a verdict is never read without it.
RUBRIC = [
    ("yes", "Yes",
     "A curb ramp is at the ring or touching it. The ramp may be partly hidden or faint, "
     "as long as you can see it is a ramp."),
    ("no", "No",
     "There is no curb ramp at that spot: the ring is on plain curb, sidewalk, street or "
     "something else, and the nearest ramp (if any) is more than roughly one ramp width "
     "away. This means the GT point is wrong or misplaced."),
    ("cant_tell", "Can't tell",
     "The source view does not let you decide (too dark, blocked, too far, too blurry). "
     "Excluded from every rate."),
]
ANSWERS = tuple(k for k, _, _ in RUBRIC)

#: The classes where the model did detect the ramp but fusion did not deliver a site at
#: it (the 58 cases handed to the labeler's clustering work, docs/multiview_48.md 9).
MERGING_CLASSES = ("association_placement", "self_detected_site_displaced")

RULES = [
    "Answer from the source view only. The other views are unmarked context and are not rated.",
    "Judge the spot under the ring, not whether a ramp exists somewhere in the crop.",
    "Add a note for anything worth recording, e.g. 'ramp 1 m left of ring' or 'ring on a "
    "driveway'.",
]


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def load_source_clicks(ramps, benchmark_root=mv.BENCHMARK):
    """{uid: (x_norm, y_norm)}: each ramp's GT point in its source pano, exactly as the
    reviewer placed it (build_ground_truth over the committed records and verdicts).
    Every residual ramp has one GT point and one source pano; anything else is refused,
    because then "the reviewer's click" would not be one point."""
    recs, verd, out = {}, {}, {}
    for r in ramps:
        if len(r["gt_refs"]) != 1:
            raise SystemExit(f"{r['uid']}: {len(r['gt_refs'])} GT points; expected 1")
        city = r["city"]
        if city not in recs:
            recs[city] = mv.read_bundle_records(city, benchmark_root)
            with open(os.path.join(benchmark_root, city, "verdicts.json"), encoding="utf-8") as f:
                verd[city] = json.load(f)["panos"]
        pid, gi = r["gt_refs"][0]
        e = verd[city][pid]
        gt = build_ground_truth(recs[city][pid]["detections"], e["dets"],
                                e.get("missed", ()), e.get("no_missed"))
        out[r["uid"]] = tuple(gt.gt_points[gi])
    return out


def angular_gap_deg(x0, y0, x1, y1):
    """Distance in degrees on the equirect grid between two normalized points, with the
    x difference wrapped at the seam."""
    dx = ((x1 - x0) + 0.5) % 1.0 - 0.5
    return math.hypot(dx * 360.0, (y1 - y0) * 180.0)


def gt_check_plan(ramps, captures_csv, clicks):
    """mv.crop_plan's crops, with the source view moved onto the reviewer's click and
    ringed, and every other view unringed. Each source item keeps the projected point it
    replaced (``x_proj``/``y_proj``) and the gap to it, so the switch is auditable."""
    plan = mv.crop_plan(ramps, captures_csv)
    for it in plan:
        if it["is_source"]:
            cx, cy = clicks[it["ramp_uid"]]
            it["x_proj"], it["y_proj"] = it["x"], it["y"]
            it["click_vs_projection_deg"] = angular_gap_deg(cx, cy, it["x"], it["y"])
            it["x"], it["y"] = cx, cy
            it["ring"] = True
        else:
            it["ring"] = False
    missing = {r["uid"] for r in ramps} - {it["ramp_uid"] for it in plan if it["is_source"]}
    if missing:
        raise SystemExit(f"no source crop for {sorted(missing)}")
    return plan


def cmd_plan(args):
    res = json.load(open(RESIDUAL_PATH, encoding="utf-8"))
    clicks = load_source_clicks(res["ramps"])
    plan = gt_check_plan(res["ramps"], CAPTURES_PATH, clicks)
    gaps = sorted(it["click_vs_projection_deg"] for it in plan if it["is_source"])
    mv.write_json(PLAN_PATH, {
        "n": len(plan), "items": plan,
        "source_ring": "reviewer's click in the source pano (build_ground_truth)",
        "click_vs_projection_deg": {"max": gaps[-1], "median": gaps[len(gaps) // 2]}})
    print(f"{len(plan)} crops ({sum(it['ring'] for it in plan)} ringed) -> {PLAN_PATH}; "
          f"click vs projected source point: median {gaps[len(gaps) // 2]:.4f} deg, "
          f"max {gaps[-1]:.4f} deg")


# --------------------------------------------------------------------------- #
# gallery
# --------------------------------------------------------------------------- #
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_digest(items, crop_sha):
    """16 hex chars over the item list and every crop's sha256, so a verdict file names
    exactly the images it was made on."""
    lines = [f"item {u}" for u in items] + [f"{n} {crop_sha[n]}" for n in sorted(crop_sha)]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def empty_verdicts(items, classes, digest):
    return {
        "task": "RampNet #48 residual misses, GT check: " + QUESTION,
        "question": QUESTION,
        "rubric": [{"key": k, "label": lab, "definition": d} for k, lab, d in RUBRIC],
        "rules": RULES,
        "rater": "jonf", "items": items, "item_class": classes,
        "manifest_digest": digest, "n_items": len(items), "n_answered": 0,
        "gallery": GALLERY_REL,
        "supersedes": "analysis_out/multiview_48/residual_taxonomy__jonf.json (no verdicts "
                      "were made on it)",
        "verdicts": {}}


def _figure(it, uid, source_width):
    name = mv.crop_name(it)
    date = html.escape(it["capture_date"] or "date unknown")
    if it["is_source"]:
        alt = f"Source view of {uid}, ring on the GT point"
        cap = f"<strong>Source view</strong> (rate this) &middot; {date}"
        cls, width = "src", source_width
    else:
        alt = f"Other view of {uid}, {it['dist_m']:.1f} m from the GT point, unmarked"
        cap = f"Other view, camera {it['dist_m']:.1f} m away &middot; {date} &middot; unmarked"
        cls, width = "ctx", mv.CROP_PX[0]
    return (f'<figure class="{cls}"><img src="crops/{html.escape(name)}" width="{width}" '
            f'height="{round(width * mv.CROP_PX[1] / mv.CROP_PX[0])}" loading="lazy" '
            f'alt="{html.escape(alt)}"><figcaption>{cap}</figcaption></figure>')


def render_gallery(ramps, plan, digest):
    by_ramp = defaultdict(list)
    for it in plan:
        by_ramp[it["ramp_uid"]].append(it)
    cards = []
    for n, r in enumerate(ramps, 1):
        uid = r["uid"]
        src = [it for it in by_ramp[uid] if it["is_source"]]
        ctx = [it for it in by_ramp[uid] if not it["is_source"]]
        key = uid.replace(":", "_")
        radios = "".join(
            f'<label><input type="radio" name="v_{key}" value="{k}"> {html.escape(lab)}</label>'
            for k, lab, _ in RUBRIC)
        ctx_html = ("".join(_figure(it, uid, 0) for it in ctx) if ctx
                    else '<p class="muted">No other capture within 18 m.</p>')
        cards.append(
            f'<section class="card" data-uid="{html.escape(uid)}" aria-labelledby="h_{key}">'
            f'<h2 id="h_{key}">{n}. {html.escape(uid)} <span class="meta">'
            f'{html.escape(r["class"].replace("_", " "))}</span></h2>'
            f'<div class="row"><div class="rate">{"".join(_figure(it, uid, 540) for it in src)}'
            f'<fieldset><legend>{html.escape(QUESTION)}</legend><div class="opts">{radios}</div>'
            f'<label class="note">Note (optional) <textarea rows="2" name="n_{key}">'
            f'</textarea></label></fieldset></div>'
            f'<div class="context" role="group" aria-label="Other views of {html.escape(uid)}, '
            f'context only"><h3>Other views: unmarked context, not rated</h3>'
            f'<div class="strip">{ctx_html}</div></div></div></section>')
    rubric_html = "".join(f"<dt>{html.escape(lab)}</dt><dd>{html.escape(d)}</dd>"
                          for _, lab, d in RUBRIC)
    rules_html = "".join(f"<li>{html.escape(x)}</li>" for x in RULES)
    meta = json.dumps({"question": QUESTION,
                       "rubric": [{"key": k, "label": lab, "definition": d}
                                  for k, lab, d in RUBRIC],
                       "rules": RULES, "items": [r["uid"] for r in ramps],
                       "item_class": {r["uid"]: r["class"] for r in ramps},
                       "manifest_digest": digest, "gallery": GALLERY_REL})
    # A JSON island is raw text to the parser: only "</" needs escaping, as "<" + "\" + "/".
    meta_js = meta.replace("</", "<" + chr(92) + "/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Residual GT check</title>
<style>
:root {{ --bg:#ffffff; --fg:#1f2328; --muted:#57606a; --line:#d0d7de; --focus:#0969da; --panel:#f6f8fa; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d1117; --fg:#e6edf3; --muted:#8d96a0; --line:#30363d; --focus:#4493f8; --panel:#161b22; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 16px 64px; max-width:1500px; }}
h1 {{ font-size:22px; }} h2 {{ font-size:16px; margin:0 0 8px; }}
h3 {{ font-size:13px; font-weight:600; color:var(--muted); margin:0 0 4px; }}
.meta, .muted {{ color:var(--muted); font-weight:normal; font-size:13px; }}
.intro {{ max-width:75ch; }}
.card {{ border-top:1px solid var(--line); padding:14px 0; }}
.card.done h2::after {{ content:" \\2713"; color:var(--muted); }}
.row {{ display:flex; flex-wrap:wrap; gap:16px; align-items:flex-start; }}
.rate {{ flex:0 1 540px; }}
.context {{ flex:1 1 360px; min-width:0; }}
.strip {{ display:flex; gap:8px; overflow-x:auto; }}
figure {{ margin:0; }} figcaption {{ color:var(--muted); font-size:12px; }}
figure.ctx img {{ opacity:.95; }}
img {{ max-width:100%; height:auto; display:block; }}
fieldset {{ border:1px solid var(--line); border-radius:6px; background:var(--panel); margin:8px 0 0; padding:8px 10px; }}
legend {{ font-weight:600; padding:0 4px; }}
.opts {{ display:flex; flex-wrap:wrap; gap:4px 18px; }}
.opts label {{ cursor:pointer; padding:2px 0; }}
.note {{ display:block; margin-top:6px; font-size:13px; color:var(--muted); }}
textarea {{ display:block; width:100%; box-sizing:border-box; font:inherit; color:var(--fg); background:var(--bg); border:1px solid var(--line); border-radius:4px; }}
:focus-visible {{ outline:3px solid var(--focus); outline-offset:2px; }}
button {{ font:inherit; padding:6px 12px; }}
dt {{ font-weight:600; }} dd {{ margin:0 0 6px 16px; }}
.bar {{ position:sticky; top:0; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--line); z-index:1; display:flex; flex-wrap:wrap; gap:8px 12px; align-items:center; }}
</style></head><body>
<h1>Residual misses (#48): GT check</h1>
<div class="intro">
<p>Each card is one ground-truth curb ramp that multi-view fusion did not deliver. Look at the
large <strong>source view</strong> on the left: the green ring is where the reviewer clicked
in that panorama. Answer one question: <strong>{html.escape(QUESTION)}</strong> The smaller
views on the right are other captures of the same corner, shown only as context, without a
ring (a projected ring would often miss by metres), and nothing is rated from them.</p>
</div>
<dl>{rubric_html}</dl>
<ul>{rules_html}</ul>
<p class="muted">Keyboard: Tab moves between cards' controls; arrow keys change the answer
within a card. With focus anywhere in a card (not in a note), press Y, N or C to answer.
Answers are saved in this browser as you go; Export writes them to a file.</p>
<div class="bar"><button type="button" id="export">Export verdicts JSON</button>
<button type="button" id="next">Next unanswered</button>
<span id="count" aria-live="polite"></span></div>
{"".join(cards)}
<script id="meta" type="application/json">{meta_js}</script>
<script>
const META = JSON.parse(document.getElementById('meta').textContent);
const KEY = "mv48_gtcheck_" + META.manifest_digest;
const saved = (() => {{ try {{ return JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ return {{}}; }} }})();
function persist() {{ try {{ localStorage.setItem(KEY, JSON.stringify(saved)); }} catch (e) {{}} }}
function answered() {{ return META.items.filter(u => saved[u] && saved[u].answer).length; }}
function update() {{
  document.getElementById('count').textContent = answered() + " of " + META.items.length + " answered";
  document.querySelectorAll('.card').forEach(c => c.classList.toggle('done', !!(saved[c.dataset.uid] || {{}}).answer));
}}
document.querySelectorAll('.card').forEach(card => {{
  const uid = card.dataset.uid;
  const cur = saved[uid] || {{}};
  card.querySelectorAll('input[type=radio]').forEach(inp => {{
    if (cur.answer === inp.value) inp.checked = true;
    inp.addEventListener('change', () => {{
      saved[uid] = Object.assign(saved[uid] || {{}}, {{answer: inp.value}});
      persist(); update();
    }});
  }});
  const ta = card.querySelector('textarea');
  ta.value = cur.note || "";
  ta.addEventListener('input', () => {{
    saved[uid] = Object.assign(saved[uid] || {{}}, {{note: ta.value}});
    persist();
  }});
  card.addEventListener('keydown', ev => {{
    if (ev.target.tagName === 'TEXTAREA' || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const k = {{y: 'yes', n: 'no', c: 'cant_tell'}}[ev.key.toLowerCase()];
    if (!k) return;
    const inp = card.querySelector('input[value="' + k + '"]');
    inp.checked = true; inp.focus();
    inp.dispatchEvent(new Event('change'));
    ev.preventDefault();
  }});
}});
update();
document.getElementById('next').addEventListener('click', () => {{
  const card = [...document.querySelectorAll('.card')].find(c => !(saved[c.dataset.uid] || {{}}).answer);
  if (card) {{ card.scrollIntoView({{block: 'start'}}); card.querySelector('input[type=radio]').focus({{preventScroll: true}}); }}
}});
document.getElementById('export').addEventListener('click', () => {{
  const verdicts = {{}};
  META.items.forEach(u => {{
    const v = saved[u];
    if (!v || (!v.answer && !(v.note || "").trim())) return;
    verdicts[u] = {{answer: v.answer || null, note: (v.note || "").trim()}};
  }});
  const out = {{task: "RampNet #48 residual misses, GT check: " + META.question,
    question: META.question, rubric: META.rubric, rules: META.rules, rater: "jonf",
    items: META.items, item_class: META.item_class, manifest_digest: META.manifest_digest,
    n_items: META.items.length, n_answered: answered(), gallery: META.gallery,
    supersedes: "analysis_out/multiview_48/residual_taxonomy__jonf.json (no verdicts were made on it)",
    exported_at: new Date().toISOString(), verdicts: verdicts}};
  const blob = new Blob([JSON.stringify(out, null, 1) + "\\n"], {{type: "application/json"}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = "{EXPORT_NAME}";
  a.click();
}});
</script></body></html>
"""


def cmd_gallery(args):
    """Copy the crops into GALLERY_DIR, write their sha256 manifest, the page, and (if
    absent) the empty per-rater verdict file."""
    res = json.load(open(RESIDUAL_PATH, encoding="utf-8"))
    plan = json.load(open(PLAN_PATH, encoding="utf-8"))["items"]
    ramps = res["ramps"]
    crops_out = os.path.join(GALLERY_DIR, "crops")
    os.makedirs(crops_out, exist_ok=True)
    crop_sha, missing = {}, []
    for it in plan:
        name = mv.crop_name(it)
        src = os.path.join(args.crops, name)
        dst = os.path.join(crops_out, name)
        if not os.path.exists(src):
            missing.append(name)
            continue
        if os.path.abspath(src) != os.path.abspath(dst):
            with open(src, "rb") as fi, open(dst, "wb") as fo:
                fo.write(fi.read())
        crop_sha[name] = sha256_file(dst)
    if missing:
        raise SystemExit(f"{len(missing)} planned crops missing from {args.crops}, "
                         f"e.g. {missing[:3]}")
    items = [r["uid"] for r in ramps]
    classes = {r["uid"]: r["class"] for r in ramps}
    digest = manifest_digest(items, crop_sha)
    mv.write_json(os.path.join(GALLERY_DIR, "manifest.json"), {
        "manifest_digest": digest, "n_crops": len(crop_sha),
        "n_ringed": sum(1 for it in plan if it["ring"]), "crops_sha256": crop_sha,
        "plan": "analysis_out/multiview_48/residual_gt_check_plan.json"})
    with open(os.path.join(GALLERY_DIR, "gallery.html"), "w", encoding="utf-8",
              newline="") as f:
        f.write(render_gallery(ramps, plan, digest))
    if not os.path.exists(VERDICTS_PATH):
        mv.write_json(VERDICTS_PATH, empty_verdicts(items, classes, digest))
    print(f"wrote {GALLERY_DIR}/gallery.html ({len(ramps)} cards, {len(crop_sha)} crops, "
          f"digest {digest})")


# --------------------------------------------------------------------------- #
# rates and agreement
# --------------------------------------------------------------------------- #
def load_verdicts(path):
    """A rater file, checked against the rubric: every answer is a rubric key or null."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    for uid, v in d.get("verdicts", {}).items():
        if v.get("answer") not in ANSWERS + (None,):
            raise ValueError(f"{path}: {uid} has answer {v.get('answer')!r}, not in {ANSWERS}")
        if uid not in d["items"]:
            raise ValueError(f"{path}: {uid} is not in the item list")
    return d


def gt_error_rate(verdicts, uids):
    """GT-error rate over ``uids``: No / (Yes + No). Can't tell and unanswered items are
    excluded and counted. Returns counts, the rate and its 95% Wilson interval."""
    c = Counter((verdicts.get(u) or {}).get("answer") or "unanswered" for u in uids)
    n = c["yes"] + c["no"]
    lo, hi = mv.wilson(c["no"], n)
    return {"n_items": len(uids), "yes": c["yes"], "no": c["no"],
            "cant_tell": c["cant_tell"], "unanswered": c["unanswered"], "n_decided": n,
            "gt_error_rate": (c["no"] / n) if n else None, "wilson_95": [lo, hi]}


def rates(d, classes=None):
    """Overall, the 58 merging cases, and per class, from one rater file."""
    classes = classes or d["item_class"]
    items = d["items"]
    v = d["verdicts"]
    by_class = defaultdict(list)
    for u in items:
        by_class[classes[u]].append(u)
    return {"rater": d.get("rater"), "manifest_digest": d.get("manifest_digest"),
            "overall": gt_error_rate(v, items),
            "merging_cases": gt_error_rate(v, [u for u in items
                                               if classes[u] in MERGING_CLASSES]),
            "by_class": {k: gt_error_rate(v, us) for k, us in sorted(by_class.items())}}


def cohen_kappa(pairs, cats):
    """Cohen's kappa for a list of (a, b) labels; None when chance agreement is 1."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in cats) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def agreement(d1, d2):
    """Pairwise agreement between two rater files made on the same gallery: over items
    both answered (all three options), and over items both answered Yes or No."""
    if d1.get("manifest_digest") != d2.get("manifest_digest"):
        raise ValueError("the two files were made on different galleries (manifest digests "
                         f"{d1.get('manifest_digest')} vs {d2.get('manifest_digest')})")
    a1 = {u: v.get("answer") for u, v in d1["verdicts"].items() if v.get("answer")}
    a2 = {u: v.get("answer") for u, v in d2["verdicts"].items() if v.get("answer")}
    both = [u for u in d1["items"] if u in a1 and u in a2]
    pairs = [(a1[u], a2[u]) for u in both]
    yn = [(a, b) for a, b in pairs if a != "cant_tell" and b != "cant_tell"]
    return {"raters": [d1.get("rater"), d2.get("rater")], "n_both": len(pairs),
            "percent_agreement": (sum(a == b for a, b in pairs) / len(pairs)) if pairs else None,
            "kappa": cohen_kappa(pairs, ANSWERS),
            "n_both_yes_no": len(yn),
            "percent_agreement_yes_no": (sum(a == b for a, b in yn) / len(yn)) if yn else None,
            "kappa_yes_no": cohen_kappa(yn, ("yes", "no")),
            "disagreements": [{"uid": u, d1.get("rater"): a1[u], d2.get("rater"): a2[u]}
                              for u in both if a1[u] != a2[u]]}


def cmd_rates(args):
    files = [load_verdicts(p) for p in args.files]
    out = {"rates": [rates(d) for d in files]}
    if len(files) == 2:
        out["agreement"] = agreement(*files)
    print(json.dumps(mv.rnd(out), indent=1, sort_keys=True))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    g = sub.add_parser("gallery")
    g.add_argument("--crops", required=True, help="dir holding the cut crops")
    r = sub.add_parser("rates")
    r.add_argument("files", nargs="+", help="one or two per-rater verdict files")
    args = ap.parse_args(argv)
    {"plan": cmd_plan, "gallery": cmd_gallery, "rates": cmd_rates}[args.cmd](args)


if __name__ == "__main__":
    main()
