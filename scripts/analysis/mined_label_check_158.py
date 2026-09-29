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
def empty_verdicts(ids, digest, rater):
    return {"task": "RampNet #158 step 4, mined-label check: " + QUESTION,
            "question": QUESTION,
            "rubric": [{"key": k, "label": lab, "definition": d} for k, lab, d in RUBRIC],
            "rules": RULES, "rater": rater, "items": ids, "manifest_digest": digest,
            "n_items": len(ids), "n_answered": 0, "gallery": GALLERY_REL, "verdicts": {}}


def _figure(it, uid, width):
    name = mv.crop_name(it)
    date = html.escape(it["capture_date"] or "date unknown")
    if it["ring"]:
        alt, cap, cls = (f"Card {uid}: the view to rate, ring on the spot",
                         f"<strong>Rate this view</strong> &middot; {date}", "src")
    else:
        alt, cap, cls = (f"Card {uid}: another capture of the corner, unmarked",
                         f"Context: another capture of the corner &middot; {date} &middot; "
                         f"unmarked, not rated", "ctx")
    img = (f'<img src="crops/{html.escape(name)}" width="{width}" '
           f'height="{round(width * mv.CROP_PX[1] / mv.CROP_PX[0])}" loading="lazy" '
           f'alt="{html.escape(alt)}">')
    if it["ring"]:
        img = f'<span class="ringwrap">{img}{R._halo(it)}</span>'
    return f'<figure class="{cls}">{img}<figcaption>{cap}</figcaption></figure>'


def render_gallery(cards, crops, digest):
    """The #48 GT-check page, re-labelled for this question. The JS is the same shape:
    rater id, per-rater localStorage, Y / N / C keys, export to the per-rater file."""
    by_card = {}
    for it in crops:
        by_card.setdefault(it["ramp_uid"], []).append(it)
    out_cards = []
    for n, c in enumerate(cards, 1):
        uid = c["id"]
        tgt = [it for it in by_card[uid] if it["ring"]]
        ctx = [it for it in by_card[uid] if not it["ring"]]
        radios = "".join(
            f'<label><input type="radio" name="v_{uid}" value="{k}"> {html.escape(lab)}</label>'
            for k, lab, _ in RUBRIC)
        ctx_html = "".join(_figure(it, uid, mv.CROP_PX[0]) for it in ctx) or \
            '<p class="muted">No context view.</p>'
        out_cards.append(
            f'<section class="card" data-uid="{uid}" aria-labelledby="h_{uid}">'
            f'<h2 id="h_{uid}">{n}. {uid}</h2>'
            f'<div class="row"><div class="rate">{"".join(_figure(it, uid, 540) for it in tgt)}'
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
            "task": "RampNet #158 step 4, mined-label check: " + QUESTION}
    return _PAGE(out_cards, meta)


def _PAGE(cards_html, meta):
    """Fill the #48 GT-check template: its CSS and script, with this task's text."""
    tmpl = R.render_gallery(
        [{"uid": "__X__", "class": "x"}],
        [{"ramp_uid": "__X__", "city": "x", "pano_id": "x", "x": 0.5, "y": 0.5,
          "is_source": True, "ring": False, "capture_date": ""}], meta["manifest_digest"])
    head, rest = tmpl.split('<section class="card"', 1)
    tail = rest[rest.index('<script id="meta"'):]
    head = head.replace("<title>Residual GT check</title>", "<title>Mined label check</title>")
    head = re.sub(r"<h1>.*?</h1>", "<h1>Mined labels (#158 step 4): is there a curb ramp at "
                  "the ring?</h1>", head, count=1, flags=re.S)
    intro = (
        '<div class="intro"><p>Each card is one label that the miner would add to RampNet\'s '
        'training data: a spot in a panorama where other captures agree there is a curb '
        'ramp, but this panorama\'s detector did not fire. Look at the large view on the '
        f'left and answer one question: <strong>{html.escape(QUESTION)}</strong> The smaller '
        'view on the right is another capture of the same corner, shown only as context, '
        'without a ring.</p></div>')
    head = re.sub(r'<div class="intro">.*?</div>', lambda _m: intro, head, count=1, flags=re.S)
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
                  'rules: META.rules, rater: rater, items: META.items, ', tail, count=1, flags=re.S)
    tail = re.sub(r'\s*supersedes: "[^"]*",', '', tail, count=1)
    tail = tail.replace(R.EXPORT_PREFIX, EXPORT_PREFIX)
    return head + "".join(cards_html) + "\n" + tail


def cmd_gallery(args):
    if args.init_rater is not None and not RATER_RE.match(args.init_rater):
        raise SystemExit(f"--init-rater {args.init_rater!r} is not a valid rater id")
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


def load_verdicts(path, ref):
    d = json.load(open(path, encoding="utf-8"))
    rater = d.get("rater")
    if not isinstance(rater, str) or not RATER_RE.match(rater):
        raise ValueError(f"{path}: rater id {rater!r} missing or invalid")
    if os.path.basename(path) != EXPORT_PREFIX + rater + EXPORT_SUFFIX:
        raise ValueError(f"{path}: file name does not match rater {rater!r}")
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


def cmd_rates(args):
    if len(args.files) > 2:
        raise SystemExit("rates takes one or two verdict files")
    ref = committed_reference()
    files = [load_verdicts(p, ref) for p in args.files]
    out = {"rates": [rates(d, ref) for d in files]}
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
    g = sub.add_parser("gallery")
    g.add_argument("--crops", required=True)
    g.add_argument("--init-rater", default=None)
    r = sub.add_parser("rates")
    r.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    {"plan": cmd_plan, "gallery": cmd_gallery, "rates": cmd_rates}[args.cmd](args)


if __name__ == "__main__":
    main()
