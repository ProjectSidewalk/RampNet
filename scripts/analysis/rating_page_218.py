"""The one-question rating page shared by #218's galleries (Seoul presence, Richmond
flat-photo detections). Same pattern as ``residual_gt_check_48.py``: a rater id, answers
kept per rater in the browser, and an Export that writes a per-rater JSON carrying the
question, rubric and rules, keyed to a manifest digest of the images rated."""
import html
import json
import os
import re

RATER_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def _ring(it):
    if not it.get("ring"):
        return ""
    fx, fy = it["ring"]
    w, h = it["w"], it["h"]
    cx, cy = fx * w, fy * h
    r = max(10.0, 0.035 * w)
    return (f'<svg class="halo" viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
            f'aria-hidden="true" focusable="false"><circle class="o" cx="{cx:.1f}" '
            f'cy="{cy:.1f}" r="{r + 2:.1f}"/><circle cx="{cx:.1f}" cy="{cy:.1f}" '
            f'r="{r:.1f}"/></svg>')


def _wrap_open(it, key, click):
    """The image wrapper. With the click mark on it is focusable and takes the mark's
    keyboard path (role=application so a screen reader passes the arrow keys through)."""
    if not click:
        return '<span class="ringwrap">'
    return (f'<span class="ringwrap clickable" tabindex="0" role="application" '
            f'aria-roledescription="ramp marker" aria-describedby="ms_{key}" '
            f'aria-label="{html.escape(it["name"])} image: click, or press Enter, to mark where '
            f'the ramp is; arrow keys move the mark (Shift for bigger steps); Delete clears it">')


def _mark_svg(it):
    """The rater's mark, not displayed until placed (an SVG ``display`` attribute: the HTML
    ``hidden`` attribute does not hide an SVG ``<g>`` in Chromium): a magenta crosshair in a diamond with a
    black under-stroke, so it reads on any photo and never looks like the green ring."""
    w, h = it["w"], it["h"]
    r = max(10.0, 0.035 * w)
    d = (f"M0 {-r:.1f} L{r:.1f} 0 L0 {r:.1f} L{-r:.1f} 0 Z "
         f"M0 {-1.6 * r:.1f} V{1.6 * r:.1f} M{-1.6 * r:.1f} 0 H{1.6 * r:.1f}")
    return (f'<svg class="mark" viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
            f'aria-hidden="true" focusable="false"><g display="none"><path class="o" d="{d}"/>'
            f'<path d="{d}"/></g></svg>')


def _mark_bar(key):
    return (f'<div class="markbar"><span class="muted" id="ms_{key}">No ramp mark.</span> '
            f'<button type="button" class="clearmark">Clear mark</button></div>')


def _caption(it):
    if not it.get("caption"):
        return ""
    return f'<p class="muted">{html.escape(it["caption"])}</p>'


def render(items, digest, cfg):
    """``items``: [{"name", "img" (relative path), "w", "h", optional "ring": (fx, fy)
    as fractions of the image, optional "caption", optional "alt"}]. ``cfg``: title, h1,
    intro (HTML), question, rubric [(key, label, definition)], rules, keys {letter: key},
    task, export_prefix, storage_prefix, gallery_rel, commit_dir; optional ``click_mark``
    (default False): when True, each card's image also takes an optional "where the ramp
    is" mark (click, or keyboard on the focused image), saved with the answer as
    ``click: [fx, fy]`` (fractions of the image width and height, 4 dp). With it off the
    page is byte-identical to the page this function wrote before the option existed."""
    CLICK = bool(cfg.get("click_mark", False))
    QUESTION, RUBRIC, RULES = cfg["question"], cfg["rubric"], cfg["rules"]
    EXPORT_PREFIX, EXPORT_SUFFIX = cfg["export_prefix"], ".json"
    GALLERY_REL = cfg["gallery_rel"]
    radios = lambda key: "".join(  # noqa: E731
        f'<label><input type="radio" name="v_{key}" value="{k}"> {html.escape(lab)}</label>'
        for k, lab, _ in RUBRIC)
    cards = []
    for n, it in enumerate(items, 1):
        key = re.sub(r"[^A-Za-z0-9_]", "_", it["name"])
        cards.append(
            f'<section class="card" data-uid="{html.escape(it["name"])}" aria-labelledby="h_{key}">'
            f'<h2 id="h_{key}">{n}. {html.escape(it["name"])}</h2>'
            f'{_wrap_open(it, key, CLICK)}<img src="{html.escape(it["img"])}" width="{it["w"]}" '
            f'height="{it["h"]}" loading="lazy" alt="{html.escape(it.get("alt", it["name"]))}">'
            f'{_ring(it)}{_mark_svg(it) if CLICK else ""}</span>'
            f'{_mark_bar(key) if CLICK else ""}'
            f'{_caption(it)}'
            f'<fieldset><legend>{html.escape(QUESTION)}</legend><div class="opts">'
            f'{radios(key)}</div><label class="note">Note (optional) '
            f'<textarea rows="2" name="n_{key}"></textarea></label></fieldset></section>')
    rubric_html = "".join(f"<dt>{html.escape(lab)}</dt><dd>{html.escape(d)}</dd>"
                          for _, lab, d in RUBRIC)
    rules_html = "".join(f"<li>{html.escape(x)}</li>" for x in RULES)
    meta = json.dumps({"question": QUESTION,
                       "rubric": [{"key": k, "label": lab, "definition": d}
                                  for k, lab, d in RUBRIC],
                       "rules": RULES, "items": [it["name"] for it in items],
                       "manifest_digest": digest, "gallery": GALLERY_REL})
    meta_js = meta.replace("</", "<" + chr(92) + "/")
    keys = cfg["keys"]
    C = _click_parts(CLICK)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(cfg['title'])}</title>
<style>
:root {{ color-scheme:light dark; --bg:#ffffff; --fg:#1f2328; --muted:#57606a; --line:#d0d7de; --focus:#0969da; --panel:#f6f8fa; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0d1117; --fg:#e6edf3; --muted:#8d96a0; --line:#30363d; --focus:#4493f8; --panel:#161b22; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 16px 64px; max-width:1100px; }}
h1 {{ font-size:22px; }} h2 {{ font-size:16px; margin:0 0 8px; }}
.muted {{ color:var(--muted); font-size:13px; }}
.intro {{ max-width:75ch; }}
.card {{ border-top:1px solid var(--line); padding:14px 0; }}
.card.done h2::after {{ content:" \\2713"; color:var(--muted); }}
img {{ max-width:100%; height:auto; display:block; }}
fieldset {{ border:1px solid var(--line); border-radius:6px; background:var(--panel); margin:8px 0 0; padding:8px 10px; }}
legend {{ font-weight:600; padding:0 4px; }}
.opts {{ display:flex; flex-wrap:wrap; gap:4px 18px; }}
.opts label {{ cursor:pointer; padding:2px 0; }}
.ringwrap {{ position:relative; display:block; width:fit-content; max-width:100%; }}
.halo {{ position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }}
.halo circle {{ fill:none; stroke:#00e676; stroke-width:2.5; vector-effect:non-scaling-stroke; }}
.halo circle.o {{ stroke:#000; stroke-width:1.2; }}{C["css"]}
.note {{ display:block; margin-top:6px; font-size:13px; color:var(--muted); }}
textarea {{ display:block; width:100%; box-sizing:border-box; font:inherit; color:var(--fg); background:var(--bg); border:1px solid var(--line); border-radius:4px; }}
:focus-visible {{ outline:3px solid var(--focus); outline-offset:2px; }}
button {{ font:inherit; padding:6px 12px; }}
input[type=text] {{ font:inherit; color:var(--fg); background:var(--bg); border:1px solid var(--line); border-radius:4px; padding:4px 6px; }}
dt {{ font-weight:600; }} dd {{ margin:0 0 6px 16px; }}
.bar {{ position:sticky; top:0; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--line); z-index:1; display:flex; flex-wrap:wrap; gap:8px 12px; align-items:center; }}
</style></head><body>
<h1>{html.escape(cfg['h1'])}</h1>
<div class="intro">{cfg['intro']}</div>
<dl>{rubric_html}</dl>
<ul>{rules_html}</ul>
<p class="muted">Enter your rater id first (lower-case letters, digits, "_" or "-"; or
<code>?rater=yourid</code> in the address). Answers are saved in this browser per rater as you
go; Export writes <code>{EXPORT_PREFIX}&lt;rater&gt;{EXPORT_SUFFIX}</code>, to be committed under
<code>{html.escape(cfg['commit_dir'])}</code>. Keys: {html.escape(", ".join(k.upper() for k in keys))} answer the card that
has focus or else the topmost card on screen (not inside a text field).{C["help"]}</p>
<div class="bar"><label for="rater">Rater id</label>
<input id="rater" type="text" size="10" autocomplete="off" spellcheck="false" aria-describedby="msg">
<button type="button" id="export">Export verdicts JSON</button>
<button type="button" id="next">Next unanswered</button>
<span id="count" aria-live="polite"></span> <span id="msg" role="status"></span></div>
{"".join(cards)}
<script id="meta" type="application/json">{meta_js}</script>
<script>
const META = JSON.parse(document.getElementById('meta').textContent);
const RATER_RE = new RegExp({json.dumps(RATER_RE.pattern)});
const LAST_RATER = "{cfg['storage_prefix']}last_rater";
function getItem(k) {{ try {{ return localStorage.getItem(k); }} catch (e) {{ return null; }} }}
function setItem(k, v) {{ try {{ localStorage.setItem(k, v); }} catch (e) {{}} }}
const CARDS = [...document.querySelectorAll('.card')];
const raterInput = document.getElementById('rater');
let rater = (new URLSearchParams(location.search).get('rater') || getItem(LAST_RATER) || "").trim().toLowerCase();
if (!RATER_RE.test(rater)) rater = "";
raterInput.value = rater;
let saved = {{}};
function storageKey() {{ return "{cfg['storage_prefix']}" + META.manifest_digest + "__" + rater; }}
function load() {{ saved = {{}}; if (!rater) return; try {{ saved = JSON.parse(getItem(storageKey()) || "{{}}") || {{}}; }} catch (e) {{ saved = {{}}; }} }}
function persist() {{ if (rater) setItem(storageKey(), JSON.stringify(saved)); }}
function say(t) {{ document.getElementById('msg').textContent = t; }}
function needRater() {{ if (rater) return false; say("Enter your rater id first."); raterInput.focus(); return true; }}
function answered() {{ return META.items.filter(u => saved[u] && saved[u].answer).length; }}
function update() {{
  document.getElementById('count').textContent = (rater ? rater + ": " : "") + answered() + " of " + META.items.length + " answered";
  CARDS.forEach(c => c.classList.toggle('done', !!(saved[c.dataset.uid] || {{}}).answer));
}}
function render() {{
  CARDS.forEach(card => {{
    const cur = saved[card.dataset.uid] || {{}};
    card.querySelectorAll('input[type=radio]').forEach(inp => {{ inp.checked = cur.answer === inp.value; }});
    card.querySelector('textarea').value = cur.note || "";{C["render"]}
  }});
  update();
}}
raterInput.addEventListener('change', () => {{
  const v = raterInput.value.trim().toLowerCase();
  if (!RATER_RE.test(v)) {{ raterInput.value = rater; say("Rater id " + JSON.stringify(v) + " refused: use lower-case letters, digits, _ or -, up to 32 characters."); return; }}
  rater = v; setItem(LAST_RATER, v); load(); render(); say("Rating as " + v + ".");
}});
CARDS.forEach(card => {{
  const uid = card.dataset.uid;
  card.querySelectorAll('input[type=radio]').forEach(inp => {{
    inp.addEventListener('change', () => {{
      if (needRater()) {{ inp.checked = false; return; }}
      saved[uid] = Object.assign(saved[uid] || {{}}, {{answer: inp.value}}); persist(); update();
    }});
  }});
  const ta = card.querySelector('textarea');
  ta.addEventListener('input', () => {{
    if (needRater()) {{ ta.value = ""; return; }}
    saved[uid] = Object.assign(saved[uid] || {{}}, {{note: ta.value}}); persist();
  }});
}});{C["js"]}
function topCard() {{
  const barBottom = document.querySelector('.bar').getBoundingClientRect().bottom;
  return CARDS.find(c => c.getBoundingClientRect().bottom > barBottom + 40);
}}
document.addEventListener('keydown', ev => {{
  const t = ev.target;
  if (t.tagName === 'TEXTAREA' || (t.tagName === 'INPUT' && t.type === 'text')) return;
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const k = {json.dumps(keys)}[(ev.key || "").toLowerCase()];
  if (!k) return;
  const card = (t.closest && t.closest('.card')) || topCard();
  if (!card) return;
  ev.preventDefault();
  if (needRater()) return;
  const inp = card.querySelector('input[value="' + k + '"]');
  inp.checked = true; if (!(t.closest && t.closest('.ringwrap'))) inp.focus(); inp.dispatchEvent(new Event('change'));
}});
load(); render();
document.getElementById('next').addEventListener('click', () => {{
  const card = CARDS.find(c => !(saved[c.dataset.uid] || {{}}).answer);
  if (card) {{ card.scrollIntoView({{block: 'start'}}); card.querySelector('input[type=radio]').focus({{preventScroll: true}}); }}
}});
document.getElementById('export').addEventListener('click', () => {{
  if (needRater()) return;
  const verdicts = {{}};
  META.items.forEach(u => {{
    const v = saved[u];
    if (!v || (!v.answer && !(v.note || "").trim(){C["export_keep"]})) return;
    verdicts[u] = {{answer: v.answer || null, note: (v.note || "").trim()}};{C["export_add"]}
  }});
  const out = {{task: {json.dumps(cfg["task"])},
    question: META.question, rubric: META.rubric, rules: META.rules, rater: rater,
    items: META.items, manifest_digest: META.manifest_digest, n_items: META.items.length,
    n_answered: answered(), gallery: META.gallery, exported_at: new Date().toISOString(),
    verdicts: verdicts}};
  const blob = new Blob([JSON.stringify(out, null, 1) + "\\n"], {{type: "application/json"}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = "{EXPORT_PREFIX}" + rater + "{EXPORT_SUFFIX}";
  a.click();
  say("Exported {EXPORT_PREFIX}" + rater + "{EXPORT_SUFFIX}.");
}});
</script></body></html>
"""


def _click_parts(click):
    """The page fragments the click mark adds; every one is empty when it is off, so the
    page without it is unchanged byte for byte."""
    if not click:
        return dict.fromkeys(("css", "help", "render", "js", "export_keep", "export_add"), "")
    return {
        "css": """
.mark { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
.mark path { fill:none; stroke:#ff2bd6; stroke-width:2.5; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
.mark path.o { stroke:#000; stroke-width:5.5; }
.ringwrap.clickable { cursor:crosshair; }
.markbar { display:flex; flex-wrap:wrap; gap:4px 12px; align-items:center; margin-top:6px; }""",
        "help": (" Optional ramp mark: click the image where the ramp is (click again to "
                 "move it, Clear mark to remove it); or Tab to the image, press Enter to put "
                 "the mark at the ring, arrow keys to move it (Shift for bigger steps), "
                 "Delete or Backspace to clear it; an answer key pressed there answers the "
                 "card and leaves focus on the image. The mark is magenta; the ring is green."),
        "render": "\n    drawMark(card, cur.click);",
        "js": r"""
function pct(f) { return Math.round(f * 100) + "%"; }
function drawMark(card, c) {
  const g = card.querySelector('.mark g');
  const st = card.querySelector('.markbar .muted');
  if (!c) { g.setAttribute('display', 'none'); st.textContent = "No ramp mark."; return; }
  const vb = card.querySelector('.mark').viewBox.baseVal;
  g.setAttribute('transform', 'translate(' + (c[0] * vb.width).toFixed(1) + ' ' + (c[1] * vb.height).toFixed(1) + ')');
  g.removeAttribute('display');
  st.textContent = "Ramp mark at " + pct(c[0]) + " across, " + pct(c[1]) + " down.";
}
function r4(x) { return Math.round(Math.min(1, Math.max(0, x)) * 10000) / 10000; }
function setMark(card, c) {
  if (needRater()) return;
  const uid = card.dataset.uid;
  if (!c && !(saved[uid] || {}).click) { say(uid + ": no ramp mark to clear."); return; }
  const cur = Object.assign(saved[uid] || {}, {});
  if (c) cur.click = [r4(c[0]), r4(c[1])]; else delete cur.click;
  saved[uid] = cur; persist(); drawMark(card, cur.click);
  say(uid + ": " + (cur.click ? "ramp mark at " + pct(cur.click[0]) + " across, " + pct(cur.click[1]) + " down." : "ramp mark cleared."));
}
function startMark(card) {
  const ring = card.querySelector('.halo circle:not(.o)');
  const vb = card.querySelector('.mark').viewBox.baseVal;
  return ring ? [ring.cx.baseVal.value / vb.width, ring.cy.baseVal.value / vb.height] : [0.5, 0.5];
}
CARDS.forEach(card => {
  const wrap = card.querySelector('.ringwrap.clickable');
  const img = wrap.querySelector('img');
  wrap.addEventListener('click', ev => {
    const r = img.getBoundingClientRect();
    if (!r.width || !r.height) return;
    setMark(card, [(ev.clientX - r.left) / r.width, (ev.clientY - r.top) / r.height]);
  });
  wrap.addEventListener('keydown', ev => {
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const cur = (saved[card.dataset.uid] || {}).click;
    const step = ev.shiftKey ? 0.05 : 0.01;
    const mv = {ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step]}[ev.key];
    if (ev.key === 'Enter') { setMark(card, cur || startMark(card)); }
    else if (mv) { const b = cur || startMark(card); setMark(card, [b[0] + mv[0], b[1] + mv[1]]); }
    else if (ev.key === 'Delete' || ev.key === 'Backspace') { setMark(card, null); }
    else return;
    ev.preventDefault(); ev.stopPropagation();
  });
  card.querySelector('.clearmark').addEventListener('click', () => setMark(card, null));
});""",
        "export_keep": " && !v.click",
        "export_add": "\n    if (v.click) verdicts[u].click = v.click;",
    }


# --------------------------------------------------------------------------- #
# reading a rater's export back (for ``rates``)
# --------------------------------------------------------------------------- #
def load_verdicts(path, reference, export_prefix):
    """A rater's export, refused unless it was made on the committed gallery under the
    current rubric (the pattern of ``residual_gt_check_48.load_verdicts``):

    - the file is named ``<export_prefix><rater>.json`` for its own, valid ``rater``;
    - its ``manifest_digest`` and item list (in order) equal ``reference``'s;
    - its ``question``, ``rubric`` and ``rules`` equal ``reference``'s, so a verdict made
      under an older rubric cannot be mixed in silently;
    - every verdict is for a listed item, with a rubric answer or null, and an optional
      ``click`` of [fx, fy] fractions in [0, 1] (the click-mark pages only).

    ``reference``: {"manifest_digest", "items", "question", "rubric" (list of
    {key, label, definition}), "rules"} -- the committed gallery, re-derived by the caller.
    """
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    rater = d.get("rater")
    if not isinstance(rater, str) or not RATER_RE.match(rater):
        raise ValueError(f"{path}: rater id {rater!r} is missing or not a valid id")
    if os.path.basename(path) != f"{export_prefix}{rater}.json":
        raise ValueError(f"{path}: file name does not match its rater id {rater!r} "
                         f"(expected {export_prefix}{rater}.json)")
    if d.get("manifest_digest") != reference["manifest_digest"]:
        raise ValueError(f"{path}: made on gallery {d.get('manifest_digest')}, but the "
                         f"committed gallery is {reference['manifest_digest']}")
    if d.get("items") != reference["items"]:
        raise ValueError(f"{path}: item list differs from the committed manifest")
    for k in ("question", "rubric", "rules"):
        if d.get(k) != reference[k]:
            raise ValueError(f"{path}: {k} differs from the committed gallery's; a verdict "
                             "made under another rubric is not comparable")
    answers = {r["key"] for r in reference["rubric"]}
    items = set(reference["items"])
    for uid, v in (d.get("verdicts") or {}).items():
        if uid not in items:
            raise ValueError(f"{path}: {uid} is not in the item list")
        if (v or {}).get("answer") not in answers | {None}:
            raise ValueError(f"{path}: {uid} has answer {v.get('answer')!r}, not in "
                             f"{sorted(answers)}")
        c = (v or {}).get("click")
        if c is not None and not (isinstance(c, list) and len(c) == 2 and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 1
                for x in c)):
            raise ValueError(f"{path}: {uid} has click {c!r}, not [fx, fy] in [0, 1]")
    return d


def agreement(va, vb, items, label=lambda a: a):
    """Two raters on the same items: raw agreement and Cohen's kappa over the items both
    decided (``label(answer)`` not None).

    >>> va = {"a": {"answer": "yes"}, "b": {"answer": "no"}, "c": {"answer": "yes"}}
    >>> vb = {"a": {"answer": "yes"}, "b": {"answer": "yes"}, "c": {"answer": "yes"}}
    >>> agreement(va, vb, ["a", "b", "c"])["n"]
    3
    """
    pairs = []
    for u in items:
        a = label(((va.get(u) or {}).get("answer")))
        b = label(((vb.get(u) or {}).get("answer")))
        if a is not None and b is not None:
            pairs.append((a, b))
    n = len(pairs)
    if not n:
        return {"n": 0, "agreement": None, "kappa": None}
    po = sum(a == b for a, b in pairs) / n
    cats = sorted({x for p in pairs for x in p}, key=str)
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n)
             for c in cats)
    return {"n": n, "agreement": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else None}
