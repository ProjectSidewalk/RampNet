"""Build a keyboard-driven tagging page for a rendered gallery (#46).

``miss_gallery.py`` and ``fp_gallery.py`` render the crops. This turns a rendered
directory into a single self-contained ``tagger.html`` next to them: one crop on
screen at a time, **one keystroke per verdict**, auto-advance, and an export that is
byte-compatible with ``benchmark/<city>/incremental_fp_tags.json`` so the result
commits and joins like every other human-tag file in this repo.

It is a local page on purpose. The crops are git-ignored panorama derivatives sitting
on disk, so the page references them by relative path; anything hosted could not see
them.

    python scripts/analysis/miss_gallery.py --bucket silent --render out/silent ...
    python scripts/analysis/make_tagger.py out/silent
    # then open out/silent/tagger.html

**Resumable and hard to lose work in.** Every keystroke writes to ``localStorage``
immediately, so a refresh or a closed tab resumes where it left off; ``--resume``
additionally preloads verdicts from an existing export, so a partly-tagged set can be
handed between machines. Nothing is written to disk until Export is pressed — the
page cannot silently overwrite a committed tag file.

The verdict schemes below are the scientific content of this tool, not UI dressing:
they are what the taxonomy's remaining buckets decompose into, and they are chosen so
that **exactly one of them is the sourcing-addressable answer** (`visible` for misses,
`real-ramp` for false positives). Everything else is a different programme.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Why RampNet produced nothing at a ramp the reviewer confirmed. Only `visible` is a
# vocabulary failure and therefore the population more training data could reach; the
# rest are capture, environment or ground-truth problems that sourcing cannot touch.
MISS_SCHEME = [
    ("visible", "Clear ramp, unobstructed", "the model simply failed — VOCABULARY"),
    ("occluded", "Blocked by vehicle / pole / vegetation / person", "capture problem"),
    ("lighting", "Deep shadow or blown highlight", "capture problem"),
    ("surface", "Debris, snow, leaves, construction over it", "environment"),
    ("not-a-ramp", "No ramp here / flush or blended transition", "GT disagreement"),
    ("unclear", "Cannot tell from this imagery", "excluded from every rate"),
]

# What the model latched onto for an isolated false positive. `real-ramp` is the one
# that is not a model error at all — it means the ground truth missed a ramp.
FP_SCHEME = [
    ("driveway", "Driveway apron", "ramp-like geometry, not a ramp"),
    ("crosswalk", "Crosswalk paint or markings", "texture confusion"),
    ("tactile-lookalike", "Manhole / grating / patterned surface", "truncated-dome lookalike"),
    ("stairs", "Steps", "ramp-adjacent furniture"),
    ("real-ramp", "Actually a ramp the GT missed", "NOT a model error — GT gap"),
    ("other", "None of the above", ""),
    ("unclear", "Cannot tell from this imagery", "excluded from every rate"),
]


def scheme_for(items):
    """Pick the verdict scheme from what the manifest actually contains.

    A false-positive manifest carries ``model``; a miss manifest carries ``bucket``
    values from the miss taxonomy. Guessing from the directory name would break the
    moment someone renames it.
    """
    if any("model" in it for it in items.values()):
        return FP_SCHEME, "fp"
    return MISS_SCHEME, "miss"


def build_html(manifest, scheme, kind, title):
    items = []
    for key, it in manifest.items():
        items.append({
            "key": key, "file": it["file"], "city": it.get("city", ""),
            "pano": it.get("pano", ""),
            "meta": _meta_line(it, kind),
            "parity": it.get("parity", ""),
        })
    payload = json.dumps(items)
    keys = json.dumps([s[0] for s in scheme])
    legend = "".join(
        f'<div class="k"><b>{i+1}</b><span class="n">{name}</span>'
        f'<span class="d">{desc}</span><span class="w">{why}</span></div>'
        for i, (name, desc, why) in enumerate(scheme))
    return (_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__ITEMS__", payload)
            .replace("__KEYS__", keys)
            .replace("__LEGEND__", legend)
            .replace("__STORE__", store_key(kind, title)))


def store_key(kind, title):
    """Stable localStorage key for one gallery.

    Deliberately not ``hash()``: Python randomizes string hashing per process, so
    regenerating the page would silently move the key and abandon a half-finished
    tagging session in the old one. Content-derived, so the same gallery always
    resumes into the same store.
    """
    import hashlib
    digest = hashlib.md5(f"{kind}:{title}".encode("utf-8")).hexdigest()[:10]
    return f"rampnet-tagger-{kind}-{digest}"


def _meta_line(it, kind):
    bits = [it.get("city", ""), f"{it.get('dist_m', '?')} m",
            f"{it.get('source_px', '?')} src px", it.get("parity", "")]
    if kind == "fp":
        bits.insert(1, str(it.get("model", "")))
        c = it.get("confidence")
        if c is not None:
            bits.insert(2, f"conf {c:.2f}")
    else:
        bits.insert(1, it.get("field", ""))
    return "  ·  ".join(str(b) for b in bits if b != "")


_TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#111; color:#ddd;
         font:13px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif; }
  header { display:flex; gap:16px; align-items:baseline; padding:8px 12px;
           background:#181818; border-bottom:1px solid #2a2a2a; position:sticky; top:0; }
  header b { color:#fff; font-size:14px; }
  #meta { color:#9a9a9a; }
  #prog { margin-left:auto; color:#9a9a9a; }
  #img { display:block; width:100%; height:auto; background:#000; }
  #wrap { padding:0 0 140px; }
  footer { position:fixed; bottom:0; left:0; right:0; background:#181818;
           border-top:1px solid #2a2a2a; padding:8px 12px; display:flex;
           flex-wrap:wrap; gap:6px 18px; align-items:center; }
  .k { display:flex; gap:6px; align-items:baseline; }
  .k b { background:#2f2f2f; border-radius:3px; padding:1px 6px; color:#fff; }
  .k .n { color:#7fd1a0; font-weight:600; }
  .k .d { color:#bbb; }
  .k .w { color:#777; font-style:italic; }
  #bar { width:100%; display:flex; gap:12px; color:#888; }
  button { background:#2a2a2a; color:#ddd; border:1px solid #3a3a3a; border-radius:4px;
           padding:4px 10px; cursor:pointer; font:inherit; }
  button:hover { background:#343434; }
  .tagged { color:#7fd1a0; }
  #done { padding:40px; text-align:center; font-size:16px; }
  .adv { color:#e0a33c; }
</style>
<header>
  <b id="title">__TITLE__</b>
  <span id="meta"></span>
  <span id="prog"></span>
</header>
<div id="wrap"><img id="img" alt=""><div id="done" hidden></div></div>
<footer>
  <div id="bar">
    <span>panels: <b>context</b> · <b>detail (source)</b> · <b>as the model saw it</b></span>
    <span>[space] skip · [backspace] undo · [x] clear this one</span>
    <button onclick="exportJSON()">Export JSON</button>
    <button onclick="if(confirm('Discard all verdicts?'))reset()">Reset</button>
    <span id="counts"></span>
  </div>
  __LEGEND__
</footer>
<script>
const ITEMS = __ITEMS__, KEYS = __KEYS__, STORE = "__STORE__";
let verdicts = {}, i = 0;
try { verdicts = JSON.parse(localStorage.getItem(STORE) || "{}"); } catch (e) { verdicts = {}; }
if (window.__RESUME__) { Object.assign(verdicts, window.__RESUME__); }

function save() { localStorage.setItem(STORE, JSON.stringify(verdicts)); }

function firstUntagged() {
  for (let j = 0; j < ITEMS.length; j++) if (!verdicts[ITEMS[j].key]) return j;
  return ITEMS.length;
}

function show() {
  const done = document.getElementById("done"), img = document.getElementById("img");
  const n = Object.keys(verdicts).length;
  document.getElementById("prog").textContent =
    `${n} / ${ITEMS.length} tagged` + (i < ITEMS.length ? `  ·  #${i + 1}` : "");
  const c = {};
  for (const v of Object.values(verdicts)) c[v] = (c[v] || 0) + 1;
  document.getElementById("counts").innerHTML = KEYS
    .filter(k => c[k]).map(k => `<span class="tagged">${k} ${c[k]}</span>`).join(" · ");
  if (i >= ITEMS.length) {
    img.hidden = true; done.hidden = false;
    done.innerHTML = "All " + ITEMS.length + " tagged. Press <b>Export JSON</b>, " +
      "then commit the file. Backspace still steps back.";
    document.getElementById("meta").textContent = "";
    return;
  }
  img.hidden = false; done.hidden = true;
  const it = ITEMS[i];
  img.src = it.file;
  const cur = verdicts[it.key] ? `  ·  [${verdicts[it.key]}]` : "";
  const par = it.parity === "advantaged"
    ? ' <span class="adv">(advantaged — compare panel 3 before calling it visible)</span>' : "";
  document.getElementById("meta").innerHTML = it.meta + par + cur;
}

function setVerdict(v) {
  if (i >= ITEMS.length) return;
  verdicts[ITEMS[i].key] = v; save(); i++; show();
}

document.addEventListener("keydown", e => {
  if (e.key === "Backspace") { e.preventDefault(); if (i > 0) { i--; show(); } return; }
  if (e.key === " ") { e.preventDefault(); if (i < ITEMS.length) { i++; show(); } return; }
  if (e.key === "x" && i < ITEMS.length) { delete verdicts[ITEMS[i].key]; save(); show(); return; }
  if (e.key === "ArrowLeft") { if (i > 0) { i--; show(); } return; }
  if (e.key === "ArrowRight") { if (i < ITEMS.length) { i++; show(); } return; }
  const d = parseInt(e.key, 10);
  if (d >= 1 && d <= KEYS.length) setVerdict(KEYS[d - 1]);
});

function exportJSON() {
  const blob = new Blob([JSON.stringify(verdicts, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "verdicts.json";
  a.click();
}
function reset() { verdicts = {}; save(); i = 0; show(); }

i = firstUntagged(); show();
</script>
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("gallery", help="Directory written by miss_gallery.py / fp_gallery.py.")
    p.add_argument("--resume", help="Existing verdicts JSON to preload.")
    p.add_argument("--title", default=None)
    args = p.parse_args(argv)

    mpath = os.path.join(args.gallery, "manifest.json")
    if not os.path.exists(mpath):
        raise SystemExit(f"{mpath}: no manifest — run a gallery script with --render first")
    with open(mpath, encoding="utf-8") as fh:
        manifest = json.load(fh)["items"]
    if not manifest:
        raise SystemExit(f"{mpath}: manifest is empty")

    scheme, kind = scheme_for(manifest)
    title = args.title or f"RampNet #46 tagger — {os.path.basename(os.path.abspath(args.gallery))}"
    html = build_html(manifest, scheme, kind, title)

    if args.resume:
        with open(args.resume, encoding="utf-8") as fh:
            html = html.replace("<script>",
                                f"<script>window.__RESUME__ = {json.dumps(json.load(fh))};",
                                1)

    out = os.path.join(args.gallery, "tagger.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"{len(manifest)} items, {kind} scheme ({len(scheme)} verdicts)")
    for n, (name, desc, why) in enumerate(scheme, 1):
        print(f"   [{n}] {name:<18} {desc}" + (f"   — {why}" if why else ""))
    print(f"\nWrote {out}")
    print("Open it in a browser. One keystroke per crop, auto-advances; every keystroke")
    print("is saved immediately, so closing the tab loses nothing. Export writes")
    print("verdicts.json keyed exactly like benchmark/<city>/incremental_fp_tags.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
