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
    ("visible", "Ramp itself is resolvable", "the model simply failed — VOCABULARY"),
    ("context-only", "Ramp not resolvable; crosswalk / apron / curb-cut cues imply one",
     "learnable, but from scene context rather than the ramp"),
    ("occluded", "Blocked by vehicle / pole / vegetation / person", "capture problem"),
    ("lighting", "Deep shadow or blown highlight", "capture problem"),
    ("surface", "Debris, snow, leaves, construction over it", "environment"),
    ("not-a-ramp", "Nothing ramp-like here — plain pavement, driveway apron",
     "GT error"),
    ("definition", "Imagery is clear; whether this CLASS counts as a curb ramp is the "
     "question (e.g. at-grade median cut-through with detectable warnings)",
     "rubric question, not a model failure"),
    ("unclear", "Cannot tell even with context", "excluded from every rate"),
]

# What the model latched onto for an isolated false positive. `real-ramp` is the one
# that is not a model error at all — it means the ground truth missed a ramp.
FP_SCHEME = [
    ("driveway", "Driveway apron", "ramp-like geometry, not a ramp"),
    ("crosswalk", "Crosswalk paint or markings", "texture confusion"),
    ("tactile-lookalike", "Manhole / grating / patterned surface", "truncated-dome lookalike"),
    ("stairs", "Steps", "ramp-adjacent furniture"),
    ("real-ramp", "Actually a ramp the GT missed", "NOT a model error — GT gap"),
    ("definition", "Real pedestrian feature, but is this CLASS a curb ramp? "
     "(e.g. at-grade median cut-through)", "rubric question, not a model failure"),
    ("other", "None of the above", ""),
    ("unclear", "Cannot tell from this imagery", "excluded from every rate"),
]


# The on-page briefing. It lives here rather than in a chat message or a doc because
# the reviewer reads it at the moment of judging, and because the ONE discipline that
# matters -- check panel 3 before calling a ramp visible -- is exactly the mistake the
# resolution-parity work exists to prevent. A guide that is not in front of the
# reviewer is a guide that does not run.
MISS_GUIDE = """
<h3>Each crop is a ramp that RampNet did not see at all</h3>
<p>A reviewer confirmed a real curb ramp at the <b class="gt">green circle</b>. RampNet produced
<b>nothing</b> there — not a weak detection, nothing, even at the 0.05 score floor. No other model
found it either. <b>Your keystroke records why.</b></p>
<h3 class="hot">Ask panel 3 only: "could I call this a ramp from THIS?"</h3>
<p>Panel 3 is the panorama at the model's own 4096&nbsp;px. It is the only panel that answers the
question we are asking. Panels 1 and 2 are there to tell you <i>what is actually on the ground</i>
so you can classify correctly — but on the 4&times; splits (bend, paterson, gainesville) panel 2
carries four times the detail the model ever had, so a ramp you can see there proves nothing. The
header turns <span class="adv">amber</span> on those.</p>
<h3>How to decide</h3>
<table>
  <tr><th>if, in panel 3</th><th>press</th></tr>
  <tr><td>the ramp itself is resolvable — you can point at the ramp, not just the place it should be</td><td><b>1</b> visible</td></tr>
  <tr><td>you cannot resolve the ramp, but a <b>crosswalk, coloured apron, or curb cut</b> tells you one is there</td><td><b>2</b> context-only</td></tr>
  <tr><td>a car, pole, vegetation or person is in the way</td><td><b>3</b> occluded</td></tr>
  <tr><td>deep shadow or a blown-out highlight</td><td><b>4</b> lighting</td></tr>
  <tr><td>debris, snow, leaves or construction over it</td><td><b>5</b> surface</td></tr>
  <tr><td>panels 1&ndash;2 show <b>nothing ramp-like</b> — plain pavement, driveway apron</td><td><b>6</b> not-a-ramp</td></tr>
  <tr><td>a real pedestrian feature is clearly there, but <b>whether it counts as a curb ramp is the question</b></td><td><b>7</b> definition</td></tr>
  <tr><td>nothing tells you either way, context included</td><td><b>8</b> unclear</td></tr>
</table>
<h3><span class="key">7</span> definition — for "is this even a curb ramp?"</h3>
<p>Distinct from <span class="key">6</span>: <code>not-a-ramp</code> means <i>nothing ramp-like is
there</i> and the ground truth is simply wrong. <code>definition</code> means the feature is real
and plainly visible, and the open question is whether <b>this class of thing</b> belongs in the
label set at all.</p>
<p>The case that prompted this: an <b>at-grade median cut-through</b> — a refuge island where the
crosswalk runs flush all the way across and there are detectable warning surfaces on both faces of
the median. There is <b>no running slope</b>; nothing ramps. It is an opening with truncated domes.
We have been labelling these as curb ramps.</p>
<p><b>Do not try to settle it while tagging — just press <span class="key">7</span>.</b> Tagged
separately, the numbers can be computed <i>both ways</i>, with and without this class in the ground
truth, and the rubric decided on real counts rather than in the abstract. A verdict here says
nothing bad about the model: it is a question about our label set.</p>
<h3><span class="key">2</span> context-only is a real answer, not a cop-out</h3>
<p>If you find yourself reasoning "there's a crosswalk here and a bit of coloured apron, so there
must be a ramp" — <b>that is <span class="key">2</span></b>, not <span class="key">1</span>. The
distinction is load-bearing: <b>RampNet sees that same context</b>, at that same resolution, across
the whole panorama. So context-only misses are still learnable — but from scene layout rather than
from the ramp's own appearance, which is a different capability and a different fix.</p>
<p><span class="key">1</span> and <span class="key">2</span> together bound what more training data
could reach; <span class="key">1</span> alone is the tight estimate. Everything else routes
elsewhere: occlusion and lighting are capture problems, <code>not-a-ramp</code> is a ground-truth
error, and none of them are helped by adding cities.</p>
<h3>Near-field first — you can stop when they run out</h3>
<p>The queue is <b>ordered near-field first</b>, and the header says <code>near</code> or
<code>far</code> for each crop. <b>The near ones are the ones that close the bracket</b>
(0.009&ndash;0.022 recall points). The far ones are a bonus question — they test whether the
far-field population really is pixel-starved as #59 assumed — so they are worth doing but not what
this pass is for.</p>
"""

FP_GUIDE = """
<h3>Each crop is a detection with no ramp under it</h3>
<p>The model fired at the <b class="gt">green circle</b>. There is no confirmed ramp within the
match radius, and it is not on the ego vehicle. <b>Your keystroke records what it latched onto.</b></p>
<h3>How to decide</h3>
<table>
  <tr><th>if you see</th><th>press</th></tr>
  <tr><td>driveway apron</td><td><b>1</b> driveway</td></tr>
  <tr><td>crosswalk paint or markings</td><td><b>2</b> crosswalk</td></tr>
  <tr><td>manhole, grating, or patterned surface resembling truncated domes</td><td><b>3</b> tactile-lookalike</td></tr>
  <tr><td>steps</td><td><b>4</b> stairs</td></tr>
  <tr><td><b>an actual curb ramp</b> the ground truth missed</td><td><b>5</b> real-ramp</td></tr>
  <tr><td>none of the above</td><td><b>6</b> other</td></tr>
  <tr><td>genuinely cannot tell</td><td><b>7</b> unclear</td></tr>
</table>
<h3 class="hot">The discipline, which cuts the opposite way here</h3>
<p>On an <span class="adv">advantaged</span> pano you may be calling a box "obviously not a ramp"
using detail <b>the model never had</b>. <b>Check panel 3</b> before dismissing it — if the thing is
ambiguous at the model's own resolution, that is a fair mistake, not a hallucination.</p>
<h3>Why it is worth the time</h3>
<p>These are the false positives geometry could not explain — 60–81% of every model's FP count.
Whether they are obvious junk or ambiguous concrete <b>sets the ceiling on the cascade arbiter in
#35</b>: an arbiter kills junk cheaply and struggles exactly where the detector did.
<b><span class="key">5</span> real-ramp is not a model error at all</b> — it means the ground truth
has a gap.</p>
"""

GUIDES = {"miss": MISS_GUIDE, "fp": FP_GUIDE}


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
    # Every verdict is both a key and a button: the keyboard is faster once you know
    # the scheme, the buttons mean you never have to learn it first.
    legend = "".join(
        f'<button class="v" data-v="{name}" onclick="setVerdict(\'{name}\')" '
        f'title="{why}"><b>{i+1}</b><span class="n">{name}</span>'
        f'<span class="d">{desc}</span></button>'
        for i, (name, desc, why) in enumerate(scheme))
    return (_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__ITEMS__", payload)
            .replace("__KEYS__", keys)
            .replace("__LEGEND__", legend)
            .replace("__GUIDE__", GUIDES.get(kind, ""))
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
  html, body { height:100%; }
  body { margin:0; background:#111; color:#ddd; display:flex; flex-direction:column;
         font:13px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif; }
  header { display:flex; gap:16px; align-items:baseline; padding:8px 12px;
           background:#181818; border-bottom:1px solid #2a2a2a; flex:0 0 auto; }
  header b { color:#fff; font-size:14px; }
  #meta { color:#9a9a9a; }
  #prog { margin-left:auto; color:#9a9a9a; white-space:nowrap; }
  /* Verdict bar sits directly under the header: it is the thing you are choosing
     from, so it should never be somewhere you have to go looking for. */
  #verdicts { display:flex; flex-wrap:wrap; gap:6px; padding:8px 12px;
              background:#141414; border-bottom:1px solid #2a2a2a; flex:0 0 auto; }
  button.v { display:flex; gap:7px; align-items:baseline; background:#242424;
             border:1px solid #363636; border-radius:5px; padding:5px 11px 5px 6px; }
  button.v:hover { background:#303030; border-color:#4a4a4a; }
  button.v b { background:#3a3a3a; border-radius:3px; padding:1px 7px; color:#fff; }
  button.v .n { color:#7fd1a0; font-weight:600; }
  button.v .d { color:#9d9d9d; }
  button.v.on { background:#1f3a2a; border-color:#3f7a56; }
  button.v.on b { background:#3f7a56; }
  /* The image fills whatever is left, so a whole crop is on screen without scrolling
     -- the panels only work as a comparison if all three are visible at once. */
  #wrap { flex:1 1 auto; min-height:0; display:flex; align-items:center;
          justify-content:center; background:#000; }
  #img { max-width:100%; max-height:100%; object-fit:contain; }
  footer { background:#181818; border-top:1px solid #2a2a2a; padding:6px 12px;
           display:flex; gap:16px; align-items:center; flex:0 0 auto; color:#888; }
  button { background:#2a2a2a; color:#ddd; border:1px solid #3a3a3a; border-radius:4px;
           padding:4px 10px; cursor:pointer; font:inherit; }
  button:hover { background:#343434; }
  .tagged { color:#7fd1a0; }
  #counts { margin-left:auto; }
  #done { padding:40px; text-align:center; font-size:16px; color:#ddd; }
  .adv { color:#e0a33c; }
  #warn { background:#4a2a12; color:#ffcf9a; padding:6px 12px;
          border-bottom:1px solid #6a3d1a; }
  #help { background:#171a19; border-bottom:1px solid #2a2a2a; padding:0 18px 18px;
          flex:0 0 auto; overflow:auto; max-height:60vh; }
  /* The dismiss control is sticky at the top of the panel, not at the bottom of it:
     the guide is taller than the panel, so a button at the end is a button you have
     to scroll to find, and the page reads as if it cannot be closed. */
  #helptop { position:sticky; top:0; background:#171a19; padding:12px 0 8px;
             display:flex; gap:12px; align-items:center; z-index:1;
             border-bottom:1px solid #232323; margin-bottom:10px; }
  #helptop b { color:#fff; }
  #help h3 { margin:16px 0 6px; font-size:13px; color:#7fd1a0; letter-spacing:.02em; }
  #help h3.hot { color:#e0a33c; }
  #help p { margin:0 0 4px; color:#c4c4c4; }
  #help table { border-collapse:collapse; margin:2px 0 4px; }
  #help td, #help th { padding:3px 14px 3px 0; text-align:left; vertical-align:top;
                       border-bottom:1px solid #232323; }
  #help th { color:#888; font-weight:500; font-size:11px; text-transform:uppercase; }
  #help td b { color:#fff; }
  #help .gt { color:#7fd1a0; }
  #help code, .key { background:#2f2f2f; border-radius:3px; padding:1px 5px; color:#fff; }
</style>
<header>
  <b id="title">__TITLE__</b>
  <span id="meta"></span>
  <span id="prog"></span>
</header>
<div id="warn" hidden></div>
<section id="help">
  <div id="helptop">
    <button onclick="hideHelp()">✕ &nbsp;Hide — start tagging</button>
    <span style="color:#777"><span class="key">?</span> or <span class="key">esc</span>
      toggles this any time</span>
  </div>
  __GUIDE__
</section>
<div id="verdicts">__LEGEND__</div>
<div id="wrap"><img id="img" alt=""><div id="done" hidden></div></div>
<footer>
  <span>panels: <b>context</b> · <b>detail (source)</b> · <b>as the model saw it</b></span>
  <button onclick="step(-1)">← back</button>
  <button onclick="step(1)">skip →</button>
  <button onclick="clearCurrent()">clear [x]</button>
  <button onclick="toggleHelp()">help [?]</button>
  <button onclick="exportJSON()">Export JSON</button>
  <button onclick="if(confirm('Discard all verdicts?'))reset()">Reset</button>
  <span id="counts"></span>
</footer>
<script>
const ITEMS = __ITEMS__, KEYS = __KEYS__, STORE = "__STORE__";
let verdicts = {}, i = 0, persist = true;

// localStorage is not guaranteed on a file:// origin -- some Chrome configurations
// throw SecurityError on both read and write. Unguarded, every keystroke would throw
// and the page would stop recording while still looking like it worked. So probe it
// once, and if it is unavailable fall back to memory and say so loudly: the work is
// still fine, but it has to be exported before the tab closes.
try {
  localStorage.setItem(STORE + "-probe", "1");
  localStorage.removeItem(STORE + "-probe");
  verdicts = JSON.parse(localStorage.getItem(STORE) || "{}");
} catch (e) {
  persist = false;
  verdicts = {};
}
if (window.__RESUME__) { Object.assign(verdicts, window.__RESUME__); }

function save() {
  if (!persist) return;
  try { localStorage.setItem(STORE, JSON.stringify(verdicts)); }
  catch (e) { persist = false; warnNoPersist(); }
}

function warnNoPersist() {
  const el = document.getElementById("warn");
  el.hidden = false;
  el.textContent = "⚠ This browser blocks storage on file:// URLs, so progress is "
    + "kept in memory only. It is NOT lost while this tab stays open — but export "
    + "before closing or reloading.";
}

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
  const cur = i < ITEMS.length ? verdicts[ITEMS[i].key] : null;
  document.querySelectorAll("button.v").forEach(b => {
    b.classList.toggle("on", b.dataset.v === cur);
    b.disabled = i >= ITEMS.length;
  });
  if (i >= ITEMS.length) {
    img.hidden = true; done.hidden = false;
    done.innerHTML = "All " + ITEMS.length + " tagged. Press <b>Export JSON</b>, then "
      + "hand off the file. <b>← back</b> still steps back if you want to revise.";
    document.getElementById("meta").textContent = "";
    return;
  }
  img.hidden = false; done.hidden = true;
  const it = ITEMS[i];
  img.src = it.file;
  const par = it.parity === "advantaged"
    ? ' <span class="adv">(advantaged — check panel 3 before calling it visible)</span>' : "";
  document.getElementById("meta").innerHTML = it.meta + par
    + (cur ? `  ·  <span class="tagged">[${cur}]</span>` : "");
}

function setVerdict(v) {
  if (i >= ITEMS.length || helpOpen()) return;
  verdicts[ITEMS[i].key] = v; save(); i++; show();
}

function step(d) {
  const j = i + d;
  if (j >= 0 && j <= ITEMS.length) { i = j; show(); }
}

function clearCurrent() {
  if (i >= ITEMS.length) return;
  delete verdicts[ITEMS[i].key]; save(); show();
}

function helpOpen() { return !document.getElementById("help").hidden; }

function hideHelp() {
  document.getElementById("help").hidden = true;
  try { localStorage.setItem(STORE + "-help", "seen"); } catch (e) {}
  window.scrollTo(0, 0);
}

function toggleHelp() {
  const el = document.getElementById("help");
  el.hidden = !el.hidden;
  if (el.hidden) window.scrollTo(0, 0);
}

// Collapsed once it has been read, so a resumed session goes straight to work. The
// briefing still matters on the first pass, so it is open by default rather than
// hidden behind a key nobody presses.
try { if (localStorage.getItem(STORE + "-help") === "seen")
        document.getElementById("help").hidden = true; } catch (e) {}

document.addEventListener("keydown", e => {
  if (e.key === "?" || (e.key === "/" && e.shiftKey)) { e.preventDefault(); toggleHelp(); return; }
  // While the briefing is up, swallow the verdict keys: a stray digit pressed while
  // reading would tag whatever crop happens to be current.
  if (helpOpen()) {
    if (e.key === "Escape" || e.key === "Enter") { e.preventDefault(); hideHelp(); }
    return;
  }
  if (e.key === "Backspace" || e.key === "ArrowLeft") { e.preventDefault(); step(-1); return; }
  if (e.key === " " || e.key === "ArrowRight") { e.preventDefault(); step(1); return; }
  if (e.key === "x") { clearCurrent(); return; }
  const d = parseInt(e.key, 10);
  if (d >= 1 && d <= KEYS.length) setVerdict(KEYS[d - 1]);
});

// A click lands focus on the button, after which the browser would fire it again on
// the next space/enter -- silently re-tagging whatever crop came next.
document.addEventListener("click", e => {
  if (e.target.closest("button")) e.target.closest("button").blur();
});

function exportJSON() {
  const blob = new Blob([JSON.stringify(verdicts, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "verdicts.json";
  a.click();
}
function reset() { verdicts = {}; save(); i = 0; show(); }

if (!persist) warnNoPersist();
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
