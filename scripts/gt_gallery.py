"""Ground-truth labeling tool: the spot-check gallery, ported onto the benchmark bundle.

Renders a self-contained, single-pano-at-a-time viewer (``index.html``) for producing
and revising the human ground truth that :mod:`rampnet.validation` scores. A reviewer
judges every model detection (correct / false positive / duplicate / unsure) *and*
scans the whole panorama for ramps the model missed, exporting a ``verdicts.json`` that
``scripts/score_validation.py`` turns into precision/recall.

This is the RampNet home of the tool that used to live in the deployment repo
(sidewalk-auto-labeler) and re-download imagery through its ``sources/``. Here it
instead consumes the **benchmark bundle** (issue #21): ``benchmark/<city>/`` with
``panos/<id>.<ext>`` (native-res images), ``records.jsonl`` (Stage-1 detections + pano
metadata), and an optional ``verdicts.json`` to prefill for revision. Nothing is
fetched from the network — the bundle is the single source of pixels and detections.

Fairness note (issue #26 #1): crops and the full-pano view are rendered at the model's
input resolution (4096x2048), **never** the native image. Mapillary panos are commonly
11000x5500; showing the reviewer more than the model saw would bias recall. The full
pano gets pan/zoom so a reviewer scanning for misses sees exactly the model's pixels.

The viewer's verdict schema and its "reviewed" gate mirror :func:`rampnet.validation.collect`
(``dets`` values ``true``/``false``/``"unsure"``/``"duplicate"``, ``missed`` marks, the
``no_missed`` false-negative confirmation) — keep the two in sync.

Usage:
    python scripts/gt_gallery.py benchmark/bend
    python scripts/gt_gallery.py benchmark/richmond --out /tmp/richmond_gallery
    python scripts/gt_gallery.py benchmark/bend --resample --sample 80   # re-sample a raw bundle

Then open the printed ``index.html`` (VS Code Live Server, ``python -m http.server``, or
just open the file), review, click "Export verdicts", and save the download back over
``benchmark/<city>/verdicts.json`` to re-score.
"""
import argparse
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

# The pipeline runs the model at a fixed 2048x4096 input, so the gallery renders at that
# same resolution — the fair comparison. Native panos (up to 16384x8192 here) are only
# ever downscaled to this; they are never shown at full size. See issue #26 #1.
MODEL_WIDTH, MODEL_HEIGHT = 4096, 2048
CROP_SIZE = 512        # per-detection close-up, taken at model resolution
TOP_N_BY_COUNT = 5     # the densest panos are always included (group "top")
RENDER_WORKERS = 8

# Native panos exceed PIL's decompression-bomb guard (16384x8192 = 134 MP). These are
# our own vetted benchmark images, so lift the cap rather than have the resize error out.
Image.MAX_IMAGE_PIXELS = None

# Two validation panos closer than this almost certainly show the same physical curb
# ramps, so the sampler keeps selected panos at least this far apart: a reviewer never
# judges the same ramps twice, and precision/recall aren't inflated by correlated
# duplicates. Well above the Mapillary thinning spacing (~5 m) since that's coverage.
DEFAULT_MIN_SPACING_M = 30


# --- Spatially de-clustered sampler (ported verbatim from the deployment tool) -------

def _haversine_m(lat1, lng1, lat2, lng2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _coords(record):
    """(lat, lng) for a record, or None if it carries no position."""
    p = record['pano']
    lat, lng = p.get('lat'), p.get('lng')
    return (lat, lng) if lat is not None and lng is not None else None


class _SpatialIndex:
    """Grid of accepted points for fast 'is anything within min_spacing?' checks.

    Cell size == min_spacing, so any point within range lies in one of the nine
    neighbouring cells — the acceptance test touches a handful of points, not the
    whole accepted set, so selection stays cheap on city-sized candidate pools.
    """

    def __init__(self, min_spacing):
        self.s = max(min_spacing, 1e-9)
        self.cells = {}

    def _key(self, lat, lng):
        clat = self.s / 111320.0
        clng = self.s / (111320.0 * max(0.01, math.cos(math.radians(lat))))
        return (math.floor(lat / clat), math.floor(lng / clng))

    def far_enough(self, lat, lng):
        kx, ky = self._key(lat, lng)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for plat, plng in self.cells.get((kx + dx, ky + dy), ()):
                    if _haversine_m(lat, lng, plat, plng) < self.s:
                        return False
        return True

    def add(self, lat, lng):
        self.cells.setdefault(self._key(lat, lng), []).append((lat, lng))


def _spread(candidates, k, index, min_spacing):
    """Greedily take up to k candidates (in the given order) that sit at least
    min_spacing metres from every already-accepted point in the shared `index`.

    Records without coordinates are accepted without constraint (graceful degradation —
    a coordinate-less run can't be de-clustered, but must still render). Returns
    (picked, rejected_for_spacing)."""
    picked, rejected = [], 0
    for rec in candidates:
        if len(picked) >= k:
            break
        c = _coords(rec)
        if c is None:
            picked.append(rec)
        elif min_spacing <= 0 or index.far_enough(*c):
            picked.append(rec)
            index.add(*c)
        else:
            rejected += 1
    return picked, rejected


def choose_panos(records, sample, empty_sample, seed, min_spacing=DEFAULT_MIN_SPACING_M):
    """Return [(record, group)] for a spatially de-clustered validation sample.

    Three strata, all held at least `min_spacing` metres apart — *across* strata as well
    as within — so a reviewer never judges the same physical ramps twice and the
    precision/recall estimate isn't inflated by correlated duplicates:
      - 'top':    the densest *distinct* intersections (up to TOP_N_BY_COUNT) — a
                  dense-scene stress test, excluded from unbiased scoring.
      - 'random': a spatially spread sample of panos with detections.
      - 'empty':  a spatially spread sample of zero-detection panos (for recall).

    Deterministic given `seed`. `min_spacing=0` disables spacing (pure random). Records
    without lat/lng are included unconstrained. City-agnostic by design.
    """
    rng = random.Random(seed)
    with_det = [r for r in records if r['detections']]
    without_det = [r for r in records if not r['detections']]
    index = _SpatialIndex(min_spacing)

    # top: densest first, but only one pano per distinct location.
    by_density = sorted(with_det, key=lambda r: len(r['detections']), reverse=True)
    top, _ = _spread(by_density, TOP_N_BY_COUNT, index, min_spacing)
    top_ids = {r['pano']['panorama_id'] for r in top}

    # random: spread across the remaining detection panos.
    rest = [r for r in with_det if r['pano']['panorama_id'] not in top_ids]
    rng.shuffle(rest)
    n_random = max(0, sample - len(top))
    random_sel, rej_r = _spread(rest, n_random, index, min_spacing)

    # empty: spread across the zero-detection panos.
    empties = list(without_det)
    rng.shuffle(empties)
    empty_sel, rej_e = _spread(empties, empty_sample, index, min_spacing)

    # No silent caps: if spacing (not scarcity) kept us short of a target, say so.
    if min_spacing > 0 and rej_r and len(random_sel) < n_random:
        print(f"  spatial sampling: kept {len(random_sel)}/{n_random} detection panos "
              f"at >= {min_spacing} m spacing (area too dense for more).")
    if min_spacing > 0 and rej_e and len(empty_sel) < empty_sample:
        print(f"  spatial sampling: kept {len(empty_sel)}/{empty_sample} empty panos "
              f"at >= {min_spacing} m spacing.")

    return ([(r, 'top') for r in top]
            + [(r, 'random') for r in random_sel]
            + [(r, 'empty') for r in empty_sel])


# --- Bundle I/O + rendering ----------------------------------------------------------

def load_bundle(bundle_dir):
    """Read a benchmark bundle: records, the panos dir, and any existing verdicts.

    Returns (records, panos_dir, verdicts_panos, run_key, run_name). verdicts_panos is
    the ``panos`` map from an existing verdicts.json (for prefill/revision) or {}.
    """
    d = Path(bundle_dir)
    records_path, panos_dir = d / "records.jsonl", d / "panos"
    if not records_path.exists():
        sys.exit(f"Bundle must contain records.jsonl: {records_path}")
    if not panos_dir.is_dir():
        sys.exit(f"Bundle must contain a panos/ directory (native-res images): {panos_dir}")

    with open(records_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    verdicts_panos, run_key, run_name = {}, d.name, d.name
    verdicts_path = d / "verdicts.json"
    if verdicts_path.exists():
        vj = json.load(open(verdicts_path, encoding="utf-8"))
        verdicts_panos = vj.get("panos", {})
        run_key = vj.get("run_key", run_key)
        run_name = vj.get("run_name", run_name)
    return records, panos_dir, verdicts_panos, run_key, run_name


def _find_pano_image(panos_dir, pid):
    """The on-disk image for a pano id (any extension), or None."""
    for p in sorted(panos_dir.glob(f"{pid}.*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            return p
    return None


def _crop_boxes(record):
    """Per-detection (crop-box, geometry-dict) at model resolution. Geometry is a pure
    function of the normalized detections and the fixed model size (no pixels needed),
    so it's shared by image rendering and the --html-only path.
    """
    pid, sx, sy = record['pano']['panorama_id'], MODEL_WIDTH, MODEL_HEIGHT
    out = []
    for i, det in enumerate(record['detections']):
        px, py = det['x_normalized'] * sx, det['y_normalized'] * sy
        half = CROP_SIZE // 2
        left = int(min(max(px - half, 0), sx - CROP_SIZE))
        top = int(min(max(py - half, 0), sy - CROP_SIZE))
        geom = {'img': f"{pid}_det{i}.jpg", 'conf': round(det['confidence'], 4),
                # detection center: normalized in the pano, and in the crop
                'x': round(det['x_normalized'], 5),
                'y': round(det['y_normalized'], 5),
                'cx': round((px - left) / CROP_SIZE, 4),
                'cy': round((py - top) / CROP_SIZE, 4)}
        out.append(((left, top, left + CROP_SIZE, top + CROP_SIZE), geom))
    return out


def entry_meta(record, group):
    """The viewer entry (pano metadata + crop geometry) for one record, without any I/O."""
    pano = record['pano']
    return {
        'pid': pano['panorama_id'],
        'source': pano.get('source', ''),
        'date': str(pano.get('capture_date', '')),
        'group': group,
        'full': f"{pano['panorama_id']}_full.jpg",
        'crops': [g for _, g in _crop_boxes(record)],
    }


def render_pano(record, group, images_dir, panos_dir):
    """Downscale one native pano to model resolution, save the clean full image and
    per-detection crops, and return the viewer entry (see :func:`entry_meta`).

    Images are saved clean (no burned-in markers) so the viewer can recolor detections
    live as verdicts change. Crops and the full image are taken at MODEL_WIDTH x
    MODEL_HEIGHT — the model's input size, never the native resolution (#26 #1).
    """
    pid = record['pano']['panorama_id']
    src = _find_pano_image(panos_dir, pid)
    if src is None:
        raise FileNotFoundError(f"no image in {panos_dir} for {pid}")

    # Downscale native -> model resolution once; all geometry is relative to that.
    img = Image.open(src).convert('RGB')
    if img.size != (MODEL_WIDTH, MODEL_HEIGHT):
        img = img.resize((MODEL_WIDTH, MODEL_HEIGHT), Image.BILINEAR)

    for box, geom in _crop_boxes(record):
        img.crop(box).save(images_dir / geom['img'], quality=85)
    img.save(images_dir / f"{pid}_full.jpg", quality=82)
    return entry_meta(record, group)


def initial_verdicts(verdicts_panos):
    """Convert a bundle verdicts.json ``panos`` map to the viewer's localStorage schema,
    so an existing bundle prefills for revision (e.g. to add the duplicate marks #26 #5
    calls out on Richmond). Bundle uses ``no_missed``; the viewer uses ``noMissed`` +
    ``seen``. dets values (incl. ``"duplicate"``) carry over unchanged.
    """
    out = {}
    for pid, e in verdicts_panos.items():
        out[pid] = {'dets': e.get('dets', []), 'missed': e.get('missed', []),
                    'noMissed': bool(e.get('no_missed', False)), 'seen': True}
    return out


HTML_TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>RampNet ground-truth labeler</title>
<style>
  :root{--ok:#1a9c3e;--bad:#d23;--dup:#8338ec;--unsure:#ff9f1c;--missed:#e534eb;--todo:#ffd400}
  body{font-family:sans-serif;margin:16px auto;max-width:1650px;background:#fafafa;color:#222}
  a{color:#06c}
  .bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px}
  .bar button,.bar select{font-size:15px;padding:6px 13px;cursor:pointer}
  .meta{color:#666;font-size:13px}
  .badge{font-size:12px;padding:2px 8px;border-radius:10px;background:#eee;color:#555}
  #intro{background:#fff6e6;border:1px solid var(--unsure);border-radius:8px;padding:10px 14px;
         margin:0 0 12px;font-size:14px;line-height:1.45}
  #intro b{color:#a35a00}
  #legend{display:flex;gap:16px;flex-wrap:wrap;align-items:center;background:#fff;border:1px solid #e2e2e2;
          border-radius:8px;padding:7px 12px;margin:0 0 12px;font-size:13px;position:sticky;top:0;z-index:5}
  #legend .k{display:inline-flex;align-items:center;gap:6px}
  #legend .sw{width:16px;height:16px;border-radius:50%;border:4px solid #000;box-sizing:border-box}
  /* Swatch modifiers are namespaced (k-*) so they never collide with the functional
     overlay/crop classes (e.g. a bare `.missed` would drag the swatch out of flow), and
     scoped under #legend so their colour out-specifies the `#legend .sw` base. */
  #legend .sw.k-todo{border-color:var(--todo)}
  #legend .sw.k-ok{border-color:var(--ok)}
  #legend .sw.k-bad{border-color:var(--bad)}
  #legend .sw.k-dup{border-color:var(--dup)}
  #legend .sw.k-unsure{border-color:var(--unsure)}
  #legend .sw.k-missed{border-color:var(--missed);border-style:dashed}
  #rev{font-weight:bold;padding:3px 10px;border-radius:12px}
  #rev.done{background:var(--ok);color:#fff}
  #rev.todo{background:#fde2e2;color:var(--bad);border:1px solid var(--bad)}
  #panoview{position:relative;width:100%;aspect-ratio:2/1;overflow:hidden;background:#111;
            border-radius:6px;cursor:crosshair;touch-action:none}
  #zoompill{position:absolute;top:8px;right:8px;background:rgba(0,0,0,.72);color:#fff;
            font:600 12px/1 sans-serif;padding:4px 9px;border-radius:12px;pointer-events:none;
            z-index:6;display:none}
  /* Transient label that names the verdict right after a click, so the state change
     is legible without hunting for the crop's caption. */
  #vflash{position:fixed;transform:translate(-50%,-100%);background:#000;color:#fff;
          font:bold 13px/1 sans-serif;padding:4px 10px;border-radius:12px;white-space:nowrap;
          pointer-events:none;opacity:0;z-index:50;box-shadow:0 1px 6px rgba(0,0,0,.5)}
  #vflash.show{animation:vflash 1.1s ease forwards}
  @keyframes vflash{0%{opacity:0}12%{opacity:1}70%{opacity:1}100%{opacity:0}}
  #panolayer{position:absolute;inset:0;transform-origin:0 0;will-change:transform}
  #panolayer img{width:100%;height:100%;display:block;user-select:none;-webkit-user-drag:none}
  .det{position:absolute;width:34px;height:34px;transform:translate(-50%,-50%) scale(var(--iz,1));
       border-radius:50%;border:4px solid var(--todo);
       box-shadow:0 0 0 3px rgba(255,255,255,.85),0 0 6px #000;cursor:pointer}
  .det.ok{border-color:var(--ok)}
  .det.bad{border-color:var(--bad)}
  .det.dup{border-color:var(--dup)}
  .det.unsure{border-color:var(--unsure)}
  .det .num{position:absolute;top:-21px;left:50%;transform:translateX(-50%);
            font:bold 13px/1 sans-serif;color:#fff;text-shadow:0 0 3px #000,0 0 3px #000}
  .missed{position:absolute;width:36px;height:36px;transform:translate(-50%,-50%) scale(var(--iz,1));
          border:4px dashed var(--missed);border-radius:50%;box-shadow:0 0 4px #000;
          cursor:pointer;background:rgba(229,52,235,.12)}
  .missed.unsure{border-color:var(--unsure);background:rgba(255,159,28,.14)}
  #zoombar{display:flex;gap:6px;align-items:center;margin:6px 0 0}
  #zoombar button{font-size:14px;padding:3px 10px;cursor:pointer}
  #fnbar{margin:10px 0 0;padding:10px 14px;border-radius:8px;display:flex;gap:12px;
         align-items:center;flex-wrap:wrap;border:2px solid}
  #fnbar.pending{background:#fde8e8;border-color:var(--bad)}
  #fnbar.done{background:#e9f7ee;border-color:var(--ok)}
  #fnbar button{font-size:15px;padding:6px 14px;cursor:pointer}
  #nomiss.active{background:var(--ok);border:1px solid var(--ok);color:#fff}
  h3.section{margin:16px 0 6px;font-size:15px}
  .crops{display:flex;flex-wrap:wrap;gap:10px;line-height:0}
  .crops figure{margin:0;text-align:center;font-size:13px;line-height:1.3;cursor:pointer}
  .cropwrap{position:relative;display:inline-block;line-height:0;border:5px solid #bbb;border-radius:4px}
  .crops img{width:256px;height:256px;border-radius:2px;display:block}
  .crops .ok .cropwrap{border-color:var(--ok)}
  .crops .bad .cropwrap{border-color:var(--bad)}
  .crops .dup .cropwrap{border-color:var(--dup)}
  .crops .unsure .cropwrap{border-color:var(--unsure)}
  .cropring{position:absolute;width:15.6%;height:15.6%;transform:translate(-50%,-50%);
            border:3px solid var(--todo);border-radius:50%;
            box-shadow:0 0 0 2px rgba(255,255,255,.85);pointer-events:none}
  .ok .cropring{border-color:var(--ok)}.bad .cropring{border-color:var(--bad)}
  .dup .cropring{border-color:var(--dup)}.unsure .cropring{border-color:var(--unsure)}
  .crops .verdict{font-weight:bold}
  .ok .verdict{color:var(--ok)}.bad .verdict{color:var(--bad)}
  .dup .verdict{color:var(--dup)}.unsure .verdict{color:var(--unsure)}
  #missedpanel figure{border:4px solid var(--missed);border-radius:4px;padding:0;background:#fff}
  #missedpanel figure.unsure{border-color:var(--unsure)}
  #missedpanel canvas{width:180px;height:180px;display:block;border-radius:2px}
  #missedpanel figcaption{font-size:12px;padding:3px}
  #missedpanel .mbtn{font-size:12px;padding:2px 6px;margin:0 2px 3px;cursor:pointer}
  #missedempty{color:#888;font-size:13px}
  #help{font-size:13px;color:#666;margin-top:16px;line-height:1.5}
  kbd{background:#eee;border:1px solid #ccc;border-radius:3px;padding:0 4px;font-size:12px}
</style>

<div id="intro">
  <b>This is a two-sided check.</b> For every pano you must do <b>both</b>:
  (1) judge each model detection — correct, false positive, duplicate (a 2nd hit on one
  ramp), or unsure; and (2) <b>scan the whole panorama</b> for curb ramps the model
  <b>missed</b>, then mark them or confirm there are none. Reviewers consistently
  under-count misses, so a pano is not <b>done</b> until every detection is judged
  <b>and</b> the missed-ramp pass is confirmed. Use pan/zoom to inspect at the model's
  full resolution.
</div>

<div id="legend">
  <span class="meta">Legend:</span>
  <span class="k"><span class="sw k-todo"></span>unjudged</span>
  <span class="k"><span class="sw k-ok"></span>correct</span>
  <span class="k"><span class="sw k-bad"></span>false positive</span>
  <span class="k"><span class="sw k-dup"></span>duplicate</span>
  <span class="k"><span class="sw k-unsure"></span>unsure</span>
  <span class="k"><span class="sw k-missed"></span>reviewer-marked missed</span>
</div>

<div class="bar">
  <button id="prev">&#8592; Prev</button>
  <button id="next">Next &#8594;</button>
  <button id="nexttodo">Next unreviewed &#8608;</button>
  <select id="filter">
    <option value="all">All panos</option>
    <option value="det">With detections</option>
    <option value="empty">Zero detections</option>
    <option value="todo">Unreviewed</option>
  </select>
  <span id="pos" class="meta"></span>
  <span id="progress" class="meta"></span>
  <span style="flex:1"></span>
  <button id="export">Export verdicts</button>
</div>

<h2 id="title" style="margin:6px 0;font-size:17px"></h2>
<div id="panoview"><div id="panolayer"><img id="panoimg" alt=""></div><div id="zoompill"></div></div>
<div id="zoombar">
  <button id="zoomout">&#8722;</button><button id="zoomin">+</button>
  <button id="zoomreset">Reset</button>
  <span class="meta">scroll to zoom, drag to pan, click empty pano to mark a missed ramp
    &mdash; this pano <b>is</b> the model's input (4096&times;2048); the pill shows how much of that
    resolution you're seeing. Zoom reaches model 1:1 but never beyond &mdash; magnifying shows the
    same pixels bigger, never more than the model saw.</span>
</div>

<div id="fnbar">
  <button id="nomiss">No missed ramps (m)</button>
  <span id="fnstate" class="meta"></span>
</div>

<h3 class="section">Detection crops <span class="meta">&mdash; click or press 1&ndash;9 to cycle a verdict</span></h3>
<div class="crops" id="crops"></div>

<h3 class="section">Reviewer-marked missed ramps</h3>
<div class="crops" id="missedpanel"></div>
<div id="missedempty">None marked yet. Scan the pano above and click any curb ramp the model missed.</div>

<p id="help">
  <kbd>&#8592;</kbd>/<kbd>&#8594;</kbd> pano &nbsp;&middot;&nbsp;
  <kbd>1</kbd>&#8211;<kbd>9</kbd> cycle a crop's verdict
  (unjudged &#8594; <span style="color:var(--ok)">correct</span> &#8594;
  <span style="color:var(--bad)">false positive</span> &#8594;
  <span style="color:var(--dup)">duplicate</span> &#8594;
  <span style="color:var(--unsure)">unsure</span>) &nbsp;&middot;&nbsp;
  click the pano to mark a <span style="color:var(--missed)">missed</span> ramp (click the marker to
  make it <span style="color:var(--unsure)">unsure</span>, again to remove), or press <kbd>m</kbd> if
  there are none &nbsp;&middot;&nbsp; <b>duplicate</b> = a redundant hit on a ramp already counted
  (scored as a false positive by default) &nbsp;&middot;&nbsp; <b>unsure</b> abstains (dropped from
  precision &amp; recall) &nbsp;&middot;&nbsp; verdicts autosave locally; Export downloads
  <span id="vname"></span> &mdash; save it back over <code>benchmark/&lt;city&gt;/verdicts.json</code>, then
  <code>python scripts/score_validation.py benchmark/&lt;city&gt;</code>
</p>

<script>
const ENTRIES = __ENTRIES__;
const MODEL_W = __MODEL_W__;   // the served pano is exactly this wide (the model's input res)
const RUN_KEY = __RUN_KEY__;
const RUN_NAME = __RUN_NAME__;
const SOURCE = __SOURCE__;
const INITIAL = __INITIAL__;   // prefill from the bundle's verdicts.json (schema below)
const STORE = 'verdicts:' + RUN_KEY;

let verdicts = JSON.parse(localStorage.getItem(STORE) || '{}');
// verdicts[pid] = {dets: [null|true|false|'unsure'|'duplicate', ...],
//                  missed: [{x, y, unsure?: bool}, ...],
//                  noMissed: bool (reviewer confirmed no missed ramps), seen: bool}
// 'unsure' (crop) and unsure:true (missed) mean "can't tell" — the scorer abstains on
// them. 'duplicate' is a redundant hit on a ramp already counted (scored as an FP by
// default). Kept in sync with rampnet.validation.collect().
// Prefill from the bundle without clobbering any local progress already in this browser.
for (const pid in INITIAL) if (!(pid in verdicts)) verdicts[pid] = INITIAL[pid];

function save() { localStorage.setItem(STORE, JSON.stringify(verdicts)); }
function v(pid, n) {
  if (!verdicts[pid]) verdicts[pid] = {dets: Array(n).fill(null), missed: [], noMissed: false, seen: false};
  const s = verdicts[pid];
  // A re-render with a different detection count (new model/threshold) invalidates the
  // stored crop verdicts; reset them (missed marks + no-missed are pano-level, kept).
  if (s.dets.length !== n) s.dets = Array(n).fill(null);
  return s;
}

let filterMode = 'all', view = ENTRIES.slice(), idx = 0;

// vcls/fnChecked/reviewed define what "reviewed" means; rampnet.validation.collect()
// applies the same gate to the exported verdicts — keep the two in sync.
function vcls(d) { return d === true ? 'ok' : d === false ? 'bad'
                       : d === 'duplicate' ? 'dup' : d === 'unsure' ? 'unsure' : ''; }
function vlabel(d) { return d === true ? 'correct' : d === false ? 'FALSE POSITIVE'
                       : d === 'duplicate' ? 'duplicate' : d === 'unsure' ? 'unsure' : 'unjudged'; }
function fnChecked(s) { return s.missed.length > 0 || !!s.noMissed; }
function reviewed(e) {
  const s = verdicts[e.pid];
  if (!s || !s.seen || !fnChecked(s)) return false;
  if (s.dets.length !== e.crops.length) return false; // stale entry, see v()
  return s.dets.every(d => d !== null);
}

function applyFilter() {
  const cur = view[idx] && view[idx].pid;
  view = ENTRIES.filter(e =>
    filterMode === 'det' ? e.crops.length > 0 :
    filterMode === 'empty' ? e.crops.length === 0 :
    filterMode === 'todo' ? !reviewed(e) : true);
  if (!view.length) { idx = 0; render(); return; }
  const keep = view.findIndex(e => e.pid === cur);
  idx = keep >= 0 ? keep : 0;
  render();
}

// --- Pan/zoom (capped at model resolution: the served image is already 4096-wide, so
// zooming never reveals more than the model saw; issue #26 #1). -----------------------
const view_el = document.getElementById('panoview');
const layer = document.getElementById('panolayer');
const panoImg = document.getElementById('panoimg');
let zoom = 1, panX = 0, panY = 0;
const MAXZOOM = 8;

function vp() { const r = view_el.getBoundingClientRect(); return {w: r.width, h: r.height}; }
function clampPan() {
  const {w, h} = vp();
  panX = Math.min(0, Math.max(w - w * zoom, panX));
  panY = Math.min(0, Math.max(h - h * zoom, panY));
}
function applyTransform() {
  clampPan();
  layer.style.transform = 'translate(' + panX + 'px,' + panY + 'px) scale(' + zoom + ')';
  // Counter-scale overlays so markers stay a constant, clickable size at any zoom.
  document.querySelectorAll('.det, .missed').forEach(m => m.style.setProperty('--iz', 1 / zoom));
  updatePill();
}

// The pill answers "am I seeing what the model saw?" — NOT a zoom factor. The served
// pano is exactly the model's input width (MODEL_W), so `frac` is the share of the
// model's horizontal resolution actually resolved on screen right now (device pixels
// included). It tops out at 1:1: you can reach model parity but never exceed it — past
// that you're only magnifying the same model pixels, seeing them bigger, not seeing more.
function updatePill() {
  const zp = document.getElementById('zoompill');
  const dpr = window.devicePixelRatio || 1;
  const frac = vp().w * zoom * dpr / MODEL_W;
  zp.style.display = 'block';
  zp.textContent = frac < 0.995 ? Math.round(frac * 100) + '% of model res'
    : 'model res 1:1' + (frac > 1.05 ? ' · ' + frac.toFixed(1) + '× magnified' : '');
}
window.addEventListener('resize', () => { if (view.length) applyTransform(); });
function resetZoom() { zoom = 1; panX = 0; panY = 0; applyTransform(); }
function zoomAt(cx, cy, factor) {
  const oldZoom = zoom;
  zoom = Math.min(MAXZOOM, Math.max(1, zoom * factor));
  panX = cx - (cx - panX) * (zoom / oldZoom);
  panY = cy - (cy - panY) * (zoom / oldZoom);
  applyTransform();
}
view_el.addEventListener('wheel', ev => {
  ev.preventDefault();
  const r = view_el.getBoundingClientRect();
  zoomAt(ev.clientX - r.left, ev.clientY - r.top, ev.deltaY < 0 ? 1.2 : 1 / 1.2);
}, {passive: false});
document.getElementById('zoomin').onclick = () => { const {w, h} = vp(); zoomAt(w / 2, h / 2, 1.4); };
document.getElementById('zoomout').onclick = () => { const {w, h} = vp(); zoomAt(w / 2, h / 2, 1 / 1.4); };
document.getElementById('zoomreset').onclick = resetZoom;

// Drag to pan; a press that barely moves is a click-to-mark instead.
let down = null;
view_el.addEventListener('pointerdown', ev => {
  if (ev.target.closest('.det, .missed')) return;   // let marker handlers run
  down = {x: ev.clientX, y: ev.clientY, panX, panY, moved: false};
  view_el.setPointerCapture(ev.pointerId);
});
view_el.addEventListener('pointermove', ev => {
  if (!down) return;
  const dx = ev.clientX - down.x, dy = ev.clientY - down.y;
  if (Math.abs(dx) > 4 || Math.abs(dy) > 4) down.moved = true;
  if (down.moved) { panX = down.panX + dx; panY = down.panY + dy; applyTransform(); }
});
view_el.addEventListener('pointerup', ev => {
  if (!down) return;
  const wasDrag = down.moved;
  down = null;
  if (wasDrag || !view.length) return;
  // Click on empty pano -> mark a missed ramp. Map screen px back through the transform.
  const r = view_el.getBoundingClientRect();
  const nx = ((ev.clientX - r.left) - panX) / (r.width * zoom);
  const ny = ((ev.clientY - r.top) - panY) / (r.height * zoom);
  if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return;
  const e = view[idx], s = v(e.pid, e.crops.length);
  s.missed.push({x: nx, y: ny});
  s.noMissed = false;
  save(); render();
});

// --- Rendering -----------------------------------------------------------------------
function render() {
  const done = ENTRIES.filter(reviewed).length;
  document.getElementById('progress').textContent = done + '/' + ENTRIES.length + ' fully reviewed';
  document.querySelectorAll('.missed, .det').forEach(m => m.remove());
  if (!view.length) {
    document.getElementById('title').textContent = 'No panos match this filter';
    panoImg.removeAttribute('src');
    document.getElementById('crops').innerHTML = '';
    document.getElementById('fnbar').style.display = 'none';
    document.getElementById('pos').textContent = '';
    renderMissed();
    return;
  }
  document.getElementById('fnbar').style.display = '';
  const e = view[idx];
  const s = v(e.pid, e.crops.length);
  s.seen = true; save();
  resetZoom();

  document.getElementById('pos').textContent = (idx + 1) + ' / ' + view.length;
  const viewerUrl = e.source === 'mapillary'
    ? 'https://www.mapillary.com/app/?pKey=' + e.pid + '&focus=photo'
    : 'https://www.google.com/maps/@?api=1&map_action=pano&pano=' + e.pid;
  const isDone = reviewed(e);
  document.getElementById('title').innerHTML =
    '<a href="' + viewerUrl + '" target="_blank">' + e.pid + '</a> ' +
    '<span class="meta">captured ' + e.date + ' &mdash; ' + e.crops.length + ' detection(s)</span> ' +
    '<span class="badge">' + e.group + '</span> ' +
    '<span id="rev" class="' + (isDone ? 'done' : 'todo') + '">' +
      (isDone ? '✓ REVIEWED' : '● NEEDS REVIEW') + '</span>';
  panoImg.src = 'images/' + e.full;

  // Detection circles overlaid on the clean pano image.
  e.crops.forEach((c, i) => {
    const d = document.createElement('div');
    d.className = ('det ' + vcls(s.dets[i])).trim();
    d.style.left = (c.x * 100) + '%';
    d.style.top = (c.y * 100) + '%';
    d.innerHTML = '<span class="num">' + (i + 1) + '</span>';
    d.title = 'detection ' + (i + 1) + ' — click to cycle verdict';
    d.onclick = ev => { ev.stopPropagation(); cycle(i, 'pano'); };
    layer.appendChild(d);
  });
  s.missed.forEach((m, i) => {
    const d = document.createElement('div');
    d.className = 'missed' + (m.unsure ? ' unsure' : '');
    d.style.left = (m.x * 100) + '%';
    d.style.top = (m.y * 100) + '%';
    d.title = m.unsure ? 'unsure missed ramp — click to remove'
                       : 'missed ramp — click to mark unsure, again to remove';
    d.onclick = ev => { ev.stopPropagation();
      if (!m.unsure) m.unsure = true; else s.missed.splice(i, 1);
      save(); render();
    };
    layer.appendChild(d);
  });
  applyTransform();

  // Missed-ramp confirmation bar — structural: it stays "pending" (red) until the
  // reviewer marks a miss or affirms none, so the FN pass can't be silently skipped.
  const affirmed = !!s.noMissed && !s.missed.length;
  const nSure = s.missed.filter(m => !m.unsure).length;
  const nUnsure = s.missed.filter(m => m.unsure).length;
  const fnbar = document.getElementById('fnbar');
  const nm = document.getElementById('nomiss');
  const fnDone = fnChecked(s);
  fnbar.className = fnDone ? 'done' : 'pending';
  nm.disabled = s.missed.length > 0;
  nm.classList.toggle('active', affirmed);
  nm.textContent = affirmed ? '✓ No missed ramps' : 'No missed ramps (m)';
  document.getElementById('fnstate').textContent =
    s.missed.length ? (nSure + ' missed ramp(s) marked' + (nUnsure ? ', ' + nUnsure + ' unsure' : '')) :
    affirmed ? 'missed-ramp check confirmed' :
    'REQUIRED: scan the whole panorama for ramps the model missed, then mark them or confirm none';

  // Detection crops with verdict state.
  const crops = document.getElementById('crops');
  crops.innerHTML = '';
  e.crops.forEach((c, i) => {
    const fig = document.createElement('figure');
    fig.className = vcls(s.dets[i]);
    fig.innerHTML = '<span class="cropwrap"><img src="images/' + c.img + '" loading="lazy">' +
      '<span class="cropring" style="left:' + (c.cx * 100) + '%;top:' + (c.cy * 100) + '%"></span></span>' +
      '<figcaption>[' + (i + 1) + '] conf ' + c.conf.toFixed(2) +
      ' &mdash; <span class="verdict">' + vlabel(s.dets[i]) + '</span></figcaption>';
    fig.onclick = () => cycle(i, 'crop');
    crops.appendChild(fig);
  });
  renderMissed();
}

// Missed-ramp panel: crop each marked location client-side from the full (model-res)
// pano so added false-negatives are reviewable/removable at a glance (issue #26 #2).
function renderMissed() {
  const panel = document.getElementById('missedpanel');
  const empty = document.getElementById('missedempty');
  panel.innerHTML = '';
  const e = view[idx];
  const s = e && verdicts[e.pid];
  const marks = (s && s.missed) || [];
  empty.style.display = (view.length && !marks.length) ? '' : 'none';
  if (!marks.length) return;
  const NW = panoImg.naturalWidth, NH = panoImg.naturalHeight;
  marks.forEach((m, i) => {
    const fig = document.createElement('figure');
    if (m.unsure) fig.className = 'unsure';
    const cv = document.createElement('canvas');
    cv.width = 180; cv.height = 180;
    if (NW && panoImg.complete) {
      const C = Math.min(512, NW, NH);
      let sx = Math.round(m.x * NW - C / 2), sy = Math.round(m.y * NH - C / 2);
      sx = Math.max(0, Math.min(sx, NW - C)); sy = Math.max(0, Math.min(sy, NH - C));
      try { cv.getContext('2d').drawImage(panoImg, sx, sy, C, C, 0, 0, 180, 180); } catch (err) {}
    }
    fig.appendChild(cv);
    const cap = document.createElement('figcaption');
    cap.innerHTML = 'missed' + (m.unsure ? ' <b style="color:var(--unsure)">(unsure)</b>' : '') + '<br>';
    const bU = document.createElement('button');
    bU.className = 'mbtn'; bU.textContent = m.unsure ? 'mark sure' : 'mark unsure';
    bU.onclick = () => { m.unsure = !m.unsure; save(); render(); };
    const bX = document.createElement('button');
    bX.className = 'mbtn'; bX.textContent = 'remove';
    bX.onclick = () => { s.missed.splice(i, 1); save(); render(); };
    cap.appendChild(bU); cap.appendChild(bX);
    fig.appendChild(cap);
    panel.appendChild(fig);
  });
}
// The full image may still be decoding when render() runs; fill the crops once it loads.
panoImg.addEventListener('load', renderMissed);

// Transient "correct / false positive / ..." label next to whatever the reviewer just
// clicked, so the new state registers immediately (issue #26 feedback).
const VCOLOR = {ok: 'var(--ok)', bad: 'var(--bad)', dup: 'var(--dup)', unsure: 'var(--unsure)'};
let flashEl = null;
function flashVerdict(anchor, d) {
  if (!anchor) return;
  if (!flashEl) { flashEl = document.createElement('div'); flashEl.id = 'vflash'; document.body.appendChild(flashEl); }
  const r = anchor.getBoundingClientRect();
  flashEl.textContent = vlabel(d);
  flashEl.style.color = VCOLOR[vcls(d)] || '#fff';
  flashEl.style.left = (r.left + r.width / 2) + 'px';
  flashEl.style.top = (r.top - 6) + 'px';
  flashEl.classList.remove('show'); void flashEl.offsetWidth; flashEl.classList.add('show');  // restart anim
}

function cycle(i, origin) {
  const e = view[idx];
  if (!e || i >= e.crops.length) return;
  const s = v(e.pid, e.crops.length);
  s.dets[i] = s.dets[i] === null ? true
            : s.dets[i] === true ? false
            : s.dets[i] === false ? 'duplicate'
            : s.dets[i] === 'duplicate' ? 'unsure'
            : null;
  save(); render();
  // Anchor the flash to whatever was clicked (the crop, or the pano circle / keyboard).
  const anchor = origin === 'crop'
    ? document.getElementById('crops').children[i]
    : layer.querySelectorAll('.det')[i];
  flashVerdict(anchor, s.dets[i]);
}

// --- Controls ------------------------------------------------------------------------
document.getElementById('prev').onclick = () => { if (view.length) { idx = (idx - 1 + view.length) % view.length; render(); } };
document.getElementById('next').onclick = () => { if (view.length) { idx = (idx + 1) % view.length; render(); } };
document.getElementById('nexttodo').onclick = () => {
  if (!view.length) return;
  for (let k = 1; k <= view.length; k++) {
    const j = (idx + k) % view.length;
    if (!reviewed(view[j])) { idx = j; render(); return; }
  }
  alert('Every pano in this view is fully reviewed. Switch the filter to "All panos" to double-check, then Export.');
};
document.getElementById('filter').onchange = ev => { filterMode = ev.target.value; applyFilter(); };
document.getElementById('nomiss').onclick = () => {
  if (!view.length) return;
  const e = view[idx], s = v(e.pid, e.crops.length);
  if (s.missed.length) return;
  s.noMissed = !s.noMissed;
  save(); render();
};
document.addEventListener('keydown', ev => {
  // Never hijack browser chords (Ctrl+M mutes, Cmd+M minimizes, Ctrl+1..9 switch tabs).
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  if (ev.target.tagName === 'SELECT') return;
  if (ev.key === 'ArrowLeft') document.getElementById('prev').click();
  else if (ev.key === 'ArrowRight') document.getElementById('next').click();
  else if (ev.key === 'm' || ev.key === 'M') document.getElementById('nomiss').click();
  else if (ev.key >= '1' && ev.key <= '9') cycle(Number(ev.key) - 1);
});

document.getElementById('export').onclick = () => {
  // Structural FN check (issue #26 #6): surface incomplete panos instead of exporting
  // silently, so a skipped missed-ramp pass is visible rather than quietly biasing recall.
  let partial = 0, unconfirmed = 0;
  for (const e of ENTRIES) {
    const s = verdicts[e.pid];
    if (!s || !s.seen) continue;
    if (s.dets.length !== e.crops.length || s.dets.some(d => d === null)) partial++;
    else if (!fnChecked(s)) unconfirmed++;
  }
  if (partial || unconfirmed) {
    const msg = 'Some engaged panos are not fully reviewed:\n' +
      (partial ? '  • ' + partial + ' with detections still unjudged\n' : '') +
      (unconfirmed ? '  • ' + unconfirmed + ' judged but the missed-ramp check was never confirmed\n' : '') +
      '\nThose are excluded from recall by the scorer. Export anyway?\n' +
      '("Next unreviewed" jumps straight to them.)';
    if (!confirm(msg)) return;
  }
  const out = {run_key: RUN_KEY, run_name: RUN_NAME, source: SOURCE,
               exported_at: new Date().toISOString(), panos: {}};
  for (const e of ENTRIES) {
    const s = verdicts[e.pid];
    if (!s || !s.seen) continue;
    out.panos[e.pid] = {group: e.group, dets: s.dets, missed: s.missed, no_missed: !!s.noMissed};
  }
  const blob = new Blob([JSON.stringify(out, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = RUN_NAME + '_verdicts.json';
  a.click();
};

document.getElementById('vname').textContent = RUN_NAME + '_verdicts.json';
save();  // persist any bundle prefill merged in above
render();
</script>
"""


def build_html(entries, initial, run_key, run_name, source):
    return (HTML_TEMPLATE
            .replace('__ENTRIES__', json.dumps(entries))
            .replace('__MODEL_W__', str(MODEL_WIDTH))
            .replace('__INITIAL__', json.dumps(initial))
            .replace('__RUN_KEY__', json.dumps(run_key))
            .replace('__RUN_NAME__', json.dumps(run_name))
            .replace('__SOURCE__', json.dumps(str(source))))


def main():
    parser = argparse.ArgumentParser(
        description="Build the RampNet ground-truth labeler over a benchmark bundle.")
    parser.add_argument("bundle", help="Benchmark bundle dir (e.g. benchmark/bend).")
    parser.add_argument("--out", type=Path,
                        help="Output gallery dir (default: <bundle>/gallery).")
    parser.add_argument("--resample", action="store_true",
                        help="Re-run the spatial sampler over records.jsonl instead of "
                             "rendering every record in the bundle (use on a raw, "
                             "un-sampled bundle).")
    parser.add_argument("--html-only", action="store_true",
                        help="Rebuild index.html from existing images (skip re-rendering "
                             "panos). For iterating on the viewer.")
    parser.add_argument("--sample", type=int, default=100,
                        help="With --resample: detection panos to include (default: %(default)s).")
    parser.add_argument("--empty-sample", type=int, default=10,
                        help="With --resample: zero-detection panos to include (default: %(default)s).")
    parser.add_argument("--seed", type=int, default=0,
                        help="With --resample: sampling seed (default: %(default)s).")
    parser.add_argument("--min-spacing", type=float, default=DEFAULT_MIN_SPACING_M,
                        help="With --resample: min metres between sampled panos "
                             "(default: %(default)s; 0 disables).")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    records, panos_dir, verdicts_panos, run_key, run_name = load_bundle(bundle)

    if args.resample:
        chosen = choose_panos(records, args.sample, args.empty_sample, args.seed, args.min_spacing)
    else:
        # The bundle's records.jsonl already *is* the validated sample; render all of it,
        # taking each pano's stratum from the existing verdicts (or 'random' if unlabeled).
        chosen = [(r, verdicts_panos.get(r['pano']['panorama_id'], {}).get('group', 'random'))
                  for r in records]
    print(f"{len(records)} records; rendering {len(chosen)} panos "
          f"({sum(1 for r, _ in chosen if r['detections'])} with detections).")

    out_dir = args.out or bundle / "gallery"
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if args.html_only:
        # Geometry is independent of the pixels, so entries rebuild without any I/O;
        # only panos whose full image is already on disk are included.
        entries = [entry_meta(r, g) for r, g in chosen
                   if (images_dir / f"{r['pano']['panorama_id']}_full.jpg").exists()]
        print(f"  --html-only: {len(entries)} panos with existing images.")
    else:
        entries, done = [], 0
        with ThreadPoolExecutor(max_workers=RENDER_WORKERS) as pool:
            futures = {pool.submit(render_pano, r, g, images_dir, panos_dir): r for r, g in chosen}
            for future in as_completed(futures):
                pid = futures[future]['pano']['panorama_id']
                done += 1
                try:
                    entries.append(future.result())
                except Exception as e:
                    print(f"  skipped {pid}: {e}")
                if done % 25 == 0 or done == len(futures):
                    print(f"  rendered {done}/{len(futures)}")

    entries.sort(key=lambda e: (-len(e['crops']), e['pid']))
    initial = initial_verdicts(verdicts_panos)
    index_path = out_dir / "index.html"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(build_html(entries, initial, run_key, run_name, bundle / "records.jsonl"))

    print(f"Gallery: {index_path}")
    if verdicts_panos:
        print(f"Prefilled {len(verdicts_panos)} panos from {bundle / 'verdicts.json'} for revision.")
    print(f"Open it, review, Export, then save the download over {bundle / 'verdicts.json'}")
    print(f"and re-score with: python scripts/score_validation.py {bundle}")


if __name__ == "__main__":
    main()
