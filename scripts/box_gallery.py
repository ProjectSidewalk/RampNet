"""Whole-apron box annotation: the second pass that turns adjudicated ramp points
into extent gold (issue #116).

#114 measured that the ``manual_labels/*.txt`` w/h are near-point marks, not object
extents — so the ecosystem has no curb-ramp extent gold at all. This tool produces it.
It is deliberately a **separate second-pass tool**, a sibling of ``gt_gallery.py``, not a
mode inside it: boxes are drawn around *adjudicated* ramps (TP det verdicts + sure missed
marks), which only exist after point review completes; the verdict protocol and its
model-resolution fairness rule (#26) stay untouched; and ``verdicts.json`` — sync-locked
to :func:`rampnet.validation.collect` — is never written, only read. Boxes live in a
separate ``boxes.json`` sidecar keyed by ``(pano_id, det:<idx> | missed:<idx>)``.

Unlike the point tool, crops here are cut at **the bundle's stored resolution**: extent
annotation is not a recall judgment, and tight edges need real pixels, so a bundle whose
``panos/`` holds a verified 1:1 native archive (richmond: 12288x6144 -> 3072 px crops) is
annotated at native scale. That is a property of the bundle, not of this tool — and it is
NOT true everywhere: ``benchmark/manual_gold`` records 4096x2048 for all 1,000 panos (the
model's input size), so its crops are 1024 px and the gold drawn there is model-resolution
gold. :func:`resolution_note` prints that distinction loudly at render time and it is
embedded in every export as ``crop_px_by_pano_dims``, because "tight at native zoom"
means something different at 1024 px than at 3072.

The viewer is **blind by construction** — it never renders a predicted crop window,
because the annotator already sees the point, so a window would leak exactly one thing: a
size prior, and size is the quantity the gold exists to measure. The algorithm-vs-gold
eyeball lives in #114's overlay gallery instead.

The box convention is explicit and versioned (``BOX_RULE`` below), shown in the viewer
and embedded in every export — so the new gold cannot reproduce the convention drift
#114 found in manual_labels. Boxes export **pano-normalized** ``{cx, cy, w, h}``
(resolution-independent; ``cx`` may sit near 0/1 with the box wrapping the
equirectangular seam, which the crop view stitches across — issue #43).

Usage:
    python scripts/box_gallery.py benchmark/richmond
    python scripts/box_gallery.py benchmark/manual_gold --from-manual-labels --sample-panos 50
    python scripts/box_gallery.py benchmark/richmond --fov 120   # wider crops, same boxes

Then open the printed ``index.html``, draw, click "Export boxes", and save the download
over ``benchmark/<city>/boxes.json``. Annotations autosave to localStorage keyed by the
bundle, survive a re-render at a different ``--fov`` (stored pano-normalized), and an
existing ``boxes.json`` prefills for revision — same round-trip contract as gt_gallery.
"""
import argparse
import hashlib
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt_gallery import _find_pano_image, load_bundle  # noqa: E402

Image.MAX_IMAGE_PIXELS = None  # our own vetted native panos exceed PIL's bomb guard

# The box convention, one short rule per line (the viewer renders them as bullets; the
# export joins them into box_rule.text). Version bumps on ANY wording change that could
# alter what an annotator draws; the scorer stratifies by it, and every export embeds it.
# v2 = v1 + the last two bullets (ruler-not-a-crop; oblique ramps), added 2026-08-14
# after the first annotation session. They clarify the same convention rather than
# changing it, so boxes drawn under v1 remain valid under v2.
BOX_RULE_VERSION = 2
BOX_RULE_LINES = [
    "One box per ramp: the whole constructed ramp surface — sloped apron + "
    "detectable-warning pad + side flares.",
    "Bottom edge: where the ramp meets the gutter or street. Top edge: where the "
    "slope levels into the sidewalk.",
    "Exclude road/gutter pavement, level sidewalk, and neighboring ramps.",
    "A corner apron that point review counted as two ramps gets two boxes, one per "
    "direction of travel — overlap is fine.",
    "Occluded is not absent: box the inferred full extent behind cars/pedestrians; "
    "truly undeterminable -> \"Can't determine extent\".",
    "Tight beats symmetric: every edge should touch the ramp's boundary at native zoom.",
    "The box is a measuring instrument, not the crop: road and surrounding context are "
    "what a crop rule adds AROUND your box and gets scored on — never include them.",
    "Oblique ramps (running toward the horizon): keep the box axis-aligned and touch "
    "the ramp's extremities — empty corners are correct, don't rotate-fit by eye.",
]
BOX_RULE = "\n".join(BOX_RULE_LINES)

# Crop field of view in degrees of pano longitude; side = width * fov/360, capped at the
# pano height. 90° comfortably contains the widest near-field apron (~62° at 2.5 m for a
# 3 m apron) with margin; boxes that touch a crop edge are auto-flagged in the viewer,
# and a re-render at --fov 120 keeps every annotation (boxes store pano-normalized).
DEFAULT_FOV_DEG = 90
RENDER_WORKERS = 8
JPEG_QUALITY = 85


# --- Item enumeration ----------------------------------------------------------------

def enumerate_items(verdicts_panos, records_by_pid):
    """[(item dict)] for every adjudicated ramp, plus human-readable warnings.

    Mirrors :func:`rampnet.validation.collect` on the what-is-a-ramp side: a ramp
    exists where a detection was judged ``True`` or the reviewer marked a sure miss.
    ``False`` isn't a ramp; ``'unsure'`` (det or missed) abstains; ``'duplicate'`` is a
    second hit on a ramp another item already covers, so boxing it would double-annotate
    one physical ramp. Panos whose det verdicts don't line up with the records are
    skipped with a warning — the same guard collect() applies, and so is the
    partially-judged skip: a pano with any ``None`` verdict is "unusable for either
    metric" there, and its missed marks in particular are untrustworthy because nobody
    finished scanning it. Boxing ramps collect() never counts would put extent gold and
    point metrics on different populations.

    Items carry ``seq`` (position within the pano) so display order inside a pano is
    stable regardless of the global shuffle.
    """
    items, warnings = [], []
    for pid in sorted(verdicts_panos):
        entry = verdicts_panos[pid]
        rec = records_by_pid.get(pid)
        if rec is None or len(rec['detections']) != len(entry.get('dets', [])):
            warnings.append(f"skipping {pid}: verdicts don't match records detections")
            continue
        if any(d is None for d in entry.get('dets', [])):
            warnings.append(f"skipping {pid}: partially judged (collect() drops it too)")
            continue
        seq = 0
        for i, (d, det) in enumerate(zip(entry.get('dets', []), rec['detections'])):
            if d is True:
                items.append({'pid': pid, 'key': f'det:{i}', 'seq': seq,
                              'x': det['x_normalized'], 'y': det['y_normalized'],
                              'conf': round(det['confidence'], 4)})
                seq += 1
        for i, m in enumerate(entry.get('missed', [])):
            if not m.get('unsure'):
                items.append({'pid': pid, 'key': f'missed:{i}', 'seq': seq,
                              'x': m['x'], 'y': m['y'], 'conf': None})
                seq += 1
    return items, warnings


def items_from_manual_labels(labels_dir, pids):
    """``gold:<i>`` items from YOLO label files — centers as prompts, w/h ignored (#114).

    The near-point marks are exactly what makes re-annotation cheap: each one says
    "there is a ramp here", and the box gets drawn around it under the explicit rule.
    Malformed lines raise — these are benchmark artifacts, not fuzzy inputs.
    """
    items = []
    for pid in pids:
        path = Path(labels_dir) / f"{pid}.txt"
        if not path.exists():
            continue
        seq = 0
        with open(path, encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"{path}:{lineno}: expected 'class cx cy w h'")
                items.append({'pid': pid, 'key': f'gold:{seq}', 'seq': seq,
                              'x': float(parts[1]), 'y': float(parts[2]), 'conf': None})
                seq += 1
    return items


# --- Crop geometry -------------------------------------------------------------------

def crop_side(width, height, fov_deg):
    """Square crop side in native pixels for a pano of the given dimensions."""
    return min(int(round(width * fov_deg / 360.0)), height, width)


# A bundle whose panos are at (or below) the model's own input size is NOT a native
# archive — its crops carry no more detail than the point-review gallery, and the box
# rule's "tight at native zoom" is a weaker instrument there.
MODEL_INPUT_WIDTH = 4096


def resolution_note(records_by_pid, pids, fov_deg):
    """({"WxH": crop_px}, message) — the resolution this bundle actually annotates at.

    The tool cuts from whatever the bundle stores, so "native resolution" is a property
    of the archive behind ``panos/``, not of this script. Report it rather than assume
    it: richmond's verified 1:1 archive gives 3072 px crops, ``benchmark/manual_gold``'s
    model-resolution bundle gives 1024 px ones, and extent drawn at those two scales is
    not the same measurement.
    """
    dims, counts = {}, {}
    for pid in pids:
        pano = records_by_pid[pid]['pano']
        width, height = pano.get('width'), pano.get('height')
        if not width or not height:
            continue
        key = f"{width}x{height}"
        dims[key] = crop_side(width, height, fov_deg)
        counts[key] = counts.get(key, 0) + 1
    parts = ", ".join(f"{k} -> {dims[k]} px ({counts[k]} panos)" for k in sorted(dims))
    message = f"Crop resolution: {parts}" if parts else "Crop resolution: unknown"
    if dims and all(int(k.split("x")[0]) <= MODEL_INPUT_WIDTH for k in dims):
        message += (f"\n  WARNING: every pano is at or below the model's own input width "
                    f"({MODEL_INPUT_WIDTH}), so this is MODEL-RESOLUTION extent gold, not "
                    f"native. Boxes drawn here are not comparable in edge precision to a "
                    f"native-archive bundle; say so wherever the numbers are reported.")
    return dims, message


def crop_rect(x, y, width, height, side):
    """(left, top) of the side x side crop centered on the normalized point.

    x wraps (equirectangular seam, #43): ``left`` may address columns past the right
    edge, taken modulo width at assembly, so a seam-split ramp appears whole in the
    crop. y clamps by shifting — no wrap at the poles.
    """
    left = int(round(x * width - side / 2)) % width
    top = int(min(max(round(y * height - side / 2), 0), height - side))
    return left, top


def cut_crop(img, left, top, side):
    """The side x side crop with x wrapped around the pano edge."""
    w = img.width
    if left + side <= w:
        return img.crop((left, top, left + side, top + side))
    out = Image.new('RGB', (side, side))
    first = w - left
    out.paste(img.crop((left, top, w, top + side)), (0, 0))
    out.paste(img.crop((0, top, left + side - w, top + side)), (first, 0))
    return out


def entry_meta(item, width, height, side, left, top, record):
    """The viewer entry for one ramp item — pure geometry, no I/O (--html-only path).

    ``cl/ct/cs`` are the crop rect in native pano pixels and ``pw/ph`` the pano dims;
    the viewer maps drawn rects through them to pano-normalized boxes, so a saved crop
    image may even be downscaled for display without touching the geometry.
    """
    pano = record['pano']
    return {
        'pid': item['pid'], 'key': item['key'], 'seq': item['seq'],
        'img': f"{item['pid']}_{item['key'].replace(':', '_')}.jpg",
        'x': round(item['x'], 5), 'y': round(item['y'], 5), 'conf': item.get('conf'),
        'pw': width, 'ph': height, 'cl': left, 'ct': top, 'cs': side,
        'source': pano.get('source', ''), 'date': str(pano.get('capture_date', '')),
    }


def render_pano_items(pid, pano_items, record, panos_dir, images_dir, fov_deg, max_side):
    """Cut every item crop for one pano from its native image (opened/decoded once).

    Geometry comes from the record's stored dimensions — the single source of truth the
    --html-only path also uses — and the on-disk image must match them exactly: these
    are verified 1:1 native archives, so a mismatch means the wrong pixels, not a
    resize opportunity. ``max_side`` (display-only downscale of the saved JPEG) never
    touches geometry: the viewer positions boxes in normalized crop units.
    """
    pano = record['pano']
    width, height = pano.get('width'), pano.get('height')
    if not width or not height:
        raise ValueError(f"{pid}: record carries no pano width/height")
    src = _find_pano_image(panos_dir, pid)
    if src is None:
        raise FileNotFoundError(f"no image in {panos_dir} for {pid}")
    img = Image.open(src).convert('RGB')
    if img.size != (width, height):
        raise ValueError(f"{pid}: disk image {img.size} != records {width}x{height} "
                         f"(native archive expected)")
    side = crop_side(width, height, fov_deg)
    entries = []
    for it in pano_items:
        left, top = crop_rect(it['x'], it['y'], width, height, side)
        crop = cut_crop(img, left, top, side)
        if max_side and side > max_side:
            crop = crop.resize((max_side, max_side), Image.BILINEAR)
        meta = entry_meta(it, width, height, side, left, top, record)
        crop.save(images_dir / meta['img'], quality=JPEG_QUALITY)
        entries.append(meta)
    return entries


# --- boxes.json I/O ------------------------------------------------------------------

def load_boxes(bundle_dir):
    """(panos map, annotator block) from an existing boxes.json, or ({}, {})."""
    p = Path(bundle_dir) / 'boxes.json'
    if not p.exists():
        return {}, {}
    with open(p, encoding='utf-8') as f:
        bj = json.load(f)
    return bj.get('panos', {}), bj.get('annotator', {})


def reconcile_initial(initial, items):
    """Drop prefill entries whose recorded point no longer matches the item's.

    boxes.json stores each ramp's prompt point precisely for this check: a revised
    verdicts.json can shift positional keys (removing a missed mark renumbers later
    ``missed:<i>``), and silently re-attaching a box to a different ramp would corrupt
    the gold. Mismatches are dropped from the viewer and reported loudly — a stale box
    does NOT survive the next export unless redrawn. Entries for (pano, key) pairs not
    rendered this session (e.g. a --sample-panos subset) are kept verbatim so a partial
    session round-trips the rest of the file.

    This is only the FILE half of the guard, and it is the half that catches nothing on
    the common path: the annotator revising in the browser they annotated in reads the
    box from localStorage, which this function never sees. ``bootstrapState`` in the
    viewer applies the same check to local state, which is what actually closes it.
    """
    by = {(it['pid'], it['key']): it for it in items}
    clean, stale = {}, []
    for pid, keys in initial.items():
        for key, rec in keys.items():
            it = by.get((pid, key))
            pt = rec.get('point') or {}
            if it is not None and pt and (
                    abs(pt.get('x', it['x']) - it['x']) > 1e-4
                    or abs(pt.get('y', it['y']) - it['y']) > 1e-4):
                stale.append(f"{pid} {key}")
                continue
            clean.setdefault(pid, {})[key] = rec
    return clean, stale


# --- Viewer --------------------------------------------------------------------------

# Kept out of the template as a standalone pure function so tests can run it under node
# (tests/test_box_gallery.py::test_state_bootstrap_*). It is the one piece of viewer JS
# that can destroy annotations, and both of its failure modes were silent:
#   * merging the prefill per PANO meant one locally-touched ramp suppressed every other
#     ramp boxes.json carried for that pano — and Export then wrote the file without them;
#   * reconciling only the prefill left the revise-in-the-same-browser path (the normal
#     one) unchecked, so a renumbered `missed:<i>` silently re-attached its box.
# `px`/`py` are the prompt point an annotation was made against; they are the whole guard,
# and the export writes them back as `point` rather than the entry's current point.
STATE_BOOTSTRAP_JS = r"""
function bootstrapState(INITIAL, local, ENTRIES) {
  const state = local || {};
  // Merge the prefill per KEY, not per pano.
  for (const pid in INITIAL) {
    for (const key in INITIAL[pid]) {
      if ((state[pid] || {})[key]) continue;
      const rec = Object.assign({}, INITIAL[pid][key]);
      if (rec.point) { rec.px = rec.point.x; rec.py = rec.point.y; delete rec.point; }
      (state[pid] = state[pid] || {})[key] = rec;
    }
  }
  // Reconcile EVERY rendered item against its current prompt point, local state included.
  // Annotations predating px/py carry no recorded point: adopt the current one rather
  // than destroying work, and report how many so they can be spot-checked.
  let staleDropped = 0, adopted = 0;
  for (const e of ENTRIES) {
    const s = (state[e.pid] || {})[e.key];
    if (!s) continue;
    if (s.px === undefined || s.py === undefined) { s.px = e.x; s.py = e.y; adopted++; }
    else if (Math.abs(s.px - e.x) > 1e-4 || Math.abs(s.py - e.y) > 1e-4) {
      delete state[e.pid][e.key]; staleDropped++;
    }
  }
  return {state: state, staleDropped: staleDropped, adopted: adopted};
}
"""

HTML_TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>RampNet box annotator</title>
<style>
  :root{--boxed:#1a9c3e;--cant:#d23;--todo:#ffd400;--edge:#ff9f1c;--target:#00e5ff;--sib:#bbb}
  body{font-family:sans-serif;margin:16px auto;max-width:1200px;background:#fafafa;color:#222}
  a{color:#06c}
  .bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px}
  .bar button,.bar select{font-size:15px;padding:6px 13px;cursor:pointer}
  .meta{color:#666;font-size:13px}
  .badge{font-size:12px;padding:2px 8px;border-radius:10px;background:#eee;color:#555}
  #rule{background:#fff6e6;border:1px solid #e0a33a;border-radius:8px;padding:7px 14px;
        margin:0 0 10px;font-size:13px;line-height:1.45}
  #rule summary{cursor:pointer;font-weight:bold;color:#a35a00;font-size:14px}
  #rule ul{margin:6px 0 2px;padding-left:22px}
  #rule li{margin:3px 0}
  #rulebar{position:sticky;top:0;z-index:5;background:#fff;border:1px solid #e2e2e2;
           border-radius:8px;padding:6px 12px;margin:0 0 10px;font-size:13px;color:#555}
  #rulebar b{color:#a35a00}
  #viewhint{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:6px 0 0;
            font-size:12.5px;color:#555}
  #viewhint .k{display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
  #viewhint .swx{position:relative;width:14px;height:14px;display:inline-block}
  #viewhint .swx:before{content:'';position:absolute;left:50%;top:0;width:2px;height:100%;
            background:var(--target);transform:translateX(-50%)}
  #viewhint .swx:after{content:'';position:absolute;top:50%;left:0;height:2px;width:100%;
            background:var(--target);transform:translateY(-50%)}
  #viewhint .swd{width:9px;height:9px;border-radius:50%;background:var(--sib);
            display:inline-block;box-shadow:0 0 2px #000}
  #viewhint .swb{width:16px;height:11px;border:2px dashed var(--sib);display:inline-block}
  #status{font-weight:bold;padding:3px 10px;border-radius:12px}
  #status.boxed{background:var(--boxed);color:#fff}
  #status.cant{background:var(--cant);color:#fff}
  #status.todo{background:#fff3bf;color:#7a6000;border:1px solid var(--todo)}
  #edgechip{display:none;background:#fff2df;color:#a35a00;border:1px solid var(--edge);
            border-radius:12px;padding:3px 10px;font-size:13px;font-weight:bold}
  /* Square is load-bearing, not cosmetic: #croplayer img is stretched to fill, so a
     non-square viewport distorts the crop anisotropically while the zoom pill (which
     reports the horizontal ratio) still says "native 1:1" — and edges drawn under a
     "tight at native zoom" rule would be quietly biased. Sizing from WIDTH with
     aspect-ratio deriving the height keeps it square at every window size. */
  #cropview{position:relative;width:min(76vh,880px,100%);aspect-ratio:1/1;
            margin:0 auto;overflow:hidden;background:#111;border-radius:6px;
            cursor:crosshair;touch-action:none}
  #notice{display:none;background:#fff2df;border:1px solid var(--edge);color:#7a4a00;
          border-radius:8px;padding:8px 14px;margin:0 0 10px;font-size:13px}
  #cropview.panmode{cursor:grab}
  #zoompill{position:absolute;top:8px;right:8px;background:rgba(0,0,0,.72);color:#fff;
            font:600 12px/1 sans-serif;padding:4px 9px;border-radius:12px;pointer-events:none;
            z-index:6}
  #croplayer{position:absolute;inset:0;transform-origin:0 0;will-change:transform}
  #croplayer img{width:100%;height:100%;display:block;user-select:none;-webkit-user-drag:none}
  /* --iz (inverse zoom) and --bw (border width) are set on #croplayer per transform so
     markers and box edges keep a constant on-screen size at any zoom. */
  .target{position:absolute;width:26px;height:26px;transform:translate(-50%,-50%) scale(var(--iz,1));
          pointer-events:none;z-index:3}
  .target:before,.target:after{content:'';position:absolute;background:var(--target);
          box-shadow:0 0 3px #000}
  .target:before{left:50%;top:0;width:2px;height:100%;transform:translateX(-50%)}
  .target:after{top:50%;left:0;height:2px;width:100%;transform:translateY(-50%)}
  .sib{position:absolute;width:10px;height:10px;transform:translate(-50%,-50%) scale(var(--iz,1));
       border-radius:50%;background:var(--sib);opacity:.75;box-shadow:0 0 3px #000;
       pointer-events:none;z-index:2}
  .sibbox{position:absolute;border:var(--bw,2px) dashed var(--sib);opacity:.6;
          pointer-events:none;z-index:1}
  #abox{position:absolute;border:var(--bw,2px) solid var(--boxed);z-index:4;cursor:move;
        box-shadow:0 0 0 var(--bw,2px) rgba(255,255,255,.35)}
  .handle{position:absolute;width:13px;height:13px;background:#fff;border:2px solid var(--boxed);
          border-radius:3px;transform:translate(-50%,-50%) scale(var(--iz,1));z-index:5}
  .handle.n{left:50%;top:0;cursor:ns-resize}.handle.s{left:50%;top:100%;cursor:ns-resize}
  .handle.e{left:100%;top:50%;cursor:ew-resize}.handle.w{left:0;top:50%;cursor:ew-resize}
  .handle.nw{left:0;top:0;cursor:nwse-resize}.handle.se{left:100%;top:100%;cursor:nwse-resize}
  .handle.ne{left:100%;top:0;cursor:nesw-resize}.handle.sw{left:0;top:100%;cursor:nesw-resize}
  #itembar{margin:10px 0 0;padding:10px 14px;border-radius:8px;display:flex;gap:12px;
           align-items:center;flex-wrap:wrap;border:2px solid #ddd;background:#fff}
  #itembar button{font-size:15px;padding:6px 14px;cursor:pointer}
  #cant.active{background:var(--cant);border:1px solid var(--cant);color:#fff}
  #noterow{display:flex;gap:8px;align-items:center;flex:1;min-width:260px;font-size:13px}
  #noterow input{flex:1;font:13px/1.45 sans-serif;padding:5px 8px;border:1px solid #ccc;
                 border-radius:5px}
  #annotator{background:#eef4ff;border:1px solid #9db8e8;border-radius:8px;padding:9px 14px;
             margin:0 0 12px;font-size:13px}
  #annotator summary{cursor:pointer;font-weight:bold;color:#24457f}
  #annotator .row{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:8px 0 0}
  #annotator label{color:#456}
  #annotator input,#annotator textarea{font:13px/1.45 sans-serif;padding:5px 7px;
             border:1px solid #b8c6de;border-radius:5px;background:#fff;color:#222}
  #annotator textarea{width:100%;box-sizing:border-box;display:block;margin-top:3px;
             resize:vertical}
  .help{font-size:13px;color:#666;margin-top:16px;line-height:1.5}
  kbd{background:#eee;border:1px solid #ccc;border-radius:3px;padding:0 4px;font-size:12px}
</style>

<div id="notice"></div>

<div id="rulebar"><b>Rule v__RULE_V__:</b> whole constructed ramp — apron + warning pad +
  flares; not road, gutter, or level sidewalk; occluded ≠ absent; tight at native zoom;
  the box is a ruler, not a crop — context is the crop rule's job.</div>

<details id="rule">
  <summary>Box rule v__RULE_V__, in full — ships inside every export</summary>
  <ul id="rulelist"></ul>
</details>

<details id="annotator">
  <summary>Annotator — exports as <code>annotator</code> in boxes.json</summary>
  <div class="row">
    <label>Name <input type="text" id="a_name" size="10" placeholder="e.g. jonf"></label>
    <label>Date <input type="text" id="a_date" size="10" placeholder="YYYY-MM-DD"></label>
  </div>
  <label>Notes — anything that fought the rule (snow, ambiguous flares, ...)
    <textarea id="a_notes" rows="3"></textarea></label>
</details>

<div class="bar">
  <button id="prev">&#8592; Prev</button>
  <button id="next">Next &#8594;</button>
  <button id="nexttodo">Next unboxed &#8608;</button>
  <select id="filter">
    <option value="all">All ramps</option>
    <option value="todo">To do</option>
    <option value="boxed">Boxed</option>
    <option value="cant">Can't determine</option>
    <option value="edge">Edge-flagged</option>
  </select>
  <span id="pos" class="meta"></span>
  <span id="progress" class="meta"></span>
  <span style="flex:1"></span>
  <button id="export">Export boxes</button>
</div>

<h2 id="title" style="margin:6px 0;font-size:17px"></h2>
<div id="cropview">
  <div id="croplayer"><img id="cropimg" alt=""><div id="overlay"></div></div>
  <div id="zoompill"></div>
</div>
<div id="viewhint">
  <span class="k"><span class="swx"></span> this ramp — draw ONE box around it</span>
  <span class="k"><span class="swd"></span> another ramp here (gets its own turn)</span>
  <span class="k"><span class="swb"></span> its box</span>
  <span class="k"><b>drag</b>&nbsp;draw &middot; <b>handles</b>&nbsp;adjust &middot;
    <kbd>Space</kbd>/<kbd>Shift</kbd>+drag or middle/right-drag&nbsp;pan &middot;
    scroll&nbsp;zoom</span>
</div>

<div id="itembar">
  <span id="status"></span>
  <button id="cant">Can't determine extent (c)</button>
  <button id="clear">Clear box (x)</button>
  <span id="edgechip">&#9888; box touches the crop edge &mdash; if the apron continues
    beyond the view, re-render with a wider --fov</span>
  <span id="noterow"><label for="itemnote">Note:</label>
    <input type="text" id="itemnote" placeholder="optional — e.g. flare buried under snow"></span>
</div>

<p class="help">
  <kbd>&#8592;</kbd>/<kbd>&#8594;</kbd> ramp &nbsp;&middot;&nbsp;
  <kbd>n</kbd> next unboxed &nbsp;&middot;&nbsp; <kbd>c</kbd> can't determine &nbsp;&middot;&nbsp;
  <kbd>x</kbd> clear &nbsp;&middot;&nbsp; <kbd>r</kbd> reset zoom &nbsp;&middot;&nbsp;
  the zoom pill shows how close you are to native pixels — box edges are only trustworthy
  near native 1:1. Annotations autosave locally and survive a re-render at a different
  --fov; Export downloads <span id="bname"></span> &mdash; save it over
  <code id="savehint"></code>.
</p>
<p class="help">
  <b>Why one ramp-centered crop instead of the full pano:</b> each judgment stays scoped to
  the prompted point (the crosshair), the view is native-resolution without decoding a 75 MP
  equirect per pano, and the crop stitches across the 360&deg; seam so an edge-split ramp is
  drawable as one box. The grey dots/boxes supply the neighbor context that actually matters
  &mdash; not double-boxing a shared apron.
  <b>Why no predicted crop window is shown:</b> the window's size is the quantity this gold
  will judge, so showing it would anchor your drawing on the very thing under test; the
  algorithm-vs-gold comparison lives in the scorer's overlay gallery instead.
</p>

<script>
const ENTRIES = __ENTRIES__;
const RUN_KEY = __RUN_KEY__;
const RUN_NAME = __RUN_NAME__;
const BOX_RULE = __BOX_RULE__;     // {version, text} — embedded verbatim in every export
const FOV_DEG = __FOV_DEG__;
const CROP_PX = __CROP_PX__;       // {"WxH": crop side px} — the resolution this gold was drawn at
const INITIAL = __INITIAL__;       // boxes.json panos map, already reconciled in Python
const INITIAL_ANNOTATOR = __INITIAL_ANNOTATOR__;
const SAVE_HINT = __SAVE_HINT__;
const STORE = 'boxes:' + RUN_KEY;
const ASTORE = 'boxannotator:' + RUN_KEY;
const MIN_PX = 6;                  // smallest drawable box side, native px

// One bullet per rule line; the export carries the same lines joined by \n. The block
// starts open, and staying collapsed is remembered per bundle once the rule is absorbed.
BOX_RULE.text.split('\n').forEach(t => {
  const li = document.createElement('li');
  li.textContent = t;
  document.getElementById('rulelist').appendChild(li);
});
const ruleEl = document.getElementById('rule');
const RSTORE = 'boxrulecollapsed:' + RUN_KEY;
ruleEl.open = localStorage.getItem(RSTORE) !== '1';
ruleEl.addEventListener('toggle', () => localStorage.setItem(RSTORE, ruleEl.open ? '' : '1'));
document.getElementById('savehint').textContent = SAVE_HINT;
document.getElementById('bname').textContent = RUN_NAME + '_boxes.json';

__STATE_BOOTSTRAP__

// state[pid][key] = {status:'boxed'|'cant'|'note', px, py, cx,cy,w,h, note?}
// cx/cy/w/h are PANO-normalized (resolution- and fov-independent); a box may wrap the
// equirect seam, in which case cx sits near 0/1 and the x-interval is [cx-w/2, cx+w/2] mod 1.
const boot = bootstrapState(INITIAL, JSON.parse(localStorage.getItem(STORE) || '{}'), ENTRIES);
let state = boot.state;
const staleDropped = boot.staleDropped, adopted = boot.adopted;

if (staleDropped || adopted) {
  const n = document.getElementById('notice');
  n.style.display = '';
  n.innerHTML =
    (staleDropped ? '<b>' + staleDropped + ' stale annotation(s) dropped</b> — their ' +
      'recorded prompt point no longer matches the ramp behind that key (a re-reviewed ' +
      'verdicts.json renumbers them). Those ramps are back in "To do"; redraw them.<br>' : '') +
    (adopted ? adopted + ' annotation(s) predate the point-stamping guard and were ' +
      'adopted at their current point — spot-check them if verdicts.json changed since.' : '');
}

function save() { localStorage.setItem(STORE, JSON.stringify(state)); }
function itemState(e) { return (state[e.pid] || {})[e.key]; }
function setItem(e, obj) {
  if (!state[e.pid]) state[e.pid] = {};
  state[e.pid][e.key] = Object.assign(obj, {px: e.x, py: e.y}); save();
}
function clearItem(e) { const s = state[e.pid]; if (s) { delete s[e.key]; save(); } }
function isDone(e) { const s = itemState(e); return !!s && (s.status === 'boxed' || s.status === 'cant'); }

// Sibling index: other adjudicated ramps on the same pano (array order == seq order).
const byPid = {};
ENTRIES.forEach(e => { (byPid[e.pid] = byPid[e.pid] || []).push(e); });

// --- Annotator block ------------------------------------------------------------------
let annotator = Object.assign({}, INITIAL_ANNOTATOR, JSON.parse(localStorage.getItem(ASTORE) || '{}'));
const AFIELDS = {name: 'a_name', date: 'a_date', notes: 'a_notes'};
for (const k in AFIELDS) document.getElementById(AFIELDS[k]).value = annotator[k] || '';
document.querySelectorAll('#annotator input, #annotator textarea').forEach(el =>
  el.addEventListener('input', () => {
    for (const k in AFIELDS) annotator[k] = document.getElementById(AFIELDS[k]).value.trim();
    localStorage.setItem(ASTORE, JSON.stringify(annotator));
  }));

// --- Geometry: crop-local [0,1] units <-> pano-normalized -----------------------------
function toPano(e, u, v) {
  return {x: (((e.cl + u * e.cs) % e.pw) + e.pw) % e.pw / e.pw,
          y: (e.ct + v * e.cs) / e.ph};
}
function toCrop(e, x, y) {
  let dx = ((x * e.pw - e.cl) % e.pw + e.pw) % e.pw;   // [0, pw)
  if (dx > e.pw / 2) dx -= e.pw;                        // nearest wrap representation
  return {u: dx / e.cs, v: (y * e.ph - e.ct) / e.cs};
}
function stateRect(e, s) {
  const c = toCrop(e, s.cx, s.cy);
  const hw = s.w * e.pw / (2 * e.cs), hh = s.h * e.ph / (2 * e.cs);
  return {u0: c.u - hw, v0: c.v - hh, u1: c.u + hw, v1: c.v + hh};
}
function rectState(e, r) {
  const p = toPano(e, (r.u0 + r.u1) / 2, (r.v0 + r.v1) / 2);
  const rd = x => Math.round(x * 1e5) / 1e5;
  return {status: 'boxed', cx: rd(p.x), cy: rd(p.y),
          w: rd((r.u1 - r.u0) * e.cs / e.pw), h: rd((r.v1 - r.v0) * e.cs / e.ph)};
}
// A boxed edge on the crop boundary means the apron may continue outside the view —
// unless that boundary is a real pano edge (top/bottom of the equirect; x always wraps).
function edgeFlag(e, s) {
  if (!s || s.status !== 'boxed') return false;
  const r = stateRect(e, s), EPS = 0.006;
  const top = r.v0 <= EPS && e.ct > 0;
  const bot = r.v1 >= 1 - EPS && e.ct + e.cs < e.ph;
  const lr = (r.u0 <= EPS || r.u1 >= 1 - EPS) && e.cs < e.pw;
  return top || bot || lr;
}

// --- Filter/nav -----------------------------------------------------------------------
let filterMode = 'all', view_entries = ENTRIES.slice(), idx = 0;
function matches(e) {
  const s = itemState(e);
  return filterMode === 'todo' ? !isDone(e) :
         filterMode === 'boxed' ? !!s && s.status === 'boxed' :
         filterMode === 'cant' ? !!s && s.status === 'cant' :
         filterMode === 'edge' ? edgeFlag(e, s) : true;
}
function applyFilter() {
  const cur = view_entries[idx];
  view_entries = ENTRIES.filter(matches);
  if (!view_entries.length) { idx = 0; render(); return; }
  const keep = cur ? view_entries.findIndex(e => e.pid === cur.pid && e.key === cur.key) : -1;
  idx = keep >= 0 ? keep : 0;
  render();
}
function curE() { return view_entries[idx]; }

// --- Pan/zoom -------------------------------------------------------------------------
const view = document.getElementById('cropview');
const layer = document.getElementById('croplayer');
const cropImg = document.getElementById('cropimg');
const overlay = document.getElementById('overlay');
let zoom = 1, panX = 0, panY = 0;

function vp() { const r = view.getBoundingClientRect(); return {w: r.width, h: r.height}; }
function maxZoom() { const e = curE(); return e ? Math.max(8, 6 * e.cs / vp().w) : 8; }
function clampPan() {
  const {w, h} = vp();
  panX = Math.min(0, Math.max(w - w * zoom, panX));
  panY = Math.min(0, Math.max(h - h * zoom, panY));
}
function applyTransform() {
  clampPan();
  layer.style.transform = 'translate(' + panX + 'px,' + panY + 'px) scale(' + zoom + ')';
  layer.style.setProperty('--iz', 1 / zoom);
  layer.style.setProperty('--bw', Math.max(2 / zoom, 0.4) + 'px');
  updatePill();
}
// The pill answers "can I trust this edge?" — the share of the crop's NATIVE pixels
// resolved on screen. Tight edges want ~1:1; beyond that is magnification only.
function updatePill() {
  const e = curE(); if (!e) return;
  const dpr = window.devicePixelRatio || 1;
  const frac = vp().w * zoom * dpr / e.cs;
  document.getElementById('zoompill').textContent =
    frac < 0.995 ? Math.round(frac * 100) + '% of native res'
                 : 'native 1:1' + (frac > 1.05 ? ' · ' + frac.toFixed(1) + '× magnified' : '');
}
function resetZoom() { zoom = 1; panX = 0; panY = 0; applyTransform(); }
function zoomAt(cx, cy, f) {
  const old = zoom;
  zoom = Math.min(maxZoom(), Math.max(1, zoom * f));
  panX = cx - (cx - panX) * (zoom / old);
  panY = cy - (cy - panY) * (zoom / old);
  applyTransform();
}
view.addEventListener('wheel', ev => {
  ev.preventDefault();
  const r = view.getBoundingClientRect();
  zoomAt(ev.clientX - r.left, ev.clientY - r.top, ev.deltaY < 0 ? 1.2 : 1 / 1.2);
}, {passive: false});
view.addEventListener('contextmenu', ev => ev.preventDefault());
window.addEventListener('resize', () => { if (view_entries.length) applyTransform(); });

// --- Drawing / editing ----------------------------------------------------------------
// One box per ramp. Left-drag on empty image draws (replacing any existing box); drag
// inside the box moves it; handles resize. Space/middle/right-drag pans. `work` is the
// rect being displayed, in crop units; commit converts it to a pano-normalized box.
let work = null;    // {u0,v0,u1,v1} or null
let drag = null, spaceHeld = false;

function typing(ev) {
  const t = ev.target.tagName;
  return t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT' || ev.target.isContentEditable;
}
document.addEventListener('keydown', ev => {
  if (ev.code === 'Space' && !typing(ev)) { spaceHeld = true; view.classList.add('panmode'); ev.preventDefault(); }
  if (ev.key === 'Shift') view.classList.add('panmode');
});
document.addEventListener('keyup', ev => {
  if (ev.code === 'Space') { spaceHeld = false; view.classList.remove('panmode'); }
  if (ev.key === 'Shift') view.classList.remove('panmode');
});

function cropPt(ev) {
  const r = view.getBoundingClientRect();
  const u = ((ev.clientX - r.left) - panX) / (r.width * zoom);
  const v = ((ev.clientY - r.top) - panY) / (r.height * zoom);
  return {u: Math.min(1, Math.max(0, u)), v: Math.min(1, Math.max(0, v))};
}
function normRect(a, b) {
  return {u0: Math.min(a.u, b.u), v0: Math.min(a.v, b.v),
          u1: Math.max(a.u, b.u), v1: Math.max(a.v, b.v)};
}
view.addEventListener('pointerdown', ev => {
  const e = curE(); if (!e) return;
  if (spaceHeld || ev.shiftKey || ev.button === 1 || ev.button === 2) {
    drag = {kind: 'pan', x: ev.clientX, y: ev.clientY, panX, panY};
    view.setPointerCapture(ev.pointerId); ev.preventDefault(); return;
  }
  if (ev.button !== 0) return;
  const p = cropPt(ev);
  const h = ev.target.closest('.handle');
  if (h) drag = {kind: 'resize', dir: h.dataset.dir, rect: Object.assign({}, work), start: p};
  else if (ev.target.closest('#abox')) drag = {kind: 'move', rect: Object.assign({}, work), start: p};
  else drag = {kind: 'draw', start: p, moved: false};
  view.setPointerCapture(ev.pointerId);
  ev.preventDefault();
});
view.addEventListener('pointermove', ev => {
  if (!drag) return;
  if (drag.kind === 'pan') {
    panX = drag.panX + (ev.clientX - drag.x); panY = drag.panY + (ev.clientY - drag.y);
    applyTransform(); return;
  }
  const e = curE(), p = cropPt(ev);
  if (drag.kind === 'draw') {
    drag.moved = true;
    work = normRect(drag.start, p);
  } else if (drag.kind === 'move') {
    let du = p.u - drag.start.u, dv = p.v - drag.start.v;
    du = Math.min(1 - drag.rect.u1, Math.max(-drag.rect.u0, du));
    dv = Math.min(1 - drag.rect.v1, Math.max(-drag.rect.v0, dv));
    work = {u0: drag.rect.u0 + du, v0: drag.rect.v0 + dv,
            u1: drag.rect.u1 + du, v1: drag.rect.v1 + dv};
  } else if (drag.kind === 'resize') {
    const r = Object.assign({}, drag.rect), d = drag.dir;
    if (d.includes('w')) r.u0 = Math.min(p.u, r.u1 - MIN_PX / e.cs);
    if (d.includes('e')) r.u1 = Math.max(p.u, r.u0 + MIN_PX / e.cs);
    if (d.includes('n')) r.v0 = Math.min(p.v, r.v1 - MIN_PX / e.cs);
    if (d.includes('s')) r.v1 = Math.max(p.v, r.v0 + MIN_PX / e.cs);
    work = r;
  }
  syncBox();
});
view.addEventListener('pointerup', () => {
  const d = drag; drag = null;
  if (!d || d.kind === 'pan') return;
  const e = curE(); if (!e) return;
  if (d.kind === 'draw' && !d.moved) return;              // click, not a drag
  if (work && (work.u1 - work.u0) * e.cs >= MIN_PX && (work.v1 - work.v0) * e.cs >= MIN_PX) {
    const prev = itemState(e) || {};
    const s = rectState(e, work);
    if (prev.note) s.note = prev.note;
    setItem(e, s);
  } else if (d.kind === 'draw') {
    const s = itemState(e);
    work = s && s.status === 'boxed' ? stateRect(e, s) : null;  // degenerate: revert
  }
  renderLight();
});

// --- Rendering ------------------------------------------------------------------------
const boxEl = document.createElement('div');
boxEl.id = 'abox';
['n','s','e','w','nw','ne','sw','se'].forEach(d => {
  const h = document.createElement('div');
  h.className = 'handle ' + d; h.dataset.dir = d;
  boxEl.appendChild(h);
});
function syncBox() {
  if (!work) { boxEl.remove(); return; }
  boxEl.style.left = (work.u0 * 100) + '%';
  boxEl.style.top = (work.v0 * 100) + '%';
  boxEl.style.width = ((work.u1 - work.u0) * 100) + '%';
  boxEl.style.height = ((work.v1 - work.v0) * 100) + '%';
  if (!boxEl.parentNode) overlay.appendChild(boxEl);
}

// One definition, used by both render paths — renderLight used to rebuild this without
// the can't-determine suffix, so the count blinked away on every state change.
function updateProgress() {
  const done = ENTRIES.filter(isDone).length;
  const cant = ENTRIES.filter(x => { const s = itemState(x); return s && s.status === 'cant'; }).length;
  document.getElementById('progress').textContent =
    done + '/' + ENTRIES.length + ' ramps annotated' + (cant ? ' (' + cant + " can't)" : '');
}

function render() {
  const e = curE();
  updateProgress();
  overlay.innerHTML = '';
  if (!e) {
    document.getElementById('title').textContent = 'No ramps match this filter';
    cropImg.removeAttribute('src');
    document.getElementById('itembar').style.display = 'none';
    document.getElementById('pos').textContent = '';
    work = null;
    return;
  }
  document.getElementById('itembar').style.display = '';
  document.getElementById('pos').textContent = (idx + 1) + ' / ' + view_entries.length;
  resetZoom();
  cropImg.src = 'images/' + e.img;

  const sibs = byPid[e.pid], k = sibs.indexOf(e);
  const viewerUrl = e.source === 'mapillary'
    ? 'https://www.mapillary.com/app/?pKey=' + e.pid + '&focus=photo'
    : 'https://www.google.com/maps/@?api=1&map_action=pano&pano=' + e.pid;
  document.getElementById('title').innerHTML =
    '<a href="' + viewerUrl + '" target="_blank">' + e.pid + '</a> ' +
    '<span class="badge">' + e.key + '</span> ' +
    '<span class="meta">ramp ' + (k + 1) + '/' + sibs.length + ' on this pano' +
    (e.conf != null ? ' &mdash; conf ' + e.conf.toFixed(2) : '') +
    (e.date ? ' &mdash; captured ' + e.date : '') + '</span>';

  // Target crosshair + siblings (their points, and their boxes as dashed outlines so a
  // neighboring apron is never boxed twice).
  const t = toCrop(e, e.x, e.y);
  const tg = document.createElement('div');
  tg.className = 'target';
  tg.style.left = (t.u * 100) + '%'; tg.style.top = (t.v * 100) + '%';
  overlay.appendChild(tg);
  sibs.forEach(o => {
    if (o === e) return;
    const c = toCrop(e, o.x, o.y);
    if (c.u < -0.02 || c.u > 1.02 || c.v < -0.02 || c.v > 1.02) return;
    const d = document.createElement('div');
    d.className = 'sib';
    d.style.left = (c.u * 100) + '%'; d.style.top = (c.v * 100) + '%';
    d.title = o.key;
    overlay.appendChild(d);
    const os = itemState(o);
    if (os && os.status === 'boxed') {
      const r = stateRect(e, os), b = document.createElement('div');
      b.className = 'sibbox';
      b.style.left = (r.u0 * 100) + '%'; b.style.top = (r.v0 * 100) + '%';
      b.style.width = ((r.u1 - r.u0) * 100) + '%'; b.style.height = ((r.v1 - r.v0) * 100) + '%';
      overlay.appendChild(b);
    }
  });

  const s = itemState(e);
  work = s && s.status === 'boxed' ? stateRect(e, s) : null;
  syncBox();
  const noteEl = document.getElementById('itemnote');
  if (document.activeElement !== noteEl) noteEl.value = (s && s.note) || '';
  renderLight();
}

// State-only refresh: chips, buttons, progress — no zoom reset, no image reload.
function renderLight() {
  const e = curE(); if (!e) return;
  const s = itemState(e);
  updateProgress();
  const st = document.getElementById('status');
  if (s && s.status === 'boxed') { st.className = 'boxed'; st.textContent = '✓ BOXED'; }
  else if (s && s.status === 'cant') { st.className = 'cant'; st.textContent = "CAN'T DETERMINE"; }
  else { st.className = 'todo'; st.textContent = '● TO DO'; }
  document.getElementById('cant').classList.toggle('active', !!s && s.status === 'cant');
  document.getElementById('edgechip').style.display = edgeFlag(e, s) ? '' : 'none';
  syncBox();
}

// --- Controls -------------------------------------------------------------------------
document.getElementById('prev').onclick = () => {
  if (view_entries.length) { idx = (idx - 1 + view_entries.length) % view_entries.length; render(); } };
document.getElementById('next').onclick = () => {
  if (view_entries.length) { idx = (idx + 1) % view_entries.length; render(); } };
document.getElementById('nexttodo').onclick = () => {
  if (!view_entries.length) return;
  for (let k = 1; k <= view_entries.length; k++) {
    const j = (idx + k) % view_entries.length;
    if (!isDone(view_entries[j])) { idx = j; render(); return; }
  }
  alert('Every ramp in this view is annotated. Switch the filter to "All ramps" to double-check, then Export.');
};
document.getElementById('filter').onchange = ev => { filterMode = ev.target.value; applyFilter(); };
document.getElementById('cant').onclick = () => {
  const e = curE(); if (!e) return;
  const s = itemState(e);
  if (s && s.status === 'cant') clearItem(e);
  else setItem(e, Object.assign({status: 'cant'}, s && s.note ? {note: s.note} : {}));
  work = null;
  renderLight();
};
document.getElementById('clear').onclick = () => {
  const e = curE(); if (!e) return;
  clearItem(e); work = null; renderLight();
};
document.getElementById('itemnote').addEventListener('input', ev => {
  const e = curE(); if (!e) return;
  const s = itemState(e) || {};
  if (ev.target.value.trim()) {
    s.note = ev.target.value;
    // A note on a not-yet-annotated ramp needs a status or the export drops it — and a
    // note explaining why an item is hard is exactly the one you don't want to lose.
    if (!s.status) s.status = 'note';
  } else {
    delete s.note;
    if (s.status === 'note') delete s.status;
  }
  if (s.status) setItem(e, s); else clearItem(e);
  renderLight();
});
document.addEventListener('keydown', ev => {
  if (ev.ctrlKey || ev.metaKey || ev.altKey || typing(ev)) return;
  if (ev.key === 'ArrowLeft') document.getElementById('prev').click();
  else if (ev.key === 'ArrowRight') document.getElementById('next').click();
  else if (ev.key === 'n' || ev.key === 'N') document.getElementById('nexttodo').click();
  else if (ev.key === 'c' || ev.key === 'C') document.getElementById('cant').click();
  else if (ev.key === 'x' || ev.key === 'X' || ev.key === 'Delete') document.getElementById('clear').click();
  else if (ev.key === 'r' || ev.key === 'R') resetZoom();
});

// --- Export ---------------------------------------------------------------------------
document.getElementById('export').onclick = () => {
  const todo = ENTRIES.filter(e => !isDone(e)).length;
  const flagged = ENTRIES.filter(e => edgeFlag(e, itemState(e))).length;
  if (todo || flagged) {
    const msg = (todo ? todo + ' ramp(s) still to do.\n' : '') +
      (flagged ? flagged + ' box(es) touch their crop edge — the apron may continue ' +
                 'beyond the view (filter: "Edge-flagged").\n' : '') +
      '\nExport anyway?';
    if (!confirm(msg)) return;
  }
  const out = {run_key: RUN_KEY, run_name: RUN_NAME,
               box_rule: BOX_RULE, crop_fov_deg: FOV_DEG,
               crop_px_by_pano_dims: CROP_PX,
               exported_at: new Date().toISOString()};
  const a = {};
  for (const k in AFIELDS) if (annotator[k]) a[k] = annotator[k];
  if (Object.keys(a).length) out.annotator = a;
  out.panos = {};
  const known = new Set(ENTRIES.map(e => e.pid + ' ' + e.key));
  for (const e of ENTRIES) {
    const s = itemState(e);
    if (!s || !s.status) continue;
    // The point is the one the annotation was MADE against (s.px/s.py), not the entry's
    // current point: re-stamping would launder a stale box past every future check.
    const rec = {point: {x: s.px !== undefined ? s.px : e.x,
                         y: s.py !== undefined ? s.py : e.y}, status: s.status};
    if (s.status === 'boxed') {
      rec.cx = s.cx; rec.cy = s.cy; rec.w = s.w; rec.h = s.h;
      if (edgeFlag(e, s)) rec.edge_flag = true;
    }
    if (s.note && s.note.trim()) rec.note = s.note.trim();
    (out.panos[e.pid] = out.panos[e.pid] || {})[e.key] = rec;
  }
  // Round-trip items this session didn't render (a --sample-panos subset must not
  // truncate the rest of the file on export). Converted back to the on-disk shape:
  // px/py is internal, `point` is the schema.
  for (const pid in state) for (const key in state[pid]) {
    if (known.has(pid + ' ' + key)) continue;
    const s = state[pid][key];
    if (!s || !s.status) continue;
    const rec = Object.assign({}, s);
    delete rec.px; delete rec.py;
    if (s.px !== undefined) rec.point = {x: s.px, y: s.py};
    (out.panos[pid] = out.panos[pid] || {})[key] = rec;
  }
  const blob = new Blob([JSON.stringify(out, null, 2)], {type: 'application/json'});
  const el = document.createElement('a');
  el.href = URL.createObjectURL(blob);
  el.download = RUN_NAME + '_boxes.json';
  el.click();
};

save();   // persist any prefill merged in above
render();
</script>
"""


def build_html(entries, initial, annotator, run_key, run_name, fov_deg, save_hint,
               crop_px=None):
    return (HTML_TEMPLATE
            .replace('__RULE_V__', str(BOX_RULE_VERSION))
            .replace('__ENTRIES__', json.dumps(entries))
            .replace('__INITIAL__', json.dumps(initial))
            .replace('__INITIAL_ANNOTATOR__', json.dumps(annotator or {}))
            .replace('__RUN_KEY__', json.dumps(run_key))
            .replace('__RUN_NAME__', json.dumps(run_name))
            .replace('__BOX_RULE__', json.dumps({'version': BOX_RULE_VERSION, 'text': BOX_RULE}))
            .replace('__FOV_DEG__', str(fov_deg))
            .replace('__CROP_PX__', json.dumps(crop_px or {}))
            .replace('__STATE_BOOTSTRAP__', STATE_BOOTSTRAP_JS)
            .replace('__SAVE_HINT__', json.dumps(str(save_hint))))


def main():
    parser = argparse.ArgumentParser(
        description="Build the whole-apron box annotator over a benchmark bundle (#116).")
    parser.add_argument("bundle", help="Benchmark bundle dir (e.g. benchmark/richmond).")
    parser.add_argument("--out", type=Path,
                        help="Output gallery dir (default: <bundle>/box_gallery).")
    parser.add_argument("--fov", type=float, default=DEFAULT_FOV_DEG,
                        help="Crop width in degrees of pano longitude (default: "
                             "%(default)s). Re-rendering wider keeps every annotation.")
    parser.add_argument("--max-side", type=int, default=0,
                        help="Downscale saved crop JPEGs to this side for disk/display "
                             "(0 = native, the default; geometry is unaffected).")
    parser.add_argument("--html-only", action="store_true",
                        help="Rebuild index.html from existing images (skip re-cutting "
                             "crops). For iterating on the viewer.")
    parser.add_argument("--from-manual-labels", nargs='?', const='', metavar='DIR',
                        help="Take items from YOLO label files (default DIR: "
                             "<repo>/manual_labels) instead of verdicts.json — the "
                             "manual_gold re-annotation mode (#114).")
    parser.add_argument("--sample-panos", type=int, default=0,
                        help="With --from-manual-labels: annotate a random subset of "
                             "this many panos (0 = all).")
    parser.add_argument("--seed", type=int, default=0,
                        help="With --sample-panos: sampling seed (default: %(default)s).")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    records, panos_dir, verdicts_panos, run_key, run_name, _ = load_bundle(bundle)
    records_by_pid = {r['pano']['panorama_id']: r for r in records}

    if args.from_manual_labels is not None:
        labels_dir = Path(args.from_manual_labels) if args.from_manual_labels else \
            Path(__file__).resolve().parent.parent / "manual_labels"
        if not labels_dir.is_dir():
            sys.exit(f"--from-manual-labels: no such directory {labels_dir}")
        pids = [p.stem for p in sorted(labels_dir.glob("*.txt")) if p.stem in records_by_pid]
        if not pids:
            sys.exit(f"--from-manual-labels: no label file in {labels_dir} matches a pano "
                     f"in {bundle}/records.jsonl")
        if args.sample_panos:            # sample before parsing, not after
            random.Random(args.seed).shuffle(pids)
            pids = sorted(pids[:args.sample_panos])
        items = items_from_manual_labels(labels_dir, pids)
        warnings = []
    else:
        if not verdicts_panos:
            sys.exit(f"No verdicts.json in {bundle} — box annotation is a second pass "
                     f"over adjudicated point review; run gt_gallery and export "
                     f"verdicts first (or use --from-manual-labels).")
        items, warnings = enumerate_items(verdicts_panos, records_by_pid)
    for w in warnings:
        print(f"  {w}")
    if not items:
        sys.exit("No adjudicated ramps to annotate.")

    initial_raw, annotator = load_boxes(bundle)
    initial, stale = reconcile_initial(initial_raw, items)
    for s in stale:
        print(f"  STALE prefill dropped (point moved — redraw it, or it is lost on the "
              f"next export): {s}")

    by_pid = {}
    for it in items:
        by_pid.setdefault(it['pid'], []).append(it)
    print(f"{len(items)} adjudicated ramps on {len(by_pid)} panos.")
    crop_px, res_message = resolution_note(records_by_pid, sorted(by_pid), args.fov)
    print(res_message)

    out_dir = args.out or bundle / "box_gallery"
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if args.html_only:
        entries = []
        for pid, its in by_pid.items():
            rec = records_by_pid[pid]
            width, height = rec['pano'].get('width'), rec['pano'].get('height')
            if not width or not height:
                continue
            side = crop_side(width, height, args.fov)
            for it in its:
                meta = entry_meta(it, width, height, side,
                                  *crop_rect(it['x'], it['y'], width, height, side), rec)
                if (images_dir / meta['img']).exists():
                    entries.append(meta)
        print(f"  --html-only: {len(entries)} ramps with existing crops.")
    else:
        entries, done = [], 0
        with ThreadPoolExecutor(max_workers=RENDER_WORKERS) as pool:
            futures = {pool.submit(render_pano_items, pid, its, records_by_pid[pid],
                                   panos_dir, images_dir, args.fov, args.max_side): pid
                       for pid, its in by_pid.items()}
            for future in as_completed(futures):
                done += 1
                try:
                    entries.extend(future.result())
                except Exception as e:
                    print(f"  skipped {futures[future]}: {e}")
                if done % 10 == 0 or done == len(futures):
                    print(f"  rendered {done}/{len(futures)} panos")

    # Stable pano shuffle (hash of pid) so nearby panos don't queue together, with a
    # pano's own ramps kept consecutive — consistent within-pano judgments and no
    # double-boxing of a shared apron.
    entries.sort(key=lambda e: (hashlib.sha1(e['pid'].encode()).hexdigest(), e['seq']))
    index_path = out_dir / "index.html"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(build_html(entries, initial, annotator, run_key, run_name, args.fov,
                           bundle / "boxes.json", crop_px))

    n_prefilled = sum(1 for keys in initial.values() for _ in keys)
    print(f"Box annotator: {index_path}")
    if n_prefilled:
        print(f"Prefilled {n_prefilled} annotations from {bundle / 'boxes.json'} for revision.")
    print(f"Rule v{BOX_RULE_VERSION} is shown in the viewer and embedded in every export.")
    print(f"Open it, draw, Export, then save the download over {bundle / 'boxes.json'}")


if __name__ == "__main__":
    main()
