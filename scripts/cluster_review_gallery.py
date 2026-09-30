"""Corner-level cluster review tool (issue #224): one review unit per screen, seeded groups
of labels to merge, split and mark, exported as ``assignments.json``.

A sibling of ``gt_gallery.py`` and ``box_gallery.py``, not a mode inside them. The bundle
is written by the auto-labeler's ``scripts/export_cluster_review.py`` into
``benchmark/<city>/cluster_review/``; this script only renders ``gallery/index.html``
beside it (crops and aerials are referenced by relative path -- nothing is copied). The
rubric is ``benchmark/RUBRICS.md`` §6 and the protocol ``docs/cluster_review_protocol.md``.

Per unit the reviewer sees an aerial panel (labels as dots coloured by their current group,
each camera as a small triangle with a thin ray to its label, the 30 m window, north arrow,
10 m scale bar, imagery attribution) and one crop strip per group. Keyboard-first:

    click crops / dots   select labels          1-9   assign selection to group N
    n   new group from the selection (split)    m     merge the selected groups
    x   not a ramp     u   unsure     0   unassign
    a   add-uncovered mode: click the aerial to add a ramp no label covers, click a mark to
        toggle unsure, shift-click to remove     p   place the selected group's ramp point
    c   unit complete (refused with an unassigned label)     z   undo     <-/->  units   ?  help

City inventory points appear only after the unit is marked complete. Each unit's
``elapsed_s`` accumulates while it is on screen and the tab is visible (1 s ticks, each
capped at 2 s so a sleeping laptop adds nothing). State autosaves to localStorage keyed by
bundle + rater + label-snapshot sha256; Export downloads the assignments file; an existing
file prefills for revision.

    python scripts/cluster_review_gallery.py benchmark/vancouver/cluster_review --pilot
        # rater A: deployed seed, exports assignments.json
    python scripts/cluster_review_gallery.py benchmark/vancouver/cluster_review --pilot \\
        --rater mikey --role b --out benchmark/vancouver/cluster_review/gallery_mikey
        # rater B: only double-rated units, each seeded with its rater_b_seed,
        # exports assignments__mikey.json
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rampnet import cluster_review as cr  # noqa: E402

SEED_ARMS = ('deployed', 'fusion')


def unit_seed_arm(unit, seed_arm, role):
    """The arm a unit opens with: an explicit arm, or `auto` = deployed for rater A and
    the unit's rater_b_seed (deployed when it has none) for rater B."""
    if seed_arm != 'auto':
        return seed_arm
    if role == 'b':
        return unit.get('rater_b_seed') or 'deployed'
    return 'deployed'


def select_units(corners, pilot=False, role='a'):
    """Units to render, interleaved by sha1(corner_id) as gt_gallery interleaves panos (a
    unit's slot says nothing about its density or stratum)."""
    us = [c for c in corners if (not pilot or c.get('pilot'))
          and (role != 'b' or c.get('double_rate'))]
    return sorted(us, key=lambda c: hashlib.sha1(c['corner_id'].encode()).hexdigest())


def viewer_unit(c, arm, rel):
    """The per-unit payload the viewer needs (paths relative to the gallery dir)."""
    return {'id': c['corner_id'], 'type': c['type'], 'has_labels': c['has_labels'],
            'pilot': bool(c.get('pilot')), 'double_rate': bool(c.get('double_rate')),
            'seed_arm': arm, 'centre': c['centre'], 'window_m': c.get('window_m', 30.0),
            'aerial': dict(c['aerial'], file=rel + c['aerial']['file']),
            'inventory': c.get('inventory', []),
            'labels': [{'key': lab['key'], 'pano_id': lab['pano_id'],
                        'user_kind': lab.get('user_kind'), 'lat': lab['lat'], 'lng': lab['lng'],
                        'camera': lab.get('camera') or {}, 'date': lab.get('capture_date'),
                        'crop': rel + lab['crop'],
                        'seed': (lab.get('seed_group') or {}).get(arm)}
                       for lab in c['labels']]}


def load_prefill(bundle, rater, snapshot):
    """(assignments or None, message). A file made on another label snapshot is not
    loaded: its keys could name different labels."""
    path = Path(bundle) / cr.rater_file_name(rater)
    if not path.exists():
        return None, None
    a = json.loads(path.read_text(encoding='utf-8'))
    if a.get('snapshot_sha256') != snapshot['labels']['sha256']:
        return None, (f'NOT prefilled from {path}: it was made on label snapshot '
                      f"{str(a.get('snapshot_sha256'))[:12]}, the bundle is "
                      f"{snapshot['labels']['sha256'][:12]}")
    return a, f'prefilled {len(a.get("corners") or {})} units from {path} for revision'


# Kept out of the template as a standalone pure function so tests run it under node
# (tests/test_cluster_review_gallery.py), as box_gallery.STATE_BOOTSTRAP_JS is: it is the
# one piece of viewer JS that can destroy review work. Its rules:
#   * a prefill from another label snapshot is ignored whole (keys may name other labels);
#   * local state wins per UNIT, the prefill fills units the browser has no state for;
#   * every rendered unit is reconciled against its current label keys -- a key that left
#     the unit is dropped, a new key is added unassigned, and either reopens the unit
#     (complete = false) rather than silently keeping an attestation that no longer holds;
#   * a unit never seen gets its seed grouping, one ramp group per seed group in label
#     order (a label with no seed group opens unassigned);
#   * state for units not rendered this session is kept verbatim, so Export round-trips it.
STATE_BOOTSTRAP_JS = r"""
function seedUnit(u) {
  const labels = {}, ramps = {}, map = {};
  let n = 0;
  for (const lab of u.labels) {
    if (lab.seed === null || lab.seed === undefined) { labels[lab.key] = null; continue; }
    if (!(lab.seed in map)) { n++; map[lab.seed] = 'r' + n; ramps['r' + n] = {placed: false}; }
    labels[lab.key] = map[lab.seed];
  }
  return {seed_arm: u.seed_arm, labels: labels, ramps: ramps, uncovered: [],
          complete: false, elapsed_s: 0, note: '', seen: false, nextRamp: n + 1};
}
function fromFile(f) {
  const ramps = {};
  let mx = 0;
  for (const r in (f.ramps || {})) {
    const p = f.ramps[r];
    ramps[r] = p.placed ? {placed: true, lat: p.lat, lng: p.lng} : {placed: false};
    mx = Math.max(mx, parseInt(r.slice(1), 10) || 0);
  }
  return {seed_arm: f.seed_arm, labels: Object.assign({}, f.labels || {}), ramps: ramps,
          uncovered: (f.uncovered || []).map(p => Object.assign({}, p)),
          complete: !!f.complete, elapsed_s: f.elapsed_s || 0, note: f.note || '',
          seen: true, nextRamp: mx + 1};
}
function bootstrapState(INITIAL, local, UNITS, SNAPSHOT) {
  const state = local || {};
  let prefilled = 0, reopened = 0, initialIgnored = false;
  if (INITIAL && INITIAL.snapshot_sha256 !== SNAPSHOT) initialIgnored = true;
  else if (INITIAL) {
    for (const cid in (INITIAL.corners || {})) {
      if (state[cid]) continue;
      state[cid] = fromFile(INITIAL.corners[cid]);
      prefilled++;
    }
  }
  for (const u of UNITS) {
    const s = state[u.id];
    if (!s) { state[u.id] = seedUnit(u); continue; }
    const keys = new Set(u.labels.map(l => l.key));
    let changed = false;
    for (const k of Object.keys(s.labels)) if (!keys.has(k)) { delete s.labels[k]; changed = true; }
    for (const k of keys) if (!(k in s.labels)) { s.labels[k] = null; changed = true; }
    if (changed) { if (s.complete) reopened++; s.complete = false; }
  }
  return {state: state, prefilled: prefilled, reopened: reopened, initialIgnored: initialIgnored};
}
"""

HTML_TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>RampNet cluster review</title>
<style>
  body{font-family:sans-serif;margin:12px auto;max-width:1500px;background:#fafafa;color:#222}
  .bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px}
  .bar button{font-size:14px;padding:5px 12px;cursor:pointer}
  .meta{color:#666;font-size:13px}
  .badge{font-size:12px;padding:2px 8px;border-radius:10px;background:#eee;color:#444}
  .badge.done{background:#1a9c3e;color:#fff}
  .badge.todo{background:#fff3bf;color:#7a6000}
  #notice{display:none;background:#fff2df;border:1px solid #ff9f1c;border-radius:8px;
          padding:8px 14px;margin:0 0 10px;font-size:13px}
  #rulebar{background:#fff;border:1px solid #e2e2e2;border-radius:8px;padding:6px 12px;
           margin:0 0 8px;font-size:13px;color:#555}
  #main{display:flex;gap:14px;align-items:flex-start}
  #left{flex:0 0 520px;position:sticky;top:6px}
  #aerialwrap{position:relative;width:512px;height:512px;background:#222;border-radius:6px;
              overflow:hidden}
  #aerialwrap img{position:absolute;left:0;top:0;width:512px;height:512px}
  #aerialwrap svg{position:absolute;left:0;top:0}
  #aerialwrap.addmode{cursor:crosshair;outline:3px solid #ff9f1c}
  #aerialwrap.placemode{cursor:crosshair;outline:3px solid #00c2ff}
  #attrib{font-size:11px;color:#666;margin:3px 0 8px}
  #right{flex:1;min-width:0}
  .group{background:#fff;border:1px solid #ddd;border-left:8px solid #999;border-radius:6px;
         margin:0 0 8px;padding:6px 8px}
  .group.gsel{box-shadow:0 0 0 3px #00c2ff}
  .ghead{font-size:13px;cursor:pointer;user-select:none;margin-bottom:4px}
  .ghead b{font-size:14px}
  .crops{display:flex;flex-wrap:wrap;gap:6px}
  .crop{width:256px;cursor:pointer;border:3px solid transparent;border-radius:4px;background:#eee}
  .crop.sel{border-color:#00c2ff}
  .crop img{width:256px;height:256px;display:block}
  .crop .cap{font-size:11px;color:#555;padding:2px 3px;word-break:break-all}
  .crop .cap .h{color:#b35c00;font-weight:bold}
  #unitnote{width:100%;box-sizing:border-box;font:13px sans-serif;padding:5px}
  #notes{background:#eef4ff;border:1px solid #9db8e8;border-radius:8px;padding:8px 12px;
         margin:0 0 10px;font-size:13px}
  #notes summary{cursor:pointer;font-weight:bold;color:#24457f}
  #notes input,#notes textarea,#notes select{font:13px sans-serif;padding:3px 6px}
  #notes textarea{width:100%;box-sizing:border-box}
  #help{display:none;position:fixed;right:20px;top:20px;background:#fff;border:1px solid #888;
        border-radius:8px;padding:12px 18px;font-size:13px;line-height:1.6;z-index:20;
        box-shadow:0 4px 18px rgba(0,0,0,.25)}
  kbd{background:#eee;border:1px solid #ccc;border-radius:3px;padding:0 4px;font-size:12px}
</style>

<div id="notice"></div>
<div id="rulebar"><b>Rubric v__RUBRIC_V__</b> (RUBRICS.md §6): one group per physical curb
  ramp; a dual-direction apron is <b>two</b> ramps; a driveway is <b>not a ramp</b>; a ramp no
  label covers gets an <b>uncovered</b> point (<kbd>a</kbd>); <b>unsure</b> abstains. Mark the
  unit complete (<kbd>c</kbd>) only when every label is assigned and every uncovered ramp in
  the window is marked. <kbd>?</kbd> for keys.</div>

<details id="notes">
  <summary>Review notes — export as <code>review_notes</code></summary>
  <div>Reviewer <input id="n_reviewer" size="10"> Date <input id="n_date" size="10"
    placeholder="YYYY-MM-DD"> Confidence <select id="n_conf"><option value=""></option>
    <option>high</option><option>medium</option><option>low</option></select></div>
  <div>Summary <textarea id="n_summary" rows="2"></textarea></div>
  <div>Caveats (one per line) <textarea id="n_caveats" rows="3"></textarea></div>
</details>

<div class="bar">
  <button id="prev">&#8592; Prev</button><button id="next">Next &#8594;</button>
  <button id="nexttodo">Next incomplete</button>
  <span id="pos" class="meta"></span><span id="progress" class="meta"></span>
  <span style="flex:1"></span>
  <span id="mode" class="badge"></span>
  <button id="export">Export</button>
</div>
<h2 id="title" style="margin:4px 0 8px;font-size:17px"></h2>
<div id="main">
  <div id="left">
    <div id="aerialwrap"><img id="aerial" alt=""><svg id="plan" width="512" height="512"></svg></div>
    <div id="attrib"></div>
    <div class="meta" id="legend">dots = labels (colour = group; ringed = selected; hollow =
      not ramp; grey = unsure/unassigned) · triangle + thin line = camera and its ray ·
      star = ramp point (hollow = mean of its labels, filled = placed) · orange X =
      uncovered ramp (dashed = unsure) · dashed circle = 30 m window · white squares =
      city inventory (after complete)</div>
    <p><textarea id="unitnote" rows="2" placeholder="Note on this unit (optional)"></textarea></p>
  </div>
  <div id="right"></div>
</div>
<div id="help">
  <b>Keys</b><br>
  click crop / dot: select label (again to deselect) · click group header: select group<br>
  <kbd>1</kbd>-<kbd>9</kbd> assign selected labels to group N · <kbd>n</kbd> new group from
  selection (split) · <kbd>m</kbd> merge selected groups (or the groups of the selected
  labels)<br>
  <kbd>x</kbd> not a ramp · <kbd>u</kbd> unsure · <kbd>0</kbd> unassign · <kbd>Esc</kbd> clear
  selection / mode<br>
  <kbd>a</kbd> add-uncovered mode: click aerial = add, click a mark = toggle unsure,
  shift-click = remove<br>
  <kbd>p</kbd> place the selected group's ramp point (next aerial click) · <kbd>P</kbd> reset
  it to the mean<br>
  <kbd>c</kbd> complete / reopen · <kbd>z</kbd> undo · <kbd>&#8592;</kbd>/<kbd>&#8594;</kbd>
  units · <kbd>?</kbd> this help
</div>

<script>
const UNITS = __UNITS__;
const SNAPSHOT = __SNAPSHOT__;          // label-snapshot sha256 every export is bound to
const CITY = __CITY__;
const RATER = __RATER__;                // null = rater A (assignments.json)
const ROLE = __ROLE__;
const RUBRIC_VERSION = __RUBRIC_V__;
const INITIAL = __INITIAL__;            // an existing assignments file, or null
const ATTRIBUTION = __ATTRIBUTION__;
const FILE_NAME = __FILE_NAME__;
const STORE = 'clusterreview:' + CITY + ':' + (RATER || 'A') + ':' + SNAPSHOT;
const NSTORE = 'clusterreviewnotes:' + CITY + ':' + (RATER || 'A') + ':' + SNAPSHOT;
const COLORS = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4','#42d4f4','#f032e6',
                '#bfef45','#fabed4','#469990','#dcbeff','#9a6324','#fffac8','#800000',
                '#aaffc3','#808000','#ffd8b1','#000075'];

__STATE_BOOTSTRAP__

function loadLocal() { try { return JSON.parse(localStorage.getItem(STORE) || '{}'); } catch (e) { return {}; } }
const boot = bootstrapState(INITIAL, loadLocal(), UNITS, SNAPSHOT);
let state = boot.state;
function save() { try { localStorage.setItem(STORE, JSON.stringify(state)); } catch (e) {} }
(function notice() {
  const bits = [];
  if (boot.initialIgnored) bits.push('<b>The assignments file was made on another label snapshot and was NOT loaded.</b>');
  if (boot.prefilled) bits.push(boot.prefilled + ' unit(s) prefilled from the assignments file.');
  if (boot.reopened) bits.push('<b>' + boot.reopened + ' complete unit(s) reopened</b>: their label set changed since they were reviewed.');
  if (bits.length) { const n = document.getElementById('notice'); n.style.display = ''; n.innerHTML = bits.join('<br>'); }
})();
document.getElementById('attrib').textContent = ATTRIBUTION;

// --- review notes --------------------------------------------------------------------
let notes = Object.assign({}, (INITIAL && !boot.initialIgnored && INITIAL.review_notes) || {});
try { Object.assign(notes, JSON.parse(localStorage.getItem(NSTORE) || '{}')); } catch (e) {}
const NF = {reviewer: 'n_reviewer', reviewed_at: 'n_date', confidence: 'n_conf', summary: 'n_summary'};
for (const k in NF) document.getElementById(NF[k]).value = notes[k] || '';
document.getElementById('n_caveats').value = (notes.caveats || []).join('\n');
function saveNotes() {
  for (const k in NF) notes[k] = document.getElementById(NF[k]).value.trim();
  notes.caveats = document.getElementById('n_caveats').value.split('\n').map(s => s.trim()).filter(Boolean);
  try { localStorage.setItem(NSTORE, JSON.stringify(notes)); } catch (e) {}
}
document.querySelectorAll('#notes input, #notes textarea, #notes select').forEach(el => el.addEventListener('input', saveNotes));

// --- geometry: lat/lng <-> aerial pixels through Web Mercator world pixels ---------------
function worldPx(lat, lng, z) {
  const n = Math.pow(2, z) * 256;
  return [(lng + 180) / 360 * n, (1 - Math.asinh(Math.tan(lat * Math.PI / 180)) / Math.PI) / 2 * n];
}
function toImg(u, lat, lng) {
  const a = u.aerial, w = a.world_px, p = worldPx(lat, lng, a.zoom);
  return [(p[0] - w.x0) / (w.x1 - w.x0) * a.px, (p[1] - w.y0) / (w.y1 - w.y0) * a.px];
}
function fromImg(u, x, y) {
  const a = u.aerial, w = a.world_px, n = Math.pow(2, a.zoom) * 256;
  const wx = w.x0 + x / a.px * (w.x1 - w.x0), wy = w.y0 + y / a.px * (w.y1 - w.y0);
  const lng = wx / n * 360 - 180;
  const lat = Math.atan(Math.sinh(Math.PI * (1 - 2 * wy / n))) * 180 / Math.PI;
  return {lat: lat, lng: lng};
}
function pxPerMetre(u) {
  const a = u.aerial, mpp = 156543.03392 * Math.cos(u.centre.lat * Math.PI / 180) / Math.pow(2, a.zoom);
  return a.px / (a.world_px.x1 - a.world_px.x0) / mpp;
}
function haversine(a, b, c, d) {
  const R = 6371008.8, p1 = a * Math.PI / 180, p2 = c * Math.PI / 180;
  const dp = p2 - p1, dl = (d - b) * Math.PI / 180;
  const h = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

// --- navigation ----------------------------------------------------------------------
let idx = 0, sel = new Set(), gsel = new Set(), mode = null, undo = {};
function cur() { return UNITS[idx]; }
function S() { return state[cur().id]; }
function groupsOf(u) {           // ramp groups in display order (by first label)
  const s = state[u.id], order = [];
  for (const lab of u.labels) { const g = s.labels[lab.key]; if (g && g.startsWith('r') && !order.includes(g)) order.push(g); }
  return order;
}
function unassigned(u) { const s = state[u.id]; return u.labels.filter(l => !s.labels[l.key]).length; }
function push() {                // snapshot for undo, before an edit
  const id = cur().id;
  (undo[id] = undo[id] || []).push(JSON.stringify(state[id]));
  if (undo[id].length > 100) undo[id].shift();
}
function edited() { S().seen = true; save(); render(); }

// --- timing: elapsed_s while shown and visible ------------------------------------------
let lastTick = performance.now(), sinceSave = 0;
setInterval(() => {
  const now = performance.now(), dt = Math.min((now - lastTick) / 1000, 2);
  lastTick = now;
  if (document.visibilityState !== 'visible' || !UNITS.length) return;
  const s = S();
  s.elapsed_s = Math.round((s.elapsed_s + dt) * 10) / 10;
  s.seen = true;
  if (++sinceSave >= 5) { sinceSave = 0; save(); }
  document.getElementById('elapsed') && (document.getElementById('elapsed').textContent = Math.round(s.elapsed_s) + ' s');
}, 1000);
document.addEventListener('visibilitychange', () => { lastTick = performance.now(); save(); });

// --- edits ---------------------------------------------------------------------------
function selectedKeys() { return [...sel]; }
function assign(keys, val) { if (!keys.length) return; push(); for (const k of keys) S().labels[k] = val; sel.clear(); edited(); }
function nextRampId(s) {       // one past the highest ramp number in use or recorded
  let mx = 0;
  for (const v of Object.keys(s.ramps).concat(Object.values(s.labels)))
    if (typeof v === 'string' && /^r\d+$/.test(v)) mx = Math.max(mx, parseInt(v.slice(1), 10));
  return 'r' + (mx + 1);
}
function newGroup() {
  const keys = selectedKeys(); if (!keys.length) return;
  push(); const s = S(), g = nextRampId(s);
  s.ramps[g] = {placed: false};
  for (const k of keys) s.labels[k] = g;
  sel.clear(); edited();
}
function merge() {
  const u = cur(), s = S();
  let gs = [...gsel];
  if (gs.length < 2) gs = [...new Set(selectedKeys().map(k => s.labels[k]).filter(g => g && g.startsWith('r')))];
  if (gs.length < 2) { alert('Select two or more groups (click their headers) or labels in two or more groups.'); return; }
  const order = groupsOf(u);
  gs.sort((a, b) => order.indexOf(a) - order.indexOf(b));
  push();
  for (const k in s.labels) if (gs.slice(1).includes(s.labels[k])) s.labels[k] = gs[0];
  for (const g of gs.slice(1)) delete s.ramps[g];
  gsel.clear(); sel.clear(); edited();
}
function toggleComplete() {
  const u = cur(), s = S();
  if (!s.complete && unassigned(u)) { alert(unassigned(u) + ' label(s) are unassigned: give each a group, x (not a ramp) or u (unsure) first.'); return; }
  push(); s.complete = !s.complete; edited();
}
function prune(u) {              // drop ramp entries no label uses
  const s = state[u.id], used = new Set(Object.values(s.labels));
  for (const g in s.ramps) if (!used.has(g)) delete s.ramps[g];
}

// --- aerial --------------------------------------------------------------------------
function rampPos(u, g) {
  const s = state[u.id], r = s.ramps[g] || {};
  if (r.placed) return {lat: r.lat, lng: r.lng, placed: true};
  const ls = u.labels.filter(l => s.labels[l.key] === g);
  if (!ls.length) return null;
  return {lat: ls.reduce((a, l) => a + l.lat, 0) / ls.length,
          lng: ls.reduce((a, l) => a + l.lng, 0) / ls.length, placed: false};
}
function colorOf(u, g) { const i = groupsOf(u).indexOf(g); return i < 0 ? '#999' : COLORS[i % COLORS.length]; }
function drawPlan() {
  const u = cur(), s = S(), svg = document.getElementById('plan'), out = [];
  const ppm = pxPerMetre(u), c = toImg(u, u.centre.lat, u.centre.lng);
  out.push('<circle cx="' + c[0] + '" cy="' + c[1] + '" r="' + (u.window_m * ppm) + '" fill="none" stroke="#fff" stroke-dasharray="6 5" stroke-width="1.5" opacity=".8"/>');
  for (const lab of u.labels) {
    const cam = lab.camera || {};
    if (cam.lat == null) continue;
    const p = toImg(u, lab.lat, lab.lng), q = toImg(u, cam.lat, cam.lng);
    out.push('<line x1="' + q[0] + '" y1="' + q[1] + '" x2="' + p[0] + '" y2="' + p[1] + '" stroke="#fff" stroke-width=".7" opacity=".55"/>');
    const h = cam.heading_deg || 0;
    out.push('<path d="M' + q[0] + ',' + (q[1] - 6) + ' l-4,9 l8,0 z" fill="#fff" stroke="#000" stroke-width=".6" transform="rotate(' + h + ' ' + q[0] + ' ' + q[1] + ')"><title>camera ' + lab.pano_id + '</title></path>');
  }
  if (s.complete) for (const p of u.inventory || []) {
    const q = toImg(u, p.lat, p.lng);
    out.push('<rect x="' + (q[0] - 6) + '" y="' + (q[1] - 6) + '" width="12" height="12" fill="none" stroke="#fff" stroke-width="2"><title>inventory ' + (p.unit_id || '') + '</title></rect>');
  }
  for (const g of groupsOf(u)) {
    const r = rampPos(u, g); if (!r) continue;
    const q = toImg(u, r.lat, r.lng), col = colorOf(u, g);
    const pts = [];
    for (let i = 0; i < 10; i++) { const rr = i % 2 ? 4 : 10, a = Math.PI / 5 * i - Math.PI / 2; pts.push((q[0] + rr * Math.cos(a)).toFixed(1) + ',' + (q[1] + rr * Math.sin(a)).toFixed(1)); }
    out.push('<polygon points="' + pts.join(' ') + '" fill="' + (r.placed ? col : 'none') + '" stroke="' + col + '" stroke-width="2"><title>' + g + '</title></polygon>');
  }
  for (const lab of u.labels) {
    const p = toImg(u, lab.lat, lab.lng), v = s.labels[lab.key];
    const col = v && v.startsWith('r') ? colorOf(u, v) : '#bbb';
    const fill = v === 'not_ramp' ? 'none' : col;
    out.push('<circle data-key="' + lab.key + '" cx="' + p[0] + '" cy="' + p[1] + '" r="5" fill="' + fill + '" stroke="' + (sel.has(lab.key) ? '#00c2ff' : (v === 'not_ramp' ? col : '#000')) + '" stroke-width="' + (sel.has(lab.key) ? 3 : 1) + '" style="cursor:pointer"><title>' + lab.key + ' · ' + (v || 'unassigned') + '</title></circle>');
  }
  s.uncovered.forEach((p, i) => {
    const q = toImg(u, p.lat, p.lng);
    out.push('<g data-unc="' + i + '" style="cursor:pointer"><path d="M' + (q[0] - 7) + ',' + (q[1] - 7) + ' l14,14 M' + (q[0] + 7) + ',' + (q[1] - 7) + ' l-14,14" stroke="#ff9f1c" stroke-width="3.5"' + (p.unsure ? ' stroke-dasharray="3 3"' : '') + '/><circle cx="' + q[0] + '" cy="' + q[1] + '" r="9" fill="transparent"/></g>');
  });
  const bar = 10 * ppm;
  out.push('<rect x="10" y="484" width="' + (bar + 8) + '" height="20" fill="rgba(0,0,0,.55)"/><line x1="14" y1="496" x2="' + (14 + bar) + '" y2="496" stroke="#fff" stroke-width="3"/><text x="16" y="492" fill="#fff" font-size="10">10 m</text>');
  out.push('<g transform="translate(490,30)"><path d="M0,-16 l7,18 l-7,-5 l-7,5 z" fill="#fff" stroke="#000"/><text x="-4" y="16" fill="#fff" font-size="12" font-weight="bold">N</text></g>');
  svg.innerHTML = out.join('');
}
document.getElementById('plan').addEventListener('click', ev => {
  const u = cur(), s = S(), svg = document.getElementById('plan'), r = svg.getBoundingClientRect();
  const x = ev.clientX - r.left, y = ev.clientY - r.top;
  const unc = ev.target.closest('[data-unc]');
  if (mode === 'add') {
    if (unc) {
      const i = +unc.dataset.unc; push();
      if (ev.shiftKey) s.uncovered.splice(i, 1); else s.uncovered[i].unsure = !s.uncovered[i].unsure;
      edited(); return;
    }
    if (ev.shiftKey) return;
    const p = fromImg(u, x, y);
    if (haversine(p.lat, p.lng, u.centre.lat, u.centre.lng) > u.window_m) { alert('That point is outside the 30 m window.'); return; }
    push(); s.uncovered.push({lat: p.lat, lng: p.lng, unsure: false}); edited(); return;
  }
  if (mode === 'place') {
    const g = [...gsel][0]; const p = fromImg(u, x, y);
    push(); s.ramps[g] = {placed: true, lat: p.lat, lng: p.lng}; mode = null; edited(); return;
  }
  const dot = ev.target.closest('[data-key]');
  if (dot) { const k = dot.dataset.key; sel.has(k) ? sel.delete(k) : sel.add(k); render(); }
});

// --- groups + crops --------------------------------------------------------------------
function cropHtml(lab) {
  return '<div class="crop' + (sel.has(lab.key) ? ' sel' : '') + '" data-key="' + lab.key + '"><img loading="lazy" src="' + lab.crop +
    '" onerror="this.style.opacity=0.15;this.alt=\'no crop\'"><div class="cap">' + lab.key + ' · ' + lab.pano_id +
    (lab.date ? ' · ' + lab.date : '') + (lab.user_kind === 'human' ? ' · <span class="h">human</span>' : '') + '</div></div>';
}
function renderGroups() {
  const u = cur(), s = S(), right = document.getElementById('right'), out = [];
  const gs = groupsOf(u);
  const sections = gs.map((g, i) => ({id: g, title: '<b>' + (i + 1) + '</b> · ramp ' + g, col: colorOf(u, g)}));
  sections.push({id: null, title: '<b>unassigned</b>', col: '#ffd400'},
                {id: 'not_ramp', title: '<b>not a ramp</b> (x)', col: '#555'},
                {id: 'unsure', title: '<b>unsure</b> (u)', col: '#bbb'});
  for (const sec of sections) {
    const ls = u.labels.filter(l => (s.labels[l.key] || null) === sec.id);
    if (!ls.length) continue;
    const dates = [...new Set(ls.map(l => l.date).filter(Boolean))].sort().join(', ');
    const isG = sec.id && sec.id.startsWith('r');
    out.push('<div class="group' + (isG && gsel.has(sec.id) ? ' gsel' : '') + '" style="border-left-color:' + sec.col + '">' +
      '<div class="ghead"' + (isG ? ' data-group="' + sec.id + '"' : '') + '>' + sec.title + ' · ' + ls.length + ' label' + (ls.length > 1 ? 's' : '') +
      (dates ? ' · ' + dates : '') + (isG && (s.ramps[sec.id] || {}).placed ? ' · point placed' : '') + '</div><div class="crops">' +
      ls.map(cropHtml).join('') + '</div></div>');
  }
  if (!u.labels.length) out.push('<p class="meta">No labels in this window. Mark every curb ramp you can see on the aerial as uncovered (<kbd>a</kbd>), then complete the unit (<kbd>c</kbd>).</p>');
  right.innerHTML = out.join('');
}
document.getElementById('right').addEventListener('click', ev => {
  const h = ev.target.closest('[data-group]');
  if (h) { const g = h.dataset.group; gsel.has(g) ? gsel.delete(g) : gsel.add(g); render(); return; }
  const c = ev.target.closest('.crop');
  if (c) { const k = c.dataset.key; sel.has(k) ? sel.delete(k) : sel.add(k); render(); }
});

function render() {
  if (!UNITS.length) { document.getElementById('title').textContent = 'No units to review.'; return; }
  const u = cur(), s = S();
  const done = UNITS.filter(x => state[x.id].complete).length;
  document.getElementById('pos').textContent = (idx + 1) + ' / ' + UNITS.length;
  document.getElementById('progress').textContent = done + ' complete';
  document.getElementById('title').innerHTML = u.id + ' <span class="badge">' + u.type + '</span> ' +
    '<span class="badge ' + (s.complete ? 'done' : 'todo') + '">' + (s.complete ? 'complete' : 'to do') + '</span> ' +
    '<span class="meta">' + u.labels.length + ' labels · seed ' + s.seed_arm + (u.pilot ? ' · pilot' : '') +
    ' · ' + s.uncovered.length + ' uncovered · on screen <span id="elapsed">' + Math.round(s.elapsed_s) + ' s</span></span>';
  const img = document.getElementById('aerial');
  if (img.getAttribute('src') !== u.aerial.file) img.src = u.aerial.file;
  const w = document.getElementById('aerialwrap');
  w.classList.toggle('addmode', mode === 'add'); w.classList.toggle('placemode', mode === 'place');
  document.getElementById('mode').textContent = mode === 'add' ? 'ADD UNCOVERED (a to leave)' : mode === 'place' ? 'PLACE RAMP POINT' :
    (sel.size ? sel.size + ' label(s) selected' : '') + (gsel.size ? ' ' + gsel.size + ' group(s) selected' : '');
  const note = document.getElementById('unitnote');
  if (document.activeElement !== note) note.value = s.note || '';
  drawPlan(); renderGroups();
}
function go(d) { save(); idx = (idx + d + UNITS.length) % UNITS.length; sel.clear(); gsel.clear(); mode = null; lastTick = performance.now(); render(); window.scrollTo(0, 0); }
document.getElementById('prev').onclick = () => go(-1);
document.getElementById('next').onclick = () => go(1);
document.getElementById('nexttodo').onclick = () => {
  for (let k = 1; k <= UNITS.length; k++) { const j = (idx + k) % UNITS.length; if (!state[UNITS[j].id].complete) { go(j - idx); return; } }
  alert('Every unit is complete. Export when ready.');
};
document.getElementById('unitnote').addEventListener('input', ev => { S().note = ev.target.value; save(); });
document.addEventListener('keydown', ev => {
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const t = ev.target.tagName;
  if (t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT') return;
  const k = ev.key, u = cur();
  if (k === 'ArrowLeft') go(-1);
  else if (k === 'ArrowRight') go(1);
  else if (k >= '1' && k <= '9') { const g = groupsOf(u)[+k - 1]; if (g) assign(selectedKeys(), g); }
  else if (k === 'n') newGroup();
  else if (k === 'm') merge();
  else if (k === 'x') assign(selectedKeys(), 'not_ramp');
  else if (k === 'u') assign(selectedKeys(), 'unsure');
  else if (k === '0' || k === 'Backspace') assign(selectedKeys(), null);
  else if (k === 'a') { mode = mode === 'add' ? null : 'add'; render(); }
  else if (k === 'p') { if (gsel.size !== 1) { alert('Select exactly one group (click its header) first.'); return; } mode = 'place'; render(); }
  else if (k === 'P') { if (gsel.size !== 1) return; push(); S().ramps[[...gsel][0]] = {placed: false}; edited(); }
  else if (k === 'c') toggleComplete();
  else if (k === 'z') { const st = undo[u.id]; if (st && st.length) { state[u.id] = JSON.parse(st.pop()); save(); render(); } }
  else if (k === 'Escape') { sel.clear(); gsel.clear(); mode = null; render(); }
  else if (k === '?') { const h = document.getElementById('help'); h.style.display = h.style.display === 'block' ? 'none' : 'block'; }
  else return;
  ev.preventDefault();
});

// --- export --------------------------------------------------------------------------
function exportUnit(u, s) {
  prune(u);
  const labels = {}, ramps = {};
  for (const lab of u.labels) { const v = s.labels[lab.key]; if (v) labels[lab.key] = v; }
  for (const g of new Set(Object.values(labels))) {
    if (!g.startsWith('r')) continue;
    const r = rampPos(u, g);
    ramps[g] = {lat: r.lat, lng: r.lng, placed: !!r.placed};
  }
  return {seed_arm: s.seed_arm, stratum: {city: CITY, type: u.type, has_labels: u.has_labels},
          labels: labels, ramps: ramps, uncovered: s.uncovered, complete: !!s.complete,
          elapsed_s: Math.round(s.elapsed_s * 10) / 10, note: (s.note || '').trim()};
}
document.getElementById('export').onclick = () => {
  const bad = UNITS.filter(u => state[u.id].complete && unassigned(u));
  if (bad.length) { alert('Export refused: ' + bad.length + ' complete unit(s) have unassigned labels, e.g. ' + bad[0].id + '.'); return; }
  const open = UNITS.filter(u => state[u.id].seen && !state[u.id].complete).length;
  if (open && !confirm(open + ' unit(s) were opened but are not complete; the scorer ignores them. Export anyway?')) return;
  saveNotes();
  const out = {schema: 'rampnet.cluster_review/1', city: CITY, snapshot_sha256: SNAPSHOT,
               rubric_version: RUBRIC_VERSION, seed_arm: null, rater: RATER, role: ROLE,
               exported_at: new Date().toISOString()};
  const rn = {};
  for (const k of ['reviewer', 'reviewed_at', 'confidence', 'summary']) if (notes[k]) rn[k] = notes[k];
  if ((notes.caveats || []).length) rn.caveats = notes.caveats;
  out.review_notes = rn;
  out.corners = {};
  const known = new Set(UNITS.map(u => u.id));
  for (const u of UNITS) { const s = state[u.id]; if (s.seen || s.complete) out.corners[u.id] = exportUnit(u, s); }
  // units not rendered this session (another filter) round-trip from the prefill verbatim
  if (INITIAL && !boot.initialIgnored) for (const cid in (INITIAL.corners || {})) if (!known.has(cid)) out.corners[cid] = INITIAL.corners[cid];
  const arms = new Set(Object.values(out.corners).map(c => c.seed_arm));
  out.seed_arm = arms.size === 1 ? [...arms][0] : (arms.size ? 'mixed' : null);
  const blob = new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = FILE_NAME; a.click();
};

save();
render();
</script>
"""


def build_html(units, snapshot, city, rater, role, initial, file_name):
    return (HTML_TEMPLATE
            .replace('__STATE_BOOTSTRAP__', STATE_BOOTSTRAP_JS)
            .replace('__UNITS__', json.dumps(units))
            .replace('__SNAPSHOT__', json.dumps(snapshot['labels']['sha256']))
            .replace('__CITY__', json.dumps(city))
            .replace('__RATER__', json.dumps(rater))
            .replace('__ROLE__', json.dumps(role))
            .replace('__RUBRIC_V__', str(cr.RUBRIC_VERSION))
            .replace('__INITIAL__', json.dumps(initial))
            .replace('__ATTRIBUTION__', json.dumps((snapshot.get('aerial') or {})
                                                   .get('attribution', '')))
            .replace('__FILE_NAME__', json.dumps(file_name)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('bundle', type=Path, help='benchmark/<city>/cluster_review')
    ap.add_argument('--rater', default=None,
                    help='rater name; exports assignments__<rater>.json (default: rater A, '
                         'assignments.json)')
    ap.add_argument('--role', choices=('a', 'b'), default='a',
                    help='b = second rater: only double-rated units (default a)')
    ap.add_argument('--seed-arm', choices=SEED_ARMS + ('auto',), default='auto',
                    help='auto = deployed for role a, the unit\'s rater_b_seed for role b')
    ap.add_argument('--pilot', action='store_true', help='only the pilot units')
    ap.add_argument('--out', type=Path, default=None, help='gallery dir (default <bundle>/gallery)')
    ap.add_argument('--html-only', action='store_true',
                    help='accepted for parity with gt_gallery; this tool only ever writes HTML')
    args = ap.parse_args(argv)

    snapshot, corners, _files = cr.load_bundle(args.bundle)
    city = snapshot['city']
    out = (args.out or args.bundle / 'gallery')
    out.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(args.bundle.resolve(), out.resolve()).replace(os.sep, '/') + '/'
    units = select_units(corners, args.pilot, args.role)
    view = [viewer_unit(c, unit_seed_arm(c, args.seed_arm, args.role), rel) for c in units]
    initial, msg = load_prefill(args.bundle, args.rater, snapshot)
    if msg:
        print(msg)
    file_name = cr.rater_file_name(args.rater)
    (out / 'index.html').write_text(build_html(view, snapshot, city, args.rater, args.role,
                                               initial, file_name), encoding='utf-8')
    arms = {}
    for v in view:
        arms[v['seed_arm']] = arms.get(v['seed_arm'], 0) + 1
    print(f"{len(view)} units ({sum(len(v['labels']) for v in view)} labels), seed arms {arms}")
    print(f"Gallery: {out / 'index.html'}")
    print(f'Open it, review, Export, and save the download as {args.bundle / file_name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
