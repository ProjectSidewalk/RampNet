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

City inventory points appear only after the unit is marked complete; the first reveal
sets a sticky ``inventory_seen`` and any later edit ``edited_after_inventory`` (both
exported; the scorer's inventory calibration drops the latter). Each unit's ``elapsed_s``
accumulates while it is on screen, the tab is visible and there has been keyboard/mouse
input in the last 60 s (1 s ticks, each capped at 2 s). State autosaves to localStorage keyed by
bundle + rater + label-snapshot sha256; Export downloads the assignments file; an existing
file prefills for revision.

    python scripts/cluster_review_gallery.py benchmark/vancouver/cluster_review --pilot
        # rater A: deployed seed, exports assignments.json
    python scripts/cluster_review_gallery.py benchmark/vancouver/cluster_review --pilot \\
        --rater mikey --role b --out benchmark/vancouver/cluster_review/gallery/mikey
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


def role_problem(bundle, rater, role):
    """Why this rater/role pair must not run, or None. Rater B must be named (else it would
    share rater A's storage, prefill from A's file and export over it), and a named rater's
    existing export must have been made under the same role."""
    if role == 'b' and not rater:
        return ('--role b needs --rater NAME: without it the gallery would use rater A\'s '
                'storage, prefill from assignments.json and export over it')
    path = Path(bundle) / cr.rater_file_name(rater)
    if path.exists():
        got = json.loads(path.read_text(encoding='utf-8')).get('role') or 'a'
        if got != role:
            return f'{path.name} was exported under --role {got}; refusing to open it as role {role}'
    return None


def corners_sha256(bundle):
    return hashlib.sha256((Path(bundle) / 'corners.jsonl').read_bytes()).hexdigest()


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
#   * local state wins per UNIT only when the reviewer has worked on it in this browser
#     (seen or complete); a unit the browser merely seeded takes the file's state (review
#     fix: a later-placed assignments file used to be silently shadowed by seeded state),
#     and every unit where local work shadows a differing file entry is reported;
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
          complete: false, cant_judge: false, cant_judge_reason: '',
          elapsed_s: 0, note: '', seen: false, nextRamp: n + 1};
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
          complete: !!f.complete, cant_judge: !!f.cant_judge,
          cant_judge_reason: f.cant_judge_reason || '',
          elapsed_s: f.elapsed_s || 0, note: f.note || '',
          inventory_seen: !!f.inventory_seen,
          edited_after_inventory: !!f.edited_after_inventory,
          seen: true, nextRamp: mx + 1};
}
function bootstrapState(INITIAL, local, UNITS, SNAPSHOT) {
  const state = local || {};
  let prefilled = 0, reopened = 0, initialIgnored = false;
  const conflicts = [];
  if (INITIAL && INITIAL.snapshot_sha256 !== SNAPSHOT) initialIgnored = true;
  else if (INITIAL) {
    for (const cid in (INITIAL.corners || {})) {
      const s = state[cid], f = INITIAL.corners[cid];
      if (s && (s.seen || s.complete)) {
        const same = JSON.stringify(s.labels) === JSON.stringify(f.labels || {}) &&
                     !!s.complete === !!f.complete && !!s.cant_judge === !!f.cant_judge &&
                     (s.uncovered || []).length === (f.uncovered || []).length;
        if (!same) conflicts.push(cid);
        continue;
      }
      state[cid] = fromFile(f);
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
  return {state: state, prefilled: prefilled, reopened: reopened, initialIgnored: initialIgnored,
          conflicts: conflicts};
}
"""

HTML_TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>RampNet cluster review</title>
<style>
  :root{--aw:min(860px, calc(100vh - 96px))}
  body{font-family:sans-serif;margin:8px 16px;background:#fafafa;color:#222}
  .bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px}
  .bar button{font-size:14px;padding:5px 12px;cursor:pointer}
  .meta{color:#666;font-size:13px}
  .badge{font-size:12px;padding:2px 8px;border-radius:10px;background:#eee;color:#444}
  .badge.done{background:#1a9c3e;color:#fff}
  .badge.todo{background:#fff3bf;color:#7a6000}
  .badge.cj{background:#6c757d;color:#fff}
  #notice{display:none;background:#fff2df;border:1px solid #ff9f1c;border-radius:8px;
          padding:8px 14px;margin:0 0 10px;font-size:13px}
  #rulebar{font-size:12px;color:#666;margin:0 0 6px}
  #unitsel{font-size:13px;padding:3px}
  #main{display:flex;gap:16px;align-items:flex-start}
  #left{flex:0 0 auto;width:var(--aw);position:sticky;top:6px}
  #aerialwrap{position:relative;width:var(--aw);height:var(--aw);background:#222;border-radius:6px;
              overflow:hidden}
  #aerialwrap img{position:absolute;left:0;top:0;width:100%;height:100%}
  #aerialwrap svg{position:absolute;left:0;top:0;width:100%;height:100%}
  #aerialwrap.addmode{cursor:crosshair;outline:3px solid #ff9f1c}
  #aerialwrap.placemode{cursor:crosshair;outline:3px solid #00c2ff}
  #attrib{font-size:11px;color:#666;margin:3px 0 8px}
  #right{flex:1;min-width:0;display:flex;flex-wrap:wrap;gap:8px;align-content:flex-start}
  .group{background:#fff;border:1px solid #ddd;border-left:8px solid #999;border-radius:6px;
         padding:6px 8px;flex:0 1 auto;max-width:100%;box-sizing:border-box}
  .group.gsel{box-shadow:0 0 0 3px #00c2ff}
  .ghead{font-size:13px;cursor:pointer;user-select:none;margin-bottom:4px}
  .ghead b{font-size:14px}
  .crops{display:flex;flex-wrap:wrap;gap:6px}
  .crop{width:160px;cursor:pointer;border:3px solid transparent;border-radius:4px;background:#eee}
  .crop.sel{border-color:#00c2ff}
  .crop.hov{border-color:#ff9f1c}
  .crop.flash{animation:flash 1.2s ease-out}
  @keyframes flash{0%{box-shadow:0 0 0 8px #ff9f1c}100%{box-shadow:0 0 0 0 rgba(255,159,28,0)}}
  .crop img{width:160px;height:160px;display:block}
  .crop .cap{font-size:11px;color:#555;padding:1px 3px}
  .crop{position:relative}
  .crop img{-webkit-user-drag:none}
  .acts{position:absolute;left:2px;right:2px;display:none;flex-wrap:wrap;gap:2px}
  .crop:hover .acts{display:flex}
  .acts.top{top:2px}
  .acts.bot{top:134px}
  .acts button{font:bold 11px sans-serif;padding:1px 5px;min-width:20px;border:1px solid #000;
               border-radius:3px;cursor:pointer;background:#fff;color:#000;opacity:.92}
  .acts button:hover{opacity:1;outline:2px solid #00c2ff}
  .drop{outline:3px dashed #00c2ff !important;outline-offset:-3px}
  #newzone{border:2px dashed #aaa;border-radius:6px;color:#777;font-size:13px;padding:18px 14px;
           align-self:stretch;display:flex;align-items:center}
  #shelf{position:fixed;left:0;right:0;bottom:0;z-index:15;background:#fffbe6;
         border-top:3px solid #ffd400;padding:4px 16px 6px;max-height:42vh;overflow:auto;
         box-shadow:0 -3px 10px rgba(0,0,0,.12)}
  #shelf .shead{font-size:13px;margin-bottom:4px}
  #shelf.min #shelfcrops{display:none}
  .h{color:#b35c00;font-weight:bold}
  #pop{position:fixed;display:none;pointer-events:none;z-index:30;background:#fff;
       border:1px solid #888;border-radius:6px;padding:4px;box-shadow:0 4px 18px rgba(0,0,0,.3)}
  #pop img{width:360px;height:360px;display:block}
  #pop .cap{font-size:12px;padding:3px 2px 0;max-width:360px}
  #pop .grid{display:grid;gap:3px}
  #pop .grid img{width:130px;height:130px}
  .ghead button{font-size:11px;padding:0 6px;margin-left:6px;cursor:pointer}
  .ghead[draggable=true]{cursor:grab}
  .group.flash{animation:flash 1.2s ease-out}
  #gcard{position:fixed;display:none;z-index:31;background:#fff;border:1px solid #888;
         border-left:8px solid #999;border-radius:6px;padding:6px 8px;
         box-shadow:0 4px 18px rgba(0,0,0,.3);max-height:80vh;overflow:auto}
  #gcard .ghd{font-size:13px;margin-bottom:5px}
  #gcard .ghd button{font-size:11px;padding:0 6px;margin-left:6px;cursor:pointer}
  #gcard .crops{display:grid;gap:6px}
  #lb{display:none;position:fixed;inset:0;z-index:40;background:rgba(0,0,0,.82);
      align-items:center;justify-content:center;gap:14px;padding:20px;box-sizing:border-box}
  #lb .big{display:flex;flex-direction:column;align-items:center}
  #lb .big img{width:min(78vh,900px);height:min(78vh,900px);border-radius:6px;background:#333}
  #lb .cap{color:#eee;font-size:14px;margin-top:6px;text-align:center}
  #lb .side{display:flex;flex-direction:column;gap:6px;max-height:90vh;overflow:auto}
  #lb .side img{width:150px;height:150px;border:3px solid transparent;border-radius:4px;cursor:pointer}
  #lb .side img.on{border-color:#ff9f1c}
  #lb .side .t{color:#ccc;font-size:12px}
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
<div id="rulebar"><b>Rubric v__RUBRIC_V__</b>: one group per physical ramp · dual-direction
  apron = two · driveway = not a ramp · missing ramp = uncovered · <kbd>?</kbd> keys, legend and
  full rubric</div>

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
  <button id="nexttodo">Next to do</button>
  <button id="cantjudge" title="the unit cannot be judged (e.g. hidden by trees, construction, no imagery); needs a reason">Can't judge…</button>
  <select id="unitsel" title="jump to a unit"></select>
  <span id="progress" class="meta"></span>
  <span style="flex:1"></span>
  <span id="mode" class="badge"></span>
  <button id="export">Export</button>
</div>
<h2 id="title" style="margin:4px 0 8px;font-size:17px"></h2>
<div id="main">
  <div id="left">
    <div id="aerialwrap"><img id="aerial" alt=""><svg id="plan" width="512" height="512"></svg></div>
    <div id="attrib"></div>
    <p><textarea id="unitnote" rows="2" placeholder="Note on this unit (optional)"></textarea></p>
  </div>
  <div id="right"></div>
</div>
<div id="shelf"><div class="shead"><b>Shelf</b> <span id="shelfn"></span> · labels still to
  decide (hover a crop for buttons, or drag it onto a group) <button id="shelftoggle">hide</button></div>
  <div class="crops" id="shelfcrops"></div></div>
<div id="pop"><img alt=""><div class="cap"></div></div>
<div id="gcard"></div>
<div id="lb"></div>
<div id="help">
  <b>Rubric v__RUBRIC_V__</b> (RUBRICS.md §6): one group per physical curb ramp; a
  dual-direction apron is <b>two</b> ramps; a driveway is <b>not a ramp</b>; a ramp no label
  covers gets an <b>uncovered</b> point; <b>unsure</b> abstains. Complete a unit only when every
  label is assigned and every uncovered ramp in the window is marked.<br><br>
  <b>Aerial</b>: dots = labels (colour = group; blue ring = selected; hollow = not ramp; grey =
  unsure / unassigned) · hover a dot = its crop, camera and ray · star = ramp point (hollow =
  mean of its labels, filled = placed) · orange X = uncovered (dashed = unsure) · dashed circle =
  30 m window · white squares = city inventory (after complete)<br><br>
  <b>Keys</b><br>
  hover a dot: see its crop · hover a crop or group header: see it on the aerial<br>
  click crop / dot: select label (again to deselect) · click group header: select group<br>
  <kbd>r</kbd> show / hide every camera and ray<br>
  double-click a crop (or <kbd>Space</kbd> over it): big view, with the rest of its group beside
  it (<kbd>←</kbd>/<kbd>→</kbd> to step, <kbd>Esc</kbd> to close)<br>
  star (a group's ramp position): hover = a card with all its crops and their buttons (move any
  of them to another group from there) · click = go to the group · drag = move the position
  (optional; used only for the inventory comparison) · double-click, <b>reset position</b> (card or
  group header) or <kbd>P</kbd> puts it back at the mean of its labels<br>
  <kbd>z</kbd> / <kbd>Ctrl</kbd>+<kbd>Z</kbd> undo · <kbd>Ctrl</kbd>+<kbd>Y</kbd> /
  <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd> redo (per unit)<br>
  group header: drag onto another group = merge · <b>not a ramp</b> = the whole group is a false
  positive<br>
  hover a crop for buttons: <b>1</b>..<b>N</b> move to that group · <b>+ new</b> a ramp of its own ·
  <b>shelf</b> not this group, decide later · <b>not ramp</b> · <b>unsure</b> (final); a selected
  crop's buttons act on the whole selection. Or drag crops onto a group, the new-group zone or the
  shelf.<br>
  <kbd>1</kbd>-<kbd>9</kbd> assign selected labels to group N · <kbd>n</kbd> new group from
  selection (split) · <kbd>m</kbd> merge selected groups (or the groups of the selected
  labels)<br>
  <kbd>x</kbd> not a ramp · <kbd>u</kbd> unsure · <kbd>0</kbd> unassign · <kbd>Esc</kbd> clear
  selection / mode<br>
  <kbd>a</kbd> add-uncovered mode: click aerial = add, click a mark = toggle unsure,
  shift-click = remove<br>
  <kbd>p</kbd> place the selected group's ramp point (next aerial click) · <kbd>P</kbd> reset
  it to the mean<br>
  <kbd>c</kbd> complete / reopen · <b>Can't judge…</b> (button): the unit cannot be judged at all
  (hidden by trees, construction, no imagery), with a required reason; excluded from scoring and
  listed; never for a unit that is merely hard · <kbd>&#8592;</kbd>/<kbd>&#8594;</kbd>
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
const CORNERS_SHA = __CORNERS_SHA__;    // sha256 of corners.jsonl: another sample never collides
const STORE = 'clusterreview:' + CITY + ':' + (RATER || 'A') + ':' + SNAPSHOT + ':' + CORNERS_SHA;
const NSTORE = 'clusterreviewnotes:' + CITY + ':' + (RATER || 'A') + ':' + SNAPSHOT + ':' + CORNERS_SHA;
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
  if (boot.conflicts.length) bits.push('<b>' + boot.conflicts.length + ' unit(s) kept this browser&#39;s work over a different entry in the assignments file</b>: ' + boot.conflicts.join(', ') + '. Clear this page&#39;s site data to take the file instead.');
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
let idx = 0, sel = new Set(), gsel = new Set(), mode = null, undo = {}, redo = {};
function cur() { return UNITS[idx]; }
function S() { return state[cur().id]; }
function groupsOf(u) {           // ramp groups in display order (by first label)
  const s = state[u.id], order = [];
  for (const lab of u.labels) { const g = s.labels[lab.key]; if (g && g.startsWith('r') && !order.includes(g)) order.push(g); }
  return order;
}
function unassigned(u) { const s = state[u.id]; return u.labels.filter(l => !s.labels[l.key]).length; }
const UNDO_FIELDS = ['labels', 'ramps', 'uncovered', 'complete', 'cant_judge', 'cant_judge_reason'];   // never elapsed_s or note
function push() {                // snapshot for undo, before an edit
  const id = cur().id, s = state[id], snap = {};
  for (const f of UNDO_FIELDS) snap[f] = s[f];
  (undo[id] = undo[id] || []).push(JSON.stringify(snap));
  if (undo[id].length > 100) undo[id].shift();
  redo[id] = [];                   // a new edit forks history: nothing left to redo
  if (s.inventory_seen) s.edited_after_inventory = true;   // an edit after the inventory showed
}
function edited() { S().seen = true; save(); render(); }
function snapOf(s) { const snap = {}; for (const f of UNDO_FIELDS) snap[f] = s[f]; return JSON.stringify(snap); }
function stepHistory(from, to) {  // undo: from = undo, to = redo; redo: the reverse
  const id = cur().id, st = from[id];
  if (!st || !st.length) return;
  const s = state[id];
  (to[id] = to[id] || []).push(snapOf(s));
  const snap = JSON.parse(st.pop());
  for (const f of UNDO_FIELDS) s[f] = snap[f];
  if (s.inventory_seen) s.edited_after_inventory = true;
  sel.clear(); save(); render();
}
function doUndo() { stepHistory(undo, redo); }
// --- big view: double-click a crop (or Space over it); its group alongside to compare ------
let lbKey = null;
function openLightbox(key) {
  const u = cur(), s = S(), lab = labOf(u, key); if (!lab) return;
  lbKey = key; hidePop(); closeCard();
  const v = s.labels[key] || null;
  const mates = u.labels.filter(l => (s.labels[l.key] || null) === v);
  const samePano = u.labels.filter(l => l.pano_id === lab.pano_id).length;
  const lb = document.getElementById('lb');
  lb.innerHTML = '<div class="big"><img src="' + lab.crop + '"><div class="cap"><b>' + groupName(u, v) + '</b>' +
    (lab.date ? ' · captured ' + lab.date : '') + (lab.user_kind === 'human' ? ' · <span class="h">human</span>' : '') +
    ' · pano ' + lab.pano_id + (samePano > 1 ? ' (' + samePano + ' labels from it here)' : '') +
    '<br><span style="color:#aaa">←/→ or click a thumbnail: the rest of this group · Esc or click outside: close</span></div></div>' +
    (mates.length > 1 ? '<div class="side">' + mates.map(l => '<div><img data-lb="' + l.key + '" class="' + (l.key === key ? 'on' : '') +
      '" src="' + l.crop + '"><div class="t">' + (l.date || '') + '</div></div>').join('') + '</div>' : '');
  lb.style.display = 'flex';
  setHover(key, null);
}
function closeLightbox() { lbKey = null; document.getElementById('lb').style.display = 'none'; setHover(null, null); }
function stepLightbox(d) {
  const u = cur(), s = S(), v = s.labels[lbKey] || null;
  const mates = u.labels.filter(l => (s.labels[l.key] || null) === v);
  const i = mates.findIndex(l => l.key === lbKey);
  openLightbox(mates[(i + d + mates.length) % mates.length].key);
}
document.getElementById('lb').addEventListener('click', ev => {
  const t = ev.target.closest('[data-lb]');
  if (t) { openLightbox(t.dataset.lb); return; }
  if (!ev.target.closest('.big img')) closeLightbox();
});
document.addEventListener('dblclick', ev => {
  const c = ev.target.closest && ev.target.closest('.crop');
  if (c && !ev.target.closest('.acts')) { ev.preventDefault(); openLightbox(c.dataset.key); }
});
function doRedo() { stepHistory(redo, undo); }

// --- timing: elapsed_s while shown and visible ------------------------------------------
const IDLE_S = 60;                // no keyboard/mouse input for this long pauses the clock
let lastTick = performance.now(), sinceSave = 0, lastInput = performance.now();
for (const ev of ['keydown', 'mousedown', 'mousemove', 'wheel', 'scroll', 'touchstart'])
  window.addEventListener(ev, () => { lastInput = performance.now(); }, {passive: true, capture: true});
setInterval(() => {
  const now = performance.now(), dt = Math.min((now - lastTick) / 1000, 2);
  lastTick = now;
  if (document.visibilityState !== 'visible' || !UNITS.length) return;
  if (now - lastInput > IDLE_S * 1000) return;
  const s = S();
  s.elapsed_s = Math.round((s.elapsed_s + dt) * 10) / 10;
  s.seen = true;
  if (++sinceSave >= 5) { sinceSave = 0; save(); }
  document.getElementById('elapsed') && (document.getElementById('elapsed').textContent = Math.round(s.elapsed_s) + ' s');
}, 1000);
document.addEventListener('visibilitychange', () => {
  lastTick = performance.now();
  if (document.visibilityState === 'hidden') save();
});
window.addEventListener('pagehide', () => save());

// --- edits ---------------------------------------------------------------------------
function selectedKeys() { return [...sel]; }
function assign(keys, val) { if (!keys.length) return; push(); for (const k of keys) S().labels[k] = val; sel.clear(); edited(); }
function nextRampId(s) {       // one past the highest ramp number in use or recorded
  let mx = 0;
  for (const v of Object.keys(s.ramps).concat(Object.values(s.labels)))
    if (typeof v === 'string' && /^r\d+$/.test(v)) mx = Math.max(mx, parseInt(v.slice(1), 10));
  return 'r' + (mx + 1);
}
function newGroup(keys) {
  keys = keys || selectedKeys(); if (!keys.length) return;
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
  if (!s.complete && s.cant_judge) { alert('This unit is marked can\'t judge. Clear that first (Can\'t judge… button).'); return; }
  if (!s.complete && unassigned(u)) { alert(unassigned(u) + ' label(s) are still on the shelf: give each a group, not ramp or unsure first.'); return; }
  push(); s.complete = !s.complete; edited();
}
function toggleCantJudge() {
  const s = S();
  if (s.cant_judge) {
    if (!confirm('Clear "can\'t judge" for this unit? (reason: ' + s.cant_judge_reason + ')')) return;
    push(); s.cant_judge = false; s.cant_judge_reason = ''; edited(); return;
  }
  const r = prompt('Why can\'t this unit be judged? (required; e.g. "ramps hidden by trees on the aerial and in every crop")\n' +
                   'Not for a unit that is merely hard: that one still gets reviewed.', '');
  if (r === null) return;
  if (!r.trim()) { alert('A reason is required.'); return; }
  push(); s.cant_judge = true; s.cant_judge_reason = r.trim(); s.complete = false; edited();
}
function done(x) { const t = state[x.id]; return t.complete || t.cant_judge; }
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
const MIN_SEP_M = 1.0;           // rampnet.cluster_review.UNCOVERED_MIN_SEP_M
function nearRamp(u, lat, lng) {
  for (const g of groupsOf(u)) { const r = rampPos(u, g); if (r && haversine(r.lat, r.lng, lat, lng) < MIN_SEP_M) return g; }
  return null;
}
function colorOf(u, g) { const i = groupsOf(u).indexOf(g); return i < 0 ? '#999' : COLORS[i % COLORS.length]; }
function drawPlan() {
  const u = cur(), s = S(), svg = document.getElementById('plan'), out = [];
  const ppm = pxPerMetre(u), c = toImg(u, u.centre.lat, u.centre.lng);
  svg.setAttribute('viewBox', '0 0 ' + u.aerial.px + ' ' + u.aerial.px);
  out.push('<circle cx="' + c[0] + '" cy="' + c[1] + '" r="' + (u.window_m * ppm) + '" fill="none" stroke="#fff" stroke-dasharray="6 5" stroke-width="1.5" opacity=".8"/>');
  // cameras + rays only on demand (r), for selected labels, or on hover (drawHover)
  for (const lab of u.labels) if (showRays || sel.has(lab.key)) out.push(rayHtml(u, lab, sel.has(lab.key)));
  if (s.complete && (u.inventory || []).length && !s.inventory_seen) { s.inventory_seen = true; save(); }
  if (s.complete) for (const p of u.inventory || []) {
    const q = toImg(u, p.lat, p.lng);
    out.push('<rect x="' + (q[0] - 6) + '" y="' + (q[1] - 6) + '" width="12" height="12" fill="none" stroke="#fff" stroke-width="2"><title>inventory ' + (p.unit_id || '') + '</title></rect>');
  }
  for (const g of groupsOf(u)) {
    const r = (starDrag && starDrag.g === g && starDrag.pos) ? Object.assign({placed: true}, starDrag.pos) : rampPos(u, g);
    if (!r) continue;
    const q = toImg(u, r.lat, r.lng), col = colorOf(u, g);
    const pts = [];
    for (let i = 0; i < 10; i++) { const rr = i % 2 ? 4 : 10, a = Math.PI / 5 * i - Math.PI / 2; pts.push((q[0] + rr * Math.cos(a)).toFixed(1) + ',' + (q[1] + rr * Math.sin(a)).toFixed(1)); }
    out.push('<polygon data-star="' + g + '" points="' + pts.join(' ') + '" fill="' + (r.placed ? col : 'rgba(0,0,0,.25)') + '" stroke="' + col + '" stroke-width="' + (gsel.has(g) ? 4 : 2.5) + '" style="cursor:grab"/>');
  }
  for (const lab of u.labels) {
    const p = toImg(u, lab.lat, lab.lng), v = s.labels[lab.key];
    const col = v && v.startsWith('r') ? colorOf(u, v) : '#bbb';
    const fill = v === 'not_ramp' ? 'none' : col;
    out.push('<circle data-key="' + lab.key + '" cx="' + p[0] + '" cy="' + p[1] + '" r="5" fill="' + fill + '" stroke="' + (sel.has(lab.key) ? '#00c2ff' : (v === 'not_ramp' ? col : '#000')) + '" stroke-width="' + (sel.has(lab.key) ? 3 : 1) + '" style="cursor:pointer"/>');
  }
  s.uncovered.forEach((p, i) => {
    const q = toImg(u, p.lat, p.lng);
    out.push('<g data-unc="' + i + '" style="cursor:pointer"><path d="M' + (q[0] - 7) + ',' + (q[1] - 7) + ' l14,14 M' + (q[0] + 7) + ',' + (q[1] - 7) + ' l-14,14" stroke="#ff9f1c" stroke-width="3.5"' + (p.unsure ? ' stroke-dasharray="3 3"' : '') + '/><circle cx="' + q[0] + '" cy="' + q[1] + '" r="9" fill="transparent"/></g>');
  });
  const bar = 10 * ppm;
  out.push('<rect x="10" y="484" width="' + (bar + 8) + '" height="20" fill="rgba(0,0,0,.55)"/><line x1="14" y1="496" x2="' + (14 + bar) + '" y2="496" stroke="#fff" stroke-width="3"/><text x="16" y="492" fill="#fff" font-size="10">10 m</text>');
  out.push('<g transform="translate(490,30)"><path d="M0,-16 l7,18 l-7,-5 l-7,5 z" fill="#fff" stroke="#000"/><text x="-4" y="16" fill="#fff" font-size="12" font-weight="bold">N</text></g>');
  out.push('<g id="hoverlayer" pointer-events="none"></g>');
  svg.innerHTML = out.join('');
}
function rayHtml(u, lab, strong) {   // camera triangle + its ray to the label
  const cam = lab.camera || {};
  if (cam.lat == null) return '';
  const p = toImg(u, lab.lat, lab.lng), q = toImg(u, cam.lat, cam.lng), h = cam.heading_deg || 0;
  return '<line x1="' + q[0] + '" y1="' + q[1] + '" x2="' + p[0] + '" y2="' + p[1] + '" stroke="' + (strong ? '#ff9f1c' : '#fff') +
    '" stroke-width="' + (strong ? 2 : .7) + '" opacity="' + (strong ? .95 : .55) + '"/>' +
    '<path d="M' + q[0] + ',' + (q[1] - 7) + ' l-5,11 l10,0 z" fill="' + (strong ? '#ff9f1c' : '#fff') + '" stroke="#000" stroke-width=".7" transform="rotate(' + h + ' ' + q[0] + ' ' + q[1] + ')"><title>camera ' + lab.pano_id + '</title></path>';
}

// --- hover linking: aerial dot <-> crop, group header -> its dots -----------------------
let hoverKey = null, hoverGroup = null, showRays = false;
function labOf(u, key) { return u.labels.find(l => l.key === key); }
function hoveredKeys() {
  const u = cur(), s = S();
  if (hoverKey) return [hoverKey];
  if (hoverGroup) return u.labels.filter(l => s.labels[l.key] === hoverGroup).map(l => l.key);
  return [];
}
function drawHover() {
  const u = cur(), layer = document.getElementById('hoverlayer');
  const keys = hoveredKeys(), out = [];
  for (const k of keys) {
    const lab = labOf(u, k); if (!lab) continue;
    out.push(rayHtml(u, lab, true));
    const p = toImg(u, lab.lat, lab.lng);
    out.push('<circle cx="' + p[0] + '" cy="' + p[1] + '" r="10" fill="none" stroke="#ff9f1c" stroke-width="3"/>');
  }
  if (layer) layer.innerHTML = out.join('');
  document.querySelectorAll('.crop.hov').forEach(e => e.classList.remove('hov'));
  for (const k of keys) document.querySelectorAll('.crop[data-key="' + CSS.escape(k) + '"]').forEach(e => e.classList.add('hov'));
}
function cropEl(key) { return document.querySelector('.crop[data-key="' + CSS.escape(key) + '"]'); }
function groupName(u, v) {
  if (!v) return 'unassigned';
  if (v === 'not_ramp' || v === 'unsure') return v.replace('_', ' ');
  return 'group ' + (groupsOf(u).indexOf(v) + 1);
}
function showPop(key, ev) {
  const u = cur(), lab = labOf(u, key), pop = document.getElementById('pop');
  if (!lab) { pop.style.display = 'none'; return; }
  const samePano = u.labels.filter(l => l.pano_id === lab.pano_id).length;
  pop.innerHTML = '<img src="' + lab.crop + '"><div class="cap"><b>' + groupName(u, S().labels[key]) + '</b>' + (lab.date ? ' · ' + lab.date : '') +
    (lab.user_kind === 'human' ? ' · <span class="h">human</span>' : '') + ' · pano ' + lab.pano_id.slice(0, 10) + '…' +
    (samePano > 1 ? ' (' + samePano + ' labels from it here)' : '') + '</div>';
  placePop(ev);
}
// Hovering a star opens an interactive card with every crop of its group (with the same
// buttons as the crops on the right). It stays open while the pointer is on the star or the
// card and closes a moment after it leaves both.
let cardGroup = null, cardHideT = null;
const CARD_HIDE_MS = 400;
function cardHtml(g) {
  const u = cur(), s = S(), ls = u.labels.filter(l => s.labels[l.key] === g);
  const cols = Math.min(4, Math.max(1, ls.length)), r = s.ramps[g] || {};
  return '<div class="ghd"><b>' + groupName(u, g) + '</b> · ' + ls.length + ' label' + (ls.length > 1 ? 's' : '') + ' · ' +
    (r.placed ? 'position placed by you' : 'position = mean of its labels') +
    (r.placed ? '<button data-cact="reset" title="back to the mean of its labels">reset position</button>' : '') +
    '<button data-cact="notramp" title="every label in this group is a false positive">not a ramp</button>' +
    '<button data-cact="goto">go to group</button></div>' +
    '<div class="crops" style="grid-template-columns:repeat(' + cols + ',166px)">' + ls.map(cropHtml).join('') + '</div>';
}
function openCard(g, starEl) {
  cancelCardHide(); hidePop();
  const card = document.getElementById('gcard'), u = cur();
  if (!u.labels.some(l => S().labels[l.key] === g)) { closeCard(); return; }
  const fresh = cardGroup !== g;
  cardGroup = g;
  card.style.borderLeftColor = colorOf(u, g);
  card.innerHTML = cardHtml(g);
  card.style.display = 'block';
  if (fresh && starEl) {          // beside the star; never moves while it stays open
    const b = starEl.getBoundingClientRect(), W = card.offsetWidth, H = card.offsetHeight;
    let x = b.right + 10, y = b.top + b.height / 2 - H / 2;
    if (x + W > window.innerWidth - 8) x = b.left - W - 10;
    y = Math.max(8, Math.min(y, window.innerHeight - H - 8));
    card.style.left = Math.max(8, x) + 'px'; card.style.top = y + 'px';
  }
}
function refreshCard() { if (cardGroup) openCard(cardGroup, null); }
function closeCard() { cancelCardHide(); cardGroup = null; document.getElementById('gcard').style.display = 'none'; }
function scheduleCardHide() { if (cardGroup && !cardHideT) cardHideT = setTimeout(() => { cardHideT = null; closeCard(); setHover(null, null); }, CARD_HIDE_MS); }
function cancelCardHide() { if (cardHideT) { clearTimeout(cardHideT); cardHideT = null; } }
(function () {
  const card = document.getElementById('gcard');
  card.addEventListener('mouseenter', () => { cancelCardHide(); if (cardGroup) setHover(null, cardGroup); });
  card.addEventListener('mouseleave', scheduleCardHide);
  card.addEventListener('mouseover', ev => { const c = ev.target.closest('.crop'); setHover(c ? c.dataset.key : null, c ? null : cardGroup); });
  card.addEventListener('click', ev => {
    const b = ev.target.closest('[data-cact]'), g = cardGroup;
    if (b) {
      const u = cur(), s = S();
      if (b.dataset.cact === 'reset') resetStar(g);
      else if (b.dataset.cact === 'notramp') applyTo(u.labels.filter(l => s.labels[l.key] === g).map(l => l.key), 'not_ramp');
      else { closeCard(); gsel.clear(); gsel.add(g); render(); revealGroup(g); }
      return;
    }
    cropClick(ev);
  });
})();
function placePop(ev) {
  const pop = document.getElementById('pop');
  pop.style.display = 'block';
  const W = pop.offsetWidth, H = pop.offsetHeight;
  let x = ev.clientX + 24, y = ev.clientY - H / 2;
  if (x + W > window.innerWidth - 8) x = ev.clientX - W - 24;
  y = Math.max(8, Math.min(y, window.innerHeight - H - 8));
  pop.style.left = x + 'px'; pop.style.top = y + 'px';
}
function hidePop() { document.getElementById('pop').style.display = 'none'; }
function setHover(key, group) {
  if (key === hoverKey && group === hoverGroup) return;
  hoverKey = key; hoverGroup = group; drawHover();
}
document.getElementById('plan').addEventListener('mousemove', ev => {
  if (starDrag) return;
  const dot = mode ? null : ev.target.closest('[data-key]');
  const star = mode || dot ? null : ev.target.closest('[data-star]');
  if (dot) { setHover(dot.dataset.key, null); showPop(dot.dataset.key, ev); }
  else if (star) { setHover(null, star.dataset.star); openCard(star.dataset.star, star); return; }
  else { if (!cardGroup) setHover(null, null); hidePop(); }
  scheduleCardHide();
});

// --- stars: click = go to the group, drag = place the ramp position ----------------------
let starDrag = null, starJustUsed = false;
function svgPoint(ev) {
  const u = cur(), r = document.getElementById('plan').getBoundingClientRect(), sc = u.aerial.px / r.width;
  return fromImg(u, (ev.clientX - r.left) * sc, (ev.clientY - r.top) * sc);
}
document.getElementById('plan').addEventListener('mousedown', ev => {
  if (mode || ev.button !== 0) return;
  const star = ev.target.closest('[data-star]'); if (!star) return;
  if (ev.target.closest('[data-key]')) return;       // a dot on top of the star wins
  ev.preventDefault(); hidePop(); closeCard();
  starDrag = {g: star.dataset.star, x: ev.clientX, y: ev.clientY, moved: false, pos: null};
});
window.addEventListener('mousemove', ev => {
  if (!starDrag) return;
  if (!starDrag.moved && Math.hypot(ev.clientX - starDrag.x, ev.clientY - starDrag.y) < 4) return;
  starDrag.moved = true; starDrag.pos = svgPoint(ev); drawPlan(); drawHover();
});
window.addEventListener('mouseup', ev => {
  if (!starDrag) return;
  const d = starDrag; starDrag = null; starJustUsed = true;
  const u = cur(), s = S();
  if (!d.moved) { gsel.clear(); gsel.add(d.g); sel.clear(); render(); revealGroup(d.g); return; }
  const p = d.pos;
  if (s.uncovered.some(q => haversine(q.lat, q.lng, p.lat, p.lng) < MIN_SEP_M)) {
    alert('That point is within ' + MIN_SEP_M + ' m of an uncovered mark; remove the mark first (a, then shift-click).'); render(); return;
  }
  push(); s.ramps[d.g] = {placed: true, lat: p.lat, lng: p.lng}; edited();
});
function revealGroup(g) {
  const e = document.querySelector('.group[data-gid="' + g + '"]'); if (!e) return;
  e.scrollIntoView({block: 'nearest', behavior: 'smooth'});
  e.classList.remove('flash'); void e.offsetWidth; e.classList.add('flash');
}
function resetStar(g) { push(); S().ramps[g] = {placed: false}; edited(); }
document.getElementById('plan').addEventListener('mouseleave', () => { if (!cardGroup) setHover(null, null); hidePop(); scheduleCardHide(); });
document.getElementById('plan').addEventListener('dblclick', ev => {   // double-click a star: reset it
  const star = ev.target.closest('[data-star]'); if (!star || mode) return;
  const g = star.dataset.star; if ((S().ramps[g] || {}).placed) resetStar(g);
});
document.getElementById('right').addEventListener('mouseover', ev => {
  const c = ev.target.closest('.crop'), h = ev.target.closest('[data-group]');
  setHover(c ? c.dataset.key : null, h ? h.dataset.group : null);
});
document.getElementById('right').addEventListener('mouseleave', () => setHover(null, null));
function revealCrop(key) {        // scroll a clicked dot's crop into view and flash it
  const e = cropEl(key); if (!e) return;
  e.scrollIntoView({block: 'nearest', behavior: 'smooth'});
  e.classList.remove('flash'); void e.offsetWidth; e.classList.add('flash');
}
document.getElementById('plan').addEventListener('click', ev => {
  const u = cur(), s = S(), svg = document.getElementById('plan'), r = svg.getBoundingClientRect();
  if (starJustUsed) { starJustUsed = false; return; }
  const sc = u.aerial.px / r.width;   // the aerial is drawn larger than its pixel grid
  const x = (ev.clientX - r.left) * sc, y = (ev.clientY - r.top) * sc;
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
    const near = nearRamp(u, p.lat, p.lng);
    if (near) { alert('That point is within ' + MIN_SEP_M + ' m of ramp ' + near + ': it is that ramp, not an uncovered one. Not added.'); return; }
    push(); s.uncovered.push({lat: p.lat, lng: p.lng, unsure: false}); edited(); return;
  }
  if (mode === 'place') {
    const g = [...gsel][0]; const p = fromImg(u, x, y);
    const clash = s.uncovered.findIndex(q => haversine(q.lat, q.lng, p.lat, p.lng) < MIN_SEP_M);
    if (clash >= 0) { alert('That point is within ' + MIN_SEP_M + ' m of an uncovered mark; remove the mark first (a, then shift-click).'); mode = null; render(); return; }
    push(); s.ramps[g] = {placed: true, lat: p.lat, lng: p.lng}; mode = null; edited(); return;
  }
  const dot = ev.target.closest('[data-key]');
  if (dot) { const k = dot.dataset.key; sel.has(k) ? sel.delete(k) : sel.add(k); render(); if (sel.has(k)) revealCrop(k); }
});

// --- groups + crops --------------------------------------------------------------------
function actsHtml(u, s, lab) {    // hover buttons: move to a group, new group, shelf, x, u
  const v = s.labels[lab.key] || null;
  const top = groupsOf(u).map((g, i) => g === v ? '' : '<button data-act="g" data-g="' + g + '" style="background:' + colorOf(u, g) +
    '" title="move to group ' + (i + 1) + '">' + (i + 1) + '</button>').join('') +
    '<button data-act="new" title="a ramp of its own: new group">+ new</button>';
  const bot = (v ? '<button data-act="shelf" title="not this group: put it on the shelf to decide later">shelf</button>' : '') +
    (v !== 'not_ramp' ? '<button data-act="x" title="not a ramp">not ramp</button>' : '') +
    (v !== 'unsure' ? '<button data-act="u" title="unsure: final, abstains from every metric">unsure</button>' : '');
  return '<div class="acts top">' + top + '</div><div class="acts bot">' + bot + '</div>';
}
function cropHtml(lab) {
  const u = cur(), s = S();
  return '<div class="crop' + (sel.has(lab.key) ? ' sel' : '') + '" draggable="true" data-key="' + lab.key + '" title="label ' + lab.key + ' · pano ' + lab.pano_id +
    '"><img loading="lazy" draggable="false" src="' + lab.crop + '" onerror="this.style.opacity=0.15;this.alt=\'no crop\'">' + actsHtml(u, s, lab) + '<div class="cap">' +
    (lab.date || '') + (lab.user_kind === 'human' ? ' · <span class="h">human</span>' : '') + '</div></div>';
}
function actKeys(key) { return sel.has(key) ? [...sel] : [key]; }   // a selected crop acts for the selection
function applyTo(keys, target) {  // target: a ramp id, 'new', null (shelf), 'not_ramp', 'unsure'
  if (!keys.length) return;
  if (target === 'new') newGroup(keys); else assign(keys, target);
}
const ACT = {g: b => b.dataset.g, new: () => 'new', shelf: () => null, x: () => 'not_ramp', u: () => 'unsure'};
function cropClick(ev) {         // true when a crop's button or the crop itself took the click
  const b = ev.target.closest('.acts button');
  if (b) { const k = b.closest('.crop').dataset.key; hidePop(); applyTo(actKeys(k), ACT[b.dataset.act](b)); return true; }
  const c = ev.target.closest('.crop');
  if (c) { const k = c.dataset.key; sel.has(k) ? sel.delete(k) : sel.add(k); render(); return true; }
  return false;
}
function renderGroups() {
  const u = cur(), s = S(), right = document.getElementById('right'), out = [];
  const gs = groupsOf(u);
  const sections = gs.map((g, i) => ({id: g, title: '<b>' + (i + 1) + '</b> · ramp ' + g, col: colorOf(u, g)}));
  sections.push({id: 'not_ramp', title: '<b>not a ramp</b> (x)', col: '#555'},
                {id: 'unsure', title: '<b>unsure</b> (u)', col: '#bbb'});
  for (const sec of sections) {
    const ls = u.labels.filter(l => (s.labels[l.key] || null) === sec.id);
    if (!ls.length) continue;
    const dates = [...new Set(ls.map(l => l.date).filter(Boolean))].sort().join(', ');
    const isG = sec.id && sec.id.startsWith('r');
    out.push('<div class="group' + (isG && gsel.has(sec.id) ? ' gsel' : '') + '" data-gid="' + sec.id + '" style="border-left-color:' + sec.col + '">' +
      '<div class="ghead"' + (isG ? ' data-group="' + sec.id + '" draggable="true" title="click: select · drag onto another group: merge"' : '') + '>' + sec.title + ' · ' + ls.length + ' label' + (ls.length > 1 ? 's' : '') +
      (dates ? ' · ' + dates : '') +
      (isG ? '<button data-hact="notramp" data-g="' + sec.id + '" title="every label in this group is a false positive">not a ramp</button>' : '') +
      (isG && (s.ramps[sec.id] || {}).placed ? '<button data-hact="reset" data-g="' + sec.id + '" title="put the ramp position back at the mean of its labels">reset position</button>' : '') +
      '</div><div class="crops">' +
      ls.map(cropHtml).join('') + '</div></div>');
  }
  if (u.labels.length) out.push('<div id="newzone" data-gid="new">drop here: new group</div>');
  if (!u.labels.length) out.push('<p class="meta">No labels in this window. Mark every curb ramp you can see on the aerial as uncovered (<kbd>a</kbd>), then complete the unit (<kbd>c</kbd>).</p>');
  right.innerHTML = out.join('');
  renderShelf();
}
let shelfMin = false;
function renderShelf() {
  const u = cur(), s = S(), ls = u.labels.filter(l => !s.labels[l.key]);
  const shelf = document.getElementById('shelf');
  document.getElementById('shelfn').textContent = '(' + ls.length + ')';
  document.getElementById('shelfcrops').innerHTML = ls.length ? ls.map(cropHtml).join('') :
    '<span class="meta">empty' + (s.complete ? '' : ': every label is decided; press c when the window is done') + '</span>';
  shelf.classList.toggle('min', shelfMin);
  document.getElementById('shelftoggle').textContent = shelfMin ? 'show' : 'hide';
  document.body.style.paddingBottom = (shelf.offsetHeight + 12) + 'px';
}
document.getElementById('shelftoggle').onclick = ev => { ev.stopPropagation(); shelfMin = !shelfMin; renderShelf(); };
document.getElementById('shelf').addEventListener('click', ev => { cropClick(ev); });
document.getElementById('shelf').addEventListener('mouseover', ev => {
  const c = ev.target.closest('.crop'); setHover(c ? c.dataset.key : null, null);
});
document.getElementById('shelf').addEventListener('mouseleave', () => setHover(null, null));

// --- drag a crop (or the selection) onto a group, the new-group zone or the shelf --------
document.addEventListener('dragstart', ev => {
  const c = ev.target.closest && ev.target.closest('.crop');
  const h = !c && ev.target.closest && ev.target.closest('.ghead[data-group]');
  if (!c && !h) return;
  hidePop(); ev.dataTransfer.effectAllowed = 'move';
  let keys;
  if (c) keys = actKeys(c.dataset.key);
  else { const u = cur(), s = S(), g = h.dataset.group; keys = u.labels.filter(l => s.labels[l.key] === g).map(l => l.key); }
  ev.dataTransfer.setData('text/plain', JSON.stringify(keys));
});
function dropTarget(ev) { return ev.target.closest && (ev.target.closest('[data-gid]') || ev.target.closest('#shelf')); }
document.addEventListener('dragover', ev => {
  const t = dropTarget(ev); if (!t) return;
  ev.preventDefault();
  document.querySelectorAll('.drop').forEach(e => { if (e !== t) e.classList.remove('drop'); });
  t.classList.add('drop');
});
document.addEventListener('dragleave', ev => { const t = dropTarget(ev); if (t && !t.contains(ev.relatedTarget)) t.classList.remove('drop'); });
document.addEventListener('drop', ev => {
  const t = dropTarget(ev); document.querySelectorAll('.drop').forEach(e => e.classList.remove('drop'));
  if (!t) return;
  ev.preventDefault();
  let keys; try { keys = JSON.parse(ev.dataTransfer.getData('text/plain')); } catch (e) { return; }
  applyTo(keys, t.id === 'shelf' ? null : t.dataset.gid);
});
document.getElementById('right').addEventListener('click', ev => {
  const hb = ev.target.closest('[data-hact]');
  if (hb) {
    const g = hb.dataset.g, u = cur(), s = S();
    if (hb.dataset.hact === 'reset') resetStar(g);
    else applyTo(u.labels.filter(l => s.labels[l.key] === g).map(l => l.key), 'not_ramp');
    return;
  }
  const h = ev.target.closest('[data-group]');
  if (h) { const g = h.dataset.group; gsel.has(g) ? gsel.delete(g) : gsel.add(g); render(); return; }
  cropClick(ev);
});

function render() {
  if (!UNITS.length) { document.getElementById('title').textContent = 'No units to review.'; return; }
  const u = cur(), s = S();
  const nDone = UNITS.filter(x => state[x.id].complete).length, nCj = UNITS.filter(x => state[x.id].cant_judge).length;
  document.getElementById('unitsel').innerHTML = UNITS.map((x, i) => '<option value="' + i + '"' + (i === idx ? ' selected' : '') + '>' +
    (i + 1) + '. ' + x.id.replace(CITY + ':', '') + ' · ' + x.type + ' · ' + x.labels.length + ' labels' +
    (state[x.id].complete ? ' ✓' : state[x.id].cant_judge ? " ✗ can't judge" : '') + '</option>').join('');
  document.getElementById('progress').textContent = (idx + 1) + ' / ' + UNITS.length + ' · ' + nDone + ' complete' + (nCj ? ' · ' + nCj + " can't judge" : '');
  document.getElementById('title').innerHTML = u.id + ' <span class="badge">' + u.type + '</span> ' +
    (s.cant_judge ? '<span class="badge cj" title="' + s.cant_judge_reason.replace(/"/g, '&quot;') + '">can\'t judge: ' +
       s.cant_judge_reason.replace(/</g, '&lt;') + '</span> ' :
       '<span class="badge ' + (s.complete ? 'done' : 'todo') + '">' + (s.complete ? 'complete' : 'to do') + '</span> ') +
    '<span class="meta">' + u.labels.length + ' labels · seed ' + s.seed_arm + (u.pilot ? ' · pilot' : '') +
    ' · ' + s.uncovered.length + ' uncovered' + (s.inventory_seen ? ' · <b>inventory seen' + (s.edited_after_inventory ? ', edited after' : '') + '</b>' : '') + ' · on screen <span id="elapsed">' + Math.round(s.elapsed_s) + ' s</span></span>';
  const img = document.getElementById('aerial');
  if (img.getAttribute('src') !== u.aerial.file) img.src = u.aerial.file;
  const w = document.getElementById('aerialwrap');
  w.classList.toggle('addmode', mode === 'add'); w.classList.toggle('placemode', mode === 'place');
  document.getElementById('mode').textContent = mode === 'add' ? 'ADD UNCOVERED (a to leave)' : mode === 'place' ? 'PLACE RAMP POINT' :
    (sel.size ? sel.size + ' label(s) selected' : '') + (gsel.size ? ' ' + gsel.size + ' group(s) selected' : '');
  const note = document.getElementById('unitnote');
  if (document.activeElement !== note) note.value = s.note || '';
  drawPlan(); renderGroups(); drawHover(); refreshCard();
}
const ISTORE = STORE + ':idx';     // the unit last on screen, restored on reload
function go(d) {
  save(); idx = (idx + d + UNITS.length) % UNITS.length; sel.clear(); gsel.clear(); mode = null;
  hoverKey = hoverGroup = null; hidePop(); closeCard(); if (lbKey) closeLightbox();
  try { localStorage.setItem(ISTORE, String(idx)); } catch (e) {}
  lastTick = performance.now(); render(); window.scrollTo(0, 0);
}
document.getElementById('unitsel').onchange = ev => { go(+ev.target.value - idx); ev.target.blur(); };
document.getElementById('prev').onclick = () => go(-1);
document.getElementById('next').onclick = () => go(1);
document.getElementById('nexttodo').onclick = () => {
  for (let k = 1; k <= UNITS.length; k++) { const j = (idx + k) % UNITS.length; if (!done(UNITS[j])) { go(j - idx); return; } }
  alert("Every unit is complete or marked can't judge. Export when ready.");
};
document.getElementById('cantjudge').onclick = ev => { ev.target.blur(); toggleCantJudge(); };
document.getElementById('unitnote').addEventListener('input', ev => { S().note = ev.target.value; save(); });
document.addEventListener('keydown', ev => {
  const t = ev.target.tagName;
  if (t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT') return;   // native undo in text fields
  if (ev.ctrlKey || ev.metaKey) {
    const kk = ev.key.toLowerCase();
    if (kk === 'z' && !ev.shiftKey) doUndo();
    else if (kk === 'y' || (kk === 'z' && ev.shiftKey)) doRedo();
    else return;
    ev.preventDefault(); return;
  }
  if (ev.altKey) return;
  if (lbKey) {
    if (ev.key === 'Escape' || ev.key === ' ') closeLightbox();
    else if (ev.key === 'ArrowRight' || ev.key === 'ArrowDown') stepLightbox(1);
    else if (ev.key === 'ArrowLeft' || ev.key === 'ArrowUp') stepLightbox(-1);
    else return;
    ev.preventDefault(); return;
  }
  if (ev.key === ' ') { if (hoverKey) { openLightbox(hoverKey); ev.preventDefault(); } return; }
  const k = ev.key, u = cur();
  if (k === 'ArrowLeft') go(-1);
  else if (k === 'ArrowRight') go(1);
  else if (k >= '1' && k <= '9') { const g = groupsOf(u)[+k - 1]; if (g) assign(selectedKeys(), g); }
  else if (k === 'n') newGroup();
  else if (k === 'm') merge();
  else if (k === 'x') assign(selectedKeys(), 'not_ramp');
  else if (k === 'u') assign(selectedKeys(), 'unsure');
  else if (k === '0' || k === 'Backspace') assign(selectedKeys(), null);
  else if (k === 'a') { mode = mode === 'add' ? null : 'add'; hidePop(); render(); }
  else if (k === 'r') { showRays = !showRays; render(); }
  else if (k === 'p') { if (gsel.size !== 1) { alert('Select exactly one group (click its header) first.'); return; } mode = 'place'; render(); }
  else if (k === 'P') { if (gsel.size !== 1) return; push(); S().ramps[[...gsel][0]] = {placed: false}; edited(); }
  else if (k === 'c') toggleComplete();
  else if (k === 'z') doUndo();
  else if (k === 'Z' || k === 'y') doRedo();
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
          cant_judge: !!s.cant_judge, cant_judge_reason: s.cant_judge ? (s.cant_judge_reason || '').trim() : '',
          elapsed_s: Math.round(s.elapsed_s * 10) / 10, note: (s.note || '').trim(),
          inventory_seen: !!s.inventory_seen, edited_after_inventory: !!s.edited_after_inventory};
}
document.getElementById('export').onclick = () => {
  const bad = UNITS.filter(u => state[u.id].complete && unassigned(u));
  if (bad.length) { alert('Export refused: ' + bad.length + ' complete unit(s) have unassigned labels, e.g. ' + bad[0].id + '.'); return; }
  const onRamp = UNITS.filter(u => state[u.id].complete && state[u.id].uncovered.some(q => nearRamp(u, q.lat, q.lng)));
  if (onRamp.length) { alert('Export refused: ' + onRamp.length + ' complete unit(s) have an uncovered mark within ' + MIN_SEP_M + ' m of a ramp point, e.g. ' + onRamp[0].id + '. Remove or move it.'); return; }
  const open = UNITS.filter(u => state[u.id].seen && !done(u)).length;
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
  for (const u of UNITS) { const s = state[u.id]; if (s.seen || s.complete || s.cant_judge) out.corners[u.id] = exportUnit(u, s); }
  // units not rendered this session (another filter) round-trip from the prefill verbatim
  if (INITIAL && !boot.initialIgnored) for (const cid in (INITIAL.corners || {})) if (!known.has(cid)) out.corners[cid] = INITIAL.corners[cid];
  const arms = new Set(Object.values(out.corners).map(c => c.seed_arm));
  out.seed_arm = arms.size === 1 ? [...arms][0] : (arms.size ? 'mixed' : null);
  const blob = new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = FILE_NAME; a.click();
};

try { const i = parseInt(localStorage.getItem(ISTORE), 10); if (i >= 0 && i < UNITS.length) idx = i; } catch (e) {}
save();
render();
</script>
"""


def build_html(units, snapshot, city, rater, role, initial, file_name, corners_sha=''):
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
            .replace('__FILE_NAME__', json.dumps(file_name))
            .replace('__CORNERS_SHA__', json.dumps(corners_sha)))


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

    problem = role_problem(args.bundle, args.rater, args.role)
    if problem:
        raise SystemExit(problem)
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
                                               initial, file_name, corners_sha256(args.bundle)),
                                    encoding='utf-8')
    arms = {}
    for v in view:
        arms[v['seed_arm']] = arms.get(v['seed_arm'], 0) + 1
    print(f"{len(view)} units ({sum(len(v['labels']) for v in view)} labels), seed arms {arms}")
    print(f"Gallery: {out / 'index.html'}")
    print(f'Open it, review, Export, and save the download as {args.bundle / file_name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
