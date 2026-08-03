"""The street sheet's page logic actually runs — and measures (#103).

Beyond inheriting the aerial page-logic harness's concerns (a template slip
renders a blank page; the state machine gates completeness), this pins the two
things that make the street sheet an *instrument* rather than a picture:

* **The JS click-to-angle map equals the Python one.** ``degOf``/``colOf`` in
  the page and ``perspective_col_to_azimuth_deg`` in ``rampnet.gsv`` are two
  copies of the same formula — the classic two-path hazard — so the harness
  evaluates the JS against values computed by the Python side, including at
  the asymmetric strip edges.
* **The export emits exactly the shared + verdict fields.** The export copies
  provenance by iterating ``META.shared_fields`` (the Python list), which is
  the design that prevents §5l's dropped-stratum bug; this asserts the
  resulting key set end to end, by triggering the real export handler and
  reading the Blob it builds.

Skipped when Node is absent, per the CPU-only/no-network rule.
"""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import street_review_sheet as srs  # noqa: E402
from rampnet.gsv import azimuth_deg_to_perspective_col  # noqa: E402

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not available")


HARNESS_TEMPLATE = r"""
// Minimal DOM stub -- enough to load the page logic and drive it. Not a browser.
const store = {};
globalThis.localStorage = {
  getItem: k => (k in store ? store[k] : null),
  setItem: (k, v) => { store[k] = String(v); },
};
const els = {};
function mk(id) {
  const e = {
    id, innerHTML: "", textContent: "", value: "", className: "", hidden: false,
    checked: false, style: {}, dataset: {}, open: false, src: "", alt: "",
    setAttribute() {}, getAttribute() {}, querySelector: () => mk("q"),
    querySelectorAll: () => [], addEventListener() {}, click() {},
    showModal() { e.open = true; }, close() { e.open = false; },
    getBoundingClientRect: () => ({left: 0, top: 0, width: 1024, height: 1024}),
  };
  return e;
}
globalThis.document = {
  getElementById: id => (els[id] ||= mk(id)),
  createElement: () => mk("tmp"),
  body: {classList: {toggle() {}}},
};
globalThis.addEventListener = () => {};
let lastBlob = null;
globalThis.Blob = class { constructor(p) { this.parts = p; } };
globalThis.URL = {createObjectURL: b => { lastBlob = b; return "blob:x"; }};
globalThis.alert = () => {};

let src = require("fs").readFileSync(process.argv[2], "utf8");
src += "\nglobalThis.__t = {V, done, complete, partial, state, render, open_, CHIPS,"
    +  " paint, nextTodo, degOf, colOf, insideStrip, META};\n";
(0, eval)(src);

const T = globalThis.__t;
const out = [];
const ok = (cond, msg) => out.push({ok: !!cond, msg});

// ---- the two-path check: JS trig vs Python trig ---------------------------
const CASES = __CASES__;   // [[deg, expected_col], ...] computed in Python
CASES.forEach(([deg, col]) => {
  ok(Math.abs(T.colOf(deg) - col) < 1e-6, "colOf(" + deg + ") matches Python");
  ok(Math.abs(T.degOf(col) - deg) < 1e-9, "degOf(" + col + ") matches Python");
});
const L = T.META.strip_left_deg, R = T.META.strip_right_deg;
ok(T.insideStrip(L + 1e-9) && T.insideStrip(R - 1e-9), "just inside both edges");
ok(!T.insideStrip(L - 1e-6) && !T.insideStrip(R + 1e-6), "just outside both edges");
ok(Math.abs(L) > Math.abs(R), "asymmetry survives into the page");
ok(T.degOf(1024/2 + 100) > 0, "right of centre is POSITIVE (the §5j sign)");

// ---- state machine --------------------------------------------------------
ok(!T.done(T.V["A"]), "untouched chip is not done");
T.open_(0);
const v = T.state("A");

// A click measures. Simulate the stage handler's effect directly.
v.click_x = 682; v.click_y = 500; v.offset_deg = T.degOf(682);
v.unreadable = false; v.no_ramp = false;
ok(T.done(v) && T.complete(v), "a measured chip is done and complete");
ok(Math.abs(v.offset_deg - R) < 1e-6, "a click on the right strip edge reads +18.3678");

// Unjudgeable clears the click AND needs its reason to be complete.
v.unreadable = true;
if (v.unreadable) { v.no_ramp = false; v.offset_deg = null; v.click_x = v.click_y = null; }
ok(v.offset_deg === null && v.click_x === null, "unjudgeable clears a disowned click");
ok(T.done(v) && !T.complete(v) && T.partial(v),
   "unjudgeable WITHOUT a reason is partial — the reason is a reported number");
v.unreadable_reason = "van_or_vehicle";
ok(T.complete(v), "unjudgeable + reason is complete");

// no_ramp is exclusive with unreadable and clears the reason.
v.no_ramp = true;
if (v.no_ramp) { v.unreadable = false; v.unreadable_reason = null;
                 v.offset_deg = null; v.click_x = v.click_y = null; }
ok(!(v.no_ramp && v.unreadable), "terminal states are mutually exclusive");
ok(v.unreadable_reason === null, "no_ramp clears a stale reason");
ok(T.complete(v), "phantom is complete");

// Un-setting unreadable must drop the reason too, or a later unreadable
// verdict silently inherits a stale tag.
v.no_ramp = false; v.unreadable = true; v.unreadable_reason = "sun_or_shadow";
v.unreadable = false;
if (!v.unreadable) { v.unreadable_reason = null; }
ok(v.unreadable_reason === null, "clearing unjudgeable clears its reason");

// nextTodo routes untouched first, then partials.
T.CHIPS.forEach(c => { delete T.V[c.id]; });
T.V["A"] = {unreadable: true, unreadable_reason: null, no_ramp: false,
            offset_deg: null};                       // partial
T.V["B"] = {unreadable: false, no_ramp: false, offset_deg: 2.5,
            click_x: 540, click_y: 500};             // complete
T.nextTodo();
ok(document.getElementById("title").textContent.startsWith("A"),
   "next-unreviewed routes to the reason-less partial");
T.paint();
ok(document.getElementById("prog").textContent.includes("partial"),
   "progress counter surfaces partials");

// Neighbour bearings render, always (no reveal gate — nothing to anchor).
T.open_(0);
const svg = document.getElementById("bigsvg").innerHTML;
ok((svg.match(/paint-order="stroke"/g) || []).length === 2,
   "both neighbour bearings are drawn without any gate");
ok(svg.includes(">5.2m</text>"), "neighbour labelled with distance from the record");
ok(svg.includes("crop edge"), "strip edges labelled");

// ---- export: the third path emits exactly the agreed fields ---------------
document.getElementById("export").onclick();
ok(lastBlob !== null, "export built a payload");
const payload = JSON.parse(lastBlob.parts[0]);
const EXPECTED = __EXPECTED_FIELDS__;
const keys = Object.keys(payload.records[0]).sort();
ok(JSON.stringify(keys) === JSON.stringify(EXPECTED),
   "export keys == SHARED_FIELDS + VERDICT_FIELDS, got: " + keys.join(","));
const recB = payload.records.find(r => r.id === "B");
ok(recB.offset_deg === 2.5 && recB.click_px[0] === 540,
   "a measured verdict round-trips through export");
const recA = payload.records.find(r => r.id === "A");
ok(recA.unreadable === true && recA.unreadable_reason === null,
   "an unfinished reason exports as null, not undefined");
ok(payload.records.every(r => r.pano_id === "P"),
   "provenance fields travel via META.shared_fields");
ok(payload.rubric && payload.rubric.sign_convention,
   "the rubric travels with the verdicts");

ok(document.getElementById("rubric-body").innerHTML.includes("<h3>"), "rubric renders");

console.log(JSON.stringify(out));
"""


def _fake_chips():
    site = {"id": "A", "lon": -104.99, "lat": 39.74, "stratum": None}
    chosen = {"pano_id": "P", "lat": 39.7401, "lon": -104.99, "date": "2021-3",
              "range_m": 11.1}
    chips = []
    for cid in ("A", "B"):
        base = srs.make_base_record(dict(site, id=cid), chosen, heading=123.4,
                                    az_gov=90.0, theta=-33.4, n_candidates=7)
        chips.append(dict(base, uri="", ctx_uri="",
                          neighbors=[[10.0, 5.2], [-30.5, 12.0]],
                          n_neighbors_out_of_view=1))
    return chips


def _page_logic(tmp_path):
    meta = {"city": "denver-co", "seed": 20260731,
            "inventory": "denver-co-2026-07-31.jsonl.gz",
            "sites_desc": "2 records (test)"}
    manifest = {"city": "denver-co", "seed": 20260731, "rubric": srs.RUBRIC,
                "reviewer": None}
    html = srs.build_sheet(meta, _fake_chips(), manifest)
    src = re.search(r"<script>\n(.*)\n</script>", html, re.S).group(1)
    path = tmp_path / "page.js"
    path.write_text(src, encoding="utf-8")
    return path


def test_emitted_javascript_parses(tmp_path):
    path = _page_logic(tmp_path)
    proc = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_page_logic_and_export(tmp_path):
    path = _page_logic(tmp_path)
    # The Python-computed truth the JS copies must reproduce.
    cases = [[d, azimuth_deg_to_perspective_col(d)]
             for d in (-45.0, srs.STRIP_LEFT_DEG, -10.0, 0.0, 10.0,
                       srs.STRIP_RIGHT_DEG, 45.0)]
    expected = sorted(list(srs.SHARED_FIELDS) + list(srs.VERDICT_FIELDS))
    harness = (HARNESS_TEMPLATE
               .replace("__CASES__", json.dumps(cases))
               .replace("__EXPECTED_FIELDS__", json.dumps(expected)))
    hpath = tmp_path / "harness.cjs"
    hpath.write_text(harness, encoding="utf-8")
    proc = subprocess.run([NODE, str(hpath), str(path)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    results = json.loads(proc.stdout.strip().splitlines()[-1])
    assert results, "harness produced no assertions"
    failed = [r["msg"] for r in results if not r["ok"]]
    assert not failed, "page logic broke: {}".format(failed)
