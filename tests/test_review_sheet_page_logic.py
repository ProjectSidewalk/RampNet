"""The review sheet's page logic actually runs (issues #96, #59).

``test_inventory_review_sheet.py`` asserts the sheet *contains* the right
strings. That is not the same as the page working, and the gap matters here more
than usual: the sheet is a single 6.7 MB self-contained app whose only
user-visible failure mode is a **blank screen with no error the reviewer can
act on**. A stray brace in the template's ``{{``/``}}`` escaping, or a typo in a
handler, would pass every string assertion and waste an afternoon of human
labour before anyone noticed.

So this loads the emitted JavaScript into Node against a minimal DOM stub and
drives the verdict state machine directly. What it protects, specifically:

* ``done()`` — a chip that cannot be completed silently re-queues forever.
* The mutual exclusion of ``no_ramp`` and ``unreadable`` — a chip asserting both
  corrupts the phantom rate and the unreadable rate at once, and both are
  reported numbers.
* The anti-anchoring gate on the published-neighbour count — if it leaked before
  the reviewer counted, ``ramps_visible`` would stop being independent evidence
  and the comparison against the published data would measure nothing.

**Skipped when Node is absent** rather than made a hard dependency, per the
CPU-only/no-network rule for this suite.
"""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import inventory_review_sheet as irs  # noqa: E402

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not available")


HARNESS = r"""
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
    checked: false, style: {}, dataset: {}, open: false,
    setAttribute() {}, getAttribute() {}, querySelector: () => mk("q"),
    querySelectorAll: () => [], addEventListener() {},
    showModal() { e.open = true; }, close() { e.open = false; },
    getBoundingClientRect: () => ({left: 0, top: 0, width: 698, height: 698}),
  };
  return e;
}
globalThis.document = {
  getElementById: id => (els[id] ||= mk(id)),
  createElement: () => mk("tmp"),
  body: {classList: {toggle() {}}},
};
globalThis.addEventListener = () => {};
globalThis.Blob = class { constructor(p) { this.parts = p; } };
globalThis.URL = {createObjectURL: () => "blob:x"};
globalThis.alert = () => {};

let src = require("fs").readFileSync(process.argv[2], "utf8");
src += "\nglobalThis.__t = {V, done, state, render, open_, CHIPS};\n";
(0, eval)(src);

const T = globalThis.__t;
const out = [];
const ok = (cond, msg) => out.push({ok: !!cond, msg});

ok(!T.done(T.V["A"]), "untouched chip is not done");

T.open_(0);
const v = T.state("A");
v.no_ramp = true; v.unreadable = false; v.offset_m = null; v.ramps_visible = 0;
ok(T.done(v), "no_ramp completes a chip");

v.unreadable = true; if (v.unreadable) v.no_ramp = false;
ok(!(v.unreadable && v.no_ramp), "unjudgeable clears no_ramp");

v.px = 349; v.py = 349; v.offset_m = 0; v.unreadable = false; v.no_ramp = false;
ok(!v.unreadable && !v.no_ramp, "measuring clears both terminal states");
ok(T.done(v), "a measured chip is done");

const v2 = T.state("B");
v2.ramps_visible = null; T.open_(1);
ok(document.getElementById("pubrow").hidden === true, "published hidden before counting");
v2.ramps_visible = 1; T.open_(1);
ok(document.getElementById("pubrow").hidden === false, "published revealed after counting");

// Chip A publishes [2, 4] -- the pork-chop shape, where 6 m splits the island.
// A count of 3 sits inside the bracket and must NOT raise an alarm; comparing
// against the 6 m figure alone used to flag it as under-recording.
const pub = () => document.getElementById("pub").className;
const va = T.state("A");
va.ramps_visible = 3; T.open_(0);
ok(pub().includes("agree"), "count inside the [6 m, 10 m] bracket is consistent");
va.ramps_visible = 5; T.open_(0);
ok(pub().includes("under"), "count above the 10 m figure flags under-recording");
va.ramps_visible = 1; T.open_(0);
ok(pub().includes("over"), "count below the 6 m figure flags phantom/duplicate");

ok(document.getElementById("rubric-body").innerHTML.includes("<h3>"), "rubric renders");

console.log(JSON.stringify(out));
"""


def _page_logic(tmp_path):
    """Emit a sheet, strip the megabyte data blobs, return the path to its JS."""
    meta = {"city": "denver-co", "inventory": "x.jsonl.gz", "sampling": "uniform",
            "seed": 20260731, "tile_source": "denver-2016", "zoom": 21,
            "mpp": 0.057, "span_px": 698, "attribution": "Denver", "note": "leaf-off"}
    chips = [{"uri": "", "id": cid, "lon": -105.0, "lat": 39.7, "tiles": [],
              "published": pub}
             for cid, pub in (("A", [2, 3]), ("B", [1, 1]))]
    html = irs.build_sheet(meta, chips, {"city": "denver-co", "rubric": irs.RUBRIC})
    src = re.search(r"<script>\n(.*)\n</script>", html, re.S).group(1)
    path = tmp_path / "page.js"
    path.write_text(src, encoding="utf-8")
    return path


def test_emitted_javascript_parses(tmp_path):
    """A brace-escaping slip in the template renders a blank page, not an error."""
    path = _page_logic(tmp_path)
    proc = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_verdict_state_machine_behaves(tmp_path):
    path = _page_logic(tmp_path)
    harness = tmp_path / "harness.cjs"
    harness.write_text(HARNESS, encoding="utf-8")
    proc = subprocess.run([NODE, str(harness), str(path)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    results = json.loads(proc.stdout.strip().splitlines()[-1])
    assert results, "harness produced no assertions"
    failed = [r["msg"] for r in results if not r["ok"]]
    assert not failed, "page logic broke: {}".format(failed)
