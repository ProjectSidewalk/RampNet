"""Generated tables in the data-sourcing docs (#145): the numbers come from committed files.

Two blocks, each between a matched pair of HTML comments that this script owns (the
prose around them is never touched):

- ``offset-tolerance`` in ``docs/location_precision_assessment_96.md`` §5g -- the Stage 1
  tolerance curve, from ``analysis_out/stage1_offset_tolerance.json["sweep"]``.
- ``inventory-snapshots`` in ``docs/curb_ramp_data_sourcing.md`` §9 -- every frozen
  inventory in ``data/inventories/``, from its ``*.manifest.json``. It sits below the
  hand-written 2026-07-31 table, which it does not replace (that table carries the
  "Paper Tab. 1" comparison notes).

Reads only committed files; numpy-free. ``--check`` re-renders, compares, and exits 1 on
drift, naming the file::

    python scripts/analysis/sourcing_tables.py            # rewrite the blocks
    python scripts/analysis/sourcing_tables.py --check    # exit 1 if any block is stale

Deliberately NOT generated (the tables stay prose, see the #145 PR):

- §5j's bearing-residual table. ``analysis_out/stage1_bearing_residual.json`` stores its
  values at 4 dp, and two of the table's cells sit exactly on a rounding tie that the
  published table resolved in opposite directions (Bend ``mean_deg`` 0.0355 printed
  +0.036, Bend ``se_mean_deg`` 0.1055 printed 0.105) -- it was rendered from unrounded
  values. No rounding rule reproduces both from the JSON, so generating it would change a
  published number. TODO(#145): store the bearing-residual JSON at 6 dp on its next
  re-run, then generate that table here too.
- TODO(#145): §5f's Denver review table could be fed from the committed
  ``analysis_out/review_denver-co/summary.json``.
- §1, §2, §3 (endpoint strings and notes are editorial), §5c, §6, §7.
"""
import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scoreboard_render import _table, splice  # noqa: E402

BEGIN = "<!-- BEGIN GENERATED: {name} (scripts/analysis/sourcing_tables.py) -->"
END = "<!-- END GENERATED: {name} -->"

OFFSET_JSON = os.path.join(REPO, "analysis_out", "stage1_offset_tolerance.json")
INVENTORY_DIR = os.path.join(REPO, "data", "inventories")
LOCATION_DOC = os.path.join(REPO, "docs", "location_precision_assessment_96.md")
SOURCING_DOC = os.path.join(REPO, "docs", "curb_ramp_data_sourcing.md")

# City slugs as the §5 prose names them.
CITY_NAME = {"denver-co": "Denver"}


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def pct(fraction):
    """A share as the §5g table prints it: 2 dp below 1%, 1 dp at or above."""
    p = 100 * fraction
    return f"{p:.2f}%" if p < 1 else f"{p:.1f}%"


def offset_tolerance_table(payload):
    """§5g: median coordinate offset -> share of Stage 1 labels lost.

    Row 1 is the measured city (``payload["city"]``) at scale 1, marked and bolded as the
    table always has been; the rest are its distribution scaled up.
    """
    rows = []
    for i, r in enumerate(payload["sweep"]):
        offset = f"{r['median_offset_m']:.2f} m"
        lost = pct(r["p_outside"])
        if i == 0:
            city = CITY_NAME.get(payload["city"], payload["city"])
            offset += f" *({city})*"
            lost = f"**{lost}**"
        rows.append([offset, lost])
    return _table(["median offset", "labels lost"], rows, ["---:", "---:"])


def _first_sentence(note):
    note = (note or "").strip()
    for i, ch in enumerate(note):
        if ch == "." and (i + 1 == len(note) or note[i + 1] == " "):
            return note[:i + 1]
    return note


def inventory_snapshot_table(manifests):
    """§9: one row per frozen inventory, sorted by snapshot stem."""
    rows = []
    for stem, m in sorted(manifests.items()):
        declared = m.get("declared_count")
        rows.append([
            f"`{stem}`",
            f"{m['records']:,}",
            "–" if declared is None else f"{declared:,}",
            f"`{m['sha256'][:12]}`",
            _first_sentence(m.get("note")).replace("|", "\\|"),
        ])
    return _table(["snapshot", "records", "declared count", "sha256 (first 12)", "note"],
                  rows, ["---", "---:", "---:", "---", "---"])


def load_manifests(directory=INVENTORY_DIR):
    out = {}
    for path in glob.glob(os.path.join(directory, "*.manifest.json")):
        stem = os.path.basename(path)[:-len(".manifest.json")]
        out[stem] = _load(path)
    return out


def blocks():
    """``{doc path: {block name: markdown}}`` -- which doc each block lives in."""
    return {
        LOCATION_DOC: {"offset-tolerance": offset_tolerance_table(_load(OFFSET_JSON))},
        SOURCING_DOC: {"inventory-snapshots": inventory_snapshot_table(load_manifests())},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="Write nothing; exit 1 if any committed block is stale or missing.")
    args = ap.parse_args()
    problems = []
    for path, tables in blocks().items():
        rel = os.path.relpath(path, REPO).replace(os.sep, "/")
        with open(path, encoding="utf-8", newline="") as fh:
            current = fh.read()
        missing = [n for n in tables if BEGIN.format(name=n) not in current]
        if missing:
            problems.append(f"{rel}: generated blocks missing: {', '.join(missing)}")
        updated = splice(current, tables, begin=BEGIN, end=END)
        if args.check:
            if updated != current:
                problems.append(f"{rel}: generated tables are stale "
                                "(re-run scripts/analysis/sourcing_tables.py)")
            else:
                print(f"{rel}: current")
        elif updated != current:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(updated)
            print(f"updated {rel}")
        else:
            print(f"{rel}: already current")
    if problems:
        print("\n".join(problems))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
