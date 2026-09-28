"""Run the labeler's ``scripts/harvest_depth.py`` on a GSV run, past its record-source check (#151).

At sidewalk-auto-labeler HEAD ``29dc605`` (2026-09-27), ``harvest_depth.main()`` refuses any
run whose records carry ``pano.source != "gsv"``. Every GSV run's records carry GSV's own
upload type in that field, ``"launch"`` (laurens_gsv 2,137 of 2,137; bend, paterson,
gainesville and sao_paulo likewise), so the check refuses every GSV run, including the four
harvested before it existed. The manifest check (``check_gsv``: ``imagery_source == "gsv"``)
is unaffected and still runs.

This wrapper loads the labeler's script unmodified and replaces only ``run_pano_ids``: it
returns source ``"gsv"`` after checking that the manifest says ``imagery_source: gsv`` and
that every record's ``pano.source`` is ``"launch"``. Everything else -- the fetch, the
``.part`` write, the hash index, the no-depth alarm, the reconcile -- is the labeler's own
code. No tracked labeler file is touched; the payloads land in ``runs/<city>/depth``, which
the labeler gitignores. Run it with the labeler's own interpreter, from the labeler root::

    cd D:/Git/sidewalk-auto-labeler
    .venv/Scripts/python.exe D:/Git/RampNet/scripts/analysis/harvest_depth_launch_151.py runs/laurens_gsv

Any argument after the script name is passed to ``harvest_depth.py`` unchanged
(``--verify``, ``--rehash``, ``--limit``). Network: the GSV metadata endpoint. Never called
from a test.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

LABELER = Path(os.environ.get("LABELER_ROOT", r"D:\Git\sidewalk-auto-labeler"))


def load_harvester(labeler_root=LABELER):
    spec = importlib.util.spec_from_file_location(
        "harvest_depth", Path(labeler_root) / "scripts" / "harvest_depth.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patch(hd):
    original = hd.run_pano_ids

    def run_pano_ids(run_dir):
        ids, _ = original(run_dir)
        manifest = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
        sources = set()
        with open(Path(run_dir) / "results.jsonl", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    sources.add(json.loads(line)["pano"].get("source"))
        if manifest.get("imagery_source") != "gsv" or sources != {"launch"}:
            sys.exit(f"not a GSV run: manifest {manifest.get('imagery_source')!r}, "
                     f"record sources {sorted(map(str, sources))}")
        return ids, "gsv"

    hd.run_pano_ids = run_pano_ids
    return hd


if __name__ == "__main__":
    harvester = patch(load_harvester())
    sys.argv = ["harvest_depth.py"] + sys.argv[1:]
    harvester.main()
