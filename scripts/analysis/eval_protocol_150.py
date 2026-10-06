"""Prove ``rampnet.eval`` is the protocol behind every committed benchmark number (#150).

Re-derives, through :func:`rampnet.eval.score_split` alone, every committed number that
has committed detections behind it, and compares:

1. **Every cell of** ``analysis_out/scoreboard.json["per_split"]`` -- precision, recall,
   F1, AP, ``ap_bundle``, tp, fp, fn, n_panos, n_gt_recall -- at the scoreboard's own
   operating point per model class (``scoreboard.OPERATING_POINT``). The scoreboard
   stores floats rounded to 6 decimals (``scoreboard_render.JSON_PRECISION``), so a
   re-derived float is rounded the same way and then compared with ``==``. For a RampNet
   cell whose AP comes from the low-floor cache (``ap_source`` = op_cache), the cache's
   detections (``analysis_out/op_cache/<split>.json``) are converted to the prediction
   format and scored against the *bundle's* ground truth for AP; ``ap_bundle`` is the AP
   of the bundle's own detections.
2. **Every number in** ``scripts/model_comparison/yolo_baseline/benchmark_eval/`` for the
   three YOLO pano arms on its ten splits, at op 0.25 / floor 0.05: the headline row of
   ``<split>.txt`` (P, its CI, R, its CI, F1, AP, tp/fp/fn/ignored, compared as the
   printed 3-decimal strings and integers), every sweep row, and ``pr_<split>/pr_<arm>.json``
   (``ap``, ``n_gt`` and the full recall/precision lists, compared exactly).

::

    python scripts/analysis/eval_protocol_150.py           # write reproduction.json
    python scripts/analysis/eval_protocol_150.py --check   # exit 1 on any difference

Reads committed files only (bundles, ``manual_labels/``, ``benchmark/model_detections/``,
``analysis_out/op_cache/``, ``analysis_out/scoreboard.json``, ``benchmark_eval/``). CPU, no
network, a few seconds. Writes ``analysis_out/eval_protocol_150/reproduction.json``.
"""
import argparse
import json
import os
import re
import sys
from time import perf_counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison", "yolo_baseline"))

import scoreboard as SB  # noqa: E402
import rescore_benchmark_eval as RBE  # noqa: E402
from export_model_cache import published_path  # noqa: E402
from rampnet import roster  # noqa: E402
from rampnet.eval import (  # noqa: E402
    all_pins, eval_sha256, load_predictions, score_split, scorer_fingerprint)
from scoreboard_render import JSON_PRECISION  # noqa: E402

OUT_DIR = os.path.join(REPO, "analysis_out", "eval_protocol_150")
OUT_JSON = os.path.join(OUT_DIR, "reproduction.json")
SCOREBOARD_JSON = os.path.join(REPO, "analysis_out", "scoreboard.json")

SCOREBOARD_FIELDS = ("precision", "recall", "f1", "ap", "ap_bundle", "tp", "fp", "fn",
                     "n_panos", "n_gt_recall")


def _r(v):
    return round(v, JSON_PRECISION) if isinstance(v, float) else v


def _bundle(split):
    return os.path.join(REPO, "benchmark", split)


def op_cache_predictions(split):
    """RampNet's low-floor extraction for ``split`` in the prediction-file format."""
    panos = SB.low_floor_panos(split)
    return {"model": "rampnet (op_cache, 0.05 floor)", "city": split,
            "detections": {pd["pano"]: [list(p) for p in pd["preds"]] for pd in panos}}


def scoreboard_cells():
    """One comparison per (leg, split) cell of the committed scoreboard."""
    with open(SCOREBOARD_JSON, encoding="utf-8") as fh:
        committed = json.load(fh)["per_split"]
    legs = {roster.published_name(leg): leg for leg in SB.legs()}
    out = []
    for name, cells in committed.items():
        leg = legs.get(name)
        for split, stored in cells.items():
            row = {"source": "scoreboard", "model": name, "split": split}
            if leg is None:
                row.update(status="unchecked", reason="not in rampnet.roster")
                out.append(row)
                continue
            op = SB.OPERATING_POINT[SB.class_of(leg)]
            row["op_threshold"] = op
            if leg.provider == "rampnet":
                preds = "rampnet"
                row["predictions"] = f"benchmark/{split}/records.jsonl"
            else:
                path = published_path(leg.label, split, publish_as=name)
                if not os.path.exists(path):
                    row.update(status="unchecked", reason=f"no file {os.path.relpath(path, REPO)}")
                    out.append(row)
                    continue
                preds = load_predictions(path)
                row["predictions"] = os.path.relpath(path, REPO).replace(os.sep, "/")
            res = score_split(_bundle(split), preds, op_threshold=op, model=name)
            got = {k: res[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn",
                                       "n_panos", "n_gt_recall")}
            got["ap_bundle"] = res["ap"]
            got["ap"] = res["ap"]
            if stored.get("ap_source", "bundle") != "bundle":
                # The scoreboard's AP for this cell is the low-floor cache's, scored here
                # against the bundle's GT through the same function.
                low = score_split(_bundle(split), op_cache_predictions(split), model=name)
                got["ap"] = low["ap"]
                row["ap_predictions"] = f"analysis_out/op_cache/{split}.json"
            row["ap_source"] = stored.get("ap_source")
            fields = {}
            for k in SCOREBOARD_FIELDS:
                g = _r(got[k])
                fields[k] = {"committed": stored.get(k), "rederived": g,
                             "equal": stored.get(k) == g}
            row["fields"] = fields
            row["status"] = "equal" if all(f["equal"] for f in fields.values()) else "differs"
            out.append(row)
    return out


_ROW = re.compile(
    r"^(?P<name>\S+)\s+(?P<p>\d\.\d{3})\s+\((?P<plo>\d\.\d{3})-(?P<phi>\d\.\d{3})\)\s+"
    r"(?P<r>\d\.\d{3})\s+\((?P<rlo>\d\.\d{3})-(?P<rhi>\d\.\d{3})\)\s+(?P<f1>\d\.\d{3})\s+"
    r"(?P<ap>\d\.\d{3}|-)\s+(?P<tp>\d+)/(?P<fp>\d+)/(?P<fn>\d+)/(?P<ign>\d+)\s*$")
_SWEEP_HEAD = re.compile(r"^\[(?P<arm>\S+)\] threshold sweep")
_SWEEP_ROW = re.compile(r"^\s+(?P<t>\d\.\d{2})\s+(?P<p>\d\.\d{3})\s+(?P<r>\d\.\d{3})\s+"
                        r"(?P<f1>\d\.\d{3})\s+(?P<tp>\d+)/(?P<fp>\d+)/(?P<fn>\d+)")


def parse_benchmark_txt(path):
    """``({arm: headline-row dict}, {arm: [sweep-row dicts]})`` from ``<split>.txt``."""
    heads, sweeps, current = {}, {}, None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            m = _ROW.match(line)
            if m and m["name"] in RBE.ARMS:
                heads[m["name"]] = m.groupdict()
                continue
            m = _SWEEP_HEAD.match(line)
            if m:
                current = m["arm"]
                sweeps[current] = []
                continue
            m = _SWEEP_ROW.match(line)
            if m and current:
                sweeps[current].append(m.groupdict())
    return heads, sweeps


def _f3(x):
    return f"{x:.3f}"


def yolo_cells():
    """One comparison per (arm, split) in ``benchmark_eval/``."""
    out = []
    for split in RBE.SPLITS:
        heads, sweeps = parse_benchmark_txt(os.path.join(RBE.BENCHMARK_EVAL, f"{split}.txt"))
        for arm in RBE.ARMS:
            row = {"source": "benchmark_eval", "model": arm, "split": split,
                   "op_threshold": RBE.OP_THRESHOLD, "floor": RBE.YOLO_FLOOR}
            path = published_path(arm, split)
            if arm not in heads or not os.path.exists(path):
                row.update(status="unchecked",
                           reason="no headline row" if arm not in heads else "no detections")
                out.append(row)
                continue
            row["predictions"] = os.path.relpath(path, REPO).replace(os.sep, "/")
            res = score_split(_bundle(split), load_predictions(path),
                              op_threshold=RBE.OP_THRESHOLD, floor=RBE.YOLO_FLOOR,
                              sweep=True, pr_curve=True, model=arm)
            h = heads[arm]
            got = {"p": _f3(res["precision"]), "plo": _f3(res["precision_ci"][0]),
                   "phi": _f3(res["precision_ci"][1]), "r": _f3(res["recall"]),
                   "rlo": _f3(res["recall_ci"][0]), "rhi": _f3(res["recall_ci"][1]),
                   "f1": _f3(res["f1"]),
                   "ap": _f3(res["ap"]) if res["ap"] is not None else "-",
                   "tp": str(res["tp"]), "fp": str(res["fp"]), "fn": str(res["fn"]),
                   "ign": str(res["ignored"])}
            fields = {k: {"committed": h[k], "rederived": got[k], "equal": h[k] == got[k]}
                      for k in got}
            # The sweep table, row by row, as printed.
            want = sweeps.get(arm, [])
            have = [{"t": f"{s['threshold']:.2f}", "p": _f3(s["precision"]),
                     "r": _f3(s["recall"]), "f1": _f3(s["f1"]), "tp": str(s["tp"]),
                     "fp": str(s["fp"]), "fn": str(s["fn"])} for s in res["sweep"]]
            fields["sweep_rows"] = {"committed": len(want), "rederived": len(have),
                                    "equal": want == have}
            # The PR-curve JSON, exactly (not rounded).
            with open(os.path.join(RBE.BENCHMARK_EVAL, f"pr_{split}", f"pr_{arm}.json"),
                      encoding="utf-8") as fh:
                pr = json.load(fh)
            fields["pr_json_ap"] = {"committed": _r(pr["ap"]), "rederived": _r(res["ap"]),
                                    "equal": pr["ap"] == res["ap"]}
            fields["pr_json_n_gt"] = {"committed": pr["n_gt"], "rederived": res["n_gt_recall"],
                                      "equal": pr["n_gt"] == res["n_gt_recall"]}
            curve = res.get("pr_curve") or {}
            fields["pr_json_curve"] = {
                "committed": len(pr["recalls"]), "rederived": len(curve.get("recalls", [])),
                "equal": (pr["recalls"] == curve.get("recalls")
                          and pr["precisions"] == curve.get("precisions"))}
            row["fields"] = fields
            row["status"] = "equal" if all(f["equal"] for f in fields.values()) else "differs"
            out.append(row)
    return out


def build():
    cells = scoreboard_cells() + yolo_cells()
    summary = {}
    for src in ("scoreboard", "benchmark_eval"):
        rows = [c for c in cells if c["source"] == src]
        summary[src] = {"cells": len(rows),
                        "equal": sum(c["status"] == "equal" for c in rows),
                        "differs": sum(c["status"] == "differs" for c in rows),
                        "unchecked": sum(c["status"] == "unchecked" for c in rows),
                        "fields_compared": sum(len(c.get("fields", {})) for c in rows)}
    return {
        "what": "Every committed scoreboard cell and benchmark_eval number re-derived "
                "through rampnet.eval.score_split (#150). See docs/eval_protocol_150.md.",
        "regenerate": "python scripts/analysis/eval_protocol_150.py [--check]",
        "float_comparison": f"scoreboard floats rounded to {JSON_PRECISION} decimals as "
                            "stored, then ==; benchmark_eval as printed (3 decimals) and "
                            "pr_<arm>.json exactly",
        "scorer_fingerprint": scorer_fingerprint(),
        "eval_sha256": eval_sha256(),
        "split_pins": all_pins(),
        "summary": summary,
        "cells": cells,
    }


def payload(result):
    return json.dumps(result, indent=1, sort_keys=True) + "\n"


def print_table(result):
    print(f"{'source':<16} {'cells':>6} {'equal':>6} {'differs':>8} {'unchecked':>10} "
          f"{'fields':>7}")
    for src, s in result["summary"].items():
        print(f"{src:<16} {s['cells']:>6} {s['equal']:>6} {s['differs']:>8} "
              f"{s['unchecked']:>10} {s['fields_compared']:>7}")
    for c in result["cells"]:
        if c["status"] == "differs":
            bad = {k: (f["committed"], f["rederived"]) for k, f in c["fields"].items()
                   if not f["equal"]}
            print(f"  DIFFERS {c['source']} {c['model']} / {c['split']}: {bad}")
        elif c["status"] == "unchecked":
            print(f"  unchecked {c['source']} {c['model']} / {c['split']}: {c['reason']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="Exit 1 if any cell differs or the committed reproduction.json "
                         "is stale; write nothing.")
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args(argv)

    t0 = perf_counter()
    result = build()
    elapsed = perf_counter() - t0
    print_table(result)
    print(f"scorer fingerprint {result['scorer_fingerprint']}, eval.py "
          f"{result['eval_sha256']}, {elapsed:.1f} s")
    differs = sum(s["differs"] for s in result["summary"].values())
    text = payload(result)
    if args.check:
        problems = []
        if differs:
            problems.append(f"{differs} cell(s) differ from the committed numbers")
        if not os.path.exists(args.out):
            problems.append(f"{args.out}: missing")
        else:
            with open(args.out, "rb") as fh:
                have = fh.read().replace(b"\r\n", b"\n").decode("utf-8")
            if have != text:
                problems.append(f"{args.out}: stale (re-run without --check)")
        for p in problems:
            print("FAIL: " + p)
        return 1 if problems else 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"wrote {args.out}")
    return 1 if differs else 0


if __name__ == "__main__":
    sys.exit(main())
