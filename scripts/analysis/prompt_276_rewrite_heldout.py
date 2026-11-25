#!/usr/bin/env python
"""#276 whole-prompt-rewrite arm: the held-out run of Argon candidate r1, from the caches.

``prompt276-argon-r1`` was the one rewrite candidate that passed the dev_select gate
(docs/vlm_prompt_276.md, "Whole-prompt-rewrite arm"), although the optimizer had rejected it
on dev_train. Jon approved the held-out run on 2026-11-13. This script scores it exactly as
Step 5 scored its arms (``prompt_276_heldout.build``): one threshold per prompt selected on
sao_paulo dev_select, the 12 held-out splits and manual_gold@250 scored at it, pools as macro
F1, and r1 against the cached zs+confidence arm by paired pano bootstrap. It never calls a
model. Exploratory: this is not a pre-registered endpoint.

    python scripts/analysis/prompt_276_rewrite_heldout.py --cache-dir <main checkout>/.model_cache
    python scripts/analysis/prompt_276_rewrite_heldout.py --check --cache-dir ...
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import prompt_276_heldout as H  # noqa: E402

REPO_ROOT = H.REPO_ROOT
OUT_JSON = os.path.join(H.OUT_DIR, "rewrite_heldout.json")
OUT_MD = os.path.join(H.OUT_DIR, "rewrite_heldout.md")
MODELS = [("gemini-4-argon", "argon")]
ARMS = {"argon": [("zs", None), ("r1", None, "prompt276-argon-r1")]}
TITLE = "# #276 rewrite arm: held-out run of Argon r1 (regenerated)"


def build(cache_dir):
    res = H.build(cache_dir, models=MODELS, arms=ARMS, primary=[], step="rewrite-heldout")
    # The zs arm's full manual_gold row belongs to Step 5's table, not to this comparison.
    res["legs"] = [r for r in res["legs"] if r["split"] != "manual_gold"]
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=os.path.join(REPO_ROOT, ".model_cache"))
    ap.add_argument("--check", action="store_true",
                    help="regenerate and compare with the committed JSON instead of writing")
    args = ap.parse_args(argv)
    res = build(args.cache_dir)
    rel = os.path.relpath(OUT_JSON, REPO_ROOT).replace("\\", "/")
    if args.check:
        with open(OUT_JSON, encoding="utf-8") as f:
            committed = json.load(f)
        if committed != res:
            print("MISMATCH: %s does not reproduce from the cache" % rel)
            return 1
        print("OK: %s reproduces from the cache (%d scored legs, %d tests)" % (
            rel, sum(r["status"] == "scored" for r in res["legs"]), len(res["tests"])))
        return 0
    os.makedirs(H.OUT_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(H.render_md(res, title=TITLE, script="prompt_276_rewrite_heldout.py", models=MODELS))
    for t in res["thresholds"]:
        print("%-7s threshold %s  dev_select F1 %s" % (
            t["arm"], t.get("threshold", "pending"), t.get("dev_select_f1", "")))
    for p in res["pool_stats"]:
        print("%-10s %-4s %s" % (p["pool"], p["arm"], ("macro F1 %.3f" % p["macro_f1"])
                                 if p["status"] == "scored" else "pending " + ",".join(p["missing"])))
    for t in res["tests"]:
        print("%-10s r1 vs zs  dF1 %+.3f [%+.3f, %+.3f] p %s" % (
            t["pool"], t["macro_delta_f1"], t["ci95"][0], t["ci95"][1], H.fmt_p(t["p"])))
    pend = [r for r in res["legs"] if r["status"] != "scored"]
    if pend:
        print("pending: " + ", ".join("%s/%s %d/%d" % (r["arm"], r["split"], r["n_cached"],
                                                       r["n_panos"]) for r in pend))
    print("wrote %s and .md" % rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
