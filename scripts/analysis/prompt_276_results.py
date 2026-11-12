#!/usr/bin/env python
"""#276 dev-split results (Step 3 battery + Step 6 thinking arm), regenerated from the caches.

Every number in docs/vlm_prompt_276.md's dev-split tables comes from here, not from a run log:
for each (model, leg) the script rebuilds the detector exactly as compare.py would, looks
every pano of the leg up in the cache, and scores the leg only when all of them are there.
It never calls a model: a leg with uncached panos is reported as pending.

    python scripts/analysis/prompt_276_results.py            # write analysis_out/prompt_opt_276/step3_results.{json,md}
    python scripts/analysis/prompt_276_results.py --check    # regenerate and compare with the committed JSON

Legs (the Step 3 protocol of the plan):
    seed            the as-run prompt on dev_train and on dev_select        (main cache)
    seed_repeat     the as-run prompt on dev_train in a fresh cache dir     (.model_cache_rep1)
    para1..para3    meaning-preserving paraphrases of the seed, dev_select
    expert          Jon's definition (V4 + V5 recall clause), dev_select
    expert-long     its 515-word structured form, dev_select
    zs_conf         seed + elicited per-box confidence, all 125 panos       (Gemini only)
    thinking_low    seed prompt at --gemini-thinking-level low, dev_select   (Step 6, Argon only)
    thinking_high   seed prompt at --gemini-thinking-level high, dev_select  (Step 6, Argon only)
    def276-<m>-r<k> the optimizer's k-th candidate on dev_train, confidence on (Step 4, as
                    prompt_optimize.py evaluate scored it; the trajectory holds the verdict)
    def276-<m>-alt  the image-free alternative rewording on dev_select            (Step 4)

The thinking legs compare against the seed dev_select leg, whose thinking level is the
model's default (unset, so absent from the signature).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "model_comparison"))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)

import compare as C  # noqa: E402
import detectors as D  # noqa: E402
from fp_taxonomy import _compare_args  # noqa: E402
from rampnet.detection_eval import aggregate, prediction_confidence, radius_sq_for, score_pano  # noqa: E402

CITY = "sao_paulo"
SPLIT_FILE = "benchmark/prompt_opt_276/sao_paulo_split.json"
OUT_DIR = os.path.join(REPO_ROOT, "analysis_out", "prompt_opt_276")
OUT_JSON = os.path.join(OUT_DIR, "step3_results.json")
OUT_MD = os.path.join(OUT_DIR, "step3_results.md")

MODELS = [
    ("gemini", "gemini-3.8-flash", "flash38"),
    ("gemini", "gemini-4-argon", "argon"),
    ("qwen", "Qwen/Qwen3-VL-32B-Instruct", "qwen32b"),
]
DEFINITIONS = ["para1", "para2", "para3", "expert", "expert-long"]
THINKING_MODELS = {"gemini-4-argon"}      # Step 6 ran the thinking arm on Argon only
THINKING_LEVELS = ["low", "high"]
DEFINITIONS_DIR = os.path.join(REPO_ROOT, "benchmark", "prompt_opt_276", "definitions")


def optimizer_definitions(short):
    """The committed Step 4 definitions of one model: (name, split, confidence), in order."""
    import glob
    names = sorted(os.path.basename(p)[:-4] for p in
                   glob.glob(os.path.join(DEFINITIONS_DIR, "def276-%s-*.txt" % short)))
    cands = [n for n in names if n.rsplit("-", 1)[1].startswith("r")]
    alts = [n for n in names if n.endswith("-alt")]
    return [(n, "dev_train", True) for n in cands] + [(n, "dev_select", False) for n in alts]


def legs_for(provider, model_id, short):
    """(leg, split, definition, confidence, thinking, cache) per model.

    Gemini adds the confidence leg and its Step 4 optimizer legs; the thinking-arm models
    add the two thinking legs.
    """
    legs = [("seed", "dev_train", None, False, None, "main"),
            ("seed_repeat", "dev_train", None, False, None, "rep1"),
            ("seed", "dev_select", None, False, None, "main")]
    legs += [(d, "dev_select", d, False, None, "main") for d in DEFINITIONS]
    if provider == "gemini":
        legs.append(("zs_conf", "all", None, True, None, "main"))
    if model_id in THINKING_MODELS:
        legs += [("thinking_" + t, "dev_select", None, False, t, "main") for t in THINKING_LEVELS]
    if provider == "gemini":
        legs += [(n, which, n, conf, None, "main") for n, which, conf in optimizer_definitions(short)]
    return legs


def split_ids(split, records, which):
    if which == "all":
        return sorted(records)
    return list(split[which])


def score_leg(points, gts, rsq):
    scores = [score_pano(points[p], gts[p], rsq) for p in gts]
    rep = aggregate(scores)
    confs = [prediction_confidence(pt) for pts in points.values() for pt in pts]
    confs = [c for c in confs if c is not None]
    out = {"precision": round(rep.precision, 4), "recall": round(rep.recall, 4),
           "f1": round(rep.f1, 4),
           "precision_ci": [round(v, 4) for v in rep.precision_ci],
           "recall_ci": [round(v, 4) for v in rep.recall_ci],
           "tp": sum(s.tp for s in scores), "fp": sum(s.fp for s in scores),
           "n_gt": sum(s.n_gt for s in scores), "n_pred": sum(len(points[p]) for p in gts)}
    if confs:
        out["n_scored_boxes"] = len(confs)
        out["n_distinct_scores"] = len(set(round(c, 4) for c in confs))
    return out


def build(main_cache, rep_cache):
    bundle = os.path.join(REPO_ROOT, "benchmark", CITY)
    records, verdicts, _panos_dir = C.load_bundle(bundle)
    gts_all = C.ground_truths_from_verdicts(records, verdicts)
    with open(os.path.join(REPO_ROOT, SPLIT_FILE), encoding="utf-8") as f:
        split = json.load(f)
    rsq = radius_sq_for()
    caches = {"main": (main_cache, C.DetectionCache(main_cache, enabled=True)),
              "rep1": (rep_cache, C.DetectionCache(rep_cache, enabled=True))}
    rows = []
    for provider, model_id, short in MODELS:
        for leg, which, definition, confidence, thinking, cache_name in legs_for(provider, model_id, short):
            cache_dir, cache = caches[cache_name]
            args = _compare_args(cache_dir)
            args.definition = definition
            if provider == "gemini":
                args.gemini_confidence = True if confidence else None
                args.gemini_thinking_level = thinking
            label, det = D.build_detector(provider, model_id, records, args)
            sig = det.signature()
            pids = split_ids(split, records, which)
            points = {p: cache.get(C.cache_key(label, sig, CITY, p)) for p in pids}
            n_cached = sum(1 for v in points.values() if v is not None)
            row = {"model": model_id, "short": short, "leg": leg, "split": which,
                   "definition": definition, "confidence": bool(confidence),
                   "thinking_level": thinking,
                   "cache": os.path.basename(os.path.normpath(cache_dir)),
                   "n_panos": len(pids), "n_cached": n_cached}
            if n_cached == len(pids):
                row["status"] = "scored"
                row.update(score_leg(points, {p: gts_all[p] for p in pids}, rsq))
            else:
                row["status"] = "pending"
            rows.append(row)
    return {"issue": "ProjectSidewalk/RampNet#276", "city": CITY, "split_file": SPLIT_FILE,
            "match_radius": 0.022, "rows": rows}


def render_md(result):
    lines = ["# #276 dev-split results: Step 3 prompt battery and Step 6 thinking arm (regenerated)", "",
             "Bundle `benchmark/%s`, splits `%s`, match radius %s. P/R with 95%% Wilson "
             "intervals; F1 at the operating point. A pending leg has uncached panos (count "
             "shown) and no number. Regenerate with "
             "`python scripts/analysis/prompt_276_results.py`." % (
                 result["city"], result["split_file"], result["match_radius"]), ""]
    for _, model_id, _short in MODELS:
        lines += ["## %s" % model_id, "",
                  "| leg | split | panos | P | R | F1 | tp/fp/n_gt | notes |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in result["rows"]:
            if r["model"] != model_id:
                continue
            if r["status"] != "scored":
                lines.append("| %s | %s | %d | | | | | pending (%d/%d cached) |" % (
                    r["leg"], r["split"], r["n_panos"], r["n_cached"], r["n_panos"]))
                continue
            note = []
            if r["confidence"]:
                note.append("%d boxes, %d distinct scores" % (
                    r.get("n_scored_boxes", 0), r.get("n_distinct_scores", 0)))
            if r.get("thinking_level"):
                note.append("thinking %s" % r["thinking_level"])
            if r["cache"] != ".model_cache":
                note.append("cache `%s`" % r["cache"])
            lines.append("| %s | %s | %d | %.3f (%.3f-%.3f) | %.3f (%.3f-%.3f) | **%.3f** | %d/%d/%d | %s |" % (
                r["leg"], r["split"], r["n_panos"],
                r["precision"], r["precision_ci"][0], r["precision_ci"][1],
                r["recall"], r["recall_ci"][0], r["recall_ci"][1],
                r["f1"], r["tp"], r["fp"], r["n_gt"], "; ".join(note)))
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=os.path.join(REPO_ROOT, ".model_cache"))
    ap.add_argument("--repeat-cache-dir", default=os.path.join(REPO_ROOT, ".model_cache_rep1"))
    ap.add_argument("--check", action="store_true",
                    help="regenerate and compare with the committed JSON instead of writing")
    args = ap.parse_args(argv)
    result = build(args.cache_dir, args.repeat_cache_dir)
    rel_json = os.path.relpath(OUT_JSON, REPO_ROOT).replace("\\", "/")
    if args.check:
        with open(OUT_JSON, encoding="utf-8") as f:
            committed = json.load(f)
        if committed != result:
            diff = [(a["short"], a["leg"], a["split"])
                    for a, b in zip(committed["rows"], result["rows"]) if a != b]
            print("MISMATCH: %d row(s) differ from %s: %s" % (len(diff), rel_json, diff[:6]))
            return 1
        print("OK: %s reproduces from the caches (%d scored legs)" % (
            rel_json, sum(r["status"] == "scored" for r in result["rows"])))
        return 0
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(render_md(result))
    for r in result["rows"]:
        tag = "%-8s %-12s %-10s" % (r["short"], r["leg"], r["split"])
        if r["status"] == "scored":
            print("%s F1 %.3f  P %.3f  R %.3f  %d/%d/%d" % (
                tag, r["f1"], r["precision"], r["recall"], r["tp"], r["fp"], r["n_gt"]))
        else:
            print("%s pending (%d/%d cached)" % (tag, r["n_cached"], r["n_panos"]))
    print("wrote %s and .md" % rel_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
