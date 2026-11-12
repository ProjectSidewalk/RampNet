"""The committed pano lists for the #276 prompt experiments: a seeded dev split of
sao_paulo and a seeded 250-pano subset of manual_gold.

Why a file at all: ``compare.py --limit N`` takes the FIRST N panos, so "tune on half
of sao_paulo and select on the other half" has no expression in the harness without a
list. ``--pano-list FILE@KEY`` reads these. Committing them (rather than deriving them
at run time) means the lists in a ledger row, a trajectory file and a published number
are the same bytes a year from now even if the bundle gains a pano.

    python scripts/analysis/prompt_opt_276_splits.py            # (re)write both files
    python scripts/analysis/prompt_opt_276_splits.py --check    # exit 1 if a file drifted

sao_paulo is the dev split by the #51 convention (the one split every tuning decision
is allowed to see). ``dev_train`` (60) feeds the optimizer its exemplars; ``dev_select``
(65) picks among candidate definitions and thresholds, once. Both halves are reported
with their GT ramp counts so a reader can see they are balanced enough; the shuffle is
plain, not stratified -- 125 panos is too few to stratify on anything but luck.

manual_gold-250 is the budgeted version of the 1,000-pano manual_gold split for the
non-zero-shot arms (the full 1,000 is run only for the zero-shot baselines).
"""
import argparse
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

SEED = 276
OUT_DIR = os.path.join(REPO, "benchmark", "prompt_opt_276")
SAO_PAULO_SPLIT = os.path.join(OUT_DIR, "sao_paulo_split.json")
MANUAL_GOLD_250 = os.path.join(OUT_DIR, "manual_gold_250.json")
DEV_TRAIN_N = 60
MANUAL_GOLD_N = 250


def _gts(bundle):
    import compare as C
    records, verdicts, _ = C.load_bundle(bundle)
    if verdicts is not None:
        return C.ground_truths_from_verdicts(records, verdicts)
    return C.load_manual_ground_truths(bundle)


def _ramps(gt):
    return len(gt.gt_points)


def build_sao_paulo_split():
    gts = _gts(os.path.join(REPO, "benchmark", "sao_paulo"))
    ids = sorted(gts)                       # sorted first: the shuffle must not depend on file order
    random.Random(SEED).shuffle(ids)
    train, select = ids[:DEV_TRAIN_N], ids[DEV_TRAIN_N:]
    return {
        "issue": "ProjectSidewalk/RampNet#276",
        "source": "benchmark/sao_paulo (verdicts.json keys, sorted, then shuffled)",
        "seed": SEED,
        "generator": "scripts/analysis/prompt_opt_276_splits.py",
        "n_panos": len(ids),
        "gt_ramps": {"dev_train": sum(_ramps(gts[p]) for p in train),
                     "dev_select": sum(_ramps(gts[p]) for p in select)},
        "dev_train": train,
        "dev_select": select,
    }


def build_manual_gold_250():
    gts = _gts(os.path.join(REPO, "benchmark", "manual_gold"))
    ids = sorted(gts)
    random.Random(SEED).shuffle(ids)
    chosen = sorted(ids[:MANUAL_GOLD_N])    # sorted for a readable diff; order is not meaningful here
    return {
        "issue": "ProjectSidewalk/RampNet#276",
        "source": "benchmark/manual_gold (gt keys, sorted, then shuffled; first 250, re-sorted)",
        "seed": SEED,
        "generator": "scripts/analysis/prompt_opt_276_splits.py",
        "n_panos": len(chosen),
        "n_source_panos": len(ids),
        "gt_ramps": sum(_ramps(gts[p]) for p in chosen),
        "panos": chosen,
    }


def _dump(obj):
    return json.dumps(obj, indent=1) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="Rebuild both lists and compare with the committed files; exit 1 on drift.")
    args = ap.parse_args(argv)
    built = {SAO_PAULO_SPLIT: build_sao_paulo_split(), MANUAL_GOLD_250: build_manual_gold_250()}
    if args.check:
        bad = 0
        for path, obj in built.items():
            if not os.path.exists(path):
                print(f"MISSING {os.path.relpath(path, REPO)}")
                bad += 1
                continue
            with open(path, encoding="utf-8", newline="") as f:
                same = f.read() == _dump(obj)
            print(("ok      " if same else "DRIFTED ") + os.path.relpath(path, REPO))
            bad += 0 if same else 1
        return 1 if bad else 0
    os.makedirs(OUT_DIR, exist_ok=True)
    for path, obj in built.items():
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(_dump(obj))
        extra = (f"dev_train {len(obj['dev_train'])} / dev_select {len(obj['dev_select'])}, "
                 f"GT ramps {obj['gt_ramps']}" if "dev_train" in obj
                 else f"{obj['n_panos']} of {obj['n_source_panos']}, GT ramps {obj['gt_ramps']}")
        print(f"wrote {os.path.relpath(path, REPO)}: {extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
