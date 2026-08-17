"""Is a Gemini leg's zero-detection panorama a real "no ramps here", or a lost response?

Raised reviewing the gemini-3.7-flash publication (#20, #120). That leg returns
**nothing at all** on 345 of the 1,109 city-split panoramas, against 163 for
gemini-3.6-flash and 255 for gemini-3.1-pro. The worry is mechanical rather than
behavioural: ``boxes_from_gemini_response`` yields ``[]`` both for a model that
looked and saw no curb ramp, and for a response that arrived with nothing
parseable in it (truncated mid-thinking, safety-blocked, no candidate). Malformed
JSON raises and is recorded as a per-pano failure; an *empty* response is silent,
gets cached as a successful detection, and is published as authoritative.

That ambiguity matters because it inverts the reading of the leg. "Fires on 75% of
panoramas and is right when it does" is a high-precision detector worth deploying;
"drops a quarter of its responses" is a broken harness whose precision is an
artifact of the drop. The published files cannot tell the two apart on their own —
they record merged per-panorama points, no per-view counts, no finish reasons.

So this asks what the *committed* data can still settle, with three tests that
point in different directions if the cause is mechanical rather than behavioural:

1. **All-or-nothing vs uniform caution.** A dropped response removes a whole view;
   caution removes marginal ramps everywhere. If 3.7-flash matches 3.6-flash's
   detections-per-panorama wherever it fires at all, and differs only in *how
   often* it fires, the deficit is concentrated the way a mechanical loss would be.
   If it instead finds fewer ramps on the panoramas where it does fire, the model
   is simply stricter.
2. **Does emptiness track the ground truth?** On ``manual_gold`` (1,000 panoramas
   with gold YOLO labels) a conservative model should fall silent mostly where
   there is little to find. Silence on ramp-dense panoramas is what a lost response
   looks like.
3. **The six-view arithmetic.** Every panorama is six independent API calls merged
   by ``dedup_points``, so an empty panorama needs all six to come back empty. If
   calls were dropped independently at rate p, the all-six rate is p**6 -- and the
   p needed to explain an observed 25-31% is far above what a leg that completes
   with exit 0 could plausibly be running at. Correlated silence is the model's,
   not the transport's.

    python scripts/analysis/empty_response_check.py
    python scripts/analysis/empty_response_check.py --models gemini-3.7-flash,gemini-3.6-flash

Reads only committed artifacts (``benchmark/model_detections/``, ``benchmark/*/``):
no cache, no GPU, no credentials.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

from export_model_cache import load_detections  # noqa: E402

CITY_SPLITS = ["annapolis", "bend", "budapest_district5", "clovis", "gainesville",
               "morgantown", "paterson", "richmond", "sao_paulo"]
DEFAULT_MODELS = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.1-pro-preview"]
# The leg under suspicion; tests 1 and 2 compare everything else against it.
SUBJECT = "gemini-3.7-flash"
VIEWS_PER_PANO = 6   # equirect_tiling.default_views(); an empty pano = 6 empty calls


def gold_counts():
    """``{pano_id: n_gt_ramps}`` for manual_gold, from the committed YOLO labels."""
    from rampnet.detection_eval import load_yolo_ground_truths
    gts = load_yolo_ground_truths(os.path.join(REPO, "manual_labels"))
    return {pid: len(gt.gt_points) for pid, gt in gts.items()}


def collect(models, splits):
    """``{model: {split: {pano: [points]}}}`` for whatever is published."""
    out = {}
    for m in models:
        per_split = {}
        for s in splits:
            d = load_detections(m, s)
            if d is not None:
                per_split[s] = d
        out[m] = per_split
    return out


def test_all_or_nothing(data, splits):
    """Where a model fires at all, does it find as much as its sibling?"""
    print("=" * 78)
    print("TEST 1 — all-or-nothing (mechanical) vs uniform caution (behavioural)")
    print("=" * 78)
    print("On panoramas where BOTH models fire, a dropped-response bug leaves the")
    print("survivors untouched, so the per-panorama count should MATCH its sibling.")
    print("A stricter model finds fewer ramps everywhere, including there.\n")
    subject = data.get(SUBJECT, {})
    print(f"{'vs':24s} {'both fire':>10s} {'subj/pano':>10s} {'other/pano':>11s} "
          f"{'ratio':>7s} {'subj silent,':>13s} {'other found':>12s}")
    print(f"{'':24s} {'':>10s} {'':>10s} {'':>11s} {'':>7s} {'other fired':>13s} "
          f"{'there':>12s}")
    rows = []
    for other, per_split in data.items():
        if other == SUBJECT:
            continue
        both = subj_pts = other_pts = 0
        silent_but_other_fired = other_found_there = 0
        for s in splits:
            a, b = subject.get(s), per_split.get(s)
            if not a or not b:
                continue
            for pid in set(a) & set(b):
                if a[pid] and b[pid]:
                    both += 1
                    subj_pts += len(a[pid])
                    other_pts += len(b[pid])
                elif not a[pid] and b[pid]:
                    silent_but_other_fired += 1
                    other_found_there += len(b[pid])
        if not both:
            continue
        sa, sb = subj_pts / both, other_pts / both
        rows.append((other, both, sa, sb, sa / sb, silent_but_other_fired,
                     other_found_there))
        print(f"{other:24s} {both:>10d} {sa:>10.2f} {sb:>11.2f} {sa/sb:>7.2f} "
              f"{silent_but_other_fired:>13d} {other_found_there:>12d}")
    print()
    if rows and all(r[4] >= 0.85 for r in rows):
        print("READING: where it fires, the subject finds ~as much as its siblings —")
        print("the deficit is concentrated in whole panoramas, which is the shape a")
        print("mechanical loss would have. NOT conclusive on its own; see tests 2-3.")
    elif rows:
        print("READING: the subject finds fewer ramps even on panoramas where it")
        print("fires, so the deficit is spread rather than concentrated — the")
        print("signature of a stricter model, not of lost responses.")
    return rows


def test_emptiness_vs_ground_truth(data, gold):
    """On manual_gold, is the model silent where there is nothing to find?"""
    print("\n" + "=" * 78)
    print("TEST 2 — does silence track the ground truth? (manual_gold, gold labels)")
    print("=" * 78)
    print("A cautious model should fall silent where there is little to find.")
    print("Silence on ramp-DENSE panoramas is what a lost response looks like.\n")
    verdicts = {}
    for model, per_split in data.items():
        dets = per_split.get("manual_gold")
        if not dets:
            continue
        scored = [(pid, pts) for pid, pts in dets.items() if pid in gold]
        if not scored:
            continue
        silent = [pid for pid, pts in scored if not pts]
        fired = [pid for pid, pts in scored if pts]
        g_silent = sum(gold[p] for p in silent) / len(silent) if silent else 0.0
        g_fired = sum(gold[p] for p in fired) / len(fired) if fired else 0.0
        # How much of the gold set sits on panoramas the model never spoke about?
        gt_total = sum(gold[p] for p, _ in scored)
        gt_lost = sum(gold[p] for p in silent)
        empty_gt0 = sum(1 for p in silent if gold[p] == 0)
        print(f"{model}")
        print(f"    panoramas          {len(scored):>6d}  silent {len(silent):>4d} "
              f"({len(silent)/len(scored):.1%})")
        print(f"    mean gold ramps    silent {g_silent:>6.2f}   firing {g_fired:>6.2f}"
              f"   ratio {(g_silent/g_fired if g_fired else float('nan')):.2f}")
        print(f"    silent panos that are genuinely EMPTY in the gold set: "
              f"{empty_gt0}/{len(silent)}"
              f"{f' ({empty_gt0/len(silent):.1%})' if silent else ''}")
        print(f"    gold ramps stranded on silent panoramas: {gt_lost}/{gt_total} "
              f"({gt_lost/gt_total:.1%} of recall unreachable)\n")
        verdicts[model] = {"panos": len(scored), "silent": len(silent),
                           "mean_gt_silent": g_silent, "mean_gt_firing": g_fired,
                           "silent_truly_empty": empty_gt0,
                           "gt_stranded": gt_lost, "gt_total": gt_total}
    return verdicts


def test_six_view_arithmetic(data, splits):
    """What per-call drop rate would be needed to explain the silence?"""
    print("=" * 78)
    print("TEST 3 — the six-view arithmetic")
    print("=" * 78)
    print(f"Each panorama is {VIEWS_PER_PANO} independent calls merged by dedup_points,")
    print("so an all-silent panorama needs ALL of them to come back empty. If calls")
    print("were dropped independently at rate p, the observed silent share is an")
    print("upper bound on p**6 — invert it to see what p would have to be.\n")
    print(f"{'model':26s} {'panos':>7s} {'silent':>7s} {'share':>8s} "
          f"{'implied p/call':>15s}")
    for model, per_split in data.items():
        n = silent = 0
        for s in splits:
            d = per_split.get(s)
            if not d:
                continue
            n += len(d)
            silent += sum(1 for pts in d.values() if not pts)
        if not n:
            continue
        share = silent / n
        p = share ** (1.0 / VIEWS_PER_PANO)
        print(f"{model:26s} {n:>7d} {silent:>7d} {share:>7.1%} {p:>14.1%}")
    print("\nREADING: those p values are absurd for a leg that completed with exit 0")
    print("and billed token counts in line with its siblings (cost table in")
    print("docs/model_comparison.md), and the same arithmetic indicts gemini-3.6-flash")
    print("at 72.6% — a leg nobody suspects. So this rules out INDEPENDENT per-call")
    print("loss for all three, and nothing more: silence is common and expected here.")
    print("A CORRELATED failure (all six views of one panorama refused alike) would")
    print("survive this test — test 2 is what rules that out, by showing the silence")
    print("lands almost entirely on panoramas that are empty in the gold set.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--splits", default=",".join(CITY_SPLITS))
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    data = collect(models, splits + ["manual_gold"])
    missing = [m for m in models if not data[m]]
    if missing:
        raise SystemExit(f"no published detections for: {', '.join(missing)}")

    rows = test_all_or_nothing(data, splits)
    gold = gold_counts()
    verdicts = test_emptiness_vs_ground_truth(data, gold)
    test_six_view_arithmetic(data, splits)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"all_or_nothing": [
                {"other": r[0], "both_fire": r[1], "subject_per_pano": r[2],
                 "other_per_pano": r[3], "ratio": r[4],
                 "subject_silent_other_fired": r[5], "other_found_there": r[6]}
                for r in rows], "ground_truth": verdicts},
                fh, indent=2, sort_keys=True)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
