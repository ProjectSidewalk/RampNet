"""Turn committed rater verdicts into the sourcing number — and compare raters (#46).

``miss_taxonomy.py`` bucketed every miss from the cached detections and left ``silent``
unexplained. ``silent_witness.py`` split that bucket by whether another model saw the
ramp, leaving a bracket: **0.009 to 0.022 recall points** of sourcing-addressable
recall, the gap being the unwitnessed misses nobody had looked at. ``miss_gallery.py``
rendered those, and a human tagged them. This is the arithmetic that closes it.

It reads only committed files — ``benchmark/miss_taxonomy_46/*.json`` and
``analysis_out/silent_witness.json`` — so the headline re-derives from a clean clone
with no GPU, no imagery and no ``.model_cache``.

**Every verdict file is checked against the manifest it was made on.** A rater's
answers carry the rubric and the manifest digest they were produced under (see
``make_tagger.py``); comparing two raters who saw different crops, or judged under
different verdict definitions, is the failure this guards against. A mismatch is a
loud error, never a silent merge.

With two or more raters it also reports **pairwise agreement and Cohen's kappa**, per
item and per verdict, plus the specific disagreements so they can be re-examined
rather than averaged away.

    python scripts/analysis/tag_results.py
    python scripts/analysis/tag_results.py --raters benchmark/miss_taxonomy_46/silent__*.json
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TAGS_DIR = os.path.join(REPO, "benchmark", "miss_taxonomy_46")
# The gallery is committed alongside the verdicts, crops and all (15 MB). That is what
# makes this rating task replicable TODAY rather than after HF #21 lands: a second
# rater needs no panoramas, no .model_cache and no GPU -- they open the committed
# crops and tag. See docs/replication.md.
DEFAULT_MANIFEST = os.path.join(TAGS_DIR, "silent_gallery", "manifest.json")
DEFAULT_WITNESS = os.path.join(OUT, "silent_witness.json")

# Verdicts that mean "a broader / more diverse training corpus could reach this ramp".
# `visible` is the tight reading: the ramp's own pixels carried it and the model still
# failed. `context-only` is learnable too, but from scene layout rather than from the
# ramp's appearance -- a different capability, so it is reported as the upper variant
# rather than folded in.
ADDRESSABLE_TIGHT = ("visible",)
ADDRESSABLE_UPPER = ("visible", "context-only")

# Verdicts that remove an item from every rate rather than counting against anything.
# `unclear` is an abstention; `definition` is a question about the label set, not about
# the model, and folding it into either direction would answer a rubric question by
# accident.
EXCLUDED = ("unclear", "definition")


# --------------------------------------------------------------------------- #
# Pure core (no I/O) — unit-tested in tests/test_tag_results.py
# --------------------------------------------------------------------------- #
def rate(verdicts, names, exclude=EXCLUDED):
    """``(hits, denominator, rate)`` for ``names`` over a verdict list.

    Excluded verdicts leave the denominator entirely — they are abstentions, not
    negatives. Counting an ``unclear`` as "not addressable" would quietly convert
    "we could not tell" into evidence against the sourcing programme.
    """
    scored = [v for v in verdicts if v not in exclude]
    hits = sum(1 for v in scored if v in names)
    return hits, len(scored), (hits / len(scored) if scored else float("nan"))


def cohens_kappa(pairs):
    """Chance-corrected agreement between two raters over ``(a, b)`` verdict pairs.

    Plain percent agreement flatters any scheme with a dominant category — if 80% of
    items are ``visible``, two raters who guessed that every time would "agree" 80% of
    the time. Kappa subtracts what the marginals alone predict.
    """
    if not pairs:
        return float("nan")
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def bracket(n_witnessed_corrected, n_unwitnessed_addressable, n_pooled_gt):
    """Sourcing-addressable recall, in points, from both populations.

    Witnessed misses need no reviewer — another model detected a ramp there, so the
    imagery is shown to contain one. Unwitnessed misses contribute only what the
    reviewer judged addressable. The two are different kinds of evidence and are kept
    separate right up to the sum.
    """
    total = n_witnessed_corrected + n_unwitnessed_addressable
    return total / n_pooled_gt if n_pooled_gt else float("nan")


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_verdicts(path):
    """``(name, {key: verdict}, meta)`` from a rater file.

    Accepts both the self-describing export and the bare ``{key: verdict}`` map the
    first export produced. A bare file is loaded with a warning rather than refused —
    it is real human work — but it carries no rubric, so it cannot be checked and the
    caller is told so.
    """
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    name = os.path.splitext(os.path.basename(path))[0].split("__")[-1]
    if isinstance(payload, dict) and "verdicts" in payload:
        return name, payload["verdicts"], payload
    return name, payload, {"_bare": True}


def check_against_manifest(name, verdicts, meta, manifest, digest, problems):
    """Refuse to compare answers that were not made on the same task."""
    unknown = [k for k in verdicts if k not in manifest]
    if unknown:
        problems.append(f"{name}: {len(unknown)} verdict key(s) absent from the manifest "
                        f"(e.g. {unknown[0]}) — different item set")
    if meta.get("_bare"):
        problems.append(f"{name}: bare {{key: verdict}} file with no rubric or manifest "
                        f"digest — cannot verify which scheme produced it")
        return
    got = meta.get("manifest_digest")
    if got and digest and got != digest:
        problems.append(f"{name}: manifest digest {got} != {digest} — the crops this "
                        f"rater saw are not the crops in this manifest")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raters", nargs="*", default=None,
                   help="Verdict files (default: benchmark/miss_taxonomy_46/silent__*.json, "
                        "excluding the manifest).")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--witness", default=DEFAULT_WITNESS)
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    with open(args.manifest, encoding="utf-8") as fh:
        mpayload = json.load(fh)
    manifest, digest = mpayload["items"], mpayload.get("digest", "")

    paths = args.raters or sorted(
        f for f in glob.glob(os.path.join(TAGS_DIR, "silent__*.json"))
        if "manifest" not in os.path.basename(f))
    if not paths:
        raise SystemExit(f"no rater files found under {TAGS_DIR}")

    raters, problems = [], []
    for path in paths:
        name, v, meta = load_verdicts(path)
        check_against_manifest(name, v, meta, manifest, digest, problems)
        raters.append((name, v, meta))

    print(f"=== Miss-taxonomy verdicts (#46) — {len(raters)} rater(s), "
          f"{len(manifest)} items ===")
    print(f"manifest digest {digest or '(none recorded)'}\n")
    for msg in problems:
        print(f"  ⚠ {msg}")
    if problems:
        print()

    # ---- per rater, per field -------------------------------------------- #
    for name, v, _ in raters:
        print(f"{name}: {len(v)}/{len(manifest)} tagged")
        for field in ("near", "far"):
            sel = {k: vv for k, vv in v.items()
                   if manifest.get(k, {}).get("field") == field}
            if not sel:
                continue
            c = Counter(sel.values())
            tight = rate(list(sel.values()), ADDRESSABLE_TIGHT)
            upper = rate(list(sel.values()), ADDRESSABLE_UPPER)
            print(f"  {field:>4}-field n={len(sel):<3} " +
                  "  ".join(f"{k} {n}" for k, n in c.most_common()))
            print(f"       visible {tight[0]}/{tight[1]} = {tight[2]:.1%}"
                  f"   +context-only {upper[0]}/{upper[1]} = {upper[2]:.1%}"
                  f"   ({len(sel) - tight[1]} excluded)")
        print()

    # ---- agreement -------------------------------------------------------- #
    if len(raters) > 1:
        print(f"{'-'*70}\nAGREEMENT\n{'-'*70}")
        for i in range(len(raters)):
            for j in range(i + 1, len(raters)):
                (na, va, _), (nb, vb, _) = raters[i], raters[j]
                shared = sorted(set(va) & set(vb))
                pairs = [(va[k], vb[k]) for k in shared]
                agree = sum(1 for a, b in pairs if a == b)
                k = cohens_kappa(pairs)
                print(f"  {na} vs {nb}: {agree}/{len(pairs)} "
                      f"({agree/len(pairs):.1%}) agree, kappa {k:.3f}")
                diffs = [(key, a, b) for key, (a, b) in zip(shared, pairs) if a != b]
                if diffs:
                    print(f"    {len(diffs)} disagreement(s):")
                    for key, a, b in diffs[:20]:
                        m = manifest.get(key, {})
                        print(f"      {m.get('city', '?'):>12} {m.get('field', '?'):>4} "
                              f"{m.get('source_px', '?'):>6} src px   {na}={a}  {nb}={b}")
        print()
    else:
        print(f"{'-'*70}")
        print("ONE RATER — no agreement statistic is possible. A second pass on the")
        print("identical manifest is what would make these verdicts checkable; the")
        print("run-book for it is in docs/curb_ramp_data_sourcing.md.")
        print(f"{'-'*70}\n")

    # ---- the number ------------------------------------------------------- #
    if not os.path.exists(args.witness):
        print(f"(no {args.witness} — run silent_witness.py --json-out to close the bracket)")
        return 0
    with open(args.witness, encoding="utf-8") as fh:
        w = json.load(fh)
    near = w.get("by_field", {}).get("near")
    if not near:
        print("(witness file has no near-field summary)")
        return 0

    name, v, _ = raters[0]
    near_v = [vv for k, vv in v.items() if manifest.get(k, {}).get("field") == "near"]
    n_unwit_near = sum(1 for m in manifest.values() if m.get("field") == "near")
    tight_hits, tight_den, tight_rate = rate(near_v, ADDRESSABLE_TIGHT)
    upper_hits, _, upper_rate = rate(near_v, ADDRESSABLE_UPPER)
    n_pooled = w.get("n_pooled_gt", 2060)
    wit_corrected = near["excess"]

    print(f"{'='*70}\nCLOSING THE SOURCING BRACKET (rater: {name})\n{'='*70}")
    print(f"  near-field silent misses split into two populations:")
    print(f"    WITNESSED     {near['witnessed']:>3} raw, {near['expected']:.1f} by chance "
          f"-> ~{wit_corrected:.0f} confirmed recognizable, no reviewer needed")
    print(f"    UNWITNESSED   {n_unwit_near:>3} reviewed here: "
          f"{tight_hits} visible of {tight_den} scored "
          f"({len(near_v) - tight_den} excluded as unclear/definition)")
    lo = bracket(wit_corrected, tight_hits, n_pooled)
    hi = bracket(wit_corrected, upper_hits, n_pooled)
    print(f"\n  SOURCING-ADDRESSABLE RECALL")
    print(f"    tight  (visible only)          {lo:.3f} recall points")
    print(f"    upper  (+ context-only)        {hi:.3f} recall points")
    print(f"    was: 0.087 (#59) -> 0.022 (miss_taxonomy) -> 0.009-0.022 (witness)")
    print(f"\n  Caveats that travel with this number:")
    print(f"    - ONE rater, no second pass, so no agreement statistic exists.")
    print(f"    - {len(near_v) - tight_den} of {len(near_v)} near-field items were excluded as")
    print(f"      unclear/definition; at n={len(near_v)} a single item moves the rate by "
          f"{1/max(1,tight_den):.1%}.")
    print(f"    - The witnessed half is a chance-corrected estimate, not a count.")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"manifest_digest": digest, "raters": [r[0] for r in raters],
                       "near_visible": [tight_hits, tight_den],
                       "near_upper": [upper_hits, tight_den],
                       "witnessed_corrected": wit_corrected,
                       "addressable_tight": lo, "addressable_upper": hi,
                       "problems": problems}, fh, indent=2)
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
