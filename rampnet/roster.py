"""The challenger roster: who is in the model comparison, since when, and how dense.

One table, ``ROSTER``, replaces three tuples that used to be kept in step by hand —
``CHALLENGERS``, ``SPARSE`` and ``DENSE`` — plus the per-provider default model ids
that were copied into four argument parsers. Adding a model to the benchmark is now
one entry here.

**Why this lives in the package rather than in an analysis script.** Both
``scripts/analysis/*`` and ``scripts/model_comparison/*`` need it, it must import
without torch so a fresh clone can score published detections on a laptop, and the
CPU-only test suite has to cover it. It follows the splits registry in
``scripts/analysis/miss_decomposition.py``, which eight scripts already import.

**The one thing in here that must not move: ``WITNESS_POOL_46``.** It is written out
literally rather than derived from ``ROSTER`` precisely so that adding a challenger
cannot touch it. See the comment on it.

Five properties of an entry are worth stating because they are easy to get wrong:

* ``density`` is **evidence, not configuration** — it comes from the measured
  boxes-per-panorama in ``docs/model_comparison.md`` (sparse models emit 1-4, the
  open-vocabulary detectors 55-88). A new arm's density is unknown until it has been
  run, so it is ``None``, and ``density_of`` raises rather than guessing. The old code
  silently treated an unclassified model as dense.
* ``standing`` separates *scored in the roster tables* from *published but not scored*.
  A leg can be run, verified and committed long before its write-up lands
  (``gemini-3.7-flash``, #120); that is an omission of a write-up, not of a run, and
  the distinction is data here rather than prose in a doc.
* ``label`` is the resolved model id: the result-table row, and the key the detection
  cache is already written under. For nearly every provider it is derivable from the
  spec; ``LABEL_OVERRIDES`` covers the ones where it is not.
* ``pins`` is what makes a registered thing a **leg** rather than a model. A knob that
  enters the detection signature splits one model id into several sets of detections:
  ``claude-sonnet-5`` at effort ``low`` and at effort ``high`` are different runs with
  different cache keys and different results. Each is its own entry, and ``pins``
  names the knob it holds, as ``(("claude_effort", "high"),)``.
* ``published_as`` is the filename stem under ``benchmark/model_detections/``, and it
  defaults to ``label``. It exists because ``label`` cannot carry a pin: the label is
  baked into cache keys that were already paid for, so renaming it orphans the
  detections. The two came apart the first time a provider had a pin, and both legs
  wanted ``claude-sonnet-5__annapolis.json``, the second silently overwriting the
  first. **Once any leg of a model needs disambiguating, give every leg of that model
  the same treatment** — a directory where one file is bare and its sibling is
  qualified reads as though the bare one is the whole model.
"""
from collections import namedtuple
import os
import re


def slug(label):
    """Filesystem-safe model id.

    ``IDEA-Research/grounding-dino-base`` -> ``IDEA-Research__grounding-dino-base``.
    Defined here rather than in the exporter because the roster is what knows the set
    of published names, and the test that checks the directory against the registry
    must not import the exporter's dependencies to spell one filename.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "__", label)

#: One registered leg. ``spec`` is the ``compare.py --models`` token; ``pins`` are the
#: extra signature-entering settings it was run with, if any.
Challenger = namedtuple(
    "Challenger", "spec label provider density standing added note pins published_as",
    defaults=((), None))

# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #
#: Every model the benchmark knows about, scored or not, in results-table order.
#:
#: ``rampnet`` is a member: it is the row every other row is read against, and having
#: it here is what makes "8 models" and "7 challengers" the same statement rather than
#: the contradiction that sat in docs/model_comparison.md and docs/replication.md.
ROSTER = (
    Challenger(
        spec="rampnet", label="rampnet", provider="rampnet",
        density="sparse", standing=True, added="2026-07-22",
        note="The subject, not a challenger. Read from the bundle's records.jsonl at "
             "the deployed threshold, not from .model_cache — it carries no detector "
             "signature. 4.2 boxes/pano."),
    Challenger(
        spec="gemini:gemini-3.6-flash", label="gemini-3.6-flash", provider="gemini",
        density="sparse", standing=True, added="2026-07-22",
        note="Hosted chat VLM, boxes without scores, so one operating point."),
    Challenger(
        spec="gemini:gemini-3.1-pro-preview", label="gemini-3.1-pro-preview",
        provider="gemini", density="sparse", standing=True, added="2026-07-22",
        note="Strongest general model on most splits until claude-opus-5 (#122)."),
    Challenger(
        spec="qwen:Qwen/Qwen3-VL-8B-Instruct", label="Qwen/Qwen3-VL-8B-Instruct",
        provider="qwen", density="sparse", standing=True, added="2026-07-22",
        note="Open-weight chat VLM. Outranks its own 32B on four splits — the "
             "benchmark's only ranking inversion; see the 32B-caution mechanism."),
    Challenger(
        spec="qwen:Qwen/Qwen3-VL-32B-Instruct", label="Qwen/Qwen3-VL-32B-Instruct",
        provider="qwen", density="sparse", standing=True, added="2026-07-22",
        note="Scaling flipped the failure mode instead of fixing it: it stops firing."),
    Challenger(
        spec="molmo:allenai/Molmo2-8B", label="allenai/Molmo2-8B", provider="molmo",
        density="sparse", standing=True, added="2026-07-23",
        note="Points rather than boxes — RampNet's native output format, so no "
             "box-to-point reduction. Needs its own env at transformers==4.57.1."),
    Challenger(
        spec="owlv2", label="google/owlv2-large-patch14-ensemble", provider="owlv2",
        density="dense", standing=True, added="2026-07-22",
        note="Open-vocabulary detector with calibrated scores, so a real PR curve. "
             "55-88 boxes/pano: most of its recall is what the match radius hands out "
             "for free, hence the chance corrections everywhere."),
    Challenger(
        spec="gdino", label="IDEA-Research/grounding-dino-base", provider="gdino",
        density="dense", standing=True, added="2026-07-22",
        note="Open-vocabulary detector, scored boxes. Dense for the same reason."),

    # --- published, but not scored in the roster tables ---------------------- #
    Challenger(
        spec="vistas:curb-cut", label="mask2former-vistas-curb-cut", provider="vistas",
        density="sparse", standing=False, added="2026-08-18",
        note="Supervised-transfer baseline (#126): the one class of challenger the "
             "roster lacked, since every other member is zero-shot. Mapillary Vistas "
             "v1.2 'Curb Cut' read off facebook/mask2former-swin-large-mapillary-"
             "vistas-semantic, no training. A BASELINE, never a supervision source — "
             "the paper (arXiv 2508.09415) already rejected these labels as too "
             "broad, driveways included. Density measured 2026-08-18 on richmond: "
             "4.48 boxes/pano, the same class as RampNet's 4.2 and an order of "
             "magnitude under the open detectors. Scored on richmond only so far."),
    Challenger(
        spec="vistas:curb-cut+curb", label="mask2former-vistas-curb-cut+curb",
        provider="vistas", density=None, standing=False, added="2026-08-18",
        note="Second #126 arm: 'Curb Cut' unioned with 'Curb'. Vistas draws that "
             "boundary somewhere we do not, so this measured whether recall hides on "
             "the other side of it. It does not: on richmond the union LOSES recall "
             "(0.697 -> 0.648) while precision collapses (0.419 -> 0.127), because "
             "'Curb' fuses adjacent ramps into one component and fires along every "
             "kerb line. Density is left unclassified on purpose: 13.31 boxes/pano "
             "sits between the sparse group (1-4) and the open detectors (55-88), so "
             "the binary does not apply and density_of should refuse rather than "
             "round. Kept as a recorded negative result, not a live arm."),
    Challenger(
        spec="gemini:gemini-3.7-flash", label="gemini-3.7-flash", provider="gemini",
        density="sparse", standing=False, added="2026-08-14",
        note="Run on all ten splits and published (#120); held out of the scored "
             "tables until its write-up lands, so every table is one consistent set. "
             "Promoting it is this entry's `standing` field plus a re-run of "
             "fp_taxonomy/null_recall — the #46 human pass is no longer affected. "
             "Density measured 2026-08-18 off the published detections: 1.90 "
             "boxes/pano over 2,109 panos, against gemini-3.6-flash's 2.34 and "
             "OWLv2's 72.77."),

    # The supervised YOLO baseline (#51). Ten splits each, scored under the
    # pre-registered #71 protocol and written up in their own table in
    # docs/model_comparison.md -- so they are published and scored, but not in the
    # roster tables, which are the zero-shot comparison. Keeping them out of
    # SCORED_SPECS is not bookkeeping: that tuple is the default --models of
    # fp_taxonomy and null_recall, and a yolo spec carries a local .pt path, so
    # promoting them would make both fail on any clone without the checkpoints.
    # (silent_witness defaults to WITNESS_POOL_46, not to SCORED_SPECS, so the
    # frozen human pass is unaffected either way -- that is the point of the
    # freeze.)
    Challenger(
        spec="yolo:yolo_ckpts/y11l_pano.pt", label="y11l_pano", provider="yolo",
        density="sparse", standing=False, added="2026-08-14",
        note="YOLO11-L, pano geometry, 60 epochs. Best pano arm on seven of ten "
             "splits. 2.11 boxes/pano at the headline conf 0.25, 4.79 at the 0.05 "
             "floor the detections are published down to. The spec's path is the "
             "convention its committed driver uses "
             "(yolo_baseline/benchmark_eval/run_yolo_pano_eval.sh): fetch the "
             "sha256-verified snapshot into yolo_ckpts/. Identity is the file stem "
             "plus a weights content hash, not the path, so the cache key survives "
             "being run from a different directory."),
    Challenger(
        spec="yolo:yolo_ckpts/y11x_pano_h200.pt", label="y11x_pano_h200",
        provider="yolo", density="sparse", standing=False, added="2026-08-14",
        note="YOLO11-X, the Tillicum-trained arm. 2.02 boxes/pano at 0.25, 4.42 at "
             "0.05. Best arm in-distribution (manual_gold 0.851) and the highest AP "
             "of the trio, at the lowest recall."),
    Challenger(
        spec="yolo:yolo_ckpts/y26_pano.pt", label="y26_pano", provider="yolo",
        density="sparse", standing=False, added="2026-08-14",
        note="YOLO26-L. The loosest of the trio -- 2.60 boxes/pano at 0.25 and 9.39 "
             "at 0.05, roughly double the YOLO11 arms -- which is why it leads only "
             "on budapest, where firing at all is the binding constraint."),

    # The four annapolis Claude legs (#122). One split, so they cannot join tables
    # the other rows report over ten; the write-up is annapolis-only for the same
    # reason. Two model ids x two efforts: the first provider whose knob splits one
    # id into several legs, hence `pins` and `published_as`.
    Challenger(
        spec="claude:claude-opus-5", label="claude-opus-5", provider="claude",
        density="sparse", standing=False, added="2026-08-15",
        pins=(("claude_effort", "low"),),
        published_as="claude-opus-5-effort-low",
        note="Top challenger on annapolis (F1 0.588), the first model to displace "
             "gemini-3.1-pro. 2.56 boxes/pano. Effort low is the provider default, "
             "so this is what a bare `claude:claude-opus-5` reproduces."),
    Challenger(
        spec="claude:claude-opus-5", label="claude-opus-5", provider="claude",
        density="sparse", standing=False, added="2026-08-15",
        pins=(("claude_effort", "high"),),
        published_as="claude-opus-5-effort-high",
        note="Same model, more thinking, worse F1 (127k thinking tokens to lose "
             "0.068). 3.73 boxes/pano against low's 2.56: effort moves the "
             "operating point, it does not raise the ceiling."),
    Challenger(
        spec="claude:claude-sonnet-5", label="claude-sonnet-5", provider="claude",
        density="sparse", standing=False, added="2026-08-15",
        pins=(("claude_effort", "low"),),
        published_as="claude-sonnet-5-effort-low",
        note="1.56 boxes/pano, the sparsest leg in the registry. Re-run after the "
             "max_tokens truncation fix, so it covers all 125 panos."),
    Challenger(
        spec="claude:claude-sonnet-5", label="claude-sonnet-5", provider="claude",
        density="sparse", standing=False, added="2026-08-15",
        pins=(("claude_effort", "high"),),
        published_as="claude-sonnet-5-effort-high",
        note="1.98 boxes/pano. Loses F1 to effort in the same direction as Opus, "
             "which is what makes that a pattern rather than one model's quirk."),
)

#: Specs whose label cannot be derived from the spec, because the ``model_id`` slot
#: carries something other than a model id.
#:
#: The Vistas arms (#126) vary by which Vistas classes are read out, not by which
#: checkpoint reads them, so their spec is ``vistas:<class-set>`` and the checkpoint
#: comes from ``--vistas-model``. Without an override, ``label_for`` would resolve
#: them to ``curb-cut``, which is not a model name and would collide across
#: checkpoints in ``benchmark/model_detections/``.
LABEL_OVERRIDES = {
    "vistas:curb-cut": "mask2former-vistas-curb-cut",
    "vistas:curb-cut+curb": "mask2former-vistas-curb-cut+curb",
}

# --------------------------------------------------------------------------- #
# The frozen pool — read the comment before touching it
# --------------------------------------------------------------------------- #
#: The witness pool the #46 human tagging pass was made under, as it stood on
#: 2026-07-31. This is ``silent_witness.py``'s default.
#:
#: **Frozen deliberately, and written out literally rather than derived from
#: ``ROSTER``.** ``silent_witness`` computes the RampNet misses that no other model
#: witnessed; a further witness can only shrink that set. The set is the item list for
#: the #46 tagging pass, which is finished, with committed per-rater verdicts at
#: ``benchmark/miss_taxonomy_46/silent__jonf.json`` (50 items, manifest digest
#: ``360b5ddf8751dcd0``). Verdicts are meaningless against a list other than the one
#: they were made on, and the breakage would be silent — the numbers would simply
#: change. So a new challenger moves the comparison tables and leaves the human pass
#: alone, by construction rather than by anyone remembering.
#:
#: To run a different pool, pass ``--models``. Do not edit this to add a model.
WITNESS_POOL_46 = (
    "gemini:gemini-3.6-flash",
    "gemini:gemini-3.1-pro-preview",
    "qwen:Qwen/Qwen3-VL-8B-Instruct",
    "qwen:Qwen/Qwen3-VL-32B-Instruct",
    "molmo:allenai/Molmo2-8B",
    "owlv2",
    "gdino",
)

# --------------------------------------------------------------------------- #
# Per-provider defaults — one definition, consumed by every parser
# --------------------------------------------------------------------------- #
#: Defaults for every ``compare.py`` argument that feeds ``build_detector`` and so
#: the detection signature and cache key. ``compare.py``'s parser, ``fp_taxonomy``'s
#: ``_compare_args`` shim, ``null_recall.py`` and ``dump_detections.py`` all read
#: these, so they cannot drift apart: a wrong default here does not crash, it changes
#: the cache key and every lookup silently misses.
PROVIDER_DEFAULTS = {
    "gemini_model": "gemini-3.6-flash",
    "claude_model": "claude-sonnet-5",
    "claude_effort": "low",
    "claude_tool_choice": "auto",
    # As-run encoding/decoding for the published Claude legs. ``None`` is what keeps
    # them OUT of the detection signature (a setting enters it only when it deviates
    # from as-run), so changing either default here silently rebuilds a different
    # cache key and every lookup misses.
    "claude_image_format": None,
    "claude_temperature": None,
    "qwen_model": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen_coord_space": "auto",
    "owlv2_model": "google/owlv2-large-patch14-ensemble",
    "gdino_model": "IDEA-Research/grounding-dino-base",
    "molmo_model": "allenai/Molmo2-8B",
    "molmo_coord_scale": "auto",
    "yolo_conf": 0.05,
    "yolo_iou": 0.5,
    "yolo_imgsz": 1024,
    # #126. The checkpoint is the 65-class Vistas v1.2 head; the arm varies by class
    # set, which is the --models spec, not a default.
    "vistas_class_set": "curb-cut",
    "vistas_model": "facebook/mask2former-swin-large-mapillary-vistas-semantic",
    "vistas_min_area_px": 16,
    "vistas_dtype": "float16",
}

#: Providers whose calls cost money -- registry knowledge, so it lives here rather
#: than in the one script that currently reads it (compare.py's refusal to spend
#: without recording it). Note this is per PROVIDER, while pricing.py prices per
#: MODEL ID and does not consult it; `test_the_paid_provider_list_covers_every_priced_model`
#: is what keeps the two from disagreeing. Token counts are the one artifact that
#: cannot be back-filled -- a re-run reads the detection cache, makes zero calls, and so can
#: never reproduce them -- which is how the four Claude legs' $28.82 ended up with
#: no committed record.
PAID_PROVIDERS = frozenset({"gemini", "claude"})

# --------------------------------------------------------------------------- #
# Derived views — nothing below is hand-maintained
# --------------------------------------------------------------------------- #
def is_default_leg(c):
    """True when this leg is what a bare ``--models <spec>`` reproduces.

    A leg whose pins all match ``PROVIDER_DEFAULTS`` needs no extra flags; anything
    else does. Only default legs go in ``BY_SPEC``, which is what keeps that mapping
    single-valued now that one spec can name several legs.
    """
    return all(PROVIDER_DEFAULTS.get(k) == v for k, v in c.pins)


def published_name(c):
    """The filename stem a leg's detections are published under."""
    return c.published_as or c.label


def published_filename(c, city):
    """``benchmark/model_detections/`` basename for one (leg, split)."""
    return slug(published_name(c)) + "__" + city + ".json"


#: The leg a bare spec resolves to. Legs with non-default pins are reachable through
#: ``ROSTER`` and ``BY_PUBLISHED``, not here: a spec alone does not identify them.
BY_SPEC = {c.spec: c for c in ROSTER if is_default_leg(c)}

#: Every leg by the name its detections are published under -- the key that is unique
#: by construction, because it is a filename.
BY_PUBLISHED = {published_name(c): c for c in ROSTER}

#: Legs with detections in ``benchmark/model_detections/``. ``rampnet`` is the one
#: member of the roster with none: it is read from each bundle's committed
#: ``records.jsonl`` and carries no detector signature.
PUBLISHED = tuple(c for c in ROSTER if c.provider != "rampnet")

# A standing leg must be reproducible from its spec alone, because that is all the
# scored tuples below carry. A pinned leg is not: `--models claude:claude-opus-5`
# gives you the LOW-effort leg whatever the roster row says, so promoting the high
# one would have the tables claim one leg while fp_taxonomy and null_recall scored
# the other's detections, and read its density off the other's measurement. Caught
# here, at import, rather than as a wrong number in a table.
for _c in ROSTER:
    if _c.standing and _c.pins:
        raise ValueError(
            f"{_c.spec!r} is standing but pinned {dict(_c.pins)!r}. A scored entry has "
            f"to be what a bare --models spec reproduces, and the pinned legs of one "
            f"spec are indistinguishable there. Give the leg its own spec first, or "
            f"score it in its own table with standing=False.")
del _c

#: Standing entries, RampNet included — the set every results table scores.
SCORED = tuple(c for c in ROSTER if c.standing)

#: Standing challengers, i.e. everything scored except RampNet itself.
CHALLENGERS = tuple(c.spec for c in SCORED if c.provider != "rampnet")

#: What ``fp_taxonomy.py --models`` defaults to: RampNet plus the challengers.
SCORED_SPECS = tuple(c.spec for c in SCORED)

#: Published and verified, but not scored in the roster tables.
OFF_ROSTER = tuple(c for c in ROSTER if not c.standing)

# Read off the entries, not by looking each spec back up in BY_SPEC: that mapping
# holds default legs only, so a spec-keyed lookup here would resolve to the wrong
# leg (or raise at import) the moment anything pinned reached this far. The check
# above makes that unreachable; deriving straight from the entry makes it moot.
#: Sparse enough that a hit is evidence rather than coverage.
SPARSE = tuple(c.spec for c in SCORED
               if c.provider != "rampnet" and c.density == "sparse")

#: So dense that a hit is mostly coverage; reported, never used for a headline.
DENSE = tuple(c.spec for c in SCORED
              if c.provider != "rampnet" and c.density == "dense")


def weights_stem(path):
    """A yolo checkpoint's identity: the file stem, not the path.

    ``detectors.YoloDetector`` keys its cache on this so the same checkpoint hits
    the same entries from a different working directory. Both spellings of a yolo
    run reach it -- ``--models yolo:<path>`` and ``--models yolo --yolo-model
    <path>`` -- because a label that agreed with the detector on one and not the
    other is the cache-key drift this registry exists to remove.
    """
    return os.path.splitext(os.path.basename(str(path).replace(chr(92), "/")))[0]


def label_for(spec, cargs=None):
    """The table row / filename label a ``--models`` spec resolves to.

    ``cargs`` is an optional namespace of provider defaults (``compare.py``'s parsed
    args, or ``fp_taxonomy._compare_args``). It is consulted so that a run with
    ``--gemini-model`` overridden labels itself with the model actually used rather
    than with the registry's default.
    """
    if spec in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[spec]
    provider, _, model_id = spec.partition(":")
    provider, model_id = provider.strip(), model_id.strip()
    if model_id:
        return weights_stem(model_id) if provider == "yolo" else model_id
    key = f"{provider}_model"
    value = getattr(cargs, key, None) if cargs is not None else None
    if value:
        # ...including when the path arrives as --yolo-model rather than in the spec.
        return weights_stem(value) if provider == "yolo" else value
    return PROVIDER_DEFAULTS.get(key, provider)


def leg_for(spec, cargs=None):
    """The registered leg a run resolves to, or ``None`` if it is not registered.

    A spec alone is not enough once pins exist, so the pinned settings are read off
    ``cargs`` and matched: ``claude:claude-opus-5`` with ``claude_effort='high'``
    is a different leg from the same spec at ``low``. This is what lets the exporter
    name a file without being told (see ``published_name``) instead of relying on
    whoever ran it to remember ``--publish-as``.
    """
    label = label_for(spec, cargs)
    candidates = [c for c in ROSTER if c.spec == spec or c.label == label]
    for c in candidates:                      # a pinned leg wins when its pins match
        if c.pins and all(getattr(cargs, k, None) == v for k, v in c.pins):
            return c
    for c in candidates:                      # otherwise the leg the bare spec names
        if not c.pins:
            return c
    return None


def density_of(spec):
    """``"sparse"`` or ``"dense"`` for a registered spec.

    Raises for anything unregistered or unmeasured. That is the point: density is
    measured boxes-per-panorama, not a setting, and the previous code silently
    treated an unknown model as dense — which would quietly move a headline, since
    only the sparse union feeds one.
    """
    entry = BY_SPEC.get(spec)
    if entry is None:
        raise KeyError(
            f"{spec!r} is not in the roster (rampnet/roster.py). Add an entry, or "
            f"pass a pool that excludes it.")
    if entry.density is None:
        raise ValueError(
            f"{spec!r} has no measured density yet, so it cannot join a witness pool: "
            f"a witness count is only meaningful against that model's own box rate. "
            f"Run it, read boxes/pano off null_recall.py, then set `density`.")
    return entry.density


def partition_by_density(specs):
    """``(sparse, dense)`` for an arbitrary pool, in the pool's own order."""
    sparse = tuple(s for s in specs if density_of(s) == "sparse")
    dense = tuple(s for s in specs if density_of(s) == "dense")
    return sparse, dense


#: Marks the generated roster table in a doc, so a test can find it and check it
#: still matches the registry. Prose that restates a roster count drifts silently —
#: docs/model_comparison.md said "all 8" and docs/replication.md "seven-model roster"
#: at the same time, and both were written by hand.
TABLE_MARKER = "<!-- roster-table: generated by `python -m rampnet.roster` -->"


def markdown_table():
    """The roster as a markdown table, for pasting under ``TABLE_MARKER`` in a doc.

    The first column is the leg's published name, not its ``label``: two legs of one
    model share a label, and a table with the same row twice is worse than no table.
    The last column is deliberately narrow -- "is this row in the tables below" --
    because off-roster covers two different situations (a write-up that has not landed
    yet, and a leg whose write-up is elsewhere in this doc), and which one applies is
    in the entry's note, not in a check mark.
    """
    rows = ["| leg | provider | density | joined | in the roster tables |",
            "| :--- | :--- | :--- | :--- | :--- |"]
    for c in ROSTER:
        scored = "✅" if c.standing else "— published, not in these tables"
        density = c.density or "not yet measured"
        rows.append(f"| `{published_name(c)}` | {c.provider} | {density} | "
                    f"{c.added} | {scored} |")
    return "\n".join(rows)


def pool_record(specs, cargs=None):
    """A JSON-able description of a model pool, to embed in an analysis artifact.

    Every published detections file already carries the detector ``signature`` that
    produced it; an analysis whose *item list* depends on which models ran needs the
    same treatment, or a verdict file can no longer be matched to the pool that
    generated its items.
    """
    specs = tuple(specs)
    sparse, dense = partition_by_density(specs)
    named = "WITNESS_POOL_46" if specs == WITNESS_POOL_46 else None
    return {"pool": named, "specs": list(specs),
            "labels": [label_for(s, cargs) for s in specs],
            "sparse": list(sparse), "dense": list(dense)}


if __name__ == "__main__":
    import sys
    # The table carries check marks, and a Windows console defaults to cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(TABLE_MARKER)
    print(markdown_table())
