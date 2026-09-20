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

  **A pin is what the leg needs to REPRODUCE, which is a superset of what enters the
  signature.** ``claude_serving_path`` (#156) is the first pin that is not a
  signature key: a Fable leg must run against ``anthropic`` because Vertex gates that
  family, but the path does not change the detections and deliberately stays out of the
  cache key (see ``ClaudeDetector.signature``). Pinning it anyway is what keeps a
  bare ``claude:claude-fable-5-1`` from resolving to a Vertex run that 403s. Note the
  consequence for naming: ``published_as`` must spell out every pin's value, so such a
  leg is ``claude-fable-5-1-effort-low-anthropic``.
* ``published_as`` is the filename stem under ``benchmark/model_detections/``, and it
  defaults to ``label``. It exists because ``label`` cannot carry a pin: the label is
  baked into cache keys that were already paid for, so renaming it orphans the
  detections. The two came apart the first time a provider had a pin, and both legs
  wanted ``claude-sonnet-5__annapolis.json``, the second silently overwriting the
  first. **Once any leg of a model needs disambiguating, give every leg of that model
  the same treatment** — a directory where one file is bare and its sibling is
  qualified reads as though the bare one is the whole model. The one exception is a
  sibling pinned only on an **opt-in** knob, one whose default is ``None`` and which
  is absent from the signature unless set (``vistas_input_size``, #163): there the
  bare file genuinely is the model at its defaults and says so in its own signature,
  so it keeps its name. ``needs_qualified_name`` is the rule; ``pin_token`` spells a
  non-scalar pin (``1024x1024``) inside the qualified name.
* A **replicate** is not a leg. It is the same leg run again — same signature, same
  cache key — on another host or in another environment, kept so the published
  numbers can be shown to reproduce (or not). ``REPLICATES`` registers them, and
  they publish under ``benchmark/model_detections/replicates/<tag>/``.
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
    # The #126 resolution-parity arm (#137), re-run and published under #163. Same
    # checkpoint and class set as the leg above; the one difference is that the
    # processor's 384x384 resize is overridden to the view's own 1024x1024, which
    # VistasDetector records in the signature only when set -- so this is a distinct
    # cache key, a distinct leg, and the bare leg above keeps its name (see
    # ``needs_qualified_name``). min_area_px=16 is shared and does NOT mean the same
    # thing here: inert at 384, exactly the smallest achievable blob at 1024.
    Challenger(
        spec="vistas:curb-cut", label="mask2former-vistas-curb-cut", provider="vistas",
        density="sparse", standing=False, added="2026-09-20",
        pins=(("vistas_input_size", (1024, 1024)),),
        published_as="mask2former-vistas-curb-cut-1024x1024",
        note="Resolution parity for the arm above (#126, #137): the checkpoint's own "
             "384x384 preprocessor is overridden so the model sees the full "
             "1024x1024 view every other tiled leg sees. First run 2026-08-18 on "
             "makelab2 into a private cache that was later lost (#163); re-run and "
             "published 2026-09-20 from the same host and env. The handicap was "
             "real and it was recall: 0.694 -> 0.884, AP 0.510 -> 0.649, precision "
             "slightly worse, F1 +0.018, so 'transfers but does not compete' "
             "stands. richmond only. Density read off the published detections: "
             "see docs/model_comparison.md, Resolution parity."),
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

    # The Claude legs (#122). Two model ids x two efforts, all four on annapolis.
    # The opus/low leg now covers eleven splits -- the nine of #139 plus both
    # Laurens arms (#151, 2026-09-04) -- so it is a complete leg on the scoreboard;
    # the other three are annapolis only and stay off every pooled table, with the
    # write-up scoping each number to the split it was measured on. None is
    # `standing`: a pinned leg cannot be, because both efforts of one model id share
    # a spec (see the check below). The first provider whose knob splits one id
    # into several legs, hence `pins` and `published_as`.
    Challenger(
        spec="claude:claude-opus-5", label="claude-opus-5", provider="claude",
        density="sparse", standing=False, added="2026-08-15",
        pins=(("claude_effort", "low"),),
        published_as="claude-opus-5-effort-low",
        note="Eleven splits: nine in #139, both Laurens arms in #151. Pooled F1 0.568 "
             "over the eight city splits, within 0.01 of gemini-3.1-pro's 0.575 -- its "
             "+0.021 annapolis lead, the only time anything displaced the top "
             "challenger, did not survive pooling, but neither did a deficit: the two "
             "split 4 wins each on the pooled eight (Opus 6 of 11 overall), and the "
             "pooled per-split gaps range from -0.069 (gainesville) to +0.086 "
             "(laurens_mapillary; the held-out laurens_gsv is wider, +0.158), so the "
             "annapolis lead was split noise, not a difference. Highest recall of any "
             "fully-pooled chat VLM "
             "(0.586), trading -0.077 precision for +0.052 recall. Best zero-shot model "
             "on BOTH Laurens arms (0.430 mapillary, 0.437 gsv) and flat across them "
             "(+0.007) where RampNet gains +0.115, which is what makes #151's "
             "rig-not-town reading sharp. 2.54 boxes/pano over eleven splits (2.56 on "
             "annapolis alone). No manual_gold row, deliberately: #144. Effort low is "
             "the provider default, so this is what a bare `claude:claude-opus-5` "
             "reproduces."),
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
    # #156. The first legs served OFF Vertex -- that family is gated there behind a
    # publisher data-sharing setting, so these ran on Anthropic's first-party API.
    # The serving path is NOT in the detection signature (it does not change the
    # answer; see ClaudeDetector.signature), but it IS pinned, because reproducing
    # these legs requires it -- which is why `published_as` spells it out.
    Challenger(
        spec="claude:claude-fable-5-1", label="claude-fable-5-1", provider="claude",
        density="sparse", standing=False, added="2026-09-05",
        pins=(("claude_effort", "low"), ("claude_serving_path", "anthropic")),
        published_as="claude-fable-5-1-effort-low-anthropic",
        note="F1 0.610 on annapolis (P 0.637 / R 0.585), 2.28 boxes/pano. Clears "
             "the 0.567 gate and displaces claude-opus-5 effort-low (0.588) -- the "
             "first general-purpose model to do so on this split. Tied with "
             "claude-fable-5 (0.611) at a MORE PRECISE operating point, which is "
             "the whole difference between them."),
    Challenger(
        spec="claude:claude-fable-5", label="claude-fable-5", provider="claude",
        density="sparse", standing=False, added="2026-09-05",
        pins=(("claude_effort", "low"), ("claude_serving_path", "anthropic")),
        published_as="claude-fable-5-effort-low-anthropic",
        note="F1 0.611 on annapolis (P 0.579 / R 0.646), 2.72 boxes/pano. "
             "Indistinguishable from claude-fable-5-1 on F1 while trading 0.058 "
             "precision for 0.061 recall: within this family the model version is "
             "an operating-point dial, the same shape as effort in #123. Also "
             "spends ~33 thinking tokens/call against 5.1's ~0.35, for no F1."),
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
    "claude_max_tokens": None,
    # Which account serves the calls. `vertex` is what all four published legs ran
    # on, so it stays the default; the Fable legs pin `anthropic` because Vertex
    # gates that family. Unlike the three settings above, this one does NOT enter
    # the detection signature -- see ClaudeDetector.signature -- so changing it
    # does not orphan the cache. It is still a pin, because reproducing a Fable
    # leg requires it.
    "claude_serving_path": "vertex",
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
    # The two #129 overrides. ``None`` keeps each OUT of the detection signature
    # (VistasDetector.signature records them only when set), which is what keeps
    # the published 384 richmond arm's cache key intact. A leg that sets one is a
    # different leg -- the 1024x1024 parity arm (#163) pins ``vistas_input_size``.
    "vistas_input_size": None,
    "vistas_revision": None,
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
def pin_value(value):
    """A pin value in comparable form.

    A pin arrives three ways -- as the registry's literal (a tuple for a size), as
    argparse's ``nargs=2`` list, and as the list JSON gave a published file's
    ``signature`` -- and ``[1024, 1024] == (1024, 1024)`` is False. Sequences are
    compared as tuples so all three spellings of one pin agree.
    """
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def pins_match(cargs, pins):
    """True when every pin in ``pins`` is what ``cargs`` carries."""
    return all(pin_value(getattr(cargs, k, None)) == pin_value(v) for k, v in pins)


def pin_token(value):
    """How a pin's value is spelled inside a published name.

    ``published_as`` has to say every pin's value so the file is self-describing
    (``claude-opus-5-effort-low``). A scalar is ``str(value)``; a size is joined with
    ``x`` -- ``(1024, 1024)`` -> ``1024x1024`` -- because ``str((1024, 1024))`` is not
    something anyone would put in a filename.
    """
    if isinstance(value, (list, tuple)):
        return "x".join(str(v) for v in value)
    return str(value)


def is_opt_in_pin(key):
    """A pin whose default is ``None``, i.e. a knob that is ABSENT from the detection
    signature unless it is set.

    That is the convention the Claude image/temperature settings and the two Vistas
    overrides follow ("a setting enters the signature only when it deviates from
    as-run"), and it decides whether a model's bare-named leg has to be renamed when
    a pinned sibling arrives -- see ``needs_qualified_name``.
    """
    return key in PROVIDER_DEFAULTS and PROVIDER_DEFAULTS[key] is None


def needs_qualified_name(c):
    """Must this leg be published under a name other than its bare label?

    Always, if it is pinned. If it is NOT pinned, only when a sibling leg (same
    label) is pinned on a knob that is in every leg's signature -- Claude's effort,
    which is ``low`` in the bare leg's signature and ``high`` in its sibling's, so a
    bare ``claude-sonnet-5__annapolis.json`` would hide which one it is.

    An unpinned leg beside siblings pinned only on opt-in knobs keeps its bare name.
    ``mask2former-vistas-curb-cut__richmond.json`` IS the arm at its defaults: its
    signature carries no ``input_size`` key at all, and the parity sibling's does, so
    the two files describe themselves and a rename would only orphan every reference
    to the published one (#163). The half-qualified directory that rule was written
    against (#122) cannot arise here, because a bare name can only ever mean "every
    opt-in knob unset".
    """
    if c.pins:
        return True
    return any(not is_opt_in_pin(k)
               for sib in ROSTER if sib.label == c.label and sib is not c
               for k, _ in sib.pins)


def is_default_leg(c):
    """True when this leg is what a bare ``--models <spec>`` reproduces.

    A leg whose pins all match ``PROVIDER_DEFAULTS`` needs no extra flags; anything
    else does. Only default legs go in ``BY_SPEC``, which is what keeps that mapping
    single-valued now that one spec can name several legs.
    """
    return all(pin_value(PROVIDER_DEFAULTS.get(k)) == pin_value(v) for k, v in c.pins)


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

# --------------------------------------------------------------------------- #
# Replicates -- the same leg, run again somewhere else
# --------------------------------------------------------------------------- #
#: A re-run of a published leg at the SAME signature, on a different host or in a
#: different environment. It is not a leg: the detections it produced have the same
#: cache key as the published ones (that is the point -- it measures whether the
#: published numbers reproduce), so it cannot be a pinned entry in ``ROSTER``, and
#: giving it one would need a pin that changes no signature field, which is what a
#: pin is not. ``of`` is the published name it replicates; ``tag`` names the run and
#: is the directory it publishes under:
#:
#:     benchmark/model_detections/replicates/<tag>/<slug(of)>__<split>.json
#:
#: written by ``export_model_cache.py --out benchmark/model_detections/replicates/<tag>``
#: with no other change -- inside its directory a replicate file is exactly a
#: published file, and ``load_detections`` reads it with ``published_dir`` pointed
#: there. What a replicate must share with the file it replicates is the header
#: (``model``, ``published_as``, ``signature``); what it may differ in is the
#: detections, and the size of that difference is the result it exists to record.
#: ``tests/test_roster.py`` holds both halves.
Replicate = namedtuple("Replicate", "of tag splits added note")

REPLICATES = (
    Replicate(
        of="mask2former-vistas-curb-cut", tag="makelab2-a40-2026-09-20",
        splits=("richmond",), added="2026-09-20",
        note="The same-environment 384x384 control for the resolution-parity arm "
             "(#126, #137, #163): the published arm ran on an RTX 3070 on transformers "
             "4.x, and the parity arm on makelab2's A40 on transformers 5.15, so "
             "without this the parity delta would confound input size with the "
             "environment. First run 2026-08-18 into a private cache that was later "
             "lost; this is the 2026-09-20 re-run, same host, same env. Read against "
             "the published file in docs/model_comparison.md, Resolution parity."),
)

#: Every replicate by its tag -- the directory name, so unique by construction.
REPLICATES_BY_TAG = {r.tag: r for r in REPLICATES}


def replicate_dir(rep, published_dir=None):
    """The directory one replicate publishes into, relative to ``published_dir``
    when given (the caller supplies the absolute ``benchmark/model_detections``)."""
    parts = ("replicates", rep.tag)
    return os.path.join(published_dir, *parts) if published_dir else os.path.join(*parts)


def replicate_filename(rep, city):
    """Basename of one (replicate, split) file: the replicated leg's own filename."""
    return slug(rep.of) + "__" + city + ".json"

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
    if provider == "vistas":
        # The model_id slot carries the CLASS SET, not a model id: two arms share one
        # checkpoint and differ only by which classes they read out. So the label is
        # derived from the class set the same way VistasDetector derives its own
        # default -- not looked up in a table that has to be kept in step with
        # detectors.VISTAS_CLASS_SETS by hand. Forgetting that entry used to return
        # the bare class set, which is not a model name and slugs into a filename
        # with the "+" mangled.
        class_set = (model_id or getattr(cargs, "vistas_class_set", None)
                     or PROVIDER_DEFAULTS["vistas_class_set"])
        return "mask2former-vistas-" + class_set
    if model_id:
        return weights_stem(model_id) if provider == "yolo" else model_id
    key = f"{provider}_model"
    value = getattr(cargs, key, None) if cargs is not None else None
    if value:
        # ...including when the path arrives as --yolo-model rather than in the spec.
        return weights_stem(value) if provider == "yolo" else value
    return PROVIDER_DEFAULTS.get(key, provider)


def legs_of(spec, cargs=None):
    """Every registered leg a spec could name, in roster order.

    ``leg_for`` picks one of these by matching pins against ``cargs``. Callers that
    need to know *why* nothing matched -- the exporter, which must not fall back to
    a bare filename when every candidate is pinned -- read the whole list.
    """
    label = label_for(spec, cargs)
    return [c for c in ROSTER if c.spec == spec or c.label == label]


def leg_for(spec, cargs=None):
    """The registered leg a run resolves to, or ``None`` if it is not registered.

    A spec alone is not enough once pins exist, so the pinned settings are read off
    ``cargs`` and matched: ``claude:claude-opus-5`` with ``claude_effort='high'``
    is a different leg from the same spec at ``low``. This is what lets the exporter
    name a file without being told (see ``published_name``) instead of relying on
    whoever ran it to remember ``--publish-as``.
    """
    candidates = legs_of(spec, cargs)
    for c in candidates:                      # a pinned leg wins when its pins match
        if c.pins and pins_match(cargs, c.pins):
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
