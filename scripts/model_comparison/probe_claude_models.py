#!/usr/bin/env python
"""Ask a serving path whether a Claude model id is reachable on our account.

Answers one question before any leg is planned: *is this model id real, and is
it enabled for us?* Both serving paths distinguish those two cases, and the
distinction is the whole point of running this.

Two paths, because #156 forced the question. Every paid Claude leg published so
far ran on **Vertex** (`--serving-path vertex`, the default), which is why that
is what this script was originally written against. Vertex gates the Fable
family behind a project setting we do not control, so the alternative is
Anthropic's **first-party API** (`--serving-path anthropic`), a different
credential and a different rate card for the same weights.

Status semantics, which differ by path:

  200  reachable now -- a leg can run today. Prints stop_reason and output
       tokens too; see PROBE_MAX_TOKENS for why those are worth reading.

  vertex:
    403  the id RESOLVED but the project is not entitled. The message says why;
         for the Fable family it is the publisher data-sharing setting, which is
         a project-level configuration change, not something a script should
         make.
    404  Vertex did not resolve the id at all -- almost always a spelling that
         does not exist (`claude-fable-5.1` vs `claude-fable-5-1`), NOT a
         permission problem.

  anthropic:
    404  the id does not exist for this account -- the same "wrong spelling"
         signal Vertex sends as a 404.
    401  the key is bad or revoked. Distinct from every model-level answer, so a
         wall of 401s is a credential problem and nothing else.
    400  the request was understood and refused. The two that matter here are an
         empty credit balance and the Fable family's data-retention requirement;
         both name themselves in the message, which is why it is printed
         verbatim.

That last row of the vertex block is what makes a 403 informative. A 403 on
every `anthropic` id would be consistent with a blanket publisher block; a 404
on a misspelling alongside 403s on real ids proves the gate is per-model and
that the ids behind it exist. Probing a deliberately-wrong spelling is therefore
part of the measurement, not a typo -- see PROBE_CONTROL. The same control earns
its place on the first-party path, where a 404 otherwise reads as "not enabled
for us" rather than "not a model".

Cost: each 200 is a real generate call, capped at --max-tokens, so a full run
costs a fraction of a cent. Every non-200 costs nothing. Nothing here writes to
the detection cache or the usage log -- this is a reachability probe, not a leg,
and its spend is too small to be worth a ledger row.

Usage:

    # Vertex (ADC, the path the four published Claude legs ran on)
    python scripts/model_comparison/probe_claude_models.py

    # Anthropic first-party (ANTHROPIC_API_KEY, from the environment or .env)
    python scripts/model_comparison/probe_claude_models.py --serving-path anthropic

    python scripts/model_comparison/probe_claude_models.py --models claude-opus-5
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The ids #156 is about, plus claude-opus-5 as a POSITIVE CONTROL: without a
# known-good id in the same run, a wall of 403s (or 401s) is indistinguishable
# from broken credentials.
PROBE_MODELS = ("claude-fable-5", "claude-fable-5-1", "claude-opus-5")

# A spelling Anthropic does not publish. Expected to 404 -- see the module
# docstring for why a deliberate miss is load-bearing here.
PROBE_CONTROL = "claude-fable-5.1"

# Deliberately tiny: this asks whether the door opens, not how the model
# performs. It is low enough that the Fable family -- which cannot disable
# thinking -- may stop at `max_tokens` before emitting any text. That is still a
# 200 and still proves reachability, which is why the stop_reason is printed
# rather than treated as a failure. Bracketing Fable's actual thinking floor is
# a separate measurement; --max-tokens is here so it needs no edit to this file.
PROBE_MAX_TOKENS = 16
PROBE_PROMPT = "Reply with the single word: ok"

SERVING_PATHS = ("vertex", "anthropic")


def _load_dotenv():
    """Reuse compare.py's .env loader so the probe reads the same credentials
    file the legs do -- GOOGLE_CLOUD_PROJECT for the vertex path,
    ANTHROPIC_API_KEY for the first-party one. Imported inside the function (the
    vertex_usage.py idiom) so this module still imports without the detector
    stack on the path."""
    try:
        from compare import load_dotenv
    except ImportError:
        return
    load_dotenv(str(REPO))


def make_client(args, ap):
    """The SDK client for one serving path, or ``ap.error`` with what is missing.

    ``max_retries=0`` on both: a 403/404/401 is the answer here, not a blip, and
    retrying it would only slow the probe down and blur the reading."""
    if args.serving_path == "vertex":
        if not args.project:
            ap.error("no project: pass --project or set GOOGLE_CLOUD_PROJECT "
                     "(and authenticate with `gcloud auth application-default "
                     "login`)")
        from anthropic import AnthropicVertex

        return AnthropicVertex(project_id=args.project, region=args.location,
                               max_retries=0)

    from anthropic import Anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        ap.error("no ANTHROPIC_API_KEY: put it in the repo-root .env (which is "
                 "gitignored) or the environment. Keys are minted at "
                 "console.anthropic.com; a key with no credit balance "
                 "authenticates and then fails each call with a 400 that says "
                 "so.")
    return Anthropic(max_retries=0)


def probe(client, model_id, max_tokens=PROBE_MAX_TOKENS):
    """``(status, detail)`` for one model id. Never raises."""
    import anthropic

    try:
        resp = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": PROBE_PROMPT}],
        )
    except anthropic.APIStatusError as e:
        # e.message is the provider's own text; it carries the actionable part
        # (which setting, which API) and is the reason this prints it verbatim
        # rather than a summary of it.
        return e.status_code, (e.message or "").strip()
    except anthropic.APIConnectionError as e:
        return "conn", f"connection error: {e}"

    # A 200 is the answer, but not the whole of it. `stop_reason` distinguishes a
    # model that answered from one that spent the whole budget thinking, and the
    # output count is the first (very coarse) datapoint on where the Fable
    # family's thinking floor sits -- the unknown that --claude-max-tokens
    # exists for on the detector.
    detail = f"served by {getattr(resp, 'model', model_id)}"
    usage = getattr(resp, "usage", None)
    if usage is not None:
        detail += (f"; {getattr(usage, 'input_tokens', 0) or 0} in / "
                   f"{getattr(usage, 'output_tokens', 0) or 0} out tokens")
    reason = getattr(resp, "stop_reason", None)
    if reason:
        detail += f"; stop_reason={reason}"
        if reason == "max_tokens":
            detail += (f" (spent the whole {max_tokens}-token budget before "
                       f"answering -- still reachable; raise --max-tokens to "
                       f"see it reply)")
    return 200, detail


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serving-path", choices=SERVING_PATHS, default="vertex",
                    help="Which account answers. `vertex` (default) is what the "
                         "four published Claude legs ran on; `anthropic` is the "
                         "first-party API, a different credential and a "
                         "different rate card for the same weights "
                         "(default: %(default)s)")
    ap.add_argument("--models", nargs="+", default=list(PROBE_MODELS),
                    help="model ids to probe (default: %(default)s)")
    ap.add_argument("--no-control", action="store_true",
                    help=f"skip the {PROBE_CONTROL} 404 control")
    ap.add_argument("--max-tokens", type=int, default=PROBE_MAX_TOKENS,
                    help="ceiling on each probe answer. Thinking bills against "
                         "it, so on the Fable family the default may be spent "
                         "before any text is emitted (default: %(default)s)")
    ap.add_argument("--project", default=None,
                    help="vertex only; defaults to GOOGLE_CLOUD_PROJECT")
    ap.add_argument("--location", default=None,
                    help="vertex only; the endpoint. `global` is what the legs "
                         "run on and is priced 10%% below regional "
                         "(default: global)")
    args = ap.parse_args(argv)

    # After parsing, so an explicit flag still wins over the file.
    _load_dotenv()
    if args.project is None:
        args.project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if args.location is None:
        args.location = os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"

    client = make_client(args, ap)

    ids = list(args.models)
    if not args.no_control and PROBE_CONTROL not in ids:
        ids.append(PROBE_CONTROL)

    if args.serving_path == "vertex":
        print(f"serving_path=vertex project={args.project} "
              f"location={args.location}\n")
    else:
        print("serving_path=anthropic (first-party API, ANTHROPIC_API_KEY)\n")

    worst = 0
    for model_id in ids:
        status, detail = probe(client, model_id, max_tokens=args.max_tokens)
        print(f"{model_id:<24} {status}")
        for line in detail.splitlines():
            print(f"{'':<24}   {line}")
        print()
        if model_id in args.models and status != 200:
            worst = 1
    return worst


if __name__ == "__main__":
    sys.exit(main())
