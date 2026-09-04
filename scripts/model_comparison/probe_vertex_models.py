#!/usr/bin/env python
"""Ask Vertex whether a Claude model id is reachable on this project.

Answers one question before any leg is planned: *is this model id real, and is
it enabled for us?* Vertex distinguishes the two, and the distinction is the
whole point of running this:

  200  reachable now -- a leg can run today.
  403  the id RESOLVED but the project is not entitled. The message says why;
       for the Fable family it is the publisher data-sharing setting, which is
       a project-level configuration change, not something a script should make.
  404  Vertex did not resolve the id at all -- almost always a spelling that
       does not exist (`claude-fable-5.1` vs `claude-fable-5-1`), NOT a
       permission problem.

That last row is what makes a 403 informative. A 403 on every `anthropic` id
would be consistent with a blanket publisher block; a 404 on a misspelling
alongside 403s on real ids proves the gate is per-model and that the ids behind
it exist. Probing a deliberately-wrong spelling is therefore part of the
measurement, not a typo -- see PROBE_CONTROL.

Cost: each 200 is a real generate call, capped at PROBE_MAX_TOKENS, so a full
run costs a fraction of a cent. 403s and 404s cost nothing. Nothing here writes
to the detection cache or the usage log -- this is a reachability probe, not a
leg.

Usage (needs the same ADC the Gemini and Claude legs use):

    python scripts/model_comparison/probe_vertex_models.py
    python scripts/model_comparison/probe_vertex_models.py --models claude-opus-5
"""
import argparse
import os
import sys

# The ids #156 is about, plus claude-opus-5 as a POSITIVE CONTROL: without a
# known-good id in the same run, a wall of 403s is indistinguishable from broken
# credentials.
PROBE_MODELS = ("claude-fable-5", "claude-fable-5-1", "claude-opus-5")

# A spelling Anthropic does not publish. Expected to 404 -- see the module
# docstring for why a deliberate miss is load-bearing here.
PROBE_CONTROL = "claude-fable-5.1"

PROBE_MAX_TOKENS = 16
PROBE_PROMPT = "Reply with the single word: ok"


def probe(client, model_id):
    """``(status, detail)`` for one model id. Never raises."""
    import anthropic

    try:
        resp = client.messages.create(
            model=model_id,
            max_tokens=PROBE_MAX_TOKENS,
            messages=[{"role": "user", "content": PROBE_PROMPT}],
        )
    except anthropic.APIStatusError as e:
        # e.message is the provider's own text; it carries the actionable part
        # (which setting, which API) and is the reason this prints it verbatim
        # rather than a summary of it.
        return e.status_code, (e.message or "").strip()
    except anthropic.APIConnectionError as e:
        return "conn", f"connection error: {e}"
    return 200, f"served by {getattr(resp, 'model', model_id)}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(PROBE_MODELS),
                    help="model ids to probe (default: %(default)s)")
    ap.add_argument("--no-control", action="store_true",
                    help=f"skip the {PROBE_CONTROL} 404 control")
    ap.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    ap.add_argument("--location",
                    default=os.environ.get("GOOGLE_CLOUD_LOCATION") or "global",
                    help="Vertex endpoint; `global` is what the legs run on and "
                         "is priced 10%% below regional (default: %(default)s)")
    args = ap.parse_args(argv)

    if not args.project:
        ap.error("no project: pass --project or set GOOGLE_CLOUD_PROJECT "
                 "(and authenticate with `gcloud auth application-default login`)")

    from anthropic import AnthropicVertex

    client = AnthropicVertex(project_id=args.project, region=args.location,
                             max_retries=0)   # a 403 is the answer, not a blip

    ids = list(args.models)
    if not args.no_control and PROBE_CONTROL not in ids:
        ids.append(PROBE_CONTROL)

    print(f"project={args.project} location={args.location}\n")
    worst = 0
    for model_id in ids:
        status, detail = probe(client, model_id)
        print(f"{model_id:<24} {status}")
        for line in detail.splitlines():
            print(f"{'':<24}   {line}")
        print()
        if model_id in args.models and status != 200:
            worst = 1
    return worst


if __name__ == "__main__":
    sys.exit(main())
