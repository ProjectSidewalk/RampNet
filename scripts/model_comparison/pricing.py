"""Per-token prices for the paid (API) models the harness can call.

Every paid-model experiment must be costable after the fact (a paper reports
what an experiment cost, not just what it scored), so the harness records token
usage per run (see compare.py --usage-log) and prices it from this table. Keep
the table verified-only: a price goes in with the date it was checked and the
page it came from, never from memory. Prices change — the recorded token counts
are the durable fact, the dollar figure is an estimate as of `as_of`, and the
cloud billing console remains the authority for actual spend.

Reconcile estimates against actuals with scripts/analysis/vertex_usage.py,
which pulls the project's real token counts from Cloud Monitoring.
"""

# USD per 1M tokens. Output prices include thinking tokens (billed as output).
#
# Source: https://ai.google.dev/gemini-api/docs/pricing (standard paid tier);
# gemini-3.1-pro-preview is the <=200k-token-prompt tier (every harness call is
# a single ~1.3k-token view, nowhere near the 200k boundary).
#
# CAVEAT ON THE SOURCE, and it matters: the measured spend these prices are applied
# to was billed through **Vertex AI**, not the Gemini Developer API. GeminiDetector
# takes the `vertexai=True` + ADC path whenever GOOGLE_GENAI_USE_VERTEXAI is set
# (which is how every leg on this project ran), and vertex_usage.py reconciles
# against Cloud Monitoring for a GCP project. The two rate cards agreed for these
# models at `as_of`, but they are separate pages that can diverge, and the
# introductory-expiry note below is an ai.google.dev fact that need not track
# Vertex. Before quoting these dollars in a paper, re-check against
# https://cloud.google.com/vertex-ai/generative-ai/pricing and stamp a new
# `as_of` — the token counts are the durable measurement, not the dollars.
PRICING = {
    "gemini-3.7-flash": {
        "input_per_m": 0.75, "output_per_m": 3.75, "as_of": "2026-08-15",
        "note": "introductory through 2026-12-31; $1.50/$7.50 from 2027-01-01",
    },
    "gemini-3.6-flash": {
        "input_per_m": 0.75, "output_per_m": 3.75, "as_of": "2026-08-15",
        "note": "introductory through 2026-12-31; $1.50/$7.50 from 2027-01-01",
    },
    "gemini-3.1-pro-preview": {
        "input_per_m": 2.00, "output_per_m": 12.00, "as_of": "2026-08-15",
        "note": "prompts <=200k tokens tier",
    },
    "gemini-2.5-flash": {
        "input_per_m": 0.30, "output_per_m": 2.50, "as_of": "2026-08-15",
        "note": None,
    },
}


def price_for(model_id):
    """The pricing entry for a model id, or None when we have no verified price
    (unknown/free/local models cost $0 in API terms but return None here so a
    missing entry is visible rather than silently priced at zero)."""
    return PRICING.get(model_id)


def estimate_cost(model_id, input_tokens, output_tokens):
    """Estimated USD for a (input, output) token count, or None if the model
    has no verified price. Output counts must already include thinking tokens
    (the SDK reports them separately; billing does not)."""
    p = price_for(model_id)
    if p is None:
        return None
    return (input_tokens * p["input_per_m"] + output_tokens * p["output_per_m"]) / 1e6
