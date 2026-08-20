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
    # Claude via Vertex (#122). Source: the Google Cloud rate card for Anthropic
    # partner models, `global` endpoint, read 2026-08-15 -- i.e. the page that
    # actually governs this billing path, unlike the Gemini rows above. Both
    # models match Anthropic's own published first-party rates exactly, and the
    # derived rates are the standard multipliers (cache hit 0.1x input, 5m cache
    # write 1.25x, 1h 2x, batch 50%) -- `cache_read_per_m` / `cache_write_per_m`
    # below are the 5-minute-TTL pair, recorded because Anthropic bills them as
    # separate SKUs that are EXCLUDED from input_tokens, so a run that ever sets
    # cache_control would otherwise be costed with the cached half missing. They
    # are zero on every run so far: the tool definition renders below Sonnet 5's
    # 1,024-token minimum cacheable prefix, so nothing is cacheable yet.
    # REGIONAL endpoints cost 10% more than
    # `global` -- these numbers are wrong if the rig is ever pointed off `global`.
    # No long-context premium: the <=200K and >200K SKUs are priced identically.
    "claude-sonnet-5": {
        "input_per_m": 2.00, "output_per_m": 10.00,
        "cache_read_per_m": 0.20, "cache_write_per_m": 2.50, "as_of": "2026-08-15",
        "note": ("Vertex `global`; promotional launch pricing through 2026-08-31, "
                 "$3.00/$15.00 after. Regional +10%."),
    },
    "claude-opus-5": {
        "input_per_m": 5.00, "output_per_m": 25.00,
        "cache_read_per_m": 0.50, "cache_write_per_m": 6.25, "as_of": "2026-08-15",
        "note": "Vertex `global`; regional +10%.",
    },
}


def price_for(model_id):
    """The pricing entry for a model id, or None when we have no verified price
    (unknown/free/local models cost $0 in API terms but return None here so a
    missing entry is visible rather than silently priced at zero)."""
    return PRICING.get(model_id)


def estimate_cost(model_id, input_tokens, output_tokens,
                  cache_read_tokens=0, cache_write_tokens=0):
    """Estimated USD for a token count, or None if the model has no verified price.

    Output counts must already include thinking tokens (the SDK reports them
    separately; billing does not).

    ``cache_*`` are separate SKUs rather than a subset of ``input_tokens``, and
    they are priced ONLY for models whose cache rates were read off the rate card
    -- never derived here. A model without them prices its plain tokens and stays
    silent about the rest, which is the same discipline as a model with no entry
    at all returning None instead of $0."""
    p = price_for(model_id)
    if p is None:
        return None
    usd = input_tokens * p["input_per_m"] + output_tokens * p["output_per_m"]
    if cache_read_tokens and "cache_read_per_m" in p:
        usd += cache_read_tokens * p["cache_read_per_m"]
    if cache_write_tokens and "cache_write_per_m" in p:
        usd += cache_write_tokens * p["cache_write_per_m"]
    return usd / 1e6


# --- compute, not tokens ----------------------------------------------------
#
# The other half of what an experiment costs (#143). Same discipline as the token
# table above: verified-only, each entry carrying the date it was checked and where
# it came from. GPU-hours are the durable measurement; the dollar figure is an
# estimate and the billing system is authoritative.
#
# `usd_per_gpu_hour` is applied to GPU-hours = elapsed wall-clock x N GPUs, which
# is how Tillicum defines the unit -- an idle GPU in a 2-GPU job bills exactly like
# a busy one. `qos_usage_factor` is Slurm's own UsageFactor per QoS.
COMPUTE_PRICING = {
    "tillicum": {
        "usd_per_gpu_hour": 0.90,
        "as_of": "2026-07-30",
        "source": ("hyak.uw.edu/docs/systems/tillicum + the 2026-07-29 provisioning "
                   "email; QoS factors from `sacctmgr show qos`, read 2026-07-31"),
        "qos_usage_factor": {
            "debug": 0.0, "normal": 1.0, "interactive": 1.0,
            "urgent": 1.0, "long": 1.0, "wide": 1.0,
        },
        # This caveat has to travel with any "free" figure quoted from `debug`:
        # Slurm bills that QoS at UsageFactor 0, but `hyakusage` charged the 2-minute
        # smoke job (198638) 0.03 GPU-hours / $0.03 anyway -- raw wall-clock x rate,
        # the factor apparently not applied -- even though its own header says
        # "billable GPU hours = raw GPU hours x QOS multiplier". The two tools state
        # different things and we do not know which one ITBill follows. Exposure is
        # capped at $0.90/job (debug is 1 h x 1 GPU), so this is unresolved, not
        # urgent. See docs/tillicum.md.
        "note": ("`debug` is UsageFactor 0 in Slurm but `hyakusage` bills it anyway; "
                 "this table follows Slurm. Never quote a debug-QoS cost without "
                 "saying which tool it came from."),
    },
    "klone": {
        # Lab-owned and scavenger partitions alike: no per-hour charge. GPU-hours
        # are still recorded, because the scientific cost of a run is its compute
        # whether or not an invoice follows, and because the #51 baseline's 496.5
        # GPU-hours is a real number that belongs in the paper.
        "usd_per_gpu_hour": 0.0,
        "as_of": "2026-07-30",
        "source": "hyak.uw.edu — condo model, no usage billing",
        "qos_usage_factor": {},
        "note": "Free at the point of use; ckpt is preemptable, which costs restarts.",
    },
}


def compute_price_for(cluster):
    """The compute-rate entry for a cluster, or None when we have no verified rate.

    None rather than $0: an unpriced cluster must be visible as unpriced, not
    silently free. klone is $0 because that was checked, which is a different
    statement."""
    return COMPUTE_PRICING.get((cluster or "").lower())


def estimate_compute_cost(cluster, gpu_hours, qos=None):
    """Estimated USD for a job's GPU-hours, or None without a verified rate.

    ``qos`` applies Slurm's UsageFactor when the cluster publishes one; an
    unknown QoS is charged at full rate rather than assumed free, because
    guessing downward is the expensive direction to be wrong in."""
    p = compute_price_for(cluster)
    if p is None:
        return None
    factor = p.get("qos_usage_factor", {}).get((qos or "").lower(), 1.0)
    return gpu_hours * p["usd_per_gpu_hour"] * factor
