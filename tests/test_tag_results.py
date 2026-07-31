"""Unit tests for the rater-verdict arithmetic and agreement stats (#46).

Pure logic — no verdict files, no manifest.

The guarantees that matter here are about *not quietly biasing the headline*:
abstentions must leave the denominator rather than count against the sourcing case,
agreement must be chance-corrected because the verdict distribution is heavily skewed
toward one category, and two raters who saw different crops must never be silently
merged.
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import tag_results as tr  # noqa: E402


# --------------------------------------------------------------------------- #
# rate — abstentions must not count as evidence either way
# --------------------------------------------------------------------------- #
def test_excluded_verdicts_leave_the_denominator():
    # Counting `unclear` as "not addressable" would convert "we could not tell" into
    # evidence against the sourcing programme.
    v = ["visible", "visible", "unclear", "unclear"]
    hits, den, r = tr.rate(v, tr.ADDRESSABLE_TIGHT)
    assert (hits, den) == (2, 2) and r == 1.0


def test_definition_is_excluded_too():
    # A rubric question is not a model failure and not a model success.
    v = ["visible", "definition", "occluded"]
    _, den, _ = tr.rate(v, tr.ADDRESSABLE_TIGHT)
    assert den == 2


def test_non_addressable_verdicts_stay_in_the_denominator():
    # Occlusion IS evidence: the ramp was there and sourcing cannot reach it.
    v = ["visible", "occluded", "lighting"]
    hits, den, r = tr.rate(v, tr.ADDRESSABLE_TIGHT)
    assert (hits, den) == (1, 3) and abs(r - 1 / 3) < 1e-9


def test_the_upper_variant_includes_context_only():
    v = ["visible", "context-only", "occluded"]
    assert tr.rate(v, tr.ADDRESSABLE_TIGHT)[0] == 1
    assert tr.rate(v, tr.ADDRESSABLE_UPPER)[0] == 2


def test_all_abstentions_yields_no_rate_rather_than_zero():
    _, den, r = tr.rate(["unclear", "definition"], tr.ADDRESSABLE_TIGHT)
    assert den == 0 and math.isnan(r)


def test_empty_input_does_not_divide_by_zero():
    assert math.isnan(tr.rate([], tr.ADDRESSABLE_TIGHT)[2])


# --------------------------------------------------------------------------- #
# cohens_kappa — percent agreement flatters a skewed scheme
# --------------------------------------------------------------------------- #
def test_perfect_agreement_is_one():
    assert abs(tr.cohens_kappa([("a", "a"), ("b", "b"), ("a", "a")]) - 1.0) < 1e-9


def test_chance_level_agreement_is_about_zero():
    # Two raters each split 50/50 with no relationship: 50% raw agreement, kappa ~0.
    pairs = [("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]
    assert abs(tr.cohens_kappa(pairs)) < 1e-9


def test_a_skewed_scheme_is_not_flattered():
    # 9 of 10 items are 'visible' and the raters differ on the tenth. Raw agreement is
    # 90%, which sounds strong; kappa must not.
    pairs = [("visible", "visible")] * 9 + [("visible", "occluded")]
    assert tr.cohens_kappa(pairs) < 0.5


def test_total_disagreement_is_negative():
    assert tr.cohens_kappa([("a", "b"), ("b", "a")]) < 0


def test_kappa_is_undefined_when_one_category_is_universal():
    # Both raters said the same thing every time: chance agreement is already 1, so
    # there is no headroom and no meaningful statistic.
    assert math.isnan(tr.cohens_kappa([("a", "a")] * 5))


def test_no_shared_items_yields_no_statistic():
    assert math.isnan(tr.cohens_kappa([]))


# --------------------------------------------------------------------------- #
# bracket — the two populations are different evidence and must stay separate
# --------------------------------------------------------------------------- #
def test_bracket_sums_both_populations_over_the_pooled_denominator():
    assert abs(tr.bracket(19, 7, 2060) - 26 / 2060) < 1e-12


def test_more_reviewer_hits_raise_the_estimate():
    assert tr.bracket(19, 8, 2060) > tr.bracket(19, 7, 2060)


def test_an_empty_population_does_not_divide_by_zero():
    assert math.isnan(tr.bracket(19, 7, 0))


# --------------------------------------------------------------------------- #
# provenance checks — never merge answers made on a different task
# --------------------------------------------------------------------------- #
def test_a_bare_file_is_flagged_as_unverifiable():
    problems = []
    tr.check_against_manifest("r", {"k": "visible"}, {"_bare": True},
                              {"k": {}}, "abc", problems)
    assert any("bare" in p for p in problems)


def test_a_mismatched_digest_is_flagged():
    problems = []
    tr.check_against_manifest("r", {"k": "visible"}, {"manifest_digest": "zzz"},
                              {"k": {}}, "abc", problems)
    assert any("digest" in p for p in problems)


def test_verdicts_for_unknown_items_are_flagged():
    problems = []
    tr.check_against_manifest("r", {"ghost": "visible"}, {"manifest_digest": "abc"},
                              {"k": {}}, "abc", problems)
    assert any("absent from the manifest" in p for p in problems)


def test_a_matching_file_raises_no_problem():
    problems = []
    tr.check_against_manifest("r", {"k": "visible"}, {"manifest_digest": "abc"},
                              {"k": {}}, "abc", problems)
    assert problems == []


# --------------------------------------------------------------------------- #
# bookkeeping
# --------------------------------------------------------------------------- #
def test_the_addressable_sets_are_nested():
    # The tight reading must be a subset of the upper one, or "bracket" is a misnomer.
    assert set(tr.ADDRESSABLE_TIGHT) < set(tr.ADDRESSABLE_UPPER)


def test_excluded_and_addressable_do_not_overlap():
    assert not (set(tr.EXCLUDED) & set(tr.ADDRESSABLE_UPPER))


def test_every_referenced_verdict_exists_in_the_tagger_scheme():
    import make_tagger as mk
    names = {n for n, _, _ in mk.MISS_SCHEME}
    for v in set(tr.ADDRESSABLE_UPPER) | set(tr.EXCLUDED):
        assert v in names, v
