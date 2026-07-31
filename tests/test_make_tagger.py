"""Unit tests for the gallery tagging page (#46).

No browser, no imagery — the page is a generated string, so what is testable is that
it contains every item, picks the right verdict scheme, and keeps its resume key
stable. That last one is the failure that would actually hurt: a moved
``localStorage`` key silently abandons a half-finished tagging session, and the tagger
would look empty rather than broken.
"""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import make_tagger as mk  # noqa: E402


def _miss(key="p_0.5_0.6"):
    return {key: {"file": f"{key}.jpg", "city": "bend", "pano": "p", "bucket": "silent",
                  "field": "near", "dist_m": 9.1, "source_px": 140.0,
                  "model_px": 35.0, "source_width": 16384, "parity": "advantaged"}}


def _fp(key="q_0.2_0.7"):
    return {key: {"file": f"{key}.jpg", "city": "bend", "pano": "q", "bucket": "isolated",
                  "model": "owlv2", "confidence": 0.83, "dist_m": 12.0,
                  "source_px": 90.0, "model_px": 22.0, "source_width": 16384,
                  "parity": "advantaged"}}


# --------------------------------------------------------------------------- #
# scheme_for — the scheme is the scientific content, so it must not be guessed wrong
# --------------------------------------------------------------------------- #
def test_a_false_positive_manifest_gets_the_fp_scheme():
    scheme, kind = mk.scheme_for(_fp())
    assert kind == "fp" and scheme is mk.FP_SCHEME


def test_a_miss_manifest_gets_the_miss_scheme():
    scheme, kind = mk.scheme_for(_miss())
    assert kind == "miss" and scheme is mk.MISS_SCHEME


def test_the_scheme_is_read_from_content_not_the_directory_name():
    # A renamed gallery directory must not change the verdict scheme.
    assert mk.scheme_for(_fp("anything_at_all"))[1] == "fp"


def test_each_scheme_has_exactly_one_tight_sourcing_verdict():
    # One answer is the tight "more data would fix this" estimate; two would make the
    # rate ambiguous. `context-only` is deliberately NOT a second one — it is the
    # upper variant, reported as `visible + context-only`, and the guide says so.
    assert [n for n, _, _ in mk.MISS_SCHEME].count("visible") == 1
    assert [n for n, _, _ in mk.FP_SCHEME].count("real-ramp") == 1


def test_the_miss_scheme_separates_resolving_a_ramp_from_inferring_one():
    # A reviewer who reasons "there is a crosswalk and a coloured apron, so there must
    # be a ramp" has NOT resolved the ramp. RampNet sees that same context at the same
    # resolution, so the two are different capabilities and must not share a verdict.
    names = [n for n, _, _ in mk.MISS_SCHEME]
    assert "context-only" in names
    assert names.index("visible") < names.index("context-only")


def test_the_guide_tells_the_reviewer_to_judge_from_the_model_panel():
    # The instrument work is worthless if the briefing does not say which panel
    # answers the question.
    assert "panel 3" in mk.MISS_GUIDE
    assert "context-only" in mk.MISS_GUIDE


def test_the_guide_flags_that_near_field_is_the_load_bearing_population():
    assert "near-field" in mk.MISS_GUIDE.lower()


def test_every_scheme_offers_an_escape_hatch():
    # Without 'unclear' a reviewer is forced to invent a verdict on bad imagery.
    for scheme in (mk.MISS_SCHEME, mk.FP_SCHEME):
        assert "unclear" in [n for n, _, _ in scheme]


def test_scheme_verdict_names_are_unique():
    for scheme in (mk.MISS_SCHEME, mk.FP_SCHEME):
        names = [n for n, _, _ in scheme]
        assert len(names) == len(set(names))


def test_schemes_fit_the_single_digit_shortcuts():
    for scheme in (mk.MISS_SCHEME, mk.FP_SCHEME):
        assert len(scheme) <= 9


# --------------------------------------------------------------------------- #
# store_key — resume must survive regeneration AND a new process
# --------------------------------------------------------------------------- #
def test_store_key_is_deterministic_within_a_process():
    assert mk.store_key("miss", "t") == mk.store_key("miss", "t")


def test_store_key_is_stable_across_processes():
    # The reason this is not hash(): PYTHONHASHSEED randomizes str hashing per
    # process, so regenerating the page would move the key and orphan the session.
    code = ("import sys; sys.path.insert(0, r'%s'); import make_tagger; "
            "print(make_tagger.store_key('miss', 'the title'))"
            % os.path.join(REPO, "scripts", "analysis"))
    seen = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        seen.add(subprocess.run([sys.executable, "-c", code], env=env,
                                capture_output=True, text=True,
                                check=True).stdout.strip())
    assert len(seen) == 1, f"store key varies with PYTHONHASHSEED: {seen}"


def test_different_galleries_get_different_stores():
    assert mk.store_key("miss", "a") != mk.store_key("miss", "b")
    assert mk.store_key("miss", "a") != mk.store_key("fp", "a")


# --------------------------------------------------------------------------- #
# build_html — every item must reach the page
# --------------------------------------------------------------------------- #
def test_every_item_is_embedded():
    manifest = {}
    for i in range(5):
        manifest.update(_miss(f"p_0.{i}_0.6"))
    html = mk.build_html(manifest, mk.MISS_SCHEME, "miss", "t")
    for key in manifest:
        assert key in html, key


def test_the_placeholders_are_all_substituted():
    html = mk.build_html(_miss(), mk.MISS_SCHEME, "miss", "my title")
    for ph in ("__ITEMS__", "__KEYS__", "__LEGEND__", "__STORE__", "__TITLE__"):
        assert ph not in html, ph
    assert "my title" in html


def test_the_embedded_items_are_valid_json():
    manifest = _miss()
    html = mk.build_html(manifest, mk.MISS_SCHEME, "miss", "t")
    blob = html.split("const ITEMS = ", 1)[1].split(", KEYS = ", 1)[0]
    items = json.loads(blob)
    assert len(items) == 1 and items[0]["key"] in manifest


def test_the_shortcut_order_matches_the_legend_order():
    html = mk.build_html(_fp(), mk.FP_SCHEME, "fp", "t")
    keys = json.loads(html.split("KEYS = ", 1)[1].split(", STORE", 1)[0])
    assert keys == [n for n, _, _ in mk.FP_SCHEME]


def test_parity_is_carried_so_the_page_can_warn_about_it():
    html = mk.build_html(_miss(), mk.MISS_SCHEME, "miss", "t")
    assert "advantaged" in html


# --------------------------------------------------------------------------- #
# _meta_line — what the reviewer reads while deciding
# --------------------------------------------------------------------------- #
def test_meta_shows_distance_and_source_pixels():
    line = mk._meta_line(list(_miss().values())[0], "miss")
    assert "9.1 m" in line and "140.0 src px" in line


def test_fp_meta_names_the_model_and_its_confidence():
    line = mk._meta_line(list(_fp().values())[0], "fp")
    assert "owlv2" in line and "conf 0.83" in line


def test_meta_omits_missing_fields_rather_than_printing_blanks():
    assert "··" not in mk._meta_line({"city": "bend"}, "miss")
