"""Guards for the benchmark export in scripts/export_benchmark.py.

Every one of these covers a way the exporter could publish something wrong *silently* -- the
failure mode that matters here, because the artifact lands on the Hub and nobody re-reads it:

1. **The split allowlist.** `benchmark/*/panos` is not a statement of intent. `manual_gold/panos`
   holds the paper's 1,000-panorama gold set, whose imagery is already inside `rampnet-dataset`;
   a bare glob swept it in as a 10th `native` split, and the manifest cross-check could not catch
   it because that directory has no `imagery_manifest.json` to compare against.

2. **The gallery join key.** A gallery crop's file stem is `<pano_id>_<x>_<y>`, so publishing it
   in a column called `pano_id` -- as the first version did -- makes the card's own documented
   join return zero rows. Exactly the trap
   `test_export_crop_dataset.py::test_the_opaque_token_is_never_published_as_a_panorama_id`
   guards for the crop set.

3. **The card's config list.** `records` mode exists so a verdict fix costs megabytes instead of
   11.41 GB, which means it runs against an --out holding only `data/records/`. Deriving the
   config list from that directory alone published a one-config README and orphaned three configs
   that were still sitting on the Hub.
"""
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "analysis"))
from export_benchmark import (  # noqa: E402
    BENCHMARK_SPLITS, GALLERIES, IMAGE_SUFFIXES, MANIFEST_NAME, MODEL_RES, NATIVE, RECORDS,
    GALLERY_FEATURES, GALLERY_SCHEMA, collect, configs_yaml, load_index, parse_gallery_id,
    split_date_range)


# --------------------------------------------------------------------------- the split allowlist

def test_split_allowlist_matches_the_analysis_registry():
    """One list of splits, not two. `miss_decomposition.ALL_SPLITS` is the repo's registry; the
    benchmark package is that minus `manual_gold`, which is a reference set rather than a split."""
    from miss_decomposition import ALL_SPLITS
    assert sorted(BENCHMARK_SPLITS) == sorted(s for s in ALL_SPLITS if s != "manual_gold")
    assert "manual_gold" not in BENCHMARK_SPLITS


def _pano(path, name):
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_bytes(b"\xff\xd8\xff\xe0not-really-a-jpeg")


def test_manual_gold_panos_are_never_collected(tmp_path, capsys):
    # The exact on-disk shape of a full checkout: a real split alongside the gold set.
    _pano(tmp_path / "bend" / "panos", "abc.jpg")
    _pano(tmp_path / "manual_gold" / "panos", "gold.jpg")
    got = {(config, city) for config, city, _ in collect(tmp_path, None, None)}
    assert got == {(NATIVE, "bend")}
    # ...and it says so, rather than dropping it silently.
    assert "skipping manual_gold" in capsys.readouterr().out


def test_an_unknown_directory_is_not_packaged_as_a_split(tmp_path):
    _pano(tmp_path / "some_scratch_dir" / "panos", "x.jpg")
    assert list(collect(tmp_path, None, None)) == []


def test_the_allowlist_applies_to_every_config(tmp_path):
    # A 4096 render dir and a gallery dir for a non-split must be ignored too, not just panos/.
    _pano(tmp_path / "r4096" / "manual_gold", "gold.jpg")
    _pano(tmp_path / "gal" / "manual_gold_incremental_fp", "gold.png")
    assert list(collect(tmp_path / "empty", tmp_path / "r4096", tmp_path / "gal")) == []


def test_non_image_files_never_reach_the_image_decoder(tmp_path):
    """A stray .DS_Store / Thumbs.db / .jpg.tmp used to reach PIL and kill an 11 GB build."""
    panos = tmp_path / "bend" / "panos"
    _pano(panos, "abc.jpg")
    for junk in ("Thumbs.db", ".DS_Store", "abc.json", "def.jpg.tmp"):
        (panos / junk).write_bytes(b"not an image")
    (_, _, items), = collect(tmp_path, None, None)
    assert [stem for stem, _, _ in items] == ["abc"]
    assert all(path.suffix.lower() in IMAGE_SUFFIXES for _, _, path in items)


# ------------------------------------------------------------------------- the gallery join key

def test_gallery_schema_does_not_publish_the_crop_tag_as_a_panorama_id():
    # low_floor_sweep.py:852 names a crop `{pano}_{x:.5f}_{y:.5f}`. Publishing that stem as
    # `pano_id` invites the join documented in the card and returns nothing.
    assert "crop_id" in GALLERY_SCHEMA.names and "crop_id" in GALLERY_FEATURES
    assert "pano_id" in GALLERY_SCHEMA.names and "pano_id" in GALLERY_FEATURES


def test_gallery_id_parses_back_to_a_real_panorama_id():
    assert parse_gallery_id("abc123_0.51234_0.44321") == ("abc123", 0.51234, 0.44321)


def test_gallery_id_keeps_underscores_that_belong_to_the_panorama_id():
    # GSV panorama ids contain underscores, so the split has to come off the right-hand end.
    pano, x, y = parse_gallery_id("A_b_C-d_e_0.10000_0.90000")
    assert pano == "A_b_C-d_e" and (x, y) == (0.1, 0.9)


def test_a_stem_that_is_not_a_tag_yields_no_panorama_id():
    # Better a null than a wrong join key: a hand-added or renamed crop is reported, not guessed.
    assert parse_gallery_id("hand_added_crop") == (None, None, None)


def test_a_bare_panorama_id_is_not_mistaken_for_a_tag():
    assert parse_gallery_id("abc123") == (None, None, None)


# ------------------------------------------------------------------------- the card's config list

def test_configs_yaml_is_emitted_in_a_fixed_order():
    yaml = configs_yaml({GALLERIES: ["bend"], NATIVE: ["bend"], RECORDS: ["bend"]})
    names = [line.split(": ", 1)[1] for line in yaml.splitlines() if line.startswith("- config")]
    assert names == [NATIVE, GALLERIES, RECORDS]


def test_configs_yaml_sorts_cities_within_a_config():
    yaml = configs_yaml({NATIVE: ["paterson", "bend"]})
    assert yaml.index("split: bend") < yaml.index("split: paterson")


def _package(tmp_path, configs, manifest=None):
    for config, cities in configs.items():
        d = tmp_path / "data" / config
        d.mkdir(parents=True, exist_ok=True)
        for city in cities:
            (d / "{}.parquet".format(city)).write_bytes(b"")
    if manifest is not None:
        (tmp_path / MANIFEST_NAME).write_text(
            json.dumps({"configs": manifest}), encoding="utf-8")
    return tmp_path


def test_a_records_only_rebuild_still_declares_the_published_imagery_configs(tmp_path):
    """The F1 case: only records/ is on this disk, but the Hub has all four."""
    out = _package(tmp_path, {RECORDS: ["bend"]},
                   manifest={NATIVE: ["bend", "paterson"], MODEL_RES: ["bend"],
                             GALLERIES: ["bend"], RECORDS: ["bend"]})
    index = load_index(out)
    assert set(index) == {NATIVE, MODEL_RES, GALLERIES, RECORDS}
    assert index[NATIVE] == ["bend", "paterson"]


def test_a_records_only_package_with_no_manifest_is_refused(tmp_path):
    out = _package(tmp_path, {RECORDS: ["bend"]})
    with pytest.raises(SystemExit) as excinfo:
        load_index(out)
    assert "dropping" in str(excinfo.value) and NATIVE in str(excinfo.value)


def test_allow_partial_is_the_explicit_way_to_publish_a_records_only_repo(tmp_path):
    out = _package(tmp_path, {RECORDS: ["bend"]})
    assert set(load_index(out, allow_partial=True)) == {RECORDS}


def test_a_city_new_on_disk_is_added_to_what_the_manifest_knew(tmp_path):
    out = _package(tmp_path, {NATIVE: ["bend", "clovis"], MODEL_RES: ["bend"],
                              GALLERIES: ["bend"]},
                   manifest={NATIVE: ["bend"], MODEL_RES: ["bend"], GALLERIES: ["bend"]})
    assert load_index(out)[NATIVE] == ["bend", "clovis"]


# ------------------------------------------------------------------------------ the card's dates

def test_the_split_date_range_is_read_from_the_committed_verdicts():
    """Hardcoded as "2026-07-22 to 2026-07-31" until sao_paulo landed on 2026-08-01 outside it."""
    benchmark = os.path.join(REPO_ROOT, "benchmark")
    assert split_date_range(benchmark, ["sao_paulo"]) == "on 2026-08-01"
    assert split_date_range(benchmark, BENCHMARK_SPLITS) == "between 2026-07-22 and 2026-08-01"


def test_the_date_range_never_invents_a_date_it_cannot_read(tmp_path):
    assert "2026" not in split_date_range(tmp_path, ["bend"])
