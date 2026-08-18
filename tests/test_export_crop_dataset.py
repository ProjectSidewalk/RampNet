"""Guards for the round-1 crop-set export in scripts/export_crop_dataset.py.

Two load-bearing rules:

1. **The labels live in the filenames.** `007mz25c_-_118_596_-_478_611.jpg` carries keypoints
   (118, 596) and (478, 611), and the export's whole value-add is turning that convention into a
   real column. If `parse_name` silently drops a keypoint, the published dataset is quietly wrong
   in a way no byte-level check would catch -- the image bytes still round-trip perfectly.

   The leading token is an opaque 8-char string from `random.choices` (download_data.py:277), NOT
   a panorama id. An earlier draft of this export published it as `pano_id`, which would have
   invited users to join it against rampnet-dataset panorama ids and get nothing back.

2. **The card describes the data, it does not assert things about it.** The first draft of the
   card claimed "0 is possible -- negative crops"; the built data has no zero-keypoint crops at
   all (all 27,704 carry at least one). Whichever way a future export comes out, the card has to
   follow the shards.
"""
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from export_crop_dataset import FEATURES, SCHEMA, keypoint_summary, parse_name  # noqa: E402


def test_parses_the_uid_and_a_single_keypoint():
    crop_uid, keypoints = parse_name("007mz25c_-_118_596")
    assert crop_uid == "007mz25c"
    assert keypoints == [{"x": 118, "y": 596}]


def test_the_opaque_token_is_never_published_as_a_panorama_id():
    # download_data.py:277 builds it with random.choices(alphabet, k=8); the source panorama is
    # not recoverable. Calling it pano_id would invite a join that silently returns nothing.
    assert "crop_uid" in SCHEMA.names and "crop_uid" in FEATURES
    assert "pano_id" not in SCHEMA.names and "pano_id" not in FEATURES


def test_parses_every_keypoint_not_just_the_first():
    # The real failure mode: a crop with several ramps losing all but one label.
    _, keypoints = parse_name("007mz25c_-_118_596_-_478_611_-_500_700")
    assert keypoints == [
        {"x": 118, "y": 596}, {"x": 478, "y": 611}, {"x": 500, "y": 700}]


def test_keeps_coordinates_verbatim_including_negatives():
    # Coordinates are stored in the crop's own pixel space and deliberately not normalised
    # or clamped -- see the module docstring on the 0.5 vs 0.515 discrepancy.
    _, keypoints = parse_name("abc_-_-4_2048")
    assert keypoints == [{"x": -4, "y": 2048}]


def test_uid_survives_a_crop_with_no_keypoints():
    crop_uid, keypoints = parse_name("007mz25c")
    assert crop_uid == "007mz25c"
    assert keypoints == []


def test_ignores_a_segment_that_is_not_a_coordinate_pair():
    # Defensive: a stray segment must not be mistaken for a label.
    _, keypoints = parse_name("abc_-_118_596_-_notacoord")
    assert keypoints == [{"x": 118, "y": 596}]


def _shard(tmp_path, counts):
    """Write a minimal one-column shard whose n_keypoints values are `counts`."""
    out = tmp_path / "data" / "train"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"n_keypoints": pa.array(counts, pa.int32())}), str(out / "train-00000.parquet"))
    return tmp_path


def test_summary_counts_keypoints_and_flags_an_all_labelled_set(tmp_path):
    out = _shard(tmp_path, [1, 1, 2, 3])
    text, total = keypoint_summary(out)
    assert total == 7
    assert "no negative crops" in text
    assert "| 1 | 2 |" in text and "| 2 | 1 |" in text and "| 3 | 1 |" in text


def test_summary_reports_negatives_when_the_data_has_them(tmp_path):
    out = _shard(tmp_path, [0, 0, 0, 1, 2])
    text, total = keypoint_summary(out)
    assert total == 3
    # The claim that was wrong in the first draft must not appear when zeros exist.
    assert "no negative crops" not in text
    assert "3 crops carry no keypoint at all" in text


def test_summary_is_empty_for_an_empty_export(tmp_path):
    text, total = keypoint_summary(tmp_path)
    assert (text, total) == ("", 0)
