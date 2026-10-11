"""``export_hf_model.py --from-hub-revision`` must accept the layout it publishes (#221).

Found by the #221 end-to-end check: the current Hub ``main`` (606a119) stores the HF
wrapper's keys (``model.feature_extractor...``), because that is what ``assemble_package``
saves, but ``load_reference_model`` loaded them straight into a bare ``KeypointModel``
and failed with every key missing. The re-export #229 asks for could not run.
CPU only; no checkpoint, no network.
"""
import os
import sys

import pytest
import torch

from rampnet.model import KeypointModel, PANO_HEATMAP_SIZE

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from export_hf_model import assemble_package, hub_state_dict_to_keypoint  # noqa: E402


def _bare():
    return KeypointModel(heatmap_size=PANO_HEATMAP_SIZE, pretrained_backbone=False)


def test_wrapper_keys_from_an_exported_package_load_strict(tmp_path):
    """The exact layout assemble_package writes round-trips into a bare KeypointModel."""
    pytest.importorskip("transformers")
    from safetensors.torch import load_file
    src = _bare()
    assemble_package(str(tmp_path), src)
    published = load_file(os.path.join(tmp_path, "model.safetensors"))
    assert all(k.startswith("model.") for k in published)      # the layout on the Hub
    dst = _bare()
    dst.load_state_dict(hub_state_dict_to_keypoint(published), strict=True)
    for k, v in src.state_dict().items():
        assert torch.equal(v, dst.state_dict()[k]), k


def test_bare_keys_pass_through_unchanged():
    sd = _bare().state_dict()
    assert hub_state_dict_to_keypoint(sd) is sd


def test_partial_prefix_is_left_for_the_strict_load_to_reject():
    sd = _bare().state_dict()
    mixed = {("model." + k if i % 2 else k): v for i, (k, v) in enumerate(sd.items())}
    out = hub_state_dict_to_keypoint(mixed)
    assert out is mixed
    with pytest.raises(RuntimeError):
        _bare().load_state_dict(out, strict=True)
