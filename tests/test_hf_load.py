"""Load smoke test for the exported HuggingFace package (issue #19).

The published package must load via the custom-code path

    AutoModel.from_pretrained(<dir>, trust_remote_code=True)

on the transformers versions we support. transformers 5.13 changed the
custom-model auto-class contract (it now calls ``register_for_auto_class`` on
the remote-code class and expects loading-related state such as
``all_tied_weights_keys``); the pre-#19 published package registered a bare
``KeypointModel`` and crashed. This test asserts the *current* exporter's
package — ``RampNetModel(PreTrainedModel)`` wrapping ``KeypointModel``, with
``post_init()`` — round-trips a load and forward pass on whatever transformers
is installed.

Run inside the project conda env, and specifically re-run it on each supported
transformers version to actually close #19:

    pip install 'transformers==5.12.1' && pytest tests/test_hf_load.py -v
    pip install 'transformers>=5.13'   && pytest tests/test_hf_load.py -v
"""
import os
import sys

import torch
from transformers import AutoModel

from rampnet.model import KeypointModel, PANO_HEATMAP_SIZE

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from export_hf_model import assemble_package  # noqa: E402


def test_hf_package_loads_with_trust_remote_code(tmp_path):
    # Random-init weights are enough: the contract under test is the load path,
    # not a specific checkpoint. Assemble exactly what the exporter ships.
    reference = KeypointModel(heatmap_size=PANO_HEATMAP_SIZE, pretrained_backbone=False).eval()
    pkg_dir = str(tmp_path / "pkg")
    assemble_package(pkg_dir, reference)

    # The exact call from issue #19's repro.
    model = AutoModel.from_pretrained(pkg_dir, trust_remote_code=True).eval()

    # Small input keeps this CPU-friendly; the Upsample head fixes output size.
    x = torch.zeros(1, 3, 128, 256)
    with torch.no_grad():
        out = model(x)
        ref_out = reference(x)
    assert out.shape == (1, 1, PANO_HEATMAP_SIZE[0], PANO_HEATMAP_SIZE[1])
    # Weights survived the assemble -> save -> from_pretrained round-trip.
    assert torch.allclose(out, ref_out, atol=1e-6)
