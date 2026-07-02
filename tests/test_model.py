"""Smoke tests for the canonical KeypointModel.

Run inside the project conda environment (torch/timm required):
    pytest tests/ -v

These guard the one invariant that matters for every released checkpoint:
the state_dict key layout of rampnet.model.KeypointModel must stay identical
to the layout the paper-era training scripts saved (feature_extractor as
Sequential over backbone.children()[:-2], plus the 4-layer head).
"""
import timm
import torch
import torch.nn as nn
import pytest

from rampnet.model import (
    BACKBONE_NAME,
    CROP_HEATMAP_SIZE,
    PANO_HEATMAP_SIZE,
    KeypointModel,
)


class LegacyTrainKeypointModel(nn.Module):
    """Verbatim copy of the paper-era train-time construction (v1.0-iccv2025
    stage_two/train.py). All released checkpoints carry this key layout."""

    def __init__(self, heatmap_size):
        super().__init__()
        backbone = timm.create_model(BACKBONE_NAME, pretrained=False)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        in_channels = backbone.num_features
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=heatmap_size, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 1, kernel_size=1)
        )


@pytest.mark.parametrize("heatmap_size", [PANO_HEATMAP_SIZE, CROP_HEATMAP_SIZE])
def test_state_dict_keys_match_paper_checkpoints(heatmap_size):
    canonical = KeypointModel(heatmap_size=heatmap_size, pretrained_backbone=False)
    legacy = LegacyTrainKeypointModel(heatmap_size=heatmap_size)
    canonical_sd = canonical.state_dict()
    legacy_sd = legacy.state_dict()
    assert list(canonical_sd.keys()) == list(legacy_sd.keys())
    for key in canonical_sd:
        assert canonical_sd[key].shape == legacy_sd[key].shape, key


def test_legacy_checkpoint_loads_strictly():
    legacy = LegacyTrainKeypointModel(heatmap_size=PANO_HEATMAP_SIZE)
    canonical = KeypointModel(heatmap_size=PANO_HEATMAP_SIZE, pretrained_backbone=False)
    canonical.load_state_dict(legacy.state_dict())  # must not raise


@pytest.mark.parametrize("heatmap_size", [PANO_HEATMAP_SIZE, CROP_HEATMAP_SIZE])
def test_forward_shape(heatmap_size):
    model = KeypointModel(heatmap_size=heatmap_size, pretrained_backbone=False).eval()
    # Small input keeps this CPU-friendly; the Upsample head fixes the output
    # size regardless of input resolution.
    x = torch.zeros(1, 3, 128, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, heatmap_size[0], heatmap_size[1])
