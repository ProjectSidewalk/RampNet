import os

import timm
import torch.nn as nn

# The backbone of every released checkpoint. Do not change it: it is the default.
BACKBONE_NAME = 'convnextv2_base.fcmae_ft_in22k_in1k_384'

# Issue #107's candidate: the same ConvNeXt-base size (87.6 M parameters), pretrained by
# DINOv3 distillation. It is ConvNeXt V1, so it has no GRN parameters and its checkpoints do
# not load into the default model, nor the default's into it. The weights are gated on the
# Hugging Face Hub and carry the DINOv3 License, which a fine-tuned checkpoint inherits.
DINOV3_CONVNEXT_BASE = 'convnext_base.dinov3_lvd1689m'

# Setting this environment variable changes the backbone KeypointModel() builds when no
# backbone_name is passed. It exists so the scorers, which all call KeypointModel() with no
# arguments, can evaluate a non-default checkpoint without a flag on each one. A mismatch
# with the checkpoint is not silent: rampnet.loading.load_checkpoint loads strictly.
BACKBONE_ENV = 'RAMPNET_BACKBONE'

# Panorama model (stage_two): equirectangular 2048x4096 input -> 512x1024 heatmap.
PANO_INPUT_SIZE = (2048, 4096)
PANO_HEATMAP_SIZE = (512, 1024)

# Crop model (stage_one): perspective 1024x352 input -> 256x88 heatmap.
CROP_INPUT_SIZE = (1024, 352)
CROP_HEATMAP_SIZE = (256, 88)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resolve_backbone(backbone_name=None):
    """Return the timm backbone id to build: the argument, else $RAMPNET_BACKBONE, else
    the released default.

    >>> resolve_backbone('convnext_base.dinov3_lvd1689m')
    'convnext_base.dinov3_lvd1689m'
    """
    return backbone_name or os.environ.get(BACKBONE_ENV) or BACKBONE_NAME


def check_normalization(backbone, backbone_name):
    """Refuse a backbone whose pretraining statistics differ from IMAGENET_MEAN/STD.

    Every training and inference transform in this repo normalizes with those constants. A
    backbone pretrained with other statistics would train and score, only worse, and that
    would read as "the backbone is worse" (#107), so it is an error here.
    """
    cfg = getattr(backbone, 'pretrained_cfg', None) or {}
    for key, expected in (('mean', IMAGENET_MEAN), ('std', IMAGENET_STD)):
        got = cfg.get(key)
        if got is None or any(abs(a - b) > 1e-6 for a, b in zip(got, expected))                 or len(got) != len(expected):
            raise ValueError(
                f"backbone {backbone_name!r} was pretrained with {key}={got}, but this repo "
                f"normalizes with {expected}; refusing to build a silently mismatched model")


class KeypointModel(nn.Module):
    """The one canonical RampNet keypoint-heatmap model.

    The state_dict key layout (feature_extractor as Sequential over
    backbone.children()[:-2], i.e. dropping norm_pre and head) is the layout
    every released checkpoint was trained and saved with — including the
    HuggingFace weights. Do not restructure the modules (e.g. features_only=True
    or named submodules): that changes the keys and breaks strict loading of
    all existing checkpoints.

    `backbone_name` (default: $RAMPNET_BACKBONE, else BACKBONE_NAME) selects another timm
    ConvNeXt for a backbone comparison (#107). A checkpoint loads only into the backbone
    it was trained with; `self.backbone_name` records which one this instance has.
    """

    def __init__(self, heatmap_size=PANO_HEATMAP_SIZE, pretrained_backbone=False,
                 backbone_name=None):
        super().__init__()
        self.backbone_name = resolve_backbone(backbone_name)
        backbone = timm.create_model(self.backbone_name, pretrained=pretrained_backbone)
        check_normalization(backbone, self.backbone_name)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        in_channels = backbone.num_features
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=heatmap_size, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, image):
        features = self.feature_extractor(image)
        heatmap = self.head(features)
        return heatmap
