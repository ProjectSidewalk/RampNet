import timm
import torch.nn as nn

BACKBONE_NAME = 'convnextv2_base.fcmae_ft_in22k_in1k_384'

# Panorama model (stage_two): equirectangular 2048x4096 input -> 512x1024 heatmap.
PANO_INPUT_SIZE = (2048, 4096)
PANO_HEATMAP_SIZE = (512, 1024)

# Crop model (stage_one): perspective 1024x352 input -> 256x88 heatmap.
CROP_INPUT_SIZE = (1024, 352)
CROP_HEATMAP_SIZE = (256, 88)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class KeypointModel(nn.Module):
    """The one canonical RampNet keypoint-heatmap model.

    The state_dict key layout (feature_extractor as Sequential over
    backbone.children()[:-2], i.e. dropping norm_pre and head) is the layout
    every released checkpoint was trained and saved with — including the
    HuggingFace weights. Do not restructure the modules (e.g. features_only=True
    or named submodules): that changes the keys and breaks strict loading of
    all existing checkpoints.
    """

    def __init__(self, heatmap_size=PANO_HEATMAP_SIZE, pretrained_backbone=False):
        super().__init__()
        backbone = timm.create_model(BACKBONE_NAME, pretrained=pretrained_backbone)
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
