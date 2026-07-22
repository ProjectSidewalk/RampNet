from transformers import PreTrainedModel

from .configuration_rampnet import RampNetConfig
# rampnet_model.py is copied verbatim from rampnet/model.py by
# scripts/export_hf_model.py at export time — it is generated, not a fork.
from .rampnet_model import KeypointModel


class RampNetModel(PreTrainedModel):
    config_class = RampNetConfig
    main_input_name = "pixel_values"

    def __init__(self, config):
        super().__init__(config)
        self.model = KeypointModel(
            heatmap_size=tuple(config.heatmap_size),
            pretrained_backbone=False,
        )
        # Required under transformers >= 5.x: initializes loading-related state
        # (e.g. all_tied_weights_keys) that from_pretrained expects on every
        # PreTrainedModel. Harmless no-op extras under 4.x.
        self.post_init()

    def forward(self, pixel_values):
        """Returns the predicted curb ramp keypoint heatmap (B, 1, H, W)."""
        return self.model(pixel_values)
