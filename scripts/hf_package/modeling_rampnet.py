import numpy as np
import torch
from transformers import PreTrainedModel

from .configuration_rampnet import RampNetConfig
# rampnet_model.py and rampnet_subcell.py are copied verbatim from rampnet/model.py and
# rampnet/subcell.py by scripts/export_hf_model.py at export time -- generated, not forks.
from .rampnet_model import KeypointModel
from .rampnet_subcell import detect_peaks


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

    def detect(self, inputs, threshold=None, decode="gaussian", min_distance=None,
               wrap_x=False):
        """Curb ramp detections as one ``(N, 3)`` array of ``(x, y, score)`` per image.

        ``inputs`` is either preprocessed ``pixel_values`` (a ``(B, 3, H, W)`` tensor; the
        model is run once, no flip TTA) or a raw heatmap this model already produced
        (``(H, W)``, ``(1, H, W)`` or ``(B, 1, H, W)``, tensor or array). Pass the heatmap
        **unclipped and single-pass**: the sub-cell decode reads the 64x128 map the
        512x1024 heatmap is an exact bilinear upsample of, and clipping or a TTA max
        breaks that. Peaks are found on ``clip(heatmap, 0, 1)`` with
        ``peak_local_max(min_distance, threshold_abs=threshold, exclude_border=False)``.

        ``x`` and ``y`` are normalized to [0, 1) (multiply by the image width / height);
        ``score`` is the clipped heatmap value at the peak.

        ``decode="gaussian"`` (the default here) refines each peak from the 8-px grid the
        argmax is quantized to (see the model card); ``decode="argmax"`` returns the
        plain ``peak_local_max`` pixel, as the published evaluation did. Needs
        scikit-image. ``threshold`` and ``min_distance`` default to the config's
        recommended values.

        A refining decode raises ``ValueError`` for ``pixel_values`` whose size is not
        ``config.input_size`` (the head upsamples to a fixed size, so another input size
        changes the x8 factor), and for a heatmap that is not an exact x8 upsample (e.g.
        one already clipped). A flip-TTA heatmap cannot be passed here
        (it is a max of two surfaces). ``rampnet.subcell.detect_peaks(..., coarse=<stack>)``
        decodes TTA output in the repo.
        """
        if threshold is None:
            threshold = self.config.recommended_threshold
        if min_distance is None:
            min_distance = self.config.recommended_min_distance
        if torch.is_tensor(inputs) and inputs.ndim == 4 and inputs.shape[1] == 3:
            if decode != "argmax" and tuple(inputs.shape[-2:]) != tuple(self.config.input_size):
                raise ValueError(
                    f"decode={decode!r} needs pixel_values of size {tuple(self.config.input_size)}"
                    f" (got {tuple(inputs.shape[-2:])}): the heatmap is only an exact x8 "
                    "upsample at the model's input size. Resize the image, or use "
                    "decode='argmax'.")
            param = next(self.parameters())
            with torch.no_grad():
                inputs = self(inputs.to(device=param.device, dtype=param.dtype))
        if torch.is_tensor(inputs):
            t = inputs.detach().cpu()
            if t.dtype in (torch.float16, torch.bfloat16):   # numpy has no bfloat16
                t = t.float()
            inputs = t.numpy()
        h = np.asarray(inputs)            # native dtype: argmax must match peak_local_max
        if not np.issubdtype(h.dtype, np.floating):
            h = h.astype(np.float64)
        if h.ndim == 2:
            h = h[None]
        elif h.ndim == 4 and h.shape[1] == 1:
            h = h[:, 0]
        if h.ndim != 3:
            raise ValueError(f"expected pixel_values (B, 3, H, W) or a heatmap, got {h.shape}")
        out = []
        for hm in h:
            rcs = detect_peaks(hm, threshold, min_distance=min_distance, decode=decode,
                               clip=True, wrap_x=wrap_x)
            H, W = hm.shape
            out.append(np.column_stack([rcs[:, 1] / W, rcs[:, 0] / H, rcs[:, 2]]))
        return out
