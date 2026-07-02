"""Shared RampNet code: canonical model, checkpoint loading, and metrics.

Import submodules directly so that pure-Python parts stay usable without the
deep-learning stack installed:

    from rampnet.model import KeypointModel        # needs torch + timm
    from rampnet.loading import load_checkpoint    # needs torch
    from rampnet.metrics import match_predictions  # pure Python
"""
