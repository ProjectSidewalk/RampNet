import hashlib
import os

import torch

_STATE_DICT_KEYS = ('model_state_dict', 'state_dict', 'model')


def load_checkpoint(model, checkpoint_path, map_location='cpu'):
    """Load a RampNet checkpoint into `model`, strictly.

    Accepts the checkpoint formats produced across this repo: a raw state_dict,
    a dict wrapping it under 'model_state_dict'/'state_dict'/'model', and
    DDP-saved dicts with a 'module.' prefix. Any missing file or key mismatch
    raises — never fall back to strict=False, which silently leaves layers
    randomly initialized and corrupts evaluation results.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in _STATE_DICT_KEYS:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
    if state_dict and all(k.startswith('module.') for k in state_dict):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model


def checkpoint_fingerprint(checkpoint_path, digest_chars=12):
    """Short content hash of a checkpoint file, for keying prediction caches.

    Cached heatmaps are only valid for the exact weights that produced them;
    keying the cache directory by this fingerprint makes switching checkpoints
    safe without manually deleting the cache.
    """
    h = hashlib.sha256()
    with open(checkpoint_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:digest_chars]
