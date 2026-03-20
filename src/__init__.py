"""
Jamuna Bankline Prediction — src package.

Patches torch.load AND torch.serialization.load at import time to work around
PyTorch 2.6+ defaulting weights_only=True, which rejects pytorch-forecasting
checkpoint objects (they contain pandas DataFrames and other complex types).

Both symbols must be patched because:
- `torch.load` is accessed by external callers via module attribute lookup
- `torch.serialization.load` is the underlying function Lightning may bind
  directly when it does `from torch.serialization import load`
"""
import torch
import torch.serialization

_orig_torch_load = torch.serialization.load  # Patch the canonical source


def _patched_torch_load(*args, **kwargs):
    # Force weights_only=False to allow unpickling complex objects.
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


# Patch both access paths
torch.load = _patched_torch_load
torch.serialization.load = _patched_torch_load
