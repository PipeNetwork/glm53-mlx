"""Load GLM-5.3 through stock mlx-lm (`glm_moe_dsa`), from the raw FP8 release or a converted checkpoint.

Thin wrapper so every script here loads the same way; per-module quantization is replayed from
`config.json` by mlx-lm itself.
"""
from __future__ import annotations
from pathlib import Path


def load_model(path, lazy: bool = False):
    from mlx_lm.utils import load_model as _load
    model, config = _load(Path(path), lazy=lazy)
    return model, config


def load(path, lazy: bool = False):
    from mlx_lm import load as _load
    return _load(str(path), lazy=lazy)
