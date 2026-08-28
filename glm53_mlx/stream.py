"""Layer-at-a-time access to the FP8 release, for the divergence ladder and the quantizer."""
from __future__ import annotations

import json
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .runtime import DeepseekV32DecoderLayer, Model, ModelArgs


def read_config(src: Path):
    raw = json.load(open(src / "config.json"))
    return raw, ModelArgs.from_dict(raw)


def shard_map(src: Path) -> dict[str, str]:
    return json.load(open(src / "model.safetensors.index.json"))["weight_map"]


def load_subset(src: Path, smap: dict[str, str], prefix: str, _cache: dict = {}) -> dict[str, mx.array]:
    """All tensors whose name starts with `prefix`, memory-mapped (nothing read until evaluated)."""
    out = {}
    for k, shard in smap.items():
        if k.startswith(prefix):
            if shard not in _cache:
                _cache.clear(); _cache[shard] = mx.load(str(src / shard))
            out[k] = _cache[shard][k]
    return out


class _Sanitizer:
    """`Model.sanitize` needs the (lazy) full model for the kv_b split; build it once."""
    def __init__(self, margs: ModelArgs):
        self.model = Model(margs)

    def __call__(self, weights: dict) -> dict:
        return self.model.sanitize(weights)


def build_layer(margs: ModelArgs, layer_i: int, sanitized: dict[str, mx.array]) -> nn.Module:
    """A single decoder layer holding the given (already sanitized) weights."""
    layer = DeepseekV32DecoderLayer(margs, layer_i)
    prefix = f"model.layers.{layer_i}."
    layer.load_weights([(k[len(prefix):], v) for k, v in sanitized.items() if k.startswith(prefix)], strict=True)
    return layer


def run_layer(layer: nn.Module, h: mx.array) -> mx.array:
    from mlx_lm.models.base import create_attention_mask
    x = h.astype(mx.bfloat16)
    mask = create_attention_mask(x, None, return_array=True)
    return layer(x, mask, None)[0].astype(mx.float32)
