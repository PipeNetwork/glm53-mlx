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


PER_EXPERT_SUFFIXES = (
    "mlp.switch_mlp.gate_proj.weight", "mlp.switch_mlp.up_proj.weight", "mlp.switch_mlp.down_proj.weight",
    "mlp.switch_mlp.gate_proj.scales", "mlp.switch_mlp.up_proj.scales", "mlp.switch_mlp.down_proj.scales",
    "mlp.switch_mlp.gate_proj.biases", "mlp.switch_mlp.up_proj.biases", "mlp.switch_mlp.down_proj.biases",
    "mlp.gate.weight", "mlp.gate.e_score_correction_bias",
)


def moe_collect(mlp, normed: mx.array, token_chunk: int = 8192):
    """The MoE block over a sequence, accumulating REAP saliency `router_weight * ||expert_output||`.

    Returns (output, saliency[E], counts[E]). Chunked over tokens because the routed activations are
    tokens x top_k x hidden.
    """
    B, L, H = normed.shape
    flat = normed.reshape(-1, H)
    E = mlp.gate.weight.shape[0]
    sal = mx.zeros((E,), dtype=mx.float32)
    cnt = mx.zeros((E,), dtype=mx.float32)
    pieces = []
    for start in range(0, flat.shape[0], token_chunk):
        chunk = flat[start:start + token_chunk][None]
        inds, scores = mlp.gate(chunk)                       # (1, n, k)
        y = mlp.switch_mlp(chunk, inds)                      # (1, n, k, H)
        contrib = scores.astype(mx.float32) * mx.sqrt((y.astype(mx.float32) ** 2).sum(-1))
        flat_idx = inds.reshape(-1)
        sal = sal.at[flat_idx].add(contrib.reshape(-1))
        cnt = cnt.at[flat_idx].add(mx.ones((flat_idx.size,), dtype=mx.float32))
        out = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if getattr(mlp, "shared_experts", None) is not None:
            out = out + mlp.shared_experts(chunk)
        pieces.append(out.reshape(-1, H))
        mx.eval(sal, cnt, pieces[-1])
    return mx.concatenate(pieces, axis=0).reshape(B, L, H), sal, cnt
