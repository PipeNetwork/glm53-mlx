"""Per-layer damage for every build in the ladder, against the FP8-dequantized bf16 reference, in one pass.

(Ported from qwen38-mlx for GLM-5.3, 744B: 8-bit is ~800 GB and 6-bit ~625 GB, so those builds cannot be
run on a 512 GB machine; a decoder layer is ~58 GB in bf16 and can.)

Most quantization work reports perplexity. That is not available here for most of the ladder: even
2-bit is 0.76 TB against 550 GB of memory, so **the full-size builds can never be run** — not by me,
not by anyone downloading them onto a single machine. Publishing them with no number attached, or
with a number borrowed from a different model, would be worse than publishing nothing.

What *is* measurable is the thing perplexity is a proxy for. A decoder layer is 51.6 GB and fits, so
each layer can be run twice — once in bf16, once quantized — on identical inputs, and the divergence
recorded. Two propagation modes, because they answer different questions:

* **teacher-forced** — every variant sees the bf16 hidden states. Isolates each layer's own damage,
  so layers can be compared to each other and the ladder ranked.
* **free-running** — each variant feeds itself. This is what actually happens at inference, and it
  is where a width that looks survivable per-layer can still diverge by layer 92.

Both are paired by construction: same layer, same inputs, same tokens, differing only in the weights.
The whole ladder is measured in a single read of the 4.89 TB source, which matters because the
source lives on a spinning external disk.

    python scripts/eval_ladder.py --src <bf16 dir> --out ladder.npz \\
        --variants 8bit,6bit,5bit,4bit,mixed-3_6bit,mixed-4_8bit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm53_mlx.runtime import ModelArgs
from glm53_mlx.stream import (
    _Sanitizer, build_layer, load_subset, read_config, run_layer,
    shard_map,
)


def parse_variant(spec: str) -> tuple[str, int, int]:
    """``4bit`` -> (name, expert bits, other bits); ``mixed-3_6bit`` -> (name, 3, 6)."""
    name = spec.strip()
    if name.startswith("mixed-"):
        e, o = name[len("mixed-"):-3].split("_")
        return name, int(e), int(o)
    if not name.endswith("bit"):
        raise ValueError(f"bad variant {spec!r}; expected e.g. 4bit or mixed-3_6bit")
    return name, int(name[:-3]), int(name[:-3])


def make_predicate(expert_bits: int, other_bits: int, group_size: int):
    def predicate(path: str, module: nn.Module):
        if not hasattr(module, "to_quantized") or module.weight.shape[-1] % group_size:
            return False
        if ".indexer." in path:
            return {"group_size": group_size, "bits": 8}
        return {"group_size": group_size, "bits": expert_bits if "switch_mlp" in path else other_bits}
    return predicate


def divergence(got: mx.array, want: mx.array) -> tuple[float, float]:
    """Per-token relative L2 error and cosine similarity, averaged over tokens.

    Relative error says how big the damage is; cosine says whether the direction survived. They
    come apart — a uniformly shrunk activation has large relative error and cosine 1.0 — and the
    pair is more informative than either alone.
    """
    a = got.reshape(-1, got.shape[-1]).astype(mx.float32)
    b = want.reshape(-1, want.shape[-1]).astype(mx.float32)
    diff = mx.sqrt(mx.sum((a - b) ** 2, axis=-1))
    norm = mx.sqrt(mx.sum(b * b, axis=-1))
    cos = mx.sum(a * b, axis=-1) / mx.maximum(
        mx.sqrt(mx.sum(a * a, axis=-1)) * norm, 1e-20)
    rel = mx.mean(diff / mx.maximum(norm, 1e-20))
    mx.eval(rel, cos)
    return float(rel.item()), float(mx.mean(cos).item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--fp8-src", help="the FP8 release dir; adds the variant `fp8` = its dequantized weights, unquantized")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--mode", choices=("teacher", "free", "both"), default="both")
    ap.add_argument("--ids", help="npy of token ids, instead of the standard set")
    ap.add_argument("--limit-layers", type=int, default=0)
    args = ap.parse_args()

    src, out_path = Path(args.src), Path(args.out)
    variants = [parse_variant(v) for v in args.variants.split(",") if v != "fp8"]
    fp8 = Path(args.fp8_src) if args.fp8_src and "fp8" in args.variants.split(",") else None
    if fp8 is not None:
        variants.append(("fp8", 0, 0))
    raw, margs = read_config(src)
    smap = shard_map(src)
    smap_fp8 = shard_map(fp8) if fp8 is not None else None
    sanitize = _Sanitizer(margs)

    ids_np = np.load(args.ids) if args.ids else np.load(Path(__file__).resolve().parents[1] / "ppl_corpus.npy")
    ids = mx.array(ids_np[: args.samples * args.seq_len].reshape(args.samples, args.seq_len))
    print(f"eval tokens: {ids.size}  variants: {[v[0] for v in variants]}")

    embed = sanitize(load_subset(src, smap, "model.embed_tokens"))["model.embed_tokens.weight"]
    # float32 carry between layers, bf16 activation rounding at 92 layer
    # boundaries costs more than the quantization damage being measured.
    h_ref = embed[ids].astype(mx.float32)
    mx.eval(h_ref)
    del embed
    mx.clear_cache()

    n_layers = args.limit_layers or margs.num_hidden_layers
    names = [v[0] for v in variants]
    tf_rel = np.zeros((n_layers, len(variants)))
    tf_cos = np.zeros((n_layers, len(variants)))
    fr_rel = np.zeros((n_layers, len(variants)))
    fr_cos = np.zeros((n_layers, len(variants)))
    h_var = {name: h_ref for name in names} if args.mode in ("free", "both") else {}

    started = time.time()
    for layer_i in range(n_layers):
        t0 = time.time()
        weights = sanitize(load_subset(src, smap, f"model.layers.{layer_i}."))
        mx.eval(*weights.values())  # dequantized bf16 layer, read once from disk

        ref_layer = build_layer(margs, layer_i, weights)
        ref_out = mx.concatenate(
            [run_layer(ref_layer, h_ref[b : b + args.batch]) for b in
             range(0, h_ref.shape[0], args.batch)], axis=0)
        mx.eval(ref_out)
        del ref_layer
        mx.clear_cache()

        for vi, (name, ebits, obits) in enumerate(variants):
            w = weights
            if name == "fp8":
                # The FP8 release, dequantized, as published: how lossy is it against bf16?
                w = sanitize(load_subset(fp8, smap_fp8, f"model.layers.{layer_i}."))
                layer = build_layer(margs, layer_i, w)
            else:
                layer = build_layer(margs, layer_i, w)
                nn.quantize(layer, args.group_size, ebits, class_predicate=make_predicate(ebits, obits, args.group_size))
            mx.eval(layer.parameters())

            if args.mode in ("teacher", "both"):
                got = mx.concatenate(
                    [run_layer(layer, h_ref[b : b + args.batch]) for b in
                     range(0, h_ref.shape[0], args.batch)], axis=0)
                mx.eval(got)
                tf_rel[layer_i, vi], tf_cos[layer_i, vi] = divergence(got, ref_out)
                del got

            if args.mode in ("free", "both"):
                hv = h_var[name]
                got = mx.concatenate(
                    [run_layer(layer, hv[b : b + args.batch]) for b in
                     range(0, hv.shape[0], args.batch)], axis=0)
                mx.eval(got)
                fr_rel[layer_i, vi], fr_cos[layer_i, vi] = divergence(got, ref_out)
                h_var[name] = got
                mx.eval(h_var[name])
                del got

            del layer, w
            mx.clear_cache()

        h_ref = ref_out
        del weights, ref_out
        mx.clear_cache()

        rate = (time.time() - started) / (layer_i + 1)
        report = "  ".join(
            f"{n}:{tf_rel[layer_i, i]:.4f}/{fr_rel[layer_i, i]:.4f}" for i, n in enumerate(names))
        print(f"layer {layer_i:2d}/{n_layers} {time.time()-t0:6.1f}s  "
              f"eta {rate*(n_layers-layer_i-1)/60:5.1f}m  tf/free  {report}", flush=True)

        np.savez(out_path, names=np.array(names), teacher_rel=tf_rel, teacher_cos=tf_cos,
                 free_rel=fr_rel, free_cos=fr_cos, layers=np.array(layer_i + 1),
                 tokens=np.array(ids.size))

    print(f"\n{'variant':>14} {'teacher rel':>12} {'free rel (final)':>17} {'free cos':>10}")
    for i, name in enumerate(names):
        print(f"{name:>14} {tf_rel[:, i].mean():>12.5f} {fr_rel[n_layers-1, i]:>17.5f} "
              f"{fr_cos[n_layers-1, i]:>10.5f}")
    print(f"\nwrote {out_path}  ({(time.time()-started)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
