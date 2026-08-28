"""Per-expert REAP saliency (`router_weight x ||expert_output||`), one decoder layer at a time.

Layer i's statistics depend only on the hidden states arriving at layer i, so the pass carries
activations forward (float32) and holds one layer (~19 GB bf16). Accumulated in two disjoint halves
of the calibration set as well as in total, so "is this enough tokens to rank 256 experts?" gets a
measured answer (split-half overlap) rather than an assumption. Resumable per layer.

    python scripts/reap_calibrate.py --src <bf16 dir> --ids calib_corpus.npy --out saliency.npz [--resume]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import mlx.core as mx
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm53_mlx.stream import _Sanitizer, build_layer, load_subset, moe_collect, read_config, shard_map
from mlx_lm.models.base import create_attention_mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ids", required=True)
    ap.add_argument("--samples", type=int, default=32); ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4); ap.add_argument("--token-chunk", type=int, default=8192)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    src, out_path = Path(args.src), Path(args.out); state_path = out_path.with_suffix(".state.safetensors")
    raw, margs = read_config(src); smap = shard_map(src); sanitize = _Sanitizer(margs)
    ids_np = np.load(args.ids)[: args.samples * args.seq_len].reshape(args.samples, args.seq_len)
    ids = mx.array(ids_np); print(f"calibration: {ids.shape[0]} x {ids.shape[1]} = {ids.size} tokens", flush=True)
    n_layers, E = margs.num_hidden_layers, margs.n_routed_experts
    saliency = np.zeros((2, n_layers, E)); counts = np.zeros((2, n_layers, E)); start_layer = 0
    if args.resume and state_path.exists():
        st = mx.load(str(state_path)); hidden = st["hidden"]; saliency = np.array(st["saliency"], dtype=np.float64)
        counts = np.array(st["counts"], dtype=np.float64); start_layer = int(st["layer"].item()); print(f"resuming at layer {start_layer}")
    else:
        embed = sanitize(load_subset(src, smap, "model.embed_tokens"))["model.embed_tokens.weight"]
        hidden = embed[ids].astype(mx.float32); mx.eval(hidden); del embed; mx.clear_cache()
    started = time.time()
    for layer_i in range(start_layer, n_layers):
        t0 = time.time()
        weights = sanitize(load_subset(src, smap, f"model.layers.{layer_i}.")); mx.eval(*weights.values())
        layer = build_layer(margs, layer_i, weights); del weights
        is_moe = hasattr(layer.mlp, "switch_mlp")
        sal = [mx.zeros((E,), dtype=mx.float32) for _ in range(2)]; cnt = [mx.zeros((E,), dtype=mx.float32) for _ in range(2)]
        outs = []; midpoint = hidden.shape[0] // 2
        for b0 in range(0, hidden.shape[0], args.batch):
            half = 0 if b0 < midpoint else 1
            x = hidden[b0:b0 + args.batch].astype(mx.bfloat16)
            h = layer.input_layernorm(x)
            r, _ = layer.self_attn(h, create_attention_mask(h, None, return_array=True), None, None)
            x = x + r
            normed = layer.post_attention_layernorm(x)
            if is_moe:
                y, s, c = moe_collect(layer.mlp, normed, args.token_chunk)
                sal[half], cnt[half] = sal[half] + s, cnt[half] + c
            else:
                y = layer.mlp(normed)
            outs.append((x + y).astype(mx.float32)); mx.eval(outs[-1], sal[half], cnt[half])
        hidden = mx.concatenate(outs, axis=0); mx.eval(hidden)
        for h in range(2):
            saliency[h, layer_i] = np.array(sal[h], dtype=np.float64); counts[h, layer_i] = np.array(cnt[h], dtype=np.float64)
        del layer, outs, sal, cnt; mx.clear_cache()
        used = int((counts[:, layer_i].sum(0) > 0).sum()); rate = (time.time() - started) / (layer_i - start_layer + 1)
        print(f"layer {layer_i:2d}/{n_layers}  {time.time()-t0:6.1f}s  experts used {used:3d}/{E}  eta {rate*(n_layers-layer_i-1)/60:5.1f} min", flush=True)
        mx.save_safetensors(str(state_path), {"hidden": hidden, "saliency": mx.array(saliency.astype(np.float32)),
                                              "counts": mx.array(counts.astype(np.float32)), "layer": mx.array([layer_i + 1])})
    total, count = saliency.sum(0), counts.sum(0)
    mean = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    halves = np.where(counts > 0, saliency / np.maximum(counts, 1), 0.0)
    np.savez(out_path, saliency=mean, total=total, counts=count, saliency_halves=halves, counts_halves=counts,
             tokens=np.array(ids.size), samples=np.array(args.samples), seq_len=np.array(args.seq_len),
             moe_layers=np.array([i for i in range(n_layers) if count[i].sum() > 0]))
    print(f"\nwrote {out_path}  ({(time.time()-started)/60:.1f} min)")
    print("\nsplit-half agreement — would each half drop the same experts?"); print(f"{'keep':>6} {'overlap':>9} {'spearman':>9}")
    moe = [i for i in range(n_layers) if count[i].sum() > 0]
    for keep in (0.75, 0.63, 0.5):
        k = max(1, int(round(E * keep))); ov, rh = [], []
        for i in moe:
            a, b = halves[0, i], halves[1, i]
            ov.append(len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k)
            ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); rh.append(np.corrcoef(ra, rb)[0, 1])
        print(f"{keep:>6.0%} {np.mean(ov):>8.1%} {np.mean(rh):>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
