"""Numerical parity of this package's `glm_moe_dsa` runtime against transformers, at tiny scale.

A random tiny model with every fragile path live: dense and sparse MoE layers, `full` and `shared`
indexer layers, a sequence longer than `index_topk` so selection is live, `index_topk_freq` and
`index_skip_topk_offset` at their release values, interleaved RoPE, more experts than top-k.
Weights are presented exactly as the release lays them out (per-expert tensors, indexer weights on
`full` layers only, MTP layer present) so the runtime's sanitize and strict loading are tested too.

    .venv/bin/python tests/test_parity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mlx.core as mx

from glm53_mlx import runtime as R

TINY = dict(
    model_type="glm_moe_dsa", vocab_size=128, hidden_size=64, intermediate_size=96,
    moe_intermediate_size=32, num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=4,
    n_shared_experts=1, n_routed_experts=8, routed_scaling_factor=2.5, kv_lora_rank=16,
    q_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=24, qk_head_dim=32, v_head_dim=16, head_dim=24,
    n_group=1, topk_group=1, num_experts_per_tok=2, norm_topk_prob=True, hidden_act="silu",
    max_position_embeddings=4096, rms_norm_eps=1e-5, first_k_dense_replace=1, moe_layer_freq=1,
    index_topk=4, index_head_dim=8, index_n_heads=16, index_topk_freq=4, index_skip_topk_offset=3,
    index_topk_pattern=None, index_share_for_mtp_iteration=True, indexer_rope_interleave=True,
    rope_interleave=True, indexer_types=["full", "full", "shared", "shared", "full", "shared"],
    mlp_layer_types=["dense"] + ["sparse"] * 5, rope_parameters={"rope_theta": 8000000.0, "rope_type": "default"},
    scoring_func="sigmoid", topk_method="noaux_tc", attention_bias=False, tie_word_embeddings=False,
    moe_router_dtype="float32", num_nextn_predict_layers=0, pad_token_id=0, eos_token_id=[1],
    ep_size=1, pretraining_tp=1,
)
B, T = 2, 14


def inputs():
    torch.manual_seed(1)
    return torch.randint(2, TINY["vocab_size"], (B, T))


def build_hf(seed=0):
    from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
    torch.manual_seed(seed)
    model = GlmMoeDsaForCausalLM(GlmMoeDsaConfig(**TINY)).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("e_score_correction_bias"):
                p.normal_(0, 1.0)
            elif "norm" in name and p.ndim == 1:
                p.add_(0.3 * torch.randn_like(p))
            elif "proj" in name and p.ndim >= 2:
                p.mul_(2.0)
    return model


def release_layout(model):
    """Tensors as the HF release ships them: per-expert weights (transformers fuses them on load)."""
    out = {}
    for k, v in model.state_dict().items():
        v = v.detach().numpy() if not v.is_floating_point() else v.detach().float().numpy()
        if k.endswith("mlp.experts.gate_up_proj"):
            base = k[: -len("gate_up_proj")]; inter = v.shape[1] // 2
            for e in range(v.shape[0]):
                out[f"{base}{e}.gate_proj.weight"] = mx.array(v[e, :inter]); out[f"{base}{e}.up_proj.weight"] = mx.array(v[e, inter:])
        elif k.endswith("mlp.experts.down_proj"):
            base = k[: -len("down_proj")]
            for e in range(v.shape[0]):
                out[f"{base}{e}.down_proj.weight"] = mx.array(v[e])
        else:
            out[k] = mx.array(v)
    return out


def build_mlx(hf, weights=None):
    model = R.Model(R.ModelArgs.from_dict(TINY))
    weights = weights if weights is not None else model.sanitize(release_layout(hf))
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters()); model.eval()
    return model


def logits_hf(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits.float().numpy()


def logits_mlx(model, ids, cache=None):
    return np.array(model(mx.array(ids.numpy()), cache=cache).astype(mx.float32))


def report(label, out, ref, tol=1e-4):
    scale = float(np.abs(ref).max()); delta = float(np.abs(out - ref).max()); ok = delta < tol * max(scale, 1.0)
    print(f"  {label:58s} max|delta| {delta:.3e}  (scale {scale:.3e})  {'OK' if ok else 'FAIL'}"); return ok


def main():
    ids = inputs(); all_ok = True
    hf = build_hf()
    keys = set(release_layout(hf))
    print(f"[0] release layout: indexer weights on layers {sorted({int(k.split('.')[2]) for k in keys if '.indexer.' in k})} (full = {[i for i, t in enumerate(TINY['indexer_types']) if t == 'full']})")
    try:
        model = build_mlx(hf); print("  strict load: OK")
    except ValueError as e:
        print("  strict load FAIL:", str(e).splitlines()[0][:120]); return 1
    ref = logits_hf(hf, ids)
    print("[1] full forward (T=14 > index_topk=4: selection live; full+shared layers; topk_freq=4, skip_offset=3)")
    all_ok &= report("logits", logits_mlx(model, ids), ref)
    torch.manual_seed(2); ids40 = torch.randint(2, TINY["vocab_size"], (B, 40))
    all_ok &= report("logits, T=40 (sorted-gather MoE path, deep sparse)", logits_mlx(model, ids40), logits_hf(hf, ids40))
    print("[2] short sequence (T <= index_topk: dense bypass)")
    all_ok &= report("logits", logits_mlx(model, ids[:, :4]), logits_hf(hf, ids[:, :4]))
    print("[3] token-by-token decode == single forward")
    cache = model.make_cache()
    steps = [logits_mlx(model, ids[:, t:t + 1], cache=cache) for t in range(T)]
    all_ok &= report("incremental vs single-shot", np.concatenate(steps, 1), logits_mlx(model, ids))
    cache = model.make_cache()
    a = logits_mlx(model, ids[:, :6], cache=cache); b_ = logits_mlx(model, ids[:, 6:], cache=cache)
    all_ok &= report("chunked prefill 6+8 vs single-shot", np.concatenate([a, b_], 1), logits_mlx(model, ids))
    print("[4] control: the shared-layer fix is load-bearing (stock behaviour = every layer runs its own indexer,")
    print("    shared layers' weights missing from the checkpoint and left at random init by a lenient load)")
    stock = R.Model(R.ModelArgs.from_dict({**TINY, "indexer_types": ["full"] * TINY["num_hidden_layers"]}))
    stock.load_weights(list(model.sanitize(release_layout(hf)).items()), strict=False); mx.eval(stock.parameters()); stock.eval()
    ref40 = logits_hf(hf, ids40)
    d = float(np.abs(logits_mlx(stock, ids40) - ref40).max()); d_short = float(np.abs(logits_mlx(stock, ids40[:, :4]) - logits_hf(hf, ids40[:, :4])).max())
    print(f"  {'T=40 (> index_topk): logits move by':58s} {d:.3e}  {'OK' if d > 1e-2 else 'FAIL'}")
    print(f"  {'T=4 (<= index_topk): unaffected, as expected':58s} {d_short:.3e}")
    all_ok &= d > 1e-2
    print("\nALL OK" if all_ok else "\nSOME CHECKS FAILED"); return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
