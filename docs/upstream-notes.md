# Notes for ml-explore/mlx-lm (`glm_moe_dsa`, GLM-5.2 / GLM-5.3)

Found by tiny-config parity against `transformers` 5.16 (`tests/test_parity.py`) and by strict-loading
the release; the port's math otherwise matches (MLA with interleaved RoPE, FP8 block dequant, routing).

1. **`indexer_types` is not modelled.** GLM-5.2/5.3 run a lightning indexer on 21 of 78 layers
   (`full` iff `max(i - index_skip_topk_offset + 1, 0) % index_topk_freq == 0`, i.e. layers 0,1,2,6,10,…,74);
   the 57 `shared` layers carry no indexer weights and reuse the most recent full layer's top-k
   (`modeling_glm_moe_dsa.py:376-377, 432-446, 739-750`). `glm_moe_dsa.py` builds an `Indexer` in
   every `DeepseekV32Attention` and its `ModelArgs` silently drops `indexer_types`, so a strict load
   fails with 285 missing parameters (57 × 5) and `mlx_lm.load` (strict=False) leaves them at random
   init. Prompts ≤ `index_topk` (2048) are unaffected because the indexer is bypassed; beyond that,
   57 layers attend to keys chosen by random projections. Fix: derive/accept `indexer_types`, build
   the indexer only on full layers, return the selection from attention and thread it through the
   decoder loop (`prev_topk_indices`), skip the `mx.depends` on the empty indexer cache for shared
   layers. With the fix, parity vs transformers is 4e-7 at T > index_topk, cached decode exact.
2. `indexer.k_norm` LayerNorm eps is 1e-6 in the reference (`modeling:191`); the port uses the
   MLX default 1e-5.
3. Indexer scores and `weights_proj` are computed in fp32 by the reference (`modeling:239-244`);
   the port runs them in bf16, which flips keys at the top-k boundary.
4. Router logits: reference `F.linear(x.float(), weight.float())` (`moe_router_dtype: float32`);
   `MoEGate` matmuls in bf16.
5. `mlx_lm.load` passing `strict=False` is what let 1. go unnoticed in published checkpoints; a
   warning listing missing parameters would have caught it.
