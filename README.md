# glm53-mlx

MLX (Apple Silicon) runtime and quantization tooling for
[**zai-org/GLM-5.3**](https://huggingface.co/zai-org/GLM-5.3) — 744B-parameter `glm_moe_dsa`
MoE (256 experts, top-8; MLA with DeepSeek-V3.2-style sparse attention), the same architecture and
config as GLM-5.2.

Published builds: **[pipenetwork/GLM-5.3 MLX](https://huggingface.co/collections/pipenetwork/glm-53-mlx-6a91e67071233946179533d5)**
(see [Measurements](#measurements)).

## Why this exists

mlx-lm carries `glm_moe_dsa` as a 53-line shim over `deepseek_v32`, unchanged since February. It
builds a lightning indexer on every one of the 78 layers. GLM-5.2 and GLM-5.3 ship indexer weights
on **21** (`indexer_types`: a layer is `full` iff `max(i − 3 + 1, 0) mod 4 == 0`; the other 57
`shared` layers carry no indexer and reuse the most recent full layer's top-k selection — in prefill
and in decode). The shim's `ModelArgs` drops `indexer_types`, so:

* a strict load of the release fails — **285 missing parameters** (57 layers × 5);
* `mlx_lm.load` loads with `strict=False` and leaves those 57 indexers at their random
  initialisation, silently. Below `index_topk` (2048 tokens) the indexer is bypassed and output is
  correct; beyond it, 57 of 78 layers attend to keys chosen by random projections.

Our own [GLM-5.2 set](https://huggingface.co/collections/pipenetwork/glm-52-mlx-6a31fa56e37a8ac73daf25b7)
was converted and smoke-tested that way — fine at short context, wrong past 2048 tokens under stock
mlx-lm. The runtime here (`glm53_mlx/runtime.py`, bundled in every checkpoint as `model_file`) builds
the indexer only on full layers and threads each full layer's selection into the shared layers that
follow, as the reference does; it also uses the reference's fp32 indexer scores and router logits and
the indexer LayerNorm epsilon (1e-6). Everything else in the shim — interleaved RoPE on the 64 rope
dims, the MLA split, FP8 block dequantization, sigmoid/noaux routing — was checked and matches.

## Validation

```bash
./scripts/run_tests.sh
```

A random tiny model with dense and sparse layers, `full` and `shared` indexer layers on the release
schedule, a sequence longer than `index_topk` so selection is live, and weights presented exactly as
the release ships them (per-expert tensors, indexer weights on full layers only), against
`transformers` 5.16:

```
[0] release layout: indexer weights on layers [0, 1, 4] (full = [0, 1, 4]);  strict load: OK
[1] full forward (T=14 > index_topk=4)     max|delta| 4.321e-07  (scale 6.473e-01)  OK
    T=40 (sorted-gather MoE, deep sparse)  max|delta| 3.874e-07  OK
[2] short sequence (dense bypass)          max|delta| 2.831e-07  OK
[3] token-by-token decode == single forward   4.247e-07 OK;  chunked prefill 6+8   0.000e+00 OK
[4] control: stock behaviour (own indexer on every layer, random where missing)
    T=40: logits move by 6.292e-01;  T=4: 4.172e-07 (unaffected below index_topk, as expected)
```

One thing that bit while writing the test: with few indexer heads, `relu`-summed scores tie at exactly
zero and `torch.topk` / `mx.argpartition` break ties differently, which looks like a large error and
is not one. Use ≥16 indexer heads in tiny configs.

## Source precision

The FP8 release is a lossy derivative of the bf16 one: dequantized FP8 weights differ from
`GLM-5.3-BF16` by up to 1.6e-2 on values of 0.46 (half an e4m3 step). The builds here are converted
from **bf16**; the ladder below measures the FP8 release itself as one more "recipe" so the cost of
converting from FP8 is a number rather than an assumption.

## Building the quants

```bash
python scripts/quantize_stream.py --src GLM-5.3-BF16-src --dst out/GLM-5.3-MLX-4bit --bits 4
python scripts/quantize_stream.py --src ... --dst out/GLM-5.3-MLX-mixed-3_6bit --bits 3 --other-bits 6
```

One decoder layer at a time (the 256 experts of a layer are stacked together, ~19 GB in bf16),
resumable at layer boundaries; the 1.5 TB source never needs to fit. Routed experts (97.5%) at
`--bits`, everything else quantizable at `--other-bits`; the lightning indexer and the MoE router stay
as stored; the MTP layer is dropped.

## Measurements

At 744B, 8-bit (~800 GB), 6-bit (~625 GB) and 5-bit (~530 GB) cannot be loaded on a 512 GB
machine — not by us, not by anyone downloading them onto one Mac. So there are two measurements:

* `scripts/eval_ladder.py` — every decoder layer run in bf16 and in each recipe on identical inputs,
  teacher-forced and free-running (ported from our Qwen3.8-2.4T work); ranks the whole ladder.
* `scripts/ppl_large.py` + `ppl_compare.py` — wikitext-2 perplexity on identical windows for the
  builds that fit (4-bit, mixed).

<!-- measurements -->
Per-layer divergence vs bf16, 16,384 tokens, 78 layers (relative L2 of the layer output; `fp8` = the FP8 release itself):

| recipe | teacher-forced (mean over layers) | free-running (final layer) | cosine (final) |
|---|---:|---:|---:|
| 8bit | 0.00685 | 0.13119 | 0.98945 |
| 6bit | 0.01465 | 0.16736 | 0.98389 |
| 5bit | 0.02651 | 0.22521 | 0.97272 |
| 4bit | 0.05161 | 0.35740 | 0.93390 |
| mixed-4_8bit | 0.02524 | 0.24951 | 0.96710 |
| mixed-3_6bit | 0.05242 | 0.42380 | 0.90624 |
| fp8 | 0.01741 | 0.17321 | 0.98320 |

Perplexity, wikitext-2 test, identical windows, builds that fit 512 GB:

| build | size | perplexity [95% CI] |
|---|---:|---|
| [4bit](https://huggingface.co/pipenetwork/GLM-5.3-MLX-4bit) | 418.6 GB | 2.8636 [2.6681, 3.0714] |
| [mixed-4_8bit](https://huggingface.co/pipenetwork/GLM-5.3-MLX-mixed-4_8bit) | 427.8 GB | 2.7420 [2.5533, 2.9477] |
| [mixed-3_6bit](https://huggingface.co/pipenetwork/GLM-5.3-MLX-mixed-3_6bit) | 332.6 GB | 3.0338 [2.8366, 3.2386] |
| [REAP25-4bit](https://huggingface.co/pipenetwork/GLM-5.3-REAP25-MLX-4bit) | 316.6 GB | 3.2872 [3.0703, 3.5184] |
| [REAP37-4bit](https://huggingface.co/pipenetwork/GLM-5.3-REAP37-MLX-4bit) | 267.2 GB | 3.8517 [3.6212, 4.0937] |
| [REAP50-4bit](https://huggingface.co/pipenetwork/GLM-5.3-REAP50-MLX-4bit) | 214.7 GB | 5.0295 [4.7571, 5.3137] |

**Recommendation.** For a 512 GB Mac, **mixed 4/8-bit** (427.7 GB): perplexity 2.7420, a paired 4.3% better than uniform 4-bit (ratio 0.9575 [0.9537, 0.9612], better on 98.6% of windows) for 9 GB more — the 2.5% of non-expert weights are worth their 8 bits, as on every model we have measured. Uniform 4-bit (418.6 GB) is the fallback when those 9 GB matter. **Mixed 3/6-bit** (332.6 GB) is the 384 GB-class option, at a real cost: 3.0338, +5.9% over 4-bit and +10.6% over mixed 4/8 — it leads the ladder for the first ten layers and then 3-bit expert damage compounds. Among the builds that cannot be run here, the ladder puts 8-bit closest to bfloat16 (free-running error 0.131), then 6-bit (0.167); the **upstream FP8 release scores 0.173, between 6-bit and 5-bit**, which is why these are converted from the bf16 release. 5-bit (0.225) sits just above mixed 4/8 (0.250) at 100 GB more.
<!-- /measurements -->

## Layout

| path | what |
|---|---|
| `glm53_mlx/runtime.py` | the runtime (mlx-lm `deepseek_v32` + GLM indexer schedule + fixes), bundled in each checkpoint |
| `glm53_mlx/stream.py` | layer-at-a-time access to the source (ladder + quantizer) |
| `scripts/quantize_stream.py` | per-layer streaming quantizer, resumable |
| `scripts/eval_ladder.py` | per-layer divergence ladder incl. the `fp8` release variant |
| `scripts/ppl_corpus.py`, `ppl_large.py`, `ppl_compare.py`, `ppl_table.py` | perplexity |
| `scripts/check_strict_load.py` | strict load through stock mlx-lm + `model_file`, then generate |
| `scripts/upload.py`, `make_collection.py`, `publish.sh` | publishing |
| `tests/test_parity.py` | validation above |
| `docs/upstream-notes.md` | findings for mlx-lm |
