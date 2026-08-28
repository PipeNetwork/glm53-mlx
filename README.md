# glm53-mlx

MLX (Apple Silicon) runtime and quantization tooling for
[**zai-org/GLM-5.3**](https://huggingface.co/zai-org/GLM-5.3) — 744B-parameter `glm_moe_dsa`
MoE (256 experts, top-8; MLA with DeepSeek-V3.2-style sparse attention), the same architecture and
config as GLM-5.2.

Published builds: **[pipenetwork/GLM-5.3 MLX](https://huggingface.co/collections/pipenetwork)**
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

**How to read this.** The ladder is the only measurement that covers the whole set: 8-bit is closest to bfloat16 (free-running error 0.131, cosine 0.989 after 78 layers), 6-bit next (0.167). The upstream **FP8 release scores 0.173 — between 6-bit and 5-bit** — so any build converted from FP8 starts below our 6-bit, which is why these are converted from bf16. 5-bit (0.225) and mixed 4/8-bit (0.250) are close, and only the mixed build fits a 512 GB Mac; uniform 4-bit (0.357) is the tight 512 GB option; **mixed 3/6-bit (0.424) loses to uniform 4-bit** — it leads for the first ten layers, then 3-bit expert damage compounds — and is published for the 384 GB fit, not for quality. Perplexity on the builds that fit this machine is below.
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
