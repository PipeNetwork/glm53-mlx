"""Publish a built GLM-5.3 MLX quant to the Hub, with a card rendered from the measurements.

    .venv/bin/python scripts/upload.py --dir <build dir> --repo pipenetwork/<name> [--yes] [--card-only]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path("/Users/david/llm/glm53-out")
UPSTREAM = "zai-org/GLM-5.3"
CODE_REPO = "https://github.com/PipeNetwork/glm53-mlx"
ORDER = ["GLM-5.3-MLX-8bit", "GLM-5.3-MLX-6bit", "GLM-5.3-MLX-5bit", "GLM-5.3-MLX-4bit", "GLM-5.3-MLX-mixed-4_8bit", "GLM-5.3-MLX-mixed-3_6bit",
         "GLM-5.3-REAP25-MLX-4bit", "GLM-5.3-REAP37-MLX-4bit", "GLM-5.3-REAP50-MLX-4bit", "GLM-5.3-REAP50-MLX-3bit"]
RAM = {"GLM-5.3-MLX-8bit": "1 TB+ (two machines)", "GLM-5.3-MLX-6bit": "768 GB (two machines)", "GLM-5.3-MLX-5bit": "768 GB (two machines)",
       "GLM-5.3-MLX-4bit": "512 GB, tight", "GLM-5.3-MLX-mixed-4_8bit": "512 GB, tight", "GLM-5.3-MLX-mixed-3_6bit": "512 GB",
       "GLM-5.3-REAP25-MLX-4bit": "384 GB-class", "GLM-5.3-REAP37-MLX-4bit": "384 GB-class", "GLM-5.3-REAP50-MLX-4bit": "256 GB Mac",
       "GLM-5.3-REAP50-MLX-3bit": "192 GB Mac"}
LADDER_NAME = {"GLM-5.3-MLX-8bit": "8bit", "GLM-5.3-MLX-6bit": "6bit", "GLM-5.3-MLX-5bit": "5bit", "GLM-5.3-MLX-4bit": "4bit",
               "GLM-5.3-MLX-mixed-4_8bit": "mixed-4_8bit", "GLM-5.3-MLX-mixed-3_6bit": "mixed-3_6bit"}

CARD = """---
license: other
license_name: glm-5.3
license_link: LICENSE
base_model: {upstream}
base_model_relation: quantized
tags:
- mlx
- apple-silicon
- glm_moe_dsa
- mixture-of-experts
- {bits_tag}
pipeline_tag: text-generation
library_name: mlx
---

# {repo_name}

MLX (Apple Silicon) build of [**GLM-5.3**](https://huggingface.co/{upstream}) — 744B-parameter
`glm_moe_dsa` MoE (256 experts, top-8; MLA with DeepSeek-V3.2-style sparse attention) — quantized
to **{recipe}**.

**These files are modified**: converted from the upstream **bfloat16** release
([GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16)) to MLX and quantized; the architecture
is unchanged. The multi-token-prediction layer (78) is not included.

## Runtime — read this

This checkpoint bundles `glm_moe_dsa.py` (declared via `model_file`) and needs it:

```bash
pip install -U mlx-lm
mlx_lm.generate --model pipenetwork/{repo_name} --trust-remote-code --prompt "..." --max-tokens 300
```

mlx-lm's own `glm_moe_dsa` builds a lightning indexer on all 78 layers, but GLM-5.2/5.3 ship
indexer weights on 21 (`indexer_types`: the other 57 "shared" layers reuse the previous full layer's
top-k selection). A strict load of the release fails with 285 missing parameters; `mlx_lm.load`
loads leniently and leaves those 57 indexers at random initialisation. Prompts up to 2048 tokens are
unaffected (the indexer is bypassed below `index_topk`); beyond that, 57 layers attend to keys
chosen by random projections. The bundled runtime implements the schedule as the reference does,
plus the reference's fp32 indexer scores and router logits and the indexer LayerNorm epsilon.
Tiny-config parity against `transformers` 5.16 is **4e-7** with the sparse path live, cached decode
exact; strict loading of this checkpoint reports zero missing and zero unexpected tensors. Details and
tests: [{code_repo}]({code_repo}).

## Size and what is quantized

**{gb:.1f} GB** on disk. RAM: {ram}.

| group | share of parameters | this build |
|---|---:|---|
| routed experts (`switch_mlp`, 75 layers × 256) | 724.8B (97.5%) | {expert_bits}-bit, group 64 |
| attention (MLA), shared experts, dense layers 0–2, embeddings, `lm_head` | 18.4B (2.5%) | {other_bits}-bit, group 64 |
| lightning indexer (21 layers), MoE router + correction bias, norms | 0.3B | as stored (bf16 / fp32) |

Source precision: the FP8 release is a lossy derivative of the bf16 one (dequantized FP8 weights
differ from bf16 by up to 1.6e-2 on values of 0.46 — half an e4m3 step). {fp8_note}

{reap_section}## Quality

Two measurements, because at 744B most of the ladder cannot be loaded on a 512 GB machine:

**Per-layer divergence vs bf16** (`scripts/eval_ladder.py`): every decoder layer run in bf16 and in
each recipe on identical inputs ({ladder_tokens:,} tokens of wikitext-2), *teacher-forced* (each layer
sees bf16 inputs — isolates its own damage) and *free-running* (each recipe feeds itself — what
inference does). Relative L2 error of the layer output; lower is better.

{ladder_table}

**Perplexity** on wikitext-2 (test), {tokens:,} tokens in {windows} windows of {seq}, for the builds
that fit this machine, scored on identical windows:

{ppl_table}

{recommendation}

Greedy generation (a collapse detector, not a ranking) is coherent on every published build.

## License

[GLM-5.3 license](LICENSE), as the upstream model. Port code: [{code_repo}]({code_repo}).
"""


def hub_sizes():
    """Sizes from the local build dirs where present, else from the Hub (local copies of the big builds are deleted after upload)."""
    out = {}
    from huggingface_hub import HfApi
    api = HfApi()
    for n in ORDER:
        d = OUT_ROOT / n
        if d.exists():
            out[n] = sum(f.stat().st_size for f in d.iterdir() if f.is_file()) / 1e9
        else:
            try:
                out[n] = sum((x.size or 0) for x in api.model_info(f"pipenetwork/{n}", files_metadata=True).siblings) / 1e9
            except Exception:
                pass
    return out


def ladder_table(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    names = [str(n) for n in z["names"]]; L = int(z["layers"])
    tf, fr, fc = z["teacher_rel"][:L], z["free_rel"][:L], z["free_cos"][:L]
    rows = ["| recipe | teacher-forced (mean over layers) | free-running (final layer) | cosine (final) |", "|---|---:|---:|---:|"]
    for i, n in enumerate(names):
        label = n if n == "fp8" else n
        rows.append(f"| {label} | {tf[:, i].mean():.5f} | {fr[L-1, i]:.5f} | {fc[L-1, i]:.5f} |")
    return "\n".join(rows), int(z["tokens"]), L


def ppl_table(results: dict, sizes: dict):
    rows = ["| build | size | perplexity [95% CI] |", "|---|---:|---|"]
    for n in ORDER:
        if n in results:
            r = results[n]
            label = n.replace("GLM-5.3-", "").replace("MLX-", "")
            rows.append(f"| [{label}](https://huggingface.co/pipenetwork/{n}) | {sizes.get(n, 0):.1f} GB | {r['perplexity']:.4f} [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}] |")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--repo", required=True)
    ap.add_argument("--results", default=str(ROOT / "ppl_results.json"))
    ap.add_argument("--ladder", default=str(OUT_ROOT / "ladder.npz"))
    ap.add_argument("--yes", action="store_true"); ap.add_argument("--card-only", action="store_true")
    args = ap.parse_args()
    d = Path(args.dir); cfg = json.load(open(d / "config.json")); q = cfg["quantization"]
    overrides = {k: v for k, v in q.items() if isinstance(v, dict)}
    expert_bits = next((v["bits"] for k, v in overrides.items() if ".switch_mlp." in k), q["bits"])
    other_bits = next((v["bits"] for k, v in overrides.items() if ".switch_mlp." not in k), q["bits"])
    recipe = f"{expert_bits}-bit" if expert_bits == other_bits else f"{expert_bits}-bit experts / {other_bits}-bit everything else"
    gb = sum(p.stat().st_size for p in d.iterdir() if p.is_file()) / 1e9
    sizes = hub_sizes()
    results = json.load(open(args.results)) if os.path.exists(args.results) else {}
    lt, ladder_tokens, L = ladder_table(Path(args.ladder)) if os.path.exists(args.ladder) else ("(ladder pending)", 0, 0)
    pt = ppl_table(results, sizes) if results else "(pending)"
    any_ppl = next((results[n] for n in ORDER if n in results), None)
    fp8_note = ("The ladder row `fp8` is the FP8 release itself measured against bf16: its error is the floor any FP8-sourced build inherits."
                if os.path.exists(args.ladder) and "fp8" in [str(n) for n in np.load(args.ladder, allow_pickle=True)["names"]] else "")
    recommendation = json.load(open(ROOT / "recommendation.json"))["text"] if (ROOT / "recommendation.json").exists() else ""
    name = args.repo.split("/")[-1]
    reap_section = ""
    if "reap" in cfg:
        r = cfg["reap"]
        sal = np.load(OUT_ROOT / r["saliency"], allow_pickle=True)
        halves = sal["saliency_halves"]; moe = [int(i) for i in sal["moe_layers"]]; keep_k = r["kept_experts"]
        ov = np.mean([len(set(np.argsort(-halves[0, i])[:keep_k]) & set(np.argsort(-halves[1, i])[:keep_k])) / keep_k for i in moe])
        reap_section = (f"## REAP pruning\n\nThis build keeps **{r['kept_experts']} of {r['original_experts']}** routed experts per MoE layer "
                        f"({r['ratio_pct']}% pruned; the 3 dense layers, attention, shared experts and the router are untouched), chosen by "
                        f"REAP saliency — mean `router_weight × ‖expert_output‖` over {r['calibration_tokens']:,} calibration tokens "
                        f"(wikitext-2 *train*, ten languages of Wikipedia and code; checked for zero 32-gram overlap with the eval set). "
                        f"Kept experts carry {100*r['saliency_retained_mean']:.1f}% of the layers' saliency mass on average. Ranking the "
                        f"experts on two disjoint halves of the calibration set picks the same kept set {100*ov:.1f}% of the time. "
                        f"The pruning was applied to the already-quantized {r.get('base', '4-bit')} build, which is exactly equivalent to "
                        f"pruning bf16 and requantizing (expert subsetting and affine groups are on different axes). Saliency retention "
                        f"is not a quality measure — the perplexity below is.\n\n")
    card = CARD.format(upstream=UPSTREAM, code_repo=CODE_REPO, repo_name=name, bits_tag=f"{expert_bits}-bit", recipe=recipe, gb=gb,
                       ram=RAM.get(name, "see table"), expert_bits=expert_bits, other_bits=other_bits, fp8_note=fp8_note,
                       ladder_tokens=ladder_tokens, ladder_table=lt, tokens=any_ppl["tokens"] if any_ppl else 0,
                       windows=any_ppl["windows"] if any_ppl else 0, seq=any_ppl["seq_len"] if any_ppl else 0, ppl_table=pt,
                       recommendation=recommendation, reap_section=reap_section)
    (d / "README.md").write_text(card)
    print(f"repo   {args.repo}\ndir    {d}\nfiles  {sum(1 for p in d.iterdir() if p.is_file())}, {gb:.1f} GB\n"); print(lt); print(); print(pt)
    if not args.yes:
        print("\ndry run — pass --yes to upload"); return 0
    from huggingface_hub import HfApi
    import time
    api = HfApi()
    if args.card_only:
        api.upload_file(path_or_fileobj=str(d / "README.md"), path_in_repo="README.md", repo_id=args.repo, repo_type="model")
        print(f"\ncard refreshed https://huggingface.co/{args.repo}"); return 0
    api.create_repo(args.repo, exist_ok=True, repo_type="model")
    for _ in range(30):
        try:
            api.model_info(args.repo); break
        except Exception:
            time.sleep(2)
    api.upload_folder(folder_path=str(d), repo_id=args.repo, repo_type="model")
    print(f"\nuploaded https://huggingface.co/{args.repo}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
