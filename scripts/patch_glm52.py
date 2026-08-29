"""Add the fixed `glm_moe_dsa` runtime to the published GLM-5.2 MLX repos (same architecture and config).

For each repo: fetch the safetensors headers from the Hub (no weight download), build our model from the
repo's own config with its quantization map replayed exactly as mlx-lm's loader does, and require that
every parameter name and shape matches the checkpoint — zero missing, zero unexpected — before pushing a
single commit with `glm_moe_dsa.py`, `config.json` (+ `model_file`) and a README note.

    python scripts/patch_glm52.py [--yes]
"""
import argparse, io, json, os, re, sys
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from huggingface_hub import CommitOperationAdd, HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm53_mlx.runtime import Model, ModelArgs

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "glm53_mlx" / "runtime.py"
NOTE = """
## Runtime — updated 2026-08-28: load with `--trust-remote-code`

This repository now bundles `glm_moe_dsa.py` (declared via `model_file` in `config.json`), a fixed runtime
for this architecture, and needs it:

```bash
mlx_lm.generate --model pipenetwork/{repo} --trust-remote-code --prompt "..." --max-tokens 300
```

mlx-lm's own `glm_moe_dsa` builds a lightning indexer on all 78 layers, but GLM-5.2 ships indexer weights
on 21 (`indexer_types`: the other 57 "shared" layers reuse the previous full layer's top-k selection).
`mlx_lm.load` loads leniently and left those 57 indexers at random initialisation. Prompts up to 2048
tokens were unaffected (the indexer is bypassed below `index_topk`); beyond that, 57 of 78 layers attended
to keys chosen by random projections. The bundled runtime implements the schedule as the reference does
(plus fp32 indexer scores and router logits and the indexer LayerNorm epsilon); tiny-config parity against
`transformers` 5.16 is 4e-7 with the sparse path live, and a strict load of this checkpoint reports zero
missing and zero unexpected tensors. Details, tests and the GLM-5.3 builds made with it:
[github.com/PipeNetwork/glm53-mlx](https://github.com/PipeNetwork/glm53-mlx). The weights are unchanged.
"""


def check(api, repo):
    meta = api.get_safetensors_metadata(repo)
    remote = {k: tuple(v.shape) for k, v in meta.weight_map.items()} if hasattr(meta, "weight_map") and isinstance(next(iter(meta.weight_map.values())), object) and False else None
    remote = {}
    for fname, fmeta in meta.files_metadata.items():
        for k, t in fmeta.tensors.items():
            remote[k] = tuple(t.shape)
    cfg = json.load(open(api.hf_hub_download(repo, "config.json", local_dir=f"/tmp/glm52cfg/{repo.split('/')[1]}")))
    model = Model(ModelArgs.from_dict(cfg))
    weights = model.sanitize(dict.fromkeys(remote))  # names only; sanitize is name-based for a converted checkpoint
    q = cfg["quantization"]
    def class_predicate(p, m):
        if p in q:
            return q[p]
        if not hasattr(m, "to_quantized"):
            return False
        return f"{p}.scales" in weights
    kw = {k: v for k, v in q.items() if not isinstance(v, dict)}
    nn.quantize(model, class_predicate=class_predicate, **kw)
    ours = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}
    missing = sorted(set(ours) - set(remote)); unexpected = sorted(set(remote) - set(ours))
    mismatch = sorted(k for k in set(ours) & set(remote) if ours[k] != remote[k])
    return cfg, missing, unexpected, mismatch, len(remote)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--yes", action="store_true"); args = ap.parse_args()
    api = HfApi()
    repos = sorted(m.id for m in api.list_models(author="pipenetwork", search="GLM-5.2") if "GGUF" not in m.id)
    ok_repos = []
    for repo in repos:
        cfg, missing, unexpected, mismatch, n = check(api, repo)
        status = "OK" if not (missing or unexpected or mismatch) else "FAIL"
        print(f"{repo:42s} {n:6d} tensors  missing {len(missing)}  unexpected {len(unexpected)}  shape-mismatch {len(mismatch)}  {status}")
        for k in (missing[:3] + unexpected[:3] + mismatch[:3]):
            print("    ", k)
        if status == "OK":
            ok_repos.append((repo, cfg))
    if not args.yes:
        print("\ndry run — pass --yes to commit to the repos that checked OK"); return
    for repo, cfg in ok_repos:
        cfg = dict(cfg); cfg["model_file"] = "glm_moe_dsa.py"
        readme = open(api.hf_hub_download(repo, "README.md", local_dir=f"/tmp/glm52cfg/{repo.split('/')[1]}")).read()
        if "## Runtime — updated" not in readme:
            m = re.search(r"^# .*$", readme, flags=re.M)
            at = m.end() if m else len(readme)
            readme = readme[:at] + "\n" + NOTE.format(repo=repo.split("/")[1]) + readme[at:]
        ops = [CommitOperationAdd(path_in_repo="glm_moe_dsa.py", path_or_fileobj=str(RUNTIME)),
               CommitOperationAdd(path_in_repo="config.json", path_or_fileobj=io.BytesIO(json.dumps(cfg, indent=2).encode())),
               CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=io.BytesIO(readme.encode()))]
        api.create_commit(repo_id=repo, operations=ops, commit_message="Bundle the fixed glm_moe_dsa runtime (shared indexer layers) via model_file; weights unchanged")
        print("patched", repo)


if __name__ == "__main__":
    main()
