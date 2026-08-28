"""Quantize GLM-5.3 one decoder layer at a time, never holding the 756 GB FP8 source.

The release stores each layer's 256 routed experts as separate FP8 tensors with 128x128 block
scales; the runtime's sanitize dequantizes them and stacks them into one `switch_mlp` tensor per
projection, which needs all 256 at once. So the index is walked by layer: the tensors of layer N
are gathered lazily from whichever shards hold them (nothing is read until evaluation),
sanitized as a group, quantized tensor by tensor, and written out. Peak memory is one layer
(~58 GB of bf16 experts plus the quantized copy).

Which modules are quantized is *derived* from the runtime: a full-size `Model` is built lazily
and every leaf that defines `to_quantized` is recorded, then filtered by the recipe:
  * routed experts (`switch_mlp`, 97.4% of parameters)   --bits / --expert-bits, group 64
  * everything else quantizable                            --bits / --other-bits, group 64
  * kept as stored: lightning indexer, MoE router + correction bias, norms
  * dropped: the multi-token-prediction layer (78)

    python scripts/quantize_stream.py --src <fp8 dir> --dst <out dir> --bits 4
    python scripts/quantize_stream.py --src ... --dst ...-mixed-3_6bit --bits 3 --other-bits 6
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm53_mlx.runtime import Model, ModelArgs

AUX_FILES = ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
             "special_tokens_map.json", "LICENSE")


def recipe(path: str, args) -> dict | None:
    for sub, bits in (args.override or []):
        if sub in path:
            return {"group_size": args.group_size, "bits": bits}
    if ".indexer." in path or path.endswith("mlp.gate"):
        return None  # selection / routing weights stay as stored (0.05% of parameters)
    if ".switch_mlp." in path:
        return {"group_size": args.group_size, "bits": args.expert_bits}
    return {"group_size": args.group_size, "bits": args.other_bits}


def quantizable_paths(margs, args) -> dict[str, dict]:
    model = Model(margs)  # lazy
    out = {}
    for path, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if not hasattr(module, "to_quantized"):
            continue
        params = recipe(path, args)
        if params is None:
            continue
        if module.weight.shape[-1] % params["group_size"]:
            print(f"  skip {path}: in-dim {module.weight.shape[-1]} not divisible by {params['group_size']}")
            continue
        out[path] = params
    return out


def materialise(x):
    with mx.stream(mx.cpu):
        mx.eval(x)
    return x


def quantize(w, group_size, bits):
    try:
        out = mx.quantize(materialise(w), group_size=group_size, bits=bits); mx.eval(out); return out
    except RuntimeError as err:
        if "Timeout" not in str(err):
            raise
        with mx.stream(mx.cpu):
            out = mx.quantize(w, group_size=group_size, bits=bits); mx.eval(out); return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--expert-bits", type=int); ap.add_argument("--other-bits", type=int)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--shard-gb", type=float, default=10.0)
    ap.add_argument("--limit-layers", type=int, default=0)
    ap.add_argument("--override", action="append", metavar="SUBSTRING=BITS")
    ap.add_argument("--resume", action="store_true", help="continue from the last clean layer boundary")
    args = ap.parse_args()
    args.expert_bits = args.expert_bits or args.bits
    args.other_bits = args.other_bits or args.bits
    args.override = [(o.split("=")[0], int(o.split("=")[1])) for o in (args.override or [])]

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    raw_cfg = json.load(open(src / "config.json"))
    margs = ModelArgs.from_dict(raw_cfg)
    model = Model(margs)
    qpaths = quantizable_paths(margs, args)
    print(f"quantizable modules: {len(qpaths)} (experts {args.expert_bits}b, other {args.other_bits}b)", flush=True)

    index = json.load(open(src / "model.safetensors.index.json"))["weight_map"]
    n_layers = margs.num_hidden_layers
    groups: dict[str, list[str]] = defaultdict(list)
    for key in index:
        m = re.match(r"model\.layers\.(\d+)\.", key)
        groups[f"layer{int(m.group(1)):03d}" if m else "top"].append(key)
    order = ["top"] + [f"layer{i:03d}" for i in range(args.limit_layers or n_layers)]
    dropped = sorted(g for g in groups if g not in order and not args.limit_layers)
    print(f"groups: {len(order)} (dropping {dropped}: multi-token-prediction layer)", flush=True)

    opened: dict[str, dict] = {}
    def fetch(keys):
        out = {}
        for k in keys:
            shard = index[k]
            if shard not in opened:
                opened[shard] = mx.load(str(src / shard))
            out[k] = opened[shard][k]
        return out

    target = args.shard_gb * 1e9
    out_index, pending, pending_bytes = {}, {}, 0
    out_n = total_out = 0
    counts = {"quantized": 0, "as_stored": 0}
    started = time.time()
    progress_path = dst / "_progress.json"
    first = 0
    if args.resume and progress_path.exists():
        pr = json.load(open(progress_path)); first, out_n, total_out, out_index = pr["next"], pr["out_n"], pr["total_out"], pr["out_index"]
        print(f"resuming at group {first}/{len(order)} ({out_n} shards, {total_out/1e9:.1f} GB written)", flush=True)

    def flush():
        nonlocal pending, pending_bytes, out_n, total_out
        if not pending:
            return
        out_n += 1
        name = f"model-{out_n:05d}.safetensors"
        mx.save_safetensors(str(dst / name), pending, metadata={"format": "mlx"})
        for key in pending:
            out_index[key] = name
        size = (dst / name).stat().st_size; total_out += size
        print(f"  -> {name}  {len(pending)} tensors  {size/1e9:.2f} GB  (total {total_out/1e9:.1f} GB, {time.time()-started:.0f}s)", flush=True)
        pending, pending_bytes = {}, 0

    for gi, g in enumerate(order):
        if gi < first:
            continue
        raw = fetch(groups[g])
        sane = model.sanitize(raw)
        print(f"[{gi+1}/{len(order)}] {g}: {len(raw)} -> {len(sane)} tensors", flush=True)
        for key, value in sane.items():
            module = key.rsplit(".", 1)[0]
            if key.endswith(".weight") and module in qpaths:
                p = qpaths[module]
                w, scales, biases = quantize(value, p["group_size"], p["bits"])
                emit = {module + ".weight": w, module + ".scales": scales, module + ".biases": biases}
                counts["quantized"] += 1
            else:
                emit = {key: materialise(value)}
                counts["as_stored"] += 1
            for k, v in emit.items():
                pending[k] = v; pending_bytes += v.nbytes
            if pending_bytes >= target:
                flush()
        del raw, sane
        opened.clear(); mx.clear_cache()
        flush()  # layer boundaries are clean resume points
        json.dump({"next": gi + 1, "out_n": out_n, "total_out": total_out, "out_index": out_index}, open(progress_path, "w"))

    quant = {"group_size": args.group_size, "bits": args.bits}
    for path, p in qpaths.items():
        if p != {"group_size": args.group_size, "bits": args.bits}:
            quant[path] = p
    cfg_out = dict(raw_cfg); cfg_out.pop("quantization_config", None)
    cfg_out["quantization"] = quant; cfg_out["quantization_config"] = quant
    cfg_out["model_file"] = "glm_moe_dsa.py"
    json.dump(cfg_out, open(dst / "config.json", "w"), indent=2)
    json.dump({"metadata": {"total_size": total_out}, "weight_map": out_index}, open(dst / "model.safetensors.index.json", "w"), indent=2)
    for name in AUX_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    shutil.copy2(Path(__file__).resolve().parents[1] / "glm53_mlx" / "runtime.py", dst / "glm_moe_dsa.py")
    progress_path.unlink(missing_ok=True)
    print(f"\n{out_n} shards, {total_out/1e9:.1f} GB, {(time.time()-started)/60:.1f} min; {counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
