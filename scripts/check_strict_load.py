"""Strict-load a built checkpoint through stock mlx-lm + the bundled `model_file`, then generate.

`mlx_lm.load` is lenient (strict=False): weights missing from the checkpoint silently stay at their
random initialisation. This is the check that must pass before anything is published.
"""
import sys, time
from pathlib import Path
import mlx.core as mx
from mlx_lm.utils import load_model, load_tokenizer
from mlx_lm import generate
try:
    mx.set_wired_limit(int(470e9))
except Exception as e:
    print("[warn]", e)
path = Path(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
t0 = time.time(); model, config = load_model(path, lazy=True, strict=True, trust_remote_code=True)
print(f"[strict] loaded {type(model).__module__}.{type(model).__name__} in {time.time()-t0:.0f}s: zero missing / zero unexpected tensors", flush=True)
tok = load_tokenizer(path, eos_token_ids=config.get("eos_token_id"))
for p in ["The capital of France is", "Write a Python function that merges overlapping intervals.", "Explain in two sentences why the sky appears blue."]:
    prompt = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
    t0 = time.time(); out = generate(model, tok, prompt=prompt, max_tokens=n, verbose=False)
    print(f"\n=== {p}\n{out}\n[{time.time()-t0:.1f}s, peak {mx.get_peak_memory()/1e9:.0f} GB]", flush=True)
