"""Greedy generation through stock mlx-lm — a collapse detector, not a quality measure."""
import sys, time
import mlx.core as mx
from mlx_lm import load, generate
try:
    mx.set_wired_limit(int(470e9))
except Exception as e:
    print("[warn]", e)
path = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
t0 = time.time(); model, tok = load(path, lazy=True); print(f"[smoke] loaded in {time.time()-t0:.0f}s", flush=True)
for p in ["The capital of France is", "Write a Python function that merges overlapping intervals.", "Explain in two sentences why the sky appears blue."]:
    prompt = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
    t0 = time.time(); out = generate(model, tok, prompt=prompt, max_tokens=n, verbose=False)
    print(f"\n=== {p}\n{out}\n[{time.time()-t0:.1f}s, peak {mx.get_peak_memory()/1e9:.0f} GB]", flush=True)
