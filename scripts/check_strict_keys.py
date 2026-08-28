"""Strict, lazy load through stock mlx-lm + `model_file`: proves the checkpoint's tensors match the
model's parameters exactly (zero missing, zero unexpected) without materialising weights — the check
for builds too large to run on this machine."""
import sys, time
from pathlib import Path
from mlx_lm.utils import load_model
t0 = time.time(); model, config = load_model(Path(sys.argv[1]), lazy=True, strict=True, trust_remote_code=True)
print(f"[strict-keys] {sys.argv[1]}: OK, zero missing / zero unexpected ({time.time()-t0:.0f}s)")
