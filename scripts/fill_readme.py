"""Replace MEASUREMENTS_TABLE (or a previously rendered block) in README.md with the ladder + ppl tables."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import upload as U
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sizes = U.hub_sizes()
parts = []
lad = U.OUT_ROOT / "ladder.npz"
if lad.exists():
    lt, toks, L = U.ladder_table(lad); parts.append(f"Per-layer divergence vs bf16, {toks:,} tokens, {L} layers (relative L2 of the layer output; `fp8` = the FP8 release itself):\n\n{lt}")
res = root / "ppl_results.json"
if res.exists():
    r = json.load(open(res)); parts.append("Perplexity, wikitext-2 test, identical windows, builds that fit 512 GB:\n\n" + U.ppl_table(r, sizes))
rec = root / "recommendation.json"
if rec.exists():
    parts.append(json.load(open(rec))["text"])
block = "<!-- measurements -->\n" + "\n\n".join(parts) + "\n<!-- /measurements -->"
p = root / "README.md"; s = p.read_text()
s = s.replace("MEASUREMENTS_TABLE", block) if "MEASUREMENTS_TABLE" in s else re.sub(r"<!-- measurements -->.*?<!-- /measurements -->", block, s, flags=re.S)
p.write_text(s); print(block)
