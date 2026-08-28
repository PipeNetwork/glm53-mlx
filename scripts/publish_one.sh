#!/bin/sh
# publish_one.sh <build> [--delete]  — upload a build (card rendered from current measurements); optionally delete the local copy after a verified upload.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; OUT=/Users/david/llm/glm53-out; b=$1; PY="$ROOT/.venv/bin/python"
echo "=== upload $b $(date)"
$PY "$ROOT/scripts/upload.py" --dir "$OUT/$b" --repo "pipenetwork/$b" --yes || { echo "UPLOAD FAILED: $b"; exit 1; }
local_gb=$($PY -c "import os; d='$OUT/$b'; print(round(sum(os.path.getsize(os.path.join(d,f)) for f in os.listdir(d) if os.path.isfile(os.path.join(d,f)))/1e9,1))")
hub_gb=$($PY -c "from huggingface_hub import HfApi; i=HfApi().model_info('pipenetwork/$b', files_metadata=True); print(round(sum((s.size or 0) for s in i.siblings)/1e9,1))")
echo "local $local_gb GB, hub $hub_gb GB"
if [ "${2:-}" = "--delete" ]; then
  if [ "$local_gb" = "$hub_gb" ]; then rm -rf "$OUT/$b" && echo "deleted local copy of $b"; else echo "SIZE MISMATCH, keeping local copy"; exit 1; fi
fi
