#!/usr/bin/env bash
# Start a dedicated vision-capable local server; keep model paths outside Git.
set -euo pipefail
: "${LLAMA_SERVER:?Set LLAMA_SERVER to the llama-server executable}"
: "${QWEN_MODEL:?Set QWEN_MODEL to the Qwen GGUF file}"
: "${QWEN_MMPROJ:?Set QWEN_MMPROJ to the matching vision projector GGUF}"
exec "$LLAMA_SERVER" -m "$QWEN_MODEL" --mmproj "$QWEN_MMPROJ" \
  --alias maestro-qwen-vl -ngl 999 -fa on --jinja \
  -c "${VLM_CONTEXT:-16384}" -np 1 --host 127.0.0.1 \
  --port "${VLM_PORT:-18082}" --reasoning off
