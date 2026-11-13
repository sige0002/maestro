#!/bin/bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
sam3_root=${MAESTRO_SAM3_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/maestro-sam3}
[[ $(git -C "$sam3_root/source" rev-parse HEAD) == f92a5efc4001ebfe9d050d43e61f6fb10e02d73c ]]
export SAM3_CHECKPOINT
SAM3_CHECKPOINT=$("$sam3_root/venv/bin/python" -c 'from huggingface_hub import hf_hub_download; print(hf_hub_download("facebook/sam3.1", "sam3.1_multiplex.pt", revision="daa63191845a41281374e725f4c9e51c7a824460", local_files_only=True))')
exec "$sam3_root/venv/bin/python" "$repo_root/servers/sam3_server/server.py"
