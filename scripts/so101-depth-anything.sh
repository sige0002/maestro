#!/bin/bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
perception_root=${MAESTRO_PERCEPTION_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/maestro-perception}
export HF_HOME="$perception_root/huggingface"
exec "$perception_root/venv/bin/python" "$repo_root/servers/depth_anything_server/server.py"
