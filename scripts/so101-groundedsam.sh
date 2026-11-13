#!/bin/bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
perception_root=${MAESTRO_PERCEPTION_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/maestro-perception}
export HF_HOME="$perception_root/huggingface" HF_HUB_OFFLINE=1
export GROUNDINGDINO_CKPT="$perception_root/models/groundingdino_swint_ogc.pth"
export SAM_CKPT="$perception_root/models/sam_vit_h_4b8939.pth"
export GROUNDEDSAM_HOST=${GROUNDEDSAM_HOST:-127.0.0.1}
export GROUNDEDSAM_PORT=${GROUNDEDSAM_PORT:-8005}
cd "$perception_root/models"
printf '%s\n' \
  '3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799  groundingdino_swint_ogc.pth' \
  'a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e  sam_vit_h_4b8939.pth' | sha256sum -c -
exec "$perception_root/venv/bin/python" "$repo_root/servers/groundedsam_server/server.py"
