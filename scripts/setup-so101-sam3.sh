#!/bin/bash
# Pin official SAM 3.1 in an isolated environment; requires approved HF access.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
sam3_root=${MAESTRO_SAM3_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/maestro-sam3}
source_revision=f92a5efc4001ebfe9d050d43e61f6fb10e02d73c
mkdir -p "$sam3_root"
if [[ ! -d "$sam3_root/source/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/facebookresearch/sam3.git "$sam3_root/source"
  git -C "$sam3_root/source" checkout "$source_revision"
fi
[[ $(git -C "$sam3_root/source" rev-parse HEAD) == "$source_revision" ]]
if [[ ! -x "$sam3_root/venv/bin/python" ]]; then uv venv --python 3.12 "$sam3_root/venv"; fi
uv pip install --python "$sam3_root/venv/bin/python" torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$sam3_root/venv/bin/python" -r "$repo_root/servers/sam3_server/requirements.txt"
uv pip install --python "$sam3_root/venv/bin/python" --no-deps -e "$sam3_root/source"
"$sam3_root/venv/bin/python" - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download('facebook/sam3.1','sam3.1_multiplex.pt',revision='daa63191845a41281374e725f4c9e51c7a824460'))
PY
