#!/bin/bash
# Native ARM64/Blackwell build of the original Maestro Grounded-SAM service.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
perception_root=${MAESTRO_PERCEPTION_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/maestro-perception}
mkdir -p "$perception_root/models"
if [[ ! -x "$perception_root/venv/bin/python" ]]; then uv venv --python 3.12 "$perception_root/venv"; fi
uv pip install --python "$perception_root/venv/bin/python" torch==2.9.1 torchvision==0.24.1 --index https://download.pytorch.org/whl/cu130
uv pip install --python "$perception_root/venv/bin/python" -r "$repo_root/servers/groundedsam_server/requirements.txt"
if [[ ! -d "$perception_root/gsa/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/IDEA-Research/Grounded-Segment-Anything.git "$perception_root/gsa"
  git -C "$perception_root/gsa" checkout 126abe633ffe333e16e4a0a4e946bc1003caf757
fi
[[ $(git -C "$perception_root/gsa" rev-parse HEAD) == 126abe633ffe333e16e4a0a4e946bc1003caf757 ]]
patch_file="$repo_root/servers/groundedsam_server/groundingdino-scalar-type.patch"
if git -C "$perception_root/gsa" apply --check "$patch_file" 2>/dev/null; then
  git -C "$perception_root/gsa" apply "$patch_file"
else
  git -C "$perception_root/gsa" apply --reverse --check "$patch_file"
fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.1} MAX_JOBS=${MAX_JOBS:-4}
export AM_I_DOCKER=true BUILD_WITH_CUDA=true
uv pip install --python "$perception_root/venv/bin/python" --no-deps --no-build-isolation \
  -e "$perception_root/gsa/GroundingDINO" -e "$perception_root/gsa/segment_anything"
if [[ ! -f "$perception_root/models/groundingdino_swint_ogc.pth" ]]; then
  curl -fL --retry 2 -o "$perception_root/models/groundingdino_swint_ogc.pth" \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
fi
if [[ ! -f "$perception_root/models/sam_vit_h_4b8939.pth" ]]; then
  curl -fL --retry 2 -o "$perception_root/models/sam_vit_h_4b8939.pth" \
    https://huggingface.co/ybelkada/segment-anything/resolve/7790786db131bcdc639f24a915d9f2c331d843ee/checkpoints/sam_vit_h_4b8939.pth
fi
export HF_HOME="$perception_root/huggingface"
"$perception_root/venv/bin/python" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
revision='86b5e0934494bd15c9632b12f734a8a67f723594'
snapshot_download('bert-base-uncased',revision=revision,
                  allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt'])
ref=Path(os.environ['HF_HOME'])/'hub/models--bert-base-uncased/refs/main'
ref.parent.mkdir(parents=True,exist_ok=True)
ref.write_text(revision)
PY
cd "$perception_root/models"
printf '%s\n' \
  '3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799  groundingdino_swint_ogc.pth' \
  'a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e  sam_vit_h_4b8939.pth' | sha256sum -c -
