#!/usr/bin/env bash
# Provision the Isaac Sim 5.1 runtime used by the GPT rollout scaffold.
# Follows TacHarness/docs/EXTERNAL_DEPENDENCIES.md; safe to re-run.
set -euo pipefail

rollout_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tacharness="${rollout_root}/TacHarness"
runtime="${tacharness}/.venv-isaacsim51"
python="${runtime}/bin/python"
isaaclab="${tacharness}/thirdparty/IsaacLab"
libero_data="${tacharness}/inputs/assets/vendor/tacbenchhub/external/tabero/datasets/Isaaclab_Libero"
tactile_data="${tacharness}/inputs/assets/vendor/tacbenchhub/external/tabero/datasets/Tactile_Manipulation_Dataset"

# 1. IsaacLab 2.3.2 at the commit pinned by TacHarness.
git -C "${tacharness}" submodule update --init --depth 1 thirdparty/IsaacLab

# 2. Python 3.11 + Isaac Sim 5.1.
[[ -x "${python}" ]] || uv venv --python 3.11 "${runtime}"
uv pip install --python "${python}" \
  --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  "isaacsim[all,extscache]==5.1.0" \
  "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" \
  "numpy==1.26.0" warp-lang huggingface_hub

# 3. IsaacLab with its dependencies, then the TacHarness provider overlays.
for package in isaaclab isaaclab_assets isaaclab_mimic isaaclab_rl isaaclab_tasks; do
  uv pip install --python "${python}" \
    --extra-index-url https://pypi.nvidia.com \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match \
    --editable "${isaaclab}/source/${package}" "numpy==1.26.0" "torch==2.7.0+cu128"
done
uv pip install --python "${python}" --no-deps \
  --editable "${tacharness}/src/providers/LIBERO" \
  --editable "${tacharness}/src/providers/RoboLab"
uv pip install --python "${python}" \
  -r "${rollout_root}/scripts/requirements.txt" "numpy==1.26.0"

# 4. Tabero scene assets (USD only; no demonstrations are used) and the sensor
#    assets its robot module needs at import time.
"${python}" - "${libero_data}" "${tactile_data}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    "NathanWu7/Isaaclab_Libero",
    repo_type="dataset",
    allow_patterns=["USD/**"],
    local_dir=sys.argv[1],
)
snapshot_download(
    "china-sae-robotics/Tactile_Manipulation_Dataset",
    repo_type="dataset",
    local_dir=sys.argv[2],
)
PY

echo "runtime ready: ${python}"
