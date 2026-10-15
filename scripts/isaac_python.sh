#!/usr/bin/env bash
# Run a command with the Isaac Sim 5.1 runtime and the paths the scaffold needs.
# Usage: scripts/isaac_python.sh -m gpt_rollout.tabero_run --task-suite libero_object --task-id 0 ...
set -euo pipefail

rollout_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tacharness="${rollout_root}/TacHarness"

# The operator reviewed NVIDIA's Omniverse agreement when requesting this runtime.
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${rollout_root}:${tacharness}:${tacharness}/src:${tacharness}/src/tac_infra:${tacharness}/src/providers/RoboLab"
# RoboLab scene assets live in the upstream checkout; the TacHarness copy ships none.
export ROBOLAB_ASSET_DIR="${ROBOLAB_ASSET_DIR:-${rollout_root}/RoboLab/assets}"

# Tabero sensor assets downloaded by scripts/setup_env.sh.
export TACBENCHHUB_TACTILE_ASSETS_DIR="${TACBENCHHUB_TACTILE_ASSETS_DIR:-${tacharness}/inputs/assets/vendor/tacbenchhub/external/tabero/datasets/Tactile_Manipulation_Dataset/assets/data}"

cd "${rollout_root}"
exec "${tacharness}/.venv-isaacsim51/bin/python" "$@"
