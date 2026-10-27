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

# Keep runtime caches and temp files next to the checkout (the system disk may be full);
# values already set in the environment win.
export TMPDIR="${TMPDIR:-${rollout_root}/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${rollout_root}/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME}/torchinductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${XDG_CACHE_HOME}/nv/ComputeCache}"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"
# System libraries the host lacks, unpacked by scripts/setup_env.sh (libGLU for the
# MDL SDK that compiles RTX materials; without it camera rendering fails).
if [[ -d "${rollout_root}/.local/lib" ]]; then
  export LD_LIBRARY_PATH="${rollout_root}/.local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Isaac Sim's CUDA interop runs a peer-to-peer test on every visible GPU at startup, and
# fails with out-of-memory on GPUs other processes have filled; expose one GPU (default 0,
# shared with the JEMM server) unless the caller picks.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Franka + TacClaw with the tactile pads' friction raised to static 5 / dynamic 4 (README 5.1);
# both benchmarks load the robot from this path.
export TACUMI_FRANKA_USD_PATH="${TACUMI_FRANKA_USD_PATH:-${rollout_root}/configs/tacclaw_pad_friction.usda}"

# Tabero sensor assets downloaded by scripts/setup_env.sh.
export TACBENCHHUB_TACTILE_ASSETS_DIR="${TACBENCHHUB_TACTILE_ASSETS_DIR:-${tacharness}/inputs/assets/vendor/tacbenchhub/external/tabero/datasets/Tactile_Manipulation_Dataset/assets/data}"

cd "${rollout_root}"
exec "${tacharness}/.venv-isaacsim51/bin/python" "$@"
