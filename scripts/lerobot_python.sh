#!/usr/bin/env bash
# Run a command with TacHarness's LeRobot 0.6 runtime, built by
# TacHarness/scripts/setup_lerobot312_runtime.sh (README section 7.1).
# Usage: scripts/lerobot_python.sh scripts/publish_rollout.py --episode-dir ...
set -euo pipefail

rollout_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tacharness="${rollout_root}/TacHarness"
runtime="${TACHARNESS_LEROBOT_ENV:-${rollout_root}/.conda/tacharness-lerobot312}"

# Keep caches and temp files next to the checkout (the system disk may be full).
export TMPDIR="${TMPDIR:-${rollout_root}/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${rollout_root}/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export HF_HUB_OFFLINE=1
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"
# tools/ holds lerobot_v3_rollouts.py, whose episode append publish_rollout.py reuses.
export PYTHONPATH="${tacharness}/src:${tacharness}/tools"

exec "${runtime}/bin/python" "$@"
