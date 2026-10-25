#!/usr/bin/env bash
# Serve JEMM for the rollout policy on one GPU (default GPU 0, int8, port 8790).
# Usage: scripts/jemm_serve.sh [--precision int8|bf16] [--port 8790]
set -euo pipefail

rollout_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Keep caches and temp files next to the checkout (the system disk may be full).
export TMPDIR="${TMPDIR:-${rollout_root}/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${rollout_root}/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME}/torchinductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${XDG_CACHE_HOME}/nv/ComputeCache}"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"
# The weights are local (models/); never reach for the Hub.
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec "${rollout_root}/.venv-jemm/bin/python" -I "${rollout_root}/scripts/jemm_serve.py" "$@"
