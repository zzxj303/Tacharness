#!/usr/bin/env bash
# Build the rollout viewer (index.html + data.json + media/) for one output root (README section 7.3).
# Usage: scripts/site/build_site.sh <output_root> [site_dir] [title]
#   e.g. scripts/site/build_site.sh outputs/trial_x2 outputs/trial_x2/site "TacClaw 两倍时长试跑"
set -euo pipefail

rollout_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
here="${rollout_root}/scripts/site"
output_root="$(cd "$1" && pwd)"
site="${2:-${output_root}/site}"
title="${3:-TacClaw Rollout 浏览}"

mkdir -p "${site}" "${TMPDIR:-${rollout_root}/tmp}"
work="$(mktemp -d "${TMPDIR:-${rollout_root}/tmp}/site.XXXXXX")"
trap 'rm -rf "${work}"' EXIT

# The prompt is rebuilt with JEMM's own contract code, so it runs in the JEMM runtime;
# the datasets are read with pyarrow in the LeRobot runtime.
"${rollout_root}/.venv-jemm/bin/python" "${here}/build_prompts.py" "${output_root}" "${work}/prompts.json"
"${rollout_root}/scripts/lerobot_python.sh" "${here}/build_site_data.py" "${output_root}" "${work}/prompts.json" "${site}" --title "${title}"
cp "${here}/index.html" "${site}/index.html"
echo "site ready: ${site}"
