#!/usr/bin/env bash
# Build the rollout viewer (index.html + data.json + media/) for one output root (README section 7.3).
# Usage: scripts/site/build_site.sh <output_root> [site_dir] [title] [case_fragment.html]
#   e.g. scripts/site/build_site.sh outputs/trial_x2 outputs/trial_x2/site "TacClaw 两倍时长试跑"
# case_fragment.html, if given, is shown as the page's case-analysis section (scripts/site/cases/).
set -euo pipefail

rollout_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
here="${rollout_root}/scripts/site"
output_root="$(cd "$1" && pwd)"
site="${2:-${output_root}/site}"
title="${3:-TacClaw Rollout 浏览}"
case_args=()
if [[ -n "${4:-}" ]]; then
  case_args=(--case "$(cd "$(dirname "$4")" && pwd)/$(basename "$4")")
fi

mkdir -p "${site}" "${TMPDIR:-${rollout_root}/tmp}"
work="$(mktemp -d "${TMPDIR:-${rollout_root}/tmp}/site.XXXXXX")"
trap 'rm -rf "${work}"' EXIT

# A JEMM prompt is rebuilt with JEMM's own contract code, so it needs the JEMM runtime; a GPT
# prompt is read from the call log with any Python. The datasets are read with pyarrow in the
# LeRobot runtime.
prompt_python=python3
if [[ -x "${rollout_root}/.venv-jemm/bin/python" ]]; then
  prompt_python="${rollout_root}/.venv-jemm/bin/python"
fi
"${prompt_python}" "${here}/build_prompts.py" "${output_root}" "${work}/prompts.json"
"${rollout_root}/scripts/lerobot_python.sh" "${here}/build_site_data.py" "${output_root}" "${work}/prompts.json" "${site}" \
  --title "${title}" "${case_args[@]}"
# Line 1 of the template is its <title>; the page's name in the browser and the artifact gallery.
{ printf '<title>%s</title>\n' "${title}"; sed 1d "${here}/index.html"; } >"${site}/index.html"
echo "site ready: ${site}"
