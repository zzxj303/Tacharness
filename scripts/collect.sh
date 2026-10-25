#!/usr/bin/env bash
# Collect rollouts for every task of one benchmark (README section 0.2): one Isaac process
# per task, 50 seeds each by default. Safe to re-run: finished seeds are skipped. The JEMM
# server (scripts/jemm_serve.sh) must be running.
# Usage: scripts/collect.sh {tabero|robolab} [output_root] [runner args, e.g. --num-seeds 1]
set -uo pipefail

rollout_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark="$1"
shift
output_root="${1:-outputs/collect}"
[[ $# -gt 0 ]] && shift
cd "${rollout_root}"
mkdir -p "${output_root}/logs"

if [[ "${benchmark}" == tabero ]]; then
  # TASK_ID indexes the suite's task list; libero_tactile lists 6 tasks.
  for suite_tasks in libero_object:10 libero_spatial:10 libero_goal:10 libero_10:10 libero_tactile:6; do
    suite="${suite_tasks%%:*}"
    for ((task_id = 0; task_id < ${suite_tasks##*:}; task_id++)); do
      log="${output_root}/logs/tabero_${suite}_task_$(printf %03d "${task_id}").log"
      scripts/isaac_python.sh -m gpt_rollout.tabero_run --task-suite "${suite}" --task-id "${task_id}" \
        --output-root "${output_root}" "$@" >>"${log}" 2>&1
      echo "[collect] tabero ${suite} task ${task_id} exit $?"
    done
  done
elif [[ "${benchmark}" == robolab ]]; then
  # The 120 benchmark tasks, by Task class name.
  tasks=$(grep -hoE '^class [A-Za-z0-9_]+Task\b' TacHarness/src/providers/RoboLab/robolab/tasks/benchmark/*.py \
    | awk '{print $2}' | sort)
  for task in ${tasks}; do
    scripts/isaac_python.sh -m gpt_rollout.robolab_run --task "${task}" \
      --output-root "${output_root}" "$@" >>"${output_root}/logs/robolab_${task}.log" 2>&1
    echo "[collect] robolab ${task} exit $?"
  done
else
  echo "usage: scripts/collect.sh {tabero|robolab} [output_root] [runner args]" >&2
  exit 64
fi
echo "[collect] ${benchmark} finished"
