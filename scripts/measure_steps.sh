#!/usr/bin/env bash
# First GPT step-count measurement (README section 4.1): one seed per task, no per-frame
# recording. Tabero runs past its own step limit (up to 200 decisions); RoboLab keeps its
# time-out. Safe to re-run: finished seeds are skipped.
# Usage: scripts/measure_steps.sh {libero_object|libero_spatial|robolab} [output_root]
set -uo pipefail

rollout_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
worker="$1"
output_root="${2:-outputs/measure_gpt}"
cd "${rollout_root}"

robolab_tasks=(
  BananaInBowlTask RubiksCubeTask YogurtInBowlTask SmallerObjectButterInBinTask
  LargerObjectRaisinBoxInBinTask FoodPacking1CansTask FruitsOnionTask ThrowAwayAppleTask
  MustardInLeftBinTask CannedFoodInBinTask
)

if [[ "${worker}" == robolab ]]; then
  for task in "${robolab_tasks[@]}"; do
    scripts/isaac_python.sh -m gpt_rollout.robolab_run --task "${task}" --num-seeds 1 \
      --output-root "${output_root}" --no-record
    echo "[measure] robolab ${task} exit $?"
  done
else
  for task_id in 0 1 2 3 4 5 6 7 8 9; do
    scripts/isaac_python.sh -m gpt_rollout.tabero_run --task-suite "${worker}" --task-id "${task_id}" --num-seeds 1 \
      --output-root "${output_root}" --disable-time-out --no-record
    echo "[measure] ${worker} task ${task_id} exit $?"
  done
fi
echo "[measure] ${worker} finished"
