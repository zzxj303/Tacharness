#!/usr/bin/env bash
# Collect one seed (0) for every task: Tabero's 46, then RoboLab's 120 benchmark tasks longest
# first, with WORKERS Isaac processes sharing one queue (README section 4.5). A task starts only
# when the disk can hold its staged frames in the worst case (every env step up to the scaled
# time limit, MB_PER_FRAME each; README 4.2 measured about 7MB) on top of what the running tasks
# may still write; staging is deleted once an episode is published. A task that cannot fit even
# with nothing else running is listed in <output_root>/.sched/skipped.
# Safe to re-run: tasks that already have a result for seed 0 are not queued again.
# TASKS limits the run to the listed tasks, queued in the order given, e.g.
# TASKS="BananaInBowlTask libero_object:0".
# Usage: [TASKS="..."] scripts/collect_parallel.sh <output_root> [runner args, e.g. --policy jemm]
set -uo pipefail

rollout_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:?usage: scripts/collect_parallel.sh <output_root> [runner args]}"
shift
workers="${WORKERS:-3}"
mb_per_frame="${MB_PER_FRAME:-7.5}"
safety_gb="${SAFETY_GB:-8}"
cd "${rollout_root}"
state="${out}/.sched"
mkdir -p "${state}/reserved" "${out}/logs"
rm -f "${state}/reserved/"* "${state}/skipped"

# One line per task: benchmark, task (suite:id for Tabero), task dir, worst-case env steps.
python3 - "${out}" >"${state}/queue" <<'PY' || exit 1
import glob, json, os, re, sys
from pathlib import Path

sys.path.insert(0, ".")
from gpt_rollout import TIME_LIMIT_SCALE

out = Path(sys.argv[1])
rows = []
# Tabero: 8 s at 20 Hz, libero_10 16 s (README 0.1).
for suite, count in (("libero_object", 10), ("libero_spatial", 10), ("libero_goal", 10), ("libero_10", 10), ("libero_tactile", 6)):
    for i in range(count):
        native = 320 if suite == "libero_10" else 160
        rows.append(("tabero", f"{suite}:{i}", out / "tabero" / suite / f"task_{i:03d}", int(native * TIME_LIMIT_SCALE)))
# RoboLab: each task's own episode_length_s at 15 Hz.
lengths = {}
for path in glob.glob("TacHarness/src/providers/RoboLab/robolab/tasks/benchmark/*.py"):
    for m in re.finditer(r"^class (\w+Task)\b.*?(?=^class |\Z)", Path(path).read_text(), re.S | re.M):
        lengths[m.group(1)] = float(re.search(r"episode_length_s\s*[:=][^=\n]*?=?\s*([\d.]+)", m.group(0)).group(1))
for task, seconds in sorted(lengths.items(), key=lambda kv: -kv[1]):
    rows.append(("robolab", task, out / "robolab" / task, int(seconds * 15 * TIME_LIMIT_SCALE)))
selected = os.environ.get("TASKS", "").split()
unknown = set(selected) - {task for _, task, _, _ in rows}
if unknown:
    sys.exit(f"unknown TASKS: {' '.join(sorted(unknown))}")
if selected:  # queued in the order given, so the longest can go first
    rows = sorted((row for row in rows if row[1] in selected), key=lambda row: selected.index(row[1]))
for bench, task, task_dir, steps in rows:
    results = task_dir / "results.jsonl"
    if results.is_file() and any(json.loads(l)["seed"] == 0 for l in results.read_text().splitlines() if l.strip()):
        continue
    print(bench, task, task_dir, steps)
PY
echo "$(date '+%F %T') queued $(wc -l <"${state}/queue") tasks, ${workers} workers" >>"${out}/logs/run.log"

need_gb() { awk -v s="$1" -v m="${mb_per_frame}" 'BEGIN { g = s * m / 1024; printf "%d", (g == int(g)) ? g : int(g) + 1 }'; }

# Prints the queue line number of the first task that fits now, or nothing.
pick() {
  local pending=0 gb dir used free n=0 bench task steps
  for r in "${state}/reserved/"*; do
    [[ -f "${r}" ]] || continue
    read -r gb dir <"${r}"
    used=$(du -s -BG "${dir}" 2>/dev/null | cut -f1 | tr -dc 0-9)
    ((pending += gb > ${used:-0} ? gb - ${used:-0} : 0))
  done
  free=$(df --output=avail -BG "${rollout_root}" | tail -1 | tr -dc 0-9)
  while read -r bench task dir steps; do
    ((n += 1))
    if (($(need_gb "${steps}") + safety_gb <= free - pending)); then
      echo "${n}"
      return
    fi
  done <"${state}/queue"
}

worker() {
  local id="$1" n line bench task dir steps rc
  shift
  sleep $((id * 30))  # stagger Isaac start-ups
  exec 9>"${state}/lock"
  while true; do
    flock 9
    if [[ ! -s "${state}/queue" ]]; then
      flock -u 9
      break
    fi
    n=$(pick)
    if [[ -z "${n}" ]]; then
      if ! compgen -G "${state}/reserved/*" >/dev/null; then
        # Nothing running and nothing fits: the disk only fills from here on.
        cat "${state}/queue" >>"${state}/skipped"
        : >"${state}/queue"
        echo "$(date '+%F %T') skipped $(wc -l <"${state}/skipped") tasks that do not fit on disk" >>"${out}/logs/run.log"
      fi
      flock -u 9
      sleep 60
      continue
    fi
    line=$(sed -n "${n}p" "${state}/queue")
    sed -i "${n}d" "${state}/queue"
    read -r bench task dir steps <<<"${line}"
    echo "$(need_gb "${steps}") ${dir}" >"${state}/reserved/${id}"
    flock -u 9

    echo "$(date '+%F %T') worker ${id} start ${bench} ${task}" >>"${out}/logs/run.log"
    if [[ "${bench}" == tabero ]]; then
      scripts/isaac_python.sh -m gpt_rollout.tabero_run --task-suite "${task%%:*}" --task-id "${task##*:}" \
        --num-seeds 1 --output-root "${out}" "$@" \
        >>"${out}/logs/tabero_${task%%:*}_task_$(printf %03d "${task##*:}").log" 2>&1
    else
      scripts/isaac_python.sh -m gpt_rollout.robolab_run --task "${task}" \
        --num-seeds 1 --output-root "${out}" "$@" >>"${out}/logs/robolab_${task}.log" 2>&1
    fi
    rc=$?
    rm -f "${state}/reserved/${id}"
    echo "$(date '+%F %T') worker ${id} exit ${rc} ${bench} ${task}" >>"${out}/logs/run.log"
  done
}

for ((i = 0; i < workers; i++)); do
  worker "${i}" "$@" &
done
wait
echo "$(date '+%F %T') finished" >>"${out}/logs/run.log"
