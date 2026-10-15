"""Print the decision log and the step counts of episodes under an output directory."""
import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
verbose = "-v" in sys.argv[2:]
for calls in sorted(root.rglob("calls.jsonl")):
    rows = [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]
    steps_by_action = defaultdict(list)
    for row in rows:
        steps_by_action[row["decision"]].append(row["frame_count"])
    episode = calls.parents[1] if calls.parent.name == "gpt_calls" else calls.parent  # failed/<episode>/calls.jsonl
    print(episode, f"decisions {len(rows)}, env steps {sum(r['frame_count'] for r in rows)}")
    for token, counts in sorted(steps_by_action.items(), key=lambda item: str(item[0])):
        print(f"  {token}: {len(counts)}x, {sum(counts)} steps, {sum(counts) / len(counts):.1f} per action")
    if verbose:
        for row in rows:
            invalid = [a["invalid"] for a in row.get("attempts", []) if "invalid" in a]
            print(f"    {row['call_index']:3d} @{row['first_frame']:4d} +{row['frame_count']:2d} {row['decision']} {row['travelled_m']} {invalid or ''}")
for results in sorted(root.rglob("results.jsonl")):
    for line in results.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        keys = ("seed", "success", "end_reason", "steps", "max_episode_steps", "success_within_max_episode_steps", "decisions", "gpt_calls", "gpt_seconds", "wall_seconds")
        print(results.parent, {k: row[k] for k in keys if k in row})
