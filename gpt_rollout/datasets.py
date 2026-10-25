"""The two datasets every episode feeds (README section 7).

- Decision level: the episode's calls in Show-Harness's rollout layout, the one
  ``RolloutWriter`` in Show-Harness/scripts/trajectory/real2sim/atomic_tokenizer.py writes
  (``agentview/NNNN.png`` + ``wrist/NNNN.png`` + ``actions.jsonl`` + ``metadata.json``).
  Each record is the observation BEFORE its token, so Show-Harness's
  ``rollouts_to_alpaca.py`` reads it unchanged. Added: the tactile mosaic, the policy's
  output and the frames each decision produced. Object positions are left out.
- Frame level: the recorder's per-frame staging, appended to a LeRobot v3 dataset by
  ``scripts/publish_rollout.py`` in TacHarness's LeRobot runtime, which then deletes the
  staged frames (about 7 MB each).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .interpreter import GRASP, MOVE_TOKENS, RELEASE

ROLLOUT_ROOT = Path(__file__).resolve().parents[1]
PUBLISH = [str(ROLLOUT_ROOT / "scripts" / "lerobot_python.sh"), str(ROLLOUT_ROOT / "scripts" / "publish_rollout.py")]
_VIEWS = (("agentview", "scene"), ("wrist", "wrist"), ("tactile", "tactile"))


def token_kind(token: str) -> str:
    """Show-Harness's ``kind`` field (atomic_tokenizer.token_kind)."""
    if token in MOVE_TOKENS:
        return "move"
    if token == GRASP:
        return "grasp"
    if token == RELEASE:
        return "release"
    raise ValueError(f"unknown token {token!r}")


def export_showharness(episode_dir: Path, rollout_dir: Path, metadata: dict[str, Any]) -> int:
    """Write the decision-level rollout; returns its number of records.

    The call images are moved, not copied, out of ``gpt_calls/``; ``calls.jsonl`` stays.
    A call without an action (an invalid GPT output, always the last call) has no record.
    """
    calls_dir = episode_dir / "gpt_calls"
    calls = [json.loads(line) for line in (calls_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines() if line]
    calls = [call for call in calls if call["decision"] is not None]
    shutil.rmtree(rollout_dir, ignore_errors=True)
    for view, _ in _VIEWS:
        (rollout_dir / view).mkdir(parents=True)
    counts: dict[str, int] = {}
    with (rollout_dir / "actions.jsonl").open("w", encoding="utf-8") as out:
        for step, call in enumerate(calls):
            if call["call_index"] != step:
                raise ValueError(f"{episode_dir}: call {call['call_index']} recorded as step {step}")
            name = f"{step:04d}.png"
            for view, image in _VIEWS:
                shutil.move(str(calls_dir / call["input_images"][image]), str(rollout_dir / view / name))
            observed = call["observation"]
            token = call["decision"]
            counts[token] = counts.get(token, 0) + 1
            record = {
                # Show-Harness's fields, in its order.
                "step": step,
                "token": token,
                "kind": token_kind(token),
                "gripper_closed": observed["gripper_closed"],
                "ee_pose": observed["position"] + observed["quaternion"],
                "gripper_width": round(float(observed["gripper_width_m"]), 5),
                "agentview": f"agentview/{name}",
                "wrist": f"wrist/{name}",
                "time": observed["time"],
                # Ours.
                "tactile": f"tactile/{name}",
                "policy_output": call["attempts"],
                "first_frame": call["first_frame"],
                "frame_count": call["frame_count"],
                "travelled_m": call["travelled_m"],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    # RolloutWriter.close adds the same two fields.
    meta = {**metadata, "num_steps": len(calls), "token_counts": counts}
    (rollout_dir / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return len(calls)


def publish_frames(
    episode_dir: Path,
    dataset_root: Path,
    *,
    benchmark: str,
    source_id: str,
    task: str,
    seed: int,
    success: bool,
) -> dict[str, Any]:
    """Append the staged frames to the benchmark's LeRobot v3 dataset, verify, delete them."""
    command = PUBLISH + [
        "--episode-dir", str(episode_dir),
        "--dataset-root", str(dataset_root),
        "--benchmark", benchmark,
        "--source-id", source_id,
        "--task", task,
        "--seed", str(seed),
        "--success", str(int(success)),
    ]  # fmt: skip
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"publish_rollout.py failed ({completed.returncode}):\n{completed.stderr[-4000:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def export_episode(
    episode_dir: Path,
    *,
    rollout_dir: Path,
    dataset_root: Path,
    benchmark: str,
    source_id: str,
    task: str,
    summary: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Both datasets for one finished episode. Run before the episode's results.jsonl line."""
    records = export_showharness(episode_dir, rollout_dir, {**metadata, **summary})
    published = publish_frames(
        episode_dir,
        dataset_root,
        benchmark=benchmark,
        source_id=source_id,
        task=task,
        seed=int(summary["seed"]),
        success=bool(summary["success"]),
    )
    return {"showharness_rollout": str(rollout_dir), "showharness_steps": records, "lerobot_v3": published}
