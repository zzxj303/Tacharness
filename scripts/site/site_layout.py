"""Where one published episode's files live under an output root (gpt_rollout/*_run.py)."""

from __future__ import annotations

import json
from pathlib import Path

TOKENS = ["MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN", "GRASP", "RELEASE"]
BENCHMARKS = ("tabero", "robolab")


def episode_dirs(root: Path, benchmark: str, source_id: str) -> tuple[Path, Path]:
    """(staging episode dir, Show-Harness rollout dir) for a source id such as
    ``tabero:libero_object:task_007:seed_000`` or ``robolab:BananaInBowlTask:seed_000``."""
    parts = source_id.split(":")
    seed = parts[-1].removeprefix("seed_")
    rel = Path(benchmark, *parts[1:-1])
    return root / rel / f"episode_{seed}", root / "showharness" / rel / f"rollout_{seed}"


def published(root: Path, benchmark: str) -> list[dict]:
    """The benchmark's published episodes, in episode_index order (empty if none)."""
    path = root / "lerobot_v3" / benchmark / "meta" / "published_rollouts.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(rows, key=lambda row: row["episode_index"])
