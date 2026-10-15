"""Ground Show-Harness meta actions in the simulator (README section 3).

The executor is Show-Harness's own ``AtomicExec`` (scripts/trajectory/real2sim/
atomic_tokenizer.py), loaded from the checkout next to this package, so one action
means exactly what it means there: ``MV_*`` moves the TCP 2 cm along one robot-base
axis, closed-loop; ``GRASP`` / ``RELEASE`` first wait for the arm to stop, then hold
the gripper command for a fixed number of control steps. Only the backend is ours:
it turns each displacement into one absolute-pose ``env.step`` of the TacClaw IK
environments, holding the episode's starting orientation (no rotation actions).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Protocol

import numpy as np

_ATOMIC_TOKENIZER = (
    Path(__file__).resolve().parents[1] / "Show-Harness" / "scripts" / "trajectory" / "real2sim" / "atomic_tokenizer.py"
)
_spec = importlib.util.spec_from_file_location("show_harness_atomic_tokenizer", _ATOMIC_TOKENIZER)
atomic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(atomic)

MOVE_TOKENS = tuple(atomic.MOVE_DIRS)
GRASP = atomic.GRASP
RELEASE = atomic.RELEASE
TOKENS = MOVE_TOKENS + (GRASP, RELEASE)


class StepAdapter(Protocol):
    gripper_open: float  # this environment's gripper action value for open
    gripper_close: float

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """(position, wxyz quaternion) of tacumi-contact-center-v1 in the robot base frame."""

    def step(self, action: np.ndarray) -> Any:
        """Run exactly one ``env.step`` with this 8-number action and record the frame."""


class EpisodeEnded(Exception):
    """The environment ended the episode in the middle of an action."""

    def __init__(self, result: Any) -> None:
        super().__init__(result.reason)
        self.result = result


class AbsolutePoseBackend:
    """The part of Show-Harness's ``AtomicSimEnv`` that ``AtomicExec`` calls."""

    def __init__(self, adapter: StepAdapter) -> None:
        self.adapter = adapter
        self.hold_quaternion = np.asarray(adapter.tcp_pose()[1], dtype=np.float64)

    def tcp_pos(self) -> np.ndarray:
        return np.asarray(self.adapter.tcp_pose()[0], dtype=np.float64)

    def apply_delta(self, delta_m: np.ndarray, grip_cmd: float, max_cmd_m: float) -> None:
        """One control step toward ``tcp + delta``, ``grip_cmd`` +1 open / -1 close.

        Each axis is capped at ``max_cmd_m``, as in Show-Harness's own backends.
        """
        delta = np.clip(np.asarray(delta_m, dtype=np.float64), -max_cmd_m, max_cmd_m)
        gripper = self.adapter.gripper_open if grip_cmd > 0 else self.adapter.gripper_close
        action = np.concatenate((self.tcp_pos() + delta, self.hold_quaternion, [gripper])).astype(np.float32)
        result = self.adapter.step(action)
        if result.done:
            raise EpisodeEnded(result)

    def gripper_width(self) -> float:
        # AtomicExec only returns this to its caller; the episode loop reads the
        # gripper state from the joints instead.
        return float("nan")


def make_executor(adapter: StepAdapter) -> Any:
    """Show-Harness's executor with its own defaults (2 cm per MV_*, 10 gripper steps),
    aiming every move at the lattice through the starting TCP as its LIBERO runner does."""
    backend = AbsolutePoseBackend(adapter)
    return atomic.AtomicExec(backend, lattice_origin=backend.tcp_pos())


def execute(executor: Any, token: str) -> float | None:
    """Run one action to completion. Returns metres travelled for an ``MV_*``."""
    if token in MOVE_TOKENS:
        return executor.move(token)
    if token == GRASP:
        executor.grasp()
    elif token == RELEASE:
        executor.release()
    else:
        raise ValueError(f"unknown action {token!r}")
    return None
