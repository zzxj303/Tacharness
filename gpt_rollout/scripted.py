"""A scripted stand-in for GPT: one pick-and-place, decided from the privileged
object positions in the same observation GPT receives.

Used to run the interpreter end to end without the GPT endpoint, and to see how many
control steps an error-free action sequence needs. Not a data source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .episode import Decision
from .interpreter import GRASP, RELEASE, atomic
from .prompts import Observation, observation_text

ALIGN_TOL_M = 0.011  # just over half an action, as Show-Harness's planners use
HOLD_RADIUS_M = 0.05  # object root this close to the TCP while closed -> held
PLACED_RADIUS_M = 0.05  # object root this close (xy) to the destination -> placed


def _toward_xy(error: np.ndarray) -> str:
    """The move along the horizontal axis with the larger error (Show-Harness's Manhattan rule)."""
    axis = int(np.argmax(np.abs(error[:2])))
    return atomic.token_for_axis(axis, error[axis])


class ScriptedPolicy:
    def __init__(self, obj: str, target: str, hover_m: float = 0.12, place_m: float = 0.15, grasp_dz: float = 0.0) -> None:
        self.obj = obj
        self.target = target
        self.hover_m = hover_m  # approach height above the object's root
        self.place_m = place_m  # release height above the destination's root
        self.grasp_dz = grasp_dz  # grasp height above the object's root

    def begin_episode(self, benchmark: str, calls_dir: Path, seed: int) -> None:
        self.calls = 0

    def decide(self, observation: Observation, recent_moves: list[str], call_index: int) -> Decision:
        self.calls += 1
        return Decision(self._token(observation), {"input_text": observation_text(observation, recent_moves)})

    def _token(self, observation: Observation) -> str:
        tcp = np.asarray(observation.position)
        obj = np.asarray(observation.objects[self.obj])
        target = np.asarray(observation.objects[self.target])
        if observation.gripper_closed:
            # Held: the object root stays near where it was below the TCP at the grasp.
            if np.linalg.norm(obj - tcp + [0.0, 0.0, self.grasp_dz]) > HOLD_RADIUS_M:
                return RELEASE  # empty or lost grasp
            carry_z = target[2] + self.place_m + 0.04
            error = target - tcp
            if np.linalg.norm(error[:2]) > ALIGN_TOL_M:
                return "MV_UP" if tcp[2] < carry_z - ALIGN_TOL_M else _toward_xy(error)
            if tcp[2] > target[2] + self.place_m + ALIGN_TOL_M:
                return "MV_DOWN"
            return RELEASE
        if np.linalg.norm((obj - target)[:2]) < PLACED_RADIUS_M:
            return "MV_UP"  # already placed: clear the object
        error = obj - tcp
        if np.linalg.norm(error[:2]) > ALIGN_TOL_M:
            return "MV_UP" if tcp[2] < obj[2] + self.hover_m - ALIGN_TOL_M else _toward_xy(error)
        if tcp[2] > obj[2] + self.grasp_dz + ALIGN_TOL_M:
            return "MV_DOWN"
        return GRASP

    def summary(self) -> dict[str, Any]:
        return {"policy": "scripted", "policy_target": [self.obj, self.target], "gpt_calls": 0}
