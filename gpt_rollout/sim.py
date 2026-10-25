"""Simulator-side helpers shared by the RoboLab and Tabero runners.

Import only after ``AppLauncher`` has started Isaac Sim.
"""

from __future__ import annotations

import importlib.metadata as metadata
import random
from typing import Any

import numpy as np
import torch
from isaaclab.utils.math import subtract_frame_transforms
from tacumi.embodiment import (
    TACUMI_CLOSE_ANGLE_RAD,
    TACUMI_DEFAULT_WRIST_CLUSTER,
    TACUMI_GRIPPER_JOINT_NAMES,
    TACUMI_POLICY_TCP_ID,
    TACUMI_WRIST_CAMERA_CONTRACT_260808_MAJORITY,
)
from tacumi.recorder import canonicalize_tactile_sample
from tacumi.visualization import render_tactile_composite

# Tabero's own open/closed test: |joint - open| above cfg.gripper_threshold.
GRIPPER_CLOSED_THRESHOLD_RAD = 0.05
# Pad separation at the two known joint angles (README section 5.1): 79.7 mm at 0 deg,
# about 58 mm at the default close angle. gripper_width_m interpolates linearly between
# them, so it is an estimate.
OPEN_PAD_GAP_M = 0.0797
CLOSED_PAD_GAP_M = 0.058


def seed_everything(env: Any, seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def uint8_rgb(value: Any) -> np.ndarray:
    array = to_numpy(value)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0.0, 255.0)
    return np.ascontiguousarray(array[..., :3].astype(np.uint8))


def tactile_composite(tactile: dict[str, dict[str, Any]]) -> np.ndarray:
    """The same left/right x INDENT/DEFORMATION/SHEAR/HSV image the recorder stores."""
    return render_tactile_composite(
        {side: canonicalize_tactile_sample(tactile[side]) for side in ("left", "right")}
    )


def contact_center_pose(env: Any) -> tuple[np.ndarray, np.ndarray]:
    """Pose the absolute IK action tracks: tacumi-contact-center-v1 in the robot base frame."""
    position, quaternion = env.action_manager.get_term("arm_action")._compute_frame_pose()
    return to_numpy(position)[0], to_numpy(quaternion)[0]


def gripper_closed(robot: Any) -> bool:
    joint_ids, _ = robot.find_joints(list(TACUMI_GRIPPER_JOINT_NAMES))
    joint_pos = to_numpy(robot.data.joint_pos)[0, joint_ids]
    return bool(np.max(np.abs(joint_pos)) > GRIPPER_CLOSED_THRESHOLD_RAD)


def scale_time_limit(env: Any, scale: float) -> int:
    """Lengthen the task's own episode time limit by ``scale`` (README 0.1 allows this when
    collecting) and return the original limit in env steps. IsaacLab reads
    ``cfg.episode_length_s`` afresh for every time-out check, so this applies after creation."""
    original_steps = int(env.max_episode_length)
    env.cfg.episode_length_s *= scale
    return original_steps


def gripper_width_m(robot: Any) -> float:
    """Estimated opening between the pads, from the mean |angle| of the two finger joints."""
    joint_ids, _ = robot.find_joints(list(TACUMI_GRIPPER_JOINT_NAMES))
    angle = float(np.mean(np.abs(to_numpy(robot.data.joint_pos)[0, joint_ids])))
    return OPEN_PAD_GAP_M - (OPEN_PAD_GAP_M - CLOSED_PAD_GAP_M) * angle / TACUMI_CLOSE_ANGLE_RAD


def positions_in_base_frame(robot: Any, positions_w: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """World-frame root positions -> robot base frame."""
    root_pos = robot.data.root_pos_w[0]
    root_quat = robot.data.root_quat_w[0]
    result = {}
    for name, position_w in positions_w.items():
        position_b, _ = subtract_frame_transforms(root_pos, root_quat, position_w.reshape(3).to(root_pos))
        result[name] = to_numpy(position_b)
    return result


def runtime_metadata(env_id: str) -> dict[str, Any]:
    return {
        "env_id": env_id,
        "isaacsim_version": metadata.version("isaacsim"),
        "isaaclab_version": metadata.version("isaaclab"),
        "policy_tcp": TACUMI_POLICY_TCP_ID,
        "wrist_camera_cluster": TACUMI_DEFAULT_WRIST_CLUSTER,
        "wrist_camera_contract": TACUMI_WRIST_CAMERA_CONTRACT_260808_MAJORITY,
        "gripper_close_angle_deg": float(np.rad2deg(TACUMI_CLOSE_ANGLE_RAD)),
    }
