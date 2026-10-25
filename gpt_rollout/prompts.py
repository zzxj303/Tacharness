"""Fixed prompt text and per-call message building (README sections 2 and 3)."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field

import cv2
import numpy as np

from .interpreter import TOKENS


@dataclass
class Observation:
    """The six inputs of README section 2 for one GPT call."""

    instruction: str
    scene_rgb: np.ndarray
    wrist_rgb: np.ndarray
    tactile_rgb: np.ndarray
    position: np.ndarray  # (3,) contact-centre position in the robot base frame, metres
    quaternion: np.ndarray  # (4,) w, x, y, z in the robot base frame
    gripper_closed: bool
    objects: dict[str, np.ndarray] = field(default_factory=dict)  # privileged; never recorded as training data


_SCENE_CAMERA = {
    "robolab": (
        "Image 1 is the scene camera: a fixed camera over the robot's left shoulder, "
        "at about (0.05, 0.57, 0.66) m in the robot base frame, looking right and down at the workspace. "
        "In this image +x points to the left and +y points toward the camera."
    ),
    "tabero": (
        "Image 1 is the scene camera: a fixed camera in front of the robot, "
        "looking back toward the robot and down at the workspace. "
        "In this image +x points toward the camera (down the image) and +y points to the right."
    ),
}
_STEP_SECONDS = {"robolab": "1/15", "tabero": "1/20"}

# README section 2's explanations, shared by every policy that reads the observation as text.
_EXPLANATION_TEMPLATE = """You control a Franka robot arm with a TacClaw two-finger gripper by choosing discrete actions. \
Each time you are called you see the current observation and your recent moves, and you choose ONE action. \
The robot executes it completely, then you are called again with a new observation.

INPUTS
- Task instruction.
- {scene_camera}
- Image 2 is the wrist camera: mounted on the gripper, fisheye lens, looking along the fingers.
- Image 3 is the tactile image of the two finger pads. The top row is the LEFT pad and the bottom row is the RIGHT pad. \
Each row has four panels: INDENT (how deep the pad is pressed), DEFORMATION (pad surface displacement arrows), \
SHEAR (shear arrows), HSV (deformation direction and magnitude as colour). An all-black row means that pad touches nothing.
- Robot state: position x, y, z in metres and orientation quaternion qw, qx, qy, qz of the gripper, \
plus whether the gripper is open or closed. The frame is the robot base frame: origin at the arm base, \
x forward, y to the robot's left, z up. The position is the point midway between the two tactile pads.
- Object positions: name and x, y, z of each object, in the same robot base frame.
- Recent moves, newest first.

ACTIONS
- MV_FWD: move the gripper 2 cm along +x.
- MV_BACK: move the gripper 2 cm along -x.
- MV_RIGHT: move the gripper 2 cm along +y.
- MV_LEFT: move the gripper 2 cm along -y.
- MV_UP: move the gripper 2 cm along +z.
- MV_DOWN: move the gripper 2 cm along -z.
- GRASP: stop the arm, then close the gripper.
- RELEASE: stop the arm, then open the gripper.
The gripper orientation stays fixed. The control step is {step_seconds} second; a move takes as many steps as \
the arm needs to travel the 2 cm. Open, the pads are 79.7 mm apart; closed, they close down to about 58 mm. \
The grip force cannot be adjusted.

HOW TO ACT
- Use the scene camera and the object positions to locate the target; use the wrist camera to fine-align when \
the target is close.
- To pick up an object, with the gripper open: move above the object, descend until the gripper position is at \
the object position, GRASP, then lift with MV_UP.
- GRASP when the object is between the two fingers. An object whose position follows the gripper and is above \
its resting height is being held. If the gripper is closed but the object is not at the gripper, the grasp \
failed: RELEASE and try again.
- To place a held object: lift it clear, move above the destination, lower it until it is just above the \
destination, then RELEASE and move up.
- Avoid choosing the opposite of your most recent move unless you overshot.
"""

_GPT_OUTPUT_TEMPLATE = """
OUTPUT
Return only this JSON object and no other text:
{{"decision": "ONE_ACTION"}}
where ONE_ACTION is one of {tokens}.
"""

# README section 3, one line per action; JEMM sees these as its candidates.
ACTION_DESCRIPTIONS = {
    "MV_FWD": "MV_FWD: move the gripper 2 cm along +x.",
    "MV_BACK": "MV_BACK: move the gripper 2 cm along -x.",
    "MV_LEFT": "MV_LEFT: move the gripper 2 cm along -y.",
    "MV_RIGHT": "MV_RIGHT: move the gripper 2 cm along +y.",
    "MV_UP": "MV_UP: move the gripper 2 cm along +z.",
    "MV_DOWN": "MV_DOWN: move the gripper 2 cm along -z.",
    "GRASP": "GRASP: stop the arm, then close the gripper.",
    "RELEASE": "RELEASE: stop the arm, then open the gripper.",
}
assert set(ACTION_DESCRIPTIONS) == set(TOKENS)

JEMM_QUESTION = "Which ONE action should the robot take next to carry out the task instruction?"


def explanation(benchmark: str) -> str:
    return _EXPLANATION_TEMPLATE.format(
        scene_camera=_SCENE_CAMERA[benchmark],
        step_seconds=_STEP_SECONDS[benchmark],
    )


def system_prompt(benchmark: str) -> str:
    return explanation(benchmark) + _GPT_OUTPUT_TEMPLATE.format(tokens=", ".join(TOKENS))


def jemm_state(benchmark: str, observation: "Observation", recent_moves: list[str]) -> str:
    """JEMM's fixed system prompt cannot carry our explanations, so they lead its State block."""
    return explanation(benchmark).rstrip() + "\n\nCURRENT OBSERVATION\n" + observation_text(observation, recent_moves)


def observation_text(observation: Observation, recent_moves: list[str]) -> str:
    position = ", ".join(f"{value:.4f}" for value in observation.position)
    quaternion = ", ".join(f"{value:.4f}" for value in observation.quaternion)
    lines = [
        f"Task instruction: {observation.instruction}",
        "Robot state:",
        f"  position x, y, z: [{position}]",
        f"  orientation qw, qx, qy, qz: [{quaternion}]",
        f"  gripper: {'closed' if observation.gripper_closed else 'open'}",
        "Object positions (x, y, z):",
    ]
    for name, xyz in observation.objects.items():
        lines.append(f"  {name}: [{', '.join(f'{value:.4f}' for value in xyz)}]")
    lines.append(f"Recent moves, newest first: {', '.join(recent_moves) or 'none'}")
    return "\n".join(lines)


CAMERA_IMAGE_SIZE = 224  # same input size as the existing VLA clients


def resize_with_pad(rgb: np.ndarray, size: int = CAMERA_IMAGE_SIZE) -> np.ndarray:
    """Pad to a square with black borders, then resize, as the VLA clients do."""
    height, width = rgb.shape[:2]
    side = max(height, width)
    top, left = (side - height) // 2, (side - width) // 2
    square = np.zeros((side, side, 3), dtype=rgb.dtype)
    square[top : top + height, left : left + width] = rgb
    return cv2.resize(square, (size, size))


def encode_png(rgb: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("PNG encoding failed")
    return buffer.tobytes()


def _image_part(png: bytes) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")},
    }


def build_messages(benchmark: str, text: str, images: list[bytes]) -> list[dict]:
    """``images`` are PNG bytes in the order scene, wrist, tactile."""
    return [
        {"role": "system", "content": system_prompt(benchmark)},
        {"role": "user", "content": [{"type": "text", "text": text}, *(_image_part(png) for png in images)]},
    ]
