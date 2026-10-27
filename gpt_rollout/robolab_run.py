"""Collect rollouts for one RoboLab benchmark task (one Isaac process per task).

Run through scripts/isaac_python.sh, which sets the Isaac runtime and paths. With
--policy gpt (the default, GPT-6 astra from configs/llm.yaml) configs/llm.yaml must exist;
with --policy jemm the JEMM server (scripts/jemm_serve.sh) must be running.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from gpt_rollout import TIME_LIMIT_SCALE

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True, help="Task class name, e.g. BananaInBowlTask")
parser.add_argument("--seed-start", type=int, default=0)
parser.add_argument("--num-seeds", type=int, default=50)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--policy", choices=("jemm", "gpt", "scripted"), default="gpt")
parser.add_argument("--jemm-endpoint", default="http://127.0.0.1:8790")
parser.add_argument(
    "--time-limit-scale",
    type=float,
    default=TIME_LIMIT_SCALE,
    help="lengthen the task's own episode time limit by this factor (README 0.1); 1 keeps the original",
)
parser.add_argument(
    "--max-decisions", type=int, default=None, help="decisions per episode; default none, the time limit ends the episode"
)
parser.add_argument("--scripted-object", help="scripted policy: object to pick, e.g. banana")
parser.add_argument("--scripted-target", help="scripted policy: where to place it, e.g. bowl")
parser.add_argument("--no-record", action="store_true", help="measurement only: keep the GPT call log and results, skip the per-frame data")
parser.add_argument("--scripted-place-m", type=float, default=0.12, help="scripted policy: release height above the target")
args = parser.parse_args()

import cv2  # noqa: E402, F401 -- must import this before isaaclab
from isaaclab.app import AppLauncher  # noqa: E402
from tacumi.isaac_plugins import ensure_lens_distortion_plugin_path  # noqa: E402

# The TacClaw wrist camera's fisheye schema must be registered before the app starts.
ensure_lens_distortion_plugin_path()
simulation_app = AppLauncher(headless=True, enable_cameras=True, device=args.device).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.world.world_state import get_world  # noqa: E402
from robolab.eval.episode import _capture_tacumi_frame, _tactile_frame_ready  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_abs_ik import (  # noqa: E402
    auto_register_droid_abs_ik_envs,
)
from tacumi.recorder import UnifiedEpisodeRecorder  # noqa: E402
from tacumi.tactile_runtime import DEFAULT_CONFIG, TacumiTactileRuntime  # noqa: E402

from gpt_rollout.actions import ACTION_LAYOUT  # noqa: E402
from gpt_rollout.datasets import export_episode  # noqa: E402
from gpt_rollout.episode import GPTPolicy, StepResult, finish_episode, load_results, run_episode  # noqa: E402
from gpt_rollout.interpreter import STEP_M  # noqa: E402
from gpt_rollout.jemm import JEMMClient, JEMMPolicy  # noqa: E402
from gpt_rollout.llm import LLMClient  # noqa: E402
from gpt_rollout.prompts import Observation  # noqa: E402
from gpt_rollout.scripted import ScriptedPolicy  # noqa: E402
from gpt_rollout.sim import (  # noqa: E402
    contact_center_pose,
    gripper_closed,
    gripper_width_m,
    positions_in_base_frame,
    runtime_metadata,
    scale_time_limit,
    seed_everything,
    tactile_composite,
    to_numpy,
    uint8_rgb,
)

SCENE_CAMERA = "over_shoulder_left_camera"
WRIST_CAMERA = "wrist_cam"
STATE_LAYOUT = [f"panda_joint{i}" for i in range(1, 8)] + [
    "gripper_close_ratio",
    "eef_x_m",
    "eef_y_m",
    "eef_z_m",
    "eef_qw",
    "eef_qx",
    "eef_qy",
    "eef_qz",
]


class RoboLabAdapter:
    benchmark = "robolab"
    # BinaryJointPositionZeroToOneAction: a value above 0.5 closes.
    gripper_open = 0.0
    gripper_close = 1.0

    def __init__(self, env, runtime, recorder, instruction: str, obs, tactile) -> None:
        self.env = env
        self.runtime = runtime
        self.recorder = recorder
        self.instruction = instruction
        self.obs = obs
        self.tactile = tactile
        self.total_reward = 0.0
        self.steps = 0

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return contact_center_pose(self.env)

    def gripper_width_m(self) -> float:
        return gripper_width_m(self.env.scene["robot"])

    def observe(self) -> Observation:
        robot = self.env.scene["robot"]
        world = get_world(self.env)
        position, quaternion = contact_center_pose(self.env)
        return Observation(
            instruction=self.instruction,
            scene_rgb=uint8_rgb(self.obs["image_obs"][SCENE_CAMERA][0]),
            wrist_rgb=uint8_rgb(self.obs["image_obs"][WRIST_CAMERA][0]),
            tactile_rgb=tactile_composite(self.tactile),
            position=position,
            quaternion=quaternion,
            gripper_closed=gripper_closed(robot),
            objects=positions_in_base_frame(
                robot,
                {name: world.get_pose(name, is_relative=False, env_id=0)[0] for name in world.objects},
            ),
        )

    def step(self, action: np.ndarray) -> StepResult:
        action_tensor = torch.from_numpy(action).to(self.env.device).reshape(1, -1)
        self.obs, reward, _, _, _ = self.env.step(action_tensor)
        self.steps += 1
        result = self.env.get_env_results()[0]["success"]
        self.tactile = _capture_tacumi_frame(self.env, self.runtime)
        reward_value = float(to_numpy(reward).reshape(-1)[0])
        self.total_reward += reward_value
        if self.recorder is not None:
            proprio = self.obs["proprio_obs"]
            position, quaternion = contact_center_pose(self.env)
            self.recorder.add_frame(
                state=np.concatenate(
                    (
                        to_numpy(proprio["arm_joint_pos"])[0].reshape(-1)[:7],
                        to_numpy(proprio["gripper_pos"])[0].reshape(-1)[:1],
                        position,
                        quaternion,
                    )
                ).astype(np.float32),
                action=action,
                wrist_rgb=uint8_rgb(self.obs["image_obs"][WRIST_CAMERA][0]),
                scene_rgb=uint8_rgb(self.obs["image_obs"][SCENE_CAMERA][0]),
                tactile=self.tactile,
                reward=reward_value,
                success=bool(result),
            )
        # RobolabEnv freezes the env when it terminates: success unless it timed out.
        if self.env.all_terminated:
            return StepResult(done=True, success=bool(result), reason="success" if result else "time_out")
        return StepResult(done=False)


def warm_up(env, runtime):
    """Hold the current pose until the tactile cameras produce frames, as run_episode does."""
    for _ in range(3):
        position, quaternion = contact_center_pose(env)
        hold = torch.zeros(1, 8, device=env.device)
        hold[0, :3] = torch.from_numpy(position)
        hold[0, 3:7] = torch.from_numpy(quaternion)
        obs, _, _, _, _ = env.step(hold)
        tactile = _capture_tacumi_frame(env, runtime)
        if _tactile_frame_ready(tactile):
            break
    else:
        raise RuntimeError("Tacumi tactile sensors did not produce warm-up frames")
    if env.recorder_manager is not None and hasattr(env.recorder_manager, "_env"):
        env.recorder_manager.clear()
    return obs, tactile


def main() -> None:
    auto_register_droid_abs_ik_envs(task=[args.task])
    (env_name,) = get_envs(task=[args.task])
    task_dir = args.output_root / "robolab" / args.task
    task_dir.mkdir(parents=True, exist_ok=True)
    set_output_dir(str(task_dir / "robolab_native"))
    os.makedirs(task_dir / "robolab_native", exist_ok=True)
    results_path = task_dir / "results.jsonl"
    done = load_results(results_path)

    env, env_cfg = create_env(env_name, device=args.device, num_envs=1, policy="gpt")
    native_steps = scale_time_limit(env, args.time_limit_scale)
    instruction = env_cfg.instruction
    fps = 1.0 / (env_cfg.sim.render_interval * env_cfg.sim.dt)
    runtime = TacumiTactileRuntime(
        config_path=os.environ.get("TACUMI_TACTILE_CONFIG", str(DEFAULT_CONFIG)),
        root_prefix="/World/envs/env_0/robot/Gripper/Tacumi",
        expected_sensors=2,
        headless=True,
    )
    if args.policy == "jemm":
        client = JEMMClient(args.jemm_endpoint)
        client.wait_ready()
        policy = JEMMPolicy(client)
    elif args.policy == "gpt":
        policy = GPTPolicy(LLMClient())
    else:
        policy = ScriptedPolicy(args.scripted_object, args.scripted_target, place_m=args.scripted_place_m)
    meta = {
        "benchmark": "robolab",
        "task": args.task,
        "instruction": instruction,
        "max_episode_steps": native_steps,
        "time_limit_scale": args.time_limit_scale,
        "episode_step_limit": int(env.max_episode_length),
        **runtime_metadata(env_name),
    }

    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        if seed in done:
            continue
        seed_everything(env, seed)
        env.reset()
        env.reset()
        obs, tactile = warm_up(env, runtime)
        if env.recorder_manager is not None and hasattr(env.recorder_manager, "set_hdf5_file"):
            env.recorder_manager.set_hdf5_file(f"run_{seed}.hdf5")
            env.recorder_manager.set_episode_index(0, env_ids=[0])
        episode_dir = task_dir / f"episode_{seed:03d}"
        recorder = None if args.no_record else UnifiedEpisodeRecorder(
            episode_dir,
            source="robolab",
            task_id=f"robolab:{args.task}",
            instruction=instruction,
            policy_contract="robolab_droid_gpt_taskspace_absolute_v1",
            state_layout=STATE_LAYOUT,
            action_layout=ACTION_LAYOUT,
            fps=fps,
            record_scene_rgb=True,
            record_rgb_dual_view=True,
            overwrite=True,
        )
        adapter = RoboLabAdapter(env, runtime, recorder, instruction, obs, tactile)
        summary = {"seed": seed, **meta, **run_episode(adapter, policy, episode_dir, args.max_decisions, seed)}
        summary["success_within_max_episode_steps"] = summary["success"] and summary["steps"] <= native_steps
        if recorder is not None:
            recorder.close(success=summary["success"], total_reward=adapter.total_reward, extra=summary)
            if args.policy != "scripted":
                summary["datasets"] = export_episode(
                    episode_dir,
                    rollout_dir=args.output_root / "showharness" / "robolab" / args.task / f"rollout_{seed:03d}",
                    dataset_root=args.output_root / "lerobot_v3" / "robolab",
                    benchmark="robolab",
                    source_id=f"robolab:{args.task}:seed_{seed:03d}",
                    task=instruction,
                    summary=summary,
                    metadata={"task": args.task, "task_text": instruction, "step_m": STEP_M},
                )
        finish_episode(results_path, summary)
        print(f"[gpt_rollout] {args.task} seed {seed}: " + json.dumps(summary), flush=True)
        env.reset_eval_state()

    env.close()


if __name__ == "__main__":
    # simulation_app.close() ends the process with status 0 and can hang after an
    # error, so report failures and exit directly.
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
