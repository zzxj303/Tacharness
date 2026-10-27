"""Collect rollouts for one Tabero task (one Isaac process per task).

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

from gpt_rollout import MAX_DECISIONS, TIME_LIMIT_SCALE

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task-suite", required=True)
parser.add_argument("--task-id", type=int, required=True)
parser.add_argument("--seed-start", type=int, default=0)
parser.add_argument("--num-seeds", type=int, default=50)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--device", default="cuda")
parser.add_argument("--policy", choices=("jemm", "gpt", "scripted"), default="gpt")
parser.add_argument("--jemm-endpoint", default="http://127.0.0.1:8790")
parser.add_argument(
    "--time-limit-scale",
    type=float,
    default=TIME_LIMIT_SCALE,
    help="lengthen the task's own episode time limit by this factor (README 0.1); 1 keeps the original",
)
parser.add_argument(
    "--max-decisions",
    type=int,
    default=None,
    help="decisions per episode; default none, the time limit ends the episode (200 with --disable-time-out)",
)
parser.add_argument(
    "--disable-time-out",
    action="store_true",
    help="measurement only: run past the task's own step limit (still recorded) until success or --max-decisions",
)
parser.add_argument("--no-record", action="store_true", help="measurement only: keep the GPT call log and results, skip the per-frame data")
parser.add_argument("--scripted-place-m", type=float, default=0.15, help="scripted policy: release height above the target")
parser.add_argument("--scripted-grasp-dz", type=float, default=0.0, help="scripted policy: grasp height above the object's root")
args = parser.parse_args()
if args.disable_time_out and args.max_decisions is None:
    args.max_decisions = MAX_DECISIONS  # nothing else would end an unsuccessful episode

ENV_ID = "Isaac-Libero-Tacumi-IK-Camera-ContactForce-v0"
NUM_SUCCESS_STEPS = 8  # openpi_inference_client.py default, part of the success rule

# The task config reads these at import time. Absolute pose actions, default reset.
os.environ["TASK_SUITE"] = args.task_suite
os.environ["TASK_ID"] = str(args.task_id)
os.environ["USE_RELATIVE_MODE"] = "False"

from isaaclab.app import AppLauncher  # noqa: E402
from tacumi.isaac_plugins import ensure_lens_distortion_plugin_path  # noqa: E402

# The TacClaw wrist camera's fisheye schema must be registered before the app starts.
ensure_lens_distortion_plugin_path()
simulation_app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import tac_manip.tasks  # noqa: E402, F401
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from tacumi.tabero_rollout import TaberoTacumiRolloutHook  # noqa: E402

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
    uint8_rgb,
)


class TaberoAdapter:
    benchmark = "tabero"
    # AbsBinaryJointPositionAction with positive_threshold: a value above 0.5 opens.
    gripper_open = 1.0
    gripper_close = 0.0

    def __init__(self, env, hook, success_term, instruction: str, object_names: list[str]) -> None:
        self.env = env
        self.hook = hook
        self.success_term = success_term
        self.instruction = instruction
        self.object_names = object_names
        self.success_step_count = 0
        self.steps = 0

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return contact_center_pose(self.env)

    def gripper_width_m(self) -> float:
        return gripper_width_m(self.env.scene["robot"])

    def observe(self) -> Observation:
        scene = self.env.scene
        position, quaternion = contact_center_pose(self.env)
        return Observation(
            instruction=self.instruction,
            scene_rgb=uint8_rgb(scene["agentview_cam"].data.output["rgb"][0]),
            wrist_rgb=uint8_rgb(scene["eye_in_hand_cam"].data.output["rgb"][0]),
            tactile_rgb=tactile_composite(self.hook.last_tactile),
            position=position,
            quaternion=quaternion,
            gripper_closed=gripper_closed(scene["robot"]),
            objects=positions_in_base_frame(
                scene["robot"],
                {name: scene[name].data.root_pos_w[0] for name in self.object_names},
            ),
        )

    def step(self, action: np.ndarray) -> StepResult:
        action_tensor = torch.from_numpy(action).to(self.env.device).reshape(1, -1)
        obs, reward, terminated, truncated, _ = self.env.step(action_tensor)
        self.steps += 1
        # Same counting rule as openpi_inference_client.py, checked every env step.
        success = False
        native_success = bool(self.success_term.func(self.env, **self.success_term.params)[0])
        if native_success:
            self.success_step_count += 1
            success = bool(terminated[0]) or self.success_step_count >= NUM_SUCCESS_STEPS
        else:
            self.success_step_count = 0
        self.hook.record_step(obs, action_tensor[0], reward, native_success)
        if success:
            return StepResult(done=True, success=True, reason="success")
        if bool(truncated[0]):
            return StepResult(done=True, reason="time_out")
        if bool(terminated[0]):
            return StepResult(done=True, reason="object_dropped")
        return StepResult(done=False)


def main() -> None:
    env_cfg = parse_env_cfg(ENV_ID, device=args.device, num_envs=1)
    env_cfg.env_name = ENV_ID
    env_cfg.recorders = {}
    # Success is counted over NUM_SUCCESS_STEPS consecutive steps below, as the
    # evaluation script does. The object-dropped termination stays on; so does
    # time_out, unless this is a --disable-time-out measurement run.
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
    if args.disable_time_out:
        env_cfg.terminations.time_out = None
    env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
    native_steps = scale_time_limit(env, args.time_limit_scale)

    task_info = env_cfg.libero_config.task_info
    instruction = task_info["language_instruction"]
    task_dir = args.output_root / "tabero" / args.task_suite / f"task_{args.task_id:03d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    results_path = task_dir / "results.jsonl"
    done = load_results(results_path)

    hook = TaberoTacumiRolloutHook(
        env=env,
        output_root=args.output_root,
        task_suite=args.task_suite,
        task_id=args.task_id,
        instruction=instruction,
        fps=1.0 / (env_cfg.sim.render_interval * env_cfg.sim.dt),
        record=not args.no_record,
        policy_contract="tabero_gpt_taskspace_absolute_v1",
        action_layout=ACTION_LAYOUT,
    )
    if args.policy == "jemm":
        client = JEMMClient(args.jemm_endpoint)
        client.wait_ready()
        policy = JEMMPolicy(client)
    elif args.policy == "gpt":
        policy = GPTPolicy(LLMClient())
    else:
        goal = env_cfg.libero_config.goals[0]
        policy = ScriptedPolicy(goal["ref_obj"], goal["target"], place_m=args.scripted_place_m, grasp_dz=args.scripted_grasp_dz)
    task_name = f"task_{args.task_id:03d}"
    meta = {
        "benchmark": "tabero",
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "instruction": instruction,
        "max_episode_steps": native_steps,
        "time_limit_scale": args.time_limit_scale,
        "episode_step_limit": None if args.disable_time_out else int(env.max_episode_length),
        "time_out_disabled": args.disable_time_out,
        "num_success_steps": NUM_SUCCESS_STEPS,
        **runtime_metadata(ENV_ID),
    }

    with torch.inference_mode():
        for seed in range(args.seed_start, args.seed_start + args.num_seeds):
            if seed in done:
                continue
            seed_everything(env, seed)
            env.reset()
            hook.start_episode(seed, instruction)
            hook.capture_tactile()
            episode_dir = task_dir / f"episode_{seed:03d}"
            adapter = TaberoAdapter(env, hook, success_term, instruction, list(env_cfg.libero_config.objects))
            summary = {"seed": seed, **meta, **run_episode(adapter, policy, episode_dir, args.max_decisions, seed)}
            summary["success_within_max_episode_steps"] = summary["success"] and summary["steps"] <= native_steps
            hook.finish_episode(success=summary["success"], experiment_index=seed, extra=summary)
            if not args.no_record and args.policy != "scripted":
                summary["datasets"] = export_episode(
                    episode_dir,
                    rollout_dir=args.output_root / "showharness" / "tabero" / args.task_suite / task_name / f"rollout_{seed:03d}",
                    dataset_root=args.output_root / "lerobot_v3" / "tabero",
                    benchmark="tabero",
                    source_id=f"tabero:{args.task_suite}:{task_name}:seed_{seed:03d}",
                    task=instruction,
                    summary=summary,
                    metadata={"task": f"{args.task_suite}/{task_name}", "task_text": instruction, "step_m": STEP_M},
                )
            finish_episode(results_path, summary)
            print(f"[gpt_rollout] {args.task_suite} task {args.task_id} seed {seed}: " + json.dumps(summary), flush=True)

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
