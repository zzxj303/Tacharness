"""Benchmark-independent rollout loop (README sections 1 and 4)."""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from . import MAX_RECALLS, RECENT_MOVES
from .actions import InvalidOutput, parse_decision
from .interpreter import MOVE_TOKENS, EpisodeEnded, StepAdapter, execute, make_executor
from .llm import LLMClient, LLMResponse, LLMTransportError
from .prompts import Observation, build_messages, encode_png, observation_text, resize_with_pad, system_prompt


# How often to retry a call whose transport failed (after llm.yaml's own
# max_attempts), and how long to wait in between.
TRANSPORT_RETRY_ROUNDS = 120
TRANSPORT_RETRY_WAIT_S = 30.0


@dataclass
class StepResult:
    done: bool
    success: bool = False
    reason: str = ""


class EpisodeAdapter(StepAdapter, Protocol):
    benchmark: str
    steps: int  # env.step calls so far in this episode

    def observe(self) -> Observation: ...

    def gripper_width_m(self) -> float:
        """Opening between the two pads, recorded with each decision (not shown to the policy)."""


@dataclass
class Decision:
    token: str | None  # None: still no valid action after MAX_RECALLS re-calls
    record: dict[str, Any] = field(default_factory=dict)  # logged to calls.jsonl


class Policy(Protocol):
    def begin_episode(self, benchmark: str, calls_dir: Path, seed: int) -> None: ...

    def decide(self, observation: Observation, recent_moves: list[str], call_index: int) -> Decision: ...

    def summary(self) -> dict[str, Any]: ...


class GPTPolicy:
    """GPT-6 picks the action from the README section 2 inputs plus the recent moves."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def begin_episode(self, benchmark: str, calls_dir: Path, seed: int) -> None:
        self.benchmark = benchmark
        self.calls_dir = calls_dir
        (calls_dir / "system_prompt.txt").write_text(system_prompt(benchmark), encoding="utf-8")
        self.usage_total: dict[str, float] = {}
        self.seconds = 0.0
        self.calls = 0
        self.transport_failures = 0

    def decide(self, observation: Observation, recent_moves: list[str], call_index: int) -> Decision:
        text = observation_text(observation, recent_moves)
        images = {
            "scene": encode_png(resize_with_pad(observation.scene_rgb)),
            "wrist": encode_png(resize_with_pad(observation.wrist_rgb)),
            "tactile": encode_png(observation.tactile_rgb),
        }
        image_files = {}
        for name, png in images.items():
            image_files[name] = f"call_{call_index:04d}_{name}.png"
            (self.calls_dir / image_files[name]).write_bytes(png)
        messages = build_messages(self.benchmark, text, list(images.values()))

        token = None
        attempts = []
        for _ in range(1 + MAX_RECALLS):
            response, failures = _chat(self.llm, messages)
            self.transport_failures += failures
            self.calls += 1
            self.seconds += response.latency_s
            for key, value in response.usage.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.usage_total[key] = self.usage_total.get(key, 0) + value
            attempt = {
                "raw_output": response.text,
                "model": response.model,
                "usage": response.usage,
                "latency_s": round(response.latency_s, 3),
            }
            attempts.append(attempt)
            try:
                token = parse_decision(response.text)
            except InvalidOutput as exc:
                attempt["invalid"] = str(exc)
                continue
            break
        return Decision(token, {"input_text": text, "input_images": image_files, "attempts": attempts})

    def summary(self) -> dict[str, Any]:
        return {
            "policy": "gpt",
            "gpt_model": self.llm.model,
            "gpt_temperature": self.llm.temperature,
            "gpt_calls": self.calls,
            "gpt_usage": self.usage_total,
            "gpt_transport_failures": self.transport_failures,
            "gpt_seconds": round(self.seconds, 2),
        }


def run_episode(
    adapter: EpisodeAdapter,
    policy: Policy,
    output_dir: Path,
    max_decisions: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Ask for one action, execute it to completion, repeat until the episode ends.

    The simulation only advances inside ``adapter.step``, so it is paused while
    the policy is deciding and during re-calls. With ``max_decisions`` None the
    environment's time limit alone ends the episode: every action takes at least
    one env step.
    """
    calls_dir = output_dir / "gpt_calls"
    calls_dir.mkdir(parents=True, exist_ok=True)
    policy.begin_episode(adapter.benchmark, calls_dir, seed)
    executor = make_executor(adapter)

    start = time.perf_counter()
    recent_moves: list[str] = []
    decisions = 0
    result = StepResult(done=True, reason="max_decisions")
    with (calls_dir / "calls.jsonl").open("w", encoding="utf-8") as log:
        for call_index in itertools.count() if max_decisions is None else range(max_decisions):
            observation = adapter.observe()
            # The observation item by item, so the decision-level export (README 7.2) can
            # leave out the privileged object positions.
            observed = {
                "time": round(time.time(), 3),
                "instruction": observation.instruction,
                "position": [round(float(v), 6) for v in observation.position],
                "quaternion": [round(float(v), 6) for v in observation.quaternion],
                "gripper_closed": bool(observation.gripper_closed),
                "gripper_width_m": round(float(adapter.gripper_width_m()), 6),
                "objects": {name: [round(float(v), 6) for v in xyz] for name, xyz in observation.objects.items()},
                "recent_moves": list(recent_moves),
            }
            decision = policy.decide(observation, recent_moves, call_index)
            decisions += 1
            first_frame = adapter.steps
            travelled = None
            step_result = StepResult(done=False)
            if decision.token is None:
                step_result = StepResult(done=True, reason="invalid_output")
            else:
                try:
                    travelled = execute(executor, decision.token)
                except EpisodeEnded as ended:
                    step_result = ended.result
            log.write(
                json.dumps(
                    {
                        "call_index": call_index,
                        "observation": observed,
                        **decision.record,
                        "decision": decision.token,
                        "first_frame": first_frame,
                        "frame_count": adapter.steps - first_frame,
                        "travelled_m": None if travelled is None else round(travelled, 4),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            log.flush()
            if step_result.done:
                result = step_result
                break
            if decision.token in MOVE_TOKENS:
                recent_moves.insert(0, decision.token)
                del recent_moves[RECENT_MOVES:]

    return {
        "success": bool(result.success),
        "end_reason": result.reason,
        "steps": adapter.steps,
        "decisions": decisions,
        "max_decisions": max_decisions,
        "wall_seconds": round(time.perf_counter() - start, 2),
        **policy.summary(),
    }


def _chat(llm: LLMClient, messages: list[dict[str, Any]]) -> tuple[LLMResponse, int]:
    """Call GPT, waiting out endpoint outages; the simulation stays paused meanwhile.

    An unreachable or timed-out endpoint is not a GPT output, so it does not
    count as one of the MAX_RECALLS re-calls.
    """
    for failures in range(TRANSPORT_RETRY_ROUNDS):
        try:
            return llm.chat(messages), failures
        except LLMTransportError as exc:
            print(f"[gpt_rollout] {exc}; retrying in {TRANSPORT_RETRY_WAIT_S:.0f}s", flush=True)
            time.sleep(TRANSPORT_RETRY_WAIT_S)
    return llm.chat(messages), TRANSPORT_RETRY_ROUNDS


def load_results(path: Path) -> dict[int, dict[str, Any]]:
    """Completed episodes of one task, keyed by seed."""
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {int(row["seed"]): row for row in rows}


def finish_episode(results_path: Path, summary: dict[str, Any]) -> None:
    """Log the outcome. Successful and failed episodes are both collected (README 0.2);
    write this only after the episode's data is exported, since resume skips the seed."""
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
