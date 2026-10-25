"""JEMM as the policy: the README section 2 inputs in, one sampled meta action out.

JEMM (``models/JEMM``, a LoRA on Qwen3.8-27B) runs in its own process
(``scripts/jemm_serve.sh``) and answers ``POST /v1/systemone`` with a probability for
every candidate. The action is drawn from those probabilities with a generator seeded
by the episode seed, so a seed reproduces its episode and the RoboLab seeds, whose
openings are identical, still differ.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .episode import Decision
from .interpreter import TOKENS
from .llm import LLMTransportError
from .prompts import ACTION_DESCRIPTIONS, JEMM_QUESTION, Observation, encode_png, jemm_state, resize_with_pad

DEFAULT_ENDPOINT = "http://127.0.0.1:8790"
DECISION_CONFIG = Path(__file__).resolve().parents[1] / "models" / "JEMM" / "decision_config.json"
# How often to retry an unreachable server, and how long to wait in between; the
# simulation stays paused meanwhile.
TRANSPORT_RETRY_ROUNDS = 120
TRANSPORT_RETRY_WAIT_S = 30.0


@dataclass
class JEMMResponse:
    probabilities: dict[str, float]
    confidence: float
    model: str
    latency_s: float
    attempts: int


class JEMMClient:
    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, timeout_s: float = 300.0, max_attempts: int = 3) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.threshold = float(json.loads(DECISION_CONFIG.read_text(encoding="utf-8"))["threshold"])

    def wait_ready(self, timeout_s: float = 1800.0, poll_s: float = 10.0) -> str:
        """Block until the server answers /health (loading the weights takes about a minute)."""
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                with urllib.request.urlopen(f"{self.endpoint}/health", timeout=10) as reply:
                    health = json.load(reply)
                if health.get("status") == "READY":
                    return str(health.get("model", ""))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise LLMTransportError(f"JEMM server at {self.endpoint} not ready after {timeout_s:.0f}s")
            print(f"[gpt_rollout] waiting for the JEMM server at {self.endpoint}", flush=True)
            time.sleep(poll_s)

    def decide(self, state: str, images: list[bytes]) -> JEMMResponse:
        """``images`` are PNG bytes in the order scene, wrist, tactile."""
        body = json.dumps(
            {
                "state": state,
                "questions": {
                    "action": {
                        "type": "choice",
                        "instructions": JEMM_QUESTION,
                        "criteria": {token: ACTION_DESCRIPTIONS[token] for token in TOKENS},
                    }
                },
                "images": [base64.b64encode(png).decode("ascii") for png in images],
            }
        ).encode("utf-8")
        start = time.perf_counter()
        error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            request = urllib.request.Request(
                f"{self.endpoint}/v1/systemone", data=body, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as reply:
                    payload = json.load(reply)
                answer = payload["answers"]["action"]
                probabilities = {token: float(answer["probabilities"][token]) for token in TOKENS}
            except urllib.error.HTTPError as exc:
                if exc.code == 422:  # the server rejected the request itself; retrying cannot help
                    raise ValueError(f"JEMM rejected the request: {exc.read()[:200]!r}") from exc
                error = exc
                continue
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, TypeError, ValueError) as exc:
                error = exc
                continue
            return JEMMResponse(
                probabilities=probabilities,
                confidence=float(answer["confidence"]),
                model=str(payload.get("model", "jemm")),
                latency_s=time.perf_counter() - start,
                attempts=attempt,
            )
        raise LLMTransportError(f"JEMM call failed after {self.max_attempts} attempts: {error!r}")


class JEMMPolicy:
    """Sample the action from JEMM's probabilities over the README section 3 actions."""

    def __init__(self, client: JEMMClient) -> None:
        self.client = client
        self.model = ""

    def begin_episode(self, benchmark: str, calls_dir: Path, seed: int) -> None:
        self.benchmark = benchmark
        self.calls_dir = calls_dir
        self.rng = np.random.default_rng(seed)
        self.seconds = 0.0
        self.calls = 0
        self.transport_failures = 0
        self.below_threshold = 0

    def decide(self, observation: Observation, recent_moves: list[str], call_index: int) -> Decision:
        state = jemm_state(self.benchmark, observation, recent_moves)
        images = {
            "scene": encode_png(resize_with_pad(observation.scene_rgb)),
            "wrist": encode_png(resize_with_pad(observation.wrist_rgb)),
            "tactile": encode_png(observation.tactile_rgb),
        }
        image_files = {}
        for name, png in images.items():
            image_files[name] = f"call_{call_index:04d}_{name}.png"
            (self.calls_dir / image_files[name]).write_bytes(png)

        response, failures = self._decide(state, list(images.values()))
        self.transport_failures += failures
        self.calls += 1
        self.seconds += response.latency_s
        self.model = response.model
        probabilities = np.array([response.probabilities[token] for token in TOKENS], dtype=np.float64)
        token = TOKENS[int(self.rng.choice(len(TOKENS), p=probabilities / probabilities.sum()))]
        below = response.confidence < self.client.threshold
        self.below_threshold += int(below)
        return Decision(
            token,
            {
                "input_text": state,
                "input_images": image_files,
                "attempts": [
                    {
                        "probabilities": {k: round(v, 6) for k, v in response.probabilities.items()},
                        "confidence": round(response.confidence, 6),
                        "below_threshold": below,
                        "model": response.model,
                        "latency_s": round(response.latency_s, 3),
                    }
                ],
            },
        )

    def _decide(self, state: str, images: list[bytes]) -> tuple[JEMMResponse, int]:
        for failures in range(TRANSPORT_RETRY_ROUNDS):
            try:
                return self.client.decide(state, images), failures
            except LLMTransportError as exc:
                print(f"[gpt_rollout] {exc}; retrying in {TRANSPORT_RETRY_WAIT_S:.0f}s", flush=True)
                time.sleep(TRANSPORT_RETRY_WAIT_S)
        return self.client.decide(state, images), TRANSPORT_RETRY_ROUNDS

    def summary(self) -> dict[str, Any]:
        return {
            "policy": "jemm",
            "policy_model": self.model,
            "policy_sampling": "categorical over JEMM probabilities, generator seeded by the episode seed",
            "policy_calls": self.calls,
            "policy_below_threshold": self.below_threshold,
            "policy_threshold": self.client.threshold,
            "policy_transport_failures": self.transport_failures,
            "policy_seconds": round(self.seconds, 2),
        }
