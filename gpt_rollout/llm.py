"""Chat-completions client configured by configs/llm.yaml."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "llm.yaml"


class LLMTransportError(RuntimeError):
    """The endpoint could not be reached within max_attempts."""


@dataclass
class LLMResponse:
    text: str
    usage: dict[str, Any]
    model: str
    latency_s: float
    attempts: int


class LLMClient:
    def __init__(self, config_path: str | Path = DEFAULT_CONFIG, profile: str = "openrouter") -> None:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["profiles"][profile]
        if config.get("api_mode", "chat") != "chat":
            raise ValueError(f"Unsupported api_mode: {config['api_mode']!r}")
        self.endpoint = str(config["endpoint"]).rstrip("/")
        self.model = str(config["model"])
        self.temperature = config.get("temperature")  # omit for models that reject it (GPT-6)
        self.timeout_s = float(config["timeout_s"])
        self.max_attempts = int(config["max_attempts"])
        self._api_key = str(config["api_key"])

    def chat(self, messages: list[dict[str, Any]]) -> LLMResponse:
        request_body: dict[str, Any] = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
        if self.temperature is not None:
            request_body["temperature"] = self.temperature
        body = json.dumps(request_body).encode("utf-8")
        start = time.perf_counter()
        error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            request = urllib.request.Request(
                f"{self.endpoint}/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as reply:
                    payload = json.load(reply)
                text = payload["choices"][0]["message"]["content"]
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError, ValueError) as exc:
                error = exc
                continue
            return LLMResponse(
                text=text if isinstance(text, str) else "",
                usage=dict(payload.get("usage") or {}),
                model=str(payload.get("model") or self.model),
                latency_s=time.perf_counter() - start,
                attempts=attempt,
            )
        raise LLMTransportError(f"LLM call failed after {self.max_attempts} attempts: {error!r}")
