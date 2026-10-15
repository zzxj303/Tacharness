"""Parse and validate the GPT output defined in README section 3."""

from __future__ import annotations

import json

from .interpreter import TOKENS

# The 8 numbers each env.step receives (and each recorded frame stores).
ACTION_LAYOUT = ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper_command"]


class InvalidOutput(ValueError):
    """The GPT output does not match the required action format."""


def parse_decision(text: str) -> str:
    """Return the one action of ``{"decision": "<ACTION>"}``.

    Characters outside the JSON object are ignored: the gateway sometimes prepends
    a stray ``_`` to replies, identically on every re-call of the same observation.
    """
    try:
        payload = json.loads(text[text.index("{") : text.rindex("}") + 1])
    except (AttributeError, TypeError, ValueError) as exc:
        raise InvalidOutput(f"not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"decision"}:
        raise InvalidOutput('output must be an object with the single key "decision"')
    token = payload["decision"]
    if token not in TOKENS:
        raise InvalidOutput(f"decision must be one of {', '.join(TOKENS)}")
    return token
