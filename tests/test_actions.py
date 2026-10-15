import json

import pytest

from gpt_rollout.actions import InvalidOutput, parse_decision
from gpt_rollout.interpreter import TOKENS


def test_vocabulary():
    assert TOKENS == ("MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN", "GRASP", "RELEASE")


@pytest.mark.parametrize("token", TOKENS)
def test_valid_decision(token):
    assert parse_decision(json.dumps({"decision": token})) == token


def test_characters_around_the_object_are_ignored():
    assert parse_decision('_{"decision": "MV_FWD"}\n') == "MV_FWD"


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        None,
        '_{"decision": "MV_FWD"} {"decision": "MV_UP"}',
        "}{",
        json.dumps({"decision": "DONE"}),
        json.dumps({"decision": "mv_up"}),
        json.dumps({"decision": "MV_UP", "reasoning": "lift"}),
        json.dumps({"actions": [[0.5, 0, 0.3, 0, 1, 0, 0, 0]] * 4}),
        json.dumps(["MV_UP"]),
    ],
)
def test_invalid_outputs(text):
    with pytest.raises(InvalidOutput):
        parse_decision(text)
