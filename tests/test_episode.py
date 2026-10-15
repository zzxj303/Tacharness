import json

import numpy as np

from gpt_rollout import MAX_RECALLS
from gpt_rollout.episode import GPTPolicy, StepResult, finish_episode, load_results, run_episode
from gpt_rollout.llm import LLMResponse
from gpt_rollout.prompts import Observation, resize_with_pad

START = np.array([0.5, 0.0, 0.3])
QUAT = np.array([0.0, 1.0, 0.0, 0.0])


def _decision(token):
    return json.dumps({"decision": token})


class FakeLLM:
    model = "fake"
    temperature = 0

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        text = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        return LLMResponse(text=text, usage={"total_tokens": 10}, model="fake", latency_s=0.0, attempts=1)


class FakeAdapter:
    """The TCP covers half of the remaining distance to the commanded target per step."""

    benchmark = "tabero"
    gripper_open = 1.0
    gripper_close = 0.0

    def __init__(self, max_steps, succeed_at=None):
        self.max_steps = max_steps
        self.succeed_at = succeed_at
        self.steps = 0
        self.position = START.copy()
        self.actions = []

    def tcp_pose(self):
        return self.position.copy(), QUAT.copy()

    def observe(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        return Observation("do it", image, image, image, self.position.copy(), QUAT.copy(), False, {})

    def step(self, action):
        self.actions.append(action)
        self.position += 0.5 * (action[:3] - self.position)
        self.steps += 1
        if self.steps == self.succeed_at:
            return StepResult(done=True, success=True, reason="success")
        if self.steps >= self.max_steps:
            return StepResult(done=True, reason="time_out")
        return StepResult(done=False)


def _calls(tmp_path):
    return [json.loads(line) for line in (tmp_path / "gpt_calls" / "calls.jsonl").read_text().splitlines()]


def test_move_is_two_centimetres_closed_loop(tmp_path):
    adapter, llm = FakeAdapter(max_steps=10_000), FakeLLM([_decision("MV_RIGHT")])
    summary = run_episode(adapter, GPTPolicy(llm), tmp_path, max_decisions=3)
    assert (summary["success"], summary["end_reason"], summary["decisions"]) == (False, "max_decisions", 3)
    np.testing.assert_allclose(adapter.position, START + [0, 0.06, 0], atol=1e-3)
    calls = _calls(tmp_path)
    assert all(c["frame_count"] > 1 and abs(c["travelled_m"] - 0.02) < 1e-3 for c in calls)
    assert calls[1]["first_frame"] == calls[0]["frame_count"]
    # Orientation is held and the gripper stays open.
    assert all(np.allclose(a[3:7], QUAT) and a[7] == 1.0 for a in adapter.actions)
    assert "Recent moves, newest first: MV_RIGHT, MV_RIGHT" in calls[2]["input_text"]


def test_grasp_holds_the_close_command(tmp_path):
    adapter, llm = FakeAdapter(max_steps=10_000), FakeLLM([_decision("GRASP")])
    run_episode(adapter, GPTPolicy(llm), tmp_path, max_decisions=1)
    assert adapter.steps >= 10 and adapter.actions[-1][7] == 0.0
    assert "Recent moves, newest first: none" in _calls(tmp_path)[0]["input_text"]


def test_episode_ends_inside_an_action(tmp_path):
    adapter, llm = FakeAdapter(max_steps=10_000, succeed_at=7), FakeLLM([_decision("MV_UP")])
    summary = run_episode(adapter, GPTPolicy(llm), tmp_path)
    assert summary["success"] and summary["steps"] == 7
    assert sum(c["frame_count"] for c in _calls(tmp_path)) == 7


def test_timeout(tmp_path):
    adapter, llm = FakeAdapter(max_steps=30), FakeLLM([_decision("MV_DOWN")])
    summary = run_episode(adapter, GPTPolicy(llm), tmp_path)
    assert (summary["success"], summary["end_reason"], summary["steps"]) == (False, "time_out", 30)


def test_recall_does_not_step_and_gives_up(tmp_path):
    adapter, llm = FakeAdapter(max_steps=10_000), FakeLLM(["oops", _decision("MV_UP"), "oops"])
    summary = run_episode(adapter, GPTPolicy(llm), tmp_path)
    assert summary["end_reason"] == "invalid_output" and not summary["success"]
    assert summary["decisions"] == 2 and llm.calls == 2 + 1 + MAX_RECALLS
    assert _calls(tmp_path)[-1]["frame_count"] == 0


def test_failed_episodes_are_set_aside(tmp_path):
    results = tmp_path / "results.jsonl"
    for seed, success in ((0, True), (1, False)):
        episode = tmp_path / f"episode_{seed:03d}"
        (episode / "frames").mkdir(parents=True)
        (episode / "frames" / "frame_000000.npz").write_bytes(b"x")
        (episode / "gpt_calls").mkdir()
        (episode / "gpt_calls" / "calls.jsonl").write_text("{}\n")
        (episode / "gpt_calls" / "call_0000_scene.png").write_bytes(b"x")
        finish_episode(results, episode, {"seed": seed, "success": success})
    assert (tmp_path / "episode_000" / "frames" / "frame_000000.npz").is_file()
    assert not (tmp_path / "episode_001").exists()
    kept = sorted(p.relative_to(tmp_path).as_posix() for p in (tmp_path / "failed").rglob("*") if p.is_file())
    assert kept == ["failed/episode_001/calls.jsonl"]
    assert set(load_results(results)) == {0, 1}


def test_resize_with_pad():
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    small = resize_with_pad(image)
    assert small.shape == (224, 224, 3) and small[0, 112].max() == 0 and small[112, 112].min() == 255
