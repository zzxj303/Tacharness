import json

import numpy as np

from gpt_rollout import MAX_RECALLS
from gpt_rollout.datasets import export_showharness
from gpt_rollout.episode import GPTPolicy, StepResult, finish_episode, load_results, run_episode
from gpt_rollout.interpreter import TOKENS
from gpt_rollout.jemm import JEMMPolicy, JEMMResponse
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


class FakeJEMM:
    """Fixed probabilities, as JEMM returns for an unchanged observation."""

    threshold = 0.99

    def __init__(self, probabilities):
        self.probabilities = dict(zip(TOKENS, probabilities))
        self.states = []

    def decide(self, state, images):
        self.states.append(state)
        confidence = max(self.probabilities.values())
        return JEMMResponse(self.probabilities, confidence, "fake-jemm", latency_s=0.0, attempts=1)


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

    def gripper_width_m(self):
        return 0.0797

    def observe(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        objects = {"milk_1": np.array([0.6, 0.1, 0.05])}
        return Observation("do it", image, image, image, self.position.copy(), QUAT.copy(), False, objects)

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
    assert "Recent actions, newest first: MV_RIGHT, MV_RIGHT" in calls[2]["input_text"]


def test_grasp_holds_the_close_command(tmp_path):
    adapter, llm = FakeAdapter(max_steps=10_000), FakeLLM([_decision("GRASP")])
    run_episode(adapter, GPTPolicy(llm), tmp_path, max_decisions=1)
    assert adapter.steps >= 10 and adapter.actions[-1][7] == 0.0
    assert "Recent actions, newest first: none" in _calls(tmp_path)[0]["input_text"]


def test_grasp_and_release_are_recent_actions(tmp_path):
    # Without them the observation after a RELEASE repeats the one before the GRASP.
    adapter = FakeAdapter(max_steps=10_000)
    llm = FakeLLM([_decision("MV_DOWN"), _decision("GRASP"), _decision("RELEASE"), _decision("MV_UP")])
    run_episode(adapter, GPTPolicy(llm), tmp_path, max_decisions=4)
    calls = _calls(tmp_path)
    assert "Recent actions, newest first: GRASP, MV_DOWN" in calls[2]["input_text"]
    assert "Recent actions, newest first: RELEASE, GRASP, MV_DOWN" in calls[3]["input_text"]
    assert calls[3]["observation"]["recent_moves"] == ["RELEASE", "GRASP", "MV_DOWN"]


def test_episode_ends_inside_an_action(tmp_path):
    adapter, llm = FakeAdapter(max_steps=10_000, succeed_at=7), FakeLLM([_decision("MV_UP")])
    summary = run_episode(adapter, GPTPolicy(llm), tmp_path)
    assert summary["success"] and summary["steps"] == 7
    assert sum(c["frame_count"] for c in _calls(tmp_path)) == 7


def test_timeout(tmp_path):
    adapter, llm = FakeAdapter(max_steps=30), FakeLLM([_decision("MV_DOWN")])
    summary = run_episode(adapter, GPTPolicy(llm), tmp_path)
    assert (summary["success"], summary["end_reason"], summary["steps"]) == (False, "time_out", 30)
    # No decision cap by default: the time limit alone ends the episode.
    assert summary["max_decisions"] is None and summary["decisions"] == len(_calls(tmp_path))


def test_recall_does_not_step_and_gives_up(tmp_path):
    adapter, llm = FakeAdapter(max_steps=10_000), FakeLLM(["oops", _decision("MV_UP"), "oops"])
    summary = run_episode(adapter, GPTPolicy(llm), tmp_path)
    assert summary["end_reason"] == "invalid_output" and not summary["success"]
    assert summary["decisions"] == 2 and llm.calls == 2 + 1 + MAX_RECALLS
    assert _calls(tmp_path)[-1]["frame_count"] == 0


def test_failed_episodes_are_kept(tmp_path):
    results = tmp_path / "results.jsonl"
    for seed, success in ((0, True), (1, False)):
        episode = tmp_path / f"episode_{seed:03d}"
        (episode / "frames").mkdir(parents=True)
        (episode / "frames" / "frame_000000.npz").write_bytes(b"x")
        finish_episode(results, {"seed": seed, "success": success})
    assert all((tmp_path / f"episode_{seed:03d}" / "frames" / "frame_000000.npz").is_file() for seed in (0, 1))
    assert set(load_results(results)) == {0, 1}


def _jemm_episode(tmp_path, seed, probabilities=(0.4, 0.1, 0.1, 0.3, 0.05, 0.05, 0.0, 0.0)):
    adapter, jemm = FakeAdapter(max_steps=10_000), FakeJEMM(probabilities)
    summary = run_episode(adapter, JEMMPolicy(jemm), tmp_path, max_decisions=12, seed=seed)
    return summary, [c["decision"] for c in _calls(tmp_path)], jemm


def test_jemm_samples_reproducibly_by_seed(tmp_path):
    _, first, _ = _jemm_episode(tmp_path / "a", seed=3)
    _, again, _ = _jemm_episode(tmp_path / "b", seed=3)
    _, other, _ = _jemm_episode(tmp_path / "c", seed=4)
    assert first == again and first != other
    # Zero-probability actions are never drawn.
    assert not {"GRASP", "RELEASE"} & set(first + other)


def test_jemm_record_and_summary(tmp_path):
    summary, _, jemm = _jemm_episode(tmp_path, seed=0)
    call = _calls(tmp_path)[0]
    attempt = call["attempts"][0]
    assert abs(sum(attempt["probabilities"].values()) - 1.0) < 1e-6 and attempt["below_threshold"]
    assert set(call["input_images"]) == {"scene", "wrist", "tactile"}
    assert call["observation"]["objects"] == {"milk_1": [0.6, 0.1, 0.05]}
    assert "CURRENT OBSERVATION" in jemm.states[0] and "milk_1" in jemm.states[0]
    assert (summary["policy"], summary["policy_calls"], summary["policy_below_threshold"]) == ("jemm", 12, 12)


def test_showharness_export(tmp_path):
    episode, rollout = tmp_path / "episode_000", tmp_path / "rollout_000"
    adapter, llm = FakeAdapter(max_steps=10_000), FakeLLM([_decision("MV_RIGHT"), _decision("GRASP")])
    summary = run_episode(adapter, GPTPolicy(llm), episode, max_decisions=2)
    assert export_showharness(episode, rollout, {"task": "t", "seed": 0}) == 2
    rows = [json.loads(line) for line in (rollout / "actions.jsonl").read_text().splitlines()]
    assert [(r["step"], r["token"], r["kind"]) for r in rows] == [(0, "MV_RIGHT", "move"), (1, "GRASP", "grasp")]
    assert rows[0]["ee_pose"] == [*START, *QUAT] and rows[0]["gripper_closed"] is False
    assert rows[1]["first_frame"] == rows[0]["frame_count"]
    assert sum(r["frame_count"] for r in rows) == summary["steps"]
    for row in rows:
        assert all((rollout / row[view]).is_file() for view in ("agentview", "wrist", "tactile"))
    # No privileged object positions in the decision-level data.
    assert "milk_1" not in (rollout / "actions.jsonl").read_text()
    meta = json.loads((rollout / "metadata.json").read_text())
    assert meta["num_steps"] == 2 and meta["token_counts"] == {"MV_RIGHT": 1, "GRASP": 1}
    assert not list((episode / "gpt_calls").glob("*.png"))


def test_resize_with_pad():
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    small = resize_with_pad(image)
    assert small.shape == (224, 224, 3) and small[0, 112].max() == 0 and small[112, 112].min() == 255
