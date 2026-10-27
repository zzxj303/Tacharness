"""Rebuild the exact request the policy received on decision 0 of the first published episode
of each benchmark. JEMM's is rebuilt with its own contract code; GPT-6's is read back from
the call log (calls.jsonl keeps the user text, gpt_calls/system_prompt.txt the system prompt).

Run in .venv-jemm for JEMM output roots (needs the jemm package), with any Python 3 for GPT
ones; scripts/site/build_site.sh picks.
Usage: build_prompts.py <output_root> <prompts.json>
"""

import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.site.site_layout import BENCHMARKS, TOKENS, episode_dirs, published  # noqa: E402

PROMPTS = Path(__file__).resolve().parents[2] / "gpt_rollout" / "prompts.py"

root, out_path = Path(sys.argv[1]), Path(sys.argv[2])
consts = {
    target.id: ast.literal_eval(node.value)
    for node in ast.parse(PROMPTS.read_text(encoding="utf-8")).body
    if isinstance(node, ast.Assign)
    for target in node.targets
    if isinstance(target, ast.Name) and target.id in ("ACTION_DESCRIPTIONS", "JEMM_QUESTION")
}


def first_call(benchmark):
    """(episode dir, Show-Harness dir, source id, decision-0 call) of the benchmark's first episode."""
    source_id = published(root, benchmark)[0]["source_id"]
    episode_dir, showharness_dir = episode_dirs(root, benchmark, source_id)
    call = json.loads((episode_dir / "gpt_calls" / "calls.jsonl").read_text(encoding="utf-8").splitlines()[0])
    return episode_dir, showharness_dir, source_id, call


benchmarks = [b for b in BENCHMARKS if published(root, b)]
episode_dir, *_ = first_call(benchmarks[0])
results = [json.loads(line) for line in (episode_dir.parent / "results.jsonl").read_text(encoding="utf-8").splitlines()]
policy = results[0].get("policy", "jemm")

if policy == "jemm":
    from jemm.contract import SYSTEM, mm_messages

    out = {"policy": policy, "system": SYSTEM, "question": consts["JEMM_QUESTION"], "candidates": consts["ACTION_DESCRIPTIONS"]}
else:
    out = {"policy": policy, "model": results[0].get("gpt_model", ""), "candidates": consts["ACTION_DESCRIPTIONS"]}

for benchmark in benchmarks:
    episode_dir, showharness_dir, source_id, call = first_call(benchmark)
    entry = {
        "source_id": source_id,
        "showharness_dir": str(showharness_dir.relative_to(root)),
        "attempt": call["attempts"][0],
        "decision": call["decision"],
        "call_index": call["call_index"],
    }
    if policy == "jemm":
        request = {
            "state": call["input_text"],
            "question": consts["JEMM_QUESTION"],
            "candidates": [{"id": t, "description": consts["ACTION_DESCRIPTIONS"][t], "action": {}} for t in TOKENS],
        }
        system, user = mm_messages(request, 3)
        assert system["content"] == SYSTEM
        entry.update(user_text=user["content"][-1]["text"], image_slots=[part["type"] for part in user["content"]])
    else:
        # gpt_rollout/prompts.py build_messages: the text first, then the three images.
        entry.update(
            system=(episode_dir / "gpt_calls" / "system_prompt.txt").read_text(encoding="utf-8"),
            user_text=call["input_text"],
            image_slots=["text", "image_url", "image_url", "image_url"],
        )
    out[benchmark] = entry
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out_path, policy, {b: len(out[b]["user_text"]) for b in benchmarks})
