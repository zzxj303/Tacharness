"""Rebuild, with JEMM's own contract code, the exact request JEMM received on decision 0 of
the first published episode of each benchmark.

Run in .venv-jemm (needs the jemm package); scripts/site/build_site.sh does this.
Usage: build_prompts.py <output_root> <prompts.json>
"""

import ast
import json
import sys
from pathlib import Path

from jemm.contract import SYSTEM, mm_messages
from site_layout import BENCHMARKS, TOKENS, episode_dirs, published

PROMPTS = Path(__file__).resolve().parents[2] / "gpt_rollout" / "prompts.py"

root, out_path = Path(sys.argv[1]), Path(sys.argv[2])
consts = {
    target.id: ast.literal_eval(node.value)
    for node in ast.parse(PROMPTS.read_text(encoding="utf-8")).body
    if isinstance(node, ast.Assign)
    for target in node.targets
    if isinstance(target, ast.Name) and target.id in ("ACTION_DESCRIPTIONS", "JEMM_QUESTION")
}
out = {"system": SYSTEM, "question": consts["JEMM_QUESTION"], "candidates": consts["ACTION_DESCRIPTIONS"]}
for benchmark in BENCHMARKS:
    episodes = published(root, benchmark)
    if not episodes:
        continue
    episode_dir, showharness_dir = episode_dirs(root, benchmark, episodes[0]["source_id"])
    call = json.loads((episode_dir / "gpt_calls" / "calls.jsonl").read_text(encoding="utf-8").splitlines()[0])
    request = {
        "state": call["input_text"],
        "question": consts["JEMM_QUESTION"],
        "candidates": [{"id": t, "description": consts["ACTION_DESCRIPTIONS"][t], "action": {}} for t in TOKENS],
    }
    system, user = mm_messages(request, 3)
    assert system["content"] == SYSTEM
    out[benchmark] = {
        "source_id": episodes[0]["source_id"],
        "showharness_dir": str(showharness_dir.relative_to(root)),
        "user_text": user["content"][-1]["text"],
        "image_slots": [part["type"] for part in user["content"]],
        "attempt": call["attempts"][0],
        "decision": call["decision"],
        "call_index": call["call_index"],
    }
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out_path, {b: len(out[b]["user_text"]) for b in BENCHMARKS if b in out})
