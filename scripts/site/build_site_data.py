"""Collect an output root's two datasets, videos and prompts into the viewer's data.json and
media/ (half-size web copies of the videos, and the three images of the prompt shown).

Run in TacHarness's LeRobot runtime (needs pyarrow); scripts/site/build_site.sh does this.
Usage: build_site_data.py <output_root> <prompts.json> <site_dir> [--title TEXT] [--case FRAGMENT.html]
"""

import argparse
import datetime
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.site.site_layout import BENCHMARKS, TOKENS, episode_dirs, published  # noqa: E402

ROLLOUT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_KEYS = {"rgb": "observation.images.rgb_dual_view", "tactile": "observation.images.tacumi_tactile_composite"}
# Web copies: about half size, H.264 with the index up front so the browser can seek.
VIDEO_HEIGHT = {("tabero", "rgb"): 384, ("robolab", "rgb"): 360, ("tabero", "tactile"): 314, ("robolab", "tactile"): 314}
SAMPLE_DECISION = 9  # the decision whose two raw records the page prints side by side

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output_root", type=Path)
parser.add_argument("prompts", type=Path)
parser.add_argument("site_dir", type=Path)
parser.add_argument("--title", default="TacClaw Rollout 浏览")
parser.add_argument("--case", type=Path, help="HTML fragment shown as the page's case-analysis section")
args = parser.parse_args()
root, site = args.output_root.resolve(), args.site_dir
media = site / "media"
shutil.rmtree(media, ignore_errors=True)
media.mkdir(parents=True)
FFMPEG = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def target_object(objects, instruction):
    """The first object named in the instruction ("pick up the X ...")."""
    text = instruction.lower()
    hits = []
    for name in objects:
        words = name.rsplit("_", 1)[0].replace("_", " ").lower()
        words = words.removeprefix("akita ")  # "akita_black_bowl" is "the black bowl"
        if words in text:
            hits.append((text.find(words), name))
    return min(hits)[1] if hits else None


def clip(src, dst, start, end, height, crf):
    subprocess.run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
         "-i", str(src), "-an", "-vf", f"scale=-2:{height}:flags=lanczos", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
        check=True,
    )  # fmt: skip


try:
    root_label = str(root.relative_to(ROLLOUT_ROOT))
except ValueError:
    root_label = str(root)
prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
data = {
    "title": args.title,
    "output_root": root_label,
    "generated": datetime.date.today().isoformat(),
    "policy": prompts["policy"],
    "benchmarks": {},
}
if args.case:
    data["case_html"] = args.case.read_text(encoding="utf-8")


def policy_fields(output):
    """What the decision table shows of one decision's policy calls (re-calls included)."""
    if "probabilities" in output[0]:  # JEMM
        return {"probs": [output[0]["probabilities"][t] for t in TOKENS], "confidence": output[0]["confidence"]}
    return {"raw": output[-1]["raw_output"], "calls": len(output), "latency_s": round(sum(a["latency_s"] for a in output), 2)}

for benchmark in BENCHMARKS:
    dataset = root / "lerobot_v3" / benchmark
    if not (dataset / "meta" / "info.json").is_file():
        continue
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    episodes_meta = {}
    for path in sorted((dataset / "meta" / "episodes").rglob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            episodes_meta[row["episode_index"]] = row
    frames_by_episode, file_of_episode = {}, {}
    for path in sorted((dataset / "data").rglob("*.parquet")):
        table = pq.read_table(
            path, columns=["episode_index", "frame_index", "action", "decision_index", "observation.tactile.summary"]
        ).to_pylist()
        for row in table:
            frames_by_episode.setdefault(row["episode_index"], []).append(row)
            file_of_episode.setdefault(row["episode_index"], path)

    episodes = []
    for entry in published(root, benchmark):
        index = entry["episode_index"]
        episode_dir, sh_dir = episode_dirs(root, benchmark, entry["source_id"])
        result = next(r for r in jsonl(episode_dir.parent / "results.jsonl") if r["seed"] == entry["seed"])
        decisions = jsonl(sh_dir / "actions.jsonl")
        calls = jsonl(episode_dir / "gpt_calls" / "calls.jsonl")
        objects0 = calls[0]["observation"]["objects"]
        target = target_object(objects0, result["instruction"])
        distances = (
            [round(math.dist(c["observation"]["position"], c["observation"]["objects"][target]), 4) for c in calls]
            if target
            else None
        )

        frames = sorted(frames_by_episode[index], key=lambda r: r["frame_index"])
        # Tabero opens at 1, RoboLab closes above 0.5 (README section 3).
        gripper_closed = [(f["action"][7] < 0.5) if benchmark == "tabero" else (f["action"][7] > 0.5) for f in frames]
        contact = [int(f["observation.tactile.summary"][0] > 0.5 or f["observation.tactile.summary"][6] > 0.5) for f in frames]

        meta = episodes_meta[index]
        videos = {}
        for short, key in VIDEO_KEYS.items():
            src = (
                dataset / "videos" / key / f"chunk-{meta[f'videos/{key}/chunk_index']:03d}"
                / f"file-{meta[f'videos/{key}/file_index']:03d}.mp4"
            )  # fmt: skip
            name = f"{benchmark}_ep{index}_{short}.mp4"
            clip(src, media / name, meta[f"videos/{key}/from_timestamp"], meta[f"videos/{key}/to_timestamp"],
                 VIDEO_HEIGHT[(benchmark, short)], 26 if short == "rgb" else 30)  # fmt: skip
            videos[short] = f"media/{name}"

        sample = min(SAMPLE_DECISION, len(decisions) - 1)
        sample_frame = next(f for f in frames if f["decision_index"] == sample)
        full_rows = [r for r in pq.read_table(file_of_episode[index]).to_pylist() if r["episode_index"] == index]
        full_row = next(r for r in full_rows if r["frame_index"] == sample_frame["frame_index"])

        episodes.append({
            "episode_index": index,
            "source_id": entry["source_id"],
            "task": result.get("task") or f"{result['task_suite']}/task_{result['task_id']:03d}",
            "instruction": result["instruction"],
            "success": result["success"],
            "end_reason": result["end_reason"],
            "steps": result["steps"],
            "max_episode_steps": result["max_episode_steps"],
            "episode_step_limit": result.get("episode_step_limit"),
            "time_limit_scale": result.get("time_limit_scale"),
            "max_decisions": result.get("max_decisions"),
            "decisions_count": result["decisions"],
            "wall_seconds": result["wall_seconds"],
            "policy_seconds": result.get("policy_seconds", result.get("gpt_seconds")),
            "fps": fps,
            "videos": videos,
            "target": target,
            "objects": objects0,
            "start_position": calls[0]["observation"]["position"],
            "distances": distances,
            # Height of the object to pick before each decision (privileged, from the call log).
            "target_z": [round(c["observation"]["objects"][target][2], 4) for c in calls] if target else None,
            "decisions": [
                {
                    "step": d["step"], "token": d["token"], "first_frame": d["first_frame"],
                    "frame_count": d["frame_count"], "travelled_m": d["travelled_m"],
                    "ee": [round(v, 4) for v in d["ee_pose"][:3]], "gripper_closed": d["gripper_closed"],
                    "gripper_width": d["gripper_width"], **policy_fields(d["policy_output"]),
                }
                for d in decisions
            ],  # fmt: skip
            "frames": {
                "x": [round(f["action"][0], 4) for f in frames],
                "y": [round(f["action"][1], 4) for f in frames],
                "z": [round(f["action"][2], 4) for f in frames],
                "closed": [int(c) for c in gripper_closed],
                "contact": contact,
                "decision": [f["decision_index"] for f in frames],
            },
            "sample_decision_record": decisions[sample],
            "sample_frame_row": {k: (v if not isinstance(v, float) else round(v, 5)) for k, v in full_row.items()},
            "showharness_dir": str(sh_dir.relative_to(root)),
            "metadata": json.loads((sh_dir / "metadata.json").read_text(encoding="utf-8")),
        })
        print(benchmark, index, entry["source_id"], len(frames), "frames", len(decisions), "decisions", flush=True)

    data["benchmarks"][benchmark] = {
        "fps": fps,
        "dataset_root": f"lerobot_v3/{benchmark}",
        "features": {k: {"dtype": v["dtype"], "shape": v["shape"], "names": v.get("names")} for k, v in info["features"].items()},
        "total_frames": info["total_frames"],
        "episodes": episodes,
    }

data["prompts"] = prompts
# The three images the policy saw on that decision 0.
for benchmark in data["benchmarks"]:
    prompts[benchmark]["images"] = {}
    for view in ("agentview", "wrist", "tactile"):
        name = f"{benchmark}_prompt_{view}.png"
        shutil.copy2(root / prompts[benchmark]["showharness_dir"] / view / "0000.png", media / name)
        prompts[benchmark]["images"][view] = f"media/{name}"

(site / "data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print("data.json", round((site / "data.json").stat().st_size / 1e6, 2), "MB")
