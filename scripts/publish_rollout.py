"""Append one rollout's staged frames to a LeRobot v3 dataset (README section 7.1).

Run through scripts/lerobot_python.sh (TacHarness's LeRobot 0.6 runtime). The episode
append itself is TacHarness's (tools/lerobot_v3_rollouts.py), so the layout, the field
names and the H.264/yuv420p videos match its rollout datasets. Ours: failed episodes are
published too (success false, reward all zero), ``action`` keeps the recorder's pose
layout, and every frame carries ``decision_index``. After the episode reads back, the
staged ``frames/`` are deleted unless --keep-frames.

Prints one JSON line: the dataset root, episode index and length.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import av
import lerobot_v3_rollouts as tacharness
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episode-dir", type=Path, required=True)
parser.add_argument("--dataset-root", type=Path, required=True)
parser.add_argument("--benchmark", choices=("robolab", "tabero"), required=True)
parser.add_argument("--source-id", required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--success", type=int, choices=(0, 1), required=True)
parser.add_argument("--keep-frames", action="store_true")

SCHEMA = "gpt_rollout.lerobot_v3.v1"
# The recorder's NPZ image key behind each TacHarness video feature.
NPZ_IMAGES = {
    "observation.images.rgb_dual_view": "observation.images.rgb_dual_view",
    "observation.images.tacumi_tactile_composite": "observation.images.tacumi_tactile_composite",
}


def _even(image: np.ndarray) -> np.ndarray:
    """yuv420p needs even sides; pad one black row/column if needed."""
    height, width = image.shape[:2]
    return np.pad(image, ((0, height % 2), (0, width % 2), (0, 0)))


class _Encoder:
    """H.264/yuv420p at the episode's fps, as tacharness._transcode_video writes."""

    def __init__(self, path: Path, fps: float) -> None:
        self.container = av.open(str(path), "w")
        self.stream = None
        self.rate = Fraction(fps).limit_denominator(1000)

    def add(self, image: np.ndarray) -> None:
        image = _even(np.ascontiguousarray(image[..., :3].astype(np.uint8)))
        if self.stream is None:
            self.stream = self.container.add_stream("libx264", rate=self.rate)
            self.stream.height, self.stream.width = image.shape[:2]
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"crf": "20", "preset": "veryfast"}
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        if self.stream is not None:
            for packet in self.stream.encode():
                self.container.mux(packet)
        self.container.close()


def _decision_index(episode_dir: Path, length: int) -> np.ndarray:
    calls = [
        json.loads(line)
        for line in (episode_dir / "gpt_calls" / "calls.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    index = np.full((length, 1), -1, dtype=np.int64)
    for call in calls:
        index[call["first_frame"] : call["first_frame"] + call["frame_count"], 0] = call["call_index"]
    covered = sum(call["frame_count"] for call in calls)
    if covered != length or (index < 0).any():
        raise ValueError(f"{episode_dir}: {length} frames but the calls cover {covered}")
    return index


def _features(info: dict, video_info: dict) -> dict:
    """TacHarness's RoboLab-style features, with the recorder's state/action layouts."""
    features = tacharness._dataset_features("robolab", video_info)
    for key in ("observation.state", "action"):
        names = list(info["features"][key]["names"])
        features[key] = {"dtype": "float32", "shape": (len(names),), "names": names}
    features["decision_index"] = {"dtype": "int64", "shape": (1,), "names": ["decision_index"]}
    return features


def _published(dataset_root: Path) -> dict[str, dict]:
    path = dataset_root / "meta" / "published_rollouts.jsonl"
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {row["source_id"]: row for row in rows}


def _verify(repo_id: str, dataset_root: Path, episode_index: int, length: int) -> None:
    """Read the episode back through LeRobot and decode its first and last frame."""
    dataset = LeRobotDataset(repo_id, root=dataset_root, episodes=[episode_index], video_backend="pyav")
    if len(dataset) != length:
        raise ValueError(f"episode {episode_index}: {len(dataset)} frames read back, expected {length}")
    for item in (dataset[0], dataset[length - 1]):
        for key in NPZ_IMAGES:
            if item[key].ndim != 3:
                raise ValueError(f"episode {episode_index}: {key} did not decode")
        if int(item["episode_index"]) != episode_index:
            raise ValueError(f"episode {episode_index}: read back episode {int(item['episode_index'])}")


def main() -> None:
    args = parser.parse_args()
    episode_dir = args.episode_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    info = json.loads((episode_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    summary = json.loads((episode_dir / "summary.json").read_text(encoding="utf-8"))
    frame_paths = sorted((episode_dir / "frames").glob("frame_*.npz"))
    length = len(frame_paths)
    if length == 0:
        raise ValueError(f"{episode_dir}: no staged frames")
    fps = float(info["fps"])
    tactile_arms = tuple(info.get("tactile_arms") or ("left",))
    decision_index = _decision_index(episode_dir, length)

    # Lock the dataset: several Isaac workers may finish episodes at the same time.
    lock = open(dataset_root.parent / f".{dataset_root.name}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    already = _published(dataset_root).get(args.source_id)
    if already is not None:
        result = {**already, "skipped": "already published"}
    else:
        source = f"gpt_rollout_{args.benchmark}"
        tacharness.SOURCE_FPS[source] = fps  # _append_preencoded_episode reads the fps from here
        repo_id = f"local/gpt-rollouts-{args.benchmark}"
        temp = Path(tempfile.mkdtemp(prefix="publish_rollout_"))
        try:
            encoders = {key: _Encoder(temp / f"{key.replace('.', '_')}.mp4", fps) for key in NPZ_IMAGES}
            states, actions, tactile = [], [], []
            for path in frame_paths:
                with np.load(path, allow_pickle=False) as frame:
                    states.append(np.asarray(frame["observation.state"], dtype=np.float32).reshape(-1))
                    actions.append(np.asarray(frame["action"], dtype=np.float32).reshape(-1))
                    tactile.append(tacharness._tactile_summary(frame, tactile_arms))
                    for key, npz_key in NPZ_IMAGES.items():
                        encoders[key].add(np.asarray(frame[npz_key]))
            for encoder in encoders.values():
                encoder.close()
            videos = {key: temp / f"{key.replace('.', '_')}.mp4" for key in NPZ_IMAGES}

            candidate = tacharness.Candidate(
                source=source,
                source_id=args.source_id,
                run_dir=episode_dir,
                task=args.task,
                task_display=args.task,
                seed=args.seed,
                success=bool(args.success),
                # No separate qualification audit here: qualified mirrors success.
                qualified=bool(args.success),
                created_at=str(info.get("created_at", "")),
                summary=summary,
                info=info,
                publication={},
            )
            values = {
                "observation.state": np.stack(states),
                "action": np.stack(actions),
                "observation.tactile.summary": np.stack(tactile).astype(np.float32),
                "decision_index": decision_index,
                **tacharness._common_values(candidate, length),
            }
            payload = tacharness.EpisodePayload(candidate=candidate, values=values, videos=videos, length=length)
            tacharness._align_payload(payload)

            if (dataset_root / "meta" / "info.json").is_file():
                dataset = LeRobotDataset.resume(repo_id=repo_id, root=dataset_root, encoder_threads=2)
            else:
                video_info = {key: tacharness._video_info(path) for key, path in videos.items()}
                dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    root=dataset_root,
                    fps=fps,
                    features=_features(info, video_info),
                    robot_type=str(info.get("robot_type", "")),
                    use_videos=True,
                    metadata_buffer_size=1,
                    encoder_threads=2,
                )
            try:
                episode_index = tacharness._append_preencoded_episode(dataset, payload, dataset_root)
            finally:
                dataset.finalize()
        finally:
            shutil.rmtree(temp, ignore_errors=True)

        info_path = dataset_root / "meta" / "info.json"
        dataset_info = json.loads(info_path.read_text(encoding="utf-8"))
        dataset_info["gpt_rollout"] = {
            "schema": SCHEMA,
            "benchmark": args.benchmark,
            "fps": fps,
            "action_layout": list(info["features"]["action"]["names"]),
            "decision_index": "the policy call (README section 7) that produced this frame",
            "frame_contract": "frame k holds action k and the observation after it",
            "published_rollouts": "meta/published_rollouts.jsonl",
            "video_codec": "h264",
            "video_pixel_format": "yuv420p",
            "end_effector_profile": info.get("end_effector_profile"),
            "tactile_encoding": tacharness.TACTILE_ENCODING,
        }
        info_path.write_text(json.dumps(dataset_info, indent=4, ensure_ascii=False), encoding="utf-8")
        _verify(repo_id, dataset_root, episode_index, length)

        result = {
            "source_id": args.source_id,
            "episode_index": int(episode_index),
            "length": length,
            "success": bool(args.success),
            "seed": args.seed,
            "published_at": datetime.now(timezone.utc).isoformat(),
        }
        with (dataset_root / "meta" / "published_rollouts.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    fcntl.flock(lock, fcntl.LOCK_UN)

    if not args.keep_frames:
        shutil.rmtree(episode_dir / "frames")
    print(json.dumps({**result, "dataset_root": str(dataset_root), "frames_deleted": not args.keep_frames}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[publish_rollout] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
