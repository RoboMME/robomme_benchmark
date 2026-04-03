#!/usr/bin/env python3
"""Export front_rgb videos with waypoint-derived keyframes highlighted."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np

DEFAULT_INPUT_DIR = Path("/data/hongzefu/data-0306")
DEFAULT_OUTPUT_DIR = Path("runs/keyframes_waypoint")
DEFAULT_EPISODES_PER_TASK = 3
DEFAULT_FPS = 30.0

_H5_RE = re.compile(r"^record_dataset_(.+)\.h5$")
_EPISODE_RE = re.compile(r"^episode_(\d+)$")
_TIMESTEP_RE = re.compile(r"^timestep_(\d+)(?:_dup(\d+))?$")


@dataclass(frozen=True)
class KeyframeDecision:
    frame_index: int
    timestep_key: str
    is_video_demo: bool
    waypoint_action: np.ndarray | None
    is_keyframe: bool
    keyframe_index: int | None


@dataclass(frozen=True)
class EpisodeSummary:
    task_name: str
    episode: int
    frame_count: int
    valid_waypoint_count: int
    keyframe_count: int
    output_path: Path


def task_name_from_h5_path(h5_path: Path) -> str:
    match = _H5_RE.match(h5_path.name)
    if match is None:
        raise ValueError(f"unexpected h5 filename: {h5_path}")
    return match.group(1)


def list_h5_paths(input_dir: Path, tasks: set[str] | None = None) -> list[Path]:
    paths: list[tuple[str, Path]] = []
    for path in sorted(input_dir.glob("record_dataset_*.h5")):
        task_name = task_name_from_h5_path(path)
        if tasks is not None and task_name not in tasks:
            continue
        paths.append((task_name, path))
    return [path for _, path in sorted(paths, key=lambda item: item[0])]


def list_episode_indices(h5: h5py.File) -> list[int]:
    indices: list[int] = []
    for key in h5.keys():
        match = _EPISODE_RE.match(key)
        if match is not None:
            indices.append(int(match.group(1)))
    return sorted(indices)


def iter_sorted_timestep_keys(episode_group: h5py.Group) -> list[str]:
    keyed: list[tuple[int, int, str]] = []
    for name in episode_group.keys():
        match = _TIMESTEP_RE.match(name)
        if match is None:
            continue
        timestep = int(match.group(1))
        dup = int(match.group(2)) if match.group(2) is not None else 0
        keyed.append((timestep, dup, name))
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [name for _, _, name in keyed]


def _read_bool(group: h5py.Group, path: str, default: bool = False) -> bool:
    if path not in group:
        return default
    value = np.asarray(group[path][()])
    flat = value.reshape(-1)
    if flat.size == 0:
        return default
    return bool(flat[0])


def read_waypoint_action(timestep_group: h5py.Group) -> np.ndarray | None:
    action_group = timestep_group.get("action")
    if action_group is None or not isinstance(action_group, h5py.Group):
        return None
    if "waypoint_action" not in action_group:
        return None
    try:
        waypoint_action = np.asarray(action_group["waypoint_action"][()]).flatten()
    except (TypeError, ValueError):
        return None
    if waypoint_action.shape != (7,):
        return None
    if not np.all(np.isfinite(waypoint_action)):
        return None
    return waypoint_action.astype(np.float64, copy=False)


def scan_episode_keyframes(episode_group: h5py.Group) -> tuple[list[KeyframeDecision], int]:
    decisions: list[KeyframeDecision] = []
    valid_waypoint_count = 0
    keyframe_count = 0
    prev_waypoint_action: np.ndarray | None = None

    for frame_index, timestep_key in enumerate(iter_sorted_timestep_keys(episode_group)):
        timestep_group = episode_group[timestep_key]
        if not isinstance(timestep_group, h5py.Group):
            continue

        is_video_demo = False
        info_group = timestep_group.get("info")
        if isinstance(info_group, h5py.Group):
            is_video_demo = _read_bool(info_group, "is_video_demo", default=False)

        waypoint_action = read_waypoint_action(timestep_group)
        is_keyframe = False
        keyframe_index: int | None = None

        if waypoint_action is not None:
            valid_waypoint_count += 1
            if prev_waypoint_action is None or not np.array_equal(waypoint_action, prev_waypoint_action):
                keyframe_count += 1
                keyframe_index = keyframe_count
                is_keyframe = True
                prev_waypoint_action = waypoint_action.copy()

        decisions.append(
            KeyframeDecision(
                frame_index=frame_index,
                timestep_key=timestep_key,
                is_video_demo=is_video_demo,
                waypoint_action=waypoint_action,
                is_keyframe=is_keyframe,
                keyframe_index=keyframe_index,
            )
        )

    return decisions, valid_waypoint_count


def _to_uint8_hwc(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.dtype in (np.float32, np.float64):
        finite = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        if finite.size and float(np.nanmax(finite)) <= 1.0 + 1e-6:
            image = (finite * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image = finite.clip(0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[0] < min(image.shape[1], image.shape[2]):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def _front_rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    rgb = _to_uint8_hwc(frame)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"front_rgb must be HWC RGB with 3 channels, got shape={rgb.shape}")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _put_text_block(frame: np.ndarray, lines: list[str], start_xy: tuple[int, int], scale: float) -> None:
    x, y = start_xy
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x + 1, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += int(28 * scale + 10)


def _format_waypoint_lines(waypoint_action: np.ndarray) -> list[str]:
    xyz = ", ".join(f"{value:.3f}" for value in waypoint_action[:3])
    rpyg = ", ".join(f"{value:.3f}" for value in waypoint_action[3:])
    return [f"wp xyz=[{xyz}]", f"wp rpyg=[{rpyg}]"]


def render_frame(
    front_rgb: np.ndarray,
    *,
    task_name: str,
    episode: int,
    decision: KeyframeDecision,
) -> np.ndarray:
    frame = _front_rgb_to_bgr(front_rgb)
    base_lines = [
        f"task={task_name}",
        f"episode={episode}  frame={decision.frame_index}",
        f"timestep={decision.timestep_key}",
        f"is_video_demo={decision.is_video_demo}",
    ]
    _put_text_block(frame, base_lines, start_xy=(10, 26), scale=0.55)

    if decision.is_keyframe:
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 8)
        keyframe_lines = [f"KEYFRAME #{decision.keyframe_index}"]
        if decision.waypoint_action is not None:
            keyframe_lines.extend(_format_waypoint_lines(decision.waypoint_action))
        _put_text_block(frame, keyframe_lines, start_xy=(10, height - 90), scale=0.7)

    return frame


def output_video_path(output_dir: Path, task_name: str, episode: int) -> Path:
    return output_dir / f"{task_name}_ep{episode}_front_rgb_keyframes.mp4"


class EpisodeVideoWriter:
    def __init__(self, out_path: Path, fps: float, frame_size: tuple[int, int]):
        self.out_path = out_path
        self.fps = fps
        self.frame_size = frame_size
        self._backend = ""
        self._imageio_writer = None
        self._cv2_writer: cv2.VideoWriter | None = None

        if out_path.exists():
            out_path.unlink()

        try:
            self._imageio_writer = imageio.get_writer(
                str(out_path),
                fps=fps,
                codec="libx264",
                format="FFMPEG",
                macro_block_size=None,
            )
            self._backend = "imageio/libx264"
            return
        except Exception:
            self._imageio_writer = None

        width, height = frame_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._cv2_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not self._cv2_writer.isOpened():
            raise RuntimeError(f"failed to open a video writer for {out_path}")
        self._backend = "cv2/mp4v"
        print(
            f"[WARN] falling back to {self._backend} for {out_path.name}; "
            "some previewers may not load this codec"
        )

    @property
    def backend(self) -> str:
        return self._backend

    def write(self, frame_bgr: np.ndarray) -> None:
        height, width = frame_bgr.shape[:2]
        expected_width, expected_height = self.frame_size
        if (width, height) != (expected_width, expected_height):
            raise ValueError(
                f"frame size mismatch: got {(width, height)} expected {(expected_width, expected_height)}"
            )

        if self._imageio_writer is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self._imageio_writer.append_data(frame_rgb)
            return

        if self._cv2_writer is None:
            raise RuntimeError("video writer is not initialized")
        self._cv2_writer.write(frame_bgr)

    def close(self) -> None:
        if self._imageio_writer is not None:
            self._imageio_writer.close()
            self._imageio_writer = None
        if self._cv2_writer is not None:
            self._cv2_writer.release()
            self._cv2_writer = None


def export_episode_video(
    h5_path: Path,
    *,
    task_name: str,
    episode: int,
    output_dir: Path,
    fps: float,
) -> EpisodeSummary:
    with h5py.File(h5_path, "r") as h5:
        episode_key = f"episode_{episode}"
        if episode_key not in h5:
            raise KeyError(f"missing {episode_key} in {h5_path}")
        episode_group = h5[episode_key]
        if not isinstance(episode_group, h5py.Group):
            raise TypeError(f"{episode_key} is not a group in {h5_path}")

        decisions, valid_waypoint_count = scan_episode_keyframes(episode_group)
        if not decisions:
            raise ValueError(f"{episode_key} has no timestep groups in {h5_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_video_path(output_dir, task_name, episode)
        writer: EpisodeVideoWriter | None = None

        try:
            for decision in decisions:
                timestep_group = episode_group[decision.timestep_key]
                obs_group = timestep_group.get("obs")
                if obs_group is None or not isinstance(obs_group, h5py.Group):
                    raise KeyError(f"{episode_key}/{decision.timestep_key} missing obs group")
                if "front_rgb" not in obs_group:
                    raise KeyError(f"{episode_key}/{decision.timestep_key} missing obs/front_rgb")

                front_rgb = np.asarray(obs_group["front_rgb"][()])
                frame = render_frame(front_rgb, task_name=task_name, episode=episode, decision=decision)

                if writer is None:
                    height, width = frame.shape[:2]
                    writer = EpisodeVideoWriter(out_path, fps, (width, height))
                writer.write(frame)
        finally:
            if writer is not None:
                writer.close()

    keyframe_count = sum(1 for decision in decisions if decision.is_keyframe)
    return EpisodeSummary(
        task_name=task_name,
        episode=episode,
        frame_count=len(decisions),
        valid_waypoint_count=valid_waypoint_count,
        keyframe_count=keyframe_count,
        output_path=out_path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episodes-per-task", type=int, default=DEFAULT_EPISODES_PER_TASK)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional task names, e.g. BinFill InsertPeg",
    )
    args = parser.parse_args(argv)
    if args.episodes_per_task <= 0:
        parser.error("--episodes-per-task must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input_dir.exists():
        print(f"[ERROR] input dir does not exist: {args.input_dir}", file=sys.stderr)
        return 1

    requested_tasks = set(args.tasks) if args.tasks else None
    h5_paths = list_h5_paths(args.input_dir, tasks=requested_tasks)
    if not h5_paths:
        print(f"[ERROR] no matching h5 files found in {args.input_dir}", file=sys.stderr)
        return 1

    total_success = 0
    failures: list[str] = []
    seen_tasks = {task_name_from_h5_path(path) for path in h5_paths}
    if requested_tasks is not None:
        missing = sorted(requested_tasks - seen_tasks)
        for task_name in missing:
            print(f"[WARN] requested task not found: {task_name}")

    for h5_path in h5_paths:
        task_name = task_name_from_h5_path(h5_path)
        try:
            with h5py.File(h5_path, "r") as h5:
                episode_indices = list_episode_indices(h5)
        except Exception as exc:
            failures.append(f"{task_name}: failed to inspect h5 ({type(exc).__name__}: {exc})")
            continue

        selected_episodes = episode_indices[: args.episodes_per_task]
        if not selected_episodes:
            failures.append(f"{task_name}: no episodes found")
            continue
        if len(selected_episodes) < args.episodes_per_task:
            print(
                f"[WARN] {task_name}: requested {args.episodes_per_task} episodes, "
                f"found only {len(selected_episodes)}"
            )

        for episode in selected_episodes:
            try:
                summary = export_episode_video(
                    h5_path,
                    task_name=task_name,
                    episode=episode,
                    output_dir=args.output_dir,
                    fps=args.fps,
                )
            except Exception as exc:
                failures.append(
                    f"{task_name} episode {episode}: {type(exc).__name__}: {exc}"
                )
                continue

            total_success += 1
            print(
                f"[OK] task={summary.task_name} episode={summary.episode} "
                f"frames={summary.frame_count} valid_waypoints={summary.valid_waypoint_count} "
                f"keyframes={summary.keyframe_count} output={summary.output_path}"
            )

    print(
        f"[SUMMARY] videos_written={total_success} "
        f"failures={len(failures)} output_dir={args.output_dir}"
    )
    for failure in failures:
        print(f"[FAIL] {failure}")

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
