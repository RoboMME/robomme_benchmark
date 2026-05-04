#!/usr/bin/env python3
"""Predict joint/gripper keyframes and compare them against waypoint keyframes."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

DEFAULT_INPUT_DIR = Path("/data/hongzefu/data-0306")
DEFAULT_OUTPUT_DIR = Path("runs/keyframes_jointangle_compare")
DEFAULT_EPISODES_PER_TASK = 3
DEFAULT_FPS = 30.0
DEFAULT_KEYFRAME_VIDEO_FPS = 1.0
DEFAULT_MOTION_LOW = 0.002
DEFAULT_MOTION_HIGH = 0.005
DEFAULT_MATCH_TOLERANCE = 2

PREDICTED_COLOR_BGR = (0, 165, 255)
NATIVE_COLOR_BGR = (0, 200, 0)
PREDICTED_COLOR_RGB = "#ff8c00"
NATIVE_COLOR_RGB = "#1a9f3f"
INFO_PANEL_WIDTH = 320
INFO_PANEL_BG_BGR = (0, 0, 0)
INFO_PANEL_TEXT_BGR = (255, 255, 255)

_H5_RE = re.compile(r"^record_dataset_(.+)\.h5$")
_EPISODE_RE = re.compile(r"^episode_(\d+)$")
_TIMESTEP_RE = re.compile(r"^timestep_(\d+)(?:_dup(\d+))?$")


@dataclass(frozen=True)
class FrameRecord:
    frame_index: int
    timestep_key: str
    is_video_demo: bool
    waypoint_action: np.ndarray | None
    joint_action: np.ndarray | None
    is_gripper_close: bool | None


@dataclass(frozen=True)
class EpisodeScan:
    frames: list[FrameRecord]
    valid_waypoint_count: int
    valid_joint_count: int
    predicted_keyframes: list[int]
    native_keyframes: list[int]


@dataclass(frozen=True)
class MatchPair:
    predicted_frame_index: int
    native_frame_index: int
    distance: int


@dataclass(frozen=True)
class ComparisonResult:
    matched_pairs: list[MatchPair]
    tp: int
    fp: int
    fn: int


@dataclass(frozen=True)
class EpisodeSummary:
    task_name: str
    episode: int
    frame_count: int
    predicted_keyframe_count: int
    native_keyframe_count: int
    timeline_png_path: Path
    timeline_json_path: Path
    compare_video_path: Path
    predicted_keyframe_video_path: Path | None
    native_keyframe_video_path: Path | None


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


def read_joint_action(timestep_group: h5py.Group) -> np.ndarray | None:
    action_group = timestep_group.get("action")
    if action_group is None or not isinstance(action_group, h5py.Group):
        return None
    if "joint_action" not in action_group:
        return None
    try:
        joint_action = np.asarray(action_group["joint_action"][()]).astype(np.float64).flatten()
    except (TypeError, ValueError):
        return None
    if joint_action.shape != (8,):
        return None
    if not np.all(np.isfinite(joint_action)):
        return None
    return joint_action


def read_is_gripper_close(timestep_group: h5py.Group) -> bool | None:
    obs_group = timestep_group.get("obs")
    if obs_group is None or not isinstance(obs_group, h5py.Group):
        return None
    if "is_gripper_close" not in obs_group:
        return None
    value = np.asarray(obs_group["is_gripper_close"][()])
    flat = value.reshape(-1)
    if flat.size != 1:
        return None
    return bool(flat[0])


def build_frame_records(episode_group: h5py.Group) -> tuple[list[FrameRecord], int, int]:
    frames: list[FrameRecord] = []
    valid_waypoint_count = 0
    valid_joint_count = 0

    for frame_index, timestep_key in enumerate(iter_sorted_timestep_keys(episode_group)):
        timestep_group = episode_group[timestep_key]
        if not isinstance(timestep_group, h5py.Group):
            continue

        info_group = timestep_group.get("info")
        is_video_demo = (
            _read_bool(info_group, "is_video_demo", default=False)
            if isinstance(info_group, h5py.Group)
            else False
        )
        waypoint_action = read_waypoint_action(timestep_group)
        joint_action = read_joint_action(timestep_group)
        is_gripper_close = read_is_gripper_close(timestep_group)

        if waypoint_action is not None:
            valid_waypoint_count += 1
        if joint_action is not None:
            valid_joint_count += 1

        frames.append(
            FrameRecord(
                frame_index=frame_index,
                timestep_key=timestep_key,
                is_video_demo=is_video_demo,
                waypoint_action=waypoint_action,
                joint_action=joint_action,
                is_gripper_close=is_gripper_close,
            )
        )

    return frames, valid_waypoint_count, valid_joint_count


def scan_native_keyframes(frames: list[FrameRecord]) -> list[int]:
    native_keyframes: list[int] = []
    prev_waypoint_action: np.ndarray | None = None
    for record in frames:
        if record.waypoint_action is None:
            continue
        if prev_waypoint_action is None or not np.array_equal(record.waypoint_action, prev_waypoint_action):
            native_keyframes.append(record.frame_index)
            prev_waypoint_action = record.waypoint_action.copy()
    return native_keyframes


def _compute_motion_deltas(frames: list[FrameRecord]) -> list[float | None]:
    deltas: list[float | None] = [None] * len(frames)
    for index in range(1, len(frames)):
        prev_joint = frames[index - 1].joint_action
        curr_joint = frames[index].joint_action
        if prev_joint is None or curr_joint is None:
            continue
        deltas[index] = float(np.max(np.abs(curr_joint[:7] - prev_joint[:7])))
    return deltas


def scan_predicted_keyframes(
    frames: list[FrameRecord],
    *,
    motion_low: float,
    motion_high: float,
) -> list[int]:
    if not frames:
        return []

    deltas = _compute_motion_deltas(frames)
    boundary_candidates: set[int] = set()
    gripper_candidates: set[int] = set()

    for index in range(1, len(frames) - 1):
        curr_delta = deltas[index]
        next_delta = deltas[index + 1]
        if curr_delta is None or next_delta is None:
            continue
        if curr_delta <= motion_low and next_delta > motion_high:
            boundary_candidates.add(index)
        if curr_delta > motion_high and next_delta <= motion_low:
            boundary_candidates.add(index + 1)

    for index in range(1, len(frames)):
        prev_gripper = frames[index - 1].is_gripper_close
        curr_gripper = frames[index].is_gripper_close
        if prev_gripper is None or curr_gripper is None:
            continue
        if prev_gripper != curr_gripper:
            gripper_candidates.add(index)

    filtered_boundaries = {
        index
        for index in boundary_candidates
        if all(abs(index - gripper_index) > 1 for gripper_index in gripper_candidates)
    }
    return sorted(filtered_boundaries | gripper_candidates)


def scan_episode_keyframes(
    episode_group: h5py.Group,
    *,
    motion_low: float = DEFAULT_MOTION_LOW,
    motion_high: float = DEFAULT_MOTION_HIGH,
) -> EpisodeScan:
    frames, valid_waypoint_count, valid_joint_count = build_frame_records(episode_group)
    predicted_keyframes = scan_predicted_keyframes(
        frames,
        motion_low=motion_low,
        motion_high=motion_high,
    )
    native_keyframes = scan_native_keyframes(frames)
    return EpisodeScan(
        frames=frames,
        valid_waypoint_count=valid_waypoint_count,
        valid_joint_count=valid_joint_count,
        predicted_keyframes=predicted_keyframes,
        native_keyframes=native_keyframes,
    )


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


def _put_panel_text_block(
    frame: np.ndarray,
    lines: list[str],
    *,
    start_xy: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = INFO_PANEL_TEXT_BGR,
) -> int:
    x, y = start_xy
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
        y += int(28 * scale + 10)
    return y


def _render_info_panel(
    *,
    height: int,
    lane_title: str,
    task_name: str,
    episode: int,
    record: FrameRecord,
    show_gripper_state: bool,
    keyframe_label: str | None,
    accent_color: tuple[int, int, int],
) -> np.ndarray:
    panel = np.full((height, INFO_PANEL_WIDTH, 3), INFO_PANEL_BG_BGR, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (INFO_PANEL_WIDTH - 1, height - 1), accent_color, 6)
    cv2.rectangle(panel, (0, 0), (11, height - 1), accent_color, -1)

    base_lines = [
        lane_title,
        "",
        f"task={task_name}",
        f"episode={episode}",
        f"frame={record.frame_index}",
        f"timestep={record.timestep_key}",
    ]
    if show_gripper_state:
        gripper_text = (
            str(record.is_gripper_close)
            if record.is_gripper_close is not None
            else "unknown"
        )
        base_lines.append(f"is_gripper_close={gripper_text}")

    _put_panel_text_block(panel, base_lines, start_xy=(26, 34), scale=0.6)

    if keyframe_label is not None:
        _put_panel_text_block(
            panel,
            [keyframe_label],
            start_xy=(26, max(height - 24, 34)),
            scale=0.72,
            color=accent_color,
        )

    return panel


def _keyframe_number_map(indices: list[int]) -> dict[int, int]:
    return {frame_index: order for order, frame_index in enumerate(indices, start=1)}


def render_lane_frame(
    front_rgb: np.ndarray,
    *,
    task_name: str,
    episode: int,
    record: FrameRecord,
    lane_title: str,
    is_keyframe: bool,
    keyframe_index: int | None,
    border_color: tuple[int, int, int],
    show_gripper_state: bool,
) -> np.ndarray:
    frame = _front_rgb_to_bgr(front_rgb)
    keyframe_label: str | None = None
    if is_keyframe and keyframe_index is not None:
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), border_color, 8)
        label = "PRED KF" if show_gripper_state else "WAYPOINT KF"
        keyframe_label = f"{label} #{keyframe_index}"

    info_panel = _render_info_panel(
        height=frame.shape[0],
        lane_title=lane_title,
        task_name=task_name,
        episode=episode,
        record=record,
        show_gripper_state=show_gripper_state,
        keyframe_label=keyframe_label,
        accent_color=border_color,
    )
    return np.hstack([info_panel, frame])


def render_comparison_frame(
    front_rgb: np.ndarray,
    *,
    task_name: str,
    episode: int,
    record: FrameRecord,
    predicted_keyframes: dict[int, int],
    native_keyframes: dict[int, int],
) -> np.ndarray:
    predicted_frame = render_lane_frame(
        front_rgb,
        task_name=task_name,
        episode=episode,
        record=record,
        lane_title="Predicted (joint + gripper)",
        is_keyframe=record.frame_index in predicted_keyframes,
        keyframe_index=predicted_keyframes.get(record.frame_index),
        border_color=PREDICTED_COLOR_BGR,
        show_gripper_state=True,
    )
    native_frame = render_lane_frame(
        front_rgb,
        task_name=task_name,
        episode=episode,
        record=record,
        lane_title="Native (waypoint)",
        is_keyframe=record.frame_index in native_keyframes,
        keyframe_index=native_keyframes.get(record.frame_index),
        border_color=NATIVE_COLOR_BGR,
        show_gripper_state=False,
    )
    return np.vstack([predicted_frame, native_frame])


def output_timeline_png_path(output_dir: Path, task_name: str, episode: int) -> Path:
    return output_dir / f"{task_name}_ep{episode}_keyframe_timeline.png"


def output_timeline_json_path(output_dir: Path, task_name: str, episode: int) -> Path:
    return output_dir / f"{task_name}_ep{episode}_keyframe_timeline.json"


def output_compare_video_path(output_dir: Path, task_name: str, episode: int) -> Path:
    return output_dir / f"{task_name}_ep{episode}_front_rgb_keyframe_compare.mp4"


def output_predicted_keyframe_video_path(output_dir: Path, task_name: str, episode: int) -> Path:
    return output_dir / f"{task_name}_ep{episode}_front_rgb_predicted_keyframes.mp4"


def output_native_keyframe_video_path(output_dir: Path, task_name: str, episode: int) -> Path:
    return output_dir / f"{task_name}_ep{episode}_front_rgb_native_keyframes.mp4"


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


def match_keyframes(
    predicted_keyframes: list[int],
    native_keyframes: list[int],
    *,
    tolerance: int,
) -> ComparisonResult:
    remaining_native = set(native_keyframes)
    matched_pairs: list[MatchPair] = []

    for predicted_index in predicted_keyframes:
        candidates = [
            native_index
            for native_index in remaining_native
            if abs(predicted_index - native_index) <= tolerance
        ]
        if not candidates:
            continue
        native_index = min(candidates, key=lambda index: (abs(predicted_index - index), index))
        remaining_native.remove(native_index)
        matched_pairs.append(
            MatchPair(
                predicted_frame_index=predicted_index,
                native_frame_index=native_index,
                distance=abs(predicted_index - native_index),
            )
        )

    tp = len(matched_pairs)
    return ComparisonResult(
        matched_pairs=matched_pairs,
        tp=tp,
        fp=len(predicted_keyframes) - tp,
        fn=len(native_keyframes) - tp,
    )


def save_timeline_plot(
    out_path: Path,
    *,
    task_name: str,
    episode: int,
    frame_count: int,
    predicted_keyframes: list[int],
    native_keyframes: list[int],
) -> None:
    fig_width = min(max(frame_count / 35.0, 8.0), 20.0)
    fig, ax = plt.subplots(figsize=(fig_width, 3.2), dpi=150)

    xmax = max(frame_count - 1, 1)
    ax.hlines(1, 0, xmax, colors="#c7c7c7", linewidth=2)
    ax.hlines(0, 0, xmax, colors="#c7c7c7", linewidth=2)
    if predicted_keyframes:
        ax.scatter(
            predicted_keyframes,
            np.full(len(predicted_keyframes), 1.0),
            color=PREDICTED_COLOR_RGB,
            s=28,
            label="Predicted (joint + gripper)",
            zorder=3,
        )
    if native_keyframes:
        ax.scatter(
            native_keyframes,
            np.full(len(native_keyframes), 0.0),
            color=NATIVE_COLOR_RGB,
            s=28,
            label="Native (waypoint)",
            zorder=3,
        )

    ax.set_xlim(-1, xmax + 1)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([1, 0], labels=["Predicted", "Native"])
    ax.set_xlabel("Frame Index")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_title(
        f"{task_name} / episode {episode} / frames={frame_count} / "
        f"predicted={len(predicted_keyframes)} / native={len(native_keyframes)}"
    )
    if predicted_keyframes or native_keyframes:
        ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_timeline_json_payload(
    *,
    task_name: str,
    episode: int,
    scan: EpisodeScan,
    compare_fps: float,
    keyframe_video_fps: float,
    motion_low: float,
    motion_high: float,
    match_tolerance: int,
    comparison: ComparisonResult,
) -> dict[str, object]:
    predicted_timestep_keys = [
        scan.frames[frame_index].timestep_key for frame_index in scan.predicted_keyframes
    ]
    native_timestep_keys = [
        scan.frames[frame_index].timestep_key for frame_index in scan.native_keyframes
    ]
    matched_pairs = [
        {
            "predicted_frame_index": pair.predicted_frame_index,
            "predicted_timestep_key": scan.frames[pair.predicted_frame_index].timestep_key,
            "native_frame_index": pair.native_frame_index,
            "native_timestep_key": scan.frames[pair.native_frame_index].timestep_key,
            "distance": pair.distance,
        }
        for pair in comparison.matched_pairs
    ]
    return {
        "task_name": task_name,
        "episode": episode,
        "frame_count": len(scan.frames),
        "params": {
            "compare_fps": compare_fps,
            "keyframe_video_fps": keyframe_video_fps,
            "motion_low": motion_low,
            "motion_high": motion_high,
            "match_tolerance": match_tolerance,
        },
        "predicted": {
            "count": len(scan.predicted_keyframes),
            "frame_indices": scan.predicted_keyframes,
            "timestep_keys": predicted_timestep_keys,
        },
        "native": {
            "count": len(scan.native_keyframes),
            "frame_indices": scan.native_keyframes,
            "timestep_keys": native_timestep_keys,
        },
        "comparison": {
            "matched_pairs": matched_pairs,
            "tp": comparison.tp,
            "fp": comparison.fp,
            "fn": comparison.fn,
        },
    }


def _load_front_rgb_frame(
    *,
    episode_group: h5py.Group,
    episode_key: str,
    record: FrameRecord,
) -> np.ndarray:
    timestep_group = episode_group[record.timestep_key]
    obs_group = timestep_group.get("obs")
    if obs_group is None or not isinstance(obs_group, h5py.Group):
        raise KeyError(f"{episode_key}/{record.timestep_key} missing obs group")
    if "front_rgb" not in obs_group:
        raise KeyError(f"{episode_key}/{record.timestep_key} missing obs/front_rgb")
    return np.asarray(obs_group["front_rgb"][()])


def export_keyframe_video(
    *,
    episode_group: h5py.Group,
    episode_key: str,
    scan: EpisodeScan,
    keyframe_indices: list[int],
    keyframe_numbers: dict[int, int],
    out_path: Path,
    fps: float,
    task_name: str,
    episode: int,
    lane_title: str,
    border_color: tuple[int, int, int],
    show_gripper_state: bool,
) -> Path | None:
    if not keyframe_indices:
        if out_path.exists():
            out_path.unlink()
        return None

    writer: EpisodeVideoWriter | None = None
    try:
        for frame_index in keyframe_indices:
            record = scan.frames[frame_index]
            front_rgb = _load_front_rgb_frame(
                episode_group=episode_group,
                episode_key=episode_key,
                record=record,
            )
            frame = render_lane_frame(
                front_rgb,
                task_name=task_name,
                episode=episode,
                record=record,
                lane_title=lane_title,
                is_keyframe=True,
                keyframe_index=keyframe_numbers[frame_index],
                border_color=border_color,
                show_gripper_state=show_gripper_state,
            )
            if writer is None:
                height, width = frame.shape[:2]
                writer = EpisodeVideoWriter(out_path, fps, (width, height))
            writer.write(frame)
    finally:
        if writer is not None:
            writer.close()

    return out_path


def export_episode_outputs(
    h5_path: Path,
    *,
    task_name: str,
    episode: int,
    output_dir: Path,
    fps: float,
    keyframe_video_fps: float,
    motion_low: float,
    motion_high: float,
    match_tolerance: int,
) -> EpisodeSummary:
    with h5py.File(h5_path, "r") as h5:
        episode_key = f"episode_{episode}"
        if episode_key not in h5:
            raise KeyError(f"missing {episode_key} in {h5_path}")
        episode_group = h5[episode_key]
        if not isinstance(episode_group, h5py.Group):
            raise TypeError(f"{episode_key} is not a group in {h5_path}")

        scan = scan_episode_keyframes(
            episode_group,
            motion_low=motion_low,
            motion_high=motion_high,
        )
        if not scan.frames:
            raise ValueError(f"{episode_key} has no timestep groups in {h5_path}")

        comparison = match_keyframes(
            scan.predicted_keyframes,
            scan.native_keyframes,
            tolerance=match_tolerance,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        timeline_png_path = output_timeline_png_path(output_dir, task_name, episode)
        timeline_json_path = output_timeline_json_path(output_dir, task_name, episode)
        compare_video_path = output_compare_video_path(output_dir, task_name, episode)
        predicted_keyframe_video_path = output_predicted_keyframe_video_path(
            output_dir, task_name, episode
        )
        native_keyframe_video_path = output_native_keyframe_video_path(
            output_dir, task_name, episode
        )

        save_timeline_plot(
            timeline_png_path,
            task_name=task_name,
            episode=episode,
            frame_count=len(scan.frames),
            predicted_keyframes=scan.predicted_keyframes,
            native_keyframes=scan.native_keyframes,
        )

        timeline_json_payload = build_timeline_json_payload(
            task_name=task_name,
            episode=episode,
            scan=scan,
            compare_fps=fps,
            keyframe_video_fps=keyframe_video_fps,
            motion_low=motion_low,
            motion_high=motion_high,
            match_tolerance=match_tolerance,
            comparison=comparison,
        )
        timeline_json_path.write_text(
            json.dumps(timeline_json_payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

        predicted_keyframes = _keyframe_number_map(scan.predicted_keyframes)
        native_keyframes = _keyframe_number_map(scan.native_keyframes)
        writer: EpisodeVideoWriter | None = None

        try:
            for record in scan.frames:
                front_rgb = _load_front_rgb_frame(
                    episode_group=episode_group,
                    episode_key=episode_key,
                    record=record,
                )
                frame = render_comparison_frame(
                    front_rgb,
                    task_name=task_name,
                    episode=episode,
                    record=record,
                    predicted_keyframes=predicted_keyframes,
                    native_keyframes=native_keyframes,
                )
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = EpisodeVideoWriter(compare_video_path, fps, (width, height))
                writer.write(frame)
        finally:
            if writer is not None:
                writer.close()

        predicted_keyframe_video = export_keyframe_video(
            episode_group=episode_group,
            episode_key=episode_key,
            scan=scan,
            keyframe_indices=scan.predicted_keyframes,
            keyframe_numbers=predicted_keyframes,
            out_path=predicted_keyframe_video_path,
            fps=keyframe_video_fps,
            task_name=task_name,
            episode=episode,
            lane_title="Predicted (joint + gripper)",
            border_color=PREDICTED_COLOR_BGR,
            show_gripper_state=True,
        )
        native_keyframe_video = export_keyframe_video(
            episode_group=episode_group,
            episode_key=episode_key,
            scan=scan,
            keyframe_indices=scan.native_keyframes,
            keyframe_numbers=native_keyframes,
            out_path=native_keyframe_video_path,
            fps=keyframe_video_fps,
            task_name=task_name,
            episode=episode,
            lane_title="Native (waypoint)",
            border_color=NATIVE_COLOR_BGR,
            show_gripper_state=False,
        )

    return EpisodeSummary(
        task_name=task_name,
        episode=episode,
        frame_count=len(scan.frames),
        predicted_keyframe_count=len(scan.predicted_keyframes),
        native_keyframe_count=len(scan.native_keyframes),
        timeline_png_path=timeline_png_path,
        timeline_json_path=timeline_json_path,
        compare_video_path=compare_video_path,
        predicted_keyframe_video_path=predicted_keyframe_video,
        native_keyframe_video_path=native_keyframe_video,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episodes-per-task", type=int, default=DEFAULT_EPISODES_PER_TASK)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--keyframe-video-fps", type=float, default=DEFAULT_KEYFRAME_VIDEO_FPS)
    parser.add_argument("--motion-low", type=float, default=DEFAULT_MOTION_LOW)
    parser.add_argument("--motion-high", type=float, default=DEFAULT_MOTION_HIGH)
    parser.add_argument("--match-tolerance", type=int, default=DEFAULT_MATCH_TOLERANCE)
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
    if args.keyframe_video_fps <= 0:
        parser.error("--keyframe-video-fps must be positive")
    if args.motion_low < 0:
        parser.error("--motion-low must be non-negative")
    if args.motion_high <= args.motion_low:
        parser.error("--motion-high must be greater than --motion-low")
    if args.match_tolerance < 0:
        parser.error("--match-tolerance must be non-negative")
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
                summary = export_episode_outputs(
                    h5_path,
                    task_name=task_name,
                    episode=episode,
                    output_dir=args.output_dir,
                    fps=args.fps,
                    keyframe_video_fps=args.keyframe_video_fps,
                    motion_low=args.motion_low,
                    motion_high=args.motion_high,
                    match_tolerance=args.match_tolerance,
                )
            except Exception as exc:
                failures.append(
                    f"{task_name} episode {episode}: {type(exc).__name__}: {exc}"
                )
                continue

            total_success += 1
            print(
                f"[OK] task={summary.task_name} episode={summary.episode} "
                f"frames={summary.frame_count} "
                f"predicted_keyframes={summary.predicted_keyframe_count} "
                f"native_keyframes={summary.native_keyframe_count} "
                f"timeline_png={summary.timeline_png_path} "
                f"timeline_json={summary.timeline_json_path} "
                f"compare_video={summary.compare_video_path} "
                f"predicted_keyframe_video={summary.predicted_keyframe_video_path} "
                f"native_keyframe_video={summary.native_keyframe_video_path}"
            )

    print(
        f"[SUMMARY] episodes_written={total_success} "
        f"failures={len(failures)} output_dir={args.output_dir}"
    )
    for failure in failures:
        print(f"[FAIL] {failure}")

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
