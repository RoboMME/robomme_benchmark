"""
Replay episodes from HDF5 datasets and save rollout videos.
Loads recorded actions from record_dataset_<Task>.h5, steps the environment
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
from pathlib import Path
from typing import Any, Dict, Literal, Union

import cv2
import h5py
import imageio
import numpy as np
import torch

from robomme.env_record_wrapper import BenchmarkEnvBuilder

GUI_RENDER = False
REPLAY_VIDEO_DIR = "runs/replay_videos"
VIDEO_FPS = 30
VIDEO_BORDER_COLOR = (255, 0, 0)
VIDEO_BORDER_THICKNESS = 10

SUBTITLE_FONT = cv2.FONT_HERSHEY_SIMPLEX
SUBTITLE_FONT_SCALE = 0.5
SUBTITLE_FONT_THICKNESS = 1
SUBTITLE_LINE_HEIGHT = 22
SUBTITLE_LEFT_PAD = 8

GOAL_FONT_COLOR = (180, 230, 180)
GOAL_TOP_PAD = 20
GOAL_MAX_LINES = 4

SUBGOAL_FONT_COLOR = (255, 255, 255)
SUBGOAL_MAX_LINES = 2

SUBTITLE_BAR_HEIGHT = (
    GOAL_TOP_PAD
    + GOAL_MAX_LINES * SUBTITLE_LINE_HEIGHT
    + 6
    + SUBGOAL_MAX_LINES * SUBTITLE_LINE_HEIGHT
    + 8
)

DEMO_TAG_TEXT = "VIDEO DEMO"
DEMO_TAG_FONT_SCALE = 0.5
DEMO_TAG_COLOR = (255, 255, 255)
DEMO_TAG_BG = (0, 0, 200)
DEMO_TAG_PAD_X = 6
DEMO_TAG_PAD_Y = 4
DEMO_TAG_MARGIN = 8

INTRO_DURATION_FRAMES = 60  # 2s @ 30 fps

TaskID = Literal[
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
]


ActionSpaceType = Literal["joint_angle", "ee_pose", "waypoint", "multi_choice"]

def _to_numpy(t) -> np.ndarray:
    return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def _wrap_text_pixels(
    text: str, font: int, scale: float, thickness: int, max_width: int
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        candidate = w if not cur else f"{cur} {w}"
        (tw, _), _ = cv2.getTextSize(candidate, font, scale, thickness)
        if tw <= max_width:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _fit_lines(
    text: str, font: int, scale: float, thickness: int,
    max_width: int, max_lines: int,
) -> list[str]:
    if not text:
        return []
    lines = _wrap_text_pixels(text, font, scale, thickness, max_width)
    if len(lines) <= max_lines:
        return lines
    lines = lines[:max_lines]
    last = lines[-1]
    while len(last) > 1:
        (tw, _), _ = cv2.getTextSize(last + "...", font, scale, thickness)
        if tw <= max_width:
            break
        last = last[:-1]
    lines[-1] = last + "..."
    return lines


def _add_subtitle_bar(
    frame: np.ndarray, goal: str = "", subgoal: str = ""
) -> np.ndarray:
    h, w = frame.shape[:2]
    bar = np.zeros((SUBTITLE_BAR_HEIGHT, w, 3), dtype=np.uint8)
    out = np.vstack([frame, bar])

    max_text_w = w - 2 * SUBTITLE_LEFT_PAD

    goal_lines = _fit_lines(
        f"Goal: {goal}" if goal else "",
        SUBTITLE_FONT, SUBTITLE_FONT_SCALE, SUBTITLE_FONT_THICKNESS,
        max_text_w, GOAL_MAX_LINES,
    )
    subgoal_lines = _fit_lines(
        f"Subgoal: {subgoal}" if subgoal else "",
        SUBTITLE_FONT, SUBTITLE_FONT_SCALE, SUBTITLE_FONT_THICKNESS,
        max_text_w, SUBGOAL_MAX_LINES,
    )

    y = h + GOAL_TOP_PAD
    for line in goal_lines:
        cv2.putText(
            out, line, (SUBTITLE_LEFT_PAD, y),
            SUBTITLE_FONT, SUBTITLE_FONT_SCALE,
            GOAL_FONT_COLOR, SUBTITLE_FONT_THICKNESS, cv2.LINE_AA,
        )
        y += SUBTITLE_LINE_HEIGHT

    # Fix the subgoal start so subgoal y position is stable across frames.
    y_sub = (
        h + GOAL_TOP_PAD + GOAL_MAX_LINES * SUBTITLE_LINE_HEIGHT + 6
        + SUBTITLE_LINE_HEIGHT - 4
    )
    for line in subgoal_lines:
        cv2.putText(
            out, line, (SUBTITLE_LEFT_PAD, y_sub),
            SUBTITLE_FONT, SUBTITLE_FONT_SCALE,
            SUBGOAL_FONT_COLOR, SUBTITLE_FONT_THICKNESS, cv2.LINE_AA,
        )
        y_sub += SUBTITLE_LINE_HEIGHT

    return out


def _draw_demo_tag(frame: np.ndarray) -> None:
    """Draw a 'VIDEO DEMO' label in the top-right corner of the frame."""
    text = DEMO_TAG_TEXT
    (tw, th), baseline = cv2.getTextSize(
        text, SUBTITLE_FONT, DEMO_TAG_FONT_SCALE, SUBTITLE_FONT_THICKNESS,
    )
    h, w = frame.shape[:2]
    x_text = w - tw - DEMO_TAG_MARGIN - DEMO_TAG_PAD_X
    y_text = DEMO_TAG_MARGIN + th + DEMO_TAG_PAD_Y
    box_x0 = x_text - DEMO_TAG_PAD_X
    box_y0 = y_text - th - DEMO_TAG_PAD_Y
    box_x1 = x_text + tw + DEMO_TAG_PAD_X
    box_y1 = y_text + baseline + DEMO_TAG_PAD_Y - 2
    cv2.rectangle(frame, (box_x0, box_y0), (box_x1, box_y1), DEMO_TAG_BG, -1)
    cv2.putText(
        frame, text, (x_text, y_text),
        SUBTITLE_FONT, DEMO_TAG_FONT_SCALE, DEMO_TAG_COLOR,
        SUBTITLE_FONT_THICKNESS, cv2.LINE_AA,
    )


def _frame_from_obs(
    front: np.ndarray | torch.Tensor,
    wrist: np.ndarray | torch.Tensor,
    is_video_demo: bool = False,
    goal: str = "",
    subgoal: str = "",
    point=None,
) -> np.ndarray:
    front_np = _to_numpy(front).astype(np.uint8)
    wrist_np = _to_numpy(wrist).astype(np.uint8)
    frame = np.hstack([front_np, wrist_np])
    if is_video_demo:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h),
                      VIDEO_BORDER_COLOR, VIDEO_BORDER_THICKNESS)
        _draw_demo_tag(frame)
    if point is not None:
        _draw_point_on_front(frame, point, front_np.shape[1])
    return _add_subtitle_bar(frame, goal=goal, subgoal=subgoal)


def _extract_frames(
    obs: dict,
    is_video_demo_fn=None,
    goal: str = "",
    subgoal: str = "",
    point=None,
) -> list[np.ndarray]:
    n = len(obs["front_rgb_list"])
    return [
        _frame_from_obs(
            obs["front_rgb_list"][i],
            obs["wrist_rgb_list"][i],
            is_video_demo=(is_video_demo_fn(i) if is_video_demo_fn else False),
            goal=goal,
            subgoal=subgoal,
            point=point,
        )
        for i in range(n)
    ]


def _is_video_demo(ts: h5py.Group) -> bool:
    info = ts.get("info")
    if info is None or "is_video_demo" not in info:
        return False
    return bool(np.reshape(np.asarray(info["is_video_demo"][()]), -1)[0])


def _is_subgoal_boundary(ts: h5py.Group) -> bool:
    info = ts.get("info")
    if info is None or "is_subgoal_boundary" not in info:
        return False
    return bool(np.reshape(np.asarray(info["is_subgoal_boundary"][()]), -1)[0])


def _decode_h5_str(raw) -> str:
    """Uniformly decode bytes / numpy bytes / str from HDF5 to str."""
    if isinstance(raw, np.ndarray):
        raw = raw.flatten()[0]
    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode("utf-8")
    return raw


def _parse_available_choices(episode_data: h5py.Group) -> list[Dict[str, Any]]:
    setup = episode_data.get("setup")
    if setup is None or "available_multi_choices" not in setup:
        return []
    raw = _decode_h5_str(setup["available_multi_choices"][()])
    try:
        choices = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(choices, list):
        return []
    result: list[Dict[str, Any]] = []
    for c in choices:
        if isinstance(c, dict) and "label" in c and "action" in c:
            result.append({
                "label": str(c["label"]),
                "action": str(c["action"]),
                "need_parameter": bool(c.get("need_parameter", False)),
            })
    return result


def _build_choice_label_map(
    available_choices: list[Dict[str, Any]],
) -> Dict[str, str]:
    """Map multi_choice label (a/b/c/...) -> human-readable action description."""
    return {c["label"].lower(): c["action"] for c in available_choices}


def _parse_seed(episode_data: h5py.Group) -> Union[int, None]:
    setup = episode_data.get("setup")
    if setup is None or "seed" not in setup:
        return None
    try:
        return int(np.asarray(setup["seed"][()]).flatten()[0])
    except (TypeError, ValueError, IndexError):
        return None


def _parse_difficulty(episode_data: h5py.Group) -> str:
    setup = episode_data.get("setup")
    if setup is None or "difficulty" not in setup:
        return ""
    return _decode_h5_str(setup["difficulty"][()])


INTRO_TITLE_SCALE = 0.7
INTRO_META_SCALE = 0.55
INTRO_HEADING_SCALE = 0.55
INTRO_CHOICE_SCALE = 0.5
INTRO_LABEL_COLOR = (0, 255, 255)
INTRO_TEXT_COLOR = (255, 255, 255)
INTRO_DIM_COLOR = (180, 180, 180)
INTRO_LEFT_PAD = 16


def _build_intro_frame(
    task_id: str,
    episode_idx: int,
    seed: Union[int, None],
    difficulty: str,
    available_choices: list[Dict[str, Any]],
    width: int,
    height: int,
) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    x = INTRO_LEFT_PAD
    y = 38

    cv2.putText(
        frame, f"Task: {task_id}", (x, y),
        SUBTITLE_FONT, INTRO_TITLE_SCALE, INTRO_TEXT_COLOR,
        SUBTITLE_FONT_THICKNESS + 1, cv2.LINE_AA,
    )
    y += 30

    meta = f"Episode #{episode_idx}"
    if seed is not None:
        meta += f"    Seed: {seed}"
    if difficulty:
        meta += f"    Difficulty: {difficulty}"
    cv2.putText(
        frame, meta, (x, y),
        SUBTITLE_FONT, INTRO_META_SCALE, INTRO_TEXT_COLOR,
        SUBTITLE_FONT_THICKNESS, cv2.LINE_AA,
    )
    y += 32

    cv2.putText(
        frame, "Available options:", (x, y),
        SUBTITLE_FONT, INTRO_HEADING_SCALE, INTRO_DIM_COLOR,
        SUBTITLE_FONT_THICKNESS, cv2.LINE_AA,
    )
    y += 26

    max_text_w = width - x - 16
    for c in available_choices:
        label_str = f"({c['label']})"
        (lw, _), _ = cv2.getTextSize(
            label_str, SUBTITLE_FONT, INTRO_CHOICE_SCALE,
            SUBTITLE_FONT_THICKNESS,
        )
        cv2.putText(
            frame, label_str, (x, y),
            SUBTITLE_FONT, INTRO_CHOICE_SCALE, INTRO_LABEL_COLOR,
            SUBTITLE_FONT_THICKNESS, cv2.LINE_AA,
        )
        suffix = "  [needs point]" if c["need_parameter"] else ""
        action_text = f"{c['action']}{suffix}"
        action_x = x + lw + 8
        action_lines = _wrap_text_pixels(
            action_text, SUBTITLE_FONT, INTRO_CHOICE_SCALE,
            SUBTITLE_FONT_THICKNESS, max_text_w - lw - 8,
        )
        for i, line in enumerate(action_lines):
            cv2.putText(
                frame, line, (action_x, y + i * 22),
                SUBTITLE_FONT, INTRO_CHOICE_SCALE, INTRO_TEXT_COLOR,
                SUBTITLE_FONT_THICKNESS, cv2.LINE_AA,
            )
        y += max(22, 22 * len(action_lines)) + 6
        if y > height - 16:
            break

    return frame


POINT_MARKER_COLOR = (0, 255, 255)
POINT_MARKER_SIZE = 12
POINT_MARKER_THICKNESS = 2


def _draw_point_on_front(
    frame_hstack: np.ndarray, point, front_width: int
) -> None:
    """Draw a cross + ring marker on the left (front camera) half of the frame.

    `point` follows the dataset schema [y, x] (pixel row, pixel col) in front_rgb.
    """
    if point is None:
        return
    try:
        y, x = int(point[0]), int(point[1])
    except (TypeError, ValueError, IndexError):
        return
    h = frame_hstack.shape[0]
    if not (0 <= x < front_width and 0 <= y < h):
        return
    cv2.drawMarker(
        frame_hstack, (x, y), POINT_MARKER_COLOR, cv2.MARKER_CROSS,
        POINT_MARKER_SIZE, POINT_MARKER_THICKNESS, cv2.LINE_AA,
    )
    cv2.circle(
        frame_hstack, (x, y), POINT_MARKER_SIZE // 2 + 1,
        POINT_MARKER_COLOR, 1, cv2.LINE_AA,
    )


def _build_action_sequence(
    episode_data: h5py.Group, action_space_type: str
) -> list[Union[np.ndarray, Dict[str, Any]]]:
    """
    Scan the entire episode and return the deduplicated action sequence:
    - joint_angle / ee_pose: actions of all non-video-demo steps (sequential, not deduplicated)
    - waypoint: remove adjacent duplicate waypoint_action (like EpisodeDatasetResolver)
    - multi_choice: choice_action (JSON dict) only for steps where is_subgoal_boundary=True
    """
    timestep_keys = sorted(
        (k for k in episode_data.keys() if k.startswith("timestep_")),
        key=lambda k: int(k.split("_")[1]),
    )

    actions: list[Union[np.ndarray, Dict[str, Any]]] = []
    prev_waypoint: np.ndarray | None = None

    for key in timestep_keys:
        ts = episode_data[key]
        if _is_video_demo(ts):
            continue

        action_grp = ts.get("action")
        if action_grp is None:
            continue

        if action_space_type == "joint_angle":
            if "joint_action" not in action_grp:
                continue
            actions.append(np.asarray(action_grp["joint_action"][()], dtype=np.float32))

        elif action_space_type == "ee_pose":
            if "eef_action" not in action_grp:
                continue
            actions.append(np.asarray(action_grp["eef_action"][()], dtype=np.float32))

        elif action_space_type == "waypoint":
            if "waypoint_action" not in action_grp:
                continue
            wa = np.asarray(action_grp["waypoint_action"][()], dtype=np.float32).flatten()
            if wa.shape != (7,) or not np.all(np.isfinite(wa)):
                continue
            # Remove adjacent duplicates
            if prev_waypoint is None or not np.array_equal(wa, prev_waypoint):
                actions.append(wa)
                prev_waypoint = wa.copy()

        elif action_space_type == "multi_choice":
            if not _is_subgoal_boundary(ts):
                continue
            if "choice_action" not in action_grp:
                continue
            raw = _decode_h5_str(action_grp["choice_action"][()])
            try:
                payload = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            choice = payload.get("choice")
            if not isinstance(choice, str) or not choice.strip():
                continue
            if "point" not in payload:
                continue
            actions.append({"choice": choice, "point": payload.get("point")})

        else:
            raise ValueError(f"Unknown action space type: {action_space_type}")

    return actions


def _save_video(
    frames: list[np.ndarray],
    task_id: str,
    episode_idx: int,
    task_goal: str,
    outcome: str,
    action_space_type: str,
) -> Path:
    video_dir = Path(REPLAY_VIDEO_DIR) / action_space_type
    video_dir.mkdir(parents=True, exist_ok=True)
    name = f"{outcome}_{task_id}_ep{episode_idx}_{task_goal}.mp4"
    path = video_dir / name
    imageio.mimsave(str(path), frames, fps=VIDEO_FPS)
    return path


def _get_episode_indices(data: h5py.File) -> list[int]:
    return sorted(
        int(key.split("_")[1])
        for key in data.keys()
        if key.startswith("episode_")
    )


def process_episode(
    env_data: h5py.File,
    episode_idx: int,
    task_id: str,
    action_space_type: ActionSpaceType,
) -> None:
    """Replay one episode from HDF5 data, record frames, and save a video."""
    episode_data = env_data[f"episode_{episode_idx}"]
    task_goal = episode_data["setup"]["task_goal"][()][0].decode()
    action_sequence = _build_action_sequence(episode_data, action_space_type)
    available_choices = _parse_available_choices(episode_data)
    choice_label_map = (
        _build_choice_label_map(available_choices)
        if action_space_type == "multi_choice" else {}
    )
    seed = _parse_seed(episode_data)
    difficulty = _parse_difficulty(episode_data)

    env = BenchmarkEnvBuilder(
        env_id=task_id,
        dataset="train",
        action_space=action_space_type,
        gui_render=GUI_RENDER,
    ).make_env_for_episode(episode_idx)

    print(f"\nTask: {task_id}, Episode: {episode_idx}")
    print(f"Task goal: {task_goal}")
    print(f"Total actions after dedup: {len(action_sequence)}")

    obs, _ = env.reset()
    sample_front = _to_numpy(obs["front_rgb_list"][0])
    sample_wrist = _to_numpy(obs["wrist_rgb_list"][0])
    intro_w = sample_front.shape[1] + sample_wrist.shape[1]
    intro_h = sample_front.shape[0] + SUBTITLE_BAR_HEIGHT
    intro_frame = _build_intro_frame(
        task_id=task_id,
        episode_idx=episode_idx,
        seed=seed,
        difficulty=difficulty,
        available_choices=available_choices,
        width=intro_w,
        height=intro_h,
    )
    frames = [intro_frame] * INTRO_DURATION_FRAMES

    frames.extend(_extract_frames(
        obs,
        is_video_demo_fn=lambda i, n=len(obs["front_rgb_list"]): i < n - 1,
        goal=task_goal,
        subgoal="",
    ))

    outcome = "unknown"
    for seq_idx, action in enumerate(action_sequence):
        point = None
        if action_space_type == "multi_choice" and isinstance(action, dict):
            label = action.get("choice", "")
            desc = choice_label_map.get(label.lower(), label)
            raw_point = action.get("point")
            if (
                isinstance(raw_point, (list, tuple)) and len(raw_point) >= 2
                and all(isinstance(p, (int, float)) for p in raw_point[:2])
            ):
                point = raw_point
                py, px = int(raw_point[0]), int(raw_point[1])
                subgoal = (
                    f"({label}) {desc} @ (x={px}, y={py})"
                )
            else:
                subgoal = f"({label}) {desc}"
        else:
            subgoal = ""
        try:
            obs, _, terminated, truncated, info = env.step(action)
            frames.extend(_extract_frames(
                obs, goal=task_goal, subgoal=subgoal, point=point,
            ))
        except Exception as e:
            print(f"Error at seq_idx {seq_idx}: {e}")
            break

        if GUI_RENDER:
            env.render()
        if terminated or truncated:
            outcome = info.get("status", "unknown")
            print(
                f"Outcome: {outcome} | task_id: {task_id} | episode: {episode_idx}"
            )
            break

    env.close()
    path = _save_video(frames, task_id, episode_idx, task_goal, outcome, action_space_type)
    print(f"Saved video to {path}\n")


def replay(
    h5_data_dir: str = "data/robomme_data_h5",
    action_space_type: ActionSpaceType = "joint_angle",
    replay_number: int = 10,
) -> None:
    """Replay episodes from HDF5 dataset files and save rollout videos."""
    for task_id in BenchmarkEnvBuilder.get_task_list():
        file_path = Path(h5_data_dir) / f"record_dataset_{task_id}.h5"

        if not file_path.exists():
            print(f"Skipping {task_id}: file not found: {file_path}")
            continue

        with h5py.File(file_path, "r") as data:
            episode_indices = _get_episode_indices(data)
            for episode_idx in episode_indices[:min(replay_number, len(episode_indices))]:
                process_episode(data, episode_idx, task_id, action_space_type)


if __name__ == "__main__":
    import tyro
    tyro.cli(replay)
