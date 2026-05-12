"""
Replay episodes from per-episode HDF5 files and save rollout videos.

Files are expected at: {h5_data_dir}/{TaskID}_ep{episode_idx}_seed{seed}.h5
Each file contains a single 'episode_0' group with the recorded trajectory.
Seed and difficulty are read directly from the file, bypassing dataset metadata.
"""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import gymnasium as gym
import h5py
import imageio
import numpy as np
import torch

GUI_RENDER = False
REPLAY_VIDEO_DIR = "runs/replay_videos"
VIDEO_FPS = 30
VIDEO_BORDER_COLOR = (255, 0, 0)
VIDEO_BORDER_THICKNESS = 10

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

# Regex: {TaskID}_ep{episode_idx}_seed{seed}.h5
_H5_PATTERN = re.compile(r"^(.+)_ep(\d+)_seed(\d+)\.h5$")


def _to_numpy(t) -> np.ndarray:
    return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def _read_subgoal_online(env) -> Tuple[str, str]:
    """Read the *online* subgoal pair from `env.unwrapped`, matching RecordWrapper.step().

    RecordWrapper.step() (RecordWrapper.py:858-859, 934-935) reads these two
    fields from `env.unwrapped` on every step:

        current_task_name_online        -> RecordWrapper writes as
                                            info.simple_subgoal_online
        current_subgoal_segment_online  -> RecordWrapper writes as
                                            info.grounded_subgoal_online (after
                                            placeholder filling via
                                            process_segmentation)

    Both attributes are updated *unconditionally* every time
    `subgoal_evaluate_func.sequential_task_check()` is invoked (see
    subgoal_evaluate_func.py:174-176), so they always reflect the *current*
    online subgoal — distinct from `current_task_name` /
    `current_subgoal_segment` which only mutate when
    `allow_subgoal_change_this_timestep == True`.

    We surface the raw (unfilled) `current_subgoal_segment_online` here: the
    placeholder-filled variant requires re-running process_segmentation, which
    needs segmentation mask + color_map plumbing that the replay env does not
    keep. The raw text still carries the same online semantics — only the
    `{red_cube}` placeholder substitution differs from RecordWrapper.
    """
    unwrapped = env.unwrapped
    simple = str(getattr(unwrapped, "current_task_name_online", "Unknown"))
    grounded = getattr(unwrapped, "current_subgoal_segment_online", None)
    grounded_str = "None" if grounded is None else str(grounded)
    return simple, grounded_str


def _overlay_subgoal_online(frame: np.ndarray, simple: str, grounded: str) -> np.ndarray:
    """Draw the online-subgoal text rows at the top-left of `frame` (mutates in-place).

    Mirrors the "ONLINE:" block that RecordWrapper renders via
    `_video_compose_planner_online_rows` (RecordWrapper.py:372-381), but here we
    draw on the simpler `front | wrist` replay frame instead of the dual-stream
    composite — the *content* (simple_subgoal_online, grounded_subgoal_online)
    matches the recorded HDF5 fields one-to-one.
    """
    lines = [
        "ONLINE:",
        f"info.simple_subgoal_online: {simple}",
        f"info.grounded_subgoal_online: {grounded}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    text_color = (0, 255, 0)
    bg_color = (0, 0, 0)
    line_height = 18
    x_left, y_top = 10, 22
    for i, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
        py = y_top + i * line_height
        cv2.rectangle(
            frame,
            (x_left - 2, py - th - 2),
            (x_left + tw + 2, py + 4),
            bg_color,
            -1,
        )
        cv2.putText(
            frame, line, (x_left, py), font, scale, text_color, thickness, cv2.LINE_AA
        )
    return frame


def _frame_from_obs(
    front: np.ndarray | torch.Tensor,
    wrist: np.ndarray | torch.Tensor,
    is_video_demo: bool = False,
    subgoal_overlay: Optional[Tuple[str, str]] = None,
) -> np.ndarray:
    frame = np.hstack([_to_numpy(front), _to_numpy(wrist)]).astype(np.uint8)
    if is_video_demo:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h),
                      VIDEO_BORDER_COLOR, VIDEO_BORDER_THICKNESS)
    if subgoal_overlay is not None:
        _overlay_subgoal_online(frame, subgoal_overlay[0], subgoal_overlay[1])
    return frame


def _extract_frames(
    obs: dict,
    is_video_demo_fn=None,
    subgoal_overlay: Optional[Tuple[str, str]] = None,
) -> list[np.ndarray]:
    n = len(obs["front_rgb_list"])
    return [
        _frame_from_obs(
            obs["front_rgb_list"][i],
            obs["wrist_rgb_list"][i],
            is_video_demo=(is_video_demo_fn(i) if is_video_demo_fn else False),
            subgoal_overlay=subgoal_overlay,
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
    video_dir_override: Optional[Path] = None,
) -> Path:
    video_dir = (
        Path(video_dir_override)
        if video_dir_override is not None
        else Path(REPLAY_VIDEO_DIR) / action_space_type
    )
    video_dir.mkdir(parents=True, exist_ok=True)
    name = f"{outcome}_{task_id}_ep{episode_idx}_{task_goal}.mp4"
    path = video_dir / name
    imageio.mimsave(str(path), frames, fps=VIDEO_FPS)
    return path


def _make_env(
    task_id: str,
    seed: int,
    difficulty: Optional[str],
    action_space_type: str,
):
    """Create a wrapped environment with an explicit seed and difficulty."""
    from robomme.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper
    from robomme.env_record_wrapper.FailAwareWrapper import FailAwareWrapper

    env_kwargs: dict = dict(
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_pos",
        render_mode="human" if GUI_RENDER else "rgb_array",
        reward_mode="dense",
        seed=seed,
    )
    if difficulty:
        env_kwargs["difficulty"] = difficulty

    env = gym.make(task_id, **env_kwargs)
    env = DemonstrationWrapper(
        env,
        max_steps_without_demonstration=10002,
        gui_render=GUI_RENDER,
    )

    if action_space_type == "ee_pose":
        from robomme.env_record_wrapper.EndeffectorDemonstrationWrapper import EndeffectorDemonstrationWrapper
        env = EndeffectorDemonstrationWrapper(env, action_repr="rpy")
    elif action_space_type == "waypoint":
        from robomme.env_record_wrapper.MultiStepDemonstrationWrapper import MultiStepDemonstrationWrapper
        env = MultiStepDemonstrationWrapper(env, gui_render=GUI_RENDER, vis=GUI_RENDER)
    elif action_space_type == "multi_choice":
        from robomme.env_record_wrapper.OraclePlannerDemonstrationWrapper import OraclePlannerDemonstrationWrapper
        env = OraclePlannerDemonstrationWrapper(env, env_id=task_id, gui_render=GUI_RENDER)

    env = FailAwareWrapper(env)
    return env


def process_episode(
    file_path: Path,
    task_id: str,
    episode_idx: int,
    action_space_type: ActionSpaceType,
    video_dir_override: Optional[Path] = None,
) -> str:
    """Replay one episode from a per-episode HDF5 file and save a video.

    Returns the outcome string read from `info["status"]` at the terminal step,
    one of {"success", "fail", "failure", "timeout", "unknown", ...}. Callers
    using this as a validation gate should require outcome == "success".
    """
    with h5py.File(file_path, "r") as f:
        ep_keys = [k for k in f.keys() if k.startswith("episode_")]
        if not ep_keys:
            raise KeyError(f"No episode group found in {file_path}")
        episode_data = f[ep_keys[0]]
        task_goal = _decode_h5_str(episode_data["setup"]["task_goal"][()][0])
        seed = int(np.asarray(episode_data["setup"]["seed"][()]).flatten()[0])
        diff_raw = episode_data["setup"].get("difficulty")
        difficulty = _decode_h5_str(diff_raw[()]) if diff_raw is not None else None
        action_sequence = _build_action_sequence(episode_data, action_space_type)

    print(f"\nTask: {task_id}, Episode: {episode_idx}, Seed: {seed}, Difficulty: {difficulty}")
    print(f"Task goal: {task_goal}")
    print(f"Total actions: {len(action_sequence)}")

    env = _make_env(task_id, seed, difficulty, action_space_type)
    obs, _ = env.reset()
    # reset 后立刻读 online subgoal —— sequential_task_check 在 reset/首个 evaluate
    # 中也会刷新 current_task_name_online / current_subgoal_segment_online,
    # 此处的读取与 RecordWrapper.step() 同一时点的读取语义一致。
    subgoal_overlay = _read_subgoal_online(env)
    frames = _extract_frames(
        obs,
        is_video_demo_fn=lambda i, n=len(obs["front_rgb_list"]): i < n - 1,
        subgoal_overlay=subgoal_overlay,
    )

    outcome = "unknown"
    for seq_idx, action in enumerate(action_sequence):
        try:
            obs, _, terminated, truncated, info = env.step(action)
            # 每个 step 完成后立刻读取实时 online subgoal,叠加到这一 step 的所有帧
            # 上。这与 RecordWrapper 在 step() 末尾读取 self.unwrapped 字段的时点
            # 完全一致,反映的是"时时读取的 subgoal"而非 planner 维护的 subgoal。
            subgoal_overlay = _read_subgoal_online(env)
            frames.extend(_extract_frames(obs, subgoal_overlay=subgoal_overlay))
        except Exception as e:
            print(f"Error at seq_idx {seq_idx}: {e}")
            break

        if GUI_RENDER:
            env.render()
        if terminated or truncated:
            outcome = info.get("status", "unknown")
            print(f"Outcome: {outcome} | task_id: {task_id} | episode: {episode_idx}")
            break

    env.close()
    path = _save_video(
        frames, task_id, episode_idx, task_goal, outcome, action_space_type,
        video_dir_override=video_dir_override,
    )
    print(f"Saved video to {path}\n")
    return outcome


def _discover_files(
    h5_data_dir: str,
) -> dict[tuple[str, int], Path]:
    """
    Scan h5_data_dir for files matching {TaskID}_ep{N}_seed{M}.h5.
    When multiple seed variants exist for the same (task, episode), keep the first
    one sorted lexicographically (lowest seed suffix wins).
    Returns: {(task_id, episode_idx): file_path}
    """
    result: dict[tuple[str, int], Path] = {}
    for p in sorted(Path(h5_data_dir).glob("*.h5")):
        m = _H5_PATTERN.match(p.name)
        if not m:
            continue
        task_id, ep_str, _ = m.groups()
        key = (task_id, int(ep_str))
        if key not in result:
            result[key] = p
    return result


_DEFAULT_ENVS: List[str] = [
    "VideoPlaceButton",
]

_DEFAULT_EPISODES: List[int] = [7]


def replay(
    h5_data_dir: str = "runs/replay_videos-legacy/hdf5_files",
    action_space_type: ActionSpaceType = "joint_angle",
    envs: List[str] = _DEFAULT_ENVS,
    episodes: List[int] = _DEFAULT_EPISODES,
) -> None:
    """
    Replay episodes from per-episode HDF5 files and save rollout videos.

    Args:
        h5_data_dir: Directory containing {TaskID}_ep{N}_seed{M}.h5 files.
        action_space_type: Which action representation to replay.
        envs: Task IDs to replay. Default: all 7 available tasks.
        episodes: Episode indices to replay. Default: 0-199.
    """
    file_map = _discover_files(h5_data_dir)

    if not file_map:
        print(f"No matching .h5 files found in {h5_data_dir}")
        return

    env_filter = set(envs)
    ep_filter = set(episodes)

    for (task_id, ep_idx), file_path in sorted(file_map.items()):
        if task_id not in env_filter:
            continue
        if ep_idx not in ep_filter:
            continue
        _ = process_episode(file_path, task_id, ep_idx, action_space_type)


if __name__ == "__main__":
    import tyro
    tyro.cli(replay)
