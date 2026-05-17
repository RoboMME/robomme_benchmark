"""
Interactive multi_choice runner for a single benchmark episode.

Prints the available options, saves the decision image so you can look at it
in a viewer, reads `<choice> [x] [y]` from stdin, steps the env, and at the
end writes the full rollout video plus the demo-only video.
"""

from pathlib import Path
from typing import Any, Literal

import cv2
import imageio
import numpy as np
import torch
import tyro

from robomme.env_record_wrapper import BenchmarkEnvBuilder

GUI_RENDER = False
VIDEO_FPS = 30
VIDEO_OUTPUT_DIR = "runs/sample_run_videos"
MAX_STEPS = 300
EPISODE_LIMITS = {"train": 100, "test": 50, "val": 50}
VIDEO_BORDER_COLOR = (255, 0, 0)
VIDEO_BORDER_THICKNESS = 10
ACTION_SPACE_TYPE = "multi_choice"

TaskID = Literal[
    "BinFill", "PickXtimes", "SwingXtimes", "StopCube",
    "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap",
    "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
]
DatasetType = Literal["train", "test", "val"]


def _to_numpy(t) -> np.ndarray:
    return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def _frame_from_obs(
    front: np.ndarray | torch.Tensor,
    wrist: np.ndarray | torch.Tensor,
    is_video_demo: bool = False,
) -> np.ndarray:
    frame = np.hstack([_to_numpy(front), _to_numpy(wrist)]).astype(np.uint8)
    if is_video_demo:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h),
                      VIDEO_BORDER_COLOR, VIDEO_BORDER_THICKNESS)
    return frame


def _extract_frames(obs: dict, is_video_demo_fn=None) -> list[np.ndarray]:
    n = len(obs["front_rgb_list"])
    return [
        _frame_from_obs(
            obs["front_rgb_list"][i],
            obs["wrist_rgb_list"][i],
            is_video_demo=(is_video_demo_fn(i) if is_video_demo_fn else False),
        )
        for i in range(n)
    ]


def _validate_episode_index(episode_idx: int, dataset: DatasetType) -> None:
    limit = EPISODE_LIMITS[dataset]
    if not 0 <= episode_idx < limit:
        raise ValueError(
            f"Invalid episode_idx {episode_idx} for '{dataset}'; allowed: [0, {limit})"
        )


def _save_image(frame: np.ndarray, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    imageio.imwrite(str(path), frame)
    return path


def _save_video(
    frames: list[np.ndarray], out_dir: Path, name: str
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    imageio.mimsave(str(path), frames, fps=VIDEO_FPS)
    return path


def _save_demo_video(reset_obs: dict, out_dir: Path) -> Path | None:
    n = len(reset_obs["front_rgb_list"])
    if n <= 1:
        return None
    demo_frames = [
        _frame_from_obs(
            reset_obs["front_rgb_list"][i],
            reset_obs["wrist_rgb_list"][i],
            is_video_demo=False,
        )
        for i in range(n - 1)
    ]
    return _save_video(demo_frames, out_dir, "demo.mp4")


def _print_options(available_choices: list[dict[str, Any]]) -> None:
    print("Available options:")
    for c in available_choices:
        suffix = "  [needs point]" if c.get("need_parameter") else ""
        print(f"  ({c['label'].upper()}) {c['action']}{suffix}")


def _prompt_choice(available_choices: list[dict[str, Any]]) -> dict[str, Any]:
    """Read one choice from stdin. Format: '<letter>' or '<letter> <x> <y>'.

    Coordinates are entered as (x, y) and converted to the env's [y, x] order.
    Retries on invalid input.
    """
    label_to_option = {
        c["label"].lower(): c for c in available_choices
    }
    while True:
        raw = input(
            "Enter choice (e.g. 'A 320 240' for x=320, y=240, "
            "or just 'A' if no coords needed): "
        ).strip()
        if not raw:
            print("  empty input, try again")
            continue

        tokens = raw.split()
        label = tokens[0].lower()
        if label not in label_to_option:
            print(f"  unknown choice '{tokens[0]}', valid: "
                  f"{[c['label'].upper() for c in available_choices]}")
            continue

        option = label_to_option[label]
        needs_point = bool(option.get("need_parameter"))
        action: dict[str, Any] = {"choice": label.upper()}

        if needs_point:
            if len(tokens) != 3:
                print("  this option needs coords — type '<letter> <x> <y>'")
                continue
            try:
                x = int(tokens[1])
                y = int(tokens[2])
            except ValueError:
                print("  coords must be integers (pixel x, pixel y)")
                continue
            action["point"] = [y, x]
        else:
            if len(tokens) > 1:
                print("  note: this option ignores coords; using letter only")

        return action


def _outcome_label(status: str) -> str:
    return {
        "success": "SUCCESS",
        "fail": "FAIL",
        "timeout": "TIMEOUT",
        "error": "ERROR",
        "ongoing": "ONGOING",
    }.get(status, status.upper() or "UNKNOWN")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    dataset: DatasetType = "test",
    task_id: TaskID = "PickXtimes",
    episode_idx: int = 0,
) -> None:
    """Run a single benchmark episode in interactive multi_choice mode.

    Args:
        dataset: Dataset split (train / test / val).
        task_id: Task identifier.
        episode_idx: Episode index within the split.
    """
    _validate_episode_index(episode_idx, dataset)

    env_builder = BenchmarkEnvBuilder(
        env_id=task_id,
        dataset=dataset,
        action_space=ACTION_SPACE_TYPE,
        gui_render=GUI_RENDER,
        max_steps=MAX_STEPS,
    )

    out_dir = Path(VIDEO_OUTPUT_DIR) / ACTION_SPACE_TYPE / f"{task_id}_ep{episode_idx}"

    print(f"\nRunning task: {task_id}, episode: {episode_idx}, "
          f"action_space: {ACTION_SPACE_TYPE}, dataset: {dataset}")
    print(f"Artifacts will be saved under: {out_dir}")

    env = env_builder.make_env_for_episode(episode_idx)
    obs, info = env.reset()

    task_goal = info["task_goal"][0]
    available_choices = info.get("available_multi_choices", []) or []
    print(f"Task goal: {task_goal}")
    _print_options(available_choices)

    demo_path = _save_demo_video(obs, out_dir)
    if demo_path is not None:
        print(f"[demo video] {demo_path}")
    else:
        print("[demo video] (no demo frames for this episode)")

    n_reset = len(obs["front_rgb_list"])
    frames = _extract_frames(
        obs, is_video_demo_fn=lambda i, n=n_reset: i < n - 1
    )

    status = "unknown"
    step_idx = 0
    try:
        while True:
            current_img = _frame_from_obs(
                obs["front_rgb_list"][-1],
                obs["wrist_rgb_list"][-1],
                is_video_demo=False,
            )
            img_path = _save_image(
                current_img, out_dir, f"step_{step_idx:02d}.png"
            )
            print(f"\n[image] step {step_idx}: {img_path}")
            _print_options(available_choices)

            action = _prompt_choice(available_choices)
            print(f"Action: {action}")

            obs, _, terminated, truncated, info = env.step(action)
            status = info.get("status", "unknown")
            if status == "error":
                print(f"Step error: {info.get('error_message', 'unknown error')}")
                frames.extend(_extract_frames(obs))
                break

            frames.extend(_extract_frames(obs))

            if GUI_RENDER:
                env.render()
            if terminated or truncated:
                break
            step_idx += 1
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted by user.")
        status = "interrupted"

    env.close()

    outcome = _outcome_label(status)
    full_path = _save_video(frames, out_dir, f"full_{outcome.lower()}.mp4")
    print(f"\n[full video] {full_path}")
    print(f"Outcome: {outcome} | task_id: {task_id} | episode: {episode_idx}\n")


if __name__ == "__main__":
    tyro.cli(main)
