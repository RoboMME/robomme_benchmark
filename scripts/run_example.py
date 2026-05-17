"""
单 episode 的 multi_choice 交互式运行脚本。

打印可选项，把决策图像保存到磁盘，从终端读取 `<字母> [x] [y]`，
推进 env，结束后保存完整 rollout 视频以及仅含 demo 部分的视频。
"""

import warnings
warnings.filterwarnings("ignore")

import sys
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


class _Tee:
    """同时写入多个流（终端 + 日志文件）。"""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def _to_numpy(t) -> np.ndarray:
    return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def _frame_from_obs(front, wrist, is_video_demo=False) -> np.ndarray:
    frame = np.hstack([_to_numpy(front), _to_numpy(wrist)]).astype(np.uint8)
    if is_video_demo:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h),
                      VIDEO_BORDER_COLOR, VIDEO_BORDER_THICKNESS)
    return frame


def _extract_frames(obs, is_video_demo_fn=None) -> list[np.ndarray]:
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
            f"无效的 episode_idx {episode_idx} (dataset='{dataset}')；允许范围: [0, {limit})"
        )


def _save_image(frame, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    imageio.imwrite(str(path), frame)
    return path


def _save_video(frames, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    imageio.mimsave(str(path), frames, fps=VIDEO_FPS)
    return path


def _save_demo_video(reset_obs, out_dir: Path) -> Path | None:
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
    print("可选项:")
    for c in available_choices:
        suffix = "  [需要坐标]" if c.get("need_parameter") else ""
        print(f"  ({c['label'].upper()}) {c['action']}{suffix}")


def _prompt_choice(
    available_choices: list[dict[str, Any]], log_file
) -> dict[str, Any]:
    """从 stdin 读取一个选项。格式: '<字母>' 或 '<字母> <x> <y>'.

    坐标按 (x, y) 输入，内部会转换为 env 期望的 [y, x]。
    """
    label_to_option = {c["label"].lower(): c for c in available_choices}
    while True:
        raw = input(
            "请输入选项 (示例: 'A 320 240' 表示 x=320 y=240；"
            "无需坐标时仅输入 'A'): "
        ).strip()
        if log_file is not None:
            log_file.write(f"{raw}\n")
            log_file.flush()

        if not raw:
            print("  空输入，请重试")
            continue
        tokens = raw.split()
        label = tokens[0].lower()
        if label not in label_to_option:
            valid = [c["label"].upper() for c in available_choices]
            print(f"  未知选项 '{tokens[0]}'，有效选项: {valid}")
            continue
        option = label_to_option[label]
        needs_point = bool(option.get("need_parameter"))
        action: dict[str, Any] = {"choice": label.upper()}
        if needs_point:
            if len(tokens) != 3:
                print("  此选项需要坐标 — 请输入 '<字母> <x> <y>'")
                continue
            try:
                x = int(tokens[1])
                y = int(tokens[2])
            except ValueError:
                print("  坐标必须是整数")
                continue
            action["point"] = [y, x]
        elif len(tokens) > 1:
            print("  提示: 此选项无需坐标，已忽略多余输入")
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
    task_id: TaskID = "Binfill",
    episode_idx: int = 0,
) -> None:
    """以交互式 multi_choice 方式运行单个 episode。"""
    _validate_episode_index(episode_idx, dataset)

    out_dir = (
        Path(VIDEO_OUTPUT_DIR) / ACTION_SPACE_TYPE
        / f"{task_id}_ep{episode_idx}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    print(f"任务: {task_id} | episode: {episode_idx} | dataset: {dataset}")

    env_builder = BenchmarkEnvBuilder(
        env_id=task_id,
        dataset=dataset,
        action_space=ACTION_SPACE_TYPE,
        gui_render=GUI_RENDER,
        max_steps=MAX_STEPS,
    )
    env = env_builder.make_env_for_episode(episode_idx)
    obs, info = env.reset()

    task_goal = info["task_goal"][0]
    available_choices = info.get("available_multi_choices", []) or []
    print(f"目标: {task_goal}")
    _print_options(available_choices)

    demo_path = _save_demo_video(obs, out_dir)
    if demo_path is not None:
        print(f"[演示视频] {demo_path}")

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
            print(f"[图像 step {step_idx}] {img_path}")

            action = _prompt_choice(available_choices, log_file=log_file)
            obs, _, terminated, truncated, info = env.step(action)
            status = info.get("status", "unknown")
            if status == "error":
                print(f"步骤错误: {info.get('error_message', '未知错误')}")
                frames.extend(_extract_frames(obs))
                break
            frames.extend(_extract_frames(obs))
            if GUI_RENDER:
                env.render()
            if terminated or truncated:
                break
            step_idx += 1
    except (KeyboardInterrupt, EOFError):
        print("\n用户中断")
        status = "interrupted"

    env.close()
    outcome = _outcome_label(status)
    full_path = _save_video(frames, out_dir, f"full_{outcome.lower()}.mp4")
    print(f"[完整视频] {full_path}")
    print(f"[日志] {log_path}")
    print(f"结果: {outcome}")

    log_file.close()


if __name__ == "__main__":
    tyro.cli(main)
