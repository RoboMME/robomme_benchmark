"""生成仅含 setup 元数据的 Robomme HDF5。

这个脚本专门为 `scripts/dev/dataset-distribution.py` 准备输入，不再执行 planner
rollout、snapshot 抓取或真实任务求解。每个 episode 的执行流程固定为：

参数解析 -> 创建环境 -> reset -> 尝试 1 次 no-op step（仅尽力产出 MP4） ->
清空 wrapper buffer -> 强制写出 HDF5 setup -> 关闭环境 -> 校验 setup HDF5。

seed 规则采用 legacy 布局：
base_seed = 1_500_000 + env_code * 100000 + episode * 100
seed = base_seed + attempt

产物语义：
- HDF5：必需产物。脚本会验证 `episode_x/setup`、`difficulty`、`task_goal`、
  `available_multi_choices` 存在，且不允许出现 `timestep_*` 数据。
- MP4：尽力产物。仅尝试 1 次 no-op step 以给 wrapper 留帧；若没有 MP4，不视为失败。
"""

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Set

import gymnasium as gym
import h5py
import numpy as np

from pickhighlight_setup_metadata import (
    PICKHIGHLIGHT_ENV_ID,
    PICKHIGHLIGHT_METADATA_DATASET,
    write_pickhighlight_setup_metadata,
)
from robomme.env_record_wrapper import RobommeRecordWrapper
from robomme.robomme_env import *  # noqa: F401,F403
from robomme.robomme_env.utils.SceneGenerationError import SceneGenerationError
from videorepick_setup_metadata import (
    VIDEOREPICK_ENV_ID,
    VIDEOREPICK_METADATA_DATASET,
    write_videorepick_setup_metadata,
)

DEFAULT_ENVS = [
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "BinFill",
    "VideoUnmaskSwap",
    "VideoUnmask",
    "ButtonUnmaskSwap",
    "ButtonUnmask",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "PickHighlight",
    "InsertPeg",
    "MoveCube",
    "PatternLock",
    "RouteStick",
]
ENV_ID_TO_CODE = {name: idx + 1 for idx, name in enumerate(DEFAULT_ENVS)}
SEED_OFFSET = 1_500_000
VALID_ENVS: Set[str] = set(DEFAULT_ENVS)
DIFFICULTY_ORDER = ("easy", "medium", "hard")
MAX_SEED_ATTEMPTS = 100
MAX_EPISODES_PER_ENV = 1000
ENV_SEED_BLOCK_SIZE = MAX_EPISODES_PER_ENV * MAX_SEED_ATTEMPTS


def _dataset_hdf5_dir(dataset_root: Path) -> Path:
    """按 RecordWrapper 规则解析当前输出对应的 HDF5 目录。"""
    base_path = dataset_root.resolve()
    if base_path.suffix in {".h5", ".hdf5"}:
        return base_path.parent / f"{base_path.stem}_hdf5_files"
    return base_path / "hdf5_files"


def _episode_h5_path(output_root: Path, env_id: str, episode: int, seed: int) -> Path:
    """返回当前 episode 对应的 HDF5 路径。"""
    return _dataset_hdf5_dir(output_root) / f"{env_id}_ep{episode}_seed{seed}.h5"


def _latest_recorded_mp4(
    output_root: Path, env_id: str, episode: int, seed: int
) -> Optional[Path]:
    """返回当前 episode 对应的最新 mp4 文件。"""
    videos_dir = output_root / "videos"
    if not videos_dir.is_dir():
        return None
    tag = f"{env_id}_ep{episode}_seed{seed}"
    candidates = [path for path in videos_dir.glob("*.mp4") if tag in path.name]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _decode_h5_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray):
        flattened = value.reshape(-1).tolist()
        if not flattened:
            return ""
        return _decode_h5_text(flattened[0])
    return str(value)


def _verify_videorepick_setup_metadata(setup_group: h5py.Group) -> tuple[bool, str]:
    if VIDEOREPICK_METADATA_DATASET not in setup_group:
        return False, f"missing setup dataset: {VIDEOREPICK_METADATA_DATASET}"

    try:
        payload_raw = _decode_h5_text(setup_group[VIDEOREPICK_METADATA_DATASET][()])
        payload = json.loads(payload_raw)
    except Exception as exc:
        return (
            False,
            f"invalid {VIDEOREPICK_METADATA_DATASET} JSON "
            f"({type(exc).__name__}: {exc})",
        )

    if not isinstance(payload, dict):
        return False, f"{VIDEOREPICK_METADATA_DATASET} is not a JSON object"

    target_color = payload.get("target_cube_1_color")
    if target_color not in {"red", "blue", "green"}:
        return False, "invalid target_cube_1_color in videorepick_metadata"

    num_repeats = payload.get("num_repeats")
    if isinstance(num_repeats, bool) or not isinstance(num_repeats, int):
        return False, "invalid num_repeats type in videorepick_metadata"
    if num_repeats < 1:
        return False, "num_repeats must be >= 1 in videorepick_metadata"

    return True, "videorepick metadata verified"


def _verify_pickhighlight_setup_metadata(setup_group: h5py.Group) -> tuple[bool, str]:
    if PICKHIGHLIGHT_METADATA_DATASET not in setup_group:
        return False, f"missing setup dataset: {PICKHIGHLIGHT_METADATA_DATASET}"

    try:
        payload_raw = _decode_h5_text(setup_group[PICKHIGHLIGHT_METADATA_DATASET][()])
        payload = json.loads(payload_raw)
    except Exception as exc:
        return (
            False,
            f"invalid {PICKHIGHLIGHT_METADATA_DATASET} JSON "
            f"({type(exc).__name__}: {exc})",
        )

    if not isinstance(payload, dict):
        return False, f"{PICKHIGHLIGHT_METADATA_DATASET} is not a JSON object"

    target_cube_colors = payload.get("target_cube_colors")
    if not isinstance(target_cube_colors, list) or not target_cube_colors:
        return False, "invalid target_cube_colors in pickhighlight_metadata"

    for color_name in target_cube_colors:
        if color_name not in {"red", "blue", "green"}:
            return False, "invalid target cube color in pickhighlight_metadata"

    return True, "pickhighlight metadata verified"


def _verify_setup_h5(h5_path: Path, env_id: str, episode: int) -> tuple[bool, str]:
    """验证 setup-only HDF5 是否满足 downstream 读取要求。"""
    if not h5_path.is_file():
        return False, f"missing HDF5 file: {h5_path}"

    episode_group_name = f"episode_{episode}"
    try:
        with h5py.File(h5_path, "r") as handle:
            if episode_group_name not in handle:
                return False, f"missing group '{episode_group_name}'"

            episode_group = handle[episode_group_name]
            if not isinstance(episode_group, h5py.Group):
                return False, f"'{episode_group_name}' is not an HDF5 group"

            setup_group = episode_group.get("setup")
            if not isinstance(setup_group, h5py.Group):
                return False, "missing setup group"

            required_datasets = (
                "difficulty",
                "task_goal",
                "available_multi_choices",
            )
            missing = [
                dataset_name
                for dataset_name in required_datasets
                if dataset_name not in setup_group
            ]
            if missing:
                return False, f"missing setup datasets: {', '.join(missing)}"

            timestep_groups = sorted(
                name for name in episode_group.keys() if name.startswith("timestep_")
            )
            if timestep_groups:
                return False, (
                    "unexpected timestep data present: "
                    f"{', '.join(timestep_groups[:3])}"
                )

            if env_id == PICKHIGHLIGHT_ENV_ID:
                return _verify_pickhighlight_setup_metadata(setup_group)

            if env_id == VIDEOREPICK_ENV_ID:
                return _verify_videorepick_setup_metadata(setup_group)
    except Exception as exc:
        return False, f"failed to inspect HDF5 ({type(exc).__name__}: {exc})"

    return True, "setup verified"


def _base_seed_for_episode(env_id: str, episode: int) -> int:
    """按 legacy 规则计算某个 env/episode 的基础 seed。"""
    if env_id not in ENV_ID_TO_CODE:
        raise ValueError(f"Environment {env_id} missing from ENV_ID_TO_CODE mapping")
    env_code = ENV_ID_TO_CODE[env_id]
    return SEED_OFFSET + env_code * ENV_SEED_BLOCK_SIZE + episode * MAX_SEED_ATTEMPTS


def _build_parser() -> argparse.ArgumentParser:
    """定义命令行参数。"""

    def parse_difficulty_ratio(value: str) -> list[int]:
        compact = value.strip().replace(":", "")
        if len(compact) != 3 or not compact.isdigit():
            raise argparse.ArgumentTypeError(
                "difficulty must be a 3-part ratio such as '211' or '2:1:1'."
            )

        ratios = [int(part) for part in compact]
        if sum(ratios) <= 0:
            raise argparse.ArgumentTypeError(
                "difficulty ratio must contain at least one non-zero part."
            )
        return ratios

    parser = argparse.ArgumentParser(
        description=(
            "Generate setup-only Robomme HDF5 files using the legacy "
            "env/episode/attempt seed layout."
        )
    )
    parser.add_argument(
        "--env",
        "-e",
        nargs="+",
        default=[
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "BinFill",
    "VideoUnmaskSwap",
    "VideoUnmask",
    "ButtonUnmaskSwap",
    "ButtonUnmask",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "PickHighlight",
    "InsertPeg",
    "MoveCube",
    "PatternLock",
    "RouteStick",
],
        choices=sorted(VALID_ENVS),
        metavar="ENV",
        help="One or more environment IDs to run in order (default: VideoPlaceButton, VideoPlaceOrder). Each env runs the same episode range; seeds are derived from env_id/episode/attempt.",
    )
    parser.add_argument(
        "--episode-number",
        type=int,
        default=300,
        metavar="N",
        help=(
            "How many consecutive episodes to run starting from index 0: "
            "episodes 0 .. N-1 (e.g. N=5 runs episodes 0,1,2,3,4). "
            "Must be < 1000. Default: 200."
        ),
    )
    parser.add_argument(
        "--difficulty",
        type=parse_difficulty_ratio,
        default=[1, 1, 1],
        help=(
            "Episode difficulty ratio in easy:medium:hard order, such as "
            "'2:1:1' or '211'. Parsed into a list like [2, 1, 1]. "
            "Default: 1:0:0."
        ),
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        choices=[0, 1],
        help="GPU id to expose via CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/replay_videos"),
        help="Directory used as setup HDF5 / best-effort video output root.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=30,
        help=(
            "Maximum number of worker processes used to parallelize episodes within "
            "the same env_id. Default: auto=min(os.cpu_count(), episode count)."
        ),
    )
    return parser


def _build_env_kwargs(episode: int, seed: int, difficulty: str) -> dict:
    """构造环境参数，并按 episode 编号启用不同级别的失败恢复。"""
    env_kwargs = dict(
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        seed=seed,
        difficulty=difficulty,
    )
    if episode <= 5:
        env_kwargs["robomme_failure_recovery"] = True
        if episode <= 2:
            env_kwargs["robomme_failure_recovery_mode"] = "z"
        else:
            env_kwargs["robomme_failure_recovery_mode"] = "xy"
    return env_kwargs


def _create_env(
    env_id: str, env_kwargs: dict, output_dir: Path, episode: int, seed: int
) -> gym.Env:
    """创建 gym 环境，并包上录屏 wrapper。"""
    env = gym.make(env_id, **env_kwargs)
    return RobommeRecordWrapper(
        env,
        dataset=str(output_dir),
        env_id=env_id,
        episode=episode,
        seed=seed,
        save_video=True,
    )


def _resolve_noop_action(env: gym.Env) -> np.ndarray:
    """基于当前 qpos 生成 1 个最小 no-op action。"""
    robot = env.unwrapped.agent.robot
    qpos = robot.get_qpos() if hasattr(robot, "get_qpos") else robot.qpos
    if hasattr(qpos, "detach"):
        qpos = qpos.detach().cpu().numpy()
    elif hasattr(qpos, "cpu"):
        qpos = qpos.cpu().numpy()
    qpos = np.asarray(qpos, dtype=np.float32).flatten()
    if qpos.size < 7:
        raise ValueError(f"Unexpected qpos size {qpos.size}; expected at least 7")

    arm_action = qpos[:7]
    action_shape = getattr(getattr(env, "action_space", None), "shape", None)
    action_dim = int(np.prod(action_shape)) if action_shape else None

    if action_dim is None:
        action_dim = 7 if qpos.size <= 7 else 8

    if action_dim == 7:
        return arm_action
    if action_dim == 8:
        return np.concatenate([arm_action, np.array([1.0], dtype=np.float32)])
    raise ValueError(f"Unsupported action dimension {action_dim}; expected 7 or 8")


def _attempt_noop_step(env: gym.Env, env_id: str, episode: int, seed: int) -> None:
    """尝试执行 1 次 no-op step，仅用于尽力产出 MP4 帧。"""
    try:
        action = _resolve_noop_action(env)
    except Exception as exc:
        print(
            f"[Setup] Warning: failed to build no-op action for env={env_id} "
            f"episode={episode} seed={seed}: {exc}"
        )
        return

    try:
        env.step(action)
    except Exception as exc:
        print(
            f"[Setup] Warning: no-op step failed for env={env_id} "
            f"episode={episode} seed={seed}: {exc}"
        )


def _prepare_setup_only_close(env: gym.Env, env_id: str, episode: int, seed: int) -> None:
    """在 close 前移除 timestep 数据，并强制 wrapper 写 setup。"""
    dropped_records = len(getattr(env, "buffer", []))
    if hasattr(env, "buffer"):
        env.buffer.clear()
    env.episode_success = True
    print(
        f"[Setup] Prepared setup-only close for env={env_id} episode={episode} "
        f"seed={seed}; cleared {dropped_records} buffered timestep record(s)."
    )


def _close_env(env: Optional[gym.Env], episode: int, seed: int) -> None:
    """安全关闭环境，避免清理阶段的异常吞掉真正的执行结果。"""
    if env is None:
        return
    try:
        env.close()
    except Exception as close_exc:
        print(
            f"Warning: Exception during env.close() for episode {episode}, "
            f"seed {seed}: {close_exc}"
        )


def _run_episode(
    env_id: str,
    episode: int,
    seed: int,
    difficulty: str,
    output_dir: Path,
) -> tuple[bool, bool]:
    """执行单个 episode。

    返回 `(success, retryable_failure)`：
    - success=True: setup HDF5 已写出并通过校验。
    - retryable_failure=True: 失败发生在 env 创建 / reset 阶段，允许换 seed 重试。
    """
    print(
        f"--- Generating setup env={env_id} episode={episode} "
        f"seed={seed} difficulty={difficulty} ---"
    )

    env: Optional[gym.Env] = None
    h5_path = _episode_h5_path(output_dir, env_id, episode, seed)

    try:
        env_kwargs = _build_env_kwargs(episode, seed, difficulty)
        env = _create_env(env_id, env_kwargs, output_dir, episode, seed)
        env.reset()
    except SceneGenerationError as exc:
        print(
            f"[Setup] Scene generation failed for env={env_id} "
            f"episode={episode} seed={seed}: {exc}"
        )
        _close_env(env, episode, seed)
        return False, True
    except Exception as exc:
        print(
            f"[Setup] Failed during env creation/reset for env={env_id} "
            f"episode={episode} seed={seed}: {type(exc).__name__}: {exc}"
        )
        _close_env(env, episode, seed)
        return False, True

    try:
        _attempt_noop_step(env, env_id, episode, seed)
        _prepare_setup_only_close(env, env_id, episode, seed)
    finally:
        _close_env(env, episode, seed)

    try:
        write_pickhighlight_setup_metadata(env, h5_path, episode)
        write_videorepick_setup_metadata(env, h5_path, episode)
    except Exception as exc:
        print(
            f"[Setup] Failed to append task-specific metadata for env={env_id} "
            f"episode={episode} seed={seed}: {type(exc).__name__}: {exc}"
        )
        return False, False

    setup_ok, setup_message = _verify_setup_h5(h5_path, env_id, episode)
    mp4_path = _latest_recorded_mp4(output_dir, env_id, episode, seed)
    status_text = "SUCCESS" if setup_ok else "FAILED"
    print(
        f"--- Finished setup env={env_id} episode={episode} seed={seed} "
        f"difficulty={difficulty} [{status_text}] ---"
    )
    print(f"[Setup] HDF5 check: {setup_message}")
    if mp4_path is not None:
        print(f"[Setup] MP4 recorded: {mp4_path.resolve()}")
    else:
        print(
            f"[Setup] Warning: no MP4 matched under {output_dir / 'videos'} "
            f"(expected filename fragment '{env_id}_ep{episode}_seed{seed}')."
        )

    return setup_ok, False


def _run_episode_with_retry(
    env_id: str,
    episode: int,
    difficulty: str,
    output_dir: Path,
) -> tuple[bool, int]:
    """按 legacy seed 规则执行单个 episode。"""
    base_seed = _base_seed_for_episode(env_id, episode)
    print(
        f"[Retry] env={env_id} episode={episode} "
        f"base_seed={base_seed} difficulty={difficulty} "
        f"max_attempts={MAX_SEED_ATTEMPTS}"
    )

    last_seed = base_seed
    for attempt in range(MAX_SEED_ATTEMPTS):
        seed = base_seed + attempt
        last_seed = seed
        print(
            f"[Retry] env={env_id} episode={episode} "
            f"attempt={attempt + 1}/{MAX_SEED_ATTEMPTS} seed={seed}"
        )
        try:
            success, retryable_failure = _run_episode(
                env_id=env_id,
                episode=episode,
                seed=seed,
                difficulty=difficulty,
                output_dir=output_dir,
            )
        except Exception as exc:
            print(
                f"[Retry] env={env_id} episode={episode} seed={seed} "
                f"raised {type(exc).__name__}: {exc}"
            )
            success = False
            retryable_failure = False

        if success:
            print(
                f"[Retry] env={env_id} episode={episode} "
                f"succeeded with seed={seed} on attempt "
                f"{attempt + 1}/{MAX_SEED_ATTEMPTS}"
            )
            return True, seed

        if not retryable_failure:
            print(
                f"[Retry] env={env_id} episode={episode} seed={seed} "
                "failed after setup generation; skip further seed retries."
            )
            return False, seed

    print(
        f"[Retry] env={env_id} episode={episode} exhausted "
        f"{MAX_SEED_ATTEMPTS} attempts; last_seed={last_seed}"
    )
    return False, last_seed


def _resolve_max_workers(requested: Optional[int], n_episodes: int) -> int:
    """计算实际启用的 worker 数量。"""
    if requested is not None and requested < 1:
        raise SystemExit("--max-workers must be at least 1.")
    cpu_count = os.cpu_count() or 1
    max_allowed = min(cpu_count, n_episodes)
    if requested is None:
        return max_allowed
    return min(requested, max_allowed)


def _print_episode_artifacts(
    output_dir: Path, env_id: str, episode: int, run_seed: int
) -> None:
    """打印当前 episode 对应的 setup HDF5 和 MP4 路径。"""
    h5_path = _episode_h5_path(output_dir, env_id, episode, run_seed)
    if h5_path.is_file():
        print(
            f"Setup HDF5 ({env_id} episode {episode}, seed={run_seed}): "
            f"{h5_path.resolve()}"
        )
    else:
        print(
            f"Missing setup HDF5 ({env_id} episode {episode}, seed={run_seed}): "
            f"{h5_path}"
        )

    mp4_path = _latest_recorded_mp4(output_dir, env_id, episode, run_seed)
    if mp4_path is not None:
        print(
            f"Best-effort MP4 ({env_id} episode {episode}, seed={run_seed}): "
            f"{mp4_path.resolve()}"
        )
    else:
        print(
            f"No MP4 matched under {output_dir / 'videos'} "
            f"(expected filename fragment '{env_id}_ep{episode}_seed{run_seed}')."
        )


def _run_env_episodes(
    env_id: str,
    episode_specs: list[tuple[int, str]],
    output_dir: Path,
    max_workers: int,
) -> list[bool]:
    """在同一个 env_id 内并行执行多个 episode。"""
    env_successes: list[tuple[int, bool]] = []

    if max_workers == 1:
        for ep, difficulty in episode_specs:
            success, used_seed = _run_episode_with_retry(
                env_id=env_id,
                episode=ep,
                difficulty=difficulty,
                output_dir=output_dir,
            )
            env_successes.append((ep, success))
            _print_episode_artifacts(output_dir, env_id, ep, used_seed)
        return [success for _, success in sorted(env_successes)]

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {
            executor.submit(
                _run_episode_with_retry,
                env_id,
                ep,
                difficulty,
                output_dir,
            ): ep
            for ep, difficulty in episode_specs
        }

        for future in as_completed(futures):
            ep = futures[future]
            try:
                success, used_seed = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Worker crashed for env={env_id} episode={ep}"
                ) from exc
            env_successes.append((ep, success))
            print(
                f"[Parent] Completed env={env_id} episode={ep} seed={used_seed} "
                f"success={success}"
            )
            _print_episode_artifacts(output_dir, env_id, ep, used_seed)

    return [success for _, success in sorted(env_successes)]


def main() -> None:
    """脚本入口：解析参数、生成 setup-only HDF5、打印结果路径。"""
    args = _build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_episodes = args.episode_number
    if n_episodes < 1:
        raise SystemExit("--episode-number must be at least 1 (run episodes 0..N-1).")
    if n_episodes >= MAX_EPISODES_PER_ENV:
        raise SystemExit(
            f"--episode-number must be less than {MAX_EPISODES_PER_ENV}; "
            "the legacy seed layout reserves 1000 episode slots per environment."
        )

    episode_numbers = list(range(0, n_episodes))
    print(f"Environments (in order): {args.env}")
    print(
        f"Episode number N={n_episodes} -> running episode indices {episode_numbers} "
        f"(0 .. {n_episodes - 1})"
    )
    print(
        "Seed policy: "
        "base_seed = 1500000 + env_code * 100000 + episode * 100; "
        f"each episode retries up to {MAX_SEED_ATTEMPTS} seeds "
        "for env creation/reset failures."
    )

    difficulty_cycle = [
        difficulty
        for difficulty, count in zip(DIFFICULTY_ORDER, args.difficulty)
        for _ in range(count)
    ]
    difficulty_preview = [
        difficulty_cycle[ep % len(difficulty_cycle)] for ep in episode_numbers
    ]
    episode_specs = list(zip(episode_numbers, difficulty_preview))
    print(f"Difficulty ratio [easy, medium, hard]: {args.difficulty}")
    print(f"Difficulty per episode: {difficulty_preview}")
    print(f"GPU: {args.gpu}")
    worker_count = _resolve_max_workers(args.max_workers, len(episode_numbers))
    print(f"Max workers per env: {worker_count}")
    print(f"Output root: {output_dir}")
    print(f"HDF5 directory: {_dataset_hdf5_dir(output_dir)}")

    successes: list[bool] = []
    for env_id in args.env:
        print(f"\n========== env={env_id} ==========")
        print(
            f"Dispatching {len(episode_numbers)} setup-only episodes for env={env_id} "
            f"with up to {worker_count} worker process(es)."
        )
        successes.extend(
            _run_env_episodes(
                env_id=env_id,
                episode_specs=episode_specs,
                output_dir=output_dir,
                max_workers=worker_count,
            )
        )

    if all(successes):
        print("Setup-only generation finished successfully (all episodes).")
    else:
        print(
            "Setup-only generation finished with failure status "
            "(one or more episodes missing verified setup HDF5)."
        )


if __name__ == "__main__":
    main()
