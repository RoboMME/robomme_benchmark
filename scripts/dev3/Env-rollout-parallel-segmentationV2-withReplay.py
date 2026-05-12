"""生成仅含 setup 元数据的 Robomme HDF5。

这个脚本专门为 `scripts/dev/dataset-distribution.py` 准备输入，不再执行 planner
rollout 或真实任务求解。每个 episode 的执行流程固定为：

参数解析 -> 创建环境 -> reset -> 尝试 1 次 no-op step（仅尽力产出 MP4） ->
清空 wrapper buffer -> 强制写出 HDF5 setup -> 关闭环境 -> 校验 setup HDF5。

seed 规则采用 legacy 布局：
base_seed = 1_500_000 + env_code * 100000 + episode * 100
seed = base_seed + attempt

产物语义：
- HDF5：必需产物。脚本会验证 `episode_x/setup`、`difficulty`、`task_goal`、
  `available_multi_choices` 存在，且不允许出现 `timestep_*` 数据。
- PNG：必需产物。reset 成功后会导出所有带 `segmentation` 的相机彩色分割图。
- JSON：必需产物。reset 成功后会导出非黑色可见对象的 world xyz 信息。
- 3D PNG：必需产物。reset 成功后会导出可见对象的独立 3D 坐标图。
- MP4：尽力产物。仅尝试 1 次 no-op step 以给 wrapper 留帧；若没有 MP4，不视为失败。
"""

import argparse
import atexit
import colorsys
import fcntl
import json
import multiprocessing as mp
import os
import shutil
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Set

os.environ.setdefault("MPLBACKEND", "Agg")

# 用于在父进程与 spawn 子进程之间传递已选 GPU id 的环境变量名
_GPU_PIN_ENV_MARKER = "_ROBOMME_GPU_PINNED"
# 默认使用 GPU 1，避免与其他任务争抢 GPU 0
DEFAULT_GPU_ID = 0
# 合法的 GPU id 范围，防止误传无效值
ALLOWED_GPU_IDS = (0, 1)


def _early_pin_gpu() -> int:
    """Pin CUDA_VISIBLE_DEVICES BEFORE any GPU-touching imports run.

    Must execute at module import time so that:
    1. Parent process: parses --gpu from sys.argv, exports CUDA_VISIBLE_DEVICES,
       and sets _ROBOMME_GPU_PINNED so spawn-children can recover the choice.
    2. Spawn worker (re-imports this module with multiprocessing's own argv):
       sees _ROBOMME_GPU_PINNED inherited from parent and re-applies the same
       CUDA_VISIBLE_DEVICES — never falling back to the argparse default.

    必须在模块导入阶段执行，原因：spawn 模式的子进程会重新 import 整个模块，
    如果此时不立刻设置 CUDA_VISIBLE_DEVICES，后续的 torch/sapien import
    就会绑定到默认 GPU（通常是 GPU 0），造成 GPU 资源冲突。
    """
    # 子进程路径：父进程已通过环境变量把选定的 GPU id 传下来，直接复用
    inherited = os.environ.get(_GPU_PIN_ENV_MARKER)
    if inherited is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = inherited
        return int(inherited)

    # 父进程路径：提前用一个只认 --gpu 的轻量 parser 解析命令行，
    # 其余未知参数用 parse_known_args 忽略，避免与完整 parser 冲突
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU_ID)
    args, _ = parser.parse_known_args()
    gpu_id = int(args.gpu)
    if gpu_id not in ALLOWED_GPU_IDS:
        raise SystemExit(
            f"--gpu must be one of {ALLOWED_GPU_IDS}; got {gpu_id}."
        )
    # 设置 CUDA_VISIBLE_DEVICES 使 PyTorch/SAPIEN 只看到指定的一块 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # 同时写入标记环境变量，让 spawn 子进程能读到相同的 gpu_id
    os.environ[_GPU_PIN_ENV_MARKER] = str(gpu_id)
    return gpu_id


_PINNED_GPU_ID = _early_pin_gpu()

import gymnasium as gym
import h5py
import imageio
import numpy as np
import torch

from robomme.env_record_wrapper import FailsafeTimeout
from robomme.robomme_env.utils.planner_fail_safe import (
    FailAwarePandaArmMotionPlanningSolver,
    ScrewPlanFailure,
)


_ACTIVE_EXECUTOR: "Optional[ProcessPoolExecutor]" = None
_SHUTDOWN_INITIATED = False


def _terminate_active_workers(grace_seconds: float = 3.0) -> None:
    """强制结束当前活跃 executor 的所有 worker 进程。

    先发 SIGTERM（terminate），等待 grace_seconds 秒后仍存活则强制 SIGKILL（kill）。
    用于 Ctrl-C / SIGTERM 信号处理和 atexit 清理，确保子进程不会变成孤儿进程。
    """
    executor = _ACTIVE_EXECUTOR
    if executor is None:
        return
    # 访问 ProcessPoolExecutor 内部的进程字典（CPython 实现细节）
    try:
        worker_processes = list(executor._processes.values())
    except (AttributeError, RuntimeError):
        # executor 已销毁或内部结构不可用，安全忽略
        worker_processes = []

    # 第一轮：向所有存活进程发 SIGTERM，允许它们做最后的清理
    for proc in worker_processes:
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass

    # 等待宽限期，轮询直到所有进程退出或超时
    deadline = time.time() + max(0.0, grace_seconds)
    while time.time() < deadline and any(p.is_alive() for p in worker_processes):
        time.sleep(0.05)

    # 第二轮：对仍存活的进程发 SIGKILL，强制立即终止
    for proc in worker_processes:
        try:
            if proc.is_alive():
                proc.kill()
        except Exception:
            pass


def _handle_termination_signal(signum, _frame) -> None:
    """SIGINT/SIGTERM 信号处理器：先结束所有 worker，再退出父进程。

    使用 _SHUTDOWN_INITIATED 标志防止重入：如果在清理期间再次收到信号，
    直接调用 os._exit 强制退出，避免无限递归。
    退出码遵循 POSIX 惯例：128 + 信号编号。
    """
    global _SHUTDOWN_INITIATED
    # 防止信号重入导致递归：第二次收到信号时立刻强制退出
    if _SHUTDOWN_INITIATED:
        os._exit(128 + signum)
    _SHUTDOWN_INITIATED = True
    print(
        f"\n[Parent] Received signal {signum}; terminating worker processes ...",
        flush=True,
    )
    _terminate_active_workers()
    print("[Parent] Worker processes terminated; exiting.", flush=True)
    # 用 sys.exit 而非 os._exit，以便触发 atexit 清理（尽管此时已手动清理过）
    sys.exit(128 + signum)


def _install_parent_signal_handlers() -> None:
    """在父进程中安装 SIGINT/SIGTERM 处理器，并注册 atexit 兜底清理。

    atexit 兜底确保即使正常退出（非信号），worker 进程也能被回收。
    """
    signal.signal(signal.SIGINT, _handle_termination_signal)
    signal.signal(signal.SIGTERM, _handle_termination_signal)
    atexit.register(_terminate_active_workers)


def _worker_initializer() -> None:
    """Worker 进程初始化：忽略 SIGINT，由父进程统一管理关闭流程。

    spawn 子进程默认会继承父进程的信号处理器，这里显式覆盖为 SIG_IGN，
    确保 Ctrl-C 只触发父进程的 _handle_termination_signal，
    而不会同时杀死子进程导致 executor 报 BrokenProcessPool。
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

SCRIPT_DIR = Path(__file__).resolve().parent
# 4 个 suite 模块住在 scripts/dev3/env_specific_extraction/；spawn 子进程不会
# 自动把脚本目录加入 sys.path，显式加入保证可 import。
ENV_SPECIFIC_EXTRACTION_DIR = SCRIPT_DIR / "env_specific_extraction"
if str(ENV_SPECIFIC_EXTRACTION_DIR) not in sys.path:
    sys.path.append(str(ENV_SPECIFIC_EXTRACTION_DIR))

# 4 个 suite 模块同位于 env_specific_extraction/，各自管自己 env 的 reset
# enrich + suite 特有的 execute 阶段钩子（imitation.select_planner）。
import counting  # noqa: E402
import imitation  # noqa: E402
import permanence  # noqa: E402
import reference  # noqa: E402

from robomme.env_record_wrapper import RobommeRecordWrapper
from robomme.robomme_env import *  # noqa: F401,F403
from robomme.robomme_env.utils.SceneGenerationError import SceneGenerationError
from robomme.robomme_env.utils.choice_action_mapping import extract_actor_position_xyz
from robomme.robomme_env.utils.segmentation_utils import create_segmentation_visuals

# 全部 16 个任务环境 ID，顺序决定 env_code（1-indexed），进而影响 seed 计算
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
# env_code 从 1 开始，便于 seed 公式中的块偏移计算
ENV_ID_TO_CODE = {name: idx + 1 for idx, name in enumerate(DEFAULT_ENVS)}

# Legacy seed 规则：
#   base_seed = SEED_OFFSET + env_code * ENV_SEED_BLOCK_SIZE + episode * MAX_SEED_ATTEMPTS
#   seed       = base_seed + attempt
# 1_500_000 作为起点，与其他 seed 空间（训练/验证等）不重叠
SEED_OFFSET = 1_500_000

VALID_ENVS: Set[str] = set(DEFAULT_ENVS)
# 难度等级固定顺序，用于 difficulty_cycle 生成
DIFFICULTY_ORDER = ("easy", "medium", "hard")
# 每个 episode 最多尝试 100 个不同 seed，避免场景生成失败时永远卡住
MAX_SEED_ATTEMPTS = 100
# 每个 env 最多 1000 个 episode slot（seed 空间按此分块）
MAX_EPISODES_PER_ENV = 1000
# screw 规划最多重试 3 次后才切换到 RRT*
DATASET_SCREW_MAX_ATTEMPTS = 3
# RRT* 规划最多重试 3 次后放弃并返回 -1
DATASET_RRT_MAX_ATTEMPTS = 3
# 每个 env 占用的 seed 总块大小 = 1000 episodes × 100 attempts
ENV_SEED_BLOCK_SIZE = MAX_EPISODES_PER_ENV * MAX_SEED_ATTEMPTS

# reset segmentation PNG 输出子目录名
RESET_SEGMENTATION_DIRNAME = "reset_segmentation_pngs"
# 导出可见对象 world xyz 的两个相机
VISIBLE_OBJECT_CAMERAS = ("base_camera", "hand_camera")
# 可见对象 JSON 文件名
VISIBLE_OBJECT_JSON_FILENAME = "visible_objects.json"
# 可见对象 3D 俯视图 PNG 文件名
VISIBLE_OBJECT_3D_PNG_FILENAME = "visible_object_positions_3d.png"
# 每个 episode 写一行的结构化结果日志文件名（位于 output_dir 根目录）
EPISODE_LOG_FILENAME = "episode_results.jsonl"


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


def _videos_success_dir(output_root: Path) -> Path:
    """返回 videos-success/ 目录路径（不创建）。"""
    return output_root / "videos-success"


def _pick_videos_success_candidate(
    output_root: Path, env_id: str, episode: int, seed: int
) -> Optional[Path]:
    """从 videos/ 挑出适合复制到 videos-success/ 的 mp4。

    排除两类不应进入"成功样本目录"的视频：
    - 以 ``FAILED_`` 开头：失败视频（含 ``FAILED_NO_OBJECT_``）。
    - 以 ``success_NO_OBJECT_`` 开头：target 全程不在视野的"成功"，
      不算合格的成功样本（详见 RecordWrapper._video_flush_episode_files）。

    与 _latest_recorded_mp4 的区别：后者是按 mtime 取最新（用于打印 best-effort
    mp4），可能错选 success_NO_OBJECT_ 视频；本函数只在合格候选中按 mtime 选。
    """
    videos_dir = output_root / "videos"
    if not videos_dir.is_dir():
        return None
    tag = f"{env_id}_ep{episode}_seed{seed}"
    candidates = [
        path
        for path in videos_dir.glob("*.mp4")
        if tag in path.name
        and not path.name.startswith("FAILED_")
        and not path.name.startswith("success_NO_OBJECT_")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _copy_to_success_dir(
    output_root: Path, mp4_path: Path
) -> Optional[Path]:
    """把成功 mp4 复制一份到 videos-success/，文件名保持不变。

    返回目标路径（成功）或 None（失败）。失败时不抛异常，只在 stderr 警告——
    与 _append_episode_log 的"日志写失败降级"同类（参见上方 _append_episode_log
    docstring）：videos-success/ 是冗余便利副本，主产物 mp4 已在 videos/，
    权威 success 状态已写入 episode_results.jsonl，复制失败不应让整个并行
    rollout 崩溃，但必须显式告知用户（不是静默降级）。catch 范围严格收敛到
    OSError，避免吞掉编程错误。
    """
    dest_dir = _videos_success_dir(output_root)
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / mp4_path.name
        # shutil.copy2 默认覆盖目标；不需要先 unlink。
        # 重复运行同一 (env_id, ep, seed) 时新覆盖旧就是"最后一次"语义。
        shutil.copy2(mp4_path, dest_path)
        return dest_path
    except OSError as exc:
        print(
            f"[warn] Failed to copy success mp4 to {dest_dir}: "
            f"{_format_exception(exc)} (source={mp4_path})",
            file=sys.stderr,
        )
        return None


def _reset_segmentation_dir(
    output_root: Path, env_id: str, episode: int, seed: int
) -> Path:
    """返回当前 episode 对应的 reset segmentation PNG 目录。"""
    return (
        output_root
        / RESET_SEGMENTATION_DIRNAME
        / f"{env_id}_ep{episode}_seed{seed}"
    )


def _episode_log_path(output_root: Path) -> Path:
    """返回 episode 结果 JSONL 的绝对路径（与其他产物共置于 output_root 下）。"""
    return output_root / EPISODE_LOG_FILENAME


def _format_exception(exc: BaseException) -> str:
    """统一异常字符串格式 '{ClassName}: {message}'，与脚本现有 print 风格一致。"""
    return f"{type(exc).__name__}: {exc}"


def _append_episode_log(log_path: Path, record: dict) -> None:
    """以独占文件锁追加一行 JSON 到 episode 结果日志。

    多 worker 并行写入时，fcntl.LOCK_EX 保证每行写入是原子的，避免内容交错；
    日志写入本身的 IO 异常不应阻断主流程，捕获后只在 stderr 警告——这是脚本里
    唯一允许的"日志写失败降级"模式，与 CLAUDE.md 禁止的业务静默失败语义不同。
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # 用 "a" 模式 + O_APPEND 语义；fcntl.LOCK_EX 防止多 worker 写交错
        with open(log_path, "a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(
                    json.dumps(record, ensure_ascii=False, default=str) + "\n"
                )
                handle.flush()
                # fsync 不是必需的：进程崩溃时 OS 仍会刷出 page cache，
                # 加 fsync 反而会拖慢并行写入。仅在用户显式断电时丢失尾部记录。
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception as log_exc:
        # 日志写失败不影响 episode 结果本身的返回值，print 到 stderr 便于排查
        print(
            f"[Log] Warning: failed to append episode log to {log_path}: "
            f"{_format_exception(log_exc)}",
            file=sys.stderr,
        )


def _to_numpy(value: object) -> np.ndarray:
    """将 tensor / array-like 转为 CPU numpy。"""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _generate_color_map(
    n: int = 10_000,
    s_min: float = 0.70,
    s_max: float = 0.95,
    v_min: float = 0.78,
    v_max: float = 0.95,
) -> dict[int, list[int]]:
    """复刻 RecordWrapper 的固定 segmentation 色表。

    使用黄金比例 phi 对色相（Hue）进行"黄金角哈希"，使相邻 seg_id 的颜色
    尽量分散，视觉上易于区分。饱和度和明度通过取模运算形成次级分层，
    避免不同 id 重叠成相同颜色。

    生成结果与 RecordWrapper 完全一致，因此可以直接用于 PNG 叠加对照。
    """
    # 黄金比例共轭，用于均匀分布色相
    phi = 0.6180339887498948
    color_map: dict[int, list[int]] = {}
    for index in range(1, n + 1):
        # 黄金角哈希：相邻 index 的色相差恒为 phi，均匀覆盖色环
        hue = (index * phi) % 1.0
        # 饱和度在 [s_min, s_max] 内按 7 档循环，增加色彩层次
        saturation = s_min + (s_max - s_min) * ((index % 7) / 6)
        # 明度在 [v_min, v_max] 内按 5 档循环，与饱和度的周期互质避免重叠
        value = v_min + (v_max - v_min) * (((index * 3) % 5) / 4)
        # HSV -> RGB 转换，结果在 [0, 1] 区间
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
        # 转为 uint8 整数，存入色表
        color_map[index] = [
            int(round(red * 255)),
            int(round(green * 255)),
            int(round(blue * 255)),
        ]
    return color_map


SEGMENTATION_COLOR_MAP = _generate_color_map()


def _get_reset_blacklisted_object_names(env: gym.Env) -> set[str]:
    """返回 reset segmentation 中需要显示为黑色的对象名称集合。"""
    blacklisted_names = {"table-workspace", "ground"}
    robot = getattr(getattr(env.unwrapped, "agent", None), "robot", None)
    for link in list(getattr(robot, "links", []) or []):
        link_name = getattr(link, "name", None)
        if link_name:
            blacklisted_names.add(str(link_name))
    return blacklisted_names


def _build_reset_segmentation_color_map(
    env: gym.Env,
    segmentation_id_map: object,
) -> dict[int, list[int]]:
    """根据 reset 时的 segmentation_id_map 应用对象级颜色覆盖。"""
    color_map = {
        seg_id: color.copy() for seg_id, color in SEGMENTATION_COLOR_MAP.items()
    }
    blacklisted_names = _get_reset_blacklisted_object_names(env)
    if isinstance(segmentation_id_map, dict):
        for obj_id, obj in segmentation_id_map.items():
            obj_name = getattr(obj, "name", None)
            if obj_name in blacklisted_names:
                color_map[int(obj_id)] = [0, 0, 0]
    return color_map


def _visible_object_json_path(reset_output_dir: Path) -> Path:
    return reset_output_dir / VISIBLE_OBJECT_JSON_FILENAME


def _visible_object_plot_path(reset_output_dir: Path) -> Path:
    return reset_output_dir / VISIBLE_OBJECT_3D_PNG_FILENAME


def _collect_reset_visible_objects(
    obs: object,
    env: gym.Env,
) -> dict[str, object]:
    """收集 base/hand camera 中非黑色可见对象的 world xyz 信息。

    遍历 VISIBLE_OBJECT_CAMERAS 中的每个相机，找出 segmentation 图里
    出现过的所有 seg_id，过滤掉桌面/地面/机器人链接等黑名单对象，
    再从 segmentation_id_map 中查出对应 Actor 并提取世界坐标。
    同一对象可能同时被两个相机看到，最终去重后记录 visible_in 字段。
    """
    if not isinstance(obs, dict):
        raise ValueError("reset observation is not a dict")

    sensor_data = obs.get("sensor_data")
    if not isinstance(sensor_data, dict):
        raise ValueError("reset observation missing sensor_data dict")

    # segmentation_id_map: seg_id -> Actor 对象，由环境在 reset 后填充
    segmentation_id_map = getattr(env.unwrapped, "segmentation_id_map", None)
    if not isinstance(segmentation_id_map, dict):
        raise ValueError("env missing segmentation_id_map dict after reset")

    blacklisted_names = _get_reset_blacklisted_object_names(env)
    # 记录每个相机看到的 seg_id 集合，用于最终写 visible_in 字段
    visible_ids_by_camera: dict[str, set[int]] = {}
    # 用 seg_id 作 key 去重，避免同一对象被两个相机重复记录
    visible_objects: dict[int, dict[str, object]] = {}

    for camera_name in VISIBLE_OBJECT_CAMERAS:
        camera_obs = sensor_data.get(camera_name)
        if not isinstance(camera_obs, dict):
            raise ValueError(f"reset observation missing camera '{camera_name}'")
        if "segmentation" not in camera_obs:
            raise ValueError(
                f"reset observation camera '{camera_name}' missing segmentation"
            )

        segmentation = _to_numpy(camera_obs["segmentation"])
        # 去掉 batch 维度（形状可能是 [1, H, W, C] 或 [H, W, C]）
        if segmentation.ndim >= 4:
            segmentation = segmentation[0]
        # squeeze 掉最后的 channel 维，得到纯 2D 的 [H, W] seg_id 矩阵
        segmentation_2d = segmentation.squeeze()
        if segmentation_2d.ndim != 2:
            raise ValueError(
                f"camera '{camera_name}' segmentation has invalid ndim "
                f"{segmentation_2d.ndim}; expected 2"
            )

        camera_visible_ids: set[int] = set()
        # 遍历该相机图像中出现的所有唯一 seg_id（已排序，便于日志对比）
        for seg_id in sorted(int(seg_value) for seg_value in np.unique(segmentation_2d)):
            # seg_id <= 0 表示背景或无效像素，跳过
            if seg_id <= 0:
                continue

            obj = segmentation_id_map.get(seg_id)
            if obj is None:
                raise ValueError(
                    f"visible segmentation id {seg_id} missing from segmentation_id_map"
                )

            obj_name = getattr(obj, "name", None)
            # 黑名单对象（桌面、地面、机器人链接）在 PNG 中显示为黑色，不计入 JSON
            if obj_name in blacklisted_names:
                continue

            # 从 Actor 对象中提取世界坐标 [x, y, z]
            position_xyz = extract_actor_position_xyz(obj)
            if position_xyz is None:
                raise ValueError(
                    f"failed to extract world xyz for seg_id={seg_id}, "
                    f"name={obj_name!r}"
                )

            camera_visible_ids.add(seg_id)
            # setdefault 确保同一 seg_id 只记录一次（多相机情况下）
            visible_objects.setdefault(
                seg_id,
                {
                    "segmentation_id": seg_id,
                    "name": str(obj_name or f"seg_{seg_id}"),
                    "object_type": type(obj).__name__,
                    "world_xyz": np.asarray(position_xyz, dtype=np.float64).tolist(),
                },
            )

        visible_ids_by_camera[camera_name] = camera_visible_ids

    # 按 seg_id 排序后构建最终列表，并附加每个相机的可见性标记
    objects_payload: list[dict[str, object]] = []
    for seg_id in sorted(visible_objects):
        item = dict(visible_objects[seg_id])
        # visible_in: {"base_camera": True/False, "hand_camera": True/False}
        item["visible_in"] = {
            camera_name: seg_id in visible_ids_by_camera.get(camera_name, set())
            for camera_name in VISIBLE_OBJECT_CAMERAS
        }
        objects_payload.append(item)

    return {
        "cameras": list(VISIBLE_OBJECT_CAMERAS),
        "objects": objects_payload,
    }


def _save_visible_objects_json(
    reset_output_dir: Path,
    env_id: str,
    episode: int,
    seed: int,
    visible_payload: dict[str, object],
    env: Optional[gym.Env] = None,
) -> Path:
    """写出 reset 可见对象 JSON。

    visible_objects.json 是 reset 阶段所有 env-specific 数据的唯一权威 sidecar。
    4 个 suite 模块各自暴露同名 enrich_visible_payload(payload, env, env_id)
    钩子，主脚本盲调 4 次：每个 suite 内部判断 env_id 是否在自己范围内，
    在则原地写入顶层字段（permanence_init_state / selected_target /
    videorepick_metadata），否则 no-op。
    """
    json_path = _visible_object_json_path(reset_output_dir)
    payload = {
        "env_id": env_id,
        "episode": episode,
        "seed": seed,
        "cameras": visible_payload["cameras"],
        "objects": visible_payload["objects"],
    }
    if env is not None:
        counting.enrich_visible_payload(payload, env=env, env_id=env_id)
        permanence.enrich_visible_payload(payload, env=env, env_id=env_id)
        reference.enrich_visible_payload(payload, env=env, env_id=env_id)
        imitation.enrich_visible_payload(payload, env=env, env_id=env_id)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return json_path


def _set_equal_xy_axes(ax, points_xyz: np.ndarray) -> None:
    """为俯视 XY scatter 设定固定坐标范围。"""
    del points_xyz
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_aspect("equal", adjustable="box")


def _xy_rot_cw_90(x_pos: float, y_pos: float) -> tuple[float, float]:
    """俯视图顺时针旋转 90°：显示 (y, -x)。"""
    return y_pos, -x_pos


def _save_visible_objects_3d_plot(
    reset_output_dir: Path,
    env_id: str,
    episode: int,
    seed: int,
    visible_payload: dict[str, object],
) -> Path:
    """写出 reset 可见对象的独立 world xyz 俯视图。"""
    import matplotlib.pyplot as plt

    plot_path = _visible_object_plot_path(reset_output_dir)
    objects = list(visible_payload["objects"])

    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111)
    points_xyz: list[list[float]] = []

    for item in objects:
        seg_id = int(item["segmentation_id"])
        point_xyz = np.asarray(item["world_xyz"], dtype=np.float64)
        color_rgb = np.asarray(
            SEGMENTATION_COLOR_MAP.get(seg_id, [255, 255, 255]),
            dtype=np.float64,
        ) / 255.0
        visible_in = item["visible_in"]
        tag = ""
        if visible_in.get("base_camera"):
            tag += "B"
        if visible_in.get("hand_camera"):
            tag += "H"
        plot_x, plot_y = _xy_rot_cw_90(point_xyz[0], point_xyz[1])

        axis.scatter(
            plot_x,
            plot_y,
            s=80,
            color=color_rgb,
            edgecolors="black",
            linewidths=0.6,
        )
        axis.text(
            plot_x,
            plot_y,
            f"{item['name']} ({tag or '-'})",
            fontsize=8,
        )
        points_xyz.append(point_xyz.tolist())

    points_array = (
        np.asarray(points_xyz, dtype=np.float64)
        if points_xyz
        else np.zeros((0, 3), dtype=np.float64)
    )
    _set_equal_xy_axes(axis, points_array)
    axis.set_xlabel("World Y")
    axis.set_ylabel("-World X")
    axis.set_title(
        f"Visible Object Positions Top-Down (Y, -X)\n{env_id} ep={episode} seed={seed}"
    )
    axis.grid(True)

    if not objects:
        axis.text(0.0, 0.0, "No visible objects", fontsize=10)

    figure.tight_layout()
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)
    return plot_path


def _save_reset_visible_object_artifacts(
    obs: object,
    env: gym.Env,
    reset_output_dir: Path,
    env_id: str,
    episode: int,
    seed: int,
) -> tuple[Path, Path]:
    """导出 reset 可见对象 JSON 与独立 3D 位置图。"""
    visible_payload = _collect_reset_visible_objects(obs=obs, env=env)
    json_path = _save_visible_objects_json(
        reset_output_dir=reset_output_dir,
        env_id=env_id,
        episode=episode,
        seed=seed,
        visible_payload=visible_payload,
        env=env,
    )
    plot_path = _save_visible_objects_3d_plot(
        reset_output_dir=reset_output_dir,
        env_id=env_id,
        episode=episode,
        seed=seed,
        visible_payload=visible_payload,
    )
    print(f"[Setup] Visible object JSON saved: {json_path.resolve()}")
    print(f"[Setup] Visible object 3D plot saved: {plot_path.resolve()}")
    return json_path, plot_path


def _save_reset_segmentation_pngs(
    obs: object,
    env: gym.Env,
    output_root: Path,
    env_id: str,
    episode: int,
    seed: int,
) -> Path:
    """导出 reset 帧中所有相机的 segmentation 彩色 PNG。

    对每个含有 segmentation 字段的相机，把 seg_id 矩阵通过 color_map
    映射成 RGB 图像并叠加在原始 rgb 帧上，写出到独立的 PNG 文件。
    返回输出目录路径（供后续写 visible_objects JSON 复用）。
    """
    if not isinstance(obs, dict):
        raise ValueError("reset observation is not a dict")

    sensor_data = obs.get("sensor_data")
    if not isinstance(sensor_data, dict):
        raise ValueError("reset observation missing sensor_data dict")

    # 建立输出目录：{output_root}/reset_segmentation_pngs/{env_id}_ep{ep}_seed{seed}/
    output_dir = _reset_segmentation_dir(output_root, env_id, episode, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 从全局色表派生本 episode 的色表（把黑名单对象覆盖为黑色）
    color_map = _build_reset_segmentation_color_map(
        env,
        getattr(env.unwrapped, "segmentation_id_map", None)
    )

    saved_paths: list[Path] = []
    for camera_name, camera_obs in sensor_data.items():
        # 跳过不含 segmentation 的相机（如深度专用相机）
        if not isinstance(camera_obs, dict) or "segmentation" not in camera_obs:
            continue
        # 有 segmentation 就必须有 rgb，否则无法生成叠加图
        if "rgb" not in camera_obs:
            raise ValueError(
                f"camera '{camera_name}' has segmentation but missing rgb frame"
            )

        segmentation = _to_numpy(camera_obs["segmentation"])
        rgb = _to_numpy(camera_obs["rgb"])
        # 去掉 batch 维，保证形状为 [H, W, C]
        if segmentation.ndim >= 4:
            segmentation = segmentation[0]
        if rgb.ndim >= 4:
            rgb = rgb[0]

        # create_segmentation_visuals 把 seg_id 矩阵按 color_map 着色，
        # 并与 rgb 叠加生成可视化图像；后两个返回值（点集、掩码）此处不需要
        segmentation_vis, _, _ = create_segmentation_visuals(
            segmentation=segmentation,
            segmentation_result=segmentation,
            base_frame=rgb,
            color_map=color_map,
            segmentation_points=[],
        )
        png_path = output_dir / f"{camera_name}_segmentation.png"
        try:
            imageio.imwrite(png_path, segmentation_vis)
        except Exception as exc:
            raise RuntimeError(
                f"failed to write segmentation png for camera '{camera_name}' "
                f"to {png_path}: {exc}"
            ) from exc
        saved_paths.append(png_path)

    # 至少要有一张 PNG，否则说明 obs 结构有问题
    if not saved_paths:
        raise ValueError("reset observation contained no camera segmentation data")

    print(f"[Setup] Reset segmentation PNGs saved: {output_dir.resolve()}")
    return output_dir


def _decode_h5_text(value: object) -> str:
    """把 HDF5 中读出的文本值统一解码为 Python str。

    HDF5 字符串数据集读出后可能是 bytes、np.bytes_ 或包含单个元素的 ndarray，
    此函数递归处理所有情况，返回一致的 str。
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        # np.bytes_ 需要先转为内置 bytes 再 decode
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray):
        # scalar 数组或 shape=(1,) 数组：展平后取第一个元素递归处理
        flattened = value.reshape(-1).tolist()
        if not flattened:
            return ""
        return _decode_h5_text(flattened[0])
    # 其他类型（如 str）直接转换
    return str(value)


def _tensor_to_bool(value) -> bool:
    """把 Tensor / ndarray / Python 标量统一转换成布尔值。

    ManiSkill 的 evaluate() 返回值可能是 torch.Tensor（GPU 上）或 np.ndarray，
    统一在 CPU 上取布尔值，避免类型判断遗漏。
    """
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        # 先 detach 再 cpu，避免在 autograd 图中产生副作用
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, np.ndarray):
        # 任意元素为 True 即返回 True（兼容 shape=(1,) 的标量数组）
        return bool(np.any(value))
    return bool(value)


def _create_planner(env: gym.Env, env_id: str):
    """按任务类型选择 stick 规划器或机械臂规划器。

    Imitation 套件 (PatternLock / RouteStick) 由 imitation.select_planner
    返回 stick planner（joint_vel_limits=0.3，避免 stick 震荡）；其余 env
    走默认 Panda 机械臂规划器。
    FailAware 前缀表示规划失败时抛 ScrewPlanFailure，而非静默返回 -1。
    """
    suite_planner = imitation.select_planner(env, env_id)
    if suite_planner is not None:
        return suite_planner
    # 默认：标准 Panda 机械臂规划器（夹爪抓取）
    return FailAwarePandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )


def _wrap_planner_with_screw_then_rrt_retry(planner) -> None:
    """给 screw 规划器的 move_to_pose_with_screw 方法打补丁，加入多次重试与 RRT* 兜底。

    策略：
    1. 先尝试 screw 规划（速度快），最多 DATASET_SCREW_MAX_ATTEMPTS 次；
    2. 若 screw 全部失败（抛 ScrewPlanFailure 或返回 -1），切换到 RRT* 规划，
       最多 DATASET_RRT_MAX_ATTEMPTS 次；
    3. RRT* 也全部失败时返回 -1，由调用方标记 episode 失败。

    用 monkey-patch 方式修改，避免改动 planner 类本身。
    """
    # 保存原始方法引用，供内部闭包调用
    original_screw = planner.move_to_pose_with_screw
    original_rrt = planner.move_to_pose_with_RRTStar

    def _retry(*args, **kwargs):
        # 第一阶段：screw 规划重试
        for attempt in range(1, DATASET_SCREW_MAX_ATTEMPTS + 1):
            try:
                result = original_screw(*args, **kwargs)
            except ScrewPlanFailure as exc:
                # ScrewPlanFailure 是预期的可重试失败类型
                print(
                    f"[Replay] screw planning failed "
                    f"(attempt {attempt}/{DATASET_SCREW_MAX_ATTEMPTS}): {exc}"
                )
                continue
            # 返回 -1 同样视为失败（规划器内部约定）
            if isinstance(result, int) and result == -1:
                print(
                    f"[Replay] screw planning returned -1 "
                    f"(attempt {attempt}/{DATASET_SCREW_MAX_ATTEMPTS})"
                )
                continue
            # 规划成功，直接返回
            return result

        # 第二阶段：screw 耗尽后降级到 RRT*
        print(
            "[Replay] screw planning exhausted; "
            f"fallback to RRT* (max {DATASET_RRT_MAX_ATTEMPTS} attempts)"
        )
        for attempt in range(1, DATASET_RRT_MAX_ATTEMPTS + 1):
            try:
                result = original_rrt(*args, **kwargs)
            except Exception as exc:
                print(
                    f"[Replay] RRT* planning failed "
                    f"(attempt {attempt}/{DATASET_RRT_MAX_ATTEMPTS}): {exc}"
                )
                continue
            if isinstance(result, int) and result == -1:
                print(
                    f"[Replay] RRT* planning returned -1 "
                    f"(attempt {attempt}/{DATASET_RRT_MAX_ATTEMPTS})"
                )
                continue
            return result

        # 两级规划全部耗尽，返回 -1 告知调用方放弃
        print("[Replay] screw->RRT* planning exhausted; return -1")
        return -1

    # 用包装后的 _retry 替换 planner 实例方法
    planner.move_to_pose_with_screw = _retry


def _execute_task_list(env: gym.Env, planner, env_id: str) -> bool:
    """按任务列表执行 evaluate -> solve -> evaluate 主循环。

    每个 task_entry 包含：
    - "name": 任务名称（仅用于日志）
    - "solve": callable(env, planner) -> int | None，执行具体规划动作

    执行流程：
    1. 调用 env.unwrapped.evaluate() 初始化状态
    2. 逐个执行 task，调用 solve 后立即 evaluate 检查 success/fail
    3. 任意 task 成功（success=True）或失败（fail=True / screw 失败）时提前退出循环
    4. 正常循环结束（for...else）时再做一次最终 evaluate

    返回 True 表示本 episode 成功（wrapper 层或底层 env 任一标记成功均可）。
    """
    # 初始化 evaluate 状态，确保内部计数器处于正确起点
    env.unwrapped.evaluate()
    tasks = list(getattr(env.unwrapped, "task_list", []) or [])
    print(f"{env_id}: Task list has {len(tasks)} tasks")

    episode_successful = False

    for idx, task_entry in enumerate(tasks):
        task_name = task_entry.get("name", f"Task {idx}")
        print(f"Executing task {idx + 1}/{len(tasks)}: {task_name}")

        solve_callable = task_entry.get("solve")
        if not callable(solve_callable):
            raise ValueError(f"Task '{task_name}' must supply a callable 'solve'.")

        # solve_complete_eval=True 触发完整的子任务状态机评估
        env.unwrapped.evaluate(solve_complete_eval=True)
        screw_failed = False
        try:
            solve_result = solve_callable(env, planner)
            # solve 返回 -1 表示规划全部耗尽
            if isinstance(solve_result, int) and solve_result == -1:
                screw_failed = True
                print(f"Screw->RRT* planning exhausted during '{task_name}'")
                # 手动写入失败标志，以便 evaluate 能正确判断
                env.unwrapped.failureflag = torch.tensor([True])
                env.unwrapped.successflag = torch.tensor([False])
                env.unwrapped.current_task_failure = True
        except ScrewPlanFailure as exc:
            # ScrewPlanFailure 在包装器未能捕获时抛出，直接视为失败
            screw_failed = True
            print(f"Screw plan failure during '{task_name}': {exc}")
            env.unwrapped.failureflag = torch.tensor([True])
            env.unwrapped.successflag = torch.tensor([False])
            env.unwrapped.current_task_failure = True
        except FailsafeTimeout as exc:
            # FailsafeTimeout：步数或时间超限，停止整个 task 序列
            print(f"Failsafe: {exc}")
            break

        # solve 完成后立即 evaluate 获取当前成功/失败状态
        evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
        fail_flag = evaluation.get("fail", False)
        success_flag = evaluation.get("success", False)

        # 整个 episode 成功（所有子任务完成），提前退出
        if _tensor_to_bool(success_flag):
            print("All tasks completed successfully.")
            episode_successful = True
            break

        # 遇到失败条件（规划失败或环境报告 fail），停止后续 task
        if screw_failed or _tensor_to_bool(fail_flag):
            print("Encountered failure condition; stopping task sequence.")
            break
    else:
        # for 循环正常结束（未 break），做最终一次 evaluate
        evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
        episode_successful = _tensor_to_bool(evaluation.get("success", False))

    # 同时检查 wrapper 层的 episode_success，取两者的 OR
    return episode_successful or _tensor_to_bool(
        getattr(env, "episode_success", False)
    )


def _mark_episode_failed(env: Optional[gym.Env], reason: str) -> None:
    """在 close 前把 wrapper 和底层 env 的状态统一压成失败。

    需要同时设置三层状态，确保 RecordWrapper 写出的 HDF5 正确标记为失败：
    - env.episode_success：wrapper 层的成功标志
    - base_env.failureflag / successflag：底层 env 评估状态
    - base_env.current_task_failure：当前子任务失败标志
    """
    if env is None:
        return
    if hasattr(env, "episode_success"):
        env.episode_success = False
    base_env = getattr(env, "unwrapped", None)
    if base_env is None:
        return
    # 使用 shape=(1,) 的 Tensor，与环境内部约定一致
    base_env.failureflag = torch.tensor([True])
    base_env.successflag = torch.tensor([False])
    base_env.current_task_failure = True
    print(f"[Replay] Episode failure forced before close: {reason}")


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
            "Generate Robomme rollout HDF5 files with replay validation: every "
            "attempt that writes an H5 is immediately replayed; only seeds whose "
            "replay outcome == 'success' are accepted. Uses the legacy "
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
        default=10,
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
        default=DEFAULT_GPU_ID,
        choices=list(ALLOWED_GPU_IDS),
        help=(
            "GPU id to expose via CUDA_VISIBLE_DEVICES. "
            f"Default: {DEFAULT_GPU_ID}. Must be parsed BEFORE any heavy imports, "
            "so the same value is also picked up by the early module-level parser."
        ),
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
            "the same env_id. Default: auto=episode count (no CPU cap)."
        ),
    )
    parser.add_argument(
        "--replay-action-space",
        type=str,
        default="joint_angle",
        choices=("joint_angle", "ee_pose", "waypoint", "multi_choice"),
        help=(
            "Action space used when replaying the freshly-generated HDF5 inside "
            "the retry loop. Each rollout attempt is only accepted when the replay "
            "outcome equals 'success'. Default: joint_angle (matches the "
            "RobommeRecordWrapper pd_joint_pos controller, most reproducible)."
        ),
    )
    return parser


def _build_env_kwargs(episode: int, seed: int, difficulty: str) -> dict:
    """构造环境参数，并按 episode 编号启用不同级别的失败恢复。

    失败恢复策略（robomme_failure_recovery）：
    - episode <= 2：仅在 Z 轴方向恢复（最严格，避免水平方向的物理干扰）
    - episode 3-5：在 XY 平面方向恢复（允许更多位置修正）
    - episode > 5：不启用失败恢复（依赖规划器自身处理）
    这是针对前几个 episode（通常用于验证/调试）的特殊照顾。
    """
    env_kwargs = dict(
        obs_mode="rgb+depth+segmentation",   # 同时获取 RGB / 深度 / 分割图
        control_mode="pd_joint_pos",          # PD 控制器驱动关节位置
        render_mode="rgb_array",              # 离屏渲染，支持 GPU 无 display
        reward_mode="dense",                  # 使用稠密奖励，便于规划器跟踪进度
        seed=seed,
        difficulty=difficulty,
    )
    if episode <= 5:
        env_kwargs["robomme_failure_recovery"] = True
        if episode <= 2:
            # 仅 Z 轴恢复：抬起后下放，避免碰撞
            env_kwargs["robomme_failure_recovery_mode"] = "z"
        else:
            # XY 平面恢复：允许水平位置的重新对齐
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


# ─────────────────────────────────────────────────────────────────────────────
# Replay validation helpers
#
# 每次 rollout 写出 H5 后，retry 循环立刻用 dataset_replay-forSegmentation 的
# process_episode 复现该 H5。outcome 必须为 "success" 才算这次 attempt 通过；
# 任何其他 outcome（fail/unknown/error/...）都换 seed 重试。
# ─────────────────────────────────────────────────────────────────────────────

# 模块级 lazy-loaded：dataset_replay-forSegmentation.py 文件名含 '-'，不能直接
# import；首个 worker attempt 内做一次 importlib.util 加载并缓存。
_REPLAY_MODULE = None


def _load_replay_module():
    """Lazy-load dataset_replay-forSegmentation.py via importlib.util."""
    global _REPLAY_MODULE
    if _REPLAY_MODULE is None:
        import importlib.util

        replay_path = Path(__file__).parent / "dataset_replay-forSegmentation.py"
        spec = importlib.util.spec_from_file_location(
            "dataset_replay_forSegmentation", replay_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Failed to build importlib spec for {replay_path}"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # 防止 dataset_replay 后续重构悄悄改掉签名/行为
        if not hasattr(module, "process_episode"):
            raise RuntimeError(
                "dataset_replay-forSegmentation.py missing process_episode entry"
            )
        _REPLAY_MODULE = module
    return _REPLAY_MODULE


def _replay_validation_video_dir(output_dir: Path) -> Path:
    """retry 阶段 replay 视频写到 output_dir/replay_validation_videos/。

    避免污染 dataset_replay 默认的 runs/replay_videos/，且与 rollout artifacts
    同根目录便于按 (env, episode) 整体清理。
    """
    return output_dir / "replay_validation_videos"


def _replay_validate_h5(
    h5_path: Path,
    env_id: str,
    episode: int,
    action_space_type: str,
    output_dir: Path,
) -> str:
    """Replay the freshly-generated H5 in-process; return the outcome string.

    Possible returns: "success" | "fail" | "failure" | "timeout" | "unknown" |
    "error"（"error" = process_episode 自身抛异常）。
    """
    if not h5_path.is_file():
        print(f"[Replay] missing h5 file {h5_path}; skipping replay")
        return "error"

    module = _load_replay_module()
    video_dir = _replay_validation_video_dir(output_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    try:
        return module.process_episode(
            h5_path,
            env_id,
            episode,
            action_space_type,
            video_dir_override=video_dir,
        )
    except Exception as exc:
        print(
            f"[Replay] process_episode raised for env={env_id} episode={episode}: "
            f"{type(exc).__name__}: {exc}"
        )
        return "error"


def _cleanup_failed_episode_artifacts(
    output_dir: Path,
    env_id: str,
    episode: int,
    seed: int,
) -> None:
    """Delete rollout artifacts for a failed seed before trying the next one.

    清理范围（避免污染 downstream 数据集 / inspect_stat）：
    - hdf5_files/{env_id}_ep{episode}_seed{seed}.h5
    - reset_segmentation_pngs/{env_id}_ep{episode}_seed{seed}/

    刻意保留（debug 价值高、不污染 downstream）：
    - videos/*{env_id}_ep{episode}_seed{seed}*.mp4 —— 所有失败 attempt 的 mp4
      （含 FAILED_ / success_NO_OBJECT_ 前缀）都留下，便于肉眼查看为什么 retry。
      videos-success/ 的复制由 _pick_videos_success_candidate 按 used_seed
      过滤，只复制 retry 最终接受 seed 的 success mp4，不会被失败 attempt 污染。
    - replay_validation_videos/{outcome}_<...>.mp4 —— retry 中每次 replay 都会
      写一个带 outcome 前缀的 mp4，全部保留。
    """
    h5_path = _episode_h5_path(output_dir, env_id, episode, seed)
    h5_path.unlink(missing_ok=True)

    reset_dir = _reset_segmentation_dir(output_dir, env_id, episode, seed)
    if reset_dir.is_dir():
        shutil.rmtree(reset_dir, ignore_errors=True)


def _append_replay_log(
    output_dir: Path,
    env_id: str,
    episode: int,
    seed: int,
    attempt: int,
    replay_outcome: str,
) -> None:
    """Append a 'kind=replay' record to episode_results.jsonl.

    Schema 与 rollout 行不同（kind 字段区分），inspect 阶段可按 kind 聚合
    rollout-fail 与 replay-fail。
    """
    now_epoch = time.time()
    record = {
        "timestamp": now_epoch,
        "timestamp_iso": time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(now_epoch)
        ),
        "kind": "replay",
        "env_id": env_id,
        "episode": episode,
        "attempt": attempt,
        "seed": seed,
        "replay_outcome": replay_outcome,
    }
    _append_episode_log(_episode_log_path(output_dir), record)


def _run_episode(
    env_id: str,
    episode: int,
    seed: int,
    difficulty: str,
    output_dir: Path,
    attempt: int = 1,
) -> tuple[bool, bool]:
    """跑一次完整 rollout：create_env → reset → save artifacts → planner → close。

    返回 `(rollout_ok, retryable_failure)`：
    - rollout_ok=True：planner 全部 task 成功 + HDF5 已写出。仅代表 rollout 通过；
      replay 验证由外层 `_run_episode_with_retry` 负责。
    - retryable_failure=True：失败发生在 env 创建 / reset 阶段。retry 不再依据
      该字段分支（任何 rollout_ok=False 都会换 seed 重试），保留它只为 JSONL
      诊断之用。

    attempt 是 1-indexed，仅用于 JSONL 日志。
    """
    print(
        f"--- [rollout] env={env_id} episode={episode} "
        f"seed={seed} difficulty={difficulty} ---"
    )

    env: Optional[gym.Env] = None
    obs: Optional[dict] = None
    # 预先计算本次 episode 对应的 HDF5 路径（env.close() 后才实际写出）
    h5_path = _episode_h5_path(output_dir, env_id, episode, seed)
    log_path = _episode_log_path(output_dir)

    # 累积式 record：每个 stage 完成后填字段，return 前 _emit() 写一行 JSONL。
    now_epoch = time.time()
    record: dict = {
        "timestamp": now_epoch,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now_epoch)),
        "kind": "rollout",
        "env_id": env_id,
        "episode": episode,
        "attempt": attempt,
        "seed": seed,
        "difficulty": difficulty,
        "scene_generation": "not_attempted",
        "scene_generation_error": None,
        "execute": "not_attempted",
        "execute_error": None,
        "artifacts": "not_attempted",
        "artifacts_error": None,
        "hdf5_verify": "not_attempted",
        "hdf5_verify_message": None,
        "mp4_recorded": False,
        "h5_path": str(h5_path),
        "final_status": "failed",
        "retryable_failure": False,
    }

    def _emit(success: bool, retryable: bool) -> None:
        record["final_status"] = "success" if success else "failed"
        record["retryable_failure"] = bool(retryable)
        _append_episode_log(log_path, record)

    # ── 1. 创建环境 + reset ─────────────────────────────────────────────────
    try:
        env_kwargs = _build_env_kwargs(episode, seed, difficulty)
        env = _create_env(env_id, env_kwargs, output_dir, episode, seed)
        obs, _ = env.reset()
    except SceneGenerationError as exc:
        print(
            f"[rollout] Scene generation failed for env={env_id} "
            f"episode={episode} seed={seed}: {exc}"
        )
        _close_env(env, episode, seed)
        record["scene_generation"] = "failed"
        record["scene_generation_error"] = _format_exception(exc)
        _emit(success=False, retryable=True)
        return False, True
    except Exception as exc:
        print(
            f"[rollout] Failed during env creation/reset for env={env_id} "
            f"episode={episode} seed={seed}: {type(exc).__name__}: {exc}"
        )
        _close_env(env, episode, seed)
        record["scene_generation"] = "failed"
        record["scene_generation_error"] = _format_exception(exc)
        _emit(success=False, retryable=True)
        return False, True

    record["scene_generation"] = "success"

    # ── 2. 导出 reset 帧的 segmentation / visible objects ──────────────────
    try:
        reset_output_dir = _save_reset_segmentation_pngs(
            obs=obs,
            env=env,
            output_root=output_dir,
            env_id=env_id,
            episode=episode,
            seed=seed,
        )
        _save_reset_visible_object_artifacts(
            obs=obs,
            env=env,
            reset_output_dir=reset_output_dir,
            env_id=env_id,
            episode=episode,
            seed=seed,
        )
    except Exception as exc:
        print(
            f"[rollout] Failed to save reset artifacts for env={env_id} "
            f"episode={episode} seed={seed}: {type(exc).__name__}: {exc}"
        )
        _close_env(env, episode, seed)
        record["artifacts"] = "failed"
        record["artifacts_error"] = _format_exception(exc)
        _emit(success=False, retryable=False)
        return False, False

    record["artifacts"] = "success"

    # ── 3. 跑 planner / task list ──────────────────────────────────────────
    episode_successful = False
    try:
        planner = _create_planner(env, env_id)
        _wrap_planner_with_screw_then_rrt_retry(planner)
        episode_successful = _execute_task_list(env, planner, env_id)
    except Exception as exc:
        print(
            f"[rollout] Planner/task execution failed for env={env_id} "
            f"episode={episode} seed={seed}: {type(exc).__name__}: {exc}"
        )
        _mark_episode_failed(env, f"exception: {type(exc).__name__}")
        episode_successful = False
        record["execute_error"] = f"exception: {_format_exception(exc)}"
    finally:
        if not episode_successful:
            _mark_episode_failed(env, "task_not_successful")
        _close_env(env, episode, seed)

    if episode_successful:
        record["execute"] = "success"
    else:
        record["execute"] = "failed"
        if record["execute_error"] is None:
            record["execute_error"] = "task_not_successful"

    mp4_path = _latest_recorded_mp4(output_dir, env_id, episode, seed)
    status_text = "SUCCESS" if episode_successful else "FAILED"
    print(
        f"--- [rollout] Finished env={env_id} episode={episode} seed={seed} "
        f"difficulty={difficulty} [{status_text}] ---"
    )
    if mp4_path is not None:
        print(f"[rollout] MP4 recorded: {mp4_path.resolve()}")
    else:
        print(
            f"[rollout] Warning: no MP4 matched under {output_dir / 'videos'} "
            f"(expected filename fragment '{env_id}_ep{episode}_seed{seed}')."
        )

    record["hdf5_verify"] = "success" if h5_path.is_file() else "failed"
    record["hdf5_verify_message"] = (
        "h5 file present"
        if h5_path.is_file()
        else f"missing HDF5 file: {h5_path}"
    )
    record["mp4_recorded"] = mp4_path is not None
    _emit(success=episode_successful, retryable=False)
    return episode_successful, False


def _run_episode_with_retry(
    env_id: str,
    episode: int,
    difficulty: str,
    output_dir: Path,
    replay_action_space: str = "joint_angle",
) -> tuple[bool, int]:
    """跑同一 episode 直到 rollout + replay 双双成功；否则按 legacy seed 序列重试。

    每个 attempt 严格顺序：

        Step 1: rollout
            _run_episode(seed) 写出 H5
            失败 → 清理本 seed artifacts，换下一个 seed

        Step 2: replay 验证
            _replay_validate_h5(h5)，outcome 必须 == "success"
            其他 outcome（fail/unknown/error/...）→ 清理本 seed artifacts，
            换下一个 seed

    seed 序列：base_seed, base_seed+1, ..., base_seed+MAX_SEED_ATTEMPTS-1。
    用尽 MAX_SEED_ATTEMPTS 仍无 (rollout+replay) 双成功 → 返回 (False, last_seed)。

    返回 (success, used_seed)。
    """
    base_seed = _base_seed_for_episode(env_id, episode)
    print(
        f"[Retry] env={env_id} episode={episode} "
        f"base_seed={base_seed} difficulty={difficulty} "
        f"max_attempts={MAX_SEED_ATTEMPTS} replay_action_space={replay_action_space}"
    )

    last_seed = base_seed
    for attempt in range(MAX_SEED_ATTEMPTS):
        seed = base_seed + attempt
        last_seed = seed
        attempt_no = attempt + 1
        print(
            f"[Retry] env={env_id} episode={episode} "
            f"attempt={attempt_no}/{MAX_SEED_ATTEMPTS} seed={seed}"
        )

        # ── Step 1: rollout ────────────────────────────────────────────────
        try:
            rollout_ok, _retryable = _run_episode(
                env_id=env_id,
                episode=episode,
                seed=seed,
                difficulty=difficulty,
                output_dir=output_dir,
                attempt=attempt_no,
            )
        except Exception as exc:
            print(
                f"[Retry] rollout raised for env={env_id} episode={episode} "
                f"seed={seed}: {type(exc).__name__}: {exc}"
            )
            rollout_ok = False

        if not rollout_ok:
            print(
                f"[Retry] rollout failed for seed={seed}; cleaning artifacts "
                "and trying next seed."
            )
            _cleanup_failed_episode_artifacts(output_dir, env_id, episode, seed)
            continue

        # ── Step 2: replay 验证 ────────────────────────────────────────────
        h5_path = _episode_h5_path(output_dir, env_id, episode, seed)
        replay_outcome = _replay_validate_h5(
            h5_path=h5_path,
            env_id=env_id,
            episode=episode,
            action_space_type=replay_action_space,
            output_dir=output_dir,
        )
        _append_replay_log(
            output_dir, env_id, episode, seed, attempt_no, replay_outcome
        )
        print(
            f"[Retry] replay outcome for env={env_id} episode={episode} "
            f"seed={seed}: {replay_outcome}"
        )

        if replay_outcome == "success":
            print(
                f"[Retry] env={env_id} episode={episode} "
                f"succeeded with seed={seed} on attempt "
                f"{attempt_no}/{MAX_SEED_ATTEMPTS}"
            )
            return True, seed

        print(
            f"[Retry] replay outcome={replay_outcome} for seed={seed}; "
            "cleaning artifacts and trying next seed."
        )
        _cleanup_failed_episode_artifacts(output_dir, env_id, episode, seed)

    print(
        f"[Retry] env={env_id} episode={episode} exhausted "
        f"{MAX_SEED_ATTEMPTS} attempts; last_seed={last_seed}"
    )
    return False, last_seed


def _resolve_max_workers(requested: Optional[int], n_episodes: int) -> int:
    """计算实际启用的 worker 数量。"""
    if requested is not None and requested < 1:
        raise SystemExit("--max-workers must be at least 1.")
    if requested is None:
        return max(n_episodes, 1)
    return min(requested, max(n_episodes, 1))


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

    # 把"合格的成功样本"额外复制一份到 videos-success/。合格性由
    # _pick_videos_success_candidate 单独判定（排除 FAILED_ 与
    # success_NO_OBJECT_ 两类前缀），独立于上面 best-effort mp4 的 mtime
    # 选择，避免在边角情况下错选 NO_OBJECT 视频。
    success_candidate = _pick_videos_success_candidate(
        output_dir, env_id, episode, run_seed
    )
    if success_candidate is not None:
        success_copy = _copy_to_success_dir(output_dir, success_candidate)
        if success_copy is not None:
            print(
                f"Success MP4 copy ({env_id} episode {episode}, "
                f"seed={run_seed}): {success_copy.resolve()}"
            )

    png_dir = _reset_segmentation_dir(output_dir, env_id, episode, run_seed)
    if png_dir.is_dir():
        print(
            f"Reset segmentation PNGs ({env_id} episode {episode}, seed={run_seed}): "
            f"{png_dir.resolve()}"
        )
    else:
        print(
            f"Missing reset segmentation PNGs ({env_id} episode {episode}, "
            f"seed={run_seed}): {png_dir}"
        )

    visible_json_path = _visible_object_json_path(png_dir)
    if visible_json_path.is_file():
        print(
            f"Visible object JSON ({env_id} episode {episode}, seed={run_seed}): "
            f"{visible_json_path.resolve()}"
        )
    else:
        print(
            f"Missing visible object JSON ({env_id} episode {episode}, "
            f"seed={run_seed}): {visible_json_path}"
        )

    visible_plot_path = _visible_object_plot_path(png_dir)
    if visible_plot_path.is_file():
        print(
            f"Visible object 3D PNG ({env_id} episode {episode}, seed={run_seed}): "
            f"{visible_plot_path.resolve()}"
        )
    else:
        print(
            f"Missing visible object 3D PNG ({env_id} episode {episode}, "
            f"seed={run_seed}): {visible_plot_path}"
        )


def _run_env_episodes(
    env_id: str,
    episode_specs: list[tuple[int, str]],
    output_dir: Path,
    max_workers: int,
    replay_action_space: str = "joint_angle",
) -> list[bool]:
    """在同一个 env_id 内并行执行多个 episode，返回按 episode 编号排序的成功列表。

    max_workers=1 时退化为串行（方便调试和单进程环境）。
    max_workers>1 时使用 ProcessPoolExecutor（spawn 模式），每个 worker 独立持有
    一个物理仿真环境，避免 SAPIEN/MuJoCo 的全局状态冲突。

    注意：spawn 模式下子进程会重新 import 整个模块，_early_pin_gpu 会在子进程中
    再次执行，并通过 _GPU_PIN_ENV_MARKER 环境变量恢复父进程设定的 gpu_id。
    """
    # (episode_idx, success) 列表，后续按 episode_idx 排序后返回
    env_successes: list[tuple[int, bool]] = []

    # 串行路径：max_workers=1，直接在当前进程中执行
    if max_workers == 1:
        for ep, difficulty in episode_specs:
            success, used_seed = _run_episode_with_retry(
                env_id=env_id,
                episode=ep,
                difficulty=difficulty,
                output_dir=output_dir,
                replay_action_space=replay_action_space,
            )
            env_successes.append((ep, success))
            _print_episode_artifacts(output_dir, env_id, ep, used_seed)
        return [success for _, success in sorted(env_successes)]

    # 并行路径：spawn 新进程池，同时处理多个 episode
    global _ACTIVE_EXECUTOR
    # spawn 模式：子进程从零开始，不继承父进程的 Python 对象（包括 GPU 上下文）
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_worker_initializer,  # 忽略 SIGINT，由父进程统一关闭
    ) as executor:
        # 将 executor 注册到全局变量，供信号处理器在 Ctrl-C 时强制关闭
        _ACTIVE_EXECUTOR = executor
        try:
            # 一次性提交所有 episode，executor 自动调度到空闲 worker
            futures = {
                executor.submit(
                    _run_episode_with_retry,
                    env_id,
                    ep,
                    difficulty,
                    output_dir,
                    replay_action_space,
                ): ep
                for ep, difficulty in episode_specs
            }

            # as_completed：哪个 future 先完成就先处理，不等顺序
            for future in as_completed(futures):
                ep = futures[future]
                try:
                    success, used_seed = future.result()
                except Exception as exc:
                    # worker 崩溃（OOM、SIGKILL 等），直接向上抛出终止整个 env
                    raise RuntimeError(
                        f"Worker crashed for env={env_id} episode={ep}"
                    ) from exc
                env_successes.append((ep, success))
                print(
                    f"[Parent] Completed env={env_id} episode={ep} seed={used_seed} "
                    f"success={success}"
                )
                _print_episode_artifacts(output_dir, env_id, ep, used_seed)
        finally:
            # 无论正常退出还是异常，清空全局引用防止 atexit 重复操作
            _ACTIVE_EXECUTOR = None

    # 按 episode 编号排序后返回，保证输出列表与 episode_specs 顺序一致
    return [success for _, success in sorted(env_successes)]


def main() -> None:
    """脚本入口：解析参数、执行 episode 生成、打印结果路径。

    执行流程：
    1. 解析命令行参数（包含对 --gpu 一致性的检查）
    2. 安装父进程信号处理器（Ctrl-C 优雅关闭）
    3. 构建 difficulty_cycle：按比例展开 easy/medium/hard 的循环序列
    4. 对每个 env_id 串行调用 _run_env_episodes（env 内部并行）
    5. 汇总并打印整体成功/失败状态
    """
    args = _build_parser().parse_args()

    # 验证完整 parser 与早期 GPU 绑定 parser 的 --gpu 一致，防止因参数顺序问题
    # 导致早期已绑定了 GPU X，但完整 parser 解析出 GPU Y 的矛盾
    if int(args.gpu) != _PINNED_GPU_ID:
        raise SystemExit(
            f"GPU mismatch: full parser saw --gpu={args.gpu} but module-level "
            f"early parser already pinned CUDA_VISIBLE_DEVICES to {_PINNED_GPU_ID}. "
            "Pass --gpu before any other arguments so it is parsed before heavy "
            "GPU-touching imports run."
        )
    print(
        f"[GPU] CUDA_VISIBLE_DEVICES pinned to '{os.environ.get('CUDA_VISIBLE_DEVICES')}' "
        f"(GPU id {_PINNED_GPU_ID}); GPU0 must remain untouched."
    )

    # 安装 SIGINT/SIGTERM 处理器，确保 Ctrl-C 时 worker 进程被回收
    _install_parent_signal_handlers()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_episodes = args.episode_number
    if n_episodes < 1:
        raise SystemExit("--episode-number must be at least 1 (run episodes 0..N-1).")
    # seed 空间按 MAX_EPISODES_PER_ENV=1000 分块，超出会导致不同 episode 的 seed 重叠
    if n_episodes >= MAX_EPISODES_PER_ENV:
        raise SystemExit(
            f"--episode-number must be less than {MAX_EPISODES_PER_ENV}; "
            "the legacy seed layout reserves 1000 episode slots per environment."
        )

    replay_action_space: str = args.replay_action_space

    episode_numbers = list(range(0, n_episodes))
    print("Mode: rollout + replay validation")
    print(f"Replay action space: {replay_action_space}")
    print(f"Environments (in order): {args.env}")
    print(
        f"Episode number N={n_episodes} -> running episode indices {episode_numbers} "
        f"(0 .. {n_episodes - 1})"
    )
    print(
        "Seed policy: "
        "base_seed = 1500000 + env_code * 100000 + episode * 100; "
        f"each episode retries up to {MAX_SEED_ATTEMPTS} seeds "
        "until rollout + replay both succeed."
    )

    # 构建难度循环序列：把 [easy, medium, hard] 按 ratio 展开成一个周期列表
    # 例如 ratio=[2,1,1] -> ["easy","easy","medium","hard","easy","easy","medium","hard",...]
    difficulty_cycle = [
        difficulty
        for difficulty, count in zip(DIFFICULTY_ORDER, args.difficulty)
        for _ in range(count)
    ]
    # 按 episode 编号对难度循环取模，得到每个 episode 的难度
    difficulty_preview = [
        difficulty_cycle[ep % len(difficulty_cycle)] for ep in episode_numbers
    ]
    # episode_specs：[(episode_idx, difficulty_str), ...]，传给 _run_env_episodes
    episode_specs = list(zip(episode_numbers, difficulty_preview))
    print(f"Difficulty ratio [easy, medium, hard]: {args.difficulty}")
    print(f"Difficulty per episode: {difficulty_preview}")
    print(f"GPU: {args.gpu}")
    worker_count = _resolve_max_workers(args.max_workers, len(episode_numbers))
    print(f"Max workers per env: {worker_count}")
    print(f"Output root: {output_dir}")
    print(f"HDF5 directory: {_dataset_hdf5_dir(output_dir)}")
    print(f"Episode results JSONL: {_episode_log_path(output_dir)}")

    successes: list[bool] = []
    # 按顺序处理每个 env_id（env 间串行，env 内并行）
    for env_id in args.env:
        print(f"\n========== env={env_id} [rollout+replay] ==========")
        print(
            f"Dispatching {len(episode_numbers)} episodes for env={env_id} "
            f"with up to {worker_count} worker process(es)."
        )
        successes.extend(
            _run_env_episodes(
                env_id=env_id,
                episode_specs=episode_specs,
                output_dir=output_dir,
                max_workers=worker_count,
                replay_action_space=replay_action_space,
            )
        )

    # 汇总打印：所有 episode 全部成功才视为整体成功
    if all(successes):
        print("Generation finished successfully (rollout + replay, all episodes).")
    else:
        print(
            "Generation finished with failure status (rollout + replay; "
            "one or more episodes missing verified output)."
        )


if __name__ == "__main__":
    main()
