import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Set
import h5py

# 将父目录添加到 Python 路径，以便导入 historybench 模块
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "scripts"))
import gymnasium as gym

# 导入 HistoryBench 相关的环境包装器和异常类
from historybench.env_record_wrapper import HistoryBenchRecordWrapper, FailsafeTimeout
from historybench.HistoryBench_env import *
from historybench.HistoryBench_env.errors import SceneGenerationError

# from util import *
import torch

# 导入规划器和相关异常类
from planner_fail_safe import (
    FailAwarePandaArmMotionPlanningSolver,
    FailAwarePandaStickMotionPlanningSolver,
    ScrewPlanFailure,
)

"""
脚本功能：并行生成 HistoryBench 环境的数据集。
该脚本支持多进程并行运行环境模拟，生成包含 RGB、深度、分割等观测数据的 HDF5 数据集。
主要功能包括：
1. 配置环境列表和参数。
2. 并行执行多个 episode 的模拟。
3. 使用 FailAware 规划器尝试解决任务。
4. 记录数据并保存为 HDF5 文件。
5. 将临时生成的多个 HDF5 文件合并为一个最终数据集。
"""

# 所有支持的环境模块名称列表
DEFAULT_ENVS =[
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
'MoveCube',
"PatternLock",
"RouteStick"
    ]

# 参考数据集 metadata 根目录：用于读取 difficulty 和 HistoryBench_seed
SOURCE_METADATA_ROOT = Path("/data/hongzefu/data_1206")
VALID_DIFFICULTIES: Set[str] = {"easy", "medium", "hard"}


def _load_env_metadata_records(
    env_id: str,
    metadata_root: Path,
) -> List[Dict[str, Any]]:
    """
    从参考目录读取某个环境的 metadata 记录，用于控制 difficulty 和 HistoryBench_seed。
    """
    metadata_path = metadata_root / f"record_dataset_{env_id}_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found for env '{env_id}': {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        payload = json.load(metadata_file)

    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(
            f"Metadata file has no valid 'records' list: {metadata_path}"
        )

    normalized_records: List[Dict[str, Any]] = []
    for idx, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise ValueError(
                f"Invalid metadata record at index {idx} in {metadata_path}"
            )
        if "episode" not in raw_record or "seed" not in raw_record or "difficulty" not in raw_record:
            raise ValueError(
                f"Metadata record missing episode/seed/difficulty at index {idx} in {metadata_path}"
            )

        try:
            episode = int(raw_record["episode"])
            seed = int(raw_record["seed"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Metadata record has non-integer episode/seed at index {idx} in {metadata_path}"
            ) from exc

        difficulty_raw = str(raw_record["difficulty"]).strip().lower()
        if difficulty_raw not in VALID_DIFFICULTIES:
            raise ValueError(
                f"Metadata record has invalid difficulty '{raw_record['difficulty']}' "
                f"at index {idx} in {metadata_path}. Expected one of {sorted(VALID_DIFFICULTIES)}."
            )

        normalized_records.append(
            {
                "episode": episode,
                "seed": seed,
                "difficulty": difficulty_raw,
            }
        )

    normalized_records.sort(key=lambda rec: rec["episode"])
    print(
        f"Loaded {len(normalized_records)} metadata records for {env_id} from {metadata_path}"
    )
    return normalized_records


def _build_seed_candidates_from_metadata(
    episode: int,
    metadata_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    构造当前 episode 的候选 (seed, difficulty) 列表。
    严格只使用 metadata 中同 episode 的 seed，不做任何跨 episode 回退。
    """
    if not metadata_records:
        return []

    same_episode_records = [rec for rec in metadata_records if rec["episode"] == episode]
    if not same_episode_records:
        return []
    if len(same_episode_records) > 1:
        raise ValueError(
            f"Found duplicated metadata records for episode {episode}. "
            "Strict mode requires exactly one source record per episode."
        )

    rec = same_episode_records[0]
    return [{"seed": int(rec["seed"]), "difficulty": rec["difficulty"]}]

def _tensor_to_bool(value) -> bool:
    """
    辅助函数：将 Tensor 或 numpy 数组转换为 Python 的 bool 类型。
    用于处理不同来源的成功/失败标志。
    """
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)


def _split_episode_indices(num_episodes: int, max_chunks: int) -> List[List[int]]:
    """
    辅助函数：将总 episode 数量分割成多个块，以便分配给不同的进程并行处理。
    
    Args:
        num_episodes: 总 episode 数量
        max_chunks: 最大分块数（通常等于 worker 数量）
        
    Returns:
        包含 episode 索引列表的列表
    """
    if num_episodes <= 0:
        return []

    chunk_count = min(max_chunks, num_episodes)
    base_size, remainder = divmod(num_episodes, chunk_count)

    chunks: List[List[int]] = []
    start = 0
    for chunk_idx in range(chunk_count):
        # 如果有余数，前 remainder 个 chunk 分配多一个 episode
        stop = start + base_size + (1 if chunk_idx < remainder else 0)
        chunks.append(list(range(start, stop)))
        start = stop

    return chunks


def _run_episode_attempt(
    env_id: str,
    episode: int,
    seed: int,
    temp_dataset_path: Path,
    save_video: bool,
    difficulty: Optional[str],
) -> bool:
    """
    运行单个 episode 的尝试并报告成功或失败。
    
    主要步骤：
    1. 初始化环境参数和 Gym 环境。
    2. 应用 HistoryBenchRecordWrapper 进行数据记录。
    3. 根据环境类型选择合适的规划器 (PandaStick 或 PandaArm)。
    4. 获取任务列表并逐个执行任务。
    5. 使用规划器解决任务，并处理可能的规划失败。
    6. 检查任务执行结果 (fail/success)。
    7. 返回 episode 是否最终成功。
    """
    print(f"--- Running simulation for episode:{episode}, seed:{seed}, env: {env_id} ---")

    env: Optional[gym.Env] = None
    try:
        # 1. 环境参数配置
        env_kwargs = dict(
            obs_mode="rgb+depth+segmentation",  # 观测模式：RGB + 深度 + 分割
            control_mode="pd_joint_pos",        # 控制模式：位置控制
            render_mode="rgb_array",            # 渲染模式
            reward_mode="dense",                # 奖励模式
            HistoryBench_seed=seed,             # 随机种子
            max_episode_steps=200,              # 最大步数
            HistoryBench_difficulty=difficulty, # 难度设置
        )
        
        # 针对前几个 episode 的特殊故障恢复设置 (仅用于测试或演示目的)
        if episode <= 5:
            env_kwargs["historybench_failure_recovery"] = True
            if episode <=2:
                env_kwargs["historybench_failure_recovery_mode"] = "z"  # z轴恢复
            else:
                env_kwargs["historybench_failure_recovery_mode"] = "xy" # xy轴恢复


        env = gym.make(env_id, **env_kwargs)
        
        # 2. 包装环境以记录数据
        env = HistoryBenchRecordWrapper(
            env,
            HistoryBench_dataset=str(temp_dataset_path), # 数据保存路径
            HistoryBench_env=env_id,
            HistoryBench_episode=episode,
            HistoryBench_seed=seed,
            save_video=save_video,

        )

        episode_successful = False


        env.reset()

        # 3. 选择规划器
        # PatternLock 和 RouteStick 需要使用 Stick 规划器，其他使用 Arm 规划器
        if env_id == "PatternLock" or env_id == "RouteStick":
            planner = FailAwarePandaStickMotionPlanningSolver(
                env,
                debug=False,
                vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_vel_limits=0.3,
            )
        else:
            planner = FailAwarePandaArmMotionPlanningSolver(
                env,
                debug=False,
                vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
            )

        env.unwrapped.evaluate()
        # 获取环境的任务列表
        tasks = list(getattr(env.unwrapped, "task_list", []) or [])

        print(f"{env_id}: Task list has {len(tasks)} tasks")

        # 4. 遍历并执行所有子任务
        for idx, task_entry in enumerate(tasks):
            task_name = task_entry.get("name", f"Task {idx}")
            print(f"Executing task {idx + 1}/{len(tasks)}: {task_name}")

            solve_callable = task_entry.get("solve")
            if not callable(solve_callable):
                raise ValueError(
                    f"Task '{task_name}' must supply a callable 'solve'."
                )

            # 在执行 solve 之前进行一次评估
            env.unwrapped.evaluate(solve_complete_eval=True)
            screw_failed = False
            try:
                # 5. 调用规划器解决当前任务
                solve_callable(env, planner)
            except ScrewPlanFailure as exc:
                # 规划失败处理
                screw_failed = True
                print(f"Screw plan failure during '{task_name}': {exc}")
                env.unwrapped.failureflag = torch.tensor([True])
                env.unwrapped.successflag = torch.tensor([False])
                env.unwrapped.current_task_failure = True
            except FailsafeTimeout as exc:
                # 超时处理
                print(f"Failsafe: {exc}")
                break

            # 任务执行后评估
            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)

            fail_flag = evaluation.get("fail", False)
            success_flag = evaluation.get("success", False)

            # 6. 检查成功/失败条件
            if _tensor_to_bool(success_flag):
                print("All tasks completed successfully.")
                episode_successful = True
                break

            if screw_failed or _tensor_to_bool(fail_flag):
                print("Encountered failure condition; stopping task sequence.")
                break

        else:
            # 如果循环正常结束（没有 break），再次检查是否成功
            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            episode_successful = _tensor_to_bool(evaluation.get("success", False))

        # 7. 优先使用 wrapper 的 success 信号 (双重检查)
        episode_successful = episode_successful or _tensor_to_bool(
            getattr(env, "episode_success", False)
        )

    except SceneGenerationError as exc:# swingxtimes 等环境可能出现场景生成失败
        print(
            f"Scene generation failed for env {env_id}, episode {episode}, seed {seed}: {exc}"
        )
        episode_successful = False
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as close_exc:
                # 即使close()失败，如果episode已经成功，仍然返回成功
                # 因为HDF5数据已经在close()之前写入（在write()方法中）
                print(f"Warning: Exception during env.close() for episode {episode}, seed {seed}: {close_exc}")
                # 如果episode已经成功，close()的异常不应该影响返回值
                # episode_successful 已经在close()之前确定

    status_text = "SUCCESS" if episode_successful else "FAILED"
    print(
        f"--- Finished Running simulation for episode:{episode}, seed:{seed}, env: {env_id} [{status_text}] ---"
    )

    return episode_successful


def run_env_dataset(
    env_id: str,
    episode_indices: Iterable[int],
    temp_folder: Path,
    save_video: bool,
    metadata_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    运行一批 episode 的数据集生成，并将数据保存到临时文件夹。
    
    Args:
        env_id: 环境 ID
        episode_indices: 需要运行的 episode 索引列表
        temp_folder: 保存数据的临时文件夹
        save_video: 是否保存视频
        metadata_records: 来源于参考数据集 metadata 的记录
    
    Returns:
        生成的 episode 元数据记录列表
    """
    temp_folder.mkdir(parents=True, exist_ok=True)
    episode_indices = list(episode_indices)
    if not episode_indices:
        return []

    if env_id not in DEFAULT_ENVS:
        raise ValueError(f"Unsupported environment: {env_id}")

    # 使用一个临时的 h5 文件路径传递给 wrapper
    # 注意：wrapper 实际上会在该路径所在的目录下的子文件夹中创建单独的 episode 文件
    temp_dataset_path = temp_folder / f"temp_chunk.h5"
    episode_records: List[Dict[str, Any]] = []

    for episode in episode_indices:
        candidate_pairs = _build_seed_candidates_from_metadata(episode, metadata_records)
        if not candidate_pairs:
            print(f"Episode {episode}: no metadata candidate seeds found, skipping.")
            continue

        episode_success = False
        for attempt_idx, candidate in enumerate(candidate_pairs, start=1):
            seed = int(candidate["seed"])
            difficulty = str(candidate["difficulty"])
            print(
                f"Episode {episode} attempt {attempt_idx}/{len(candidate_pairs)} "
                f"with metadata seed={seed}, difficulty={difficulty}"
            )

            try:
                success = _run_episode_attempt(
                    env_id=env_id,
                    episode=episode,
                    seed=seed,
                    temp_dataset_path=temp_dataset_path,
                    save_video=save_video,
                    difficulty=difficulty,
                )

                if success:
                    # 记录成功的 episode 信息
                    episode_records.append(
                        {
                            "task": env_id,
                            "episode": episode,
                            "seed": seed,
                            "difficulty": difficulty,
                        }
                    )
                    episode_success = True
                    break  # 成功则跳出当前 episode 的 seed 尝试
                print(
                    f"Episode {episode} failed with required source metadata seed {seed}; skipping this episode."
                )
            except Exception as exc:
                print(
                    f"Episode {episode} exception with required source metadata seed {seed}: {exc}; skipping this episode."
                )

        if not episode_success:
            print(
                f"Episode {episode} failed with strict source metadata seed; "
                "metadata will not be recorded for this episode."
            )

    return episode_records


def _merge_dataset_from_folder(
    env_id: str,
    temp_folder: Path,
    final_dataset_path: Path,
) -> None:
    """
    将临时文件夹中的所有 episode 文件合并到最终的数据集中。
    
    Args:
        env_id: 环境 ID
        temp_folder: 包含 episode 文件的临时文件夹
        final_dataset_path: 最终输出的 HDF5 文件路径
    """
    if not temp_folder.exists() or not temp_folder.is_dir():
        print(f"Warning: Temporary folder {temp_folder} does not exist")
        return

    final_dataset_path.parent.mkdir(parents=True, exist_ok=True)

    # 查找 HistoryBenchRecordWrapper 创建的子文件夹
    # 它通常创建以 "_hdf5_files" 结尾的目录
    hdf5_folders = list(temp_folder.glob("*_hdf5_files"))

    if not hdf5_folders:
        print(f"Warning: No HDF5 folders found in {temp_folder}")
        return

    print(f"Merging episodes from {temp_folder} into {final_dataset_path}")

    # 打开最终的 HDF5 文件进行追加模式写入
    with h5py.File(final_dataset_path, "a") as final_file:
        for hdf5_folder in sorted(hdf5_folders):
            # 获取文件夹中所有的 h5 文件
            h5_files = sorted(hdf5_folder.glob("*.h5"))

            if not h5_files:
                print(f"Warning: No h5 files found in {hdf5_folder}")
                continue

            print(f"Found {len(h5_files)} episode files in {hdf5_folder.name}")

            # 合并每个 episode 文件
            for h5_file in h5_files:
                print(f"  - Merging {h5_file.name}")

                try:
                    with h5py.File(h5_file, "r") as episode_file:
                        file_keys = list(episode_file.keys())
                        if len(file_keys) == 0:
                            print(f"    Warning: {h5_file.name} is empty, skipping...")
                            continue
                        
                        for env_group_name, src_env_group in episode_file.items():
                            episode_keys = list(src_env_group.keys()) if isinstance(src_env_group, h5py.Group) else []
                            if len(episode_keys) == 0:
                                print(f"    Warning: {env_group_name} in {h5_file.name} has no episodes, skipping...")
                                continue
                            
                            # 如果环境组（例如 'PickXtimes'）不存在，直接复制
                            if env_group_name not in final_file:
                                final_file.copy(src_env_group, env_group_name)
                                continue

                            dest_env_group = final_file[env_group_name]
                            if not isinstance(dest_env_group, h5py.Group):
                                print(f"    Warning: {env_group_name} is not a group, skipping...")
                                continue

                            # 如果环境组已存在，逐个复制 episode
                            for episode_name in src_env_group.keys():
                                if episode_name in dest_env_group:
                                    print(f"    Warning: Episode {episode_name} already exists, overwriting...")
                                    del dest_env_group[episode_name]
                                src_env_group.copy(episode_name, dest_env_group, name=episode_name)
                except Exception as e:
                    print(f"    Error merging {h5_file.name}: {e}")
                    continue

    # 保留视频：wrapper 会将视频写到临时目录的 videos 下，清理前先转移到最终目录
    temp_videos_dir = temp_folder / "videos"
    final_videos_dir = final_dataset_path.parent / "videos"
    if temp_videos_dir.exists() and temp_videos_dir.is_dir():
        final_videos_dir.mkdir(parents=True, exist_ok=True)
        moved_count = 0
        for video_path in sorted(temp_videos_dir.glob("*.mp4")):
            target_path = final_videos_dir / video_path.name
            if target_path.exists():
                stem = target_path.stem
                suffix = target_path.suffix
                index = 1
                while True:
                    candidate = final_videos_dir / f"{stem}_dup{index}{suffix}"
                    if not candidate.exists():
                        target_path = candidate
                        break
                    index += 1
            try:
                shutil.move(str(video_path), str(target_path))
                moved_count += 1
            except Exception as exc:
                print(f"Warning: Failed to move video {video_path.name}: {exc}")
        if moved_count > 0:
            print(f"Moved {moved_count} videos to {final_videos_dir}")

    # 合并成功后清理临时文件夹
    try:
        shutil.rmtree(temp_folder)
        print(f"Cleaned up temporary folder: {temp_folder}")
    except Exception as e:
        print(f"Warning: Failed to remove temporary folder {temp_folder}: {e}")


def _save_episode_metadata(
    records: List[Dict[str, Any]],
    metadata_path: Path,
    env_id: str,
) -> None:
    """保存每个 episode 的种子/难度元数据到 JSON 文件。"""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_records = sorted(records, key=lambda rec: rec.get("episode", -1))
    metadata = {
        "env_id": env_id,
        "record_count": len(sorted_records),
        "records": sorted_records,
    }
    try:
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
        print(f"Saved episode metadata to {metadata_path}")
    except Exception as exc:
        print(f"Warning: Failed to save episode metadata to {metadata_path}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HistoryBench 数据集生成器")
    parser.add_argument(
        "--env",
        "-e",
        type=str,
        nargs="+",
        default=None,
        help="要运行的环境 ID。提供一个或多个值；默认为所有内置 HistoryBench 环境。",
    )
    parser.add_argument(
        "--episodes",
        "-n",
        type=int,
        default=50,
        help="每个环境生成的 episode 数量 (默认: 50)",
    )
    parser.add_argument(
        "--save-video",
        dest="save_video",
        action="store_true",
        default=True,
        help="启用通过 HistoryBenchRecordWrapper 进行视频录制 (默认: 启用)。",
    )
    parser.add_argument(
        "--no-save-video",
        dest="save_video",
        action="store_false",
        help="禁用视频录制。",
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=25,
        help="运行多个环境时的并行 worker 数量。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 只生成前 50 条
    args.episodes = min(50, args.episodes)
    env_inputs = args.env or DEFAULT_ENVS
    env_ids: List[str] = []
    # 解析环境列表参数，支持逗号分隔
    for raw_env in env_inputs:
        env_ids.extend(env.strip() for env in raw_env.split(",") if env.strip())

    if not env_ids:
        env_ids = DEFAULT_ENVS.copy()

    num_workers = max(1, args.max_workers)
    episode_indices = list(range(args.episodes))

    for env_id in env_ids:
        source_metadata_records = _load_env_metadata_records(
            env_id=env_id,
            metadata_root=SOURCE_METADATA_ROOT,
        )

        # 为所有 episode 创建共享的临时文件夹
        temp_folder =  Path(f"/data/hongzefu/dataset_generate/temp_{env_id}_episodes")
        final_dataset_path =  Path(f"/data/hongzefu/dataset_generate/record_dataset_{env_id}.h5")
        #final_dataset_path =  Path(f"/data/hongzefu/dataset_generate/record_dataset_{env_id}.h5")

        print(f"\n{'='*80}")
        print(f"Environment: {env_id}")
        print(f"Episodes: {args.episodes}")
        print(f"Workers: {num_workers}")
        print(f"Temporary folder: {temp_folder}")
        print(f"Final dataset: {final_dataset_path}")
        print(f"{'='*80}\n")

        episode_records: List[Dict[str, Any]] = []

        if num_workers > 1:
            # 1. 拆分任务块
            episode_chunks = _split_episode_indices(args.episodes, num_workers)

            if len(episode_chunks) <= 1:
                # 单个 chunk，直接运行
                chunk = episode_chunks[0] if episode_chunks else []
                episode_records = run_env_dataset(
                    env_id,
                    chunk,
                    temp_folder,
                    args.save_video,
                    source_metadata_records,
                )
            else:
                worker_count = len(episode_chunks)
                print(
                    f"Running {env_id} with {worker_count} workers across {args.episodes} episodes..."
                )

                # 2. 并行执行
                # 每个 worker 写入同一个临时文件夹（但使用不同的文件/目录名）
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    future_to_chunk = {
                        executor.submit(
                            run_env_dataset,
                            env_id,
                            chunk,
                            temp_folder,  # 所有 worker 使用相同的临时文件夹路径
                            args.save_video,
                            source_metadata_records,
                        ): chunk
                        for chunk in episode_chunks
                    }

                    for future in as_completed(future_to_chunk):
                        chunk = future_to_chunk[future]
                        chunk_label = (chunk[0], chunk[-1]) if chunk else ("?", "?")
                        try:
                            records = future.result()
                            episode_records.extend(records)
                            print(f"✓ Completed episodes {chunk_label[0]}-{chunk_label[1]} for {env_id}")
                        except Exception as exc:
                            print(
                                f"✗ Environment {env_id} failed on episodes "
                                f"{chunk_label[0]}-{chunk_label[1]} with error: {exc}"
                            )

            # 3. 合并所有 episode 文件到最终数据集
            print(f"\nMerging all episodes into final dataset...")
            _merge_dataset_from_folder(
                env_id,
                temp_folder,
                final_dataset_path,
            )
        else:
            # 单 worker 模式
            episode_records = run_env_dataset(
                env_id,
                episode_indices,
                temp_folder,
                args.save_video,
                source_metadata_records,
            )

            # 合并 episode 到最终数据集
            print(f"\nMerging all episodes into final dataset...")
            _merge_dataset_from_folder(
                env_id,
                temp_folder,
                final_dataset_path,
            )

        # 4. 保存元数据
        metadata_path = final_dataset_path.with_name(
            f"{final_dataset_path.stem}_metadata.json"
        )
        _save_episode_metadata(episode_records, metadata_path, env_id)

        print(f"\n✓ Finished! Final dataset saved to: {final_dataset_path}\n")

    print("✓ All requested environments processed.")


if __name__ == "__main__":
    main()
