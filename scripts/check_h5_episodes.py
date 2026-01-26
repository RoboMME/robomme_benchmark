#!/usr/bin/env python3
"""
检查 solve_3.5_parallel_multi_loop_v4.py 生成的 h5 文件是否包含所有期望的 episode。

用法示例:
    # 检查单个文件
    python check_h5_episodes.py --h5-file /path/to/record_dataset_RouteStick.h5 --expected-episodes 50
    
    # 检查目录下所有 h5 文件
    python check_h5_episodes.py --h5-file /path/to/dataset_dir --expected-episodes 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Set

import h5py


def extract_episode_number(episode_key: str) -> Optional[int]:
    """
    从 episode 键名中提取 episode 编号。
    
    Args:
        episode_key: episode 键名，例如 "episode_0", "episode_123"
    
    Returns:
        episode 编号，如果无法解析则返回 None
    """
    if not episode_key.startswith("episode_"):
        return None
    try:
        return int(episode_key.split("_")[-1])
    except (ValueError, IndexError):
        return None


def read_episodes_from_h5(h5_file_path: Path, env_id: Optional[str] = None) -> tuple[Set[int], str, dict[int, int]]:
    """
    从 h5 文件中读取所有存在的 episode 编号及其 timestep 长度。
    
    Args:
        h5_file_path: h5 文件路径
        env_id: 环境 ID（可选，如果未提供则自动检测）
    
    Returns:
        (episode 编号集合, 环境名称, episode_timestep_lengths字典)
    
    Raises:
        FileNotFoundError: 如果文件不存在
        KeyError: 如果找不到环境组
    """
    if not h5_file_path.exists():
        raise FileNotFoundError(f"H5 文件不存在: {h5_file_path}")
    
    episodes: Set[int] = set()
    episode_timestep_lengths: dict[int, int] = {}
    env_name: Optional[str] = None
    env_group: Optional[h5py.Group] = None
    
    with h5py.File(h5_file_path, "r") as handle:
        # 如果指定了 env_id，尝试多种可能的组名格式
        if env_id:
            # 根据 solve_3.5_parallel_multi_loop_v4.py，组名通常是环境 ID 本身
            possible_names = [env_id, f"env_{env_id}"]
            
            for name in possible_names:
                if name in handle:
                    obj = handle[name]
                    if isinstance(obj, h5py.Group):
                        env_group = obj
                        env_name = name
                        break
            
            if env_group is None:
                raise KeyError(f"在 h5 文件中找不到环境组: 尝试了 {possible_names}")
        else:
            # 自动检测：优先查找 env_* 格式，然后查找其他组
            # 首先尝试查找 env_* 格式的组
            for key in handle.keys():
                obj = handle[key]
                if key.startswith("env_") and isinstance(obj, h5py.Group):
                    env_group = obj
                    env_name = key
                    break
            
            # 如果没找到 env_* 格式，查找第一个非空组（可能是直接的环境名）
            if env_group is None:
                for key in handle.keys():
                    obj = handle[key]
                    if isinstance(obj, h5py.Group):
                        # 检查是否有子键（跳过空组）
                        try:
                            if len(obj.keys()) > 0 and key != "setup":
                                env_group = obj
                                env_name = key
                                break
                        except (AttributeError, TypeError):
                            # 如果 obj.keys() 不可用，跳过
                            continue
            
            if env_group is None or env_name is None:
                raise KeyError("在 h5 文件中找不到环境组")
        
        # 确保 env_group 和 env_name 不为 None（类型检查）
        assert env_group is not None and env_name is not None, "环境组和环境名应该已设置"
        
        # 遍历所有 episode
        # type: ignore[union-attr] - env_group 已通过 assert 确保不为 None
        for episode_key in env_group.keys():
            # 跳过 setup 组
            if episode_key == "setup":
                continue
            
            episode_num = extract_episode_number(episode_key)
            if episode_num is not None:
                episodes.add(episode_num)
                
                # 计算该 episode 的 timestep 数量
                episode_group = env_group[episode_key]
                if isinstance(episode_group, h5py.Group):
                    # 统计以 record_timestep_ 开头的键
                    timestep_count = sum(
                        1 for key in episode_group.keys() 
                        if key.startswith("record_timestep_")
                    )
                    episode_timestep_lengths[episode_num] = timestep_count
    
    # 确保 env_name 不为 None
    if env_name is None:
        raise RuntimeError("环境名未设置，这不应该发生")
    
    return episodes, env_name, episode_timestep_lengths


def get_expected_episode_count(expected_episodes_arg: Optional[int]) -> Optional[int]:
    """
    获取期望的 episode 数量。
    
    Args:
        expected_episodes_arg: 命令行参数指定的期望数量
    
    Returns:
        期望的 episode 数量，如果未指定则返回 None
    """
    return expected_episodes_arg


def check_episodes(
    h5_file_path: Path,
    env_id: Optional[str] = None,
    expected_episodes: Optional[int] = None,
) -> dict:
    """
    检查 h5 文件中的 episode 完整性。
    
    Args:
        h5_file_path: h5 文件路径
        env_id: 环境 ID（可选）
        expected_episodes: 期望的 episode 数量（必需）
    
    Returns:
        包含检查结果的字典
    """
    result = {
        "h5_file": str(h5_file_path),
        "env_name": None,
        "existing_episodes": set(),
        "episode_timestep_lengths": {},
        "expected_count": None,
        "missing_episodes": [],
        "status": "unknown",
        "error": None,
    }
    
    try:
        # 读取 h5 文件中的 episode
        existing_episodes, env_name, episode_timestep_lengths = read_episodes_from_h5(h5_file_path, env_id)
        result["existing_episodes"] = existing_episodes
        result["env_name"] = env_name
        result["episode_timestep_lengths"] = episode_timestep_lengths
        
        # 如果 env_id 未指定，从检测到的环境名提取
        if not env_id and env_name:
            # 移除 env_ 前缀（如果有）
            if env_name.startswith("env_"):
                env_id = env_name.replace("env_", "")
            else:
                env_id = env_name
        
        # 获取期望的 episode 数量
        expected_count = get_expected_episode_count(expected_episodes)
        result["expected_count"] = expected_count
        
        # 比较和计算缺失的 episode
        if expected_count is not None:
            all_expected = set(range(expected_count))
            missing = sorted(all_expected - existing_episodes)
            result["missing_episodes"] = missing
            
            if len(missing) == 0:
                result["status"] = "complete"
            else:
                result["status"] = "incomplete"
        else:
            result["status"] = "unknown_expected"
    
    except FileNotFoundError as e:
        result["error"] = str(e)
        result["status"] = "file_not_found"
    except KeyError as e:
        result["error"] = str(e)
        result["status"] = "env_not_found"
    except Exception as e:
        result["error"] = str(e)
        result["status"] = "error"
    
    return result


def print_report(result: dict, verbose: bool = False) -> None:
    """
    打印检查报告。
    
    Args:
        result: check_episodes 返回的结果字典
        verbose: 是否显示详细信息
    """
    print("=" * 80)
    print(f"检查 h5 文件: {result['h5_file']}")
    print("=" * 80)
    
    if result["error"]:
        print(f"❌ 错误: {result['error']}")
        return
    
    if result["env_name"]:
        print(f"环境: {result['env_name']}")
    
    existing_count = len(result["existing_episodes"])
    print(f"实际存在的 episode 数量: {existing_count}")
    
    # 打印每个 episode 的 timestep 长度
    episode_timestep_lengths = result.get("episode_timestep_lengths", {})
    if episode_timestep_lengths:
        print(f"\n每个 episode 的 timestep 长度:")
        sorted_episodes = sorted(episode_timestep_lengths.keys())
        # 按行显示，每行显示多个
        for i in range(0, len(sorted_episodes), 10):
            batch = sorted_episodes[i : i + 10]
            episode_info = [f"ep{ep}:{episode_timestep_lengths[ep]}" for ep in batch]
            print(f"  {', '.join(episode_info)}")
        
        # 统计信息
        if sorted_episodes:
            timestep_lengths = [episode_timestep_lengths[ep] for ep in sorted_episodes]
            min_len = min(timestep_lengths)
            max_len = max(timestep_lengths)
            avg_len = sum(timestep_lengths) / len(timestep_lengths)
            print(f"\n  Timestep 长度统计: 最小={min_len}, 最大={max_len}, 平均={avg_len:.1f}")
    
    if result["expected_count"] is not None:
        expected_count = result["expected_count"]
        print(f"期望的 episode 数量: {expected_count}")
        
        missing_count = len(result["missing_episodes"])
        if missing_count == 0:
            print("✅ 所有 episode 都存在！")
        else:
            print(f"❌ 缺失 {missing_count} 个 episode")
            
            if verbose or missing_count <= 20:
                # 显示所有缺失的 episode
                print(f"\n缺失的 episode 列表:")
                missing_list = result["missing_episodes"]
                # 按行显示，每行显示多个
                for i in range(0, len(missing_list), 10):
                    batch = missing_list[i : i + 10]
                    print(f"  {', '.join(map(str, batch))}")
            else:
                # 只显示前 20 个
                print(f"\n缺失的 episode 列表（前 20 个）:")
                for i in range(0, min(20, len(result["missing_episodes"])), 10):
                    batch = result["missing_episodes"][i : i + 10]
                    print(f"  {', '.join(map(str, batch))}")
                print(f"  ... 还有 {missing_count - 20} 个缺失的 episode")
    else:
        print("⚠️  未指定期望的 episode 数量")
        print("   提示: 使用 --expected-episodes 参数指定期望数量")
        if verbose:
            existing_list = sorted(result["existing_episodes"])
            print(f"\n实际存在的 episode: {existing_list[:20]}")
            if len(existing_list) > 20:
                print(f"  ... 还有 {len(existing_list) - 20} 个")
    
    print("=" * 80)


def main() -> int:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="检查 h5 文件是否包含所有期望的 episode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法（自动检测环境）
  python check_h5_episodes.py --h5-file /path/to/record_dataset_RouteStick.h5 --expected-episodes 50
  
  # 指定环境 ID
  python check_h5_episodes.py --h5-file /path/to/record_dataset_RouteStick.h5 --env-id RouteStick --expected-episodes 50
        """,
    )
    
    parser.add_argument(
        "--h5-file",
        type=Path,
        default='/nfs/turbo/coe-chaijy-unreplicated/hongzefu/dataset_generate',
        help="要检查的 h5 文件路径或包含 h5 文件的目录",
    )
    
    parser.add_argument(
        "--env-id",
        type=str,
        default=None,
        help="环境 ID（如果未指定，将自动检测）",
    )
    
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=100,
        help="期望的 episode 数量（必需）",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="显示详细信息",
    )
    
    args = parser.parse_args()
    
    # 确定要检查的文件列表
    h5_files: list[Path] = []
    if args.h5_file.is_file() and args.h5_file.suffix in (".h5", ".hdf5"):
        # 单个文件
        h5_files = [args.h5_file]
    elif args.h5_file.is_dir():
        # 目录，查找所有 .h5 文件
        h5_files = sorted(args.h5_file.glob("*.h5")) + sorted(args.h5_file.glob("*.hdf5"))
        if not h5_files:
            print(f"错误: 在目录 {args.h5_file} 中未找到任何 .h5 或 .hdf5 文件", file=sys.stderr)
            return 1
        print(f"找到 {len(h5_files)} 个 h5 文件\n")
    else:
        print(f"错误: {args.h5_file} 不是有效的文件或目录", file=sys.stderr)
        return 1
    
    # 对每个文件执行检查
    all_results: list[dict] = []
    total_incomplete = 0
    total_errors = 0
    
    for h5_file in h5_files:
        # 如果未指定 env_id，尝试从文件名推断
        env_id = args.env_id
        if env_id is None:
            # 从文件名提取环境 ID（例如 record_dataset_RouteStick.h5 -> RouteStick）
            stem = h5_file.stem
            if stem.startswith("record_dataset_"):
                env_id = stem.replace("record_dataset_", "")
        
        # 执行检查
        result = check_episodes(
            h5_file_path=h5_file,
            env_id=env_id,
            expected_episodes=args.expected_episodes,
        )
        all_results.append(result)
        
        # 打印报告
        print_report(result, verbose=args.verbose)
        
        # 统计
        if result["status"] == "incomplete":
            total_incomplete += 1
        elif result["status"] in ("file_not_found", "env_not_found", "error"):
            total_errors += 1
        
        # 文件之间添加分隔
        if len(h5_files) > 1:
            print()
    
    # 如果有多个文件，打印汇总
    if len(h5_files) > 1:
        print("=" * 80)
        print("汇总:")
        print(f"  总文件数: {len(h5_files)}")
        complete_count = len([r for r in all_results if r["status"] == "complete"])
        print(f"  完整文件: {complete_count}")
        print(f"  不完整文件: {total_incomplete}")
        print(f"  错误文件: {total_errors}")
        print("=" * 80)
    
    # 返回适当的退出码
    if total_errors > 0:
        return 2
    elif total_incomplete > 0:
        return 1
    elif all(r["status"] == "complete" for r in all_results):
        return 0
    else:
        return 3


if __name__ == "__main__":
    sys.exit(main())
