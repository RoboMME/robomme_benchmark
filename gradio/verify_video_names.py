#!/usr/bin/env python3
"""
验证和修复视频文件名，确保与 env_id 正确对应
"""
import json
import os
from pathlib import Path

def get_all_env_ids():
    """从 user_tasks_og.json 获取所有唯一的 env_id"""
    tasks_file = 'user_tasks_og.json'
    with open(tasks_file, 'r') as f:
        data = json.load(f)
    
    env_ids = set()
    for tasks in data.values():
        for task in tasks:
            env_ids.add(task['env_id'])
    return sorted(env_ids)

def verify_video_files(videos_dir='videos'):
    """验证视频文件名是否正确对应"""
    env_ids = get_all_env_ids()
    videos_path = Path(videos_dir)
    
    if not videos_path.exists():
        print(f"错误: 目录 {videos_dir} 不存在")
        return
    
    # 获取现有文件
    existing_files = {f.name.lower(): f for f in videos_path.glob('*.mp4')}
    
    print("=" * 80)
    print("视频文件名验证结果")
    print("=" * 80)
    print(f"{'Env ID':<25} {'期望文件名':<35} {'状态':<10}")
    print("-" * 80)
    
    correct_files = []
    missing_files = []
    incorrect_files = []
    
    for env_id in env_ids:
        expected_filename = env_id.lower() + '.mp4'
        expected_lower = expected_filename.lower()
        
        if expected_lower in existing_files:
            actual_file = existing_files[expected_lower]
            if actual_file.name == expected_filename:
                status = "✓ 正确"
                correct_files.append((env_id, expected_filename))
            else:
                status = f"⚠ 大小写不匹配: {actual_file.name}"
                incorrect_files.append((env_id, expected_filename, actual_file.name))
        else:
            status = "✗ 缺失"
            missing_files.append((env_id, expected_filename))
        
        print(f"{env_id:<25} {expected_filename:<35} {status:<10}")
    
    print("=" * 80)
    print(f"\n总结:")
    print(f"  ✓ 正确匹配: {len(correct_files)} 个")
    print(f"  ✗ 缺失文件: {len(missing_files)} 个")
    print(f"  ⚠ 需要修复: {len(incorrect_files)} 个")
    
    if incorrect_files:
        print(f"\n需要重命名的文件:")
        for env_id, expected, actual in incorrect_files:
            print(f"  {actual} -> {expected}")
    
    if missing_files:
        print(f"\n缺失的视频文件 (这些 env_id 没有对应的视频):")
        for env_id, expected in missing_files:
            print(f"  {env_id} -> {expected}")
    
    return correct_files, missing_files, incorrect_files

def fix_incorrect_names(videos_dir='videos', dry_run=True):
    """修复不正确的文件名"""
    env_ids = get_all_env_ids()
    videos_path = Path(videos_dir)
    
    if not videos_path.exists():
        print(f"错误: 目录 {videos_dir} 不存在")
        return
    
    existing_files = {f.name.lower(): f for f in videos_path.glob('*.mp4')}
    
    fixed = []
    for env_id in env_ids:
        expected_filename = env_id.lower() + '.mp4'
        expected_lower = expected_filename.lower()
        
        if expected_lower in existing_files:
            actual_file = existing_files[expected_lower]
            if actual_file.name != expected_filename:
                # 文件名大小写不匹配，需要重命名
                new_path = actual_file.parent / expected_filename
                if dry_run:
                    print(f"[DRY RUN] 将重命名: {actual_file.name} -> {expected_filename}")
                else:
                    try:
                        actual_file.rename(new_path)
                        print(f"✓ 已重命名: {actual_file.name} -> {expected_filename}")
                        fixed.append((actual_file.name, expected_filename))
                    except Exception as e:
                        print(f"✗ 重命名失败 {actual_file.name}: {e}")
                fixed.append((actual_file.name, expected_filename))
    
    if not fixed:
        print("没有需要修复的文件名")
    elif dry_run:
        print(f"\n[DRY RUN 模式] 共 {len(fixed)} 个文件需要重命名")
        print("运行时不加 --dry-run 参数以执行实际重命名")
    
    return fixed

if __name__ == '__main__':
    import sys
    
    if '--fix' in sys.argv:
        dry_run = '--dry-run' in sys.argv or '--fix' not in sys.argv
        fix_incorrect_names(dry_run=dry_run)
    else:
        verify_video_files()