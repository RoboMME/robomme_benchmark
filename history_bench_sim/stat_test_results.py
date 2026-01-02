#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计每个任务的结果：成功数、失败数、API错误数
"""
import os
import glob
from pathlib import Path
from collections import defaultdict

def count_episode_results(task_dir):
    """统计单个任务目录的结果"""
    task_path = Path(task_dir)
    
    # 统计成功的episode（success*.mp4文件）
    success_files = [f for f in task_path.glob('*.mp4') if f.name.startswith('success')]
    success_count = len(success_files)
    
    # 统计失败的episode（fail*.mp4文件）
    fail_files = [f for f in task_path.glob('*.mp4') if f.name.startswith('fail')]
    fail_count = len(fail_files)
    
    # 统计API错误的episode（api_error*.mp4文件）
    api_error_files = [f for f in task_path.glob('*.mp4') if f.name.startswith('api_error')]
    api_error_count = len(api_error_files)
    
    # 总数
    total_count = success_count + fail_count + api_error_count
    
    return {
        'success': success_count,
        'fail': fail_count,
        'api_error': api_error_count,
        'total': total_count
    }

def main():
    base_dir = Path('/home/hongzefu/oracle_planning_results/local')
    
    if not base_dir.exists():
        print(f"错误：目录不存在 {base_dir}")
        return
    
    # 获取所有任务目录
    task_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    task_dirs.sort()
    
    # 统计结果
    total_success = 0
    total_fail = 0
    total_api_error = 0
    total_all = 0
    
    # 收集所有任务的统计信息
    task_stats = []
    for task_dir in task_dirs:
        task_name = task_dir.name
        stats = count_episode_results(task_dir)
        task_stats.append((task_name, stats))
        
        total_success += stats['success']
        total_fail += stats['fail']
        total_api_error += stats['api_error']
        total_all += stats['total']
    
    # 分别统计 success
    print("\n" + "=" * 80)
    print("成功统计 (SUCCESS)")
    print("=" * 80)
    print(f"{'任务名称':<30} {'成功数':<10}")
    print("-" * 80)
    for task_name, stats in task_stats:
        if stats['success'] > 0:
            print(f"{task_name:<30} {stats['success']:<10}")
    print("-" * 80)
    print(f"{'总计':<30} {total_success:<10}")
    print("=" * 80)
    
    # 分别统计 fail
    print("\n" + "=" * 80)
    print("失败统计 (FAIL)")
    print("=" * 80)
    print(f"{'任务名称':<30} {'失败数':<10}")
    print("-" * 80)
    for task_name, stats in task_stats:
        if stats['fail'] > 0:
            print(f"{task_name:<30} {stats['fail']:<10}")
    print("-" * 80)
    print(f"{'总计':<30} {total_fail:<10}")
    print("=" * 80)
    
    # 分别统计 api_error
    print("\n" + "=" * 80)
    print("API错误统计 (API ERROR)")
    print("=" * 80)
    print(f"{'任务名称':<30} {'API错误数':<10}")
    print("-" * 80)
    for task_name, stats in task_stats:
        if stats['api_error'] > 0:
            print(f"{task_name:<30} {stats['api_error']:<10}")
    print("-" * 80)
    print(f"{'总计':<30} {total_api_error:<10}")
    print("=" * 80)

if __name__ == '__main__':
    main()