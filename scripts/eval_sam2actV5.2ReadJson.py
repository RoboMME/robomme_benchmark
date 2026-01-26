"""
评估数据集关键点V2版本脚本

该脚本用于从pred_actions.json文件读取预测动作并评估HistoryBench数据集中的任务。
主要功能包括：
1. 从pred_actions.json文件读取预测动作
2. 直接创建仿真环境并初始化（不使用EpisodeConfigResolver）
3. 直接控制任务的seed和difficulty
4. 使用运动规划器执行预测动作
5. 评估任务完成情况

使用场景：
- 评估模型在特定任务上的表现
- 批量测试多个episode
- 记录评估结果用于分析
"""

import os
import sys
import json  # 用于JSON文件读写
import numpy as np  # 数值计算库
import sapien  # SAPIEN物理仿真引擎
import argparse  # 命令行参数解析
from pathlib import Path  # 路径操作
from collections import defaultdict  # 用于分组数据

# 将父目录添加到Python路径中，以便导入项目模块
# 这样可以导入historybench等自定义模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym  # Gymnasium强化学习环境库
from gymnasium.utils.save_video import save_video  # 视频保存工具（本脚本中未使用）

# 导入HistoryBench相关模块
from historybench.HistoryBench_env import *
from historybench.HistoryBench_env.util import task_goal  # 任务目标语言描述工具
from historybench.env_record_wrapper import DemonstrationWrapper  # 演示包装器

# 导入ManiSkill运动规划工具函数
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,  # 通过OBB（定向包围盒）计算抓取信息
    get_actor_obb,  # 获取actor的OBB
)

import torch  # PyTorch深度学习框架

# 导入自定义的运动规划器，包含失败处理机制
from planner_fail_safe import (
    FailAwarePandaArmMotionPlanningSolver,  # Panda机械臂运动规划求解器（带失败感知）
    FailAwarePandaStickMotionPlanningSolver,  # Panda机械臂+棍子运动规划求解器（带失败感知）
    ScrewPlanFailure,  # 螺旋运动规划失败异常
)

# 输出根目录：脚本所在目录的父目录
OUTPUT_ROOT = Path(__file__).resolve().parents[1]


def read_pred_actions(pred_actions_path):
    """
    从pred_actions.json文件读取所有预测动作
    
    该函数读取包含预测动作的JSON文件，并按episode_idx和keypoint_pair_idx分组。
    
    Args:
        pred_actions_path: pred_actions.json文件的路径（字符串或Path对象）
            文件格式示例：
            [
                {
                    "task_name": "BinFill",
                    "episode_idx": 0,
                    "keypoint_pair_idx": 0,
                    "pred_action": [x, y, z, qx, qy, qz, qw, grip, coll],
                    ...
                },
                ...
            ]
        
    Returns:
        dict: 按(task_name, episode_idx)分组的字典
            格式: {(task_name, episode_idx): [按keypoint_pair_idx排序的动作列表]}
            如果文件不存在或读取失败，返回空字典
    """
    # 检查文件是否存在
    if not Path(pred_actions_path).exists():
        print(f"Pred actions file not found: {pred_actions_path}")
        return {}
    
    # 读取JSON文件
    with open(pred_actions_path, 'r') as f:
        pred_actions = json.load(f)
    
    # 按(task_name, episode_idx)分组，并按keypoint_pair_idx排序
    grouped = defaultdict(list)
    for action_record in pred_actions:
        task_name = action_record.get('task_name')
        episode_idx = action_record.get('episode_idx')
        keypoint_pair_idx = action_record.get('keypoint_pair_idx')
        pred_action = action_record.get('pred_action')
        gt_action = action_record.get('gt_action')  # 读取gt_action
        
        if task_name is not None and episode_idx is not None and pred_action is not None:
            grouped[(task_name, episode_idx)].append({
                'keypoint_pair_idx': keypoint_pair_idx,
                'pred_action': pred_action,
                'gt_action': gt_action,  # 保存gt_action
                'record': action_record  # 保留完整记录以便后续使用
            })
    
    # 对每个episode的动作按keypoint_pair_idx排序
    for key in grouped:
        grouped[key].sort(key=lambda x: x['keypoint_pair_idx'] if x['keypoint_pair_idx'] is not None else -1)
    
    return grouped


def main():
    """
    主函数：从pred_actions.json读取预测动作并运行仿真评估
    
    该函数是脚本的入口点，主要执行以下步骤：
    1. 解析命令行参数（seed、difficulty等）
    2. 读取pred_actions.json文件，获取所有预测动作
    3. 按episode分组处理
    4. 对每个episode：
       a. 直接创建并初始化环境（不使用EpisodeConfigResolver）
       b. 初始化运动规划器
       c. 执行评估循环（使用pred_action执行动作）
       d. 评估任务完成情况
    5. 输出评估结果
    
    工作流程：
    - 从pred_actions.json读取每个episode的预测动作
    - 按keypoint_pair_idx顺序执行动作
    - 如果任务成功完成，提前结束episode
    """
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description='Run evaluation using pred_actions from JSON file')
    # 随机种子，默认42
    parser.add_argument('--seed', type=int, default=40000, help='Random seed for environment')
    # 难度等级，默认easy
    parser.add_argument('--difficulty', type=str, default='easy', choices=['easy', 'medium', 'hard'], help='Difficulty level')
    # pred_actions.json文件路径
    parser.add_argument('--pred_actions_path', type=str, 
                       default='/data/hongzefu/historybench-v5.7.5-sam2act7-full-dataset/scripts/pred_actions.json',
                       help='Path to pred_actions.json file')
    args = parser.parse_args()
    
    # 提取参数值
    seed = args.seed  # 随机种子
    difficulty = args.difficulty  # 难度等级
    pred_actions_path = Path(args.pred_actions_path)  # pred_actions.json文件路径
    
    # ========== 读取预测动作 ==========
    grouped_actions = read_pred_actions(pred_actions_path)
    if not grouped_actions:
        print(f"No pred actions found in {pred_actions_path}")
        return
    
    print(f"Found {len(grouped_actions)} episodes in pred_actions.json")
    
    # ========== 遍历所有episode ==========
    for (task_name, episode_idx), action_list in grouped_actions.items():
        env_id = task_name  # 环境ID就是任务名称
        
        print(f"--- Running simulation for episode:{episode_idx}, env: {env_id}, seed: {seed}, difficulty: {difficulty} ---")
        print(f"Found {len(action_list)} actions for this episode")
        
        # ========== 直接创建环境并重置 ==========
        # 不使用EpisodeConfigResolver，直接使用gym.make创建环境
        env_kwargs = dict(
            obs_mode="rgb+depth+segmentation",  # 观测模式：RGB + 深度 + 分割
            control_mode="pd_joint_pos",        # 控制模式：位置控制
            render_mode="human",            # 渲染模式
            reward_mode="dense",                # 奖励模式
            HistoryBench_seed=seed,             # 随机种子
            max_episode_steps=200,              # 最大步数
            HistoryBench_difficulty=difficulty, # 难度设置
        )
        
        env = gym.make(env_id, **env_kwargs)
        # 使用DemonstrationWrapper包装环境
        # max_steps_without_demonstration: 演示轨迹的最大步数
        # gui_render: 是否使用GUI渲染（当前使用rgb_array模式，设为False）
        env = DemonstrationWrapper(
            env,
            max_steps_without_demonstration=200,  # 与max_episode_steps保持一致
            gui_render=True,  # 需要GUI渲染
            save_video=True  # 保存视频
        )
        # 重置环境，获取初始观测和信息
        # DemonstrationWrapper会在reset时自动生成演示轨迹
        obs, info = env.reset(seed=seed)
        
        # ========== 初始化运动规划器 ==========
        # 根据任务类型选择不同的运动规划器
        # PatternLock和RouteStick任务需要使用带棍子的规划器（因为涉及棍子操作）
        if env_id in ("PatternLock", "RouteStick"):
            # 使用Panda机械臂+棍子的运动规划求解器
            planner = FailAwarePandaStickMotionPlanningSolver(
                env,  # 环境对象
                debug=False,  # 是否开启调试模式
                vis=True,  # 是否可视化规划过程
                base_pose=env.unwrapped.agent.robot.pose,  # 机器人基座位姿
                visualize_target_grasp_pose=False,  # 是否可视化目标抓取位姿（棍子任务不需要）
                print_env_info=False,  # 是否打印环境信息
                joint_vel_limits=0.3,  # 关节速度限制（rad/s）
            )
        else:
            # 使用标准Panda机械臂运动规划求解器（适用于大多数任务）
            planner = FailAwarePandaArmMotionPlanningSolver(
                env,
                debug=False,
                vis=True,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=True,  # 可视化目标抓取位姿（有助于调试）
                print_env_info=False,
            )
            
        # ========== 获取语言目标描述 ==========
        # 从任务配置中获取自然语言描述的任务目标
        # 例如："将红色方块放入蓝色盒子中"
        lang_goal = task_goal.get_language_goal(env, env_id)
        print(f"Language Goal: {lang_goal}")
        
        # ========== 执行循环 ==========
        # 遍历该episode的所有预测动作
        for step, action_data in enumerate(action_list):
            keypoint_pair_idx = action_data['keypoint_pair_idx']
            pred_action = action_data['gt_action']
            #pred_action = action_data['pred_action']#选择预测动作
            
            print(f"Step {step}/{len(action_list)} (keypoint_pair_idx: {keypoint_pair_idx})")
            
            # ========== 执行动作 ==========
            try:
                # 动作格式：[x, y, z, qx, qy, qz, qw, grip, coll]
                # 前7个元素：末端执行器目标位姿（位置x,y,z + 四元数qx,qy,qz,qw）
                # 第8个元素：夹爪动作（>0.5表示打开，<=0.5表示闭合）
                # 第9个元素：碰撞相关（本脚本中未使用）
                pose_list = pred_action[:7]  # 提取位姿部分（位置+四元数）
                gripper_action = pred_action[7]  # 提取夹爪动作
                
                # 构建SAPIEN Pose对象
                # p: 位置向量 [x, y, z]
                # 模型返回的四元数通常是 [x, y, z, w]
                pos = pose_list[:3]
                quat_xyzw = pose_list[3:7] # [qx, qy, qz, qw]
                
                # SAPIEN 需要 [w, x, y, z]
                quat_wxyz = [
                    quat_xyzw[3], # w
                    quat_xyzw[0], # x
                    quat_xyzw[1], # y
                    quat_xyzw[2]  # z
                ]
                
                pose = sapien.Pose(p=pos, q=quat_wxyz)
                print(f"pose: {pos}, quat(wxyz): {quat_wxyz} gripper_action: {gripper_action}")
                                    
                # ========== 执行运动规划并移动到目标位姿 ==========
                try:
                    # 使用螺旋运动（screw motion）规划并执行移动到目标位姿
                    # 螺旋运动是一种平滑的轨迹规划方法，适合机械臂运动
                    planner.move_to_pose_with_RRTStar(pose)
                    
                    # 如果动作指示需要闭合夹爪，在移动完成后闭合
                    if gripper_action <= 0.5:
                        try:
                            planner.close_gripper()
                        except Exception as e:
                            print(f"  Stick action ")
                    else:
                        try:
                            planner.open_gripper()
                        except Exception as e:
                            print(f"  Stick action ")
                        planner.open_gripper()
                        
                except ScrewPlanFailure as exc:
                    # 捕获螺旋运动规划失败异常
                    # 这通常发生在目标位姿不可达或存在碰撞时
                    print(f"    Screw plan failure: {exc}")
                    # 注意：这里可以选择break（退出）或continue（继续下一步）
                    # 当前实现是继续执行，但可以考虑改为break以提高效率
                except Exception as exc:
                    # 捕获其他执行错误
                    print(f"    Error executing action: {exc}")
                    break  # 发生错误，退出循环
                    
                # ========== 更新观测数据 ==========
                # 执行动作后，获取新的环境观测
                obs = env.unwrapped.get_obs()
                
                # ========== 检查任务是否成功完成 ==========
                # 评估当前状态是否满足任务目标
                # solve_complete_eval=True表示进行完整的成功评估
                evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
                # 从评估结果中提取成功标志
                # 如果evaluation中没有"success"键，默认返回False
                success = evaluation.get("success", torch.tensor([False])).item()
                
                # 如果任务成功完成，提前结束episode
                if success:
                    print(f"Episode {episode_idx} Success!")
                    break  # 任务完成，退出循环
                    
            except Exception as e:
                # 捕获执行循环中的任何其他异常
                print(f"Error in step loop: {e}")
                import traceback
                traceback.print_exc()  # 打印完整的错误堆栈信息，便于调试
                break  # 发生异常，退出循环
        
        # ========== 最终评估 ==========
        # 无论episode是否提前结束，都进行最终评估
        # 这可以确保即使任务未成功完成，也能获得完整的评估结果
        evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
        print(f"Final evaluation for episode {episode_idx}: {evaluation}")
        
        # ========== 清理资源 ==========
        # 关闭环境，释放资源（包括图形界面、物理引擎等）
        env.close()
        print(f"--- Finished Running simulation for episode:{episode_idx}, env: {env_id} ---")


# ========== 脚本入口 ==========
# 当脚本被直接运行时（而非作为模块导入），执行main函数
if __name__ == "__main__":
    main()
