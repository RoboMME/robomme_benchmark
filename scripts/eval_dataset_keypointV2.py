"""
评估数据集关键点V2版本脚本

该脚本用于通过SAM2ACT Agent API评估HistoryBench数据集中的任务。
主要功能包括：
1. 从metadata文件读取episode配置
2. 创建仿真环境并初始化
3. 通过HTTP API调用SAM2ACT模型获取动作
4. 使用运动规划器执行动作
5. 评估任务完成情况

使用场景：
- 评估模型在特定任务上的表现
- 批量测试多个episode
- 记录评估结果用于分析
"""

import os
import sys
import json  # 用于JSON文件读写
import h5py  # 用于HDF5文件操作（虽然本脚本中未直接使用）
import numpy as np  # 数值计算库
import sapien  # SAPIEN物理仿真引擎
import requests  # HTTP请求库，用于调用API
import argparse  # 命令行参数解析
from pathlib import Path  # 路径操作

# 将父目录添加到Python路径中，以便导入项目模块
# 这样可以导入historybench等自定义模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym  # Gymnasium强化学习环境库
from gymnasium.utils.save_video import save_video  # 视频保存工具（本脚本中未使用）

# 导入HistoryBench相关模块
from historybench.env_record_wrapper import HistoryBenchRecordWrapper, EpisodeConfigResolver
from historybench.HistoryBench_env import *
from historybench.HistoryBench_env.util import task_goal  # 任务目标语言描述工具

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


class NumpyEncoder(json.JSONEncoder):
    """
    自定义JSON编码器，用于处理NumPy数据类型
    
    由于标准JSON编码器不支持NumPy的数据类型（如np.integer、np.floating、np.ndarray），
    需要自定义编码器将这些类型转换为Python原生类型，以便进行JSON序列化。
    
    转换规则：
    - np.integer -> int: NumPy整数类型转换为Python整数
    - np.floating -> float: NumPy浮点数类型转换为Python浮点数
    - np.ndarray -> list: NumPy数组转换为Python列表
    """
    def default(self, o):
        # 处理NumPy整数类型
        if isinstance(o, np.integer):
            return int(o)
        # 处理NumPy浮点数类型
        elif isinstance(o, np.floating):
            return float(o)
        # 处理NumPy数组类型
        elif isinstance(o, np.ndarray):
            return o.tolist()
        # 其他类型使用默认的JSON编码器处理
        return json.JSONEncoder.default(self, o)


def get_model_input(env, obs, timestep, lang_goal):
    """
    从环境观测数据构建模型API所需的输入字典
    
    该函数将HistoryBench环境的观测数据转换为SAM2ACT模型API期望的格式。
    主要转换包括：
    1. 相机数据（RGB、深度、点云）
    2. 机器人状态（关节位置、夹爪状态、末端执行器位姿）
    3. 相机参数（内参、外参）
    
    Args:
        env: 环境实例，用于访问机器人对象
        obs: 当前时刻的环境观测数据字典，包含：
            - sensor_data: 传感器数据（相机RGB、深度、点云等）
            - sensor_param: 传感器参数（相机内参、外参等）
            - extra: 额外信息（如TCP位姿）
            - agent: 智能体信息（关节位置等）
        timestep: 当前时间步（本函数中未使用，但保留以保持接口一致性）
        lang_goal: 语言目标描述字符串（本函数中未使用，但保留以保持接口一致性）
        
    Returns:
        dict: 包含以下键的字典：
            - front_rgb: 前置相机RGB图像（numpy数组）
            - front_depth: 前置相机深度图（numpy数组）
            - front_point_cloud: 前置相机点云数据（numpy数组，格式为xyzw）
            - wrist_rgb: 腕部相机RGB图像（numpy数组）
            - wrist_depth: 腕部相机深度图（numpy数组）
            - wrist_point_cloud: 腕部相机点云数据（numpy数组，格式为xyzw）
            - joint_positions: 机械臂关节位置（7维数组，对应Panda机械臂的7个关节）
            - gripper_joint_positions: 夹爪关节位置（2维数组）
            - gripper_open: 夹爪开合状态（0.0或1.0，1.0表示打开）
            - gripper_pose: 末端执行器位姿（7维数组：[x, y, z, qx, qy, qz, qw]）
            - misc: 其他信息字典，包含相机内参和外参
    """
    
    # 从观测数据中提取传感器数据、传感器参数和额外信息
    sensor_data = obs['sensor_data']  # 传感器数据：相机RGB、深度、点云等
    sensor_param = obs['sensor_param']  # 传感器参数：相机内参、外参等
    extra = obs['extra']  # 额外信息：TCP位姿等
    
    # 初始化输出字典
    obs_obj = {}
    
    # ========== 相机数据映射 ==========
    # 将环境中的相机名称映射到模型API期望的名称
    # Base camera（基座相机）-> front（前置相机）
    if 'base_camera' in sensor_data:
        # 将PyTorch张量转换为numpy数组，并取第一个batch的元素
        obs_obj['front_rgb'] = sensor_data['base_camera']['rgb'].cpu().numpy()[0]
        obs_obj['front_depth'] = sensor_data['base_camera']['depth'].cpu().numpy()[0]

    
    # Hand camera（手部相机）-> wrist（腕部相机）
    if 'hand_camera' in sensor_data:
        obs_obj['wrist_rgb'] = sensor_data['hand_camera']['rgb'].cpu().numpy()[0]
        obs_obj['wrist_depth'] = sensor_data['hand_camera']['depth'].cpu().numpy()[0]

    
    # ========== 机器人状态提取 ==========
    # 获取环境中的智能体和机器人对象
    agent = env.unwrapped.agent
    robot = agent.robot
    
    # 提取关节位置
    # 注意：机器人可能有更多关节，但通常只需要机械臂关节
    # 对于Panda机器人，通常是前7个关节为机械臂，后2个为夹爪
    # obs['agent']['qpos']包含所有关节位置（包括机械臂和夹爪）
    qpos = obs['agent']['qpos'].cpu().numpy()[0]  # 转换为numpy数组并取第一个batch
    
    # 提取机械臂关节位置（前7个关节，对应Panda的7自由度机械臂）
    # 假设Panda机器人结构：前7个关节为机械臂，后2个为夹爪
    # 通常模型期望的'joint_positions'是7维的
    if len(qpos) >= 7:
        obs_obj['joint_positions'] = qpos[:7]
        
    # 提取夹爪关节位置（第8和第9个关节）
    if len(qpos) >= 9:
        obs_obj['gripper_joint_positions'] = qpos[7:9]
    
    # 计算夹爪开合状态（0.0表示闭合，1.0表示打开）
    # 基于夹爪宽度或关节位置计算
    # 最大宽度通常约为0.08m（每个手指0.04m）
    # 如果两个夹爪关节位置之和大于阈值（0.002m），则认为夹爪是打开的
    gripper_width = np.sum(obs_obj.get('gripper_joint_positions', [0, 0]))
    obs_obj['gripper_open'] = 1.0 if gripper_width > 0.002 else 0.0
    

    # 将位置和四元数拼接成7维向量：[x, y, z, qx, qy, qz, qw]
    # 先转换为numpy数组并展平为一维，确保维度匹配
    position = agent.tcp.pose.p.cpu().numpy().flatten()
    quaternion = agent.tcp.pose.q.cpu().numpy().flatten()
    obs_obj['gripper_pose'] = np.concatenate([position, quaternion])
    
    # ========== 其他数据（相机内参、外参） ==========
    misc = {}
    
    # 前置相机参数
    if 'base_camera' in sensor_param:
        # 相机内参矩阵（3x3，使用OpenCV格式）
        misc['front_camera_intrinsics'] = sensor_param['base_camera']['intrinsic_cv'].cpu().numpy()[0]
        # 相机外参矩阵（4x4，从相机坐标系到世界坐标系的变换矩阵）
        misc['front_camera_extrinsics'] = sensor_param['base_camera']['extrinsic_cv'].cpu().numpy()[0]
        
    # 腕部相机参数
    if 'hand_camera' in sensor_param:
        misc['wrist_camera_intrinsics'] = sensor_param['hand_camera']['intrinsic_cv'].cpu().numpy()[0]
        misc['wrist_camera_extrinsics'] = sensor_param['hand_camera']['extrinsic_cv'].cpu().numpy()[0]
        
    obs_obj['misc'] = misc
    
    return obs_obj


def read_metadata(metadata_path):
    """
    从metadata JSON文件读取所有episode配置信息
    
    该函数读取HistoryBench数据集的metadata文件，该文件包含了所有需要评估的episode的配置信息。
    每个episode记录通常包含任务类型、episode编号、随机种子、难度等级等信息。
    
    Args:
        metadata_path: metadata JSON文件的路径（字符串或Path对象）
            文件格式示例：
            {
                "records": [
                    {
                        "task": "BinFill",
                        "episode": 0,
                        "seed": 42,
                        "difficulty": "easy"
                    },
                    ...
                ]
            }
        
    Returns:
        list: 包含所有episode记录的列表
            每个记录是一个字典，通常包含以下键：
            - task: 任务类型（如"BinFill"）
            - episode: episode编号（整数）
            - seed: 随机种子（整数，可选）
            - difficulty: 难度等级（字符串，如"easy"、"medium"、"hard"，可选）
            如果文件不存在或读取失败，返回空列表
    """
    # 检查文件是否存在
    if not Path(metadata_path).exists():
        print(f"Metadata file not found: {metadata_path}")
        return []
    
    # 读取JSON文件
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        # 从metadata中提取records字段，如果不存在则返回空列表
        episode_records = metadata.get('records', [])
        return episode_records


def main():
    """
    主函数：使用SAM2ACT Agent API运行仿真评估
    
    该函数是脚本的入口点，主要执行以下步骤：
    1. 解析命令行参数（API地址、最大步数等）
    2. 遍历指定的环境任务列表
    3. 读取每个任务的metadata文件，获取所有episode配置
    4. 对每个episode：
       a. 创建并初始化环境
       b. 初始化运动规划器
       c. 执行评估循环（调用API获取动作并执行）
       d. 评估任务完成情况
    5. 输出评估结果
    
    工作流程：
    - 每个episode最多执行max_steps步
    - 每一步：获取观测 -> 调用API获取动作 -> 执行动作 -> 检查是否成功
    - 如果任务成功完成，提前结束episode
    """
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description='Run evaluation using SAM2ACT Agent API')
    # API服务器地址，默认本地8000端口
    parser.add_argument('--api_url', type=str, default='http://141.212.48.176:8002', help='API URL')
    # 每个episode的最大执行步数，默认25步
    parser.add_argument('--max_steps', type=int, default=100, help='Max steps per episode')
    args = parser.parse_args()
    
    # 提取参数值
    api_url = args.api_url  # API服务器URL
    max_steps = args.max_steps  # 每个episode的最大步数
    
    # ========== 配置参数 ==========
    # 要评估的环境任务列表（可以包含多个任务，如["BinFill", "PatternLock"]）
    env_id_list = ["BinFill"]
    # 数据集根目录路径，包含metadata文件和episode数据
    dataset_root = Path("/data/hongzefu/historybench-v5.7.4-sam2act-generate128/dataset_json")
    
    # ========== 遍历所有环境任务 ==========
    for env_id in env_id_list:
        # 构建metadata文件路径
        # 格式：record_dataset_{环境ID}_metadata.json
        metadata_path = dataset_root / f"record_dataset_{env_id}_metadata.json"
        
        # 读取metadata文件，获取所有episode配置
        episode_records = read_metadata(metadata_path)
        if not episode_records:
            print(f"No episode records found for {env_id}; skipping")
            continue  # 如果没有episode记录，跳过该环境
            
        print(f"Found {len(episode_records)} episodes for {env_id}")
        
        # ========== 初始化Episode配置解析器 ==========
        # EpisodeConfigResolver用于根据episode配置创建对应的环境实例
        resolver = EpisodeConfigResolver(
            env_id=env_id,  # 环境ID（任务类型）
            dataset=None,  # 数据集对象（None表示从metadata文件读取）
            metadata_path=metadata_path,  # metadata文件路径
            render_mode="rgb_array",  # 渲染模式："rgb_array"表示无头渲染（不创建窗口）
            gui_render=False,  # 是否启用GUI渲染（False表示不显示图形界面）
            max_steps_without_demonstration=200,  # 没有演示时的最大步数（用于安全限制）
        )
        
        # ========== 遍历所有episode ==========
        for episode_record in episode_records:
            # 从episode记录中提取信息
            episode = episode_record['episode']  # episode编号
            seed = episode_record.get('seed')  # 随机种子（可选）
            difficulty = episode_record.get('difficulty')  # 难度等级（可选）
            
            print(f"--- Running simulation for episode:{episode}, env: {env_id}, seed: {seed}, difficulty: {difficulty} ---")
            
            # ========== 创建环境并重置 ==========
            # 根据episode编号创建对应的环境实例
            # 返回：环境对象、episode数据集、解析后的种子、解析后的难度
            env, episode_dataset, resolved_seed, resolved_difficulty = resolver.make_env_for_episode(episode)
            # 重置环境，获取初始观测和信息
            obs, info = env.reset()
            
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
            # 在每个episode中，最多执行max_steps步
            for step in range(max_steps):
                print(f"Step {step}/{max_steps}")
                
                # ========== 准备API输入数据 ==========
                try:
                    # 将环境观测数据转换为模型API期望的格式
                    obs_dict = get_model_input(env, obs, step, lang_goal)
                    
                    # ========== 调用SAM2ACT API获取动作 ==========
                    # 构建API请求的payload（载荷）
                    payload = {
                        "obs_obj": obs_dict,  # 观测数据字典（包含相机、机器人状态等）
                        "curr_idx": step,  # 当前时间步索引
                        "lang_goal": lang_goal,  # 语言目标描述
                        "episode_length": max_steps  # episode总长度
                    }
                    
                    # 发送POST请求到API服务器
                    # 使用NumpyEncoder处理numpy类型数据，确保可以正确序列化为JSON
                    response = requests.post(
                        f"{api_url}/act",  # API端点：/act（获取动作）
                        data=json.dumps(payload, cls=NumpyEncoder),  # 将payload序列化为JSON字符串
                        headers={'Content-Type': 'application/json'}  # 设置请求头
                    )
                    
                    # 检查API响应状态
                    if response.status_code != 200:
                        print(f"API Error: {response.status_code} - {response.text}")
                        break  # API调用失败，退出循环
                        
                    # 解析API返回的JSON响应
                    result = response.json()
                    action = result.get('action')  # 提取动作数据
                    
                    # 检查是否成功获取动作
                    if action is None:
                        print("No action returned from API")
                        break  # 没有返回动作，退出循环
                        
                    # ========== 执行动作 ==========
                    # 动作格式：[x, y, z, qx, qy, qz, qw, grip, coll]
                    # 前7个元素：末端执行器目标位姿（位置x,y,z + 四元数qx,qy,qz,qw）
                    # 第8个元素：夹爪动作（>0.5表示打开，<=0.5表示闭合）
                    # 第9个元素：碰撞相关（本脚本中未使用）
                    pose_list = action[:7]  # 提取位姿部分（位置+四元数）
                    gripper_action = action[7]  # 提取夹爪动作
                    
                    # 构建SAPIEN Pose对象
                    # p: 位置向量 [x, y, z]
                    # q: 四元数 [qx, qy, qz, qw]（注意顺序）
                    pose = sapien.Pose(p=pose_list[:3], q=pose_list[3:7])
                    
                    # ========== 处理夹爪动作 ==========
                    # 如果夹爪动作值大于0.5，表示需要打开夹爪
                    if gripper_action > 0.5:
                        planner.open_gripper()  # 打开夹爪
                    else:
                        # 延迟闭合：先移动到目标位姿，然后再闭合夹爪
                        # 这样可以避免在移动过程中夹爪闭合导致碰撞
                        pass
                        
                    # ========== 执行运动规划并移动到目标位姿 ==========
                    try:
                        # 使用螺旋运动（screw motion）规划并执行移动到目标位姿
                        # 螺旋运动是一种平滑的轨迹规划方法，适合机械臂运动
                        planner.move_to_pose_with_screw(pose)
                        
                        # 如果动作指示需要闭合夹爪，在移动完成后闭合
                        if gripper_action <= 0.5:
                            planner.close_gripper()
                            
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
                        print(f"Episode {episode} Success!")
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
            print(f"Final evaluation for episode {episode}: {evaluation}")
            
            # ========== 清理资源 ==========
            # 关闭环境，释放资源（包括图形界面、物理引擎等）
            env.close()
            print(f"--- Finished Running simulation for episode:{episode}, env: {env_id} ---")


# ========== 脚本入口 ==========
# 当脚本被直接运行时（而非作为模块导入），执行main函数
if __name__ == "__main__":
    main()
