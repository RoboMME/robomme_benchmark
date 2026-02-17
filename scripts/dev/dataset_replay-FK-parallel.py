"""
从 HDF5 数据集回放 episode 并保存视频。

从 record_dataset_<Task>.h5 中读取已录制的关节动作 (joint_action)，
通过正向运动学 (FK) 转换为末端执行器位姿动作 (EE pose action)，
输入到以 EE_POSE_ACTION_SPACE 包裹的环境中执行回放，
最后将前置/腕部相机的左右拼接视频写入磁盘。
"""

import os
from typing import Optional, Tuple

import cv2
import h5py
import imageio
import numpy as np
import sapien
import torch

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)

from robomme.robomme_env import *
from robomme.robomme_env.utils import *
from robomme.env_record_wrapper import BenchmarkEnvBuilder
from robomme.robomme_env.utils import EE_POSE_ACTION_SPACE
from robomme.robomme_env.utils.rpy_util import build_endeffector_pose_dict

# --- 配置 ---
GUI_RENDER = False
REPLAY_VIDEO_DIR = "replay_videos"
VIDEO_FPS = 30


def _init_fk_planner(env) -> Tuple:
    """创建 PandaArmMotionPlanningSolver 并返回 FK 所需的辅助对象。

    返回:
        (mplib_planner, ee_link_idx, robot_base_pose)
        - mplib_planner: mplib.Planner 实例，用于 FK 计算
        - ee_link_idx: 末端执行器在 pinocchio 模型中的 link 索引
        - robot_base_pose: 机器人基座在世界坐标系下的位姿
    """
    solver = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    mplib_planner = solver.planner
    ee_link_idx = mplib_planner.link_name_2_idx[mplib_planner.move_group]
    robot_base_pose = env.unwrapped.agent.robot.pose

    print(f"[FK] move_group: {mplib_planner.move_group}, "
          f"ee_link_idx: {ee_link_idx}, "
          f"link_names: {mplib_planner.user_link_names}")
    return mplib_planner, ee_link_idx, robot_base_pose


def _joint_action_to_ee_pose(
    mplib_planner,
    joint_action: np.ndarray,
    robot_base_pose: sapien.Pose,
    ee_link_idx: int,
    prev_ee_quat_wxyz: Optional[torch.Tensor] = None,
    prev_ee_rpy_xyz: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """通过正向运动学 (FK) 将 8 维关节动作转换为 7 维末端执行器位姿动作。

    参数:
        mplib_planner: mplib.Planner 实例（来自 PandaArmMotionPlanningSolver）。
        joint_action: 8 维数组 [q1..q7, gripper]。
        robot_base_pose: 机器人基座在世界坐标系下的 Sapien Pose。
        ee_link_idx: 末端执行器在 pinocchio 模型中的 link 索引。
        prev_ee_quat_wxyz: 上一帧的四元数缓存（用于符号对齐）。
        prev_ee_rpy_xyz: 上一帧的 RPY 缓存（用于连续性展开）。

    返回:
        ee_action: 7 维 [x, y, z, roll, pitch, yaw, gripper]。
        new_prev_quat: 更新后的四元数缓存。
        new_prev_rpy: 更新后的 RPY 缓存。
    """
    action = np.asarray(joint_action, dtype=np.float64).flatten()
    arm_qpos = action[:7]
    gripper = float(action[7]) if action.size > 7 else -1.0

    # 构建完整 qpos：7 个手臂关节 + 2 个夹爪手指关节
    finger_pos = max(gripper, 0.0) if gripper >= 0 else 0.04
    full_qpos = np.concatenate([arm_qpos, [finger_pos, finger_pos]])

    # 在机器人基座坐标系下计算正向运动学
    pmodel = mplib_planner.pinocchio_model
    pmodel.compute_forward_kinematics(full_qpos)
    fk_result = pmodel.get_link_pose(ee_link_idx)  # 7 维 [x,y,z, qw,qx,qy,qz]

    p_base = fk_result[:3]
    q_base_wxyz = fk_result[3:]  # wxyz 四元数格式

    # 基座坐标系 -> 世界坐标系 变换
    pose_in_base = sapien.Pose(p_base, q_base_wxyz)
    world_pose = robot_base_pose * pose_in_base

    # 使用共享工具构建连续 RPY（含四元数归一化、符号对齐、RPY 展开）
    position_t = torch.as_tensor(
        np.asarray(world_pose.p, dtype=np.float64), dtype=torch.float64
    )
    quat_wxyz_t = torch.as_tensor(
        np.asarray(world_pose.q, dtype=np.float64), dtype=torch.float64
    )
    pose_dict, new_prev_quat, new_prev_rpy = build_endeffector_pose_dict(
        position_t, quat_wxyz_t,
        prev_ee_quat_wxyz, prev_ee_rpy_xyz,
    )

    # 拼接为 7 维 EE 位姿动作：[位置(3), RPY(3), 夹爪(1)]
    pos_np = pose_dict["pose"].detach().cpu().numpy().flatten()[:3]
    rpy_np = pose_dict["rpy"].detach().cpu().numpy().flatten()[:3]
    ee_action = np.concatenate([pos_np, rpy_np, [gripper]]).astype(np.float64)

    return ee_action, new_prev_quat, new_prev_rpy


def _frame_from_obs(obs: dict, is_video_frame: bool = False) -> np.ndarray:
    """从前置相机和腕部相机的观测中构建一帧左右拼接图像。"""
    front = obs["front_camera"][0].cpu().numpy()
    wrist = obs["wrist_camera"][0].cpu().numpy()
    frame = np.concatenate([front, wrist], axis=1).astype(np.uint8)
    if is_video_frame:
        # 视频演示帧加红色边框标注
        frame = cv2.rectangle(
            frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 10
        )
    return frame


def _first_execution_step(episode_data) -> int:
    """返回第一个非视频演示步的索引（即实际执行开始的步数）。"""
    step_idx = 0
    while episode_data[f"timestep_{step_idx}"]["info"]["is_video_demo"][()]:
        step_idx += 1
    return step_idx


def process_episode(
    h5_file_path: str, episode_idx: int, env_id: str, gui_render: bool = False,
) -> None:
    """回放 HDF5 中的一个 episode：读取关节动作、FK 转换、环境执行、保存视频。

    每个 worker 进程独立打开 HDF5 文件，避免跨进程共享文件句柄。
    """
    with h5py.File(h5_file_path, "r") as env_data:
        episode_data = env_data[f"episode_{episode_idx}"]
        task_goal = episode_data["setup"]["task_goal"][()].decode()
        total_steps = sum(1 for k in episode_data.keys() if k.startswith("timestep_"))

        step_idx = _first_execution_step(episode_data)
        print(f"[ep{episode_idx}] 执行起始步索引: {step_idx}")

        # 以 EE_POSE_ACTION_SPACE 创建环境（会包裹 EndeffectorDemonstrationWrapper）
        env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="train",
            action_space=EE_POSE_ACTION_SPACE,
            gui_render=gui_render,
        )
        env = env_builder.make_env_for_episode(episode_idx)
        print(f"[ep{episode_idx}] 任务: {env_id}, 目标: {task_goal}")

        obs, info = env.reset()

        # 初始化 FK 规划器（需要在 env.reset() 之后调用）
        mplib_planner, ee_link_idx, robot_base_pose = _init_fk_planner(env)

        # 观测列表：长度为 1 表示无视频，长度 > 1 表示含视频演示；最后一个元素是当前帧
        frames = []
        n_obs = len(obs["front_camera"])
        for i in range(n_obs):
            single_obs = {k: [v[i]] for k, v in obs.items()}
            frames.append(_frame_from_obs(single_obs, is_video_frame=(i < n_obs - 1)))
        print(f"[ep{episode_idx}] 初始帧数（视频 + 当前帧）: {len(frames)}")

        outcome = "unknown"
        prev_quat: Optional[torch.Tensor] = None
        prev_rpy: Optional[torch.Tensor] = None
        try:
            while step_idx < total_steps:
                # 从 HDF5 读取关节动作
                joint_action = np.asarray(
                    episode_data[f"timestep_{step_idx}"]["action"]["joint_action"][()],
                    dtype=np.float64,
                )

                # 正向运动学：joint_action -> ee_pose action
                ee_action, prev_quat, prev_rpy = _joint_action_to_ee_pose(
                    mplib_planner, joint_action, robot_base_pose, ee_link_idx,
                    prev_ee_quat_wxyz=prev_quat,
                    prev_ee_rpy_xyz=prev_rpy,
                )

                # 第一步打印调试信息，方便验证 FK 转换结果
                if step_idx == _first_execution_step(episode_data):
                    print(f"[ep{episode_idx}][FK] 第一步 joint_action: {joint_action}")
                    print(f"[ep{episode_idx}][FK] 第一步 ee_action:    {ee_action}")

                # 将 EE 位姿动作输入环境执行
                obs, _, terminated, _, info = env.step(ee_action)
                frames.append(_frame_from_obs(obs))

                if gui_render:
                    env.render()

                # TODO: hongze 修正嵌套列表的处理
                if terminated:
                    if info.get("success", False)[-1][-1]:
                        outcome = "success"
                    if info.get("fail", False)[-1][-1]:
                        outcome = "fail"
                    break
                step_idx += 1
        finally:
            env.close()

    # 保存回放视频
    safe_goal = task_goal.replace(" ", "_").replace("/", "_")
    os.makedirs(REPLAY_VIDEO_DIR, exist_ok=True)
    video_name = f"{outcome}_{env_id}_ep{episode_idx}_{safe_goal}_step-{len(frames)}.mp4"
    video_path = os.path.join(REPLAY_VIDEO_DIR, video_name)
    imageio.mimsave(video_path, frames, fps=VIDEO_FPS)
    print(f"[ep{episode_idx}] 视频已保存到 {video_path}")


def _worker_init(gpu_id_queue) -> None:
    """Pool worker 初始化函数，在 CUDA 初始化前绑定 GPU。

    每个 worker 进程启动时从队列中取出一个 GPU ID 并设置环境变量，
    确保该进程的所有后续 CUDA 操作都在指定 GPU 上执行。
    """
    gpu_id = gpu_id_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[Worker PID {os.getpid()}] 绑定 GPU {gpu_id}")


def _process_episode_worker(args: Tuple[str, int, str, bool]) -> str:
    """multiprocessing worker 入口，解包参数并调用 process_episode。"""
    h5_file_path, episode_idx, env_id, gui_render = args
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    try:
        process_episode(h5_file_path, episode_idx, env_id, gui_render=gui_render)
        return f"OK: {env_id} ep{episode_idx} (GPU {gpu_id})"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"FAIL: {env_id} ep{episode_idx} (GPU {gpu_id}): {e}"


def replay(
    h5_data_dir: str = "/data/hongzefu/data_0214",
    num_workers: int = 20,
    gui_render: bool = False,
    gpu_ids: str = "0,1",
) -> None:
    """遍历指定目录下所有任务的 HDF5 文件，对每个 env 的多 episode 并行回放。

    参数:
        h5_data_dir: HDF5 数据集所在目录。
        num_workers: 每个 env 的并行 worker 数量。
        gui_render: 是否开启 GUI 渲染（多进程下建议关闭）。
        gpu_ids: 逗号分隔的 GPU ID 列表，worker 按轮询方式交替使用。
                 例如 "0,1" 表示在 GPU 0 和 GPU 1 之间交替分配。
    """
    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    gpu_id_list = [int(g.strip()) for g in gpu_ids.split(",")]
    print(f"使用 GPU: {gpu_id_list}, workers: {num_workers}")

    env_id_list = BenchmarkEnvBuilder.get_task_list()
    for env_id in env_id_list:
        file_name = f"record_dataset_{env_id}.h5"
        file_path = os.path.join(h5_data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"跳过 {env_id}: 文件不存在: {file_path}")
            continue

        # 快速读取 episode 列表后关闭文件
        with h5py.File(file_path, "r") as data:
            episode_indices = sorted(
                int(k.split("_")[1])
                for k in data.keys()
                if k.startswith("episode_")
            )
        print(f"任务: {env_id}, 共 {len(episode_indices)} 个 episode, "
              f"workers: {num_workers}, GPUs: {gpu_id_list}")

        # 构造 worker 参数列表
        worker_args = [
            (file_path, ep_idx, env_id, gui_render)
            for ep_idx in episode_indices
        ]

        # 每轮创建新的 GPU 分配队列，worker 启动时各取一个 GPU ID
        gpu_id_queue = ctx.Queue()
        for i in range(num_workers):
            gpu_id_queue.put(gpu_id_list[i % len(gpu_id_list)])

        # 并行回放（initializer 在每个 worker 进程启动时绑定 GPU）
        with ctx.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(gpu_id_queue,),
        ) as pool:
            results = pool.map(_process_episode_worker, worker_args)

        for r in results:
            print(r)


if __name__ == "__main__":
    import tyro
    tyro.cli(replay)
