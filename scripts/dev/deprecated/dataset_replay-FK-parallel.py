"""
Replay episodes from an HDF5 dataset and save videos.

Read recorded joint actions (joint_action) from record_dataset_<Task>.h5,
convert them to end-effector pose actions (EE pose actions) via forward kinematics (FK),
replay them in an environment wrapped by EE_POSE_ACTION_SPACE,
and finally save side-by-side front/wrist camera videos to disk.
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

# --- Configuration ---
GUI_RENDER = False
REPLAY_VIDEO_DIR = "replay_videos"
VIDEO_FPS = 30
MAX_STEPS = 1000


def _init_fk_planner(env) -> Tuple:
    """Create PandaArmMotionPlanningSolver and return helper objects needed for FK.

    Returns:
        (mplib_planner, ee_link_idx, robot_base_pose)
        - mplib_planner: mplib.Planner instance used for FK computation
        - ee_link_idx: end-effector link index in the pinocchio model
        - robot_base_pose: robot base pose in world coordinates
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
    """Convert 8D joint action to 7D end-effector pose action via forward kinematics (FK).

    Args:
        mplib_planner: mplib.Planner instance (from PandaArmMotionPlanningSolver).
        joint_action: 8D array [q1..q7, gripper].
        robot_base_pose: robot base pose as a Sapien Pose.
        ee_link_idx: end-effector link index in the pinocchio model.
        prev_ee_quat_wxyz: previous-frame quaternion cache (for sign alignment).
        prev_ee_rpy_xyz: previous-frame RPY cache (for continuity unwrapping).

    Returns:
        ee_action: 7D [x, y, z, roll, pitch, yaw, gripper].
        new_prev_quat: updated quaternion cache.
        new_prev_rpy: updated RPY cache.
    """
    action = np.asarray(joint_action, dtype=np.float64).flatten()
    arm_qpos = action[:7]
    gripper = float(action[7]) if action.size > 7 else -1.0

    # Build full qpos: 7 arm joints + 2 gripper finger joints
    finger_pos = max(gripper, 0.0) if gripper >= 0 else 0.04
    full_qpos = np.concatenate([arm_qpos, [finger_pos, finger_pos]])

    # Compute forward kinematics in the robot-base coordinate frame
    pmodel = mplib_planner.pinocchio_model
    pmodel.compute_forward_kinematics(full_qpos)
    fk_result = pmodel.get_link_pose(ee_link_idx)  # 7D [x,y,z, qw,qx,qy,qz]

    p_base = fk_result[:3]
    q_base_wxyz = fk_result[3:]  # wxyz quaternion format

    # base frame -> world frame transform
    pose_in_base = sapien.Pose(p_base, q_base_wxyz)
    world_pose = robot_base_pose * pose_in_base

    # Use shared utilities to build continuous RPY (quaternion normalization, sign alignment, RPY unwrapping)
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

    # Concatenate into 7D EE pose action: [position(3), RPY(3), gripper(1)]
    pos_np = pose_dict["pose"].detach().cpu().numpy().flatten()[:3]
    rpy_np = pose_dict["rpy"].detach().cpu().numpy().flatten()[:3]
    ee_action = np.concatenate([pos_np, rpy_np, [gripper]]).astype(np.float64)

    return ee_action, new_prev_quat, new_prev_rpy


def _frame_from_obs(obs: dict, is_video_frame: bool = False) -> np.ndarray:
    """Build one side-by-side frame from front and wrist camera observations."""
    front = obs["front_camera"][0].cpu().numpy()
    wrist = obs["wrist_camera"][0].cpu().numpy()
    frame = np.concatenate([front, wrist], axis=1).astype(np.uint8)
    if is_video_frame:
        # Mark video-demo frames with a red border
        frame = cv2.rectangle(
            frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 10
        )
    return frame


def _first_execution_step(episode_data) -> int:
    """Return the first non-video-demo step index (actual execution start step)."""
    step_idx = 0
    while episode_data[f"timestep_{step_idx}"]["info"]["is_video_demo"][()]:
        step_idx += 1
    return step_idx


def process_episode(
    h5_file_path: str, episode_idx: int, env_id: str, gui_render: bool = False,
) -> None:
    """Replay one episode in HDF5: read joint actions, run FK conversion, execute the environment, and save video.

    Each worker process opens the HDF5 file independently to avoid cross-process shared file handles.
    """
    with h5py.File(h5_file_path, "r") as env_data:
        episode_data = env_data[f"episode_{episode_idx}"]
        task_goal = episode_data["setup"]["task_goal"][()].decode()
        total_steps = sum(1 for k in episode_data.keys() if k.startswith("timestep_"))

        step_idx = _first_execution_step(episode_data)
        print(f"[ep{episode_idx}] execution start step index: {step_idx}")

        # Create environment with EE_POSE_ACTION_SPACE (wrapped by EndeffectorDemonstrationWrapper)
        env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="train",
            action_space=EE_POSE_ACTION_SPACE,
            gui_render=gui_render,
        )
        env = env_builder.make_env_for_episode(episode_idx, max_steps=MAX_STEPS)
        print(f"[ep{episode_idx}] task: {env_id}, goal: {task_goal}")

        obs, info = env.reset()

        # Initialize FK planner (must be called after env.reset())
        mplib_planner, ee_link_idx, robot_base_pose = _init_fk_planner(env)

        # Observation list: length 1 means no demo video, length >1 means includes demo video; last element is current frame
        frames = []
        n_obs = len(obs["front_camera"])
        for i in range(n_obs):
            single_obs = {k: [v[i]] for k, v in obs.items()}
            frames.append(_frame_from_obs(single_obs, is_video_frame=(i < n_obs - 1)))
        print(f"[ep{episode_idx}] initial frame count (demo video + current frame): {len(frames)}")

        outcome = "unknown"
        prev_quat: Optional[torch.Tensor] = None
        prev_rpy: Optional[torch.Tensor] = None
        try:
            while step_idx < total_steps:
                # Read joint action from HDF5
                joint_action = np.asarray(
                    episode_data[f"timestep_{step_idx}"]["action"]["joint_action"][()],
                    dtype=np.float64,
                )

                # Forward kinematics: joint_action -> ee_pose action
                ee_action, prev_quat, prev_rpy = _joint_action_to_ee_pose(
                    mplib_planner, joint_action, robot_base_pose, ee_link_idx,
                    prev_ee_quat_wxyz=prev_quat,
                    prev_ee_rpy_xyz=prev_rpy,
                )

                # Print debug info on the first step to verify FK conversion
                if step_idx == _first_execution_step(episode_data):
                    print(f"[ep{episode_idx}][FK] first step joint_action: {joint_action}")
                    print(f"[ep{episode_idx}][FK] first step ee_action:    {ee_action}")

                # Execute EE pose action in the environment
                obs, _, terminated, _, info = env.step(ee_action)
                frames.append(_frame_from_obs(obs))

                if gui_render:
                    env.render()

                # TODO: hongze fix nested-list handling
                if terminated:
                    if info.get("success", False)[-1][-1]:
                        outcome = "success"
                    if info.get("fail", False)[-1][-1]:
                        outcome = "fail"
                    break
                step_idx += 1
        finally:
            env.close()

    # Save replay video
    safe_goal = task_goal.replace(" ", "_").replace("/", "_")
    os.makedirs(REPLAY_VIDEO_DIR, exist_ok=True)
    video_name = f"{outcome}_{env_id}_ep{episode_idx}_{safe_goal}_step-{len(frames)}.mp4"
    video_path = os.path.join(REPLAY_VIDEO_DIR, video_name)
    imageio.mimsave(video_path, frames, fps=VIDEO_FPS)
    print(f"[ep{episode_idx}] Video saved to {video_path}")


def _worker_init(gpu_id_queue) -> None:
    """Pool worker initializer that binds a GPU before CUDA initialization.

    When each worker starts, it takes one GPU ID from the queue and sets env vars,
    ensuring all later CUDA ops in that process run on the assigned GPU.
    """
    gpu_id = gpu_id_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[Worker PID {os.getpid()}] bind GPU {gpu_id}")


def _process_episode_worker(args: Tuple[str, int, str, bool]) -> str:
    """multiprocessing worker entrypoint: unpack args and call process_episode."""
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
    """Iterate through all task HDF5 files in the given directory and replay multiple episodes per env in parallel.

    Args:
        h5_data_dir: Directory containing HDF5 datasets.
        num_workers: Number of parallel workers per env.
        gui_render: Whether to enable GUI rendering (recommended off in multiprocessing).
        gpu_ids: Comma-separated GPU ID list; workers use them in round-robin order.
                 For example, "0,1" alternates assignment between GPU 0 and GPU 1.
    """
    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    gpu_id_list = [int(g.strip()) for g in gpu_ids.split(",")]
    print(f"Using GPUs: {gpu_id_list}, workers: {num_workers}")

    env_id_list = BenchmarkEnvBuilder.get_task_list()
    for env_id in env_id_list:
        file_name = f"record_dataset_{env_id}.h5"
        file_path = os.path.join(h5_data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Skip {env_id}: file does not exist: {file_path}")
            continue

        # Quickly read episode list and close file
        with h5py.File(file_path, "r") as data:
            episode_indices = sorted(
                int(k.split("_")[1])
                for k in data.keys()
                if k.startswith("episode_")
            )
        print(f"task: {env_id}, total {len(episode_indices)} episodes, "
              f"workers: {num_workers}, GPUs: {gpu_id_list}")

        # Build worker argument list
        worker_args = [
            (file_path, ep_idx, env_id, gui_render)
            for ep_idx in episode_indices
        ]

        # Create a new GPU assignment queue for each round; each worker grabs one GPU ID at startup
        gpu_id_queue = ctx.Queue()
        for i in range(num_workers):
            gpu_id_queue.put(gpu_id_list[i % len(gpu_id_list)])

        # Parallel replay (initializer binds GPU when each worker starts)
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
