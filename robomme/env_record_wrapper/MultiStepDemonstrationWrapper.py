"""
MultiStepDemonstrationWrapper：封装 DemonstrationWrapper，对外提供 keypoint step 接口。

每次 step(action) 接收 action = keypoint_p(3) + rpy(3) + gripper_action(1)，共 7 维。
内部将 RPY 转为 quat 后通过 planner_denseStep 调用 move_to_pose_with_screw 与 close_gripper/open_gripper，
其中 PatternLock/RouteStick 会强制跳过 close_gripper/open_gripper。
返回统一批次（obs/info 为字典列式列表，reward/terminated/truncated 为一维张量）。
调用方需保证 scripts/ 在 sys.path 中，以便导入 planner_fail_safe。
"""
import numpy as np
import sapien
import torch
import gymnasium as gym

from . import planner_denseStep
from .rpy_util import rpy_xyz_to_quat_wxyz_torch


class RRTPlanFailure(RuntimeError):
    """当 move_to_pose_with_RRTStar 返回 -1（规划失败）时抛出。"""


class MultiStepDemonstrationWrapper(gym.Wrapper):
    """
    封装 DemonstrationWrapper。step(action) 会把 action 解释为
    (keypoint_p, rpy, gripper_action) 共 7 维，内部将 RPY 转为 quat 后
    通过 planner_denseStep 执行规划，返回统一批次。
    """

    def __init__(self, env, gui_render=True, vis=True, **kwargs):
        super().__init__(env)
        self._planner = None
        self._gui_render = gui_render
        self._vis = vis
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

    @staticmethod
    def _batch_to_steps(batch):
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = batch
        n = int(reward_batch.numel())
        steps = []
        obs_keys = list(obs_batch.keys())
        info_keys = list(info_batch.keys())
        for idx in range(n):
            obs = {k: obs_batch[k][idx] for k in obs_keys}
            info = {k: info_batch[k][idx] for k in info_keys}
            reward = reward_batch[idx]
            terminated = terminated_batch[idx]
            truncated = truncated_batch[idx]
            steps.append((obs, reward, terminated, truncated, info))
        return steps

    def _get_planner(self):
        if self._planner is not None:
            return self._planner
        from planner_fail_safe import (
            FailAwarePandaArmMotionPlanningSolver,
            FailAwarePandaStickMotionPlanningSolver,
        )

        env_id = self.env.unwrapped.spec.id
        base_pose = self.env.unwrapped.agent.robot.pose
        if env_id in ("PatternLock", "RouteStick"):
            self._planner = FailAwarePandaStickMotionPlanningSolver(
                self.env,
                debug=False,
                vis=self._vis,
                base_pose=base_pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_vel_limits=0.3,
            )
        else:
            self._planner = FailAwarePandaArmMotionPlanningSolver(
                self.env,
                debug=False,
                vis=self._vis,
                base_pose=base_pose,
                visualize_target_grasp_pose=True,
                print_env_info=False,
            )
        return self._planner

    def _current_tcp_p(self):
        current_pose = self.env.unwrapped.agent.tcp.pose
        p = current_pose.p
        if hasattr(p, "cpu"):
            p = p.cpu().numpy()
        p = np.asarray(p).flatten()
        return p

    def _no_op_step(self):
        """使用当前 qpos + gripper 执行一步，不移动机械臂，仅获取观测。"""
        robot = self.env.unwrapped.agent.robot
        qpos = robot.get_qpos().cpu().numpy().flatten()
        arm = qpos[:7]
        gripper = float(qpos[7]) if len(qpos) > 7 else 0.0
        action = np.hstack([arm, gripper])
        return self.env.step(action)

    def step(self, action):
        """执行关键点 step：RRT* 移动 + 可选夹爪动作，返回统一批次。"""
        action = np.asarray(action, dtype=np.float64).flatten()
        if action.size < 7:
            raise ValueError(f"action must have at least 7 elements, got {action.size}")
        keypoint_p = action[:3]
        rpy = action[3:6]
        gripper_action = float(action[6])

        # RPY → quat (wxyz) for sapien.Pose
        rpy_t = torch.as_tensor(rpy, dtype=torch.float64)
        keypoint_q = rpy_xyz_to_quat_wxyz_torch(rpy_t).numpy()

        pose = sapien.Pose(p=keypoint_p, q=keypoint_q)
        planner = self._get_planner()
        is_stick_env = self.env.unwrapped.spec.id in ("PatternLock", "RouteStick")

        current_p = self._current_tcp_p()
        dist = np.linalg.norm(current_p - keypoint_p)

        collected_steps = []
        # if dist < 0.001:
        #     collected_steps.append(self._no_op_step())
        # else:
        move_steps = planner_denseStep._collect_dense_steps(
            planner, lambda: planner.move_to_pose_with_screw(pose)
        )
        if move_steps == -1:
            raise RRTPlanFailure("move_to_pose_with_screw failed (returned -1)")
        collected_steps.extend(move_steps)

        # PatternLock/RouteStick 强制跳过夹爪动作（即使规划器对象存在同名方法）。
        if not is_stick_env:
            if gripper_action == -1:
                if hasattr(planner, "close_gripper"):
                    result = planner_denseStep.close_gripper(planner)
                    if result != -1:
                        collected_steps.extend(self._batch_to_steps(result))
            elif gripper_action == 1:
                if hasattr(planner, "open_gripper"):
                    result = planner_denseStep.open_gripper(planner)
                    if result != -1:
                        collected_steps.extend(self._batch_to_steps(result))

        return planner_denseStep.to_step_batch(collected_steps)

    def reset(self, **kwargs):
        self._planner = None
        return self.env.reset(**kwargs)

    def close(self):
        self._planner = None
        return self.env.close()
