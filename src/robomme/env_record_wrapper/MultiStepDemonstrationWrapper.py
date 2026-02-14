"""
MultiStepDemonstrationWrapper: Wraps DemonstrationWrapper to provide keypoint step interface.

Each step(action) receives action = keypoint_p(3) + rpy(3) + gripper_action(1), total 7 dimensions.
Internally converts RPY to quat then calls move_to_pose_with_screw and close_gripper/open_gripper via planner_denseStep,
where PatternLock/RouteStick will force skip close_gripper/open_gripper.
Returns unified batch (obs/info as dictionary of lists, reward/terminated/truncated as 1D tensors).
Caller must ensure scripts/ is in sys.path to import planner_fail_safe.
"""
import numpy as np
import sapien
import torch
import gymnasium as gym

from ..robomme_env.utils import planner_denseStep
from ..robomme_env.utils.rpy_util import rpy_xyz_to_quat_wxyz_torch


class RRTPlanFailure(RuntimeError):
    """Raised when move_to_pose_with_RRTStar returns -1 (planning failed)."""


class MultiStepDemonstrationWrapper(gym.Wrapper):
    """
    Wraps DemonstrationWrapper. step(action) interprets action as
    (keypoint_p, rpy, gripper_action) total 7 dims, internally converts RPY to quat,
    executes planning via planner_denseStep, and returns unified batch.
    """

    def __init__(self, env, gui_render=True, vis=True, **kwargs):
        super().__init__(env)
        self._planner = None
        self._gui_render = gui_render
        self._vis = vis
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
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
        from ..robomme_env.utils.planner_fail_safe import (
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
        """Execute one step using current qpos + gripper, without moving arm, only to get observation."""
        robot = self.env.unwrapped.agent.robot
        qpos = robot.get_qpos().cpu().numpy().flatten()
        arm = qpos[:7]
        gripper = float(qpos[7]) if len(qpos) > 7 else 0.0
        action = np.hstack([arm, gripper])
        return self.env.step(action)

    def step(self, action):
        """Execute keypoint step: RRT* movement + optional gripper action, return unified batch."""
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

        # PatternLock/RouteStick force skip gripper action (even if planner object has method with same name).
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
