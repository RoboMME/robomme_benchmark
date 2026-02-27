"""
EndeffectorDemonstrationWrapper: Outer wrapper, receiving ee_pose/ee_quat action and converting to joint action via IK.

- Supports two external interfaces:
  1) rpy mode: action = [ee_p(3), rpy(3), gripper(1)], total 7 dimensions
  2) quat mode: action = [ee_p(3), quat(4), gripper(1)], total 8 dimensions
- PatternLock/RouteStick: Internal ignores gripper, passes down 7-dimensional joint action
"""
import numpy as np
import torch
import gymnasium as gym
from typing import Literal

from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)
from ..robomme_env.utils.rpy_util import rpy_xyz_to_quat_wxyz_torch


class EndeffectorDemonstrationWrapper(gym.Wrapper):
    """
    Wrap an environment expecting joint actions. step(action) receives ee pose:
    - rpy mode: action = [ee_p(3), rpy(3), gripper(1)] (7 dim)
    - quat mode: action = [ee_p(3), quat(4), gripper(1)] (8 dim)
    - rpy mode internally converts RPY to quat (wxyz); quat mode directly uses input quat
    - PatternLock/RouteStick compatible with no-gripper input, and ignores gripper internally
    Internally performs IK to get joint_action, then calls inner env.step(joint_action), returning:
    (obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch).
    """

    # stick environment internally ignores gripper and passes down 7-dimensional joint action.
    _EE_POSE_7D_ENV_IDS = ("PatternLock", "RouteStick")

    def __init__(self, env, action_repr: Literal["rpy", "quat"] = "rpy"):
        super().__init__(env)
        if action_repr not in ("rpy", "quat"):
            raise ValueError(f"Unsupported action_repr '{action_repr}'. Allowed: ['quat', 'rpy']")
        self.action_repr = action_repr
        self._ee_pose_planner = None

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).flatten()
        env_spec = getattr(self.env.unwrapped, "spec", None)
        env_id = getattr(env_spec, "id", "<unknown_env>")
        no_gripper_env = env_id in self._EE_POSE_7D_ENV_IDS

        if no_gripper_env:
            required = 6 if self.action_repr == "rpy" else 7
            if action.size < required:
                detail = "ee_p, rpy" if self.action_repr == "rpy" else "ee_p, quat"
                raise ValueError(
                    f"[{env_id}] action must have at least {required} elements ({detail}) "
                    f"for no-gripper env, got {action.size}"
                )
        else:
            required = 7 if self.action_repr == "rpy" else 8
            if action.size < required:
                detail = "ee_p, rpy, gripper" if self.action_repr == "rpy" else "ee_p, quat, gripper"
                raise ValueError(
                    f"[{env_id}] action must have at least {required} elements ({detail}), got {action.size}"
                )

        ee_p = action[:3]
        if self.action_repr == "rpy":
            rpy = action[3:6]
            # RPY → quat (wxyz)
            rpy_t = torch.as_tensor(rpy, dtype=torch.float64)
            ee_q = rpy_xyz_to_quat_wxyz_torch(rpy_t).numpy()
            gripper_idx = 6
        else:
            ee_q = action[3:7]
            gripper_idx = 7
        gripper = None if no_gripper_env else float(action[gripper_idx])

        if self._ee_pose_planner is None:
            if no_gripper_env:
                self._ee_pose_planner = PandaStickMotionPlanningSolver(
                    self.env,
                    debug=False,
                    vis=False,
                    base_pose=self.env.unwrapped.agent.robot.pose,
                    visualize_target_grasp_pose=False,
                    print_env_info=False,
                    joint_vel_limits=0.3,
                )
            else:
                self._ee_pose_planner = PandaArmMotionPlanningSolver(
                    self.env,
                    debug=False,
                    vis=False,
                    base_pose=self.env.unwrapped.agent.robot.pose,
                    visualize_target_grasp_pose=False,
                    print_env_info=False,
                )
        planner = self._ee_pose_planner
        goal_world = np.concatenate([ee_p, ee_q])
        goal_base = planner.planner.transform_goal_to_wrt_base(goal_world)
        current_qpos = planner.robot.get_qpos().cpu().numpy()[0]
        ik_status, ik_solutions = planner.planner.IK(goal_base, current_qpos)
        if ik_status != "Success" or len(ik_solutions) == 0:
            error_msg = f"ee step ({self.action_repr}): IK failed (status={ik_status}, num_solutions={len(ik_solutions)})"
            return ({}, 0.0, True, False, {"status": "error", "error_message": error_msg})
        qpos = np.asarray(ik_solutions[0][:7], dtype=np.float64)
        if no_gripper_env:
            joint_action = qpos
        else:
            joint_action = np.hstack([qpos, gripper])
        
        return self.env.step(joint_action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def close(self):
        return self.env.close()
