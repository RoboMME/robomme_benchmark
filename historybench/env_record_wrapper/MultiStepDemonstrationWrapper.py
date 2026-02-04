"""
MultiStepDemonstrationWrapper: wraps DemonstrationWrapper and exposes a keypoint-step API.

Each step(action) accepts action = keypoint_p (3) + keypoint_q (4) + gripper_action (1) = 8 dims.
Uses planner_denseStep to run move_to_pose_with_RRTStar and close_gripper/open_gripper,
returning 5 lists (obs_list, reward_list, terminated_list, truncated_list, info_list).
Callers must have scripts/ on sys.path for planner_fail_safe import.
"""
import numpy as np
import sapien
import gymnasium as gym

from . import planner_denseStep


class RRTPlanFailure(RuntimeError):
    """Raised when move_to_pose_with_RRTStar returns -1 (planning failed)."""


class MultiStepDemonstrationWrapper(gym.Wrapper):
    """
    Wraps DemonstrationWrapper; step(action) interprets action as (keypoint_p, keypoint_q, gripper_action)
    and runs the planner via planner_denseStep, returning 5 lists (obs, reward, terminated, truncated, info).
    """

    def __init__(self, env, gui_render=True, vis=True, **kwargs):
        super().__init__(env)
        self._planner = None
        self._gui_render = gui_render
        self._vis = vis
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

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
        """One step with current qpos + gripper to get obs without moving."""
        robot = self.env.unwrapped.agent.robot
        qpos = robot.get_qpos().cpu().numpy().flatten()
        arm = qpos[:7]
        gripper = float(qpos[7]) if len(qpos) > 7 else 0.0
        action = np.hstack([arm, gripper])
        return self.env.step(action)

    def step(self, action):
        """Keypoint step: runs RRT* move + optional gripper via planner_denseStep, returns 5 lists."""
        action = np.asarray(action, dtype=np.float64).flatten()
        if action.size < 8:
            raise ValueError(f"action must have at least 8 elements, got {action.size}")
        keypoint_p = action[:3]
        keypoint_q = action[3:7]
        gripper_action = float(action[7])

        pose = sapien.Pose(p=keypoint_p, q=keypoint_q)
        planner = self._get_planner()

        current_p = self._current_tcp_p()
        dist = np.linalg.norm(current_p - keypoint_p)

        if dist < 0.001:
            obs, reward, terminated, truncated, info = self._no_op_step()
            obs_list = [obs]
            reward_list = [reward]
            terminated_list = [terminated]
            truncated_list = [truncated]
            info_list = [info]
        else:
            result = planner_denseStep.move_to_pose_with_RRTStar(planner, pose)
            if result == -1:
                raise RRTPlanFailure("move_to_pose_with_RRTStar failed (returned -1)")
            obs_list, reward_list, terminated_list, truncated_list, info_list = result

        if gripper_action == -1:
            result = planner_denseStep.close_gripper(planner)
            if result != -1:
                go_obs, go_r, go_t, go_tr, go_i = result
                obs_list.extend(go_obs)
                reward_list.extend(go_r)
                terminated_list.extend(go_t)
                truncated_list.extend(go_tr)
                info_list.extend(go_i)
        elif gripper_action == 1:
            result = planner_denseStep.open_gripper(planner)
            if result != -1:
                go_obs, go_r, go_t, go_tr, go_i = result
                obs_list.extend(go_obs)
                reward_list.extend(go_r)
                terminated_list.extend(go_t)
                truncated_list.extend(go_tr)
                info_list.extend(go_i)

        return obs_list, reward_list, terminated_list, truncated_list, info_list

    def reset(self, **kwargs):
        self._planner = None
        return self.env.reset(**kwargs)

    def close(self):
        self._planner = None
        return self.env.close()
