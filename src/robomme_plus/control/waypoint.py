# Adapted for robomme-plus: private imports, task IDs, and instance hooks.
"""
MultiStepDemonstrationWrapper: Wraps DemonstrationWrapper to provide waypoint step interface.

Each step(action) receives action = waypoint_p(3) + rpy(3) + gripper_action(1), total 7 dimensions.
Internally converts RPY to quat then calls move_to_pose_with_screw and close_gripper/open_gripper via planner_denseStep,
where PatternLock/RouteStick will force skip close_gripper/open_gripper.
Returns obs as dictionary-of-lists, and reward/terminated/truncated as the last step value.
"""

import gymnasium as gym
import numpy as np
import sapien
import torch

import robomme_plus.evaluation.batches as planner_denseStep
from robomme_plus.control.planner import ScrewPlanFailure
from robomme_plus.control.rotations import rpy_xyz_to_quat_wxyz_torch

DATASET_SCREW_MAX_ATTEMPTS = 3
DATASET_RRT_MAX_ATTEMPTS = 3


class RRTPlanFailure(RuntimeError):
    """Raised when move_to_pose_with_RRTStar returns -1 (planning failed)."""


class MultiStepDemonstrationWrapper(gym.Wrapper):
    """
    Wraps DemonstrationWrapper. step(action) interprets action as
    (waypoint_p, rpy, gripper_action) total 7 dims, internally converts RPY to quat,
    executes planning via planner_denseStep, and returns last-step signals.
    """

    def __init__(self, env, gui_render=False, vis=False):
        super().__init__(env)
        self._planner = None
        self._gui_render = gui_render
        self._vis = vis
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
        )

    @staticmethod
    def _take_last_step_value(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 0 or value.ndim == 0:
                return value
            return value.reshape(-1)[-1]
        if isinstance(value, np.ndarray):
            if value.size == 0 or value.ndim == 0:
                return value
            return value.reshape(-1)[-1]
        if isinstance(value, (list, tuple)):
            return value[-1] if value else value
        return value

    def _get_planner(self):
        trial = getattr(self.env.unwrapped, "_execution_trial", 1)
        if getattr(self, "_planner_trial", trial) != trial:
            self._planner = None
        self._planner_trial = trial
        if self._planner is not None:
            return self._planner
        from robomme_plus.control.planner import (
            FailAwarePandaArmMotionPlanningSolver,
            FailAwarePandaStickMotionPlanningSolver,
        )

        env_id = getattr(self.env.unwrapped, "task_name", self.env.unwrapped.spec.id)
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

    def _at_pose(self, pose, position_tolerance=1e-5):
        current = self.env.unwrapped.agent.tcp.pose
        p = current.p.detach().cpu().numpy().reshape(-1)
        q = current.q.detach().cpu().numpy().reshape(-1)
        return (
            np.linalg.norm(p - pose.p) < position_tolerance
            and min(np.linalg.norm(q - pose.q), np.linalg.norm(q + pose.q)) < 1e-4
        )

    def _hold_step(self):
        qpos = (
            self.env.unwrapped.agent.robot.get_qpos().detach().cpu().numpy().reshape(-1)
        )
        # Finger joint positions are metres; the action uses a normalized command.
        # Retain the planner's open/closed command just as a movement does.
        action = (
            qpos[:7]
            if len(qpos) == 7
            else np.r_[qpos[:7], self._get_planner().gripper_state]
        )
        return self.env.step(action)

    def step(self, action):
        """Execute waypoint step and return last-step signals for reward/terminated/truncated."""
        action = np.asarray(action, dtype=np.float64).flatten()
        if not np.isfinite(action).all():
            raise ValueError("Waypoint action values must be finite")
        if action.size < 7:
            raise ValueError(f"action must have at least 7 elements, got {action.size}")
        waypoint_p = action[:3]
        rpy = action[3:6]
        gripper_action = float(action[6])

        # RPY → quat (wxyz) for sapien.Pose
        rpy_t = torch.as_tensor(rpy, dtype=torch.float64)
        waypoint_q = rpy_xyz_to_quat_wxyz_torch(rpy_t).numpy()

        pose = sapien.Pose(p=waypoint_p, q=waypoint_q)
        planner = self._get_planner()
        is_stick_env = getattr(
            self.env.unwrapped, "task_name", self.env.unwrapped.spec.id
        ) in ("PatternLock", "RouteStick")

        collected_steps = []
        if self._at_pose(pose):
            # An already-reached waypoint is a physical hold, not an empty plan.
            move_steps = [self._hold_step()]
        else:
            move_steps = -1
            for attempt in range(1, DATASET_SCREW_MAX_ATTEMPTS + 1):
                try:
                    result = planner_denseStep._collect_dense_steps(
                        planner, lambda: planner.move_to_pose_with_screw(pose)
                    )
                except ScrewPlanFailure as exc:
                    print(
                        f"[MultiStep] screw planning failed (attempt {attempt}/{DATASET_SCREW_MAX_ATTEMPTS}): {exc}"
                    )
                    continue

                if isinstance(result, int) and result == -1:
                    print(
                        f"[MultiStep] screw planning returned -1 (attempt {attempt}/{DATASET_SCREW_MAX_ATTEMPTS})"
                    )
                    continue

                move_steps = result
                break

            if move_steps == -1:
                print(
                    f"[MultiStep] screw planning exhausted; fallback to RRT* (max {DATASET_RRT_MAX_ATTEMPTS} attempts)"
                )
                for attempt in range(1, DATASET_RRT_MAX_ATTEMPTS + 1):
                    try:
                        result = planner_denseStep._collect_dense_steps(
                            planner, lambda: planner.move_to_pose_with_RRTStar(pose)
                        )
                    except Exception as exc:
                        print(
                            f"[MultiStep] RRT* planning failed (attempt {attempt}/{DATASET_RRT_MAX_ATTEMPTS}): {exc}"
                        )
                        continue

                    if isinstance(result, int) and result == -1:
                        print(
                            f"[MultiStep] RRT* planning returned -1 (attempt {attempt}/{DATASET_RRT_MAX_ATTEMPTS})"
                        )
                        continue

                    move_steps = result
                    break

            if move_steps == -1:
                if self._at_pose(pose, position_tolerance=1.001e-3):
                    # Preserve precise moves when planning succeeds, but do
                    # not penalize a failed sub-millimetre hold adjustment.
                    move_steps = [self._hold_step()]
                else:
                    raise RRTPlanFailure("Both screw and RRTStar planning exhausted.")
        collected_steps.extend(move_steps)

        # PatternLock/RouteStick force skip gripper action (even if planner object has method with same name).
        ended = bool(
            collected_steps and planner_denseStep.interrupts_action(collected_steps[-1])
        )
        if not is_stick_env and not ended:
            if gripper_action == -1:
                if hasattr(planner, "close_gripper"):
                    result = planner_denseStep.close_gripper(planner)
                    if result != -1:
                        collected_steps.extend(
                            planner_denseStep._output_to_steps(result)
                        )
            elif gripper_action == 1:
                if hasattr(planner, "open_gripper"):
                    result = planner_denseStep.open_gripper(planner)
                    if result != -1:
                        collected_steps.extend(
                            planner_denseStep._output_to_steps(result)
                        )

        if not collected_steps:
            # Successful empty plans still need an observation and a clock tick.
            collected_steps.append(self._hold_step())

        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = (
            planner_denseStep.to_step_batch(collected_steps)
        )
        info_flat = planner_denseStep.flatten_info(info_batch)
        return (
            obs_batch,
            self._take_last_step_value(reward_batch),
            self._take_last_step_value(terminated_batch),
            self._take_last_step_value(truncated_batch),
            info_flat,
        )

    def reset(self, **kwargs):
        self._planner = None
        return self.env.reset(**kwargs)

    def close(self):
        self._planner = None
        return self.env.close()
