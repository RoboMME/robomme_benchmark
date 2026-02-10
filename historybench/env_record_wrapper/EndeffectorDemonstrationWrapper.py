"""
EndeffectorDemonstrationWrapper：外层包装器，接收 ee_pose 动作并经 IK 转成关节动作。

- 统一外部接口：action = [ee_p(3), rpy(3), gripper(1)]，共 7 维
- 内部将 RPY 转换为 quat（wxyz）后执行 IK
- PatternLock/RouteStick：内部忽略 gripper，并下传 7 维 joint action
"""
import numpy as np
import torch
import gymnasium as gym

from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from .rpy_util import rpy_xyz_to_quat_wxyz_torch


class EndeffectorDemonstrationWrapper(gym.Wrapper):
    """
    封装一个期望关节动作的环境。step(action) 接收 ee pose：
    - 统一外部输入：action = [ee_p(3), rpy(3), gripper(1)]（7 维）
    - 内部将 RPY 转换为 quat（wxyz）后执行 IK
    - PatternLock/RouteStick 兼容 len>=6 输入，并在内部忽略 gripper
    内部先做 IK 得到 joint_action，再调用内层 env.step(joint_action)，并返回：
    (obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch)。
    """

    # stick 环境内部忽略 gripper 并下传 7 维 joint action。
    _EE_POSE_7D_ENV_IDS = ("PatternLock", "RouteStick")

    def __init__(self, env):
        super().__init__(env)
        self._ee_pose_planner = None

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).flatten()
        env_spec = getattr(self.env.unwrapped, "spec", None)
        env_id = getattr(env_spec, "id", "<unknown_env>")
        no_gripper_env = env_id in self._EE_POSE_7D_ENV_IDS

        if no_gripper_env:
            if action.size < 6:
                raise ValueError(
                    f"[{env_id}] action must have at least 6 elements (ee_p, rpy) for no-gripper env, got {action.size}"
                )
        elif action.size < 7:
            raise ValueError(
                f"[{env_id}] action must have at least 7 elements (ee_p, rpy, gripper), got {action.size}"
            )

        ee_p = action[:3]
        rpy = action[3:6]
        gripper = None if no_gripper_env else float(action[6])

        # RPY → quat (wxyz)
        rpy_t = torch.as_tensor(rpy, dtype=torch.float64)
        ee_q = rpy_xyz_to_quat_wxyz_torch(rpy_t).numpy()

        if self._ee_pose_planner is None:
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
            raise RuntimeError(
                f"ee_pose step: IK failed (status={ik_status}, num_solutions={len(ik_solutions)}), "
                f"goal_base={goal_base.tolist()}, current_qpos={current_qpos.tolist()}"
            )
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
