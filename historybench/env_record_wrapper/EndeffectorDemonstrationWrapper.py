"""
EndeffectorDemonstrationWrapper：外层包装器，接收 ee_pose 动作并经 IK 转成关节动作。

- 常规环境：action = [ee_p(3), ee_q(4), gripper(1)]，共 8 维
- PatternLock/RouteStick：action = [ee_p(3), ee_q(4)]，共 7 维（无 gripper）
"""
import numpy as np
import gymnasium as gym

from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver


class EndeffectorDemonstrationWrapper(gym.Wrapper):
    """
    封装一个期望关节动作的环境。step(action) 接收 ee pose：
    - 常规环境：action = [ee_p(3), ee_q(4), gripper(1)]（8 维）
    - PatternLock/RouteStick：action = [ee_p(3), ee_q(4)]（7 维）
    内部先做 IK 得到 joint_action，再调用内层 env.step(joint_action)，并返回：
    (obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch)。
    """

    # 与 EpisodeDatasetResolver 保持一致：这两个环境只使用 7 维 ee_pose（无 gripper）。
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
            if action.size < 7:
                raise ValueError(
                    f"[{env_id}] action must have at least 7 elements (ee_p, ee_q) for no-gripper env, got {action.size}"
                )
        elif action.size < 8:
            raise ValueError(
                f"[{env_id}] action must have at least 8 elements (ee_p, ee_q, gripper), got {action.size}"
            )

        ee_p = action[:3]
        ee_q = action[3:7]
        gripper = None if no_gripper_env else float(action[7])

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
