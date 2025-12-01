import os
import sys

import gymnasium as gym
import numpy as np
import sapien
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ROOT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from historybench.HistoryBench_env import *
# from util import *
from historybench.HistoryBench_env.util.planner import (
    grasp_and_lift_peg_side,
    insert_peg,
)
from mani_skill.envs.tasks import PegInsertionSideEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
    quaternion_multiply,
)




def main():
    for historybench_seed in range(100):
        env = gym.make(
            "PickPeg",
            obs_mode="none",
            control_mode="pd_joint_pos",
            render_mode="human",
            reward_mode="dense",
            HistoryBench_seed=historybench_seed,
        )
        env.reset()
        try:
            res = solve(env, debug=False, vis=True)

        finally:
            env.close()


def solve(env: PegInsertionSideEnv, debug=False, vis=False):
    env.reset()
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    FINGER_LENGTH = 0.025
    env = env.unwrapped

    peg_init_pose = env.peg.pose

    grasp_and_lift_peg_side(env, planner, env.peg_tail)
    insert_peg(env, planner, env.current_grasp_pose_p,env.current_grasp_pose_q,env.peg_init_pose,direction=-1)

    planner.close()
    return None


if __name__ == "__main__":
    main()
