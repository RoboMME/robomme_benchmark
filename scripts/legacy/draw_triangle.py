import sapien
import gymnasium as gym


from mani_skill.envs.tasks import DrawTriangleEnv
from mani_skill.examples.motionplanning.panda.motionplanner_stick import \
    PandaStickMotionPlanningSolver


def solve(env: DrawTriangleEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaStickMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.3,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    rot = list(env.agent.tcp.pose.get_q()[0].cpu().numpy())

    # -------------------------------------------------------------------------- #
    # Move to first vertex
    # -------------------------------------------------------------------------- #

    reach_pose = sapien.Pose(p=list(env.vertices[0, 0].numpy()), q=rot)
    res = planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Move to second vertex
    # -------------------------------------------------------------------------- #

    reach_pose = sapien.Pose(p=list(env.vertices[0, 1]), q=rot)
    res = planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Move to third vertex
    # -------------------------------------------------------------------------- #

    reach_pose = sapien.Pose(p=list(env.vertices[0, 2]), q=rot)
    res = planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Move back to first vertex
    # -------------------------------------------------------------------------- #

    reach_pose = sapien.Pose(p=list(env.vertices[0, 0]), q=rot)
    res = planner.move_to_pose_with_screw(reach_pose)

    planner.close()
    return res

def main():
    env: DrawTriangleEnv = gym.make(
        "DrawTriangle-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
      #  reward_mode="None",
    )
    for seed in range(100):
        res = solve(env, seed=seed, debug=False, vis=True)
        print(res[-1])
    env.close()




if __name__ == "__main__":
    main()
