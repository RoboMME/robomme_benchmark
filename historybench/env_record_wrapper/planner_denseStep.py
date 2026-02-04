"""
Dense-step planner: wrappers around the ManiSkill planner that collect every
env.step result during move_to_pose_with_RRTStar / move_to_pose_with_screw /
close_gripper / open_gripper and return 5 lists (obs_list, reward_list,
terminated_list, truncated_list, info_list) instead of only the last step.
On planning failure, return -1.
"""


def _run_with_dense_collection(planner, fn):
    """
    Run fn() while intercepting planner.env.step to collect every (obs, reward,
    terminated, truncated, info). Return 5 lists built from collected steps, or -1
    if fn returns -1.
    """
    collected = []
    original_step = planner.env.step

    def _step(action):
        out = original_step(action)
        collected.append(out)
        return out

    planner.env.step = _step
    try:
        result = fn()
        if result == -1:
            return -1
        obs_list = [x[0] for x in collected]
        reward_list = [x[1] for x in collected]
        terminated_list = [x[2] for x in collected]
        truncated_list = [x[3] for x in collected]
        info_list = [x[4] for x in collected]
        return (obs_list, reward_list, terminated_list, truncated_list, info_list)
    finally:
        planner.env.step = original_step


def move_to_pose_with_RRTStar(planner, pose):
    """
    Run planner.move_to_pose_with_RRTStar(pose) and return 5 lists
    (obs_list, reward_list, terminated_list, truncated_list, info_list), one entry
    per internal env.step. Return -1 on planning failure.
    """
    return _run_with_dense_collection(
        planner, lambda: planner.move_to_pose_with_RRTStar(pose)
    )


def move_to_pose_with_screw(planner, pose):
    """
    Run planner.move_to_pose_with_screw(pose) and return 5 lists
    (obs_list, reward_list, terminated_list, truncated_list, info_list), one entry
    per internal env.step. Return -1 on planning failure.
    """
    return _run_with_dense_collection(
        planner, lambda: planner.move_to_pose_with_screw(pose)
    )


def close_gripper(planner):
    """
    Run planner.close_gripper() and return 5 lists
    (obs_list, reward_list, terminated_list, truncated_list, info_list), one entry
    per internal env.step. Return -1 on failure.
    """
    return _run_with_dense_collection(planner, lambda: planner.close_gripper())


def open_gripper(planner):
    """
    Run planner.open_gripper() and return 5 lists
    (obs_list, reward_list, terminated_list, truncated_list, info_list), one entry
    per internal env.step. Return -1 on failure.
    """
    return _run_with_dense_collection(planner, lambda: planner.open_gripper())
