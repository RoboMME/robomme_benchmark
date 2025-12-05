from .base import SYSTEM_PROMPT_with_DEMO, GROUNDED_SUBGOAL_INFORMATION, SYSTEM_PROMPT_ORACLE_PLANNER

notes = """
Notes:
1. You need to find out which part (near the robot or far from the robot) of the peg is being grasped from the pre-recorded video
2. You need to find out which side (left side of the robot base frame or right side of the robot base frame) of the box is the peg being inserted from the pre-recorded video. If it looks insert from right in the video, then it's actually inserting from the left side (from the robot itself).
3. Based on the pre-recorded video, generate the correct subgoal sequence for the task
4. If generate points, need to pinpoint the point at the correct end of the peg
"""

subgoals = """
- pick up the peg by grasping the [near/far] end
- insert the peg from the [left/right] side of the box
"""

example = """
Given the trajectory of the robot in the pre-recorded video, the task subgoal sequence could be:
1. pick up the peg by grasping the near end
2. insert the peg from the left side of the box
"""

InsertPeg_SYSTEM_PROMPT = SYSTEM_PROMPT_with_DEMO.format(
    subgoals=subgoals,
    example=example + notes,
)



subgoals_grounded = """
- pick up the peg by grasping the [near/far] end at <y, x>
- insert the peg from the [left/right] side of the box at <y, x>
"""

example_grounded = """
Given the trajectory of the robot in the pre-recorded video, the task subgoal sequence could be:
1. pick up the peg by grasping the near end at <356, 499>
2. insert the peg from the left side of the box at <472, 700>
"""

InsertPeg_SYSTEM_PROMPT_GROUNDED = SYSTEM_PROMPT_with_DEMO.format(
    subgoals=subgoals_grounded,
    example=example_grounded + notes,
) + GROUNDED_SUBGOAL_INFORMATION



subgoals_oracle_planner = """
- {"action": "pick up the peg by grasping one end", "point": [y, x]}
- {"action": "insert the peg from the left side", "point": null}
- {"action": "insert the peg from the right side", "point": null}
"""

example_oracle_planner = """
Given the trajectory of the robot in the pre-recorded video, the task subgoal sequence could be:
1. {"action": "pick up the peg by grasping one end", "point": [356, 499]}
2. {"action": "insert the peg from the right side", "point": null}
"""

InsertPeg_SYSTEM_PROMPT_ORACLE_PLANNER = SYSTEM_PROMPT_ORACLE_PLANNER.format(
    subgoals=subgoals_oracle_planner,
    example=example_oracle_planner + notes,
)