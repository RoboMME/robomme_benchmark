from .base import SYSTEM_PROMPT_with_DEMO, GROUNDED_SUBGOAL_INFORMATION, SYSTEM_PROMPT_ORACLE_PLANNER

notes = """
Notes:
1. Make sure only the picking up is finished, then place the cube onto the correct target.
2. You have to fully understand how the cube is being placed mulitiple times in the pre-recorded video, and remember where is the correct target.
3. The targets are purple disks.
"""

subgoals = """
- pick up the cube
- place the cube onto the correct target
"""

example = """
Given the trajectory of the robot in the pre-recorded video, the task subgoal sequence should alwasy be:
1. pick up the cube
3. place the cube onto the correct target
"""

VideoPlaceOrder_SYSTEM_PROMPT = SYSTEM_PROMPT_with_DEMO.format(
    subgoals=subgoals,
    example=example + notes,
)



subgoals_grounded = """
- pick up the cube at <y, x>
- place the cube onto the correct target at <y, x>
"""

example_grounded = """
Given the trajectory of the robot in the pre-recorded video, the task subgoal sequence should alwasy be:
1. pick up the cube at <356, 345>
2. place the cube onto the correct target at <244, 532>
"""

VideoPlaceOrder_SYSTEM_PROMPT_GROUNDED = SYSTEM_PROMPT_with_DEMO.format(
    subgoals=subgoals_grounded,
    example=example_grounded + notes,
) + GROUNDED_SUBGOAL_INFORMATION


subgoals_oracle_planner = """
- {"action": "pick up the cube", "point": [y, x]}
- {"action": "drop onto", "point": [y, x]}
"""

example_oracle_planner = """
Given the trajectory of the robot in the pre-recorded video, the task subgoal sequence should alwasy be:
1. {"action": "pick up the cube", "point": [356, 345]}
2. {"action": "drop onto", "point": [244, 532]}
"""

VideoPlaceOrder_SYSTEM_PROMPT_ORACLE_PLANNER = SYSTEM_PROMPT_ORACLE_PLANNER.format(
    subgoals=subgoals_oracle_planner,
    example=example_oracle_planner + notes,
)