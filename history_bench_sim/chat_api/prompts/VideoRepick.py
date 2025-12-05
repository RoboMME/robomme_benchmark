from .base import SYSTEM_PROMPT, GROUNDED_SUBGOAL_INFORMATION, SYSTEM_PROMPT_ORACLE_PLANNER

subgoals = """
- pick up the correct cube for the [first/second/third/...] time
- put it down
- press the button to finish
"""

example = """
Given the pre-recorded video, You need to remember which cube has been picked up before, and then pick it up again.

If the task goal is the pick up the green cube two times, a typical task subgoal sequence for this task could be:
1. pick up the correct cube for the first time
2. put it down
3. pick up the correct cube for the second time
4. put it down
5. press the button to finish
"""


VideoRepick_SYSTEM_PROMPT = SYSTEM_PROMPT.format(
    subgoals=subgoals,
    example=example,
)


subgoals_grounded = """
- pick up the correct cube at <y, x> for the [first/second/third/...] time
- put it down
- press the button at <y, x> to finish
"""

example_grounded = """
Given the trajectory of the robot in the pre-recorded video, the task subgoal sequence should alwasy be:
1. pick up the correct cube at <356, 345> for the first time
2. put it down
3. press the button at <180, 376> to finish
"""

VideoRepick_SYSTEM_PROMPT_GROUNDED = SYSTEM_PROMPT.format(
    subgoals=subgoals_grounded,
    example=example_grounded,
) + GROUNDED_SUBGOAL_INFORMATION


subgoals_oracle_planner = """
- {"action": "pick up the cube", "point": [y, x]}
- {"action": "put it down", "point": null}
- {"action": "press the button to finish", "point": null}
"""

example_oracle_planner = """
Given the trajectory of the robot in the pre-recorded video, the task subgoal sequence should alwasy be:
1. {"action": "pick up the cube", "point": [356, 345]}
2. {"action": "put it down", "point": null}
3. {"action": "press the button to finish", "point": null}
"""

VideoRepick_SYSTEM_PROMPT_ORACLE_PLANNER = SYSTEM_PROMPT_ORACLE_PLANNER.format(
    subgoals=subgoals_oracle_planner,
    example=example_oracle_planner,
)