from .base import SYSTEM_PROMPT, GROUNDED_SUBGOAL_INFORMATION, SYSTEM_PROMPT_ORACLE_PLANNER

notes = """
Notes:
1. The targets are two white grey disks.
2. When the robot has reached the target, the target will turn red.
3. The coordinate origin is the robot base frame. The right target is on the right side of the robot base frame, which appears on the left side of the camera. The left target is on the left side of the robot base frame, which appears on the right side of the camera.
"""

subgoals = """
- pick up the [color] cube
- move to the top of the right-side target for the [first/second/third/...] time
- move to the top of the left-side target for the [first/second/third/...] time
- put the [color] cube on the table
- press the button
"""

example = """
If the task goal is the swing two times, a typical task subgoal sequence for this task could be:
1. pick up the green cube
2. move to the top of the right-side target for the first time
3. move to the top of the left-side target for the first time
4. move to the top of the right-side target for the second time
5. move to the top of the left-side target for the second time
4. put the green cube on the table
5. press the button
"""


SwingXtimes_SYSTEM_PROMPT = SYSTEM_PROMPT.format(
    subgoals=subgoals,
    example=example+notes,
)


subgoals_grounded = """
- pick up the [color] cube at <y, x>
- move to the top of the right target at <y, x> for the [first/second/third/...] time
- move to the top of the left target at <y, x> for the [first/second/third/...] time
- put the [color] cube on the table
- press the button at <y, x>
"""

example_grounded = """
If the task goal is the swing two times, a typical task subgoal sequence for this task could be:
1. pick up the green cube at <36, 81>
2. move to the top of the right-side target at <356, 49> for the first time
3. move to the top of the left-side target at <352, 624> for the first time
4. move to the top of the right-side target at <356, 49> for the second time
5. move to the top of the left-side target at <352, 624> for the second time
6. put the green cube on the table
7. press the button at <180, 376>
"""

SwingXtimes_SYSTEM_PROMPT_GROUNDED = SYSTEM_PROMPT.format(
    subgoals=subgoals_grounded,
    example=example_grounded+notes,
) + GROUNDED_SUBGOAL_INFORMATION


subgoals_oracle_planner = """
- {"action": "pick up the cube", "point": [y, x]}
- {"action": "move to the top of the target", "point": [y, x]}
- {"action": "put the cube on the table", "point": null}
- {"action": "press the button", "point": null}
"""

example_oracle_planner = """
If the task goal is the swing two times, a typical task subgoal sequence for this task could be:
1. {"action": "pick up the cube", "point": [36, 81]}
2. {"action": "move to the top of the target", "point": [356, 49]}
3. {"action": "move to the top of the target", "point": [352, 624]}
4. {"action": "move to the top of the target", "point": [356, 49]}
5. {"action": "move to the top of the target", "point": [352, 624]}
6. {"action": "put the cube on the table", "point": null}
7. {"action": "press the button", "point": null}
"""

SwingXtimes_SYSTEM_PROMPT_ORACLE_PLANNER = SYSTEM_PROMPT_ORACLE_PLANNER.format(
    subgoals=subgoals_oracle_planner,
    example=example_oracle_planner+notes,
)