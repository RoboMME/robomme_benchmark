from .base import SYSTEM_PROMPT_with_DEMO, SYSTEM_PROMPT_ORACLE_PLANNER

notes = """
Notes:
The robot base frame is the origin. The front camera is looking at the robot. 'move left' means gripper moves to the right in the front camera. 'move right' means gripper moves to the left in the front camera.
"""


subgoals = """
- move to the nearest [left/right] target by circling around the stick [clockwise/counterclockwise]
"""

example = """
Given the trajectory of the robot in the pre-recorded video, the task subgoal sequence could be:
1. move to the nearest left target by circling around the stick clockwise
2. move to the nearest right target by circling around the stick counterclockwise
3. move to the nearest left target by circling around the stick clockwise
4. move to the nearest right target by circling around the stick counterclockwise
"""

RouteStick_SYSTEM_PROMPT = SYSTEM_PROMPT_with_DEMO.format(
    subgoals=subgoals,
    example=example + notes,
)


RouteStick_SYSTEM_PROMPT_GROUNDED = RouteStick_SYSTEM_PROMPT

subgoals_oracle_planner = """
- {"action": "move to the nearest [left/right] target by circling around the stick [clockwise/counterclockwise]", "point": null}
"""

example_oracle_planner = """
Given the trajectory of the robot in the pre-recorded video, the task subgoal sequence could be:
1. {"action": "move to the nearest left target by circling around the stick clockwise", "point": null}
2. {"action": "move to the nearest right target by circling around the stick counterclockwise", "point": null}
3. {"action": "move to the nearest left target by circling around the stick clockwise", "point": null}
4. {"action": "move to the nearest right target by circling around the stick counterclockwise", "point": null}
"""

RouteStick_SYSTEM_PROMPT_ORACLE_PLANNER = SYSTEM_PROMPT_ORACLE_PLANNER.format(
    subgoals=subgoals_oracle_planner,
    example=example_oracle_planner + notes,
)