from .base import SYSTEM_PROMPT, GROUNDED_SUBGOAL_INFORMATION, SYSTEM_PROMPT_ORACLE_PLANNER

notes = """
Notes:
1. The target is a purple disk.
"""

subgoals = """
- pick up the [color] cube for the [first/second/third/...] time
- place the [color] cube onto the target
- press the button to stop
"""

example = """
If the task goal is the pick up the green cube two times, a typical task subgoal sequence for this task could be:
1. pick up the green cube for the first time
2. place the green cube onto the target
3. pick up the green cube for the second time
4. place the green cube onto the target
5. press the button to stop
"""


PickXtimes_SYSTEM_PROMPT = SYSTEM_PROMPT.format(
    subgoals=subgoals,
    example=example + notes,
)


subgoals_grounded = """
- pick up the [color] cube at <y, x> for the [first/second/third/...] time
- place the [color] cube onto the target at <y, x>
- press the button to stop at <y, x>
"""

example_grounded = """
If the task goal is the pick up the green cube two times, a typical task subgoal sequence for this task could be:
1. pick up the green cube at <356, 499> for the first time
2. place the green cube onto the target at <472, 700>
3. pick up the green cube at <472, 700> for the second time
4. place the green cube onto the target at <472, 700>
5. press the button to stop at <180, 376>
"""

PickXtimes_SYSTEM_PROMPT_GROUNDED = SYSTEM_PROMPT.format(
    subgoals=subgoals_grounded,
    example=example_grounded + notes,
) + GROUNDED_SUBGOAL_INFORMATION


subgoals_oracle_planner = """
- {"action": "pick up the cube", "point": [y, x]}
- {"action": "place the cube onto the target", "point": null}
- {"action": "press the button to stop", "point": null}
"""


subgoals_oracle_planner = """
- {"action": "pick up the cube", "point": [y, x]}
- {"action": "place the cube onto the target", "point": null}
- {"action": "press the button to stop", "point": null}
"""

example_oracle_planner = """
If the task goal is the pick up the green cube two times, a typical task subgoal sequence for this task could be:
1. {"action": "pick up the cube", "point": [356, 499]} 
2. {"action": "place the cube onto the target", "point": null}
3. {"action": "pick up the cube", "point": [156, 483]}
4. {"action": "place the cube onto the target", "point": null}
5. {"action": "press the button to stop", "point": null}
"""

PickXtimes_SYSTEM_PROMPT_ORACLE_PLANNER = SYSTEM_PROMPT_ORACLE_PLANNER.format(
    subgoals=subgoals_oracle_planner,
    example=example_oracle_planner + notes,
)