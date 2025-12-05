from .base import SYSTEM_PROMPT, GROUNDED_SUBGOAL_INFORMATION, SYSTEM_PROMPT_ORACLE_PLANNER



notes = """
Notes:
1. The bin is a box with black holes on the top, it will suck all the cubes that have been put into it, so you need to count by yourself how many cubes have been put into the bin.
"""

subgoals = """
- pick up the [first/second/third/...][red/blue/green] cube
- put it into the bin
- press the button
"""

example = """
If the task goal is to 'put two red cubes into the bin', a typical task subgoal sequence for this task could be: 
1. pick up the first red cube 
2. put it into the bin 
3. pick up the second red cube 
4. put it into the bin 
5. press the button
"""



BinFill_SYSTEM_PROMPT = SYSTEM_PROMPT.format(
    subgoals=subgoals,
    example=example + notes
)




subgoals_grounded = """
- pick up the [first/second/third/...][red/blue/green] cube at <y, x>
- put it into the bin at <y, x>
- press the button at <y, x>
"""

example_grounded = """
If the task goal is to 'put two red cubes into the bin', a typical task subgoal sequence for this task could be: 
1. pick up the first red cube at <356, 499>
2. put it into the bin at <472, 700>
3. pick up the second red cube at <156, 483>
4. put it into the bin at <472, 700>
5. press the button at <180, 376>
"""


BinFill_SYSTEM_PROMPT_GROUNDED = SYSTEM_PROMPT.format(
    subgoals=subgoals_grounded,
    example=example_grounded + notes,
) + GROUNDED_SUBGOAL_INFORMATION


subgoals_oracle_planner = """
- {"action": "pick up the cube", "point": [y, x]}
- {"action": "put it into the bin", "point": null}
- {"action": "press the button", "point": null}
"""

example_oracle_planner = """
If the task goal is to 'put two red cubes into the bin', a typical task subgoal sequence for this task could be: 
1. {"action": "pick up the cube", "point": [356, 499]}
2. {"action": "put it into the bin", "point": null}
3. {"action": "pick up the cube", "point": [156, 483]}
4. {"action": "put it into the bin", "point": null}
5. {"action": "press the button", "point": null}
"""

BinFill_SYSTEM_PROMPT_ORACLE_PLANNER = SYSTEM_PROMPT_ORACLE_PLANNER.format(
    subgoals=subgoals_oracle_planner,
    example=example_oracle_planner + notes,
)