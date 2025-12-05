from .base import SYSTEM_PROMPT_DYNAMIC_CHANGE, GROUNDED_SUBGOAL_INFORMATION, SYSTEM_PROMPT_DYNAMIC_CHANGE_ORACLE_PLANNER

notes = """
Notes:
1. remember to finish the press the button (the gripper approachs the button, close the gripper and press the button, and then move up and away from the button, then open the gripper), then move to the next subgoal.
2. If there is only one highlighted cube, you dont need to add first/second/third those words.
"""


PICK_HIGHLIGHT_IMAGE_TEXT_QUERY = "The task goal is '{task_goal}'. Given this first input image, please describe the first subgoal, usually it is pressing the button, and there are not enough information for you to know the which are the highlighted cubes"

PICK_HIGHLIGHT_VIDEO_TEXT_QUERY = "Given the current video clip of the robot execution, if there are any cubes that has been highlighted with white area in the table, count the total number of highlighted cubes and decompose the task goal into a sequence of following subgoals, and then predict the next subgoal"




subgoals = """
- pick up the [first/second/third/...] highlighted cube, which is [red/blue/green]
- place the cube onto the table
- press the button
"""

example = """
A typical task subgoal sequence for this task could be:
1. press the button
2. pick up the first highlighted cube, which is red
3. place the cube onto the table
4. pick up the second highlighted cube, which is blue
5. place the cube onto the table
"""

PickHighlight_SYSTEM_PROMPT = SYSTEM_PROMPT_DYNAMIC_CHANGE.format(
    subgoals=subgoals,
    example=example + notes,
)


subgoals_grounded = """
- pick up the [first/second/third/...] highlighted cube at <y, x>, which is [red/blue/green]
- place the cube onto the table
- press the button at <y, x>
"""

example_grounded = """
Given the trajectory of the robot in the pre-recorded video, the task subgoal sequence could be:
1. press the button at <180, 376>
2. pick up the first highlighted cube at <356, 499>, which is red 
3. place the cube onto the table
4. pick up the second highlighted cube at <156, 483>, which is blue 
5. place the cube onto the table
"""

PickHighlight_SYSTEM_PROMPT_GROUNDED = SYSTEM_PROMPT_DYNAMIC_CHANGE.format(
    subgoals=subgoals_grounded,
    example=example_grounded + notes,
) + GROUNDED_SUBGOAL_INFORMATION


subgoals_oracle_planner = """
- {"action": "pick up the highlighted cube", "point": [y, x]}
- {"action": "place the cube onto the table", "point": null}
- {"action": "press button", "point": null}
"""

example_oracle_planner = """
Given the trajectory of the robot in the pre-recorded video, the task subgoal sequence could be:
1. {"action": "press button", "point": null}
2. {"action": "pick up the highlighted cube", "point": [356, 499]}
3. {"action": "place the cube onto the table", "point": null}
4. {"action": "pick up the highlighted cube", "point": [156, 483]}
5. {"action": "place the cube onto the table", "point": null}
"""

PickHighlight_SYSTEM_PROMPT_ORACLE_PLANNER = SYSTEM_PROMPT_DYNAMIC_CHANGE_ORACLE_PLANNER.format(
    subgoals=subgoals_oracle_planner,
    example=example_oracle_planner + notes,
)