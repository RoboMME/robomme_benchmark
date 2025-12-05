from .base import SYSTEM_PROMPT_with_DEMO, GROUNDED_SUBGOAL_INFORMATION, SYSTEM_PROMPT_with_DEMO_ORACLE_PLANNER


notes = """
Notes:
1. You need to find out which color cube is being hidden in the pre-recorded video, and then pick them up during the execution.
2. You need to continue pick up the container until the this task subgoal is completed. You should memorize the location of the target cube, and then pick it up during the execution.
3. The container will be swapped during the execution, so you need to memorize the location of the target cubes, and then pick them up during the execution.
"""

subgoals = """
- pick up the container that hides the [red/blue/green] cube
- put down the container
"""

example = """
Given the trajectory of the robot in the pre-recorded video, the task subgoal sequence could be:
1. pick up the container that hides the red cube
3. put down the container
4. pick up the container that hides the blue cube
5. put down the container
"""

VideoUnmaskSwap_SYSTEM_PROMPT = SYSTEM_PROMPT_with_DEMO.format(
    subgoals=subgoals,
    example=example + notes,
)

subgoals_grounded = """
- pick up the container at <y, x> that hides the [red/blue/green] cube
- put down the container
"""

example_grounded = """
Given the trajectory of the robot in the pre-recorded video, the task subgoal sequence should alwasy be:
1. pick up the container at <356, 345> that hides the red cube
2. put down the container
"""

VideoUnmaskSwap_SYSTEM_PROMPT_GROUNDED = SYSTEM_PROMPT_with_DEMO.format(
    subgoals=subgoals_grounded,
    example=example_grounded + notes,
) + GROUNDED_SUBGOAL_INFORMATION


subgoals_oracle_planner = """
- {"action": "pick up the container", "point": [y, x]}
- {"action": "put down the container", "point": null}
""" 

example_oracle_planner = """
Given the trajectory of the robot in the pre-recorded video, the task subgoal sequence should alwasy be:
1. {"action": "pick up the container", "point": [356, 345]}
2. {"action": "put down the container", "point": null}
"""

VideoUnmaskSwap_SYSTEM_PROMPT_ORACLE_PLANNER = SYSTEM_PROMPT_with_DEMO_ORACLE_PLANNER.format(
    subgoals=subgoals_oracle_planner,
    example=example_oracle_planner + notes,
)