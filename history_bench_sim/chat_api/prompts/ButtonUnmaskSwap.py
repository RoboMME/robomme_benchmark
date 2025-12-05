from .base import SYSTEM_PROMPT, GROUNDED_SUBGOAL_INFORMATION, SYSTEM_PROMPT_ORACLE_PLANNER

notes = """
Notes:
1. You need to press the button first, which you are doing this, be sure to remember where those cubes are, and then they will be hidden behind the white container boxes.
2. You need to find out which color cube is being hidden, and then pick them up during the execution.
3. You need to continue press the button until the this task subgoal is completed. 
4. The container will be swapped during the execution, so you need to memorize the location of the target cubes, and then pick them up during the execution.
"""

subgoals = """
- press the first button
- press the second button
- pick up the container that hides the [red/blue/green] cube
- put down the container
"""

example = """
If the task goal is pick up the container hiding blue and red cube, a typical task subgoal sequence for this task could be:
1. press the first button
2. press the second button
3. pick up the container that hides the blue cube
4. put down the container
5. pick up the container that hides the red cube
6. put down the container
"""


ButtonUnmaskSwap_SYSTEM_PROMPT = SYSTEM_PROMPT.format(
    subgoals=subgoals,
    example=example + notes,
)


subgoals_grounded = """
- press the first button at <y, x>
- press the second button at <y, x>
- pick up the container at <y, x> that hides the [red/blue/green] cube
- put down the container
"""

example_grounded = """
If the task goal is pick up the container hiding blue and red cube, a typical task subgoal sequence for this task could be:
1. press the first button at <247, 303>
2. press the second button at <428, 345>
2. pick up the container at <356, 499> that hides the blue cube
3. put down the container
4. pick up the container at <156, 483> that hides the red cube
5. put down the container
"""

ButtonUnmaskSwap_SYSTEM_PROMPT_GROUNDED = SYSTEM_PROMPT.format(
    subgoals=subgoals_grounded,
    example=example_grounded + notes,
) + GROUNDED_SUBGOAL_INFORMATION


subgoals_oracle_planner = """
- {"action": "press the first button", "point": null}
- {"action": "press the second button", "point": null}
- {"action": "pick up the container", "point": [y, x]}
- {"action": "put down the container", "point": null}
"""

example_oracle_planner = """
If the task goal is pick up the container hiding blue and red cube, a typical task subgoal sequence for this task could be:
1. {"action": "press the first button", "point": null}
2. {"action": "press the second button", "point": null}
3. {"action": "pick up the container", "point": [356, 499]}
4. {"action": "put down the container", "point": null}
5. {"action": "pick up the container", "point": [156, 483]}
6. {"action": "put down the container", "point": null}
"""

ButtonUnmaskSwap_SYSTEM_PROMPT_ORACLE_PLANNER = SYSTEM_PROMPT_ORACLE_PLANNER.format(
    subgoals=subgoals_oracle_planner,
    example=example_oracle_planner + notes,
)