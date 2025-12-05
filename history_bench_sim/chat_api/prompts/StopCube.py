from historybench.HistoryBench_env.util import subgoal_language
from .base import SYSTEM_PROMPT, GROUNDED_SUBGOAL_INFORMATION, SYSTEM_PROMPT_ORACLE_PLANNER

notes = """
Notes:
1. You need to recon the speed of the cube to decide when to remain static and when to press the button to stop the cube.
2. You need to count the number of times the cube has reached the target to decide when to press the button to stop the cube.
3. The video clips I give you are continuous, so do not re-count the number of times if cube has just reached the target for both previous and current video clips.
4. If you think the cube is about to reach the target for the expected number of times, you should press the button to stop the cube directly.
5. The target is a purple disk.
"""

subgoals = """
- move to the top of the button to prepare
- remain static
- press the button to stop the cube on the target
"""

example = """
A typical task subgoal sequence for this task could be:
1. move to the top of the button to prepare
2. remain static
3. press the button to stop the cube on the target
"""


StopCube_SYSTEM_PROMPT = SYSTEM_PROMPT.format(
    subgoals=subgoals,
    example=example+notes,
)


subgoals_grounded = """
- move to the top of the button at <y, x> to prepare
- remain static
- press the button to stop the cube on the target at <y, x>
"""

example_grounded = """
If the task goal is to stop the cube on the target, a typical task subgoal sequence for this task could be:
1. move to the top of the button at <356, 499> to prepare
2. remain static
3. press the button to stop the cube on the target at <472, 700>
"""

StopCube_SYSTEM_PROMPT_GROUNDED = SYSTEM_PROMPT.format(
    subgoals=subgoals_grounded,
    example=example_grounded+notes,
) + GROUNDED_SUBGOAL_INFORMATION


subgoals_oracle_planner = """
- {"action": "move to the top of the button to prepare", "point": null}
- {"action": "remain static", "point": null}
- {"action": "press button to stop the cube", "point": null}
"""

example_oracle_planner = """
If the task goal is to stop the cube on the target, a typical task subgoal sequence for this task could be:
1. {"action": "move to the top of the button to prepare", "point": null}
2. {"action": "remain static", "point": null}
3. {"action": "press button to stop the cube", "point": null}
"""

StopCube_SYSTEM_PROMPT_ORACLE_PLANNER = SYSTEM_PROMPT_ORACLE_PLANNER.format(
    subgoals=subgoals_oracle_planner,
    example=example_oracle_planner+notes,
)