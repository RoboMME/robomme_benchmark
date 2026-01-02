from .base import SYSTEM_PROMPT_with_DEMO, GROUNDED_SUBGOAL_INFORMATION, SYSTEM_PROMPT_ORACLE_PLANNER


# notes = """
# Notes:
# 1. You need to reconstruct the whole trajectory of the robot by watching the pre-recorded video. then issue the subgoal sequence during execution

# 2. When the subgoal target (white grey disk) is reached, the target will turn red.
# """

notes = """
Notes:
1. The robot base frame is the origin. The video or image I give to you is the front camera is looking at the robot. 'move forward' means gripper moves away from the robot base (appears moving backward in the front camera). 'move backward' means gripper moves toward the robot base (appears moving forward in the front camera). 'move left' means gripper moves to the right in the front camera. 'move right' means gripper moves to the left in the front camera. Be sure to understand this transformation.

2. You need to reconstruct the whole trajectory of the robot by watching the pre-recorded video. then issue the subgoal sequence during execution

3. When the subgoal target (white grey disk) is reached, the target will turn red.

"""


subgoals = """
- move forward
- move left
- move right
- move backward
- move forward-left
- move forward-right
- move backward-left
- move backward-right
"""

example = """
Given the trajectory of the robot in the pre-recorded video, the task subgoal sequence could be:
1. move forward
2. move left
3. move forward-left
4. move forward-right
"""

PatternLock_SYSTEM_PROMPT = SYSTEM_PROMPT_with_DEMO.format(
    subgoals=subgoals,
    example=example + notes,
)


PatternLock_SYSTEM_PROMPT_GROUNDED = PatternLock_SYSTEM_PROMPT


subgoals_oracle_planner = """
- {"action": "move forward", "point": null}
- {"action": "move left", "point": null}
- {"action": "move right", "point": null}
- {"action": "move backward", "point": null}
- {"action": "move forward-left", "point": null}
- {"action": "move forward-right", "point": null}
- {"action": "move backward-left", "point": null}
- {"action": "move backward-right", "point": null}
"""

example_oracle_planner = """
Given the trajectory of the robot in the pre-recorded video, the task subgoal sequence could be:
1. {"action": "move forward", "point": null}
2. {"action": "move left", "point": null}
3. {"action": "move forward-left", "point": null}
4. {"action": "move forward-right", "point": null}
5. {"action": "move backward-left", "point": null}
6. {"action": "move backward-right", "point": null}


This task do not need any grounding information.
"""

PatternLock_SYSTEM_PROMPT_ORACLE_PLANNER = SYSTEM_PROMPT_ORACLE_PLANNER.format(
    subgoals=subgoals_oracle_planner,
    example=example_oracle_planner + notes,
)