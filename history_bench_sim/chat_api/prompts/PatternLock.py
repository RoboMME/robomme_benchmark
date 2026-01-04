from .base import SYSTEM_PROMPT_PATTERNLOCK_ORACLE


notes = """
Notes:
1. The robot base frame is the origin. The video or image I give to you is the front camera is looking at the robot. 
'move forward' means gripper moves away from the robot base (appears moving backward in the front camera). 
'move backward' means gripper moves toward the robot base (appears moving forward in the front camera). 
'move left' means gripper moves to the right in the front camera. 
'move right' means gripper moves to the left in the front camera. 
Be sure to understand this transformation.

2. You need to reconstruct the whole trajectory of the robot by watching the pre-recorded video. then issue the subgoal sequence during execution.

3. Each subgoal corresponds to reaching a single target. When a subgoal target (the white and gray disk) is reached, it will turn red. One subgoal equals reaching one target. Be sure to clearly separate, count, and order all subgoals correctly.

4. When outputing the subgoal sequence, you need to first think about how many subgoals are there, then output the subgoal sequence accordingly. Be sure to follow this reasoning process.

5. There could not be more than 10 subgoals! Be sure to follow this rule.
"""




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
Given the trajectory of the robot in the pre-recorded video, The robot has travelled to six diffrent white and gray disks, thus there are six subgoals.
The task subgoal sequence could be:
1. {"action": "move forward", "point": null}
2. {"action": "move left", "point": null}
3. {"action": "move forward-left", "point": null}
4. {"action": "move forward-right", "point": null}
5. {"action": "move backward-left", "point": null}
6. {"action": "move backward-right", "point": null}


This task do not need any grounding information.
"""

PatternLock_SYSTEM_PROMPT_ORACLE_PLANNER = SYSTEM_PROMPT_PATTERNLOCK_ORACLE.format(
    subgoals=subgoals_oracle_planner,
    example=example_oracle_planner + notes,
)