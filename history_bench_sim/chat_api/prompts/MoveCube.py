from .base import SYSTEM_PROMPT_with_DEMO, GROUNDED_SUBGOAL_INFORMATION, SYSTEM_PROMPT_with_DEMO_ORACLE_PLANNER

notes = """
Notes:
1. The target is a purple disk.
2. You need to fully understand the video how the cube is being moved to the target
3. Be sure to analyze and differentiate the three ways of moving the cube to the target.
4. If you think the last subgoal is not finished yet, you should continue to generate the last subgoal. 
"""

subgoals = """
- close gripper and push the cube to the target
- pick up the cube
- place the cube onto the target
- pick up the peg 
- hook the cube to the target with the peg
"""

example = """
Given the trajectory of the robot in the pre-recorded video, you need to understand how the cube is being moved to the target, there are intotal three ways:
A) push to the target with the gripper, the subgoals should be just: 
1. close gripper and push the cube to the target

B) pick & place to the target, the subgoals should be: 
1. pick up the cube 
2. place the cube onto the target

C) hook to the target with the peg, the subgoals should be: 
1. pick up the peg 
2. hook the cube to the target with the peg
"""

MoveCube_SYSTEM_PROMPT = SYSTEM_PROMPT_with_DEMO.format(
    subgoals=subgoals,
    example=example + notes,
)

subgoals_grounded = """
- pick up the cube at <y, x>
- place the cube onto the target at <y, x>
- pick up the peg at <y, x>
- hook the cube at <y, x> to the target at <y, x> with the peg
- close gripper and push the cube at <y, x> to the target at <y, x>
"""

example_grounded = """
Given the trajectory of the robot in the pre-recorded video, you need to understand how the cube is being moved to the target, there are intotal three ways:
A) push to the target with the gripper, the subgoals should be just: 
1. close gripper and push the cube at <156, 483> to the target at <472, 700>

B) pick & place to the target, the subgoals should be: 
1. pick up the cube at <156, 483>
2. place the cube onto the target at <472, 700>

C) hook to the target with the peg, the subgoals should be: 
1. pick up the peg at <356, 499>
2. hook the cube at <156, 483> to the target at <472, 700> with the peg
"""

MoveCube_SYSTEM_PROMPT_GROUNDED = SYSTEM_PROMPT_with_DEMO.format(
    subgoals=subgoals_grounded,
    example=example_grounded + notes,
) + GROUNDED_SUBGOAL_INFORMATION



subgoals_oracle_planner = """
- {"action": "pick up the cube", "point": null}
- {"action": "place the cube onto the target", "point": null}
- {"action": "pick up the peg", "point": null}
- {"action": "hook the cube to the target with the peg", "point": null}
- {"action": "close gripper and push the cube to the target", "point": null}
"""

example_oracle_planner = """
Given the trajectory of the robot in the pre-recorded video, you need to understand how the cube is being moved to the target, there are intotal three ways:
A) push to the target with the gripper, the subgoals should be just: 
1. {"action": "close gripper and push the cube to the target", "point": null}

B) pick & place to the target, the subgoals should be: 
1. {"action": "pick up the cube", "point": null}
2. {"action": "place the cube onto the target", "point": null}

C) hook to the target with the peg, the subgoals should be: 
1. {"action": "pick up the peg", "point": null}
2. {"action": "hook the cube to the target with the peg", "point": null}
"""

MoveCube_SYSTEM_PROMPT_ORACLE_PLANNER = SYSTEM_PROMPT_with_DEMO_ORACLE_PLANNER.format(
    subgoals=subgoals_oracle_planner,
    example=example_oracle_planner + notes,
)