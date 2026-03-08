"""
Note content management module
Manages Coordinate Information and Task Hint content
"""


def get_coordinate_information():
    """
    Get coordinate information content (Note 1)

    Returns:
        str: Coordinate information in Markdown format
    """
    return """
The coordinate system differs based on the camera perspective.

In the base camera view, the lateral axis is inverted relative to the robot: the right side of the camera frame corresponds to the robot's left side, and vice versa.

Conversely, the wrist camera view is fully aligned with the robot's motion frame. Directional movements are consistent, meaning 'right' in the camera view corresponds to the robot's right, and 'forward' implies forward movement

select "Ground Truth Action" if you need help, and "Execute" it
"""


def get_task_hint(env_id):
    """
    Get task hint content based on environment ID (Note 2)

    Args:
        env_id (str): Environment ID, e.g., "VideoPlaceOrder", "PickXtimes", etc.

    Returns:
        str: Task hint in Markdown format
    """
    # Return different hints based on env_id
    # Order follows solve_3.5_parallel_multi_loop_v4.py DEFAULT_ENVS list
    hints = {
        "PickXtimes": """\
For example, to pick up red cubes twice, a typical action sequence is:
    1. pick up the cube (click to select the correct color)
    2. place it onto the target.
    3. pick up the cube (click to select the correct color)
    4. place it onto the target.
    5. press the button to stop.

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "StopCube": """\
For example, to stop the cube on the target on its third visit, a typical action sequence is:
    1. move above the button to prepare.
    2. remain static (count how many times the cube passes the target, may select "remain static" multiple times).
    3. when the cube is about to reach the target for the third time, press the button to stop. You need to anticipate the time duration of pressing.

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "SwingXtimes": """\
For example, to swing right-to-left twice, a typical action sequence is:
    1. pick up the cube (click to select the correct color).
    2. move to the right target, then to the left target (click to select each).
    3. move to the right target again, then to the left target again (click to select each).
    4. put the cube on the table and press the button to stop.

Spatial directions (left, right) follow the robot base frame.
Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "BinFill": """\
Suppose the task is to pick two red cubes and put them into the bin, a typical sequence:
    1. pick up a red cube (click to select), then put it in the bin.
    2. pick up another red cube (click to select), then put it in the bin.
    3. press the button to stop.

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "VideoUnmaskSwap": """\
Watch the video where the cubes are hidden by containers. Memorize each cube's color. Track the swap of containers.
Typical sequence:
    1. pick up a container (click to select), then drop it.
    Repeat for a second container if the goal is to find two cubes.

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "VideoUnmask": """\
Watch the video where the cubes are hidden by containers. Memorize each cube's color.
Typical sequence:
    1. pick up a container (click to select), then put it down.
    Repeat for a second container if the goal is to find two cubes.

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "ButtonUnmaskSwap": """\
Press the buttons. While doing so, cubes are hidden in containers. Memorize each cube's color. Track the swap of containers.
Typical sequence:
    1. press the first button, then the second.
    2. pick up a container (click to select), then drop it.
    Repeat for a second container if the goal is to find two cubes.

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "ButtonUnmask": """\
Press the button first. While doing so, cubes are hidden in containers. Memorize each cube's color.
Typical sequence:
    1. press the button.
    2. pick up a container (click to select), then drop it.
    Repeat for a second container if the goal is to find two cubes.

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "VideoRepick": """\
Remember which cube was picked in the video, then pick it again. Cube positions may be swapped.
Typical sequence:
    1. pick up the correct cube (click to select by color)
    2. put it on the table.
    3. repeat step 1-2 for the required number of times.
    4. press the button to stop.

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "VideoPlaceButton": """\
The video shows a robot placing a cube on different targets and pressing the button in sequence. Targets may change positions.
Typical sequence:
    1. pick up the correct cube (click to select)
    2. drop it onto the target (click to select target).

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "VideoPlaceOrder": """\
The video shows a robot placing a cube on different targets and pressing the button in sequence. Targets may change positions.
Typical sequence:
    1. pick up the correct cube (click to select)
    2. drop it onto the target (click to select target).

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "PickHighlight": """\
While pressing the button, some cubes are highlighted with white discs. Remember them.
Typical sequence:
    1. press the button.
    2. pick up each highlighted cube (click to select)
    3. place the cube onto the table.
Repeat for all highlighted cubes.

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "InsertPeg": """\
The video shows a robot inserting a peg into a hole. The peg has two colored parts. Pick the correct part and insert from the correct side.
Typical sequence:
    1. pick up the peg (click to select correct peg and part).
    2. insert it into the hole on the left.

Spatial directions (left, right) follow the robot base frame.
Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "MoveCube": """\
The video shows a robot moving a cube to a target by either (1) pick-and-place, (2) pushing with the gripper, or (3) hooking with a peg.
Remember which method was used and reproduce it.

Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "PatternLock": """\
The video shows a robot drawing a pattern. Remember the movements and reproduce them faithfully.

Spatial directions (left, right, forward, backward) follow the robot base frame.
Select "Ground Truth Action" if you need help, then "Execute" it.
""",

        "RouteStick": """\
The video shows a robot moving between targets by circling around a stick (clockwise or counter-clockwise; move left or right around the stick).
Remember the action sequence and reproduce it.

Spatial directions (left, right) follow the robot base frame.
Select "Ground Truth Action" if you need help, then "Execute" it.
""",

    }

    # Normalize env_id to handle case-insensitive matching
    # First try direct lookup
    if env_id in hints:
        return hints[env_id]

    # Create a mapping from lowercase to standard format for case-insensitive lookup
    # This handles cases where env_id might be passed as lowercase (e.g., "pickxtimes", "binfill")
    env_id_lower_to_standard = {
        key.lower(): key for key in hints.keys()
    }

    # Try case-insensitive lookup
    if env_id:
        env_id_lower = env_id.lower()
        if env_id_lower in env_id_lower_to_standard:
            standard_key = env_id_lower_to_standard[env_id_lower]
            return hints[standard_key]

    # Return default hint if not found
    return """///

select "Ground Truth Action" if you need help, and "Execute" it
"""
