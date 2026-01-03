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
        "PickXtimes": """
       Typical action selection sequence for the environment:

       1.Pick up the cube.Use mouse click to select the cube with the correct color.

       2.Place the cube on the target.Repeat the action X times.

       3.Press the button to stop.
        """,
        
        "StopCube": """
        Typical action selection sequence for the environment:

       1.Move to the top of the button to prepare.
        
       2.Remain static. For each static phase, the robot will hold its position for a random duration. Repeat the action X times.

       3.Press the button to stop the cube. Observe the cube's movement via the Execution LiveStream, and press the button slightly before the cube reaches the target to account for the robot's reaction time!
        """,

        "SwingXtimes": """

        Typical action selection sequence for the environment:

       1.Pick up the cube.Use mouse click to select the cube with the correct color.

       2.Swing it on the right side target,then swing it on the left side target, repeating this action X times. 
       The targets are the two white-grey disks. Select "Move to the top of the target" and then click the target to trigger the swing.
        
       3.Put the cube down on the table 
        
       4.Press the button to stop.

        """,
        
        "BinFill": """
        Typical action selection sequence for the environment:

       1.Pick up the cube.Use mouse click to select the cube with the correct color.

       2.Place the cube on the target.

       3. Repeat 1 and 2 for X times.

       4.Press the button to stop.
        """,
        
        "VideoUnmaskSwap": """
        Watch the video carefully. Cubes will be hidden by containers, and you need to memorize the color of the cube inside each one.

        You must also track the containers as they swap positions!

        Typical action selection sequence for the environment:

        1.Pick up the container.Use mouse click to select the container.

        2.Drop the container down.

        3. Pick up another container if needed.
        """,
        
        "VideoUnmask": """
        Watch the video carefully. Cubes will be hidden by containers, and you need to memorize the color of the cube inside each one.

        Typical action selection sequence for the environment:

        1.Pick up the container.Use mouse click to select the container.

        2.Drop the container down.

        3. Pick up another container if needed.
        """,
        
        "ButtonUnmaskSwap": 
        f"""
        Press the buttons sequentially. While the robot is pressing the buttons, cubes will be hidden inside the containers, and you need to memorize the color of the cube inside each one.

        You also need to track the containers as they swap positions!

        Typical action selection sequence for the environment:

        1.Press the first button.

        2.Press the second button.

        3.Pick up the container.Use mouse click to select the container.

        4.Drop the container down.

        5. Pick up another container if needed.
        
        """,
        
        "ButtonUnmask":"""
        Press the buttons sequentially. While the robot is pressing the buttons, cubes will be hidden inside the containers, and you need to memorize the color of the cube inside each one.

        Typical action selection sequence for the environment:

        1.Press the button.

        3.Pick up the container.Use mouse click to select the container.

        4.Drop the container down.

        5. Pick up another container if needed.
        
        """,
        
        "VideoRepick": """
        Watch the video carefully. It shows a robot picking up and putting down one particular cube ONLY ONCE.

        After the robot picking up and putting down the cube, the cubes might be swapped positions. You need to track the cubes as they swap!

        Typical action selection sequence for the environment:

        1.Pick up the cube.Use mouse click to select the cube with the correct color.

        2.Put the cube down on the table.

        3. Repeat 1 and 2 for X times.

        4.Press the button to stop.
        """,
        
        "VideoPlaceButton": 
        """
        Watch the video carefully. It shows a robot placing a cube on different targets in sequence.

        The robot will also press the button once during the action.

        Remember the order of the targets and the button press!

        After the placements, some targets may change positions. Keep track of the targets as they swap!

        Typical action selection sequence for the environment:

        1.Pick up the cube.Use mouse click to select the cube with the correct color.

        2.Put the cube down on the target. Use mouse click to select the target.

        """
        ,
        
        "VideoPlaceOrder":  """
        Watch the video carefully. It shows a robot placing a cube on different targets in sequence.

        The robot will also press the button once during the action.

        Remember the order of the targets and the button press!

        After the placements, some targets may change positions. Keep track of the targets as they swap!

        Typical action selection sequence for the environment:

        1.Pick up the cube.Use mouse click to select the cube with the correct color.

        2.Put the cube down on the target. Use mouse click to select the target.
        """,
        
        "PickHighlight": """
        Press the button. When the robots is pressing button, cubes will be highlighted with white discs.

        After pressing the buttons, you need to pick up each highlighted cube. THE ORDER OF THE CUBES IS NOT IMPORTANT.
        
        Typical action selection sequence for the environment:

        1.Press the button.

        2.Pick up the cube.Use mouse click to select the cube with the correct color.
.
        3.Put the cube down on the table. 

        4. Repeat 2 and 3 for X times with DIFFERENT CUBES.
        """,
        
        "InsertPeg": """
        Watch the video carefully. It shows a robot picking up a peg and then inserting it into a hole.

        After watching the video, you need to pick up and insert the peg into the hole exactly as shown.

        The peg consists of two parts with different colors; you need to pick up the correct part of the peg.

        Typical action selection sequence for the environment:

        1.Pick up the peg.Use mouse click to select the correct peg and the correct part of the peg.

        2.Insert the peg into the hole either on the left or right side.
        """,
        
        "MoveCube": """
        Watch the video carefully. It shows a robot moving a cube to a target using different methods.

        The robot might pick up and place the cube, push it with the gripper, or hook it using a peg.

        Remember the order of the actions and the specific method used to move the cube!

        After watching the video, you need to move the cube to the target exactly as shown.
        
        """,
        
        "PatternLock": """
        Watch the video carefully. It shows a robot tracing a pattern with a stick.

        Remember the sequence of movements and the path of the tracing!

        After watching the video, you need to trace the pattern with the stick exactly as shown.

        """,
        
        "RouteStick": """
        Watch the video carefully. It shows a robot navigating from one target to another by circling around a stick.

        The movement can be clockwise or counter-clockwise, and the stick may be on the left or right side.

        Remember the sequence of actions!

        After watching the video, you need to navigate around the sticks exactly as shown.

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
    return """///"""

