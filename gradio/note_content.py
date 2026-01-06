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
        "PickXtimes": """Suppose the task goal is to pick up red cubes for two times, a typical action sequence could be:  
       1. pick up the cube (use mouse click to select the cube with the correct color) 
       2. place the cube onto the target. 
       3. pick up the cube (use mouse click to select the cube with the correct color) 
       4. place the cube onto the target. 
       5. press the button to stop.  
       """,
        
        "StopCube": """Suppose the task goal is to stop the cube on the target for three times, a typical action sequence could be:     
       1. move to the top of the button to prepare    
       2. remain static  (it will execute for a fixed time duration, you need to count the times the cube has passed the target)    
       3. remain static   
       4. remain static    
       5. remain static  (Suppose you feel the cube is about to reach the target for the expected number of times, you should press the button to stop the cube directly)  
       6. press the button to stop.   
       """,

        "SwingXtimes": """Suppose the task goal is to swing the back and forth for two times, a typical action sequence could be:
       1. pick up the cube (use mouse click to select the cube with the correct color)    
       2. move to the top of the target (use mouse click to select the right-side target)  
       3. move to the top of the target (use mouse click to select the left-side target)  
       4. move to the top of the target (use mouse click to select the right-side target)  
       5. move to the top of the target (use mouse click to select the left-side target)  
       6. put the cube onto the table  
       7. press the button to stop.    
        """,
        
        "BinFill": """Suppose the task goal is to pick two red cubes in the bin, a typical action sequence could be:   
       1. pick up the cube (use mouse click to select the cube with the correct color)  
       2. put it into the bin.   
       3. pick up the cube (use mouse click to select the cube with the correct color)  
       4. put it into the bin.   
       5. press the button to stop.  
        """,
        
        "VideoUnmaskSwap": """Watch the video carefully. Cubes will be hidden by containers, and you need to memorize the color of the cube inside each one.  
        You need to track the containers since they swap positions!  
        A typical action sequence could be:  
        1. pick up the container (use mouse click to select the container)  
        2. drop the container down.  
        
        pick up another container if the task goal is to find two containers.
        """,
        
        "VideoUnmask": """Watch the video carefully. Cubes will be hidden by containers, and you need to memorize the color of the cube inside each one.  
        A typical action sequence could be:  
        1. pick up the container (use mouse click to select the container)  
        2. drop the container down.  
        
        pick up another container if the task goal is to find two containers.
        """,
        
        "ButtonUnmaskSwap": 
        """Press the buttons sequentially. While pressing the buttons, the cubes will be hidden inside the containers, and you need to memorize the color of the cube inside each one.  
        You need to track the containers since they swap positions!  
        A typical action sequence could be:  
        1. press the first button.   
        2. press the second button.   
        3. pick up the container (use mouse click to select the container)   
        4. drop the container down.   
        
        pick up another container if the task goal is to find two containers.  
        """,
        
        "ButtonUnmask":"""Press the buttons sequentially. While pressing the buttons, the cubes will be hidden inside the containers, and you need to memorize the color of the cube inside each one.   
        A typical action sequence could be:   
        1. press the button.   
        2. pick up the container (use mouse click to select the container)   
        3. drop the container down.   
        
        pick up another container if the task goal is to find two containers.  
        """,
        
        "VideoRepick": """Remember the cube that has been picked up before, and then pick it up again. The cubes might be swapped positions.    
        A typical action sequence could be:   
        1. pick up the cube (use mouse click to select the correct cube with the correct color)   
        2. put the cube down on the table.  
        (repeat 1 and 2 for the expected number of times)  
        3. press the button to stop.    
        """,
        
        "VideoPlaceButton": 
        """The video shows a robot placing a cube on different targets and pressing the button in a sequence. The targets may change positions.
        A typical action sequence could be:  
        1. pick up the cube (use mouse click to select the correct cube with the correct color)  
        2. put the cube down on the target (use mouse click to select the target)  
        """
        ,
        
        "VideoPlaceOrder":  """The video shows a robot placing a cube on different targets and pressing the button in a sequence. The targets may change positions.
        A typical action sequence could be:  
        1. pick up the cube (use mouse click to select the correct cube with the correct color)  
        2. put the cube down on the target (use mouse click to select the target)  
        """,
        
        "PickHighlight": """While the robot is pressing the button, some cubes will be highlighted with white discs on the table. Remember them.     
        A typical action sequence could be:      
        1. press the button.    
        2. pick up the cube (use mouse click to select the correct cube with the correct color)     
        3. put the cube down on the table.     
        (Repeat 2 and 3 for with the rest of highlighted cubes)   
        """,
        
        "InsertPeg": """The video shows a robot picking up and inserting a peg into a hole.    
        The peg consists of two parts with different colors; you need to pick up the correct part of the peg and insert it into the hole from the correct side.    
        A typical action sequence could be:   
        1. pick up the peg (use mouse click to select the correct peg and the correct part of the peg)   
        2. insert the peg into the hole on the left side  
        """,
        
        "MoveCube": """The video shows a robot moving a cube to a target using different methods.    
        The robot might (1) pick up and place the cube, (2) push it with the gripper, or (3) hook it using a peg.    
        Remember the way the robot moves the cube and choose the correct action to execute.   
        """,
        
        "PatternLock": """The video shows a robot tracing a pattern with a stick.    
        Remember the movements and reproduce them by choosing correct actions.  
        The correct directions (e.g., left, right, forward, backward) are as given near the base camera view.
        """,
        
        "RouteStick": """The video shows a robot navigating from one target to another by circling around a stick.  
        The movement can be clockwise or counter-clockwise, and the stick may be on the left or right side.  
        Remember the sequence of actions and choose the correct action to execute.  
        The correct directions (e.g., left, right, forward, backward) are as given near the base camera view.
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