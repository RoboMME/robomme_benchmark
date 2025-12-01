import numpy as np
def get_reset_panda_param(way,gripper=None):
    qpos = np.array(
   [0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4, 
                    0.04,
                    0.04,
                ],)
    
    action=np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4, 1.0]
                )
    #action 1 corresponds to qpos 0.04 0.04 == open gripper

    # remove last two joints (gripper) → keep only first 7
    qpos_no_gripper = qpos[:7]
    action_no_gripper = action[:7]

    if gripper == "stick":
        if way == "qpos":
            return qpos_no_gripper
        elif way == "action":
            return action_no_gripper
        
    else:
        if way=="qpos":
            return qpos
        elif way=="action":
            return action
