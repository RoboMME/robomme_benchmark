import numpy as np
import sapien
import gymnasium as gym

import torch
from historybench.HistoryBench_env import *
from . import reset_panda

from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)

from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
    quaternion_multiply,
)
from historybench.HistoryBench_env.util import *
 
import random
from ...logging_utils import logger

# Probability for deliberately triggering a failed hover before pickup.
FAILED_HOVER_PROB = 0.03


def grasp_and_lift_peg_side(env, planner,obj):
    planner.open_gripper()
    """Move to the peg tail, close gripper, lift, and keep holding."""
    pose = obj.pose
    lift_height=0.2

    grasp_pose_p = pose.p
    if isinstance(grasp_pose_p, torch.Tensor):
        grasp_pose_p = grasp_pose_p.detach().cpu().numpy()
    grasp_pose_p = np.asarray(grasp_pose_p, dtype=np.float32).reshape(-1)

    grasp_pose_q = pose.q
    if isinstance(grasp_pose_q, torch.Tensor):
        grasp_pose_q = grasp_pose_q.detach().cpu().numpy()
    grasp_pose_q = np.asarray(grasp_pose_q, dtype=np.float32).reshape(-1)

    flip_angles = torch.tensor([[np.pi, 0.0, 0.0]], dtype=torch.float32)
    flip_matrix = euler_angles_to_matrix(flip_angles, convention="XYZ")
    flip_quat = matrix_to_quaternion(flip_matrix)[0]

    grasp_pose_q_tensor = torch.from_numpy(grasp_pose_q).to(
        dtype=torch.float32, device=flip_quat.device
    )
    rotated_quat = quaternion_multiply(
        grasp_pose_q_tensor.unsqueeze(0), flip_quat.unsqueeze(0)
    ).squeeze(0)
    grasp_pose_q = rotated_quat.detach().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(grasp_pose_q)
    if norm > 0:
        grasp_pose_q /= norm

    lifted_pose_p = grasp_pose_p.copy()
    lifted_pose_p[2] = lift_height
    planner.move_to_pose_with_screw(sapien.Pose(p=lifted_pose_p, q=grasp_pose_q))

    planner.move_to_pose_with_screw(sapien.Pose(p=grasp_pose_p, q=grasp_pose_q))
    planner.close_gripper()

    lifted_pose_p = grasp_pose_p.copy()
    lifted_pose_p[2] = lift_height
    planner.move_to_pose_with_screw(sapien.Pose(p=lifted_pose_p, q=grasp_pose_q))

    planner.close_gripper()
    current_grasp_pose=sapien.Pose(p=grasp_pose_p, q=grasp_pose_q)
    env.current_grasp_pose = current_grasp_pose


def return_to_original_pose(env,planner,current_grasp_pose):

    grasp_pose_p=current_grasp_pose.p
    grasp_pose_q=current_grasp_pose.q
    lifted_pose_p = grasp_pose_p.copy()
    lifted_pose_p[2] = 0.2
    planner.move_to_pose_with_screw(sapien.Pose(p=lifted_pose_p, q=grasp_pose_q))
    planner.move_to_pose_with_screw(sapien.Pose(p=grasp_pose_p, q=grasp_pose_q))
    planner.open_gripper()

# def insert_peg(env, planner,current_grasp_pose,peg_init_pose,direction,obj):
# """Insert the peg into the box."""
# grasp_pose = current_grasp_pose

# insert_pose = env.box.pose * peg_init_pose.inv() * grasp_pose

# if obj==-1:
#     insert_pose=insert_pose*sapien.Pose(q=[0, 0, 0, 1])

# if direction==-1:
#     insert_pose=insert_pose*sapien.Pose(q=[0, 0, 0, 1])

# # pre_insert_pose=sapien.Pose(p=[-0.1,0.15,0.2],q=insert_pose.q.tolist()[0])
# # pre_insert_pose=pre_insert_pose*sapien.Pose([0.05*direction, 0, 0])
# # planner.move_to_pose_with_screw(pre_insert_pose)


# if obj==-1:
#     if direction==-1:
#         bias=0.1
#     else:
#         bias=0
#     insert_pose_p = np.asarray(insert_pose.p, dtype=np.float32).reshape(-1)
#     insert_pose_q = np.asarray(insert_pose.q, dtype=np.float32).reshape(-1)
#     pre_pose = sapien.Pose(
#         p=[insert_pose_p[0] + 0.2 + bias, insert_pose_p[1], 0.2],
#         q=insert_pose_q
#     )
#     planner.move_to_pose_with_screw(pre_pose)
#     planner.move_to_pose_with_screw(insert_pose * sapien.Pose([0.2+bias, 0, 0]))
#     planner.move_to_pose_with_screw(insert_pose * sapien.Pose([0.15+bias, 0, 0]))
#     for i in range(5):
#         planner.move_to_pose_with_screw(insert_pose * sapien.Pose([0.05+bias, 0, 0]))
#     planner.move_to_pose_with_screw(insert_pose * sapien.Pose([-0.1+bias, 0, 0]))
#     planner.move_to_pose_with_screw(insert_pose * sapien.Pose([0.15+bias, 0, 0]))
    

# else:
#     pass
#     if direction==-1:
#         bias=0.1
#     else:
#         bias=0
#     dx = -0.2 - bias
# legacy comment
#     rel = insert_pose * sapien.Pose([dx, 0, 0])
#     rel_p = np.asarray(rel.p, dtype=np.float32).reshape(-1)
#     rel_q = np.asarray(rel.q, dtype=np.float32).reshape(-1)

# legacy comment
#     ready = sapien.Pose(p=[rel_p[0], rel_p[1], 0.2], q=rel_q)

#     planner.move_to_pose_with_screw(ready)
#     planner.move_to_pose_with_screw(insert_pose * sapien.Pose([-0.2-bias, 0, 0]))
#     planner.move_to_pose_with_screw(insert_pose * sapien.Pose([-0.15-bias, 0, 0]))
#     for i in range(5):
#         planner.move_to_pose_with_screw(insert_pose * sapien.Pose([-0.05-bias, 0, 0]))
#     # planner.move_to_pose_with_screw(insert_pose * sapien.Pose([-0.05-bias, 0, 0]))
#     planner.move_to_pose_with_screw(insert_pose * sapien.Pose([0.1-bias, 0, 0]))
#     planner.move_to_pose_with_screw(insert_pose * sapien.Pose([-0.15-bias, 0, 0]))

#    insert_pose = env.box.pose * insert_obj.pose.inv() * env.agent.tcp.pose






# def insert_peg(env, planner,direction,obj,insert_obj=None):
#     """Insert the peg into the box."""

#     if insert_obj is None:
#         raise ValueError("insert_obj must be provided to compute the insert pose.")

#     def _compute_insert_pose():
#         pose = env.box.pose * insert_obj.pose.inv() * env.agent.tcp.pose
#         if obj == -1:
#             pose = pose * sapien.Pose(q=[0, 0, 0, 1])
#         if direction == -1:
#             pose = pose * sapien.Pose(q=[0, 0, 0, 1])
#         return pose

#     def _pose_components():
#         pose = _compute_insert_pose()
#         pose_p = np.asarray(pose.p, dtype=np.float32).reshape(-1)
#         pose_q = np.asarray(pose.q, dtype=np.float32).reshape(-1)
#         return pose, pose_p, pose_q

#     def _move_with_offset(offset):
#         # Recompute insert pose each time before moving to stay aligned with the box.
#         current_pose = _compute_insert_pose()
#         planner.move_to_pose_with_screw(current_pose * sapien.Pose(offset))
# ##########################
#     if obj==-1:

#         _, insert_pose_p, insert_pose_q = _pose_components()
#         pre_pose = sapien.Pose(
#             p=[0, 0, 0.2],
#             q=insert_pose_q
#         )
# legacy comment
#         planner.move_to_pose_with_screw(pre_pose)

#         _move_with_offset([0.2 , 0, 0])
#         _move_with_offset([0.15 , 0, 0])
#         for i in range(5):
#             _move_with_offset([0.05 , 0, 0])
#         _move_with_offset([-0.1 , 0, 0])
#         _move_with_offset([0.15 , 0, 0])
        

#     else:#obj=1

# legacy comment
#         _, insert_pose_p, insert_pose_q = _pose_components()
#         pre_pose = sapien.Pose(
#             p=[0,0, 0.2],
#             q=insert_pose_q
#         )
# legacy comment
#         planner.move_to_pose_with_screw(pre_pose)

#         _move_with_offset([-0.2 , 0, 0])
#         _move_with_offset([-0.15 , 0, 0])
#         for i in range(5):
#             _move_with_offset([-0.05 , 0, 0])
#         _move_with_offset([0.1 , 0, 0])
#         _move_with_offset([-0.15 , 0, 0])

    
def insert_peg(env, planner,direction,obj,insert_obj=None,cut_retreat=False):
    """Insert the peg into the box."""

    if insert_obj is None:
        raise ValueError("insert_obj must be provided to compute the insert pose.")

    def _compute_insert_pose():
        pose = env.box.pose * insert_obj.pose.inv() * env.agent.tcp.pose
        if obj == -1:
            pose = pose * sapien.Pose(q=[0, 0, 0, 1])
        if direction == -1:
            pose = pose * sapien.Pose(q=[0, 0, 0, 1])
        return pose

    def _pose_components():
        pose = _compute_insert_pose()
        pose_p = np.asarray(pose.p, dtype=np.float32).reshape(-1)
        pose_q = np.asarray(pose.q, dtype=np.float32).reshape(-1)
        return pose, pose_p, pose_q

    def _move_with_offset(offset):
        # Recompute insert pose each time before moving to stay aligned with the box.
        current_pose = _compute_insert_pose()
        offset_vec = np.asarray(offset, dtype=np.float32).reshape(-1).copy()
        if obj == 1 and direction == -1 and offset_vec[0] < 0:
            # Tail grasp + left insert: compensate head-tail gap so we still move far enough.
            relative_pose = insert_obj.pose.inv() * env.agent.tcp.pose
            relative_p = np.asarray(relative_pose.p, dtype=np.float32).reshape(-1)
            offset_vec[0] += relative_p[0]
        planner.move_to_pose_with_screw(current_pose * sapien.Pose(offset_vec.tolist()))

    def _move_with_offset_with_break(offset):
        """Move with an interrupt check. Interrupt when elapsed_steps > end_steps + 3; if end_steps is None, execute directly."""
        end_steps = getattr(env, "end_steps", None)
        while end_steps is None or int(getattr(env, "elapsed_steps", 0)) <= end_steps + 3:
            # legacy comment
            current_pose = _compute_insert_pose()
            offset_vec = np.asarray(offset, dtype=np.float32)
            target_pose = current_pose * sapien.Pose(offset_vec.tolist())

            # legacy comment
            pose_for_plan = planner._transform_pose_for_planning(target_pose)
            pose_p = np.asarray(pose_for_plan.p, dtype=np.float32).reshape(-1)
            pose_q = np.asarray(pose_for_plan.q, dtype=np.float32).reshape(-1)
            result = planner.planner.plan_screw(
                np.concatenate([pose_p, pose_q]),
                planner.robot.get_qpos().cpu().numpy()[0],
                time_step=planner.base_env.control_timestep,
                use_point_cloud=planner.use_point_cloud,
            )
            if result["status"] != "Success":
                return False

            # legacy comment
            n_step = result["position"].shape[0]
            for i in range(n_step):
                if end_steps is not None and int(getattr(env, "elapsed_steps", 0)) > end_steps + 3:
                    logger.debug("break early")
                    return True  # legacy comment
                qpos = result["position"][i]
                if planner.control_mode == "pd_joint_pos_vel":
                    qvel = result["velocity"][i]
                    action = np.hstack([qpos, qvel, planner.gripper_state])
                else:
                    action = np.hstack([qpos, planner.gripper_state])
                planner.env.step(action)
                planner.elapsed_steps += 1
            return True  # legacy comment
        return True  # legacy comment
##########################
    if obj==-1:

        _, insert_pose_p, insert_pose_q = _pose_components()


        _move_with_offset([0.2 , 0, -0.15])
        _move_with_offset([0.2 , 0, 0])
        _move_with_offset([0.15 , 0, 0])
        # for i in range(5):
        #     _move_with_offset([0.05 , 0, 0])
        if cut_retreat!=True:
            _move_with_offset([-0.05 , 0, 0])
        else:
            logger.debug(f"cut_retreat mode (obj=-1): elapsed_steps={int(getattr(env, 'elapsed_steps', 0))}, end_steps={env.end_steps}")
            _move_with_offset_with_break([-0.05, 0, 0])


    else:#obj=1
        
        _, insert_pose_p, insert_pose_q = _pose_components()

        _move_with_offset([-0.2 , 0, -0.15])
        _move_with_offset([-0.2 , 0, 0])
        _move_with_offset([-0.15 , 0, 0])
        # for i in range(5):
        #     _move_with_offset([-0.05 , 0, 0])
        if cut_retreat!=True:
            _move_with_offset([-0.05 , 0, 0])
        else:
            logger.debug(f"cut_retreat mode (obj=1): elapsed_steps={int(getattr(env, 'elapsed_steps', 0))}, end_steps={env.end_steps}")
            _move_with_offset_with_break([-0.05, 0, 0])


def _zero_action_for_space(space):
    if isinstance(space, gym.spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)
    if isinstance(space, gym.spaces.Dict):
        return {k: _zero_action_for_space(subspace) for k, subspace in space.spaces.items()}
    sample = space.sample()
    if isinstance(sample, np.ndarray):
        return np.zeros_like(sample)
    if isinstance(sample, dict):
        return {k: np.zeros_like(v) for k, v in sample.items()}
    raise NotImplementedError("Unsupported action space type for zero action generation")


def _flag_to_bool(flag):
    if flag is None:
        return False
    if isinstance(flag, (bool, np.bool_, np.bool8)):
        return bool(flag)
    if isinstance(flag, torch.Tensor):
        return bool(flag.detach().cpu().bool().any())
    if isinstance(flag, np.ndarray):
        return bool(flag.any())
    return bool(flag)

def solve_liftup_Xdistance(env,planner,distance):
    original_pose = env.agent.tcp.pose
    lift_pose_p=original_pose.p.tolist()[0]
    lift_pose_q=original_pose.q.tolist()[0]
    lift_pose_p[2]+=distance
    planner.move_to_pose_with_screw(sapien.Pose(p=lift_pose_p,q=lift_pose_q))
# def solve_push_to_target(env, planner, obj=None, target=None):
#     planner.open_gripper()
#     FINGER_LENGTH = 0.025
#     env = env.unwrapped

# legacy comment
#     obj_pos = obj.pose.sp.p if hasattr(obj.pose.sp.p, '__iter__') else np.array(obj.pose.sp.p)
#     target_pos = target.pose.sp.p if hasattr(target.pose.sp.p, '__iter__') else np.array(target.pose.sp.p)
    
# legacy comment
# legacy comment
# legacy comment

# legacy comment
#     push_direction_3d = np.array([push_direction_xy[0], push_direction_xy[1], 0])

# legacy comment
# legacy comment
# legacy comment
# legacy comment
# legacy comment

# legacy comment
#     x_axis = np.array([push_direction_xy[0], push_direction_xy[1], 0])

# legacy comment
#     z_axis = np.array([0, 0, -1])

# legacy comment
#     y_axis = np.cross(z_axis, x_axis)
#     y_norm = np.linalg.norm(y_axis)
#     if y_norm < 1e-6:
#         raise ValueError("Push direction invalid; cannot construct gripper frame.")
#     y_axis = y_axis / y_norm

# legacy comment
#     rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

# legacy comment
#     rotation_matrix_torch = torch.from_numpy(rotation_matrix).float().unsqueeze(0)
#     push_quat = matrix_to_quaternion(rotation_matrix_torch)[0]

#     # -------------------------------------------------------------------------- #
# legacy comment
#     # -------------------------------------------------------------------------- #
#     if  target_pos[0] < obj_pos[0] :
#         z_rotation_angles = torch.deg2rad(
#             torch.tensor([0.0, 0.0, 180], dtype=torch.float32)
#         )
#         z_rotation_matrix = euler_angles_to_matrix(z_rotation_angles, convention="XYZ")
#         z_rotation_quat = matrix_to_quaternion(z_rotation_matrix.unsqueeze(0))[0]

# legacy comment
#         push_quat = quaternion_multiply(
#             push_quat.unsqueeze(0), z_rotation_quat.unsqueeze(0)
#         )[0]


# legacy comment
#     push_pose = sapien.Pose(p=obj_pos, q=push_quat.detach().cpu().numpy())
    
#     # -------------------------------------------------------------------------- #
# legacy comment
#     # -------------------------------------------------------------------------- #
# legacy comment
#     start_pos = obj_pos - push_direction_3d * offset_distance
# legacy comment
    
#     reach_pose_q = push_pose.q.tolist() if hasattr(push_pose.q, 'tolist') else list(push_pose.q)
    
# legacy comment
#     #planner.move_to_pose_with_screw(sapien.Pose(p=[0,0,0.1], q=reach_pose_q))
#     planner.move_to_pose_with_screw(sapien.Pose(p=start_pos.tolist(), q=reach_pose_q))
    
# legacy comment
#     planner.close_gripper()
    
#     # -------------------------------------------------------------------------- #
# legacy comment
#     # -------------------------------------------------------------------------- #
# legacy comment
#     end_pos = target_pos.copy()
# legacy comment
    
        
#     planner.move_to_pose_with_screw(sapien.Pose(p=end_pos.tolist(), q=reach_pose_q))
    
# legacy comment
#     planner.open_gripper()
def solve_push_to_target(env, planner, obj=None, target=None):
    planner.open_gripper()
    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # legacy comment
    obj_pos = obj.pose.sp.p if hasattr(obj.pose.sp.p, '__iter__') else np.array(obj.pose.sp.p)
    target_pos = target.pose.sp.p if hasattr(target.pose.sp.p, '__iter__') else np.array(target.pose.sp.p)
    
    # legacy comment
    push_direction_xy = target_pos[:2] - obj_pos[:2]  # legacy comment
    push_direction_xy = push_direction_xy / np.linalg.norm(push_direction_xy)  # legacy comment

    # legacy comment
    push_direction_3d = np.array([push_direction_xy[0], push_direction_xy[1], 0])

    # legacy comment
    # legacy comment
    # legacy comment
    # legacy comment
    # legacy comment

    # legacy comment
    x_axis = np.array([push_direction_xy[0], push_direction_xy[1], 0])

    # legacy comment
    z_axis = np.array([0, 0, -1])

    # legacy comment
    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-6:
        raise ValueError("Push direction invalid; cannot construct gripper frame.")
    y_axis = y_axis / y_norm

    # legacy comment
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

    # legacy comment
    rotation_matrix_torch = torch.from_numpy(rotation_matrix).float().unsqueeze(0)
    push_quat = matrix_to_quaternion(rotation_matrix_torch)[0]

    # -------------------------------------------------------------------------- #
    # legacy comment
    # -------------------------------------------------------------------------- #
    if  target_pos[0] < obj_pos[0] :
        z_rotation_angles = torch.deg2rad(
            torch.tensor([0.0, 0.0, 180], dtype=torch.float32)
        )
        z_rotation_matrix = euler_angles_to_matrix(z_rotation_angles, convention="XYZ")
        z_rotation_quat = matrix_to_quaternion(z_rotation_matrix.unsqueeze(0))[0]

        # legacy comment
        push_quat = quaternion_multiply(
            push_quat.unsqueeze(0), z_rotation_quat.unsqueeze(0)
        )[0]


    # legacy comment
    push_pose = sapien.Pose(p=obj_pos, q=push_quat.detach().cpu().numpy())
    
    # -------------------------------------------------------------------------- #
    # legacy comment
    # -------------------------------------------------------------------------- #
    offset_distance = 0.05  # legacy comment
    start_pos = obj_pos - push_direction_3d * offset_distance
    start_pos[2] = push_pose.p[2]  # legacy comment
    
    reach_pose_q = push_pose.q.tolist() if hasattr(push_pose.q, 'tolist') else list(push_pose.q)
    
    # legacy comment
    #planner.move_to_pose_with_screw(sapien.Pose(p=[0,0,0.1], q=reach_pose_q))
    planner.move_to_pose_with_screw(sapien.Pose(p=start_pos.tolist(), q=reach_pose_q))
    
    # legacy comment
    planner.close_gripper()
    
    # -------------------------------------------------------------------------- #
    # legacy comment
    # -------------------------------------------------------------------------- #
    # legacy comment
    end_pos = target_pos.copy() - push_direction_3d * env.cube_half_size
    end_pos[2] = start_pos[2]  # legacy comment
    
        
    planner.move_to_pose_with_screw(sapien.Pose(p=end_pos.tolist(), q=reach_pose_q))
    
    # legacy comment
    planner.open_gripper()



# def solve_push_to_target_with_peg(env, planner, obj=None, target=None,direction=None,obj_flag=None):


#     FINGER_LENGTH = 0.025
#     env = env.unwrapped

# legacy comment
#     obj_pos = obj.pose.sp.p if hasattr(obj.pose.sp.p, '__iter__') else np.array(obj.pose.sp.p)
#     target_pos = target.pose.sp.p if hasattr(target.pose.sp.p, '__iter__') else np.array(target.pose.sp.p)

    
    
# legacy comment
# legacy comment
# legacy comment

# legacy comment
#     push_direction_3d = np.array([push_direction_xy[0], push_direction_xy[1], 0])

# legacy comment
# legacy comment
# legacy comment
# legacy comment
# legacy comment

# legacy comment
#     x_axis = np.array([push_direction_xy[0], push_direction_xy[1], 0])

# legacy comment
#     z_axis = np.array([0, 0, -1])

# legacy comment
#     y_axis = np.cross(z_axis, x_axis)
#     y_norm = np.linalg.norm(y_axis)
#     if y_norm < 1e-6:
#         raise ValueError("Push direction invalid; cannot construct gripper frame.")
#     y_axis = y_axis / y_norm

# legacy comment
#     rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

# legacy comment
#     rotation_matrix_torch = torch.from_numpy(rotation_matrix).float().unsqueeze(0)
#     base_quat = matrix_to_quaternion(rotation_matrix_torch)[0]

# legacy comment
#     z_rotation_angles = torch.deg2rad(torch.tensor([0.0, 0.0, 90.0*direction*obj_flag], dtype=torch.float32))
#     z_rotation_matrix = euler_angles_to_matrix(z_rotation_angles, convention="XYZ")
#     z_rotation_quat = matrix_to_quaternion(z_rotation_matrix.unsqueeze(0))[0]

# legacy comment
#     push_quat = quaternion_multiply(base_quat.unsqueeze(0), z_rotation_quat.unsqueeze(0))[0].cpu().numpy()

# legacy comment
#     push_pose = sapien.Pose(p=obj_pos, q=push_quat)

#     # -------------------------------------------------------------------------- #
# legacy comment
#     # -------------------------------------------------------------------------- #
# legacy comment
#     start_pos = obj_pos - push_direction_3d * offset_distance
# legacy comment

#     reach_pose_q = push_pose.q.tolist() if hasattr(push_pose.q, 'tolist') else list(push_pose.q)

# legacy comment
#     planner.move_to_pose_with_screw(sapien.Pose(p=[0, 0, 0.1], q=reach_pose_q))
#     start_pos=start_pos.tolist()
#     start_pos[1]-=0.1*direction
# legacy comment
#     start_ready_pos[2]=0.1
#     planner.move_to_pose_with_screw(sapien.Pose(p=start_ready_pos, q=reach_pose_q))
#     planner.move_to_pose_with_screw(sapien.Pose(p=start_pos, q=reach_pose_q))

# legacy comment
#     planner.close_gripper()

#     # -------------------------------------------------------------------------- #
# legacy comment
#     # -------------------------------------------------------------------------- #
# legacy comment
#     end_pos = target_pos.copy()
# legacy comment
#     end_pos[1]-=0.1*direction
#     planner.move_to_pose_with_screw(sapien.Pose(p=end_pos.tolist(), q=reach_pose_q))

# legacy comment
#     planner.open_gripper()
def solve_push_to_target_with_peg(env, planner, obj=None, target=None, direction=None, obj_flag=None):

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # -------------------------------------------------------------------------- #
    # legacy comment
    # -------------------------------------------------------------------------- #
    obj_pos = obj.pose.sp.p if hasattr(obj.pose.sp.p, '__iter__') else np.array(obj.pose.sp.p)
    target_pos = target.pose.sp.p if hasattr(target.pose.sp.p, '__iter__') else np.array(target.pose.sp.p)

    # -------------------------------------------------------------------------- #
    # legacy comment
    # -------------------------------------------------------------------------- #
    push_direction_xy = target_pos[:2] - obj_pos[:2]
    push_direction_xy = push_direction_xy / np.linalg.norm(push_direction_xy)

    # legacy comment
    push_direction_3d = np.array([push_direction_xy[0], push_direction_xy[1], 0])

    # -------------------------------------------------------------------------- #
    # legacy comment
    # legacy comment
    # legacy comment
    # legacy comment
    # -------------------------------------------------------------------------- #
    x_axis = np.array([push_direction_xy[0], push_direction_xy[1], 0])   # legacy comment
    z_axis = np.array([0, 0, -1])                                        # legacy comment
    y_axis = np.cross(z_axis, x_axis)                                    # legacy comment

    # legacy comment
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-6:
        raise ValueError("Invalid push direction; failed to construct gripper frame.")
    y_axis = y_axis / y_norm

    # legacy comment
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

    # -------------------------------------------------------------------------- #
    # legacy comment
    # -------------------------------------------------------------------------- #
    rotation_matrix_torch = torch.from_numpy(rotation_matrix).float().unsqueeze(0)
    base_quat = matrix_to_quaternion(rotation_matrix_torch)[0]

    # -------------------------------------------------------------------------- #
    # legacy comment
    # -------------------------------------------------------------------------- #
    z_rotation_angles = torch.deg2rad(
        torch.tensor([0.0, 0.0, 90.0 * direction * obj_flag], dtype=torch.float32)
    )
    z_rotation_matrix = euler_angles_to_matrix(z_rotation_angles, convention="XYZ")
    z_rotation_quat = matrix_to_quaternion(z_rotation_matrix.unsqueeze(0))[0]

    # legacy comment
    push_quat = quaternion_multiply(
        base_quat.unsqueeze(0), z_rotation_quat.unsqueeze(0)
    )[0].cpu().numpy()

    # legacy comment
    push_pose = sapien.Pose(p=obj_pos, q=push_quat)

    # -------------------------------------------------------------------------- #
    # legacy comment
    # -------------------------------------------------------------------------- #
    offset_distance = 0.1                                            # legacy comment
    start_pos = obj_pos - push_direction_3d * offset_distance
    start_pos[2] = obj_pos[2]                                        # legacy comment

    # legacy comment
    lateral_unit = np.array([-push_direction_xy[1], push_direction_xy[0], 0], dtype=np.float32)
    lateral_norm = np.linalg.norm(lateral_unit[:2])

    # legacy comment
    if lateral_norm < 1e-6:
        lateral_unit = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        lateral_unit /= lateral_norm

    # legacy comment
    lateral_distance = 0.1 * direction
    start_pos = start_pos - lateral_unit * lateral_distance

    # legacy comment
    reach_pose_q = push_pose.q.tolist() if hasattr(push_pose.q, "tolist") else list(push_pose.q)

    # -------------------------------------------------------------------------- #
    # legacy comment
    #    
    # legacy comment
    # -------------------------------------------------------------------------- #
    start_ready_pos = start_pos.copy()
  
    # legacy comment
    planner.move_to_pose_with_screw(sapien.Pose(p=start_ready_pos.tolist(), q=reach_pose_q))
    planner.move_to_pose_with_screw(sapien.Pose(p=start_pos.tolist(), q=reach_pose_q))

    # -------------------------------------------------------------------------- #
    # legacy comment
    # -------------------------------------------------------------------------- #
    planner.close_gripper()

        # -------------------------------------------------------------------------- #
    # legacy comment
    # -------------------------------------------------------------------------- #
    # legacy comment
    end_pos = target_pos - push_direction_3d * 0.03 # legacy comment
    end_pos[2] = start_pos[2]  # legacy comment
    end_pos = end_pos -  lateral_unit * lateral_distance
    planner.move_to_pose_with_screw(sapien.Pose(p=end_pos.tolist(), q=reach_pose_q))

    # legacy comment
    planner.open_gripper()

def move_to_avoid(env, planner):

    original_pose = env.agent.tcp.pose
    lift_pose_p=original_pose.p.tolist()[0]
    lift_pose_q=original_pose.q.tolist()[0]
    lift_pose_p=[-0.1,0,0.1]
    planner.move_to_pose_with_screw(sapien.Pose(p=lift_pose_p,q=lift_pose_q))

    return None


def solve_pickup_fail(env, planner, obj=None,z_offset=None,xy_offset=None,obj_type="cube",mode=None):
    """Hover directly above grasp pose with slight random +z, close then reopen gripper."""
    if obj is None:
        return None

    env = getattr(env, "unwrapped", env)
    planner.open_gripper()

    # Build the same grasp pose as the normal pickup, but stop above it with random z lift <= 0.1.
    obb = get_actor_obb(obj)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=0.025,
    )
    closing = grasp_info["closing"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, obj.pose.sp.p)
    if obj_type=="bin":
        grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.01])

    fail_pose_p = np.asarray(grasp_pose.p, dtype=np.float32).reshape(-1).tolist()
    fail_pose_q = np.asarray(grasp_pose.q, dtype=np.float32).reshape(-1).tolist()

    normalized_mode = mode.lower() if isinstance(mode, str) else mode

    # Randomly pick whether to perturb the failed hover in XY or Z.
    #mode = random.choice(["xy", "z"])
    #mode="xy"
    if normalized_mode == "xy":
        env.fail="xy"
        xy_offset = np.asarray(xy_offset, dtype=np.float32).reshape(-1)
        if xy_offset.size == 1:
            xy_offset = np.repeat(xy_offset, 2)
        # allow +/- adjustments or no change per axis, but ensure at least one axis moves
        signs = [random.choice((-1, 0, 1)) for _ in range(2)]
        while signs[0] == 0 and signs[1] == 0:
            signs = [random.choice((-1, 0, 1)) for _ in range(2)]
        signed_offset = xy_offset * np.array(signs, dtype=np.float32)
        fail_pose_p[0] += float(signed_offset[0])
        fail_pose_p[1] += float(signed_offset[1])
    elif normalized_mode == "z":
        env.fail="z"
        z_shift = z_offset 
        fail_pose_p[2] += z_shift
    else:
        raise ValueError(f"Invalid fail mode: {mode}")
    
    ready_pose_p = fail_pose_p.copy()
    ready_pose_p[2] = 0.15
    if obj_type=="bin":
        ready_pose_p[2] = 0.2
    ready_pose = sapien.Pose(p=ready_pose_p, q=fail_pose_q)
    planner.move_to_pose_with_screw(ready_pose)
    fail_pose = sapien.Pose(p=fail_pose_p, q=fail_pose_q)
    planner.move_to_pose_with_screw(fail_pose)
    planner.close_gripper()
    planner.move_to_pose_with_screw(ready_pose)
    planner.open_gripper()

    env.use_fail_planner=True

    return None


def solve_pickup(env, planner, obj=None,fail_grasp=False,mode=None):
    # 10% chance to perform a deliberate failed hover before the normal pickup.
    planner.open_gripper()
    if(env.use_demonstrationwrapper==False):
        if fail_grasp==True:
            solve_pickup_fail(env, planner, obj,z_offset=env.cube_half_size*2,xy_offset=env.cube_half_size*2,obj_type="cube",mode=mode)

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(obj)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, obj.pose.sp.p)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose_p = grasp_pose.p.tolist() if hasattr(grasp_pose.p, 'tolist') else list(grasp_pose.p)
    reach_pose_q = grasp_pose.q.tolist() if hasattr(grasp_pose.q, 'tolist') else list(grasp_pose.q)
    reach_pose_p[2]=0.15
    planner.move_to_pose_with_screw(sapien.Pose(p=reach_pose_p,q=reach_pose_q))
    planner.open_gripper()
    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose_P=obj.pose.p.tolist()[0]
    goal_pose_P[2]=0.15
    goal_pose = sapien.Pose(goal_pose_P, grasp_pose.q)
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res

def solve_pickup_bin(env, planner, obj=None, fail_grasp=False, mode=None):
    planner.open_gripper()
    # 10% chance to perform a deliberate failed hover before the normal pickup.
    if(env.use_demonstrationwrapper==False):
        if fail_grasp==True:
            solve_pickup_fail(env, planner, obj,z_offset=0.035,xy_offset=0.035,obj_type="bin", mode=mode)

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(obj)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, obj.pose.sp.p)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.15])
    reach_pose_p=reach_pose.p.tolist()
    reach_pose_p[2]=0.2
    reach_pose_q=reach_pose.q.tolist()
    reach_pose_fix=sapien.Pose(reach_pose_p,reach_pose_q)
    planner.move_to_pose_with_screw(reach_pose_fix)
    planner.open_gripper()

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    grasp_pose_up=grasp_pose * sapien.Pose([0, 0, -0.01])
    planner.move_to_pose_with_screw(grasp_pose_up)
    planner.close_gripper()
    # grasp_pose_up=grasp_pose * sapien.Pose([0, 0.1,0])#test
    # planner.move_to_pose_with_screw(grasp_pose_up)#test

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose_P=obj.pose.p.tolist()[0]
    goal_pose_P[2]=0.2
    goal_pose = sapien.Pose(goal_pose_P, grasp_pose.q)
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res


# def solve_putdown_bin(env, planner, obj=None):
#     planner.open_gripper()
#     FINGER_LENGTH = 0.025
#     env = env.unwrapped

#     planner.close_gripper()
#     goal_pose_p=env.agent.tcp.pose.p.tolist()[0]
#     goal_pose_p[2]=0.15
#     goal_pose_p[0]+=0.1
#     goal_pose_q=env.agent.tcp.pose.q.tolist()[0]
#     goal_pose = sapien.Pose(goal_pose_p,goal_pose_q)
#     res = planner.move_to_pose_with_screw(goal_pose)

#     goal_pose_p[2]=0
#     goal_pose = sapien.Pose(goal_pose_p,goal_pose_q)
#     res = planner.move_to_pose_with_screw(goal_pose)

#     planner.open_gripper()
#     planner.close()
#     return res



# def solve_putdown_whenhold(env, planner, obj=None):
#     FINGER_LENGTH = 0.025
#     env = env.unwrapped

#     # retrieves the object oriented bounding box (trimesh box object)
#     obb = get_actor_obb(obj)

#     approaching = np.array([0, 0, -1])
#     # get transformation matrix of the tcp pose, is default batched and on torch
#     target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
#     # we can build a simple grasp pose using this information for Panda
#     grasp_info = compute_grasp_info_by_obb(
#         obb,
#         approaching=approaching,
#         target_closing=target_closing,
#         depth=FINGER_LENGTH,
#     )
#     closing, center = grasp_info["closing"], grasp_info["center"]
#     grasp_pose = env.agent.build_grasp_pose(approaching, closing, obj.pose.sp.p)


#     goal_pose_P=obj.pose.p.tolist()[0]
#     goal_pose_P[2]=0
#     goal_pose = sapien.Pose(goal_pose_P, grasp_pose.q)
#     res = planner.move_to_pose_with_screw(goal_pose)
#     planner.open_gripper()
#     planner.close()
#     return res


def solve_putonto_whenhold(env, planner,target=None):
    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # # retrieves the object oriented bounding box (trimesh box object)
    # obb = get_actor_obb(obj)

    # approaching = np.array([0, 0, -1])
    # # get transformation matrix of the tcp pose, is default batched and on torch
    # target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # # we can build a simple grasp pose using this information for Panda
    # grasp_info = compute_grasp_info_by_obb(
    #     obb,
    #     approaching=approaching,
    #     target_closing=target_closing,
    #     depth=FINGER_LENGTH,
    # )
    # closing, center = grasp_info["closing"], grasp_info["center"]
    # grasp_pose = env.agent.build_grasp_pose(approaching, closing, obj.pose.sp.p)

    grasp_pose_q=env.agent.tcp.pose.q.tolist()[0]

    goal_pose_P_prepare=target.pose.p.tolist()[0]
    goal_pose_P_prepare[2]=0.15
    goal_pose = sapien.Pose(goal_pose_P_prepare, grasp_pose_q)
    res = planner.move_to_pose_with_screw(goal_pose)

    goal_pose_P=target.pose.p.tolist()[0]
    goal_pose = sapien.Pose(goal_pose_P, grasp_pose_q)
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.open_gripper()

    goal_pose_P=target.pose.p.tolist()[0]
    goal_pose_P[2]=0.15
    goal_pose = sapien.Pose(goal_pose_P, grasp_pose_q)
    res = planner.move_to_pose_with_screw(goal_pose)
    planner.close()
    return res
def solve_swingonto_whenhold(env, planner,target=None,height=0.05):
    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # # retrieves the object oriented bounding box (trimesh box object)
    # obb = get_actor_obb(obj)

    # approaching = np.array([0, 0, -1])
    # # get transformation matrix of the tcp pose, is default batched and on torch
    # target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # # we can build a simple grasp pose using this information for Panda
    # grasp_info = compute_grasp_info_by_obb(
    #     obb,
    #     approaching=approaching,
    #     target_closing=target_closing,
    #     depth=FINGER_LENGTH,
    # )
    # closing, center = grasp_info["closing"], grasp_info["center"]
    # grasp_pose = env.agent.build_grasp_pose(approaching, closing, obj.pose.sp.p)

    grasp_pose_q=env.agent.tcp.pose.q.tolist()[0]
    goal_pose_P=target.pose.p.tolist()[0]
    goal_pose_P[2]=height
    goal_pose = sapien.Pose(goal_pose_P, grasp_pose_q)
    res = planner.move_to_pose_with_screw(goal_pose)
    planner.close()
    return res
def solve_swingonto_withDirection(env, planner, target=None, radius=0.1, direction="counterclockwise"):
    """Planar arc motion at z=0.07 from current TCP to target.

    direction: "counterclockwise" means the left side of t_start->t_end (positive cross product); "clockwise" means the right side."""
    if target is None:
        raise ValueError("target must be provided for swing onto motion.")

    start_pos = env.agent.tcp.pose.p.reshape(-1, 3)[0]
    end_pos = target.pose.p.reshape(-1, 3)[0]
    start_xy = np.asarray(start_pos[:2], dtype=np.float32)
    end_xy = np.asarray(end_pos[:2], dtype=np.float32)

    chord_vec = end_xy - start_xy
    chord_len = np.linalg.norm(chord_vec)
    current_qpos = env.agent.tcp.pose.q.reshape(-1, 4)[0].tolist()

    # Initial joint pose used to dry-run each segment in order, then concatenate into one trajectory for execution
    init_qpos_tensor = planner.robot.get_qpos()
    qpos_device = init_qpos_tensor.device if hasattr(init_qpos_tensor, "device") else None
    qpos_dtype = init_qpos_tensor.dtype if hasattr(init_qpos_tensor, "dtype") else torch.float32
    init_qpos = (
        init_qpos_tensor.detach().cpu().numpy() if hasattr(init_qpos_tensor, "detach") else np.asarray(init_qpos_tensor)
    ).reshape(-1)
    plan_start_qpos = init_qpos.copy()


    waypoints = []
    if chord_len < 1e-6:
        goal_p = end_pos.tolist()
        goal_p[2] = 0.07
        waypoints.append(sapien.Pose(goal_p, current_qpos))
    else:
        # radius only controls lateral offset; forward span comes from the chord length.
        lateral_offset = float(max(radius, 1e-4))
        dir_unit = chord_vec / chord_len
        perp = np.array([-dir_unit[1], dir_unit[0]])
        direction_l = str(direction).lower()
        if direction_l == "counterclockwise":
            sign = -1.0
        elif direction_l == "clockwise":
            sign = 1.0
        else:
            # compatible with legacy left/right notation
            sign = 1.0 if direction_l == "left" else -1.0
        control_xy = (start_xy + end_xy) / 2.0 + sign * lateral_offset * perp

        # Use a quadratic Bezier curve (start -> control -> end) to create a smoother bend than a circular arc.
        num_steps = 45
        for t in np.linspace(0.0, 1.0, num_steps):
            one_minus_t = 1.0 - t
            waypoint_xy = (
                (one_minus_t ** 2) * start_xy
                + 2 * one_minus_t * t * control_xy
                + (t ** 2) * end_xy
            )
            goal_p = [float(waypoint_xy[0]), float(waypoint_xy[1]), 0.07]
            waypoints.append(sapien.Pose(goal_p, current_qpos))
    if waypoints:
        # Hold the final pose a bit longer by repeating it 5 times.
        last_p = np.asarray(waypoints[-1].p, dtype=np.float32).reshape(-1).tolist()
        last_q = np.asarray(waypoints[-1].q, dtype=np.float32).reshape(-1).tolist()
        for _ in range(5):
            waypoints.append(sapien.Pose(last_p, last_q))
    logger.debug(" get waypoint")
    # Directly connect each waypoint IK solution into a discrete path without extra interpolation/planning
    positions = []
    last_res = None
    # legacy comment
    plan_start_qpos_full = planner.planner.pad_qpos(plan_start_qpos.copy())
    for idx, wp in enumerate(waypoints):
        pose_for_plan = planner._transform_pose_for_planning(wp)
        pose_p = np.asarray(pose_for_plan.p, dtype=np.float32).reshape(-1)
        pose_q = np.asarray(pose_for_plan.q, dtype=np.float32).reshape(-1)
        goal_world = np.concatenate([pose_p, pose_q])
        goal_base = planner.planner.transform_goal_to_wrt_base(goal_world)

        ik_status, ik_solutions = planner.planner.IK(
            goal_base,
            plan_start_qpos_full.copy(),
        )
        if ik_status != "Success" or len(ik_solutions) == 0:
            logger.debug(f"IK failed at waypoint {idx}: {ik_status}")
            continue

        # Take the first IK solution and append directly; do not call plan_qpos_to_qpos for interpolation
        qpos_sol = ik_solutions[0]
        padded_qpos = plan_start_qpos_full.copy()
        padded_qpos[: qpos_sol.shape[0]] = qpos_sol
        positions.append(padded_qpos)
        # Update full qpos as the next segment start to avoid base/joint drift between segments
        plan_start_qpos_full = padded_qpos

    if len(positions) == 0:
        logger.debug("No IK solutions found for waypoints, aborting follow_path.")
        return None

    traj_res = {
        "status": "Success",
        "position": np.stack(positions, axis=0),
    }
    if planner.control_mode == "pd_joint_pos_vel":
        traj_res["velocity"] = np.zeros_like(traj_res["position"])

    last_res = planner.follow_path(traj_res)
    return last_res
# def solve_swingonto_withDirection(env, planner, target=None, radius=0.1, direction="counterclockwise"):
#     """Planar arc motion at z=0.07 from current TCP to target.

#     direction: "counterclockwise" means the left side of t_start->t_end (positive cross product); "clockwise" means the right side."""
#     if target is None:
#         raise ValueError("target must be provided for swing onto motion.")

#     start_pos = env.agent.tcp.pose.p.reshape(-1, 3)[0]
#     end_pos = target.pose.p.reshape(-1, 3)[0]
#     start_xy = np.asarray(start_pos[:2], dtype=np.float32)
#     end_xy = np.asarray(end_pos[:2], dtype=np.float32)

#     chord_vec = end_xy - start_xy
#     chord_len = np.linalg.norm(chord_vec)
#     current_qpos = env.agent.tcp.pose.q.reshape(-1, 4)[0].tolist()

#     if chord_len < 1e-6:
#         goal_p = end_pos.tolist()
#         goal_p[2] = 0.07
#         try:
#             return planner.move_to_pose_with_screw(sapien.Pose(goal_p, current_qpos))
#         except Exception as e:
#             print(f"move_to_pose_with_screw failed: {e}")
#             return None

#     # Keep radius feasible for chord length
#     radius = float(max(radius, chord_len / 2.0 + 1e-4))
#     half_chord = chord_len / 2.0
#     sagitta = math.sqrt(max(radius ** 2 - half_chord ** 2, 0.0))

#     dir_unit = chord_vec / chord_len
#     perp = np.array([-dir_unit[1], dir_unit[0]])
#     direction_l = str(direction).lower()
#     if direction_l == "counterclockwise":
#         sign = 1.0
#     elif direction_l == "clockwise":
#         sign = -1.0
#     else:
#         # compatible with legacy left/right notation
#         sign = 1.0 if direction_l == "left" else -1.0
#     center_xy = (start_xy + end_xy) / 2.0 + sign * sagitta * perp

#     start_angle = math.atan2(start_xy[1] - center_xy[1], start_xy[0] - center_xy[0])
#     end_angle = math.atan2(end_xy[1] - center_xy[1], end_xy[0] - center_xy[0])
#     angle_diff = end_angle - start_angle
#     if sign > 0 and angle_diff <= 0:
#         angle_diff += 2 * math.pi
#     if sign < 0 and angle_diff >= 0:
#         angle_diff -= 2 * math.pi

#     num_steps = 10
#     res = None
#     for ang in np.linspace(start_angle, start_angle + angle_diff, num_steps):
#         waypoint_xy = center_xy + radius * np.array([math.cos(ang), math.sin(ang)])
#         goal_p = [float(waypoint_xy[0]), float(waypoint_xy[1]), 0.07]
#         try:
#             res = planner.move_to_pose_with_screw(sapien.Pose(goal_p, current_qpos))
#         except Exception as e:
#             print(f"move_to_pose_with_screw failed: {e}")
#             res = None
#     return res

def solve_swingonto(env, planner,target=None,record_swing_qpos=False):
    env = env.unwrapped


    # if horizontal==True:
    #     # use current gripper pose and rotate 90° around Z axis
    #     current_qpos = torch.tensor([-7.3356e-08,  1.0000e+00, -2.0862e-07, -1.8728e-09])
    #     z_rot = torch.tensor([[0.0, 0.0, np.pi / 2]], dtype=torch.float32, device=current_qpos.device)
    #     z_rot = matrix_to_quaternion(euler_angles_to_matrix(z_rot, convention="XYZ"))[0]
    #     current_qpos = quaternion_multiply(current_qpos.unsqueeze(0), z_rot.unsqueeze(0))[0]
    #     current_qpos = current_qpos.tolist()
    # else:
    #     current_qpos = env.agent.tcp.pose.q.reshape(-1, 4)[0]
    
    current_qpos = env.agent.tcp.pose.q.reshape(-1, 4)[0]

    
    goal_pose_P=target.pose.p.tolist()[0]
    goal_pose_P[2]=0.07
    goal_pose = sapien.Pose(goal_pose_P, current_qpos)
    for i in range(2):
        res = planner.move_to_pose_with_screw(goal_pose)
    try:
        planner.close_gripper()
    except:
        AttributeError


    if record_swing_qpos==True:
        env.swing_qpos=env.agent.robot.qpos

    return res

def solve_strong_reset(env, planner,timestep=30,gripper=None,action=None):
    try:
        planner.open_gripper()
    except:
        AttributeError
    if action==None:
        action=reset_panda.get_reset_panda_param("action",gripper=gripper)
    for i in range(timestep):
        env.step(action)
        logger.debug("strong reset!!")
        env.unwrapped.reset_in_proecess=True
        env.unwrapped.after_demo=True
    env.unwrapped.reset_in_proecess=False
def solve_reset(env, planner):
    pose_p=[0,0,0.2]
    pose_q=env.agent.tcp.pose.q.tolist()[0]
    planner.move_to_pose_with_screw(sapien.Pose(p=pose_p,q=pose_q))
    planner.open_gripper()

def solve_putdown_whenhold(env, planner,release_z=0.07):
    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # # retrieves the object oriented bounding box (trimesh box object)
    # obb = get_actor_obb(obj)

    # approaching = np.array([0, 0, -1])
    # # get transformation matrix of the tcp pose, is default batched and on torch
    # target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # # we can build a simple grasp pose using this information for Panda
    # grasp_info = compute_grasp_info_by_obb(
    #     obb,
    #     approaching=approaching,
    #     target_closing=target_closing,
    #     depth=FINGER_LENGTH,
    # )
    # closing, center = grasp_info["closing"], grasp_info["center"]
    # grasp_pose = env.agent.build_grasp_pose(approaching, closing, obj.pose.sp.p)
    grasp_pose_q=env.agent.tcp.pose.q.tolist()[0]

    goal_pose_P=env.agent.tcp.pose.p.tolist()[0]
    goal_pose_P[2]=release_z
    #goal_pose_P[0]+=0.1#test
    goal_pose = sapien.Pose(goal_pose_P,grasp_pose_q)
    res = planner.move_to_pose_with_screw(goal_pose)
    planner.open_gripper()

    goal_pose_P=env.agent.tcp.pose.p.tolist()[0]
    goal_pose_P[2]=0.15

    goal_pose = sapien.Pose(goal_pose_P, grasp_pose_q)
    res = planner.move_to_pose_with_screw(goal_pose)
    planner.close()
    return res


def solve_putonto_whenhold_binspecial(env, planner,target=None):
    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # # retrieves the object oriented bounding box (trimesh box object)
    # obb = get_actor_obb(obj)

    # approaching = np.array([0, 0, -1])
    # # get transformation matrix of the tcp pose, is default batched and on torch
    # target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # # we can build a simple grasp pose using this information for Panda
    # grasp_info = compute_grasp_info_by_obb(
    #     obb,
    #     approaching=approaching,
    #     target_closing=target_closing,
    #     depth=FINGER_LENGTH,
    # )
    # closing, center = grasp_info["closing"], grasp_info["center"]
    #grasp_pose = env.agent.build_grasp_pose(approaching, closing, obj.pose.sp.p)

    grasp_pose_q=env.agent.tcp.pose.q.tolist()[0]

    goal_pose_P=target.pose.p.tolist()[0]
    goal_pose_P[2]=0.2
    goal_pose = sapien.Pose(goal_pose_P, grasp_pose_q)
    res = planner.move_to_pose_with_screw(goal_pose)
    planner.open_gripper()

    goal_pose_P=target.pose.p.tolist()[0]
    goal_pose_P[0]=goal_pose_P[0]-0.1
    goal_pose_P[2]=0.2
    goal_pose = sapien.Pose(goal_pose_P, grasp_pose_q)
    res = planner.move_to_pose_with_screw(goal_pose)
    
    planner.close()
    return res

def solve_hold_obj(env, planner, static_steps,close=False):
    start_step = int(getattr(env, "elapsed_steps", 0))
    target_step = start_step + static_steps
    while int(getattr(env, "elapsed_steps", 0)) < target_step:
        if close:
                try:
                     planner.close_gripper()
                except:
                    AttributeError

        else:
                try:
                     planner.open_gripper()
                except:
                    AttributeError

        current_step = int(getattr(env, "elapsed_steps", 0))
        #print(f"<plannner>:Holding object:{current_step}/{target_step})")
    return None

def solve_hold_obj_absTimestep(env,planner,absTimestep):
    while int(getattr(env, "elapsed_steps", 0)) < absTimestep:
        planner.close_gripper()
    return None

def solve_button(env, planner,obj,steps_press=None,interval=20,without_hold=False):
    # if steps_press:
    #     while env.elapsed_steps<steps_press-interval:
    #         current_step = int(getattr(env, "elapsed_steps", 0))
    #         print(f"Waiting to press button:{current_step}/{steps_press-interval})")
    #         planner.close_gripper()

    FINGER_LENGTH = 0.025
    env=env.unwrapped
    position=obj.pose.p.tolist()[0]
    ready_position=position.copy()
    ready_position[2]=0.15

    angles = torch.deg2rad(torch.tensor([180.0, 0.0, 0.0], dtype=torch.float32))  # (3,)
    rotate = matrix_to_quaternion(
        euler_angles_to_matrix(angles, convention="XYZ")
    )
    if without_hold==False:
        planner.move_to_pose_with_screw(sapien.Pose(p=ready_position,q=rotate))
    planner.close_gripper()
    steps=env.elapsed_steps.item()
    logger.debug(f"press button at step {steps}")
    planner.move_to_pose_with_screw(sapien.Pose(p=position,q=rotate))

    planner.move_to_pose_with_screw(sapien.Pose(p=ready_position, q=rotate))

def solve_button_ready(env, planner,obj):
    FINGER_LENGTH = 0.025
    env=env.unwrapped
    position=obj.pose.p.tolist()[0]
    ready_position=position.copy()
    ready_position[2]=0.15

    angles = torch.deg2rad(torch.tensor([180.0, 0.0, 0.0], dtype=torch.float32))  # (3,)
    rotate = matrix_to_quaternion(
        euler_angles_to_matrix(angles, convention="XYZ")
    )
    
    planner.move_to_pose_with_screw(sapien.Pose(p=ready_position,q=rotate))
    planner.close_gripper()
