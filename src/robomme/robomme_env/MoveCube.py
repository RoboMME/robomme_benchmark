from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import SO100, Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Actor
#Robomme
import matplotlib.pyplot as plt

from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
import copy
from .utils import *
from .utils.difficulty import normalize_robomme_difficulty
from .utils.subgoal_evaluate_func import static_check
from .utils import subgoal_language
from .utils.object_generation import spawn_fixed_cube, build_board_with_hole
from .utils import reset_panda
from ..logging_utils import logger


PICK_CUBE_DOC_STRING = """**Task Description:**
A simple task where the objective is to grasp a red cube with the {robot_id} robot and move it to a target goal position. This is also the *baseline* task to test whether a robot with manipulation
capabilities can be simulated and trained properly. Hence there is extra code for some robots to set them up properly in this environment as well as the table scene builder.

**Randomizations:**
- the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
- the cube's z-axis rotation is randomized to a random angle
- the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

**Success Conditions:**
- the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
- the robot is static (q velocity < 0.2)
"""


@register_env("MoveCube")
class MoveCube(BaseEnv):

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
        "so100",
        "widowxai",
    ]
    agent: Union[Panda]
    goal_thresh = 0.025
    cube_spawn_half_size = 0.05
    cube_spawn_center = (0, 0)
    _clearance = 0.01

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0,seed=0,Robomme_video_episode=None,Robomme_video_path=None,
                     **kwargs):
        self.reset_in_proecess=False
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]
        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]

        self.seed = seed
        self._hb_generator = torch.Generator()
        self._hb_generator.manual_seed(int(self.seed))

        self.robomme_failure_recovery = bool(
            kwargs.pop("robomme_failure_recovery", False)
        )
        self.robomme_failure_recovery_mode = kwargs.pop(
            "robomme_failure_recovery_mode", None
        )
        if isinstance(self.robomme_failure_recovery_mode, str):
            self.robomme_failure_recovery_mode = (
                self.robomme_failure_recovery_mode.lower()
            )
        normalized_robomme_difficulty = normalize_robomme_difficulty(
            kwargs.pop("difficulty", None)
        )
        if normalized_robomme_difficulty is not None:
            self.difficulty = normalized_robomme_difficulty
        else:
            seed_mod = seed % 3
            if seed_mod == 0:
                self.difficulty = "easy"
            elif seed_mod == 1:
                self.difficulty = "medium"
            else:
                self.difficulty = "hard"

        self.restore_flag=False
        self.use_demonstrationwrapper=False
        self.demonstration_record_traj=False
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        camera_eye=[0.3,0,0.4]
        camera_target =[0,0,-0.2]
        pose = sapien_utils.look_at(
            eye=camera_eye, target=camera_target
        )
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

   

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=0
        )
        self.table_scene.build()

        length_tensor = torch.rand(1, generator=self._hb_generator)
        radius_tensor = torch.rand(1, generator=self._hb_generator)
        self.length = (0.1 + (0.05 - 0.05) * length_tensor).item()
        self.radius = (0.01 + (0.005 - 0.005) * radius_tensor).item()

        # Create a single peg
        #peg_spawn_translation = np.array([self.length / 2, 0.0, self.radius], dtype=np.float32)
        base_y = -0.2 if torch.rand(1, generator=self._hb_generator).item() < 0.5 else 0.2

        peg_spawn_translation = np.array([0.0, base_y, 0.0], dtype=np.float32)

        # Generate [-0.05, 0.05] random offset (using torch generator)
        x_jitter = (torch.rand(1, generator=self._hb_generator).item() - 0.5) * 0.1
        y_jitter = (torch.rand(1, generator=self._hb_generator).item() - 0.5) * 0.1

        # Apply offset
        peg_spawn_translation[:2] += np.array([x_jitter, y_jitter], dtype=np.float32)
        self.peg1_basex=peg_spawn_translation[0]
        self.peg1_basey=peg_spawn_translation[1]

        initial_yaw = torch.rand(1, generator=self._hb_generator).item() * (np.pi / 2) - (np.pi / 4)
        yaw_angles = torch.tensor([[0.0, 0.0, initial_yaw]], dtype=torch.float32)
        yaw_matrix = euler_angles_to_matrix(yaw_angles, convention="XYZ")
        yaw_quat = matrix_to_quaternion(yaw_matrix)[0].detach().cpu().numpy().tolist()

        peg_initial_pose = sapien.Pose(
            p=peg_spawn_translation.tolist(),
            q=yaw_quat,
        )

        self.peg, self.peg_head, self.peg_tail = build_peg(
            self,
            length=self.length,
            radius=self.radius,
            initial_pose=peg_initial_pose,
            name='peg',
            head_color= "#EC7357",
            tail_color= "#EC7357",
        )

        # Create lists for backward compatibility
        self.pegs = [self.peg]
        self.peg_heads = [self.peg_head]
        self.peg_tails = [self.peg_tail]



        # Store initial poses for all pegs
        self.peg_init_poses = [peg.pose for peg in self.pegs]
        self.peg_init_pose = self.pegs[0].pose  # Keep backward compatibility

        #generate another set of pose for another reset
        base_y = -0.2 if torch.rand(1, generator=self._hb_generator).item() < 0.5 else 0.2

        peg_spawn_translation = np.array([0.0, base_y, 0.0], dtype=np.float32)
        x_jitter = (torch.rand(1, generator=self._hb_generator).item() - 0.5) * 0.1
        y_jitter = (torch.rand(1, generator=self._hb_generator).item() - 0.5) * 0.1
        peg_spawn_translation[:2] += np.array([x_jitter, y_jitter], dtype=np.float32)

        initial_yaw = torch.rand(1, generator=self._hb_generator).item() * (np.pi / 2) - (np.pi / 4)
        yaw_angles = torch.tensor([[0.0, 0.0, initial_yaw]], dtype=torch.float32)
        yaw_matrix = euler_angles_to_matrix(yaw_angles, convention="XYZ")
        yaw_quat = matrix_to_quaternion(yaw_matrix)[0].detach().cpu().numpy().tolist()

        self.peg2_basex=peg_spawn_translation[0]
        self.peg2_basey=peg_spawn_translation[1]
        peg_initial_pose = sapien.Pose(
            p=peg_spawn_translation.tolist(),
            q=yaw_quat,
        )
        self.peg_init_poses_2=[peg_initial_pose]
        
        self.finish_return_flag=False

                # Define task list, each task contains a dictionary with function, name, demonstration flag, and optional failure_func
        obj_sample = torch.randint(0, 2, (1,), generator=self._hb_generator)
        self.obj_flag = -1 if obj_sample.item() == 0 else 1
        dir_sample = torch.randint(0, 2, (1,), generator=self._hb_generator)
        #self.direction = -1 if dir_sample.item() == 0 else 1


        self.goal_site = spawn_random_target(
                        self,
                        avoid=None,  # Use current avoidance list, containing all spawned cubes
                        include_existing=False,  # Manually maintain list
                        include_goal=False,  # Manually maintain list
                        region_center=[0.0, 0.0],
                        region_half_size=0.15,
                        radius=self.cube_half_size*2,  # Use radius instead of half_size
                        thickness=0.005,  # target thickness
                        min_gap=self.cube_half_size*1,  # Gap requirement same as cube
                        name_prefix=f"goal_site",
                        generator=self._hb_generator
                        )
        self.goal_site_2 = spawn_random_target(
                self,
                avoid=None,  # Use current avoidance list, containing all spawned cubes
                include_existing=False,  # Manually maintain list
                include_goal=False,  # Manually maintain list
                region_center=[0.0, 0.0],
                region_half_size=0.1,
                radius=self.cube_half_size*2,  # Use radius instead of half_size
                thickness=0.005,  # target thickness
                min_gap=self.cube_half_size*1,  # Gap requirement same as cube
                name_prefix=f"goal_site_2",
                generator=self._hb_generator
                )
        


        max_cube_spawn_trials = 128

        goal_pos = self.goal_site.pose.p
        goal_xy = np.asarray(goal_pos)
        goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(-1)[:2]

        def _sample_cube_center(required_distance: float):
            for _ in range(max_cube_spawn_trials):
                sampled_x = torch.rand(1, generator=self._hb_generator).item() * 0.2 -0.1
                #direction = -1.0 if -self.peg1_basey < 0 else 1.0
                #sampled_y = torch.rand(1, generator=self._hb_generator).item() * 0.2 * direction
                sampled_y = torch.rand(1, generator=self._hb_generator).item() * 0.2 -0.1
                candidate_xy = np.array([sampled_x, sampled_y], dtype=np.float64)
                if np.linalg.norm(candidate_xy - goal_xy) > required_distance:
                    return candidate_xy
            return None

        cube_center = _sample_cube_center(self.cube_half_size*5)

        cube_x, cube_y = float(cube_center[0]), float(cube_center[1])

        self.cube = spawn_random_cube(
                            self,
                            region_center=[cube_x, cube_y],
                            color=(1, 0, 0, 1),
                            name_prefix="fixed_cube",
                            region_half_size=0.05,
                            generator=self._hb_generator,
                            half_size=self.cube_half_size,
                        )
        
        self.cube_init_pose=self.cube.pose



        goal_pos = self.goal_site_2.pose.p
        goal_xy = np.asarray(goal_pos)
        goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(-1)[:2]
        def _sample_cube_center(required_distance: float):
            for _ in range(max_cube_spawn_trials):
                sampled_x = torch.rand(1, generator=self._hb_generator).item() * 0.2 -0.1
                #direction = -1.0 if -self.peg2_basey < 0 else 1.0
                #sampled_y = torch.rand(1, generator=self._hb_generator).item() * 0.2 * direction
                sampled_y = torch.rand(1, generator=self._hb_generator).item()  * 0.2 -0.1
                candidate_xy = np.array([sampled_x, sampled_y], dtype=np.float64)
                if np.linalg.norm(candidate_xy - goal_xy) > required_distance:
                    return candidate_xy
            return None

        cube_center = _sample_cube_center(self.cube_half_size*5)

        cube_x, cube_y = float(cube_center[0]), float(cube_center[1])
        self.cube_2 = spawn_random_cube(
                            self,
                            region_center=[cube_x, cube_y],
                            color=(1, 0, 0, 1),
                            name_prefix="fixed_cube_2",
                            region_half_size=0.05,
                            generator=self._hb_generator,
                            half_size=self.cube_half_size,
                        )
        
        self.cube_init_pose_2=self.cube_2.pose
        #only need the pose! teleport away in 

        goal2_p = np.array(self.goal_site_2.pose.p.detach().cpu().numpy(), dtype=np.float64, copy=True)
        self.goal_site_2_pose_p = goal2_p

        goal2_q = np.array(self.goal_site_2.pose.q.detach().cpu().numpy(), dtype=np.float64, copy=True)
        self.goal_site_2_pose_q = goal2_q

        goal1_p = np.array(self.goal_site.pose.p.detach().cpu().numpy(), dtype=np.float64, copy=True)
        self.goal_site_1_pose_p = goal1_p

        goal1_q = np.array(self.goal_site.pose.q.detach().cpu().numpy(), dtype=np.float64, copy=True)
        self.goal_site_1_pose_q = goal1_q



    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)

            if not hasattr(self, "pegs"):
                return

            # Initialize all 3 pegs

            qpos = np.array(
            [
                0.0,
                0,
                0,
                -np.pi * 4 / 8,
                0,
                np.pi * 2 / 4,
                np.pi / 4,
                0.04,
                0.04,
            ],
            dtype=np.float32,
            )
            self.ways=["peg_push","gripper_push","grasp_putdown"]
            way_idx = torch.randint(len(self.ways), (1,), generator=self._hb_generator).item()
            self.way = self.ways[way_idx]
            #self.way="gripper_push"

            self.agent.reset(qpos)            
            self.cube_2.set_pose(sapien.Pose(p=[10,10,1]))#only need the pose!
            self.goal_site_2.set_pose(sapien.Pose(p=[10, -10, 1]))

            

    def evaluate(self,solve_complete_eval=False):
        timestep = self.elapsed_steps
        # flag=is_A_pickup_notB(self,self.peg_head,self.peg_tail)
        # flag2=is_A_pickup_notB(self,self.peg_tail,self.peg_head)
        # flag=is_A_insert_notB(self,self.peg_head,self.peg_tail,self.box)

        self.successflag=torch.tensor([False])
        self.failureflag = torch.tensor([False])
        

        self.obj_flag=-1
        if self.obj_flag==-1:
            self.grasp_target=self.peg_tail
            self.grasp_target_false=self.peg_head

        else:
            self.grasp_target=self.peg_head
            self.grasp_target_false=self.peg_tail

        self.direction1 = 1 if self.cube_init_pose.p[0][1]-self.goal_site_1_pose_p[0][1] > 0 else -1# relative position
        self.direction2 = 1 if self.cube_init_pose_2.p[0][1]-self.goal_site_2_pose_p[0][1]  > 0 else -1
        # direction -1 push from left 
        # direction 1 push from right +y side / table right side from camera view -> treated as push from right


        if self.way=="peg_push":
            tasks = [
                {
                "func": lambda: is_any_obj_pickup_flag_currentpickup(self, objects=[self.grasp_target,self.grasp_target_false]),
                "name": f"Pick up the peg",
                "subgoal_segment":f"Pick up the peg at <>",
                "choice_label": "pick up the peg",
                "demonstration": True,
                "failure_func":   lambda:[
                                           is_obj_pickup(self, obj=self.cube), 
                                           is_obj_pushed_onto(self,self.cube,self.goal_site,distance_threshold=self.cube_half_size*2*1.2),],
                "solve": lambda env, planner:grasp_and_lift_peg_side(env, planner, env.grasp_target),
                "segment":self.grasp_target
                },
                {
                "func": lambda:  is_obj_pushed_onto(self,self.cube,self.goal_site,distance_threshold=self.cube_half_size*2*1.2,must_gripper_open=True),
                "name": f"Hook the cube to the target with the peg",
                "subgoal_segment":f"Hook the cube at <> to the target at <> with the peg",
                "choice_label": "hook the cube to the target with the peg",
                "demonstration": True,
                "failure_func": lambda:None,
                "solve": lambda env, planner:solve_push_to_target_with_peg(env,planner,self.cube,self.goal_site,self.direction1,self.obj_flag),
                "segment":[self.cube,self.goal_site],
                },
                                                {
                "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=30),
                "name": "static",
                "subgoal_segment":f"static",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: [solve_hold_obj(env, planner, static_steps=30)],
            },
                  {
                    "func": lambda:reset_check(self),
                    "name": "NO RECORD",
                    "subgoal_segment":f"NO RECORD",
                    "demonstration": True,
                    "failure_func": None,
                    "specialflag":"reset pegs",
                    "solve": lambda env, planner: [solve_strong_reset(env,planner)],
                    },
                
                {
                "func": lambda: is_any_obj_pickup_flag_currentpickup(self, objects=[self.grasp_target,self.grasp_target_false]),
                "name": "Pick up the peg",
                "subgoal_segment":f"Pick up the peg at <>",
                "choice_label": "pick up the peg",
                "demonstration": False,
                "failure_func":   lambda:[
                                           is_obj_pickup(self, obj=self.cube), 
                                           is_obj_pushed_onto(self,self.cube,self.goal_site,distance_threshold=self.cube_half_size*2*1.2),],
                "solve": lambda env, planner:grasp_and_lift_peg_side(env, planner, env.grasp_target),
                "segment":self.grasp_target
                },
                {
                "func": lambda:  is_obj_pushed_onto(self,self.cube,self.goal_site,distance_threshold=self.cube_half_size*2*1.2,must_gripper_open=True),
                "name": f"Hook the cube to the target with the peg",
                "subgoal_segment":f"Hook the cube at <> to the target at <> with the peg",
                "choice_label": "hook the cube to the target with the peg",
                "demonstration": False,
                "failure_func": lambda:None,
                "solve": lambda env, planner:solve_push_to_target_with_peg(env,planner,self.cube,self.goal_site,self.direction2,self.obj_flag),
                "segment":[self.cube,self.goal_site],
                },
                ]
            
            #test using gripper/grasp =false

        if self.way=="gripper_push":
            tasks = [{
                                "func": lambda: is_obj_pushed_onto(self,self.cube,self.goal_site,distance_threshold=self.cube_half_size*2*1.2,must_gripper_open=True),
                                "name": "Close the gripper and push the cube to the target",
                                "subgoal_segment":f"Close the gripper and push the cube at <> to the target at <>",
                                "choice_label": "close gripper and push the cube to the target",
                                "demonstration": True,
                                "failure_func":  lambda: [is_obj_pickup(self, obj=self.cube),
                                                          is_obj_pickup(self, obj=self.grasp_target),
                                                          is_obj_pickup(self, obj=self.grasp_target_false)],
                                "solve": lambda env, planner:solve_push_to_target(env,planner,self.cube,self.goal_site),
                                "segment":[self.cube,self.goal_site],
                                },
                                                                {
                "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=60),
                "name": "static",
                "subgoal_segment":f"static",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: [solve_hold_obj(env, planner, static_steps=60)],
            },
                                {
                                "func": lambda:reset_check(self),
                                "name": "NO RECORD",
                                "subgoal_segment":f"NO RECORD",
                                "demonstration": True,
                                "failure_func": None,
                                "specialflag":"reset pegs",
                                "solve": lambda env, planner: [solve_strong_reset(env,planner)],
                                },
                                {
                                "func": lambda: is_obj_pushed_onto(self,self.cube,self.goal_site,distance_threshold=self.cube_half_size*2*1.2,must_gripper_open=True),
                                "name": "Close the gripper and push the cube to the target",
                                "subgoal_segment":f"Close the gripper and push the cube at <> to the target at <>",
                                "choice_label": "close gripper and push the cube to the target",
                                "demonstration": False,
                                "failure_func":  lambda: [is_obj_pickup(self, obj=self.cube),
                                                          is_obj_pickup(self, obj=self.grasp_target),
                                                          is_obj_pickup(self, obj=self.grasp_target_false)],
                                "solve": lambda env, planner:solve_push_to_target(env,planner,self.cube,self.goal_site),
                                "segment":[self.cube,self.goal_site],
                                },
                                
                                
                                ]
            
        if self.way=="grasp_putdown":
            tasks = [
                {
                        "func": lambda: is_obj_pickup(self, obj=self.cube),
                        "name": "Pick up the cube",
                        "subgoal_segment":f"Pick up the cube at <>",
                        "choice_label": "pick up the cube",
                        "demonstration": True,
                        "failure_func": lambda: [is_obj_pushed_onto(self,self.cube,self.goal_site,distance_threshold=self.cube_half_size*2*1.2), 
                                                 is_obj_pickup(self, obj=self.grasp_target),
                                                 is_obj_pickup(self, obj=self.grasp_target_false)],
                        "solve": lambda env, planner:[solve_pickup(env, planner, obj=self.cube),],
                        "segment":[self.cube],
                        },
                        {
                    "func": (lambda: is_obj_dropped_onto(self,obj=self.cube,target=self.goal_site)),
                    "name": "place the cube onto the target",
                    "subgoal_segment":f"place the cube onto the target at <>",
                    "choice_label": "place the cube onto the target",
                    "demonstration": True,
                    "failure_func":  None, 
                    "solve": lambda env, planner: [solve_putonto_whenhold(env, planner,target=self.goal_site)],
                                        "segment":[self.goal_site],
                                        },

                                {
                "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=60),
                "name": "static",
                "subgoal_segment":f"static",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: [solve_hold_obj(env, planner, static_steps=60)],
            },
                                                    {
                                "func": lambda:reset_check(self),
                                "name": "NO RECORD",
                                "subgoal_segment":f"NO RECORD",
                                "demonstration": True,
                                "failure_func": None,
                                "specialflag":"reset pegs",
                                "solve": lambda env, planner: [solve_strong_reset(env,planner)],
                                },
                {
                        "func": lambda: is_obj_pickup(self, obj=self.cube),
                        "name": "Pick up the cube",
                        "subgoal_segment":f"Pick up the cube at <>",
                        "choice_label": "pick up the cube",
                        "demonstration": False,
                        "failure_func": lambda: [is_obj_pushed_onto(self,self.cube,self.goal_site,distance_threshold=self.cube_half_size*2*1.2), 
                                                 is_obj_pickup(self, obj=self.grasp_target),
                                                 is_obj_pickup(self, obj=self.grasp_target_false)],
                        "solve": lambda env, planner:[solve_pickup(env, planner, obj=self.cube),],
                        "segment":[self.cube],
                        },
                        {
                    "func": (lambda: is_obj_dropped_onto(self,obj=self.cube,target=self.goal_site)),
                    "name": "place the cube onto the target",
                    "subgoal_segment":f"place the cube onto the target at <>",
                    "choice_label": "place the cube onto the target",
                    "demonstration": False,
                    "failure_func":  None, 
                    "solve": lambda env, planner: [solve_putonto_whenhold(env, planner,target=self.goal_site)],
                                        "segment":[self.goal_site],
                                        },

            ]


                            


        # Store task list for RecordWrapper use
        self.task_list = tasks

        # Use encapsulated sequence task check function
        if(self.use_demonstrationwrapper==False):# change subgoal after planner ends during recording
            if solve_complete_eval==True:
                allow_subgoal_change_this_timestep=True
            else:
                allow_subgoal_change_this_timestep=False
        else:# during demonstration, video needs to call evaluate(solve_complete_eval), video ends and flag changes in demonstrationwrapper
            if solve_complete_eval==True or self.demonstration_record_traj==False:
                allow_subgoal_change_this_timestep=True
            else:
                allow_subgoal_change_this_timestep=False

        #allow_subgoal_change_this_timestep=True
        all_tasks_completed, current_task_name, task_failed,_ = sequential_task_check(self, tasks,allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep)

        if task_failed:
            self.failureflag = torch.tensor([True])
            logger.debug(f"Task failed: {current_task_name}")

        # If static_check succeeds or all tasks completed, set success flag
        if all_tasks_completed and not task_failed:
            self.successflag = torch.tensor([True])

        return {
            "success": self.successflag,
            "fail": self.failureflag,
        }
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.agent.tcp_pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward*0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5


#Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):


        obs, reward, terminated, truncated, info = super().step(action)
        timestep = int(info["elapsed_steps"])

            
        if self.reset_in_proecess==True:
            for i, peg in enumerate(self.pegs):
                peg.set_pose(self.peg_init_poses_2[i])
                if peg.dof > 0:
                    zero = np.zeros(peg.dof)
                    peg.set_qpos(zero)
                    peg.set_qvel(zero)

            self.cube.set_pose(self.cube_init_pose_2)
            #self.goal_site_2.set_pose(sapien.Pose(p=self.goal_site_2_pose_p[0],q=self.goal_site_2_pose_q[0]))
            goal2_p = np.array(self.goal_site_2_pose_p, copy=True)
            goal2_q = np.array(self.goal_site_2_pose_q, copy=True)
            self.goal_site.set_pose(sapien.Pose(p=goal2_p[0],q=goal2_q[0]))
            #print("reset goal site to",goal2_p[0],goal2_q[0])



        return obs, reward, terminated, truncated, info
