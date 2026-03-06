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

from .utils import *
from .utils.difficulty import normalize_robomme_difficulty
from .utils.subgoal_evaluate_func import static_check
from .utils import subgoal_language
from .utils.object_generation import spawn_fixed_cube, build_board_with_hole
from .utils import reset_panda
from .utils import subgoal_evaluate_func
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


@register_env("InsertPeg")
class InsertPeg(BaseEnv):

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
        self.use_demonstrationwrapper=False
        self.demonstration_record_traj=False
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
        # self.length = (0.05 + (0.0 - 0.0) * length_tensor).item()
        # self.radius = (0.01 + (0.0 - 0.0) * radius_tensor).item()
        self.length = (0.05 + (0.01 - 0.01) * length_tensor).item()
        self.radius = (0.01 + (0.005 - 0.005) * radius_tensor).item()

        # Create 3 identical pegs with different x-axis coordinates
        self.pegs = []
        self.peg_heads = []
        self.peg_tails = []
        self._peg_initial_poses = []

        offsets = [0.1,0,-0.1]  # X-axis differences for the 3 pegs
        # Sample a single pair of colors so all pegs share the same appearance per seed.
        peg_head_color = torch.rand(3, generator=self._hb_generator).tolist()
        # Use complementary tail color so head/tail are contrasting.
        peg_tail_color = [1.0 - c for c in peg_head_color]

        for offset in offsets:
            peg_spawn_translation = np.array([self.length / 2 , -0.15-offset, self.radius], dtype=np.float32)

            #initial_yaw = (torch.rand(1, generator=self._hb_generator).item() * 2 * np.pi) - np.pi
            initial_yaw =  0
            yaw_angles = torch.tensor([[0.0, 0.0, initial_yaw]], dtype=torch.float32)
            yaw_matrix = euler_angles_to_matrix(yaw_angles, convention="XYZ")
            yaw_quat = matrix_to_quaternion(yaw_matrix)[0].detach().cpu().numpy().tolist()

            peg_initial_pose = sapien.Pose(
                p=peg_spawn_translation.tolist(),
                q=yaw_quat,
            )

            peg, peg_head, peg_tail = build_peg(
                self,
                length=self.length,
                radius=self.radius,
                initial_pose=peg_initial_pose,
                name=f"peg_{len(self.pegs)}",
                head_color=peg_head_color,
                tail_color=peg_tail_color,
            )

            self.pegs.append(peg)
            self.peg_heads.append(peg_head)
            self.peg_tails.append(peg_tail)
            self._peg_initial_poses.append(peg_initial_pose)

        # Randomly select one peg from the 3 pegs
        random_peg_idx = int(torch.randint(0, 3, (1,), generator=self._hb_generator).item())
        random_peg_idx=0
        self.peg = self.pegs[random_peg_idx]

        self.peg_head = self.peg_heads[random_peg_idx]
        self.peg_tail = self.peg_tails[random_peg_idx]
        self._peg_initial_pose = self._peg_initial_poses[random_peg_idx]


        self.box=build_box_with_hole(self,inner_radius=self.radius*1.7,outer_radius=self.radius*4,depth=self.length,center=[0,0])
        
        self.reset_in_proecess=False
   

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.end_steps=None
            self.table_scene.initialize(env_idx)
            # Reset highlight state at the start of each episode
            self._insert_highlight_start = None
            self._insert_highlight_active = False

            if not hasattr(self, "pegs"):
                return

            base_translation = (0, 0)
            x_jitter_2 = (torch.rand(1, generator=self._hb_generator).item() - 0.5) * 0.2
            y_jitter_2 = (torch.rand(1, generator=self._hb_generator).item() - 0.5) * 0.2
            # x_jitter_2=0
            # y_jitter_2=0
            box_translation = [base_translation[0] + x_jitter_2, base_translation[1] + y_jitter_2, self.radius * 4]
            box_yaw = np.pi / 2 + (torch.rand(1, generator=self._hb_generator).item() * 2 - 1) * np.radians(20)
            box_angles = torch.tensor([[0.0, 0.0, box_yaw]], dtype=torch.float32)
            box_matrix = euler_angles_to_matrix(box_angles, convention="XYZ")
            box_quat = matrix_to_quaternion(box_matrix)[0].detach().cpu().numpy().tolist()
            self.box.set_pose(sapien.Pose(p=box_translation, q=box_quat))

            box_xy = np.array(box_translation[:2], dtype=np.float32)
            sampled_xy_positions = []
            max_sampling_attempts = 512

            #Initialize all 3 pegs with constrained random placements
            for i, peg in enumerate(self.pegs):
                candidate_xy = None
                for _ in range(max_sampling_attempts):
                    x_sample = (torch.rand(1, generator=self._hb_generator).item() * 0.4) - 0.2
                    y_sample = (torch.rand(1, generator=self._hb_generator).item() * 0.6) - 0.3
                    sampled_xy = np.array([x_sample, y_sample], dtype=np.float32)

                    if np.linalg.norm(sampled_xy - box_xy) <= self.radius * 6:
                        continue

                    if any(np.linalg.norm(sampled_xy - prev_xy) <= self.length * 1.5 for prev_xy in sampled_xy_positions):
                        continue

                    candidate_xy = sampled_xy
                    break

                if candidate_xy is None:
                    raise RuntimeError("Failed to sample peg positions satisfying placement constraints.")

                yaw_value = (torch.rand(1, generator=self._hb_generator).item() * 2 - 1) * np.radians(45)

                yaw_angles = torch.tensor([[0.0, 0.0, yaw_value]], dtype=torch.float32)
                yaw_matrix = euler_angles_to_matrix(yaw_angles, convention="XYZ")
                yaw_quat = matrix_to_quaternion(yaw_matrix)[0].detach().cpu().numpy().tolist()

                pose = sapien.Pose(p=[float(candidate_xy[0]), float(candidate_xy[1]), 0.0], q=yaw_quat)
                peg.set_pose(pose)
                sampled_xy_positions.append(candidate_xy)



            # Store initial poses for all pegs
            self.peg_init_poses = []
            for peg in self.pegs:
                pose = peg.pose
                pose_p = np.asarray(pose.p, dtype=np.float32).reshape(-1).copy()
                pose_q = np.asarray(pose.q, dtype=np.float32).reshape(-1).copy()
                self.peg_init_poses.append(sapien.Pose(p=pose_p, q=pose_q))


            # robomme-v2.7/robomme/robomme_env/PickPeg.py:243
            pose = self.peg.pose
            pose_p = np.asarray(pose.p, dtype=np.float32).reshape(-1).copy()
            pose_q = np.asarray(pose.q, dtype=np.float32).reshape(-1).copy()
            self.peg_init_pose = sapien.Pose(p=pose_p, q=pose_q)
            self.peg_init_pose = sapien.Pose(p=pose_p, q=pose_q)


                    # Define task list, each task contains a dictionary with function, name, demonstration flag, and optional failure_func
            obj_sample = torch.randint(0, 2, (1,), generator=self._hb_generator)
            dir_sample = torch.randint(0, 2, (1,), generator=self._hb_generator)
            self.obj_flag = -1 if obj_sample.item() == 0 else 1
            self.direction = -1 if dir_sample.item() == 0 else 1
            
            # if self.seed<30:
            #     self.obj_flag=-1
            #     self.direction=1
            # elif self.seed<60:
            #     self.obj_flag=-1
            #     self.direction=-1
            # elif self.seed<90:
            #     self.obj_flag=1
            #     self.direction=1
            # #elif self.seed<60:
            # else:
            #     self.obj_flag=1
            #     self.direction=-1


            # self.obj_flag=1
            # self.direction=-1


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

            self.agent.reset(qpos)            
            if self.obj_flag==-1:
                self.grasp_target=self.peg_head
                self.insert_target=self.peg_tail
            else:
                self.grasp_target=self.peg_tail
                self.insert_target=self.peg_head

            agent_x = self.agent.robot.pose.p.tolist()[0][0]
            head_x = float(self.peg_head.pose.p.tolist()[0][0])
            tail_x = float(self.peg_tail.pose.p.tolist()[0][0])
            logger.debug(f"agent_x: {agent_x}, head_x: {head_x}, tail_x: {tail_x}")
            near_link = self.peg_head if abs(head_x - agent_x) <= abs(tail_x - agent_x) else self.peg_tail

            self.grasp_target_distance = "near" if self.grasp_target is near_link else "far"
            logger.debug(f"grasp_target_distance: {self.grasp_target_distance}")

            self.insert_way="left" if self.direction == -1 else "right"
            tasks = [
                {
                    "func": lambda: is_A_pickup_notB(self, self.grasp_target, self.insert_target),
                    "name": f"Pick up the peg by grasping the {self.grasp_target_distance} end",
                    "subgoal_segment":f"Pick up the peg by grasping the {self.grasp_target_distance} end at <>",
                    "choice_label": "pick up the peg by grasping one end",
                    "demonstration": True,
                    "failure_func": lambda: is_A_pickup_notB(self, self.insert_target, self.grasp_target),
                    "solve": lambda env, planner: grasp_and_lift_peg_side(env, planner, env.grasp_target),
                    "segment":self.grasp_target
                },
                {
                    "func": lambda: is_A_insert_notB(self, self.insert_target, self.grasp_target, self.box,direction=self.direction),
                    "name": f"Insert the peg from the {self.insert_way} side of the box",
                    "subgoal_segment":f"Insert the peg from the {self.insert_way} side of the box at <>",
                    "choice_label": f"insert the peg from the {self.insert_way} side",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: insert_peg(env, planner,  direction=self.direction,obj=self.obj_flag,insert_obj=self.insert_target),
                    "segment":self.box
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
                    "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=100),
                    "name": "NO RECORD",
                    "subgoal_segment":"NO RECORD",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [solve_hold_obj(env, planner, static_steps=100,close=False)],
                },

            {
                    "func": lambda: is_A_pickup_notB(self, self.grasp_target, self.insert_target),
                    "name": f"Pick up the peg by grasping the {self.grasp_target_distance} end",
                    "subgoal_segment":f"Pick up the peg by grasping the {self.grasp_target_distance} end at <>",
                    "choice_label": "pick up the peg by grasping one end",
                    "demonstration": False,
                    "failure_func": lambda: [
                        is_A_pickup_notB(self, self.insert_target, self.grasp_target),
                        is_any_obj_pickup(self, [head for i, head in enumerate(self.peg_heads) if self.pegs[i] is not self.peg] +
                                            [tail for i, tail in enumerate(self.peg_tails) if self.pegs[i] is not self.peg])
                    ],
                    "solve": lambda env, planner: grasp_and_lift_peg_side(env, planner, env.grasp_target),
                    "segment":self.grasp_target
                },
                {
                    "func": lambda: is_A_insert_notB(self, self.insert_target, self.grasp_target,self.box,direction=self.direction,mark_end_flag=True),
                    "name": f"Insert the peg from the {self.insert_way} side",
                    "subgoal_segment":f"Insert the peg from the {self.insert_way} side at <>",
                    "choice_label": f"insert the peg from the {self.insert_way} side",
                    "demonstration": False,
                    "failure_func": lambda: [
                        is_A_insert_notB(self, self.grasp_target, self.insert_target, self.box),
                        is_A_insert_notB(self, self.insert_target, self.grasp_target, self.box, direction=-self.direction),
                        is_any_obj_pickup(self, [head for i, head in enumerate(self.peg_heads) if self.pegs[i] is not self.peg] +
                                            [tail for i, tail in enumerate(self.peg_tails) if self.pegs[i] is not self.peg])
                    ],
                    "solve": lambda env, planner: insert_peg(env, planner,  direction=self.direction,obj=self.obj_flag,insert_obj=self.insert_target,cut_retreat=True),
                    "segment":self.box
                },
            ]

            # Store task list for RecordWrapper use
            self.task_list = tasks

    def evaluate(self,solve_complete_eval=False):
        timestep = self.elapsed_steps


        self.successflag=torch.tensor([False])
        self.failureflag = torch.tensor([False])
        


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

        all_tasks_completed, current_task_name, task_failed,_ = sequential_task_check(self, self.task_list,allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep)


        if self.end_steps!=None:# truncate tail, also truncate tail in planner
            logger.debug(
                "elapsed_steps=%s, end_steps=%s",
                self.elapsed_steps,
                self.end_steps,
            )
            if int(getattr(self, "elapsed_steps", 0))>=self.end_steps+3:
                 self.successflag = torch.tensor([True])

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
        cur_step = int(self.elapsed_steps[0].item())



        if self.reset_in_proecess==True:
            for i, peg in enumerate(self.pegs):
                peg.set_pose(self.peg_init_poses[i])
                if peg.dof > 0:
                    zero = np.zeros(peg.dof)
                    peg.set_qpos(zero)
                    peg.set_qvel(zero)
              
            logger.debug("reset peg!")



        if is_A_insert_notB(self, self.insert_target, self.grasp_target, self.box,direction=self.direction):
            self.start_step=cur_step


        color=sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5)
        if getattr(self, "start_step", None) is not None :
            if cur_step <=  self.start_step + 20 and cur_step>= self.start_step:
               color=[1.0, 0.0, 0.0, 1.0]


        highlight_obj(
                self,
                self.box,
                start_step= 0,
                end_step= 99999,
                cur_step=cur_step,
                disk_radius=0.015,
                disk_half_length=0.055,
                highlight_color=color,)
        return obs, reward, terminated, truncated, info
