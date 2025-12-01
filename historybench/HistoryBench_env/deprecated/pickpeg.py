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
#HistoryBench
import matplotlib.pyplot as plt
import h5py
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)

from util import *
from util.evaluate import static_check
from util.object_generation import build_peg


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


@register_env("PickPeg", max_episode_steps=200)
class PickPeg(BaseEnv):

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

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0,HistoryBench_seed=0,HistoryBench_video_episode=None,HistoryBench_video_path=None,
                     **kwargs):
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

        self.HistoryBench_seed = HistoryBench_seed
        self._hb_generator = torch.Generator()
        self._hb_generator.manual_seed(int(self.HistoryBench_seed))


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
        self.length = (0.065 + (0.125 - 0.085) * length_tensor).item()
        self.radius = (0.015 + (0.025 - 0.015) * radius_tensor).item()

        self._peg_spawn_translation = np.array([self.length / 2, 0.0, self.radius], dtype=np.float32)

        initial_yaw = (torch.rand(1, generator=self._hb_generator).item() * 2 * np.pi) - np.pi
        yaw_angles = torch.tensor([[0.0, 0.0, initial_yaw]], dtype=torch.float32)
        yaw_matrix = euler_angles_to_matrix(yaw_angles, convention="XYZ")
        yaw_quat = matrix_to_quaternion(yaw_matrix)[0].detach().cpu().numpy().tolist()

        self._peg_initial_pose = sapien.Pose(
            p=self._peg_spawn_translation.tolist(),
            q=yaw_quat,
        )

        self.peg, self.peg_head, self.peg_tail = build_peg(
            self,
            length=self.length,
            radius=self.radius,
            initial_pose=self._peg_initial_pose,
        )

        self.box=build_box_with_hole(self,inner_radius=self.radius*2,outer_radius=self.radius*2.5,depth=self.length/2,center=[0,0])
        

   

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)

            if not hasattr(self, "peg"):
                return

            # peg_translation = getattr(
            #     self,
            #     "_peg_spawn_translation",
            #     np.array([self.length / 2, 0.0, self.radius], dtype=np.float32),
            # )

            yaw_override = None
            if options and isinstance(options, dict):
                yaw_override = options.get("peg_yaw")

            if isinstance(yaw_override, (list, tuple)) and len(yaw_override) > 0:
                yaw_value = float(yaw_override[0])
            elif isinstance(yaw_override, (int, float)):
                yaw_value = float(yaw_override)
            else:
                yaw_value =  (torch.rand(1, generator=self._hb_generator).item() * 2 - 1) * np.radians(20)

            yaw_angles = torch.tensor([[0.0, 0.0, yaw_value]], dtype=torch.float32)
            yaw_matrix = euler_angles_to_matrix(yaw_angles, convention="XYZ")
            yaw_quat = matrix_to_quaternion(yaw_matrix)[0].detach().cpu().numpy().tolist()

            pose = sapien.Pose(p=[0,-0.3,0], q=yaw_quat)
            self.peg.set_pose(pose)
            self._peg_initial_pose = pose

            if self.peg.dof > 0:
                zero = np.zeros(self.peg.dof)
                self.peg.set_qpos(zero)
                self.peg.set_qvel(zero)


            box_translation = [0, 0, self.radius * 4]
            box_yaw = np.pi / 2 + (torch.rand(1, generator=self._hb_generator).item() * 2 - 1) * np.radians(20)
            box_angles = torch.tensor([[0.0, 0.0, box_yaw]], dtype=torch.float32)
            box_matrix = euler_angles_to_matrix(box_angles, convention="XYZ")
            box_quat = matrix_to_quaternion(box_matrix)[0].detach().cpu().numpy().tolist()
            self.box.set_pose(sapien.Pose(p=box_translation, q=box_quat))

            self.peg_init_pose = self.peg.pose
            self.finish_return_flag=False

                    # 定义任务列表，每个任务包含一个带函数、名称、demonstration 标志和可选 failure_func 的字典
            obj_sample = torch.randint(0, 2, (1,), generator=self._hb_generator)
            self.obj_flag = -1 if obj_sample.item() == 0 else 1
            dir_sample = torch.randint(0, 2, (1,), generator=self._hb_generator)
            self.direction = -1 if dir_sample.item() == 0 else 1

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

    def evaluate(self,solve_complete_eval=False):
        timestep = self.elapsed_steps
        # flag=is_A_pickup_notB(self,self.peg_head,self.peg_tail)
        # flag2=is_A_pickup_notB(self,self.peg_tail,self.peg_head)
        # flag=is_A_insert_notB(self,self.peg_head,self.peg_tail,self.box)

        self.successflag=torch.tensor([False])
        self.failureflag = torch.tensor([False])
        
        

        if self.obj_flag==-1:
            self.grasp_target=self.peg_head
            self.insert_target=self.peg_tail
        else:
            self.grasp_target=self.peg_tail
            self.insert_target=self.peg_head
        tasks = [
            {
                "func": lambda: is_A_pickup_notB(self, self.grasp_target, self.insert_target),
                "name": f"Pick up peg",
                "demonstration": True,
                "failure_func": lambda: is_A_pickup_notB(self, self.insert_target, self.grasp_target),
                "solve": lambda env, planner: grasp_and_lift_peg_side(env, planner, env.grasp_target),
            },
            {
                "func": lambda: is_A_insert_notB(self, self.insert_target, self.grasp_target, self.box),
                "name": f"insert peg",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: insert_peg(env, planner, env.current_grasp_pose, env.peg_init_pose, direction=self.direction,obj=self.obj_flag),
            },
            {
                "func": lambda: restore_finish(self),
                "name": f"return",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: return_to_original_pose(env, planner, env.current_grasp_pose),
            },
            {
                "func": lambda: is_A_pickup_notB(self, self.grasp_target, self.insert_target),
                "name": f"Pick up peg",
                "demonstration": False,
                "failure_func": lambda: is_A_pickup_notB(self, self.insert_target, self.grasp_target),
                "solve": lambda env, planner: grasp_and_lift_peg_side(env, planner, env.grasp_target),
            },
            {
                "func": lambda: is_A_insert_notB(self, self.insert_target, self.grasp_target, self.box),
                "name": f"insert peg",
                "demonstration": False,
                "failure_func": None,
                "solve": lambda env, planner: insert_peg(env, planner, env.current_grasp_pose, env.peg_init_pose, direction=self.direction,obj=self.obj_flag),
            },
        ]

        # 存储任务列表供RecordWrapper使用
        self.task_list = tasks

        # 使用封装的序列任务检查函数
        if(self.use_demonstrationwrapper==False):#record时候planner结束再改变subgoal
            if solve_complete_eval==True:
                allow_subgoal_change_this_timestep=True
            else:
                allow_subgoal_change_this_timestep=False
        else:#demonstration时候无论如何都为true
            allow_subgoal_change_this_timestep=True
        all_tasks_completed, current_task_name, task_failed = sequential_task_check(self, tasks,allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep)

        if task_failed:
            self.failureflag = torch.tensor([True])
            print(f"Task failed: {current_task_name}")

        # 如果static_check成功或者所有任务完成，则设置成功标志
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


#HistoryBench
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):


        obs, reward, terminated, truncated, info = super().step(action)
        timestep = int(info["elapsed_steps"])


        return obs, reward, terminated, truncated, info
