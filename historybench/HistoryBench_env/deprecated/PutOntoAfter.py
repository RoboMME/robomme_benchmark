from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

#HistoryBench
import matplotlib.pyplot as plt
import h5py
import random
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)

from util import *
from util.evaluate import static_check
from util.object_generation import spawn_fixed_cube, build_board_with_hole


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


@register_env("PutOntoAfter", max_episode_steps=2000)
class PutOntoAfter(BaseEnv):

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

        # 使用 seed 随机确定需要重复的次数 (1-5)
        generator = torch.Generator()
        generator.manual_seed(HistoryBench_seed)
        self.num_repeats = torch.randint(1, 6, (1,), generator=generator).item()
        print(f"Task will repeat {self.num_repeats} times (pickup-drop cycles)")

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
        generator = torch.Generator()
        generator.manual_seed(self.HistoryBench_seed)

        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()



        self.goal_site = actors.build_sphere(
            self.scene,
            radius=0.01,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0,0,0]),
        )
        self._hidden_objects.append(self.goal_site)

        # Create manual OBB for goal_site collision avoidance (since it has no collision mesh)
        goal_site_obb = (
            np.array([0.0, 0.0], dtype=np.float64),  # center at origin
            np.eye(2, dtype=np.float64),  # axes (identity for axis-aligned)
            np.array([0.1, 0.1], dtype=np.float64)  # half-sizes (radius of sphere)
        )

        button_obb = build_button(
            self,
            center_xy=(-0.3, 0),
            scale=1.5,
            generator=generator,
        )
        avoid = [button_obb]
        
        avoid = [button_obb, goal_site_obb]

        self.all_cubes = []  # 保存所有 cube 对象

        # 生成 1 个红色 cube
        try:
            red_cube = spawn_random_cube(
                self,
                color=(1, 0, 0, 1),  # 红色
                avoid=avoid,
                include_existing=False,
                include_goal=False,
                region_center=[-0.1, 0],
                region_half_size=0.3,
                half_size=self.cube_half_size,
                min_gap=self.cube_half_size,
                random_yaw=True,
                name_prefix="cube_red",
                generator=generator
            )
            self.all_cubes.append(red_cube)
            self.cube_red = red_cube
            avoid.append(red_cube)
            print("Generated 1 red cube")
        except RuntimeError as e:
            print(f"生成红色 cube 失败：{e}")

        # 生成 10 个灰色 cubes
        for i in range(3):
            try:
                gray_cube = spawn_random_cube(
                    self,
                    color=(0.5, 0.5, 0.5, 1),  # 灰色
                    avoid=avoid,
                    include_existing=False,
                    include_goal=False,
                    region_center=[-0.1, 0],
                    region_half_size=0.3,
                    half_size=self.cube_half_size,
                    min_gap=self.cube_half_size,
                    random_yaw=True,
                    name_prefix=f"cube_gray_{i}",
                    generator=generator
                )
                self.all_cubes.append(gray_cube)
                setattr(self, f"cube_gray_{i}", gray_cube)
                avoid.append(gray_cube)
            except RuntimeError as e:
                print(f"生成灰色 cube {i} 失败：{e}")
                break

        print(f"Generated {len(self.all_cubes)} cubes total (1 red + {len(self.all_cubes)-1} gray)")

     
        # 将红色 cube 设为 target
        self.target_cube = self.cube_red
        self.target_cube_name = "cube_red"
        print(f"Target cube: {self.target_cube_name}")

        # 所有非目标 cube（灰色）列表，用于失败检测
        self.non_target_cubes = [cube for cube in self.all_cubes if cube is not self.target_cube]

        # 生成targets
        self.targets = []
        for i in range(4):
            try:
                target = spawn_random_target(
                    self,
                    avoid=avoid,  # 使用当前避让清单，包含所有已生成的cubes
                    include_existing=False,  # 手动维护清单
                    include_goal=False,  # 手动维护清单
                    region_center=[-0.1, 0],
                    region_half_size=0.3,
                    radius=self.cube_half_size*2,  # 使用radius而不是half_size
                    thickness=0.005,  # target的厚度
                    min_gap=self.cube_half_size*2,  # 与cube相同的间隙要求
                    name_prefix=f"target_{i}",
                    generator=generator
                )
            except RuntimeError as e:
                print(f"第 {i + 1} 个target采样失败：{e}")
                break

            self.targets.append(target)
            # 将target赋值给self.target_0, self.target_1等属性
            setattr(self, f"target_{i}", target)
            # 将新生成的target加入避让清单
            avoid.append(target)

        # 将除 target_1 外的所有 targets 放入列表
        self.targets_not_true = [t for i, t in enumerate(self.targets) if i != 2]



    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Randomly set flag to True or False
            self.flag = torch.rand(1).item() < 0.5

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

    def _get_obs_extra(self, info: Dict):
        return dict()



    def evaluate(self,solve_complete_eval=False):
        self.successflag=torch.tensor([False])
        self.failureflag = torch.tensor([False])

        tasks = []
        if self.flag==True:
            tasks.append({
                    "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                    "name": "Pick up red cube",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: solve_pickup(env, planner, obj=self.target_cube),
                })
            tasks.append({
                    "func": (lambda: is_obj_dropped_onto(self,obj=self.target_cube,target=self.target_0)),
                    "name": "Drop cube onto target",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: solve_putonto_whenhold(env, planner, obj=self.target_cube,target=self.target_0),
            })
        tasks.append({
                "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                "name": "Pick up red cube",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: solve_pickup(env, planner, obj=self.target_cube),
            })
        tasks.append({
                "func": (lambda: is_obj_dropped_onto(self,obj=self.target_cube,target=self.target_1)),
                "name": "Drop cube onto target",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: solve_putonto_whenhold(env, planner, obj=self.target_cube,target=self.target_1),
         })
        tasks.append({
                "func": (lambda: is_button_pressed(self,obj=self.button)),
                "name": "Button Press",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: solve_button(env, planner, self.button),
         })
        tasks.append({
                "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                "name": "Pick up red cube",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: solve_pickup(env, planner, obj=self.target_cube),
            })
        tasks.append({
                "func": (lambda: is_obj_dropped_onto(self,obj=self.target_cube,target=self.target_2)),
                "name": "Drop cube onto target",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: solve_putonto_whenhold(env, planner, obj=self.target_cube,target=self.target_2),
         })
        tasks.append({
                "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                "name": "Pick up red cube",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: solve_pickup(env, planner, obj=self.target_cube),
            })
        tasks.append({
                "func": (lambda: is_obj_dropped_onto(self,obj=self.target_cube,target=self.goal_site)),
                "name": "Drop cube",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: solve_putonto_whenhold(env, planner, obj=self.target_cube,target=self.goal_site),
         })
        


        tasks.append({
                "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                "name": "Pick up red cube",
                "demonstration": False,
                "failure_func":lambda: is_any_obj_pickup(self, self.non_target_cubes),
                "solve": lambda env, planner: solve_pickup(env, planner, obj=self.target_cube),
            })
        tasks.append({
                "func": (lambda: is_obj_dropped_onto(self,obj=self.target_cube,target=self.target_2)),
                "name": "Drop cube onto target",
                "demonstration": False,
                "failure_func": (lambda: is_obj_dropped_onto_any(self,obj=self.target_cube,target=self.targets_not_true)),
                "solve": lambda env, planner: [solve_putonto_whenhold(env, planner, obj=self.target_cube,target=self.target_2)
                                                ,solve_liftup_Xdistance(env,planner,0.2)
                ]
                                               
         })



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

        # 如果任务失败，立即标记失败
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
        # timestep = int(info["elapsed_steps"])
        #
        # highlight_obj(self,self.target_cube, start_step=0, end_step=100, cur_step=timestep)

        return obs, reward, terminated, truncated, info
