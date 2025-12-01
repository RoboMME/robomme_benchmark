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


@register_env("VideoRepickXtimes_hard", max_episode_steps=2000)
class VideoRepickXtimes_hard(BaseEnv):

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

    #HistoryBench
    subgoal_repeat = 1
    subgoal_timestep = 0
    subgoal_list = [
        'pickup',
        'drop',
        'static(success/failure time limited)'
    ]  # will be expanded by subgoal_repeat
    # HistoryBench
    success_timer=0
    failure_timer=0

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


        button_obb = build_button(
            self,
            center_xy=(-0.2, 0),
            scale=1.5,
            generator=generator,
        )
        avoid = [button_obb]


        # 定义三组颜色：红色、蓝色、绿色
        color_groups = [
            {"color": (1, 0, 0, 1), "name": "red"},      # 红色
            {"color": (0, 0, 1, 1), "name": "blue"},     # 蓝色
            {"color": (0, 1, 0, 1), "name": "green"}     # 绿色
        ]

        self.all_cubes = []  # 保存所有 cube 对象
        cube_names = []  # 保存所有 cube 的名称
        cube_color_labels = []  # 记录每个 cube 的颜色标签

        for group_idx, group_info in enumerate(color_groups):
            color = group_info["color"]
            color_name = group_info["name"]

            for i in range(3):
                try:
                    cube = spawn_random_cube(
                        self,
                        color=color,
                        avoid=avoid,
                        include_existing=False,
                        include_goal=False,
                        region_center=[-0.1, 0],
                        region_half_size=0.25,
                        half_size=self.cube_half_size,
                        min_gap=self.cube_half_size,
                        random_yaw=True,
                        name_prefix=f"cube_{color_name}_{i}",
                        generator=generator
                    )
                   # print(self.cube_half_size)
                except RuntimeError as e:
                    print(f"生成 {color_name} cube {i} 失败：{e}")
                    break

                self.all_cubes.append(cube)
                cube_name = f"cube_{color_name}_{i}"
                cube_names.append(cube_name)
                cube_color_labels.append(color_name)
                # 将cube赋值给对应的属性，例如 self.cube_red_0, self.cube_blue_1 等
                setattr(self, cube_name, cube)
                avoid.append(cube)

        # 预设目标及交换相关的成员变量
        self.target_cube = None
        self.target_cube_name = None
        self.target_cube_color = None
        self.non_target_cubes = []
        self.swap_groups = []


        generator = torch.Generator()
        generator.manual_seed(self.HistoryBench_seed)

        target_idx = torch.randint(0, len(self.all_cubes), (1,), generator=generator).item()
        self.target_cube = self.all_cubes[target_idx]
        self.target_cube_name = cube_names[target_idx]
        self.target_cube_color = cube_color_labels[target_idx]
        print(
            f"Selected target cube: color={self.target_cube_color}, name={self.target_cube_name}"
        )

        # 保存所有非目标 cube，用于失败检测等用途
        self.non_target_cubes = [
            cube for idx, cube in enumerate(self.all_cubes) if idx != target_idx
        ]

        perm = torch.randperm(len(self.all_cubes), generator=generator)
        swap_pairs_indices = []
        for pair_idx in range(3):
            perm_idx_a = 2 * pair_idx
            perm_idx_b = 2 * pair_idx + 1
            swap_idx_a = perm[perm_idx_a].item()
            swap_idx_b = perm[perm_idx_b].item()
            swap_pairs_indices.append((swap_idx_a, swap_idx_b))

        moving_indices = {idx for pair in swap_pairs_indices for idx in pair}
        static_cubes = [
            cube for idx, cube in enumerate(self.all_cubes) if idx not in moving_indices
        ]

        for pair_idx, (swap_idx_a, swap_idx_b) in enumerate(swap_pairs_indices, start=1):
            cube_a = self.all_cubes[swap_idx_a]
            cube_b = self.all_cubes[swap_idx_b]
            group_info = {
                "cube_a": cube_a,
                "cube_b": cube_b,
                "other_cubes": static_cubes,
            }
            self.swap_groups.append(group_info)
            print(
                f"Swap group {pair_idx}: "
                f"{group_info['cube_a']} <-> {group_info['cube_b']}"
            )

        self.num_repeats = torch.randint(3, 4, (1,), generator=generator).item()



    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
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

        # 定义任务列表，每个任务包含一个带函数、名称、demonstration 标志和可选 failure_func 的字典
        tasks = []
        target_label = getattr(self, "target_cube_color", None)
        if not target_label:
            target_label = getattr(self, "target_cube_name", "target")
        tasks.append({
                        "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=220),
                        "name": "Static",
                        "demonstration": True,
                        "failure_func": None,
                        "solve": lambda env, planner: solve_hold_obj(env, planner, static_steps=220),
        })
        for i in range (self.num_repeats):
            tasks.append({
                "func": (lambda : is_obj_pickup(self, obj=self.target_cube)),
                "name": f"Pick up {target_label} cube",
                "demonstration": False,
                "failure_func": lambda: is_any_obj_pickup(self, self.non_target_cubes),
                "solve": lambda env, planner: solve_pickup(env, planner, obj=self.target_cube),
            })
            tasks.append({
                "func": (lambda: is_obj_dropped(self,obj=self.target_cube)),
                "name": f"Drop {target_label} cube",
                "demonstration": False,
                "failure_func": lambda: is_any_obj_pickup(self, self.non_target_cubes),
                "solve": lambda env, planner: solve_putdown_whenhold(env, planner, obj=self.target_cube)
             })
        tasks.append({
                "func": lambda: timewindow(
                    self,
                    lambda: is_button_pressed(self, obj=self.button),
                    min_steps=1,
                    max_steps=200,
                    timewindow_timer=0,
                ),
                "name": "button press",
                "demonstration": False,
                "failure_func":False,
                "solve": lambda env, planner: solve_button(env, planner, obj=self.button),
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

        if self.target_cube is not None:
            highlight_obj(self, self.target_cube, start_step=0, end_step=30, cur_step=timestep)

        if getattr(self, "swap_groups", None):
            swap_start = 20 + 50 * 2
            swap_end = 20 + 50 * 3
            for group in self.swap_groups:
                other_bins = group["other_cubes"]
                swap_flat_two_lane(
                    self,
                    cube_a=group["cube_a"],
                    cube_b=group["cube_b"],
                    start_step=swap_start,
                    end_step=swap_end,
                    cur_step=timestep,
                    lane_offset=0.1,
                    smooth=True,
                    keep_upright=True,
                    other_cube=other_bins,
                )
        return obs, reward, terminated, truncated, info
