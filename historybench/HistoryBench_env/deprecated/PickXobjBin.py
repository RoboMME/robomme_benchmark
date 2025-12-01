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


@register_env("PickXobjBin", max_episode_steps=2000)
class PickXobjBinEnv(BaseEnv):

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
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # 创建generator用于所有随机化
        generator = torch.Generator()
        generator.manual_seed(self.HistoryBench_seed)

        button_obb = build_button(
            self,
            center_xy=(-0.3, 0),
            scale=1.5,
            generator=generator,
        )
        avoid = [button_obb]

        # 创建带正方形洞的正方形板子
        x_var = torch.rand(1, generator=generator).item() * 0.2 - 0.2  # [-0.25, 0.25]
        y_var = torch.rand(1, generator=generator).item() * 0.4 - 0.2  # [-0.25, 0.25]
        z_rot_deg = (torch.rand(1, generator=generator).item() * 40.0 - 20.0)  # [-20, 20] degrees
        z_rot_rad = torch.deg2rad(torch.tensor(z_rot_deg))
        # Create rotation quaternion for z-axis rotation
        rot_mat = euler_angles_to_matrix(torch.tensor([[0.0, 0.0, z_rot_rad]]), convention="XYZ")
        rot_quat = matrix_to_quaternion(rot_mat)[0]  # [w, x, y, z]
        self.board_with_hole = build_board_with_hole(
            self,
            board_side=0.1,  # 正方形板子的边长
            hole_side=0.08,   # 正方形洞的边长，稍大于cube尺寸以便cube能通过
            thickness=0.05,   # 板子厚度
            position=[0.15 + x_var, 0.0 + y_var, 0.0],  # 板子位置
            rotation_quat=rot_quat.tolist(),  # z轴旋转
            name="board_with_hole"
        )
        avoid += [self.board_with_hole]


        self.all_cubes = []  # 保存所有 cube 对象

        self.red_cubes = []
        self.red_cube_names = []
        total_cubes = 10
        for idx in range(total_cubes):
            try:
                red_cube = spawn_random_cube(
                    self,
                    color=(1, 0, 0, 1),
                    avoid=avoid,
                    include_existing=False,
                    include_goal=False,
                    region_center=[-0.1, 0],
                    region_half_size=0.3,
                    half_size=self.cube_half_size,
                    min_gap=self.cube_half_size,
                    random_yaw=True,
                    name_prefix=f"cube_red_{idx}",
                    generator=generator,
                )
            except RuntimeError as e:
                print(f"生成红色 cube {idx} 失败：{e}")
                break

            self.all_cubes.append(red_cube)
            self.red_cubes.append(red_cube)
            cube_name = f"cube_red_{idx}"
            self.red_cube_names.append(cube_name)
            setattr(self, cube_name, red_cube)
            avoid.append(red_cube)

        print(f"Generated {len(self.red_cubes)} red cubes total")

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


        # 动态生成 N 次 pickup-drop 循环的任务列表
        tasks = []
        generator = torch.Generator()
        generator.manual_seed(self.HistoryBench_seed)
        available = len(self.red_cubes)

        if available == 0:
            print("No red cubes generated; marking episode as failure")
            self.failureflag = torch.tensor([True])
            return {
                "success": self.successflag,
                "fail": self.failureflag,
            }


        for cube_idx in range(self.num_repeats):
            cube = self.red_cubes[cube_idx]
            cube_name = self.red_cube_names[cube_idx]
            tasks.append({
                "func": (lambda c=cube: is_obj_pickup(self, obj=c)),
                "name": f"Pick up red cube",
                "demonstration": False,
                "failure_func": None,
                "solve": lambda env, planner, c=cube: solve_pickup(env, planner, obj=c),
            })

            tasks.append({
                "func": (lambda c=cube: is_obj_dropped_onto_delete(self, obj=c, target=self.board_with_hole)),
                "name": f"Place red cube in bin",
                "demonstration": False,
                "failure_func": None,
                "solve": lambda env, planner, c=cube: solve_putonto_whenhold_binspecial(env, planner, obj=c, target=self.board_with_hole),
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
                #"solve": lambda env, planner: [solve_hold_obj(env, planner, 300),solve_button(env, planner, obj=self.button)]
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
