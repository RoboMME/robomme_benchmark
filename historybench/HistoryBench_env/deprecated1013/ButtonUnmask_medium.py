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


@register_env("ButtonUnmask_medium", max_episode_steps=2000)
class ButtonUnmask_medium(BaseEnv):

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

        button_obb_1 = build_button(
            self,
            center_xy=(-0.2, 0.15),
            scale=1.5,
            generator=generator,
            name="button_left",
            randomize=True,
            randomize_range=(0.1, 0.1)
        )
        # Store first button before building second one
        self.button_left = self.button
        self.button_joint_1 = self.button_joint

    

        avoid = [button_obb_1]


         # 生成3个bins
        spawned_bins = []
        for i in range(5):
            try:
                bin_actor = spawn_random_bin(
                    self,
                    avoid=avoid,  # 使用当前避让清单，包含所有已生成的对象
                    region_center=[0.0, 0],
                    region_half_size=0.2,
                    min_gap=self.cube_half_size*3,  # bins需要更大的间隙，增加到6倍避免碰撞
                    name_prefix=f"bin_{i}",
                    max_trials=256,
                    generator=generator
                )
            except RuntimeError as e:
                break

            spawned_bins.append(bin_actor)
            # 将bin赋值给self.bin_0, self.bin_1等属性
            setattr(self, f"bin_{i}", bin_actor)
            # 将新生成的bin加入避让清单
            avoid.append(bin_actor)


        # 在每个bin下方生成3个动态cube（使用固定位置，颜色为红、绿、蓝）
        spawned_dynamic_cubes = []
        cube_colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]  # 红、绿、蓝
        color_names = ["red", "green", "blue"]

        # 使用 seed 随机打乱颜色顺序

        shuffle_indices = torch.randperm(len(cube_colors), generator=generator).tolist()
        cube_colors = [cube_colors[i] for i in shuffle_indices]
        color_names = [color_names[i] for i in shuffle_indices]

        # 存储 color_names 以便 RecordWrapper 访问
        self.color_names = color_names

        # 只为前3个bin生成cube
        for i, bin_actor in enumerate(spawned_bins[:3]):
            # 获取bin的位置
            bin_pos = bin_actor.pose.p
            if isinstance(bin_pos, torch.Tensor):
                bin_pos = bin_pos[0].detach().cpu().numpy()

            cube_position = [bin_pos[0], bin_pos[1]]
            # 使用固定位置生成cube，颜色为红、绿、蓝
            cube_actor = spawn_fixed_cube(
                self,
                position=cube_position,
                half_size=self.cube_half_size,
                color=cube_colors[i],  # 按顺序使用红、绿、蓝
                name_prefix=f"target_cube_{color_names[i]}",
                yaw=0.0,  # 不旋转
            )

            spawned_dynamic_cubes.append(cube_actor)
            # 将cube赋值给self.target_cube_red, self.target_cube_green, self.target_cube_blue等属性
            setattr(self, f"target_cube_{color_names[i]}", cube_actor)
            # 同时也用数字索引存储，方便访问
            setattr(self, f"target_cube_{i}", cube_actor)
            # 将新生成的cube加入避让清单
            avoid.append(cube_actor)

        # Randomly select two pairs of bin indices for swapping
        self.spawned_bins = spawned_bins
        self.other_bins = []
        for idx, bin_actor in enumerate(spawned_bins):
            other_list = [b for j, b in enumerate(spawned_bins) if j != idx]
            self.other_bins.append(other_list)
            setattr(self, f"other_bins_{idx}", other_list)
            setattr(self, f"otherbins_{idx}", other_list)
        self.otherbins = self.other_bins
        num_bins = len(spawned_bins)


        # First pair: randomly select two different bins
        perm1 = torch.randperm(num_bins, generator=generator)
        self.swap_pair1_idx1 = perm1[0].item()
        self.swap_pair1_idx2 = perm1[1].item()

        # Second pair: independently randomly select two different bins
        perm2 = torch.randperm(num_bins, generator=generator)
        self.swap_pair2_idx1 = perm2[0].item()
        self.swap_pair2_idx2 = perm2[1].item()

        # Third pair: independently randomly select two different bins
        perm3 = torch.randperm(num_bins, generator=generator)
        self.swap_pair3_idx1 = perm3[0].item()
        self.swap_pair3_idx2 = perm3[1].item()

        print(f"Swap pair 1: bin_{self.swap_pair1_idx1} <-> bin_{self.swap_pair1_idx2}")
        print(f"Swap pair 2: bin_{self.swap_pair2_idx1} <-> bin_{self.swap_pair2_idx2}")
        print(f"Swap pair 3: bin_{self.swap_pair3_idx1} <-> bin_{self.swap_pair3_idx2}")




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
        tasks = [
            {
                "func": lambda: is_button_pressed(self, obj=self.button_left),
                "name": "button press",
                "demonstration": False,
                "failure_func":None,
                "solve": lambda env, planner: solve_button(env, planner, obj=self.button_left),
            },
        
            {
                "func": (lambda: is_bin_pickup(self, obj=self.bin_0)),
                "name": "Pick up bin",
                "demonstration": False,
                "failure_func": lambda: is_any_bin_pickup(self, [self.bin_2,self.bin_1]),
                "solve": lambda env, planner: solve_pickup_bin(env, planner, obj=self.bin_0),
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

    def _get_other_bins_for_pair(self, idx_a: int, idx_b: int):
        """Return bins that are not part of the provided pair indices."""
        if not hasattr(self, "spawned_bins"):
            return []

        total_bins = len(self.spawned_bins)
        if idx_a >= total_bins or idx_b >= total_bins:
            return []

        # Prefer precomputed lists when available
        if hasattr(self, "otherbins") and idx_a < len(self.otherbins):
            other_candidates = [
                bin_actor
                for bin_actor in self.otherbins[idx_a]
                if bin_actor is not self.spawned_bins[idx_b]
            ]
            return other_candidates

        return [
            bin_actor
            for i, bin_actor in enumerate(self.spawned_bins)
            if i not in (idx_a, idx_b)
        ]


#HistoryBench
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):


        obs, reward, terminated, truncated, info = super().step(action)
        timestep = int(info["elapsed_steps"])
        
        #Lift and drop bin_0
        lift_and_drop_objects_back_to_original(
            self,
            obj=self.bin_0,
            start_step=0,
            end_step=20,
            cur_step=timestep,
        )
        lift_and_drop_objects_back_to_original(
            self,
            obj=self.bin_1,
            start_step=0,
            end_step=20,
            cur_step=timestep,
        )   
        lift_and_drop_objects_back_to_original(
            self,
            obj=self.bin_2,
            start_step=0,
            end_step=20,
            cur_step=timestep,
        ) 


        # Use randomly selected bin pairs for swapping
        other_bins_pair1 = self._get_other_bins_for_pair(
            self.swap_pair1_idx1, self.swap_pair1_idx2
        )
        swap_flat_two_lane(
            self,
            cube_a=self.spawned_bins[self.swap_pair1_idx1],
            cube_b=self.spawned_bins[self.swap_pair1_idx2],
            start_step=20,
            end_step=20+50,
            cur_step=timestep,
            lane_offset=0.1,
            smooth=True,
            keep_upright=True,
            other_cube=other_bins_pair1,  # Keep all other bins in place to prevent collision during swap
        )


        lift_and_drop_objectA_onto_objectB(
            self,
            obj_a=self.target_cube_0,
            obj_b=self.bin_0,
            start_step=20,
            end_step=70,
            cur_step=timestep,
        )
        lift_and_drop_objectA_onto_objectB(
            self,
            obj_a=self.target_cube_1,
            obj_b=self.bin_1,
            start_step=20,
            end_step=70,
            cur_step=timestep,
        )
        lift_and_drop_objectA_onto_objectB(
            self,
            obj_a=self.target_cube_2,
            obj_b=self.bin_2,
            start_step=20,
            end_step=70,
            cur_step=timestep,
        )


        return obs, reward, terminated, truncated, info
