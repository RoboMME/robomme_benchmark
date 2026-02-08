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

from .util import *
from .util.evaluate import static_check
from .util.object_generation import spawn_fixed_cube, build_board_with_hole
from .util import reset_panda
from .util.difficulty import normalize_historybench_difficulty


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


@register_env("VideoUnmask", max_episode_steps=2000)
class VideoUnmask(BaseEnv):

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

    config_hard = {
    'bin':15,
    "pick":2,
    }

    config_easy = {
    'bin':3,
    "pick":1,
    }

    config_medium = {
    'bin':5,
    "pick":1,
    }

    # 组合成一个字典
    configs = {
        'hard': config_hard,
        'easy': config_easy,
        'medium': config_medium
    }


    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0,HistoryBench_seed=0,HistoryBench_video_episode=None,HistoryBench_video_path=None,
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

        self.HistoryBench_seed = HistoryBench_seed

        self.historybench_failure_recovery = bool(
            kwargs.pop("historybench_failure_recovery", False)
        )
        self.historybench_failure_recovery_mode = kwargs.pop(
            "historybench_failure_recovery_mode", None
        )
        if isinstance(self.historybench_failure_recovery_mode, str):
            self.historybench_failure_recovery_mode = (
                self.historybench_failure_recovery_mode.lower()
            )

        normalized_historybench_difficulty = normalize_historybench_difficulty(
            kwargs.pop("HistoryBench_difficulty", None)
        )
        if normalized_historybench_difficulty is not None:
            self.difficulty = normalized_historybench_difficulty
        else:
            seed_mod = HistoryBench_seed % 3
            if seed_mod == 0:
                self.difficulty = "easy"
            elif seed_mod == 1:
                self.difficulty = "medium"
            else:  # seed_mod == 2
                self.difficulty = "hard"

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

        avoid=[]


         # 生成3个bins
        self.spawned_bins = []
        for i in range(self.configs[self.difficulty]['bin']):
            try:
                bin_actor = spawn_random_bin(
                    self,
                    avoid=avoid,  # 使用当前避让清单，包含所有已生成的对象
                    region_center=[0, 0],
                    region_half_size=0.2,
                    min_gap=self.cube_half_size*2,  # bins需要更大的间隙，增加到6倍避免碰撞
                    name_prefix=f"bin_{i}",
                    max_trials=256,
                    generator=generator
                )
                print(f"Spawned bin_{i} at position {bin_actor.pose.p}")
            except RuntimeError as e:
                break

            self.spawned_bins.append(bin_actor)
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
        for i, bin_actor in enumerate(self.spawned_bins[:3]):
            # 获取bin的位置
            bin_pos = bin_actor.pose.p
            if isinstance(bin_pos, torch.Tensor):
                bin_pos = bin_pos[0].detach().cpu().numpy()

            cube_position = [bin_pos[0], bin_pos[1]]
            # 使用固定位置生成cube，颜色为红、绿、蓝
            cube_actor = spawn_fixed_cube(
                self,
                position=cube_position,
                half_size=self.cube_half_size/1.2,
                color=cube_colors[i],  # 按顺序使用红、绿、蓝
                name_prefix=f"target_cube_{color_names[i]}",
                yaw=0.0,  # 不旋转
                dynamic=True
            )

            spawned_dynamic_cubes.append(cube_actor)
            # 将cube赋值给self.target_cube_red, self.target_cube_green, self.target_cube_blue等属性
            setattr(self, f"target_cube_{color_names[i]}", cube_actor)
            # 同时也用数字索引存储，方便访问
            setattr(self, f"target_cube_{i}", cube_actor)
            # 将新生成的cube加入避让清单
            avoid.append(cube_actor)

        tasks = [
             {
                        "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=64),
                        "name": "static",
                        "subgoal_segment": "static",
                        "demonstration": True,
                        "failure_func": None,
                        "solve": lambda env, planner: solve_hold_obj(env, planner, static_steps=64),
                        },
            

            
            {
                "func": (lambda: is_bin_pickup(self, obj=self.bin_0)),
                "name": f"pick up the container that hides the {self.color_names[0]} cube",
                "subgoal_segment":f"pick up the container at <> that hides the {self.color_names[0]} cube",
                "demonstration": False,
                "failure_func": lambda: is_any_bin_pickup(self,[bin for bin in self.spawned_bins if bin != self.bin_0]),
                "solve": lambda env, planner: solve_pickup_bin(env, planner, obj=self.bin_0),
                "segment":self.bin_0,
            },]
        if self.configs[self.difficulty]['pick']>1:
            tasks.append({
                    "func": (lambda: is_bin_putdown(self, obj=self.bin_0)),
                    "name": "put down the container",
                    "subgoal_segment":"put down the container",
                    "demonstration": False,
                    "failure_func": lambda:is_any_bin_pickup(self,[bin for bin in self.spawned_bins if bin != self.bin_0]),
                    "solve": lambda env, planner: solve_putdown_whenhold(env, planner),

                })
            tasks.append(
                {
                    "func": (lambda: is_bin_pickup(self, obj=self.bin_1)),
                    "name": f"pick up the container that hides the {self.color_names[1]} cube",
                    "subgoal_segment":f"pick up the container at <> that hides the {self.color_names[1]} cube",
                    "demonstration": False,
                    "failure_func": lambda: is_any_bin_pickup(self,[bin for bin in self.spawned_bins if bin != self.bin_1]),
                    "solve": lambda env, planner: solve_pickup_bin(env, planner,obj=self.bin_1),
                    "segment":self.bin_1,
                })
        



        # 存储任务列表供RecordWrapper使用
        self.task_list = tasks

        # 记录用于恢复的 pickup 相关任务索引和条目
        self.recovery_pickup_indices, self.recovery_pickup_tasks = task4recovery(self.task_list)
        if self.historybench_failure_recovery:
            # Only inject an intentional failed grasp when recovery mode is enabled
            self.fail_grasp_task_index = inject_fail_grasp(
                self.task_list,
                generator=generator,
                mode=self.historybench_failure_recovery_mode,
            )
        else:
            self.fail_grasp_task_index = None

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos=reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)


    def _get_obs_extra(self, info: Dict):
        return dict()



    def evaluate(self,solve_complete_eval=False):
        self.successflag=torch.tensor([False])
        self.failureflag = torch.tensor([False])
        test=[bin for bin in self.spawned_bins if bin != self.bin_0]
       

        # 使用封装的序列任务检查函数
        if(self.use_demonstrationwrapper==False):#record时候planner结束再改变subgoal
            if solve_complete_eval==True:
                allow_subgoal_change_this_timestep=True
            else:
                allow_subgoal_change_this_timestep=False
        else:#demonstration时候video需要call evaluate(solve_complete_eval) video结束在demonstrationwrapper里面改变flag
            if solve_complete_eval==True or self.demonstration_record_traj==False:
                allow_subgoal_change_this_timestep=True
            else:
                allow_subgoal_change_this_timestep=False
        all_tasks_completed, current_task_name, task_failed,self.current_task_specialflag = sequential_task_check(self, self.task_list,allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep)

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


        timestep = self.elapsed_steps        
        #Lift and drop bins (bin_0 to bin_4 if they exist)
        for i in range(15):
            bin_attr = f"bin_{i}"
            if hasattr(self, bin_attr):
                lift_and_drop_objects_back_to_original(
                    self,
                    obj=getattr(self, bin_attr),
                    start_step=0,
                    end_step=32*2,
                    cur_step=timestep,
                ) 



        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info
