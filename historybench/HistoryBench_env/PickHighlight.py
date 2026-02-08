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
from .util import subgoal_language
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


@register_env("PickHighlight", max_episode_steps=2000)
class PickHighlight(BaseEnv):

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
        'spawn': 6,
        "pickup": 3
    }

    config_easy = {
        'spawn': 3,
        "pickup": 1
    }

    config_medium = {
        'spawn': 4,
        "pickup": 2
    }

    # 组合成一个字典
    configs = {
        'hard': config_hard,
        'easy': config_easy,
        'medium': config_medium
    }


    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0,HistoryBench_seed=0,HistoryBench_video_episode=None,HistoryBench_video_path=None,
                     **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.use_demonstrationwrapper=False
        self.demonstration_record_traj=False
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
        self.generator = torch.Generator()
        self.generator.manual_seed(self.HistoryBench_seed)

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
            # Determine difficulty based on seed % 3
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
        self.generator.manual_seed(self.HistoryBench_seed)
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()


        button_obb = build_button(
            self,
            center_xy=(-0.2, 0),
            scale=1.5,
            generator=self.generator,
        )
        avoid = [button_obb]

        self.all_cubes = []  # 保存所有 cube 对象
        self.all_cube_names = []
        self.all_cube_colors = []

        # 可用颜色列表
        available_colors = [
            {"color": (1, 0, 0, 1), "name": "red"},
            {"color": (0, 0, 1, 1), "name": "blue"},
            {"color": (0, 1, 0, 1), "name": "green"}
        ]

        # 根据难度获取需要生成的cube数量
        num_cubes_to_spawn = self.configs[self.difficulty]['spawn']

        # 生成指定数量的cube，每个cube颜色随机选择
        for cube_idx in range(num_cubes_to_spawn):
            # 随机选择一个颜色
            color_choice_idx = torch.randint(0, len(available_colors), (1,), generator=self.generator).item()
            chosen_color = available_colors[color_choice_idx]

            try:
                cube = spawn_random_cube(
                    self,
                    color=chosen_color["color"],
                    avoid=avoid,
                    include_existing=False,
                    include_goal=False,
                    region_center=[-0.1, 0],
                    region_half_size=0.2,
                    half_size=self.cube_half_size,
                    min_gap=self.cube_half_size*2,
                    random_yaw=True,
                    name_prefix=f"cube_{chosen_color['name']}_{cube_idx}",
                    generator=self.generator,
                )

                cube_name = f"cube_{chosen_color['name']}_{cube_idx}"

                # Add cube immediately after successful creation
                self.all_cubes.append(cube)
                self.all_cube_names.append(cube_name)
                self.all_cube_colors.append(chosen_color["name"])
                setattr(self, cube_name, cube)
                avoid.append(cube)

            except RuntimeError as e:
                print(f"生成 cube {cube_idx} ({chosen_color['name']}) 失败：{e}")
                break

        print(f"Generated {len(self.all_cubes)} cubes total")



         # Randomly select one cube from all available cubes as the target
        target_cube_indices = torch.randperm(len(self.all_cubes), generator=self.generator)[:self.configs[self.difficulty]['pickup']]

        self.target_cubes = [self.all_cubes[idx] for idx in target_cube_indices]
        self.target_cube_names = [self.all_cube_names[idx] for idx in target_cube_indices]
        self.target_cube_colors = [self.all_cube_colors[idx] for idx in target_cube_indices]
        self.target_labels = [
            (color or name or "target") 
            for color, name in zip(self.target_cube_colors, self.target_cube_names)
        ]
        # 记录每个目标方块被抓起的次数；所有目标方块计数均大于1才算成功
        self.target_cube_pickup_counts = {name: 0 for name in self.target_cube_names}

        # 定义任务列表，每个任务包含一个带函数、名称、demonstration 标志和可选 failure_func 的字典
        tasks = []
        target_label = getattr(self, "target_cube_color", None) or getattr(
            self, "target_cube_name", None
        ) or getattr(self, "target_label", None) or "target"
        self.target_label = target_label

        tasks.append({
            "func": lambda: is_button_pressed(self, obj=self.button),
                "name": "press the button",
                "subgoal_segment":"press the button at <>",
                "demonstration": False,
                "failure_func":is_any_obj_pickup(self,[cube for cube in self.all_cubes]),
                "solve": lambda env, planner:solve_button(env, planner, obj=self.button),
                 "segment":self.cap_link,
            })
        # 每个目标方块各挑一次，循环内的 lambda 显式捕获当前 cube，避免闭包引用同一对象
        num_targets = len(self.target_cubes)
        for cube_idx, cube in enumerate(self.target_cubes):
                # 如果只有一个目标cube，不显示序号
                if num_targets == 1:
                    task_name = f"pick up the highlighted cube, which is {self.target_labels[cube_idx]}"
                    task_subgoal = f"pick up the highlighted cube at <>, which is {self.target_labels[cube_idx]}"
                else:
                    task_name = subgoal_language.get_subgoal_with_index(cube_idx, "pick up the {idx} highlighted cube, which is {color}", color=self.target_labels[cube_idx])
                    task_subgoal = subgoal_language.get_subgoal_with_index(cube_idx, "pick up the {idx} highlighted cube at <>, which is {color}", color=self.target_labels[cube_idx])

                tasks.append({
                    "func": (lambda c=cube: is_any_obj_pickup_flag_currentpickup(self, objects=[c])),
                    "name": task_name,
                    "subgoal_segment": task_subgoal,
                    "demonstration": False,
                    "failure_func": lambda idx=cube_idx:
                        [is_any_obj_pickup(self,[cube for cube in self.all_cubes if cube not in self.target_cubes] ),
                       ],
                    "solve": lambda env, planner, c=cube: solve_pickup(env, planner, obj=c),
                    "segment":cube,
                })
                if cube_idx!=num_targets-1:
                    tasks.append({
                        "func": (lambda :is_obj_dropped_currentpickup(self,self.target_cubes)),
                        "name": f"place the cube onto the table",
                        "subgoal_segment":"place the cube onto the table",
                        "demonstration": False,
                        "failure_func": lambda idx=cube_idx:
                        [ is_any_obj_pickup(self,[cube for cube in self.all_cubes if cube not in self.target_cubes] ),
                           ],
                        "solve": lambda env, planner, c=cube: [solve_putdown_whenhold(env, planner, release_z=0.01),
                                                        # solve_pickup(env, planner, obj=c),
                                                        # solve_putdown_whenhold(env, planner, obj=c,release_z=0.01)#测试用
                                                        ],
                        "segment":None,
                    })
            
        


        # 存储任务列表供RecordWrapper使用
        self.task_list = tasks            


        # 记录用于恢复的 pickup 相关任务索引和条目
        self.recovery_pickup_indices, self.recovery_pickup_tasks = task4recovery(self.task_list)
        if self.historybench_failure_recovery:
            # Only inject an intentional failed grasp when recovery mode is enabled
            self.fail_grasp_task_index = inject_fail_grasp(
                self.task_list,
                generator=self.generator,
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
        # 保留之前的失败状态（一旦失败，永远失败）
        if not hasattr(self, 'failureflag') or self.failureflag is None:
            self.failureflag = torch.tensor([False])
        previous_failure = bool(self.failureflag.detach().cpu().item()) if isinstance(self.failureflag, torch.Tensor) else False
        # 如果之前已经失败，不要重置，保持失败状态；否则重置
        if previous_failure:
            # 保持失败状态，不重置
            pass
        else:
            self.failureflag = torch.tensor([False])



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
            
        # 使用封装的序列任务检查函数
        all_tasks_completed, current_task_name, task_failed ,self.current_task_specialflag= sequential_task_check(self, self.task_list,allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep)

        if task_failed:
            self.failureflag = torch.tensor([True])
            print(f"Task failed: {current_task_name}")
        
        # 如果之前已经失败，保持失败状态
        if previous_failure:
            self.failureflag = torch.tensor([True])


        #############上升沿检测 必须放在fail检测之前
        target_cubes = getattr(self, "target_cubes", [])
        target_cube_names = getattr(self, "target_cube_names", [])

        if target_cubes and not hasattr(self, "target_cube_pickup_counts"):
            self.target_cube_pickup_counts = {name: 0 for name in target_cube_names}
            self.target_cube_pickup_active = {name: False for name in target_cube_names}


        if target_cubes and not hasattr(self, "target_cube_pickup_active"):
            self.target_cube_pickup_active = {name: False for name in target_cube_names}

        # 仅在某个方块从“未抓取”变为“抓取”时计数，避免同一次抓取在多帧内重复累计
        for cube, name in zip(target_cubes, target_cube_names):
            pickup_tensor = is_obj_pickup(self, cube)
            if isinstance(pickup_tensor, torch.Tensor):
                picked_now = bool(pickup_tensor.detach().cpu().any())
            else:
                picked_now = bool(pickup_tensor)

            was_picked = self.target_cube_pickup_active.get(name, False)
            if picked_now and not was_picked:
                self.target_cube_pickup_counts[name] = (
                    self.target_cube_pickup_counts.get(name, 0) + 1
                )
            self.target_cube_pickup_active[name] = picked_now

        pickup_counts = getattr(self, "target_cube_pickup_counts", {})
        counts_satisfied = (
            len(pickup_counts) > 0
            and all(count >= 1 for count in pickup_counts.values())
        )
        #############上升沿检测 必须放在fail检测之前


        # 任何情况下只要都 pick 至少一次即刻成功（计数记录离散抓取事件）
        if counts_satisfied:
            self.successflag = torch.tensor([True])
       
       #planner全执行完仍然不成功fail
        if all_tasks_completed and not counts_satisfied:
            self.failureflag = torch.tensor([True])
            print(f"Pickup counts not satisfied: {pickup_counts}")

        if self.failureflag == torch.tensor([True]):
            pass
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


      
        timestep = self.elapsed_steps
        target_cubes = getattr(self, "target_cubes", [])

      

        highlight_count = min(self.configs[self.difficulty]["pickup"], len(target_cubes))
        for i in range(highlight_count):
            highlight_obj(
                self,
                target_cubes[i],
                start_step=10,
                end_step=100,
                cur_step=timestep,
            )
        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info
