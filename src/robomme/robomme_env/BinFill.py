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
from mani_skill.utils.structs import Actor, Link
#Robomme
import matplotlib.pyplot as plt
import random
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from .utils import *
from .utils.subgoal_evaluate_func import static_check
from .utils.object_generation import spawn_fixed_cube, build_board_with_hole
from .utils import reset_panda
from .utils import subgoal_language
from .utils.difficulty import normalize_robomme_difficulty

from ..logging_utils import logger


@register_env("BinFill")
class BinFill(BaseEnv):

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

    config_easy = {
    'color': 1, 
    'spawn_cubes':[4,6],
    "put_in_color":[1,1],
    "put_in_numbers":[1,3]
    }

    config_medium = {
    'color': 2, 
    'spawn_cubes':[6,8],
    "put_in_color":[1,2],
    "put_in_numbers":[2,4]
    }


    config_hard = {
    'color': 3, 
    'spawn_cubes':[8,10],
    "put_in_color":[2,3],
    "put_in_numbers":[3,5]
    }




    # Combine into a dictionary
    configs = {
        'hard': config_hard,
        'easy': config_easy,
        'medium': config_medium
    }

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0,seed=0,Robomme_video_episode=None,Robomme_video_path=None,
                     **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.use_demonstrationwrapper=False
        self.demonstration_record_traj=False
        normalized_robomme_difficulty = normalize_robomme_difficulty(
            kwargs.pop("difficulty", None)
        )
        self.robomme_failure_recovery = bool(
            kwargs.pop("robomme_failure_recovery", False)
        )
        self.robomme_failure_recovery_mode = kwargs.pop(
            "robomme_failure_recovery_mode", None
        )
        if isinstance(self.robomme_failure_recovery_mode, str):
            self.robomme_failure_recovery_mode = self.robomme_failure_recovery_mode.lower()

        if normalized_robomme_difficulty is not None:
            self.difficulty = normalized_robomme_difficulty
        else:
            # Determine difficulty based on seed % 3
            seed_mod = seed % 3
            if seed_mod == 0:
                self.difficulty = "easy"
            elif seed_mod == 1:
                self.difficulty = "medium"
            else:  # seed_mod == 2
                self.difficulty = "hard"
        #self.difficulty = "hard"

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
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.task_generator = torch.Generator()
        self.task_generator.manual_seed(seed + 1003)
        self.dynamic=bool(torch.randint(0, 2, (1,), generator=self.generator).item())

        # Track the color order and counts used to describe the language goal.
        self.binfill_language_sequence = []

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
        camera_eye=[1,0,0.4]
        camera_target =[0,0,0.4]
        pose = sapien_utils.look_at(
            eye=camera_eye, target=camera_target
        )

        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # 所有场景随机都统一使用同一个生成器，保证同 seed 可复现
        generator = self.generator

        button_obb = build_button(
            self,
            center_xy=(-0.2, 0),
            scale=1.5,
            generator=generator,
        )
        # avoid 中维护当前已占用区域，后续采样方块时需要避开这些障碍
        avoid = [button_obb]

        # 随机生成带孔挡板的位置和朝向
        x_var = torch.rand(1, generator=generator).item() * 0.2 - 0.2  # [-0.2, 0.0)
        y_var = torch.rand(1, generator=generator).item() * 0.4 - 0.2  # [-0.2, 0.2)
        z_rot_deg = (torch.rand(1, generator=generator).item() * 40.0 - 20.0)  # [-20, 20) 度
        z_rot_rad = torch.deg2rad(torch.tensor(z_rot_deg))
        # 将 z 轴旋转角转换成四元数，供场景物体初始化使用
        rot_mat = euler_angles_to_matrix(torch.tensor([[0.0, 0.0, z_rot_rad]]), convention="XYZ")
        rot_quat = matrix_to_quaternion(rot_mat)[0]
        self.board_with_hole = build_board_with_hole(
            self,
            board_side=0.1,  # 挡板边长
            hole_side=0.08,   # 中间孔洞边长，略大于方块尺寸
            thickness=0.05,   # 挡板厚度
            position=[0.15 + x_var, 0.0 + y_var, 0.0],  # 挡板中心位置
            rotation_quat=rot_quat.tolist(),  # 挡板绕 z 轴旋转
            name="board_with_hole"
        )
        avoid += [self.board_with_hole]

        # 任务要求的颜色数和每种颜色目标数使用独立随机源，避免受场景采样顺序影响
        task_generator = self.task_generator

        # 根据难度读取颜色数、生成数和放入目标数的采样范围
        config = self.configs[self.difficulty]
        num_colors = config['color']  # 本局允许出现的颜色种类数
        spawn_range = config['spawn_cubes']  # 总生成方块数范围
        put_in_color_range = config['put_in_color']
        # 先随机选出本局可能出现的颜色集合
        color_pool = torch.randperm(3, generator=task_generator).tolist()[:num_colors]
        # 再随机本局任务要求多少种颜色放入 bin
        put_in_color = torch.randint(
            put_in_color_range[0], put_in_color_range[1] + 1, (1,), generator=task_generator
        ).item()
        put_in_color = max(1, min(3, put_in_color))
        put_in_color = min(put_in_color, max(1, num_colors))
        # active_color_indices 是真正参与放入目标分配的颜色
        active_color_indices = color_pool[:put_in_color]
        put_in_range = config['put_in_numbers']  # 总目标放入数量范围

        # target_numbers 表示红蓝绿三种颜色分别需要放入 bin 的数量
        target_numbers = [0, 0, 0]
        if put_in_color == 1:
            # 单颜色任务：只给该颜色分配目标数
            selected_idx = active_color_indices[0]
            target_numbers[selected_idx] = torch.randint(
                put_in_range[0],
                put_in_range[1] + 1,
                (1,),
                generator=task_generator,
            ).item()
        else:
            # 多颜色任务：先保证每个参与颜色至少分到 1 个，再分配剩余目标数
            min_required_targets = len(active_color_indices)
            if put_in_range[1] < min_required_targets:
                raise ValueError(
                    f"BinFill 配置无效：put_in_numbers={put_in_range} 无法覆盖 {min_required_targets} 个目标颜色"
                )
            total_target = torch.randint(
                max(put_in_range[0], min_required_targets),
                put_in_range[1] + 1,
                (1,),
                generator=task_generator,
            ).item()
            for color_idx in active_color_indices:
                target_numbers[color_idx] += 1
            for _ in range(total_target - min_required_targets):
                idx = torch.randint(0, len(active_color_indices), (1,), generator=task_generator).item()
                target_numbers[active_color_indices[idx]] += 1

        self.red_cubes_target_number = target_numbers[0]
        self.blue_cubes_target_number = target_numbers[1]
        self.green_cubes_target_number = target_numbers[2]

        # 再生成实际摆在桌面上的方块数量，并保证生成数不少于目标数
        total_spawn = torch.randint(spawn_range[0], spawn_range[1] + 1, (1,), generator=generator).item()

        if num_colors == 1:
            # 单颜色场景：只有一个颜色会生成方块
            spawn_numbers = [0, 0, 0]
            active_idx = next((i for i in color_pool if target_numbers[i] > 0), color_pool[0])
            spawn_numbers[active_idx] = max(total_spawn, target_numbers[active_idx])
        else:
            # 多颜色场景：每个参与生成的颜色至少放 1 个，并满足生成数 >= 目标数
            spawn_numbers = [0, 0, 0]
            for i in color_pool:
                spawn_numbers[i] = max(target_numbers[i], 1)
            used_spawn = sum(spawn_numbers[i] for i in color_pool)
            remaining = total_spawn - used_spawn
            # 剩余方块数继续随机分配到已参与的颜色中
            for _ in range(max(0, remaining)):
                idx = torch.randint(0, len(color_pool), (1,), generator=generator).item()
                spawn_numbers[color_pool[idx]] += 1

        self.red_cubes_spawn_number = spawn_numbers[0]
        self.blue_cubes_spawn_number = spawn_numbers[1]
        self.green_cubes_spawn_number = spawn_numbers[2]

        logger.debug(f"Target numbers - Red: {self.red_cubes_target_number}, Blue: {self.blue_cubes_target_number}, Green: {self.green_cubes_target_number}")
        logger.debug(f"Spawn numbers - Red: {self.red_cubes_spawn_number}, Blue: {self.blue_cubes_spawn_number}, Green: {self.green_cubes_spawn_number}")

        self.all_cubes = []
        self.red_cubes, self.blue_cubes, self.green_cubes = [], [], []

        color_info = [
            {"color": (1, 0, 0, 1), "name": "red", "list": self.red_cubes, "spawn_num": self.red_cubes_spawn_number},
            {"color": (0, 0, 1, 1), "name": "blue", "list": self.blue_cubes, "spawn_num": self.blue_cubes_spawn_number},
            {"color": (0, 1, 0, 1), "name": "green", "list": self.green_cubes, "spawn_num": self.green_cubes_spawn_number}
        ]

        # 先整理出所有待生成方块，再打乱顺序，避免颜色生成顺序固定
        cube_tasks = []
        for info in color_info:
            for idx in range(info["spawn_num"]):
                cube_tasks.append({"color": info["color"], "name": info["name"], "list": info["list"], "idx": idx})

        shuffle_order = torch.randperm(len(cube_tasks), generator=generator).tolist()
        cube_tasks = [cube_tasks[i] for i in shuffle_order]

        # 逐个生成方块；每成功生成一个，就加入 avoid，避免后续方块重叠
        for task in cube_tasks:
            try:
                cube = spawn_random_cube(
                    self, color=task["color"], avoid=avoid,
                    include_existing=False, include_goal=False,
                    region_center=[-0.1, 0], region_half_size=[0.2, 0.25],
                    half_size=self.cube_half_size, min_gap=self.cube_half_size,
                    random_yaw=True, name_prefix=f"cube_{task['name']}_{task['idx']}",
                    generator=generator,
                )
                self.all_cubes.append(cube)
                task["list"].append(cube)
                avoid.append(cube)
            except RuntimeError as e:
                logger.debug(f"Failed to spawn {task['name']} cube {task['idx']}: {e}")

        logger.debug(f"Generated {len(self.all_cubes)} cubes total (red: {len(self.red_cubes)}, blue: {len(self.blue_cubes)}, green: {len(self.green_cubes)})")




    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos=reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)

            tasks=[]
            self.red_cubes_in_bin=0
            self.blue_cubes_in_bin=0
            self.green_cubes_in_bin=0
            self.binfill_language_sequence = []
            color_task_definitions = [
                ("blue", self.blue_cubes, self.blue_cubes_target_number),
                ("red", self.red_cubes, self.red_cubes_target_number),
                ("green", self.green_cubes, self.green_cubes_target_number),
            ]
            color_order = torch.randperm(len(color_task_definitions), generator=self.generator).tolist()
            for color_idx in color_order:
                color_name, cube_collection, target_number = color_task_definitions[color_idx]
                if target_number <= 0:
                    continue
                self.binfill_language_sequence.append((color_name, target_number))
                for i in range(target_number):
                    cube = cube_collection[i]
                    tasks.append({
                        "func": lambda c=self.all_cubes: is_any_obj_pickup_flag_currentpickup(self,objects=c),
                        "name": subgoal_language.get_subgoal_with_index(i, "pick up the {idx} {color} cube", color=color_name),
                        "subgoal_segment": subgoal_language.get_subgoal_with_index(i, "pick up the {idx} {color} cube at <>", color=color_name),
                        "choice_label": "pick up the cube",
                        "demonstration": False,
                        "failure_func":  lambda:is_button_pressed(self, obj=self.button),
                        "solve": lambda env, planner, c=cube: solve_pickup(env, planner, obj=c),
                        "segment":[cube_collection[i]]
                    })
                    tasks.append({
                        "func": lambda c=self.all_cubes: is_any_obj_dropped_onto_delete(self, objects=c, target=self.board_with_hole),
                        "name": f"put it into the bin",
                        "subgoal_segment":"put it into the bin at <>",
                        "choice_label": "put it into the bin",
                        "demonstration": False,
                        "failure_func":  lambda:is_button_pressed(self, obj=self.button),
                        "solve": lambda env, planner, c=cube: [
                            solve_putonto_whenhold_binspecial(env, planner, target=self.board_with_hole),
                        ],
                        "segment":[self.board_with_hole]
                    })
            tasks.append({
                "func": lambda: is_button_pressed(self, obj=self.button),
                "name": "press the button",
                "subgoal_segment":"press the button at <>",
                "choice_label": "press the button",
                "demonstration": False,
                "failure_func":lambda  c=self.all_cubes:[not check_in_bin_number(self,in_bin_list= [self.red_cubes_in_bin, self.blue_cubes_in_bin, self.green_cubes_in_bin],
                                                            total_number_list=[self.red_cubes_target_number, self.blue_cubes_target_number, self.green_cubes_target_number])
                ,is_any_obj_dropped_onto_delete(self, objects=c, target=self.board_with_hole)],
                "solve": lambda env, planner: [solve_button(env, planner, obj=self.button)],
                  "segment":self.cap_link 
            })
            self.task_list=tasks
            # Record pickup related task indices and items for recovery
            self.recovery_pickup_indices, self.recovery_pickup_tasks = task4recovery(self.task_list)
            if self.robomme_failure_recovery:
                # Only inject an intentional failed grasp when recovery mode is enabled
                self.fail_grasp_task_index = inject_fail_grasp(
                    self.task_list,
                    generator=self.generator,
                    mode=self.robomme_failure_recovery_mode,
                )
            else:
                self.fail_grasp_task_index = None


    def _get_obs_extra(self, info: Dict):
        return dict()



    def evaluate(self,solve_complete_eval=False):
        self.successflag=torch.tensor([False])
        # Save current_task_failure state before calling sequential_task_check
        # This is because failure might be detected during step(), but sequential_task_check might reset it
        previous_failure = getattr(self, "current_task_failure", False)
        self.failureflag = torch.tensor([False])



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

            
        # Use encapsulated sequence task check function
        all_tasks_completed, current_task_name, task_failed ,self.current_task_specialflag= sequential_task_check(self, self.task_list,allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep)

        # If task failed, mark as failed immediately
        # Or if failure was detected previously (previous_failure), also mark as failed
        if task_failed or previous_failure:
            self.failureflag = torch.tensor([True])
            if task_failed:
                logger.debug(f"Task failed: {current_task_name}")
            elif previous_failure:
                # If marked failed due to previous_failure, ensure current_task_failure is also set
                self.current_task_failure = True

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
        self.vis_obj_id_list=[]
        
        timestep = self.elapsed_steps
        if self.dynamic:
            # Dynamically lift cubes for each color (starting from 2nd cube)
            for cube_list in [self.red_cubes, self.blue_cubes, self.green_cubes]:
                for idx in range(1, len(cube_list)):
                    lift_and_drop_objects_back_to_original(
                        self,
                        obj=cube_list[idx],
                        start_step=0,
                        end_step=idx * 100,
                        cur_step=timestep,
                    )
                
        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info
