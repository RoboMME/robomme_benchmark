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
#HistoryBench
import matplotlib.pyplot as plt
import h5py
import random
import json
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from .util import *
from .util.evaluate import static_check
from .util.object_generation import spawn_fixed_cube, build_board_with_hole
from .util import reset_panda
from .util import subgoal_language
from .util.difficulty import normalize_historybench_difficulty



@register_env("BinFill", max_episode_steps=2000)
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

    # config_hard = {
    # 'color': 3, 
    # 'spawn_cubes':4,
    #     "put_in_color":3,
    # }

    # config_easy = {
    #     'color': 1, 
    # 'spawn_cubes':8,
    #     "put_in_color":1,
    # }

    # config_medium = {
    #     'color': 3, 
    # 'spawn_cubes':4,
    #     "put_in_color":1,
    # }

    config_easy = {
    'color': 1, 
    'spawn_cubes':[4,6],
    "put_in_color":[1,1],
    "put_in_numbers":[1,3]
    }

    config_medium = {
    'color': 2, 
    'spawn_cubes':[8,10],
    "put_in_color":[1,2],
    "put_in_numbers":[2,4]
    }


    config_hard = {
    'color': 3, 
    'spawn_cubes':[10,12],
    "put_in_color":[2,3],
    "put_in_numbers":[3,5]
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

        self.HistoryBench_seed = HistoryBench_seed
        self.generator = torch.Generator()
        self.generator.manual_seed(HistoryBench_seed)
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
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # 创建generator用于所有随机化
        generator = self.generator

        button_obb = build_button(
            self,
            center_xy=(-0.2, 0),
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

        ###
        ###
        ###
        ###
        ###
        # 先生成target_number (put_in):
        # 如果put_in_color == 1：随机选一个颜色，给它分配[put_in_range[0], put_in_range[1]]范围内的target数量
        # 如果put_in_color == 3：
            # 先从put_in_range生成总的target数量total_target
            # 从[0, 0, 0]开始，随机分配到三个颜色（不要求每个颜色至少1个）

        # 再生成spawn_number:
        # 如果num_colors == 1：只有有target的那个颜色会spawn cube，spawn数量 = max(total_spawn, target数量)
        # 如果num_colors == 3：每个颜色的spawn至少等于target，剩余的spawn随机分配
        # 这样确保了 spawn >= target 对于每个颜色都成立。


        # 获取当前难度的配置
        config = self.configs[self.difficulty]
        num_colors = config['color']  # 1 或 3
        spawn_range = config['spawn_cubes']  # [min, max]
        put_in_color_range = config['put_in_color']
        color_pool = torch.randperm(3, generator=generator).tolist()[:num_colors]
        put_in_color = torch.randint(
            put_in_color_range[0], put_in_color_range[1] + 1, (1,), generator=generator
        ).item()
        put_in_color = max(1, min(3, put_in_color))
        put_in_color = min(put_in_color, max(1, num_colors))
        active_color_indices = color_pool[:put_in_color]
        put_in_range = config['put_in_numbers']  # [min, max]

        # 先生成target_number (put_in)
        target_numbers = [0, 0, 0]
        if put_in_color == 1:
            # 只有一个颜色需要放入bin
            selected_idx = active_color_indices[0]
            target_numbers[selected_idx] = torch.randint(put_in_range[0], put_in_range[1] + 1, (1,), generator=generator).item()
        else:
            # 3个颜色都需要放入bin，先生成总数再分配
            total_target = torch.randint(put_in_range[0], put_in_range[1] + 1, (1,), generator=generator).item()
            # 随机分配target数量到三个颜色
            for _ in range(total_target):
                idx = torch.randint(0, len(active_color_indices), (1,), generator=generator).item()
                target_numbers[active_color_indices[idx]] += 1

        self.red_cubes_target_number = target_numbers[0]
        self.blue_cubes_target_number = target_numbers[1]
        self.green_cubes_target_number = target_numbers[2]

        # 再生成spawn_number，确保spawn >= target
        total_spawn = torch.randint(spawn_range[0], spawn_range[1] + 1, (1,), generator=generator).item()

        if num_colors == 1:
            # 只有一个颜色有cube，选择有target的那个颜色（如果都没有则用color_pool的第一个颜色）
            spawn_numbers = [0, 0, 0]
            active_idx = next((i for i in color_pool if target_numbers[i] > 0), color_pool[0])
            # spawn数量至少等于target数量
            spawn_numbers[active_idx] = max(total_spawn, target_numbers[active_idx])
        else:
            # num_colors 控制 1/2/3 种颜色：确保每个被选中的颜色至少有1个spawn，且spawn >= target
            spawn_numbers = [0, 0, 0]
            for i in color_pool:
                spawn_numbers[i] = max(target_numbers[i], 1)
            used_spawn = sum(spawn_numbers[i] for i in color_pool)
            remaining = total_spawn - used_spawn
            # 随机分配剩余的spawn数量
            for _ in range(max(0, remaining)):
                idx = torch.randint(0, len(color_pool), (1,), generator=generator).item()
                spawn_numbers[color_pool[idx]] += 1

        self.red_cubes_spawn_number = spawn_numbers[0]
        self.blue_cubes_spawn_number = spawn_numbers[1]
        self.green_cubes_spawn_number = spawn_numbers[2]

        print(f"Target numbers - Red: {self.red_cubes_target_number}, Blue: {self.blue_cubes_target_number}, Green: {self.green_cubes_target_number}")
        print(f"Spawn numbers - Red: {self.red_cubes_spawn_number}, Blue: {self.blue_cubes_spawn_number}, Green: {self.green_cubes_spawn_number}")


        ###
        ###
        ###
        ###
        ###
        self.all_cubes = []
        self.red_cubes, self.blue_cubes, self.green_cubes = [], [], []

        color_info = [
            {"color": (1, 0, 0, 1), "name": "red", "list": self.red_cubes, "spawn_num": self.red_cubes_spawn_number},
            {"color": (0, 0, 1, 1), "name": "blue", "list": self.blue_cubes, "spawn_num": self.blue_cubes_spawn_number},
            {"color": (0, 1, 0, 1), "name": "green", "list": self.green_cubes, "spawn_num": self.green_cubes_spawn_number}
        ]

        # 生成所有cube的任务列表并打乱顺序
        cube_tasks = []
        for info in color_info:
            for idx in range(info["spawn_num"]):
                cube_tasks.append({"color": info["color"], "name": info["name"], "list": info["list"], "idx": idx})

        # 打乱生成顺序
        shuffle_order = torch.randperm(len(cube_tasks), generator=generator).tolist()
        cube_tasks = [cube_tasks[i] for i in shuffle_order]

        # 按打乱后的顺序生成cube
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
                print(f"生成{task['name']} cube {task['idx']} 失败：{e}")

        print(f"Generated {len(self.all_cubes)} cubes total (red: {len(self.red_cubes)}, blue: {len(self.blue_cubes)}, green: {len(self.green_cubes)})")




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
                        "demonstration": False,
                        "failure_func":  lambda:is_button_pressed(self, obj=self.button),
                        "solve": lambda env, planner, c=cube: [solve_pickup(env, planner, obj=c),],
                        "segment":[cube_collection[i]],

                    })
                    tasks.append({
                        "func": lambda c=self.all_cubes: is_any_obj_dropped_onto_delete(self, objects=c, target=self.board_with_hole),
                        "name": f"put it into the bin",
                        "subgoal_segment":"put it into the bin at <>",
                        "demonstration": False,
                        "failure_func":  lambda:is_button_pressed(self, obj=self.button),
                        "solve": lambda env, planner, c=cube: [
                            solve_putonto_whenhold_binspecial(env, planner, obj=c, target=self.board_with_hole),
                        ],
                        "segment":[self.board_with_hole]
                    })
            tasks.append({
                "func": lambda: is_button_pressed(self, obj=self.button),
                "name": "press the button",
                "subgoal_segment":"press the button at <>",
                "demonstration": False,
                "failure_func":lambda :not check_in_bin_number(self,in_bin_list= [self.red_cubes_in_bin, self.blue_cubes_in_bin, self.green_cubes_in_bin],
                                                            total_number_list=[self.red_cubes_target_number, self.blue_cubes_target_number, self.green_cubes_target_number]),
                "solve": lambda env, planner: solve_button(env, planner, obj=self.button),
                  "segment":self.cap_link 
            })
            self.task_list=tasks

    def _get_obs_extra(self, info: Dict):
        return dict()



    def evaluate(self,solve_complete_eval=False):
        self.successflag=torch.tensor([False])
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
        self.vis_obj_id_list=[]
        
        timestep = self.elapsed_steps
        # if self.dynamic:
        #     # 对每个颜色的cube进行动态lift操作（从第2个cube开始）
        #     for cube_list in [self.red_cubes, self.blue_cubes, self.green_cubes]:
        #         for idx in range(1, len(cube_list)):
        #             lift_and_drop_objects_back_to_original(
        #                 self,
        #                 obj=cube_list[idx],
        #                 start_step=0,
        #                 end_step=idx * 100,
        #                 cur_step=timestep,
        #             )
                
        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info
