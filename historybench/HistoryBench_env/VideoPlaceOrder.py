import json
import os
import uuid
from pathlib import Path
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


@register_env("VideoPlaceOrder", max_episode_steps=2000)
class VideoPlaceOrder(BaseEnv):

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
        #"place":2,
        "swap":False,
        "targets":4
    }
    config_medium= {
        'color': 3, 
       # "place":2,
        "swap":False,
        "targets":4
    }
    config_hard = {
        'color': 3, 
       # "place":4,
        "swap":True,
        "targets":4
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
        self.generator = torch.Generator()
        self.generator.manual_seed(HistoryBench_seed)
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
            self.difficulty = "easy"
        


        self.onto_goalsite=False
        self.start_step=99999
        self.end_step=99999
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

        yaw=0
        rotate = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])  # z轴旋转的四元数
        angles = torch.deg2rad(torch.tensor([0.0, 90.0, 0.0], dtype=torch.float32))  # (3,)
        rotate = matrix_to_quaternion(
            euler_angles_to_matrix(angles, convention="XYZ")
        )
        self.goal_site = spawn_random_target(
                        self,
                        avoid=None,  # 使用当前避让清单，包含所有已生成的cubes
                        include_existing=False,  # 手动维护清单
                        include_goal=False,  # 手动维护清单
                        region_center=[-0.2, 0],
                        region_half_size=0.05,
                        radius=self.cube_half_size*2,  # 使用radius而不是half_size
                        thickness=0.005,  # target的厚度
                        min_gap=self.cube_half_size*1,  # 与cube相同的间隙要求
                        name_prefix=f"goal_site",
                        generator=self.generator
                        )
        avoid=[]
        avoid.append(self.goal_site)    
        button_obb = build_button(
            self,
            center_xy=(-0.1, 0),
            scale=1.5,
            generator=self.generator,
        )
        avoid.append(button_obb)

                # 生成targets
       



        self.all_cubes = []  # 保存所有 cube 对象

        # Initialize storage for each color group
        self.red_cubes = []
        self.red_cube_names = []
        self.blue_cubes = []
        self.blue_cube_names = []
        self.green_cubes = []
        self.green_cube_names = []

        cubes_per_color = 1
        color_groups = [
            {"color": (1, 0, 0, 1), "name": "red", "list": self.red_cubes, "name_list": self.red_cube_names},
            {"color": (0, 0, 1, 1), "name": "blue", "list": self.blue_cubes, "name_list": self.blue_cube_names},
            {"color": (0, 1, 0, 1), "name": "green", "list": self.green_cubes, "name_list": self.green_cube_names}
        ]
        shuffle_indices = torch.randperm(len(color_groups), generator=self.generator).tolist()
        color_groups = [color_groups[i] for i in shuffle_indices]


        self.target_color_name = color_groups[0]["name"]
        print(f"Target color selected: {self.target_color_name}")

            # Generate 5 cubes for each color group
        for idx, group in enumerate(color_groups):
            if idx < self.configs[self.difficulty]['color']:
                for idx in range(cubes_per_color):
                    try:
                        cube = spawn_random_cube(
                            self,
                            color=group["color"],
                            avoid=avoid,
                            include_existing=False,
                            include_goal=False,
                            region_center=[-0.1, 0],
                            region_half_size=0.2,
                            half_size=self.cube_half_size,
                            min_gap=self.cube_half_size,
                            random_yaw=True,
                            name_prefix=f"cube_{group['name']}_{idx}",
                            generator=self.generator,
                        )
                    except RuntimeError as e:
                        print(f"生成{group['name']} cube {idx} 失败：{e}")
                        break

                    self.all_cubes.append(cube)
                    group["list"].append(cube)
                    cube_name = f"cube_{group['name']}_{idx}"
                    group["name_list"].append(cube_name)
                    setattr(self, cube_name, cube)
                    avoid.append(cube)

            print(f"Generated {len(group['list'])} {group['name']} cubes")

        print(f"Generated {len(self.all_cubes)} cubes total (red: {len(self.red_cubes)}, blue: {len(self.blue_cubes)}, green: {len(self.green_cubes)})")

        self.targets = []
        for i in range(4):
            if i < self.configs[self.difficulty]['targets']:
                try:
                    target = spawn_random_target(
                        self,
                        avoid=avoid,  # 使用当前避让清单，包含所有已生成的cubes
                        include_existing=False,  # 手动维护清单
                        include_goal=False,  # 手动维护清单
                        region_center=[0, 0],
                        region_half_size=0.2,
                        radius=self.cube_half_size*2,  # 使用radius而不是half_size
                        thickness=0.005,  # target的厚度
                        min_gap=self.cube_half_size*1,  # 与cube相同的间隙要求
                        name_prefix=f"target_{i}",
                        generator=self.generator
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
        





 # Randomly select one cube from all available cubes as the target
        if len(self.all_cubes) > 0:
            target_cube_idx = torch.randint(0, len(self.all_cubes), (1,), generator=self.generator).item()
            self.target_cube = self.all_cubes[target_cube_idx]

            # Determine the color of the selected target cube
            if self.target_cube in self.red_cubes:
                self.target_color_name = "red"
            elif self.target_cube in self.blue_cubes:
                self.target_color_name = "blue"
            elif self.target_cube in self.green_cubes:
                self.target_color_name = "green"


            print(f"Target cube selected: {self.target_color_name} cube (index {target_cube_idx} in all_cubes)")
        else:
            self.target_cube = None
            self.target_color_name = None
            print("No cubes generated, no target cube selected")

        # Create list of non-target cubes for failure checking
        self.non_target_cubes = [cube for cube in self.all_cubes if cube != self.target_cube]
        print(f"Non-target cubes: {len(self.non_target_cubes)}")




        # 预设交换目标
        self.swap_target_a = None
        self.swap_target_b = None
        self.swap_target_other = []

        if self.configs[self.difficulty]['swap']==True:
            if len(self.targets) >= 2:
                perm = torch.randperm(len(self.targets), generator=self.generator)
                swap_idx_a = perm[0].item()
                swap_idx_b = perm[1].item()
                self.swap_target_a = self.targets[swap_idx_a]
                self.swap_target_b = self.targets[swap_idx_b]
                self.swap_target_other = [
                    target
                    for idx, target in enumerate(self.targets)
                    if idx not in (swap_idx_a, swap_idx_b)
                ]
                print(
                    f"Swap targets selected: target_{swap_idx_a} <-> target_{swap_idx_b}"
                )
        # 随机选择1-4个targets
        num_targets_to_pick = torch.randint(2, len(self.targets) + 1, (1,), generator=self.generator).item()

        #num_targets_to_pick = torch.randint(4, len(self.targets) + 1, (1,), generator=self.generator).item()
        indices = torch.randperm(len(self.targets), generator=self.generator)[:num_targets_to_pick]

        self.which_targets_to_pick = [self.targets[i] for i in indices]


        self.which_in_subset=torch.randint(1,len(self.which_targets_to_pick)+1,(1,),generator=self.generator).item()
        #self.which_targets_to_pick = torch.ran


        print("self.which_in_subset:",self.which_in_subset)
        self.target_target=self.which_targets_to_pick[self.which_in_subset-1]

        self.targets_not_true = [t for i, t in enumerate(self.targets) if self.targets[i]!=self.target_target]

        if len(self.which_targets_to_pick) > 0:
            k = torch.randint(0, len(self.which_targets_to_pick), (1,), generator=self.generator).item()
            self.button_task_index = k * 2 + 2  # each pair contributes pickup + drop
        else:
            self.button_task_index = 0

        def _actor_to_name(actor):
            if actor is None:
                return None
            if hasattr(actor, "name"):
                return actor.name
            return str(actor)

        target_debug_payload = {
            "historybench_seed": self.HistoryBench_seed,
            "which_in_subset": self.which_in_subset,
            "num_targets_to_pick": num_targets_to_pick,
            "which_targets_to_pick": [_actor_to_name(target) for target in self.which_targets_to_pick],
            "target_target": _actor_to_name(self.target_target),
            "button_task_index": self.button_task_index,
        }

        try:
            log_path = Path(__file__).resolve().parent / "target_selection.json"
            payload_list = []
            if log_path.exists():
                try:
                    with open(log_path, "r") as fp:
                        existing_payload = json.load(fp)
                    if isinstance(existing_payload, list):
                        payload_list = existing_payload
                    elif existing_payload is not None:
                        payload_list = [existing_payload]
                except json.JSONDecodeError:
                    # 如果文件被破坏，保留备份并重新开始
                    log_path.with_suffix(".json.bak").write_text(log_path.read_text())
                    payload_list = []
            payload_list.append(target_debug_payload)
            with open(log_path, "w") as fp:
                json.dump(payload_list, fp, indent=2)
        except Exception as exc:
            print(f"Failed to write target selection log: {exc}")
        
        


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos=reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)

            pose_p=self.goal_site.pose.p.tolist()[0]
            pose_q=self.goal_site.pose.q.tolist()[0]
            pose_p[2]=-0.05
            self.goal_site.set_pose(sapien.Pose(p=pose_p,q=pose_q))  
            #print(self.goal_site.pose.p)         


            tasks = []

            # 1) 先生成所有“拾取+放置”的组合到临时列表
            pair_tasks = []
            for i in self.which_targets_to_pick:
                # 1.1 拾取
                pair_tasks.append({
                    "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                "name": f"pick up the cube",
                "subgoal_segment":f"pick up the cube at <>",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: solve_pickup(env, planner, obj=self.target_cube),
                    "segment":self.target_cube, 
                })

                # 1.2 放置（注意用 i=i 绑定当前目标）
                pair_tasks.append({
                    "func": (lambda i=i: is_obj_dropped_onto(self, obj=self.target_cube, target=i)),
                    "name": "drop the cube onto target",
                    "subgoal_segment":f"drop the cube onto target at <>",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner, i=i: solve_putonto_whenhold(env, planner,  target=i),
                    "segment":i,
                })

            # 2) 定义“按按钮”任务
            button_task = {
                "func": (lambda: is_button_pressed(self, obj=self.button)),
                "name": "press the button",
                "subgoal_segment":f"press the button at <>",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: solve_button(env, planner, self.button),
                "segment":self.cap_link 
            }

            # 4) 组装最终 tasks
            tasks = pair_tasks[: self.button_task_index] + [button_task] + pair_tasks[self.button_task_index :]
    ############
            tasks.append({
                    "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                        "name": f"pick up the cube",
                        "subgoal_segment":f"pick up the cube at <>",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: solve_pickup(env, planner, obj=self.target_cube),
                    "segment":self.target_cube,
                })
            tasks.append({
                    "func": (lambda: is_obj_dropped_onto(self,obj=self.target_cube,target=self.goal_site)),
                "name": "drop the cube onto table",
                "subgoal_segment":f"drop the cube onto table",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [solve_putonto_whenhold(env, planner,target=self.goal_site)],
             
            })
            tasks.append(       {
                                "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=20),
                                "name": "static",
                                "subgoal_segment":f"static",
                                "demonstration": True,
                                "failure_func": None,

                                "solve": lambda env, planner: [solve_reset(env,planner), solve_hold_obj(env, planner, static_steps=20)],
                                },)
            tasks.append(             {
                                "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=60),
                                "name": "static",
                                "subgoal_segment":f"static",
                                "demonstration": True,
                                "failure_func": None,
                                "specialflag":"swap",
                                "solve": lambda env, planner: [solve_hold_obj(env, planner, static_steps=60)],
                                },)

            tasks.append({
                                "func": lambda:reset_check(self),
                                "name": "NO RECORD",
                                "subgoal_segment":f"NO RECORD",
                                "demonstration": True,
                                "failure_func": None,
                                "solve": lambda env, planner: [ solve_strong_reset(env,planner)],
                                },)


            tasks.append({
                    "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                    "name": f"pick up the cube",
                    "subgoal_segment":f"pick up the cube at <>",
                    "demonstration": False,
                    "failure_func":lambda: is_any_obj_pickup(self, self.non_target_cubes),
                    "solve": lambda env, planner: [solve_pickup(env, planner, obj=self.target_cube)],
                    "segment":self.target_cube,
                })
            tasks.append({
                    "func": (lambda: is_obj_dropped_onto(self,obj=self.target_cube,target=self.target_target)),
                    "name": "place the cube onto the correct target",
                    "subgoal_segment":f"place the cube onto the correct target at <>",
                    "demonstration": False,
                    "failure_func": (lambda: is_obj_dropped_onto_any(self,obj=self.target_cube,target=self.targets_not_true)),
                    "solve": lambda env, planner: [solve_putonto_whenhold(env, planner,target=self.target_target),
                                                ],
                    "segment":self.target_target
            
            })



            # 存储任务列表供RecordWrapper使用
            self.task_list = tasks
            try:
                task_entries = []
                for task in tasks:
                    if isinstance(task, dict):
                        task_name = task.get("name", "Unknown")
                    elif isinstance(task, (list, tuple)):
                        if len(task) >= 2:
                            task_name = task[1]
                        else:
                            task_name = str(task)
                    else:
                        task_name = str(task)
                    task_entries.append(
                        {
                            "name": task_name,
                        }
                    )

                self._task_log_dir = Path(__file__).resolve().parent / "task_name_logs"
                self._task_log_dir.mkdir(parents=True, exist_ok=True)

                payload = {
                    "historybench_seed": int(self.HistoryBench_seed),
                    "difficulty": self.difficulty,
                    "tasks": task_entries,
                }

                self._task_log_payload = payload
                self._task_log_written = False
            except Exception as exc:
                print(f"Failed to write task names log: {exc}")

    def _get_obs_extra(self, info: Dict):
        return dict()



    def evaluate(self,solve_complete_eval=False):

    

        self.successflag=torch.tensor([False])
        self.failureflag = torch.tensor([False])


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
        all_tasks_completed, current_task_name, task_failed ,self.current_task_specialflags= sequential_task_check(self, self.task_list,allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep)

        # 如果任务失败，立即标记失败
        if task_failed:
            self.failureflag = torch.tensor([True])
            print(f"Task failed: {current_task_name}")
            self._write_task_log(status="failed")

        # 如果static_check成功或者所有任务完成，则设置成功标志
        if all_tasks_completed and not task_failed:
            self.successflag = torch.tensor([True])
            self._write_task_log(status="success")

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




        # highlight_obj(self,self.target_cube, start_step=0, end_step=100, cur_step=timestep)
        
        if self.current_task_specialflag=="swap":
            if self.onto_goalsite==False:
                self.onto_goalsite=True
                self.start_step=int(self.elapsed_steps.item())
                self.end_step=int(self.elapsed_steps.item())+50


        if self.swap_target_a is not None and self.swap_target_b is not None:
            other_bins = self.swap_target_other if self.swap_target_other else None
            swap_flat_two_lane(
                self,
                cube_a=self.swap_target_a,
                cube_b=self.swap_target_b,
                start_step=self.start_step,
                end_step=self.end_step,
                cur_step=self.elapsed_steps,
                lane_offset=0.1,
                smooth=True,
                keep_upright=True,
                other_cube=other_bins,
            )
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info

    def _write_task_log(self, status: str) -> None:
        if getattr(self, "_task_log_written", False):
            return

        payload = getattr(self, "_task_log_payload", None)
        log_dir = getattr(self, "_task_log_dir", None)

        if not payload or log_dir is None:
            return

        payload_to_write = dict(payload)
        payload_to_write["status"] = status

        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / (
            f"task_names_seed_{payload_to_write.get('historybench_seed', 'unknown')}_"
            f"pid_{os.getpid()}_{uuid.uuid4().hex}.json"
        )

        try:
            with open(log_file, "w", encoding="utf-8") as fp:
                json.dump(payload_to_write, fp, indent=2)
            self._task_log_written = True
        except Exception as exc:
            print(f"Failed to write task names log: {exc}")
