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

from .errors import SceneGenerationError
from .util import *
from .util.evaluate import static_check, too_many_swings
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


@register_env("SwingXtimes", max_episode_steps=2000)
class SwingXtimes(BaseEnv):

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
    'color': 3, 
    'number_min': 3,
    'number_max':3,
    }

    config_easy = {
        'color': 1, 
    'number_min': 1,
    'number_max':3
    }

    config_medium = {
        'color': 3, 
    'number_min': 1,
    'number_max':2
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
            # Determine difficulty based on seed % 3
            seed_mod = HistoryBench_seed % 3
            if seed_mod == 0:
                self.difficulty = "easy"
            elif seed_mod == 1:
                self.difficulty = "medium"
            else:  # seed_mod == 2
                self.difficulty = "hard"

               # 使用 seed 随机确定需要重复的次数 (1-5)
        generator = torch.Generator()
        generator.manual_seed(HistoryBench_seed)
        self.num_repeats = torch.randint(self.configs[self.difficulty]['number_min'], self.configs[self.difficulty]['number_max']+1, (1,), generator=generator).item()
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

        try:
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
            shuffle_indices = torch.randperm(len(color_groups), generator=generator).tolist()
            color_groups = [color_groups[i] for i in shuffle_indices]

            # Randomly select target color using generator
            target_color_idx = torch.randint(0, len(color_groups), (1,), generator=generator).item()
            self.target_color_name = color_groups[target_color_idx]["name"]
            print(f"Target color selected: {self.target_color_name}")

            # Generate cubes for each color group
            for idx, group in enumerate(color_groups):
                if idx < self.configs[self.difficulty]['color']:
                    for cube_idx in range(cubes_per_color):
                        try:
                            cube = spawn_random_cube(
                                self,
                                color=group["color"],
                                avoid=avoid,
                                include_existing=False,
                                include_goal=False,
                                region_center=[-0.1, 0],
                                region_half_size=0.25,
                                half_size=self.cube_half_size,
                                min_gap=self.cube_half_size,
                                random_yaw=True,
                                name_prefix=f"cube_{group['name']}_{cube_idx}",
                                generator=generator,
                            )
                        except RuntimeError as e:
                            raise SceneGenerationError(
                                f"生成{group['name']} cube {cube_idx} 失败：{e}"
                            ) from e

                        self.all_cubes.append(cube)
                        group["list"].append(cube)
                        cube_name = f"cube_{group['name']}_{cube_idx}"
                        group["name_list"].append(cube_name)
                        setattr(self, cube_name, cube)
                        avoid.append(cube)

                print(f"Generated {len(group['list'])} {group['name']} cubes")

            print(f"Generated {len(self.all_cubes)} cubes total (red: {len(self.red_cubes)}, blue: {len(self.blue_cubes)}, green: {len(self.green_cubes)})")

            # Generate first target
            try:
                temp_target_0 = spawn_random_target(
                    self,
                    avoid=avoid,  # 使用当前避让清单，包含所有已生成的cubes
                    include_existing=False,  # 手动维护清单
                    include_goal=False,  # 手动维护清单
                    region_center=[-0.1, -0.2],
                    region_half_size=0.1,
                    radius=self.cube_half_size*2,  # 使用radius而不是half_size
                    thickness=0.005,  # target的厚度
                    min_gap=self.cube_half_size*1,  # 与cube相同的间隙要求
                    name_prefix=f"temp_target_0",
                    generator=generator,
                    target_style="gray"
                )
                avoid.append(temp_target_0)
                print(f"Generated first target")
            except RuntimeError as e:
                raise SceneGenerationError("First target采样失败") from e

            # Generate second target
            try:
                temp_target_1 = spawn_random_target(
                    self,
                    avoid=avoid,  # 使用当前避让清单，包含所有已生成的cubes和第一个target
                    include_existing=False,  # 手动维护清单
                    include_goal=False,  # 手动维护清单
                    region_center=[-0.1, 0.2],
                    region_half_size=0.1,
                    radius=self.cube_half_size*2,  # 使用radius而不是half_size
                    thickness=0.005,  # target的厚度
                    min_gap=self.cube_half_size*1,  # 与cube相同的间隙要求
                    name_prefix=f"temp_target_1",
                    generator=generator,
                    target_style="gray"
                )
                avoid.append(temp_target_1)
                print(f"Generated second target")
            except RuntimeError as e:
                raise SceneGenerationError("Second target采样失败") from e

            # Swap names if necessary to ensure target_0.y < target_1.y
            temp_0_y = temp_target_0.pose.p[0, 1].item()  # Get y coordinate
            temp_1_y = temp_target_1.pose.p[0, 1].item()  # Get y coordinate

            if temp_0_y < temp_1_y:
                # No swap needed
                self.target_right = temp_target_0
                self.target_left = temp_target_1
                print(f"target_0 y={temp_0_y:.3f}, target_1 y={temp_1_y:.3f} (no swap needed)")
            else:
                # Swap the assignments
                self.target_right = temp_target_1
                self.target_left = temp_target_0
                print(f"Swapped: target_0 y={temp_1_y:.3f}, target_1 y={temp_0_y:.3f} (swapped to ensure target_0.y < target_1.y)")

            # Randomly select one cube from all available cubes as the target
            if len(self.all_cubes) > 0:
                target_cube_idx = torch.randint(0, len(self.all_cubes), (1,), generator=generator).item()
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



        except SceneGenerationError:
            raise
        except Exception as exc:
            raise SceneGenerationError(
                f"Failed to load SwingXtimes scene for seed {self.HistoryBench_seed}"
            ) from exc
        

        tasks = []
        tasks.append({
                "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                "name": f"pick up the {self.target_color_name} cube",
                "subgoal_segment":f"pick up the {self.target_color_name} cube at <>",
                "demonstration": False,
                "failure_func": lambda: [is_any_obj_pickup(self, self.non_target_cubes),is_button_pressed(self, obj=self.button),too_many_swings(self)],
                "solve": lambda env, planner: solve_pickup(env, planner, obj=self.target_cube),
                'segment':self.target_cube,
            })

        ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
        for i in range(self.num_repeats):
            ordinal = ordinals[i] if i < len(ordinals) else f"{i+1}th"
            tasks.append({
                "func": (lambda: is_obj_swing_onto(self,obj=self.target_cube,target=self.target_right,distance_threshold=0.03,z_threshold=0.12)),
                "name": f"move to the top of the right-side target for the {ordinal} time",
                "subgoal_segment":f"move to the top of the right-side target at <> for the {ordinal} time",
                "demonstration": False,
                "failure_func": lambda:  [is_any_obj_pickup(self, self.non_target_cubes),is_button_pressed(self, obj=self.button),too_many_swings(self)],
                # "solve": lambda env, planner: [solve_swingonto_whenhold(env, planner,target=self.target_right,height=0.1),
                #                             ],
                "solve": lambda env, planner: [solve_swingonto_whenhold(env, planner,target=self.target_right,height=0.1),
                                                # solve_swingonto_whenhold(env, planner,target=self.target_right,height=0.15),
                                                # solve_swingonto_whenhold(env, planner,target=self.target_right,height=0.1),
                                            ],
                'segment':self.target_right,
            })
            tasks.append({
                "func": (lambda: is_obj_swing_onto(self,obj=self.target_cube,target=self.target_left,distance_threshold=0.03,z_threshold=0.12)),
                "name": f"move to the top of the left-side target for the {ordinal} time",
                "subgoal_segment":f"move to the top of the left-side target at <> for the {ordinal} time",
                "demonstration": False,
                "failure_func": lambda:  [is_any_obj_pickup(self, self.non_target_cubes),is_button_pressed(self, obj=self.button),too_many_swings(self)],
                "solve": lambda env, planner: [solve_swingonto_whenhold(env, planner, target=self.target_left,height=0.1),
                                            ],
                'segment':self.target_left,
            })


        tasks.append({
                "func": (lambda: is_obj_dropped(self, obj=self.target_cube)),
                "name": f"put the {self.target_color_name} cube on the table",
                "subgoal_segment":f"put the {self.target_color_name} cube on the table",
                "demonstration": False,
                "failure_func":  lambda: [is_any_obj_pickup(self, self.non_target_cubes),is_button_pressed(self, obj=self.button),too_many_swings(self)],
                "solve": lambda env, planner: solve_putdown_whenhold(env, planner,),
            })
        tasks.append({
                "func": lambda: is_button_pressed(self, obj=self.button),
                "name": "press the button",
                "subgoal_segment":"press the button at <>",
                "demonstration": False,
                "failure_func":lambda:[is_any_obj_pickup(self, self.non_target_cubes),too_many_swings(self)],
                "solve": lambda env, planner: solve_button(env, planner, obj=self.button),
                "segment":self.cap_link 
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
            self.highlight_right_start = None
            self.highlight_left_start = None
            # 摆动计数初始化：
            # swing_count     记录累计摆动/落点次数（左右各算一次）
            # swing_over_limit 标记是否超过允许次数，一旦为 True 就判定失败
            # _was_on_right/_was_on_left 用于做“边沿检测”，防止同一落点连续多帧重复计数
            self.swing_count = 0
            self.swing_over_limit = False
            self._was_on_right = False
            self._was_on_left = False
            # 期望的最大摆动次数（左右各 self.num_repeats 次）
            self.max_swings = self.num_repeats * 2

    def _get_obs_extra(self, info: Dict):
        return dict()




    def evaluate(self,solve_complete_eval=False):
        previous_failure = getattr(self, "failureflag", None)
        self.successflag = torch.tensor([False])
        if previous_failure is not None and bool(previous_failure.item()):
            self.failureflag = previous_failure
        else:
            self.failureflag = torch.tensor([False])

        # 为测试“超出摆动上限”场景，强行降低上限（例如 1 次）
        # 这样第二次落点就会触发 too_many_swings 失败逻辑，便于验证
        #self.max_swings = 2



       
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


#HistoryBench
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):


        obs, reward, terminated, truncated, info = super().step(action)
        # 先检测当前帧是否"落"在左右目标上，用于高亮与计数
        # 注意：policy 在 z 轴上下抖动时，is_obj_swing_onto 的 z_threshold 判定可能导致 on/off 反复翻转，
        # 从而触发多次 "False->True" 边沿，造成重复计数。这里用 enter/exit 滞回阈值缓解抖动。
        # 为防止 z 轴在阈值附近抖动造成 on/off 反复翻转、重复计数：
        # 使用 enter/exit 两套阈值（滞回 hysteresis）。
        # - enter 更严格：进入目标区域才算一次落点
        # - exit 更宽松：在目标区域内的小抖动不会被误判为离开
        swing_enter_distance_threshold = 0.03
        swing_exit_distance_threshold = 0.04  # >= enter
        swing_enter_z_threshold = 0.12
        swing_exit_z_threshold = 0.3  # >= enter
        
        if self._was_on_right:
            on_right = is_obj_swing_onto(
                self,
                obj=self.target_cube,
                target=self.target_right,
                distance_threshold=swing_exit_distance_threshold,
                z_threshold=swing_exit_z_threshold,
            )
        else:
            on_right = is_obj_swing_onto(
                self,
                obj=self.target_cube,
                target=self.target_right,
                distance_threshold=swing_enter_distance_threshold,
                z_threshold=swing_enter_z_threshold,
            )

        if self._was_on_left:
            on_left = is_obj_swing_onto(
                self,
                obj=self.target_cube,
                target=self.target_left,
                distance_threshold=swing_exit_distance_threshold,
                z_threshold=swing_exit_z_threshold,
            )
        else:
            on_left = is_obj_swing_onto(
                self,
                obj=self.target_cube,
                target=self.target_left,
                distance_threshold=swing_enter_distance_threshold,
                z_threshold=swing_enter_z_threshold,
            )
        if on_right:
             self.highlight_right_start=int(self.elapsed_steps[0].item())
             # 只在“首次”落点时累计一次摆动次数，避免连续帧重复累计
             if not self._was_on_right:
                self.swing_count += 1
        if on_left:
             self.highlight_left_start=int(self.elapsed_steps[0].item())
             # 只在“首次”落点时累计一次摆动次数，避免连续帧重复累计
             if not self._was_on_left:
                self.swing_count += 1

        # 边沿检测记录完后更新状态
        self._was_on_right = on_right
        self._was_on_left = on_left

        if self.swing_count > self.max_swings:
            if not self.swing_over_limit:
                # 只打印一次，提示摆动次数已超上限
                print(f"Swing count exceeded: {self.swing_count}>{self.max_swings}")
            self.swing_over_limit = True
             
        if self.highlight_right_start is not None:
            cur_step = int(self.elapsed_steps[0].item())

            highlight_obj(
                    self,
                    self.target_right,
                    start_step=self.highlight_right_start,
                    end_step=self.highlight_right_start+20,
                    cur_step=cur_step,
                    disk_radius=self.cube_half_size*2*1.003,
                    disk_half_length=0.005*2*1.4,
                    use_target_style=True,
                    highlight_color=[1.0, 0.0, 0.0, 1.0],
                )
        if self.highlight_left_start is not None:
            cur_step = int(self.elapsed_steps[0].item())

            highlight_obj(
                    self,
                    self.target_left,
                    start_step=self.highlight_left_start,
                    end_step=self.highlight_left_start+20,
                    cur_step=cur_step,
                    disk_radius=self.cube_half_size*2*1.003,
                    disk_half_length=0.005*2*1.4,
                    use_target_style=True,
                    highlight_color=[1.0, 0.0, 0.0, 1.0],
                )
        return obs, reward, terminated, truncated, info
