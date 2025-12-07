


from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
import math
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
import json
import os
import random
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)

from .util import *
from .util.evaluate import *
from .util.object_generation import *
from .util import reset_panda
from .util.route import *
from .util.planner import *
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


#方向如果反了 修改evaluate和solve

@register_env("RouteStick", max_episode_steps=2000)
class RouteStick(BaseEnv):

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
    'length':[2,3],
    'backtrack':False,
    }
    config_medium = {
    'length':[4,5],
    'backtrack':False,
    }
    config_hard = {
    'length':[4,7],
    'backtrack':True,
    }

    # 组合成一个字典
    configs = {
        'hard': config_hard,
        'easy': config_easy,
        'medium': config_medium
    }

    def __init__(self, *args, robot_uids="panda_stick", robot_init_qpos_noise=0,HistoryBench_seed=0,HistoryBench_video_episode=None,HistoryBench_video_path=None,
                     **kwargs):
        self.achieved_list=[]
        self.use_demonstrationwrapper=False
        self.demonstration_record_traj=False
        self.match=False
        self.after_demo=False
        self.direction_mistake_flag=False
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
            self.difficulty = "easy"
               # 使用 seed 随机确定需要重复的次数 (1-5)
        generator = torch.Generator()
        generator.manual_seed(HistoryBench_seed)




        self.highlight_starts = {}  # 使用字典存储每个按钮的高亮开始时间
        self._first_non_record_step = None  # 延迟高亮的起始timestep

        self.z_threshold=0.15
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

        # Generate 3x3 grid of buttons (九宫格按钮)
        grid_center = [-0.1, 0]  # 九宫格中心位置
        grid_spacing_x = 0.07 # 按钮之间的间距
        grid_spacing_y=0.07

        self.buttons_grid = []
        self.button_joints_grid = []
        avoid = []
        button_index = 0

        num_rows, num_cols = 1, 9
        row_center = (num_rows - 1) / 2
        col_center = (num_cols - 1) / 2


        theta = math.radians(
        (torch.rand(1, generator=generator).item() * 60) - 30
            )
        #theta=0
        for row in range(num_rows):
            for col in range(num_cols):  # 5列 (y方向)
                x_pos = grid_center[0] + (row - row_center) * grid_spacing_x
                y_pos = grid_center[1] + (col - col_center) * grid_spacing_y

                orig_x, orig_y = x_pos, y_pos
                x_pos = orig_x * math.cos(theta) - orig_y * math.sin(theta)
                y_pos = orig_x * math.sin(theta) + orig_y * math.cos(theta)

                target_name = f"target_{button_index}"

                # Create rotation quaternion for vertical target
                angles = torch.deg2rad(torch.tensor([0.0, 90.0, 0.0], dtype=torch.float32))
                rotate = matrix_to_quaternion(
                    euler_angles_to_matrix(angles, convention="XYZ")
                )

                # Build purple and white target
                raised_indices = {0, 2, 4, 6, 8}
                z_pos = 0.01 if button_index in raised_indices else -0.01
                target = build_gray_white_target(
                    scene=self.scene,
                    radius=0.02,
                    thickness=0.01,
                    name=target_name,
                    body_type="kinematic",
                    add_collision=False,
                    initial_pose=sapien.Pose(p=[x_pos, y_pos, z_pos], q=rotate),
                )

                self.buttons_grid.append(target)
                # Note: purple_white_target doesn't have joints, so we append None
                self.button_joints_grid.append(None)
                print(f"Generated target {button_index} at position ({x_pos:.3f}, {y_pos:.3f})")
                button_index += 1

        self.targets_grid = self.buttons_grid

        # Spawn white cubes on specific targets to create fixed obstacles.
        target_cube_indices = [1, 3, 5,7]
        self.target_cube_indices = target_cube_indices
        self.target_cubes = {}
        self.cubes_on_targets = []

        for target_idx in target_cube_indices:
            if target_idx >= len(self.targets_grid):
                print(f"[SwingAvoid] Skip cube spawn for target {target_idx}: index out of range.")
                continue

            target_actor = self.targets_grid[target_idx]
            target_pos = (
                target_actor.pose.p
                if hasattr(target_actor, "pose")
                else target_actor.get_pose().p
            )

            if isinstance(target_pos, torch.Tensor):
                target_pos = target_pos.detach().cpu().numpy()

            target_pos = np.asarray(target_pos, dtype=np.float64).reshape(-1)

            cube_position = [float(target_pos[0]), float(target_pos[1])]

            cylinder_radius = 0.015
            cylinder_height = 0.1
            cylinder_half_length = cylinder_height / 2.0

            # Keep the cylinder centered so that it stands upright on the table surface.
            cylinder_angles = torch.deg2rad(torch.tensor([0.0, 90.0, 0.0], dtype=torch.float32))
            builder = self.scene.create_actor_builder()

            cylinder_material = sapien.render.RenderMaterial()
            random_rgb = torch.rand(3, generator=generator).tolist()
            cylinder_material.set_base_color((*random_rgb, 1))

            # Rotate upright then around its own z-axis to align with the target line.
            z_twist_mat = euler_angles_to_matrix(
                torch.tensor([0.0, 0.0, theta], dtype=torch.float32), convention="XYZ"
            )
            base_upright_mat = euler_angles_to_matrix(
                cylinder_angles, convention="XYZ"
            )
            final_rot_mat = z_twist_mat @ base_upright_mat
            cylinder_quat = matrix_to_quaternion(final_rot_mat)

            builder.set_initial_pose(
                sapien.Pose(
                    p=[
                        cube_position[0],
                        cube_position[1],
                        cylinder_half_length,
                    ],
                    q=cylinder_quat.detach().cpu().numpy(),
                )
            )

            rect_length = 0.03 #0.03
            rect_width = 0.015
            # Keep height the same as the previous cylinder; rotation keeps height along world z.
            builder.add_box_visual(
                half_size=[cylinder_half_length, rect_width, rect_length],
                material=cylinder_material,
            )
            builder.add_box_collision(
                half_size=[cylinder_half_length, rect_width, rect_length],
            )

            cube_actor = builder.build_kinematic(name=f"target_cube_{target_idx}")

            self.cubes_on_targets.append(cube_actor)
            self.target_cubes[target_idx] = cube_actor
            setattr(self, f"target_cube_{target_idx}", cube_actor)

        tasks = []



        tasks=[]

        # Use the actual button actors corresponding to indices 0,2,4,6,8
        button_indices = [0, 2, 4, 6, 8]
        self.route_button_indices = button_indices

        cfg = self.configs.get(getattr(self, "difficulty", "easy"), self.config_easy)
        length_min, length_max = cfg.get("length")
        steps = int(torch.randint(length_min, length_max + 1, (1,), generator=generator).item())
        allow_backtracking = bool(cfg.get("backtrack", True))

        traj=generate_dynamic_walk(button_indices,steps=steps,allow_backtracking=allow_backtracking,generator=generator)#生成轨迹
        self.selected_buttons = [self.buttons_grid[i] for i in traj]

        def _stick_side(actor, ref_actor=None):
            """
            Determine whether a target lies on the left or right of a reference
            target based on their y positions. When no reference is provided,
            fall back to the workspace center (y=0).
            """
            def _get_y(a):
                pos = a.pose.p if hasattr(a, "pose") else a.get_pose().p
                pos_flat = np.asarray(pos).reshape(-1)
                return pos_flat[1] if pos_flat.size >= 2 else None

            y_val = _get_y(actor)
            ref_y = _get_y(ref_actor) if ref_actor is not None else 0.0
            if y_val is None or ref_y is None:
                # Fallback to right to avoid indexing errors; should not happen.
                return "right"
            return "left" if y_val > ref_y else "right"#按照机器人视角反向！

        # 随机为每个 solve_swingonto_withDirection 决定顺/逆时针方向并记录
        self.swing_directions = []
        for _ in self.selected_buttons[1:]:
            dir_flag = "clockwise" if torch.rand(1, generator=generator).item() < 0.5 else "counterclockwise"
            self.swing_directions.append(dir_flag)
        print(f"[RouteStick] swing direction list: {self.swing_directions}")

        current_target=self.selected_buttons[0]
        tasks.append({
            "func":   lambda t=current_target: is_obj_swing_onto(self, obj=self.agent.tcp, target=t, distance_threshold=0.03, z_threshold=self.z_threshold),
            "name":  "NO RECORD",
            "subgoal_segment":f"NO RECORD",
            "demonstration": True,
            "failure_func":  None,
            "solve": lambda env, planner, t=current_target: solve_swingonto(env, planner, target=t,record_swing_qpos=True),

        })  
        for i, current_target in enumerate(self.selected_buttons[1:]):
            direction = self.swing_directions[i]
            prev_target = self.selected_buttons[i]
            stick_side = _stick_side(current_target, prev_target)
             #task_name=f"rotate around the {stick_side} stick {direction}"
            task_name=f"move to the nearest {stick_side} target by circling around the stick {direction}"
            tasks.append({#减小threashold 试试看回放有没有出现
            "func":   lambda t=current_target: is_obj_swing_onto(self, obj=self.agent.tcp, target=t, distance_threshold=0.03, z_threshold=self.z_threshold),
            "name": task_name,
            "subgoal_segment":task_name,
            "demonstration": True,
            "failure_func":  (lambda expected=current_target, last=prev_target: self._wrong_button_touch(expected_button=expected, last_button=last)),
            "expected_dir": direction,
            "solve": lambda env, planner, t=current_target, d=direction: solve_swingonto_withDirection(env, planner, target=t,radius=0.2,direction=d),
                })  
        tasks.append({
                    "func": lambda:reset_check(self,gripper="stick"),
                    "name": "NO RECORD",
                    "subgoal_segment":"NO RECORD",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [solve_strong_reset(env,planner,timestep=200,gripper="stick")],
                    },),
        
        current_target=self.selected_buttons[0]
        tasks.append({
            "func":   lambda:reset_check(self,gripper="stick",target_qpos=self.swing_qpos),
            "name":  "NO RECORD",
            "subgoal_segment":f"NO RECORD",
            "demonstration": True,
            "failure_func":  None,
            "solve": lambda env, planner, t=current_target: [solve_strong_reset(env, planner,gripper="stick",action=self.swing_qpos)],
        })  
        for i, current_target in enumerate(self.selected_buttons[1:]):
            direction = self.swing_directions[i]
            prev_target = self.selected_buttons[i]
            stick_side = _stick_side(current_target, prev_target)
            #task_name=f"rotate around the {stick_side} stick {direction}"
            task_name=f"move to the nearest {stick_side} target by circling around the stick {direction}"
            tasks.append({
            "func":   lambda t=current_target: is_obj_swing_onto(self, obj=self.agent.tcp, target=t, distance_threshold=0.03, z_threshold=self.z_threshold),
            "name": task_name,
            "subgoal_segment":task_name,
            "demonstration": False,
            "failure_func":  (lambda expected=current_target, last=prev_target: [self._wrong_button_touch(expected_button=expected, last_button=last), self.direction_fail()]),
            "expected_dir": direction,
            "solve": lambda env, planner, t=current_target, d=direction: solve_swingonto_withDirection(env, planner, target=t,radius=0.2,direction=d),
        })  
            # 存储任务列表供RecordWrapper使用
        self.task_list = tasks



    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):

            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos=reset_panda.get_reset_panda_param("qpos",gripper="stick")
            self.agent.reset(qpos)

     


    def _get_obs_extra(self, info: Dict):
        return dict()




    def evaluate(self,solve_complete_eval=False):
        self.successflag=torch.tensor([False])
        self.failureflag = torch.tensor([False])

        after_demo_active = bool(getattr(self, "after_demo", False))
        # 仅在示范结束后开始记录轨迹与检测，且在首次开启后清空历史
        if after_demo_active and not getattr(self, "_after_demo_initialized", False):
            self._swing_success_history = []
            self._gripper_xy_trace = []
            self._last_swing_pair_reported_step = None
            self._after_demo_initialized = True

        cur_step = int(self.elapsed_steps[0].item())
        prev_timestep = getattr(self, "timestep", 0)
        prev_task = None
        if hasattr(self, "task_list") and 0 <= prev_timestep < len(self.task_list):
            prev_task = self.task_list[prev_timestep]

        # 记录当前步的夹爪xy位置（仅在 after_demo 开启后才记录）
        if after_demo_active:
            gripper_xy = torch.as_tensor(self.agent.tcp.pose.p[0][:2]).detach().cpu()
            self._gripper_xy_trace.append((cur_step, gripper_xy))

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


        # 检测是否刚完成一个摆动任务
        if after_demo_active:
            new_timestep = getattr(self, "timestep", prev_timestep)
            task_completed = prev_task is not None and new_timestep > prev_timestep
            if task_completed:
                func = prev_task.get("func") if isinstance(prev_task, dict) else None
                if callable(func) and hasattr(func, "__code__") and "is_obj_swing_onto" in func.__code__.co_names:
                    target = func.__defaults__[0] if func.__defaults__ else None
                    history = self._swing_success_history
                    if target is not None and (not history or history[-1]["target"] is not target):
                        # 记录这次摆动完成的时间戳与目标；仅保留最近两条（必须目标不同）
                        expected_dir = None
                        if isinstance(prev_task, dict):
                            expected_dir = prev_task.get("expected_dir")
                        history.append({"step": cur_step, "target": target, "expected_dir": expected_dir})
                        self._swing_success_history = history[-2:]

            # 当存在最近两次摆动成功（目标不同）时，判断夹爪轨迹在两目标连线的左右侧并打印
            if len(self._swing_success_history) == 2:
                first, second = self._swing_success_history
                pair_key = (first["step"], second["step"])
                if first["target"] is not second["target"] and self._last_swing_pair_reported_step != pair_key:
                    start, end = first["step"], second["step"]
                    segment = [(s, xy) for s, xy in self._gripper_xy_trace if start <= s <= end]
                    # 只保留从第一次时间戳开始的轨迹，避免列表无限增长
                    self._gripper_xy_trace = [(s, xy) for s, xy in self._gripper_xy_trace if s >= start]

                    t1 = torch.as_tensor(first["target"].pose.p[0][:2]).detach().cpu()
                    t2 = torch.as_tensor(second["target"].pose.p[0][:2]).detach().cpu()
                    line_vec = t2 - t1
                    cross_vals = []
                    for _, xy in segment:
                        rel = xy - t1
                        cross_vals.append(float(line_vec[0] * rel[1] - line_vec[1] * rel[0]))

                    if cross_vals:
                        avg_cross = sum(cross_vals) / len(cross_vals)
                        # 根据当前坐标系，正叉积方向应视为顺时针
                        side = "clockwise" if avg_cross > 0 else "counterclockwise" if avg_cross <0 else "on the line"
                        expected_dir = second.get("expected_dir")
                        if expected_dir and side != "on the line" and side != expected_dir:
                            print("direction mistake!!!")
                            self.direction_mistake_flag=True
                        print(f"Gripper path from step {start} to {end} stayed on the {side} side of the directed line between the last two targets.")
                    self._last_swing_pair_reported_step = pair_key

        return {
            "success": self.successflag,
            "fail": self.failureflag,
        }

    def direction_fail(self):
        """
        方向错误失败检查：一旦检测到方向错误，返回 True 供 sequential_task_check 标记失败。
        """
        flag=bool(getattr(self,"direction_mistake_flag",False))
        if flag:
            print("direction_fail triggered")
        return flag


    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):

        reward=torch.tensor([0])
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5


#HistoryBench
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):

        obs, reward, terminated, truncated, info = super().step(action)

        cur_step = int(self.elapsed_steps[0].item())
        highlight_position(
            self,
            self.agent.tcp.pose.p,
            start_step=cur_step,
            end_step=cur_step + 40,
            cur_step=cur_step,
            disk_radius=0.005,
        )


        for idx, button in enumerate(self.buttons_grid):
            if is_obj_swing_onto(self, obj=self.agent.tcp, target=button,distance_threshold=0.03):
                # 更新开始时间以便重复触发时刷新高亮效果
                self.highlight_starts[idx] = cur_step

        for idx, button in enumerate(self.buttons_grid):
            start_step = self.highlight_starts.get(idx)
            if start_step is not None:
                highlight_obj(
                    self,
                    button,
                    start_step=start_step,
                    end_step=start_step + 40,
                    cur_step=cur_step,
                    disk_radius=0.02*1.002,
                    disk_half_length=0.01*2*1.002,
                    highlight_color=[1.0, 0.0, 0.0, 1.0],
                    use_target_style=True,
                )
        return obs, reward, terminated, truncated, info
    

    def _wrong_button_touch(self, expected_button, last_button=None):
        # is_obj_swing_onto 触碰到的按钮既不是当前期望目标，也不是上一个按钮（消抖）时判为错误
        for button in self.buttons_grid:
            if button is expected_button:
                continue
            if last_button is not None and button is last_button:
                continue
            if is_obj_swing_onto(self, obj=self.agent.tcp, target=button):
                return True
        return False
