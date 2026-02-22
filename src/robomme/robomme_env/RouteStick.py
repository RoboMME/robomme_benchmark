


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

#Robomme
import matplotlib.pyplot as plt
import random
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)

from .utils import *
from .utils.subgoal_evaluate_func import *
from .utils.object_generation import *
from .utils import reset_panda
from .utils.route import *
from .utils.subgoal_planner_func import *
from .utils.difficulty import normalize_robomme_difficulty

from ..logging_utils import logger

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


# If direction is reversed, modify evaluate and solve

@register_env("RouteStick")
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

    # Combine into a dictionary
    configs = {
        'hard': config_hard,
        'easy': config_easy,
        'medium': config_medium
    }

    def __init__(self, *args, robot_uids="panda_stick", robot_init_qpos_noise=0,seed=0,Robomme_video_episode=None,Robomme_video_path=None,
                     **kwargs):
        self.achieved_list=[]
        self.use_demonstrationwrapper=False
        self.demonstration_record_traj=False
        self.match=False
        self.after_demo=False
        self.current_task_demonstration = False
        self._gripper_xy_trace=[]
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

        self.seed = seed

        self.robomme_failure_recovery = bool(
            kwargs.pop("robomme_failure_recovery", False)
        )
        self.robomme_failure_recovery_mode = kwargs.pop(
            "robomme_failure_recovery_mode", None
        )
        if isinstance(self.robomme_failure_recovery_mode, str):
            self.robomme_failure_recovery_mode = (
                self.robomme_failure_recovery_mode.lower()
            )
        normalized_robomme_difficulty = normalize_robomme_difficulty(
            kwargs.pop("difficulty", None)
        )
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
            self.difficulty = "easy"
               # Use seed to randomly determine number of repetitions (1-5)
        generator = torch.Generator()
        generator.manual_seed(seed)




        self.highlight_starts = {}  # Use dictionary to store highlight start time for each button
        self._first_non_record_step = None  # Start timestep for delayed highlight

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
        camera_eye=[0.3,0,0.4]
        camera_target =[0,0,-0.2]
        pose = sapien_utils.look_at(
            eye=camera_eye, target=camera_target
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Generate 3x3 grid of buttons
        grid_center = [-0.1, 0]  # Grid center position
        grid_spacing_x = 0.07 # Spacing between buttons
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
            for col in range(num_cols):  # Columns (y direction)
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
                logger.debug(f"Generated target {button_index} at position ({x_pos:.3f}, {y_pos:.3f})")
                button_index += 1

        self.targets_grid = self.buttons_grid

        # Spawn white cubes on specific targets to create fixed obstacles.
        target_cube_indices = [1, 3, 5,7]
        self.target_cube_indices = target_cube_indices
        self.target_cubes = {}
        self.cubes_on_targets = []

        for target_idx in target_cube_indices:
            if target_idx >= len(self.targets_grid):
                logger.debug(f"[SwingAvoid] Skip cube spawn for target {target_idx}: index out of range.")
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

        traj=generate_dynamic_walk(button_indices,steps=steps,allow_backtracking=allow_backtracking,generator=generator)# Generate trajectory
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
            return "left" if y_val > ref_y else "right"# Reverse according to robot perspective!

        # Randomly decide and record clockwise/counterclockwise direction for each solve_swingonto_withDirection
        self.swing_directions = []
        for _ in self.selected_buttons[1:]:
            dir_flag = "clockwise" if torch.rand(1, generator=generator).item() < 0.5 else "counterclockwise"
            self.swing_directions.append(dir_flag)
        logger.debug(f"[RouteStick] swing direction list: {self.swing_directions}")

        current_target=self.selected_buttons[0]
        tasks.append({
            "func":   lambda t=current_target: is_obj_swing_onto(self, obj=self.agent.tcp, target=t, distance_threshold=0.03, z_threshold=self.z_threshold),
            "name":  "NO RECORD",
            "subgoal_segment":f"NO RECORD",
            "choice_label": "pick up the stick",
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
            tasks.append({# Decrease threshold to see if replay appears
            "func":   lambda t=current_target: is_obj_swing_onto(self, obj=self.agent.tcp, target=t, distance_threshold=0.03, z_threshold=self.z_threshold),
            "name": task_name,
            "subgoal_segment":task_name,
            "choice_label": task_name,
            "demonstration": True,
            "failure_func":  (lambda expected=current_target, last=prev_target: self._wrong_button_touch(expected_button=expected, last_button=last)),
            "expected_dir": direction,
            "solve": lambda env, planner, t=current_target, d=direction: solve_swingonto_withDirection(env, planner, target=t,radius=0.2,direction=d),
                })  
        tasks.append({
                    "func": lambda:reset_check(self,gripper="stick"),
                    "name": "NO RECORD",
                    "subgoal_segment":"NO RECORD",
                    "choice_label": "place the stick into the tube",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [solve_strong_reset(env,planner,timestep=200,gripper="stick")],
                    },),
        
        current_target=self.selected_buttons[0]
        tasks.append({
            "func":   lambda:reset_check(self,gripper="stick",target_qpos=self.swing_qpos),
            "name":  "NO RECORD",
            "subgoal_segment":f"NO RECORD",
            "choice_label": "pick up the stick",
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
            "func":   lambda t=current_target,list=[current_target, prev_target,direction]: is_obj_swing_onto(self, obj=self.agent.tcp, target=t, distance_threshold=0.03, z_threshold=self.z_threshold,judge_direction_list=list),
            "name": task_name,
            "subgoal_segment":task_name,
            "choice_label": task_name,
            "demonstration": False,
            "failure_func":  (lambda expected=current_target, last=prev_target: [self._wrong_button_touch(expected_button=expected, last_button=last)]),
            "expected_dir": direction,
            "solve": lambda env, planner, t=current_target, d=direction: solve_swingonto_withDirection(env, planner, target=t,radius=0.2,direction=d),
        })  
            # Store task list for RecordWrapper use
        self.task_list = tasks



    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):

            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos=reset_panda.get_reset_panda_param("qpos",gripper="stick")
            self.agent.reset(qpos)
            self.failureflag = torch.tensor([False])

     


    def _get_obs_extra(self, info: Dict):
        return dict()




    def evaluate(self,solve_complete_eval=False):
        previous_failure = getattr(self, "failureflag", torch.tensor([False]))
        if isinstance(previous_failure, torch.Tensor):
            failure_latched = bool(previous_failure.detach().cpu().item())
        else:
            failure_latched = bool(previous_failure)
        self.successflag = torch.tensor([False])
        self.failureflag = torch.tensor([True]) if failure_latched else torch.tensor([False])
        had_latched_fail = failure_latched



        # Record gripper xy position of current step (only record after demonstration is enabled)
        if  self.current_task_demonstration == False:
            gripper_xy = torch.as_tensor(self.agent.tcp.pose.p[0][:2]).detach().cpu()
            self._gripper_xy_trace.append((self.elapsed_steps, gripper_xy))

        # Use encapsulated sequence task check function
        if(self.use_demonstrationwrapper==False):# change subgoal after planner ends during recording
            if solve_complete_eval==True:
                allow_subgoal_change_this_timestep=True
            else:
                allow_subgoal_change_this_timestep=False
        else:# during demonstration, video needs to call evaluate(solve_complete_eval) video ends and flag changes in demonstrationwrapper
            if solve_complete_eval==True or self.demonstration_record_traj==False:
                allow_subgoal_change_this_timestep=True
            else:
                allow_subgoal_change_this_timestep=False
        all_tasks_completed, current_task_name, task_failed,self.current_task_specialflag = sequential_task_check(self, self.task_list,allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep)
     
        

        # If task failed, mark as failed immediately, and keep fail thereafter
        if task_failed:
            self.failureflag = torch.tensor([True])
            if not had_latched_fail:
                logger.debug(f"Task failed: {current_task_name}")

        # If static_check succeeds or all tasks completed, set success flag
        if all_tasks_completed and not task_failed:
            self.successflag = torch.tensor([True])


        # # Check if a swing task has just been completed
        # if after_demo_active:
        #     # If current_task_name jumps, record adjacent two targets in order
        #     last_logged_name = getattr(self, "_last_logged_task_name", None)
        #     if current_task_name != last_logged_name:
        #         change_idx = getattr(self, "_swing_task_change_idx", 0)
        #         buttons = getattr(self, "selected_buttons", [])
        #         if change_idx + 1 < len(buttons):
        #             first_target = buttons[change_idx]
        #             second_target = buttons[change_idx + 1]
        #             expected_dir = (
        #                 self.swing_directions[change_idx]
        #                 if change_idx < len(getattr(self, "swing_directions", []))
        #                 else None
        #             )
        #             self._swing_success_history = [
        #                 {"step": cur_step, "target": first_target, "expected_dir": None},
        #                 {"step": cur_step, "target": second_target, "expected_dir": expected_dir},
        #             ]
        #             self._swing_task_change_idx = change_idx + 1
        #         self._last_logged_task_name = current_task_name

        #     # When there are two recent successful swings (different targets), judge gripper trajectory on left/right side of line between two targets and print
        #     if len(self._swing_success_history) == 2:
        #         first, second = self._swing_success_history
        #         pair_key = (first["step"], second["step"])
        #         if first["target"] is not second["target"] and self._last_swing_pair_reported_step != pair_key:
        #             start, end = first["step"], second["step"]
        #             segment = [(s, xy) for s, xy in self._gripper_xy_trace if start <= s <= end]
        #             # Only keep trajectory from first timestamp, avoid list growing infinitely
        #             self._gripper_xy_trace = [(s, xy) for s, xy in self._gripper_xy_trace if s >= start]

        #             t1 = torch.as_tensor(first["target"].pose.p[0][:2]).detach().cpu()
        #             t2 = torch.as_tensor(second["target"].pose.p[0][:2]).detach().cpu()
        #             line_vec = t2 - t1
        #             cross_vals = []
        #             for _, xy in segment:
        #                 rel = xy - t1
        #                 cross_vals.append(float(line_vec[0] * rel[1] - line_vec[1] * rel[0]))

        #             if cross_vals:
        #                 avg_cross = sum(cross_vals) / len(cross_vals)
        #                 # According to current coordinate system, positive cross product direction should be considered clockwise
        #                 side = "clockwise" if avg_cross > 0 else "counterclockwise" if avg_cross <0 else "on the line"
        #                 expected_dir = second.get("expected_dir")
        #                 if expected_dir and side != "on the line" and side != expected_dir:
        #                     print("direction mistake!!!")
        #                 print(f"Gripper path from step {start} to {end} stayed on the {side} side of the directed line between the last two targets.")
        #             self._last_swing_pair_reported_step = pair_key

        return {
            "success": self.successflag,
            "fail": self.failureflag,
        }

    def direction_fail(self,judge_direction_list=None):
        if judge_direction_list is None:
            return True

        # judge_direction_list format is [current_target, prev_target, expected_dir]
        try:
            current_target, prev_target, expected_dir = judge_direction_list
        except Exception:
            # Unable to judge direction when parameter is abnormal, keep unfinished status
            self.failureflag = torch.tensor([True])
            return False

        trace = getattr(self, "_gripper_xy_trace", [])

        # Calculate vector from previous target to current target
        prev_xy = torch.as_tensor(prev_target.pose.p[0][:2]).detach().cpu()
        curr_xy = torch.as_tensor(current_target.pose.p[0][:2]).detach().cpu()
        line_vec = curr_xy - prev_xy

        # If two targets coincide or no trajectory, unable to judge direction
        if torch.norm(line_vec) < 1e-6 or len(trace) == 0:
            self.failureflag = torch.tensor([True])
            return False

        cross_vals = []
        for _, xy in trace:
            xy = torch.as_tensor(xy).detach().cpu()
            rel = xy - prev_xy
            cross_vals.append(float(line_vec[0] * rel[1] - line_vec[1] * rel[0]))

        # Clear trajectory immediately after consumption, convenient for next segment judgment
        self._gripper_xy_trace = []

        if len(cross_vals) == 0:
            self.failureflag = torch.tensor([True])
            return False

        avg_cross = sum(cross_vals) / len(cross_vals)
        # Positive cross product in current coordinate system is considered clockwise
        side = "clockwise" if avg_cross > 0 else "counterclockwise" if avg_cross < 0 else "on the line"

        # Unable to judge direction along line; need to retry
        if side == "on the line":
            self.failureflag = torch.tensor([True])
            return False

        expected_dir = str(expected_dir).lower()
        if side != expected_dir:
            logger.debug(f"direction mistake: expected {expected_dir}, got {side}")
            self.failureflag = torch.tensor([True])
            return False

        return True





    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):

        reward=torch.tensor([0])
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5


#Robomme
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
                # Update start time to refresh highlight effect when triggered repeatedly
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
        # Judge as error when touched button is neither current expected target nor previous button (debounce)
        for button in self.buttons_grid:
            if button is expected_button:
                continue
            if last_button is not None and button is last_button:
                continue
            if is_obj_swing_onto(self, obj=self.agent.tcp, target=button):
                return True
        return False
