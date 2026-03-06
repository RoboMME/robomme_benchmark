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

#Robomme
import matplotlib.pyplot as plt

import random
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)

from .utils.SceneGenerationError import SceneGenerationError
from .utils import *
from .utils.subgoal_evaluate_func import static_check
from .utils.object_generation import spawn_fixed_cube, build_board_with_hole
from .utils import reset_panda
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


@register_env("VideoRepick")
class VideoRepick(BaseEnv):

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
        "cube":3,
        "swap_min":1,
        "swap_max":2,
    }
    config_medium= {
        "cube":3,
        "swap_min":2,
        "swap_max":3,
    }
    config_hard = {
        "cluster":True,
        "swap":None,
        "swap_min":0,
        "swap_max":0,
    }


    # Combine into a dictionary
    configs = {
        'hard': config_hard,
        'easy': config_easy,
        'medium': config_medium
    }


    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0,seed=0,Robomme_video_episode=None,Robomme_video_path=None,
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

        np.random.seed(seed)
        normalized_robomme_difficulty = normalize_robomme_difficulty(
            kwargs.pop("difficulty", None)
        )
        if normalized_robomme_difficulty is not None:
            self.difficulty = normalized_robomme_difficulty
        else:
            seed_mod = seed % 3
            if seed_mod == 0:
                self.difficulty = "easy"
            elif seed_mod == 1:
                self.difficulty = "medium"
            else:  # seed_mod == 2
                self.difficulty = "hard"
        #self.difficulty = "hard"
        # Use seed to randomly determine number of repetitions (1-5)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.num_repeats = torch.randint(1, 4, (1,), generator=self.generator).item()
        logger.debug(f"Task will repeat {self.num_repeats} times (pickup-drop cycles)")

        self.swap_times = torch.randint(self.configs[self.difficulty]['swap_min'], self.configs[self.difficulty]['swap_max']+1, (1,), generator=self.generator).item()
        logger.debug(f"Task will swap {self.swap_times} times")


        self.static_flag=False
        self.start_step=99999
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
        try:
            self.table_scene = TableSceneBuilder(
                self, robot_init_qpos_noise=self.robot_init_qpos_noise
            )
            self.table_scene.build()

            button_obb_1 = build_button(
                self,
                center_xy=(-0.2, 0),
                scale=1.5,
                generator=self.generator,
                name="button",
                randomize=True,
                randomize_range=(0.1, 0.1)
            )
            # Store first button before building second one
            self.button_left = self.button
            self.button_joint_1 = self.button_joint

            avoid = [button_obb_1]

            options = [
                {"color": (1, 0, 0, 1), "name": "red"},
                {"color": (0, 0, 1, 1), "name": "blue"},
                {"color": (0, 1, 0, 1), "name": "green"},
            ]
            if self.difficulty == "hard":
                self.spawned_cubes = []

                for idx in range(5):
                    shuffle_indices = torch.randperm(len(options), generator=self.generator).tolist()
                    new_options = [options[i] for i in shuffle_indices]
                    for group in new_options:
                        try:
                            cube = spawn_random_cube(
                                self,
                                color=group["color"],
                                avoid=avoid,
                                include_existing=False,
                                include_goal=False,
                                region_center=[-0.1, 0],
                                region_half_size=[0.2, 0.25],
                                half_size=self.cube_half_size,
                                min_gap=self.cube_half_size,
                                random_yaw=True,
                                name_prefix=f"cube_{group['name']}_{idx}",
                                generator=self.generator,
                            )
                        except RuntimeError as e:
                            raise SceneGenerationError(
                                f"Failed to generate {group['name']} cube {idx}"
                            ) from e

                        self.spawned_cubes.append(cube)
                        avoid.append(cube)

                if not self.spawned_cubes:
                    raise SceneGenerationError("Failed to generate any cube")

                target_idx = torch.randint(0, len(self.spawned_cubes), (1,), generator=self.generator).item()
                logger.debug("target index: %s", target_idx)
                self.target_cube_1 = self.spawned_cubes[target_idx]

            else:
                idx = torch.randint(0, len(options), (1,), generator=self.generator).item()
                chosen_color = options[idx]["color"]

                cube_colors = [chosen_color] * 4
                shuffle_indices = torch.randperm(len(cube_colors), generator=self.generator).tolist()
                cube_colors = [cube_colors[i] for i in shuffle_indices]

                self.spawned_cubes = []

                region4 = [[-0.05, -0.1], [-0.05, 0.1], [0.1, 0.1], [0.1, -0.1]]
                region3_tri = [[-0.05, -0.1], [-0.05, 0.1], [0.1, 0]]
                region3_line = [[0, -0.15], [0, 0.15], [0, 0]]

                region3_choice = torch.randint(0, 2, (1,), generator=self.generator).item()
                region3 = region3_tri if region3_choice == 0 else region3_line

                if self.configs[self.difficulty]['cube'] == 4:
                    region = region4
                else:
                    region = region3
                angle, region = rotate_points_random(region, (0, 180), self.generator)

                for i in range(self.configs[self.difficulty]['cube']):
                    try:
                        cube_actor = spawn_random_cube(
                            self,
                            avoid=avoid,
                            region_center=region[i],
                            region_half_size=0.07,
                            min_gap=self.cube_half_size * 1,
                            half_size=self.cube_half_size,
                            name_prefix=f"bin_{i}",
                            max_trials=256,
                            color=cube_colors[i],
                            generator=self.generator

                        )
                    except RuntimeError as e:
                        raise SceneGenerationError(f"Failed to generate bin_{i}") from e

                    self.spawned_cubes.append(cube_actor)
                    setattr(self, f"bin_{i}", cube_actor)
                    avoid.append(cube_actor)

                if not self.spawned_cubes:
                    raise SceneGenerationError("Failed to generate any bin")

                target_indices = torch.randperm(len(self.spawned_cubes), generator=self.generator)[:1].tolist()
                self.target_cube_1 = self.spawned_cubes[target_indices[0]]

                if self.difficulty != "hard":
                    remaining_indices = [i for i in range(len(self.spawned_cubes)) if i not in target_indices]
                    if len(remaining_indices) < 2:
                        raise SceneGenerationError("Not enough cubes for swapping")

                    selected_remaining = torch.randperm(len(remaining_indices), generator=self.generator)[:2].tolist()
                    selected_indices = [remaining_indices[i] for i in selected_remaining]
                    swap_indices = target_indices + selected_indices

                    self.swap_pair1_idx1 = self.spawned_cubes[swap_indices[0]]
                    self.swap_pair2_idx1 = self.spawned_cubes[swap_indices[1]]
                    self.swap_pair3_idx1 = self.spawned_cubes[swap_indices[2]]
                    self.swap_pair1_idx2 = None
                    self.swap_pair2_idx2 = None
                    self.swap_pair3_idx2 = None
                    self._refresh_swap_schedule()
        except SceneGenerationError:
            raise
        except Exception as exc:
            raise SceneGenerationError(
                f"Failed to load VideoRepick scene for seed {self.seed}"
            ) from exc




    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos=reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)
            tasks = [
            {
                "func": (lambda: is_obj_pickup(self, obj=self.target_cube_1)),
                "name": f"pick up the cube",
                "subgoal_segment":f"pick up the cube at <>",
                "choice_label": "pick up the cube",
                "demonstration": True,
                "failure_func": lambda:None,
                "solve": lambda env, planner: [solve_pickup(env, planner, obj=self.target_cube_1)],
                'segment':self.target_cube_1,
            },{
                "func": (lambda: is_obj_dropped(self, obj=self.target_cube_1)),
                "name": "drop the cube on the table",
                "subgoal_segment":f"drop the cube on the table",
                "choice_label": "put it down",
                "demonstration": True,
                "failure_func": lambda: None,
                "solve": lambda env, planner: [solve_putdown_whenhold(env, planner,release_z=0.03)]
                }, 
            ]
            if self.swap_times>=1:
                tasks.append(   {
                                "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=20),
                                "name": "static",
                                "subgoal_segment":"static",
                                "demonstration": True,
                                "failure_func": None,
                                "solve": lambda env, planner: [solve_reset(env,planner),solve_hold_obj(env, planner, static_steps=20)],
                                },)
            if self.swap_times>=1:
                for count in range(self.swap_times):
                    tasks.append(   {
                                "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=self.swap_schedule[-1][3]-self.swap_schedule[-1][2]),
                                "name": "static",
                                "subgoal_segment":"static",
                                "demonstration": True,
                                "failure_func": None,
                                "specialflag":"swap",
                                "solve": lambda env, planner: [solve_hold_obj(env, planner, static_steps=self.swap_schedule[-1][3]-self.swap_schedule[-1][2])],
                                },)
                
            tasks.append(             {
                                "func": lambda:reset_check(self),
                                "name": "NO RECORD",
                                "subgoal_segment":"NO RECORD",
                                "demonstration": True,
                                "failure_func": None,
                                "solve": lambda env, planner: [ solve_strong_reset(env,planner)],
                                },)
            ordinal_words = [
                "first",
                "second",
                "third",
                "fourth",
                "fifth",
                "sixth",
                "seventh",
                "eighth",
                "ninth",
                "tenth",
            ]
            for i in range(self.num_repeats):
                ordinal = ordinal_words[i] if i < len(ordinal_words) else f"{i+1}th"
                tasks.append(  {
                        "func": (lambda: is_obj_pickup(self, obj=self.target_cube_1)),
                        "name": f"pick up the correct cube for the {ordinal} time" ,
                        "subgoal_segment":f"pick up the correct cube at <> for the {ordinal} time" ,
                        "choice_label": "pick up the cube",
                        "demonstration": False,
                        "failure_func": lambda: [
                            is_any_obj_pickup(self,[cube for cube in self.spawned_cubes if cube != self.target_cube_1]),
                            timewindow(self, lambda: is_button_pressed(self, obj=self.button_left),min_steps=50,max_steps=500,timewindow_timer=2,),],
                        "solve": lambda env, planner: [solve_pickup(env, planner, obj=self.target_cube_1)],
                        'segment':self.target_cube_1,
                    },)
                
                tasks.append({
                        "func": lambda: is_obj_dropped(self,obj=self.target_cube_1),
                    "name": "put it down",
                    "subgoal_segment":f"put it down",
                    "choice_label": "put it down",
                        "demonstration": False,
                        "failure_func": lambda:[
                            is_any_obj_pickup(self,[cube for cube in self.spawned_cubes if cube != self.target_cube_1]),
                            timewindow(self, lambda: is_button_pressed(self, obj=self.button_left),min_steps=50,max_steps=500,timewindow_timer=3,),], 
                        "solve": lambda env, planner: solve_putdown_whenhold(env, planner,release_z=0.01)
                    })

            tasks.append({
                    "func": lambda: is_button_pressed(self, obj=self.button_left),
                    "name": "press the button to finish",
                    "subgoal_segment":f"press the button at <> to finish",
                    "choice_label": "press the button to finish",
                    "demonstration": False,
                    "failure_func":lambda: is_any_obj_pickup(self,[cube for cube in self.spawned_cubes]),
                    "solve": lambda env, planner: solve_button(env, planner, obj=self.button_left),
                    "segment":self.cap_link 
                })



        # Store task list for RecordWrapper use
        self.task_list = tasks

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
        previous_failure = getattr(self, "failureflag", None)
        self.successflag = torch.tensor([False])
        if previous_failure is not None and bool(previous_failure.item()):
            self.failureflag = previous_failure
        else:
            self.failureflag = torch.tensor([False])


        # Use encapsulated sequence task check function
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
        all_tasks_completed, current_task_name, task_failed,self.current_task_specialflag = sequential_task_check(self, self.task_list,allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep)

        # If task failed, mark as failed immediately
        if task_failed:
            self.failureflag = torch.tensor([True])
            logger.debug(f"Task failed: {current_task_name}")

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

    def _get_other_bins_for_pair(self, idx_a: int, idx_b: int):
        """Return bins that are not part of the provided pair indices."""
        if not hasattr(self, "spawned_bins"):
            return []

        total_bins = len(self.spawned_cubes)
        if idx_a >= total_bins or idx_b >= total_bins:
            return []

        # Prefer precomputed lists when available
        if hasattr(self, "otherbins") and idx_a < len(self.otherbins):
            other_candidates = [
                bin_actor
                for bin_actor in self.otherbins[idx_a]
                if bin_actor is not self.spawned_cubes[idx_b]
            ]
            return other_candidates

        return [
            bin_actor
            for i, bin_actor in enumerate(self.spawned_cubes)
            if i not in (idx_a, idx_b)
        ]

    def _get_actor_position(self, actor):
        """Return actor position as a numpy array."""
        if actor is None:
            return np.zeros(3, dtype=np.float32)

        pos = actor.pose.p if hasattr(actor, "pose") else actor.get_pose().p
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().numpy()

        pos = np.asarray(pos, dtype=np.float32).reshape(-1)
        if pos.size < 3:
            padded = np.zeros(3, dtype=np.float32)
            padded[: pos.size] = pos
            return padded
        return pos

    def _compute_dynamic_swap_candidates(self, positions):
        """Compute nearest-neighbour swap candidates using provided positions."""
        candidate_map = {}
        num_positions = len(positions)
        if num_positions <= 1:
            return candidate_map

        for idx, pos in enumerate(positions):
            distances = []
            for other_idx, other_pos in enumerate(positions):
                if other_idx == idx:
                    continue
                dist = np.linalg.norm(pos[:2] - other_pos[:2])
                distances.append((other_idx, dist))

            distances.sort(key=lambda item: item[1])
            candidate_map[idx] = [j for j, _ in distances[:2]]

        return candidate_map

    def _select_swap_pair_from_positions(self, positions, generator=None):
        """Select one swap pair given current planned positions."""
        num_bins = len(positions)
        if num_bins < 2:
            return None

        candidate_map = self._compute_dynamic_swap_candidates(positions)
        valid_indices = [idx for idx, cands in candidate_map.items() if cands]
        if not valid_indices:
            return None

        if generator is None:
            generator = self.generator

        first_idx = valid_indices[
            int(torch.randint(0, len(valid_indices), (1,), generator=generator).item())
        ]
        candidates = candidate_map[first_idx]
        second_idx = candidates[
            int(torch.randint(0, len(candidates), (1,), generator=generator).item())
        ]

        distance = float(
            np.linalg.norm(positions[first_idx][:2] - positions[second_idx][:2])
        )

        return {"idx1": first_idx, "idx2": second_idx, "distance": distance}

    def _refresh_swap_schedule(self,start_step=400):
        if self.swap_times==1:
                    self.swap_schedule = [
                        (self.swap_pair1_idx1, self.swap_pair1_idx2, start_step, start_step + 50),
                        ]# Final swap order
        elif self.swap_times==2:
            self.swap_schedule = [
                    (self.swap_pair1_idx1, self.swap_pair1_idx2, start_step, start_step + 50),
                    (self.swap_pair2_idx1, self.swap_pair2_idx2, start_step + 50, start_step + 50 * 2),
                ]# Final swap order
        elif self.swap_times==3:
            self.swap_schedule = [
                    (self.swap_pair1_idx1, self.swap_pair1_idx2, start_step, start_step + 50),
                    (self.swap_pair2_idx1, self.swap_pair2_idx2, start_step + 50, start_step + 50 * 2),
                    (self.swap_pair3_idx1, self.swap_pair3_idx2, start_step + 50 * 2, start_step + 50 * 3),
            ]

#Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):


       
        if self.current_task_specialflag=="swap":
            if self.static_flag==False:
                self.static_flag=True
                self.start_step=int(self.elapsed_steps.item())
                self._refresh_swap_schedule(self.start_step)
                logger.debug("tag!")
             
        if self.static_flag==True:
            for i in range(len(self.swap_schedule)):
                start = self.swap_schedule[i][2]
                end = self.swap_schedule[i][3]
                if self.elapsed_steps in range (start,end):
                    # Select corresponding swap pair based on index
                    pair_idx1 = getattr(self, f'swap_pair{i+1}_idx1')
                    pair_idx2 = getattr(self, f'swap_pair{i+1}_idx2')

                    if pair_idx2 is None and pair_idx1 is not None:
                        reference_pos = self._get_actor_position(pair_idx1)
                        closest_actor = None
                        closest_dist = float("inf")
                        for candidate in self.spawned_cubes:
                            if candidate is None or candidate is pair_idx1:
                                continue
                            candidate_pos = self._get_actor_position(candidate)
                            dist = np.linalg.norm(reference_pos[:2] - candidate_pos[:2])
                            if dist < closest_dist:
                                closest_dist = dist
                                closest_actor = candidate
                        if closest_actor is not None:
                            setattr(self, f'swap_pair{i+1}_idx2', closest_actor)
                            self._refresh_swap_schedule(self.start_step)


            for idx_a, idx_b, start_step, end_step in self.swap_schedule:
                if idx_a is None or idx_b is None:
                    continue
                if self.elapsed_steps >= int(start_step) and self.elapsed_steps <= int(end_step):
                        
                    swap_flat_two_lane(
                                    self,
                                    cube_a=idx_a,
                                    cube_b=idx_b,
                                    start_step=start_step,
                                    end_step=end_step,
                                    cur_step=self.elapsed_steps,
                                    lane_offset=0.07,
                                    smooth=True,
                                    keep_upright=True,
                                    other_cube=[b for b in self.spawned_cubes if b not in (idx_a, idx_b)],  # Keep all other bins in place to prevent collision during swap
                                )



        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info
