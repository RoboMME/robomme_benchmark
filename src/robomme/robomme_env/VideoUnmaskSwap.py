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

from .utils import *
from .utils.subgoal_evaluate_func import static_check
from .utils.object_generation import spawn_fixed_cube, build_board_with_hole, build_bin
from .utils import reset_panda
from .utils.difficulty import normalize_robomme_difficulty
from .utils.SceneGenerationError import SceneGenerationError
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


@register_env("VideoUnmaskSwap")
class VideoUnmaskSwap(BaseEnv):

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
        "bin":3,
        "swap_min":1,
        "swap_max":2,
        "pick_min":1,
        "pick_max":2
    }
    config_medium= {
        "bin":4,
        "swap_min":1,
        "swap_max":2,
        "pick_min":1,
        "pick_max":1
    }
    config_hard = {
        "bin":4,
        "swap_min":2,
        "swap_max":3,
        "pick_min":2,
        "pick_max":2
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

        # Use seed to randomly determine number of repetitions (1-5)
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.swap_times = torch.randint(self.configs[self.difficulty]['swap_min'], self.configs[self.difficulty]['swap_max']+1, (1,), generator=generator).item()
        logger.debug(f"Task will swap {self.swap_times} times")


        self.pick_times = torch.randint(self.configs[self.difficulty]['pick_min'], self.configs[self.difficulty]['pick_max']+1, (1,), generator=generator).item()
        logger.debug(f"Task will pick {self.pick_times} times")



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
        generator.manual_seed(self.seed)

        # --- 1. 通过 permanance_task_pos_generator 拿位置（n_buttons=0, n_swaps=swap_times）
        n_bins = self.configs[self.difficulty]['bin']
        n_swaps = self.swap_times
        n_pickups = self.pick_times
        result = permanance_task_pos_generator(
            n_bins=n_bins, n_swaps=n_swaps, n_buttons=0,
            n_pickups=n_pickups, seed=self.seed,
        )
        if "fail" in result:
            raise SceneGenerationError(result["fail"])

        bin_positions = result["bin_positions"]
        bin_colors = result["bin_colors"]
        pickup_map = result["pickup_map"]
        swap_pairs_raw = result["swap_pairs"]

        # --- 2. pickup-first 重排：pickup 0..n_pickups-1 → 剩余 colored → no_cube
        inv_pickup = {order: raw for raw, order in pickup_map.items()}
        pickup_indices_raw = [inv_pickup[k] for k in range(n_pickups)]
        other_colored_raw = sorted(
            raw for raw, c in enumerate(bin_colors)
            if c != "no_cube" and raw not in pickup_indices_raw
        )
        no_cube_raw = sorted(
            raw for raw, c in enumerate(bin_colors) if c == "no_cube"
        )
        perm = pickup_indices_raw + other_colored_raw + no_cube_raw
        inv_perm = {raw: new for new, raw in enumerate(perm)}

        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # --- 3. 按重排顺序建 bins
        self.spawned_bins = []
        for new_idx, raw_idx in enumerate(perm):
            x, y = bin_positions[raw_idx]
            z_rotation = float(torch.rand(1, generator=generator).item() * 90.0)
            bin_actor = build_bin(
                self,
                callsign=f"bin_{new_idx}",
                position=[x, y, 0.002],
                z_rotation_deg=z_rotation,
            )
            self.spawned_bins.append(bin_actor)
            setattr(self, f"bin_{new_idx}", bin_actor)

        # --- 4. 颜色重排 + 建 cubes（前 3 个 bin 必为 colored）
        cube_color_to_rgba = {
            "red": (1, 0, 0, 1),
            "green": (0, 1, 0, 1),
            "blue": (0, 0, 1, 1),
        }
        permuted_bin_colors = [bin_colors[perm[i]] for i in range(n_bins)]
        # color_names[i] 对应 self.bin_i 的 cube 颜色（前 3 个 bin 都有 cube）
        self.color_names = permuted_bin_colors[:3]

        spawned_dynamic_cubes = []
        self.cube_bin_pairs = []
        self.bin_to_cube = {}
        self.bin_to_color = {}
        self.spawned_dynamic_cubes = spawned_dynamic_cubes

        # --- 5. selected_bins / selected_bin_indices（长度 3：3 个 colored bin，
        # 前 n_pickups 个是 pickup 目标，task_list 引用 selected_bins[0/1]）
        self.selected_bin_indices = list(range(min(3, n_bins)))
        self.selected_bins = self.spawned_bins[:3]

        for i in range(min(3, n_bins)):
            color = permuted_bin_colors[i]
            bin_actor = self.spawned_bins[i]
            bin_pos = bin_actor.pose.p
            if isinstance(bin_pos, torch.Tensor):
                bin_pos = bin_pos[0].detach().cpu().numpy()

            cube_position = [bin_pos[0], bin_pos[1]]
            cube_actor = spawn_fixed_cube(
                self,
                position=cube_position,
                half_size=self.cube_half_size / 1.2,
                color=cube_color_to_rgba[color],
                name_prefix=f"target_cube_{color}",
                yaw=0.0,
                dynamic=True,
            )

            spawned_dynamic_cubes.append(cube_actor)
            setattr(self, f"target_cube_{color}", cube_actor)
            setattr(self, f"target_cube_{i}", cube_actor)
            setattr(self, f"target_cube_for_bin_{i}", cube_actor)
            self.cube_bin_pairs.append((cube_actor, bin_actor))
            self.bin_to_cube[i] = cube_actor
            self.bin_to_color[i] = color

        self.cube_bins = self.selected_bins[:]
        self.cube_bin_indices = self.selected_bin_indices[:]

        # --- 6. target_* 字段（保留原 _load_scene 的 target_choice / target_indices 段）
        self.target_bin = None
        self.target_bin_index = None
        self.target_cube = None
        self.target_cube_color = None
        self.other_cube_bins = []
        self.other_cube_bin_indices = []
        self.other_cubes = []

        if self.cube_bin_pairs:
            target_choice = int(
                torch.randint(
                    len(self.cube_bin_pairs),
                    (1,),
                    generator=generator,
                ).item()
            )
            target_cube_actor, target_bin_actor = self.cube_bin_pairs[target_choice]
            self.target_cube = target_cube_actor
            self.target_bin = target_bin_actor
            self.target_bin_index = self.selected_bin_indices[target_choice]
            self.target_cube_color = self.color_names[target_choice]
            self.target_cube_name = (
                getattr(target_cube_actor, "name", None)
                or f"target_cube_{self.target_cube_color}"
            )
            self.target_label = self.target_cube_color or self.target_cube_name or "target"

            for idx_i, (cube_actor, bin_actor) in enumerate(self.cube_bin_pairs):
                if idx_i == target_choice:
                    continue
                self.other_cube_bins.append(bin_actor)
                self.other_cube_bin_indices.append(self.selected_bin_indices[idx_i])
                self.other_cubes.append(cube_actor)
        else:
            self.target_cube = None
            self.target_bin = None
            self.target_bin_index = None
            self.target_cube_color = None
            self.target_cube_name = None
            self.target_label = "target"

        # Randomly select 2 unique bins as target_bin_1 and target_bin_2
        target_indices = torch.randperm(
            len(self.selected_bin_indices), generator=generator,
        )[:2]
        self.target_bin_1 = self.selected_bins[target_indices[0]]
        self.target_bin_2 = self.selected_bins[target_indices[1]]
        self.target_bin_1_cube_color = self.color_names[target_indices[0].item()]
        self.target_bin_2_cube_color = self.color_names[target_indices[1].item()]

        # --- 7. swap_pairs 索引重映射 + 设属性
        for pair_idx in range(1, 4):
            setattr(self, f"swap_pair{pair_idx}_idx1", None)
            setattr(self, f"swap_pair{pair_idx}_idx2", None)
        for k, (raw_a, raw_b) in enumerate(swap_pairs_raw):
            setattr(
                self,
                f"swap_pair{k + 1}_idx1",
                self.spawned_bins[inv_perm[raw_a]],
            )
            setattr(
                self,
                f"swap_pair{k + 1}_idx2",
                self.spawned_bins[inv_perm[raw_b]],
            )
        self._refresh_swap_schedule()

        tasks = [
             {
                        "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=self.swap_schedule[-1][3]),
                        "name": "static",
                        "subgoal_segment": "static",
                        "choice_label": "static",
                        "demonstration": True,
                        "failure_func": None,
                        "solve": lambda env, planner: solve_hold_obj(env, planner, static_steps=self.swap_schedule[-1][3]),
                        },

            
            {
                "func": (lambda: is_bin_pickup(self, obj=self.selected_bins[0])),
                "name": f"pick up the container that hides the {self.color_names[0]} cube",
                "subgoal_segment":f"pick up the container at <> that hides the {self.color_names[0]} cube",
                "choice_label": "pick up the container",
                "demonstration": False,
                "failure_func": lambda: is_any_bin_pickup(self, [bin for bin in self.spawned_bins if bin != self.selected_bins[0]]),
                "solve": lambda env, planner: solve_pickup_bin(env, planner, obj=self.selected_bins[0]),
                "segment":self.selected_bins[0],
            },
        ]
        if self.pick_times==2:
            tasks.append({
                    "func": (lambda: is_bin_putdown(self, obj=self.selected_bins[0])),
                    "name": "put down the container",
                    "subgoal_segment":"put down the container",
                    "choice_label": "put down the container",
                    "demonstration": False,
                    "failure_func": lambda:is_any_bin_pickup(self,[bin for bin in self.spawned_bins if bin != self.selected_bins[0]]),
                    "solve": lambda env, planner: solve_putdown_whenhold(env, planner,),

                })
            tasks.append(
                {
                    "func": (lambda: is_bin_pickup(self, obj=self.selected_bins[1])),
                    "name": f"pick up the container that hides the {self.color_names[1]} cube",
                    "subgoal_segment":f"pick up the container at <> that hides the {self.color_names[1]} cube",
                    "choice_label": "pick up the container",
                    "demonstration": False,
                    "failure_func": lambda: is_any_bin_pickup(self,[bin for bin in self.spawned_bins if bin != self.selected_bins[1]]),
                    "solve": lambda env, planner: solve_pickup_bin(env, planner, obj=self.selected_bins[1]),
                    "segment":self.selected_bins[1],
                })

        # Store task list for RecordWrapper use
        self.task_list = tasks

        # Record pickup related task indices and items for recovery
        self.recovery_pickup_indices, self.recovery_pickup_tasks = task4recovery(self.task_list)
        if self.robomme_failure_recovery:
            # Only inject an intentional failed grasp when recovery mode is enabled
            self.fail_grasp_task_index = inject_fail_grasp(
                self.task_list,
                generator=generator,
                mode=self.robomme_failure_recovery_mode,
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



    def _refresh_swap_schedule(self):
        if self.swap_times==1:
                    self.swap_schedule = [
                            (self.swap_pair1_idx1, self.swap_pair1_idx2, 64, 64 + 50),
                        ]# Final swap order
        elif self.swap_times==2:
            self.swap_schedule = [
                    (self.swap_pair1_idx1, self.swap_pair1_idx2, 64, 64 + 50),
                    (self.swap_pair2_idx1, self.swap_pair2_idx2, 64 + 50, 64 + 50 * 2),
                ]# Final swap order
        elif self.swap_times==3:
            self.swap_schedule = [
                    (self.swap_pair1_idx1, self.swap_pair1_idx2, 64, 64 + 50),
                    (self.swap_pair2_idx1, self.swap_pair2_idx2, 64 + 50, 64 + 50 * 2),
                    (self.swap_pair3_idx1, self.swap_pair3_idx2, 64 + 50 * 2, 64+ 50 * 3),
            ]



#Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):


        timestep = self.elapsed_steps        
        # Keep all spawned bins in their original placement during the pre-swap window
        for bin_actor in getattr(self, "spawned_bins", []):
            lift_and_drop_objects_back_to_original(
                self,
                obj=bin_actor,
                start_step=0,
                end_step=32*2,
                cur_step=timestep,
            )

        for idx_a, idx_b, start_step, end_step in self.swap_schedule:
            
            if idx_a is None or idx_b is None:
                continue

            swap_flat_two_lane(
                self,
                cube_a=idx_a,
                cube_b=idx_b,
                start_step=start_step,
                end_step=end_step,
                cur_step=timestep,
                lane_offset=0.05,
                smooth=True,
                keep_upright=True,
                other_cube=[b for b in self.spawned_bins if b not in (idx_a, idx_b)],  # Keep all other bins in place to prevent collision during swap
            )

        for cube_actor, bin_actor in getattr(self, "cube_bin_pairs", []):
            if cube_actor is None or bin_actor is None:
                continue
            
            lift_and_drop_objectA_onto_objectB(
                self,
                obj_a=cube_actor,
                obj_b=bin_actor,
                start_step=32*2,
                end_step=self.swap_schedule[-1][3],
                cur_step=timestep,
            )

        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info
