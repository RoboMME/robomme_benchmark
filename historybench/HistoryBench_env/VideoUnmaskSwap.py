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


@register_env("VideoUnmaskSwap", max_episode_steps=2000)
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

        # 使用 seed 随机确定需要重复的次数 (1-5)
        generator = torch.Generator()
        generator.manual_seed(HistoryBench_seed)
        self.swap_times = torch.randint(self.configs[self.difficulty]['swap_min'], self.configs[self.difficulty]['swap_max']+1, (1,), generator=generator).item()
        print(f"Task will swap {self.swap_times} times")


        self.pick_times = torch.randint(self.configs[self.difficulty]['pick_min'], self.configs[self.difficulty]['pick_max']+1, (1,), generator=generator).item()
        print(f"Task will pick {self.pick_times} times")



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
        region4=[[-0.05,-0.1],[-0.05,0.1],[0.1,0.1],[0.1,-0.1]]
        region3_tri=[[-0.05,-0.1],[-0.05,0.1],[0.1,0]]
        region3_line=[[0,-0.15],[0,0.15],[0,0]]

        # 使用 generator 随机选择 region3_tri 或 region3_line
        region3_choice = torch.randint(0, 2, (1,), generator=generator).item()
        region3 = region3_tri if region3_choice == 0 else region3_line

        if self.configs[self.difficulty]['bin']==4:
            region=region4
        else:
             region=region3
        angle, region = rotate_points_random(region,(0,180),generator)
        
        for i in range(self.configs[self.difficulty]['bin']):
            try:
                bin_actor = spawn_random_bin(
                    self,
                    avoid=avoid,  # 使用当前避让清单，包含所有已生成的对象
                    region_center=region[i],
                    region_half_size=0.07,
                    min_gap=self.cube_half_size*1,  # bins需要更大的间隙，增加到6倍避免碰撞
                    name_prefix=f"bin_{i}",
                    max_trials=256,
                    generator=generator
                )
            except RuntimeError as e:
                break

            self.spawned_bins.append(bin_actor)
            # 将bin赋值给self.bin_0, self.bin_1等属性
            setattr(self, f"bin_{i}", bin_actor)
            # 将新生成的bin加入避让清单
            avoid.append(bin_actor)


        # 在每个bin下方生成3个动态cube（使用固定位置，颜色为红、绿、蓝）
        spawned_dynamic_cubes = []
        self.cube_bin_pairs = []
        self.bin_to_cube = {}
        self.bin_to_color = {}
        self.spawned_dynamic_cubes = spawned_dynamic_cubes
        cube_colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]  # 红、绿、蓝
        color_names = ["red", "green", "blue"]

        # 使用 seed 随机打乱颜色顺序

        shuffle_indices = torch.randperm(len(cube_colors), generator=generator).tolist()
        cube_colors = [cube_colors[i] for i in shuffle_indices]
        color_names = [color_names[i] for i in shuffle_indices]

        # 存储 color_names 以便 RecordWrapper 访问
        self.color_names = color_names

        # 从所有bin中随机选择3个bin生成cube
        num_bins_to_select = min(3, len(self.spawned_bins))
        selected_bin_indices = torch.randperm(3, generator=generator)[:num_bins_to_select].tolist()
        selected_bins = [self.spawned_bins[idx] for idx in selected_bin_indices]
        self.selected_bin_indices = selected_bin_indices
        self.selected_bins = selected_bins  # 保存选中的bin，与color_names顺序对应

        for i, (bin_idx, bin_actor) in enumerate(zip(selected_bin_indices, selected_bins)):
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
            setattr(self, f"target_cube_for_bin_{bin_idx}", cube_actor)
            self.cube_bin_pairs.append((cube_actor, bin_actor))
            self.bin_to_cube[bin_idx] = cube_actor
            self.bin_to_color[bin_idx] = color_names[i]

            # 将新生成的cube加入避让清单
            avoid.append(cube_actor)

        self.cube_bins = selected_bins
        self.cube_bin_indices = selected_bin_indices
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
            self.target_bin_index = selected_bin_indices[target_choice]
            self.target_cube_color = color_names[target_choice]
            self.target_cube_name = (
                getattr(target_cube_actor, "name", None)
                or f"target_cube_{self.target_cube_color}"
            )
            self.target_label = self.target_cube_color or self.target_cube_name or "target"

            for idx_i, (cube_actor, bin_actor) in enumerate(self.cube_bin_pairs):
                if idx_i == target_choice:
                    continue
                self.other_cube_bins.append(bin_actor)
                self.other_cube_bin_indices.append(selected_bin_indices[idx_i])
                self.other_cubes.append(cube_actor)
        else:
            self.target_cube = None
            self.target_bin = None
            self.target_bin_index = None
            self.target_cube_color = None
            self.target_cube_name = None
            self.target_label = "target"

       # 随机选择2个不重复的 bin 作为 target_bin_1 和 target_bin_2
        # target_indices 是对 selected_bin_indices 的索引（0, 1, 2）
        target_indices = torch.randperm(len(selected_bin_indices), generator=generator)[:2]
        # 使用 selected_bins 来获取正确的 bin（与 color_names 顺序对应）
        self.target_bin_1=self.selected_bins[target_indices[0]]
        self.target_bin_2=self.selected_bins[target_indices[1]]
        # 记录这两个 bin 对应的 cube 颜色，使用 color_names 直接索引
        self.target_bin_1_cube_color = color_names[target_indices[0].item()]
        self.target_bin_2_cube_color = color_names[target_indices[1].item()]
        # swap_indices 必须包含 target_indices，再从剩余索引中选择1个
        remaining_indices = [i for i in range(len(self.spawned_bins)) if i not in target_indices.tolist()]
        if remaining_indices:
            third_idx = remaining_indices[torch.randint(0, len(remaining_indices), (1,), generator=generator).item()]
            swap_indices = torch.cat([target_indices, torch.tensor([third_idx])])
        else:
            swap_indices = target_indices
        self.swap_pair1_idx1=self.spawned_bins[swap_indices[0]]
        self.swap_pair2_idx1=self.spawned_bins[swap_indices[1]]
        self.swap_pair3_idx1=self.spawned_bins[swap_indices[2]]
        self.swap_pair1_idx2=None
        self.swap_pair2_idx2=None
        self.swap_pair3_idx2=None
        self._refresh_swap_schedule()

        tasks = [
             {
                        "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=self.swap_schedule[-1][3]),
                        "name": "static",
                        "subgoal_segment": "static",
                        "demonstration": True,
                        "failure_func": None,
                        "solve": lambda env, planner: solve_hold_obj(env, planner, static_steps=self.swap_schedule[-1][3]),
                        },

            
            {
                "func": (lambda: is_bin_pickup(self, obj=self.selected_bins[0])),
                "name": f"pick up the container that hides the {self.color_names[0]} cube",
                "subgoal_segment":f"pick up the container at <> that hides the {self.color_names[0]} cube",
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
                    "demonstration": False,
                    "failure_func": lambda:is_any_bin_pickup(self,[bin for bin in self.spawned_bins if bin != self.selected_bins[0]]),
                    "solve": lambda env, planner: solve_putdown_whenhold(env, planner,),

                })
            tasks.append(
                {
                    "func": (lambda: is_bin_pickup(self, obj=self.selected_bins[1])),
                    "name": f"pick up the container that hides the {self.color_names[1]} cube",
                    "subgoal_segment":f"pick up the container at <> that hides the {self.color_names[1]} cube",
                    "demonstration": False,
                    "failure_func": lambda: is_any_bin_pickup(self,[bin for bin in self.spawned_bins if bin != self.selected_bins[1]]),
                    "solve": lambda env, planner: solve_pickup_bin(env, planner, obj=self.selected_bins[1]),
                    "segment":self.selected_bins[1],
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

    def _select_swap_pair_from_positions(self, positions, generator):
        """Select one swap pair given current planned positions."""
        num_bins = len(positions)
        if num_bins < 2:
            return None

        candidate_map = self._compute_dynamic_swap_candidates(positions)
        valid_indices = [idx for idx, cands in candidate_map.items() if cands]
        if not valid_indices:
            return None

        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(self.HistoryBench_seed))
            self._swap_rng = generator

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


    def _refresh_swap_schedule(self):
        if self.swap_times==1:
                    self.swap_schedule = [
                            (self.swap_pair1_idx1, self.swap_pair1_idx2, 64, 64 + 50),
                        ]#最后的swap顺序
        elif self.swap_times==2:
            self.swap_schedule = [
                    (self.swap_pair1_idx1, self.swap_pair1_idx2, 64, 64 + 50),
                    (self.swap_pair2_idx1, self.swap_pair2_idx2, 64 + 50, 64 + 50 * 2),
                ]#最后的swap顺序
        elif self.swap_times==3:
            self.swap_schedule = [
                    (self.swap_pair1_idx1, self.swap_pair1_idx2, 64, 64 + 50),
                    (self.swap_pair2_idx1, self.swap_pair2_idx2, 64 + 50, 64 + 50 * 2),
                    (self.swap_pair3_idx1, self.swap_pair3_idx2, 64 + 50 * 2, 64+ 50 * 3),
            ]



#HistoryBench
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

        for i in range(len(self.swap_schedule)):
            start = self.swap_schedule[i][2]
            end = self.swap_schedule[i][3]
            if timestep in range (start,end):
                # 根据索引选择对应的swap pair
                pair_idx1 = getattr(self, f'swap_pair{i+1}_idx1')
                pair_idx2 = getattr(self, f'swap_pair{i+1}_idx2')

                if pair_idx2 is None and pair_idx1 is not None:
                    reference_pos = self._get_actor_position(pair_idx1)
                    closest_actor = None
                    closest_dist = float("inf")
                    for candidate in self.spawned_bins:
                        if candidate is None or candidate is pair_idx1:
                            continue
                        candidate_pos = self._get_actor_position(candidate)
                        dist = np.linalg.norm(reference_pos[:2] - candidate_pos[:2])
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_actor = candidate
                    if closest_actor is not None:
                        setattr(self, f'swap_pair{i+1}_idx2', closest_actor)
                        self._refresh_swap_schedule()




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
                lane_offset=0.07,
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
