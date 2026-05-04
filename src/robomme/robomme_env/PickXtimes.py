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
from .utils import subgoal_language
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


@register_env("PickXtimes")
class PickXtimes(BaseEnv):

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
    _COLOR_ORDER_SEED_OFFSET = 100_003
    _TARGET_SELECTION_SEED_OFFSET = 200_003
    _REPEAT_COUNT_SEED_OFFSET = 300_003
    
    config_hard = {
    'color': 3, 
    'number_min': 4,
    'number_max':5,
    }

    config_easy = {
        'color': 1, 
    'number_min': 1,
    'number_max':3
    }

    config_medium = {
        'color': 3, 
    'number_min': 1,
    'number_max':3
    }

    # Combine into a dictionary
    configs = {
        'hard': config_hard,
        'easy': config_easy,
        'medium': config_medium
    }

    @staticmethod
    def _splitmix64(x: int) -> int:
        """Splitmix64 bit mixing to break linear seed correlation."""
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return (x ^ (x >> 31)) & 0xFFFFFFFFFFFFFFFF

    def _make_generator(self, seed_offset: int = 0) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self._splitmix64(self.seed + seed_offset))
        return generator

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
        self.seed = seed
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
        """
        构建 PickXtimes 任务场景。

        这个函数主要完成以下几件事：
        1. 将“场景几何随机”和“任务语义随机”拆成独立 generator。
        2. 搭建桌面和按钮等基础场景物体。
        3. 按难度生成若干颜色的方块，并记录每种颜色对应的对象列表。
        4. 在不与现有物体重叠的前提下采样目标放置区域。
        5. 从所有已生成方块中随机选出一个真正的目标方块。
        6. 基于目标方块和重复次数，动态生成“抓取-放置-结束”任务序列。
        """

        # scene_generator 只负责几何与位置相关的随机采样；
        # 任务语义相关的随机性使用独立 generator，避免受位置采样消耗次数影响。
        scene_generator = self._make_generator()
        color_order_generator = self._make_generator(self._COLOR_ORDER_SEED_OFFSET)
        target_selection_generator = self._make_generator(
            self._TARGET_SELECTION_SEED_OFFSET
        )
        repeat_generator = self._make_generator(self._REPEAT_COUNT_SEED_OFFSET)

        # 重复次数单独走任务语义随机源，不受后续场景采样影响。
        self.num_repeats = torch.randint(
            self.configs[self.difficulty]['number_min'],
            self.configs[self.difficulty]['number_max'] + 1,
            (1,),
            generator=repeat_generator,
        ).item()
        logger.debug(
            f"Task will repeat {self.num_repeats} times (pickup-drop cycles)"
        )

        # 先搭建桌面场景，机器人和后续物体都依赖这个基础环境。
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # 在桌面上放置一个按钮。按钮既是场景元素，也是最终“停止任务”的交互目标。
        button_obb = build_button(
            self,
            center_xy=(-0.2, 0),
            scale=1.5,
            generator=scene_generator,
        )
        # avoid 记录当前已经占用的区域，后面采样方块和目标点时需要避开这些区域，
        # 以减少物体之间初始重叠或过近导致的无效场景。
        avoid = [button_obb]

        # all_cubes 保存本场景生成的全部方块对象，便于后续统一随机抽取目标方块。
        self.all_cubes = []

        # 分颜色维护方块对象和名字列表。
        # 这样后面既可以按颜色生成语言描述，也可以方便地做失败检查和目标归类。
        self.red_cubes = []
        self.red_cube_names = []
        self.blue_cubes = []
        self.blue_cube_names = []
        self.green_cubes = []
        self.green_cube_names = []

        # 当前实现里每种激活颜色只生成一个方块。
        # 保留这个变量可以让后续扩展到“每种颜色多个方块”时不用重写主逻辑。
        cubes_per_color = 1
        color_groups = [
            {"color": (1, 0, 0, 1), "name": "red", "list": self.red_cubes, "name_list": self.red_cube_names},
            {"color": (0, 0, 1, 1), "name": "blue", "list": self.blue_cubes, "name_list": self.blue_cube_names},
            {"color": (0, 1, 0, 1), "name": "green", "list": self.green_cubes, "name_list": self.green_cube_names}
        ]

        # 随机打乱颜色组顺序。
        # 这样即使难度只允许部分颜色出现，也不会总是优先生成固定颜色。
        shuffle_indices = torch.randperm(
            len(color_groups), generator=color_order_generator
        ).tolist()
        color_groups = [color_groups[i] for i in shuffle_indices]

        # 根据难度控制本局实际启用多少种颜色。
        # easy / medium / hard 会通过 configs[self.difficulty]['color'] 决定参与生成的颜色数量。
        for group_idx, group in enumerate(color_groups):
            if group_idx < self.configs[self.difficulty]['color']:
                # 在当前颜色组内生成若干个方块。
                # 每次采样都会避开按钮和此前已生成的物体。
                for cube_idx in range(cubes_per_color):
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
                            name_prefix=f"cube_{group['name']}_{cube_idx}",
                            generator=scene_generator,
                            # Halton 索引：seed → 一一映射的 (x, y)；
                            # group_idx 区分同一 episode 内的多个 cube。
                            halton_index=int(self.seed) * 3 + int(group_idx),
                            halton_bases=(2, 3),
                        )
                    except RuntimeError as e:
                        # 如果当前颜色的某个方块采样失败，就停止继续生成这一颜色组，
                        # 但保留前面已经成功生成的物体，避免整局场景直接构建失败。
                        logger.debug(f"Failed to generate {group['name']} cube {cube_idx}: {e}")
                        break

                    # 同时把方块登记到总列表和颜色专属列表中，
                    # 便于后续统一抽取目标，或按颜色做逻辑判断。
                    self.all_cubes.append(cube)
                    group["list"].append(cube)
                    cube_name = f"cube_{group['name']}_{cube_idx}"
                    group["name_list"].append(cube_name)
                    # 额外挂到实例属性上，便于外部通过固定名字访问该方块。
                    setattr(self, cube_name, cube)
                    # 新生成的方块也要加入避障列表，防止后续采样与其重叠。
                    avoid.append(cube)

            logger.debug(f"Generated {len(group['list'])} {group['name']} cubes")

        logger.debug(f"Generated {len(self.all_cubes)} cubes total (red: {len(self.red_cubes)}, blue: {len(self.blue_cubes)}, green: {len(self.green_cubes)})")

        try:
            # 在桌面区域内采样一个放置目标点（target）。
            # 这里同样显式传入 avoid，保证目标位置不会与按钮或已生成方块冲突。
            target = spawn_random_target(
                self,
                avoid=avoid,
                include_existing=False,
                include_goal=False,
                region_center=[-0.1, 0],
                region_half_size=0.2,
                radius=self.cube_half_size*2,
                thickness=0.005,
                min_gap=self.cube_half_size*2,
                name_prefix=f"target",
                generator=scene_generator,
                # Halton 索引：seed → target XY 一一映射；与 cube 走互不重叠的基底流。
                halton_index=int(self.seed),
                halton_bases=(5, 7),
            )
        except RuntimeError as e:
            logger.debug(f"Target sampling failed: {e}")

        # 这个任务只有一个目标点，因此直接挂到 self.target 上。
        setattr(self, f"target", target)
        # 目标点也加入避障列表，虽然当前后面不再继续采样场景物体，
        # 但这样能保持 avoid 的语义完整。
        avoid.append(target)


        # 从所有已生成方块中随机抽一个，作为真正需要重复抓取和放置的目标方块。
        if len(self.all_cubes) > 0:
            target_cube_idx = torch.randint(
                0,
                len(self.all_cubes),
                (1,),
                generator=target_selection_generator,
            ).item()
            self.target_cube = self.all_cubes[target_cube_idx]



            # 根据目标方块所属颜色列表，确定任务中使用的颜色名字，
            # 使语言描述和任务逻辑与实际目标方块保持一致。
            if self.target_cube in self.red_cubes:
                self.target_color_name = "red"
            elif self.target_cube in self.blue_cubes:
                self.target_color_name = "blue"
            elif self.target_cube in self.green_cubes:
                self.target_color_name = "green"


            logger.debug(f"Target cube selected: {self.target_color_name} cube (index {target_cube_idx} in all_cubes)")
        else:
            # 理论上如果一个方块都没生成出来，就没有可执行的抓取目标。
            # 这里保留空值，方便后续逻辑识别异常场景。
            self.target_cube = None
            self.target_color_name = None
            logger.debug("No cubes generated, no target cube selected")

        # 维护“非目标方块”列表。
        # 在任务执行时，如果抓起了错误的方块，failure_func 会用它来判定失败。
        self.non_target_cubes = [cube for cube in self.all_cubes if cube != self.target_cube]
        logger.debug(f"Non-target cubes: {len(self.non_target_cubes)}")

        # 按照 num_repeats 动态构造子任务序列：
        # 每一轮都包含“抓起目标方块”与“放到目标点上”两个阶段；
        # 最后追加“按按钮结束任务”作为收尾动作。
        tasks = []
        for i in range(self.num_repeats):
            # 第 1 类子任务：抓起目标颜色的方块。
            tasks.append({
                "func": (lambda i=i: is_obj_pickup(self, obj=self.target_cube)),
                "name": subgoal_language.get_subgoal_with_index(i, "pick up the {color} cube for the {idx} time", color=self.target_color_name),
                 "subgoal_segment": subgoal_language.get_subgoal_with_index(i, "pick up the {color} cube at <> for the {idx} time", color=self.target_color_name),
                "choice_label": "pick up the cube",
                "demonstration": False,
                # 失败条件包括：
                # 1. 抓起了任意非目标方块；
                # 2. 在不该结束时提前按下按钮。
                "failure_func": lambda: [is_any_obj_pickup(self, self.non_target_cubes),is_button_pressed(self, obj=self.button)],
                "solve": lambda env, planner: solve_pickup(env,planner,self.target_cube),
                "segment":self.target_cube
            })
            # 第 2 类子任务：将当前抓着的目标方块放到 target 上。
            tasks.append({
                "func": (lambda: is_obj_dropped_onto(self,obj=self.target_cube,target=self.target)),
                "name": f"place the {self.target_color_name} cube onto the target",
                "subgoal_segment": f"place the {self.target_color_name} cube onto the target at <>",
                "choice_label": "place the cube onto the target",
                "demonstration": False,
                "failure_func": lambda: [is_any_obj_pickup(self, self.non_target_cubes),is_button_pressed(self, obj=self.button)],
                "solve": lambda env, planner: solve_putonto_whenhold(env, planner, target=self.target),
                "segment":self.target
            })

        # 所有重复轮次结束后，要求机器人按按钮来显式结束任务。
        tasks.append( {
                "func": lambda:is_button_pressed(self, obj=self.button),
                "name": "press the button to stop",
                "subgoal_segment": "press the button at <> to stop",
                "choice_label": "press the button to stop",
                "demonstration": False,
                "failure_func":lambda:is_any_obj_pickup(self, self.all_cubes),
                "solve": lambda env, planner: solve_button(env, planner, obj=self.button),
                "segment":self.cap_link 
            })


        # 保存完整任务列表，供评测逻辑和 RecordWrapper 等上层模块使用。
        self.task_list = tasks

        # 记录和“抓取”有关的任务索引，供失败恢复逻辑使用。
        self.recovery_pickup_indices, self.recovery_pickup_tasks = task4recovery(self.task_list)
        if self.robomme_failure_recovery:
            # 只有在开启失败恢复模式时，才注入一次可控的失败抓取任务，
            # 用于测试或训练恢复能力。
            self.fail_grasp_task_index = inject_fail_grasp(
                self.task_list,
                generator=scene_generator,
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
            logger.debug(self.agent.robot.qpos)

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
        all_tasks_completed, current_task_name, task_failed,self.current_task_specialflag= sequential_task_check(self, self.task_list,allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep)

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


#Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):


        
        # highlight_obj(self,self.target_cube, start_step=0, end_step=30, cur_step=timestep)
        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info
