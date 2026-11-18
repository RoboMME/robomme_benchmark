"""Insert Peg: scene construction, native scoring and RoboMME-Plus task configuration."""

import copy
from typing import Union

import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Panda
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS

# Robomme
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder

from robomme_plus.simulation.base import MemoryTask
from robomme_plus.simulation.difficulty import (
    is_newvalue_difficulty,
    normalize_robomme_difficulty,
    require_xhard4_only,
)
from robomme_plus.simulation.specification import EpisodeSpecError, SpecRecorder
from robomme_plus.common.errors import SceneGenerationError
from robomme_plus.common.logging_utils import logger
from robomme_plus.simulation.actors import build_peg
from robomme_plus.control.planning import (
    grasp_and_lift_peg_side,
    insert_peg,
    solve_hold_obj,
    solve_strong_reset,
)
from robomme_plus.scoring.predicates import is_A_insert_notB, is_A_pickup_notB, is_any_obj_pickup, reset_check, static_check
from robomme_plus.common.errors import SamplingConfigError


SCENE_CONFIG = {'parameters': {'peg_size': {'length_expression': '0.05 + (0.01 - 0.01) * rand()',
                             'radius_expression': '0.01 + (0.005 - 0.005) * rand()',
                             'note': 'The two rand() draws are multiplied by 0 and cancel out, but '
                                     'must be kept so the RNG stream does not shift (hard rule '
                                     'R8)'},
                'head_color': {'sampler': 'torch.rand', 'shape': [3]},
                'tail_color_rule': '1 - head',
                'target_peg': {'sampler': 'torch.randint(0, 3)',
                               'overridden_to': 0,
                               'note': 'Drawn, then immediately overwritten by 0; the number of '
                                       'draws is kept'},
                'obj_selection': {'sampler': 'torch.randint(0, 2)', 'mapping': [-1, 1]},
                'direction_selection': {'sampler': 'torch.randint(0, 2)', 'mapping': [-1, 1]},
                'recovery': 'This env has no inject_fail_grasp; it only accepts the entry recovery '
                            'mode, and the actual recovery action is null'},
 'positions': {'construction_peg': {'y_base': -0.15,
                                    'initial_yaw': 0,
                                    'note': 'Temporary pose used at construction; the real input '
                                            'is the pose re-sampled at each initialization'},
               'box': {'base_translation': [0, 0],
                       'jitter_span': 0.2,
                       'yaw_center_rad': 'np.pi / 2',
                       'yaw_half_span_deg': 20,
                       'z_factor': 4,
                       'inner_radius_factor': 1.7,
                       'outer_radius_factor': 4},
               'peg_sampling': {'x_span': 0.4,
                                'x_offset': -0.2,
                                'y_span': 0.6,
                                'y_offset': -0.3,
                                'min_distance_to_box_factor': 6,
                                'min_distance_between_pegs_factor': 1.5,
                                'max_attempts': 512}},
 'decision': {'peg_count': 3,
              'peg_offsets': [0.1, 0, -0.1],
              'near_target_distractor': None,
              'peg_yaw_range': {'half_span_deg': 45},
              'xhard4': {'peg_count': 4,
                         'peg_offsets': [0.1, 0, -0.1, -0.2],
                         'peg_yaw_range': {'half_span_deg': 180},
                         'peg_min_pair_gap_m': 0.03,
                         'peg_box_min_gap_m': 0.01,
                         'peg_x_max_m': 0.1}}}


def _rect_corners(center, yaw, half):
    """Oriented rectangle (center, yaw, half size) -> 4 corners (counter-clockwise), float64."""
    c = np.asarray(center, dtype=np.float64).reshape(2)
    u = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    v = np.array([-u[1], u[0]], dtype=np.float64)
    a = u * float(half[0])
    b = v * float(half[1])
    return np.stack([c + a + b, c - a + b, c - a - b, c + a - b])


def _point_segment_distance(p, a, b):
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = 0.0 if denom <= 0.0 else min(1.0, max(0.0, float(np.dot(p - a, ab)) / denom))
    return float(np.linalg.norm(p - (a + t * ab)))


def _rects_overlap(ca, cb):
    """Separating axis theorem: return True if the two rectangles share any point (boundary contact included)."""
    for corners in (ca, cb):
        for k in range(2):
            edge = corners[k + 1] - corners[k]
            axis = np.array([-edge[1], edge[0]], dtype=np.float64)
            pa = ca @ axis
            pb = cb @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def footprint_gap(rect_a, rect_b):
    """Exact minimum distance (meters) between two oriented rectangle footprints.

    ``rect = (center_xy, yaw, half_xy)``. Returns 0 when SAT finds an intersection (contact included); otherwise
    the rectangles are disjoint and the minimum distance is attained between a vertex and an edge of the other
    rectangle, so the result is the minimum over the 4x4x2 = 32 vertex-to-edge distances.
    Callers must use a strict inequality (``gap > g``): the gap is always 0 on intersection, so ``>= 0`` would impose no constraint.
    """
    ca = _rect_corners(*rect_a)
    cb = _rect_corners(*rect_b)
    if _rects_overlap(ca, cb):
        return 0.0
    best = float("inf")
    for src, dst in ((ca, cb), (cb, ca)):
        for p in src:
            for j in range(4):
                best = min(best, _point_segment_distance(p, dst[j], dst[(j + 1) % 4]))
    return best


def peg_footprint(root_xy, yaw, length, radius):
    """Table footprint of the whole peg: center root - (length/2)*u, half size (length, radius)."""
    root = np.asarray(root_xy, dtype=np.float64).reshape(2)
    u = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    return (root - 0.5 * float(length) * u, float(yaw), (float(length), float(radius)))


def box_footprint(box_xy, box_yaw, depth, outer_radius):
    """Table footprint of the box with hole: center box_xy, half size (depth, outer_radius), orientation box_yaw."""
    return (
        np.asarray(box_xy, dtype=np.float64).reshape(2),
        float(box_yaw),
        (float(depth), float(outer_radius)),
    )


class _TaskScene(MemoryTask):
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
    _clearance = 0.01


    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0,
        seed=0,
        Robomme_video_episode=None,
        Robomme_video_path=None,
        sampling_config=None,
        native_episode_spec=None,
        **kwargs,
    ):
        # Must come before any RNG call and super().__init__()
        self._sampling = copy.deepcopy(SCENE_CONFIG)
        self._spec = SpecRecorder(
            native_episode_spec,
            "InsertPeg",
            {"seed": seed},
            difficulty=kwargs.get("difficulty"),
        )
        self._native_init_index = -1
        self.use_demonstrationwrapper = False
        self.demonstration_record_traj = False
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
        self._hb_generator = torch.Generator()
        self._hb_generator.manual_seed(int(self.seed))

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
            else:
                self.difficulty = "hard"
        require_xhard4_only(self.difficulty, "InsertPeg")
        self._xhard_peg_yaw_reduction = True

        self.restore_flag = False
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        # ManiSkill performs a construction reset. Preserve the stream at the
        # first public reset so existing benchmark seeds keep their scenes.
        self._hb_episode_rng_state = self._hb_generator.get_state()

    def _load_scene(self, options: dict):
        self._hb_generator.manual_seed(int(self.seed))
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0)
        self.table_scene.build()

        length_tensor = torch.rand(1, generator=self._hb_generator)
        radius_tensor = torch.rand(1, generator=self._hb_generator)
        # self.length = (0.05 + (0.0 - 0.0) * length_tensor).item()
        # self.radius = (0.01 + (0.0 - 0.0) * radius_tensor).item()
        self.length = (0.05 + (0.01 - 0.01) * length_tensor).item()
        self.radius = (0.01 + (0.005 - 0.005) * radius_tensor).item()

        # Create 3 identical pegs with different x-axis coordinates
        self.pegs = []
        self.peg_heads = []
        self.peg_tails = []
        self._peg_initial_poses = []

        decision_cfg = self._sampling["decision"]
        decision_cfg = decision_cfg[self.difficulty]
        native_pos = self._sampling["positions"]
        offsets = list(decision_cfg["peg_offsets"])  # X-axis differences for the 3 pegs
        # Sample a single pair of colors so all pegs share the same appearance per seed.
        peg_head_color = self._spec.value(
            "objects.head_rgb", torch.rand(3, generator=self._hb_generator).tolist()
        )
        # Use complementary tail color so head/tail are contrasting.
        peg_tail_color = [1.0 - c for c in peg_head_color]

        for offset in offsets:
            peg_spawn_translation = np.array(
                [
                    self.length / 2,
                    native_pos["construction_peg"]["y_base"] - offset,
                    self.radius,
                ],
                dtype=np.float32,
            )

            # initial_yaw = (torch.rand(1, generator=self._hb_generator).item() * 2 * np.pi) - np.pi
            initial_yaw = native_pos["construction_peg"]["initial_yaw"]
            yaw_angles = torch.tensor([[0.0, 0.0, initial_yaw]], dtype=torch.float32)
            yaw_matrix = euler_angles_to_matrix(yaw_angles, convention="XYZ")
            yaw_quat = (
                matrix_to_quaternion(yaw_matrix)[0].detach().cpu().numpy().tolist()
            )

            peg_initial_pose = sapien.Pose(
                p=peg_spawn_translation.tolist(),
                q=yaw_quat,
            )

            peg, peg_head, peg_tail = build_peg(
                self,
                length=self.length,
                radius=self.radius,
                initial_pose=peg_initial_pose,
                name=f"peg_{len(self.pegs)}",
                head_color=peg_head_color,
                tail_color=peg_tail_color,
            )

            self.pegs.append(peg)
            self.peg_heads.append(peg_head)
            self.peg_tails.append(peg_tail)
            self._peg_initial_poses.append(peg_initial_pose)

        # Randomly select one peg from the 3 pegs
        target_cfg = self._sampling["parameters"]["target_peg"]
        # The result is overridden by 0, but the draw must still happen to keep the RNG stream aligned.
        random_peg_idx = self._spec.value(
            "objects.sampling_trace.random_peg_idx",
            int(
                torch.randint(
                    0, decision_cfg["peg_count"], (1,), generator=self._hb_generator
                ).item()
            ),
            decision_key=f"{self.difficulty}.peg_count"
            if is_newvalue_difficulty(self.difficulty)
            else None,
        )
        random_peg_idx = target_cfg["overridden_to"]
        self.peg = self.pegs[random_peg_idx]

        self.peg_head = self.peg_heads[random_peg_idx]
        self.peg_tail = self.peg_tails[random_peg_idx]
        self._peg_initial_pose = self._peg_initial_poses[random_peg_idx]

        box_cfg = native_pos["box"]
        self.box = self._build_insertion_fixture(
            inner_radius=self.radius * box_cfg["inner_radius_factor"],
            outer_radius=self.radius * box_cfg["outer_radius_factor"],
            depth=self.length,
            center=list(box_cfg["base_translation"]),
        )

        self.reset_in_proecess = False

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if hasattr(self, "_hb_episode_rng_state"):
            self._hb_generator.set_state(self._hb_episode_rng_state)
        self._native_init_index = getattr(self, "_native_init_index", -1) + 1
        with torch.device(self.device):
            self.end_steps = None
            self.table_scene.initialize(env_idx)
            # Reset highlight state at the start of each episode
            self._insert_highlight_start = None
            self._insert_highlight_active = False

            if not hasattr(self, "pegs"):
                return

            box_cfg = self._sampling["positions"]["box"]
            peg_sampling = self._sampling["positions"]["peg_sampling"]
            base_translation = tuple(box_cfg["base_translation"])
            x_jitter_2 = (
                torch.rand(1, generator=self._hb_generator).item() - 0.5
            ) * box_cfg["jitter_span"]
            y_jitter_2 = (
                torch.rand(1, generator=self._hb_generator).item() - 0.5
            ) * box_cfg["jitter_span"]
            # x_jitter_2=0
            # y_jitter_2=0
            x_jitter_2, y_jitter_2 = self._spec.value(
                f"initializations.{self._native_init_index}.box_jitter",
                [x_jitter_2, y_jitter_2],
            )
            box_translation = [
                base_translation[0] + x_jitter_2,
                base_translation[1] + y_jitter_2,
                self.radius * box_cfg["z_factor"],
            ]
            box_yaw = np.pi / 2 + (
                torch.rand(1, generator=self._hb_generator).item() * 2 - 1
            ) * np.radians(box_cfg["yaw_half_span_deg"])
            box_yaw = self._spec.value(
                f"initializations.{self._native_init_index}.box_yaw", box_yaw
            )
            box_angles = torch.tensor([[0.0, 0.0, box_yaw]], dtype=torch.float32)
            box_matrix = euler_angles_to_matrix(box_angles, convention="XYZ")
            box_quat = (
                matrix_to_quaternion(box_matrix)[0].detach().cpu().numpy().tolist()
            )
            self.box.set_pose(sapien.Pose(p=box_translation, q=box_quat))

            box_xy = np.array(box_translation[:2], dtype=np.float32)
            sampled_xy_positions = []
            max_sampling_attempts = peg_sampling["max_attempts"]

            xhard = is_newvalue_difficulty(self.difficulty)
            xhard_cfg = self._sampling["decision"][self.difficulty]
            yaw_half_span_deg = xhard_cfg["peg_yaw_range"]["half_span_deg"]
            uniform_pegs = []
            yaw_dk = f"{self.difficulty}.peg_yaw_range"

            # Initialize all 3 pegs with constrained random placements
            for i, peg in enumerate(uniform_pegs):
                candidate_xy = None
                for _ in range(max_sampling_attempts):
                    x_sample = (
                        torch.rand(1, generator=self._hb_generator).item()
                        * peg_sampling["x_span"]
                    ) + peg_sampling["x_offset"]
                    y_sample = (
                        torch.rand(1, generator=self._hb_generator).item()
                        * peg_sampling["y_span"]
                    ) + peg_sampling["y_offset"]
                    sampled_xy = np.array([x_sample, y_sample], dtype=np.float32)

                    if (
                        np.linalg.norm(sampled_xy - box_xy)
                        <= self.radius * peg_sampling["min_distance_to_box_factor"]
                    ):
                        continue

                    if any(
                        np.linalg.norm(sampled_xy - prev_xy)
                        <= self.length
                        * peg_sampling["min_distance_between_pegs_factor"]
                        for prev_xy in sampled_xy_positions
                    ):
                        continue

                    candidate_xy = sampled_xy
                    break

                if candidate_xy is None:
                    raise SceneGenerationError(
                        f"InsertPeg xhard: no valid position found for peg_{i} within {max_sampling_attempts} attempts"
                    )

                yaw_value = (
                    torch.rand(1, generator=self._hb_generator).item() * 2 - 1
                ) * np.radians(yaw_half_span_deg)

                candidate_xy_list, yaw_value = self._spec.value(
                    f"initializations.{self._native_init_index}.pegs.{i}",
                    [[float(candidate_xy[0]), float(candidate_xy[1])], yaw_value],
                    decision_key=yaw_dk,
                )
                candidate_xy = np.array(candidate_xy_list, dtype=np.float32)
                yaw_angles = torch.tensor([[0.0, 0.0, yaw_value]], dtype=torch.float32)
                yaw_matrix = euler_angles_to_matrix(yaw_angles, convention="XYZ")
                yaw_quat = (
                    matrix_to_quaternion(yaw_matrix)[0].detach().cpu().numpy().tolist()
                )

                pose = sapien.Pose(
                    p=[float(candidate_xy[0]), float(candidate_xy[1]), 0.0], q=yaw_quat
                )
                peg.set_pose(pose)
                sampled_xy_positions.append(candidate_xy)

            self._xhard_sample_pegs(box_xy, box_yaw, xhard_cfg)

            # Store initial poses for all pegs
            self.peg_init_poses = []
            for peg in self.pegs:
                pose = peg.pose
                pose_p = np.asarray(pose.p, dtype=np.float32).reshape(-1).copy()
                pose_q = np.asarray(pose.q, dtype=np.float32).reshape(-1).copy()
                self.peg_init_poses.append(sapien.Pose(p=pose_p, q=pose_q))

            # robomme-v2.7/robomme/robomme_env/PickPeg.py:243
            pose = self.peg.pose
            pose_p = np.asarray(pose.p, dtype=np.float32).reshape(-1).copy()
            pose_q = np.asarray(pose.q, dtype=np.float32).reshape(-1).copy()
            self.peg_init_pose = sapien.Pose(p=pose_p, q=pose_q)
            self.peg_init_pose = sapien.Pose(p=pose_p, q=pose_q)

            # Define task list, each task contains a dictionary with function, name, demonstration flag, and optional failure_func
            obj_sample = self._spec.value(
                f"initializations.{self._native_init_index}.obj_sample",
                int(torch.randint(0, 2, (1,), generator=self._hb_generator).item()),
            )
            dir_sample = self._spec.value(
                f"initializations.{self._native_init_index}.dir_sample",
                int(torch.randint(0, 2, (1,), generator=self._hb_generator).item()),
            )
            self.obj_flag = -1 if obj_sample == 0 else 1
            self.direction = -1 if dir_sample == 0 else 1

            self._peg_grasp_flipped = False
            self._peg_grasp_flip_log = []

            # if self.seed<30:
            #     self.obj_flag=-1
            #     self.direction=1
            # elif self.seed<60:
            #     self.obj_flag=-1
            #     self.direction=-1
            # elif self.seed<90:
            #     self.obj_flag=1
            #     self.direction=1
            # #elif self.seed<60:
            # else:
            #     self.obj_flag=1
            #     self.direction=-1

            # self.obj_flag=1
            # self.direction=-1

            qpos = np.array(
                [
                    0.0,
                    0,
                    0,
                    -np.pi * 4 / 8,
                    0,
                    np.pi * 2 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ],
                dtype=np.float32,
            )

            self.agent.reset(qpos)
            if self.obj_flag == -1:
                self.grasp_target = self.peg_head
                self.insert_target = self.peg_tail
            else:
                self.grasp_target = self.peg_tail
                self.insert_target = self.peg_head

            agent_x = self.agent.robot.pose.p.tolist()[0][0]
            head_x = float(self.peg_head.pose.p.tolist()[0][0])
            tail_x = float(self.peg_tail.pose.p.tolist()[0][0])
            logger.debug(f"agent_x: {agent_x}, head_x: {head_x}, tail_x: {tail_x}")
            base_xy = np.asarray(
                self.agent.robot.pose.p.tolist()[0][:2], dtype=np.float64
            )
            head_xy = np.asarray(
                self.peg_head.pose.p.tolist()[0][:2], dtype=np.float64
            )
            tail_xy = np.asarray(
                self.peg_tail.pose.p.tolist()[0][:2], dtype=np.float64
            )
            head_dist = float(np.linalg.norm(head_xy - base_xy))
            tail_dist = float(np.linalg.norm(tail_xy - base_xy))
            logger.debug(
                f"xhard near/far by euclidean XY: head_dist={head_dist:.5f}, tail_dist={tail_dist:.5f}"
            )
            near_link = self.peg_head if head_dist <= tail_dist else self.peg_tail

            self.grasp_target_distance = (
                "near" if self.grasp_target is near_link else "far"
            )
            logger.debug(f"grasp_target_distance: {self.grasp_target_distance}")

            self.insert_way = "left" if self.direction == -1 else "right"
            tasks = [
                {
                    "func": lambda: is_A_pickup_notB(
                        self, self.grasp_target, self.insert_target
                    ),
                    "name": f"Pick up the peg by grasping the {self.grasp_target_distance} end",
                    "subgoal_segment": f"Pick up the peg by grasping the {self.grasp_target_distance} end at <>",
                    "choice_label": "pick up the peg by grasping one end",
                    "demonstration": True,
                    "failure_reason": "wrong_peg_end",
                    "failure_func": lambda: is_A_pickup_notB(
                        self, self.insert_target, self.grasp_target
                    ),
                    "solve": lambda env, planner: grasp_and_lift_peg_side(
                        env, planner, env.grasp_target
                    ),
                    "segment": self.grasp_target,
                },
                {
                    "func": lambda: is_A_insert_notB(
                        self,
                        self.insert_target,
                        self.grasp_target,
                        self.box,
                        direction=self.direction,
                    ),
                    "name": f"Insert the peg from the {self.insert_way} side of the box",
                    "subgoal_segment": f"Insert the peg from the {self.insert_way} side of the box at <>",
                    "choice_label": f"insert the peg from the {self.insert_way} side",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: insert_peg(
                        env,
                        planner,
                        direction=self.direction,
                        obj=self.obj_flag,
                        insert_obj=self.insert_target,
                    ),
                    "segment": self.box,
                },
                {
                    "func": lambda: reset_check(self),
                    "name": "NO RECORD",
                    "subgoal_segment": f"NO RECORD",
                    "demonstration": True,
                    "failure_func": None,
                    "specialflag": "reset pegs",
                    "solve": lambda env, planner: [solve_strong_reset(env, planner)],
                },
                {
                    "func": lambda: static_check(
                        self, timestep=int(self.elapsed_steps), static_steps=100
                    ),
                    "name": "NO RECORD",
                    "subgoal_segment": "NO RECORD",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [
                        solve_hold_obj(env, planner, static_steps=100, close=False)
                    ],
                },
                {
                    "func": lambda: is_A_pickup_notB(
                        self, self.grasp_target, self.insert_target
                    ),
                    "name": f"Pick up the peg by grasping the {self.grasp_target_distance} end",
                    "subgoal_segment": f"Pick up the peg by grasping the {self.grasp_target_distance} end at <>",
                    "choice_label": "pick up the peg by grasping one end",
                    "demonstration": False,
                    "failure_reason": ["wrong_peg_end", "wrong_peg"],
                    "failure_func": lambda: [
                        is_A_pickup_notB(self, self.insert_target, self.grasp_target),
                        is_any_obj_pickup(
                            self,
                            [
                                head
                                for i, head in enumerate(self.peg_heads)
                                if self.pegs[i] is not self.peg
                            ]
                            + [
                                tail
                                for i, tail in enumerate(self.peg_tails)
                                if self.pegs[i] is not self.peg
                            ],
                        ),
                    ],
                    "solve": lambda env, planner: grasp_and_lift_peg_side(
                        env, planner, env.grasp_target
                    ),
                    "segment": self.grasp_target,
                },
                {
                    "func": lambda: is_A_insert_notB(
                        self,
                        self.insert_target,
                        self.grasp_target,
                        self.box,
                        direction=self.direction,
                        mark_end_flag=True,
                    ),
                    "name": f"Insert the peg from the {self.insert_way} side",
                    "subgoal_segment": f"Insert the peg from the {self.insert_way} side at <>",
                    "choice_label": f"insert the peg from the {self.insert_way} side",
                    "demonstration": False,
                    "failure_reason": ["wrong_insertion_end", "wrong_insertion_side", "wrong_peg"],
                    "failure_func": lambda: [
                        is_A_insert_notB(
                            self, self.grasp_target, self.insert_target, self.box
                        ),
                        is_A_insert_notB(
                            self,
                            self.insert_target,
                            self.grasp_target,
                            self.box,
                            direction=-self.direction,
                        ),
                        is_any_obj_pickup(
                            self,
                            [
                                head
                                for i, head in enumerate(self.peg_heads)
                                if self.pegs[i] is not self.peg
                            ]
                            + [
                                tail
                                for i, tail in enumerate(self.peg_tails)
                                if self.pegs[i] is not self.peg
                            ],
                        ),
                    ],
                    "solve": lambda env, planner: insert_peg(
                        env,
                        planner,
                        direction=self.direction,
                        obj=self.obj_flag,
                        insert_obj=self.insert_target,
                        cut_retreat=True,
                    ),
                    "segment": self.box,
                },
            ]

            # Store task list for RecordWrapper use
            self.task_list = tasks

    def _xhard_sample_pegs(self, box_xy, box_yaw, xhard_cfg):
        """V5 xhard: sample all 4 peg poses in one loop with the same rules.

        Each peg gets at most ``max_attempts`` (512) attempts; each attempt:
          1. Draw one rand each for x and y as the native sampling does (x upper bound tightened to ``peg_x_max_m``; y as native);
          2. Native root criteria (<= radius*6 from the box center, or <= length*1.5 from any placed peg root -> redraw);
          3. Only after passing, draw yaw (lazily, +-half_span_deg);
          4. Peg footprint vs box footprint ``footprint_gap <= peg_box_min_gap_m`` -> redraw;
          5. ``footprint_gap <= peg_min_pair_gap_m`` against any placed peg -> redraw (strict inequality).
        If the attempts are exhausted, ``SceneGenerationError`` is raised (task-level failure; a different seed may succeed).
        ``_spec.value`` is called only for the accepted attempt; the attempt count and measured minimum gaps are
        recorded via ``record``; when replaying a frozen spec both gaps are re-checked on the frozen values and
        ``EpisodeSpecError`` is raised on violation. The target is always peg_0 (merely the first one drawn).
        """
        peg_sampling = self._sampling["positions"]["peg_sampling"]
        max_attempts = peg_sampling["max_attempts"]
        x_lo = peg_sampling["x_offset"]
        x_span = float(xhard_cfg["peg_x_max_m"]) - x_lo
        if not x_span > 0:
            raise SamplingConfigError(
                f"InsertPeg xhard: peg_x_max_m must be greater than the x lower bound {x_lo}"
            )
        yaw_half = np.radians(xhard_cfg["peg_yaw_range"]["half_span_deg"])
        pair_gap = float(xhard_cfg["peg_min_pair_gap_m"])
        box_gap = float(xhard_cfg["peg_box_min_gap_m"])
        box_to_root = self.radius * peg_sampling["min_distance_to_box_factor"]
        root_to_root = self.length * peg_sampling["min_distance_between_pegs_factor"]
        box_cfg = self._sampling["positions"]["box"]
        box_fp = box_footprint(
            box_xy, box_yaw, self.length, self.radius * box_cfg["outer_radius_factor"]
        )
        prefix = f"initializations.{self._native_init_index}"

        placed_xy = []
        placed_fp = []
        attempts_log = []
        for i, peg in enumerate(self.pegs):
            accepted = None
            attempts = 0
            for attempts in range(1, max_attempts + 1):
                x_sample = (
                    torch.rand(1, generator=self._hb_generator).item() * x_span
                ) + x_lo
                y_sample = (
                    torch.rand(1, generator=self._hb_generator).item()
                    * peg_sampling["y_span"]
                ) + peg_sampling["y_offset"]
                sampled_xy = np.array([x_sample, y_sample], dtype=np.float32)
                if np.linalg.norm(sampled_xy - box_xy) <= box_to_root:
                    continue
                if any(
                    np.linalg.norm(sampled_xy - prev_xy) <= root_to_root
                    for prev_xy in placed_xy
                ):
                    continue
                yaw_try = (
                    torch.rand(1, generator=self._hb_generator).item() * 2 - 1
                ) * yaw_half
                fp = peg_footprint(sampled_xy, yaw_try, self.length, self.radius)
                if footprint_gap(fp, box_fp) <= box_gap:
                    continue
                if any(footprint_gap(fp, prev_fp) <= pair_gap for prev_fp in placed_fp):
                    continue
                accepted = (sampled_xy, yaw_try)
                break
            if accepted is None:
                raise SceneGenerationError(
                    f"InsertPeg xhard: no valid position (peg root + footprint gap) found for peg_{i} within {max_attempts} attempts"
                )
            attempts_log.append(attempts)
            candidate_xy, yaw_value = accepted
            candidate_xy_list, yaw_value = self._spec.value(
                f"{prefix}.pegs.{i}",
                [[float(candidate_xy[0]), float(candidate_xy[1])], yaw_value],
                decision_key=f"{self.difficulty}.peg_yaw_range",
            )
            candidate_xy = np.array(candidate_xy_list, dtype=np.float32)
            fp = peg_footprint(candidate_xy, yaw_value, self.length, self.radius)
            gap_box = footprint_gap(fp, box_fp)
            if not gap_box > box_gap:
                raise EpisodeSpecError(
                    f"InsertPeg xhard: footprint gap {gap_box:.4f} m between peg_{i} and the box is not greater than {box_gap} m (frozen spec violates the V5 rule)"
                )
            for j, prev_fp in enumerate(placed_fp):
                gap_pair = footprint_gap(fp, prev_fp)
                if not gap_pair > pair_gap:
                    raise EpisodeSpecError(
                        f"InsertPeg xhard: footprint gap {gap_pair:.4f} m between peg_{i} and peg_{j} is not greater than {pair_gap} m (frozen spec violates the V5 rule)"
                    )
            yaw_angles = torch.tensor([[0.0, 0.0, yaw_value]], dtype=torch.float32)
            yaw_quat = (
                matrix_to_quaternion(
                    euler_angles_to_matrix(yaw_angles, convention="XYZ")
                )[0]
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
            peg.set_pose(
                sapien.Pose(
                    p=[float(candidate_xy[0]), float(candidate_xy[1]), 0.0], q=yaw_quat
                )
            )
            placed_xy.append(candidate_xy)
            placed_fp.append(fp)

        self._spec.record(f"{prefix}.peg_attempts", attempts_log)
        self._spec.record(
            f"{prefix}.min_pair_gap_m",
            min(
                (
                    footprint_gap(placed_fp[a], placed_fp[b])
                    for a in range(len(placed_fp))
                    for b in range(a + 1, len(placed_fp))
                ),
                default=None,
            ),
        )
        self._spec.record(
            f"{prefix}.min_box_gap_m",
            min((footprint_gap(item, box_fp) for item in placed_fp), default=None),
        )


    # Robomme


from robomme_plus.registry import TASKS
from robomme_plus.tasks.imitation.utils.environment import ImitationMixin


@register_env(TASKS["InsertPeg"].gym_id)
class InsertPeg(ImitationMixin, _TaskScene):
    """Repeat the demonstrated peg, grasp end, and approach side on a turned stand with a moved hole."""

    PERCEPTUAL_VARIATIONS = True
    task_name = "InsertPeg"


from robomme_plus.tasks.imitation.utils.controllers import setup_insert

setup_insert(InsertPeg)
