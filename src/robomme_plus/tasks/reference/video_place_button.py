"""Video Place Button: scene construction, native scoring and RoboMME-Plus task configuration."""

import copy
from typing import Dict, Union

import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Panda
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder

import robomme_plus.simulation.difficulty as difficulty_utils
import robomme_plus.control.reset as reset_panda
from robomme_plus.simulation.base import MemoryTask
from robomme_plus.simulation.difficulty import normalize_robomme_difficulty
from robomme_plus.simulation.specification import SpecRecorder
from robomme_plus.common.errors import SceneGenerationError
from robomme_plus.common.logging_utils import logger

# Robomme
from robomme_plus.simulation.actors import (
    build_button,
    spawn_random_cube,
)
from robomme_plus.control.planning import (
    solve_button,
    solve_hold_obj,
    solve_pickup,
    solve_reset,
    solve_strong_reset,
)
from robomme_plus.scoring.predicates import (
    is_any_obj_pickup,
    is_button_pressed,
    is_obj_dropped_onto,
    is_obj_dropped_onto_any,
    is_obj_pickup,
    reset_check,
    sequential_task_check,
    static_check,
)
from robomme_plus.simulation.choreography import swap_flat_two_lane
from robomme_plus.simulation.geometry import cube_obb2d_exact
from robomme_plus.tasks.reference.utils.home_sites import (
    build_home_sites,
    home_pose_record,
    validate_demo_plan,
    validate_place_sequence,
)
from robomme_plus.tasks.reference.utils.place_swaps import (
    clocked_put as solve_putonto_whenhold,
)


SCENE_CONFIG = {'parameters': {'color_pool': [{'rgba': [1, 0, 0, 1], 'name': 'red'},
                               {'rgba': [0, 0, 1, 1], 'name': 'blue'},
                               {'rgba': [0, 1, 0, 1], 'name': 'green'}],
                'color_order': {'sampler': 'torch.randperm(3)'},
                'target_selection': {'sampler': 'torch.randint(0, len(all_cubes))'},
                'task_mapping': 'before→target_0, after→target_1; demo '
                                'target_0→button→target_1→goal_site',
                'swap_duration_steps': 50,
                'goal_site_z_override': -0.05,
                'recovery': 'reuses the fail recover mode given at the entry point and the '
                            'original generator',
                'target_slots': 4,
                'color': {'xhard1': 3, 'xhard3': 3}},
 'positions': {'goal': {'region_center': [-0.1, 0],
                        'region_half_size': 0.1,
                        'radius_factor': 3,
                        'thickness': 0.005},
               'button': {'center_xy': [0.1, 0], 'scale': 1.5, 'randomize_range': [0.05, 0.3]},
               'cubes': {'region_center': [0, 0], 'region_half_size': 0.2, 'random_yaw': True},
               'targets': {'region_center': [0, 0],
                           'region_half_size': 0.2,
                           'radius_factor': 2,
                           'thickness': 0.005,
                           'min_gap_factor': 1}},
 'decision': {'demo_object_count': 1,
              'demo_return_policy': 'native_random_goal_site',
              'targets': {'xhard1': 4, 'xhard3': 5},
              'swap': {'xhard1': True, 'xhard3': True},
              'additional_place': {'xhard1': False, 'xhard3': False},
              'xhard1': {'demo_object_count': 1,
                         'demo_return_policy': 'return_to_origin',
                         'extra_place_before': 0,
                         'extra_place_after': 1},
              'xhard3': {'demo_object_count': 2,
                         'demo_return_policy': 'return_to_origin',
                         'extra_place_before': 1,
                         'extra_place_after': 0}}}


def _is_newvalue_difficulty(value):
    """Use the shared difficulty-family predicate when available; otherwise fall back to a local set (supports targeted tests before the VP copy is merged)."""
    predicate = getattr(difficulty_utils, "is_newvalue_difficulty", None)
    if predicate is not None:
        return bool(predicate(value))
    return isinstance(value, str) and value.strip().lower() in {
        "xhard1",
        "xhard2",
        "xhard3",
        "xhard4",
    }


def vpb_target_placement_count(tier_cfg):
    """V6 gradient counts only placements onto a target; the final return to home is not counted."""
    return (
        2 * int(tier_cfg["demo_object_count"])
        + int(tier_cfg.get("extra_place_before", 0))
        + int(tier_cfg.get("extra_place_after", 0))
    )


def _extra_place_owners(demo_count, before_count, after_count, generator):
    sides = ["before"] * int(before_count) + ["after"] * int(after_count)
    if not sides:
        return sides, []
    if demo_count == 1:
        return sides, [0] * len(sides)
    if demo_count != 2 or len(sides) > 2:
        raise SceneGenerationError(
            "VPB V6 extra placement supports only 1-2 demo cubes and at most 2 extra segments"
        )
    if len(sides) == 1:
        owner_ids = [
            int(torch.randint(0, demo_count, (1,), generator=generator).item())
        ]
    elif len(sides) == 2:
        owner_ids = torch.randperm(demo_count, generator=generator).tolist()
    else:
        owner_ids = []
    return sides, [int(owner) for owner in owner_ids]


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


    # xhard3/4 need 5 targets: with 4, extra-placement candidates always equal the after-target set (a no-op placement).

    # Combine into a dictionary

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
        # Must run before any random-number call and before super().__init__()
        self._sampling = copy.deepcopy(SCENE_CONFIG)
        self._spec = SpecRecorder(
            native_episode_spec,
            "VideoPlaceButton",
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
        self._episode_rng = torch.Generator()
        self._episode_rng.manual_seed(seed)

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

        # Use seed to randomly determine number of repetitions (1-5)
        generator = torch.Generator()
        generator.manual_seed(seed)

        self.onto_goalsite = False
        self.start_step = 99999
        self.end_step = 99999
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options: dict):
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        try:
            self.table_scene = TableSceneBuilder(
                self, robot_init_qpos_noise=self.robot_init_qpos_noise
            )
            self.table_scene.build()
            try:
                self.goal_site = self._spawn_target(
                    avoid=None,  # Use current avoidance list, containing all spawned cubes
                    include_existing=False,  # Manually maintain list
                    include_goal=False,  # Manually maintain list
                    region_center=list(
                        self._sampling["positions"]["goal"]["region_center"]
                    ),
                    region_half_size=self._sampling["positions"]["goal"][
                        "region_half_size"
                    ],
                    radius=self.cube_half_size
                    * self._sampling["positions"]["goal"][
                        "radius_factor"
                    ],  # Use radius instead of half_size
                    thickness=0.005,  # target thickness
                    min_gap=self.cube_half_size * 1,  # Gap requirement same as cube
                    name_prefix=f"goal_site",
                    recorder=self._spec,
                    spec_path="layout.goal_xy",
                    generator=generator,
                )
            except RuntimeError as exc:
                raise SceneGenerationError("goal_site sampling failed") from exc
            avoid = [self.goal_site]

            button_cfg = self._sampling["positions"]["button"]
            cubes_cfg = self._sampling["positions"]["cubes"]
            targets_cfg = self._sampling["positions"]["targets"]
            decision_cfg = self._sampling["decision"]
            button_obb = build_button(
                self,
                center_xy=tuple(button_cfg["center_xy"]),
                scale=button_cfg["scale"],
                generator=generator,
                randomize_range=tuple(button_cfg["randomize_range"]),
                recorder=self._spec,
                spec_path="layout.button_xy",
            )
            avoid.append(button_obb)

            self.all_cubes = []  # Save all cube objects

            # Initialize storage for each color group
            self.red_cubes = []
            self.red_cube_names = []
            self.blue_cubes = []
            self.blue_cube_names = []
            self.green_cubes = []
            self.green_cube_names = []

            cubes_per_color = 1
            color_groups = [
                {
                    "color": (1, 0, 0, 1),
                    "name": "red",
                    "list": self.red_cubes,
                    "name_list": self.red_cube_names,
                },
                {
                    "color": (0, 0, 1, 1),
                    "name": "blue",
                    "list": self.blue_cubes,
                    "name_list": self.blue_cube_names,
                },
                {
                    "color": (0, 1, 0, 1),
                    "name": "green",
                    "list": self.green_cubes,
                    "name_list": self.green_cube_names,
                },
            ]
            self._spec.identity.setdefault(
                "difficulty", getattr(self, "difficulty", None)
            )
            shuffle_indices = self._spec.value(
                "objects.color_order",
                torch.randperm(len(color_groups), generator=generator).tolist(),
            )
            color_groups = [color_groups[i] for i in shuffle_indices]

            self.target_color_name = color_groups[0]["name"]
            logger.debug(f"Target color selected: {self.target_color_name}")

            # Generate cubes for each color group
            for idx, group in enumerate(color_groups):
                if idx < self._sampling["parameters"]["color"][self.difficulty]:
                    for cube_idx in range(cubes_per_color):
                        try:
                            cube = spawn_random_cube(
                                self,
                                color=group["color"],
                                avoid=avoid,
                                include_existing=False,
                                include_goal=False,
                                region_center=list(cubes_cfg["region_center"]),
                                region_half_size=cubes_cfg["region_half_size"],
                                half_size=self.cube_half_size,
                                min_gap=self.cube_half_size,
                                random_yaw=cubes_cfg["random_yaw"],
                                name_prefix=f"cube_{group['name']}_{cube_idx}",
                                generator=generator,
                                recorder=self._spec,
                                spec_path=f"layout.cubes.{group['name']}_{cube_idx}",
                            )
                        except RuntimeError as exc:
                            raise SceneGenerationError(
                                f"Failed to generate {group['name']} cube {cube_idx}: {exc}"
                            ) from exc

                        self.all_cubes.append(cube)
                        group["list"].append(cube)
                        cube_name = f"cube_{group['name']}_{cube_idx}"
                        group["name_list"].append(cube_name)
                        setattr(self, cube_name, cube)
                        avoid.append(
                            cube_obb2d_exact(cube.initial_pose, self.cube_half_size)
                        )

                logger.debug(f"Generated {len(group['list'])} {group['name']} cubes")

            logger.debug(
                f"Generated {len(self.all_cubes)} cubes total (red: {len(self.red_cubes)}, blue: {len(self.blue_cubes)}, green: {len(self.green_cubes)})"
            )

            self.targets = []
            target_slots = int(self._sampling["parameters"]["target_slots"])
            target_slots = max(
                target_slots, int(decision_cfg["targets"][self.difficulty])
            )
            for i in range(target_slots):
                if i < decision_cfg["targets"][self.difficulty]:
                    try:
                        target = self._spawn_target(
                            avoid=avoid,  # Use current avoidance list, containing all spawned cubes
                            include_existing=False,  # Manually maintain list
                            include_goal=False,  # Manually maintain list
                            region_center=list(targets_cfg["region_center"]),
                            region_half_size=targets_cfg["region_half_size"],
                            radius=self.cube_half_size
                            * targets_cfg[
                                "radius_factor"
                            ],  # Use radius instead of half_size
                            thickness=targets_cfg["thickness"],  # target thickness
                            min_gap=self.cube_half_size
                            * targets_cfg[
                                "min_gap_factor"
                            ],  # Gap requirement same as cube
                            name_prefix=f"target_{i}",
                            generator=generator,
                            recorder=self._spec,
                            spec_path=f"layout.targets.{i}",
                        )
                    except RuntimeError as exc:
                        raise SceneGenerationError(
                            f"Target {i + 1} sampling failed: {exc}"
                        ) from exc

                    self.targets.append(target)
                    setattr(self, f"target_{i}", target)
                    avoid.append(target)

            self._load_scene_xhard_tail(generator)
            return

        except SceneGenerationError:
            raise
        except Exception as exc:
            raise SceneGenerationError(
                f"Failed to load VideoPlaceButton scene for seed {self.seed}"
            ) from exc

    # ------------------------------------------------------------------
    # xhard: demo two cubes, each put back at its original position; everything else unchanged
    # ------------------------------------------------------------------
    def _load_scene_xhard_tail(self, generator):
        """xhard only: take over the second half of _load_scene starting at target-cube selection.

        Demonstration template (looped per object; the original difficulties' inline sequence stays verbatim
        in _load_scene)::

            for each demo cube k: pick(k) → drop(before_targets[k]) ← before the button
            press button
            for each demo cube k: pick(k) → drop(after_targets[k])  ← after the button
            for each demo cube k: pick(k) → drop(home_k)             ← back to original position
            static20 → static60(swap) → NO RECORD → pick(answer cube) → drop(target_target)

        Answer for "before / after" (task_mapping generalized to 2 objects; not settled in the plan, may be
        overturned by the user): draw task_flag by the original rule to pick before/after, then draw anew which
        demo cube is asked about; before/after refer to that cube's randomly
        assigned target at the corresponding time relative to the button.

        Random stream (N5, shifts only xhard itself): randperm[:count] replaces the original randint target
        selection -> swap randperm (unchanged) -> task_flag (unchanged) -> new answer_demo_index; landing-site
        actors draw no random numbers and are placed after all spawns.
        """
        decision_cfg = self._sampling["decision"]
        tier_key = "xhard" if self.difficulty == "xhard" else self.difficulty
        xhard_cfg = decision_cfg[tier_key]
        demo_count, return_policy = validate_demo_plan(
            xhard_cfg["demo_object_count"],
            xhard_cfg["demo_return_policy"],
            self.difficulty,
            len(self.all_cubes),
        )
        is_v6_tier = _is_newvalue_difficulty(self.difficulty)
        pass
        extra_before = int(xhard_cfg.get("extra_place_before", 0)) if is_v6_tier else 0
        extra_after = int(xhard_cfg.get("extra_place_after", 0)) if is_v6_tier else 0
        target_placements = (
            vpb_target_placement_count(xhard_cfg) if is_v6_tier else 2 * demo_count
        )
        expected_placements = {"xhard1": 3, "xhard2": 4, "xhard3": 5, "xhard4": 6}
        if is_v6_tier and target_placements != expected_placements[self.difficulty]:
            raise SceneGenerationError(
                f"VideoPlaceButton {self.difficulty} needs target placement count "
                f"{expected_placements[self.difficulty]}, got {target_placements} from the config"
            )
        self.target_placement_count = target_placements
        if len(self.targets) < 2 * demo_count:
            raise SceneGenerationError(
                f"VideoPlaceButton xhard needs targets ≥ 2×demo_object_count={2 * demo_count}, got {len(self.targets)}"
            )

        demo_ids = self._spec.value(
            "objects.demo_ids",
            torch.randperm(len(self.all_cubes), generator=generator)[
                :demo_count
            ].tolist(),
            decision_key=f"{tier_key}.demo_object_count",
        )
        demo_ids = [int(i) for i in demo_ids]
        if len(demo_ids) != demo_count:
            raise SceneGenerationError(
                f"requested {demo_count} demo cubes, got {len(demo_ids)}"
            )
        self.demo_cubes = [self.all_cubes[i] for i in demo_ids]

        self.swap_target_a = None
        self.swap_target_b = None
        self.swap_target_other = []
        if decision_cfg["swap"][self.difficulty] == True and len(self.targets) >= 2:
            perm = self._spec.value(
                "objects.swap_pair_ids",
                torch.randperm(len(self.targets), generator=generator).tolist(),
                decision_key=f"swap.{self.difficulty}",
            )
            swap_idx_a, swap_idx_b = int(perm[0]), int(perm[1])
            self.swap_target_a = self.targets[swap_idx_a]
            self.swap_target_b = self.targets[swap_idx_b]
            self.swap_target_other = [
                target
                for idx, target in enumerate(self.targets)
                if idx not in (swap_idx_a, swap_idx_b)
            ]
        self.pre_flag = 0
        self.post_flag = 0
        self.task_flag = bool(
            self._spec.value(
                "objects.task_flag",
                torch.rand(1, generator=generator).item() < 0.5,
            )
        )
        answer_index = int(
            self._spec.value(
                "objects.answer_demo_index",
                torch.randint(0, demo_count, (1,), generator=generator).item(),
                decision_key=f"{tier_key}.demo_object_count",
            )
        )

        # Every ring is eligible for either temporal role, including rings
        # added by the RoboMME-Plus level. Keep this stream independent of native draws.
        role_ids = np.random.default_rng(int(self.seed) + 872432).permutation(len(self.targets))
        before_ids = role_ids[:2 * demo_count:2].tolist()
        after_ids = role_ids[1:2 * demo_count:2].tolist()
        self._spec.record("objects.demo_before_target_ids", before_ids)
        self._spec.record("objects.demo_after_target_ids", after_ids)
        self.demo_before_targets = [self.targets[i] for i in before_ids]
        self.demo_after_targets = [self.targets[i] for i in after_ids]
        self.target_cube = self.demo_cubes[answer_index]
        for color_name, group in (
            ("red", self.red_cubes),
            ("blue", self.blue_cubes),
            ("green", self.green_cubes),
        ):
            if self.target_cube in group:
                self.target_color_name = color_name
        self.target_target_language = "before" if self.task_flag == 1 else "after"
        self.non_target_cubes = [
            cube for cube in self.all_cubes if cube != self.target_cube
        ]

        # Occupancy follows the full task sequence; empty candidates fail generation (no silent fallback); random draw count is unchanged.
        n_targets = len(self.targets)
        occupancy = {i: None for i in range(n_targets)}
        location = {}
        for k in range(demo_count):
            occupancy[before_ids[k]] = k
            location[k] = before_ids[k]
        last_before_target = dict(enumerate(before_ids))
        self.demo_extra_place_before = []
        self.demo_extra_place_after = []

        def _apply_formal_after():
            for k in range(demo_count):
                holder = occupancy[after_ids[k]]
                if holder is not None and holder != k:
                    raise SceneGenerationError(
                        f"VideoPlaceButton {self.difficulty} after-button target {after_ids[k]} is still occupied by cube {holder} (two cubes on one target)"
                    )
                occupancy[location[k]] = None
                occupancy[after_ids[k]] = k
                location[k] = after_ids[k]

        def _answer_target_idx():
            if self.task_flag == 1:
                return int(last_before_target[answer_index])
            return int(after_ids[answer_index])

        answer_target_idx = None
        sides, owner_ids_drawn = _extra_place_owners(
            demo_count, extra_before, extra_after, generator
        )
        owner_ids = self._spec.value(
            "objects.extra_place_owner_ids",
            owner_ids_drawn,
            decision_key=f"{tier_key}.extra_place_before/after",
        )
        before_ids_drawn, after_ids_drawn = [], []
        after_phase_applied = False
        for side, owner in zip(sides, owner_ids):
            owner = int(owner)
            if not 0 <= owner < demo_count:
                raise SceneGenerationError(
                    f"VPB extra placement owner out of range: {owner}"
                )
            if side == "after" and not after_phase_applied:
                answer_target_idx = _answer_target_idx()
                _apply_formal_after()
                after_phase_applied = True
            if side == "before":
                excluded = set(after_ids) | {before_ids[answer_index]}
            else:
                excluded = {answer_target_idx}
            candidates = [
                idx
                for idx in range(n_targets)
                if occupancy[idx] is None and idx not in excluded
            ]
            if not candidates:
                raise SceneGenerationError(
                    f"VideoPlaceButton {self.difficulty} {side} extra placement has no free target that is neither the answer nor an after target"
                )
            target_idx_drawn = candidates[
                int(
                    torch.randint(
                        0, len(candidates), (1,), generator=generator
                    ).item()
                )
            ]
            target_idx = int(
                self._spec.value(
                    f"objects.extra_place_target_ids.{side}.{len(before_ids_drawn if side == 'before' else after_ids_drawn)}",
                    target_idx_drawn,
                    decision_key=f"{tier_key}.extra_place_{side}",
                )
            )
            if target_idx not in candidates:
                raise SceneGenerationError(
                    f"VPB replayed {side} extra target={target_idx} is occupied, is an after target, or is the answer target"
                )
            occupancy[location[owner]] = None
            occupancy[target_idx] = owner
            location[owner] = target_idx
            item = (self.demo_cubes[owner], self.targets[target_idx])
            if side == "before":
                self.demo_extra_place_before.append(item)
                before_ids_drawn.append(target_idx)
                last_before_target[owner] = target_idx
            else:
                self.demo_extra_place_after.append(item)
                after_ids_drawn.append(target_idx)
        if not after_phase_applied:
            answer_target_idx = _answer_target_idx()
            _apply_formal_after()
        self._spec.record("actions.target_placement_count", target_placements)
        place_sequence = [(k, before_ids[k]) for k in range(demo_count)]
        place_sequence += [
            (self.demo_cubes.index(c), self.targets.index(t))
            for c, t in self.demo_extra_place_before
        ]
        place_sequence += [(k, after_ids[k]) for k in range(demo_count)]
        place_sequence += [
            (self.demo_cubes.index(c), self.targets.index(t))
            for c, t in self.demo_extra_place_after
        ]
        validate_place_sequence(place_sequence, n_targets)
        self._spec.record(
            "actions.place_sequence", [[int(c), int(t)] for c, t in place_sequence]
        )
        self.target_target = self.targets[answer_target_idx]
        self.targets_not_true = [t for t in self.targets if t != self.target_target]
        self._spec.record("actions.target_target_id", int(answer_target_idx))

        # Must run after all spawns; builds sites from each cube's initial pose directly (not via spawn_random_target).
        self._build_xhard_final_sites(return_policy, generator)

        tasks = []
        for cube, target in zip(self.demo_cubes, self.demo_before_targets):
            tasks.extend(self._xhard_pick_place(cube, target))
        for cube, target in self.demo_extra_place_before:
            tasks.extend(self._xhard_pick_place(cube, target))
        tasks.append(
            {
                "func": (lambda: is_button_pressed(self, obj=self.button)),
                "name": "press the button",
                "subgoal_segment": f"press the button at <>",
                "choice_label": "press the button",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: solve_button(env, planner, self.button),
                "segment": self.cap_link,
            }
        )
        for cube, target in zip(self.demo_cubes, self.demo_after_targets):
            tasks.extend(self._xhard_pick_place(cube, target))
        for cube, target in self.demo_extra_place_after:
            tasks.extend(self._xhard_pick_place(cube, target))
        for cube, site, kind in self._xhard_final_sites:
            tasks.extend(self._xhard_pick_place(cube, site, home=kind))
        tasks.extend(self._xhard_closing_tasks())

        self.task_list = tasks
        self.fail_grasp_task_index = None

    def _build_xhard_final_sites(self, return_policy, generator):
        """Return every demonstrated cube to its initial pose."""
        self.xhard_home_sites, self._xhard_home_checks = build_home_sites(
            self, self.demo_cubes, generator
        )
        self._spec.record(
            "actions.return_pose_by_object_id",
            home_pose_record(self.demo_cubes, self.xhard_home_sites),
        )
        self.xhard_goal_drop_sites = []
        self._xhard_final_sites = [
            (c, h, "home") for c, h in zip(self.demo_cubes, self.xhard_home_sites)
        ]
        return

    def _xhard_pick_place(self, cube, target, home=False):
        """One pick + drop pair of a demo segment (the closure binds the current cube and site via default args).

        ``home``: False / "target" = drop on the target; True / "home" = put back at the original position;
        "goal" = not put back, dropped onto the table (goal_site region, text reuses the original
        difficulties' "drop the cube onto table").
        """
        if home == "goal":
            name = "drop the cube onto table"
            segment_text = "drop the cube onto table"
        elif home is True or home == "home":
            name = "put the cube back to its original position"
            # Home sites are hidden (_hidden_objects), so no coordinates can be filled in; V5 xhard specs keep the old ``at <>`` text.
            segment_text = (
                name
                if _is_newvalue_difficulty(self.difficulty)
                else "put the cube back to its original position at <>"
            )
        else:
            name = "drop the cube onto target"
            segment_text = "drop the cube onto target at <>"
        return [
            {
                "func": (lambda cube=cube: is_obj_pickup(self, obj=cube)),
                "name": f"pick up the cube",
                "subgoal_segment": f"pick up the cube at <>",
                "choice_label": "pick up the cube",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner, cube=cube: solve_pickup(
                    env, planner, obj=cube
                ),
                "segment": cube,
            },
            {
                "func": (
                    lambda cube=cube, target=target: is_obj_dropped_onto(
                        self, obj=cube, target=target
                    )
                ),
                "name": name,
                "subgoal_segment": segment_text,
                "choice_label": "drop onto",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner, target=target: solve_putonto_whenhold(
                    env, planner, target=target
                ),
                "segment": target,
            },
        ]

    def _xhard_closing_tasks(self):
        """Demo closing + execution segment, same structure as the original difficulties (static20 → swap → NO RECORD → pick the answer cube and drop it on the correct target)."""
        return [
            {
                "func": lambda: static_check(
                    self, timestep=int(self.elapsed_steps), static_steps=20
                ),
                "name": "static",
                "subgoal_segment": f"static",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: [
                    solve_reset(env, planner),
                    solve_hold_obj(env, planner, static_steps=20),
                ],
            },
            {
                "func": lambda: static_check(
                    self, timestep=int(self.elapsed_steps), static_steps=60
                ),
                "name": "static",
                "subgoal_segment": f"static",
                "specialflag": "swap",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: [
                    solve_hold_obj(env, planner, static_steps=60)
                ],
            },
            {
                "func": lambda: reset_check(self),
                "name": "NO RECORD",
                "subgoal_segment": f"NO RECORD",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: [solve_strong_reset(env, planner)],
            },
            {
                "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                "name": f"pick up the cube",
                "subgoal_segment": f"pick up the cube at <>",
                "choice_label": "pick up the cube",
                "demonstration": False,
                "failure_reason": "wrong_cube",
                "failure_func": lambda: is_any_obj_pickup(self, self.non_target_cubes),
                "solve": lambda env, planner: [
                    solve_pickup(env, planner, obj=self.target_cube)
                ],
                "segment": self.target_cube,
            },
            {
                "func": (
                    lambda: is_obj_dropped_onto(
                        self, obj=self.target_cube, target=self.target_target
                    )
                ),
                "name": "place the cube onto the correct target",
                "subgoal_segment": f"place the cube onto the correct target at <>",
                "choice_label": "drop onto",
                "demonstration": False,
                "failure_reason": "wrong_placement",
                "failure_func": (
                    lambda: is_obj_dropped_onto_any(
                        self, obj=self.target_cube, target=self.targets_not_true
                    )
                ),
                "solve": lambda env, planner: [
                    solve_putonto_whenhold(env, planner, target=self.target_target)
                ],
                "segment": self.target_target,
            },
        ]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._native_init_index = getattr(self, "_native_init_index", -1) + 1
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos = reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)
            pose_p = self.goal_site.pose.p.tolist()[0]
            pose_q = self.goal_site.pose.q.tolist()[0]
            pose_p[2] = -0.05
            self.goal_site.set_pose(sapien.Pose(p=pose_p, q=pose_q))
            # print(self.goal_site.pose.p)

    def evaluate(self, solve_complete_eval=False):
        self.successflag = torch.tensor([False])
        self.failureflag = torch.tensor([False])

        # Use encapsulated sequence task check function
        if (
            self.use_demonstrationwrapper == False
        ):  # change subgoal after planner ends during recording
            if solve_complete_eval == True:
                allow_subgoal_change_this_timestep = True
            else:
                allow_subgoal_change_this_timestep = False
        else:  # during demonstration, video needs to call evaluate(solve_complete_eval), video ends and flag changes in demonstrationwrapper
            if solve_complete_eval == True or self.demonstration_record_traj == False:
                allow_subgoal_change_this_timestep = True
            else:
                allow_subgoal_change_this_timestep = False
        (
            all_tasks_completed,
            current_task_name,
            task_failed,
            self.current_task_specialflag,
        ) = sequential_task_check(
            self,
            self.task_list,
            allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep,
        )

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

    # Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):

        # highlight_obj(self,self.target_cube, start_step=0, end_step=100, cur_step=timestep)

        if self.current_task_specialflag == "swap":
            if self.onto_goalsite == False:
                self.onto_goalsite = True
                self.start_step = int(self.elapsed_steps.item())
                self.end_step = int(self.elapsed_steps.item()) + 50

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


from robomme_plus.registry import TASKS
from robomme_plus.tasks.reference.utils.place_environment import (
    VideoPlaceMixin,
)


@register_env(TASKS["VideoPlaceButton"].gym_id)
class VideoPlaceButton(VideoPlaceMixin, _TaskScene):
    """Remember a requested cube's target relative to the demonstrated button press."""
    PERCEPTUAL_VARIATIONS = True

    task_name = "VideoPlaceButton"
