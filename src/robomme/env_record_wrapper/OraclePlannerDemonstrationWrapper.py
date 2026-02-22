import gymnasium as gym
import numpy as np
import torch

from robomme.robomme_env.utils.vqa_options import get_vqa_options
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)
from ..robomme_env.utils import planner_denseStep
from ..robomme_env.utils.oracle_action_matcher import (
    find_exact_label_option_index,
    normalize_and_clip_point_xy,
    select_target_with_point,
)
from ..logging_utils import logger


# -----------------------------------------------------------------------------
# Module: Oracle Planner Demonstration Wrapper
# Connect Robomme Oracle planning logic in Gym environment, support step-by-step observation collection.
# Oracle logic below is inlined from history_bench_sim.oracle_logic, cooperating with
# planner_denseStep, aggregating multiple internal env.step calls into a unified batch return.
# -----------------------------------------------------------------------------


class OraclePlannerDemonstrationWrapper(gym.Wrapper):
    """
    Wrap Robomme environment with Oracle planning logic into Gym Wrapper for demonstration/evaluation;
    Input to step is command_dict (containing label and optional point).
    step returns obs as dict-of-lists and reward/terminated/truncated as last-step values.
    """

    def __init__(self, env, env_id, gui_render=True):
        super().__init__(env)
        self.env_id = env_id
        self.gui_render = gui_render

        self.planner = None
        self.language_goal = None

        # State: segmentation map, frame buffer, current available options
        self.seg_vis = None
        self.seg_raw = None
        self.base_frames = []
        self.wrist_frames = []
        self.available_options = []
        self._oracle_screw_max_attempts = 3
        self._oracle_rrt_max_attempts = 3

        # Action/Observation space (Empty Dict here, agreed externally)
        self.action_space = gym.spaces.Dict({})
        self.observation_space = gym.spaces.Dict({})

    def _wrap_planner_with_screw_then_rrt_retry(self, planner, screw_failure_exc):
        original_move_to_pose_with_screw = planner.move_to_pose_with_screw
        original_move_to_pose_with_rrt = planner.move_to_pose_with_RRTStar

        def _move_to_pose_with_screw_then_rrt_retry(*args, **kwargs):
            for attempt in range(1, self._oracle_screw_max_attempts + 1):
                try:
                    result = original_move_to_pose_with_screw(*args, **kwargs)
                except screw_failure_exc as exc:
                    logger.debug(
                        f"[OraclePlannerWrapper] screw planning failed "
                        f"(attempt {attempt}/{self._oracle_screw_max_attempts}): {exc}"
                    )
                    continue

                if isinstance(result, int) and result == -1:
                    logger.debug(
                        f"[OraclePlannerWrapper] screw planning returned -1 "
                        f"(attempt {attempt}/{self._oracle_screw_max_attempts})"
                    )
                    continue

                return result

            logger.debug(
                "[OraclePlannerWrapper] screw planning exhausted; "
                f"fallback to RRT* (max {self._oracle_rrt_max_attempts} attempts)"
            )
            for attempt in range(1, self._oracle_rrt_max_attempts + 1):
                try:
                    result = original_move_to_pose_with_rrt(*args, **kwargs)
                except Exception as exc:
                    logger.debug(
                        f"[OraclePlannerWrapper] RRT* planning failed "
                        f"(attempt {attempt}/{self._oracle_rrt_max_attempts}): {exc}"
                    )
                    continue

                if isinstance(result, int) and result == -1:
                    logger.debug(
                        f"[OraclePlannerWrapper] RRT* planning returned -1 "
                        f"(attempt {attempt}/{self._oracle_rrt_max_attempts})"
                    )
                    continue

                return result

            raise RuntimeError(
                "[OraclePlannerWrapper] screw->RRT* planning exhausted; "
                f"screw_attempts={self._oracle_screw_max_attempts}, "
                f"rrt_attempts={self._oracle_rrt_max_attempts}"
            )

        planner.move_to_pose_with_screw = _move_to_pose_with_screw_then_rrt_retry
        return planner

    def reset(self, **kwargs):
        # Prefer fail-aware planners; fallback to base planners if import fails.
        try:
            from ..robomme_env.utils.planner_fail_safe import (
                FailAwarePandaArmMotionPlanningSolver,
                FailAwarePandaStickMotionPlanningSolver,
                ScrewPlanFailure,
            )
        except Exception as exc:
            logger.debug(
                "[OraclePlannerWrapper] Warning: failed to import planner_fail_safe, "
                f"fallback to base planners: {exc}"
            )
            FailAwarePandaArmMotionPlanningSolver = PandaArmMotionPlanningSolver
            FailAwarePandaStickMotionPlanningSolver = PandaStickMotionPlanningSolver

            class ScrewPlanFailure(RuntimeError):
                """Placeholder exception type when fail-aware planner import is unavailable."""

        # Select stick or arm planner based on env_id and initialize.
        if self.env_id in ("PatternLock", "RouteStick"):
            self.planner = FailAwarePandaStickMotionPlanningSolver(
                self.env,
                debug=False,
                vis=self.gui_render,
                base_pose=self.env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_vel_limits=0.3,
            )
        else:
            self.planner = FailAwarePandaArmMotionPlanningSolver(
                self.env,
                debug=False,
                vis=self.gui_render,
                base_pose=self.env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
            )
        self._wrap_planner_with_screw_then_rrt_retry(
            self.planner,
            screw_failure_exc=ScrewPlanFailure,
        )
        ret = self.env.reset(**kwargs)
        if isinstance(ret, tuple) and len(ret) == 2:
            obs, info = ret
        else:
            obs, info = ret, {}
        self._build_step_options()
        if isinstance(info, dict):
            info["available_multi_choices"] = self.available_options
        return obs, info

    @staticmethod
    def _flatten_info_batch(info_batch: dict) -> dict:
        return {k: v[-1] if isinstance(v, list) and v else v for k, v in info_batch.items()}

    @staticmethod
    def _take_last_step_value(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 0 or value.ndim == 0:
                return value
            return value.reshape(-1)[-1]
        if isinstance(value, np.ndarray):
            if value.size == 0 or value.ndim == 0:
                return value
            return value.reshape(-1)[-1]
        if isinstance(value, (list, tuple)):
            return value[-1] if value else value
        return value

    @staticmethod
    def _empty_target():
        return {
            "obj": None,
            "name": None,
            "seg_id": None,
            "click_point": None,
            "centroid_point": None,
            "selection_mode": None,
            "used_random_fallback": False,
        }

    def _extract_seg_raw(self):
        obs = self.env.unwrapped.get_obs(unflattened=True)
        seg = obs["sensor_data"]["base_camera"]["segmentation"]
        seg = seg.cpu().numpy() if hasattr(seg, "cpu") else np.asarray(seg)
        seg_raw = (seg[0] if seg.ndim > 2 else seg).squeeze().astype(np.int64)
        return seg_raw

    def _build_step_options(self):
        selected_target = self._empty_target()
        solve_options = get_vqa_options(self.env, self.planner, selected_target, self.env_id)
        self.available_options = [
            {"label": opt.get("label"), "action": opt.get("action", "Unknown"), "need_parameter": bool(opt.get("available"))}
            for opt in solve_options
        ]
        return selected_target, solve_options

    def _resolve_command(self, command_dict, solve_options):
        if not isinstance(command_dict, dict):
            return None, None
        if "label" not in command_dict:
            return None, None

        target_label = command_dict.get("label")
        if not isinstance(target_label, str) or not target_label:
            return None, None

        found_idx = find_exact_label_option_index(target_label, solve_options)
        if found_idx == -1:
            logger.debug(
                f"Error: Label '{target_label}' not found in current options by exact label match."
            )
            return None, None

        return found_idx, command_dict.get("point")

    def _apply_click_target(self, selected_target, option, target_point, seg_raw):
        if target_point is None or seg_raw is None:
            return

        h, w = seg_raw.shape[:2]
        seg_id_map = getattr(self.env.unwrapped, "segmentation_id_map", {}) or {}
        best_cand = select_target_with_point(
            seg_raw=seg_raw,
            seg_id_map=seg_id_map,
            available=option.get("available"),
            point_like=target_point,
        )
        if best_cand is not None:
            selected_target.update(best_cand)
            return

        click_point = normalize_and_clip_point_xy(target_point, width=w, height=h)
        if click_point is not None:
            selected_target["click_point"] = click_point

    def _execute_selected_option(self, option_idx, solve_options):
        option = solve_options[option_idx]
        logger.debug(f"Executing option: {option_idx + 1} - {option.get('action')}")

        result = planner_denseStep._run_with_dense_collection(
            self.planner,
            lambda: option.get("solve")(),
        )
        if result == -1:
            action_text = option.get("action", "Unknown")
            raise RuntimeError(
                f"Oracle solve failed after screw->RRT* retries for env '{self.env_id}', "
                f"action '{action_text}' (index {option_idx + 1})."
            )
        return result

    def _post_eval(self):
        self.env.unwrapped.evaluate()
        evaluation = self.env.unwrapped.evaluate(solve_complete_eval=True)
        logger.debug(f"Evaluation result: {evaluation}")

    @staticmethod
    def _frame_count_from_obs_batch(obs_batch) -> int:
        if not isinstance(obs_batch, dict):
            return 0
        front_rgb_list = obs_batch.get("front_rgb_list")
        if isinstance(front_rgb_list, list):
            return len(front_rgb_list)
        return 0

    @staticmethod
    def _build_fallback_blue_box_mask(
        frame_count: int,
        used_random_fallback: bool,
    ) -> list[bool]:
        n = max(0, int(frame_count))
        if n == 0:
            return []
        if not used_random_fallback:
            return [False] * n
        return [True] + ([False] * (n - 1))

    def _format_step_output(self, batch, used_random_fallback: bool = False):
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = batch
        info_flat = self._flatten_info_batch(info_batch)
        frame_count = self._frame_count_from_obs_batch(obs_batch)
        info_flat["oracle_random_fallback_used"] = bool(used_random_fallback)
        info_flat["oracle_random_fallback_blue_box_mask"] = self._build_fallback_blue_box_mask(
            frame_count=frame_count,
            used_random_fallback=used_random_fallback,
        )
        info_flat["available_multi_choices"] = getattr(self, "available_options", [])
        return (
            obs_batch,
            self._take_last_step_value(reward_batch),
            self._take_last_step_value(terminated_batch),
            self._take_last_step_value(truncated_batch),
            info_flat,
        )

    def step(self, action):
        """
        Execute one step: action is command_dict, must contain "label", optional "point".
        Return last-step signals for reward/terminated/truncated while keeping obs as dict-of-lists.
        """
        # 1) Read the latest segmentation map from current observation for click-to-target grounding.
        self.seg_raw = self._extract_seg_raw()
        # 2) Build solver options once and prepare a mutable selected_target holder for solve() closures.
        selected_target, solve_options = self._build_step_options()
        # 3) Validate/resolve the incoming command into (option index, optional click point).
        found_idx, target_point = self._resolve_command(action, solve_options)

        # 4) For invalid command or unmatched label, keep legacy behavior: return an empty dense batch.
        if found_idx is None:
            return self._format_step_output(planner_denseStep.empty_step_batch())

        # 5) If a point is provided, map it to a concrete candidate target (or fallback click point only).
        self._apply_click_target(
            selected_target=selected_target,
            option=solve_options[found_idx],
            target_point=target_point,
            seg_raw=self.seg_raw,
        )
        used_random_fallback = bool(selected_target.get("used_random_fallback", False))
        # 6) Execute selected solve() with dense step collection; raise on solve == -1.
        batch = self._execute_selected_option(found_idx, solve_options)
        # 7) Run post-solve environment evaluation to keep existing side effects and logging.
        self._post_eval()
        # 8) Convert batch to wrapper output contract (last reward/terminated/truncated + flattened info).
        return self._format_step_output(batch, used_random_fallback=used_random_fallback)
