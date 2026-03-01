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
)
from ..robomme_env.utils.choice_action_mapping import select_target_with_pixel
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
    Input to step is command_dict (containing choice and optional pixel point).
    step returns obs as dict-of-lists and reward/terminated/truncated as last-step values.
    """

    def __init__(self, env, env_id, gui_render=True):
        super().__init__(env)
        self.env_id = env_id
        self.gui_render = gui_render

        self.planner = None
        self.language_goal = None

        # State: current available options
        self.available_options = []
        self._oracle_screw_max_attempts = 3
        self._oracle_rrt_max_attempts = 3
        self._front_camera_intrinsic_cv = None
        self._front_camera_extrinsic_cv = None
        self._front_rgb_shape = None

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
        self._update_front_camera_cache(obs_like=obs, info_like=info)
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
    def _to_numpy(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _take_last_columnar(value):
        if isinstance(value, list):
            return value[-1] if value else None
        return value

    @classmethod
    def _normalize_intrinsic_cv(cls, intrinsic_like):
        intrinsic = cls._to_numpy(intrinsic_like)
        if intrinsic is None:
            return None
        intrinsic = intrinsic.reshape(-1)
        if intrinsic.size < 9:
            return None
        intrinsic = intrinsic[:9].reshape(3, 3)
        if not np.all(np.isfinite(intrinsic)):
            return None
        return intrinsic.astype(np.float64, copy=False)

    @classmethod
    def _normalize_extrinsic_cv(cls, extrinsic_like):
        extrinsic = cls._to_numpy(extrinsic_like)
        if extrinsic is None:
            return None
        extrinsic = extrinsic.reshape(-1)
        if extrinsic.size < 12:
            return None
        extrinsic = extrinsic[:12].reshape(3, 4)
        if not np.all(np.isfinite(extrinsic)):
            return None
        return extrinsic.astype(np.float64, copy=False)

    def _update_front_camera_cache(self, obs_like=None, info_like=None):
        obs_dict = obs_like if isinstance(obs_like, dict) else {}
        info_dict = info_like if isinstance(info_like, dict) else {}

        front_rgb = self._take_last_columnar(obs_dict.get("front_rgb_list"))
        front_rgb_np = self._to_numpy(front_rgb)
        if front_rgb_np is not None and front_rgb_np.ndim >= 2:
            self._front_rgb_shape = tuple(front_rgb_np.shape[:2])

        front_extrinsic = self._take_last_columnar(
            obs_dict.get("front_camera_extrinsic_list")
        )
        front_extrinsic_np = self._normalize_extrinsic_cv(front_extrinsic)
        if front_extrinsic_np is not None:
            self._front_camera_extrinsic_cv = front_extrinsic_np

        front_intrinsic = self._take_last_columnar(info_dict.get("front_camera_intrinsic"))
        front_intrinsic_np = self._normalize_intrinsic_cv(front_intrinsic)
        if front_intrinsic_np is not None:
            self._front_camera_intrinsic_cv = front_intrinsic_np

    @staticmethod
    def _empty_target():
        return {
            "obj": None,
            "name": None,
            "seg_id": None,
            "position": None,
            "match_distance": None,
            "selection_mode": None,
        }

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
        if "choice" not in command_dict:
            return None, None

        target_choice = command_dict.get("choice")
        if not isinstance(target_choice, str):
            return None, None
        target_choice = target_choice.strip()
        if not target_choice:
            return None, None
        target_label = target_choice.lower()

        found_idx = find_exact_label_option_index(target_label, solve_options)
        if found_idx == -1:
            logger.debug(
                f"Error: Choice '{target_choice}' not found in current options by exact label match."
            )
            return None, None

        point = command_dict.get("point")
        if point is None:
            return found_idx, None
        if not isinstance(point, (list, tuple, np.ndarray)) or len(point) < 2:
            return found_idx, None
        try:
            y = float(point[0])
            x = float(point[1])
        except (TypeError, ValueError):
            return found_idx, None
        if not np.isfinite(x) or not np.isfinite(y):
            return found_idx, None
        # select_target_with_pixel expects [x, y].
        return found_idx, [int(np.rint(x)), int(np.rint(y))]

    def _apply_position_target(self, selected_target, option, target_pixel):
        if target_pixel is None:
            return

        best_cand = select_target_with_pixel(
            available=option.get("available"),
            pixel_like=target_pixel,
            intrinsic_cv=self._front_camera_intrinsic_cv,
            extrinsic_cv=self._front_camera_extrinsic_cv,
            image_shape=self._front_rgb_shape,
        )
        if best_cand is not None:
            selected_target.update(best_cand)

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

    def _format_step_output(self, batch):
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = batch
        self._update_front_camera_cache(obs_like=obs_batch, info_like=info_batch)
        info_flat = self._flatten_info_batch(info_batch)
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
        Execute one step: action is command_dict, must contain "choice", optional
        pixel `point=[y, x]` in front_rgb.
        Return last-step signals for reward/terminated/truncated while keeping obs as dict-of-lists.
        """
        # 1) Build solver options once and prepare a mutable selected_target holder for solve() closures.
        selected_target, solve_options = self._build_step_options()
        # 2) Validate/resolve the incoming command into (option index, optional target position).
        found_idx, target_pixel = self._resolve_command(action, solve_options)

        # 3) For invalid command or unmatched choice, keep legacy behavior: return an empty dense batch.
        if found_idx is None:
            return self._format_step_output(planner_denseStep.empty_step_batch())

        # 4) If a point is provided, map it to the nearest candidate target.
        option = solve_options[found_idx]
        self._apply_position_target(
            selected_target=selected_target,
            option=option,
            target_pixel=target_pixel,
        )

        requires_target = "available" in option
        if requires_target:
            if target_pixel is None:
                raise ValueError(
                    f"Multi-choice action '{option.get('action', 'Unknown')}' requires "
                    "a target pixel point=[y, x], but command did not provide it."
                )
            if selected_target.get("obj") is None:
                raise ValueError(
                    f"Multi-choice action '{option.get('action', 'Unknown')}' could not match "
                    f"any available candidate from point={target_pixel}."
                )

        # 5) Execute selected solve() with dense step collection; raise on solve == -1.
        batch = self._execute_selected_option(found_idx, solve_options)
        # 6) Run post-solve environment evaluation to keep existing side effects and logging.
        self._post_eval()
        # 7) Convert batch to wrapper output contract (last reward/terminated/truncated + flattened info).
        return self._format_step_output(batch)
