import gymnasium as gym
import numpy as np
import torch
import cv2

from robomme.robomme_env.utils.vqa_options import get_vqa_options
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)
from ..robomme_env.utils import planner_denseStep


# -----------------------------------------------------------------------------
# Module: Oracle Planner Demonstration Wrapper
# Connect Robomme Oracle planning logic in Gym environment, support step-by-step observation collection.
# Oracle logic below is inlined from history_bench_sim.oracle_logic, cooperating with
# planner_denseStep, aggregating multiple internal env.step calls into a unified batch return.
# -----------------------------------------------------------------------------


def _find_best_semantic_match(user_query, options):
    """
    Use rapidfuzz edit distance to find the best match for user_query in options labels;
    Used for semantic fallback when exact action name is not found.
    Return (best_idx, best_score), if no match or exception, best_idx=-1, score=0.0.
    """
    from rapidfuzz import process, fuzz
    if not options:
        return -1, 0.0
    labels = [opt.get("label", "") for opt in options]
    query_text = str(user_query or "").strip()
    try:
        result = process.extractOne(query_text, labels, scorer=fuzz.ratio)
        if result:
            match_text, score, best_idx = result
            best_score = score / 100.0
        else:
            return -1, 0.0
    except Exception as exc:
        print(f"  [NLP] Edit distance match failed ({exc}), fallback to option 1.")
        return 0, 0.0
    print(f"  [NLP] Nearest semantic match (edit distance): '{query_text}' -> '{labels[best_idx]}' (Score: {best_score:.4f})")
    return best_idx, best_score


def step_after(env, planner, env_id, seg_raw, command_dict):
    """
    Execute one Oracle action based on command_dict (containing action and optional point),
    Return unified dense batch (obs/info values are list, reward/terminated/truncated are 1D tensor).
    """
    selected_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
    solve_options = get_vqa_options(env, planner, selected_target, env_id)
    target_action = command_dict.get("action")
    target_param = command_dict.get("point")
    # Return empty batch if no action
    if "action" not in command_dict:
        return planner_denseStep.empty_step_batch()
    if target_action is None:
        return planner_denseStep.empty_step_batch()
    found_idx = -1
    # Find action index in solution options by label or index
    for i, opt in enumerate(solve_options):
        if opt.get("label") == target_action or str(i + 1) == str(target_action):
            found_idx = i
            break
    # If not hit and is string and not digit, try semantic match
    if found_idx == -1 and isinstance(target_action, str) and not target_action.isdigit():
        print(f"Attempting semantic match for '{target_action}'...")
        found_idx, score = _find_best_semantic_match(target_action, solve_options)
    if found_idx == -1:
        print(f"Error: Action '{target_action}' not found in current options.")
        return planner_denseStep.empty_step_batch()
    # If click coordinates provided and segmentation map exists, parse nearest object and fill selected_target
    if target_param is not None and seg_raw is not None:
        cx, cy = target_param
        h, w = seg_raw.shape[:2]
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))
        seg_id_map = getattr(env.unwrapped, "segmentation_id_map", {}) or {}
        candidates = []

        def _collect(item):
            """Recursively collect actionable objects in available."""
            if isinstance(item, (list, tuple)):
                for x in item:
                    _collect(x)
            elif isinstance(item, dict):
                for x in item.values():
                    _collect(x)
            else:
                if item:
                    candidates.append(item)

        avail = solve_options[found_idx].get("available")
        if avail:
            _collect(avail)
            best_cand = None
            min_dist = float("inf")
            # Find object nearest to click point in segmentation map
            for actor in candidates:
                target_ids = [sid for sid, obj in seg_id_map.items() if obj is actor]
                for tid in target_ids:
                    tid = int(tid)
                    mask = seg_raw == tid
                    if np.any(mask):
                        ys, xs = np.nonzero(mask)
                        center_x, center_y = xs.mean(), ys.mean()
                        dist = (center_x - cx) ** 2 + (center_y - cy) ** 2
                        if dist < min_dist:
                            min_dist = dist
                            best_cand = {
                                "obj": actor,
                                "name": getattr(actor, "name", f"id_{tid}"),
                                "seg_id": tid,
                                "click_point": (int(cx), int(cy)),
                                "centroid_point": (int(center_x), int(center_y)),
                            }
            if best_cand:
                selected_target.update(best_cand)
            else:
                selected_target["click_point"] = (int(cx), int(cy))
        else:
            selected_target["click_point"] = (int(cx), int(cy))
    print(f"Executing option: {found_idx + 1} - {solve_options[found_idx].get('label')}")

    # Wrap solve() with dense collection, collecting results of all intermediate env.step calls
    result = planner_denseStep._run_with_dense_collection(
        planner,
        lambda: solve_options[found_idx].get("solve")()
    )

    if result == -1:
        print("Warning: solve() failed (returned -1)")
        return planner_denseStep.empty_step_batch()

    env.unwrapped.evaluate()
    evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
    print(f"Evaluation result: {evaluation}")
    return result




class OraclePlannerDemonstrationWrapper(gym.Wrapper):
    """
    Wrap Robomme environment with Oracle planning logic into Gym Wrapper for demonstration/evaluation;
    Input to step is command_dict (containing action and optional point).
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

        # Action/Observation space (Empty Dict here, agreed externally)
        self.action_space = gym.spaces.Dict({})
        self.observation_space = gym.spaces.Dict({})


    def reset(self, **kwargs):
        # Select stick or arm planner based on env_id and initialize
        if self.env_id in ("PatternLock", "RouteStick"):
            self.planner = PandaStickMotionPlanningSolver(
                self.env,
                debug=False,
                vis=self.gui_render,
                base_pose=self.env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_vel_limits=0.3,
            )
        else:
            self.planner = PandaArmMotionPlanningSolver(
                self.env,
                debug=False,
                vis=self.gui_render,
                base_pose=self.env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
            )
        return self.env.reset(**kwargs)

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

    def step(self, action):
        """
        Execute one step: action is command_dict, must contain "action", optional "point".
        Return last-step signals for reward/terminated/truncated while keeping obs as dict-of-lists.
        """
        command_dict = action

        # Directly get current observation and segmentation map (no separate step_before)
        obs = self.env.unwrapped.get_obs(unflattened=True)
        seg = obs["sensor_data"]["base_camera"]["segmentation"]
        seg = seg.cpu().numpy() if hasattr(seg, "cpu") else np.asarray(seg)
        self.seg_raw = (seg[0] if seg.ndim > 2 else seg).squeeze().astype(np.int64)

        dummy_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
        raw_options = get_vqa_options(self.env, self.planner, dummy_target, self.env_id)
        self.available_options = [
            {"action": opt.get("label", "Unknown"), "need_parameter": bool(opt.get("available"))}
            for opt in raw_options
        ]

        # Call step_after to execute action and get unified batch
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = step_after(
            self.env, self.planner, self.env_id, self.seg_raw, command_dict
        )
        info_flat = self._flatten_info_batch(info_batch)
        return (
            obs_batch,
            self._take_last_step_value(reward_batch),
            self._take_last_step_value(terminated_batch),
            self._take_last_step_value(truncated_batch),
            info_flat,
        )
