import gymnasium as gym
import numpy as np
import torch
import cv2
import colorsys
from rapidfuzz import process, fuzz

from historybench.HistoryBench_env.util.vqa_options import get_vqa_options
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)

def _generate_color_map(n=10000, s_min=0.70, s_max=0.95, v_min=0.78, v_max=0.95):
    phi = 0.6180339887498948
    color_map = {}
    for i in range(1, n + 1):
        h = (i * phi) % 1.0
        s = s_min + (s_max - s_min) * ((i % 7) / 6)
        v = v_min + (v_max - v_min) * (((i * 3) % 5) / 4)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        color_map[i] = [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))]
    return color_map

def _sync_table_color(env, color_map):
    seg_id_map = getattr(env.unwrapped, "segmentation_id_map", None)
    if not isinstance(seg_id_map, dict):
        return
    for obj_id, obj in seg_id_map.items():
        if getattr(obj, "name", None) == "table-workspace":
            color_map[obj_id] = [0, 0, 0]


# -----------------------------------------------------------------------------
# Oracle logic (inlined from history_bench_sim.oracle_logic)
# -----------------------------------------------------------------------------

def _prepare_frame(frame):
    """Preprocess frame to uint8."""
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        max_val = float(np.max(frame)) if frame.size else 0.0
        if max_val <= 1.0:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    return frame


def _prepare_segmentation_visual(segmentation, color_map, target_hw):
    """Convert segmentation to visual image."""
    if segmentation is None:
        return None, None
    seg = segmentation
    if hasattr(seg, "cpu"):
        seg = seg.cpu().numpy()
    seg = np.asarray(seg)
    if seg.ndim > 2:
        seg = seg[0]
    seg_2d = seg.squeeze().astype(np.int64)
    h, w = seg_2d.shape[:2]
    seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    unique_ids = np.unique(seg_2d)
    for seg_id in unique_ids:
        if seg_id <= 0:
            continue
        color = color_map.get(int(seg_id))
        if color is None:
            continue
        seg_rgb[seg_2d == seg_id] = color
    seg_bgr = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)
    target_h, target_w = target_hw
    if seg_bgr.shape[:2] != (target_h, target_w):
        seg_bgr = cv2.resize(seg_bgr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return seg_bgr, seg_2d


def _fetch_segmentation(env):
    """Get segmentation from env."""
    obs = env.unwrapped.get_obs(unflattened=True)
    return obs["sensor_data"]["base_camera"]["segmentation"]


def _build_solve_options(env, planner, selected_target, env_id):
    """Build available action options."""
    return get_vqa_options(env, planner, selected_target, env_id)


def _find_best_semantic_match(user_query, options):
    """Find best option by character edit distance (rapidfuzz)."""
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
        print(f"  [NLP] Edit Distance match failed ({exc}); defaulting to option 1.")
        return 0, 0.0
    print(f"  [NLP] Closest Match (Edit Distance): '{query_text}' -> '{labels[best_idx]}' (Score: {best_score:.4f})")
    return best_idx, best_score


def step_before(env, planner, env_id, color_map, use_segmentation=False):
    """Called before executing action; returns current state and available options."""
    base_frames = getattr(env, "frames", [])
    if not base_frames:
        base_frames = getattr(env.unwrapped, "frames", []) or []
    wrist_frames = getattr(env, "wrist_frames", [])
    if not wrist_frames:
        wrist_frames = getattr(env.unwrapped, "wrist_frames", []) or []
    seg_data = _fetch_segmentation(env)
    seg_hw = (255, 255)
    if base_frames and len(base_frames) > 0:
        seg_hw = base_frames[-1].shape[:2]
    elif seg_data is not None:
        try:
            temp = seg_data
            if hasattr(temp, "cpu"):
                temp = temp.cpu().numpy()
            temp = np.asarray(temp)
            if temp.ndim > 2:
                temp = temp[0]
            seg_hw = temp.shape[:2]
        except Exception:
            pass
    seg_vis = None
    seg_raw = None
    if use_segmentation:
        seg_vis, seg_raw = _prepare_segmentation_visual(seg_data, color_map, seg_hw)
    else:
        _, seg_raw = (_prepare_segmentation_visual(seg_data, color_map, seg_hw)
                      if seg_data is not None else (None, None))
        if base_frames:
            vis_frame = _prepare_frame(base_frames[-1])
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            if vis_frame.shape[:2] != seg_hw:
                vis_frame = cv2.resize(vis_frame, (seg_hw[1], seg_hw[0]), interpolation=cv2.INTER_LINEAR)
            seg_vis = vis_frame
    if seg_vis is None:
        seg_vis = np.zeros((seg_hw[0], seg_hw[1], 3), dtype=np.uint8)
    dummy_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
    raw_options = _build_solve_options(env, planner, dummy_target, env_id)
    available_options = [
        {"action": opt.get("label", "Unknown"), "need_parameter": bool(opt.get("available"))}
        for opt in raw_options
    ]
    return seg_vis, seg_raw, base_frames, wrist_frames, available_options


def step_after(env, planner, env_id, seg_vis, seg_raw, base_frames, wrist_frames, command_dict):
    """Execute action from command_dict and return evaluation."""
    selected_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
    solve_options = _build_solve_options(env, planner, selected_target, env_id)
    target_action = command_dict.get("action")
    target_param = command_dict.get("point")
    if "action" not in command_dict:
        return None
    if target_action is None:
        return None
    found_idx = -1
    for i, opt in enumerate(solve_options):
        if opt.get("label") == target_action or str(i + 1) == str(target_action):
            found_idx = i
            break
    if found_idx == -1 and isinstance(target_action, str) and not target_action.isdigit():
        print(f"Attempting semantic match for: '{target_action}'")
        found_idx, score = _find_best_semantic_match(target_action, solve_options)
    if found_idx == -1:
        print(f"Error: Action '{target_action}' not found in current options.")
        return None
    if target_param is not None and seg_raw is not None:
        cx, cy = target_param
        h, w = seg_raw.shape[:2]
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))
        seg_id_map = getattr(env.unwrapped, "segmentation_id_map", {}) or {}
        candidates = []

        def _collect(item):
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
    print(f"Executing Option: {found_idx + 1} - {solve_options[found_idx].get('label')}")
    solve_options[found_idx].get("solve")()
    env.unwrapped.evaluate()
    evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
    print(f"Evaluation: {evaluation}")
    return evaluation


def _tensor_to_bool(value):
    """Convert tensor or other types to boolean."""
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)

class OraclePlannerDemonstrationWrapper(gym.Wrapper):
    """
    Gym Environment Wrapper for Oracle Planner Logic.
    Wraps the environment to provide oracle planner capabilities.
    """
    def __init__(self, env, env_id, gui_render=True):
        super().__init__(env)
        self.env_id = env_id
        self.gui_render = gui_render
        
        self.planner = None
        self.color_map = None
        self.language_goal = None
        
        # State variables
        self.seg_vis = None
        self.seg_raw = None
        self.base_frames = []
        self.wrist_frames = []
        self.available_options = []
        
        # Metadata
        self.action_space = gym.spaces.Dict({}) 
        self.observation_space = gym.spaces.Dict({})

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # Initialization logic
        try:
            self.language_goal = self.env.unwrapped.demonstration_data.get('language goal')
        except AttributeError:
             # Fallback if demonstration_data is missing, possibly read from metadata if available
             # or simply default to None/Empty
            self.language_goal = None
            print(f"Warning: {self.env_id} object has no attribute 'demonstration_data'. 'language_goal' set to None.")

        # Generate semantic segmentation color map
        self.color_map = _generate_color_map()
        _sync_table_color(self.env, self.color_map)

        # Initialize Planner
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

        # Initial step_before
        self.seg_vis, self.seg_raw, self.base_frames, self.wrist_frames, self.available_options = \
            step_before(self.env, self.planner, self.env_id, self.color_map)
            
        info["available_options"] = self.available_options
        info["seg_vis"] = self.seg_vis
        info["seg_raw"] = self.seg_raw
        return self._get_obs(obs), info

    def step(self, action):
        """
        Args:
            action: command_dict containing "action" and "point"
        """
        command_dict = action

        # Determine where to get the data from
        env_with_data = self.env
        if not hasattr(env_with_data, "frames"):
             env_with_data = self.env.unwrapped

        # Record start index before execution
        start_idx = len(getattr(env_with_data, "frames", []))

        # Get current state and options (step_before) before executing action
        self.seg_vis, self.seg_raw, self.base_frames, self.wrist_frames, self.available_options = \
            step_before(self.env, self.planner, self.env_id, self.color_map)

        # Execute action (step_after)
        evaluation = step_after(
            self.env, 
            self.planner, 
            self.env_id, 
            self.seg_vis, 
            self.seg_raw, 
            self.base_frames, 
            self.wrist_frames, 
            command_dict
        )
        
        success = False
        fail = False
        
        if evaluation:
            fail = _tensor_to_bool(evaluation.get("fail", False))
            success = _tensor_to_bool(evaluation.get("success", False))
            
        terminated = success or fail
        truncated = False 
        reward = 1.0 if success else 0.0
        
        info = {}
        if evaluation:
            info.update(evaluation)

        info["available_options"] = self.available_options
        info["seg_vis"] = self.seg_vis
        info["seg_raw"] = self.seg_raw

        # Retrieve sliced lists (data generated during this step)
        step_frames = getattr(env_with_data, "frames", [])[start_idx:]
        step_wrist_frames = getattr(env_with_data, "wrist_frames", [])[start_idx:]
        step_actions = getattr(env_with_data, "actions", [])[start_idx:]
        step_states = getattr(env_with_data, "states", [])[start_idx:]
        step_velocity = getattr(env_with_data, "velocity", [])[start_idx:]
        step_subgoal = getattr(env_with_data, "subgoal", [])[start_idx:]
        step_subgoal_grounded = getattr(env_with_data, "subgoal_grounded", [])[start_idx:]

        # Update info with step-specific subgoals
        info["subgoal"] = step_subgoal
        info["subgoal_grounded"] = step_subgoal_grounded

        try:
            base_obs = self.env.unwrapped.get_obs(unflattened=True)
        except Exception:
            base_obs = {}
        if not isinstance(base_obs, dict):
            base_obs = {}
        
        obs = self._get_obs(base_obs)
        
        # Override obs with step-specific data
        obs["frames"] = step_frames
        obs["wrist_frames"] = step_wrist_frames
        obs["actions"] = step_actions
        obs["states"] = step_states
        obs["velocity"] = step_velocity

        return obs, reward, terminated, truncated, info

    def _get_obs(self, base_obs=None):
        if base_obs is None or not isinstance(base_obs, dict):
            base_obs = {}
        base_obs["language_goal"] = self.language_goal
        base_obs["frames"] = self.base_frames
        base_obs["wrist_frames"] = self.wrist_frames
        return base_obs
