import os
import sys
import numpy as np
import gymnasium as gym
import cv2
import colorsys
import torch
from pathlib import Path
from PIL import Image

# --- Setup Paths ---
# Ensure we can import local project packages from parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# --- NLP Imports ---
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    print("Loading NLP Model (all-MiniLM-L6-v2)...")
    _NLP_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    print("NLP Model loaded.")
except ImportError:
    print("Warning: sentence-transformers not found. NLP matching will fail.")
    _NLP_MODEL = None
except Exception as e:
    print(f"Error loading NLP model: {e}")
    _NLP_MODEL = None

# --- Project Imports ---
from robomme.env_record_wrapper import BenchmarkEnvBuilder
from robomme.robomme_env import *  # noqa: F401,F403; ensure gym envs are registered
from robomme.robomme_env.utils.vqa_options import get_vqa_options
from robomme.robomme_env.utils.oracle_action_matcher import (
    find_exact_label_option_index,
    map_action_text_to_option_label,
)
from robomme.robomme_env.utils.choice_action_mapping import (
    extract_actor_position_xyz,
    project_world_to_pixel,
    select_target_with_position,
)
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.motionplanner_stick import PandaStickMotionPlanningSolver

# --- FailAware Planner Imports ---
try:
    from robomme.robomme_env.utils.planner_fail_safe import (
        FailAwarePandaArmMotionPlanningSolver,
        FailAwarePandaStickMotionPlanningSolver,
        ScrewPlanFailure,
    )
except ImportError as e:
    print(f"Warning: Failed to import robomme fail-aware planners: {e}")
    # Fallback to regular planners
    FailAwarePandaArmMotionPlanningSolver = PandaArmMotionPlanningSolver
    FailAwarePandaStickMotionPlanningSolver = PandaStickMotionPlanningSolver
    ScrewPlanFailure = RuntimeError

# --- Constants ---
ROBOMME_METADATA_ROOT_ENV = "ROBOMME_METADATA_ROOT"
# For backward compatibility with process_session constructor naming.
# Semantics: optional override root for metadata json files.
DEFAULT_DATASET_ROOT = os.environ.get(ROBOMME_METADATA_ROOT_ENV)

# --- Helper Functions from Script ---

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

def _tensor_to_bool(value):
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)

def _prepare_frame(frame):
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
    obs = env.unwrapped.get_obs(unflattened=True)
    return obs["sensor_data"]["base_camera"]["segmentation"]

def _build_solve_options(env, planner, selected_target, env_id):
    return get_vqa_options(env, planner, selected_target, env_id)

def _extract_last_text(value, default="Unknown Goal"):
    if isinstance(value, str):
        text = value.strip()
        return text or default
    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            if item is None:
                continue
            text = str(item).strip()
            if text:
                return text
    return default

def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []

def _to_frame_list(frames_like):
    if frames_like is None:
        return []
    if isinstance(frames_like, list):
        return frames_like
    if isinstance(frames_like, tuple):
        return list(frames_like)
    if isinstance(frames_like, torch.Tensor):
        arr = frames_like.detach().cpu().numpy()
        if arr.ndim == 3:
            return [arr]
        if arr.ndim == 4:
            return [x for x in arr]
        return []
    if isinstance(frames_like, np.ndarray):
        if frames_like.ndim == 3:
            return [frames_like]
        if frames_like.ndim == 4:
            return [x for x in frames_like]
        return []
    return []

def _iter_env_chain(env, max_depth=16):
    current = env
    seen = set()
    for _ in range(max_depth):
        if current is None:
            return
        env_id = id(current)
        if env_id in seen:
            return
        seen.add(env_id)
        yield current
        current = getattr(current, "env", None)

def _extract_obs_front_frames(env):
    """
    Strict path: only use wrapper-produced obs batch front_rgb_list.
    Returns (front_list, obs_ref_id) or (None, None) if unavailable.
    """
    for wrapped in _iter_env_chain(env):
        for attr_name in ("_last_obs", "last_obs"):
            obs_candidate = getattr(wrapped, attr_name, None)
            if not isinstance(obs_candidate, dict):
                continue
            if "front_rgb_list" not in obs_candidate:
                continue
            front_list = _to_frame_list(obs_candidate.get("front_rgb_list"))
            return front_list, id(obs_candidate)
    return None, None

def _collect_front_frames_from_step_output(step_output):
    """
    Extract front camera frames from a single env.step(...) output.
    Supports both classic step tuple and dense batch tuple.
    """
    if not (isinstance(step_output, tuple) and len(step_output) == 5):
        return []
    obs = step_output[0]
    if not isinstance(obs, dict):
        return []
    return _to_frame_list(obs.get("front_rgb_list"))


def _collect_choice_segment_candidates(item, out):
    if isinstance(item, (list, tuple)):
        for child in item:
            _collect_choice_segment_candidates(child, out)
        return
    if isinstance(item, dict):
        for child in item.values():
            _collect_choice_segment_candidates(child, out)
        return
    if item is not None:
        out.append(item)


def _extract_choice_segment_position_xyz(current_segment):
    candidates = []
    _collect_choice_segment_candidates(current_segment, candidates)
    for candidate in candidates:
        pos = extract_actor_position_xyz(candidate)
        if pos is not None:
            return pos.astype(np.float64)
    return None


def _find_actor_segmentation_id(segmentation_id_map, actor):
    if not isinstance(segmentation_id_map, dict):
        return None
    for seg_id, obj in segmentation_id_map.items():
        if obj is actor:
            try:
                return int(seg_id)
            except Exception:
                continue
    return None


def _compute_segmentation_centroid_xy(segmentation, seg_id):
    if segmentation is None:
        return None
    try:
        seg_arr = np.asarray(segmentation)
    except Exception:
        return None
    if seg_arr.ndim > 2:
        seg_arr = np.squeeze(seg_arr)
    if seg_arr.ndim != 2:
        return None
    mask = seg_arr == int(seg_id)
    if not np.any(mask):
        return None
    ys, xs = np.nonzero(mask)
    x = int(np.rint(xs.mean()))
    y = int(np.rint(ys.mean()))
    return [x, y]

def _extract_demonstration_payload(demonstration_data):
    """
    Compatible with both legacy dict payloads and current DemonstrationWrapper tuple batch:
    - dict style: {"language goal": "...", "frames": [...]}
    - tuple/list style: (obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch)
    """
    default_goal = "Unknown Goal"
    default_frames = []

    if isinstance(demonstration_data, dict):
        goal_candidate = (
            demonstration_data.get("language goal")
            or demonstration_data.get("language_goal")
            or demonstration_data.get("task_goal")
        )
        frames_candidate = demonstration_data.get("frames")
        if frames_candidate is None:
            frames_candidate = demonstration_data.get("front_rgb_list")
        return _extract_last_text(goal_candidate, default_goal), _ensure_list(frames_candidate)

    if isinstance(demonstration_data, (tuple, list)):
        obs_batch = demonstration_data[0] if len(demonstration_data) >= 1 else None
        info_batch = demonstration_data[4] if len(demonstration_data) >= 5 else None
        if info_batch is None and len(demonstration_data) >= 2 and isinstance(demonstration_data[1], dict):
            # Fallback for (obs, info) shaped payloads
            info_batch = demonstration_data[1]

        frames_candidate = None
        if isinstance(obs_batch, dict):
            frames_candidate = obs_batch.get("front_rgb_list")

        goal_candidate = None
        if isinstance(info_batch, dict):
            goal_candidate = info_batch.get("task_goal")
            if goal_candidate is None:
                goal_candidate = info_batch.get("language goal")
            if goal_candidate is None:
                goal_candidate = info_batch.get("language_goal")

        return _extract_last_text(goal_candidate, default_goal), _ensure_list(frames_candidate)

    return default_goal, default_frames

def _find_best_semantic_match(user_query, options):
    if _NLP_MODEL is None:
        return -1, 0.0
    
    if not options:
        return -1, 0.0

    labels = [opt.get("label", "") for opt in options]
    query_text = str(user_query or "").strip()

    try:
        query_embedding = _NLP_MODEL.encode(query_text, convert_to_tensor=True)
        corpus_embeddings = _NLP_MODEL.encode(labels, convert_to_tensor=True)
        cos_scores = st_util.cos_sim(query_embedding, corpus_embeddings)[0]
        best_idx = torch.argmax(cos_scores).item()
        best_score = cos_scores[best_idx].item()
    except Exception as exc:
        print(f"  [NLP] Semantic match failed ({exc}); defaulting to option 1.")
        return 0, 0.0
    
    return best_idx, best_score

# --- Core Logic Wrapper ---

class OracleSession:
    def __init__(self, dataset_root=DEFAULT_DATASET_ROOT, gui_render=False):
        """
        gui_render: If True, uses 'human' render mode (pops up window). 
                    For Gradio, we usually want False (rgb_array).
        """
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.gui_render = gui_render # Usually False for web app
        self.render_mode = "human" if gui_render else "rgb_array"
        
        self.env = None
        self.planner = None
        self.color_map = None
        self.env_id = None
        self.episode_idx = None
        self.language_goal = ""
        self.difficulty = None
        self.seed = None
        self.history = [] # Logs interaction steps
        
        # State caches
        self.seg_vis = None
        self.seg_raw = None
        self.base_frames = []
        self.wrist_frames = []
        self.demonstration_frames = []
        self.available_options = []
        self.raw_solve_options = []
        # Track frame indices for incremental video updates
        self.last_base_frame_idx = 0
        self.last_wrist_frame_idx = 0
        self.non_demonstration_task_length = None  # 从 DemonstrationWrapper 读取
        # Track latest obs-batch object and consumed indices to avoid duplicate appends.
        self._last_obs_ref_id = None
        self._last_obs_front_consumed = 0

    def _resolve_metadata_override_root(self):
        if self.dataset_root:
            return self.dataset_root
        env_root = os.environ.get(ROBOMME_METADATA_ROOT_ENV)
        if env_root:
            return Path(env_root)
        return None

    def load_episode(self, env_id, episode_idx):
        """Initialize environment for a specific episode."""
        if self.env:
            self.env.close()

        try:
            metadata_override_root = self._resolve_metadata_override_root()
            builder = BenchmarkEnvBuilder(
                env_id=env_id,
                dataset="train",
                # Gradio uses local oracle solve() directly (not env.step(command_dict)),
                # so we must keep a low-level stepping wrapper chain.
                # "multi_choice" inserts OraclePlannerDemonstrationWrapper, which expects
                # dict commands and may swallow planner low-level action arrays.
                action_space="joint_angle",
                gui_render=self.gui_render,
                #gui_render=True,
                override_metadata_path=metadata_override_root,
                max_steps=3000,
            )

            episode_num = builder.get_episode_num()
            if episode_num <= 0:
                if metadata_override_root:
                    expected = metadata_override_root / f"record_dataset_{env_id}_metadata.json"
                    return None, f"Dataset metadata not found or empty: {expected}"
                return None, f"Dataset metadata not found or empty for env '{env_id}' in split 'test'"

            if episode_idx < 0 or episode_idx >= episode_num:
                return None, f"Episode index out of range for {env_id}: {episode_idx} (valid 0-{episode_num - 1})"

            seed, difficulty = builder.resolve_episode(episode_idx)
            self.env = builder.make_env_for_episode(episode_idx)
            self.env.reset()
            self.env_id = env_id
            self.episode_idx = episode_idx
            self.difficulty = difficulty
            self.seed = seed
            
            # Demonstration data
            demonstration_data = getattr(self.env, "demonstration_data", None)
            self.language_goal, self.demonstration_frames = _extract_demonstration_payload(demonstration_data)
            
            # Setup Color Map
            self.color_map = _generate_color_map()
            _sync_table_color(self.env, self.color_map)
            
            # Initialize Planner (using FailAware versions)
            if env_id in ("PatternLock", "RouteStick"):
                self.planner = FailAwarePandaStickMotionPlanningSolver(
                    self.env, debug=False, vis=self.gui_render,
                    base_pose=self.env.unwrapped.agent.robot.pose,
                    visualize_target_grasp_pose=False, print_env_info=False,
                    joint_vel_limits=0.3,
                )
            else:
                self.planner = FailAwarePandaArmMotionPlanningSolver(
                    self.env, debug=False, vis=self.gui_render,
                    base_pose=self.env.unwrapped.agent.robot.pose,
                    visualize_target_grasp_pose=False, print_env_info=False,
                )
            
            self.env.unwrapped.evaluate() # Initial eval check
            
            # 从 DemonstrationWrapper 读取 non_demonstration_task_length（如果存在）
            self.non_demonstration_task_length = getattr(self.env, 'non_demonstration_task_length', None)
            
            # Reset logs
            self.history = []
            
            # Reset frame indices
            self.last_base_frame_idx = 0
            self.last_wrist_frame_idx = 0
            self.base_frames = []
            self.wrist_frames = []
            self._last_obs_ref_id = None
            self._last_obs_front_consumed = 0
            
            # Initial Observation
            return self.update_observation()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error initializing episode: {e}"

    def update_observation(self, use_segmentation=True):
        """Captures current state, updates segmentation, and generates options."""
        if not self.env:
            return None, "Environment not initialized"

        # 1. Capture Frames (strict path: only front_rgb_list from wrapper obs batch)
        front_frames, obs_ref_id = _extract_obs_front_frames(self.env)
        self.wrist_frames = []
        if front_frames is not None:
            front_frames = front_frames or []
            if obs_ref_id != self._last_obs_ref_id:
                self._last_obs_ref_id = obs_ref_id
                self._last_obs_front_consumed = 0
            new_front = front_frames[self._last_obs_front_consumed:]
            self._last_obs_front_consumed = len(front_frames)
            if new_front:
                self.base_frames.extend(_prepare_frame(frame) for frame in new_front if frame is not None)
        else:
            self.base_frames = []
            self._last_obs_ref_id = None
            self._last_obs_front_consumed = 0

        seg_data = _fetch_segmentation(self.env)
        
        # 2. Determine Resolution
        seg_hw = (255, 255) # Default
        if self.base_frames and len(self.base_frames) > 0:
            seg_hw = self.base_frames[-1].shape[:2]
        elif seg_data is not None:
             # Try to guess from seg data
            try:
                temp = seg_data
                if hasattr(temp, "cpu"): temp = temp.cpu().numpy()
                temp = np.asarray(temp)
                if temp.ndim > 2: temp = temp[0]
                seg_hw = temp.shape[:2]
            except: pass

        # 3. Process Segmentation/Image
        if use_segmentation:
            self.seg_vis, self.seg_raw = _prepare_segmentation_visual(seg_data, self.color_map, seg_hw)
        else:
            # If not using segmentation view, use RGB but scale to match seg logic
             seg_vis_from_seg, self.seg_raw = (
                 _prepare_segmentation_visual(seg_data, self.color_map, seg_hw)
                 if seg_data is not None
                 else (None, None)
             )
             if self.base_frames:
                vis_frame = _prepare_frame(self.base_frames[-1])
                vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR) # Keep consistent BGR internally
                if vis_frame.shape[:2] != seg_hw:
                    vis_frame = cv2.resize(vis_frame, (seg_hw[1], seg_hw[0]), interpolation=cv2.INTER_LINEAR)
                self.seg_vis = vis_frame
             elif seg_vis_from_seg is not None:
                 # 没有 RGB 原始帧时，回退到 segmentation 可视化，避免首屏空白。
                 self.seg_vis = seg_vis_from_seg
             else:
                 self.seg_vis = np.zeros((seg_hw[0], seg_hw[1], 3), dtype=np.uint8)

        # 4. Generate Options
        dummy_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
        self.raw_solve_options = _build_solve_options(self.env, self.planner, dummy_target, self.env_id)
        
        # Format for UI
        self.available_options = []
        for i, opt in enumerate(self.raw_solve_options):
            opt_label = str(opt.get("label", f"Option {i + 1}")).strip()
            opt_action = str(opt.get("action", "")).strip()
            if opt_label and opt_action:
                ui_label = f"{opt_label}. {opt_action}"
            else:
                ui_label = opt_label or opt_action or f"Option {i + 1}"
            self.available_options.append((ui_label, i)) # Tuple for Gradio Radio/Dropdown

        return self.get_pil_image(), "Ready"

    def get_pil_image(self, use_segmented=True):
        """
        获取PIL图像
        
        Args:
            use_segmented: 如果为True，返回分割视图(seg_vis)；如果为False，返回原图(base_frames)
        """
        if use_segmented:
            # 返回分割视图
            if self.seg_vis is None:
                return Image.new('RGB', (255, 255), color='gray')
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb = cv2.cvtColor(self.seg_vis, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        else:
            # 返回原图
            if not self.base_frames or len(self.base_frames) == 0:
                return Image.new('RGB', (255, 255), color='gray')
            # 获取最后一帧
            frame = self.base_frames[-1]
            # 准备帧（确保格式正确）
            frame = _prepare_frame(frame)
            # frame 已经是 RGB 格式，直接转换为 PIL Image
            return Image.fromarray(frame)

    def close(self):
        if self.env:
            self.env.close()

    def _get_front_camera_projection_params(self):
        if not self.env:
            return None, None, None

        intrinsic = None
        extrinsic = None
        image_shape = None

        try:
            obs = self.env.unwrapped.get_obs(unflattened=True)
        except Exception:
            obs = None

        if isinstance(obs, dict):
            try:
                cam_param = obs.get("sensor_param", {}).get("base_camera", {})
                intrinsic = np.asarray(cam_param.get("intrinsic_cv")).reshape(-1)[:9].reshape(3, 3)
                extrinsic = np.asarray(cam_param.get("extrinsic_cv")).reshape(-1)[:12].reshape(3, 4)
            except Exception:
                intrinsic = None
                extrinsic = None

            try:
                rgb = obs.get("sensor_data", {}).get("base_camera", {}).get("rgb")
                if rgb is not None and hasattr(rgb, "cpu"):
                    rgb = rgb.cpu().numpy()
                rgb = np.asarray(rgb)
                if rgb.ndim == 4:
                    image_shape = (int(rgb.shape[1]), int(rgb.shape[2]))
                elif rgb.ndim == 3:
                    image_shape = (int(rgb.shape[0]), int(rgb.shape[1]))
            except Exception:
                image_shape = None

        if image_shape is None and self.seg_raw is not None:
            try:
                seg = np.asarray(self.seg_raw)
                image_shape = (int(seg.shape[0]), int(seg.shape[1]))
            except Exception:
                image_shape = None

        if image_shape is None and self.base_frames:
            frame = np.asarray(self.base_frames[-1])
            image_shape = (int(frame.shape[0]), int(frame.shape[1]))

        return intrinsic, extrinsic, image_shape

    def get_reference_action(self):
        if not self.env:
            return {
                "ok": False,
                "option_idx": None,
                "option_label": "",
                "option_action": "",
                "need_coords": False,
                "coords_xy": None,
                "message": "Environment not initialized.",
            }

        target_action_text = getattr(self.env.unwrapped, "current_choice_label", "")
        if not isinstance(target_action_text, str) or not target_action_text.strip():
            return {
                "ok": False,
                "option_idx": None,
                "option_label": "",
                "option_action": "",
                "need_coords": False,
                "coords_xy": None,
                "message": "Current step has no ground truth action text.",
            }

        selected_target = {
            "obj": None,
            "name": None,
            "seg_id": None,
            "click_point": None,
            "centroid_point": None,
        }
        try:
            current_options = _build_solve_options(self.env, self.planner, selected_target, self.env_id)
        except Exception as exc:
            return {
                "ok": False,
                "option_idx": None,
                "option_label": "",
                "option_action": "",
                "need_coords": False,
                "coords_xy": None,
                "message": f"Failed to build options: {exc}",
            }

        if not current_options:
            return {
                "ok": False,
                "option_idx": None,
                "option_label": "",
                "option_action": "",
                "need_coords": False,
                "coords_xy": None,
                "message": "No available options for current step.",
            }

        matched_label = map_action_text_to_option_label(target_action_text, current_options)
        if matched_label is None:
            return {
                "ok": False,
                "option_idx": None,
                "option_label": "",
                "option_action": "",
                "need_coords": False,
                "coords_xy": None,
                "message": f"Cannot map ground truth action '{target_action_text}' to option label.",
            }

        option_idx = find_exact_label_option_index(matched_label, current_options)
        if option_idx < 0:
            return {
                "ok": False,
                "option_idx": None,
                "option_label": "",
                "option_action": "",
                "need_coords": False,
                "coords_xy": None,
                "message": f"Mapped label '{matched_label}' not found in current options.",
            }

        option = current_options[option_idx]
        option_label = str(option.get("label", "")).strip()
        option_action = str(option.get("action", "")).strip()
        need_coords = bool(option.get("available"))

        if not need_coords:
            return {
                "ok": True,
                "option_idx": int(option_idx),
                "option_label": option_label,
                "option_action": option_action,
                "need_coords": False,
                "coords_xy": None,
                "message": "Ground truth action resolved.",
            }

        reference_position = _extract_choice_segment_position_xyz(
            getattr(self.env.unwrapped, "current_segment", None)
        )
        if reference_position is None:
            return {
                "ok": False,
                "option_idx": int(option_idx),
                "option_label": option_label,
                "option_action": option_action,
                "need_coords": True,
                "coords_xy": None,
                "message": "Cannot resolve reference target position from current segment.",
            }

        best_candidate = select_target_with_position(option.get("available"), reference_position)
        if best_candidate is None or best_candidate.get("obj") is None:
            return {
                "ok": False,
                "option_idx": int(option_idx),
                "option_label": option_label,
                "option_action": option_action,
                "need_coords": True,
                "coords_xy": None,
                "message": "Cannot match reference target to available candidates.",
            }

        actor = best_candidate.get("obj")
        segmentation_id_map = getattr(self.env.unwrapped, "segmentation_id_map", {}) or {}
        seg_id = _find_actor_segmentation_id(segmentation_id_map, actor)
        coords_xy = None
        if seg_id is not None:
            coords_xy = _compute_segmentation_centroid_xy(self.seg_raw, seg_id)

        if coords_xy is None:
            world_xyz = best_candidate.get("position")
            if world_xyz is None:
                world_xyz = extract_actor_position_xyz(actor)
            intrinsic, extrinsic, image_shape = self._get_front_camera_projection_params()
            if world_xyz is not None and intrinsic is not None and extrinsic is not None and image_shape is not None:
                coords_xy = project_world_to_pixel(
                    world_xyz=world_xyz,
                    intrinsic_cv=intrinsic,
                    extrinsic_cv=extrinsic,
                    image_shape=image_shape,
                )

        if coords_xy is None:
            return {
                "ok": False,
                "option_idx": int(option_idx),
                "option_label": option_label,
                "option_action": option_action,
                "need_coords": True,
                "coords_xy": None,
                "message": "Failed to compute pixel coordinates for reference target.",
            }

        coords_xy = [int(coords_xy[0]), int(coords_xy[1])]
        return {
            "ok": True,
            "option_idx": int(option_idx),
            "option_label": option_label,
            "option_action": option_action,
            "need_coords": True,
            "coords_xy": coords_xy,
            "message": f"Ground truth action resolved at ({coords_xy[0]}, {coords_xy[1]}).",
        }

    def execute_action(self, action_idx, click_coords):

# 用户点击EXECUTE
#   ↓
# execute_step() 调用 session.execute_action()
#   ↓
# execute_action() 执行 solve()
#   ↓ (在solve()执行过程中，step()可能检测到失败)
#   ↓
# evaluate(solve_complete_eval=True) 被调用
#   ↓
# BinFill.evaluate() 检查失败状态
#   - 保存 previous_failure
#   - 调用 sequential_task_check
#   - 如果 previous_failure=True 或 task_failed=True，设置 failureflag=True
#   ↓
# oracle_logic.py 获取 evaluation 结果
#   - 如果 is_fail=False，额外检查 failureflag 和 current_task_failure
#   - 设置 done = is_success or is_fail
#   ↓
# execute_step() 检查 done
#   - 如果 done=True，调用 complete_current_task()
#   ↓
# complete_current_task() 更新任务索引
#   - current_idx: 0 -> 1 (episode: 0 -> 1)


        """
        The real step logic.
        """
        if not self.env: return None, "No Env", False
        
        # 1. Re-create options with a persistent target dict that we can modify
        target_ref = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
        current_options = _build_solve_options(self.env, self.planner, target_ref, self.env_id)
        
        if action_idx < 0 or action_idx >= len(current_options):
             return self.get_pil_image(), "Invalid Action Index", False
             
        chosen_opt = current_options[action_idx]
        
        # 2. Resolve Target (Click -> Object)
        if click_coords:
             # Reuse logic from step() above, applying to target_ref
             cx, cy = click_coords
             h, w = self.seg_raw.shape[:2]
             cx = max(0, min(cx, w-1))
             cy = max(0, min(cy, h-1))
             
             seg_id_map = getattr(self.env.unwrapped, "segmentation_id_map", {}) or {}
             
             candidates = []
             def _collect(item):
                if isinstance(item, (list, tuple)):
                    for x in item: _collect(x)
                elif isinstance(item, dict):
                    for x in item.values(): _collect(x)
                else:
                    if item: candidates.append(item)
            
             avail = chosen_opt.get("available")
             if avail:
                 _collect(avail)
                 best_cand = None
                 min_dist = float('inf')
                 for actor in candidates:
                    target_ids = [sid for sid, obj in seg_id_map.items() if obj is actor]
                    for tid in target_ids:
                        tid = int(tid)
                        mask = (self.seg_raw == tid)
                        if np.any(mask):
                            ys, xs = np.nonzero(mask)
                            center_x, center_y = xs.mean(), ys.mean()
                            dist = (center_x - cx)**2 + (center_y - cy)**2
                            if dist < min_dist:
                                min_dist = dist
                                best_cand = {
                                    "obj": actor,
                                    "name": getattr(actor, "name", f"id_{tid}"),
                                    "seg_id": tid,
                                    "click_point": (int(cx), int(cy)),
                                    "centroid_point": (int(center_x), int(center_y))
                                }
                 if best_cand:
                    target_ref.update(best_cand)
                 else:
                    target_ref["click_point"] = (int(cx), int(cy))
             else:
                  target_ref["click_point"] = (int(cx), int(cy))

        # 3. Execute Solve
        # 异常处理流程：
        #   任何异常发生 (ScrewPlanFailure 或其他异常)
        #   ↓
        #   oracle_logic.py: 重新抛出异常
        #   ↓
        #   process_session.py: 捕获并传递到主进程
        #   ↓
        #   gradio_callbacks.py: 捕获并显示弹窗 (gr.Info)
        status_msg = f"Executing: {chosen_opt.get('label')}"
        before_elapsed_steps = getattr(self.env.unwrapped, "elapsed_steps", None)
        # Collect intermediate front-camera frames during solve() so livestream
        # can show the full execution process instead of only the final frame.
        original_step = self.env.step
        captured_front_frames = []
        stream_frame_callback = getattr(self, "stream_frame_callback", None)
        self._execute_streamed_frame_count = 0

        def _step_with_capture(action):
            step_output = original_step(action)
            step_front_frames = _collect_front_frames_from_step_output(step_output)
            if step_front_frames:
                prepared_frames = [
                    _prepare_frame(frame) for frame in step_front_frames if frame is not None
                ]
                if prepared_frames:
                    captured_front_frames.extend(prepared_frames)
                    if callable(stream_frame_callback):
                        try:
                            stream_frame_callback(prepared_frames)
                            self._execute_streamed_frame_count += len(prepared_frames)
                        except Exception:
                            # Keep solve path robust even if streaming callback fails.
                            pass
            return step_output

        self.env.step = _step_with_capture
        try:
            chosen_opt.get("solve")()
        except ScrewPlanFailure as e:
            # Re-raise ScrewPlanFailure so it can be handled in process_session and displayed as popup
            print(f"Screw Plan Failure")
            raise
        except Exception as e:
            # Re-raise all other exceptions so they can be displayed as popup too
            print(f"Execution Error")
            raise
        finally:
            self.env.step = original_step

        if captured_front_frames:
            self.base_frames.extend(captured_front_frames)
        print(f"[execute_action] captured_front_frames={len(captured_front_frames)}")
        after_elapsed_steps = getattr(self.env.unwrapped, "elapsed_steps", None)
        print(
            "[execute_action] elapsed_steps: "
            f"{before_elapsed_steps} -> {after_elapsed_steps}"
        )
            
        # 4. Evaluate
        self.env.unwrapped.evaluate()
        evaluation = self.env.unwrapped.evaluate(solve_complete_eval=True)
        
        is_success = _tensor_to_bool(evaluation.get("success", False))
        is_fail = _tensor_to_bool(evaluation.get("fail", False))
        
        # 如果evaluate()没有检测到失败，但环境已经设置了failureflag，则使用failureflag
        # 这是因为失败可能在solve()执行过程中的step()里被检测到，但evaluate()可能还没有反映
        failureflag = getattr(self.env.unwrapped, "failureflag", None)
        current_task_failure = getattr(self.env.unwrapped, "current_task_failure", False)
        
        if not is_fail:
            if failureflag is not None:
                failureflag_bool = _tensor_to_bool(failureflag)
                if failureflag_bool:
                    is_fail = True
            elif current_task_failure:
                is_fail = True
        
        if is_success: status_msg += " | SUCCESS"
        if is_fail: status_msg += " | FAILED"
        
        # 5. Update State for next step
        img, _ = self.update_observation()
        
        done = is_success or is_fail
        return img, status_msg, done
