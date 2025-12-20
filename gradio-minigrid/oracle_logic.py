import os
import sys
import numpy as np
import gymnasium as gym
import cv2
import colorsys
import json
import torch
from pathlib import Path
from PIL import Image

# --- Setup Paths ---
# Ensure we can import historybench and mani_skill from parent directory
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
from historybench.env_record_wrapper import EpisodeConfigResolver
from historybench.HistoryBench_env.util.vqa_options import get_vqa_options
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.motionplanner_stick import PandaStickMotionPlanningSolver

# --- Constants ---
DEFAULT_DATASET_ROOT = Path(parent_dir) / "dataset_json"

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
                    For Gradio, we usually want False (rgb_array) unless debugging locally.
        """
        self.dataset_root = Path(dataset_root)
        self.gui_render = gui_render # Usually False for web app
        self.render_mode = "human" if gui_render else "rgb_array"
        
        self.env = None
        self.planner = None
        self.color_map = None
        self.env_id = None
        self.episode_idx = None
        self.language_goal = ""
        self.difficulty = None
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

    def load_episode(self, env_id, episode_idx):
        """Initialize environment for a specific episode."""
        if self.env:
            self.env.close()
            
        metadata_path = self.dataset_root / f"record_dataset_{env_id}_metadata.json"
        if not metadata_path.exists():
            return None, f"Dataset metadata not found for {env_id}"

        resolver = EpisodeConfigResolver(
            env_id=env_id,
            dataset=None,
            metadata_path=metadata_path,
            render_mode=self.render_mode,
            gui_render=self.gui_render,
            max_steps_without_demonstration=3000,
        )

        try:
            self.env, episode_dataset, seed, difficulty = resolver.make_env_for_episode(episode_idx)
            self.env.reset()
            self.env_id = env_id
            self.episode_idx = episode_idx
            self.difficulty = difficulty
            
            # Demonstration data
            demonstration_data = getattr(self.env, "demonstration_data", {}) or {}
            self.language_goal = demonstration_data.get('language goal', 'Unknown Goal')
            self.demonstration_frames = demonstration_data.get('frames', [])
            
            # Setup Color Map
            self.color_map = _generate_color_map()
            _sync_table_color(self.env, self.color_map)
            
            # Initialize Planner
            if env_id in ("PatternLock", "RouteStick"):
                self.planner = PandaStickMotionPlanningSolver(
                    self.env, debug=False, vis=self.gui_render,
                    base_pose=self.env.unwrapped.agent.robot.pose,
                    visualize_target_grasp_pose=False, print_env_info=False,
                    joint_vel_limits=0.3,
                )
            else:
                self.planner = PandaArmMotionPlanningSolver(
                    self.env, debug=False, vis=self.gui_render,
                    base_pose=self.env.unwrapped.agent.robot.pose,
                    visualize_target_grasp_pose=False, print_env_info=False,
                )
            
            self.env.unwrapped.evaluate() # Initial eval check
            
            # Reset logs
            self.history = []
            
            # Reset frame indices
            self.last_base_frame_idx = 0
            self.last_wrist_frame_idx = 0
            
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

        # 1. Capture Frames
        self.base_frames = getattr(self.env, "frames", []) or []
        self.wrist_frames = getattr(self.env, "wrist_frames", []) or []
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
             _, self.seg_raw = (_prepare_segmentation_visual(seg_data, self.color_map, seg_hw) if seg_data is not None else (None, None))
             if self.base_frames:
                vis_frame = _prepare_frame(self.base_frames[-1])
                vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR) # Keep consistent BGR internally
                if vis_frame.shape[:2] != seg_hw:
                    vis_frame = cv2.resize(vis_frame, (seg_hw[1], seg_hw[0]), interpolation=cv2.INTER_LINEAR)
                self.seg_vis = vis_frame
             else:
                 self.seg_vis = np.zeros((seg_hw[0], seg_hw[1], 3), dtype=np.uint8)

        # 4. Generate Options
        dummy_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
        self.raw_solve_options = _build_solve_options(self.env, self.planner, dummy_target, self.env_id)
        
        # Format for UI
        self.available_options = []
        for i, opt in enumerate(self.raw_solve_options):
            label = opt.get("label", f"Option {i+1}")
            self.available_options.append((label, i)) # Tuple for Gradio Radio/Dropdown

        return self.get_pil_image(), "Ready"

    def step(self, action_idx, click_coords=None):
        """
        Executes a step based on user selection.
        action_idx: Index of selected option (int)
        click_coords: (x, y) tuple or None. Coordinates are relative to the image size.
        """
        if not self.env:
            return None, "No active session", False

        # 1. Identify Target Action
        if action_idx is None or action_idx < 0 or action_idx >= len(self.raw_solve_options):
            return self.get_pil_image(), "Invalid action selection", False

        selected_option = self.raw_solve_options[action_idx]
        target_action = selected_option.get("label")
        
        # 2. Logic to Find Target Object (Replicating _prompt_next_task_gui logic)
        # We need to simulate the "selection" process
        
        # Initialize simulation state
        selected_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
        
        # Logic to "click" and select object if coordinates provided
        if click_coords:
            cx, cy = click_coords
            # Note: _handle_logic_click in script relies on `state` dictionary and modifying `selected_target`
            # We recreate a simplified version here
            
            # Assuming seg_vis/seg_raw are current
            h, w = self.seg_raw.shape[:2]
            # Ensure coords in bounds
            cx = max(0, min(cx, w-1))
            cy = max(0, min(cy, h-1))
            
            seg_id_map = getattr(self.env.unwrapped, "segmentation_id_map", {}) or {}
            
            # Find all available targets for this option
            # Helper to collect candidates from option
            candidates = []
            def _collect(item):
                if isinstance(item, (list, tuple)):
                    for x in item: _collect(x)
                elif isinstance(item, dict):
                    for x in item.values(): _collect(x)
                else:
                    if item: candidates.append(item)
            
            avail = selected_option.get("available")
            _collect(avail)
            
            # Find best candidate near click
            best_cand = None
            min_dist = float('inf')
            
            for actor in candidates:
                # Find actor ID in seg map
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
                selected_target.update(best_cand)
            elif click_coords:
                 # Fallback: Just click point
                 selected_target["click_point"] = (int(cx), int(cy))

        # 3. Construct Simulated Inputs
        # Format: [Option_Idx+1, Coords, Option_Idx+1] (Logic from step_after)
        # Script logic: 
        # simulated_inputs.append(str(found_idx + 1))
        # if target_param is not None: simulated_inputs.append(f"{target_param[0]} {target_param[1]}")
        # else: if solve_options[found_idx].get("available"): simulated_inputs.append("0 0")
        # simulated_inputs.append(str(found_idx + 1))
        
        # Actually, we can bypass the string parsing and text interface of `_prompt_next_task_gui`
        # if we just invoke the solver directly, BUT `_prompt_next_task_gui` does important target resolution.
        # Ideally, we should reuse the logic or replicate it. 
        # For robustness, we will perform the ACTION directly if we have resolved the target.
        
        # However, the `solve()` function of the option might depend on `selected_target` being set correctly in a shared scope?
        # The script passes `selected_target` dict into `_build_solve_options`. 
        # Wait, `_build_solve_options` is called BEFORE the step. 
        # The closures inside `solve_options` capture `selected_target`? 
        # Let's check `get_vqa_options` in `historybench`.
        # Usually these closures refer to the mutable `selected_target` dictionary passed in.
        
        # So:
        # 1. We have `self.raw_solve_options` which were built with a `dummy_target` in `update_observation`.
        #    BUT that `dummy_target` is now stale/disconnected if we rebuilt options.
        #    Actually, we need to rebuild options with the *current* `selected_target` object 
        #    before calling solve, OR rely on the fact that we modify the dictionary that was used to create them.
        
        # In `update_observation`, we created `dummy_target` and passed it to `_build_solve_options`.
        # We need to preserve this dictionary instance so that when we modify it now, the `solve` lambda sees the change.
        # But we threw it away in `update_observation`.
        # CORRECTION: We should store `self.current_target_ref` in `update_observation`.
        
        # Re-run `_build_solve_options` to be safe, or store the ref.
        # Storing ref is better.
        
        pass 

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
        status_msg = f"Executing: {chosen_opt.get('label')}"
        try:
            chosen_opt.get("solve")()
        except Exception as e:
            print(f"Execution Error: {e}")
            return self.get_pil_image(), f"Error: {e}", False

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
