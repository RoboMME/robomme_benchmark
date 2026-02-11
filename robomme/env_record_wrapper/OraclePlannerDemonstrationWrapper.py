import gymnasium as gym
import numpy as np
import torch
import cv2
from rapidfuzz import process, fuzz

from robomme.robomme_env.util.vqa_options import get_vqa_options
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)
from . import planner_denseStep


# -----------------------------------------------------------------------------
# 本模块：Oracle 规划器演示包装器
# 在 Gym 环境中接入 Robomme 的 Oracle 规划逻辑，支持按步收集观测。
# 以下 Oracle 逻辑内联自 history_bench_sim.oracle_logic，与 planner_denseStep
# 配合，将规划器内部多次 env.step 聚合成统一批次返回。
# -----------------------------------------------------------------------------


def _find_best_semantic_match(user_query, options):
    """
    用 rapidfuzz 的字符编辑距离在 options 的 label 中找与 user_query 最匹配的一项；
    用于精确 action 名未命中时的语义回退。
    返回 (best_idx, best_score)，无匹配或异常时 best_idx=-1、score=0.0。
    """
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
        print(f"  [NLP] 编辑距离匹配失败 ({exc})，回退到选项 1。")
        return 0, 0.0
    print(f"  [NLP] 最近语义匹配（编辑距离）：'{query_text}' -> '{labels[best_idx]}' (得分: {best_score:.4f})")
    return best_idx, best_score


def step_after(env, planner, env_id, seg_raw, command_dict):
    """
    根据 command_dict（含 action 与可选 point）执行一次 Oracle 动作，
    返回统一 dense batch（obs/info 字典值为 list，reward/terminated/truncated 为 1D tensor）。
    """
    selected_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
    solve_options = get_vqa_options(env, planner, selected_target, env_id)
    target_action = command_dict.get("action")
    target_param = command_dict.get("point")
    # 无 action 则直接返回空批次
    if "action" not in command_dict:
        return planner_denseStep.empty_step_batch()
    if target_action is None:
        return planner_denseStep.empty_step_batch()
    found_idx = -1
    # 在解题选项中按 label 或序号查找动作索引
    for i, opt in enumerate(solve_options):
        if opt.get("label") == target_action or str(i + 1) == str(target_action):
            found_idx = i
            break
    # 未命中且为字符串且非数字时，尝试语义匹配
    if found_idx == -1 and isinstance(target_action, str) and not target_action.isdigit():
        print(f"正在对 '{target_action}' 尝试语义匹配…")
        found_idx, score = _find_best_semantic_match(target_action, solve_options)
    if found_idx == -1:
        print(f"错误：当前选项中未找到动作 '{target_action}'。")
        return planner_denseStep.empty_step_batch()
    # 若提供了点击坐标且存在分割图，则解析最近物体并填充 selected_target
    if target_param is not None and seg_raw is not None:
        cx, cy = target_param
        h, w = seg_raw.shape[:2]
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))
        seg_id_map = getattr(env.unwrapped, "segmentation_id_map", {}) or {}
        candidates = []

        def _collect(item):
            """递归收集 available 中的可操作对象。"""
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
            # 在分割图中找与点击点距离最近的物体
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
    print(f"执行选项：{found_idx + 1} - {solve_options[found_idx].get('label')}")

    # 用 dense 收集包装 solve()，收集中间所有 env.step 的结果
    result = planner_denseStep._run_with_dense_collection(
        planner,
        lambda: solve_options[found_idx].get("solve")()
    )

    if result == -1:
        print("警告：solve() 执行失败（返回 -1）")
        return planner_denseStep.empty_step_batch()

    env.unwrapped.evaluate()
    evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
    print(f"评估结果：{evaluation}")
    return result




class OraclePlannerDemonstrationWrapper(gym.Wrapper):
    """
    将带 Oracle 规划逻辑的 Robomme 环境包装成 Gym Wrapper，用于演示/评估；
    step 的输入为 command_dict（含 action 与可选 point），输出为统一 dense batch。
    """
    def __init__(self, env, env_id, gui_render=True):
        super().__init__(env)
        self.env_id = env_id
        self.gui_render = gui_render

        self.planner = None
        self.language_goal = None

        # 状态：分割图、帧缓存、当前可用选项
        self.seg_vis = None
        self.seg_raw = None
        self.base_frames = []
        self.wrist_frames = []
        self.available_options = []

        # 动作/观测空间（此处为空 Dict，由外部约定）
        self.action_space = gym.spaces.Dict({})
        self.observation_space = gym.spaces.Dict({})


    def reset(self, **kwargs):
        # 按 env_id 选择棍状或机械臂规划器并初始化
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

    def step(self, action):
        """
        执行一步：action 为 command_dict，需包含 "action"，可选 "point"。
        返回统一 dense batch，对应规划器执行过程中每一帧的数据。
        """
        command_dict = action

        # 直接获取当前观测与分割图（不单独 step_before）
        obs = self.env.unwrapped.get_obs(unflattened=True)
        seg = obs["sensor_data"]["base_camera"]["segmentation"]
        seg = seg.cpu().numpy() if hasattr(seg, "cpu") else np.asarray(seg)
        self.seg_raw = (seg[0] if seg.ndim > 2 else seg).squeeze().astype(np.int64)

        dummy_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
        raw_options = get_vqa_options(self.env, self.planner, dummy_target, self.env_id)
        self.available_options = [
            {"action": opt.get("label", "未知"), "need_parameter": bool(opt.get("available"))}
            for opt in raw_options
        ]

        # 调用 step_after 执行动作并得到统一批次
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = step_after(
            self.env, self.planner, self.env_id, self.seg_raw, command_dict
        )
        return obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch
