"""
核心理念：将“整个 Episode”视为一个原子操作。仅保留与环境交互部分，无 query model。

阶段 A 初始化 -> 阶段 B 步骤循环 (step_before -> command_dict -> step_after) -> 阶段 C 保存状态。
若发生异常：env.close(), del env, gc.collect()，记录 sim_error 后继续下一 episode；无 episode 重试。
"""
import os
import sys

# 添加项目根目录到 Python 路径，以便正确导入模块
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import json
import shutil
import torch
import gc
import re

from historybench.env_record_wrapper import EpisodeConfigResolver, EpisodeDatasetResolver
from pathlib import Path


# =============================================================================
# 辅助函数
# =============================================================================

def _tensor_to_bool(value):
    """将 tensor 或其他类型转换为布尔值"""
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)




def _frame_to_uint8_rgb(f):
    """Convert a single frame (tensor or ndarray) to uint8 RGB."""
    if hasattr(f, "cpu") and callable(getattr(f, "cpu", None)):
        arr = f.cpu().numpy()
    else:
        arr = np.asarray(f)
    arr = np.asarray(arr).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr


def save_frames_to_video(frames, output_path, fps=12):
    """Save a list of frames (numpy or tensor) to an MP4 file. Frames assumed RGB."""
    if not frames:
        return
    try:
        import imageio
        processed = [_frame_to_uint8_rgb(f) for f in frames]
        imageio.mimwrite(output_path, processed, fps=fps, quality=8, macro_block_size=None, codec="libx264")  # type: ignore[arg-type]
    except Exception as e:
        try:
            import cv2
            arr0 = _frame_to_uint8_rgb(frames[0])
            h, w = arr0.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
            out = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))
            if out.isOpened():
                for f in frames:
                    a = _frame_to_uint8_rgb(f)
                    out.write(cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
                out.release()
        except Exception as e2:
            print(f"Failed to save video to {output_path}: {e}, {e2}")


def get_num_episodes(dataset_root, env_id):
    metadata_path = dataset_root / f"record_dataset_{env_id}_metadata.json"
    if not metadata_path.exists():
        print(f"[{env_id}] Metadata {metadata_path} not found.")
        return 0
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            return payload.get("record_count", 0)
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return 0

def main():
    # Dataset Root
    dataset_root = Path("/data/hongzefu/dataset_generate")
    
    env_id_list = [
        # "PickXtimes",
        # "StopCube",
    #"SwingXtimes",
      #  "BinFill",

    #     "VideoUnmaskSwap",
    #     "VideoUnmask",
    #     "ButtonUnmaskSwap",
    #     "ButtonUnmask",

         "VideoRepick",
    #     "VideoPlaceButton",
    #      "VideoPlaceOrder",
    #     "PickHighlight",

    #     "InsertPeg",
    #     'MoveCube',
    #     "PatternLock",
    #     "RouteStick"
    ]

    for env_id in env_id_list:
        num_episodes = get_num_episodes(dataset_root, env_id)

        # Initialize Resolver per Env
        metadata_path = dataset_root / f"record_dataset_{env_id}_metadata.json"
        resolver = EpisodeConfigResolver(
            env_id=env_id,
            metadata_path=metadata_path,
            render_mode="human",
            gui_render=True,
            max_steps_without_demonstration=1000,
            action_space="oracle_planner"
        )

        #for episode in range(num_episodes):
        for episode in range(1):
            # if episode !=1:
            #     continue
            
            model_name = "env_only"
            save_dir = '/data/hongzefu/dataset_generate/videos'
            
            # 使用 EpisodeConfigResolver
            env, seed, difficulty = resolver.make_env_for_episode(episode)
            
            h5_path = dataset_root / f"record_dataset_{env_id}.h5"
            dataset_resolver = EpisodeDatasetResolver(env_id, episode, h5_path)

            obs_list, reward_list, terminated_list, truncated_list, info_list = env.reset()

              # ---------- 从每个 obs 读取 frame 等，构建列表 ----------
            frames = []
            wrist_frames = []
            actions = []
            states = []
            velocity = []
            for o in obs_list:
                if o:
                    frames.extend(o.get("frames", []))
                    wrist_frames.extend(o.get("wrist_frames", []))
                    actions.extend(o.get("actions", []))
                    states.extend(o.get("states", []))
                    velocity.extend(o.get("velocity", []))
            language_goal = obs_list[0].get("language_goal") if obs_list and obs_list[0] else None

            # ---------- 从每个 info 读取子目标等 ----------
            subgoal = []
            subgoal_grounded = []
            for i in info_list:
                if i:
                    subgoal.extend(i.get("subgoal", []))
                    subgoal_grounded.extend(i.get("subgoal_grounded", []))

            
            success = "fail"

            os.makedirs(save_dir, exist_ok=True)

            step_idx = 0
            max_query_times = 10

            # 阶段 B：执行步骤循环
            while True:
                if step_idx >= max_query_times:
                    print(f"Max query times ({max_query_times}) reached, stopping.")
                    break



                # 每次获得的 frames 单独保存为视频
                if frames:
                    video_path = os.path.join(save_dir, f"frames_step_{step_idx}.mp4")
                    save_frames_to_video(frames, video_path)
                if wrist_frames:
                    wrist_path = os.path.join(save_dir, f"wrist_frames_step_{step_idx}.mp4")
                    save_frames_to_video(wrist_frames, wrist_path)


                subgoal_text = dataset_resolver.get_grounded_subgoal(step_idx)
                if subgoal_text is None:
                    print(f"No more subgoals at step {step_idx}. Exiting loop.")
                    break
                
                point = None
                match = re.search(r'<(\d+),\s*(\d+)>', subgoal_text)
                if match:
                    point = [int(match.group(1)), int(match.group(2))]

                command_dict = {"action": subgoal_text, "point": point}

                if command_dict["point"] is not None:
                    command_dict["point"] = command_dict["point"][::-1]

                print(f"\nCommand: {command_dict}")

                step_idx += 1

                # 步骤 2：执行 step (相当于 step_after + step_before)
                obs_list, reward_list, terminated_list, truncated_list, info_list = env.step(command_dict)

                # 从每个 obs 读取 frame 等，构建列表
                frames = []
                wrist_frames = []
                actions = []
                states = []
                velocity = []
                for o in obs_list:
                    if o:
                        frames.extend(o.get('frames', []))
                        wrist_frames.extend(o.get('wrist_frames', []))
                        actions.extend(o.get('actions', []))
                        states.extend(o.get('states', []))
                        velocity.extend(o.get('velocity', []))
                language_goal = obs_list[0].get('language_goal') if obs_list and obs_list[0] else None

                # 从每个 info 读取
                subgoal = []
                subgoal_grounded = []
                for i in info_list:
                    if i:
                        subgoal.extend(i.get('subgoal', []))
                        subgoal_grounded.extend(i.get('subgoal_grounded', []))


                


                # 用最后一步的 terminated/truncated/info 做循环判断
                terminated = terminated_list[-1] if terminated_list else False
                truncated = truncated_list[-1] if truncated_list else False
                info = info_list[-1] if info_list else {}


                # 步数达到上限
                if truncated:
                    print(f"[{env_id}] episode {episode} 步数超限，步 {step}。")
                    break
                # 任务结束（成功或失败）
                if terminated.any():
                    if info.get("success") == torch.tensor([True]) or (
                        isinstance(info.get("success"), torch.Tensor) and info.get("success").item()
                    ):
                        print(f"[{env_id}] episode {episode} 成功。")
                    elif info.get("fail", False):
                        print(f"[{env_id}] episode {episode} 失败。")
                    break


            

            # 保存回放视频（frames + subgoal_grounded）
            out_video_path = os.path.join(dataset_root, "videos", f"replay_ee_{env_id}_ep{episode}.mp4")
            env.save_video(out_video_path)
            print(f"Saved video: {out_video_path}")



            # 阶段 C：成功标记与保存
            env.close()
            dataset_resolver.close()
                      
    # oracle_resolver.close() # No longer needed as we create per loop
    
if __name__ == "__main__":
    main()
