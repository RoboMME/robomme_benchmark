import numpy as np
import gymnasium as gym
import dataclasses
import collections
import time
from pathlib import Path
import shutil
import json

from openpi_client import websocket_client_policy as _websocket_client_policy
from history_bench_sim.utils import suppress_warnings, pack_buffer, check_args

from historybench.env_record_wrapper import DemonstrationWrapper
from historybench.HistoryBench_env import *

from history_bench_sim.utils import RolloutRecorder

TASK_NAME_LIST=  [      
    "BinFill",
    "StopCube",
    "PickXtimes",
    "SwingXtimes",
    
    "ButtonUnmask",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmaskSwap",
    
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick"
]

TASKS_WITH_STICK_GRIPPER = [
    "PatternLock",
    "RouteStick"
]

TASKS_WITH_DYNAMIC_CHANGE_DURING_EXEC = [
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
]

suppress_warnings()

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8011
    obs_horizon: int = 16
    max_steps: int = 1000
    render: bool = False

    save_path: str = "runs/new_evaluation_v2"
    policy_name: str = "historypi05_bench-bg512-input-static-drop"
        
    use_gemini: bool = False
    symbolic_memory: str | None = None # [simple_subgoal, grounded_subgoal, action_history]
    use_history: bool = True
    overwrite: bool = False
    
    model_seed: int = 42
    model_ckpt_id: int = 40000


def evaluate(args: Args):
    check_args(args)
    
    save_dir = Path(args.save_path) / args.policy_name / f"ckpt{args.model_ckpt_id}" / f"seed{args.model_seed}"
    if save_dir.exists():
        ans = input(f"{save_dir} already exists. continue (y/n): ")
        if ans.lower() != "y":
            raise ValueError(f"{save_dir} already exists. Aborting...")
    save_dir.mkdir(parents=True, exist_ok=True)
    video_save_dir = save_dir / "videos"
    
    log_dict = {}
    
    for task_name in TASK_NAME_LIST[:1]:
        log_dict[task_name] = {}
        for episode_id in range(3):
            try:
                env= gym.make(
                    task_name,
                    obs_mode="rgb",
                    control_mode="pd_joint_pos",
                    render_mode="human" if args.render else None,
                    reward_mode="dense",
                    HistoryBench_seed=episode_id,
                    max_episode_steps=99999,
                )
            except Exception as e:
                print(f"Error creating environment for task {task_name} episode {episode_id}: {e}")
                time.sleep(1)
                continue
            env = DemonstrationWrapper(env, max_steps_without_demonstration=args.max_steps, gui_render=args.render)
            env.reset()
            print(f"[historybench] env for task {task_name} episode {episode_id} setup finished")
            
            client = _websocket_client_policy.HistoryVLAWebsocketClientPolicy(args.host, args.port)
        
            success_flag = eval_one_episode(env, args, client, episode_id, task_name, save_dir, video_save_dir)
            log_dict[task_name][episode_id] = success_flag == "success"
            env.close()
            del env
    
    log_dict["success_rate"] = {task_name: sum(log_dict[task_name].values()) / len(log_dict[task_name].values()) for task_name in log_dict.keys()}
    log_dict["total_success_rate"] = sum(log_dict["success_rate"].values()) / len(log_dict["success_rate"].values())
    with open(save_dir / "log.json", "w") as f:
        json.dump(log_dict, f)


def pack_state(state):
    if len(state) == 8:
        return state
    else:
        return np.concatenate([state, np.array([0.0])], axis=0, dtype=np.float32)
    

def eval_one_episode(
    env, args, client,
    episode_id: int,  env_name: str,
    save_dir: str, video_save_dir: str,
):
    resp = client.reset()
    while resp.get("reset_finished", False) is False:
        time.sleep(0.1)
    
    action_plan = collections.deque() # for action execution
    
    image_buffer = []
    wrist_image_buffer = []
    state_buffer = []
    
    pre_traj = env.demonstration_data
    task_goal = pre_traj["language goal"]
    print(f"\ntask_goal: {task_goal}")
    
    recoder = RolloutRecorder(video_save_dir, task_goal, fps=30)
    
    assert args.obs_horizon in [8, 16]
    
    if args.obs_horizon == 8:
        final_indices = list(range(0, len(pre_traj["frames"]), 2)) + [len(pre_traj["frames"])-1]
    else:
        final_indices = list(range(len(pre_traj["frames"])))
    # print(f"final_indices: {final_indices}")
    
    for i in final_indices:
        image_buffer.append(pre_traj["frames"][i])
        wrist_image_buffer.append(pre_traj["wrist_frames"][i])
        state_buffer.append(pack_state(pre_traj["states"][i][0, :8]))
        recoder.record(
            image=pre_traj["frames"][i].copy(), 
            wrist_image=pre_traj["wrist_frames"][i].copy(), 
            state=pack_state(pre_traj["states"][i][0, :8]).copy(), 
            is_video=True
    )
        
    if args.symbolic_memory in ["simple_subgoal", "grounded_subgoal"]:
        if args.use_gemini:
            from history_bench_sim.chat_api.gemini import Gemini as RobotInstructionAgent
            gemini_save_ep_dir = save_dir / "conversation" / env_name
            gemini_save_ep_dir.mkdir(parents=True, exist_ok=True)
            system_prompt = RobotInstructionAgent.get_system_prompt(task_goal)
            agent = RobotInstructionAgent(system_prompt, save_dir=gemini_save_ep_dir, subgoal_type=args.symbolic_memory)
            print(f"[historybench] Gemini agent setup finished")
        else:
            print(f"[historybench] Use Oracle subgoal")
                
    exec_start_idx = len(image_buffer)-1
    print(f"exec_start_idx: {exec_start_idx}")

    img = image_buffer[-1]
    wrist_img = wrist_image_buffer[-1]
    state = state_buffer[-1]
    prompt = task_goal        
    success_flag = "unknown"
    
    count = 0
    while True:            
        if not action_plan:
            if args.use_history:                 
                resp = client.add_buffer(pack_buffer(image_buffer, state_buffer, exec_start_idx))
                while resp.get("add_buffer_finished", False) is False:
                    time.sleep(0.1)
            
            element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state,
                    "prompt": prompt,
                }
            if args.symbolic_memory in ["simple_subgoal", "grounded_subgoal"]:
                if args.use_gemini:
                    subgoal = agent.process(
                        image_buffer,
                        "What should the robot do next?"
                    ) # TODO: use gemini
                    element['simple_subgoal'] = subgoal
                    element['grounded_subgoal'] = subgoal
                else:
                    if args.symbolic_memory == "simple_subgoal":
                        subgoal = env.subgoal
                    else:
                        subgoal = env.subgoal_grounded
                    element['simple_subgoal'] = subgoal
                    element['grounded_subgoal'] = subgoal
            else:
                subgoal = None
                
            action_chunk = client.infer(element)["actions"]

            assert len(action_chunk) == args.obs_horizon // 8 * 2 + args.obs_horizon
            action_plan.extend(action_chunk[: args.obs_horizon])
            
            image_buffer.clear()
            wrist_image_buffer.clear()
            state_buffer.clear()
            exec_start_idx = 0 # reset as 0, so client will not update this value
        
        action = action_plan.popleft()
        if env_name in TASKS_WITH_STICK_GRIPPER:
            action = action[:7]
        obs, _, terminated, truncated, info = env.step(action)
        
        if args.obs_horizon == 8 and env_name in TASKS_WITH_DYNAMIC_CHANGE_DURING_EXEC:
            # run 2x steps to sync dynamic animation
            obs, _, terminated, truncated, info = env.step(action)
            
        count += 1
        
        if count > args.max_steps:
            success_flag = "timeout"
            break
            
        img = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
        wrist_img = obs['sensor_data']['hand_camera']['rgb'][0].cpu().numpy()
        state = env.agent.robot.qpos.cpu().numpy() if hasattr(env.agent.robot.qpos, 'cpu') else env.agent.robot.qpos
        state = pack_state(state[0, :8]) # only use the first 8 dimensions
        
            
        if len(action) == 8:
            eef_vel = np.asarray(env.agent.robot.links[9].get_linear_velocity()[0].tolist() + [action[-1]], dtype=np.float32)
        else: # for no gripper task, such as PatternLock, RouteStick
            eef_vel = np.asarray(env.agent.robot.links[9].get_linear_velocity()[0].tolist() + [-1.0], dtype=np.float32)
        resp = client.add_action_history({"add_action_history": True, "data": eef_vel})
        while resp.get("add_action_history_finished", False) is False:
            time.sleep(0.01)

        image_buffer.append(img.copy())
        wrist_image_buffer.append(wrist_img.copy())
        state_buffer.append(state.copy())
        recoder.record(image=img.copy(), wrist_image=wrist_img.copy(), state=state.copy(), action=action.copy(), subgoal=subgoal)
        
        if args.render:
            env.render()

        if truncated:
            print("time limit!")
            success_flag = "timeout"
            break
        elif terminated:
            if info.get("success", False):
                success_flag = "success"
            if info.get("fail", False):
                success_flag = "fail"
            break    
    
    recoder.save_video(f"{env_name}_ep{episode_id}_{success_flag}_{task_goal}.mp4")    
    if args.use_gemini:
        agent.save_conversation(gemini_save_ep_dir / f"ep{episode_id}_{success_flag}_{task_goal}.json")
        del agent
    
    return success_flag

if __name__ == "__main__":
    import tyro
    tyro.cli(evaluate)
    
