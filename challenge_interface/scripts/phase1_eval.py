"""
Sample script to evaluate participant's remote policy for the challenge.

This is used by RoboMME challenge organizers to evaluate the policy for Phase 1.

"""
import collections
import hashlib
import json
import os
import time
import imageio
import cv2
import numpy as np
import argparse
from robomme.env_record_wrapper import BenchmarkEnvBuilder
from challenge_interface.client import PolicyClient
from challenge_interface.client_http import PolicyHTTPClient




# Participant parameters (you will need to submit this parameters at eval.ai)
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a policy for the CVPR challenge.")
    parser.add_argument(
        "--transport",
        type=str,
        choices=("websocket", "http"),
        default="websocket",
        help="Transport used by policy server (default: %(default)s).",
    )
    parser.add_argument("--action_space", type=str, default="joint_angle", help="Action space to use.")
    parser.add_argument(
        "--use_depth",
        action="store_true",
        help="Whether to use depth images.",
    )
    parser.add_argument(
        "--use_camera_params",
        action="store_true",
        help="Whether to use camera parameters.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host/IP to connect to the policy server.")
    parser.add_argument("--port", type=int, default=8001, help="Port to connect to the policy server.")
    parser.add_argument("--team_id", type=str, default="team_0000", help="Team ID.")
    parser.add_argument("--max_steps", type=int, default=1500, help="Maximum number of steps per episode. We set 1500 for RoboMME Challenge.")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate. We will use 10 for RoboMME Challenge Phase 1 evaluation")
    return parser.parse_args()


VALID_ACTION_SPACES = ("joint_angle", "ee_pose", "waypoint")
EXPECTED_ACTION_SHAPES = {
    "joint_angle": (8,),
    "ee_pose": (7,),
    "waypoint": (7,),
}

def _is_success(outcome: object) -> bool:
    """
    Best-effort success detection across possible status strings.
    """
    if outcome is None:
        return False
    s = str(outcome).strip().lower()
    return s == "success" or ("success" in s and "fail" not in s)


def _config_fingerprint(config: dict) -> str:
    normalized = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_json_atomic(path: str, obj: dict) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _build_inputs(obs, info, use_camera_params):
    """Build the observation buffer sent to the remote policy server."""
    buffer = {
        "task_goal": info["task_goal"],
        "is_first_step": True,
    }
    for key in obs:
        buffer[key] = obs[key]
        
    if use_camera_params:
        buffer["front_camera_intrinsic"] = info["front_camera_intrinsic"]
        buffer["wrist_camera_intrinsic"] = info["wrist_camera_intrinsic"]
    return buffer


def _update_inputs(buffer, obs):
    """Append new observation data into the buffer."""
    for key in obs:
        buffer[key].extend(obs[key])


def _clear_inputs(buffer, obs):
    """Clear observation arrays in the buffer (keep task_goal, is_first_step)."""
    buffer["is_first_step"] = False
    for key in obs:
        buffer[key].clear()


def run_episode(
    client,
    env_builder,
    episode_idx,
    env_id,
    *,
    use_depth: bool,
    use_camera_params: bool,
    action_space: str,
):
    """Run one episode: reset env, stream obs to policy, step until done."""
    
    resp = client.reset()
    while not resp.get("reset_finished", False):
        time.sleep(0.1)
    print(f"Reset finished for policy server, env id: {env_id}, episode idx: {episode_idx}")
    
    env = env_builder.make_env_for_episode(
        episode_idx=episode_idx,
        include_front_depth=use_depth,
        include_wrist_depth=use_depth,
        include_front_camera_extrinsic=use_camera_params,
        include_wrist_camera_extrinsic=use_camera_params,
        include_front_camera_intrinsic=use_camera_params,
        include_wrist_camera_intrinsic=use_camera_params,
    )
    action_plan = collections.deque()
    obs, info = env.reset()
    inputs = _build_inputs(obs, info, use_camera_params)
    expected_shape = EXPECTED_ACTION_SHAPES[action_space]
    
    video_frames = []
    exec_start_idx = len(obs["front_rgb_list"]) - 1
    
    for i in range(len(obs["front_rgb_list"])):
        video_frames.append(np.hstack([obs["front_rgb_list"][i], obs["wrist_rgb_list"][i]]))
        if i < exec_start_idx: # add red border to indicate the conditioned video frames
            video_frames[-1] = cv2.rectangle(video_frames[-1], (0, 0), (video_frames[-1].shape[1], video_frames[-1].shape[0]), (255, 0, 0), 10)

    while True:
        if not action_plan:
            outputs = client.infer(inputs)
            action_chunk = outputs["actions"]
            action_plan.extend(action_chunk)
            _clear_inputs(inputs, obs)

        action = action_plan.popleft()
        assert action.shape == expected_shape, f"Expected {expected_shape}, got {action.shape}"

        obs, _, terminated, truncated, info = env.step(action)
        video_frames.append(np.hstack([obs["front_rgb_list"][-1], obs["wrist_rgb_list"][-1]]))
        _update_inputs(inputs, obs)

        if terminated or truncated:
            break

    outcome = info.get("status", "unknown")
    env.close()
    del env
    return outcome, video_frames, info["task_goal"][0]


def main() -> None:
    args = parse_args()
    assert args.action_space in VALID_ACTION_SPACES, (
        f"ACTION_SPACE must be one of {VALID_ACTION_SPACES}"
    )
    if args.transport == "http":
        client = PolicyHTTPClient(host=args.host, port=args.port)
    else:
        client = PolicyClient(host=args.host, port=args.port)

    output_dir = f"challenge_results/{args.team_id}"
    os.makedirs(output_dir, exist_ok=True)

    video_output_dir = os.path.join(output_dir, "videos")
    os.makedirs(video_output_dir, exist_ok=True)

    progress_path = os.path.join(output_dir, "progress.json")
    metrics_path = os.path.join(output_dir, "metrics.json")

    config = {
        "transport": args.transport,
        "action_space": args.action_space,
        "use_depth": bool(args.use_depth),
        "use_camera_params": bool(args.use_camera_params),
        "num_episodes": int(args.num_episodes),
        "max_steps": int(args.max_steps),
        "host": args.host,
        "port": int(args.port),
        "dataset": "heldout",
    }
    config_fp = _config_fingerprint(config)

    progress = _load_json(progress_path)
    if progress is None or progress.get("config_fingerprint") != config_fp:
        progress = {
            "team_id": args.team_id,
            "config": config,
            "config_fingerprint": config_fp,
            "completed": {},  # {task_id: {episode_idx(str): {"outcome": str, "video_path": str}}}
            "current": {"task_id": None, "episode_idx": None, "outcome": None},
            "finished": False,
            "metrics": None,
            "updated_at": time.time(),
        }
        _save_json_atomic(progress_path, progress)

    task_list = BenchmarkEnvBuilder.get_task_list()
    for env_id in task_list:
        env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="heldout",
            action_space=args.action_space,
            max_steps=args.max_steps,
        )

        for episode_idx in range(args.num_episodes):
            completed_for_task = progress["completed"].setdefault(env_id, {})
            ep_key = str(episode_idx)
            if ep_key in completed_for_task:
                progress["current"] = {
                    "task_id": env_id,
                    "episode_idx": episode_idx,
                    "outcome": completed_for_task[ep_key].get("outcome", None),
                }
                progress["updated_at"] = time.time()
                _save_json_atomic(progress_path, progress)
                print(f"[SKIP] {env_id} episode {episode_idx} already completed.")
                continue

            progress["current"] = {"task_id": env_id, "episode_idx": episode_idx, "outcome": None}
            progress["updated_at"] = time.time()
            _save_json_atomic(progress_path, progress)

            outcome, video_frames, task_goal = run_episode(
                client,
                env_builder,
                episode_idx,
                env_id,
                use_depth=args.use_depth,
                use_camera_params=args.use_camera_params,
                action_space=args.action_space,
            )

            video_path = os.path.join(
                video_output_dir,
                f"{env_id}_ep_{episode_idx}_{outcome}_{task_goal}.mp4",
            )
            imageio.mimsave(video_path, video_frames, fps=30)
            print(f"Outcome: {outcome} (task={env_id}, episode={episode_idx})")

            completed_for_task[ep_key] = {"outcome": outcome, "video_path": video_path}
            progress["current"]["outcome"] = outcome
            progress["updated_at"] = time.time()
            _save_json_atomic(progress_path, progress)

    # When all tasks/episodes are done, write metrics.
    all_done = True
    for env_id in task_list:
        completed_for_task = progress.get("completed", {}).get(env_id, {})
        for episode_idx in range(args.num_episodes):
            if str(episode_idx) not in completed_for_task:
                all_done = False
                break
        if not all_done:
            break

    if all_done:
        per_task_metrics = {}
        total_success = 0
        total_episodes = len(task_list) * args.num_episodes

        for env_id in task_list:
            completed_for_task = progress["completed"].get(env_id, {})
            success_count = 0
            for episode_idx in range(args.num_episodes):
                outcome = completed_for_task[str(episode_idx)].get("outcome", None)
                if _is_success(outcome):
                    success_count += 1
            avg_success = success_count / max(1, args.num_episodes)
            per_task_metrics[env_id] = {
                "avg_success": avg_success,
                "success_count": success_count,
                "num_episodes": args.num_episodes,
            }
            total_success += success_count

        overall_avg_success = total_success / max(1, total_episodes)
        metrics = {
            "team_id": args.team_id,
            "config": config,
            "per_task": per_task_metrics,
            "overall": {
                "avg_success": overall_avg_success,
                "total_success": total_success,
                "total_episodes": total_episodes,
            },
            "updated_at": time.time(),
        }

        _save_json_atomic(metrics_path, metrics)
        progress["finished"] = True
        progress["metrics"] = metrics
        progress["updated_at"] = time.time()
        _save_json_atomic(progress_path, progress)
    else:
        print("Evaluation not finished yet; metrics.json will be written once all tasks/episodes are complete.")


if __name__ == "__main__":
    main()
