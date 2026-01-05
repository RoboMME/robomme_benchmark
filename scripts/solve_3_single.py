import os


import sys
import numpy as np
import sapien
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym
from gymnasium.utils.save_video import save_video

from historybench.env_record_wrapper import HistoryBenchRecordWrapper
from historybench.HistoryBench_env import *


from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)

# from util import *
import torch

from planner_fail_safe import (
    FailAwarePandaArmMotionPlanningSolver,
    FailAwarePandaStickMotionPlanningSolver,
    ScrewPlanFailure,
)

OUTPUT_ROOT = Path(__file__).resolve().parents[1]



def main():
    """
    Main function to run the simulation and record data for multiple seeds.
    """

    num_episodes = 1
    env_id_list=["VideoRepick"]
    for env_id in env_id_list:
        dataset_path = Path(f"/data/hongzefu/dataset_generate/record_dataset_{env_id}.h5")
        for episode in range(num_episodes):
            seed=episode
            print(f"--- Running simulation for episode:{episode},env: {env_id} ---")

            # Initialize the environment with the specified seed for recording
            env_kwargs = dict(
                obs_mode="rgb+depth+segmentation",
                control_mode="pd_joint_pos",
                render_mode="human",  
                reward_mode="dense",
                HistoryBench_seed=33000,
                max_episode_steps=200,
                HistoryBench_difficulty="hard",
            )

            env_kwargs["historybench_failure_recovery"] = False
            #env_kwargs["historybench_failure_recovery_mode"] = "xy"


            env = gym.make(env_id, **env_kwargs)

            
            env = HistoryBenchRecordWrapper(env,HistoryBench_dataset=str(dataset_path),HistoryBench_env=env_id,HistoryBench_episode=episode,HistoryBench_seed=seed,
                                            save_video=True)
            env.reset()

            if env_id=="PatternLock" or env_id == "RouteStick":
                planner = FailAwarePandaStickMotionPlanningSolver(
                        env,
                        debug=False,
                        vis=True,
                        base_pose=env.unwrapped.agent.robot.pose,
                        visualize_target_grasp_pose=False,
                        print_env_info=False,
                        joint_vel_limits=0.3,
                    )
            else:
                planner = FailAwarePandaArmMotionPlanningSolver(
                        env,
                        debug=False,
                        vis=True,
                        base_pose=env.unwrapped.agent.robot.pose,
                        visualize_target_grasp_pose=False,
                        print_env_info=False,
                    )
                
                
            # 预先进行一次评估以初始化任务列表
            env.unwrapped.evaluate()
            tasks = list(getattr(env.unwrapped, "task_list", []) or [])

            if not tasks:
                print("No tasks defined for this environment; skipping execution")
            else:
                print(f"{env_id}: Task list has {len(tasks)} tasks")

                def _tensor_to_bool(value):
                    if value is None:
                        return False
                    if isinstance(value, torch.Tensor):
                        return bool(value.detach().cpu().bool().item())
                    if isinstance(value, np.ndarray):
                        return bool(np.any(value))
                    return bool(value)

                for idx, task_entry in enumerate(tasks):
                    task_name = task_entry.get("name", f"Task {idx}")
                    print(f"Executing task {idx+1}/{len(tasks)}: {task_name}")

                    solve_callable = task_entry.get("solve")
                    if not callable(solve_callable):
                        raise ValueError(f"Task '{task_name}' must supply a callable 'solve'.")


                    evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
                    screw_failed = False
                    try:
                        solve_callable(env, planner)
                    except ScrewPlanFailure as exc:
                        screw_failed = True
                        print(f"Screw plan failure during '{task_name}': {exc}")
                        env.unwrapped.failureflag = torch.tensor([True])
                        env.unwrapped.successflag = torch.tensor([False])
                        env.unwrapped.current_task_failure = True
                    evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
                    fail_flag = evaluation.get("fail", False)
                    success_flag = evaluation.get("success", False)
                    # 也检查 current_task_failure 标志
                    current_task_failure = getattr(env.unwrapped, 'current_task_failure', False)

                    print(evaluation)

                    if screw_failed or _tensor_to_bool(fail_flag) or current_task_failure:
                        print("Encountered failure condition; stopping task sequence.")
                        break

                    if _tensor_to_bool(success_flag):
                        print("All tasks completed successfully.")
                        break

                # 任务循环结束后再进行一次评估以刷新状态
                env.unwrapped.evaluate()


    
            env.close()

            print(f"--- Finished Running simulation for episode:{episode},env: {env_id} ---")


if __name__ == "__main__":
    main()
