import argparse
import h5py
import sys
import numpy as np

DEFAULT_PATH = "/data/hongzefu/data_0225/record_dataset_VideoUnmaskSwap.h5"

def _is_equal(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    try:
        return bool(np.all(a == b))
    except:
        return False

def inspect_actions(filepath, target_timestep=None, window=10):
    print(f"Inspecting HDF5 file: {filepath}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Sort episodes naturally (e.g., episode_0, episode_1, etc.)
            episodes = [k for k in f.keys() if k.startswith('episode_')]
            episodes.sort(key=lambda x: int(x.split('_')[1]))
            
            for ep_name in episodes:
                ep_group = f[ep_name]
                print(f"--- {ep_name} ---")
                
                # Sort timesteps naturally
                timesteps = [k for k in ep_group.keys() if k.startswith('timestep_')]
                timesteps.sort(key=lambda x: int(x.split('_')[1]))
                
                last_choice_action = None
                last_is_keyframe = None
                last_waypoint_action = None
                last_is_video_demo = None
                skip_count = 0
                
                for ts_name in timesteps:
                    ts_group = ep_group[ts_name]
                    
                    # Try to find the requested fields
                    choice_action = None
                    is_keyframe = None
                    is_video_demo = None
                    waypoint_action = None
                    
                    # Action group typically contains choice_action and waypoint_action
                    if 'action' in ts_group:
                        action_group = ts_group['action']
                        if 'choice_action' in action_group:
                            choice_action = np.array(action_group['choice_action'])
                        if 'waypoint_action' in action_group:
                            waypoint_action = np.array(action_group['waypoint_action'])
                            
                    # info group typically contains is_keyframe and is_video_demo
                    if 'info' in ts_group:
                        info_group = ts_group['info']
                        if 'is_keyframe' in info_group:
                            is_keyframe = np.array(info_group['is_keyframe'])
                        if 'is_video_demo' in info_group:
                            is_video_demo = np.array(info_group['is_video_demo'])
                            
                    # For safety, check if they are directly under timestep or elsewhere
                    if choice_action is None and 'choice_action' in ts_group:
                        choice_action = np.array(ts_group['choice_action'])
                    if is_keyframe is None and 'is_keyframe' in ts_group:
                        is_keyframe = np.array(ts_group['is_keyframe'])
                    if is_video_demo is None and 'is_video_demo' in ts_group:
                        is_video_demo = np.array(ts_group['is_video_demo'])
                    if waypoint_action is None and 'waypoint_action' in ts_group:
                        waypoint_action = np.array(ts_group['waypoint_action'])

                    should_skip = False
                    if last_choice_action is not None:
                        same_choice = _is_equal(choice_action, last_choice_action)
                        same_kf = _is_equal(is_keyframe, last_is_keyframe)
                        same_vd = _is_equal(is_video_demo, last_is_video_demo)
                        same_wp = _is_equal(waypoint_action, last_waypoint_action)
                        
                        should_skip = same_choice and same_kf and same_vd and same_wp
                        
                        if target_timestep is not None:
                            ts_idx = int(ts_name.split('_')[1])
                            if abs(ts_idx - target_timestep) <= window:
                                should_skip = False
                        
                    if should_skip:
                        skip_count += 1
                        continue

                    if skip_count > 0:
                        print(f"  ... ({skip_count} identical timesteps skipped) ...")
                        skip_count = 0

                    print(f"  {ts_name}:")
                    print(f"    choice_action:   {choice_action}")
                    print(f"    is_keyframe:     {is_keyframe}")
                    print(f"    is_video_demo:   {is_video_demo}")
                    print(f"    waypoint_action: {waypoint_action}")

                    last_choice_action = choice_action
                    last_is_keyframe = is_keyframe
                    last_is_video_demo = is_video_demo
                    last_waypoint_action = waypoint_action

                if skip_count > 0:
                    print(f"  ... ({skip_count} identical timesteps skipped) ...")
                    skip_count = 0

    except Exception as e:
        print(f"Error reading HDF5 file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect HDF5 dataset actions.")
    parser.add_argument("filepath", type=str, nargs="?", default=DEFAULT_PATH,
                        help="Path to the HDF5 file.")
    parser.add_argument("-t", "--timestep", type=int, default=168,
                        help="Specific timestep to not omit even if identical.")
    parser.add_argument("-w", "--window", type=int, default=10,
                        help="Window around specified timestep to not omit (default: 10).")
    args = parser.parse_args()

    inspect_actions(args.filepath, target_timestep=args.timestep, window=args.window)
