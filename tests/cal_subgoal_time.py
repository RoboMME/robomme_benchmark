
import h5py
import os
import numpy as np
from collections import defaultdict

# Directories to search (each is treated as a separate dataset)
DIRS = [
    '/data/hongzefu/data_1206',
    '/data/hongzefu/dataset_generate-b4'
]

# EnvIDs to specifically analyze (leave empty to analyze all)
TARGET_ENVIDS = [
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "BinFill",
    "VideoUnmaskSwap",
    "VideoUnmask",
    "ButtonUnmaskSwap",
    "ButtonUnmask",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "PickHighlight",
    "InsertPeg",
    "MoveCube",
    "PatternLock",
    "RouteStick",
]


def get_dataset_name(directory):
    """Get a short display name for a dataset directory."""
    return os.path.basename(directory)


def get_h5_files_by_dir(directories):
    """Return a dict: { dir_path: [h5_file_paths] }"""
    result = {}
    for d in directories:
        if not os.path.exists(d):
            continue
        files = []
        for f in os.listdir(d):
            if f.endswith('.h5') and f.startswith('record_dataset_'):
                files.append(os.path.join(d, f))
        result[d] = sorted(files)
    return result


def extract_episode_data(file_path, env_id):
    """
    Extract episode data from an h5 file.
    Returns a dict: { episode_num: { 'seed': ..., 'difficulty': ..., 'total_timesteps': ..., 'subgoals': [...] } }
    """
    results = {}

    try:
        with h5py.File(file_path, 'r') as f:
            # Locate where episodes are
            episodes_group = None
            env_group_name = f"env_{env_id}"

            if env_group_name in f:
                episodes_group = f[env_group_name]
            else:
                if any(k.startswith('episode_') for k in f.keys()):
                    episodes_group = f

            if episodes_group is None:
                return results

            # Get all episode keys
            episode_keys = [k for k in episodes_group.keys() if k.startswith('episode_')]
            if not episode_keys:
                return results

            # Sort episodes numerically
            episode_keys.sort(key=lambda x: int(x.split('_')[-1]))
            episode_keys = episode_keys[5:20]  # episode 6-10

            for ep_key in episode_keys:
                ep_num = int(ep_key.split('_')[-1])
                parent_group = episodes_group[ep_key]

                # Extract seed and difficulty from setup group
                seed = "N/A"
                difficulty = "N/A"
                if 'setup' in parent_group:
                    setup = parent_group['setup']
                    if 'seed' in setup:
                        val = setup['seed'][()]
                        seed = val.decode('utf-8') if isinstance(val, bytes) else str(val)
                    if 'difficulty' in setup:
                        val = setup['difficulty'][()]
                        difficulty = val.decode('utf-8') if isinstance(val, bytes) else str(val)

                # Determine timestep keys
                all_keys = list(parent_group.keys())
                timestep_keys = [k for k in all_keys if k.startswith('record_timestep_')]
                if not timestep_keys:
                    timestep_keys = [k for k in all_keys if k.startswith('timestep_')]

                if not timestep_keys:
                    results[ep_num] = {
                        'seed': seed,
                        'difficulty': difficulty,
                        'total_timesteps': 0,
                        'subgoals': []
                    }
                    continue

                timestep_keys.sort(key=lambda x: int(x.split('_')[-1]))
                total_timesteps = len(timestep_keys)

                subgoals = []
                for k in timestep_keys:
                    ts_group = parent_group[k]
                    sg = None

                    if 'info' in ts_group:
                        info = ts_group['info']
                        if 'simple_subgoal' in info:
                            sg = info['simple_subgoal'][()]
                        elif 'subgoal' in info:
                            sg = info['subgoal'][()]

                    if sg is None and 'simple_subgoal' in ts_group:
                        sg = ts_group['simple_subgoal'][()]

                    if sg is None:
                        sg = b"None"

                    subgoals.append(sg)

                # Decode subgoals
                decoded_subgoals = []
                for sg in subgoals:
                    if isinstance(sg, bytes):
                        decoded_subgoals.append(sg.decode('utf-8'))
                    else:
                        decoded_subgoals.append(str(sg))

                # Consolidate consecutive identical subgoals
                consolidated = []
                if decoded_subgoals:
                    current_sg = decoded_subgoals[0]
                    count = 0
                    for sg in decoded_subgoals:
                        if sg == current_sg:
                            count += 1
                        else:
                            consolidated.append((current_sg, count))
                            current_sg = sg
                            count = 1
                    consolidated.append((current_sg, count))

                results[ep_num] = {
                    'seed': seed,
                    'difficulty': difficulty,
                    'total_timesteps': total_timesteps,
                    'subgoals': consolidated
                }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()

    return results


def main():
    # Structure: { env_id: { dataset_name: { ep_num: episode_data } } }
    all_data = defaultdict(dict)

    files_by_dir = get_h5_files_by_dir(DIRS)

    for dir_path, h5_files in files_by_dir.items():
        dataset_name = get_dataset_name(dir_path)
        for file_path in h5_files:
            filename = os.path.basename(file_path)
            env_id = filename.replace('record_dataset_', '').replace('.h5', '')

            if TARGET_ENVIDS and env_id not in TARGET_ENVIDS:
                continue

            episode_data = extract_episode_data(file_path, env_id)
            if episode_data:
                all_data[env_id][dataset_name] = episode_data

    if not all_data:
        print("No matching data found.")
        return

    # Print interleaved: for each env_id, for each episode, alternate datasets
    dataset_names = [get_dataset_name(d) for d in DIRS if d in files_by_dir]

    # Build output lines (all results) and different lines (timesteps mismatch)
    lines = []
    diff_lines = []

    for env_id in sorted(all_data.keys()):
        env_header = [
            "=" * 70,
            f"EnvID: {env_id}",
            "=" * 70,
        ]
        lines.extend(env_header)

        datasets_for_env = all_data[env_id]

        # Collect all episode numbers across all datasets for this env
        all_ep_nums = set()
        for ds_name, ep_dict in datasets_for_env.items():
            all_ep_nums.update(ep_dict.keys())

        env_has_diff = False
        env_diff_episode_lines = []

        for ep_num in sorted(all_ep_nums):
            ep_lines = []
            ep_lines.append(f"\n  episode_{ep_num}:")
            ep_lines.append(f"  {'-' * 60}")

            # Collect total_timesteps from each dataset for this episode
            timesteps_per_ds = {}
            for ds_name in dataset_names:
                if ds_name not in datasets_for_env:
                    continue
                ep_dict = datasets_for_env[ds_name]
                if ep_num not in ep_dict:
                    continue

                ep = ep_dict[ep_num]
                timesteps_per_ds[ds_name] = ep['total_timesteps']
                ep_lines.append(f"    [{ds_name}]  seed={ep['seed']}, difficulty={ep['difficulty']}, total_timesteps={ep['total_timesteps']}")
                for sg, duration in ep['subgoals']:
                    ep_lines.append(f"      '{sg}': {duration} steps")

            ep_lines.append(f"  {'-' * 60}")

            # Add to all-results
            lines.extend(ep_lines)

            # Check if total_timesteps differ across datasets
            unique_timesteps = set(timesteps_per_ds.values())
            if len(timesteps_per_ds) >= 2 and len(unique_timesteps) > 1:
                env_has_diff = True

                # Build diff lines showing only differing subgoals
                diff_ep_lines = []
                diff_ep_lines.append(f"\n  episode_{ep_num}:")
                diff_ep_lines.append(f"  {'-' * 60}")

                # Show header from each dataset
                ds_subgoals = {}
                for ds_name in dataset_names:
                    if ds_name not in datasets_for_env:
                        continue
                    ep_dict = datasets_for_env[ds_name]
                    if ep_num not in ep_dict:
                        continue
                    ep = ep_dict[ep_num]
                    diff_ep_lines.append(f"    [{ds_name}]  seed={ep['seed']}, difficulty={ep['difficulty']}, total_timesteps={ep['total_timesteps']}")
                    ds_subgoals[ds_name] = ep['subgoals']

                # Compare subgoals between datasets and only print differing ones
                if len(ds_subgoals) == 2:
                    ds_names_list = list(ds_subgoals.keys())
                    sg1 = ds_subgoals[ds_names_list[0]]
                    sg2 = ds_subgoals[ds_names_list[1]]

                    max_len = max(len(sg1), len(sg2))
                    for i in range(max_len):
                        if i < len(sg1) and i < len(sg2):
                            name1, dur1 = sg1[i]
                            name2, dur2 = sg2[i]
                            if name1 == name2 and dur1 != dur2:
                                diff_ep_lines.append(f"      '{name1}': {dur1} vs {dur2} steps")
                            elif name1 != name2:
                                diff_ep_lines.append(f"      [{ds_names_list[0]}] '{name1}': {dur1} steps")
                                diff_ep_lines.append(f"      [{ds_names_list[1]}] '{name2}': {dur2} steps")
                        elif i < len(sg1):
                            name1, dur1 = sg1[i]
                            diff_ep_lines.append(f"      [{ds_names_list[0]}] only: '{name1}': {dur1} steps")
                        else:
                            name2, dur2 = sg2[i]
                            diff_ep_lines.append(f"      [{ds_names_list[1]}] only: '{name2}': {dur2} steps")

                diff_ep_lines.append(f"  {'-' * 60}")
                env_diff_episode_lines.extend(diff_ep_lines)

        if env_has_diff:
            diff_lines.extend(env_header)
            diff_lines.extend(env_diff_episode_lines)

    # Print to console
    output = "\n".join(lines)
    print(output)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Write all results to txt file
    output_path = os.path.join(base_dir, "subgoal_time_result.txt")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write(output + "\n")
    print(f"\nAll results saved to {output_path}")

    # Write different total_timesteps to txt file
    diff_output_path = os.path.join(base_dir, "subgoal_time_resultdifferent.txt")
    if diff_lines:
        diff_output = "\n".join(diff_lines)
        with open(diff_output_path, 'w', encoding='utf-8') as f_out:
            f_out.write(diff_output + "\n")
        print(f"Different timesteps saved to {diff_output_path}")
    else:
        with open(diff_output_path, 'w', encoding='utf-8') as f_out:
            f_out.write("No episodes with different total_timesteps found.\n")
        print(f"No differences found. Empty result saved to {diff_output_path}")


if __name__ == "__main__":
    main()
