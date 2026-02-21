import os
import sys
import json
import h5py
import numpy as np
import argparse
from pathlib import Path
from typing import Any

from robomme.robomme_env.utils.rpy_util import summarize_and_print_rpy_sequence

def _write_split_rpy_summaries_json(
    path: str,
    demo_summaries: list[dict[str, Any]],
    non_demo_summaries: list[dict[str, Any]],
) -> None:
    """
    Summarize both demo and non-demo parts and write to JSON.
    """
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "demo_summaries": demo_summaries,
        "non_demo_summaries": non_demo_summaries,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_is_video_demo(ts_group: h5py.Group) -> bool:
    """Read info/is_video_demo from timestep group, default to False if missing."""
    info_grp = ts_group.get("info")
    if info_grp is not None and "is_video_demo" in info_grp:
        val = info_grp["is_video_demo"][()]
        if isinstance(val, (bytes, np.bytes_)):
            return val in (b"True", b"true", b"1")
        return bool(val)
    return False


def _extract_rpy_from_timestep(ts_group: h5py.Group) -> list[np.ndarray]:
    """Extract RPY vector list from a single timestep."""
    if (
        "action" in ts_group
        and "eef_action_raw" in ts_group["action"]
        and "rpy" in ts_group["action"]["eef_action_raw"]
    ):
        rpy_data = ts_group["action"]["eef_action_raw"]["rpy"][()]
        rpy_arr = np.asarray(rpy_data, dtype=np.float64)
        if rpy_arr.ndim == 1:
            rpy_arr = rpy_arr.reshape(1, -1)
        else:
            rpy_arr = rpy_arr.reshape(-1, rpy_arr.shape[-1])
        if rpy_arr.shape[-1] == 3:
            return [row.copy() for row in rpy_arr]
    return []


def main():
    # Hardcoded dataset directory as requested
    DATASET_DIR = Path("/data/hongzefu/dataset_generate")
    
    parser = argparse.ArgumentParser(description="Read generated HDF5 dataset and verify RPY consistency.")
    parser.add_argument("--dataset_path", type=str, default=str(DATASET_DIR), help="Path to the HDF5 file or directory to verify.")
    args = parser.parse_args()

    input_path = Path(args.dataset_path).resolve()
    
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)

    # Determine files to process
    files_to_process = []
    if input_path.is_file():
        if input_path.suffix in ['.h5', '.hdf5']:
            files_to_process.append(input_path)
    elif input_path.is_dir():
        files_to_process.extend(sorted(input_path.glob("*.h5")))
        files_to_process.extend(sorted(input_path.glob("*.hdf5")))
    
    if not files_to_process:
        print(f"No HDF5 files found in {input_path}")
        sys.exit(0)

    print(f"Found {len(files_to_process)} files to process in {input_path}")

    for dataset_path in files_to_process:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset_path}")
        print(f"{'='*50}")
        
        # Generate output JSON path
        output_json_path = Path("/data/hongzefu/dataset_replay") / f"{dataset_path.stem}_rpy_summary.json"

        demo_summaries: list[dict[str, Any]] = []
        non_demo_summaries: list[dict[str, Any]] = []

        try:
            with h5py.File(dataset_path, "r") as f:
                # Iterate through environments (e.g., env_PickXtimes...)
                env_groups = [key for key in f.keys() if key.startswith("env_")]
                env_groups.sort()

                if not env_groups:
                    print(f"Warning: No 'env_*' groups found in {dataset_path.name}")

                for env_group_name in env_groups:
                    env_group = f[env_group_name]
                    print(f"Processing environment group: {env_group_name}")
                    
                    # Extract env_id from group name (remove 'env_' prefix)
                    env_id = env_group_name[4:]

                    # Iterate through episodes
                    episode_keys = [key for key in env_group.keys() if key.startswith("episode_")]
                    # Sort numerically by episode ID
                    episode_keys.sort(key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else x)

                    for episode_key in episode_keys:
                        print(f"  Processing {episode_key}...")
                        episode_group = env_group[episode_key]
                        try:
                            episode_idx = int(episode_key.split('_')[1])
                        except (IndexError, ValueError):
                             episode_idx = -1
                        
                        # Iterate through timesteps to reconstruct sequence
                        timestep_keys = [key for key in episode_group.keys() if key.startswith("record_timestep_")]
                        
                        def get_timestep_idx(key):
                            parts = key.split('_')
                            try:
                                return int(parts[2])
                            except (IndexError, ValueError):
                                return -1
                        
                        timestep_keys.sort(key=get_timestep_idx)

                        # Separate RPY sequences by is_video_demo flag
                        demo_rpy_seq: list[np.ndarray] = []
                        non_demo_rpy_seq: list[np.ndarray] = []

                        for ts_key in timestep_keys:
                            ts_group = episode_group[ts_key]
                            rpy_rows = _extract_rpy_from_timestep(ts_group)
                            if rpy_rows:
                                if _read_is_video_demo(ts_group):
                                    demo_rpy_seq.extend(rpy_rows)
                                else:
                                    non_demo_rpy_seq.extend(rpy_rows)

                        # Summarize demo portion
                        if demo_rpy_seq:
                            demo_summary = summarize_and_print_rpy_sequence(
                                demo_rpy_seq,
                                label=f"[{env_id}] episode {episode_idx} (demo)",
                            )
                            demo_summaries.append({
                                "order_index": len(demo_summaries),
                                "env_id": env_id,
                                "episode": episode_idx,
                                "action_space": "eef_pose",
                                "summary": demo_summary,
                            })

                        # Summarize non-demo portion
                        if non_demo_rpy_seq:
                            non_demo_summary = summarize_and_print_rpy_sequence(
                                non_demo_rpy_seq,
                                label=f"[{env_id}] episode {episode_idx} (non-demo)",
                            )
                            non_demo_summaries.append({
                                "order_index": len(non_demo_summaries),
                                "env_id": env_id,
                                "episode": episode_idx,
                                "action_space": "eef_pose",
                                "summary": non_demo_summary,
                            })

        except Exception as e:
            print(f"An error occurred while reading {dataset_path.name}: {e}")
            import traceback
            traceback.print_exc()

        # Write summary to JSON
        if demo_summaries or non_demo_summaries:
            _write_split_rpy_summaries_json(str(output_json_path), demo_summaries, non_demo_summaries)
            print(f"Saved split RPY summaries to: {output_json_path}")
            print(f"  demo entries: {len(demo_summaries)}, non-demo entries: {len(non_demo_summaries)}")
        else:
            print(f"No summaries generated for {dataset_path.name}")

if __name__ == "__main__":
    main()
