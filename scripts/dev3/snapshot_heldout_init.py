"""
Use override_metadata_path to load heldout seed/difficulty,
reset the env, and save the initial observation image.
"""

from pathlib import Path
import numpy as np
import imageio

from robomme.env_record_wrapper import BenchmarkEnvBuilder

REPO_ROOT   = Path(__file__).resolve().parents[2]
METADATA_DIR = REPO_ROOT / "runs/replay_videos/heldout_metadata"
OUT_DIR      = REPO_ROOT / "runs/dev3_snapshots"

TASK        = "BinFill"
EPISODE_IDX = 8   # change as needed


def main():
    builder = BenchmarkEnvBuilder(
        env_id=TASK,
        dataset="test",               # dataset arg is required but overridden below
        action_space="joint_angle",
        gui_render=False,
        override_metadata_path=METADATA_DIR,
    )

    seed, difficulty = builder.resolve_episode(EPISODE_IDX)
    print(f"Episode {EPISODE_IDX}: seed={seed}, difficulty={difficulty}")

    env = builder.make_env_for_episode(EPISODE_IDX)
    obs, info = env.reset()
    env.close()

    # obs["front_rgb_list"] and obs["wrist_rgb_list"] are lists of frames
    # (video-demo frames + the current live frame); last entry is the live view
    front = np.asarray(obs["front_rgb_list"][-1])
    wrist = np.asarray(obs["wrist_rgb_list"][-1])
    combined = np.hstack([front, wrist])   # side-by-side

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{TASK}_ep{EPISODE_IDX}_seed{seed}_{difficulty}_init.png"
    imageio.imwrite(str(out_path), combined)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
