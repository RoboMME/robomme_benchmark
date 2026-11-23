# RoboMME-Plus training data

The training split contains **50 successful episodes, 57,056 frames at 20 Hz**:
one easy, one medium and one hard episode for each of 15 tasks, plus one for each
of MoveCube's five methods. Some episodes include a perceptual or mechanism
variation. The [H5](https://huggingface.co/datasets/Yinpei/robomme_plus_data_h5)
and [LeRobot](https://huggingface.co/datasets/Yinpei/robomme_plus_data_lerobot)
downloads contain the same recordings.

Use these recordings as offline data in the `Demo-only` and `Hybrid` settings.

## Dataset Replay

Replay the saved joint and end-effector actions to check the data and export
front and wrist videos:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/plus_eval/dataset_replay.py \
  --h5-data-dir data/plus_train/h5 --tasks PatternLock --episodes 0
```

Omit `--tasks` and `--episodes` to replay all 50 episodes in both action spaces,
or pass `--action-space joint_angle` or `--action-space ee_pose` to check one.
Videos are saved to `runs/plus_dataset_replay/<Task>/episode_<k>_<action_space>.mp4`
at the recorded control rate, with a red border on conditioning-video frames. The
script checks episode metadata, stops each replay when the episode ends, and exits
with an error if any replay fails.

## Seeds and episode mapping

Training seeds are listed in
[`src/robomme_plus/env_metadata/train/record_dataset_<Task>_metadata.json`](../../src/robomme_plus/env_metadata/train).
Record `k` is H5 `episode_<k>`, and `BenchmarkEnvBuilder(task, dataset="train")`
loads it. The dataset's `train_plan.json`, `train_seeds.csv` and `metadata/train/`
list each episode's task, index, seed, difficulty or method, and execution length.
Episodes with a variation name it in `perceptual_variation` or
`mechanism_variation`; the variation does not change the seed or difficulty.
In LeRobot, `meta/source_episodes.json` maps global episode indices back to these rows.

## Conventions

- Observations precede their paired actions: `obs[t+1]` follows `action[t]`.
- Conditioning-video frames come first (`is_video_demo`), then execution. The
  robot's return to its home pose between the two is not recorded, so the native
  step index can jump.
- In the simulator, the conditioning-video frames returned by `reset()` are shifted
  one frame relative to H5. The first execution observation matches the last frame
  returned by `reset()`.
- Positions use metres, angles radians, quaternions `wxyz`, and extrinsic `xyz` Euler angles.

## H5

Each `h5/record_dataset_<Task>.h5` contains `episode_<k>/setup` and
`timestep_<t>/{obs,action,info}`.

| Group/field | Content |
| --- | --- |
| `setup` | Goal, seed, difficulty, episode metadata, camera intrinsics and phase lengths |
| `obs/front_rgb`, `obs/wrist_rgb` | uint8 RGB, 256×256×3 |
| `obs/front_depth`, `obs/wrist_depth` | int16 depth in millimetres, 256×256×1 |
| `obs/*_camera_extrinsic` | World-to-camera `[R\|t]`, 3×4 |
| `obs/joint_state`, `obs/gripper_state` | 7 arm joints; two finger widths in metres |
| `obs/eef_state` | Native TCP `[x,y,z,roll,pitch,yaw]` |
| `action/native_joint_action` | Exact simulator command: 7 values for sticks, 8 normally, 15 for MoveCube |
| `action/joint_action` | Standard 8-value joint target + gripper command |
| `action/eef_action` | Desired TCP `[x,y,z,roll,pitch,yaw,gripper]` for the `ee_pose` interface |
| `action/waypoint_action` | Current 7-value keyframe target when `waypoint_action_valid` is true |
| `info` | Subgoal text, grounding, demo/completion/boundary flags and native control step |

MoveCube native commands are `[q(7), dq(7), gripper]`; other tasks omit the
velocities. Gripper commands use −1 for closed and +1 for open; observed finger
widths range from 0 to 0.04 m. Choice actions are unused. To replay an episode,
send `action/joint_action` or `action/eef_action` after `reset()`; reset already
plays the conditioning video.

## LeRobot

The dataset root contains `meta/`, `data/` and `videos/`. Each episode has one
Parquet file and front/wrist H264 videos. Each row corresponds to one H5 timestep.

| Feature | Meaning |
| --- | --- |
| `observation.images.front`, `.wrist` | 256×256 RGB video at 20 FPS |
| `observation.state` | 7 joint positions + mean finger width in metres |
| `action` | 7 joint targets + gripper command |
| `action.eef` | Desired TCP position, Euler angles and gripper command (7 values) |
| `action.native_sequence` | 75 values: reshape to 5×15, use the first `native_action_count` rows |
| `native_action_count`, `native_frame_index` | Always 1; corresponding H5 frame |
| `is_video_demo`, `control_freq` | Phase flag; 20 Hz |
| `task_index` | Goal text in `meta/tasks.parquet` |
| `subtask*_index` | Plain/grounded subgoal variants in `meta/subtasks.parquet` |
| `episode_index`, `frame_index`, `timestamp`, `index` | Episode, local frame, seconds and global row |
| `next.done` | True only on the last row |

`meta/native_action_format.json` describes how native commands are packed.
LeRobot omits depth, camera extrinsics, observed TCP poses and waypoints; use H5 for
these. H264 video is lossy, so its frames are not pixel-identical to the H5 RGB arrays.

All 50 episodes pass joint and end-effector replay (100/100 checks) as well as
export and loader checks. The published manifests record file checksums and the
matching code revision.
