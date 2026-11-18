# RoboMME-Plus training data

The v2 training split contains **50 successful episodes, 57,056 frames at 20 Hz**:
one easy, medium and hard episode for 15 tasks, and five MoveCube methods.
One episode per non-MoveCube task has an [atomic variation](../src/robomme_plus/README.md#task-changes):
13 appearance changes and 2 mechanism changes. The other 35 episodes have no added variation flags.
H5 and LeRobot represent the same recordings. Download [H5](https://huggingface.co/datasets/Yinpei/robomme_plus_data_h5) and
[LeRobot v3](https://huggingface.co/datasets/Yinpei/robomme_plus_data_lerobot)
from `main`. The original release is preserved unchanged on the
[H5 v1](https://huggingface.co/datasets/Yinpei/robomme_plus_data_h5/tree/v1) and
[LeRobot v1](https://huggingface.co/datasets/Yinpei/robomme_plus_data_lerobot/tree/v1) branches.

## Replay

Replay both saved joint and end-effector actions through the simulator and export
front/wrist videos:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/plus_eval/dataset_replay.py \
  --h5-data-dir data/plus_train/h5 --tasks PatternLock --episodes 0
```

Omit `--tasks` and `--episodes` to replay all 50 training episodes in both action
spaces (100 replays). Use `--action-space joint_angle` or `--action-space ee_pose`
to check only one format. Videos are saved to
`runs/plus_dataset_replay/<Task>/episode_<k>_<action_space>.mp4` at the recorded control rate;
demonstration frames have a red border. The script checks episode metadata and
stops each replay when the simulator terminates. It exits nonzero if a replay fails.

## Seeds and episode mapping

The source of training seeds is
[`src/robomme_plus/env_metadata/train/`](../src/robomme_plus/env_metadata/train),
in `record_dataset_<Task>_metadata.json`. Record `k` corresponds to H5
`episode_<k>`; `BenchmarkEnvBuilder(task, dataset="train")` loads it.
Dataset copies of `train_plan.json`, `train_seeds.csv` and `metadata/train/`
include the task, episode, seed, difficulty/method and measured execution length.
Selected rows also include a versioned `perceptual_variation` or
`mechanism_variation` atomic preset. Seed values and difficulty levels are unchanged.
LeRobot `meta/source_episodes.json` maps global episode indices back to these rows.

## Conventions

- Observations precede their paired actions: `obs[t+1]` follows `action[t]`.
- Conditioning-video frames come first (`is_video_demo`); execution follows.
  Return-to-home motion between phases is unrecorded. The native step index can jump.
- Evaluator demo observations follow their actions, so their image sequence is
  shifted one frame relative to H5. The first execution observation matches the
  final reset observation.
- Positions use metres, angles radians, quaternions `wxyz`, and extrinsic `xyz` Euler angles.
- VideoPlace recordings include release and at least 5 cm of upward gripper retreat.
- Some MoveCube recordings retain settling frames after first success. A public
  evaluator replay stops at first success.

PatternLock and RouteStick show the demo's final activated target for two control
steps (0.1 s). Execution keeps its original success checks without an added wait.

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

MoveCube native commands are `[q(7), dq(7), gripper]`; other tasks omit velocities.
Action gripper commands use −1 closed and +1 open. Observed finger widths use
0–0.04 m. Choice actions are unused. Replay execution with `action/joint_action`
or `action/eef_action`; environment reset already performs the demonstration.

## LeRobot v3

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

`meta/native_action_format.json` describes native command packing. LeRobot omits
depth, camera extrinsics, observed TCP poses and waypoints; use H5 for these. H264 images
are lossy, so they are not pixel-identical to the H5 RGB arrays.

All 50 episodes passed joint and end-effector replay (100/100 checks),
export consistency and dataset-loader checks. Published manifests record
file checksums and the matching code revision.
