# HDF5 Training Data Format

Structure inside each `record_dataset_<EnvID>.h5` file:

```text
episode_1/
  setup/
  timestep_1/
    obs/
    action/
    info/
  timestep_2/
    obs/
    action/
    info/
  ...
...
```

Each episode contains:
- `setup/`: episode-level configuration.
- `timestep_<K>/`: per-timestep data.

## `setup/` fields (episode configuration)

| Field | Type | Description |
|-------|------|-------------|
| `seed` | `int` | Environment seed (fixed for benchmarking) |
| `difficulty` | `str` | Difficulty level (fixed for benchmarking) |
| `task_goal` | `list[str]` | Possible language goals for the task |
| `front_camera_intrinsic` | `float32 (3, 3)` | Front camera intrinsic matrix |
| `wrist_camera_intrinsic` | `float32 (3, 3)` | Wrist camera intrinsic matrix |
| `available_multi_choices` | `str` | Available options for the multi-choice Video-QA problem |

## `obs/` fields (observations)

| Field | Type / shape | Description |
|-------|---------------|-------------|
| `front_rgb` | `uint8 (256, 256, 3)` | Front camera RGB |
| `wrist_rgb` | `uint8 (256, 256, 3)` | Wrist camera RGB |
| `front_depth` | `int16 (256, 256, 1)` | Front camera depth (mm) |
| `wrist_depth` | `int16 (256, 256, 1)` | Wrist camera depth (mm) |
| `joint_state` | `float32 (7,)` | Absoluate joint positions (7 joints) |
| `eef_state` | `float32 (6,)` | Absoluate end-effector pose `[x, y, z, roll, pitch, yaw]` |
| `gripper_state` | `float32 (2,)` | Gripper opening width in [0, 0.04] |
| `is_gripper_close` | `bool` | Whether gripper is closed |
| `front_camera_extrinsic` | `float32 (3, 4)` | Front camera extrinsic matrix |
| `wrist_camera_extrinsic` | `float32 (3, 4)` | Wrist camera extrinsic matrix |

## `action/` fields

| Field | Type / shape | Description |
|-------|---------------|-------------|
| `joint_action` | `float32 (8,)` | Absolute joint-space action: 7 joint angles + gripper |
| `eef_action` | `float32 (7,)` | Absolute end-effector action `[x, y, z, roll, pitch, yaw, gripper]` |
| `waypoint_action` | `float32 (7,)` | Absolute end-effector action at discrete time steps; a subtask may contain multiple waypoint actions. Used for data generation. |
| `choice_action` | `str` | JSON string for multi-choice selection with an optional grounded pixel location on the front image, e.g., `{"choice": "A", "point": [y, x]}` |

In RoboMME, a gripper action of -1 means close and 1 means open.

## `info/` fields (metadata)

| Field | Type | Description |
|-------|------|-------------|
| `simple_subgoal` | `bytes (UTF-8)` | Simple subgoal text (built-in planner view) |
| `simple_subgoal_online` | `bytes (UTF-8)` | Simple subgoal text (online view; may advance to the next subgoal earlier than planner view) |
| `grounded_subgoal` | `bytes (UTF-8)` | Grounded subgoal text (built-in planner view) |
| `grounded_subgoal_online` | `bytes (UTF-8)` | Grounded subgoal text (online view; may advance to the next subgoal earlier than planner view) |
| `is_video_demo` | `bool` | Whether this frame is from the conditioning video shown before execution |
| `is_subgoal_boundary` | `bool` | Whether this is a keyframe (i.e., a boundary between subtasks) |
| `is_completed` | `bool` | Whether the task is finished |
