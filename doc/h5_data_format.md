# HDF5 Training Dataset Format

Structure inside each `record_dataset_<EnvID>.h5` file:

```text
episode_<N>/
  timestep_<K>/
    obs/
    action/
    info/
  setup/
```

For each episode data:
- `timestep_<K>/`: per-timestep data.
- `setup/`: episode-level configuration.

## `setup/` fields (episode configuration)

| Field | Type | Description |
|-------|------|-------------|
| `seed` | `int` | Environment seed |
| `difficulty` | `str` | Difficulty level |
| `task_goal` | `list[str]` | A list of possible language goals for the task |
| `front_camera_intrinsic` | `float32 (3, 3)` | Front camera intrinsic matrix |
| `wrist_camera_intrinsic` | `float32 (3, 3)` | Wrist camera intrinsic matrix |
| `available_multi_choices` | `str` | Current available options for multi-choice action |

## `obs/` fields (observations)

| Field | Type / shape | Description |
|-------|---------------|-------------|
| `front_rgb` | `uint8 (256, 256, 3)` | Front camera RGB |
| `wrist_rgb` | `uint8 (256, 256, 3)` | Wrist camera RGB |
| `front_depth` | `int16 (256, 256, 1)` | Front camera depth (mm) |
| `wrist_depth` | `int16 (256, 256, 1)` | Wrist camera depth (mm) |
| `joint_state` | `float32 (7,)` | Joint state (7 joints) |
| `eef_state` | `float32 (6,)` | End-effector state `[x, y, z, roll, pitch, yaw]` |
| `gripper_state` | `float32 (2,)` | Gripper state |
| `is_gripper_close` | `bool` | Whether gripper is closed |
| `front_camera_extrinsic` | `float32 (3, 4)` | Front camera extrinsic matrix |
| `wrist_camera_extrinsic` | `float32 (3, 4)` | Wrist camera extrinsic matrix |

## `action/` fields (actions)

| Field | Type / shape | Description |
|-------|---------------|-------------|
| `joint_action` | `float32 (8,)` or `str "None"` | Joint-space action |
| `eef_action` | `float32 (7,)` | End-effector action `[x, y, z, roll, pitch, yaw, gripper]` |
| `waypoint_action` | `float32 (7,)` | End-effector action at discrete time steps. A subtask may contain multiple waypoint actions|
| `choice_action` | `str` | dict string for multi-choice selection and optional grounded pixel, e.g. `{"choice": "A", "point": [x, y]}`|

## `info/` fields (metadata)

| Field | Type | Description |
|-------|------|-------------|
| `simple_subgoal` | `bytes (UTF-8)` | Simple subgoal text (built-in planner view) |
| `simple_subgoal_online` | `bytes (UTF-8)` | Simple subgoal text (online view; may be earlier than planner view) |
| `grounded_subgoal` | `bytes (UTF-8)` | Grounded subgoal text |
| `grounded_subgoal_online` | `bytes (UTF-8)` | Online grounded subgoal text |
| `is_video_demo` | `bool` | Whether this frame is from the conditioning video input before execution |
| `is_keyframe` | `bool` | Whether this is a keyframe (i.e. a boundary between subtasks)|
| `is_completed` | `bool` | Whether the task is finished |
