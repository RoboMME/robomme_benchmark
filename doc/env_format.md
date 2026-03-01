# Environment Input/Output

On RoboMME, a key difference from traditional Gym-like envs is that every observation value is a **list** rather than a single item. This is because some RoboMME tasks use conditioning video input, and for discrete action types (e.g. waypoint or multi_choice) we also return intermediate observations for potential use with video-based policy models.


## Env Input Format

We support four `ACTION_SPACE` types:

- `joint_angle`: 7 joint angles + gripper open/close
- `ee_pose`: 3 position (xyz) + 3 rotation (rpy) + gripper open/close
- `waypoint`: Same format as ee_pose, but executed in discrete keyframe steps
- `multi_choice`: Command dict, e.g. `{"choice": "A", "point": [y, x]}`; the total choices can be found in `info["available_multi_choices"]`, where the `point` is the pixel location on the front image. this action is designed for Video-QA research.

Note: Gripper closed is -1, gripper open is 1.


## Env Output Format

When calling the `step` function:

```python
obs, reward, terminated, truncated, info = env.step(action)
```

| Return | Description | Typical type |
|--------|-------------|--------------|
| `obs` | Observation dict | `dict[str, list]` |
| `info` | Info dict | `dict[str, Any]` |
| `reward` | Reward value (not used) | scalar tensor |
| `terminated` | Termination flag | scalar boolean tensor |
| `truncated` | Truncation flag | scalar boolean tensor |

### `obs` dict

| Key | Meaning | Typical content |
|-----|---------|-----------------|
| `maniskill_obs` | The original raw env observation from ManiSkill | Raw observation dict |
| `front_rgb_list` | Front camera RGB List | Image frames, e.g. `(H, W, 3)` |
| `wrist_rgb_list` | Wrist camera RGB List | Image frames, e.g. `(H, W, 3)` |
| `front_depth_list` | Front camera depth List | Depth map, e.g. `(H, W, 1)` |
| `wrist_depth_list` | Wrist camera depth List | Depth map, e.g. `(H, W, 1)` |
| `eef_state_list` | End-effector state List | `[x, y, z, roll, pitch, yaw]` |
| `joint_state_list` | Robot joint state List | Joint vector, often 7-D |
| `gripper_state_list` | Robot gripper state List | 2-D |
| `front_camera_extrinsic_list` | Front camera extrinsic List | Camera extrinsic matrix |
| `wrist_camera_extrinsic_list` | Wrist camera extrinsic List | Camera extrinsic matrix |


To use only the current (latest) observation, use `obs[key][-1]`.

### Optional field switches (`include_*`)

`BenchmarkEnvBuilder.make_env_for_episode(...)` controls optional observation/info fields through `include_*` flags.

Default behavior:
- All `include_*` flags default to `False`.
- Without extra flags, env returns RGB + state related fields only.

Mapping:

| Flag | Added key |
|------|-----------|
| `include_maniskill_obs` | `obs["maniskill_obs"]` |
| `include_front_depth` | `obs["front_depth_list"]` |
| `include_wrist_depth` | `obs["wrist_depth_list"]` |
| `include_front_camera_extrinsic` | `obs["front_camera_extrinsic_list"]` |
| `include_wrist_camera_extrinsic` | `obs["wrist_camera_extrinsic_list"]` |
| `include_available_multi_choices` | `info["available_multi_choices"]` |
| `include_front_camera_intrinsic` | `info["front_camera_intrinsic"]` |
| `include_wrist_camera_intrinsic` | `info["wrist_camera_intrinsic"]` |

Special case:
- If `action_space="multi_choice"`, front camera parameters are forced on internally:
  - `front_camera_extrinsic_list`
  - `front_camera_intrinsic`
  Even if the corresponding `include_front_camera_*` flags are `False`.

Example:

```python
from robomme.env_record_wrapper import BenchmarkEnvBuilder

builder = BenchmarkEnvBuilder(
    env_id="VideoUnmaskSwap",
    dataset="test",
    action_space="joint_angle",
    gui_render=False,
)

env = builder.make_env_for_episode(
    episode_idx=0,
    max_steps=1000,
    include_maniskill_obs=False,
    include_front_depth=True,
    include_wrist_depth=False,
    include_front_camera_extrinsic=True,
    include_wrist_camera_extrinsic=False,
    include_available_multi_choices=False,
    include_front_camera_intrinsic=True,
    include_wrist_camera_intrinsic=False,
)

obs, info = env.reset()
```

### `info` dict

| Key | Meaning | Typical content |
|-----|---------|-----------------|
| `task_goal` | Task goal list | `list[str]` |
| `simple_subgoal_online` | Oracle online simple subgoal | Description of the current simple subgoal |
| `grounded_subgoal_online` | Oracle online grounded subgoal | Description of the current grounded subgoal |
| `available_multi_choices` | Current available options for multi-choice action | List of e.g. `{"label: "a/b/...", "action": str, "need_parameter": bool}`, need_parameter means this action needs grounding info like `[y, x]` |
| `front_camera_intrinsic` | Front camera intrinsic | Camera intrinsic matrix |
| `wrist_camera_intrinsic` | Wrist camera intrinsic | Camera intrinsic matrix |
| `status` | Status flag | One of `success`, `fail`, `timeout`, `ongoing`, `error` |
