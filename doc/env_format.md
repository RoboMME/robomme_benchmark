# Environment Input/Output

A key difference from traditional Gym-like envs is that every observation value is a **list** rather than a single item. This is because some RoboMME tasks use conditioning video input, and for discrete action types (e.g. keypoint or multi_choice) we also return intermediate observations for potential use.


## Action Space Type

We support four `ACTION_SPACE` types:

- `joint_angle`: 7 joint angles + gripper open/close
- `ee_pose`: 3 position (xyz) + 3 rotation (rpy) + gripper open/close
- `keypoint`: Same format as ee_pose, but executed in discrete keyframe steps
- `multi_choice`: Command dict, e.g. `{"choice": "A", "point": [x, y]}`; intended for human studies or Video-QA research

Note: Gripper closed is -1, gripper open is 1.


## Step output

When calling `step`:

```python
obs, reward, terminated, truncated, info = env.step(action)
```

| Return | Description | Typical type |
|--------|-------------|--------------|
| `obs` | Observation dict | `dict[str, list]` |
| `info` | Info dict | `dict[str, list]` |
| `reward` | Reward values (not used) | 1D tensor |
| `terminated` | Termination flags | 1D boolean tensor |
| `truncated` | Truncation flags | 1D boolean tensor |

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

### `info` dict

| Key | Meaning | Typical content |
|-----|---------|-----------------|
| `task_goal` | Task goal text | Natural-language string |
| `simple_subgoal_online` | Oracle online simple subgoal | Description of the current subgoal |
| `grounded_subgoal_online` | Oracle online grounded subgoal | Subgoal with object grounding |
| `available_multi_choices` | Current available options for multi-choice action | List of e.g. `{"choice: "A/B/...", "action": str, "need_grounding": bool}` |
| `front_camera_intrinsic` | Front camera intrinsic | Camera intrinsic matrix |
| `wrist_camera_intrinsic` | Wrist camera intrinsic | Camera intrinsic matrix |
| `status` | Status flag | One of `success`, `fail`, `timeout` |

