# ASPIRE: RoboMME-Plus input and control reference

Read the shared benchmark overview and demonstration format together with
ASPIRE's original API reference and development instructions. This document
specifies the adapter's public data contract; it does not supply a task policy.

## Live observations and conditioning video

`get_observation()` returns an ASPIRE **`Observation` object**.
`robomme_plus.get_conditioning_video()` returns an ordered sequence of those
same objects, one per conditioning-video frame. It returns an empty sequence
when the episode has no conditioning video. It excludes the first execution
observation; obtain the live observation separately.

A video frame is not an image array, a path, a dictionary of image arrays, or
a concatenated front/wrist JPEG. Access its cameras by name:

```python
video = robomme_plus.get_conditioning_video()
if len(video) > 0:
    frame = video[0]                     # Observation
    front_rgb = frame.cameras["front"].rgb
    wrist_rgb = frame.cameras["wrist"].rgb
    front_depth_m = frame.cameras["front"].depth_m
    front_intrinsics = frame.cameras["front"].intrinsics
    front_pose = frame.cameras["front"].camera_pose
    recorded_robot_state = frame.robot_state
    video_time_s = frame.timestamp_s

live = get_observation()                # Current execution Observation
goal = get_task_context().language
```

Both camera RGB arrays are uint8 `(H, W, 3)` in RGB order, currently 256×256.
Depth is `(H, W)` in meters. Intrinsics are a 3×3 matrix. `camera_pose` is a
`Pose` with `position`, `quaternion_wxyz` and `frame`, expressing the camera in
`robot_base`. Images and depth use the camera's image coordinates; use the
documented geometry tools to convert them to robot coordinates. Arrays are
read-only. Camera pose and robot state belong to that particular frame.

Long conditioning videos are uniformly sampled to at most **512 timestamps**,
including both endpoints. Both cameras, depth, calibration and robot state are
retained at each selected timestamp. Use `timestamp_s` for elapsed time: consecutive
retained frames need not be one control tick apart. Videos of 512 frames or fewer
are unchanged. The live execution clock starts separately at zero; do not subtract
a video timestamp from an execution timestamp. Input is streamed in bounded pieces.
Task metadata includes `conditioning_video_sampling` with original and retained counts.
Depth/state/calibration in this
runtime input are distinct from the RGB-only released-demonstration export.

## Robot state, actions and time

`get_robot_state()` returns `RobotState`. The arm key is `"primary"`; its
`embodiment` is `"robomme_plus"`. Joint positions are seven absolute angles in
radians. `end_effector_poses["primary"]` is a `Pose` in `robot_base`, using
ASPIRE's Panda-hand reference plus local `[0, 0, 0.1]` meters. The measured
`gripper_positions["primary"]` is normalized: **0 closed, 1 open**. A partially
closed reading reports finger position, not a task-success flag.

For direct `step`, use `ArmCommand(mode="joint_position", ...)` with that
embodiment. A low-level joint command advances one native control tick, currently
20 Hz. High-level motion and gripper helpers may execute multiple ticks or return
early. Measure progress using returned state timestamps and status, rather than
counting Python calls. `robomme_plus.get_controller_metadata()` describes frames
and action conventions; `robomme_plus.get_task_metadata()` provides the task, language,
control frequency and overall `max_steps`. See the shared `PUBLIC_RULES.md` for
the published button deadlines, which can terminate before that horizon.

The existing API documentation defines the permitted constructors, helper
functions and sandbox syntax. Use actual public contract objects when testing
parsers offline; a hand-written image-only mock is not equivalent to an
`Observation`. Offline checks do not authorize a simulator start in demo-only.

After a motion or gripper helper reports `terminated` or `truncated`, finish the
program using the result already returned. Further state/image reads may raise
"Episode has ended". Guard optional final diagnostics so they cannot discard a
completed program's result. Helper completion, the program's report and native
task success are separate outcomes.
