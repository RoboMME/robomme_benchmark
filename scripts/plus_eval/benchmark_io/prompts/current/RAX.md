# RAX: RoboMME-Plus input and control reference

Read the shared benchmark overview and demonstration format together with RAX's
original development instructions and robot-interface skill. RAX retains its
native exploration, registration, rehearsal and submission workflow. All actual
training episode starts, including frozen-bundle rehearsals, consume the assigned
task budget.

## Observation packets and the initial video

The robot tools return **JSON metadata and image attachments**. An observation
packet includes `episode_id`, `step_id`, `instruction`, `phase` and `state`.
RGB attachments identify the `front` and `wrist` cameras separately. With the
RGB-D profile, depth metadata, `camera_intrinsics` and `camera_poses` are also
present. Check the advertised observation profile rather than assuming that
geometry exists in an RGB-only configuration.

For an MCP tool result, decode the JSON text block in `content`; its `attachments`
descriptors identify image blocks by **`content_index`**, `camera` and `kind`.
Those indices count all content blocks, including the JSON text block at zero.
Match views by `camera`, not by an assumed attachment order. Depth PNGs encode
uint16 millimeters; multiply by `meters_per_unit` (0.001); zero is invalid.

The exact calibration schema for one view is:

```json
{
  "camera_intrinsics": {
    "front": {"matrix": [[128, 0, 128], [0, 128, 128], [0, 0, 1]], "width": 256, "height": 256}
  },
  "camera_poses": {
    "front": {"position_m": [1, 2, 3], "quaternion_wxyz": [1, 0, 0, 0], "frame": "world", "convention": "OpenGL"}
  }
}
```

These numbers are invented format examples. Both views have their own entries.
The position field is **`position_m`**, not `position`; the intrinsic matrix is
under **`matrix`**. Quaternion ordering is **w, x, y, z**. For a depth pixel
`(u, v, d)`, the OpenGL camera vector is
`[(u-cx)*d/fx, -(v-cy)*d/fy, -d]`; rotate by the camera quaternion and add
`position_m` to obtain world meters. Alternatively use `robomme_pixel_to_position`.
Do not silently substitute guessed calibration when these fields are present.

The inline images show the **final observation of a reply**, not the entire
conditioning video. A `frame_sequence` descriptor points to a readable JSON
`manifest_path`. Read that manifest to inspect the preceding frames. The initial
sequence is supplied at episode startup and by `robomme_observe` at execution
step zero; the frozen runner publishes it before the controller starts.

The initial manifest's `frames` list contains the retained conditioning frames followed
by the first execution frame. Use `phase == "video_demo"` for conditioning
footage and `phase == "execution"` for the live boundary. A task without an
initial video has only the execution observation. `frame_index` orders the
sequence; `step_id` is the execution clock and may be zero for every reset frame.
Do not use it to sort or deduplicate conditioning frames.

Each frame contains `files`, whose entries have `camera`, `kind` (`rgb` or
`depth`) and `path`. Published paths are relative to the agent workspace root,
not the manifest's directory. Use the returned paths instead of guessing private
simulator directories. This example starts with the JSON packet decoded by your
existing RAX tool client:

```python
import json
from pathlib import Path
from PIL import Image

workspace_root = Path("/workspace")
sequence = packet["frame_sequence"]
manifest = json.loads((workspace_root / sequence["manifest_path"]).read_text())
for frame in manifest["frames"]:
    if frame["phase"] != "video_demo":
        continue
    files = {(f["camera"], f["kind"]): f["path"] for f in frame["files"]}
    with Image.open(workspace_root / files[("front", "rgb")]) as image:
        front = image.convert("RGB")
        # This frame can now be inspected by your own program.
```

Long reset videos are uniformly sampled to at most **512 timestamps**, including
both endpoints and both cameras at each timestamp. Shorter videos are unchanged.
`source_frame_index` and `timestamp_s` preserve the original video clock. The
reported `frequency_hz` describes the source clock, not the spacing of sampled
frames; use timestamp differences for speed estimates. `video_sampling` in the
initial transition records original and retained counts. The first live execution
observation is always retained separately. Motion sequences normally sample every
five control steps. The depth PNGs use
uint16 millimeters: multiply by their `meters_per_unit` (currently 0.001); zero is
invalid. Intrinsics describe image projection. Camera poses use world-frame
positions, `quaternion_wxyz` and the declared OpenGL convention. Use the existing
pixel-to-position API or the documented conversion, rather than mixing it with
ASPIRE's frame conventions.

## Robot state, actions and time

`state` has eight numbers: native end-effector position `[0:3]` in world meters,
orientation `[3:6]` as an axis-angle vector in radians, and two finger positions
`[6:8]` in meters. There are no object poses or joint-angle arrays in this state.
Each finger ranges from 0 to 0.04 meters; their sum is the gripper opening.

Each `robomme_step` action row has seven numbers in `[-1, 1]`: world-frame
translation deltas `[0:3]` scaled by 0.05 meters, world-frame rotation-vector
deltas `[3:6]` scaled by 0.5 radians, and gripper `[6]`, where **-1 opens and +1
closes**. These are relative commands, not absolute poses. The adapter translates
them against the current measured pose. A row normally advances one 20 Hz
control tick; reported step counts remain authoritative when native error
handling consumes additional ticks. Stick tools can ignore gripper commands.

Use the robot status reply for the current action/state layouts, episode limit
and terminal state. Repeated observation queries do not advance physics. The
published recording, actual terminal outcome and your program's own assessment
are different things; a completed controller process is not necessarily success.

The limits are `episode_limits.episode_step_limit` and
`episode_limits.remaining_episode_steps`; use the reported `step_id`. A requested
delta is not a measurement of completed motion. Read measured `state` after
movement, recompute the remaining displacement, and respect remaining steps.
Do not infer contact, grasp or placement from sending a command alone.

`episode_ended` indicates termination. The ending transition's `steps` entries
carry `episode_ended` and `task_complete`; inspect that transition before another
robot call. A terminal failure is a normal episode outcome. A JSON-RPC `error`
instead identifies an interface/runtime failure and includes `data` describing
its stage and operation; it is not an observation. Never resume or reset an ended
episode to evade the interaction budget. A clean controller exit also does not
imply `task_complete`.

## Offline format checks

`fixtures/README.md` lists complete synthetic MCP replies, public frame manifests,
both camera views and terminal/error examples. They pass through the same packet
serializers as the adapter and contain invented sensor values, no task solution.
Use them to check decoding and calibration before a real attempt. Such checks
cannot validate perception, contact physics or task success. RAX's native exact
bundle rehearsal requirement still applies to interaction-based submission;
offline checks do not replace that rehearsal or grant additional starts.
