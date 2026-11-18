---
name: robot-interface
description: The robot - one Franka Panda arm with a parallel-jaw gripper - its official action rows (end-effector position and rotation deltas plus a gripper command), its state vector, frames, cameras and benchmark control conventions. Read before composing robot actions.
---

# Robot interface: Panda + parallel gripper

One Franka Panda arm is mounted at the edge of the table and ends in a two-finger
parallel-jaw gripper. `{{ robot.status }}` reports the exact action and state layouts;
this skill explains the RoboMME-Plus mapping.

## One action row

Each `{{ robot.step }}` row is one control step of 50 ms ({{ robot.hz }} per second). A row is a
**relative command**, every value in [-1, 1]:

| Slice | Meaning |
|---|---|
| `[0:3]` | end-effector position change, world frame: 1.0 asks for 0.05 m |
| `[3:6]` | end-effector rotation change as an axis-angle vector, world frame: 1.0 asks for 0.5 rad |
| `[6]` | gripper: -1 opens, +1 closes |

The adapter translates each row against the latest measured native TCP into the
benchmark's absolute `ee_pose` command. The benchmark's IK and joint controller
execute it for one native control tick. This is not LIBERO's OSC controller;
LIBERO response/lag measurements do not apply. Inspect actual observations after
commands. Recoverable native IK errors can consume additional penalty steps;
the reported step clock is authoritative.

PatternLock and RouteStick use a fixed drawing stick and ignore gripper commands.
Every started episode, including rehearsals and debugging, consumes the study budget.

## The measured state

`state` has 8 values:

| Slice | Meaning |
|---|---|
| `[0:3]` | end-effector position (m, world frame): the benchmark-native TCP (tool frame on stick tasks) |
| `[3:6]` | end-effector orientation as an axis-angle vector (rad, world frame) |
| `[6:8]` | the two finger positions (m); their sum is the opening (each 0 to 0.04 m; zero on stick tasks) |

There are no joint angles and no object positions in the state; find objects in the
images.

## Frames and controller behavior

The native TCP position is in world metres. Orientation is an axis-angle vector
in world radians. Read current positions, orientation and geometry from the
observations; do not assume the LIBERO reset pose, table height or tracking lag.
Near angle pi, compare orientations by relative rotation rather than component
subtraction. No scene-object poses or oracle information are exposed.

## Cameras

{% for camera in robot.cameras %}
- `{{ camera }}`{% if camera == 'wrist' %}: on the gripper, looking along the fingers, moving with it{% else %}: the third-person view of the table and the arm{% endif %}.
{% endfor %}

Images are 256x256 RGB. Motion replies return the final frame inline and save
every fifth control step. Reset saves **every** conditioning-video frame at 20 Hz,
followed by the first execution observation. Its manifest records the phase;
read the full reset sequence when the task supplies a video.
{% if camera_geometry %}
Depth images and `camera_intrinsics` / `camera_poses` (world frame) come with every
observation; see [object-localization](../object-localization/SKILL.md).
{% else %}
For locating objects without depth, see [object-localization](../object-localization/SKILL.md).
{% endif %}

{% if multi_env %}
## Environment selection

- Pass the selected `env_id` on every `robomme_*` call, and on `start_episode`,
  `evaluate` and `finish`; for example `{{ robot.step }}(env_id=i, actions=...)`.
{% if agent_formal %}
  `{{ robot.status }}` reports one environment; `status` and `gemini_generate` take no `env_id`.
{% else %}
  `{{ robot.status }}` reports one environment; `status`, `gemini_generate`, `register`,
  `rehearse` and `submit` take no `env_id`.
{% endif %}
- Keep multi-call sequences in one environment, as
  [multi-env exploration](../multi-env-exploration/SKILL.md) describes.
{% endif %}

## Related skills

- [object-localization](../object-localization/SKILL.md) for where objects are.
{% if not formal_episode %}
- [references/pose-records.md](references/pose-records.md) for how poses and executed
  rows are recorded for reuse.
{% endif %}
