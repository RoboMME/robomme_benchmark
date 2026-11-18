# Pose and action records: Panda + parallel gripper

{% if formal_episode %}
How [experience memory](../../experience-memory/SKILL.md) records a motion of this robot for
reuse.
{% else %}
How [experience memory](../../experience-memory/SKILL.md) records a motion of this robot for
reuse. These fields fill the robot-specific part of an example in
[its format](../../experience-memory/references/format.md).
{% endif %}

## Encode poses

- Rows are relative commands, so a recorded row sequence only repeats a motion from the
  same measured start. Record the reusable recipe as end-effector targets relative to a
  visible anchor (where to go), and the row values that got there as how.
- Store targets relative to a visible anchor or the measured end-effector position,
  together with how to find the anchor again; object placement changes between layouts,
  so absolute coordinates do not transfer.
- Record the gripper state (open or closed) and the finger positions after a grasp.
- Compare orientations by their relative rotation: the axis-angle vector of the
  downward-pointing gripper sits near angle pi, where components can flip sign.

## Example fields

```json
{
  "action_kind": "rows",
  "executed_action": {
    "tool": "{{ robot.step }}",
    "representation": "official_delta_rows_20hz",
    "rows": "transition.steps[*].executed_action"
  },
  "pose_recipe": {
    "representation": "relative_world",
    "anchor": "<visible point or measured end-effector position>",
    "frame": "world",
    "position_offset_m": [0.0, 0.0, 0.05],
    "gripper": "open | closed"
  },
  "point_reacquisition": {
{% if camera_geometry %}
    "tool": "{{ robot.pixel }}",
{% else %}
    "method": "RGB visual alignment with measured end-effector positions",
{% endif %}
    "visible_feature": "<feature description>",
    "preferred_views": {{ robot.cameras | list | tojson }},
    "offset_m": [0.0, 0.0, 0.05]
  }
}
```

Record executed rows as `official_delta_rows_20hz`: the rows from
`transition.steps[*].executed_action`, each paired with the observation it was sent from,
step `step_id - 1`. Replaying rows only reproduces an effect from a matching measured state
and layout.
