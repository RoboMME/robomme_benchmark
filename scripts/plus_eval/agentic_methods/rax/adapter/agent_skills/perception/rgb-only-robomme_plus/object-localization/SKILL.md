---
name: object-localization
description: Locate objects from RGB images and measured end-effector positions, without depth, camera intrinsics or camera poses. Use to judge where objects are relative to the gripper before reaching, grasping or placing.
---

# Object localization: RGB only

Only RGB images and the measured state are supplied: no depth, camera intrinsics or
camera poses. One image gives a direction, not a metric position, so never turn image
pixels alone into coordinates. Do not seek simulator files or hidden camera or object
state.

The state's end-effector position is the point between the fingertips (see
[robot-interface](../robot-interface/SKILL.md)); it moves with the gripper.

## Find the target

- Use `{{ robot.overview_camera }}` for the overview: where the target is relative to the gripper,
  the table edges and other objects. Images are 256x256 and upright.
- Learn how the image maps to the world: move the empty gripper a small, known
  displacement at a safe height and see how it moves in a fresh image. Do not assume
  image axes equal world axes.
- The wrist camera resolves finer detail near the gripper. Recheck after moving.

## Align with measured motion

- Bring the empty gripper near the target at a safe height, then compare it with the
  target in fresh images and correct with small moves.
- Judge height and alignment from more than one view: a pixel overlap in one image
  does not prove contact or alignment (parallax). Confirm in the other camera or
  after a small motion before contact.
- A learned image-to-motion mapping holds only near that height and camera; refit
  after large changes. Alignment motions consume episode steps; keep them few and small.
{% if not formal_episode %}
Record in memory what you aligned against, the views and motions used, and the
outcome, not coordinates.
{% endif %}
