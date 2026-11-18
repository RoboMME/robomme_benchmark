# Controller checklist (swing-v1)

| Step | Preconditions (checked from fresh obs) | Progress measure | Stall / timeout | Escalation |
|---|---|---|---|---|
| PERCEIVE (controller.perceive) | front RGB + depth/API | blobs found | single pass | raise arm + re-perceive (run) |
| GRASP (Ctl.grasp) | cube located, gripper open | TCP distance to target, finger sum | move_to max_iter, stall<2mm | re-perceive, retry x3, STOP |
| VISIT (Ctl.visit) | finger sum > thr (held) | TCP xy error to target | move_to limits | drop -> re-perceive + regrasp (max 2) |
| PUTDOWN (Ctl.put_down) | held | TCP z | move_to limits | none (open + retreat) |
| BUTTON (Ctl.press_button) | instruction has 'button', button located | TCP z vs button top | 3 depths, stall check | none |

Irreversible actions: release (put_down; only after the visit loop), button press (only after put_down,
only if the instruction mentions a button and a non-cube elevated blob was found).
Switches: none; no env vars or files outside the project are read.
