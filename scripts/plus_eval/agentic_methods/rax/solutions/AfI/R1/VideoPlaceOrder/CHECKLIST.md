# vpo controller checklist (v1)

| Step | Preconditions (fresh obs) | Progress measure | Stall / timeout | Escalation |
|---|---|---|---|---|
| P0 sense (`Controller.run`) | observe has 8-D state | n/a | abort if no state | calibrate() picks back-projection variant vs `robomme_pixel_to_position`; default OpenGL if API fails |
| P1 video (`analyse_video_states`) | manifest frames with `state` | n/a | n/a | colour-tracking fallback (`video_color_fallback`) |
| start_low rise | EE < table+0.10 | z | move_to stall(6 calls)/40 steps | relocate cube after rise |
| P3 pick (`Controller.pick`) | gripper open, cube estimate | TCP distance | move_to stall 6 calls, max 160 steps | wrist refine; API-sampled points if no depth; 3 attempts w/ re-perception; last-resort cube = last placement xy |
| yaw (`Robot.rotate_yaw`) | wrist yaw estimate, |dyaw|>12deg | rot error | 3 calls w/o 0.5deg gain | flip sign once, else give up (grasp anyway) |
| grasp verify | finger sum | width in (0.45*cube, 0.078) | n/a | open, relocate, retry |
| P4 place (`Controller.place`) | holding_width true | TCP distance | same as move_to | lost cube -> relocate + repick; verify after release (3.5 cm) -> repick once |
| P5 idle | - | - | 15 zero rows | return |

Irreversible: release (checked: over target xy within 3 mm, at expert release z + 6 mm).
Global: every tool error -> re-observe (never blind repeat), >8 errors -> stop; wall budget 1050 s;
EpisodeOver on episode_ended/task_complete -> stop immediately.
