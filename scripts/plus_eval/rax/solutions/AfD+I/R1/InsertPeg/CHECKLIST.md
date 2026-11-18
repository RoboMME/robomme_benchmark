# insertpeg-v1 checklist (checked against code before registering)

| Step | Preconditions (fresh obs) | Progress measure | Stall / timeout | Escalation |
|---|---|---|---|---|
| INIT (controller.main) | observe ok, state parsed | - | tool errors caught | END |
| DECIDE (replay.states_usable) | >=30 video states, TCP range >3 cm, gripper ptp >4 mm, start within 6 cm | - | - | VISION |
| probe_rotation (rio) | state | measured yaw response | 10-step undo | adaptive _check_rot_convention |
| REPLAY (replay.run_replay) | waypoints | per-wp position/rotation error | catch-up <=3/wp, budget 150 | chunking if slow (chunk_size) |
| grip transition | video width change | finger width settles | 25 steps max | check_grasp -> one regrasp |
| PUSH (replay.final_push) | not complete, closed gripper | advance along insertion dir | goto stall (6) | spiral offsets <=6 mm |
| VISION (vision.run_vision) | depth+calibration (validated vs tool) | see vision.py | goto stall | hole hypothesis scan |

Irreversible: closing the gripper (only at video grasp pose), pushing into the box (bounded 18 mm beyond video end).
Never stops on parsed `remaining` alone; stops on episode_ended / task_complete / 6 consecutive step errors / 1050 s.
