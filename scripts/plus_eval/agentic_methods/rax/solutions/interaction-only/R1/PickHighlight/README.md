# pickhl — PickHighlight controller (RAX, interaction-only, R1)

Written fully offline (R1: the only simulator start is the rehearsal). Nothing here is tuned on live data.

## Transition table (controller.py:run)
| Step | Preconditions | Success exit | Failure / timeout exit |
|---|---|---|---|
| S0 init | observe ok, state parsed | S1 | no state -> return |
| S1 perceive (front exec frame from reset manifest, depth back-projection, convention verified with pixel tool; wrist start view adds hidden cubes) | depth+calib, else tool-sampled cloud | S2 | no cubes -> return |
| S2 highlight (per-cube colour-change score vs exec frame over video frames, depth-gated, shadow-filtered, cross-talk removed; target = score > max(0.05, 0.3*max); instruction count/singular/first/last) | video frames | S2b | no video highlight -> live watch 40 steps; still none -> top-1 |
| S2b probe (translation map, rotation frame, gripper sign; ~25 steps) | arm at start | S3 | exception -> logged, defaults |
| S3 per target: hover -> wrist refine (tool-checked calib, table-height check) -> yaw align (clearance-aware) -> pre -> descend -> close -> width check -> lift -> task_complete? -> place back -> open -> retreat | | next target | miss -> reacquire (home + front re-perceive) -> offset retry; 3 attempts |
| S3b borderline cubes (0.03 < score < thr), max 2 | not complete | | |
| S4 re-pick last target and hold lifted until episode end / wall deadline | not complete | | |

Every tool error: logged, observe+status, no blind retry; >6 consecutive errors or episode_ended -> stop.
move_to: closed loop on measured state, stall detector (4 chunks <15% progress), row budget, wall deadline.

## Irreversible actions
- close gripper: precondition TCP at grasp height above perceived cube (descend result logged); verified by finger width.
- release (place): cube lowered to table+h/2+8mm; checked by width afterwards implicitly at next close.
- picking a wrong cube may end the episode: only cubes with highlight evidence are picked (borderline only after all targets).
