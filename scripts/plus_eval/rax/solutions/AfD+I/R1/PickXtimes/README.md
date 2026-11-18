# pickx — PickXtimes controller

Entry: controller.py:main(ctx). Modules: vision.py (numpy HSV/blob perception), mcpio.py (defensive reply parsing).
Built entirely offline (R1 budget: the one simulator start is the rehearsal).

## Step checklist
| Step | Preconditions | Progress measure | Stall/timeout | Escalation |
|---|---|---|---|---|
| INIT | observe reply has state+front image | — | — | defaults: colour green, N=1 if unparsable; table_z fallback |
| TARGET_REF | target seen in front | TCP distance to hover | move_to: 3 no-progress calls / 16 calls | keep front estimate |
| CUBE_REF | cube xyz known | wrist detection within 7 cm gate | same | re-detect in front view from home pose |
| GRASP | hover above cube, gripper open | finger sum after close; wrist colour fraction after lift | move_to stall (contact) | open, re-hover, re-detect (max 6 fails) |
| PLACE | cube held | cube xy vs target xy after release (<3 cm) | move_to stall = contact -> release anyway | not counted -> re-grasp from measured spot |
| BUTTON | count == N | z descent / episode end | 4 stalled chunks | 7 attempts with ±6-10 mm xy offsets, deeper each time |
Irreversible: button press only after count==N (press_button called once after the loop).
Wall guard: cycles stop after 1100 s.
Rotation monitor: if world-frame rotation commands increase error twice, switch to tool frame; then disable rotation.
