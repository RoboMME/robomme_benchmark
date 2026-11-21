# swing controller changelog

## swing-v1 (submitted candidate, demo-only, offline-only)
Evidence: 3 training demos (runtime/demonstrations/episode_0..2) + kinematic mock
(/workspace/experiments/mock). No simulator interaction (demo-only, R0).

Phases: perceive (front RGB-D fixtures: button + 2 targets) -> grasp (front cube, wrist
refine, yaw align, close, lift, finger check) -> swing (2N alternating hovers, first side from
instruction; image-left target = robot right) -> place (back at pick spot) -> press button with
closed gripper, increasing depths until episode ends.

Changes during offline development (each driven by mock or demo evidence):
- stall detection relative to free-motion ratio, gated near expected z (false contact at track 0.25)
- adaptive gain in move_to from tracking EMA; TRAVEL_CLEAR 0.085->0.065, SPEED_CARRY 0.020 (step use)
- retry condition: red<0.03 & neutral>0.25 (old cond. referenced other target, wrong)
- wrist yaw sampling margin=0 + top-face z filter (disc sampling lost orientation)
- finger-axis rows 100/130; fallback to R[:,1] when rows disagree >10 deg
- button press: fixed depths 12/22/32 mm instead of stop-on-stall
- lit check at visit bottom (indicator lasts ~1 s)
- hover 8 mm above target by default (ep0 expert hovers); switch to contact if a contact retry registers
- partially labelled RGB images: unlabelled one assigned to the remaining camera

Mock regression (all complete): ep0/1/2, track 0.5/0.4, rot_local, label_mode none, lit_height 0.003.
