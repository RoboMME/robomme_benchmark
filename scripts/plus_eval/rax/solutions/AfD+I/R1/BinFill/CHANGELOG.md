# binfill controller changelog

## binfill-v1 (registered for the single rehearsal)
Offline-only development (1 total simulator start = the rehearsal). Evidence: mock simulator
experiments/mock/sim.py (ray-cast BinFill stand-in), logs experiments/mock/log*.txt.
- Belt velocity voting from front-camera RGB-D detections; intercept with feed-forward.
- Wrist refinement of xy / top z / yaw (sign self-check), descent tracking belt, close, lift, finger check.
- Held colour check (wrong colour -> release off-bin), carry to bin (rim-top grid centre), release, count.
- Button press with closed gripper (grid-fit cap), retried twice lower.
Fixes during mock dev: approach exit check used future step; bin centre bias (ring/wrist occlusion);
descent overshoot (z interpolation from updating z); cube top-z mask; camera classifier
(bottom-corner saturation, 0/370 errors on demo frames).
Robustness: WALL_STOP 960 s guard inside Robot.step and loop; no-state branch keeps stepping
(avoid 180 s no-progress timeout) and gives up after 40; main() never raises.
Mock results: seeds 0,1,2 success; variants (6 cubes, speed .07, frac .5 fmt B, fmt C, dark) success.
Fault injection (experiments/mock/run_fault.py, seed 3): p2w errors 15% -> success; step errors 5% -> success;
garbage replies 5% / no images / invalid p2w / no state -> no crash, graceful termination.
