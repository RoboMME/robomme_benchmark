# Controller changelog
## v1 (2026-11-16)
- Evidence: ep1 manual (failed by wrong-bin guess, but revealed two-wave timeline); offline replay of ep1 frames
  (tracker finds all 7 cubes; bins match cube xy within 1.2 cm); mock ctx full run + injected tool errors OK.
- Pipeline: press button (front-depth localisation) -> park (-0.12,0,0.33) -> watch single steps until all
  target colours seen+covered+static 10 frames -> per target: locate bins, nearest to cube last xy, yaw from
  min-area rect of top points, choose finger axis with most clearance, grasp at ztop-0.015, lift 0.18, place back.
- Untested live: place-back, yaw > 0.2 rad grasps, second/third picks, success signal.

## v2 (2026-11-16)
Evidence: rehearsal 1 (v1, bundle 2bb789563d9541dd8c8298ba4a3593d6) failed: 3 cube waves (ep1 had 2), v1 stopped
watching at step 98 and tracked one instance per colour; pillar occluded the front camera; red lift at step 130
(during wave 3) apparently ignored, so the correct blue lift at 208 ended the episode as failure.
Changes:
- tracker: per-pixel HSV classes (no background difference), multiple instances per colour, both cameras,
  tighter hue tolerance for red/orange/yellow; parse "all ... cubes/containers".
- watch from the start pose (wrist sees the whole table top-down), press early (step>=14) and return.
- watch ends only after every target is seen and no cube was visible for 30 steps (hiding finished).
- travel height from obstacle point cloud (pillar) ; bins from wrist+front at park, refine with wrist
  above the bin (rejected if shift >2 cm or size off).
- robust step error recovery (repeated observe retries), watch errors fall through to picks.
Offline: mockctx (ep1 replay) full run, fault injection (calls 5,50-52,300,301), early end at 30.
- PRESS_START_MIN 14->20 so every wave-1 cube gets several sightings before the arm leaves park.
