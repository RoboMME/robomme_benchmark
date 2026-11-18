# Changelog
- insertpeg-v1 (2026-11-16): first and only registered version (R1 budget: no live
  exploration possible). Offline-tested on a fake kinematic simulator only
  (experiments/fakesim.py, run_fake.py, run_faults.py, stdio_driver.py).
  Fixes during offline testing: grasp-end index, calib u/v hypothesis, top-face centroid
  bias -> mid-range centre, gripper-flip symmetry, identity-branch flip, consecutive
  error counting, gripper event onset snapping (slow-gripper look-ahead fired early).
