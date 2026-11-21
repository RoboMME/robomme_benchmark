# stopcube controller changelog

## stopcube-v1 (registered for the single R1 rehearsal)
Built offline only (R1: the single simulator start is the mandatory rehearsal), tested
against my synthetic scene experiments/fake_sim.py (not the benchmark).
- Perception: RGB-D back-projection (calibration variants checked against
  robomme_pixel_to_position, linear fit fallback), table-plane height map, raised
  components, flat colour patches as target candidates, cube = component that moves.
- Logic: N parsed from the instruction; visit counting with hysteresis; predictive press
  trigger (constant-velocity closest approach within latency L); late fallback when inside.
- Motion: approach above button (SAFE_DZ 0.10), wrist refine, latency calibration
  (PRESS_DZ -0.5), hover at top+0.025, press with escalation to dz=-1, verify stop, hold.
- Evidence: experiments/all_cases.json sweep: all succeed except 3 definitional cases
  (target at path end, start on target, video with > N visits). Slow-robot early
  first-visit (N=1) also fails.
