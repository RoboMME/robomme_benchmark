# StopCube controller changelog
- v1 (2026-11-16): single per-step loop; colour cube tracking (front cam) + geometric fallback; arrival counter
  (excursions < 0.025 m from target); approach -> wrist refine at z 0.11 -> hover cap_top+12 mm -> press when
  count==N-1 and predicted closest approach <= 1.5(+gap/0.015) steps ahead. Evidence: Ep1 live script success (step 253).
- v1.1: retry presses only on a later arrival (pressed_counts), not repeatedly within the same arrival;
  geometric target fallback. Offline tests: experiments/offline_test.py (6 cases) + geom-only cube case, all pass.
- v1.2: (rehearsal 1 of v1.1 failed: pink cube, colour mask latched onto static table pixels, never counted)
  geometric cube detection primary (front depth, world z 0.020-0.07, >7 cm from button), colour+height fallback;
  APPROACH_CLIP 0.8->1.0, refine at cap_top+0.07, DESCEND_CLIP 0.5; press lead from hover gap (press_lead);
  target mask restricted to disk height. Regression experiments/regression.py on Ep1 + R1 recordings,
  N=3/4/5, frozen world, injected error, never-registering press: 0 failures; press within -0.5..+0.4 steps of crossing.
