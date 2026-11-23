# Runtime limits

These resource limits apply equally during development rollouts and frozen tests.
They are wall-clock safeguards, independent of the task's native control-step limit.

- Conditioning videos: at most 512 uniformly sampled timestamps, both endpoints,
  both cameras and original timestamps retained. Current live observations are
  not downsampled. Released demonstration files remain reusable offline evidence.
- Simulator startup: at most 600 seconds. One simulator RPC: at most 180 seconds.
- Autonomous controller: at most 1,200 seconds; at most 600 seconds before its
  first control step, and 180 seconds without advancing control after that.
  Repeated reads, logging and image processing do not reset the progress timer.
- ASPIRE provider calls have bounded timeouts. A hung shared provider is restarted;
  an interrupted call is recorded rather than silently replayed.
- The outer evaluation worker has a 2,100-second deadline including cleanup.

Handle tool failures explicitly. Do not retry an uncertain robot action: physics
may already have advanced. Stop after the simulator reports a terminal episode.
Every development simulator start still counts against R, including timeouts and
failed rehearsals. Resource limits do not authorize additional attempts.

R0 and R10 evaluations retain rollout artifacts; R5 retains results and logs.
The operator records sampled frame counts, policy hashes, outcomes and errors.
Frozen evaluation cannot modify the policy or feed test results into development.
