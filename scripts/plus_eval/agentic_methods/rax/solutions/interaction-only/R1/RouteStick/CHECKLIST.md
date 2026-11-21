# routestick controller checklist (v1)

| Step | Preconditions (checked from fresh data) | Progress measure | Stall / timeout | Escalation |
|---|---|---|---|---|
| PERCEIVE (`run`, `traj_from_states`, `states_vs_depth`, `traj_from_images`, `scene_shift`) | observe packet has state; manifest readable; >=5 video frames | n frames, path spread >= 2 cm, states-vs-depth consistency | all pre-motion; bounded loops over <=512 frames | states -> images (if states constant or <30% consistent with depth) -> safe hold |
| APPROACH (`go_to`) | path start >1 cm from live TCP | distance to target | 60/120/60 step caps, wall deadline | lift -> xy -> lower; proceeds to FOLLOW regardless |
| FOLLOW (`Follower.run`) | plan built | tracking error vs time-parameterised plan, plan index k | err>2.5 cm -> catch-up without advancing time; after 6 stalled calls plan time creeps; >5 consecutive tool failures or >3 stale states -> stop | gain adaptation (0.35-1.3), rotation convention world->body->off |
| SETTLE | FOLLOW done and episode not ended/succeeded | final distance | 15 steps + 10 hold steps | none |

Irreversible actions: none besides motion; each robot step is relative to the measured pose; rows are clipped to [-1,1], non-finite -> 0.
Switches: none (no env vars, no files outside the project read). Output only to /workspace/output.
