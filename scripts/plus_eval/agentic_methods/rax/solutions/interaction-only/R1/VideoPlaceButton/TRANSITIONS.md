# vpb controller — transition table and checklist (version vpb-1.x)

| Step | Preconditions (fresh) | Success exit | Failure / timeout exit |
|---|---|---|---|
| INIT | robomme_observe reply | -> VIDEO-LOAD | no state: one zero row (gripper close) then continue |
| VIDEO-LOAD | frame_sequence manifest (reply, status, or glob of runs/*/frames) | -> SCENE0 | no manifest: events empty |
| SCENE0 | front depth + map (tool fit, else calibration) | tz, comps, blobs | scene None: tz from video events, else 0 |
| VIDEO-EVENTS | >=5 states | events/hist -> target | no placements -> IMAGE-FALLBACK (colour track + button occupancy); still none -> blob fallback target |
| HOLDCHK | finger sum | holding -> PLACE | not holding -> PICK |
| PICK (<=3 attempts) | target known | fingers 0.006..0.07 after close and after lift -> PLACE | miss: open, lift, next attempt (re-perceive, ungated) |
| PLACE | holding | release + retreat -> VERIFY | stall on descent: release anyway (contact) |
| VERIFY | front scene | cube <3 cm from target or episode ended -> IDLE | cube >5 cm away -> PICK at seen location; unknown -> IDLE |
| IDLE | — | 12 hold rows, return | — |

Global guards: Robot.step raises EpisodeOver on episode_ended / remaining<=0 / wall deadline 900 s /
>25 tool errors. move_to has step caps and stall detectors (near: 6 calls <2 mm; far: 18 calls <3 mm).
Rotation control self-checks and degrades world -> body -> off.

Irreversible actions:
- release (PLACE): precondition = TCP within tol above target xy at release z or stalled on contact; checked in place().
- grasp close: precondition = at grasp z over cube; verified by finger sum (+ visual after lift, logged).
