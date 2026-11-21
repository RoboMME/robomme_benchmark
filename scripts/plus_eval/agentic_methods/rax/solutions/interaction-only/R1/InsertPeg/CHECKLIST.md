# insertpeg controller — transition table and checklist (VERSION insertpeg-v1)

Phases: SETUP -> REPLAY -> PUSH -> END ; SETUP -> FALLBACK -> END ; any -> END on
episode_ended / task_complete / wall (WALL_LIMIT_S=1000 s) / 6 consecutive or 60 total step errors.

| Phase | Preconditions | Progress measure | Timeout / stall | Escalation |
|---|---|---|---|---|
| SETUP (run) | observe ok (3 read-only retries) | - | - | no state -> END; no manifest/video state -> FALLBACK |
| calibrate | exec-frame front depth | fit rms of pixel_to_position samples | 1 call | json calib if fit rms > 4 mm; none -> identity transforms |
| scene.analyse | calib, video frame 0 depth | depth agreement fractions | - | any exception -> identity (pure replay) |
| approach (converge) | state | pos/rot error | 120 steps | proceed anyway (logged) |
| pre-grasp converge | before close event | pos err < 1.2 mm, rot < 1.5 deg | 40 steps | proceed (logged) |
| grasp settle | after close run | finger width change | 12 steps | warn if width << demo width |
| pre-insert converge | start of final straight segment | pos < 1 mm, rot < 1 deg | 40 steps | proceed |
| replay loop | per demo index | servo error per step | demo length; stride 2 if steps/wall short | - |
| push | demo ends holding | - | ~30 steps | - |
| fallback | calib + depth | - | fixed caps per move | returns when no peg/box found |

Irreversible actions: gripper close (preceded by converge at the demo's close pose);
gripper open (converge first). No action is resent after an error; the loop recomputes
from a refreshed state.
