# patternlock controller (v1) — written blind (interaction-only, R1: the only simulator start is the rehearsal)

Entry: controller.py:main(ctx). Modules: plcore.py (pure python parsing/math/waypoints),
plvision.py (numpy/PIL/scipy fallbacks). Logs: stdout + /workspace/output/controller_log.txt,
output/{packet0,status0,manifest_summary,video_states,plan_waypoints,result_controller}.json.

## Transition table
| step | success -> | failure/timeout -> |
|---|---|---|
| P0 observe (state, manifest) | P1 | no state: exit |
| P1 plan A_state (video frame `state` EE positions, range>=2 cm, sanity vs depth) | P2 replay | vision_plan |
| P1 vision: calibrate front cam (pinhole candidates/affine vs robomme_pixel_to_position, residual<1.5 cm) | track tip | exit plan None -> P3 |
| P1 A_vision tip track (lowest fg depth pt + exec TCP offset; near-table -> press) range>=3 cm | P2 replay | B |
| P1 B highlight order (RGB change blobs, first-appearance order) 2..12 targets | P2 touch | P3 |
| P2 replay follower | P3 | episode_ended/task_complete: exit; wall>900 s: P3 |
| P3 settle (10 hold steps) | exit | exit |

## Checklist (step: precondition / progress / stall / timeout / escalation)
- keypoint (start,end,z-min,corner,dwell): measured dist<4 mm / dist per chunk / <1 mm over 2 chunks & d<2 cm -> accept / 40 steps -> accept / next waypoint
- pass-through waypoint: predicted dist<6 mm / measured dist / no 3 mm improvement in 4 chunks -> skip to next key
- orientation: error to target (demo if it varies >=0.05 rad else initial) / error / error growing >0.05 over 6 corrections and >0.25 -> disable rotation
- step tool error: no retry; robomme_status + observe; >=4 errors -> end
- wall: 900 s controller budget; chunk size grows if calls are slow
Tested only offline against experiments/mock_env.py (synthetic camera/grid; NOT the simulator).
