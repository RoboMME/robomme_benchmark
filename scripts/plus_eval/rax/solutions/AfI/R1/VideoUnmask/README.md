# vunmask — VideoUnmask controller (RAX, AfI, R1)

Written blind (no live exploration possible: the single simulator start is the rehearsal).
Entry: `controller.py:main(ctx)`. Modules: `vu_io.py` (reply parsing), `vu_geom.py`
(rotations, camera models), `vu_percep.py` (colour blobs, tracking, point-cloud objects).
Outputs: `/workspace/output/controller_log.txt`, `summary.json`, `observe0_summary.json`,
`status_reply_summary.json`, `debug_video.png`, `wrist_view_*.png`.

## Transition table
| step | success -> | failure -> |
|---|---|---|
| INIT (status, observe step 0) | CALIB | no state: return |
| CALIB (own back-projection vs robomme_pixel_to_position) | VIDEO | tool mode / flatness fallback |
| VIDEO (track instruction colours + seed colours) | SCENE | 'any' saturated / none -> SCENE |
| SCENE (front cloud, table z, objects, nearest to cube, seed split) | PICK | guess object nearest object-centroid |
| PICK: view_refine -> grasp_lift | lifted -> PLACEBACK (if more targets) / HOLD | miss: yaw+90, then deeper; slip: deeper; max 3 |
| PLACEBACK | next PICK | log, next PICK |
| HOLD | return | return |

## Checklist (per step: precondition / progress / stall / timeout / escalation)
- move_to: measured pose each call; error logged; max_calls bound; rotation non-convergence -> frame switch.
- CLOSE: fingers settle loop (<=6x3 steps); width < 6 mm = miss.
- LIFT: width after lift < 6 mm = slip; task_complete watched every reply.
- Episode end / wall limit (1080 s): EpisodeOver raised from Robot.step, main returns.
- Irreversible: lifting a container (wrong one cannot be undone) — precondition is the
  video-track + nearest-container decision, logged in summary.json before motion.
