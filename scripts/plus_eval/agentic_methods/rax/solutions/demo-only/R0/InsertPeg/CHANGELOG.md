# insertpeg controller changelog
## v1 (2026-11-16) — submitted candidate, offline only (demo-only, zero simulator starts)
Evidence: 3 released recordings show video layout == execution layout and the expert repeating
the video's grasp/insert; mock-simulator logic tests in /workspace/experiments (not sim evidence).
- Replay the conditioning video's per-frame EE state trajectory with closed-loop absolute pose
  tracking; gripper events from finger widths; convergence gates with stall detection;
  one regrasp if the grasp closes empty; push 3/6/9 mm along the final straight approach;
  retract/offset search (<=4 mm) only if the insertion is blocked.
- Fallback when the reset manifest has no moving per-frame state: no motion (known limitation).

## Checklist (step: precondition / progress / stall / timeout / escalation)
- LOAD: manifest with >=10 video states and >=2 cm EE range / - / - / - / FALLBACK.
- TRACK gates: converged pose / pose error / STALL_WINDOW no-improvement / GATE_MAX_WAIT / continue.
- GRASP (irreversible close): converged at video close frame / finger width / settle check /
  GRIP_SETTLE_MAX / width<3 mm while video held >8 mm -> open, redo once from previous gate.
- FINISH: push depth along insertion axis / blocked if short by >4 mm / ladder offsets / wall
  limit 1080 s / end at deepest offset.
