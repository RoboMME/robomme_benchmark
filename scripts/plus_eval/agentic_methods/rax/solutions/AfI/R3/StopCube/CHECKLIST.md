# Step checklist (controller.py)
| step | precondition | progress measure | stall/timeout | escalation |
|---|---|---|---|---|
| perceive (start) | step-0 obs | target (purple) / button (low-sat blob, x>-0.45) found | - | target geom fallback; button default |
| APPROACH | button xy | dist EE->(bxy,0.11) | 60 steps -> REFINE | - |
| REFINE | EE above cap | wrist cap pixels | 1 step | keep front estimate if wrist result off by >3 cm |
| DESCEND | refined cap | dist EE->hover | 40 steps -> HOVER | - |
| HOVER | - | cube arrivals counted every step | episode step limit | - |
| PRESS (irreversible) | count>=N-1, not yet pressed at this count, EE xy within 1.5 cm of cap, gap<8 cm, ta<=lead or inside | EE z, episode_ended | 7 rows / z<=cap_top-1cm | RETREAT, retry on next arrival (max 3) |
| loop | any exception -> log, re-observe, continue; >20 errors -> return; wall >1100 s -> return |
