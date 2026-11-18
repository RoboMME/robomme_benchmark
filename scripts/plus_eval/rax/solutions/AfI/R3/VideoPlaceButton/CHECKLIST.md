# vpb controller checklist / transitions
S0 observe@0 -> S1 (manifest from frame_sequence; fallback glob of runs/*/frames/sequence_000000)
S1 plan: video events (finger width) + target slots/swaps (swaps.py) + instruction -> goal slot (identity followed through swaps)
   fail -> goal=release pos (no identity) or largest target
S2 locate cube (front colour+depth; prior = last release of that colour)
S3 hover 0.15 -> wrist refine xy+yaw (3 iters) at 0.10
S4 descend 0.048 -> 0.018, close 8 steps, lift 0.15; finger sum < 0.012 -> reopen, retry S2 (max 3 attempts)
S5 carry 0.15 over goal, descend 0.054 -> 0.024, open 8 steps (episode ends here on placement), lift
S6 verify cube xy vs goal from front; >2 cm -> retry S2
Irreversible: release over a target ends the episode -> goal computed once, logged in output/vpb_log.txt
