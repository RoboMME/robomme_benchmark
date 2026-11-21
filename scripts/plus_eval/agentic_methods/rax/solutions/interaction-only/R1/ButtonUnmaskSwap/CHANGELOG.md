# bums controller changelog
## v1 (2026-11-16) — first and only registered version (R1: one simulator start total)
Evidence: offline synthetic mock only (experiments/mock_sim.py, batch.sh): 20/21 variants + fault injection OK.
No real-simulator evidence before the rehearsal.
Checklist (step: precondition / progress / timeout / escalation):
- init: observe parsed, front rgb+depth found / camera fit rms<2cm else parsed calib / - / phases skipped if fails
- prewatch: arm still; motion = bin set changes / static counter / cap step 95 / -> press
- press: button found / TCP z vs target, stall detect / step-bounded moves / late-press warning
- watch: home reached / static counter, target cube visibility / cap step 330 or 140 steps reserve / -> decide
- decide: last frame with bins / reverse tracks + covering events / - / fallbacks nearest-last-seen, elimination, guess
- pick: finger_sum after close >4mm = held / lift height / 3 attempts with z offsets / -> hold
- hold: until episode end / step limit / wall limit 1100 s
Irreversible: lifting a bin (wrong bin = failure): only after decide; press only above detected button.
