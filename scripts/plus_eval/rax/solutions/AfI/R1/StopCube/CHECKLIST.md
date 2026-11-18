# pre-freeze checklist (stopcube-v1)
- [x] main(ctx) catches all exceptions, logs, holds until episode end (post) — fault tests pass
- [x] imports: stdlib, numpy, PIL; scipy/cv2 optional with fallbacks
- [x] wall budget 1100 s < controller limit; no step-free phase > 90 s (video backtrack capped)
- [x] outputs to /workspace/output (run_log.jsonl, images capped at 40)
- [x] full offline sweep re-run right before registration (same known-failure set)
