---
trigger: always_on
---

This workspace is managed by uv.


## Always run from repo root ✅
- Install deps exactly from lock:
  - uv sync --locked
- Run Python:
  - uv run -- python <file.py>
  - uv run -- python -m <module>
- Run tests:
  - uv run -- pytest

## Run a specific package (workspace member)
- uv run --package <package_name> -- python <file.py>
- uv run --package <package_name> -- python -m <module>
- uv run --package <package_name> -- pytest

## Dependency changes (if needed)
- Add to one member:
  - uv add --project path/to/member <dep>
- After any dep change:
  - uv lock
  - uv sync
