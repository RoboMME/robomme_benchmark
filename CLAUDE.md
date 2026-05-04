# CLAUDE.md — RoboMME Benchmark Agent Guide

## Environment & Package Management (uv)

This repo uses **uv** exclusively. Never use bare `pip`, `python`, or `conda` unless explicitly asked.

```bash
# First-time setup
uv sync

# With optional server dependencies (challenge interface)
uv sync --group server

# Add a dependency
uv add <package>
uv remove <package>

# Always run commands through uv
uv run python scripts/run_example.py
uv run python -m pytest tests/
```

Python version is pinned at **3.11** (`.python-version`). `pyproject.toml` is the single source of truth; `uv.lock` must stay consistent with it.

`mani-skill` is sourced from a specific Git commit — do not change `[tool.uv.sources]` without a reason.

---

## Error Handling Philosophy

**No silent fallbacks.** Every error path must raise explicitly. No `except Exception: pass`, no quiet defaults when something is missing, no returning `None` where a value is expected. If something can go wrong at a system boundary, raise a typed exception with a clear message. `info["status"] = "error"` in env wrappers is the one sanctioned soft-error pattern (see `FailAwareWrapper`).

---

## Repository Layout

```
src/robomme/
  robomme_env/          # 16 task environments (one file per task)
  robomme_env/utils/    # shared helpers: planners, object generation, scoring
  env_record_wrapper/   # RecordWrapper, DemonstrationWrapper, BenchmarkEnvBuilder
  env_metadata/         # per-task episode metadata (train/val/test JSON)
  logging_utils.py

scripts/
  run_example.py        # quick single-env smoke test
  evaluation.py         # full eval harness
  dataset_replay.py     # replay downloaded HDF5 data for sanity check

challenge_interface/    # CVPR 2026 challenge server/client stubs
  policy.py             # participants implement Policy.infer() / Policy.reset()
  server_http.py / server.py
  scripts/deploy.py / phase1_eval.py

tests/
  lightweight/          # fast unit tests, no physics engine
  dataset/              # heavy e2e tests, real Mujoco sim + HDF5 recording
  _shared/              # shared fixtures and dataset generation cache
  conftest.py           # session-level repo_root / src_root fixtures
```

---

## Running Tests

All test commands go through `uv run`.

```bash
# Full suite
uv run python -m pytest tests/

# Fast logic-only tests (no GPU, no physics engine)
uv run python -m pytest tests/lightweight/

# Heavy e2e tests (requires display/GPU — uses real sim)
uv run python -m pytest tests/dataset/

# By pytest mark
uv run python -m pytest -m lightweight
uv run python -m pytest -m dataset
uv run python -m pytest -m gpu
uv run python -m pytest -m slow

# Single file or test
uv run python -m pytest tests/lightweight/test_TaskGoal.py
uv run python -m pytest tests/lightweight/test_TaskGoal.py::test_binfill_two_colors

# With live output
uv run python -m pytest tests/ -s
```

### Test categories

| Directory | Marks | Cost | What it covers |
|-----------|-------|------|----------------|
| `tests/lightweight/` | `lightweight` | Seconds | Pure logic: label matching, planners, AST checks, mocks |
| `tests/dataset/` | `dataset`, `gpu` | Minutes | Full env loop: sim + wrapper + HDF5 record/replay |

`tests/dataset/` uses a **session-scoped hash cache** (`_shared/dataset_generation.py`) so identical env/seed/difficulty combinations only generate data once per pytest session.

### Preferred testing approach

Prefer **end-to-end tests** in `tests/dataset/` for anything that touches env wrappers, HDF5 recording, or observation shapes. Lightweight tests are for pure functions that have no sim dependency. Do not mock the physics engine or the wrapper stack — divergence from real behavior has caused silent bugs before.

---

## Key APIs

### Instantiating an environment

```python
from robomme.env_record_wrapper import BenchmarkEnvBuilder

builder = BenchmarkEnvBuilder(
    env_id="PickXtimes",   # task name
    dataset="test",         # "train" | "val" | "test"
    action_space="joint_angle",  # "joint_angle" | "ee_pose" | "waypoint" | "multi_choice"
    gui_render=False,
)
env = builder.make_env_for_episode(episode_idx=0)
obs, info = env.reset()
task_goal = info["task_goal"][0]

obs, reward, terminated, truncated, info = env.step(action)
# info["status"] == "error" means the step raised internally — do not ignore it
```

### Challenge interface (CVPR 2026)

```bash
uv sync --group server

# Terminal 0: policy server
uv run python -m challenge_interface.scripts.deploy --port 8001

# Terminal 1: local evaluation client
uv run python -m challenge_interface.scripts.phase1_eval --port 8001
```

Participants implement `Policy.infer()` and `Policy.reset()` in `challenge_interface/policy.py`.

---

## Dependency Groups

| Group | Purpose | Install command |
|-------|---------|-----------------|
| (default) | core sim + training | `uv sync` |
| `dev` | pytest + opencv dev tools | `uv sync --group dev` |
| `server` | Flask + msgpack + websockets (challenge interface) | `uv sync --group server` |

---

## Headless / GPU Notes

Tasks require a display for Vulkan rendering. For headless machines:

```bash
# CPU rendering fallback (slower, no Vulkan required)
export SAPIEN_RENDER_DEVICE=cpu
export MUJOCO_GL=osmesa
```

Or use the provided Docker image:

```bash
docker build -t robomme:cuda12.8 .
docker run --rm -it --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
  -v "$PWD/runs:/app/runs" \
  robomme:cuda12.8
```

Tests marked `gpu` will fail without a working display or the env vars above.

---

## Adding New Tests

- **Pure logic** → `tests/lightweight/test_<topic>.py`, mark `@pytest.mark.lightweight`
- **Sim-dependent** → `tests/dataset/test_<topic>.py`, mark `@pytest.mark.dataset`
- Register any new dataset fixture in `tests/dataset/conftest.py` following the existing `dataset_factory` pattern
- Do not skip or `xfail` without a comment explaining why — prefer fixing the root cause
